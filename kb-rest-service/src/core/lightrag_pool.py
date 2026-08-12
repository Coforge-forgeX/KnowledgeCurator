"""
Process-wide pool of initialized `LightRAGService` instances.

Why this exists
---------------
`LightRAGService.initialize()` is expensive: it probes the Neo4j database, runs
PostgreSQL schema/column/index checks and builds the embedding client. Measured
cold cost is ~8s per knowledge base (and ~50s on the very first call in a fresh
worker). Before this pool, `MultiKBStrategy` built a fresh service per KB on
every request and dropped it at the end, so that cost was paid again on every
single query — for a 2-KB workspace, ~16s of pure setup per request.

Serverless safety
-----------------
This service is deployed functionless (Azure Functions, AWS Lambda, GCP Cloud
Functions) as well as container-hosted, and those hosts differ in a way that
matters a great deal for cached async state:

* Azure Functions (v2 async model) and FastAPI/uvicorn run every invocation on
  ONE event loop per worker process. Pooled services are reused across requests
  and the optimization pays off fully.
* `aws_lambda_handler.lambda_handler` and `gcp_function_main.entrypoint` call
  `asyncio.run()`, which creates and closes a NEW event loop per invocation.
  Anything holding loop-bound state (asyncio locks, asyncpg pools, the Neo4j
  async driver) becomes invalid the moment that loop closes; reusing it raises
  "attached to a different loop" / "Event loop is closed".

So the pool is keyed to the running event loop. When the loop changes, the
whole pool is dropped and rebuilt. On Azure/FastAPI that never happens and
every request after the first is warm. On Lambda/GCP it happens every
invocation, which means no speedup there — but also no corruption, and no
regression: `LightRAGService.close()` only cleared `_rag`/`_initialized` (it
never closed the underlying drivers), so dropping the reference does exactly
what the old per-request cleanup did.

Entries also expire (`POOL_ENTRY_TTL_SECONDS`) and can be invalidated on query
failure. A consumption-plan worker can be frozen between invocations for long
enough that the database drops the connection underneath a pooled service;
without those two escape hatches a stale entry would wedge that KB for the rest
of the worker's life.
"""
import asyncio
import os
import time
from typing import Dict, Optional, Tuple

from src.core.config import settings
from src.core.lightrag_service import LightRAGService
from src.core.logging import get_logger

logger = get_logger(__name__)


def _default_max_services() -> int:
    """
    Ceiling on retained instances, lower under serverless.

    Each entry holds an embedding client plus live Neo4j/Postgres connections.
    A function instance runs with far less memory than a container and scales
    out horizontally, so the same ceiling costs N times the connections against
    Postgres/Neo4j once the platform fans out. A small cap still covers the
    common case completely — a workspace with 2 KBs on one agent occupies 2
    entries — while bounding a busy multi-tenant worker.
    """
    if bool(getattr(settings.database, "SERVERLESS", True)):
        return 6
    return 32


MAX_POOLED_SERVICES = int(
    os.getenv("LIGHTRAG_POOL_MAX_SERVICES", "") or _default_max_services()
)

# Rebuild an entry older than this. Guards against a frozen serverless worker
# resuming with server-side-closed database connections.
POOL_ENTRY_TTL_SECONDS = int(os.getenv("LIGHTRAG_POOL_TTL_SECONDS", "900"))

# (working_dir, workspace_label, workspace_id, agent_id) — every field that
# feeds LightRAGService._build_runtime_signature(). Keying on less would defeat
# the pool: `set_runtime_context` invalidates an instance whenever workspace_id
# or agent_id changes, so two workspaces sharing one KB label would thrash a
# single entry back into re-initialization on every alternating request.
PoolKey = Tuple[str, str, Optional[int], Optional[int]]

_pool: Dict[PoolKey, "_PoolEntry"] = {}
_pool_loop: Optional[asyncio.AbstractEventLoop] = None
_pool_lock: Optional[asyncio.Lock] = None


class _PoolEntry:
    __slots__ = ("service", "created_at")

    def __init__(self, service: LightRAGService) -> None:
        self.service = service
        self.created_at = time.time()

    def is_expired(self) -> bool:
        return (time.time() - self.created_at) > POOL_ENTRY_TTL_SECONDS


def _lock_for_current_loop() -> asyncio.Lock:
    """
    Return a lock owned by the currently running event loop.

    The lock is NOT created at import time on purpose. An `asyncio.Lock` binds
    to a loop on first use and raises if it is later awaited from a different
    one, which is precisely what happens on Lambda/GCP where every invocation
    gets a fresh loop from `asyncio.run()`.
    """
    global _pool_loop, _pool_lock

    running_loop = asyncio.get_running_loop()
    if _pool_lock is None or _pool_loop is not running_loop:
        if _pool:
            logger.info(
                "Event loop changed; dropping pooled LightRAG services",
                dropped=len(_pool),
            )
            _pool.clear()
        _pool_loop = running_loop
        _pool_lock = asyncio.Lock()

    return _pool_lock


async def get_pooled_lightrag_service(
    *,
    working_dir: str,
    workspace_label: str,
    workspace_id: Optional[int] = None,
    agent_id: Optional[int] = None,
) -> LightRAGService:
    """
    Return an initialized `LightRAGService` for this (KB, workspace, agent).

    The returned service is shared — callers must NOT `close()` it. Closing
    would clear `_rag`/`_initialized` and force the next caller to pay full
    initialization again, which is exactly what this pool exists to avoid. To
    retire a service that has gone bad, call `invalidate_pooled_service`.

    Initialization is serialized by the pool lock: `LightRAGService.initialize`
    configures the Neo4j/Postgres/embedding clients through process-level
    `os.environ` writes, so two concurrent initializations would clobber each
    other's credentials. The lock is only meaningfully held on a cold entry —
    for a warm one `initialize()` short-circuits on its runtime signature.
    """
    key: PoolKey = (working_dir or "", workspace_label or "", workspace_id, agent_id)

    async with _lock_for_current_loop():
        entry = _pool.get(key)

        if entry is not None and entry.is_expired():
            logger.info(
                "Pooled LightRAG service expired; rebuilding",
                workspace_label=workspace_label,
                age_seconds=round(time.time() - entry.created_at),
            )
            _pool.pop(key, None)
            entry = None

        if entry is not None:
            # Cheap no-ops when nothing changed; keeps behaviour identical to a
            # freshly built service if the runtime signature ever does change.
            entry.service.set_runtime_context(workspace_id=workspace_id, agent_id=agent_id)
            await entry.service.initialize()
            logger.debug(
                "Reused pooled LightRAG service",
                workspace_label=workspace_label,
                pool_size=len(_pool),
            )
            return entry.service

        _evict_if_full()

        service = LightRAGService(working_dir=working_dir, workspace=workspace_label)
        service.set_runtime_context(workspace_id=workspace_id, agent_id=agent_id)
        await service.initialize()
        _pool[key] = _PoolEntry(service)

        logger.info(
            "Initialized and pooled LightRAG service",
            workspace_label=workspace_label,
            working_dir=working_dir,
            pool_size=len(_pool),
        )
        return service


def invalidate_pooled_service(
    *,
    working_dir: str,
    workspace_label: str,
    workspace_id: Optional[int] = None,
    agent_id: Optional[int] = None,
) -> None:
    """
    Drop one pooled service so the next request rebuilds it.

    Called when a query against a pooled service fails: the failure may be a
    dead database connection held by that instance, and without this the same
    broken instance would be handed to every subsequent request. Synchronous
    and lock-free by design so it is safe to call from an exception handler.
    """
    key: PoolKey = (working_dir or "", workspace_label or "", workspace_id, agent_id)
    if _pool.pop(key, None) is not None:
        logger.info("Invalidated pooled LightRAG service", workspace_label=workspace_label)


def _evict_if_full() -> None:
    """Drop the oldest entry when the pool is at capacity. Caller holds the lock."""
    while len(_pool) >= MAX_POOLED_SERVICES:
        evicted_key = next(iter(_pool))
        _pool.pop(evicted_key, None)
        logger.info("Evicted pooled LightRAG service", workspace_label=evicted_key[1])


async def clear_lightrag_pool() -> None:
    """Close and drop every pooled service (shutdown / test teardown)."""
    for entry in list(_pool.values()):
        try:
            await entry.service.close()
        except Exception:
            pass
    _pool.clear()
    logger.info("Cleared LightRAG service pool")
