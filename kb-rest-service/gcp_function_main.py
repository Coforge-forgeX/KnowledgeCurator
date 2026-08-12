"""GCP Cloud Functions HTTP entrypoint using provider-agnostic handlers."""

import asyncio
from typing import Any, Optional

from src.adapters.cloud_function_adapter import (
    abstract_response_to_http_tuple,
    dispatch_request,
    from_gcp_request,
)

# One event loop per warm instance, reused across invocations.
#
# `asyncio.run()` would create AND CLOSE a loop per invocation, which throws
# away every cached async resource the process holds — most importantly the
# pooled LightRAG services (`core.lightrag_pool`) and their Neo4j/Postgres
# connections, whose transports are bound to the loop that created them. With a
# persistent loop a warm instance skips the ~8s-per-KB LightRAG initialization
# that a fresh loop would force it to repeat.
_loop: Optional[asyncio.AbstractEventLoop] = None


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop


async def _handle(request: Any):
    abstract_req, abstract_ctx = from_gcp_request(request)
    abstract_resp = await dispatch_request(abstract_req, abstract_ctx)
    body, status_code, headers, mimetype = abstract_response_to_http_tuple(abstract_resp)
    return body, status_code, headers


def entrypoint(request: Any):
    # GCP Functions accepts Flask-style return tuples: (body, status, headers)
    return _get_loop().run_until_complete(_handle(request))
