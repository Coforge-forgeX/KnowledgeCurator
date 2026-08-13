"""
Shared Document Deletion Utilities

Common deletion operations for indexed documents across LightRAG, Neo4j, blob storage, and ORM.
Used by both delete_files_by_id and delete_all_indexed_documents handlers.
"""
import asyncio
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, select, text

from src.core.config import settings
from src.core.database import DocumentMetadata, FileTask, get_async_session
from src.core.logging import get_logger
from src.core.neo4j_driver import get_neo4j_driver
from src.core.redis import redis_manager
from src.storage import get_storage_adapter

logger = get_logger(__name__)

FILE_KEY_PREFIX = "query_file:"
TABLE_CANDIDATES = [
    "lightrag_vdb_chunks",
    "lightrag_vdb_relation",
    "lightrag_vdb_entity",
]


def normalize_path(path: str) -> str:
    """Normalize file path for consistent matching across stores."""
    return str(path or "").strip().replace("\\", "/")


def build_path_variants(file_path: str) -> List[str]:
    """Build path variants for matching (with and without leading slash)."""
    normalized_file_path = normalize_path(file_path)
    variants: List[str] = [normalized_file_path]

    if normalized_file_path.startswith("/"):
        variants.append(normalized_file_path.lstrip("/"))
    else:
        variants.append(f"/{normalized_file_path}")
    return list(dict.fromkeys([v for v in variants if v]))


def build_path_suffix(path: str) -> str:
    """Build ILIKE pattern for path matching."""
    return f"%{path}"


def is_not_found_error(exc: Exception) -> bool:
    """Check if exception indicates resource not found."""
    message = str(exc).lower()
    return any(token in message for token in ["not found", "does not exist", "404", "blobnotfound"])


def is_retryable_db_error(exc: Exception) -> bool:
    """Detect transient DB disconnects that are safe to retry once."""
    message = str(exc).lower()
    return any(
        token in message
        for token in [
            "connection was closed in the middle of operation",
            "connectiondoesnotexisterror",
            "server closed the connection unexpectedly",
            "connection is closed",
        ]
    )


async def run_with_db_retry(coro_factory, *, operation_name: str):
    """Retry a DB operation once when connection drops transiently."""
    try:
        return await coro_factory()
    except Exception as exc:
        if not is_retryable_db_error(exc):
            raise
        logger.warning(
            "Transient DB error detected; retrying operation once",
            operation=operation_name,
            error=str(exc),
        )
        return await coro_factory()


def summarize_lightrag_counts(by_table: Dict[str, int]) -> Dict[str, int]:
    """Summarize LightRAG deletion counts by type."""
    total = sum(int(v or 0) for v in (by_table or {}).values())
    return {
        "total": total,
        "chunks": int((by_table or {}).get("lightrag_vdb_chunks", 0) or 0),
        "relations": int(
            ((by_table or {}).get("lightrag_vdb_relation", 0) or 0)
            + ((by_table or {}).get("lightrag_vdb_relations", 0) or 0)
        ),
        "entities": int((by_table or {}).get("lightrag_vdb_entity", 0) or 0),
    }


async def delete_table_rows_by_file_path(table_name: str, file_path: str) -> int:
    """Delete rows from a LightRAG table using file_path match variants."""
    path_variants = build_path_variants(file_path)

    async with get_async_session() as session:
        total_deleted = 0
        for path in path_variants:
            try:
                result = await session.execute(
                    text(
                        """
                        DELETE FROM {table_name}
                        WHERE
                            LOWER(REPLACE(file_path, '\\\\', '/')) = LOWER(:path)
                            OR LOWER(file_path) = LOWER(:path)
                            OR file_path ILIKE :path_suffix
                        """
                        .replace("{table_name}", table_name)
                    ),
                    {
                        "path": path,
                        "path_suffix": build_path_suffix(path),
                    },
                )
                total_deleted += int(result.rowcount or 0)
            except Exception:
                continue

    return total_deleted


async def delete_from_lightrag_tables(file_path: str) -> Dict[str, int]:
    """Delete rows from LightRAG PG tables by file_path."""
    deleted_by_table: Dict[str, int] = {}

    for table_name in TABLE_CANDIDATES:
        try:
            deleted_by_table[table_name] = await delete_table_rows_by_file_path(table_name, file_path)
        except Exception:
            deleted_by_table[table_name] = 0

    return deleted_by_table


async def count_lightrag_rows_by_file_path(file_path: str) -> Dict[str, int]:
    """Count remaining LightRAG rows by path markers (post-delete verification)."""
    remaining_by_table: Dict[str, int] = {}
    path_variants = build_path_variants(file_path)

    for table_name in TABLE_CANDIDATES:
        try:
            async with get_async_session() as session:
                total = 0
                for path in path_variants:
                    try:
                        result = await session.execute(
                            text(
                                f"SELECT COUNT(*) FROM {table_name} "
                                "WHERE "
                                "LOWER(REPLACE(file_path, '\\\\', '/')) = LOWER(:path) "
                                "OR LOWER(file_path) = LOWER(:path) "
                                "OR file_path ILIKE :path_suffix"
                            ),
                            {
                                "path": path,
                                "path_suffix": build_path_suffix(path),
                            },
                        )
                        total += int(result.scalar() or 0)
                    except Exception:
                        continue
                remaining_by_table[table_name] = total
        except Exception:
            remaining_by_table[table_name] = 0

    return remaining_by_table


async def execute_neo_delete_query(
    *,
    query: str,
    parameters: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Execute Neo4j write query with lazy connect fallback."""
    neo4j_driver = get_neo4j_driver()
    try:
        return await neo4j_driver.execute_write_query(query, parameters)
    except RuntimeError as exc:
        if "Driver not connected" not in str(exc):
            raise
        await neo4j_driver.connect()
        return await neo4j_driver.execute_write_query(query, parameters)


async def delete_from_graph(
    *,
    file_path: str,
    workspace_labels: Optional[List[str]] = None,
) -> Dict[str, int]:
    """Delete graph rows by file_path with workspace-scoped label fallback."""
    if not settings.database.NEO4J_URI:
        raise RuntimeError("Neo4j is not configured")

    workspace_labels = workspace_labels or []
    base_params: Dict[str, Any] = {
        "file_path": file_path,
    }

    deleted_nodes = 0
    deleted_relations = 0

    async def _run_for_scope(label: Optional[str]) -> Dict[str, int]:
        if label:
            node_query = f"""
            MATCH (n:`{label}`)
            WHERE n.file_path = $file_path
            OR n.source = $file_path
            OR n.source_id = $file_path
            OPTIONAL MATCH (n)-[r]-()
            WITH collect(DISTINCT n) AS nodes, count(DISTINCT r) AS rel_count
            FOREACH (node IN nodes | DETACH DELETE node)
            RETURN size(nodes) AS deleted_count, rel_count AS deleted_relations
            """
        else:
            node_query = """
            MATCH (n)
            WHERE n.file_path = $file_path
            OR n.source = $file_path
            OR n.source_id = $file_path
            OPTIONAL MATCH (n)-[r]-()
            WITH collect(DISTINCT n) AS nodes, count(DISTINCT r) AS rel_count
            FOREACH (node IN nodes | DETACH DELETE node)
            RETURN size(nodes) AS deleted_count, rel_count AS deleted_relations
            """

        node_result = await execute_neo_delete_query(query=node_query, parameters=base_params)

        scope_nodes = int((node_result[0].get("deleted_count", 0) if node_result else 0) or 0)
        scope_relations = int((node_result[0].get("deleted_relations", 0) if node_result else 0) or 0)
        return {"nodes": scope_nodes, "relations": scope_relations}

    # Try label-scoped cleanup first
    for label in workspace_labels:
        scoped = await _run_for_scope(label)
        deleted_nodes += scoped["nodes"]
        deleted_relations += scoped["relations"]

    # Fallback for legacy/unlabeled data
    if deleted_nodes == 0 and deleted_relations == 0:
        unscoped = await _run_for_scope(None)
        deleted_nodes += unscoped["nodes"]
        deleted_relations += unscoped["relations"]

    return {
        "nodes_deleted": deleted_nodes,
        "relations_deleted": deleted_relations,
    }


async def delete_blob(
    *,
    file_path: str,
) -> Dict[str, Any]:
    """Delete blob/object and convert not-found to warning."""
    storage = get_storage_adapter()

    try:
        await storage.delete(file_path)
        return {"storage_deleted": True, "storage_warning": None}
    except Exception as storage_exc:
        if is_not_found_error(storage_exc):
            logger.warning(
                "Blob/object already missing during delete",
                file_path=file_path,
            )
            return {
                "storage_deleted": False,
                "storage_warning": "Blob/object was already missing",
            }
        raise


async def delete_orm_metadata(*, workspace_id: int, file_path: str) -> Dict[str, int]:
    """Delete file-task and document-metadata rows for this workspace and path."""
    async with get_async_session() as session:
        metadata_result = await session.execute(
            delete(DocumentMetadata).where(
                DocumentMetadata.workspace_id == workspace_id,
                DocumentMetadata.file_path == file_path,
            )
        )
        task_result = await session.execute(
            delete(FileTask).where(
                FileTask.workspace_id == workspace_id,
                FileTask.file_path == file_path,
            )
        )

    return {
        "document_metadata_deleted": int(metadata_result.rowcount or 0),
        "file_tasks_deleted": int(task_result.rowcount or 0),
    }


async def delete_orm_metadata_for_workspace(*, workspace_id: int) -> Dict[str, int]:
    """Delete ALL file-task and document-metadata rows for this workspace."""
    async with get_async_session() as session:
        metadata_result = await session.execute(
            delete(DocumentMetadata).where(
                DocumentMetadata.workspace_id == workspace_id,
            )
        )
        task_result = await session.execute(
            delete(FileTask).where(
                FileTask.workspace_id == workspace_id,
            )
        )

    return {
        "document_metadata_deleted": int(metadata_result.rowcount or 0),
        "file_tasks_deleted": int(task_result.rowcount or 0),
    }


async def delete_single_document(
    *,
    workspace_id: int,
    file_path: str,
    file_name: str,
    workspace_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Delete a single document with full cleanup across all systems.

    Returns summary of deletion operations.
    """
    workspace_labels = workspace_labels or []

    # Run blob + PG LightRAG + Neo4j cleanup in parallel
    blob_result, lightrag_result, graph_result = await asyncio.gather(
        delete_blob(file_path=file_path),
        run_with_db_retry(
            lambda: delete_from_lightrag_tables(file_path=file_path),
            operation_name="delete_lightrag_tables",
        ),
        delete_from_graph(file_path=file_path, workspace_labels=workspace_labels),
        return_exceptions=True,
    )

    # Check for exceptions
    if isinstance(blob_result, Exception):
        raise blob_result
    if isinstance(lightrag_result, Exception):
        raise lightrag_result
    if isinstance(graph_result, Exception):
        raise graph_result

    # Delete ORM metadata
    orm_deleted = await run_with_db_retry(
        lambda: delete_orm_metadata(
            workspace_id=workspace_id,
            file_path=file_path,
        ),
        operation_name="delete_orm_metadata",
    )

    # Verify cleanup
    lightrag_remaining = await run_with_db_retry(
        lambda: count_lightrag_rows_by_file_path(file_path=file_path),
        operation_name="verify_lightrag_cleanup",
    )

    lightrag_deleted_summary = summarize_lightrag_counts(lightrag_result)
    lightrag_remaining_summary = summarize_lightrag_counts(lightrag_remaining)

    return {
        "file_path": file_path,
        "file_name": file_name,
        "cleanup": {
            "storage_deleted": bool(blob_result.get("storage_deleted", False)),
            "storage_warning": blob_result.get("storage_warning"),
            "lightrag_deleted": lightrag_deleted_summary,
            "lightrag_remaining": lightrag_remaining_summary,
            "neo4j_deleted": {
                "nodes": int(graph_result.get("nodes_deleted", 0) or 0),
                "relations": int(graph_result.get("relations_deleted", 0) or 0),
            },
            "orm_deleted": {
                "document_metadata": int(orm_deleted.get("document_metadata_deleted", 0) or 0),
                "file_tasks": int(orm_deleted.get("file_tasks_deleted", 0) or 0),
            },
        },
        "status": "deleted",
    }


async def get_all_workspace_documents(workspace_id: int) -> List[Dict[str, Any]]:
    """Fetch all indexed documents for a workspace from file_tasks and document_metadata."""
    async with get_async_session() as session:
        # Get all file tasks for workspace
        file_tasks_stmt = select(FileTask).where(FileTask.workspace_id == workspace_id)
        file_task_rows = (await session.execute(file_tasks_stmt)).scalars().all()

        # Get all document metadata for workspace
        metadata_stmt = select(DocumentMetadata).where(
            DocumentMetadata.workspace_id == workspace_id
        )
        metadata_rows = (await session.execute(metadata_stmt)).scalars().all()

    # Build unique list of file paths
    file_paths = set()
    documents = []

    for task in file_task_rows:
        if task.file_path:
            normalized = normalize_path(task.file_path)
            if normalized and normalized not in file_paths:
                file_paths.add(normalized)
                documents.append({
                    "file_path": normalized,
                    "file_name": task.file_name or normalized.split("/")[-1],
                    "container_name": task.container_name,
                })

    for meta in metadata_rows:
        if meta.file_path:
            normalized = normalize_path(meta.file_path)
            if normalized and normalized not in file_paths:
                file_paths.add(normalized)
                documents.append({
                    "file_path": normalized,
                    "file_name": meta.file_name or normalized.split("/")[-1],
                    "container_name": None,
                })

    return documents


def clear_redis_file_cache(file_id: Optional[str] = None, clear_all: bool = False) -> int:
    """
    Clear Redis file cache entries.

    Args:
        file_id: Specific file_id to clear (optional)
        clear_all: Clear all file cache entries (optional)

    Returns:
        Number of keys cleared
    """
    if not redis_manager.is_available:
        return 0

    try:
        if clear_all:
            keys = redis_manager.scan_keys(f"{FILE_KEY_PREFIX}*")
            if keys:
                return int(redis_manager.delete(*keys) or 0)
        elif file_id:
            redis_manager.delete(f"{FILE_KEY_PREFIX}{file_id}")
            return 1
    except Exception as e:
        logger.warning(f"Failed to clear Redis cache: {e}")

    return 0
