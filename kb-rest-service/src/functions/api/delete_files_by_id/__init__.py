"""Delete indexed files by opaque file_id tokens or direct file paths."""
import asyncio
from typing import Any, Dict, List, Optional

from sqlalchemy import delete, select, text

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, require_auth
from src.core.config import settings
from src.core.database import DocumentMetadata, FileTask, get_async_session
from src.core.exceptions import AuthorizationException, ValidationException
from src.core.logging import get_logger
from src.core.neo4j_driver import get_neo4j_driver
from src.core.redis import redis_manager
from src.helpers.file_token import decode_signed_file_id
from src.helpers.workspace_permissions import require_workspace_admin_curator
from src.helpers.workspace_helpers import get_workspace_storage_paths
from src.common import create_error_response, create_internal_error_response, create_success_response, parse_request
from src.storage import get_storage_adapter

from .payloads import DeleteFilesByIdRequest

logger = get_logger(__name__)

FILE_KEY_PREFIX = "query_file:"
TABLE_CANDIDATES = [
    "lightrag_vdb_chunks",
    "lightrag_vdb_relation",
    "lightrag_vdb_entity",
]


def _build_path_variants(file_path: str) -> List[str]:
    normalized_file_path = _normalize_path(file_path)
    variants: List[str] = [normalized_file_path]

    if normalized_file_path.startswith("/"):
        variants.append(normalized_file_path.lstrip("/"))
    else:
        variants.append(f"/{normalized_file_path}")
    return list(dict.fromkeys([v for v in variants if v]))


def _build_path_suffix(path: str) -> str:
    return f"%{path}"


async def _delete_table_rows_by_file_path(table_name: str, file_path: str) -> int:
    """Delete rows from a LightRAG table using file_path match variants only."""
    path_variants = _build_path_variants(file_path)

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
                        "path_suffix": _build_path_suffix(path),
                    },
                )
                total_deleted += int(result.rowcount or 0)
            except Exception:
                # Table/schema can vary by deployment.
                continue

    return total_deleted


async def _validate_curate_permission(user_id: int, workspace_id: int) -> None:
    """Require user to be admin in workspace and can_curate_kb=True."""
    await require_workspace_admin_curator(
        user_id=user_id,
        workspace_id=workspace_id,
        action_description="delete files",
    )


async def _load_token_mapping(file_id: str) -> Dict[str, Any]:
    signed_mapping = decode_signed_file_id(file_id)
    if signed_mapping:
        return signed_mapping

    mapping_str = redis_manager.get(f"{FILE_KEY_PREFIX}{file_id}")
    if not mapping_str:
        raise ValidationException(message=f"file_id '{file_id}' is invalid or expired")

    import json

    try:
        mapping = json.loads(mapping_str)
    except Exception as exc:
        raise ValidationException(message=f"file_id '{file_id}' mapping is invalid") from exc

    return mapping if isinstance(mapping, dict) else {}


def _normalize_path(path: str) -> str:
    """Normalize file path for consistent matching across stores."""
    return str(path or "").strip().replace("\\", "/")


async def _delete_from_lightrag_tables_by_file_path(file_path: str) -> Dict[str, int]:
    """Delete rows from LightRAG PG tables by file_path."""
    deleted_by_table: Dict[str, int] = {}

    for table_name in TABLE_CANDIDATES:
        try:
            deleted_by_table[table_name] = await _delete_table_rows_by_file_path(table_name, file_path)
        except Exception:
            # Table can be missing or inaccessible depending on deployment mode.
            deleted_by_table[table_name] = 0

    return deleted_by_table


async def _count_lightrag_rows_by_file_path(file_path: str) -> Dict[str, int]:
    """Count remaining LightRAG rows by path markers (post-delete verification)."""
    remaining_by_table: Dict[str, int] = {}
    path_variants = _build_path_variants(file_path)

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
                                "path_suffix": _build_path_suffix(path),
                            },
                        )
                        total += int(result.scalar() or 0)
                    except Exception:
                        continue
                remaining_by_table[table_name] = total
        except Exception:
            remaining_by_table[table_name] = 0

    return remaining_by_table


def _summarize_lightrag_counts(by_table: Dict[str, int]) -> Dict[str, int]:
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


async def _resolve_target(
    *,
    target: Dict[str, Any],
    workspace_id: int,
    default_container: str,
) -> Dict[str, Any]:
    """Normalize and resolve delete target for either file_id or file_path."""
    source_type = str(target.get("source_type") or "file_id")
    file_id = str(target.get("file_id") or "")

    mapping: Dict[str, Any] = {}
    if source_type == "file_id":
        mapping = await _load_token_mapping(file_id)
        file_path = _normalize_path(mapping.get("blob_path"))
        file_name = str(mapping.get("file_name") or "").strip()
        provider = str(mapping.get("provider") or settings.storage.STORAGE_PROVIDER or "azure")
        container_name = str(mapping.get("container_name") or default_container)
        mapping_workspace_id = mapping.get("workspace_id")

        if mapping_workspace_id is not None and int(mapping_workspace_id) != workspace_id:
            raise ValidationException(message="file_id is invalid for this workspace")
    else:
        file_path = _normalize_path(target.get("file_path"))
        file_name = str(target.get("file_name") or "").strip() or file_path.split("/")[-1]
        provider = str(settings.storage.STORAGE_PROVIDER or "azure")
        container_name = str(default_container)
        file_id = file_id or file_path

    if not file_path:
        if source_type == "file_id":
            raise ValidationException(message=f"file_id '{file_id}' has no file path")
        raise ValidationException(message="file_path is required")

    return {
        "source_type": source_type,
        "file_id": file_id,
        "file_name": file_name,
        "file_path": file_path,
        "provider": provider,
        "container_name": container_name,
    }


async def _validate_target_indexed_for_workspace(*, source_type: str, workspace_id: int, file_path: str) -> None:
    """Validate file_path indexing for file_path-based deletes."""
    if source_type != "file_path":
        return

    index_presence = await _run_with_db_retry(
        lambda: _get_index_presence(
            workspace_id=workspace_id,
            file_path=file_path,
        ),
        operation_name="validate_file_path_index_presence",
    )
    in_workspace = bool(index_presence.get("in_workspace"))
    if not in_workspace:
        raise ValidationException(message="file_path is not indexed in this workspace")


async def _execute_neo_delete_query(
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


async def _delete_from_graph(
    *,
    file_path: str,
    workspace_labels: List[str],
) -> Dict[str, int]:
    """Delete graph rows by file_path with workspace-scoped label fallback."""
    if not settings.database.NEO4J_URI:
        raise RuntimeError("Neo4j is not configured (missing NEO4J_DATABASE_NEO4J_BOLT_URI)")

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

        node_result = await _execute_neo_delete_query(query=node_query, parameters=base_params)

        scope_nodes = int((node_result[0].get("deleted_count", 0) if node_result else 0) or 0)
        scope_relations = int((node_result[0].get("deleted_relations", 0) if node_result else 0) or 0)
        return {"nodes": scope_nodes, "relations": scope_relations}

    # First try label-scoped cleanup to avoid expensive global scans.
    for label in workspace_labels:
        scoped = await _run_for_scope(label)
        deleted_nodes += scoped["nodes"]
        deleted_relations += scoped["relations"]

    # Fallback for legacy/unlabeled data.
    if deleted_nodes == 0 and deleted_relations == 0:
        unscoped = await _run_for_scope(None)
        deleted_nodes += unscoped["nodes"]
        deleted_relations += unscoped["relations"]

    return {
        "nodes_deleted": deleted_nodes,
        "relations_deleted": deleted_relations,
    }


def _is_not_found_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(token in message for token in ["not found", "does not exist", "404", "blobnotfound"])


def _is_retryable_db_error(exc: Exception) -> bool:
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


async def _run_with_db_retry(coro_factory, *, operation_name: str):
    """Retry a DB operation once when connection drops transiently."""
    try:
        return await coro_factory()
    except Exception as exc:
        if not _is_retryable_db_error(exc):
            raise
        logger.warning(
            "Transient DB error detected; retrying operation once",
            operation=operation_name,
            error=str(exc),
        )
        return await coro_factory()


async def _get_index_presence(
    *,
    workspace_id: int,
    file_path: str,
) -> Dict[str, bool]:
    """Check whether a file is indexed in this workspace and/or another workspace."""
    async with get_async_session() as session:
        task_in_workspace = await session.execute(
            select(FileTask.id).where(
                FileTask.workspace_id == workspace_id,
                FileTask.file_path == file_path,
            )
        )
        metadata_in_workspace = await session.execute(
            select(DocumentMetadata.id).where(
                DocumentMetadata.workspace_id == workspace_id,
                DocumentMetadata.file_path == file_path,
            )
        )
        task_in_other_workspace = await session.execute(
            select(FileTask.id).where(
                FileTask.workspace_id != workspace_id,
                FileTask.file_path == file_path,
            )
        )
        metadata_in_other_workspace = await session.execute(
            select(DocumentMetadata.id).where(
                DocumentMetadata.workspace_id != workspace_id,
                DocumentMetadata.file_path == file_path,
            )
        )

    in_workspace = bool(
        task_in_workspace.scalar_one_or_none() or metadata_in_workspace.scalar_one_or_none()
    )
    in_other_workspace = bool(
        task_in_other_workspace.scalar_one_or_none() or metadata_in_other_workspace.scalar_one_or_none()
    )
    return {"in_workspace": in_workspace, "in_other_workspace": in_other_workspace}


async def _delete_blob(
    *,
    provider: str,
    container_name: str,
    file_path: str,
) -> Dict[str, Any]:
    """Delete blob/object and convert not-found to warning."""
    # Use service's provider-agnostic storage adapter
    storage = get_storage_adapter()

    try:
        await storage.delete(file_path)
        return {"storage_deleted": True, "storage_warning": None}
    except Exception as storage_exc:
        if _is_not_found_error(storage_exc):
            logger.warning(
                "Blob/object already missing during delete",
                file_path=file_path,
            )
            return {
                "storage_deleted": False,
                "storage_warning": "Blob/object was already missing",
            }
        raise


async def _delete_orm_metadata(*, workspace_id: int, file_path: str) -> Dict[str, int]:
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


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """Delete files by file_id or file_path with workspace authorization checks."""
    correlation_id = context.correlation_id
    user_id = get_user_id(req)

    payload, error_response = parse_request(req, DeleteFilesByIdRequest)
    if error_response:
        return error_response

    workspace_id = int(payload.workspace_id)

    try:
        await _validate_curate_permission(user_id, workspace_id)

        storage_paths = await get_workspace_storage_paths(workspace_id)
        default_container = str((storage_paths or {}).get("container") or settings.storage.STORAGE_CONTAINER_NAME or "")

        deleted: List[Dict[str, Any]] = []
        failed: List[Dict[str, Any]] = []

        targets: List[Dict[str, Any]] = []
        for file_id in (payload.file_id or []):
            targets.append({"source_type": "file_id", "file_id": file_id})
        for file_path in (payload.file_path or []):
            targets.append({"source_type": "file_path", "file_path": _normalize_path(file_path)})

        for target in targets:
            try:
                resolved = await _resolve_target(
                    target=target,
                    workspace_id=workspace_id,
                    default_container=default_container,
                )

                source_type = str(resolved["source_type"])
                file_id = str(resolved["file_id"])
                file_name = str(resolved["file_name"])
                file_path = str(resolved["file_path"])
                provider = str(resolved["provider"])
                container_name = str(resolved["container_name"])

                await _validate_target_indexed_for_workspace(
                    source_type=source_type,
                    workspace_id=workspace_id,
                    file_path=file_path,
                )

                # Run blob + PG LightRAG + Neo4j cleanup in parallel.
                blob_result, lightrag_result, graph_result = await asyncio.gather(
                    _delete_blob(
                        provider=provider,
                        container_name=container_name,
                        file_path=file_path,
                    ),
                    _run_with_db_retry(
                        lambda: _delete_from_lightrag_tables_by_file_path(file_path=file_path),
                        operation_name="delete_lightrag_tables_by_file_path",
                    ),
                    _delete_from_graph(file_path=file_path, workspace_labels=[]),
                    return_exceptions=True,
                )

                if isinstance(blob_result, Exception):
                    raise blob_result
                if isinstance(lightrag_result, Exception):
                    raise lightrag_result
                if isinstance(graph_result, Exception):
                    raise graph_result

                orm_deleted = await _run_with_db_retry(
                    lambda: _delete_orm_metadata(
                        workspace_id=workspace_id,
                        file_path=file_path,
                    ),
                    operation_name="delete_orm_metadata",
                )

                lightrag_remaining = await _run_with_db_retry(
                    lambda: _count_lightrag_rows_by_file_path(file_path=file_path),
                    operation_name="verify_lightrag_cleanup",
                )

                # Invalidate token mapping after delete for file_id requests.
                if source_type == "file_id":
                    redis_manager.delete(f"{FILE_KEY_PREFIX}{file_id}")

                lightrag_deleted_summary = _summarize_lightrag_counts(lightrag_result)
                lightrag_remaining_summary = _summarize_lightrag_counts(lightrag_remaining)

                deleted.append(
                    {
                        "source_type": source_type,
                        "file_id": file_id,
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
                )

            except Exception as e:
                failed.append(
                    {
                        "source_type": str(target.get("source_type") or "file_id"),
                        "file_id": str(target.get("file_id") or "") or None,
                        "file_path": _normalize_path(locals().get("file_path") or target.get("file_path")) or None,
                        "error": str(e),
                    }
                )

        requested_count = len(targets)
        deleted_count = len(deleted)
        failed_count = len(failed)

        response_data = {
            "workspace_id": workspace_id,
            "requested": requested_count,
            "deleted_count": deleted_count,
            "failed_count": failed_count,
            "deleted": deleted,
            "failed": failed,
        }

        # If every requested file failed, the overall operation should be reported as failed.
        if requested_count > 0 and deleted_count == 0 and failed_count > 0:
            return create_error_response(
                message="File delete operation failed",
                error_code="DELETE_FILES_BY_ID_FAILED",
                details=response_data,
                status_code=400,
                correlation_id=correlation_id,
            )

        if failed_count > 0:
            return create_success_response(
                message="File delete operation completed with warnings",
                data=response_data,
                status_code=207,
                correlation_id=correlation_id,
            )

        return create_success_response(
            message="File delete operation completed",
            data=response_data,
            correlation_id=correlation_id,
        )

    except AuthorizationException as e:
        return create_error_response(
            message=e.message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
            correlation_id=correlation_id,
        )
    except ValidationException as e:
        return create_error_response(
            message=e.message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            correlation_id=correlation_id,
        )
    except Exception as e:
        logger.error("Delete by file_id failed", error=e, workspace_id=workspace_id)
        return create_internal_error_response(
            message="Failed to delete files",
            error=e,
            error_code="DELETE_FILES_BY_ID_FAILED",
            correlation_id=correlation_id,
        )
