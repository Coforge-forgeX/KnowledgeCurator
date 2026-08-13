"""
Delete indexed files by opaque file_id tokens or direct file paths.

Refactored to use shared deletion utilities from src.helpers.deletion.
"""
import json
from typing import Any, Dict, List

from sqlalchemy import select

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, require_auth
from src.core.config import settings
from src.core.database import FileTask, get_async_session
from src.core.exceptions import AuthorizationException, ValidationException
from src.core.logging import get_logger
from src.core.redis import redis_manager
from src.helpers.file_token import decode_signed_file_id
from src.helpers.workspace_permissions import require_workspace_admin_curator
from src.helpers.workspace_helpers import get_workspace_storage_paths
from src.helpers.deletion import (
    clear_redis_file_cache,
    delete_single_document,
    normalize_path,
    run_with_db_retry,
)
from src.common import create_error_response, create_internal_error_response, create_success_response, parse_request

from .payloads import DeleteFilesByIdRequest

logger = get_logger(__name__)

FILE_KEY_PREFIX = "query_file:"


async def _validate_curate_permission(user_id: int, workspace_id: int) -> None:
    """Require user to be admin in workspace and can_curate_kb=True."""
    await require_workspace_admin_curator(
        user_id=user_id,
        workspace_id=workspace_id,
        action_description="delete files",
    )


async def _load_token_mapping(file_id: str) -> Dict[str, Any]:
    """Load file mapping from signed token or Redis cache."""
    signed_mapping = decode_signed_file_id(file_id)
    if signed_mapping:
        return signed_mapping

    mapping_str = redis_manager.get(f"{FILE_KEY_PREFIX}{file_id}")
    if not mapping_str:
        raise ValidationException(message=f"file_id '{file_id}' is invalid or expired")

    try:
        mapping = json.loads(mapping_str)
    except Exception as exc:
        raise ValidationException(message=f"file_id '{file_id}' mapping is invalid") from exc

    return mapping if isinstance(mapping, dict) else {}


async def _get_index_presence(
    *,
    workspace_id: int,
    file_path: str,
) -> Dict[str, bool]:
    """Check whether a file is indexed in this workspace and/or another workspace."""
    async with get_async_session() as session:
        from src.core.database import DocumentMetadata

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
        file_path = normalize_path(mapping.get("blob_path"))
        file_name = str(mapping.get("file_name") or "").strip()
        provider = str(mapping.get("provider") or settings.storage.STORAGE_PROVIDER)
        container_name = str(mapping.get("container_name") or default_container)
        mapping_workspace_id = mapping.get("workspace_id")

        if mapping_workspace_id is not None and int(mapping_workspace_id) != workspace_id:
            raise ValidationException(message="file_id is invalid for this workspace")
    else:
        file_path = normalize_path(target.get("file_path"))
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

    index_presence = await run_with_db_retry(
        lambda: _get_index_presence(
            workspace_id=workspace_id,
            file_path=file_path,
        ),
        operation_name="validate_file_path_index_presence",
    )
    in_workspace = bool(index_presence.get("in_workspace"))
    if not in_workspace:
        raise ValidationException(message="file_path is not indexed in this workspace")


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
            targets.append({"source_type": "file_path", "file_path": normalize_path(file_path)})

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

                await _validate_target_indexed_for_workspace(
                    source_type=source_type,
                    workspace_id=workspace_id,
                    file_path=file_path,
                )

                # Use shared deletion utility for the actual deletion
                result = await delete_single_document(
                    workspace_id=workspace_id,
                    file_path=file_path,
                    file_name=file_name,
                    workspace_labels=[],
                )

                # Invalidate token mapping after delete for file_id requests
                if source_type == "file_id":
                    clear_redis_file_cache(file_id=file_id)

                # Add source_type and file_id to result
                result["source_type"] = source_type
                result["file_id"] = file_id
                deleted.append(result)

            except Exception as e:
                failed.append(
                    {
                        "source_type": str(target.get("source_type") or "file_id"),
                        "file_id": str(target.get("file_id") or "") or None,
                        "file_path": normalize_path(locals().get("file_path") or target.get("file_path")) or None,
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
