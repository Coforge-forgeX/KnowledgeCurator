"""Queue indexing for all existing files in a workspace's blob path."""

import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sqlalchemy import select

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, require_auth
from src.core.database import FileTask, User, get_async_session
from src.core.logging import get_logger
from src.helpers.file_validation import validate_file_extension
from src.helpers.workspace_helpers import get_workspace_storage_paths
from src.helpers.workspace_kb_helpers import get_kb_id_for_upload
from src.helpers.workspace_permissions import require_workspace_admin_curator
from src.common import create_error_response, create_internal_error_response, create_success_response, parse_request
from src.functions.api.upload_and_index.__init__ import enqueue_indexing_job
from src.storage import get_storage_adapter

from .payloads import (
    IndexedFileTaskResponse,
    IndexWorkspaceFilesRequest,
    IndexWorkspaceFilesResponse,
)

logger = get_logger(__name__)


def _normalize_prefix(prefix: str) -> str:
    cleaned = (prefix or "").strip().strip("/")
    if not cleaned:
        return ""
    return f"{cleaned}/"


async def _list_blob_paths(container_name: str, prefix: str) -> List[str]:
    """List blob paths from configured storage using provider-agnostic adapter."""
    try:
        storage = get_storage_adapter()
        normalized_prefix = _normalize_prefix(prefix)

        # Use storage adapter's list_files method
        blob_paths = await storage.list_files(prefix=normalized_prefix if normalized_prefix else None)

        logger.info(
            "Listed blob paths using storage adapter",
            provider=storage.provider_name,
            container=container_name,
            prefix=normalized_prefix or "(all)",
            count=len(blob_paths),
        )

        return blob_paths

    except Exception as e:
        logger.error(f"Failed to list blob paths: {e}", exc_info=True)
        return []


async def _get_uploader_name(user_id: int) -> str:
    """Resolve display name for file task created-by metadata."""
    async with get_async_session() as session:
        stmt = select(User.first_name, User.last_name).where(User.user_id == user_id)
        result = await session.execute(stmt)
        row = result.first()

    if not row:
        return str(user_id)

    first_name = row[0] or ""
    last_name = row[1] or ""
    full_name = f"{first_name} {last_name}".strip()
    return full_name if full_name else str(user_id)


async def _create_file_task_for_blob(
    workspace_id: int,
    user_display_name: str,
    container_name: str,
    upload_path: str,
    domain: str,
    kb_name: str,
    file_path: str,
) -> Optional[int]:
    """Create a file task row for an already-uploaded blob."""
    try:
        async with get_async_session() as session:
            file_task = FileTask(
                container_name=container_name,
                upload_path=upload_path,
                domain=domain,
                kb_name=kb_name,
                file_path=file_path,
                workspace_id=workspace_id,
                status="queued",
                uploaded_by=user_display_name,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            session.add(file_task)
            await session.flush()
            task_id = file_task.id

            logger.info(
                "File task created for existing blob",
                task_id=task_id,
                workspace_id=workspace_id,
                file_path=file_path,
            )

            return task_id
    except Exception as e:
        logger.error(
            "Failed to create file task for existing blob",
            error=e,
            workspace_id=workspace_id,
            file_path=file_path,
        )
        return None


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """
    Enqueue indexing jobs for all existing files under workspace blob path.

    POST /api/v2/workspaces/index-files
    Body: { "workspace_id": 123 }
    """
    payload, error_response = parse_request(req, IndexWorkspaceFilesRequest)
    if error_response:
        return error_response

    workspace_id = int(payload.workspace_id)
    user_id = get_user_id(req)

    await require_workspace_admin_curator(
        user_id=user_id,
        workspace_id=workspace_id,
        action_description="index workspace files",
    )

    workspace_paths = await get_workspace_storage_paths(workspace_id)
    if not workspace_paths:
        return create_error_response(
            message="Failed to retrieve workspace information",
            error_code="WORKSPACE_NOT_FOUND",
            status_code=404,
            correlation_id=context.correlation_id,
        )

    container_name = str(workspace_paths.get("container") or "")
    upload_path = str(workspace_paths.get("upload_path") or "")
    domain = str(workspace_paths.get("domain") or "")
    kb_name = str(workspace_paths.get("kb_name") or "")
    is_kg = bool(workspace_paths.get("is_kg"))
    kb_id_for_upload = await get_kb_id_for_upload(workspace_id)

    if is_kg and kb_id_for_upload is None:
        return create_error_response(
            message="KG workspace must have one linked knowledge base",
            error_code="KB_MAPPING_MISSING",
            status_code=400,
            correlation_id=context.correlation_id,
        )

    try:
        blob_paths = await _list_blob_paths(container_name=container_name, prefix=upload_path)
    except Exception as e:
        logger.error(
            "Failed to list workspace blobs",
            error=e,
            workspace_id=workspace_id,
            container=container_name,
            prefix=upload_path,
        )
        return create_internal_error_response(
            message="Failed to list workspace files from storage",
            error=e,
            error_code="BLOB_LIST_FAILED",
            correlation_id=context.correlation_id,
        )

    if not blob_paths:
        response_data = IndexWorkspaceFilesResponse(
            success=True,
            message="No files found to index",
            workspace_id=workspace_id,
            total_blobs_scanned=0,
            queued_files=0,
            tasks=[],
            failed_files=[],
            skipped_files=[],
            kb_id=kb_id_for_upload,
        )
        return create_success_response(
            message=response_data.message,
            data=response_data.model_dump(),
            status_code=200,
            correlation_id=context.correlation_id,
        )

    user_display_name = await _get_uploader_name(user_id)

    queued_tasks: List[IndexedFileTaskResponse] = []
    failed_files: List[str] = []
    skipped_files: List[str] = []

    for file_path in blob_paths:
        file_name = os.path.basename(file_path)
        if not file_name:
            skipped_files.append(file_path)
            continue

        try:
            validate_file_extension(file_name)
        except Exception:
            skipped_files.append(f"{file_name}: unsupported file type")
            continue

        task_id = await _create_file_task_for_blob(
            workspace_id=workspace_id,
            user_display_name=user_display_name,
            container_name=container_name,
            upload_path=upload_path,
            domain=domain,
            kb_name=kb_name,
            file_path=file_path,
        )

        if not task_id:
            failed_files.append(f"{file_name}: failed to create task")
            continue

        success, enqueue_error = await enqueue_indexing_job(
            task_id=task_id,
            workspace_id=workspace_id,
            user_id=user_id,
            file_path=file_path,
            domain=domain,
            kb_name=kb_name,
            container_name=container_name,
            kb_id=kb_id_for_upload,
        )

        if not success:
            failed_files.append(f"{file_name}: {enqueue_error}")
            continue

        queued_tasks.append(
            IndexedFileTaskResponse(
                task_id=task_id,
                file_name=file_name,
                file_path=file_path,
                status="queued",
            )
        )

    message = f"Successfully queued {len(queued_tasks)} file(s) for indexing"
    if not queued_tasks and failed_files:
        return create_error_response(
            message="Failed to queue workspace files for indexing",
            error_code="QUEUE_FAILED",
            details={
                "failed_files": failed_files,
                "skipped_files": skipped_files,
            },
            status_code=500,
            correlation_id=context.correlation_id,
        )

    response_data = IndexWorkspaceFilesResponse(
        success=True,
        message=message,
        workspace_id=workspace_id,
        total_blobs_scanned=len(blob_paths),
        queued_files=len(queued_tasks),
        tasks=queued_tasks,
        failed_files=failed_files,
        skipped_files=skipped_files,
        kb_id=kb_id_for_upload,
    )

    return create_success_response(
        message=response_data.message,
        data=response_data.model_dump(),
        status_code=202,
        correlation_id=context.correlation_id,
    )