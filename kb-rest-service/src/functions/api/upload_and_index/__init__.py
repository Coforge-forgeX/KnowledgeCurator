"""
Upload and Index API - Optimized Implementation

This endpoint:
1. Validates payload and user permissions
2. Checks workspace access and can_curate_kb permission
3. Uploads files to Azure Blob Storage
4. Creates file_tasks records for tracking
5. Enqueues indexing jobs to Azure Storage Queue
6. Returns immediately with task IDs (non-blocking)
"""

import base64
import hashlib
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy import select

from shared.adapters.storage import get_storage_adapter as _get_storage_adapter
from shared.adapters.queue import get_queue_adapter as _get_queue_adapter

from src.core import (
    AuthorizationException,
    get_logger,
    get_user_id,
    get_workspace_ids,
    require_auth,
)
from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.database import DocumentMetadata, FileTask, Workspace, User, UserMap, get_async_session
from src.helpers.file_validation import get_content_type
from src.helpers.workspace_helpers import get_workspace_storage_paths
from src.helpers.workspace_kb_helpers import get_kb_id_for_upload
from src.shared import (
    create_error_response,
    create_success_response,
    parse_request,
)

from .payloads import FileTaskResponse, UploadAndIndexRequest, UploadAndIndexResponse

logger = get_logger(__name__)


def should_skip_duplicate_check() -> bool:
    """Allow temporary duplicate-check bypass for debugging uploads."""
    from src.core.config import settings

    # Any truthy value enables bypass: true, 1, yes, on
    raw = getattr(settings, "SKIP_DUPLICATE_CHECK", None)
    if raw is None:
        return False
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


# Helper functions to get configured adapters
def get_storage_adapter(container_name: Optional[str] = None):
    """Get storage adapter configured with kb-rest-service settings"""
    from src.core.config import settings

    return _get_storage_adapter(
        provider=settings.storage.STORAGE_PROVIDER or "azure",
        connection_string=settings.storage.AZURE_BLOB_STORAGE_CONNECTION_STRING,
        container_name=container_name or settings.storage.STORAGE_CONTAINER_NAME,
    )


def get_queue_adapter():
    """Get queue adapter configured with kb-rest-service settings"""
    from src.core.config import settings

    queue_provider = settings.active_queue_provider
    return _get_queue_adapter(
        provider=queue_provider,
        connection_string=settings.active_queue_connection,
        queue_name=settings.active_queue_name,
        queue_url=settings.SQS_QUEUE_URL if queue_provider == "aws" else None,
        region_name=settings.AWS_REGION if queue_provider == "aws" else None,
    )


async def get_workspace_context(workspace_id: int) -> Optional[Dict]:
    """
    Get workspace context including namespace and workspace name.
    These are used to derive domain and KB name.

    Returns:
        Dict with workspace_name, namespace, or None if not found
    """
    try:
        async with get_async_session() as session:
            stmt = select(Workspace).where(
                Workspace.workspace_id == workspace_id,
                Workspace.is_active == True
            )
            result = await session.execute(stmt)
            workspace = result.scalar_one_or_none()

            if not workspace:
                logger.warning("Workspace not found", workspace_id=workspace_id)
                return None

            return {
                "workspace_id": workspace.workspace_id,
                "workspace_name": workspace.workspace_name or f"workspace_{workspace_id}",
                "namespace": workspace.namespace or "default",
            }

    except Exception as e:
        logger.error("Failed to get workspace context", error=e, workspace_id=workspace_id)
        return None


async def check_user_permission(
    workspace_id: int, user_id: int
) -> bool:
    """
    Check if user has can_curate_kb permission in workspace.

    Returns:
        True if user has can_curate_kb permission, False otherwise
    """
    try:
        async with get_async_session() as session:
            # Query workspace_users_mapping table for can_curate_kb permission directly
            stmt = select(UserMap.can_curate_kb).where(
                UserMap.workspace_id == workspace_id,
                UserMap.user_id == user_id,
                UserMap.is_active == True
            )

            result = await session.execute(stmt)
            row = result.first()

            if not row:
                logger.warning(
                    "User not found in workspace",
                    user_id=user_id,
                    workspace_id=workspace_id,
                )
                return False

            has_permission = row[0] is True

            logger.info(
                "Permission check",
                user_id=user_id,
                workspace_id=workspace_id,
                can_curate_kb=has_permission,
            )

            return has_permission

    except Exception as e:
        logger.error(
            "Permission check failed",
            error=e,
            user_id=user_id,
            workspace_id=workspace_id,
        )
        return False


async def upload_file_to_storage(
    container: str,
    blob_path: str,
    file_content: Optional[str],
    file_name: str,
    file_bytes: Optional[bytes] = None,
) -> Tuple[bool, Optional[str], Optional[int]]:
    """
    Upload file to cloud storage (platform-agnostic).

    Args:
        container: Container name
        blob_path: Full blob path
        file_content: Base64 encoded file content
        file_name: Original file name

    Returns:
        (success, error_message, file_size_bytes)
    """
    try:
        storage = get_storage_adapter(container_name=container)

        # Decode base64 content only when raw bytes are not provided.
        if file_bytes is None:
            try:
                if not file_content:
                    return False, "Missing file content", None

                # Handle data URL format
                if file_content.startswith("data:"):
                    file_content = file_content.split(",", 1)[1]

                file_bytes = base64.b64decode(file_content)
            except Exception as e:
                return False, f"Invalid base64 content: {str(e)}", None

        # Calculate actual file size from decoded bytes
        file_size_bytes = len(file_bytes)

        # Determine content type from extension
        content_type = get_content_type(file_name)

        # Upload using storage adapter bound to the target container
        await storage.upload(blob_path, file_bytes, content_type)

        logger.info(
            "File uploaded to storage",
            provider=storage.provider_name,
            container=container,
            blob_path=blob_path,
            size_bytes=file_size_bytes
        )

        return True, None, file_size_bytes

    except Exception as e:
        logger.error("Storage upload failed", error=e, container=container, blob_path=blob_path)
        return False, str(e), None


def decode_file_content(file_content: str) -> Tuple[bool, Optional[bytes], Optional[str]]:
    """Decode request file content into raw bytes."""
    try:
        if not file_content:
            return False, None, "Missing file content"

        normalized = file_content.split(",", 1)[1] if file_content.startswith("data:") else file_content
        return True, base64.b64decode(normalized), None
    except Exception as e:
        return False, None, f"Invalid base64 content: {str(e)}"


def compute_content_hash(file_bytes: bytes) -> str:
    """Compute stable SHA-256 hash for duplicate detection."""
    return hashlib.sha256(file_bytes).hexdigest()


async def is_duplicate_document(
    workspace_id: int,
    kb_id: Optional[int],
    content_hash: str,
) -> bool:
    """Check if a document with the same content already exists in the target scope."""
    try:
        async with get_async_session() as session:
            stmt = select(DocumentMetadata.id).where(DocumentMetadata.content_hash == content_hash)

            if kb_id is not None:
                # KG documents are shared by KB, so deduplicate at KB scope.
                stmt = stmt.where(DocumentMetadata.kb_id == kb_id)
            else:
                # Non-KG documents are workspace-local, so deduplicate at workspace scope.
                stmt = stmt.where(
                    DocumentMetadata.workspace_id == workspace_id,
                    DocumentMetadata.kb_id.is_(None),
                )

            result = await session.execute(stmt.limit(1))
            return result.scalar_one_or_none() is not None
    except Exception as e:
        logger.error(
            "Duplicate check failed",
            error=e,
            workspace_id=workspace_id,
            kb_id=kb_id,
        )
        # Fail open so uploads are not blocked by transient read errors.
        return False


async def create_file_task(
    workspace_id: int,
    user_id: int,
    container_name: str,
    upload_path: str,
    domain: str,
    kb_name: str,
    file_name: str,
    file_path: str,
    file_size: Optional[int] = None,
) -> Optional[int]:
    """
    Create file_tasks record for tracking.

    Returns:
        task_id or None on error
    """
    try:
        async with get_async_session() as session:
            # Get user's full name for uploaded_by field using ORM
            stmt = select(User.first_name, User.last_name).where(User.user_id == user_id)
            result = await session.execute(stmt)
            user_row = result.first()

            if user_row:
                first_name = user_row[0] or ""
                last_name = user_row[1] or ""
                full_name = f"{first_name} {last_name}".strip()
                uploaded_by = full_name if full_name else str(user_id)
            else:
                uploaded_by = str(user_id)

            # Format file size with units
            file_size_str = None
            if file_size:
                if file_size < 1024:
                    file_size_str = f"{file_size} Bytes"
                elif file_size < 1024 * 1024:
                    file_size_str = f"{file_size / 1024:.2f} KB"
                elif file_size < 1024 * 1024 * 1024:
                    file_size_str = f"{file_size / (1024 * 1024):.2f} MB"
                else:
                    file_size_str = f"{file_size / (1024 * 1024 * 1024):.2f} GB"

            # Create file task
            file_task = FileTask(
                container_name=container_name,
                upload_path=upload_path,
                domain=domain,
                kb_name=kb_name,
                file_path=file_path,
                workspace_id=workspace_id,
                status="pending",
                file_size=file_size_str,
                uploaded_by=uploaded_by,
                created_at=datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            session.add(file_task)
            await session.flush()  # Get ID without committing

            task_id = file_task.id

            logger.info(
                "File task created",
                task_id=task_id,
                file_name=file_name,
                workspace_id=workspace_id,
            )

            return task_id

    except Exception as e:
        logger.error("Failed to create file task", error=e, file_name=file_name)
        return None


async def enqueue_indexing_job(
    task_id: int,
    workspace_id: int,
    file_path: str,
    domain: str,
    kb_name: str,
    container_name: str,
    kb_id: Optional[int] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Enqueue indexing job to message queue (platform-agnostic).

    Returns:
        (success, error_message)
    """
    try:
        queue = get_queue_adapter()

        # Create job message
        job_message = {
            "job_id": str(task_id),
            "task_id": task_id,
            "workspace_id": workspace_id,
            "file_path": file_path,
            "domain": domain,
            "kb_name": kb_name,
            "container_name": container_name,
            "kb_id": kb_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Send message to queue (platform-agnostic)
        message_id = await queue.send_message(job_message)

        logger.info(
            "Indexing job enqueued",
            provider=queue.provider_name,
            task_id=task_id,
            message_id=message_id,
            queue=queue.queue_name,
            kb_id=kb_id,
        )

        return True, None

    except Exception as e:
        logger.error("Failed to enqueue indexing job", error=e, task_id=task_id)
        return False, str(e)


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """
    Upload and index documents endpoint.

    POST /api/upload-and-index
    Headers:
        Authorization: Bearer <token>
    Body:
        {
            "workspace_id": 1,
            "files": [
                {
                    "file_name": "document.pdf",
                    "file_content": "<base64>",
                    "file_size": 12345
                }
            ]
        }

    Domain, KB name, and container are derived from workspace context.
    Supported file types: .pdf, .docx, .doc, .txt, .md

    Returns:
        202 Accepted: Files queued for indexing
        400 Bad Request: Invalid payload
        403 Forbidden: No permission
        404 Not Found: Workspace not found
        500 Server Error
    """
    # Parse request
    payload, error_response = parse_request(req, UploadAndIndexRequest)
    print("Request parsed")
    if error_response:
        return error_response

    user_id = get_user_id(req)
    user_workspaces = get_workspace_ids(req)
    
    workspace_id = payload.workspace_id

    # Check workspace access
    if workspace_id not in user_workspaces:
        logger.warning(
            "Unauthorized workspace access",
            user_id=user_id,
            workspace_id=workspace_id,
        )
        raise AuthorizationException(
            message="You do not have access to this workspace"
        )

    # Check can_curate_kb permission
    has_permission = await check_user_permission(
        workspace_id=workspace_id, user_id=user_id
    )

    if not has_permission:
        logger.warning(
            "User lacks can_curate_kb permission",
            user_id=user_id,
            workspace_id=workspace_id,
        )
        raise AuthorizationException(
            message="You do not have permission to curate knowledge base in this workspace"
        )
    print("All Auth Success")
    # Get workspace storage paths (container, upload_path, domain, kb_name)
    workspace_paths = await get_workspace_storage_paths(workspace_id)
    print(f"workspace path: {workspace_paths}")
    if not workspace_paths:
        logger.error("Failed to get workspace storage paths", workspace_id=workspace_id)
        return create_error_response(
            message="Failed to retrieve workspace information",
            error_code="WORKSPACE_NOT_FOUND",
            status_code=404,
            correlation_id=context.correlation_id,
        )

    container_name = workspace_paths["container"]
    upload_path_base = workspace_paths["upload_path"]
    domain = workspace_paths["domain"]
    kb_name = workspace_paths["kb_name"]
    is_kg = workspace_paths["is_kg"]
    kb_id_for_upload = await get_kb_id_for_upload(workspace_id)

    if is_kg and kb_id_for_upload is None:
        logger.error(
            "KG workspace has no linked KB for upload",
            workspace_id=workspace_id,
        )
        return create_error_response(
            message="KG workspace must have one linked knowledge base",
            error_code="KB_MAPPING_MISSING",
            status_code=400,
            correlation_id=context.correlation_id,
        )

    logger.info(
        "Processing upload request",
        workspace_id=workspace_id,
        container=container_name,
        upload_path=upload_path_base,
        domain=domain,
        kb_name=kb_name,
        is_kg=is_kg,
        kb_id=kb_id_for_upload,
        file_count=len(payload.files),
    )

    # Process files
    tasks: List[FileTaskResponse] = []
    failed_files: List[str] = []

    for file in payload.files:
        file_name = file.file_name
        blob_path = f"{upload_path_base}/{file_name}"

        try:
            decode_ok, file_bytes, decode_error = decode_file_content(file.file_content)
            if not decode_ok or file_bytes is None:
                failed_files.append(f"{file_name}: {decode_error}")
                continue

            content_hash = compute_content_hash(file_bytes)

            # Skip duplicates before storage upload and queueing unless bypass is enabled.
            if should_skip_duplicate_check():
                logger.warning(
                    "Duplicate check bypassed for debugging",
                    workspace_id=workspace_id,
                    kb_id=kb_id_for_upload,
                    file_name=file_name,
                )
            elif await is_duplicate_document(workspace_id, kb_id_for_upload, content_hash):
                logger.info(
                    "Duplicate document detected - skipping indexing",
                    workspace_id=workspace_id,
                    kb_id=kb_id_for_upload,
                    file_name=file_name,
                )
                failed_files.append(f"{file_name}: Duplicate document already indexed")
                continue

            # 1. Upload to storage (returns calculated file size)
            success, error, file_size_bytes = await upload_file_to_storage(
                container=container_name,
                blob_path=blob_path,
                file_content=None,
                file_name=file_name,
                file_bytes=file_bytes,
            )

            if not success:
                logger.error("Storage upload failed", file_name=file_name, error=error)
                failed_files.append(f"{file_name}: {error}")
                continue

            # 2. Create file task record with calculated size
            task_id = await create_file_task(
                workspace_id=workspace_id,
                user_id=user_id,
                container_name=container_name,
                upload_path=upload_path_base,
                domain=domain,
                kb_name=kb_name,
                file_name=file_name,
                file_path=blob_path,
                file_size=file_size_bytes,
            )

            if not task_id:
                logger.error("Failed to create file task", file_name=file_name)
                failed_files.append(f"{file_name}: Failed to create tracking record")
                continue

            # 3. Enqueue indexing job
            success, error = await enqueue_indexing_job(
                task_id=task_id,
                workspace_id=workspace_id,
                file_path=blob_path,
                domain=domain,
                kb_name=kb_name,
                container_name=container_name,
                kb_id=kb_id_for_upload,
            )

            if not success:
                logger.error("Failed to enqueue job", file_name=file_name, error=error)
                # Update task status to failed
                async with get_async_session() as session:
                    stmt = select(FileTask).where(FileTask.id == task_id)
                    result = await session.execute(stmt)
                    file_task = result.scalar_one_or_none()
                    if file_task:
                        file_task.status = "failed"
                        file_task.error_message = f"Failed to enqueue: {error}"
                        await session.commit()

                failed_files.append(f"{file_name}: Failed to enqueue indexing job")
                continue

            # Success - add to tasks
            tasks.append(
                FileTaskResponse(
                    task_id=task_id,
                    file_name=file_name,
                    file_path=blob_path,
                    status="pending",
                )
            )

        except Exception as e:
            logger.error(
                "File processing error",
                file_name=file_name,
                error=e,
                exc_info=True,
            )
            failed_files.append(f"{file_name}: {str(e)}")

    # Build response
    if not tasks and failed_files:
        # All files failed
        return create_error_response(
            message="All files failed to upload and index",
            error_code="UPLOAD_FAILED",
            details={"failed_files": failed_files},
            status_code=500,
            correlation_id=context.correlation_id,
        )

    response_data = UploadAndIndexResponse(
        success=True,
        message=f"Successfully queued {len(tasks)} file(s) for indexing",
        workspace_id=workspace_id,
        total_files=len(payload.files),
        tasks=tasks,
        failed_files=failed_files,
    )

    return create_success_response(
        message=response_data.message,
        data=response_data.model_dump(),
        status_code=202,  # Accepted
        correlation_id=context.request_id,
    )
