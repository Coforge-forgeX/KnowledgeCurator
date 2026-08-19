"""
RAG Service - Extended Knowledge Base Operations

Handles RAG queries, document indexing, blob storage, and workspace management.
Mirrors functionality from KnowledgeCurator MCP tools.
"""
import asyncio
import base64
import os
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select

from src.core.config import settings
from src.core.database import DocumentMetadata, FileTask, User, get_async_session
from src.core.exceptions import LightRAGException
from src.helpers.file_validation import get_content_type
from src.core.logging import get_logger
from src.functions.api.upload_and_index.payloads import FileUpload
from src.helpers.queue_helpers import get_indexing_queue_helper
from src.storage import get_storage_adapter

from src.functions.api.upload_and_index.payloads import FileTaskResponse

logger = get_logger(__name__)


class RAGService:
    """
    Extended RAG service for knowledge base operations.

    Provides comprehensive document management including:
    - RAG query execution
    - Document upload and indexing
    - Workspace indexing management
    - Blob storage operations
    - Indexing status tracking
    """

    def __init__(self):
        self._queue_helper = None  # Lazy-loaded only when needed
        self._storage = None  # Lazy-loaded storage adapter

    @property
    def queue_helper(self):
        """Lazy-load queue helper only when needed (for indexing operations)"""
        if self._queue_helper is None:
            self._queue_helper = get_indexing_queue_helper()
        return self._queue_helper

    @property
    def storage(self):
        """Lazy-load storage adapter only when needed"""
        if self._storage is None:
            self._storage = get_storage_adapter()
        return self._storage

    # ========================================================================
    # Document Upload & Indexing
    # ========================================================================

    async def upload_and_index_tool(
        self,
        files: FileUpload,
        workspace_id: int,
        user_id: int,
        upload_path: str,
        container_name: str,
        kb_ids: List[int],
        is_kg: bool,
        domain: Optional[str] = None,
        kb_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload documents to blob storage and queue for indexing.

        Uses background tasks for non-blocking upload and indexing.
        Returns task IDs immediately.

        Args:
            files: List of files containing:
                file_name: List of file names
                file_content: List of file contents (base64 encoded)
            workspace_id: Workspace identifier
            user_id: User identifier
            domain: Optional domain/industry name
            kb_name: Optional knowledge base name

        Returns:
            Dict with task_ids and uploaded file paths
        """

        # Process files
        tasks: List[FileTaskResponse] = []
        failed_files: List[str] = []

        # Get first kb_id for duplicate checking
        kb_id_for_upload = None

        if is_kg and kb_ids:
            kb_id_for_upload = kb_ids[0]

        for file in files:
            file_name = file.file_name
            file_content = file.file_content
            try:
                decode_ok, file_bytes, decode_error = self.decode_file_content(file_content)
                if not decode_ok or file_bytes is None:
                    failed_files.append(f"{file_name}: {decode_error}")
                    continue

                content_hash = self.compute_content_hash(file_bytes)
                file_size_bytes = len(file_bytes)

                # Skip duplicates before storage upload and queueing unless bypass is enabled.
                if self.should_skip_duplicate_check():
                    logger.warning(
                        "Duplicate check bypassed for debugging",
                        workspace_id=workspace_id,
                        kb_id=kb_id_for_upload,
                        file_name=file_name,
                    )
                elif await self.is_duplicate_document(workspace_id, kb_id_for_upload, content_hash):
                    logger.info(
                        "Duplicate document detected - skipping indexing",
                        workspace_id=workspace_id,
                        kb_id=kb_id_for_upload,
                        file_name=file_name,
                    )
                    failed_files.append(f"{file_name}: Duplicate document already indexed")
                    continue

                # Construct blob path
                full_blob_path = f"{upload_path}/{file_name}"

                # 1. Create file task record first (status: pending)
                task_id = await self._create_file_task(
                    workspace_id=workspace_id,
                    user_id=user_id,
                    container_name=container_name,
                    upload_path=upload_path,
                    domain=domain or "",
                    kb_name=kb_name or "",
                    file_name=file_name,
                    file_path=full_blob_path,
                    file_size=file_size_bytes,
                )

                if not task_id:
                    logger.error("Failed to create file task", file_name=file_name)
                    failed_files.append(f"{file_name}: Failed to create tracking record")
                    continue

                # 2. Start background upload and enqueue (non-blocking)
                asyncio.create_task(
                    self._background_upload_and_index(
                        task_id=task_id,
                        container_name=container_name,
                        blob_path=full_blob_path,
                        file_bytes=file_bytes,
                        file_name=file_name,
                        workspace_id=workspace_id,
                        user_id=user_id,
                        domain=domain or "",
                        kb_name=kb_name or "",
                        kb_id=kb_id_for_upload,
                    )
                )

                logger.info(
                    "Background upload task started",
                    task_id=task_id,
                    file_name=file_name,
                )

                # 3. Add to tasks immediately (upload happens in background)
                tasks.append(
                    FileTaskResponse(
                        task_id=task_id,
                        file_name=file_name,
                        file_path=full_blob_path,
                        status="pending",  # Will change to: uploading -> queued -> processing -> indexed
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
        task_ids = [t.task_id for t in tasks] if tasks else []

        return {
            "response": f"Successfully uploaded {len(tasks)} file(s)",
            "task_ids": task_ids,
            "uploaded_files": [t.file_path for t in tasks],
            "failed_files": failed_files,
        }

    async def _create_file_task(
        self,
        workspace_id: int,
        user_id: int,
        container_name: str,
        upload_path: str,
        domain: str,
        kb_name: str,
        file_name: str,
        file_path: str,
        file_size: int,
    ) -> Optional[int]:
        """
        Create a file task record for tracking upload/indexing progress.
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

    async def _background_upload_and_index(
        self,
        task_id: int,
        container_name: str,
        blob_path: str,
        file_bytes: bytes,
        file_name: str,
        workspace_id: int,
        user_id: int,
        domain: Optional[str],
        kb_name: Optional[str],
        kb_id: Optional[int],
    ) -> None:
        """
        Background task to upload file and enqueue indexing job.

        Runs asynchronously without blocking. Updates file_task status:
        - pending -> uploading -> queued -> (worker picks up) -> processing -> indexed
        """
        try:
            # Update status to uploading
            await self.update_file_task_status_direct(task_id, "uploading")

            content_type = get_content_type(file_name)

            # Upload to storage using adapter with workspace-specific container override
            storage = get_storage_adapter(container_override=container_name)
            try:
                await storage.upload(
                    filename=blob_path,
                    data=file_bytes,
                    content_type=content_type,
                )
            except Exception as upload_error:
                logger.error("Background upload failed", file_name=file_name, error=str(upload_error))
                await self.update_file_task_status_direct(task_id, "failed", str(upload_error))
                return

            logger.info("Background upload completed", task_id=task_id, file_name=file_name)

            # Update status to queued (upload complete, ready for indexing)
            await self.update_file_task_status_direct(task_id, "queued")

            # Enqueue indexing job
            await self._queue_document_for_indexing(
                task_id=task_id,
                workspace_id=workspace_id,
                file_path=blob_path,
                file_name=file_name,
                user_id=user_id,
                domain=domain,
                kb_name=kb_name,
                kb_id=kb_id,
                container_name=container_name,
            )

            logger.info("Background upload and enqueue completed", task_id=task_id)

        except Exception as e:
            logger.error(
                "Background upload task failed",
                task_id=task_id,
                error=str(e),
                exc_info=True,
            )
            try:
                await self.update_file_task_status_direct(task_id, "failed", str(e))
            except Exception as update_error:
                logger.error("Failed to update task status after error", error=update_error)

    async def update_file_task_status_direct(
        self,
        task_id: int,
        status: str,
        error_message: Optional[str] = None,
    ) -> None:
        """
        Update file_task status directly (for background tasks).
        """
        try:
            async with get_async_session() as session:
                stmt = select(FileTask).where(FileTask.id == task_id)
                result = await session.execute(stmt)
                file_task = result.scalar_one_or_none()

                if file_task:
                    file_task.status = status
                    if error_message:
                        if hasattr(file_task, "error_message"):
                            file_task.error_message = error_message
                    file_task.updated_at = datetime.now(timezone.utc)
                    await session.commit()

                    logger.debug("Task status updated", task_id=task_id, status=status)
        except Exception as e:
            logger.error("Failed to update task status", task_id=task_id, error=e)

        except Exception as e:
            logger.error("Upload and index failed", error=e)
            raise LightRAGException(
                message=f"Failed to upload and index documents: {str(e)}",
                operation="upload_and_index"
            )



    @staticmethod
    def decode_file_content(
        file_content: str,
    ) -> Tuple[bool, Optional[bytes], Optional[str]]:
        """Decode request file content into raw bytes."""

        if not file_content:
            return False, None, "Missing file content"

        import binascii

        try:
            if file_content.startswith("data:"):
                parts = file_content.split(",", 1)
                if len(parts) != 2:
                    return False, None, "Invalid data URL format"
                normalized = parts[1]
            else:
                normalized = file_content

            decoded = base64.b64decode(normalized, validate=True)
            return True, decoded, None

        except (binascii.Error, ValueError) as e:
            return False, None, f"Invalid base64 content: {str(e)}"

    @staticmethod
    def compute_content_hash(file_bytes: bytes) -> str:
        """Compute stable SHA-256 hash for duplicate detection."""
        return hashlib.sha256(file_bytes).hexdigest()


    @staticmethod
    def should_skip_duplicate_check() -> bool:
        """Allow temporary duplicate-check bypass for debugging uploads."""

        # Any truthy value enables bypass: true, 1, yes, on
        raw = settings.SKIP_DUPLICATE_CHECK
        if raw is None:
            return False
        return str(raw).strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
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


    async def start_workspace_indexing(
        self,
        workspace_id: int,
        user_id: int,
        role_id: int,
    ) -> Dict[str, Any]:
        """
        Start indexing all documents in a workspace.

        Args:
            workspace_id: Workspace identifier
            user_id: User identifier
            role_id: Role identifier

        Returns:
            Dict with status and task count
        """
        try:
            logger.info(
                "Starting workspace indexing",
                workspace_id=workspace_id,
            )

            # Get all pending/failed tasks for this workspace
            async with get_async_session() as session:
                stmt = select(FileTask).where(
                    FileTask.workspace_id == workspace_id,
                    FileTask.status.in_(["pending", "failed"])
                )
                result = await session.execute(stmt)
                tasks = result.scalars().all()

            if not tasks:
                return {
                    "status": "no_pending_tasks",
                    "message": "No pending tasks found for this workspace",
                    "task_count": 0,
                }

            # Queue all tasks for indexing
            queued_count = 0
            for task in tasks:
                try:
                    # Extract file_name from file_path (file_name removed from FileTask)
                    file_name = os.path.basename(task.file_path) if task.file_path else None

                    await self._queue_document_for_indexing(
                        task_id=task.id,
                        workspace_id=workspace_id,
                        file_path=task.file_path,
                        file_name=file_name,
                    )
                    queued_count += 1
                except Exception as e:
                    logger.warning(
                        "Failed to queue task",
                        error=e,
                        task_id=task.id,
                    )

            logger.info(
                "Workspace indexing started",
                workspace_id=workspace_id,
                queued_count=queued_count,
            )

            return {
                "status": "indexing_started",
                "message": f"Queued {queued_count} document(s) for indexing",
                "task_count": queued_count,
            }

        except Exception as e:
            logger.error("Failed to start workspace indexing", error=e)
            raise LightRAGException(
                message=f"Failed to start workspace indexing: {str(e)}",
                operation="start_workspace_indexing"
            )

    # ========================================================================
    # Helper Methods
    # ========================================================================

    async def _queue_document_for_indexing(
        self,
        task_id: int,
        workspace_id: int,
        file_path: str,
        file_name: str,
        user_id: Optional[int] = None,
        domain: Optional[str] = None,
        kb_name: Optional[str] = None,
        kb_id: Optional[int] = None,
        container_name: Optional[str] = None,
    ) -> None:
        """Queue a document for background indexing"""
        try:
            message = {
                "task_id": task_id,
                "workspace_id": workspace_id,
                "user_id": user_id,
                "file_path": file_path,
                "file_name": file_name,
                "domain": domain or None,
                "kb_name": kb_name or None,
                "kb_id": kb_id,
                "container_name": container_name,
                "queued_at": str(datetime.now(timezone.utc)),
            }

            await self.queue_helper.send_message(message)

            # Update task status to processing
            async with get_async_session() as session:
                stmt = select(FileTask).where(FileTask.id == task_id)
                result = await session.execute(stmt)
                task = result.scalar_one_or_none()

                if task:
                    task.status = "processing"
                    task.updated_at = datetime.now(timezone.utc)
                    await session.commit()

            logger.debug(
                "Queued document for indexing",
                task_id=task_id,
                file_name=file_name,
            )

        except Exception as e:
            logger.error(
                "Failed to queue document",
                error=e,
                task_id=task_id,
            )
            raise


# ============================================================================
# Singleton Instance
# ============================================================================

_rag_service_instance: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    """Get or create singleton RAG service instance"""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance
