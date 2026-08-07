"""
RAG Service - Extended Knowledge Base Operations

Handles RAG queries, document indexing, blob storage, and workspace management.
Mirrors functionality from KnowledgeCurator MCP tools.
"""
import asyncio
import base64
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from azure.storage.blob import (
    BlobSasPermissions,
    BlobServiceClient,
    generate_blob_sas,
)
from sqlalchemy import select

from src.core.config import settings
from src.core.database import FileTask, get_async_session
from src.core.exceptions import LightRAGException, ValidationException
from src.core.lightrag_service import get_lightrag_service
from src.core.logging import get_logger
from src.helpers.queue_helpers import get_indexing_queue_helper

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
        self.lightrag_service = get_lightrag_service()
        self._queue_helper = None  # Lazy-loaded only when needed
        self.blob_connection_string = settings.storage.AZURE_BLOB_STORAGE_CONNECTION_STRING
        self.blob_container_name = settings.storage.STORAGE_CONTAINER_NAME

    @property
    def queue_helper(self):
        """Lazy-load queue helper only when needed (for indexing operations)"""
        if self._queue_helper is None:
            self._queue_helper = get_indexing_queue_helper()
        return self._queue_helper

    # ========================================================================
    # Document Upload & Indexing
    # ========================================================================

    async def upload_and_index_tool(
        self,
        file_names: List[str],
        file_contents: List[str],
        workspace_id: int,
        user_id: int,
        role_id: int,
        domain: Optional[str] = None,
        kb_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Upload documents to blob storage and queue for indexing.

        Args:
            file_names: List of file names
            file_contents: List of file contents (base64 encoded)
            workspace_id: Workspace identifier
            user_id: User identifier
            role_id: Role identifier
            domain: Optional domain/industry name
            kb_name: Optional knowledge base name

        Returns:
            Dict with task_ids and uploaded file paths
        """
        try:
            if len(file_names) != len(file_contents):
                raise ValidationException(
                    message="file_names and file_contents must have the same length"
                )

            logger.info(
                "Uploading and indexing documents",
                workspace_id=workspace_id,
                file_count=len(file_names),
            )

            # Construct blob path
            blob_path = self._construct_blob_path(workspace_id, domain, kb_name)

            # Upload files to blob storage
            uploaded_files = []
            task_ids = []

            if not self.blob_connection_string:
                raise ValidationException(
                    message="Azure Blob Storage connection string not configured"
                )

            blob_service_client = BlobServiceClient.from_connection_string(
                self.blob_connection_string
            )
            container_client = blob_service_client.get_container_client(
                self.blob_container_name
            )

            for file_name, file_content in zip(file_names, file_contents):
                try:
                    # Decode base64 content
                    if isinstance(file_content, str):
                        try:
                            file_bytes = base64.b64decode(file_content)
                        except Exception:
                            # If not base64, assume it's already bytes/string
                            file_bytes = file_content.encode() if isinstance(file_content, str) else file_content
                    else:
                        file_bytes = file_content

                    # Upload to blob
                    full_blob_path = f"{blob_path}/{file_name}"
                    blob_client = container_client.get_blob_client(full_blob_path)
                    blob_client.upload_blob(file_bytes, overwrite=True)

                    uploaded_files.append(full_blob_path)

                    # Create file task for indexing
                    async with get_async_session() as session:
                        file_task = FileTask(
                            container_name=self.blob_container_name,
                            upload_path=blob_path,
                            domain=domain or "",
                            kb_name=kb_name or "",
                            file_path=full_blob_path,
                            file_name=file_name,
                            workspace_id=workspace_id,
                            status="pending",
                            file_size=len(file_bytes),
                            uploaded_by=str(user_id),
                            created_at=datetime.now(timezone.utc),
                            updated_at=datetime.now(timezone.utc),
                        )
                        session.add(file_task)
                        await session.flush()
                        task_id = file_task.id
                        task_ids.append(task_id)

                    # Queue for indexing
                    await self._queue_document_for_indexing(
                        task_id=task_id,
                        workspace_id=workspace_id,
                        file_path=full_blob_path,
                        file_name=file_name,
                    )

                    logger.info(
                        "Uploaded and queued document",
                        file_name=file_name,
                        task_id=task_id,
                    )

                except Exception as e:
                    logger.error(
                        "Failed to upload file",
                        error=e,
                        file_name=file_name,
                    )
                    # Mark task as failed if it was created
                    if task_ids and len(task_ids) > 0:
                        async with get_async_session() as session:
                            stmt = select(FileTask).where(FileTask.id == task_ids[-1])
                            result = await session.execute(stmt)
                            task = result.scalar_one_or_none()
                            if task:
                                task.status = "failed"
                                task.error_message = str(e)
                                await session.commit()

            return {
                "response": f"Successfully uploaded {len(uploaded_files)} file(s)",
                "task_ids": task_ids,
                "uploaded_files": uploaded_files,
            }

        except Exception as e:
            logger.error("Upload and index failed", error=e)
            raise LightRAGException(
                message=f"Failed to upload and index documents: {str(e)}",
                operation="upload_and_index"
            )

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

    async def index_uploaded_files(
        self,
        file_paths: List[str],
        workspace_id: int,
        user_id: int,
    ) -> Dict[str, Any]:
        """
        Index specific uploaded files.

        Args:
            file_paths: List of blob file paths to index
            workspace_id: Workspace identifier
            user_id: User identifier

        Returns:
            Dict with status and task_ids
        """
        try:
            logger.info(
                "Indexing uploaded files",
                workspace_id=workspace_id,
                file_count=len(file_paths),
            )

            task_ids = []

            for file_path in file_paths:
                # Check if file task exists
                async with get_async_session() as session:
                    stmt = select(FileTask).where(
                        FileTask.file_path == file_path,
                        FileTask.workspace_id == workspace_id,
                    )
                    result = await session.execute(stmt)
                    task = result.scalar_one_or_none()

                    if task:
                        # Extract file_name from file_path (file_name removed from FileTask)
                        file_name = os.path.basename(task.file_path) if task.file_path else None

                        # Queue existing task
                        await self._queue_document_for_indexing(
                            task_id=task.id,
                            workspace_id=workspace_id,
                            file_path=file_path,
                            file_name=file_name,
                        )
                        task_ids.append(task.id)
                    else:
                        # Create new task (file_name removed from FileTask schema)
                        file_name = os.path.basename(file_path)
                        new_task = FileTask(
                            container_name=self.blob_container_name,
                            file_path=file_path,
                            workspace_id=workspace_id,
                            status="pending",
                            uploaded_by=str(user_id),
                        )
                        session.add(new_task)
                        await session.flush()
                        task_id = new_task.id
                        task_ids.append(task_id)

                        await self._queue_document_for_indexing(
                            task_id=task_id,
                            workspace_id=workspace_id,
                            file_path=file_path,
                            file_name=file_name,
                        )

            return {
                "status": "indexing_queued",
                "message": f"Queued {len(task_ids)} file(s) for indexing",
                "task_ids": task_ids,
            }

        except Exception as e:
            logger.error("Failed to index uploaded files", error=e)
            raise LightRAGException(
                message=f"Failed to index uploaded files: {str(e)}",
                operation="index_uploaded_files"
            )

    # ========================================================================
    # Indexing Status & Monitoring
    # ========================================================================

    async def check_specific_indexing_status(
        self,
        task_ids: List[int],
        workspace_id: int,
    ) -> List[Dict[str, Any]]:
        """
        Check indexing status for specific tasks.

        Args:
            task_ids: List of task IDs
            workspace_id: Workspace identifier

        Returns:
            List of task status dictionaries
        """
        try:
            logger.info(
                "Checking indexing status",
                workspace_id=workspace_id,
                task_count=len(task_ids),
            )

            statuses = []

            async with get_async_session() as session:
                for task_id in task_ids:
                    stmt = select(FileTask).where(
                        FileTask.id == task_id,
                        FileTask.workspace_id == workspace_id,
                    )
                    result = await session.execute(stmt)
                    task = result.scalar_one_or_none()

                    if task:
                        # Extract file_name from file_path (file_name removed from FileTask)
                        file_name = os.path.basename(task.file_path) if task.file_path else None

                        statuses.append({
                            "task_id": task.id,
                            "file_name": file_name,
                            "file_path": task.file_path,
                            "status": task.status,
                            "workspace_id": task.workspace_id,
                            "file_size": task.file_size,
                            "error_message": task.error_message,
                            "created_at": str(task.created_at) if task.created_at else None,
                            "updated_at": str(task.updated_at) if task.updated_at else None,
                        })
                    else:
                        statuses.append({
                            "task_id": task_id,
                            "status": "not_found",
                            "error_message": "Task not found",
                        })

            logger.info(
                "Retrieved indexing statuses",
                found_count=len([s for s in statuses if s.get("status") != "not_found"]),
            )

            return statuses

        except Exception as e:
            logger.error("Failed to check indexing status", error=e)
            raise LightRAGException(
                message=f"Failed to check indexing status: {str(e)}",
                operation="check_indexing_status"
            )

    # ========================================================================
    # Blob Storage Operations
    # ========================================================================

    async def fetch_blob_structure(
        self,
        workspace_id: Optional[int] = None,
    ) -> Dict[str, List[str]]:
        """
        Fetch directory structure from Azure Blob Storage.

        Returns:
            Dict mapping domains to lists of knowledge bases
            Example: {"domain1": ["kb1", "kb2"], "domain2": ["kb1"]}
        """
        try:
            logger.info("Fetching blob structure", workspace_id=workspace_id)

            if not self.blob_connection_string:
                return {"error": "Azure Blob Storage not configured"}

            blob_service_client = BlobServiceClient.from_connection_string(
                self.blob_connection_string
            )
            container_client = blob_service_client.get_container_client(
                self.blob_container_name
            )

            structure = {}

            # List all blobs and extract domain/kb structure
            async for blob in container_client.list_blobs():
                # Expecting path like "domain/kb/filename"
                parts = blob.name.split("/")
                if len(parts) >= 2:
                    domain, kb = parts[0], parts[1]
                    if domain not in structure:
                        structure[domain] = set()
                    structure[domain].add(kb)

            # Convert sets to lists for JSON serializability
            structure = {domain: list(kbs) for domain, kbs in structure.items()}

            logger.info(
                "Fetched blob structure",
                domain_count=len(structure),
            )

            return structure

        except Exception as e:
            logger.error("Failed to fetch blob structure", error=e)
            return {"error": str(e)}

    async def delete_files_from_blob(
        self,
        file_names: List[str],
        workspace_id: int,
        role_id: int,
        domain: Optional[str] = None,
        kb_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Delete files from Azure Blob Storage.

        Args:
            file_names: List of file names to delete
            workspace_id: Workspace identifier
            role_id: Role identifier
            domain: Optional domain name
            kb_names: Optional list of KB names

        Returns:
            Dict with deleted files list
        """
        try:
            logger.info(
                "Deleting files from blob",
                workspace_id=workspace_id,
                file_count=len(file_names),
            )

            if not self.blob_connection_string:
                raise ValidationException(
                    message="Azure Blob Storage not configured"
                )

            blob_service_client = BlobServiceClient.from_connection_string(
                self.blob_connection_string
            )
            container_client = blob_service_client.get_container_client(
                self.blob_container_name
            )

            deleted_files = []
            kb_list = kb_names or [""]

            for kb in kb_list:
                for file_name in file_names:
                    # Construct blob path
                    if domain and kb:
                        blob_path = f"{domain}/{kb}/{file_name}"
                    else:
                        blob_path = self._construct_blob_path(workspace_id, domain, kb) + f"/{file_name}"

                    blob_client = container_client.get_blob_client(blob_path)

                    try:
                        blob_client.delete_blob()
                        deleted_files.append(blob_path)
                        logger.info("Deleted blob", blob_path=blob_path)

                        # Update file task status
                        async with get_async_session() as session:
                            stmt = select(FileTask).where(
                                FileTask.file_path == blob_path,
                                FileTask.workspace_id == workspace_id,
                            )
                            result = await session.execute(stmt)
                            task = result.scalar_one_or_none()

                            if task:
                                await session.delete(task)
                                await session.commit()

                    except Exception as e:
                        logger.warning(
                            "Failed to delete blob",
                            error=e,
                            blob_path=blob_path,
                        )

            return {
                "status": "completed",
                "message": f"Deleted {len(deleted_files)} file(s)",
                "deleted_files": deleted_files,
            }

        except Exception as e:
            logger.error("Failed to delete files from blob", error=e)
            raise LightRAGException(
                message=f"Failed to delete files from blob: {str(e)}",
                operation="delete_files_from_blob"
            )

    # ========================================================================
    # Helper Methods
    # ========================================================================

    async def _get_workspace_working_dir(
        self,
        workspace_id: int,
        domain: Optional[str] = None,
        kb_name: Optional[str] = None
    ) -> str:
        """
        Get LightRAG working directory for workspace.

        If domain/kb_name not provided, fetches from database for proper scoping.

        Args:
            workspace_id: Workspace identifier
            domain: Optional domain (fetched from DB if not provided)
            kb_name: Optional KB name (fetched from DB if not provided)

        Returns:
            Full working directory path for LightRAG

        Raises:
            Exception: If workspace storage paths cannot be retrieved
        """
        from src.helpers.workspace_helpers import get_workspace_storage_paths
        from shared.workspace_helpers import get_workspace_working_dir

        base_dir = settings.lightrag.LIGHTRAG_WORKING_DIR

        # Fetch domain/kb_name from database if not provided
        if domain is None or kb_name is None:
            storage_paths = await get_workspace_storage_paths(workspace_id)
            if not storage_paths:
                logger.error(
                    "Failed to get workspace storage paths for working directory",
                    workspace_id=workspace_id
                )
                # Fallback to basic working dir if storage paths unavailable
                return get_workspace_working_dir(workspace_id, base_dir)

            domain = storage_paths.get("domain")
            kb_name = storage_paths.get("kb_name")

            logger.debug(
                "Retrieved domain/kb_name from database for working directory",
                workspace_id=workspace_id,
                domain=domain,
                kb_name=kb_name
            )

        # Build working directory with domain/kb_name for proper scoping
        working_dir = get_workspace_working_dir(
            workspace_id=workspace_id,
            base_dir=base_dir,
            domain=domain,
            kb_name=kb_name
        )

        logger.debug(
            "Built workspace working directory",
            workspace_id=workspace_id,
            working_dir=working_dir
        )

        return working_dir

    def _construct_blob_path(
        self,
        workspace_id: int,
        domain: Optional[str] = None,
        kb_name: Optional[str] = None,
    ) -> str:
        """Construct blob storage path"""
        parts = []

        if domain:
            parts.append(domain)
        else:
            parts.append(f"workspace_{workspace_id}")

        if kb_name:
            parts.append(kb_name)
        else:
            parts.append("default")

        return "/".join(parts)

    async def _queue_document_for_indexing(
        self,
        task_id: int,
        workspace_id: int,
        file_path: str,
        file_name: str,
    ) -> None:
        """Queue a document for background indexing"""
        try:
            message = {
                "task_id": task_id,
                "workspace_id": workspace_id,
                "file_path": file_path,
                "file_name": file_name,
                "queued_at": str(datetime.now(timezone.utc)),
            }

            await self.queue_helper.send_message_async(message)

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
