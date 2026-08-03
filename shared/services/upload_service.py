"""
Upload Service

Handles file upload to blob storage and enqueueing indexing jobs.
Following Single Responsibility and Dependency Inversion principles.
"""

import base64
import json
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import psycopg2
from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.storage.queue.aio import QueueServiceClient

from ..models.upload_models import FileMetadata, IndexingJob, UploadResponse


class UploadService:
    """
    Service for uploading files and enqueueing indexing jobs.

    Responsibilities:
    - Upload files to Azure Blob Storage
    - Create file_tasks records for tracking
    - Enqueue indexing jobs to Azure Storage Queue
    - Generate SAS URLs for uploaded files

    Dependencies:
    - Azure Blob Storage (via connection string)
    - Azure Storage Queue (via connection string)
    - PostgreSQL (for file_tasks tracking)
    """

    def __init__(self, connection_string: str, container_name: str, queue_name: str):
        """
        Initialize upload service.

        Args:
            connection_string: Azure Storage connection string
            container_name: Blob container name
            queue_name: Queue name for indexing jobs
        """
        self.connection_string = connection_string
        self.container_name = container_name
        self.queue_name = queue_name

    async def upload_files_and_enqueue(
        self,
        workspace_id: int,
        user_id: int,
        role_id: int,
        domain: str,
        kb_name: str,
        files: List[FileMetadata],
    ) -> UploadResponse:
        """
        Upload files to blob storage and enqueue indexing jobs.

        Args:
            workspace_id: Workspace identifier
            user_id: User identifier
            role_id: User role identifier
            domain: Domain/industry name
            kb_name: Knowledge base name
            files: List of files to upload

        Returns:
            UploadResponse with task IDs and status
        """
        try:
            # Construct blob path based on role and workspace
            upload_path = self._construct_upload_path(
                domain, kb_name, workspace_id, role_id
            )

            # Upload files to blob storage
            uploaded_files = await self._upload_to_blob(upload_path, files)

            # Create file_tasks records for tracking
            task_ids = []
            for file_meta, blob_url in uploaded_files:
                task_id = self._create_file_task(
                    workspace_id=workspace_id,
                    user_id=user_id,
                    container_name=self.container_name,
                    upload_path=upload_path,
                    domain=domain,
                    kb_name=kb_name,
                    file_name=file_meta.file_name,
                    file_size=file_meta.file_size,
                    status="uploaded",
                )
                if task_id:
                    task_ids.append(task_id)

            # Enqueue indexing jobs
            await self._enqueue_indexing_jobs(
                workspace_id=workspace_id,
                domain=domain,
                kb_name=kb_name,
                uploaded_files=uploaded_files,
                user_id=user_id,
            )

            return UploadResponse(
                status="success",
                message=f"Successfully uploaded {len(uploaded_files)} file(s) and queued for indexing",
                task_ids=task_ids,
                failed_files=[],
            )

        except Exception as e:
            return UploadResponse(
                status="error",
                message=f"Upload failed: {str(e)}",
                task_ids=[],
                failed_files=[f.file_name for f in files],
            )

    def _construct_upload_path(
        self, domain: str, kb_name: str, workspace_id: int, role_id: int
    ) -> str:
        """
        Construct blob storage path based on user role and workspace.

        SME (role_id=34): domain/kb_name/knowledge_bases/
        Others: domain/kb_name/knowledge_bases/{workspace_id}/

        Args:
            domain: Domain name
            kb_name: KB name
            workspace_id: Workspace ID
            role_id: User role ID

        Returns:
            str: Blob path
        """
        base_path = f"{domain}/{kb_name}/knowledge_bases"

        if role_id == 34:  # SME
            return base_path
        else:  # Workspace user
            return f"{base_path}/{workspace_id}"

    async def _upload_to_blob(
        self, upload_path: str, files: List[FileMetadata]
    ) -> List[Tuple[FileMetadata, str]]:
        """
        Upload files to Azure Blob Storage.

        Args:
            upload_path: Base path in blob container
            files: List of files to upload

        Returns:
            List of (file_metadata, blob_url) tuples
        """
        blob_service_client = BlobServiceClient.from_connection_string(
            self.connection_string
        )
        container_client = blob_service_client.get_container_client(
            self.container_name
        )

        uploaded = []

        for file_meta in files:
            try:
                # Decode base64 content
                file_bytes = self._decode_file_content(file_meta.file_content)

                # Determine content type
                content_type = self._get_content_type(file_meta.file_name)

                # Upload to blob
                blob_path = f"{upload_path}/{file_meta.file_name}"
                blob_client = container_client.get_blob_client(blob_path)

                blob_client.upload_blob(
                    file_bytes,
                    overwrite=True,
                    content_settings=ContentSettings(content_type=content_type),
                )

                blob_url = blob_client.url
                uploaded.append((file_meta, blob_url))

            except Exception as e:
                print(f"Failed to upload {file_meta.file_name}: {e}")
                continue

        return uploaded

    async def _enqueue_indexing_jobs(
        self,
        workspace_id: int,
        domain: str,
        kb_name: str,
        uploaded_files: List[Tuple[FileMetadata, str]],
        user_id: int,
    ) -> None:
        """
        Enqueue indexing jobs to Azure Storage Queue.

        Args:
            workspace_id: Workspace identifier
            domain: Domain name
            kb_name: KB name
            uploaded_files: List of uploaded file metadata and URLs
            user_id: User who uploaded
        """
        queue_service = QueueServiceClient.from_connection_string(
            self.connection_string
        )
        queue_client = queue_service.get_queue_client(self.queue_name)

        # Create queue if not exists
        try:
            await queue_client.create_queue()
        except Exception:
            pass  # Queue already exists

        for file_meta, blob_url in uploaded_files:
            try:
                # Create indexing job
                job = IndexingJob(
                    job_id=str(uuid.uuid4()),
                    workspace_id=workspace_id,
                    document_url=blob_url,
                    kb_id=f"{domain}_{kb_name}",
                    domain=domain,
                    kb_name=kb_name,
                    file_name=file_meta.file_name,
                    user_id=user_id,
                    created_at=datetime.utcnow(),
                )

                # Send to queue
                message = job.json()
                await queue_client.send_message(message)

                print(f"Enqueued job {job.job_id} for {file_meta.file_name}")

            except Exception as e:
                print(f"Failed to enqueue job for {file_meta.file_name}: {e}")

    def _create_file_task(
        self,
        workspace_id: int,
        user_id: int,
        container_name: str,
        upload_path: str,
        domain: str,
        kb_name: str,
        file_name: str,
        file_size: Optional[int],
        status: str = "uploading",
    ) -> Optional[int]:
        """
        Create file_tasks record in PostgreSQL for tracking.

        Args:
            workspace_id: Workspace ID
            user_id: User ID
            container_name: Blob container name
            upload_path: Upload path
            domain: Domain name
            kb_name: KB name
            file_name: File name
            file_size: File size in bytes
            status: Initial status

        Returns:
            int: Task ID if created, None otherwise
        """
        try:
            conn = psycopg2.connect(
                host=os.environ["POSTGRES_HOST"],
                user=os.environ["POSTGRES_USER"],
                password=os.environ["POSTGRES_PASSWORD"],
                dbname=os.environ.get("POSTGRESQL_DATABASE_DATABASE_2"),
            )

            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO file_tasks
                    (container_name, upload_path, domain, kb_name, file_path, workspace_id, status, file_size, uploaded_by, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                    RETURNING id
                    """,
                    (
                        container_name,
                        upload_path,
                        domain,
                        kb_name,
                        f"{upload_path}/{file_name}",
                        workspace_id,
                        status,
                        file_size,
                        user_id,
                    ),
                )
                task_id = cur.fetchone()[0]
                conn.commit()
                return task_id

        except Exception as e:
            print(f"Error creating file_tasks record: {e}")
            return None
        finally:
            if conn:
                conn.close()

    @staticmethod
    def _decode_file_content(content: str) -> bytes:
        """
        Decode base64 file content.

        Args:
            content: Base64 encoded string

        Returns:
            bytes: Decoded file content
        """
        # Handle data URLs (e.g., "data:application/pdf;base64,...")
        if content.startswith("data:"):
            content = content.split("base64,", 1)[-1]

        try:
            return base64.b64decode(content, validate=True)
        except Exception:
            # If not base64, treat as UTF-8 text
            return content.encode("utf-8")

    @staticmethod
    def _get_content_type(file_name: str) -> str:
        """
        Get MIME content type from file extension.

        Args:
            file_name: File name with extension

        Returns:
            str: MIME content type
        """
        ext = os.path.splitext(file_name)[1].lower()

        content_types = {
            ".pdf": "application/pdf",
            ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ".doc": "application/msword",
            ".txt": "text/plain",
            ".csv": "text/csv",
            ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            ".xls": "application/vnd.ms-excel",
            ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            ".ppt": "application/vnd.ms-powerpoint",
        }

        return content_types.get(ext, "application/octet-stream")
