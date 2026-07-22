"""
Azure Blob Storage Adapter

Implementation of StorageAdapter for Azure Blob Storage.
"""
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import (
    BlobSasPermissions,
    BlobServiceClient,
    ContentSettings,
    generate_blob_sas,
)
from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient

from core.exceptions import StorageException
from core.logging import get_logger

from .base import BlobContent, BlobMetadata, StorageAdapter

logger = get_logger(__name__)


class AzureBlobStorageAdapter(StorageAdapter):
    """
    Azure Blob Storage adapter implementation.

    Provides async operations for Azure Blob Storage using the
    azure-storage-blob SDK.
    """

    def __init__(
        self,
        container_name: str,
        connection_string: Optional[str] = None,
        account_name: Optional[str] = None,
        account_key: Optional[str] = None,
        **config
    ):
        """
        Initialize Azure Blob Storage adapter.

        Args:
            container_name: Azure container name
            connection_string: Azure Storage connection string
            account_name: Storage account name (alternative to connection_string)
            account_key: Storage account key (alternative to connection_string)
            **config: Additional configuration
        """
        super().__init__(container_name, **config)

        from core.config import settings

        self.connection_string = (
            connection_string or settings.storage.AZURE_BLOB_STORAGE_CONNECTION_STRING
        )
        self.account_name = account_name
        self.account_key = account_key

        if not self.connection_string and not (self.account_name and self.account_key):
            raise StorageException(
                message="Azure Blob Storage credentials not provided",
                operation="initialize",
            )

        self._sync_client: Optional[BlobServiceClient] = None
        self._async_client: Optional[AsyncBlobServiceClient] = None

        logger.info(
            "Azure Blob Storage adapter initialized",
            container_name=container_name,
        )

    def _get_sync_client(self) -> BlobServiceClient:
        """Get or create synchronous blob service client"""
        if not self._sync_client:
            if self.connection_string:
                self._sync_client = BlobServiceClient.from_connection_string(
                    self.connection_string
                )
            else:
                self._sync_client = BlobServiceClient(
                    account_url=f"https://{self.account_name}.blob.core.windows.net",
                    credential=self.account_key,
                )
        return self._sync_client

    def _get_async_client(self) -> AsyncBlobServiceClient:
        """Get or create asynchronous blob service client"""
        if not self._async_client:
            if self.connection_string:
                self._async_client = AsyncBlobServiceClient.from_connection_string(
                    self.connection_string
                )
            else:
                self._async_client = AsyncBlobServiceClient(
                    account_url=f"https://{self.account_name}.blob.core.windows.net",
                    credential=self.account_key,
                )
        return self._async_client

    async def upload_file(
        self,
        file_path: str,
        content: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        overwrite: bool = True,
    ) -> BlobMetadata:
        """Upload file to Azure Blob Storage"""
        try:
            client = self._get_async_client()
            container_client = client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(file_path)

            content_settings = None
            if content_type:
                content_settings = ContentSettings(content_type=content_type)

            # Upload blob
            await blob_client.upload_blob(
                content,
                overwrite=overwrite,
                content_settings=content_settings,
                metadata=metadata,
            )

            # Get metadata
            properties = await blob_client.get_blob_properties()

            logger.info(
                "File uploaded to Azure Blob Storage",
                file_path=file_path,
                size=len(content),
            )

            return BlobMetadata(
                name=os.path.basename(file_path),
                path=file_path,
                size=properties.size,
                content_type=properties.content_settings.content_type or "",
                created_at=properties.creation_time,
                updated_at=properties.last_modified,
                metadata=properties.metadata,
                etag=properties.etag,
            )

        except Exception as e:
            logger.error("Failed to upload file to Azure Blob Storage", error=e)
            raise StorageException(
                message=f"Failed to upload file: {str(e)}",
                operation="upload_file",
            )

    async def download_file(self, file_path: str) -> BlobContent:
        """Download file from Azure Blob Storage"""
        try:
            client = self._get_async_client()
            container_client = client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(file_path)

            # Download blob
            download_stream = await blob_client.download_blob()
            content = await download_stream.readall()

            # Get properties
            properties = await blob_client.get_blob_properties()

            logger.info(
                "File downloaded from Azure Blob Storage",
                file_path=file_path,
                size=len(content),
            )

            metadata = BlobMetadata(
                name=os.path.basename(file_path),
                path=file_path,
                size=properties.size,
                content_type=properties.content_settings.content_type or "",
                created_at=properties.creation_time,
                updated_at=properties.last_modified,
                metadata=properties.metadata,
                etag=properties.etag,
            )

            return BlobContent(data=content, metadata=metadata)

        except ResourceNotFoundError:
            raise StorageException(
                message=f"File not found: {file_path}",
                operation="download_file",
            )
        except Exception as e:
            logger.error("Failed to download file from Azure Blob Storage", error=e)
            raise StorageException(
                message=f"Failed to download file: {str(e)}",
                operation="download_file",
            )

    async def delete_file(self, file_path: str) -> bool:
        """Delete file from Azure Blob Storage"""
        try:
            client = self._get_async_client()
            container_client = client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(file_path)

            await blob_client.delete_blob()

            logger.info("File deleted from Azure Blob Storage", file_path=file_path)
            return True

        except ResourceNotFoundError:
            logger.warning("File not found for deletion", file_path=file_path)
            return False
        except Exception as e:
            logger.error("Failed to delete file from Azure Blob Storage", error=e)
            raise StorageException(
                message=f"Failed to delete file: {str(e)}",
                operation="delete_file",
            )

    async def list_files(
        self,
        prefix: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[BlobMetadata]:
        """List files in Azure Blob Storage"""
        try:
            client = self._get_async_client()
            container_client = client.get_container_client(self.container_name)

            blobs = []
            async for blob in container_client.list_blobs(
                name_starts_with=prefix,
                results_per_page=max_results,
            ):
                blobs.append(
                    BlobMetadata(
                        name=blob.name.split("/")[-1],
                        path=blob.name,
                        size=blob.size,
                        content_type=blob.content_settings.content_type or ""
                        if blob.content_settings
                        else "",
                        created_at=blob.creation_time,
                        updated_at=blob.last_modified,
                        metadata=blob.metadata,
                        etag=blob.etag,
                    )
                )

            logger.info(
                "Listed files from Azure Blob Storage",
                count=len(blobs),
                prefix=prefix,
            )

            return blobs

        except Exception as e:
            logger.error("Failed to list files from Azure Blob Storage", error=e)
            raise StorageException(
                message=f"Failed to list files: {str(e)}",
                operation="list_files",
            )

    async def file_exists(self, file_path: str) -> bool:
        """Check if file exists in Azure Blob Storage"""
        try:
            client = self._get_async_client()
            container_client = client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(file_path)

            return await blob_client.exists()

        except Exception as e:
            logger.error("Failed to check file existence", error=e)
            return False

    async def get_file_url(
        self,
        file_path: str,
        expiry_seconds: int = 3600,
    ) -> str:
        """Generate signed URL for Azure Blob Storage file"""
        try:
            sync_client = self._get_sync_client()

            account_name = sync_client.account_name
            account_key = sync_client.credential.account_key

            if not account_key:
                raise StorageException(
                    message="Account key required for SAS token generation",
                    operation="get_file_url",
                )

            # Generate SAS token
            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=self.container_name,
                blob_name=file_path,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.now(timezone.utc) + timedelta(seconds=expiry_seconds),
            )

            url = f"https://{account_name}.blob.core.windows.net/{self.container_name}/{file_path}?{sas_token}"

            logger.info("Generated signed URL", file_path=file_path)
            return url

        except Exception as e:
            logger.error("Failed to generate signed URL", error=e)
            raise StorageException(
                message=f"Failed to generate signed URL: {str(e)}",
                operation="get_file_url",
            )

    async def copy_file(
        self,
        source_path: str,
        destination_path: str,
    ) -> BlobMetadata:
        """Copy file within Azure Blob Storage"""
        try:
            client = self._get_async_client()
            container_client = client.get_container_client(self.container_name)

            source_blob = container_client.get_blob_client(source_path)
            dest_blob = container_client.get_blob_client(destination_path)

            # Copy blob
            source_url = source_blob.url
            await dest_blob.start_copy_from_url(source_url)

            # Wait for copy to complete
            properties = await dest_blob.get_blob_properties()
            while properties.copy.status == "pending":
                await asyncio.sleep(0.5)
                properties = await dest_blob.get_blob_properties()

            logger.info(
                "File copied in Azure Blob Storage",
                source=source_path,
                destination=destination_path,
            )

            return BlobMetadata(
                name=os.path.basename(destination_path),
                path=destination_path,
                size=properties.size,
                content_type=properties.content_settings.content_type or "",
                created_at=properties.creation_time,
                updated_at=properties.last_modified,
                metadata=properties.metadata,
                etag=properties.etag,
            )

        except Exception as e:
            logger.error("Failed to copy file in Azure Blob Storage", error=e)
            raise StorageException(
                message=f"Failed to copy file: {str(e)}",
                operation="copy_file",
            )

    async def get_file_metadata(self, file_path: str) -> BlobMetadata:
        """Get file metadata from Azure Blob Storage"""
        try:
            client = self._get_async_client()
            container_client = client.get_container_client(self.container_name)
            blob_client = container_client.get_blob_client(file_path)

            properties = await blob_client.get_blob_properties()

            return BlobMetadata(
                name=os.path.basename(file_path),
                path=file_path,
                size=properties.size,
                content_type=properties.content_settings.content_type or "",
                created_at=properties.creation_time,
                updated_at=properties.last_modified,
                metadata=properties.metadata,
                etag=properties.etag,
            )

        except ResourceNotFoundError:
            raise StorageException(
                message=f"File not found: {file_path}",
                operation="get_file_metadata",
            )
        except Exception as e:
            logger.error("Failed to get file metadata", error=e)
            raise StorageException(
                message=f"Failed to get file metadata: {str(e)}",
                operation="get_file_metadata",
            )

    async def close(self) -> None:
        """Close Azure Blob Storage clients"""
        if self._async_client:
            await self._async_client.close()
            self._async_client = None

        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

        logger.info("Azure Blob Storage clients closed")
