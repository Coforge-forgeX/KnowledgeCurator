"""Azure Blob Storage adapter"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from azure.storage.blob import (
    BlobServiceClient,
    BlobSasPermissions,
    ContentSettings,
    generate_blob_sas,
)

from ..models import BlobInfo
from ..protocols import StorageAdapter

logger = logging.getLogger(__name__)


class AzureBlobStorageAdapter(StorageAdapter):
    """Azure Blob Storage implementation"""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        container_name: Optional[str] = None
    ):
        """
        Initialize Azure Blob Storage adapter.

        Args:
            connection_string: Azure Storage connection string
            container_name: Container name
        """
        if not connection_string:
            raise ValueError("Azure Blob Storage connection string is required")
        if not container_name:
            raise ValueError("Container name is required")

        self.connection_string = connection_string
        self._container_name = container_name

        # Initialize blob service client
        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.connection_string
        )
        self.container_client = self.blob_service_client.get_container_client(
            self._container_name
        )

        logger.info(
            f"Azure Blob Storage adapter initialized for container: {self._container_name}"
        )

    async def upload(
        self, filename: str, data: bytes, content_type: Optional[str] = None
    ) -> BlobInfo:
        """Upload file to Azure Blob Storage"""
        if not filename:
            raise ValueError("Filename cannot be empty")

        try:
            blob_client = self.container_client.get_blob_client(filename)

            # Upload with content type
            await asyncio.to_thread(
                blob_client.upload_blob,
                data,
                overwrite=True,
                content_settings=ContentSettings(
                    content_type=content_type or "application/octet-stream"
                ),
            )

            blob_url = blob_client.url

            logger.info(
                f"File uploaded to Azure Blob Storage: {filename}, size: {len(data)}"
            )

            return BlobInfo(
                container=self._container_name,
                blob_name=filename,
                blob_url=blob_url,
                provider="azure",
                size_bytes=len(data)
            )

        except Exception as e:
            logger.error(
                f"Failed to upload file to Azure Blob Storage: {filename}, error: {e}",
                exc_info=True
            )
            raise

    async def download(self, filename: str) -> bytes:
        """Download file from Azure Blob Storage"""
        try:
            blob_client = self.container_client.get_blob_client(filename)

            # Check if blob exists
            exists = await asyncio.to_thread(blob_client.exists)
            if not exists:
                raise FileNotFoundError(f"Blob not found: {filename}")

            # Download blob
            download_stream = await asyncio.to_thread(blob_client.download_blob)
            data = await asyncio.to_thread(download_stream.readall)

            logger.info(
                f"File downloaded from Azure Blob Storage: {filename}, size: {len(data)}"
            )

            return data

        except FileNotFoundError:
            raise
        except Exception as e:
            logger.error(
                f"Failed to download file from Azure Blob Storage: {filename}, error: {e}",
                exc_info=True
            )
            raise

    async def blob_exists(self, filename: str) -> bool:
        """Check if blob exists"""
        try:
            blob_client = self.container_client.get_blob_client(filename)
            exists = await asyncio.to_thread(blob_client.exists)
            return exists
        except Exception as e:
            logger.error(f"Failed to check blob existence: {filename}, error: {e}")
            return False

    async def delete(self, filename: str) -> bool:
        """Delete blob"""
        try:
            blob_client = self.container_client.get_blob_client(filename)

            # Check if exists
            exists = await self.blob_exists(filename)
            if not exists:
                return False

            # Delete
            await asyncio.to_thread(blob_client.delete_blob)

            logger.info(f"Blob deleted: {filename}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete blob: {filename}, error: {e}")
            return False

    async def generate_download_url(
        self, filename: str, expiry_minutes: Optional[int] = None
    ) -> str:
        """Generate SAS URL for download"""
        try:
            blob_client = self.container_client.get_blob_client(filename)

            # Extract account name and key from connection string
            conn_parts = dict(part.split("=", 1) for part in self.connection_string.split(";") if "=" in part)
            account_name = conn_parts.get("AccountName")
            account_key = conn_parts.get("AccountKey")

            if not account_name or not account_key:
                raise ValueError("Could not extract account credentials from connection string")

            # Generate SAS token
            sas_token = generate_blob_sas(
                account_name=account_name,
                container_name=self._container_name,
                blob_name=filename,
                account_key=account_key,
                permission=BlobSasPermissions(read=True),
                expiry=datetime.now(timezone.utc) + timedelta(minutes=expiry_minutes or 60)
            )

            # Build URL
            sas_url = f"{blob_client.url}?{sas_token}"

            logger.debug(f"SAS URL generated for: {filename}")
            return sas_url

        except Exception as e:
            logger.error(f"Failed to generate SAS URL: {filename}, error: {e}")
            raise

    @property
    def provider_name(self) -> str:
        return "azure"

    @property
    def container_name(self) -> str:
        return self._container_name
