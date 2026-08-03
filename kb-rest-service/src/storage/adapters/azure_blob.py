"""Azure Blob Storage adapter"""

import asyncio
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobSasPermissions, BlobServiceClient, generate_blob_sas

from core.exceptions import ConfigurationException
from core.logging import get_logger

from ..models import BlobInfo
from ..protocols import StorageAdapter

logger = get_logger(__name__)


class AzureBlobStorageAdapter(StorageAdapter):
    """Azure Blob Storage implementation"""

    def __init__(self) -> None:
        from core.config import settings

        conn_str = settings.storage.AZURE_BLOB_STORAGE_CONNECTION_STRING
        if not conn_str:
            raise ConfigurationException(
                "AZURE_BLOB_STORAGE_CONNECTION_STRING not configured",
                config_key="AZURE_BLOB_STORAGE_CONNECTION_STRING",
            )

        self._service = BlobServiceClient.from_connection_string(conn_str)
        self._container = settings.storage.AZURE_BLOB_STORAGE_CONTAINER_NAME or "documents"
        self._path_prefix = getattr(settings.storage, "BLOB_PATH_PREFIX", "").strip("/")
        self._expiry_minutes = getattr(settings, "BLOB_URL_EXPIRY_MINUTES", 60)

        logger.info(
            "Azure Blob Storage adapter initialized",
            container=self._container,
            prefix=self._path_prefix,
        )

    @property
    def provider_name(self) -> str:
        return "azure"

    @property
    def container_name(self) -> str:
        return self._container

    def _build_blob_name(self, filename: str) -> str:
        """Build full blob path with prefix"""
        if not self._path_prefix:
            return filename
        return f"{self._path_prefix}/{filename}"

    async def upload(
        self, filename: str, data: bytes, content_type: Optional[str] = None, container: Optional[str] = None
    ) -> BlobInfo:
        """Upload file to Azure Blob Storage

        Args:
            filename: Name/path of the file
            data: File content as bytes
            content_type: MIME type
            container: Optional container override (uses default if not specified)
        """
        if not filename or not filename.strip():
            raise ValueError("filename cannot be empty")

        # Use specified container or fall back to default
        target_container = container or self._container

        blob_name = self._build_blob_name(filename.strip())
        container_client = self._service.get_container_client(target_container)

        # Create container if it doesn't exist
        try:
            await asyncio.to_thread(container_client.create_container)
            logger.info(f"Created container: {target_container}")
        except Exception:
            pass  # Container already exists

        blob_client = container_client.get_blob_client(blob_name)

        # Upload blob (using asyncio.to_thread for sync SDK)
        await asyncio.to_thread(
            blob_client.upload_blob,
            data,
            overwrite=True,
            content_type=content_type or "application/octet-stream",
        )

        size_bytes = len(data)

        logger.info(
            "File uploaded to Azure",
            blob_name=blob_name,
            size_bytes=size_bytes,
            content_type=content_type,
            container=target_container,
        )

        return BlobInfo(
            container=target_container,
            blob_name=blob_name,
            blob_url=blob_client.url,
            provider="azure",
            size_bytes=size_bytes,
        )

    async def generate_download_url(
        self, filename: str, expiry_minutes: Optional[int] = None, container: Optional[str] = None
    ) -> str:
        """Generate SAS URL for Azure blob

        Args:
            filename: Name/path of the file
            expiry_minutes: URL expiry time in minutes
            container: Optional container override
        """
        target_container = container or self._container
        blob_name = self._build_blob_name(filename.strip())
        expiry = datetime.now(timezone.utc) + timedelta(
            minutes=max(1, expiry_minutes or self._expiry_minutes)
        )

        credential = self._service.credential
        account_key = getattr(credential, "account_key", None)
        if not account_key:
            raise ConfigurationException(
                "Azure account key unavailable; SAS URL generation requires connection string with account key",
                config_key="AZURE_BLOB_STORAGE_CONNECTION_STRING",
            )

        account_name = self._service.account_name
        if not account_name:
            raise ConfigurationException(
                "Azure account name unavailable; SAS URL generation requires valid account name",
                config_key="AZURE_BLOB_STORAGE_CONNECTION_STRING",
            )

        sas_token = generate_blob_sas(
            account_name=account_name,
            container_name=target_container,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=expiry,
        )

        blob_client = self._service.get_blob_client(target_container, blob_name)
        return f"{blob_client.url}?{sas_token}"

    async def blob_exists(self, filename: str, container: Optional[str] = None) -> bool:
        """Check if blob exists in Azure

        Args:
            filename: Name/path of the file
            container: Optional container override
        """
        target_container = container or self._container
        blob_name = self._build_blob_name(filename.strip())
        blob_client = self._service.get_blob_client(target_container, blob_name)
        try:
            exists = await asyncio.to_thread(blob_client.exists)
            return bool(exists)
        except Exception as e:
            logger.warning(f"Error checking blob existence: {e}")
            return False

    async def delete(self, filename: str, container: Optional[str] = None) -> bool:
        """Delete blob from Azure

        Args:
            filename: Name/path of the file
            container: Optional container override
        """
        target_container = container or self._container
        blob_name = self._build_blob_name(filename.strip())
        blob_client = self._service.get_blob_client(target_container, blob_name)
        try:
            await asyncio.to_thread(blob_client.delete_blob)
            logger.info(f"Deleted blob: {blob_name} from container: {target_container}")
            return True
        except ResourceNotFoundError:
            logger.warning(f"Blob not found for deletion: {blob_name}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete blob {blob_name}: {e}")
            raise

    async def download(self, filename: str, container: Optional[str] = None) -> bytes:
        """Download blob content from Azure

        Args:
            filename: Name/path of the file
            container: Optional container override
        """
        target_container = container or self._container
        blob_name = self._build_blob_name(filename.strip())
        blob_client = self._service.get_blob_client(target_container, blob_name)
        try:
            blob_data = await asyncio.to_thread(blob_client.download_blob)
            content = await asyncio.to_thread(blob_data.readall)
            logger.info(f"Downloaded blob: {blob_name}, size: {len(content)} bytes from container: {target_container}")
            return content
        except ResourceNotFoundError:
            raise FileNotFoundError(f"Blob not found: {blob_name}")
        except Exception as e:
            logger.error(f"Failed to download blob {blob_name}: {e}")
            raise
