"""Azure Blob Storage client"""
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

from azure.storage.blob import BlobSasPermissions, BlobServiceClient, generate_blob_sas

from core.exceptions import ConfigurationException


@dataclass(frozen=True)
class BlobInfo:
    container: str
    blob_name: str
    blob_url: str


class BlobClient:
    """Upload files to Azure Blob Storage + generate download URLs"""

    def __init__(self) -> None:
        from core.config import settings

        conn_str = settings.blob.AZURE_STORAGE_CONNECTION_STRING
        if not conn_str:
            raise ConfigurationException(
                "AZURE_STORAGE_CONNECTION_STRING not configured",
                config_key="AZURE_STORAGE_CONNECTION_STRING",
            )

        self._service = BlobServiceClient.from_connection_string(conn_str)
        self._container = settings.blob.BLOB_CONTAINER_NAME
        self._path_prefix = settings.blob.BLOB_PATH_PREFIX.strip("/")
        self._expiry_minutes = settings.blob.BLOB_URL_EXPIRY_MINUTES

    @property
    def container_name(self) -> str:
        return self._container

    def build_blob_name(self, filename: str) -> str:
        if not self._path_prefix:
            return filename
        return f"{self._path_prefix}/{filename}"

    def upload(self, filename: str, data: bytes, content_type: str = None) -> BlobInfo:
        """Upload file to blob storage"""
        if not filename or not filename.strip():
            raise ValueError("filename cannot be empty")

        blob_name = self.build_blob_name(filename.strip())
        container = self._service.get_container_client(self._container)
        try:
            container.create_container()
        except Exception:
            pass  # Container exists

        blob = container.get_blob_client(blob_name)
        blob.upload_blob(
            data, overwrite=True, content_type=content_type or "application/octet-stream"
        )
        return BlobInfo(
            container=self._container,
            blob_name=blob_name,
            blob_url=blob.url,
        )

    def generate_download_url(
        self, filename: str, expiry_minutes: Optional[int] = None
    ) -> str:
        """Generate SAS URL for downloading blob"""
        blob_name = self.build_blob_name(filename.strip())
        expiry = datetime.now(timezone.utc) + timedelta(
            minutes=max(1, expiry_minutes or self._expiry_minutes)
        )

        credential = self._service.credential
        account_key = getattr(credential, "account_key", None)
        if not account_key:
            raise ConfigurationException(
                "Blob account key unavailable; SAS URL generation requires connection string with account key",
                config_key="AZURE_STORAGE_CONNECTION_STRING",
            )

        sas = generate_blob_sas(
            account_name=self._service.account_name,
            container_name=self._container,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=expiry,
        )
        blob = self._service.get_blob_client(self._container, blob_name)
        return f"{blob.url}?{sas}"

    def blob_exists(self, filename: str) -> bool:
        """Check if blob exists"""
        blob_name = self.build_blob_name(filename.strip())
        blob = self._service.get_blob_client(self._container, blob_name)
        try:
            return bool(blob.exists())
        except Exception:
            return False


_client: Optional[BlobClient] = None


def get_blob_client() -> BlobClient:
    """Get singleton blob client"""
    global _client
    if _client is None:
        _client = BlobClient()
    return _client
