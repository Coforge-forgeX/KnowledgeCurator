"""Abstract storage adapter interface"""

from abc import ABC, abstractmethod
from typing import Optional

from .models import BlobInfo


class StorageAdapter(ABC):
    """Abstract interface for cloud storage operations"""

    @abstractmethod
    async def upload(
        self, filename: str, data: bytes, content_type: Optional[str] = None
    ) -> BlobInfo:
        """
        Upload file to cloud storage.

        Args:
            filename: Target filename (path prefix applied automatically)
            data: File data as bytes
            content_type: MIME type (defaults to application/octet-stream)

        Returns:
            BlobInfo with upload details including size

        Raises:
            ValueError: If filename is empty
            Exception: If upload fails
        """
        pass

    @abstractmethod
    async def generate_download_url(
        self, filename: str, expiry_minutes: Optional[int] = None
    ) -> str:
        """
        Generate signed/temporary download URL.

        Args:
            filename: Filename to generate URL for
            expiry_minutes: URL validity duration (uses default if None)

        Returns:
            Signed URL (Azure: SAS, AWS: presigned, GCP: signed, Local: file:// or http://)

        Raises:
            Exception: If URL generation fails
        """
        pass

    @abstractmethod
    async def blob_exists(self, filename: str) -> bool:
        """
        Check if blob/file exists.

        Args:
            filename: Filename to check

        Returns:
            True if exists, False otherwise
        """
        pass

    @abstractmethod
    async def delete(self, filename: str) -> bool:
        """
        Delete blob/file.

        Args:
            filename: Filename to delete

        Returns:
            True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def download(self, filename: str) -> bytes:
        """
        Download blob/file content.

        Args:
            filename: Filename to download

        Returns:
            File content as bytes

        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If download fails
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get storage provider name (azure/aws/gcp/local)"""
        pass

    @property
    @abstractmethod
    def container_name(self) -> str:
        """Get container/bucket name"""
        pass
