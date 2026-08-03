"""
Cloud storage client wrapper (platform-agnostic).

This module provides a simplified interface to the storage abstraction layer,
maintaining backward compatibility with existing code while using the new
factory pattern underneath.
"""
from typing import Optional

from storage import BlobInfo, get_storage_adapter
from storage.protocols import StorageAdapter


class BlobClient:
    """
    Platform-agnostic storage client wrapper.

    This class wraps the storage adapter factory to maintain backward compatibility
    with existing code that uses BlobClient directly.
    """

    def __init__(self) -> None:
        """Initialize with storage adapter from factory"""
        self._storage: StorageAdapter = get_storage_adapter()

    @property
    def container_name(self) -> str:
        """Get container/bucket name"""
        return self._storage.container_name

    async def upload(
        self, filename: str, data: bytes, content_type: Optional[str] = None
    ) -> BlobInfo:
        """
        Upload file to cloud storage.

        Args:
            filename: Target filename
            data: File data as bytes
            content_type: MIME type

        Returns:
            BlobInfo with upload details
        """
        return await self._storage.upload(filename, data, content_type)

    async def generate_download_url(
        self, filename: str, expiry_minutes: Optional[int] = None
    ) -> str:
        """
        Generate signed download URL.

        Args:
            filename: Filename to generate URL for
            expiry_minutes: URL validity duration

        Returns:
            Signed URL
        """
        return await self._storage.generate_download_url(filename, expiry_minutes)

    async def blob_exists(self, filename: str) -> bool:
        """
        Check if file exists.

        Args:
            filename: Filename to check

        Returns:
            True if exists, False otherwise
        """
        return await self._storage.blob_exists(filename)

    async def delete(self, filename: str) -> bool:
        """
        Delete file.

        Args:
            filename: Filename to delete

        Returns:
            True if deleted, False if not found
        """
        return await self._storage.delete(filename)

    async def download(self, filename: str) -> bytes:
        """
        Download file content.

        Args:
            filename: Filename to download

        Returns:
            File content as bytes
        """
        return await self._storage.download(filename)

    @property
    def provider_name(self) -> str:
        """Get storage provider name"""
        return self._storage.provider_name


# Singleton instance
_client: Optional[BlobClient] = None


def get_blob_client() -> BlobClient:
    """
    Get singleton blob client.

    Returns:
        BlobClient instance (wraps storage adapter)
    """
    global _client
    if _client is None:
        _client = BlobClient()
    return _client
