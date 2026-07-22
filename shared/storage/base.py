"""
Base Storage Adapter Interface

Defines the contract for cloud storage adapters (Azure, AWS, GCP).
Follows the Adapter Pattern for pluggable storage backends.
"""
from abc import ABC, abstractmethod
from typing import Any, BinaryIO, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class BlobMetadata:
    """Metadata for a blob/file in storage"""
    name: str
    path: str
    size: int
    content_type: str
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, str]] = None
    etag: Optional[str] = None


@dataclass
class BlobContent:
    """Content and metadata for a downloaded blob"""
    data: bytes
    metadata: BlobMetadata


class StorageAdapter(ABC):
    """
    Abstract base class for cloud storage adapters.

    Implementations:
    - AzureBlobStorageAdapter (Azure Blob Storage)
    - S3StorageAdapter (AWS S3)
    - GCSStorageAdapter (GCP Cloud Storage)

    Design Pattern: Adapter Pattern
    Purpose: Provide unified interface for different cloud storage providers
    """

    def __init__(self, container_name: str, **config):
        """
        Initialize storage adapter.

        Args:
            container_name: Container/bucket name
            **config: Provider-specific configuration
        """
        self.container_name = container_name
        self.config = config

    @abstractmethod
    async def upload_file(
        self,
        file_path: str,
        content: bytes,
        content_type: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        overwrite: bool = True,
    ) -> BlobMetadata:
        """
        Upload a file to storage.

        Args:
            file_path: Path within container (e.g., "workspace_1/doc.pdf")
            content: File content as bytes
            content_type: MIME type
            metadata: Custom metadata
            overwrite: Whether to overwrite existing file

        Returns:
            BlobMetadata: Uploaded file metadata
        """
        pass

    @abstractmethod
    async def download_file(
        self,
        file_path: str,
    ) -> BlobContent:
        """
        Download a file from storage.

        Args:
            file_path: Path within container

        Returns:
            BlobContent: Downloaded file content and metadata
        """
        pass

    @abstractmethod
    async def delete_file(
        self,
        file_path: str,
    ) -> bool:
        """
        Delete a file from storage.

        Args:
            file_path: Path within container

        Returns:
            bool: True if deleted, False if not found
        """
        pass

    @abstractmethod
    async def list_files(
        self,
        prefix: Optional[str] = None,
        max_results: Optional[int] = None,
    ) -> List[BlobMetadata]:
        """
        List files in storage.

        Args:
            prefix: Filter by path prefix
            max_results: Maximum number of results

        Returns:
            List[BlobMetadata]: List of file metadata
        """
        pass

    @abstractmethod
    async def file_exists(
        self,
        file_path: str,
    ) -> bool:
        """
        Check if a file exists in storage.

        Args:
            file_path: Path within container

        Returns:
            bool: True if exists, False otherwise
        """
        pass

    @abstractmethod
    async def get_file_url(
        self,
        file_path: str,
        expiry_seconds: int = 3600,
    ) -> str:
        """
        Generate a signed URL for direct file access.

        Args:
            file_path: Path within container
            expiry_seconds: URL validity duration

        Returns:
            str: Signed URL
        """
        pass

    @abstractmethod
    async def copy_file(
        self,
        source_path: str,
        destination_path: str,
    ) -> BlobMetadata:
        """
        Copy a file within storage.

        Args:
            source_path: Source file path
            destination_path: Destination file path

        Returns:
            BlobMetadata: Destination file metadata
        """
        pass

    @abstractmethod
    async def get_file_metadata(
        self,
        file_path: str,
    ) -> BlobMetadata:
        """
        Get metadata for a file without downloading it.

        Args:
            file_path: Path within container

        Returns:
            BlobMetadata: File metadata
        """
        pass

    async def close(self) -> None:
        """
        Close connections and cleanup resources.
        Override if adapter needs cleanup.
        """
        pass
