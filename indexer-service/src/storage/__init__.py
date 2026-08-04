"""Storage adapters - using shared implementation

All storage operations now use the shared adapters from services/shared/adapters/storage.
This module provides a configured get_storage_adapter() that automatically uses
indexer-service settings.
"""
from typing import Optional

from shared.adapters.storage import (
    get_storage_adapter as _get_shared_adapter,
    StorageAdapter,
    BlobInfo,
    StorageProvider,
)

# Singleton instance
_storage_adapter: Optional[StorageAdapter] = None


def get_storage_adapter(force_recreate: bool = False) -> StorageAdapter:
    """
    Get storage adapter configured with indexer-service settings.

    Args:
        force_recreate: If True, recreate the adapter even if one exists

    Returns:
        StorageAdapter instance configured from settings

    Example:
        storage = get_storage_adapter()
        blob_info = await storage.upload("file.pdf", file_bytes)
    """
    global _storage_adapter

    if _storage_adapter is None or force_recreate:
        from src.core.config import settings

        _storage_adapter = _get_shared_adapter(
            provider=settings.storage.STORAGE_PROVIDER or "azure",
            connection_string=settings.storage.AZURE_BLOB_STORAGE_CONNECTION_STRING,
            container_name=settings.storage.STORAGE_CONTAINER_NAME,
        )

    return _storage_adapter


__all__ = [
    "get_storage_adapter",
    "StorageAdapter",
    "BlobInfo",
    "StorageProvider",
]
