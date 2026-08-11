"""
Backward compatibility wrapper for storage adapter.

This module provides backward compatibility for code that imports from storage_adapter.
New code should import directly from shared adapters.

Deprecated: Use shared adapters directly.
"""
from shared.adapters.storage import (
    get_storage_adapter as _get_storage_adapter,
    StorageAdapter,
)


def get_storage_adapter():
    """Get storage adapter configured with indexer-service settings"""
    from src.core.config import settings

    connection_string = (
        settings.storage.AZURE_BLOB_STORAGE_CONNECTION_STRING
        or settings.azure.AZURE_STORAGE_CONNECTION_STRING
    )

    return _get_storage_adapter(
        provider=settings.storage.STORAGE_PROVIDER or "azure",
        connection_string=connection_string,
        container_name=settings.storage.STORAGE_CONTAINER_NAME,
    )


# Alias for backward compatibility
AzureBlobAdapter = StorageAdapter

__all__ = [
    "get_storage_adapter",
    "StorageAdapter",
    "AzureBlobAdapter",
]
