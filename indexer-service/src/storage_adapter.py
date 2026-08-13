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
    """Get storage adapter configured with indexer-service settings (provider-agnostic)"""
    from src.core.config import settings

    # Determine provider from settings
    provider = (settings.storage.STORAGE_PROVIDER or settings.CLOUD_PROVIDER).lower()

    # Build kwargs based on provider
    kwargs = {}
    if provider == "azure":
        connection_string = (
            settings.storage.AZURE_BLOB_STORAGE_CONNECTION_STRING
            or settings.azure.AZURE_STORAGE_CONNECTION_STRING
        )
        kwargs["connection_string"] = connection_string
        kwargs["container_name"] = settings.storage.STORAGE_CONTAINER_NAME
    elif provider == "aws":
        kwargs["bucket_name"] = settings.storage.S3_BUCKET_NAME
        kwargs["region_name"] = settings.storage.AWS_REGION
        kwargs["access_key_id"] = settings.storage.AWS_ACCESS_KEY_ID
        kwargs["secret_access_key"] = settings.storage.AWS_SECRET_ACCESS_KEY
    elif provider == "gcp":
        kwargs["bucket_name"] = settings.storage.GCS_BUCKET_NAME
        kwargs["project_id"] = settings.storage.GCP_PROJECT_ID
        kwargs["credentials_path"] = settings.storage.GCP_CREDENTIALS_PATH
    elif provider == "local":
        pass
    else:
        raise ValueError(f"Unsupported storage provider: {provider}")

    return _get_storage_adapter(provider=provider, **kwargs)


# Alias for backward compatibility
AzureBlobAdapter = StorageAdapter

__all__ = [
    "get_storage_adapter",
    "StorageAdapter",
    "AzureBlobAdapter",
]
