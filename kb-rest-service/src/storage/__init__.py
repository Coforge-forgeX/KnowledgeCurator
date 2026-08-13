"""Storage adapters - using shared implementation

All storage operations now use the shared adapters from services/shared/adapters/storage.
This module provides a configured get_storage_adapter() that automatically uses
kb-rest-service settings.
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


def get_storage_adapter(
    force_recreate: bool = False,
    container_override: Optional[str] = None
) -> StorageAdapter:
    """
    Get storage adapter configured with kb-rest-service settings.

    Args:
        force_recreate: If True, recreate the adapter even if one exists
        container_override: Override the default container name (e.g., for workspace-specific uploads)

    Returns:
        StorageAdapter instance configured from settings

    Example:
        storage = get_storage_adapter()
        blob_info = await storage.upload("file.pdf", file_bytes)

        # With container override:
        storage = get_storage_adapter(container_override="workspace")
    """
    global _storage_adapter

    # When using container override, create a new instance without caching
    use_cache = container_override is None

    if use_cache and _storage_adapter is not None and not force_recreate:
        return _storage_adapter

    from src.core.config import settings

    provider = (settings.storage.STORAGE_PROVIDER or settings.CLOUD_PROVIDER).lower()

    # Build kwargs based on provider
    kwargs = {}

    if provider == "azure":
        kwargs["connection_string"] = settings.storage.AZURE_STORAGE_CONNECTION_STRING
        kwargs["container_name"] = container_override or settings.storage.STORAGE_CONTAINER_NAME
    elif provider == "aws":
        kwargs["bucket_name"] = container_override or settings.storage.S3_BUCKET_NAME
        kwargs["region_name"] = settings.storage.AWS_REGION
        kwargs["aws_access_key_id"] = settings.storage.AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = settings.storage.AWS_SECRET_ACCESS_KEY
        if settings.storage.S3_PATH_PREFIX:
            kwargs["path_prefix"] = settings.storage.S3_PATH_PREFIX
    elif provider == "gcp":
        kwargs["bucket_name"] = container_override or settings.storage.GCS_BUCKET_NAME
        kwargs["project_id"] = settings.storage.GCP_PROJECT_ID
        if settings.storage.GCP_CREDENTIALS_PATH:
            kwargs["credentials_path"] = settings.storage.GCP_CREDENTIALS_PATH
        if settings.storage.GCS_PATH_PREFIX:
            kwargs["path_prefix"] = settings.storage.GCS_PATH_PREFIX
    elif provider == "local":
        kwargs["base_path"] = settings.storage.LOCAL_STORAGE_PATH
        if settings.storage.LOCAL_STORAGE_PATH_PREFIX:
            kwargs["path_prefix"] = settings.storage.LOCAL_STORAGE_PATH_PREFIX
    else:
        # Default to Azure with container name
        kwargs["container_name"] = container_override or settings.storage.STORAGE_CONTAINER_NAME
        if settings.storage.AZURE_STORAGE_CONNECTION_STRING:
            kwargs["connection_string"] = settings.storage.AZURE_STORAGE_CONNECTION_STRING

    adapter = _get_shared_adapter(provider=provider, **kwargs)

    # Only cache if not using container override
    if use_cache:
        _storage_adapter = adapter

    return adapter


__all__ = [
    "get_storage_adapter",
    "StorageAdapter",
    "BlobInfo",
    "StorageProvider",
]
