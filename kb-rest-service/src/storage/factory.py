"""
Factory for creating storage adapters (Factory Pattern).

Provides simple interface for creating appropriate storage adapter based on configuration.
"""
import logging
from enum import Enum
from typing import Optional

from core.config import settings
from core.logging import get_logger

from .protocols import StorageAdapter

logger = get_logger(__name__)


class StorageProvider(str, Enum):
    """Supported storage providers"""

    AZURE = "azure"
    AWS = "aws"
    GCP = "gcp"
    LOCAL = "local"


class StorageFactory:
    """
    Factory for creating storage adapters.

    Usage:
        # Azure Blob Storage (default for production)
        storage = StorageFactory.create()

        # AWS S3
        storage = StorageFactory.create(provider="aws")

        # GCP Cloud Storage
        storage = StorageFactory.create(provider="gcp")

        # Local filesystem (for development)
        storage = StorageFactory.create(provider="local")
    """

    @staticmethod
    def create(provider: Optional[str] = None) -> StorageAdapter:
        """
        Create storage adapter based on provider.

        Args:
            provider: Storage provider ("azure", "aws", "gcp", "local")
                     If None, uses CLOUD_PROVIDER or STORAGE_PROVIDER env var

        Returns:
            StorageAdapter instance

        Raises:
            ValueError: If provider is unknown
            ConfigurationException: If provider credentials are not configured

        Examples:
            # Azure (production)
            storage = StorageFactory.create("azure")

            # AWS S3
            storage = StorageFactory.create("aws")

            # GCP Cloud Storage
            storage = StorageFactory.create("gcp")

            # Local (development)
            storage = StorageFactory.create("local")
        """
        # Determine provider from parameter or settings
        if provider is None:
            provider = settings.STORAGE_PROVIDER or settings.CLOUD_PROVIDER

        # Normalize provider name
        provider = provider.lower().strip()

        # Convert string to enum
        try:
            provider_enum = StorageProvider(provider)
        except ValueError:
            logger.warning(
                f"Unknown storage provider '{provider}', defaulting to Azure"
            )
            provider_enum = StorageProvider.AZURE

        logger.info(f"Creating storage adapter for provider: {provider_enum.value}")

        # Create adapter based on provider
        if provider_enum == StorageProvider.AZURE:
            from .adapters.azure_blob import AzureBlobStorageAdapter
            return AzureBlobStorageAdapter()

        elif provider_enum == StorageProvider.AWS:
            from .adapters.aws_s3 import AWSS3StorageAdapter
            return AWSS3StorageAdapter()

        elif provider_enum == StorageProvider.GCP:
            from .adapters.gcp_storage import GCPStorageAdapter
            return GCPStorageAdapter()

        elif provider_enum == StorageProvider.LOCAL:
            from .adapters.local_storage import LocalStorageAdapter
            return LocalStorageAdapter()

        else:
            # Should never reach here due to enum validation
            logger.error(f"Unhandled storage provider: {provider_enum}")
            from .adapters.azure_blob import AzureBlobStorageAdapter
            return AzureBlobStorageAdapter()

    @staticmethod
    def create_from_env() -> StorageAdapter:
        """
        Create storage adapter from environment variables.

        Environment variables:
            CLOUD_PROVIDER or STORAGE_PROVIDER: "azure", "aws", "gcp", or "local"

        Returns:
            Configured StorageAdapter

        Example:
            # Set in environment: CLOUD_PROVIDER=aws
            storage = StorageFactory.create_from_env()
        """
        return StorageFactory.create()


# Singleton instance for global access (lazy initialization)
_storage_adapter: Optional[StorageAdapter] = None


def get_storage_adapter(force_recreate: bool = False) -> StorageAdapter:
    """
    Get singleton storage adapter instance.

    Args:
        force_recreate: If True, recreate the adapter even if one exists

    Returns:
        StorageAdapter instance

    Example:
        storage = get_storage_adapter()
        blob_info = await storage.upload("file.pdf", file_bytes)
    """
    global _storage_adapter

    if _storage_adapter is None or force_recreate:
        _storage_adapter = StorageFactory.create_from_env()

    return _storage_adapter
