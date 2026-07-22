"""
Storage Adapter Factory

Factory for creating storage adapter instances based on configuration.
Implements the Factory Pattern for storage adapter creation.
"""
from enum import Enum
from typing import Dict, Optional

from core.config import settings
from core.exceptions import ConfigurationException
from core.logging import get_logger

from .azure_adapter import AzureBlobStorageAdapter
from .base import StorageAdapter
from .s3_adapter import S3StorageAdapter

logger = get_logger(__name__)


class StorageProvider(str, Enum):
    """Supported storage providers"""
    AZURE = "azure"
    AWS = "aws"
    GCP = "gcp"


class StorageAdapterFactory:
    """
    Factory for creating storage adapters.

    Design Pattern: Factory Pattern
    Purpose: Centralize storage adapter creation and configuration
    """

    @staticmethod
    def create_adapter(
        provider: StorageProvider = StorageProvider.AZURE,
        container_name: Optional[str] = None,
        **config
    ) -> StorageAdapter:
        """
        Create a storage adapter instance.

        Args:
            provider: Storage provider (azure, aws, gcp)
            container_name: Container/bucket name
            **config: Provider-specific configuration

        Returns:
            StorageAdapter: Configured storage adapter instance

        Raises:
            ConfigurationException: If provider is invalid or config is missing
        """
        # Get container name from config
        if not container_name:
            container_name = settings.storage.STORAGE_CONTAINER_NAME
            if not container_name:
                raise ConfigurationException(
                    message="Container name not provided",
                    config_key="STORAGE_CONTAINER_NAME",
                )

        logger.info(
            "Creating storage adapter",
            provider=provider,
            container_name=container_name,
        )

        if provider == StorageProvider.AZURE:
            return AzureBlobStorageAdapter(
                container_name=container_name,
                **config
            )

        elif provider == StorageProvider.AWS:
            return S3StorageAdapter(
                container_name=container_name,
                **config
            )

        elif provider == StorageProvider.GCP:
            # GCP adapter not yet implemented
            # from .gcs_adapter import GCSStorageAdapter
            # return GCSStorageAdapter(container_name=container_name, **config)
            raise ConfigurationException(
                message="GCP storage adapter not yet implemented",
                config_key="STORAGE_PROVIDER",
            )

        else:
            raise ConfigurationException(
                message=f"Unknown storage provider: {provider}",
                config_key="STORAGE_PROVIDER",
            )

    @staticmethod
    def create_from_config() -> StorageAdapter:
        """
        Create storage adapter from configuration settings.

        Uses settings.storage for all configuration:
            - STORAGE_PROVIDER: azure, aws, or gcp (default: azure)
            - STORAGE_CONTAINER_NAME: Container/bucket name
            - Provider-specific credentials

        Returns:
            StorageAdapter: Configured storage adapter
        """
        provider_str = settings.storage.STORAGE_PROVIDER.lower()

        try:
            provider = StorageProvider(provider_str)
        except ValueError:
            raise ConfigurationException(
                message=f"Invalid storage provider: {provider_str}",
                config_key="STORAGE_PROVIDER",
            )

        container_name = settings.storage.STORAGE_CONTAINER_NAME

        logger.info(
            "Creating storage adapter from config",
            provider=provider,
            container_name=container_name,
        )

        return StorageAdapterFactory.create_adapter(
            provider=provider,
            container_name=container_name,
        )


# Convenience function
def get_storage_adapter(
    provider: Optional[str] = None,
    container_name: Optional[str] = None,
    **config
) -> StorageAdapter:
    """
    Get a storage adapter instance.

    Args:
        provider: Storage provider (azure, aws, gcp). If None, uses config.
        container_name: Container/bucket name. If None, uses config.
        **config: Provider-specific configuration

    Returns:
        StorageAdapter: Configured storage adapter
    """
    if provider is None:
        return StorageAdapterFactory.create_from_config()

    try:
        provider_enum = StorageProvider(provider.lower())
    except ValueError:
        raise ConfigurationException(
            message=f"Invalid storage provider: {provider}",
            config_key="provider",
        )

    return StorageAdapterFactory.create_adapter(
        provider=provider_enum,
        container_name=container_name,
        **config
    )
