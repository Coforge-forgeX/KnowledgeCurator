"""
Factory for creating storage adapters (Factory Pattern).

Provides simple interface for creating appropriate storage adapter based on configuration.
"""
import logging
from enum import Enum
from typing import Optional

from .protocols import StorageAdapter

logger = logging.getLogger(__name__)


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
        # Azure Blob Storage
        storage = StorageFactory.create(
            provider="azure",
            connection_string=conn_str,
            container_name="my-container"
        )

        # AWS S3
        storage = StorageFactory.create(
            provider="aws",
            bucket_name="my-bucket"
        )
    """

    @staticmethod
    def create(
        provider: str = "azure",
        connection_string: Optional[str] = None,
        container_name: Optional[str] = None,
        **kwargs
    ) -> StorageAdapter:
        """
        Create storage adapter based on provider.

        Args:
            provider: Storage provider ("azure", "aws", "gcp", "local")
            connection_string: Storage connection string (provider-specific)
            container_name: Container/bucket name
            **kwargs: Additional provider-specific arguments

        Returns:
            StorageAdapter instance

        Raises:
            ValueError: If provider is unknown or required config is missing
        """
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
            if not container_name:
                raise ValueError("container_name is required for Azure storage")

            from .adapters.azure_blob import AzureBlobStorageAdapter
            return AzureBlobStorageAdapter(
                connection_string=connection_string,
                container_name=container_name,
                **kwargs
            )

        elif provider_enum == StorageProvider.AWS:
            from .adapters.aws_s3 import AWSS3StorageAdapter
            return AWSS3StorageAdapter(**kwargs)

        elif provider_enum == StorageProvider.GCP:
            from .adapters.gcp_storage import GCPStorageAdapter
            return GCPStorageAdapter(**kwargs)

        elif provider_enum == StorageProvider.LOCAL:
            from .adapters.local_storage import LocalStorageAdapter
            return LocalStorageAdapter(**kwargs)

        else:
            # Should never reach here due to enum validation
            logger.error(f"Unhandled storage provider: {provider_enum}")
            raise ValueError(f"Unsupported provider: {provider}")


def get_storage_adapter(
    provider: str = "azure",
    connection_string: Optional[str] = None,
    container_name: Optional[str] = None,
    **kwargs
) -> StorageAdapter:
    """
    Get a storage adapter instance.

    Args:
        provider: Storage provider ("azure", "aws", "gcp", "local")
        connection_string: Storage connection string (provider-specific)
        container_name: Container/bucket name
        **kwargs: Additional provider-specific arguments

    Returns:
        StorageAdapter instance

    Example:
        storage = get_storage_adapter(
            provider="azure",
            connection_string=conn_str,
            container_name="my-container"
        )
        blob_info = await storage.upload("file.pdf", file_bytes)
    """
    return StorageFactory.create(
        provider=provider,
        connection_string=connection_string,
        container_name=container_name,
        **kwargs
    )
