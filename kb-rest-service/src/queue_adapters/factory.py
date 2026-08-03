"""
Factory for creating queue adapters (Factory Pattern).

Provides simple interface for creating appropriate queue adapter based on configuration.
"""
import logging
from enum import Enum
from typing import Optional

from core.config import settings
from core.logging import get_logger

from .protocols import QueueAdapter

logger = get_logger(__name__)


class QueueProvider(str, Enum):
    """Supported queue providers"""

    AZURE = "azure"
    AWS = "aws"
    REDIS = "redis"


class QueueFactory:
    """
    Factory for creating queue adapters.

    Usage:
        # Azure Storage Queue (default for production)
        queue = QueueFactory.create()

        # AWS SQS
        queue = QueueFactory.create(provider="aws")

        # Redis Queue
        queue = QueueFactory.create(provider="redis")
    """

    @staticmethod
    def create(provider: Optional[str] = None) -> QueueAdapter:
        """
        Create queue adapter based on provider.

        Args:
            provider: Queue provider ("azure", "aws", "redis")
                     If None, uses QUEUE_PROVIDER or CLOUD_PROVIDER env var

        Returns:
            QueueAdapter instance

        Raises:
            ValueError: If provider is unknown
            ConfigurationException: If provider credentials are not configured

        Examples:
            # Azure (production)
            queue = QueueFactory.create("azure")

            # AWS SQS
            queue = QueueFactory.create("aws")

            # Redis
            queue = QueueFactory.create("redis")
        """
        # Determine provider from parameter or settings
        if provider is None:
            provider = settings.QUEUE_PROVIDER or settings.CLOUD_PROVIDER

        # Normalize provider name
        provider = provider.lower().strip()

        # Convert string to enum
        try:
            provider_enum = QueueProvider(provider)
        except ValueError:
            logger.warning(
                f"Unknown queue provider '{provider}', defaulting to Azure"
            )
            provider_enum = QueueProvider.AZURE

        logger.info(f"Creating queue adapter for provider: {provider_enum.value}")

        # Create adapter based on provider
        if provider_enum == QueueProvider.AZURE:
            from .adapters.azure_queue import AzureQueueAdapter
            return AzureQueueAdapter()

        elif provider_enum == QueueProvider.AWS:
            from .adapters.aws_sqs import AWSSQSAdapter
            return AWSSQSAdapter()

        elif provider_enum == QueueProvider.REDIS:
            from .adapters.redis_queue import RedisQueueAdapter
            return RedisQueueAdapter()

        else:
            # Should never reach here due to enum validation
            logger.error(f"Unhandled queue provider: {provider_enum}")
            from .adapters.azure_queue import AzureQueueAdapter
            return AzureQueueAdapter()

    @staticmethod
    def create_from_env() -> QueueAdapter:
        """
        Create queue adapter from environment variables.

        Environment variables:
            CLOUD_PROVIDER or QUEUE_PROVIDER: "azure", "aws", or "redis"

        Returns:
            Configured QueueAdapter

        Example:
            # Set in environment: CLOUD_PROVIDER=aws
            queue = QueueFactory.create_from_env()
        """
        return QueueFactory.create()


# Singleton instance for global access (lazy initialization)
_queue_adapter: Optional[QueueAdapter] = None


def get_queue_adapter(force_recreate: bool = False) -> QueueAdapter:
    """
    Get singleton queue adapter instance.

    Args:
        force_recreate: If True, recreate the adapter even if one exists

    Returns:
        QueueAdapter instance

    Example:
        queue = get_queue_adapter()
        message_id = await queue.send_message({"job_id": "123", "file": "doc.pdf"})
    """
    global _queue_adapter

    if _queue_adapter is None or force_recreate:
        _queue_adapter = QueueFactory.create_from_env()

    return _queue_adapter
