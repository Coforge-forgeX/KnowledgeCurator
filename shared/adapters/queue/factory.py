"""
Factory for creating queue adapters (Factory Pattern).

Provides simple interface for creating appropriate queue adapter based on configuration.
"""
from enum import Enum
from typing import Optional
import logging

def get_logger(name: str):
    return logging.getLogger(name)

from .protocols import QueueAdapter

logger = get_logger(__name__)


class QueueProvider(str, Enum):
    """Supported queue providers"""

    AZURE = "azure"
    AZURE_SERVICE_BUS = "azure_service_bus"
    AWS = "aws"
    REDIS = "redis"


class QueueFactory:
    """
    Factory for creating queue adapters.

    Usage:
        # Azure Storage Queue
        queue = QueueFactory.create(
            provider="azure",
            connection_string=conn_str,
            queue_name="my-queue"
        )

        # AWS SQS
        queue = QueueFactory.create(
            provider="aws",
            queue_url="https://sqs..."
        )
    """

    @staticmethod
    def create(
        provider: str = "azure",
        connection_string: Optional[str] = None,
        queue_name: Optional[str] = None,
        **kwargs
    ) -> QueueAdapter:
        """
        Create queue adapter based on provider.

        Args:
            provider: Queue provider ("azure", "aws", "redis")
            connection_string: Queue connection string (provider-specific)
            queue_name: Queue name
            **kwargs: Additional provider-specific arguments

        Returns:
            QueueAdapter instance

        Raises:
            ValueError: If provider is unknown or required config is missing
        """
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
            if not queue_name:
                raise ValueError("queue_name is required for Azure queue")

            from .adapters.azure_queue import AzureQueueAdapter
            return AzureQueueAdapter(
                connection_string=connection_string,
                queue_name=queue_name,
            )

        elif provider_enum == QueueProvider.AZURE_SERVICE_BUS:
            topic_name = kwargs.get("topic_name")
            subscription_name = kwargs.get("subscription_name")
            max_concurrent_calls = kwargs.get("max_concurrent_calls", 1)
            max_lock_renewal_duration = kwargs.get("max_lock_renewal_duration", 1800)

            if not queue_name and not topic_name:
                raise ValueError("Either queue_name or topic_name is required for Azure Service Bus")

            from .adapters.azure_service_bus import AzureServiceBusAdapter
            return AzureServiceBusAdapter(
                connection_string=connection_string,
                queue_name=queue_name,
                topic_name=topic_name,
                subscription_name=subscription_name,
                max_concurrent_calls=max_concurrent_calls,
                max_lock_renewal_duration=max_lock_renewal_duration,
            )

        elif provider_enum == QueueProvider.AWS:
            from .adapters.aws_sqs import AWSSQSAdapter
            return AWSSQSAdapter(
                connection_string=connection_string,
                queue_name=queue_name,
                queue_url=kwargs.get("queue_url"),
                region_name=kwargs.get("region_name") or kwargs.get("aws_region"),
            )

        elif provider_enum == QueueProvider.REDIS:
            from .adapters.redis_queue import RedisQueueAdapter
            return RedisQueueAdapter(
                connection_string=connection_string,
                queue_name=queue_name,
                **kwargs,
            )

        else:
            # Should never reach here due to enum validation
            logger.error(f"Unhandled queue provider: {provider_enum}")
            raise ValueError(f"Unsupported provider: {provider}")


def get_queue_adapter(
    provider: str = "azure",
    connection_string: Optional[str] = None,
    queue_name: Optional[str] = None,
    **kwargs
) -> QueueAdapter:
    """
    Get a queue adapter instance.

    Args:
        provider: Queue provider ("azure", "aws", "redis")
        connection_string: Queue connection string (provider-specific)
        queue_name: Queue name
        **kwargs: Additional provider-specific arguments

    Returns:
        QueueAdapter instance

    Example:
        queue = get_queue_adapter(
            provider="azure",
            connection_string=conn_str,
            queue_name="my-queue"
        )
        message_id = await queue.send_message({"job_id": "123"})
    """
    return QueueFactory.create(
        provider=provider,
        connection_string=connection_string,
        queue_name=queue_name,
        **kwargs
    )
