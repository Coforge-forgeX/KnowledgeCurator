"""Queue adapters - using shared implementation

All queue operations now use the shared adapters from services/shared/adapters/queue.
This module provides a configured get_queue_adapter() that automatically uses
indexer-service settings.
"""
from typing import Optional

from shared.adapters.queue import (
    get_queue_adapter as _get_shared_adapter,
    QueueAdapter,
    QueueMessage,
    QueueProvider,
)

# Singleton instance
_queue_adapter: Optional[QueueAdapter] = None


def get_queue_adapter(
    force_recreate: bool = False, queue_name: Optional[str] = None
) -> QueueAdapter:
    """
    Get queue adapter configured with indexer-service settings.

    Args:
        force_recreate: If True, recreate the adapter even if one exists
        queue_name: Queue name (optional, uses default from settings)

    Returns:
        QueueAdapter instance configured from settings

    Example:
        queue = get_queue_adapter()
        message_id = await queue.send_message({"job_id": "123"})
    """
    global _queue_adapter

    if _queue_adapter is None or force_recreate:
        from core.config import settings

        # Select appropriate connection string based on provider
        provider = settings.active_queue_provider
        if provider == "azure_service_bus":
            connection_string = settings.azure.SERVICE_BUS_CONNECTION_STRING
            kwargs = {
                "topic_name": settings.azure.SERVICE_BUS_TOPIC_NAME,
                "subscription_name": settings.azure.SERVICE_BUS_SUBSCRIPTION_NAME,
                "max_lock_renewal_duration": settings.MAX_LOCK_RENEWAL_DURATION,
            }
        else:
            connection_string = settings.azure.AZURE_STORAGE_CONNECTION_STRING
            kwargs = {}

        _queue_adapter = _get_shared_adapter(
            provider=provider,
            connection_string=connection_string,
            queue_name=queue_name or settings.azure.INDEXING_QUEUE_NAME,
            **kwargs,
        )

    return _queue_adapter


__all__ = [
    "get_queue_adapter",
    "QueueAdapter",
    "QueueMessage",
    "QueueProvider",
]
