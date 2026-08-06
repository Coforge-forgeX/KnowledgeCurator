"""Queue adapters - using shared implementation

All queue operations now use the shared adapters from services/shared/adapters/queue.
This module provides a configured get_queue_adapter() that automatically uses
kb-rest-service settings.

Supports:
- Azure Storage Queue (legacy)
- Azure Service Bus (recommended for production)
- AWS SQS
- Redis Queue
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
    Get queue adapter configured with kb-rest-service settings.

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
        from src.core.config import settings

        queue_provider = settings.active_queue_provider.lower()

        # Build kwargs for provider-specific options
        kwargs = {}

        # Service Bus specific
        if queue_provider == "azure_service_bus":
            topic_name = getattr(settings.azure, "SERVICE_BUS_TOPIC_NAME", None)
            if topic_name:
                kwargs["topic_name"] = topic_name

        # AWS SQS specific
        elif queue_provider == "aws":
            kwargs["queue_url"] = settings.SQS_QUEUE_URL
            kwargs["region_name"] = settings.AWS_REGION

        _queue_adapter = _get_shared_adapter(
            provider=queue_provider,
            connection_string=settings.active_queue_connection,
            queue_name=queue_name or settings.active_queue_name,
            **kwargs
        )

    return _queue_adapter


__all__ = [
    "get_queue_adapter",
    "QueueAdapter",
    "QueueMessage",
    "QueueProvider",
]
