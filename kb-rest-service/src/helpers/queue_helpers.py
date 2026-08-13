"""Queue helper functions - provider-agnostic wrapper around queue adapters"""
from typing import Any, Dict

from src.core.config import settings
from src.core.exceptions import QueueException
from src.core.logging import get_logger
from src.queue_adapters import get_queue_adapter

logger = get_logger(__name__)


def get_indexing_queue_helper():
    """
    Get queue adapter for indexing queue.

    Returns:
        QueueAdapter: Configured queue adapter for indexing (provider-agnostic)

    Note:
        This function returns the shared queue adapter which automatically
        uses the correct provider (Azure Queue, Azure Service Bus, AWS SQS, or Redis)
        based on the QUEUE_PROVIDER setting.
    """
    try:
        queue_name = settings.active_queue_name
        adapter = get_queue_adapter(queue_name=queue_name)
        logger.info(
            "Indexing queue adapter initialized",
            provider=settings.active_queue_provider,
            queue_name=queue_name,
        )
        return adapter
    except Exception as e:
        logger.error(
            "Failed to initialize indexing queue adapter",
            error=str(e),
            provider=settings.active_queue_provider,
        )
        raise QueueException(
            message=f"Failed to initialize queue adapter: {str(e)}",
            operation="initialize",
        ) from e
