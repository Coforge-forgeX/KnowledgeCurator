"""Queue adapter factory for indexer service.

Supports:
- Azure Storage Queue (legacy)
- Azure Service Bus (recommended for production)
"""
import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

# Add src and shared to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class AzureStorageQueueAdapter:
    """Azure Storage Queue adapter for indexing jobs (legacy)."""

    def __init__(self):
        from azure.storage.queue.aio import QueueClient

        self.provider_name = "azure_storage_queue"
        self.connection_string = settings.azure.AZURE_STORAGE_CONNECTION_STRING
        self.queue_name = settings.azure.INDEXING_QUEUE_NAME
        self.poll_interval = settings.azure.QUEUE_POLL_INTERVAL
        self.visibility_timeout = settings.MESSAGE_VISIBILITY_TIMEOUT

        # Initialize queue client
        self.queue_client: Optional[QueueClient] = None

    async def initialize(self):
        """Initialize the queue client."""
        from azure.storage.queue.aio import QueueClient

        if not self.queue_client:
            self.queue_client = QueueClient.from_connection_string(
                self.connection_string,
                self.queue_name,
            )
            # Ensure queue exists
            try:
                await self.queue_client.create_queue()
            except Exception:
                # Queue likely already exists
                pass

    async def receive_messages(
        self,
        max_messages: int = 10,
        visibility_timeout: int = 300,
        wait_time_seconds: int = 20,
    ) -> List[Any]:
        """Receive multiple messages from the queue.

        Args:
            max_messages: Maximum number of messages to receive
            visibility_timeout: How long messages should be hidden from other consumers (seconds)
            wait_time_seconds: Long polling wait time (not used in Azure, for API compatibility)

        Returns:
            List of message objects with .content and .receipt_handle attributes
        """
        if not self.queue_client:
            await self.initialize()

        try:
            messages = []
            async for msg in self.queue_client.receive_messages(
                visibility_timeout=visibility_timeout, max_messages=max_messages
            ):
                # Create a simple object with content and receipt_handle
                class Message:
                    def __init__(self, msg):
                        try:
                            self.content = json.loads(msg.content)
                        except json.JSONDecodeError:
                            self.content = {"raw": msg.content}
                        self.receipt_handle = f"{msg.id}:{msg.pop_receipt}"
                        self.message_id = msg.id
                        self.pop_receipt = msg.pop_receipt
                        self.dequeue_count = msg.dequeue_count

                messages.append(Message(msg))

                if len(messages) >= max_messages:
                    break

            if messages:
                logger.info(
                    f"Received {len(messages)} message(s) from queue '{self.queue_name}'"
                )
            return messages

        except Exception as e:
            logger.error(f"Error receiving messages from queue: {e}")
            return []

    async def delete_message(self, receipt_handle: str, pop_receipt: str = None):
        """Delete a message from the queue.

        Args:
            receipt_handle: Either "message_id:pop_receipt" or just message_id
            pop_receipt: Optional pop_receipt if not included in receipt_handle
        """
        if not self.queue_client:
            await self.initialize()

        try:
            # Parse receipt_handle if it contains both id and pop_receipt
            if ":" in receipt_handle and pop_receipt is None:
                message_id, pop_receipt = receipt_handle.split(":", 1)
            else:
                message_id = receipt_handle

            await self.queue_client.delete_message(message_id, pop_receipt)
            logger.debug(f"Deleted message {message_id} from queue")
        except Exception as e:
            logger.error(f"Error deleting message: {e}")

    async def move_to_dead_letter(
        self, receipt_handle: Any, reason: str = "MaxRetriesExceeded", error_description: str = None
    ) -> None:
        """Move a message to dead letter queue.

        Note: Azure Storage Queue does not have native DLQ support.
        This method deletes the message and logs it as a failed message.
        Consider using Azure Service Bus for proper DLQ support.

        Args:
            receipt_handle: Message receipt handle or message object
            reason: Reason for dead lettering
            error_description: Optional error description
        """
        logger.warning(
            f"Azure Storage Queue does not support DLQ - deleting message instead "
            f"(reason={reason}, description={error_description})"
        )

        # Extract message_id and pop_receipt from message object if available
        if hasattr(receipt_handle, 'message_id'):
            message_id = receipt_handle.message_id
            pop_receipt = receipt_handle.pop_receipt
            await self.delete_message(f"{message_id}:{pop_receipt}")
        else:
            await self.delete_message(receipt_handle)

    async def close(self):
        """Close the queue client."""
        if self.queue_client:
            await self.queue_client.close()
            self.queue_client = None


def get_queue_adapter():
    """Get queue adapter based on configuration.

    Returns appropriate adapter based on QUEUE_PROVIDER setting:
    - azure_service_bus: Azure Service Bus (recommended)
    - azure: Azure Storage Queue (legacy)
    """
    queue_provider = settings.active_queue_provider.lower()

    logger.info(
        "Initializing queue adapter",
        provider=queue_provider,
    )

    if queue_provider == "azure_service_bus":
        # Use shared Service Bus adapter directly
        from shared.adapters.queue import get_queue_adapter as _get_shared_adapter

        connection_string = (
            settings.azure.SERVICE_BUS_CONNECTION_STRING
            or settings.azure.AZURE_STORAGE_CONNECTION_STRING
        )

        topic_name = settings.azure.SERVICE_BUS_TOPIC_NAME
        subscription_name = settings.azure.SERVICE_BUS_SUBSCRIPTION_NAME
        queue_name = settings.azure.INDEXING_QUEUE_NAME

        logger.info(
            "Using shared Azure Service Bus adapter",
            topic=topic_name,
            subscription=subscription_name,
            queue=queue_name,
        )

        return _get_shared_adapter(
            provider="azure_service_bus",
            connection_string=connection_string,
            queue_name=queue_name if not topic_name else None,
            topic_name=topic_name,
            subscription_name=subscription_name,
            max_concurrent_calls=settings.MAX_CONCURRENT_JOBS,
            max_lock_renewal_duration=settings.MAX_LOCK_RENEWAL_DURATION,
        )
    else:
        logger.info("Using Azure Storage Queue adapter (legacy)")
        return AzureStorageQueueAdapter()
