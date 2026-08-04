"""Azure Storage Queue adapter"""

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional


def get_logger(name: str):
    return logging.getLogger(name)

from azure.storage.queue import QueueServiceClient, QueueClient
from azure.core.exceptions import ResourceExistsError, ResourceNotFoundError

from ..models import QueueMessage
from ..protocols import QueueAdapter

logger = get_logger(__name__)


class AzureQueueAdapter(QueueAdapter):
    """Azure Storage Queue implementation"""

    def __init__(
        self,
        connection_string: Optional[str] = None,
        queue_name: Optional[str] = None
    ):
        """
        Initialize Azure Queue adapter.

        Args:
            connection_string: Azure Storage connection string
            queue_name: Queue name
        """
        if not connection_string:
            raise ValueError("Azure Storage connection string is required")
        if not queue_name:
            raise ValueError("Queue name is required")

        self.connection_string = connection_string
        self._queue_name = queue_name

        if not self.connection_string:
            raise ValueError("Azure Storage connection string not configured")

        # Initialize queue service client
        self.queue_service_client = QueueServiceClient.from_connection_string(
            self.connection_string
        )
        self.queue_client = self.queue_service_client.get_queue_client(
            self._queue_name
        )

        # Ensure the queue exists for local/dev runs (e.g., Azurite).
        try:
            self.queue_client.create_queue()
        except ResourceExistsError:
            pass

        logger.info(f"Azure Queue adapter initialized (queue={self._queue_name})")

    async def send_message(
        self, message: Dict[str, Any], delay_seconds: int = 0
    ) -> str:
        """Send message to Azure Storage Queue"""
        try:
            # Encode message as JSON
            message_text = json.dumps(message)

            # Send message
            result = await asyncio.to_thread(
                self.queue_client.send_message,
                message_text,
                visibility_timeout=delay_seconds if delay_seconds > 0 else None
            )

            message_id = result.id

            logger.info(
                f"Message sent to Azure Queue (message_id={message_id}, queue={self._queue_name})"
            )

            return message_id

        except Exception as e:
            logger.error(
                f"Failed to send message to Azure Queue: {str(e)}",
                exc_info=True
            )
            raise

    async def receive_messages(
        self,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> List[QueueMessage]:
        """Receive messages from Azure Storage Queue"""
        try:
            # Azure Queue doesn't support long polling directly
            # We can simulate it with a loop if wait_time_seconds > 0
            messages_raw = await asyncio.to_thread(
                self.queue_client.receive_messages,
                messages_per_page=max_messages,
                visibility_timeout=visibility_timeout
            )

            messages = []
            for msg in messages_raw:
                try:
                    # Decode JSON content
                    content = json.loads(msg.content)

                    messages.append(
                        QueueMessage(
                            content=content,
                            message_id=msg.id,
                            receipt_handle=f"{msg.id}:{msg.pop_receipt}"
                        )
                    )
                except json.JSONDecodeError as e:
                    logger.warning(
                        f"Failed to decode message content (message_id={msg.id}, error={str(e)})"
                    )
                    # Skip malformed messages
                    continue

            if messages:
                logger.info(
                    f"Messages received from Azure Queue (count={len(messages)}, queue={self._queue_name})"
                )

            return messages

        except Exception as e:
            logger.error(
                "Failed to receive messages from Azure Queue",
                error=str(e),
                exc_info=True
            )
            raise

    async def delete_message(self, receipt_handle: str) -> bool:
        """Delete message from Azure Storage Queue"""
        try:
            # Parse receipt handle (format: "message_id:pop_receipt")
            # Azure uses message_id and pop_receipt for deletion
            if ":" in receipt_handle:
                message_id, pop_receipt = receipt_handle.split(":", 1)
            else:
                # Assume the entire string is the pop_receipt
                # (This is a simplified approach; actual implementation may vary)
                pop_receipt = receipt_handle
                message_id = None

            if not message_id:
                logger.error("Missing Azure message_id in receipt_handle")
                return False

            await asyncio.to_thread(
                self.queue_client.delete_message,
                message_id,
                pop_receipt,
            )

            logger.debug(f"Message deleted from Azure Queue (receipt_handle={receipt_handle})")
            return True

        except ResourceNotFoundError:
            logger.warning(f"Message not found for deletion (receipt_handle={receipt_handle})")
            return False
        except Exception as e:
            logger.error(
                f"Failed to delete message from Azure Queue (error={str(e)}, receipt_handle={receipt_handle})"
            )
            return False

    async def get_queue_size(self) -> int:
        """Get approximate number of messages in queue"""
        try:
            properties = await asyncio.to_thread(
                self.queue_client.get_queue_properties
            )
            return properties.approximate_message_count or 0

        except Exception as e:
            logger.error(f"Failed to get queue size: {str(e)}")
            return 0

    async def purge_queue(self) -> bool:
        """Delete all messages in queue"""
        try:
            await asyncio.to_thread(
                self.queue_client.clear_messages
            )

            logger.info(f"Queue purged (queue={self._queue_name})")
            return True

        except Exception as e:
            logger.error(f"Failed to purge queue: {str(e)}")
            return False

    @property
    def provider_name(self) -> str:
        return "azure"

    @property
    def queue_name(self) -> str:
        return self._queue_name
