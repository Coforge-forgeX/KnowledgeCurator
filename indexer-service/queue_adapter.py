"""Queue adapter for indexer service"""
import asyncio
import json
import os
import sys
from typing import Optional, Dict, Any
from azure.storage.queue.aio import QueueServiceClient, QueueClient

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from core.config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class AzureQueueAdapter:
    """Azure Storage Queue adapter for indexing jobs"""

    def __init__(self):
        self.provider_name = "azure"
        self.connection_string = settings.azure.AZURE_STORAGE_CONNECTION_STRING
        self.queue_name = settings.azure.INDEXING_QUEUE_NAME
        self.poll_interval = settings.azure.QUEUE_POLL_INTERVAL
        self.visibility_timeout = settings.MESSAGE_VISIBILITY_TIMEOUT

        # Initialize queue client
        self.queue_client: Optional[QueueClient] = None

    async def initialize(self):
        """Initialize the queue client"""
        if not self.queue_client:
            # Don't use base64 encoding - messages are plain JSON
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

    async def receive_message(self) -> Optional[tuple[str, Dict[str, Any]]]:
        """
        Receive a message from the queue.

        Returns:
            Tuple of (message_id, message_content) or None if no message
        """
        if not self.queue_client:
            await self.initialize()

        try:
            messages = []
            async for msg in self.queue_client.receive_messages(
                visibility_timeout=self.visibility_timeout, max_messages=1
            ):
                messages.append(msg)

            if messages:
                msg = messages[0]
                try:
                    content = json.loads(msg.content)
                    return (msg.id, content)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode message content: {e}")
                    # Delete invalid message
                    await self.delete_message(msg.id, msg.pop_receipt)
                    return None
            return None
        except Exception as e:
            logger.error(f"Error receiving message from queue: {e}")
            return None

    async def receive_messages(self, max_messages: int = 10, visibility_timeout: int = 300, wait_time_seconds: int = 20):
        """
        Receive multiple messages from the queue.

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
                visibility_timeout=visibility_timeout,
                max_messages=max_messages
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

                messages.append(Message(msg))

                if len(messages) >= max_messages:
                    break

            if messages:
                logger.info(f"Received {len(messages)} message(s) from queue '{self.queue_name}'")
            return messages

        except Exception as e:
            logger.error(f"Error receiving messages from queue: {e}")
            return []

    async def delete_message(self, receipt_handle: str, pop_receipt: str = None):
        """
        Delete a message from the queue.

        Args:
            receipt_handle: Either "message_id:pop_receipt" or just message_id
            pop_receipt: Optional pop_receipt if not included in receipt_handle
        """
        if not self.queue_client:
            await self.initialize()

        try:
            # Parse receipt_handle if it contains both id and pop_receipt
            if ':' in receipt_handle and pop_receipt is None:
                message_id, pop_receipt = receipt_handle.split(':', 1)
            else:
                message_id = receipt_handle

            await self.queue_client.delete_message(message_id, pop_receipt)
            logger.debug(f"Deleted message {message_id} from queue")
        except Exception as e:
            logger.error(f"Error deleting message: {e}")

    async def close(self):
        """Close the queue client"""
        if self.queue_client:
            await self.queue_client.close()
            self.queue_client = None


def get_queue_adapter() -> AzureQueueAdapter:
    """Get queue adapter based on configuration"""
    # For now, only Azure Storage Queue is supported
    # Can be extended to support AWS SQS, GCP Pub/Sub, etc.
    return AzureQueueAdapter()
