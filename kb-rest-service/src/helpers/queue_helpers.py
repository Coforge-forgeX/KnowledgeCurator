"""Azure Queue Storage helper functions"""
import base64
import json
from typing import Any, Dict, Optional

from azure.storage.queue import QueueClient, QueueMessage
from azure.storage.queue.aio import QueueClient as AsyncQueueClient

from core.config import settings
from core.exceptions import QueueException
from core.logging import get_logger

logger = get_logger(__name__)


class QueueHelper:
    """Helper class for Azure Queue Storage operations"""

    def __init__(
        self,
        queue_name: str,
        connection_string: Optional[str] = None,
    ):
        """
        Initialize Queue Helper.

        Args:
            queue_name: Name of the queue
            connection_string: Optional connection string override
        """
        self.queue_name = queue_name
        self.connection_string = (
            connection_string or settings.azure.QUEUE_STORAGE_CONNECTION_STRING
        )

        if not self.connection_string:
            raise QueueException(
                message="Queue storage connection string not configured",
                operation="initialize",
            )

        self._sync_client: Optional[QueueClient] = None
        self._async_client: Optional[AsyncQueueClient] = None

        logger.info("Queue helper initialized", queue_name=queue_name)

    def _get_sync_client(self) -> QueueClient:
        """Get or create synchronous queue client"""
        if not self._sync_client:
            self._sync_client = QueueClient.from_connection_string(
                conn_str=self.connection_string,
                queue_name=self.queue_name,
            )
        return self._sync_client

    def _get_async_client(self) -> AsyncQueueClient:
        """Get or create asynchronous queue client"""
        if not self._async_client:
            self._async_client = AsyncQueueClient.from_connection_string(
                conn_str=self.connection_string,
                queue_name=self.queue_name,
            )
        return self._async_client

    def send_message(
        self,
        message: Dict[str, Any],
        visibility_timeout: Optional[int] = None,
        time_to_live: Optional[int] = None,
    ) -> str:
        """
        Send a message to the queue (synchronous).

        Args:
            message: Message dictionary to send
            visibility_timeout: Time in seconds before message becomes visible
            time_to_live: Time in seconds before message expires

        Returns:
            Message ID

        Raises:
            QueueException: If send operation fails
        """
        try:
            client = self._get_sync_client()

            # Encode message as JSON
            message_text = json.dumps(message)

            # Send message
            result = client.send_message(
                content=message_text,
                visibility_timeout=visibility_timeout,
                time_to_live=time_to_live,
            )

            logger.info(
                "Message sent to queue",
                queue_name=self.queue_name,
                message_id=result.id,
            )

            return result.id

        except Exception as e:
            logger.error(
                "Failed to send message to queue",
                error=e,
                queue_name=self.queue_name,
            )
            raise QueueException(
                message=f"Failed to send message: {str(e)}",
                operation="send_message",
            )

    async def send_message_async(
        self,
        message: Dict[str, Any],
        visibility_timeout: Optional[int] = None,
        time_to_live: Optional[int] = None,
    ) -> str:
        """
        Send a message to the queue (asynchronous).

        Args:
            message: Message dictionary to send
            visibility_timeout: Time in seconds before message becomes visible
            time_to_live: Time in seconds before message expires

        Returns:
            Message ID

        Raises:
            QueueException: If send operation fails
        """
        try:
            client = self._get_async_client()

            # Encode message as JSON
            message_text = json.dumps(message)

            # Send message
            result = await client.send_message(
                content=message_text,
                visibility_timeout=visibility_timeout,
                time_to_live=time_to_live,
            )

            logger.info(
                "Message sent to queue",
                queue_name=self.queue_name,
                message_id=result.id,
            )

            return result.id

        except Exception as e:
            logger.error(
                "Failed to send message to queue",
                error=e,
                queue_name=self.queue_name,
            )
            raise QueueException(
                message=f"Failed to send message: {str(e)}",
                operation="send_message_async",
            )

    def receive_messages(
        self,
        max_messages: int = 1,
        visibility_timeout: Optional[int] = None,
    ) -> list:
        """
        Receive messages from the queue (synchronous).

        Args:
            max_messages: Maximum number of messages to receive
            visibility_timeout: Time in seconds before message becomes visible again

        Returns:
            List of queue messages

        Raises:
            QueueException: If receive operation fails
        """
        try:
            client = self._get_sync_client()

            messages = client.receive_messages(
                messages_per_page=max_messages,
                visibility_timeout=visibility_timeout,
            )

            result = []
            for message in messages:
                try:
                    content = json.loads(message.content)
                    result.append(
                        {
                            "id": message.id,
                            "content": content,
                            "pop_receipt": message.pop_receipt,
                            "dequeue_count": message.dequeue_count,
                        }
                    )
                except json.JSONDecodeError:
                    # Handle non-JSON messages
                    result.append(
                        {
                            "id": message.id,
                            "content": message.content,
                            "pop_receipt": message.pop_receipt,
                            "dequeue_count": message.dequeue_count,
                        }
                    )

            logger.info(
                "Messages received from queue",
                queue_name=self.queue_name,
                count=len(result),
            )

            return result

        except Exception as e:
            logger.error(
                "Failed to receive messages from queue",
                error=e,
                queue_name=self.queue_name,
            )
            raise QueueException(
                message=f"Failed to receive messages: {str(e)}",
                operation="receive_messages",
            )

    def delete_message(self, message_id: str, pop_receipt: str) -> None:
        """
        Delete a message from the queue (synchronous).

        Args:
            message_id: Message ID
            pop_receipt: Pop receipt from received message

        Raises:
            QueueException: If delete operation fails
        """
        try:
            client = self._get_sync_client()
            client.delete_message(message=message_id, pop_receipt=pop_receipt)

            logger.info(
                "Message deleted from queue",
                queue_name=self.queue_name,
                message_id=message_id,
            )

        except Exception as e:
            logger.error(
                "Failed to delete message from queue",
                error=e,
                queue_name=self.queue_name,
                message_id=message_id,
            )
            raise QueueException(
                message=f"Failed to delete message: {str(e)}",
                operation="delete_message",
            )

    async def delete_message_async(self, message_id: str, pop_receipt: str) -> None:
        """
        Delete a message from the queue (asynchronous).

        Args:
            message_id: Message ID
            pop_receipt: Pop receipt from received message

        Raises:
            QueueException: If delete operation fails
        """
        try:
            client = self._get_async_client()
            await client.delete_message(message=message_id, pop_receipt=pop_receipt)

            logger.info(
                "Message deleted from queue",
                queue_name=self.queue_name,
                message_id=message_id,
            )

        except Exception as e:
            logger.error(
                "Failed to delete message from queue",
                error=e,
                queue_name=self.queue_name,
                message_id=message_id,
            )
            raise QueueException(
                message=f"Failed to delete message: {str(e)}",
                operation="delete_message_async",
            )

    async def close(self) -> None:
        """Close queue clients"""
        if self._async_client:
            await self._async_client.close()
            self._async_client = None

        if self._sync_client:
            self._sync_client.close()
            self._sync_client = None

        logger.info("Queue clients closed", queue_name=self.queue_name)


# Helper function to create indexing queue helper
def get_indexing_queue_helper() -> QueueHelper:
    """
    Get queue helper for indexing queue.

    Returns:
        QueueHelper: Configured queue helper for indexing
    """
    return QueueHelper(
        queue_name=settings.azure.INDEXING_QUEUE_NAME,
    )
