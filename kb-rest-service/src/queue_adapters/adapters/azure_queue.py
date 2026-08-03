"""Azure Storage Queue adapter"""

import json
import os
from typing import Any, Dict, List

from azure.storage.queue.aio import QueueServiceClient

from core.exceptions import ConfigurationException
from core.logging import get_logger

from ..models import QueueMessage
from ..protocols import QueueAdapter

logger = get_logger(__name__)


class AzureQueueAdapter(QueueAdapter):
    """Azure Storage Queue implementation"""

    def __init__(self) -> None:
        from core.config import settings

        # Try queue-specific connection string first, fall back to general Azure storage
        conn_str = (
            settings.azure.QUEUE_STORAGE_CONNECTION_STRING
            or settings.azure.AZURE_STORAGE_CONNECTION_STRING
        )
        if not conn_str:
            raise ConfigurationException(
                "AZURE_QUEUE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_CONNECTION_STRING not configured",
                config_key="AZURE_QUEUE_STORAGE_CONNECTION_STRING",
            )

        self._queue_name = settings.azure.INDEXING_QUEUE_NAME or "indexing-jobs"
        self._service_client = QueueServiceClient.from_connection_string(conn_str)
        self._queue_client = self._service_client.get_queue_client(self._queue_name)

        logger.info(
            "Azure Queue adapter initialized",
            queue_name=self._queue_name,
        )

    @property
    def provider_name(self) -> str:
        return "azure"

    @property
    def queue_name(self) -> str:
        return self._queue_name

    async def _ensure_queue_exists(self) -> None:
        """Create queue if it doesn't exist"""
        try:
            await self._queue_client.create_queue()
            logger.info(f"Created queue: {self._queue_name}")
        except Exception:
            pass  # Queue already exists

    async def send_message(
        self, message: Dict[str, Any], delay_seconds: int = 0
    ) -> str:
        """Send message to Azure Storage Queue"""
        try:
            await self._ensure_queue_exists()

            message_content = json.dumps(message)
            result = await self._queue_client.send_message(
                message_content, visibility_timeout=delay_seconds
            )

            logger.info(
                "Message sent to Azure Queue",
                message_id=result.id,
                queue=self._queue_name,
            )

            return result.id

        except Exception as e:
            logger.error(f"Failed to send message to Azure Queue: {e}")
            raise

    async def receive_messages(
        self,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> List[QueueMessage]:
        """Receive messages from Azure Storage Queue"""
        try:
            messages = []
            async for msg in self._queue_client.receive_messages(
                messages_per_page=max_messages,
                visibility_timeout=visibility_timeout,
            ):
                try:
                    content = json.loads(msg.content)
                except json.JSONDecodeError:
                    content = {"raw": msg.content}

                messages.append(
                    QueueMessage(
                        content=content,
                        message_id=msg.id,
                        receipt_handle=msg.pop_receipt,
                    )
                )

                if len(messages) >= max_messages:
                    break

            logger.info(
                f"Received {len(messages)} message(s) from Azure Queue",
                queue=self._queue_name,
            )

            return messages

        except Exception as e:
            logger.error(f"Failed to receive messages from Azure Queue: {e}")
            raise

    async def delete_message(self, receipt_handle: str) -> bool:
        """Delete message from Azure Storage Queue"""
        try:
            # Azure uses message_id and pop_receipt together
            # receipt_handle should be in format: "message_id:pop_receipt"
            if ":" in receipt_handle:
                message_id, pop_receipt = receipt_handle.split(":", 1)
            else:
                # Fallback: treat entire handle as pop_receipt
                message_id = None
                pop_receipt = receipt_handle

            await self._queue_client.delete_message(message_id, pop_receipt)

            logger.info(
                "Message deleted from Azure Queue",
                message_id=message_id,
                queue=self._queue_name,
            )

            return True

        except Exception as e:
            logger.error(f"Failed to delete message from Azure Queue: {e}")
            return False

    async def get_queue_size(self) -> int:
        """Get approximate message count from Azure Queue"""
        try:
            properties = await self._queue_client.get_queue_properties()
            return properties.approximate_message_count or 0
        except Exception as e:
            logger.error(f"Failed to get queue size: {e}")
            return 0

    async def purge_queue(self) -> bool:
        """Clear all messages from Azure Queue"""
        try:
            await self._queue_client.clear_messages()
            logger.info(f"Purged queue: {self._queue_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to purge queue: {e}")
            return False
