"""Abstract queue adapter interface"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .models import QueueMessage


class QueueAdapter(ABC):
    """Abstract interface for message queue operations"""

    @abstractmethod
    async def send_message(
        self, message: Dict[str, Any], delay_seconds: int = 0
    ) -> str:
        """
        Send message to queue.

        Args:
            message: Message content as dictionary (will be JSON-encoded)
            delay_seconds: Delay before message becomes visible (default: 0)

        Returns:
            Message ID

        Raises:
            Exception: If send fails
        """
        pass

    @abstractmethod
    async def receive_messages(
        self,
        max_messages: int = 1,
        visibility_timeout: int = 30,
        wait_time_seconds: int = 0,
    ) -> List[QueueMessage]:
        """
        Receive messages from queue.

        Args:
            max_messages: Maximum number of messages to receive (1-10)
            visibility_timeout: How long messages are hidden from other consumers (seconds)
            wait_time_seconds: Long polling wait time (0-20 seconds)

        Returns:
            List of QueueMessage objects

        Raises:
            Exception: If receive fails
        """
        pass

    @abstractmethod
    async def delete_message(self, receipt_handle: str) -> bool:
        """
        Delete message from queue.

        Args:
            receipt_handle: Receipt handle from received message

        Returns:
            True if deleted, False otherwise

        Raises:
            Exception: If delete fails
        """
        pass

    @abstractmethod
    async def get_queue_size(self) -> int:
        """
        Get approximate number of messages in queue.

        Returns:
            Approximate message count

        Note:
            This is an approximation and may not be exact due to distributed nature
        """
        pass

    @abstractmethod
    async def purge_queue(self) -> bool:
        """
        Delete all messages in queue.

        Returns:
            True if purged successfully

        Raises:
            Exception: If purge fails
        """
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get queue provider name (azure/aws/redis)"""
        pass

    @property
    @abstractmethod
    def queue_name(self) -> str:
        """Get queue name"""
        pass
