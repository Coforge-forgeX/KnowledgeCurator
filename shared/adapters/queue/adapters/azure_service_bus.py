"""Azure Service Bus queue adapter.

This adapter publishes and consumes messages from Azure Service Bus.
Supports both queue mode and topic/subscription mode.
"""
import json
import logging
from typing import Any, Dict, List, Optional

from azure.servicebus import ServiceBusMessage
from azure.servicebus.aio import ServiceBusClient, ServiceBusSender, ServiceBusReceiver, AutoLockRenewer
from azure.servicebus.exceptions import MessageLockLostError

logger = logging.getLogger(__name__)


class ServiceBusQueueMessage:
    """Wrapper for Service Bus message to match common queue interface."""

    def __init__(self, message):
        """Initialize message wrapper.

        Args:
            message: Azure Service Bus message
        """
        self._message = message
        try:
            self.content = json.loads(str(message))
        except json.JSONDecodeError:
            logger.error(f"Failed to decode Service Bus message (message_id={message.message_id})")
            self.content = {"raw": str(message)}

        self.message_id = message.message_id
        self.receipt_handle = message.message_id
        self.dequeue_count = message.delivery_count or 1

    def __getattr__(self, name):
        """Delegate attribute access to underlying message."""
        return getattr(self._message, name)


class AzureServiceBusAdapter:
    """Azure Service Bus adapter for message queuing.

    Supports both producer (send) and consumer (receive) operations.
    Works in queue mode or topic/subscription mode.

    Benefits over Azure Storage Queue:
    - Better reliability and message delivery guarantees
    - Dead letter queue support
    - Message sessions and ordering
    - Better scalability
    - Advanced message properties
    """

    def __init__(
        self,
        connection_string: str,
        queue_name: Optional[str] = None,
        topic_name: Optional[str] = None,
        subscription_name: Optional[str] = None,
        max_concurrent_calls: int = 1,
        max_lock_renewal_duration: int = 1800,
    ):
        """Initialize Service Bus adapter.

        Args:
            connection_string: Azure Service Bus connection string
            queue_name: Queue name (for queue mode)
            topic_name: Topic name (for topic/subscription mode)
            subscription_name: Subscription name (for consumer in topic mode)
            max_concurrent_calls: Max concurrent message processing (for consumer)
            max_lock_renewal_duration: Max duration (seconds) to auto-renew message locks.
                Set to max expected job duration. Default: 1800s (30 min)
        """
        if not queue_name and not topic_name:
            raise ValueError("Either queue_name or topic_name must be provided")

        self.provider_name = "azure_service_bus"
        self.connection_string = connection_string
        self.queue_name = queue_name
        self.topic_name = topic_name
        self.subscription_name = subscription_name
        self.max_concurrent_calls = max_concurrent_calls
        self.max_lock_renewal_duration = max_lock_renewal_duration
        self.entity_type = "topic" if topic_name else "queue"

        self._client: Optional[ServiceBusClient] = None
        self._sender: Optional[ServiceBusSender] = None
        self._receiver: Optional[ServiceBusReceiver] = None
        self._lock_renewer: Optional[AutoLockRenewer] = None

    async def initialize(self) -> None:
        """Initialize Service Bus client and sender."""
        if not self._client:
            self._client = ServiceBusClient.from_connection_string(
                self.connection_string
            )

        if not self._sender:
            if self.entity_type == "topic":
                self._sender = self._client.get_topic_sender(topic_name=self.topic_name)
                logger.info(
                    f"Azure Service Bus topic sender initialized (topic={self.topic_name})"
                )
            else:
                self._sender = self._client.get_queue_sender(queue_name=self.queue_name)
                logger.info(
                    f"Azure Service Bus queue sender initialized (queue={self.queue_name})"
                )

    async def initialize_receiver(self) -> None:
        """Initialize Service Bus receiver for consuming messages."""
        if not self._client:
            self._client = ServiceBusClient.from_connection_string(
                self.connection_string
            )

        if not self._receiver:
            if self.entity_type == "topic":
                if not self.subscription_name:
                    raise ValueError("subscription_name required for topic receiver")

                self._receiver = self._client.get_subscription_receiver(
                    topic_name=self.topic_name,
                    subscription_name=self.subscription_name,
                    max_wait_time=20,
                )
                logger.info(
                    f"Azure Service Bus subscription receiver initialized "
                    f"(topic={self.topic_name}, subscription={self.subscription_name})"
                )
            else:
                self._receiver = self._client.get_queue_receiver(
                    queue_name=self.queue_name,
                    max_wait_time=20,
                )
                logger.info(
                    f"Azure Service Bus queue receiver initialized (queue={self.queue_name})"
                )

        # Initialize auto lock renewer for long-running jobs
        if not self._lock_renewer:
            # Auto-renew locks up to configured max duration
            # This allows long-running indexing jobs to complete without lock expiration
            self._lock_renewer = AutoLockRenewer(max_lock_renewal_duration=self.max_lock_renewal_duration)
            logger.info(f"AutoLockRenewer initialized (max_duration={self.max_lock_renewal_duration}s)")

    async def send_message(self, message_content: Dict[str, Any]) -> str:
        """Send a message to Service Bus.

        Args:
            message_content: Message payload as dictionary

        Returns:
            Message ID

        Raises:
            Exception: If message sending fails
        """
        if not self._sender:
            await self.initialize()

        try:
            # Create Service Bus message
            message = ServiceBusMessage(
                json.dumps(message_content, default=str),
                content_type="application/json",
            )

            # Add custom properties for routing/filtering if needed
            if "task_id" in message_content:
                message.application_properties = {
                    "task_id": str(message_content["task_id"]),
                    "workspace_id": str(message_content.get("workspace_id", "")),
                }

            # Send message
            await self._sender.send_messages(message)

            entity_name = self.topic_name if self.entity_type == "topic" else self.queue_name
            message_id = message.message_id or str(message_content.get("task_id", "unknown"))

            logger.info(
                f"Message sent to Service Bus (message_id={message_id}, {self.entity_type}={entity_name})"
            )

            return message_id

        except Exception as exc:
            entity_name = self.topic_name if self.entity_type == "topic" else self.queue_name
            logger.error(
                f"Failed to send message to Service Bus ({self.entity_type}={entity_name}): {exc}",
                exc_info=True,
            )
            raise

    async def receive_messages(
        self,
        max_messages: int = 10,
        visibility_timeout: int = 300,
        wait_time_seconds: int = 20,
    ) -> List[ServiceBusQueueMessage]:
        """Receive messages from Service Bus.

        Args:
            max_messages: Maximum messages to receive
            visibility_timeout: Message lock duration (not used in Service Bus)
            wait_time_seconds: Long polling wait time

        Returns:
            List of ServiceBusQueueMessage objects
        """
        if not self._receiver:
            await self.initialize_receiver()

        try:
            messages = await self._receiver.receive_messages(
                max_message_count=max_messages,
                max_wait_time=wait_time_seconds,
            )

            if messages:
                wrapped_messages = [ServiceBusQueueMessage(msg) for msg in messages]

                # Register each message with the auto lock renewer
                # This prevents lock expiration during long-running jobs
                for wrapped_msg in wrapped_messages:
                    self._lock_renewer.register(
                        self._receiver,
                        wrapped_msg._message,
                        max_lock_renewal_duration=self.max_lock_renewal_duration
                    )

                entity_name = (
                    f"{self.topic_name}/{self.subscription_name}"
                    if self.entity_type == "topic"
                    else self.queue_name
                )
                logger.info(
                    f"Received {len(messages)} message(s) from Service Bus "
                    f"({self.entity_type}={entity_name})"
                )
                return wrapped_messages

            return []

        except Exception as exc:
            entity_name = (
                f"{self.topic_name}/{self.subscription_name}"
                if self.entity_type == "topic"
                else self.queue_name
            )
            logger.error(
                f"Error receiving messages from Service Bus ({self.entity_type}={entity_name}): {exc}",
                exc_info=True,
            )
            return []

    async def delete_message(self, receipt_handle: Any, pop_receipt: str = None) -> None:
        """Complete (acknowledge) a message.

        In Service Bus, completing a message removes it from the queue.

        Args:
            receipt_handle: Message ID or ServiceBusQueueMessage object
            pop_receipt: Not used (for compatibility with other adapters)
        """
        if not self._receiver:
            await self.initialize_receiver()

        try:
            if hasattr(receipt_handle, '_message'):
                # It's a ServiceBusQueueMessage wrapper
                await self._receiver.complete_message(receipt_handle._message)
            else:
                logger.warning(
                    f"Cannot complete message - message object not available (receipt={receipt_handle})"
                )

        except MessageLockLostError as exc:
            # Lock expired - message will return to queue automatically
            # This is not a critical error; log as warning
            logger.warning(
                f"Message lock expired before completion - message will retry automatically "
                f"(message_id={getattr(receipt_handle, 'message_id', 'unknown')})"
            )
        except Exception as exc:
            logger.error(
                f"Error completing message (receipt={receipt_handle}): {exc}",
                exc_info=True,
            )

    async def move_to_dead_letter(
        self, receipt_handle: Any, reason: str = "MaxRetriesExceeded", error_description: str = None
    ) -> None:
        """Move a message to the dead letter queue.

        Args:
            receipt_handle: Message ID or ServiceBusQueueMessage object
            reason: Reason for dead lettering (e.g., "MaxRetriesExceeded")
            error_description: Optional detailed error description
        """
        if not self._receiver:
            await self.initialize_receiver()

        try:
            if hasattr(receipt_handle, '_message'):
                # It's a ServiceBusQueueMessage wrapper
                await self._receiver.dead_letter_message(
                    receipt_handle._message,
                    reason=reason,
                    error_description=error_description or f"Message failed after maximum retries"
                )
                entity_name = (
                    f"{self.topic_name}/{self.subscription_name}"
                    if self.entity_type == "topic"
                    else self.queue_name
                )
                logger.warning(
                    f"Message moved to dead letter queue "
                    f"({self.entity_type}={entity_name}, reason={reason}, message_id={receipt_handle.message_id})"
                )
            else:
                logger.warning(
                    f"Cannot dead letter message - message object not available (receipt={receipt_handle})"
                )

        except MessageLockLostError as exc:
            # Lock expired - message will return to queue automatically and retry
            logger.warning(
                f"Message lock expired before dead lettering - message will retry automatically "
                f"(message_id={getattr(receipt_handle, 'message_id', 'unknown')}, reason={reason})"
            )
        except Exception as exc:
            logger.error(
                f"Error moving message to dead letter queue (receipt={receipt_handle}): {exc}",
                exc_info=True,
            )

    async def close(self) -> None:
        """Close Service Bus client, sender, and receiver."""
        if self._lock_renewer:
            await self._lock_renewer.close()
            self._lock_renewer = None

        if self._sender:
            await self._sender.close()
            self._sender = None

        if self._receiver:
            await self._receiver.close()
            self._receiver = None

        if self._client:
            await self._client.close()
            self._client = None

        logger.debug("Service Bus adapter closed")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
