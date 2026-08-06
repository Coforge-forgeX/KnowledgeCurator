"""
Azure Service Bus Consumer Example

This example shows how to consume progress events from Azure Service Bus
and forward them to WebSocket clients, logging systems, or other services.

Usage:
    # Queue consumer
    python azure_servicebus_consumer.py --mode queue --queue progress-events

    # Topic subscriber
    python azure_servicebus_consumer.py --mode topic --topic agent-progress --subscription websocket-relay

Environment Variables:
    EVENT_BUS_CONNECTION_STRING or SERVICE_BUS_CONNECTION_STRING
"""

import asyncio
import json
import os
from typing import Any, Dict

from azure.servicebus import ServiceBusClient, ServiceBusMessage
from azure.servicebus.aio import ServiceBusClient as AsyncServiceBusClient


class ProgressEventConsumer:
    """Consumes progress events from Azure Service Bus."""

    def __init__(
        self,
        connection_string: str,
        entity_name: str,
        entity_type: str = "queue",
        subscription_name: str | None = None,
    ):
        self.connection_string = connection_string
        self.entity_name = entity_name
        self.entity_type = entity_type.lower()
        self.subscription_name = subscription_name

    async def process_message(self, message: ServiceBusMessage) -> None:
        """
        Process a single progress event.

        Override this method to implement your custom logic:
        - Forward to WebSocket clients
        - Store in analytics database
        - Trigger notifications
        - Update dashboards
        """
        try:
            # Parse event payload
            body = str(message)
            event = json.loads(body)

            # Log event
            print(f"\n{'=' * 80}")
            print(f"Operation: {event.get('operation')}")
            print(f"Status: {event.get('status')}")
            print(f"Message: {event.get('message')}")
            print(f"User: {event.get('user_id')}")
            print(f"Timestamp: {event.get('timestamp')}")

            if event.get("metadata"):
                print(f"Metadata: {json.dumps(event['metadata'], indent=2)}")

            # Example: Forward to WebSocket
            # await self.broadcast_to_websockets(event)

            # Example: Store in database
            # await self.store_event(event)

            # Example: Trigger notifications
            # if event.get('status') == 'failed':
            #     await self.send_alert(event)

            print(f"{'=' * 80}\n")

        except Exception as exc:
            print(f"Error processing message: {exc}")
            # Don't raise - let the message complete so it doesn't retry infinitely

    async def broadcast_to_websockets(self, event: Dict[str, Any]) -> None:
        """Example: Forward event to WebSocket clients."""
        # Implementation depends on your WebSocket framework
        # FastAPI WebSockets, Socket.IO, etc.
        pass

    async def store_event(self, event: Dict[str, Any]) -> None:
        """Example: Store event in analytics database."""
        # Store in PostgreSQL, MongoDB, ClickHouse, etc.
        pass

    async def send_alert(self, event: Dict[str, Any]) -> None:
        """Example: Send alerts for failed operations."""
        # Send to Slack, email, PagerDuty, etc.
        pass

    async def run_queue_consumer(self) -> None:
        """Run consumer for queue mode."""
        print(f"Starting queue consumer: {self.entity_name}")

        async with AsyncServiceBusClient.from_connection_string(
            self.connection_string
        ) as client:
            receiver = client.get_queue_receiver(queue_name=self.entity_name)

            async with receiver:
                print(f"Listening for messages on queue '{self.entity_name}'...")

                async for message in receiver:
                    await self.process_message(message)
                    await receiver.complete_message(message)

    async def run_topic_consumer(self) -> None:
        """Run consumer for topic subscription mode."""
        if not self.subscription_name:
            raise ValueError("subscription_name required for topic mode")

        print(
            f"Starting topic subscriber: {self.entity_name} / {self.subscription_name}"
        )

        async with AsyncServiceBusClient.from_connection_string(
            self.connection_string
        ) as client:
            receiver = client.get_subscription_receiver(
                topic_name=self.entity_name, subscription_name=self.subscription_name
            )

            async with receiver:
                print(
                    f"Listening for messages on topic '{self.entity_name}' "
                    f"subscription '{self.subscription_name}'..."
                )

                async for message in receiver:
                    await self.process_message(message)
                    await receiver.complete_message(message)

    async def run(self) -> None:
        """Run the consumer based on entity type."""
        try:
            if self.entity_type == "queue":
                await self.run_queue_consumer()
            else:
                await self.run_topic_consumer()
        except KeyboardInterrupt:
            print("\nShutting down consumer...")
        except Exception as exc:
            print(f"Consumer error: {exc}")
            raise


class WebSocketRelayConsumer(ProgressEventConsumer):
    """
    Example: Consumer that forwards events to WebSocket clients.

    This is a common pattern where:
    1. kb-rest-service publishes progress events to Azure Service Bus
    2. This consumer receives events
    3. Events are broadcast to connected WebSocket clients
    """

    def __init__(self, *args, websocket_manager=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.websocket_manager = websocket_manager

    async def process_message(self, message: ServiceBusMessage) -> None:
        """Forward event to WebSocket clients."""
        try:
            body = str(message)
            event = json.loads(body)

            # Extract routing info
            user_id = event.get("user_id")
            conversation_id = event.get("conversation_id")

            # Broadcast to relevant clients
            if self.websocket_manager:
                await self.websocket_manager.broadcast(
                    event, user_id=user_id, conversation_id=conversation_id
                )

            # Also log for debugging
            print(
                f"Forwarded: {event.get('operation')} - "
                f"{event.get('status')} - {event.get('message')}"
            )

        except Exception as exc:
            print(f"Error forwarding message: {exc}")


class AnalyticsConsumer(ProgressEventConsumer):
    """
    Example: Consumer that stores events in analytics database.

    Use case: Build dashboards showing:
    - Operations per user
    - Success/failure rates
    - Performance metrics
    - Usage patterns
    """

    def __init__(self, *args, analytics_db=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.analytics_db = analytics_db

    async def process_message(self, message: ServiceBusMessage) -> None:
        """Store event in analytics database."""
        try:
            body = str(message)
            event = json.loads(body)

            if self.analytics_db:
                await self.analytics_db.insert_event(event)

            print(f"Stored: {event.get('operation')} - {event.get('status')}")

        except Exception as exc:
            print(f"Error storing message: {exc}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Azure Service Bus Progress Consumer")
    parser.add_argument(
        "--mode",
        choices=["queue", "topic"],
        default="queue",
        help="Consumer mode: queue or topic",
    )
    parser.add_argument("--queue", help="Queue name (for queue mode)")
    parser.add_argument("--topic", help="Topic name (for topic mode)")
    parser.add_argument("--subscription", help="Subscription name (for topic mode)")

    args = parser.parse_args()

    # Get connection string
    connection_string = os.getenv("EVENT_BUS_CONNECTION_STRING") or os.getenv(
        "SERVICE_BUS_CONNECTION_STRING"
    )

    if not connection_string:
        print("Error: EVENT_BUS_CONNECTION_STRING or SERVICE_BUS_CONNECTION_STRING required")
        return

    # Determine entity name
    if args.mode == "queue":
        entity_name = args.queue or os.getenv("PROGRESS_QUEUE", "progress-events")
        subscription_name = None
    else:
        entity_name = args.topic or os.getenv("PROGRESS_TOPIC", "agent-progress")
        subscription_name = args.subscription
        if not subscription_name:
            print("Error: --subscription required for topic mode")
            return

    # Create and run consumer
    consumer = ProgressEventConsumer(
        connection_string=connection_string,
        entity_name=entity_name,
        entity_type=args.mode,
        subscription_name=subscription_name,
    )

    asyncio.run(consumer.run())


if __name__ == "__main__":
    main()
