from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from src.core.logging import Logger

from .models import ProgressEvent

logger = Logger("progress-transports")


class LogProgressPublisher:
    """Safe default transport that emits progress in application logs."""

    async def publish(self, event: ProgressEvent) -> None:
        logger.info(
            "ProgressEvent",
            operation=event.operation,
            status=event.status,
            user_id=event.user_id,
            conversation_id=event.conversation_id,
            job_id=event.job_id,
            correlation_id=event.correlation_id,
            event_message=event.message,
            metadata=event.metadata,
        )


class AzureServiceBusProgressPublisher:
    """Azure Service Bus adapter (topic/event fanout)."""

    def __init__(self, connection_string: str, entity_name: str, entity_type: str = "topic"):
        self._connection_string = connection_string
        self._entity_name = entity_name
        self._entity_type = (entity_type or "topic").strip().lower()

    async def publish(self, event: ProgressEvent) -> None:
        try:
            from azure.servicebus.aio import ServiceBusClient
            from azure.servicebus import ServiceBusMessage
        except ImportError as exc:
            raise RuntimeError(
                "azure-servicebus is not installed; cannot use ASB progress backend"
            ) from exc

        payload = json.dumps(event.to_dict(), default=str)
        async with ServiceBusClient.from_connection_string(self._connection_string) as client:
            if self._entity_type == "queue":
                sender = client.get_queue_sender(queue_name=self._entity_name)
            else:
                sender = client.get_topic_sender(topic_name=self._entity_name)
            async with sender:
                await sender.send_messages(ServiceBusMessage(payload))


class AwsEventBridgeProgressPublisher:
    """AWS EventBridge adapter for cloud-agnostic portability."""

    def __init__(self, bus_name: str, source: str = "forgex.businessanalyst"):
        self._bus_name = bus_name
        self._source = source

    async def publish(self, event: ProgressEvent) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is not installed; cannot use EventBridge progress backend"
            ) from exc

        detail = json.dumps(event.to_dict(), default=str)
        client = boto3.client("events")
        client.put_events(
            Entries=[
                {
                    "Source": self._source,
                    "DetailType": f"progress.{event.operation}",
                    "Detail": detail,
                    "EventBusName": self._bus_name,
                }
            ]
        )


class LocalRelayProgressPublisher:
    """Local transport that posts events to a relay service for WebSocket fanout."""

    def __init__(self, publish_url: str, timeout_s: float = 5.0, max_retries: int = 2):
        self._publish_url = publish_url
        self._timeout_s = timeout_s
        self._max_retries = max_retries

    async def publish(self, event: ProgressEvent) -> None:
        payload = event.to_dict()
        attempts = self._max_retries + 1
        last_exc: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                    response = await client.post(self._publish_url, json=payload)
                    response.raise_for_status()
                return
            except Exception as exc:
                last_exc = exc
                if attempt < attempts:
                    await asyncio.sleep(0.25 * attempt)

        if last_exc:
            raise last_exc


class InMemoryWebSocketProgressPublisher:
    """Local adapter for direct testing via callable emitter.

    emitter(event_dict) can bridge to any in-process websocket manager.
    """

    def __init__(self, emitter: Any):
        self._emitter = emitter

    async def publish(self, event: ProgressEvent) -> None:
        await self._emitter(event.to_dict())


