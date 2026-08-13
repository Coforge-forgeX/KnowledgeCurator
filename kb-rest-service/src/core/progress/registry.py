from __future__ import annotations

from src.core.config import settings
from src.core.logging import Logger

from .cloud_provider import CloudProvider, resolve_cloud_provider
from .ports import CompositeProgressPublisher, NullProgressPublisher, ProgressPublisher
from .transports import (
    AwsEventBridgeProgressPublisher,
    AzureServiceBusProgressPublisher,
    LocalRelayProgressPublisher,
    LogProgressPublisher,
)

logger = Logger("progress-registry")


def _resolve_backend() -> str:
    """Resolve progress backend from settings."""
    explicit_backend = settings.progress.PROGRESS_BACKEND.strip().lower()
    if explicit_backend and explicit_backend != "auto":
        return explicit_backend

    bus_provider = (settings.progress.EVENT_BUS_PROVIDER or "").strip().lower()
    if bus_provider in {
        "azure_service_bus",
        "aws_eventbridge",
        "local_relay",
        "log",
        "none",
    }:
        return bus_provider

    # Auto-detect from cloud provider
    cloud = resolve_cloud_provider(
        explicit_provider=settings.CLOUD_PROVIDER,
        aws_region=settings.storage.AWS_REGION,
        azure_website_name=None,  # Not available in settings
        azure_webjobs_storage=settings.storage.AZURE_STORAGE_CONNECTION_STRING,
        gcp_project_id=settings.storage.GCP_PROJECT_ID,
    )
    if cloud == CloudProvider.AZURE:
        return "azure_service_bus"
    if cloud == CloudProvider.AWS:
        return "aws_eventbridge"
    if cloud == CloudProvider.GCP:
        return "log"  # Or add GCP Pub/Sub support
    return "log"


def get_progress_publisher() -> ProgressPublisher:
    """Resolve progress transport from configuration settings."""

    backend = _resolve_backend()

    if backend == "none":
        return NullProgressPublisher()

    if backend == "log":
        return LogProgressPublisher()

    if backend == "local_relay":
        publish_url = settings.progress.PROGRESS_LOCAL_RELAY_URL
        return CompositeProgressPublisher(
            [LogProgressPublisher(), LocalRelayProgressPublisher(publish_url)]
        )

    if backend == "azure_service_bus":
        connection_string = (
            settings.progress.EVENT_BUS_CONNECTION_STRING
            or settings.progress.SERVICE_BUS_CONNECTION_STRING
            or ""
        )
        queue_name = (settings.progress.PROGRESS_QUEUE or "").strip()
        topic_name = settings.progress.PROGRESS_TOPIC.strip()

        if not connection_string:
            logger.warning(
                "Progress backend fallback to log",
                backend=backend,
                reason="missing_connection_string",
            )
            return LogProgressPublisher()

        if queue_name:
            logger.info(
                "Progress backend resolved",
                backend=backend,
                entity_type="queue",
                entity_name=queue_name,
            )
            return CompositeProgressPublisher(
                [
                    LogProgressPublisher(),
                    AzureServiceBusProgressPublisher(connection_string, queue_name, "queue"),
                ]
            )

        logger.info(
            "Progress backend resolved",
            backend=backend,
            entity_type="topic",
            entity_name=topic_name,
        )
        return CompositeProgressPublisher(
            [
                LogProgressPublisher(),
                AzureServiceBusProgressPublisher(connection_string, topic_name, "topic"),
            ]
        )

    if backend == "aws_eventbridge":
        bus_name = settings.progress.PROGRESS_EVENT_BUS
        return CompositeProgressPublisher(
            [LogProgressPublisher(), AwsEventBridgeProgressPublisher(bus_name)]
        )

    return LogProgressPublisher()
