from __future__ import annotations

from typing import Protocol

from .models import ProgressEvent


class ProgressPublisher(Protocol):
    async def publish(self, event: ProgressEvent) -> None:
        """Publish a progress event to the configured transport."""


class NullProgressPublisher:
    async def publish(self, event: ProgressEvent) -> None:
        _ = event


class CompositeProgressPublisher:
    def __init__(self, publishers: list[ProgressPublisher]):
        self._publishers = publishers

    async def publish(self, event: ProgressEvent) -> None:
        for publisher in self._publishers:
            await publisher.publish(event)
