from __future__ import annotations

import asyncio
import uuid
from typing import Any, Dict

from src.core.logging import Logger

from .models import ProgressEvent
from .ports import ProgressPublisher

logger = Logger("progress-reporter")


class ProgressReporter:
    """Use-case level helper to standardize progress events."""

    def __init__(
        self,
        publisher: ProgressPublisher,
        *,
        operation: str,
        user_id: str,
        conversation_id: str | None,
        job_id: str | None,
        correlation_id: str | None,
    ):
        self._publisher = publisher
        self._operation = operation
        self._user_id = user_id
        self._conversation_id = conversation_id
        self._job_id = job_id
        self._correlation_id = correlation_id

        self._event_seq = 0
        self._last_progress = 0
        self._emit_lock = asyncio.Lock()
        self._run_id = uuid.uuid4().hex

    async def emit(self, *, status: str, message: str, metadata: Dict[str, Any] | None = None) -> bool:
        async with self._emit_lock:
            payload = dict(metadata or {})
            current_progress = int(payload.get("progress_percent", self._last_progress))
            if current_progress < self._last_progress:
                current_progress = self._last_progress
            self._last_progress = current_progress

            self._event_seq += 1
            payload["progress_percent"] = current_progress
            payload["run_id"] = self._run_id
            payload["event_seq"] = self._event_seq

            event = ProgressEvent(
                operation=self._operation,
                status=status,
                message=message,
                user_id=self._user_id,
                conversation_id=self._conversation_id,
                job_id=self._job_id,
                correlation_id=self._correlation_id,
                metadata=payload,
            )
            try:
                await self._publisher.publish(event)
                return True
            except Exception as exc:
                logger.error(
                    "Progress publish failed",
                    error=exc,
                    operation=self._operation,
                    status=status,
                    user_id=self._user_id,
                )
                return False
