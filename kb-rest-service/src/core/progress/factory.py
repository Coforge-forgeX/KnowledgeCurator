from __future__ import annotations

from typing import Any

from .registry import get_progress_publisher
from .reporter import ProgressReporter


def create_progress_reporter(*, operation: str, user_id: str, payload: Any, req: Any | None) -> ProgressReporter:
    correlation_id = None
    if req is not None:
        correlation_id = req.headers.get("x-correlation-id") or req.headers.get("X-Correlation-ID")

    return ProgressReporter(
        get_progress_publisher(),
        operation=operation,
        user_id=str(user_id),
        conversation_id=str(getattr(payload, "conversation_id", "") or "") or None,
        job_id=str(getattr(payload, "job_id", "") or "") or None,
        correlation_id=correlation_id,
    )
