from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict


@dataclass(slots=True)
class ProgressEvent:
    """Canonical progress message emitted by long-running operations."""

    operation: str
    status: str
    message: str
    user_id: str
    correlation_id: str | None = None
    conversation_id: str | None = None
    job_id: str | None = None
    provider: str = "business-analyst"
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
