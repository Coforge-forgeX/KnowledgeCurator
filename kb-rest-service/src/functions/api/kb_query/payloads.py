"""Request payload model for POST /api/kb/query."""
from typing import Optional

from src.shared.payloads import BasePayload, NonEmptyStr


class KBQueryRequest(BasePayload):
    """Payload for POST /api/kb/query."""

    workspace_id: int
    query: NonEmptyStr
    kb_id: Optional[int] = None
    top_k: int = 5
