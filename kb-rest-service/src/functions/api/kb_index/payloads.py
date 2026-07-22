"""Request payload model for POST /api/kb/index."""
from typing import Optional

from shared.payloads import BasePayload, NonEmptyStr


class KBIndexRequest(BasePayload):
    """Payload for POST /api/kb/index - enqueue indexing job."""

    workspace_id: int
    document_url: NonEmptyStr
    kb_id: Optional[int] = None
