"""Request payload model for POST /api/kb/chat."""
from typing import List, Optional

from pydantic import Field
from src.shared.payloads import BasePayload, NonEmptyStr


class Message(BasePayload):
    """Chat message model."""

    role: NonEmptyStr
    content: NonEmptyStr


class KBChatRequest(BasePayload):
    """Payload for POST /api/kb/chat."""

    workspace_id: int
    messages: List[Message] = Field(min_length=1)
    kb_id: Optional[int] = None
