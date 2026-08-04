"""Request payload model for POST /api/llm/route."""
from typing import Any, Dict, List, Optional

from src.shared.payloads import BasePayload, NonEmptyStr


class LLMRouteRequest(BasePayload):
    """Payload for POST /api/llm/route."""

    messages: List[Dict[str, Any]]
    model: Optional[NonEmptyStr] = None
    temperature: float = 0.7
    max_tokens: Optional[int] = None
