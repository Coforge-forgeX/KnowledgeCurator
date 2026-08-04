"""Request payload model for GET /api/sharepoint/list."""
from typing import Optional

from src.shared.payloads import BasePayload


class SharePointListRequest(BasePayload):
    """Payload for GET /api/sharepoint/list."""

    workspace_id: Optional[int] = None
    site_url: Optional[str] = None
