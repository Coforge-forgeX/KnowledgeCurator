"""Payload models for grouped workspace documents API."""
from pydantic import Field

from src.shared.payloads import BasePayload


class WorkspaceDocumentsGroupedRequest(BasePayload):
    """Request payload for grouped workspace document listing."""

    workspace_id: int = Field(..., gt=0, description="Workspace ID")
