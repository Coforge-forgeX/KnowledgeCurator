"""Payload models for List Indexed Documents API"""
from pydantic import Field

from src.shared.payloads import BasePayload


class ListIndexedDocumentsRequest(BasePayload):
    """Request payload for listing indexed documents"""

    workspace_id: int = Field(..., gt=0, description="Workspace ID")
    limit: int = Field(default=100, gt=0, le=1000, description="Maximum documents to return per page")
    offset: int = Field(default=0, ge=0, description="Number of documents to skip for pagination")
