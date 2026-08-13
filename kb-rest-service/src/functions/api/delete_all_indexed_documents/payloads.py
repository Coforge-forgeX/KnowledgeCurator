"""
Payloads for Delete All Indexed Documents API

Request/response models for deleting all indexed documents in a workspace.
"""
from pydantic import BaseModel, Field


class DeleteAllIndexedDocumentsRequest(BaseModel):
    """Request model for deleting all indexed documents in a workspace."""

    workspace_id: int = Field(..., gt=0, description="Workspace ID (must be > 0)")


class DeleteAllIndexedDocumentsResponse(BaseModel):
    """Response model for delete all indexed documents operation."""

    success: bool
    message: str
    workspace_id: int
    deleted_count: int
    failed_count: int
    cleanup_summary: dict
