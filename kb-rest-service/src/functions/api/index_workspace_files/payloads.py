"""Payload models for indexing existing workspace files."""

from typing import List, Optional

from pydantic import BaseModel, Field


class IndexWorkspaceFilesRequest(BaseModel):
    """Request payload for queueing all existing workspace files for indexing."""

    workspace_id: int = Field(..., gt=0, description="Workspace ID")


class IndexedFileTaskResponse(BaseModel):
    """Queued file task details."""

    task_id: int
    file_name: str
    file_path: str
    status: str


class IndexWorkspaceFilesResponse(BaseModel):
    """Response payload for index workspace files endpoint."""

    success: bool
    message: str
    workspace_id: int
    total_blobs_scanned: int
    queued_files: int
    tasks: List[IndexedFileTaskResponse]
    failed_files: List[str] = Field(default_factory=list)
    skipped_files: List[str] = Field(default_factory=list)
    kb_id: Optional[int] = None