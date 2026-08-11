"""Payload models for file task status API."""
from typing import List, Optional

from pydantic import Field, model_validator

from src.common.payloads import BasePayload


class FileTasksStatusRequest(BasePayload):
    """Request payload for file task status lookup."""

    file_tasks_id: Optional[List[int]] = Field(
        default=None,
        description="Preferred list of file_tasks.id values",
        min_length=1,
        max_length=500,
    )
    workspace_id: Optional[int] = Field(
        default=None,
        gt=0,
        description="Workspace ID used only when file_tasks_id is not provided",
    )

    @model_validator(mode="after")
    def validate_request(self):
        if self.file_tasks_id:
            return self
        if self.workspace_id is None:
            raise ValueError("Either file_tasks_id or workspace_id must be provided")
        return self
