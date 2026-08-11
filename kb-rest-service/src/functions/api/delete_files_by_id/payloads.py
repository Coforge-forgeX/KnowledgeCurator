"""Payload models for deleting files by file_id or direct file_path."""
from typing import List, Optional, Union

from pydantic import Field, field_validator, model_validator

from src.common.payloads import BasePayload


class DeleteFilesByIdRequest(BasePayload):
    """Request payload for deleting files by file_id(s) or file_path(s)."""

    workspace_id: int = Field(..., gt=0, description="Workspace ID")
    file_id: Optional[List[str]] = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="File token(s) returned by documents/query endpoints",
    )
    file_path: Optional[List[str]] = Field(
        default=None,
        min_length=1,
        max_length=200,
        description="Direct file path(s) for recovery cleanup when file_id is unavailable",
    )

    @field_validator("file_id", mode="before")
    @classmethod
    def normalize_file_id(cls, value: Union[str, List[str]]):
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("file_id must be a string or list of strings")
        normalized = [str(item).strip() for item in value if str(item).strip()]
        if not normalized:
            raise ValueError("file_id cannot be empty")
        return normalized

    @field_validator("file_path", mode="before")
    @classmethod
    def normalize_file_path(cls, value: Union[str, List[str]]):
        if value is None:
            return None
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list):
            raise ValueError("file_path must be a string or list of strings")
        normalized = [str(item).strip() for item in value if str(item).strip()]
        if not normalized:
            raise ValueError("file_path cannot be empty")
        return normalized

    @model_validator(mode="after")
    def validate_inputs_present(self):
        if not self.file_id and not self.file_path:
            raise ValueError("Either file_id or file_path must be provided")
        return self
