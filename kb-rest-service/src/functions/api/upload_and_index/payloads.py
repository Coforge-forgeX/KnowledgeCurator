"""
Upload and Index API - Request/Response Models

Optimized payload validation for file upload and indexing.
"""

from typing import List
from pydantic import BaseModel, Field, field_validator

from helpers.file_validation import validate_file_extension


class FileUpload(BaseModel):
    """Individual file upload"""

    file_name: str = Field(..., min_length=1, max_length=255)
    file_content: str = Field(..., description="Base64 encoded file content")

    @field_validator("file_name")
    @classmethod
    def validate_file_name(cls, v: str) -> str:
        """Validate file extension using centralized validation"""
        return validate_file_extension(v)


class UploadAndIndexRequest(BaseModel):
    """
    POST /api/upload-and-index request

    Simplified payload - workspace_id is the only required context.
    Domain, KB name, and container info are derived from workspace.
    """

    workspace_id: int = Field(..., gt=0, description="Workspace ID")
    files: List[FileUpload] = Field(..., min_length=1, description="Files to upload")

    @field_validator("files")
    @classmethod
    def validate_files_limit(cls, v: List[FileUpload]) -> List[FileUpload]:
        """Limit number of files per request"""
        max_files = 10
        if len(v) > max_files:
            raise ValueError(f"Maximum {max_files} files allowed per request")
        return v

    @field_validator("files")
    @classmethod
    def validate_duplicate_filenames(cls, v: List[FileUpload]) -> List[FileUpload]:
        """Check for duplicate file names"""
        file_names = [f.file_name for f in v]
        if len(file_names) != len(set(file_names)):
            raise ValueError("Duplicate file names not allowed")
        return v


class FileTaskResponse(BaseModel):
    """Individual file task response"""

    task_id: int
    file_name: str
    file_path: str
    status: str


class UploadAndIndexResponse(BaseModel):
    """POST /api/upload-and-index response"""

    success: bool
    message: str
    workspace_id: int
    total_files: int
    tasks: List[FileTaskResponse]
    failed_files: List[str] = Field(default_factory=list)
