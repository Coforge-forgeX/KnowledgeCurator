"""
Upload API Models

Pydantic models for file upload REST endpoint.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, validator

from src.helpers.file_validation import SUPPORTED_FILE_EXTENSIONS


class FileMetadata(BaseModel):
    """File metadata for upload"""

    file_name: str = Field(..., min_length=1)
    file_content: str = Field(..., description="Base64 encoded file content")
    file_size: Optional[int] = Field(None, ge=0)

    @validator("file_name")
    def validate_file_extension(cls, v):
        """Validate file has supported extension - uses centralized validation"""
        if "." not in v:
            raise ValueError("File name must have extension")

        # Use centralized supported extensions list
        if not any(v.lower().endswith(ext) for ext in SUPPORTED_FILE_EXTENSIONS):
            raise ValueError(
                f"File extension not supported. "
                f"Valid: {', '.join(SUPPORTED_FILE_EXTENSIONS)}"
            )
        return v


class UploadRequest(BaseModel):
    """POST /upload-document request"""

    workspace_id: int = Field(..., gt=0)
    user_id: int = Field(..., gt=0)
    role_id: int = Field(..., gt=0)
    domain: str = Field(...)
    kb_name: str = Field(...)
    files: List[FileMetadata] = Field(..., min_items=1)
    container_name: Optional[str] = None

    @validator("files")
    def validate_files_limit(cls, v):
        if len(v) > 10:
            raise ValueError("Maximum 10 files per upload")
        return v


class UploadResponse(BaseModel):
    """POST /upload-document response"""

    status: str
    message: str
    task_ids: List[int] = Field(default_factory=list)
    failed_files: List[str] = Field(default_factory=list)
