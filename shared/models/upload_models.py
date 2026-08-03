"""
Upload & Indexing Pydantic Models

Data models for file upload and indexing operations.
"""

from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, validator


class IndexingStatusEnum(str, Enum):
    """Indexing job status"""
    PENDING = "pending"
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    QUEUED = "queued"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class FileMetadata(BaseModel):
    """File metadata for upload"""

    file_name: str = Field(..., min_length=1, description="File name with extension")
    file_content: str = Field(..., description="File content (base64 encoded)")
    file_size: Optional[int] = Field(None, ge=0, description="File size in bytes")
    content_type: Optional[str] = Field(None, description="MIME type")

    @validator("file_name")
    def validate_file_extension(cls, v):
        """Validate file has extension"""
        if "." not in v:
            raise ValueError("File name must have extension")

        valid_extensions = [".pdf", ".docx", ".doc", ".txt", ".xlsx", ".xls", ".pptx", ".ppt", ".csv"]
        if not any(v.lower().endswith(ext) for ext in valid_extensions):
            raise ValueError(f"File extension not supported. Valid: {', '.join(valid_extensions)}")
        return v


class UploadRequest(BaseModel):
    """Request payload for file upload"""

    workspace_id: int = Field(..., description="Workspace identifier", gt=0)
    user_id: int = Field(..., description="User identifier", gt=0)
    role_id: int = Field(..., description="User role identifier", gt=0)
    domain: str = Field(..., description="Domain/Industry name")
    kb_name: str = Field(..., description="Knowledge base name")
    files: List[FileMetadata] = Field(..., min_items=1, description="Files to upload")
    container_name: Optional[str] = Field(None, description="Azure blob container name")

    @validator("files")
    def validate_files_not_empty(cls, v):
        if not v:
            raise ValueError("At least one file required")
        if len(v) > 10:
            raise ValueError("Maximum 10 files per upload")
        return v


class UploadResponse(BaseModel):
    """Response payload for file upload"""

    status: str = Field(..., description="Upload status")
    message: str = Field(..., description="Status message")
    task_ids: List[int] = Field(default_factory=list, description="Created task IDs for tracking")
    failed_files: List[str] = Field(default_factory=list, description="Files that failed to upload")


class IndexingJob(BaseModel):
    """Indexing job payload for queue"""

    job_id: str = Field(..., description="Unique job identifier")
    workspace_id: int = Field(..., description="Workspace identifier")
    document_url: str = Field(..., description="Document blob URL to index")
    kb_id: str = Field(..., description="Knowledge base identifier")
    domain: str = Field(..., description="Domain name")
    kb_name: str = Field(..., description="Knowledge base name")
    file_name: str = Field(..., description="Original file name")
    user_id: Optional[int] = Field(None, description="User who uploaded")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Job creation timestamp")

    class Config:
        """Pydantic config"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class IndexingStatus(BaseModel):
    """Indexing job status"""

    task_id: int = Field(..., description="Database task ID")
    job_id: Optional[str] = Field(None, description="Queue job ID")
    file_name: str = Field(..., description="File name")
    status: IndexingStatusEnum = Field(..., description="Current status")
    progress: Optional[int] = Field(None, ge=0, le=100, description="Progress percentage")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    created_at: datetime = Field(..., description="Task creation timestamp")
    updated_at: datetime = Field(..., description="Last update timestamp")

    class Config:
        """Pydantic config"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class IndexingStatusRequest(BaseModel):
    """Request to check indexing status"""

    workspace_id: int = Field(..., description="Workspace identifier", gt=0)
    user_id: int = Field(..., description="User identifier", gt=0)
    task_ids: List[int] = Field(..., min_items=1, description="Task IDs to check")


class IndexingStatusResponse(BaseModel):
    """Response with indexing statuses"""

    statuses: List[IndexingStatus] = Field(..., description="List of task statuses")
