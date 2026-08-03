"""
Indexing Models (Shared between kb-rest-service and indexer-service)

These models are used by both services:
- kb-rest-service: Creates IndexingJob and enqueues to Azure Queue
- indexer-service: Reads IndexingJob from queue and processes
"""

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class IndexingStatusEnum(str, Enum):
    """Indexing job status"""
    PENDING = "pending"
    UPLOADING = "uploading"
    UPLOADED = "uploaded"
    QUEUED = "queued"
    INDEXING = "indexing"
    INDEXED = "indexed"
    FAILED = "failed"


class IndexingJob(BaseModel):
    """
    Indexing job payload for Azure Storage Queue.

    Used by:
    - kb-rest-service: Creates and enqueues job
    - indexer-service: Receives and processes job
    """

    job_id: str = Field(..., description="Unique job identifier (UUID)")
    workspace_id: int = Field(..., description="Workspace identifier")
    document_url: str = Field(..., description="Document blob URL to index")
    kb_id: str = Field(..., description="Knowledge base identifier (domain_kb_name)")
    domain: str = Field(..., description="Domain/industry name")
    kb_name: str = Field(..., description="Knowledge base/sub-industry name")
    file_name: str = Field(..., description="Original file name")
    user_id: int = Field(None, description="User who uploaded the file")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Job creation timestamp")

    class Config:
        """Pydantic config"""
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
