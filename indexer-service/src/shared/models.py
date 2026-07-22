"""Shared Pydantic models for Indexer Service"""
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Job processing status"""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class IndexingJob(BaseModel):
    """Indexing job model from queue message"""

    job_id: str
    workspace_id: int
    document_url: str
    kb_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class JobResult(BaseModel):
    """Job processing result"""

    success: bool
    job_id: str
    message: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: Optional[float] = None
    document_id: Optional[int] = None
    chunks_processed: Optional[int] = None
    entities_extracted: Optional[int] = None


class DocumentMetadata(BaseModel):
    """Document metadata model"""

    document_id: Optional[int] = None
    workspace_id: int
    kb_id: Optional[int] = None
    document_url: str
    document_name: Optional[str] = None
    document_type: Optional[str] = None
    size_bytes: Optional[int] = None
    status: JobStatus = JobStatus.QUEUED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    error_message: Optional[str] = None


class ProcessingStats(BaseModel):
    """Processing statistics"""

    total_chunks: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    processing_time_seconds: float = 0.0


class QueueMessage(BaseModel):
    """Azure Storage Queue message wrapper"""

    id: str
    content: str
    dequeue_count: int = 0
    insertion_time: Optional[datetime] = None
    expiration_time: Optional[datetime] = None
    next_visible_time: Optional[datetime] = None
