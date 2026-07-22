"""
Indexing State Management

Tracks indexing job state and checkpoints for retry/resume functionality.
"""
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class IndexingState(str, Enum):
    """Indexing job states"""

    PENDING = "pending"  # Job received, not started
    DOWNLOADING = "downloading"  # Downloading file from storage
    DOWNLOADED = "downloaded"  # File downloaded successfully
    EXTRACTING = "extracting"  # Extracting text from document
    EXTRACTED = "extracted"  # Text extracted successfully
    INDEXING = "indexing"  # Indexing with LightRAG
    INDEXED = "indexed"  # Indexed successfully
    UPDATING_METADATA = "updating_metadata"  # Updating database
    COMPLETED = "completed"  # All steps completed
    FAILED = "failed"  # Job failed
    RETRYING = "retrying"  # Job being retried


class CheckpointData(BaseModel):
    """Data stored at each checkpoint"""

    # Download checkpoint
    file_downloaded: bool = False
    file_size: Optional[int] = None
    content_type: Optional[str] = None

    # Extraction checkpoint
    text_extracted: bool = False
    extracted_text_path: Optional[str] = None  # Path to cached extracted text
    extraction_method: Optional[str] = None
    page_count: Optional[int] = None

    # Indexing checkpoint
    indexed: bool = False
    doc_id: Optional[str] = None
    chunks_processed: Optional[int] = None

    # Metadata checkpoint
    metadata_updated: bool = False

    # Additional context
    error_message: Optional[str] = None
    retry_count: int = 0
    last_retry_at: Optional[datetime] = None


class IndexingJobState(BaseModel):
    """Complete state of an indexing job"""

    job_id: str
    workspace_id: int
    document_url: str
    kb_id: Optional[int] = None

    # State tracking
    state: IndexingState = IndexingState.PENDING
    checkpoint: CheckpointData = Field(default_factory=CheckpointData)

    # Timing
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Retry configuration
    max_retries: int = 3
    retry_count: int = 0
    retry_backoff_seconds: int = 60  # Base backoff (exponential)

    # Error tracking
    last_error: Optional[str] = None
    error_history: list[Dict[str, Any]] = Field(default_factory=list)

    class Config:
        use_enum_values = True

    def can_retry(self) -> bool:
        """Check if job can be retried"""
        return self.retry_count < self.max_retries

    def should_resume(self) -> bool:
        """Check if job should resume from checkpoint"""
        return self.state in [
            IndexingState.DOWNLOADED,
            IndexingState.EXTRACTED,
            IndexingState.INDEXED,
        ]

    def get_retry_delay(self) -> int:
        """Calculate exponential backoff delay in seconds"""
        return self.retry_backoff_seconds * (2**self.retry_count)

    def record_error(self, error: str, state: IndexingState) -> None:
        """Record error and update state"""
        self.last_error = error
        self.error_history.append(
            {
                "error": error,
                "state": state.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "retry_count": self.retry_count,
            }
        )
        self.updated_at = datetime.now(timezone.utc)

    def increment_retry(self) -> None:
        """Increment retry count"""
        self.retry_count += 1
        self.checkpoint.retry_count = self.retry_count
        self.checkpoint.last_retry_at = datetime.now(timezone.utc)
        self.state = IndexingState.RETRYING
        self.updated_at = datetime.now(timezone.utc)

    def mark_started(self) -> None:
        """Mark job as started"""
        if not self.started_at:
            self.started_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def mark_completed(self) -> None:
        """Mark job as completed"""
        self.state = IndexingState.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)

    def mark_failed(self) -> None:
        """Mark job as failed"""
        self.state = IndexingState.FAILED
        self.updated_at = datetime.now(timezone.utc)
