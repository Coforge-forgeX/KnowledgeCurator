"""Test Pydantic models"""
import pytest

from shared.models import (
    DocumentMetadata,
    IndexingJob,
    JobResult,
    JobStatus,
    ProcessingStats,
)


def test_indexing_job_creation(sample_job_data):
    """Test IndexingJob model creation"""
    job = IndexingJob(**sample_job_data)
    assert job.job_id == "test-job-123"
    assert job.workspace_id == 1
    assert job.document_url == "https://example.com/document.pdf"
    assert job.kb_id == 1


def test_indexing_job_optional_fields():
    """Test IndexingJob with optional fields"""
    job = IndexingJob(
        job_id="test-123", workspace_id=1, document_url="https://example.com/doc.pdf"
    )
    assert job.kb_id is None
    assert job.metadata is None


def test_job_result_success():
    """Test JobResult for successful job"""
    result = JobResult(
        success=True,
        job_id="test-123",
        message="Processing completed",
        duration_seconds=5.2,
        document_id=1,
        chunks_processed=10,
    )
    assert result.success is True
    assert result.job_id == "test-123"
    assert result.error is None


def test_job_result_failure():
    """Test JobResult for failed job"""
    result = JobResult(success=False, job_id="test-123", error="Processing failed")
    assert result.success is False
    assert result.error == "Processing failed"


def test_document_metadata_creation(sample_document_metadata):
    """Test DocumentMetadata model creation"""
    metadata = DocumentMetadata(**sample_document_metadata)
    assert metadata.document_id == 1
    assert metadata.workspace_id == 1
    assert metadata.status == JobStatus.QUEUED


def test_document_metadata_status_enum():
    """Test DocumentMetadata status enum"""
    metadata = DocumentMetadata(
        workspace_id=1,
        document_url="https://example.com/doc.pdf",
        status=JobStatus.PROCESSING,
    )
    assert metadata.status == JobStatus.PROCESSING
    assert metadata.status.value == "processing"


def test_processing_stats_defaults():
    """Test ProcessingStats default values"""
    stats = ProcessingStats()
    assert stats.total_chunks == 0
    assert stats.total_entities == 0
    assert stats.total_relationships == 0
    assert stats.processing_time_seconds == 0.0


def test_processing_stats_with_values():
    """Test ProcessingStats with values"""
    stats = ProcessingStats(
        total_chunks=50,
        total_entities=100,
        total_relationships=200,
        processing_time_seconds=10.5,
    )
    assert stats.total_chunks == 50
    assert stats.total_entities == 100
    assert stats.total_relationships == 200
    assert stats.processing_time_seconds == 10.5
