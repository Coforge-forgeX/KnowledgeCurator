"""Pytest configuration and fixtures for Indexer Service tests"""
from datetime import datetime

import pytest

# Import paths come from `pythonpath` in pyproject.toml [tool.pytest.ini_options].


@pytest.fixture
def sample_job_data():
    """Sample indexing job data for testing"""
    return {
        "job_id": "test-job-123",
        "workspace_id": 1,
        "document_url": "https://example.com/document.pdf",
        "kb_id": 1,
    }


@pytest.fixture
def sample_queue_message():
    """Sample Azure queue message for testing"""
    return {
        "id": "msg-123",
        "content": '{"job_id": "test-job-123", "workspace_id": 1, "document_url": "https://example.com/doc.pdf"}',
        "dequeue_count": 0,
    }


@pytest.fixture
def sample_document_metadata():
    """Sample document metadata for testing"""
    return {
        "document_id": 1,
        "workspace_id": 1,
        "kb_id": 1,
        "document_url": "https://example.com/document.pdf",
        "document_name": "document.pdf",
        "document_type": "application/pdf",
        "size_bytes": 102400,
        "status": "queued",
    }


@pytest.fixture
def mock_azure_queue_message():
    """Mock Azure Storage Queue message object"""

    class MockMessage:
        def __init__(self, id="msg-123", content="", dequeue_count=0):
            self.id = id
            self.content = content
            self.dequeue_count = dequeue_count
            self.insertion_time = datetime.utcnow()
            self.expiration_time = None
            self.next_visible_time = None

    return MockMessage
