"""Pytest configuration and fixtures for KB REST service tests"""
import pytest

# Import paths come from `pythonpath` in pyproject.toml [tool.pytest.ini_options].


@pytest.fixture
def sample_request_data():
    """Sample request data for testing"""
    return {"workspace_id": "test-workspace", "kb_id": "test-kb", "query": "test query"}


@pytest.fixture
def mock_azure_function_request():
    """Mock Azure Function request object"""

    class MockRequest:
        def __init__(self, params=None, body=None, method="POST"):
            self.params = params or {}
            self._body = body
            self.method = method

        def get_json(self):
            return self._body

        def get_body(self):
            return self._body

    return MockRequest


@pytest.fixture
def sample_kb_data():
    """Sample knowledge base data for testing"""
    return {
        "workspace_id": "ws-001",
        "kb_id": "kb-001",
        "documents": [
            {"id": "doc-1", "content": "Test document 1", "metadata": {"source": "test"}},
            {"id": "doc-2", "content": "Test document 2", "metadata": {"source": "test"}},
        ],
    }


@pytest.fixture
def sample_query_data():
    """Sample query data for testing"""
    return {
        "workspace_id": "ws-001",
        "kb_id": "kb-001",
        "query": "What is the test about?",
        "top_k": 5,
    }
