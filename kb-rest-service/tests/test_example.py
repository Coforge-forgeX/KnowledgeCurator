"""Example test file for KB REST service"""
import pytest

from shared.models import APIResponse, KBQueryRequest
from shared.payloads import BasePayload, NonEmptyStr, parse_request


def test_api_response_creation():
    """Test APIResponse model creation"""
    response = APIResponse(success=True, message="Test successful", data={"key": "value"})
    assert response.success is True
    assert response.message == "Test successful"
    assert response.data == {"key": "value"}


def test_kb_query_request_validation():
    """Test KBQueryRequest validation"""
    valid_data = {
        "workspace_id": "ws-001",
        "kb_id": "kb-001",
        "query": "test query",
        "top_k": 5,
    }
    request = KBQueryRequest(**valid_data)
    assert request.workspace_id == "ws-001"
    assert request.kb_id == "kb-001"
    assert request.query == "test query"
    assert request.top_k == 5


def test_kb_query_request_default_top_k():
    """Test KBQueryRequest default top_k value"""
    data = {"workspace_id": "ws-001", "kb_id": "kb-001", "query": "test query"}
    request = KBQueryRequest(**data)
    assert request.top_k == 5


def test_base_payload_forbids_extra_fields():
    """Test that BasePayload rejects unknown fields"""

    class TestPayload(BasePayload):
        field1: str

    with pytest.raises(Exception):
        TestPayload(field1="value", unknown_field="should fail")


def test_parse_request_valid(mock_azure_function_request):
    """Test parse_request with valid data"""

    class TestPayload(BasePayload):
        workspace_id: NonEmptyStr
        query: str

    valid_data = {"workspace_id": "ws-001", "query": "test"}
    mock_req = mock_azure_function_request(body=valid_data)

    payload, error = parse_request(mock_req, TestPayload)
    assert error is None
    assert payload is not None
    assert payload.workspace_id == "ws-001"
    assert payload.query == "test"


def test_parse_request_invalid(mock_azure_function_request):
    """Test parse_request with invalid data"""

    class TestPayload(BasePayload):
        workspace_id: NonEmptyStr
        query: str

    invalid_data = {"workspace_id": "", "query": "test"}  # empty workspace_id
    mock_req = mock_azure_function_request(body=invalid_data)

    payload, error = parse_request(mock_req, TestPayload)
    assert payload is None
    assert error is not None
    assert error.status_code == 400
