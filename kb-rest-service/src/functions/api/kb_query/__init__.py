"""KB query endpoint - TODO: Port from kb_adapter_tool.py"""
from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.shared.payloads import parse_request
from src.shared.response_utils import create_error_response

from .payloads import KBQueryRequest


async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """
    Query knowledge base.

    POST /api/kb/query
    Body: {
        "workspace_id": 1,
        "query": "What is...",
        "kb_id": 1,
        "top_k": 5
    }
    """
    # Validate request payload
    payload, error = parse_request(req, KBQueryRequest)
    if error:
        return error

    # TODO: Implement query logic from kb_adapter_tool.py
    return create_error_response(
        message="TODO: Port query_kb from kb_adapter_tool.py",
        error_code="NOT_IMPLEMENTED",
        status_code=501,
        correlation_id=context.correlation_id,
    )
