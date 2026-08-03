"""KB chatbot endpoint - TODO: Port from kb_curator_chatbot.py"""
from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.shared.payloads import parse_request
from src.shared.response_utils import create_error_response

from .payloads import KBChatRequest


async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """
    KB chatbot endpoint.

    POST /api/kb/chat
    Body: {
        "workspace_id": 1,
        "messages": [{"role": "user", "content": "Hello"}],
        "kb_id": 1
    }
    """
    # Validate request payload
    payload, error = parse_request(req, KBChatRequest)
    if error:
        return error

    # TODO: Implement chatbot logic from kb_curator_chatbot.py
    return create_error_response(
        message="TODO: Port chatbot from kb_curator_chatbot.py",
        error_code="NOT_IMPLEMENTED",
        status_code=501,
        correlation_id=context.correlation_id,
    )
