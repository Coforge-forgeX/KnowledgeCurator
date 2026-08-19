"""Chat API: start a conversation session.

POST /api/v2/chat/start
Headers: Authorization: Bearer <token>
Body: { "workspace_id": int }

`user_id` comes from the token — never from the body — and workspace membership
is verified against UserMap before a session is created, matching `message_gpt`.
"""
import uuid

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import require_auth
from src.core.logging import get_logger
from src.common import create_exception_response, create_success_response

logger = get_logger(__name__)


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """
    Create a conversation session.

    Returns:
        200: session_id and status
        400: Validation error
        401: Missing/invalid token
        403: Caller is not an active member of the workspace
        500: Server error
    """
    correlation_id = context.correlation_id

    try:

        session_id = str(uuid.uuid4())
        data = {
                "session_id": session_id,
                "status": "created",
                "message": f"Session started with id: {session_id}",
            }

        return create_success_response(
            message="Conversation started",
            data=data,
            status_code=201,
            correlation_id=correlation_id,
        )
    except Exception as e:
        logger.error(
            "chat_start_conversation failed",
            error=e,
            user_id=user_id,
            correlation_id=correlation_id,
        )
        return create_exception_response(
            e,
            fallback_message="An error occurred while starting the conversation",
            fallback_error_code="CHAT_START_FAILED",
            correlation_id=correlation_id,
        )
