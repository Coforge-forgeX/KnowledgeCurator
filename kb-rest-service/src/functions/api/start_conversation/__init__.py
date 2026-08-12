"""Chat API: start a conversation session.

POST /api/v2/chat/start
Headers: Authorization: Bearer <token>
Body: { "workspace_id": int }

`user_id` comes from the token — never from the body — and workspace membership
is verified against UserMap before a session is created, matching `message_gpt`.
"""

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, require_auth
from src.core.logging import get_logger
from src.models.chat_models import StartConversationRequest
from src.services.chat import get_chat_access_validator
from src.services.chat_service import get_chat_service
from src.common import create_exception_response, create_success_response, parse_request

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
    user_id = get_user_id(req)

    try:
        payload, error = parse_request(req, StartConversationRequest)
        if error:
            return error

        await get_chat_access_validator().validate_membership(
            user_id=user_id,
            workspace_id=payload.workspace_id,
        )

        service = get_chat_service()
        await service.initialize()
        data = await service.start_conversation(
            workspace_id=payload.workspace_id,
            user_id=user_id,
        )

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
