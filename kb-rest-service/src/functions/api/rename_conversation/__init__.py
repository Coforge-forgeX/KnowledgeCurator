"""Chat API: rename conversation session.

POST /api/v2/chat/session/rename
Headers: Authorization: Bearer <token>
Body: { "workspace_id": int, "session_id": str, "title": str }

`user_id` comes from the token, so a caller cannot rename another user's
conversation by passing their id.
"""

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, require_auth
from src.core.logging import get_logger
from src.models.chat_models import SessionRenameRequest
from src.services.chat import get_chat_access_validator
from src.services.chat_service import get_chat_service
from src.common import create_exception_response, create_success_response, parse_request

logger = get_logger(__name__)


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """
    Rename a conversation session.

    Returns:
        200: session_id + new title
        400: Validation error
        401: Missing/invalid token
        403: Caller is not an active member of the workspace
        404: No such conversation for this user/workspace
        500: Server error
    """
    correlation_id = context.correlation_id
    user_id = get_user_id(req)

    try:
        payload, error = parse_request(req, SessionRenameRequest)
        if error:
            return error

        await get_chat_access_validator().validate_membership(
            user_id=user_id,
            workspace_id=payload.workspace_id,
        )

        service = get_chat_service()
        await service.initialize()
        data = await service.rename_conversation(
            session_id=payload.session_id,
            workspace_id=payload.workspace_id,
            user_id=user_id,
            title=payload.title,
        )

        return create_success_response(
            message="Conversation renamed",
            data=data,
            status_code=200,
            correlation_id=correlation_id,
        )
    except Exception as e:
        logger.error(
            "chat_rename_conversation failed",
            error=e,
            user_id=user_id,
            correlation_id=correlation_id,
        )
        return create_exception_response(
            e,
            fallback_message="An error occurred while renaming the conversation",
            fallback_error_code="CHAT_RENAME_FAILED",
            correlation_id=correlation_id,
        )
