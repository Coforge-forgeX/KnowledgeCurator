"""Chat API: delete conversation session.

DELETE /api/v2/chat/session/delete
Headers: Authorization: Bearer <token>
Body: { "workspace_id": int, "session_id": str }

DELETE is the verb for this operation; `parse_request` reads the JSON body for
DELETE just as it does for POST.

`user_id` comes from the token, so the delete is always scoped to the caller's
own conversation, and a session that does not exist returns 404 rather than a
misleading success.
"""

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, require_auth
from src.core.logging import get_logger
from src.models.chat_models import SessionDeleteRequest
from src.services.chat import get_chat_access_validator
from src.services.chat_service import get_chat_service
from src.common import create_exception_response, create_success_response, parse_request

logger = get_logger(__name__)


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """
    Delete a conversation session and its messages.

    Returns:
        200: deletion status
        400: Validation error
        401: Missing/invalid token
        403: Caller is not an active member of the workspace
        404: No such conversation for this user/workspace
        500: Server error
    """
    correlation_id = context.correlation_id
    user_id = get_user_id(req)

    try:
        payload, error = parse_request(req, SessionDeleteRequest)
        if error:
            return error

        await get_chat_access_validator().validate_membership(
            user_id=user_id,
            workspace_id=payload.workspace_id,
        )

        service = get_chat_service()
        await service.initialize()
        data = await service.delete_conversation(
            session_id=payload.session_id,
            workspace_id=payload.workspace_id,
            user_id=user_id,
        )

        return create_success_response(
            message="Conversation deleted",
            data=data,
            status_code=200,
            correlation_id=correlation_id,
        )
    except Exception as e:
        logger.error(
            "chat_delete_conversation failed",
            error=e,
            user_id=user_id,
            correlation_id=correlation_id,
        )
        return create_exception_response(
            e,
            fallback_message="An error occurred while deleting the conversation",
            fallback_error_code="CHAT_DELETE_FAILED",
            correlation_id=correlation_id,
        )
