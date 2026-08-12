"""Chat API: load one conversation + a page of its messages.

GET /api/v2/chat/load?session_id=..&workspace_id=..&page=1&page_size=50&order=desc
Headers: Authorization: Bearer <token>

`user_id` comes from the token, so the (session, workspace, user) scope the
service queries on can only ever match the caller's own conversation.

The transcript is paginated: `order=desc` (the default) puts the newest messages
on page 1 so opening a long conversation stays cheap — see
`LoadConversationRequest`.
"""

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, require_auth
from src.core.logging import get_logger
from src.models.chat_models import LoadConversationRequest
from src.services.chat import get_chat_access_validator
from src.services.chat_service import get_chat_service
from src.common import (
    create_exception_response,
    create_paginated_response,
    parse_request,
)

logger = get_logger(__name__)


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """
    Load a conversation and one page of its messages.

    Returns:
        200: messages in `data.items`, with `session_id`/`session_metadata`
             beside them inside `data`, and the counters under `pagination`
        400: Validation error
        401: Missing/invalid token
        403: Caller is not an active member of the workspace
        404: No such conversation for this user/workspace
        500: Server error
    """
    correlation_id = context.correlation_id
    user_id = get_user_id(req)

    try:
        payload, error = parse_request(req, LoadConversationRequest)
        if error:
            return error

        await get_chat_access_validator().validate_membership(
            user_id=user_id,
            workspace_id=payload.workspace_id,
        )

        service = get_chat_service()
        await service.initialize()
        data = await service.load_conversation(
            session_id=payload.session_id,
            workspace_id=payload.workspace_id,
            user_id=user_id,
            page=payload.page,
            page_size=payload.page_size,
            newest_first=payload.newest_first,
        )

        return create_paginated_response(
            message="Conversation loaded",
            items=data["messages"],
            page=data["page"],
            page_size=data["page_size"],
            total_count=data["total_count"],
            status_code=200,
            correlation_id=correlation_id,
            extra={
                "session_id": data["session_id"],
                "session_metadata": data["session_metadata"],
                "message_count": data["message_count"],
                "order": payload.order,
            },
        )
    except Exception as e:
        logger.error(
            "chat_load_conversation failed",
            error=e,
            user_id=user_id,
            correlation_id=correlation_id,
        )
        return create_exception_response(
            e,
            fallback_message="An error occurred while loading the conversation",
            fallback_error_code="CHAT_LOAD_FAILED",
            correlation_id=correlation_id,
        )
