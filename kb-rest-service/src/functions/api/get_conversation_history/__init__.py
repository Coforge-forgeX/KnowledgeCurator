"""Chat API: list conversation sessions (history), paginated.

GET /api/v2/chat/history?workspace_id=..&page=1&page_size=20
Headers: Authorization: Bearer <token>

`user_id` comes from the token, so a caller can only ever list their own
conversations. `limit` is still accepted as a deprecated alias for `page_size`.
"""

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, require_auth
from src.core.logging import get_logger
from src.models.chat_models import ConversationHistoryRequest
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
    List the caller's conversation sessions in a workspace.

    Returns:
        200: sessions in `data.items`, with page/page_size/total_count/
             total_pages/has_next/has_previous under `pagination`
        400: Validation error
        401: Missing/invalid token
        403: Caller is not an active member of the workspace
        500: Server error
    """
    correlation_id = context.correlation_id
    user_id = get_user_id(req)

    try:
        payload, error = parse_request(req, ConversationHistoryRequest)
        if error:
            return error

        page_size = payload.effective_page_size

        await get_chat_access_validator().validate_membership(
            user_id=user_id,
            workspace_id=payload.workspace_id,
        )

        service = get_chat_service()
        await service.initialize()
        result = await service.get_conversation_history(
            workspace_id=payload.workspace_id,
            user_id=user_id,
            page=payload.page,
            page_size=page_size,
        )

        return create_paginated_response(
            message="Conversation history retrieved",
            items=result["items"],
            page=result["page"],
            page_size=result["page_size"],
            total_count=result["total_count"],
            status_code=200,
            correlation_id=correlation_id,
        )
    except Exception as e:
        logger.error(
            "chat_get_conversation_history failed",
            error=e,
            user_id=user_id,
            correlation_id=correlation_id,
        )
        return create_exception_response(
            e,
            fallback_message="An error occurred while retrieving the conversation history",
            fallback_error_code="CHAT_HISTORY_FAILED",
            correlation_id=correlation_id,
        )
