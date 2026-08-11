"""Chat API: list conversation sessions (history).

GET /api/v2/chat/history?workspace_id=..&user_id=..&limit=..
"""

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.exceptions import ValidationException
from src.core.logging import get_logger
from src.services.chat_service import get_chat_service
from src.shared import create_error_response, create_success_response

logger = get_logger(__name__)


async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    correlation_id = context.correlation_id
    try:
        workspace_id_raw = req.get_query_param("workspace_id")
        user_id_raw = req.get_query_param("user_id")
        limit_raw = req.get_query_param("limit")

        if not workspace_id_raw or not user_id_raw:
            raise ValidationException(message="workspace_id and user_id are required")

        workspace_id = int(workspace_id_raw)
        user_id = int(user_id_raw)
        limit = int(limit_raw) if limit_raw else None

        service = get_chat_service()
        await service.initialize()
        data = await service.get_conversation_history(
            workspace_id=workspace_id,
            user_id=user_id,
            limit=limit,
        )

        return create_success_response(
            message="Conversation history retrieved",
            data=data,
            status_code=200,
            correlation_id=correlation_id,
        )
    except Exception as e:
        logger.error("chat_get_conversation_history failed", error=e)
        return create_error_response(
            message=str(e),
            error_code="CHAT_HISTORY_FAILED",
            status_code=500,
            correlation_id=correlation_id,
        )
