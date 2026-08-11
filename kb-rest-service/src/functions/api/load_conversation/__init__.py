"""Chat API: load one conversation + messages.

GET /api/v2/chat/load?session_id=..&workspace_id=..&user_id=..
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
        session_id = (req.get_query_param("session_id") or "").strip()
        workspace_id_raw = req.get_query_param("workspace_id")
        user_id_raw = req.get_query_param("user_id")

        if not session_id or not workspace_id_raw or not user_id_raw:
            raise ValidationException(message="session_id, workspace_id and user_id are required")

        workspace_id = int(workspace_id_raw)
        user_id = int(user_id_raw)

        service = get_chat_service()
        await service.initialize()
        data = await service.load_conversation(
            session_id=session_id,
            workspace_id=workspace_id,
            user_id=user_id,
        )

        return create_success_response(
            message="Conversation loaded",
            data=data,
            status_code=200,
            correlation_id=correlation_id,
        )
    except Exception as e:
        logger.error("chat_load_conversation failed", error=e)
        return create_error_response(
            message=str(e),
            error_code="CHAT_LOAD_FAILED",
            status_code=500,
            correlation_id=correlation_id,
        )
