"""Chat API: rename conversation session.

POST /api/v2/chat/session/rename
Body: { "workspace_id": int, "user_id": int, "session_id": str, "title": str }
"""

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.logging import get_logger
from src.models.chat_models import SessionRenameRequest
from src.services.chat_service import get_chat_service
from src.shared import create_error_response, create_success_response, parse_request

logger = get_logger(__name__)


async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    correlation_id = context.correlation_id
    try:
        payload, error = parse_request(req, SessionRenameRequest)
        if error:
            return error

        service = get_chat_service()
        await service.initialize()
        data = await service.rename_conversation(
            session_id=payload.session_id,
            workspace_id=payload.workspace_id,
            user_id=payload.user_id,
            title=payload.title,
        )

        return create_success_response(
            message="Conversation renamed",
            data=data,
            status_code=200,
            correlation_id=correlation_id,
        )
    except Exception as e:
        logger.error("chat_rename_conversation failed", error=e)
        return create_error_response(
            message=str(e),
            error_code="CHAT_RENAME_FAILED",
            status_code=500,
            correlation_id=correlation_id,
        )
