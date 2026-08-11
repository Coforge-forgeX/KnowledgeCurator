"""Chat API: start a conversation session.

POST /api/v2/chat/start
Body: { "workspace_id": int, "user_id": int }

Note: This endpoint intentionally does not require auth yet because other
kb-rest-service routes are anonymous in Azure Functions entrypoint.
If you want auth, we can add @require_auth() and derive user_id from token.
"""

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.exceptions import ValidationException
from src.core.logging import get_logger
from src.models.chat_models import ConversationHistoryRequest
from src.services.chat_service import get_chat_service
from src.shared import create_error_response, create_success_response, parse_request

logger = get_logger(__name__)


async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    correlation_id = context.correlation_id
    try:
        # Reuse ConversationHistoryRequest because it already defines workspace_id/user_id.
        payload, error = parse_request(req, ConversationHistoryRequest)
        if error:
            return error

        if not payload.workspace_id or not payload.user_id:
            raise ValidationException(message="workspace_id and user_id are required")

        service = get_chat_service()
        await service.initialize()
        data = await service.start_conversation(
            workspace_id=payload.workspace_id,
            user_id=payload.user_id,
        )

        return create_success_response(
            message="Conversation started",
            data=data,
            status_code=200,
            correlation_id=correlation_id,
        )
    except Exception as e:
        logger.error("chat_start_conversation failed", error=e)
        return create_error_response(
            message=str(e),
            error_code="CHAT_START_FAILED",
            status_code=500,
            correlation_id=correlation_id,
        )
