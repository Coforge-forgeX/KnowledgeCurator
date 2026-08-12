"""
Cancel Chat Message API Endpoint

Cancels an in-flight `message_gpt` request for a session, mirroring
KnowledgeCurator's stop-button behavior. Uses `common_adapters.cancel_convesation`,
the same cancellation primitive `message_gpt`'s orchestrator registers its
running task against (see `services.chat.orchestrator.ChatOrchestrator._run_cancellable`).

POST /api/message-gpt/cancel
Headers: Authorization: Bearer <token>
Body: CancelChatRequest
"""
from common_adapters.cancel_convesation import cancel_conversation

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, require_auth
from src.core.exceptions import AuthorizationException, ValidationException
from src.core.logging import get_logger
from src.models.chat_models import CancelChatRequest
from src.common import create_error_response, create_success_response, parse_request

logger = get_logger(__name__)


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    correlation_id = context.correlation_id
    authenticated_user_id = get_user_id(req)

    try:
        payload, error_response = parse_request(req, CancelChatRequest)
        if error_response:
            return error_response

        logger.info(
            "Cancelling chat message",
            correlation_id=correlation_id,
            user_id=authenticated_user_id,
            workspace_id=payload.workspace_id,
            session_id=payload.session_id,
        )

        cancel_conversation(
            conversation_id=payload.session_id,
            workspace_id=str(payload.workspace_id),
            user_id=str(authenticated_user_id),
            reason=payload.reason,
        )

        return create_success_response(
            message="Cancellation requested",
            data={"cancelled": True, "session_id": payload.session_id},
            status_code=200,
            correlation_id=correlation_id,
        )

    except ValidationException as e:
        return create_error_response(
            message=e.message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            correlation_id=correlation_id,
        )

    except AuthorizationException as e:
        return create_error_response(
            message=e.message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
            correlation_id=correlation_id,
        )

    except Exception as e:
        logger.error("cancel_chat_message failed", error=e, correlation_id=correlation_id)
        return create_error_response(
            message="An error occurred while cancelling the message",
            error_code="INTERNAL_ERROR",
            details={"error": str(e)},
            status_code=500,
            correlation_id=correlation_id,
        )
