"""
Cancel Chat Message API Endpoint

Cancels an in-flight `message_gpt` request for a session, mirroring
KnowledgeCurator's stop-button behavior. Uses `common_adapters.cancel_convesation`,
the same cancellation primitive `message_gpt`'s orchestrator registers its
running task against (see `services.chat.orchestrator.ChatOrchestrator._run_cancellable`).

Security: Validates that the authenticated user owns the session before cancellation
to prevent DoS attacks where one user cancels another user's requests.

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
from src.common import create_error_response, create_internal_error_response, create_success_response, parse_request
from src.services.chat.access_validator import get_chat_access_validator
from src.services.mongodb_service import get_mongodb_service

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

        # Validate workspace membership to prevent cross-workspace attacks
        validator = get_chat_access_validator()
        await validator.validate_membership(
            user_id=authenticated_user_id,
            workspace_id=payload.workspace_id,
        )

        # Validate session ownership to prevent users from cancelling each other's sessions
        # This prevents DoS attacks where User A cancels User B's request
        mongo = get_mongodb_service()
        await mongo.initialize()

        session = await mongo.get_session(
            session_id=payload.session_id,
            workspace_id=payload.workspace_id,
            user_id=authenticated_user_id,
        )

        if not session:
            logger.warning(
                "Attempted to cancel non-existent or unauthorized session",
                correlation_id=correlation_id,
                user_id=authenticated_user_id,
                workspace_id=payload.workspace_id,
                session_id=payload.session_id,
            )
            raise AuthorizationException(
                message="Session not found or you do not have permission to cancel it"
            )

        # Security checks passed - proceed with cancellation
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
        return create_internal_error_response(
            message="An error occurred while cancelling the message",
            error=e,
            error_code="INTERNAL_ERROR",
            correlation_id=correlation_id,
        )
