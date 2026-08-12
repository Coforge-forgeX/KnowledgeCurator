"""
message_gpt API Endpoint

REST equivalent of KnowledgeCurator's `message_gpt` MCP tool. Follows the
same conventions as `query_rag`/`upload_and_index`:
- @require_auth() for authentication
- parse_request() for payload validation
- a single access-validation pass (see `services.chat.access_validator`)
  shared by both SEARCH and UPDATE modes, instead of each mode re-validating
  user/workspace access independently.

POST /api/message-gpt
Headers: Authorization: Bearer <token>
Body: ChatRequest (see src/models/chat_models.py)
"""
from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, require_auth
from src.core.exceptions import AuthorizationException, ValidationException
from src.core.logging import get_logger
from src.models.chat_models import ChatRequest
from src.services.chat import get_chat_orchestrator
from src.common import create_error_response, create_internal_error_response, create_success_response, parse_request

logger = get_logger(__name__)


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """
    Process a chatbot message.

    Security:
    1. Authentication via @require_auth (Bearer token).
    2. `user_id` is taken from the token, not the body — there is nothing to
       spoof and nothing to cross-check.
    3. Workspace membership, role_id, can_curate_kb and the workspace's
       domain/kb_name/knowledge bases are resolved once from the database
       (see ChatAccessValidator), never trusted from the request body.

    Returns:
        200: ChatResponse with response/sources/task_ids/session_id
        400: Validation error
        403: Not authorized (workspace access or curator permission)
        500: Server error
    """
    correlation_id = context.correlation_id
    authenticated_user_id = get_user_id(req)

    payload = None
    try:
        payload, error_response = parse_request(req, ChatRequest)
        if error_response:
            return error_response

        logger.info(
            "message_gpt request received",
            correlation_id=correlation_id,
            user_id=authenticated_user_id,
            workspace_id=payload.workspace_id,
            session_id=payload.session_id,
            mode=payload.mode,
        )

        orchestrator = get_chat_orchestrator()
        response = await orchestrator.handle_message(payload, user_id=authenticated_user_id)

        return create_success_response(
            message="Message processed successfully",
            data=response.dict(),
            status_code=200,
            correlation_id=correlation_id,
        )

    except ValidationException as e:
        logger.warning("Validation error", error=e.message, correlation_id=correlation_id)
        return create_error_response(
            message=e.message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            correlation_id=correlation_id,
        )

    except AuthorizationException as e:
        logger.warning(
            "Authorization error",
            error=e.message,
            user_id=authenticated_user_id,
            workspace_id=payload.workspace_id if payload else None,
            correlation_id=correlation_id,
        )
        return create_error_response(
            message=e.message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403,
            correlation_id=correlation_id,
        )

    except Exception as e:
        logger.error("message_gpt failed", error=e, correlation_id=correlation_id)
        return create_internal_error_response(
            message="An error occurred while processing your message",
            error=e,
            error_code="INTERNAL_ERROR",
            correlation_id=correlation_id,
        )
