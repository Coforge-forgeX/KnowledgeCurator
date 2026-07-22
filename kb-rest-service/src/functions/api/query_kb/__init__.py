"""Query Knowledge Base API - Following SOLID & Clean Architecture"""
import azure.functions as func

from core import (
    get_logger,
    get_user_id,
    get_workspace_ids,
    require_auth,
    azure_http_decorator,
    ValidationException,
    AuthorizationException,
)
from services.kb_service import get_kb_service
from shared import (
    ErrorMessages,
    SuccessMessages,
    create_query_response,
    create_error_response,
    parse_request,
)
from shared.payloads import QueryRequestPayload

logger = get_logger(__name__)


@azure_http_decorator()
@require_auth()
async def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """
    Query knowledge base endpoint.

    POST /api/query-kb
    Headers: Authorization: Bearer <token>
    Body: {
        "query": "What is LightRAG?",
        "workspace_id": 1,
        "mode": "hybrid",
        "only_need_context": false
    }

    Returns:
        200: Query results with answer and sources
        400: Validation error
        403: Not authorized for workspace
        500: Server error
    """
    user_id = get_user_id(req)
    user_workspaces = get_workspace_ids(req)

    # Parse and validate request
    payload, error_response = parse_request(req, QueryRequestPayload)
    if error_response:
        return error_response

    workspace_id = payload.workspace_id

    # Authorization check (SOLID: Single Responsibility)
    if workspace_id and workspace_id not in user_workspaces:
        logger.warning(
            "Unauthorized workspace access",
            user_id=user_id,
            workspace_id=workspace_id,
        )
        raise AuthorizationException(
            message=ErrorMessages.UNAUTHORIZED_ACCESS
        )

    try:
        # Business logic delegated to service layer (SOLID: Dependency Inversion)
        kb_service = get_kb_service()
        result = await kb_service.query_knowledge_base(
            query=payload.query,
            workspace_id=workspace_id or user_workspaces[0],
            mode=payload.mode,
            only_context=payload.only_need_context,
        )

        # Return structured response
        return create_query_response(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            retrieved_chunks=result.get("retrieved_chunks", []),
            metadata={"mode": payload.mode},
            correlation_id=context.correlation_id,
        )

    except Exception as e:
        logger.error(
            "Query failed",
            error=e,
            user_id=user_id,
            workspace_id=workspace_id,
        )
        return create_error_response(
            message=ErrorMessages.QUERY_EXECUTION_ERROR,
            error_code="QUERY_FAILED",
            details={"error": str(e)},
            status_code=500,
            correlation_id=context.correlation_id,
        )
