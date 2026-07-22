"""Get Knowledge Graph API - Clean & Optimized"""
import azure.functions as func

from core import (
    get_logger,
    get_user_id,
    get_workspace_ids,
    require_auth,
    azure_http_decorator,
    AuthorizationException,
)
from services.kb_service import get_kb_service
from shared import (
    ErrorMessages,
    create_success_response,
    create_error_response,
    parse_request,
)
from .payloads import GetKnowledgeGraphRequest

logger = get_logger(__name__)


@azure_http_decorator()
@require_auth()
async def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """Get knowledge graph for a workspace."""
    user_id = get_user_id(req)
    user_workspaces = get_workspace_ids(req)

    payload, error_response = parse_request(req, GetKnowledgeGraphRequest)
    if error_response:
        return error_response

    workspace_id = payload.workspace_id

    # Authorization check
    if workspace_id not in user_workspaces:
        raise AuthorizationException(message=ErrorMessages.UNAUTHORIZED_ACCESS)

    try:
        kb_service = get_kb_service()
        kg = await kb_service.get_knowledge_graph(workspace_id=workspace_id)

        return create_success_response(
            message="Knowledge graph retrieved successfully",
            data=kg,
            correlation_id=context.correlation_id,
        )

    except Exception as e:
        logger.error("Get KG failed", error=e, user_id=user_id)
        return create_error_response(
            message="Failed to retrieve knowledge graph",
            error_code="GET_KG_FAILED",
            details={"error": str(e)},
            status_code=500,
            correlation_id=context.correlation_id,
        )
