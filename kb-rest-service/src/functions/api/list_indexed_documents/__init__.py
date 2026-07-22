"""List Indexed Documents API - Clean & Optimized"""
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
from .payloads import ListIndexedDocumentsRequest

logger = get_logger(__name__)


@azure_http_decorator()
@require_auth()
async def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """List indexed documents for a workspace."""
    user_id = get_user_id(req)
    user_workspaces = get_workspace_ids(req)

    payload, error_response = parse_request(req, ListIndexedDocumentsRequest)
    if error_response:
        return error_response

    workspace_id = payload.workspace_id

    # Authorization check
    if workspace_id not in user_workspaces:
        raise AuthorizationException(message=ErrorMessages.UNAUTHORIZED_ACCESS)

    try:
        kb_service = get_kb_service()
        documents = await kb_service.get_indexed_documents(
            workspace_id=workspace_id,
            limit=payload.limit,
        )

        return create_success_response(
            message="Documents retrieved successfully",
            data={
                "documents": documents,
                "count": len(documents),
                "workspace_id": workspace_id,
            },
            correlation_id=context.correlation_id,
        )

    except Exception as e:
        logger.error("List documents failed", error=e, user_id=user_id)
        return create_error_response(
            message=ErrorMessages.QUERY_EXECUTION_ERROR,
            error_code="LIST_DOCS_FAILED",
            details={"error": str(e)},
            status_code=500,
            correlation_id=context.correlation_id,
        )
