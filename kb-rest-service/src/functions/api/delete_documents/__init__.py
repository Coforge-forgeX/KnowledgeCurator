"""Delete Documents API - Clean & Optimized"""
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
    SuccessMessages,
    create_batch_response,
    create_error_response,
    parse_request,
    DocumentDeletePayload,
)

logger = get_logger(__name__)


@azure_http_decorator()
@require_auth()
async def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """Delete documents from knowledge base."""
    user_id = get_user_id(req)
    user_workspaces = get_workspace_ids(req)

    payload, error_response = parse_request(req, DocumentDeletePayload)
    if error_response:
        return error_response

    workspace_id = payload.workspace_id

    # Authorization check
    if workspace_id not in user_workspaces:
        raise AuthorizationException(message=ErrorMessages.UNAUTHORIZED_ACCESS)

    try:
        kb_service = get_kb_service()
        result = await kb_service.delete_documents(
            doc_ids=payload.doc_ids,
            workspace_id=workspace_id,
        )

        return create_batch_response(
            message=SuccessMessages.DOCUMENTS_DELETED.format(
                count=result["successful"]
            ),
            successful=result["successful"],
            failed=result["failed"],
            total=result["total"],
            details=result.get("errors"),
            correlation_id=context.correlation_id,
        )

    except Exception as e:
        logger.error("Delete failed", error=e, user_id=user_id)
        return create_error_response(
            message=ErrorMessages.DOCUMENT_DELETION_ERROR,
            error_code="DELETE_FAILED",
            details={"error": str(e)},
            status_code=500,
            correlation_id=context.correlation_id,
        )
