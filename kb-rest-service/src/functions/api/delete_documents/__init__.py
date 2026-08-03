"""Delete Documents API - Clean & Optimized"""
from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, get_workspace_ids, require_auth
from src.core.exceptions import AuthorizationException
from src.core.logging import get_logger
from src.services.kb_service import get_kb_service
from src.shared import (
    DocumentDeletePayload,
    ErrorMessages,
    SuccessMessages,
    create_batch_response,
    create_error_response,
    parse_request,
)

logger = get_logger(__name__)


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
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
