"""List Indexed Documents API - Clean & Optimized"""
from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, get_workspace_ids, require_auth
from src.core.exceptions import AuthorizationException
from src.core.logging import get_logger
from src.services.kb_service import get_kb_service
from src.shared import ErrorMessages, create_error_response, create_success_response, parse_request

from .payloads import ListIndexedDocumentsRequest

logger = get_logger(__name__)


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
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
        result = await kb_service.get_indexed_documents(
            workspace_id=workspace_id,
            limit=payload.limit,
            offset=payload.offset,
        )

        return create_success_response(
            message="Documents retrieved successfully",
            data={
                **result,  # includes: documents, total, limit, offset, has_more, page, total_pages
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
