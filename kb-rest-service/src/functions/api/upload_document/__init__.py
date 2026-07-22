"""Upload & Index Document API - Optimized with Best Practices"""
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
    create_success_response,
    create_error_response,
    parse_request,
)
from .payloads import UploadDocumentRequest

logger = get_logger(__name__)


@azure_http_decorator()
@require_auth()
async def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """
    Upload document and queue for indexing.

    POST /api/upload-document
    Headers: Authorization: Bearer <token>
    Body: {
        "workspace_id": 1,
        "document_text": "...",
        "file_name": "document.pdf",
        "metadata": {"source": "upload"}
    }

    Returns:
        200: Document queued successfully
        400: Validation error
        403: Not authorized
        500: Server error
    """
    user_id = get_user_id(req)
    user_workspaces = get_workspace_ids(req)

    # Parse request
    payload, error_response = parse_request(req, UploadDocumentRequest)
    if error_response:
        return error_response

    workspace_id = payload.workspace_id

    # Authorization (DRY: Reusable pattern)
    if workspace_id not in user_workspaces:
        logger.warning(
            "Unauthorized upload attempt",
            user_id=user_id,
            workspace_id=workspace_id,
        )
        raise AuthorizationException(message=ErrorMessages.UNAUTHORIZED_ACCESS)

    try:
        # Delegate to service layer (SOLID)
        kb_service = get_kb_service()
        message_id = await kb_service.queue_document_for_indexing(
            document_text=payload.document_text,
            workspace_id=workspace_id,
            file_name=payload.file_name,
            metadata=payload.metadata,
        )

        return create_success_response(
            message=SuccessMessages.INDEXING_QUEUED,
            data={
                "message_id": message_id,
                "file_name": payload.file_name,
                "workspace_id": workspace_id,
            },
            status_code=202,  # Accepted
            correlation_id=context.correlation_id,
        )

    except Exception as e:
        logger.error(
            "Upload failed",
            error=e,
            user_id=user_id,
            workspace_id=workspace_id,
        )
        return create_error_response(
            message=ErrorMessages.DOCUMENT_INDEXING_ERROR,
            error_code="UPLOAD_FAILED",
            details={"error": str(e)},
            status_code=500,
            correlation_id=context.correlation_id,
        )
