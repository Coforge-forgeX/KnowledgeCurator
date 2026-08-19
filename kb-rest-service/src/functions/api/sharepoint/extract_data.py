"""Extract SharePoint Data Handler for V2 API."""

from src.common import create_error_response, create_success_response, parse_request
from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import require_auth
from src.core.logging import get_logger
from src.services.sharepoint_service import get_sharepoint_service

from .payloads import ExtractSharePointDataRequest, ExtractSharePointDataResponse

logger = get_logger(__name__)


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """
    Extract text and metadata from SharePoint documents.

    POST /api/v2/sharepoint/extract-data
    """
    payload, error_response = parse_request(req, ExtractSharePointDataRequest)
    if error_response:
        return error_response

    try:
        service = get_sharepoint_service()
        creds_dict = payload.credentials.model_dump() if payload.credentials else None

        res_dict = await service.extract_data(
            workspace_id=payload.workspace_id,
            user_id=payload.user_id,
            conversation_id=payload.conversation_id,
            folder_path=payload.folder_path,
            file_types=payload.file_types,
            name_contains=payload.name_contains,
            min_size=payload.min_size,
            max_size=payload.max_size,
            created_after=payload.created_after,
            created_before=payload.created_before,
            modified_after=payload.modified_after,
            modified_before=payload.modified_before,
            tags=payload.tags,
            credentials=creds_dict,
        )

        response_data = ExtractSharePointDataResponse(
            success=res_dict.get("success", False),
            documents=res_dict.get("documents", []),
            count=res_dict.get("count", 0),
            error=res_dict.get("error"),
        )

        if response_data.success:
            return create_success_response(
                message=f"Successfully extracted {response_data.count} document(s) from SharePoint",
                data=response_data.model_dump(),
                status_code=200,
                correlation_id=context.correlation_id,
            )
        else:
            return create_error_response(
                message=response_data.error or "Failed to extract SharePoint data",
                error_code="SHAREPOINT_EXTRACTION_FAILED",
                status_code=400,
                correlation_id=context.correlation_id,
            )

    except Exception as e:
        logger.error(f"Error extracting SharePoint data: {e}", exc_info=True)
        return create_error_response(
            message=f"Failed to extract SharePoint data: {str(e)}",
            error_code="SHAREPOINT_EXTRACT_ERROR",
            status_code=500,
            correlation_id=context.correlation_id,
        )
