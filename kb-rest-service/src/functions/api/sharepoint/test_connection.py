"""Test SharePoint Connection Handler for V2 API."""

from src.common import create_error_response, create_success_response, parse_request
from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import require_auth
from src.core.logging import get_logger
from src.services.sharepoint_service import get_sharepoint_service

from .payloads import TestSharePointConnectionRequest, TestSharePointConnectionResponse

logger = get_logger(__name__)


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """
    Test SharePoint connection credentials.

    POST /api/v2/sharepoint/test-connection
    """
    payload, error_response = parse_request(req, TestSharePointConnectionRequest)
    if error_response:
        return error_response

    try:
        service = get_sharepoint_service()
        res_dict = await service.test_connection(
            workspace_id=payload.workspace_id,
            user_id=payload.user_id,
            data=payload.data.model_dump(),
        )

        response_data = TestSharePointConnectionResponse(
            status=res_dict.get("status", "error"),
            message=res_dict.get("message", ""),
            exists=res_dict.get("exists"),
        )

        if res_dict.get("status") == "success":
            return create_success_response(
                message=response_data.message,
                data=response_data.model_dump(),
                status_code=200,
                correlation_id=context.correlation_id,
            )
        else:
            return create_error_response(
                message=response_data.message or "SharePoint connection test failed",
                error_code="SHAREPOINT_CONNECTION_FAILED",
                status_code=400,
                correlation_id=context.correlation_id,
            )

    except Exception as e:
        logger.error(f"Error testing SharePoint connection: {e}", exc_info=True)
        return create_error_response(
            message=f"Failed to test SharePoint connection: {str(e)}",
            error_code="SHAREPOINT_TEST_ERROR",
            status_code=500,
            correlation_id=context.correlation_id,
        )
