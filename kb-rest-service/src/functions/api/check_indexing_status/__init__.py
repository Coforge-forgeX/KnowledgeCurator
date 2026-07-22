"""Check Indexing Status API - Clean & Optimized"""
import azure.functions as func

from core import (
    get_logger,
    get_user_id,
    require_auth,
    azure_http_decorator,
)
from services.kb_service import get_kb_service
from shared import (
    ErrorMessages,
    create_success_response,
    create_error_response,
    parse_request,
)
from .payloads import CheckIndexingStatusRequest

logger = get_logger(__name__)


@azure_http_decorator()
@require_auth()
async def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """Check indexing status for specific task IDs."""
    user_id = get_user_id(req)

    payload, error_response = parse_request(req, CheckIndexingStatusRequest)
    if error_response:
        return error_response

    try:
        kb_service = get_kb_service()
        statuses = await kb_service.check_indexing_status(
            task_ids=payload.task_ids,
        )

        return create_success_response(
            message=f"Status retrieved for {len(statuses)} tasks",
            data={
                "statuses": statuses,
                "requested_count": len(payload.task_ids),
                "found_count": len(statuses),
            },
            correlation_id=context.correlation_id,
        )

    except Exception as e:
        logger.error("Check status failed", error=e, user_id=user_id)
        return create_error_response(
            message=ErrorMessages.QUERY_EXECUTION_ERROR,
            error_code="CHECK_STATUS_FAILED",
            details={"error": str(e)},
            status_code=500,
            correlation_id=context.correlation_id,
        )
