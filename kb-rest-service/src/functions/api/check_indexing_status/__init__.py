"""Check Indexing Status API - Clean & Optimized"""
from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, require_auth
from src.core.logging import get_logger
from src.services.kb_service import get_kb_service
from src.shared import ErrorMessages, create_error_response, create_success_response, parse_request

from .payloads import CheckIndexingStatusRequest

logger = get_logger(__name__)


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
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
