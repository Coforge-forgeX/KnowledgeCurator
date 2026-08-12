"""Check Indexing Status API - Clean & Optimized"""
import json

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, require_auth
from src.core.exceptions import ValidationException
from src.core.logging import get_logger
from src.services.kb_service import get_kb_service
from src.common import ErrorMessages, create_error_response, create_internal_error_response, create_success_response, parse_request

from .payloads import CheckIndexingStatusRequest

logger = get_logger(__name__)


def _parse_task_ids_from_get(req: AbstractRequest) -> list[str]:
    """Parse task IDs from GET query params.

    Supports:
    - task_ids=1,2,3
    - task_ids=["1","2"]
    - task_id=1 (single fallback)
    """
    raw_task_ids = req.get_query_param("task_ids")
    if not raw_task_ids:
        raw_task_ids = req.get_query_param("task_id")

    if not raw_task_ids:
        raise ValidationException(message="task_ids query parameter is required")

    value = str(raw_task_ids).strip()
    if not value:
        raise ValidationException(message="task_ids query parameter cannot be empty")

    if value.startswith("["):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                task_ids = [str(item).strip() for item in parsed if str(item).strip()]
            else:
                task_ids = []
        except json.JSONDecodeError:
            task_ids = []
    else:
        task_ids = [chunk.strip() for chunk in value.split(",") if chunk.strip()]

    if not task_ids:
        raise ValidationException(message="task_ids query parameter is invalid")

    return task_ids


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """Check indexing status for specific task IDs."""
    user_id = get_user_id(req)

    try:
        if req.method.upper() == "GET":
            task_ids = _parse_task_ids_from_get(req)
            payload = CheckIndexingStatusRequest.model_validate({"task_ids": task_ids})
        else:
            payload, error_response = parse_request(req, CheckIndexingStatusRequest)
            if error_response:
                return error_response

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
        return create_internal_error_response(
            message=ErrorMessages.QUERY_EXECUTION_ERROR,
            error=e,
            error_code="CHECK_STATUS_FAILED",
            correlation_id=context.correlation_id,
        )
