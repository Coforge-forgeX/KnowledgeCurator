"""
Update Configuration Handler for V2 API.

POST /api/v2/config/update
Headers: Authorization: Bearer <token>
Body: { "workspace_id": int | str, "data": dict }
"""

from src.common import create_exception_response, create_success_response, parse_request
from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.auth import get_user_id, require_auth
from src.core.logging import get_logger
from src.functions.api.config.payloads import UpdateConfigRequest
from src.services.user_config_service import get_user_config_service

logger = get_logger(__name__)


@require_auth()
async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """
    Update user configuration for workspace.
    `user_id` is extracted strictly from JWT claims for security.
    """
    correlation_id = context.correlation_id
    user_id = get_user_id(req)

    try:
        payload, error = parse_request(req, UpdateConfigRequest)
        if error:
            return error

        service = get_user_config_service()
        result = await service.update_config(
            workspace_id=payload.workspace_id,
            user_id=user_id,
            data=payload.data,
        )

        return create_success_response(
            message="User configuration updated successfully",
            data=result,
            status_code=200,
            correlation_id=correlation_id,
        )
    except Exception as e:
        logger.error(
            "update_config handler failed",
            error=e,
            user_id=user_id,
            correlation_id=correlation_id,
        )
        return create_exception_response(
            e,
            fallback_message="Failed to update user configuration",
            fallback_error_code="UPDATE_CONFIG_FAILED",
            correlation_id=correlation_id,
        )