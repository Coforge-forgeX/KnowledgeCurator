"""SharePoint list endpoint - TODO: Port from sharepoint_agent.py"""
from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.shared.payloads import parse_request
from src.shared.response_utils import create_error_response

from .payloads import SharePointListRequest


async def main(req: AbstractRequest, context: AbstractContext) -> AbstractResponse:
    """
    List SharePoint sites.

    GET /api/sharepoint/list
    Query params: ?workspace_id=1&site_url=https://...
    """
    # Validate request payload
    payload, error = parse_request(req, SharePointListRequest)
    if error:
        return error

    # TODO: Implement SharePoint listing logic from sharepoint_agent.py
    return create_error_response(
        message="TODO: Port from sharepoint_agent.py",
        error_code="NOT_IMPLEMENTED",
        status_code=501,
        correlation_id=context.correlation_id,
    )
