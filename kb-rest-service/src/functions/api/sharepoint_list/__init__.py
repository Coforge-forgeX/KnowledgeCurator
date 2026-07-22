"""SharePoint list endpoint - TODO: Port from sharepoint_agent.py"""
import json

import azure.functions as func

from shared.payloads import parse_request
from functions.api.sharepoint_list.payloads import SharePointListRequest


async def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
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
    return func.HttpResponse(
        json.dumps({"error": "TODO: Port from sharepoint_agent.py"}),
        status_code=501,
        mimetype="application/json",
    )
