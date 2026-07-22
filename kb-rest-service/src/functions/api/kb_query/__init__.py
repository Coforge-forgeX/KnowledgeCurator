"""KB query endpoint - TODO: Port from kb_adapter_tool.py"""
import json

import azure.functions as func

from shared.payloads import parse_request
from functions.api.kb_query.payloads import KBQueryRequest


async def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """
    Query knowledge base.

    POST /api/kb/query
    Body: {
        "workspace_id": 1,
        "query": "What is...",
        "kb_id": 1,
        "top_k": 5
    }
    """
    # Validate request payload
    payload, error = parse_request(req, KBQueryRequest)
    if error:
        return error

    # TODO: Implement query logic from kb_adapter_tool.py
    return func.HttpResponse(
        json.dumps({"error": "TODO: Port query_kb from kb_adapter_tool.py"}),
        status_code=501,
        mimetype="application/json",
    )
