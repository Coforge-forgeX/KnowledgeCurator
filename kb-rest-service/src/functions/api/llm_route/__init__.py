"""LLM router endpoint - TODO: Port from llm_router_tool.py"""
import json

import azure.functions as func

from shared.payloads import parse_request
from functions.api.llm_route.payloads import LLMRouteRequest


async def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """
    LLM router endpoint.

    POST /api/llm/route
    Body: {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "gpt-4",
        "temperature": 0.7
    }
    """
    # Validate request payload
    payload, error = parse_request(req, LLMRouteRequest)
    if error:
        return error

    # TODO: Implement LLM routing logic from llm_router_tool.py
    return func.HttpResponse(
        json.dumps({"error": "TODO: Port from llm_router_tool.py"}),
        status_code=501,
        mimetype="application/json",
    )
