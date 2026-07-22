"""KB chatbot endpoint - TODO: Port from kb_curator_chatbot.py"""
import json

import azure.functions as func

from shared.payloads import parse_request
from functions.api.kb_chat.payloads import KBChatRequest


async def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """
    KB chatbot endpoint.

    POST /api/kb/chat
    Body: {
        "workspace_id": 1,
        "messages": [{"role": "user", "content": "Hello"}],
        "kb_id": 1
    }
    """
    # Validate request payload
    payload, error = parse_request(req, KBChatRequest)
    if error:
        return error

    # TODO: Implement chatbot logic from kb_curator_chatbot.py
    return func.HttpResponse(
        json.dumps({"error": "TODO: Port chatbot from kb_curator_chatbot.py"}),
        status_code=501,
        mimetype="application/json",
    )
