"""Azure Functions HTTP entrypoint using explicit per-route decorators.

This follows the Python v2 programming model so `func start` lists each route,
similar to user-mgmnt-service.
"""

import os
import sys

# Match main.py: app root on the path so `src.*` resolves.
# MUST be done before any local imports.
# WARNING: Do not update this path
# Use pip install -e .. [To include the shared package in the current repo]
main_dir = os.path.dirname(os.path.abspath(__file__))
if main_dir not in sys.path:
    sys.path.insert(0, main_dir)

import azure.functions as func

# Configure Windows console for UTF-8 encoding (prevents Unicode crashes in the
# Functions worker, whose stdout/stderr wrappers default to cp1252).
from shared.windows_encoding import configure_windows_console_encoding
configure_windows_console_encoding()

from src.adapters.cloud_function_adapter import (
    abstract_response_to_http_tuple,
    dispatch_request,
    from_azure_request,
)

app = func.FunctionApp()


async def _handle(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    abstract_req, abstract_ctx = from_azure_request(req)
    # Keep function name for better diagnostics in handler context.
    abstract_ctx._function_name = getattr(context, "function_name", "http_trigger")
    abstract_resp = await dispatch_request(abstract_req, abstract_ctx)
    body, status_code, headers, mimetype = abstract_response_to_http_tuple(abstract_resp)
    return func.HttpResponse(body=body, status_code=status_code, headers=headers, mimetype=mimetype)


@app.route(route="health", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET"])
async def health(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)


@app.route(route="api/v2/kb/index", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
async def kb_index(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)


@app.route(route="api/v2/documents/upload", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
async def upload_and_index(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)


@app.route(route="api/v2/workspaces/index-files", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
async def index_workspace_files(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)


@app.route(route="api/v2/documents/status", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET"])
async def documents_status(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)


@app.route(route="api/v2/workspaces/documents", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET"])
async def workspace_documents(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)


@app.route(route="api/v2/files", auth_level=func.AuthLevel.ANONYMOUS, methods=["DELETE"])
async def delete_files_by_id(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)


@app.route(route="api/v2/workspaces/documents/all", auth_level=func.AuthLevel.ANONYMOUS, methods=["DELETE"])
async def delete_all_indexed_documents(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)


@app.route(route="api/v2/kb/graph", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
async def get_knowledge_graph(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)


@app.route(route="api/v2/kb/graph/mutate", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
async def mutate_knowledge_graph(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)


@app.route(route="api/v2/query-kb", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
async def query_kb_v2(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)


@app.route(route="api/v2/files/{file_id}/download", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET"])
async def query_sources_download_url(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)


@app.route(route="api/v2/chat/start", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET"])
async def chat_start_conversation(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)


@app.route(route="api/v2/chat/history", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET"])
async def chat_get_conversation_history(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)


@app.route(route="api/v2/chat/load", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET"])
async def chat_load_conversation(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)


@app.route(route="api/v2/chat/session/rename", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
async def chat_rename_conversation(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)


@app.route(route="api/v2/chat/session/delete", auth_level=func.AuthLevel.ANONYMOUS, methods=["DELETE"])
async def chat_delete_conversation(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)


@app.route(route="api/v2/chat/message", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
async def message_gpt(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)


@app.route(route="api/v2/chat/message/cancel", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
async def cancel_chat_message(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)

@app.route(route="api/v2/sharepoint/test-connection", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
async def sharepoint_test_connection(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)


@app.route(route="api/v2/sharepoint/toggle-connection", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
async def sharepoint_toggle_connection(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)


@app.route(route="api/v2/sharepoint/extract-data", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
async def sharepoint_extract_data(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)


@app.route(route="api/v2/workspaces/download-zip", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
async def workspace_download_zip(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)

@app.route(route="api/v2/config/get", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST", "GET"])
async def get_config(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)


@app.route(route="api/v2/config/update", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
async def update_config(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)

@app.route(route="api/v2/kb/extract-keywords", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
async def extract_keywords(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    return await _handle(req, context)