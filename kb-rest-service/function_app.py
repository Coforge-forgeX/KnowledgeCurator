"""Azure Functions HTTP entrypoint using explicit per-route decorators.

This follows the Python v2 programming model so `func start` lists each route,
similar to user-mgmnt-service.
"""

import os
import sys

# Match main.py import precedence so shared.adapters resolves correctly.
# MUST be done before any local imports that depend on shared.
main_dir = os.path.dirname(os.path.abspath(__file__))
services_path = os.path.dirname(main_dir)  # .../Kb/services
src_path = os.path.join(main_dir, "src")
sys.path = [p for p in sys.path if p not in {services_path, src_path}]
sys.path.insert(0, src_path)
sys.path.insert(0, services_path)

import azure.functions as func

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
