"""
KB REST Azure Function App

Handles knowledge base operations, queries, chatbot, LLM routing.
Enqueues indexing jobs to Azure Storage Queue for indexer service.
"""
import logging
import os
import sys

import azure.functions as func

# Add src and shared folders to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "shared"))

from functions.api.kb_query import main as kb_query_main
from functions.api.kb_chat import main as kb_chat_main
from functions.api.kb_index import main as kb_index_main
from functions.api.llm_route import main as llm_route_main
from functions.api.sharepoint_list import main as sharepoint_list_main
from functions.api.query_kb import main as query_kb_main
from functions.api.upload_document import main as upload_document_main
from functions.api.delete_documents import main as delete_documents_main
from functions.api.list_indexed_documents import main as list_indexed_documents_main
from functions.api.check_indexing_status import main as check_indexing_status_main
from functions.api.get_knowledge_graph import main as get_knowledge_graph_main

logging.basicConfig(level=logging.INFO)

app = func.FunctionApp()

INDEXING_QUEUE_NAME = "kb-indexing-jobs"


@app.route(route="health", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET"])
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint"""
    import json

    response = {
        "status": "healthy",
        "service": "kb-rest-api",
    }

    return func.HttpResponse(
        json.dumps(response), status_code=200, mimetype="application/json"
    )


@app.route(route="kb/query", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
async def kb_query(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """Query knowledge base"""
    return await kb_query_main(req, context)


@app.route(route="kb/chat", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
async def kb_chat(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """KB chatbot endpoint"""
    return await kb_chat_main(req, context)


@app.route(route="kb/index", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
@app.queue_output(
    arg_name="msg",
    queue_name=INDEXING_QUEUE_NAME,
    connection="AzureWebJobsStorage",
)
async def kb_index(
    req: func.HttpRequest, context: func.Context, msg: func.Out[str]
) -> func.HttpResponse:
    """Enqueue indexing job to background worker"""
    return await kb_index_main(req, context, msg)


@app.route(route="llm/route", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST"])
async def llm_route(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """LLM router endpoint"""
    return await llm_route_main(req, context)


@app.route(
    route="sharepoint/list", auth_level=func.AuthLevel.ANONYMOUS, methods=["GET"]
)
async def sharepoint_list(
    req: func.HttpRequest, context: func.Context
) -> func.HttpResponse:
    """List SharePoint sites"""
    return await sharepoint_list_main(req, context)


# ============================================================================
# New Optimized KB REST APIs - Following SOLID & Best Practices
# ============================================================================

@app.route(route="query-kb", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST", "OPTIONS"])
async def query_kb(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """Query knowledge base with LightRAG"""
    return await query_kb_main(req, context)


@app.route(route="upload-document", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST", "OPTIONS"])
async def upload_document(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """Upload document and queue for indexing"""
    return await upload_document_main(req, context)


@app.route(route="delete-documents", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST", "OPTIONS"])
async def delete_documents(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """Delete documents from knowledge base"""
    return await delete_documents_main(req, context)


@app.route(route="list-indexed-documents", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST", "OPTIONS"])
async def list_indexed_documents(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """List indexed documents for a workspace"""
    return await list_indexed_documents_main(req, context)


@app.route(route="check-indexing-status", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST", "OPTIONS"])
async def check_indexing_status(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """Check indexing status for specific task IDs"""
    return await check_indexing_status_main(req, context)


@app.route(route="get-knowledge-graph", auth_level=func.AuthLevel.ANONYMOUS, methods=["POST", "OPTIONS"])
async def get_knowledge_graph(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """Get knowledge graph for a workspace"""
    return await get_knowledge_graph_main(req, context)
