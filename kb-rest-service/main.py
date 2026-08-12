"""
KB REST Service - FastAPI Application

Multi-cloud deployment entry point for Azure Container Apps, AWS App Runner/Lambda, and GCP Cloud Run.

Key features:
- Lazy module imports (optimize cold starts)
- Abstract request/response handling (provider-agnostic)
- Unified logging with correlation IDs
- Comprehensive middleware (CORS, security headers, request validation, error handling)
- Health check endpoint
- Knowledge base operations (query, chat, indexing, document management)

Author: Architecture Team
Date: 2026-07-28
Version: 1.0.0
"""
import json
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, APIRouter, Request, Response, status, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware.base import BaseHTTPMiddleware

# Put the app root on the import path so `src.*` resolves when the process is
# started from another working directory. The `shared` package is resolved
# normally: installed (pip install -e ..) for local dev, vendored at the app
# root by the deploy pipeline in production.
main_dir = os.path.dirname(os.path.abspath(__file__))
if main_dir not in sys.path:
    sys.path.insert(0, main_dir)

# Configure Windows console for UTF-8 encoding (prevents Unicode crashes)
from shared.windows_encoding import configure_windows_console_encoding
configure_windows_console_encoding()

from src.core.config import settings
from src.core.exceptions import (
    APIException,
    AuthenticationException,
    AuthorizationException,
    ValidationException,
)
from src.core.logging import get_logger, setup_logging
from src.common.response_utils import build_error_body, build_success_body
from src.registry import get_handler
from src.functions.api.index_workspace_files.payloads import IndexWorkspaceFilesRequest
from src.functions.api.upload_and_index.payloads import UploadAndIndexRequest, UploadAndIndexResponse
from src.functions.api.kb_index.payloads import KBIndexRequest
from src.functions.api.workspace_documents_grouped.payloads import WorkspaceDocumentsGroupedRequest
from src.functions.api.delete_files_by_id.payloads import DeleteFilesByIdRequest
from src.functions.api.fetch_graph.payloads import FetchGraphRequest, FetchGraphResponse
from src.functions.api.mutate_knowledge_graph.payloads import MutateKnowledgeGraphRequest
from src.models.chat_models import (
    CancelChatRequest,
    CancelChatResponse,
    ChatRequest,
    ChatResponse,
    SessionDeleteRequest,
    SessionRenameRequest,
    StartConversationRequest,
)

# Setup logging BEFORE creating app (important for cold start tracking)
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup/shutdown tasks.

    Startup: Log app startup with environment info
    Shutdown: Cleanup resources (currently no resources to cleanup)

    Note: We don't initialize database/Redis here to keep cold start fast.
    They initialize lazily on first use.
    """
    # Startup
    logger.info(
        "KB REST Service starting",
        version="2.0.0",
        environment=settings.ENVIRONMENT,
        cloud_provider=settings.CLOUD_PROVIDER,
        storage_provider=settings.STORAGE_PROVIDER or settings.CLOUD_PROVIDER,
        queue_provider=settings.QUEUE_PROVIDER or settings.CLOUD_PROVIDER,
        port=settings.PORT,
    )

    yield

    # Shutdown
    logger.info("KB REST Service shutting down")


# Create FastAPI app with minimal initialization (fast cold start)
app = FastAPI(
    title="Knowledge Base REST Service",
    description="Multi-cloud Knowledge Base and RAG API with document indexing, querying, and chat",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    swagger_ui_parameters={"persistAuthorization": True},
)


# ============================================================================
# Middleware Configuration
# ============================================================================


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""

    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
    }

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        for header, value in self.SECURITY_HEADERS.items():
            response.headers[header] = value
        return response


class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """
    Inject correlation ID into request state and response headers.

    Picks up X-Correlation-ID from request headers or generates new UUID.
    Makes it available via request.state.correlation_id for logging.
    """

    async def dispatch(self, request: Request, call_next):
        # Get or generate correlation ID
        correlation_id = (
            request.headers.get("x-correlation-id")
            or request.headers.get("X-Correlation-ID")
            or request.headers.get("x-request-id")
            or request.headers.get("X-Request-ID")
            or str(uuid.uuid4())
        )

        # Store in request state for handlers to access
        request.state.correlation_id = correlation_id

        # Log request start
        logger.info(
            "Request started",
            correlation_id=correlation_id,
            method=request.method,
            url=str(request.url),
            client_host=request.client.host if request.client else None,
        )

        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate execution time
        execution_time_ms = (time.time() - start_time) * 1000

        # Add correlation ID to response headers
        response.headers["X-Correlation-ID"] = correlation_id

        # Log request completion
        logger.info(
            "Request completed",
            correlation_id=correlation_id,
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            execution_time_ms=execution_time_ms,
        )

        return response


class RequestSizeMiddleware(BaseHTTPMiddleware):
    """Validate request size doesn't exceed maximum"""

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > settings.MAX_REQUEST_SIZE:
                    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
                    logger.warning(
                        "Request too large",
                        correlation_id=correlation_id,
                        content_length=content_length,
                        max_size=settings.MAX_REQUEST_SIZE,
                    )
                    return JSONResponse(
                        content={
                            "success": False,
                            "error": "REQUEST_TOO_LARGE",
                            "message": f"Request payload exceeds maximum size of {settings.MAX_REQUEST_SIZE} bytes",
                            "correlation_id": correlation_id,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        headers={"X-Correlation-ID": correlation_id},
                    )
            except (ValueError, TypeError):
                pass

        return await call_next(request)


# Every handler below builds its body with `build_error_body` so the framework
# layer emits byte-for-byte the same error envelope as the handlers do, including
# the "hide 5xx internals" policy.
def _error_json_response(
    request: Request,
    message: str,
    error_code: str,
    status_code: int,
    details: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))
    return JSONResponse(
        content=build_error_body(
            message=message,
            error_code=error_code,
            details=details,
            status_code=status_code,
            correlation_id=correlation_id,
        ),
        status_code=status_code,
        headers={"X-Correlation-ID": correlation_id},
    )


# Global exception handler for APIException
@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    """Handle APIException globally (validation, auth, database errors)"""
    logger.warning(
        "API exception",
        correlation_id=getattr(request.state, "correlation_id", None),
        error_code=exc.error_code,
        error_message=exc.message,
        status_code=exc.status_code,
    )

    return _error_json_response(
        request,
        message=exc.message,
        error_code=exc.error_code,
        status_code=exc.status_code,
        details=exc.details,
    )


# Global exception handler for ValidationException
@app.exception_handler(ValidationException)
async def validation_exception_handler(request: Request, exc: ValidationException):
    """Handle ValidationException globally"""
    logger.warning(
        "Validation exception",
        correlation_id=getattr(request.state, "correlation_id", None),
        error_message=exc.message,
        details=exc.details,
    )

    return _error_json_response(
        request,
        message=exc.message,
        error_code="VALIDATION_ERROR",
        status_code=status.HTTP_400_BAD_REQUEST,
        details=exc.details,
    )


# Global exception handler for AuthorizationException
@app.exception_handler(AuthorizationException)
async def authorization_exception_handler(request: Request, exc: AuthorizationException):
    """Handle AuthorizationException globally"""
    logger.warning(
        "Authorization exception",
        correlation_id=getattr(request.state, "correlation_id", None),
        error=exc.message,
    )

    return _error_json_response(
        request,
        message=exc.message,
        error_code="AUTHORIZATION_ERROR",
        status_code=status.HTTP_403_FORBIDDEN,
    )


# FastAPI's own request validation (path/query/body params declared on the route)
@app.exception_handler(RequestValidationError)
async def request_validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Translate FastAPI's `{"detail": [...]}` 422 into the service envelope.

    Without this, a bad query param on a FastAPI-declared route answers in a
    shape no other endpoint uses — and with a 422 where `parse_request` returns
    400 for the very same mistake.
    """
    errors = [
        {
            "field": ".".join(str(p) for p in err.get("loc", ()) if p != "query"),
            "message": err.get("msg", ""),
            "type": err.get("type", ""),
        }
        for err in exc.errors()
    ]
    first = errors[0] if errors else {"field": "", "message": "Invalid request"}
    field = first["field"] or "payload"

    logger.warning(
        "Request validation error",
        correlation_id=getattr(request.state, "correlation_id", None),
        errors=errors,
    )

    return _error_json_response(
        request,
        message=f"Invalid request: {field} - {first['message']}",
        error_code="VALIDATION_ERROR",
        status_code=status.HTTP_400_BAD_REQUEST,
        details={"validation_errors": errors},
    )


# Starlette/FastAPI HTTPException (unknown route 404, 405, raised HTTPExceptions)
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Translate `{"detail": ...}` HTTP errors into the service envelope."""
    detail = exc.detail if isinstance(exc.detail, str) else "Request failed"
    error_codes = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED", 401: "AUTHENTICATION_ERROR"}

    return _error_json_response(
        request,
        message=detail,
        error_code=error_codes.get(exc.status_code, "HTTP_ERROR"),
        status_code=exc.status_code,
        details=None if isinstance(exc.detail, str) else {"detail": exc.detail},
    )


# Global exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions"""
    logger.error(
        "Unhandled exception",
        correlation_id=getattr(request.state, "correlation_id", None),
        error=str(exc),
        error_type=type(exc).__name__,
        exc_info=True,
    )

    return _error_json_response(
        request,
        message="An internal server error occurred. Please contact support with the correlation ID.",
        error_code="INTERNAL_SERVER_ERROR",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


# Add middleware (order matters! CORS first, then custom middleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.security.CORS_ORIGINS,
    allow_credentials=settings.security.CORS_ALLOW_CREDENTIALS,
    allow_methods=settings.security.CORS_ALLOW_METHODS,
    allow_headers=settings.security.CORS_ALLOW_HEADERS,
)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(CorrelationIDMiddleware)
app.add_middleware(RequestSizeMiddleware)


# ============================================================================
# Health Check Endpoint
# ============================================================================


@app.get("/health", tags=["System"])
async def health_check(response: Response) -> Dict[str, Any]:
    """
    Health check endpoint for load balancers and monitoring.

    Probes Postgres, Neo4j, Redis and MongoDB and reports overall status.

    Uses the same envelope as every other endpoint, with the report under `data`.
    An unhealthy service still answers in the success envelope and signals the
    problem through the 503 status and `data.status`, rather than switching to
    the error shape — one endpoint should not return two different body shapes
    for callers to branch on.
    """
    from src.core.health import run_health_checks

    checks, overall_status = await run_health_checks()
    response.status_code = 200 if overall_status == "healthy" else 503

    return build_success_body(
        message=overall_status,
        data={
            "status": overall_status,
            "service": "kb-rest-api",
            "version": "2.0.0",
            "cloud_provider": settings.CLOUD_PROVIDER,
            "storage_provider": settings.STORAGE_PROVIDER or settings.CLOUD_PROVIDER,
            "queue_provider": settings.QUEUE_PROVIDER or settings.CLOUD_PROVIDER,
            "checks": checks,
        },
    )


# ============================================================================
# Helper Function for Handler Invocation with Lazy Imports
# ============================================================================


async def invoke_handler(request: Request, handler_name: str):
    """
    Generic handler invoker using type-safe registry.

    This function:
    1. Gets handler module from registry (validated at startup)
    2. Wraps FastAPI request in AbstractRequest
    3. Creates AbstractContext (with correlation ID from middleware)
    4. Invokes handler with abstract interfaces
    5. Converts AbstractResponse to FastAPI JSONResponse

    Args:
        request: FastAPI Request object (contains correlation_id in state)
        handler_name: Handler name from registry (e.g., "upload_and_index", "kb_query")

    Returns:
        FastAPI JSONResponse
    """
    from src.adapters.fastapi_adapter import FastAPIContext, FastAPIRequest
    from src.core.abstractions import AbstractResponse

    try:
        # Get handler from type-safe registry (no string-based imports!)
        handler_module = get_handler(handler_name)
        handler_main = handler_module.main

        # Wrap request in abstract interfaces
        abstract_req = FastAPIRequest(request)
        abstract_ctx = FastAPIContext(request)

        # Call handler - returns AbstractResponse
        handler_response = await handler_main(abstract_req, abstract_ctx)

        # Convert AbstractResponse to FastAPI JSONResponse
        if isinstance(handler_response, AbstractResponse):
            # Parse body if it's a JSON string
            body = handler_response.body
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except (json.JSONDecodeError, ValueError):
                    # Keep as string if not valid JSON
                    pass

            # Add correlation ID to headers if not present
            headers = dict(handler_response.headers)
            if "X-Correlation-ID" not in headers:
                correlation_id = getattr(
                    request.state, "correlation_id", str(uuid.uuid4())
                )
                headers["X-Correlation-ID"] = correlation_id

            return JSONResponse(
                content=body,
                status_code=handler_response.status_code,
                headers=headers,
                media_type=handler_response.mimetype,
            )
        else:
            # Fallback for unexpected response type
            logger.warning(
                "Handler returned unexpected type",
                handler=handler_name,
                response_type=type(handler_response).__name__,
            )
            return JSONResponse(
                content={
                    "success": False,
                    "error": "Invalid response type from handler",
                },
                status_code=500,
            )

    except Exception:
        # Let global exception handler deal with it (will add correlation ID, etc.)
        raise


# ============================================================================
# Custom OpenAPI Schema with Bearer Token Authentication
# ============================================================================


def custom_openapi():
    """
    Custom OpenAPI schema with Bearer token authentication.
    Adds security scheme to all endpoints so Swagger shows "Authorize" button.
    """
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Add Bearer token security scheme
    if "components" not in openapi_schema:
        openapi_schema["components"] = {}
    if "securitySchemes" not in openapi_schema["components"]:
        openapi_schema["components"]["securitySchemes"] = {}

    openapi_schema["components"]["securitySchemes"]["HTTPBearer"] = {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "JWT",
        "description": "Enter your JWT token (without 'Bearer' prefix)",
    }

    # Apply security globally to all endpoints
    for path in openapi_schema["paths"].values():
        for operation in path.values():
            if isinstance(operation, dict) and "security" not in operation:
                operation["security"] = [{"HTTPBearer": []}]

    app.openapi_schema = openapi_schema
    return app.openapi_schema


# Set custom OpenAPI schema
app.openapi = custom_openapi


# ============================================================================
# API Endpoints - Pydantic Validation & Full Swagger Documentation
# ============================================================================

# Create API router with /api/v2 prefix
api_router = APIRouter(prefix="/api/v2", tags=["API v2"])


@api_router.post(
    "/query",
    tags=["Knowledge Base"],
    summary="Query RAG System (Optimized)",
    description="Optimized RAG query endpoint with caching, security, and multi-KB support"
)
async def query(request: Request):
    """
    Query the RAG system with advanced features.

    Enhancements over /kb/query:
    - **Automatic workspace resolution**: Domain and KB name fetched from DB
    - **Redis caching**: 60%+ hit ratio for faster responses
    - **Multi-KB support**: Query across multiple knowledge bases
    - **Enhanced security**: User-workspace membership validation
    - **Retrieved chunks**: Returns context chunks for evaluation

    Supports query modes:
    - **naive**: Simple keyword search
    - **local**: Local context-aware search
    - **global**: Global knowledge graph search
    - **hybrid**: Combines local and global (recommended)
    - **mix**: Mixed mode combining all strategies
    """
    return await invoke_handler(request, "query_rag")


@api_router.get(
    "/files/{file_id}/download",
    tags=["Knowledge Base"],
    summary="Generate File Download URL",
    description="Generate a signed file download URL with 5 minute TTL from opaque file_id"
)
async def query_source_download_url(file_id: str, request: Request):
    """Create short-lived signed URL for a source returned by query endpoint."""
    return await invoke_handler(request, "query_source_download_url")


@api_router.post(
    "/kb/index",
    tags=["Knowledge Base"],
    summary="Index Document",
    description="Add a document to the knowledge base for indexing"
)
async def kb_index(request: Request, payload: KBIndexRequest):
    """Index a document in the knowledge base"""
    request.state.parsed_payload = payload
    return await invoke_handler(request, "kb_index")


@api_router.post(
    "/documents/upload",
    tags=["Documents"],
    summary="Upload and Index Files",
    description="Upload multiple files and queue them for background indexing",
    response_model=UploadAndIndexResponse,
    responses={
        200: {
            "description": "Files uploaded and queued successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "message": "2 files uploaded and queued for indexing",
                        "workspace_id": 123,
                        "total_files": 2,
                        "tasks": [
                            {
                                "task_id": 1,
                                "file_name": "document.pdf",
                                "file_path": "workspace_123/document.pdf",
                                "status": "queued"
                            }
                        ],
                        "failed_files": []
                    }
                }
            }
        },
        401: {"description": "Missing or invalid authentication"},
        400: {"description": "Invalid request payload"}
    }
)
async def upload_and_index(request: Request, payload: UploadAndIndexRequest):
    """
    Upload files and queue for background indexing.

    **Request Body:**
    - `workspace_id`: Workspace ID (required, must be > 0)
    - `files`: List of files to upload (1-10 files)
      - `file_name`: Name of the file (with extension)
      - `file_content`: Base64 encoded file content

    **Example:**
    ```json
    {
        "workspace_id": 123,
        "files": [
            {
                "file_name": "document.pdf",
                "file_content": "JVBERi0xLjQKJeLjz9MKMSAwIG9iago8PC9UeXBlL0..."
            }
        ]
    }
    ```
    """
    # Store parsed payload in request state so handler can access it without re-parsing
    request.state.parsed_payload = payload
    return await invoke_handler(request, "upload_and_index")


@api_router.post(
    "/workspaces/index-files",
    tags=["Documents"],
    summary="Index All Existing Workspace Files",
    description="Queue all existing files from workspace blob path for indexing without uploading again"
)
async def index_workspace_files(request: Request, payload: IndexWorkspaceFilesRequest):
    """Queue existing workspace files for indexing."""
    request.state.parsed_payload = payload
    return await invoke_handler(request, "index_workspace_files")


@api_router.get(
    "/documents/status",
    tags=["Documents"],
    summary="Get Document Status",
    description="Get status by file task IDs or by workspace ID via query params"
)
async def indexing_status_get(request: Request):
    """Get file task indexing status with file_tasks_id-first behavior via GET query params.

    Not bound to a typed FastAPI payload: the handler accepts `task_ids`/`task_id`
    aliases and comma-separated or JSON-list values for `file_tasks_id`
    (see FileTasksStatusRequest normalization in the handler), which a strict
    FastAPI query-dependency binding would reject before the handler ever runs.
    """
    return await invoke_handler(request, "file_tasks_status")


@api_router.get(
    "/workspaces/documents",
    tags=["Documents"],
    summary="List Workspace Documents",
    description="Return workspace documents, including linked KB documents, grouped in response by workspace/KB"
)
async def workspace_documents(request: Request, payload: WorkspaceDocumentsGroupedRequest = Depends()):
    """Get workspace documents with grouping in response payload."""
    request.state.parsed_payload = payload
    return await invoke_handler(request, "workspace_documents")


@api_router.delete(
    "/files",
    tags=["Documents"],
    summary="Delete Files By File ID",
    description="Delete indexed files by file_id token(s) with workspace ownership and curate permission checks"
)
async def delete_files_by_id(request: Request, payload: DeleteFilesByIdRequest):
    """Delete indexed files by opaque file_id tokens."""
    request.state.parsed_payload = payload
    return await invoke_handler(request, "delete_files_by_id")


@api_router.post(
    "/kb/graph",
    tags=["Knowledge Base"],
    summary="Fetch Filtered Graph Data",
    description="Fetch graph data filtered by LLM to show only nodes relevant to the provided answer",
    response_model=FetchGraphResponse,
)
async def knowledge_graph(request: Request, payload: FetchGraphRequest):
    """Fetch filtered graph data for a workspace query and answer."""
    request.state.parsed_payload = payload
    return await invoke_handler(request, "fetch_graph")


@api_router.post(
    "/kb/graph/mutate",
    tags=["Knowledge Base"],
    summary="Mutate Knowledge Graph",
    description="Create, update, or delete graph nodes/relationships scoped to indexed workspace data"
)
async def mutate_knowledge_graph(request: Request, payload: MutateKnowledgeGraphRequest):
    """Mutate Neo4j graph and sync LightRAG VDB tables for a workspace-scoped file."""
    request.state.parsed_payload = payload
    return await invoke_handler(request, "mutate_knowledge_graph")


@api_router.post(
    "/kb/graph-data",
    tags=["Knowledge Base"],
    summary="Fetch Filtered Graph Data",
    description="Fetch graph data filtered by LLM to show only nodes relevant to the answer",
    response_model=FetchGraphResponse,
)
async def fetch_graph_data(request: Request, payload: FetchGraphRequest):
    """
    Fetch filtered graph data for a query and answer.

    This endpoint:
    1. Checks Redis cache for existing filtered graph
    2. If not cached, fetches context from LightRAG
    3. Uses LLM to filter only relevant nodes/relationships
    4. Caches the result for future requests
    5. Returns filtered graph data
    """
    request.state.parsed_payload = payload
    return await invoke_handler(request, "fetch_graph")


@api_router.post(
    "/chat/start",
    tags=["Chat"],
    summary="Start Conversation",
    description="Create a new conversation session in a workspace for the authenticated user",
    status_code=201,
)
async def chat_start_conversation(request: Request, payload: StartConversationRequest):
    """Start a conversation session. `user_id` is taken from the Bearer token."""
    request.state.parsed_payload = payload
    return await invoke_handler(request, "chat_start_conversation")


@api_router.get(
    "/chat/history",
    tags=["Chat"],
    summary="List Conversation Sessions",
    description=(
        "List the authenticated user's conversation sessions in a workspace. "
        "Paginated: returns page/page_size/total_count/total_pages/has_next/has_previous."
    ),
)
async def chat_get_conversation_history(
    request: Request,
    workspace_id: int,
    page: int = 1,
    page_size: int = 20,
    limit: Optional[int] = None,
):
    """List conversation sessions. Params are read from the query string by the handler."""
    return await invoke_handler(request, "chat_get_conversation_history")


@api_router.get(
    "/chat/load",
    tags=["Chat"],
    summary="Load Conversation",
    description=(
        "Load a single conversation session with one page of its messages. "
        "order=desc (default) puts the newest messages on page 1 and pages "
        "backwards in time; order=asc starts from the oldest. Messages within a "
        "page are always oldest-first."
    ),
)
async def chat_load_conversation(
    request: Request,
    session_id: str,
    workspace_id: int,
    page: int = 1,
    page_size: int = 50,
    order: str = "desc",
):
    """Load one conversation and a page of its messages."""
    return await invoke_handler(request, "chat_load_conversation")


@api_router.post(
    "/chat/session/rename",
    tags=["Chat"],
    summary="Rename Conversation",
    description="Rename a conversation session",
)
async def chat_rename_conversation(request: Request, payload: SessionRenameRequest):
    """Rename a conversation session."""
    request.state.parsed_payload = payload
    return await invoke_handler(request, "chat_rename_conversation")


@api_router.delete(
    "/chat/session/delete",
    tags=["Chat"],
    summary="Delete Conversation",
    description=(
        "Delete a conversation session and its messages. Returns 404 when the "
        "conversation does not exist for the authenticated user."
    ),
)
async def chat_delete_conversation(request: Request, payload: SessionDeleteRequest):
    """Delete a conversation session."""
    request.state.parsed_payload = payload
    return await invoke_handler(request, "chat_delete_conversation")


@api_router.post(
    "/chat/message",
    tags=["Chat"],
    summary="Send Chat Message",
    description="Process a chatbot message (SEARCH or UPDATE mode) against the workspace knowledge base",
    response_model=ChatResponse,
)
async def message_gpt(request: Request, payload: ChatRequest):
    """
    Send a message to the chatbot.

    `user_id`, workspace membership, curate permission and the workspace's
    domain/kb_name are resolved server-side from the Bearer token and DB —
    they are never taken from the body.

    Modes:
    - **SEARCH**: answer from the knowledge base
    - **UPDATE**: ingest the attached files into the knowledge base
    """
    request.state.parsed_payload = payload
    return await invoke_handler(request, "message_gpt")


@api_router.post(
    "/chat/message/cancel",
    tags=["Chat"],
    summary="Cancel Chat Message",
    description="Cancel an in-flight chat message for a session (stop button)",
    response_model=CancelChatResponse,
)
async def cancel_chat_message(request: Request, payload: CancelChatRequest):
    """Cancel the running message_gpt task for a session."""
    request.state.parsed_payload = payload
    return await invoke_handler(request, "cancel_chat_message")


# ============================================================================
# Register API Router
# ============================================================================

# Include the API router with all v2 endpoints
app.include_router(api_router)


# ============================================================================
# Application Entry Point (for local development)
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    import logging

    # Suppress watchfiles change detection logs
    logging.getLogger("watchfiles").setLevel(logging.WARNING)

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=False,
        reload_excludes=["**/__pycache__/**", "**/*.pyc", "**/*.pyo", "**/.git/**", "**/.venv/**"],
        log_level="info",
        access_log=True,
    )
