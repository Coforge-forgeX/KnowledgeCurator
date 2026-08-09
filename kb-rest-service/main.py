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
from typing import Any, Dict

from fastapi import FastAPI, APIRouter, Request, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

# Add services/ and src/ to import path with deterministic precedence.
# Keep services/ before src/ so `shared.*` resolves to services/shared, not src/shared.
main_dir = os.path.dirname(os.path.abspath(__file__))
services_path = os.path.dirname(main_dir)  # Parent of kb-rest-service
src_path = os.path.join(main_dir, "src")

# Remove any existing entries first so we can reinsert in the correct order.
sys.path = [p for p in sys.path if p not in {services_path, src_path}]

sys.path.insert(0, src_path)
sys.path.insert(0, services_path)
print(f"[STARTUP] Added to sys.path (priority): {services_path}, {src_path}")

print(f"[STARTUP] sys.path[0:3] = {sys.path[0:3]}")

# Verify shared.adapters is importable
try:
    import shared.adapters
    print("[STARTUP] ✓ shared.adapters is importable")
except ImportError as e:
    print(f"[STARTUP] ✗ Failed to import shared.adapters: {e}")

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
from src.registry import get_handler
from src.models.api_models import (
    KBQueryRequest,
    KBQueryResponse,
    KBChatRequest,
    KBChatResponse,
    KBIndexRequest,
    KBIndexResponse,
    UploadAndIndexRequest,
    UploadAndIndexResponse,
    IndexingStatusRequest,
    IndexingStatusResponse,
    ListDocumentsRequest,
    ListDocumentsResponse,
    DeleteDocumentsRequest,
    DeleteDocumentsResponse,
    KnowledgeGraphRequest,
    KnowledgeGraphResponse,
    LLMRouteRequest,
    LLMRouteResponse,
    StatusResponse,
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


# Global exception handler for APIException
@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    """Handle APIException globally (validation, auth, database errors)"""
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))

    logger.warning(
        "API exception",
        correlation_id=correlation_id,
        error_code=exc.error_code,
        error_message=exc.message,
        status_code=exc.status_code,
    )

    return JSONResponse(
        content={
            "success": False,
            "error": exc.error_code,
            "message": exc.message,
            "details": exc.details,
            "correlation_id": correlation_id,
            "timestamp": datetime.utcnow().isoformat(),
        },
        status_code=exc.status_code,
        headers={"X-Correlation-ID": correlation_id},
    )


# Global exception handler for ValidationException
@app.exception_handler(ValidationException)
async def validation_exception_handler(request: Request, exc: ValidationException):
    """Handle ValidationException globally"""
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))

    logger.warning(
        "Validation exception",
        correlation_id=correlation_id,
        error_message=exc.message,
        details=exc.details,
    )

    return JSONResponse(
        content={
            "success": False,
            "error": "VALIDATION_ERROR",
            "message": exc.message,
            "details": exc.details,
            "correlation_id": correlation_id,
            "timestamp": datetime.utcnow().isoformat(),
        },
        status_code=status.HTTP_400_BAD_REQUEST,
        headers={"X-Correlation-ID": correlation_id},
    )


# Global exception handler for AuthorizationException
@app.exception_handler(AuthorizationException)
async def authorization_exception_handler(request: Request, exc: AuthorizationException):
    """Handle AuthorizationException globally"""
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))

    logger.warning(
        "Authorization exception",
        correlation_id=correlation_id,
        error=exc.message,
    )

    return JSONResponse(
        content={
            "success": False,
            "error": "AUTHORIZATION_ERROR",
            "message": exc.message,
            "correlation_id": correlation_id,
            "timestamp": datetime.utcnow().isoformat(),
        },
        status_code=status.HTTP_403_FORBIDDEN,
        headers={"X-Correlation-ID": correlation_id},
    )


# Global exception handler for unhandled exceptions
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions"""
    correlation_id = getattr(request.state, "correlation_id", str(uuid.uuid4()))

    logger.error(
        "Unhandled exception",
        correlation_id=correlation_id,
        error=str(exc),
        error_type=type(exc).__name__,
        exc_info=True,
    )

    return JSONResponse(
        content={
            "success": False,
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An internal server error occurred. Please contact support with the correlation ID.",
            "correlation_id": correlation_id,
            "timestamp": datetime.utcnow().isoformat(),
        },
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        headers={"X-Correlation-ID": correlation_id},
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
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint for load balancers and monitoring.

    Returns:
        Status information including service name and version
    """
    return {
        "status": "healthy",
        "service": "kb-rest-api",
        "version": "2.0.0",
        "cloud_provider": settings.CLOUD_PROVIDER,
        "storage_provider": settings.STORAGE_PROVIDER or settings.CLOUD_PROVIDER,
        "queue_provider": settings.QUEUE_PROVIDER or settings.CLOUD_PROVIDER,
    }


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
    "/kb/query",
    tags=["Knowledge Base"],
    summary="Query Knowledge Base",
    description="Search the knowledge base using LightRAG with proper request/response validation"
)
async def kb_query(request: Request):
    """
    Query knowledge base with LightRAG.

    Supports multiple query modes:
    - **naive**: Simple keyword search
    - **local**: Local context-aware search
    - **global**: Global knowledge graph search
    - **hybrid**: Combines local and global (recommended)
    """
    return await invoke_handler(request, "kb_query")


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


@api_router.post(
    "/kb/chat",
    tags=["Knowledge Base"],
    summary="Chat with Knowledge Base",
    description="Conversational interface to the knowledge base with context retention"
)
async def kb_chat(request: Request):
    """Chat with the knowledge base using conversational AI"""
    return await invoke_handler(request, "kb_chat")


@api_router.post(
    "/kb/index",
    tags=["Knowledge Base"],
    summary="Index Document",
    description="Add a document to the knowledge base for indexing"
)
async def kb_index(request: Request):
    """Index a document in the knowledge base"""
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
    "/documents/status",
    tags=["Documents"],
    summary="Check Indexing Status",
    description="Check the status of document indexing tasks"
)
async def indexing_status(request: Request):
    """Check the status of document indexing tasks"""
    return await invoke_handler(request, "check_indexing_status")


@api_router.post(
    "/documents/list",
    tags=["Documents"],
    summary="List Indexed Documents",
    description="Get a paginated list of indexed documents for a workspace"
)
async def list_documents(request: Request):
    """List all indexed documents for a workspace with pagination"""
    return await invoke_handler(request, "list_indexed_documents")


@api_router.delete(
    "/documents",
    tags=["Documents"],
    summary="Delete Documents",
    description="Delete documents from the knowledge base"
)
async def delete_documents(request: Request):
    """Delete documents from the knowledge base"""
    return await invoke_handler(request, "delete_documents")


@api_router.post(
    "/kb/graph",
    tags=["Knowledge Base"],
    summary="Get Knowledge Graph",
    description="Retrieve the knowledge graph for visualization"
)
async def knowledge_graph(request: Request):
    """Get the knowledge graph structure for a workspace"""
    return await invoke_handler(request, "get_knowledge_graph")


@api_router.post(
    "/llm/route",
    tags=["LLM"],
    summary="Route LLM Request",
    description="Intelligent routing of LLM requests to appropriate models"
)
async def llm_route(request: Request):
    """Route LLM requests to the appropriate model based on task type"""
    return await invoke_handler(request, "llm_route")


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
