"""
Middleware components for Azure Functions request/response interception.

This module provides middleware functionality specifically designed for Azure Functions.
Note: FastAPI/Starlette middleware classes are NOT compatible with Azure Functions v1
programming model. Instead, this module uses decorators and helper functions.

Active middleware features:
- azure_http_decorator: Main decorator for Azure Functions with built-in:
  * Request/response logging
  * Correlation ID tracking
  * Security headers
  * CORS headers
  * Request size validation
  * Error handling
  * OPTIONS preflight handling
"""
import inspect
import json
import time
import uuid
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar, cast

import azure.functions as func
from typing_extensions import ParamSpec

from .config import settings
from .exceptions import APIException
from .logging import Logger


class AzureFunctionMiddleware:
    """Middleware wrapper for Azure Functions context"""

    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger or Logger("azure-function")

    def log_function_start(
        self, req: func.HttpRequest, context: func.Context
    ) -> Dict[str, Any]:
        """Log function execution start"""
        # Use existing correlation ID from request header, or generate new one
        correlation_id = req.headers.get("X-Correlation-ID") or req.headers.get(
            "x-correlation-id"
        )
        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        start_time = time.time()

        self.logger.info(
            "Azure Function Started",
            function_name=context.function_name,
            invocation_id=context.invocation_id,
            correlation_id=correlation_id,
            method=req.method,
            url=req.url,
            headers=dict(req.headers),
        )

        return {
            "correlation_id": correlation_id,
            "start_time": start_time,
            "function_name": context.function_name,
            "invocation_id": context.invocation_id,
        }

    def log_function_end(
        self,
        execution_context: Dict[str, Any],
        status_code: int,
        response_data: Any = None,
    ):
        """Log function execution end"""
        execution_time_ms = (time.time() - execution_context["start_time"]) * 1000

        self.logger.info(
            "Azure Function Completed",
            function_name=execution_context["function_name"],
            invocation_id=execution_context["invocation_id"],
            correlation_id=execution_context["correlation_id"],
            status_code=status_code,
            execution_time_ms=execution_time_ms,
        )

    def log_function_error(self, execution_context: Dict[str, Any], error: Exception):
        """Log function execution error"""
        execution_time_ms = (time.time() - execution_context["start_time"]) * 1000

        self.logger.error(
            "Azure Function Error",
            error=error,
            function_name=execution_context["function_name"],
            invocation_id=execution_context["invocation_id"],
            correlation_id=execution_context["correlation_id"],
            execution_time_ms=execution_time_ms,
        )


P = ParamSpec("P")
R = TypeVar("R")


# Security headers applied to every Azure Function HTTP response by the interceptor.
# The Starlette SecurityMiddleware above never runs under the Azure Functions v1
# model (function.json triggers), so headers must be set here to actually apply.
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "X-XSS-Protection": "1; mode=block",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}


def _get_cors_headers(origin: str = None) -> Dict[str, str]:
    """Generate CORS headers based on configuration."""
    cors_headers = {}

    # Determine allowed origin
    allowed_origins = settings.security.CORS_ORIGINS
    if "*" in allowed_origins or not origin:
        cors_headers["Access-Control-Allow-Origin"] = "*"
    elif origin in allowed_origins:
        cors_headers["Access-Control-Allow-Origin"] = origin
        if settings.security.CORS_ALLOW_CREDENTIALS:
            cors_headers["Access-Control-Allow-Credentials"] = "true"

    # Add other CORS headers
    cors_headers["Access-Control-Allow-Methods"] = ", ".join(
        settings.security.CORS_ALLOW_METHODS
    )
    cors_headers["Access-Control-Allow-Headers"] = ", ".join(
        settings.security.CORS_ALLOW_HEADERS
    )
    cors_headers["Access-Control-Max-Age"] = "86400"  # 24 hours

    return cors_headers


def _apply_security_headers(response: func.HttpResponse, origin: str = None) -> None:
    """Add standard security and CORS headers to an HttpResponse (best-effort)."""
    try:
        response.headers.update(SECURITY_HEADERS)
        response.headers.update(_get_cors_headers(origin))
    except Exception:
        pass


def azure_http_decorator(
    func_mw: "AzureFunctionMiddleware" = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator to auto-log Azure Function start/end/error.

    Usage:
        @azure_http_decorator()
        def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
            ...

    Notes:
    - Expects signature (req: func.HttpRequest, context: func.Context, ...)
    - Injects X-Correlation-ID header into successful/failed HttpResponse when possible.
    - Automatically wraps exceptions in standardized error responses.
    - Applies security headers to all responses.
    """

    def _decorator(fn: Callable[P, R]) -> Callable[P, R]:
        mw = func_mw or azure_function_middleware

        def _start(args, kwargs):
            # Support calling as main(req, context) or main(req=req, context=context)
            req = (
                kwargs.get("req")
                if "req" in kwargs
                else (args[0] if len(args) > 0 else None)
            )
            context = (
                kwargs.get("context")
                if "context" in kwargs
                else (args[1] if len(args) > 1 else None)
            )

            if isinstance(req, func.HttpRequest) and context is not None:
                # Validate request size
                content_length = req.headers.get("content-length")
                if content_length:
                    try:
                        if int(content_length) > settings.MAX_REQUEST_SIZE:
                            raise APIException(
                                status_code=413,
                                error_code="REQUEST_TOO_LARGE",
                                message=f"Request payload exceeds maximum size of {settings.MAX_REQUEST_SIZE} bytes",
                            )
                    except (ValueError, TypeError):
                        pass

                execution_context = mw.log_function_start(req, context)
                # Store origin for CORS headers
                execution_context["origin"] = req.headers.get(
                    "origin"
                ) or req.headers.get("Origin")

                # Expose correlation id to handlers for structured API payloads.
                try:
                    setattr(
                        context, "correlation_id", execution_context["correlation_id"]
                    )
                except Exception:
                    pass
                return execution_context
            return None

        def _finish(result, execution_context, req=None):
            # Handle CORS preflight requests
            if req and req.method == "OPTIONS":
                origin = execution_context.get("origin") if execution_context else None
                preflight_response = func.HttpResponse(
                    status_code=204,
                    headers={
                        "X-Correlation-ID": execution_context.get(
                            "correlation_id", str(uuid.uuid4())
                        )
                    },
                )
                _apply_security_headers(preflight_response, origin)
                return preflight_response

            if execution_context is not None:
                # Try to infer status_code for logging.
                status_code = getattr(result, "status_code", 200)
                mw.log_function_end(execution_context, int(status_code))

                # Best-effort add correlation id to response header.
                try:
                    if isinstance(result, func.HttpResponse):
                        result.headers["X-Correlation-ID"] = execution_context[
                            "correlation_id"
                        ]
                except Exception:
                    pass

            if isinstance(result, func.HttpResponse):
                origin = execution_context.get("origin") if execution_context else None
                _apply_security_headers(result, origin)

            return result

        def _on_error(e, execution_context):
            # Centralized error handling: log with correlation id, then return a
            # consistent JSON error response carrying that same correlation id.
            correlation_id = (
                execution_context["correlation_id"]
                if execution_context
                else str(uuid.uuid4())
            )
            if execution_context is not None:
                mw.log_function_error(execution_context, e)
            else:
                mw.logger.error(
                    "Azure Function Error", error=e, correlation_id=correlation_id
                )

            if isinstance(e, APIException):
                # Known API exceptions - safe to expose message to client
                status_code = e.status_code
                body = {
                    "success": False,
                    "error": e.error_code,
                    "message": e.message,
                    "details": e.details,
                }
            else:
                # Unknown/unexpected exceptions - log full details server-side
                # but only send generic message to client
                mw.logger.error(
                    "Unhandled exception in Azure Function",
                    error=e,
                    correlation_id=correlation_id,
                    error_type=type(e).__name__,
                    error_details=str(e),
                )
                status_code = 500
                body = {
                    "success": False,
                    "error": "INTERNAL_SERVER_ERROR",
                    "message": "An internal server error occurred. Please contact support with the correlation ID.",
                }

            body["correlation_id"] = correlation_id
            body["timestamp"] = datetime.utcnow().isoformat()

            error_response = func.HttpResponse(
                json.dumps(body),
                status_code=status_code,
                mimetype="application/json",
                headers={"X-Correlation-ID": correlation_id},
            )
            origin = execution_context.get("origin") if execution_context else None
            _apply_security_headers(error_response, origin)
            return error_response

        _wrapped: Callable[P, R]

        if inspect.iscoroutinefunction(fn):
            async_fn = cast(Callable[P, Awaitable[Any]], fn)

            async def _async_wrapped(*args: P.args, **kwargs: P.kwargs) -> Any:
                # Get request object
                req = (
                    kwargs.get("req")
                    if "req" in kwargs
                    else (args[0] if len(args) > 0 else None)
                )

                execution_context = _start(args, kwargs)

                # Handle OPTIONS preflight before calling the actual function
                if isinstance(req, func.HttpRequest) and req.method == "OPTIONS":
                    return _finish(None, execution_context, req)

                try:
                    result = await async_fn(*args, **kwargs)
                    return _finish(result, execution_context, req)
                except Exception as e:
                    return _on_error(e, execution_context)

            _wrapped = cast(Callable[P, R], _async_wrapped)

        else:

            def _sync_wrapped(*args: P.args, **kwargs: P.kwargs) -> Any:
                # Get request object
                req = (
                    kwargs.get("req")
                    if "req" in kwargs
                    else (args[0] if len(args) > 0 else None)
                )

                execution_context = _start(args, kwargs)

                # Handle OPTIONS preflight before calling the actual function
                if isinstance(req, func.HttpRequest) and req.method == "OPTIONS":
                    return _finish(None, execution_context, req)

                try:
                    result = fn(*args, **kwargs)
                    return _finish(result, execution_context, req)
                except Exception as e:
                    return _on_error(e, execution_context)

            _wrapped = cast(Callable[P, R], _sync_wrapped)

        # Preserve name/docstring for Azure Functions indexing and logs.
        _wrapped.__name__ = getattr(fn, "__name__", "wrapped")
        _wrapped.__doc__ = getattr(fn, "__doc__", None)
        return _wrapped

    return _decorator


# Backward-compatible alias: older code imports/uses azure_http_interceptor.
# Keep this to avoid breaking existing deployments; prefer azure_http_decorator.
azure_http_interceptor = azure_http_decorator


# Create Azure Function middleware instance
azure_function_middleware = AzureFunctionMiddleware()
