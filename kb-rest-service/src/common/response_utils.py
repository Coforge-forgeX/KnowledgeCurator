"""
Utility functions for creating standardized API responses.

Every response this service returns has the same envelope, so a client can be
written once against it:

    success:      { "success": true,  "message": str, "data": {...},
                    "pagination": {...} (paginated endpoints only),
                    "timestamp": iso8601, "correlation_id": str }

    error:        { "success": false, "error": CODE, "message": str,
                    "details": {...} (subject to the policy below),
                    "timestamp": iso8601, "correlation_id": str }

`data` is ALWAYS a JSON object, never a bare list or scalar — a list-returning
endpoint puts its page under `data.items` (see `create_paginated_response`).
That keeps room to add sibling fields to any payload later without changing its
type, which a top-level array cannot do.
"""
import json
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

from src.core.abstractions import AbstractResponse

# Key holding the page of records for every list/paginated endpoint. `pagination`
# always describes this list.
ITEMS_KEY = "items"


def _as_data_object(data: Optional[Any]) -> Dict[str, Any]:
    """
    Coerce a handler's payload into the envelope's object-shaped `data`.

    Dicts pass through. A list is the common mistake — it becomes
    `{"items": [...]}` rather than a top-level array, so the envelope stays
    extensible. Anything else is a scalar and gets a `value` key.
    """
    if data is None:
        return {}
    if isinstance(data, dict):
        return data
    if isinstance(data, (list, tuple, set)):
        return {ITEMS_KEY: list(data)}
    return {"value": data}


def _should_include_error_details(status_code: int, explicit: Optional[bool] = None) -> bool:
    """Determine whether error details should be returned to API callers.

    Policy:
    - 4xx: include details by default (client-correctable input/auth issues)
    - 5xx: hide details by default to avoid leaking internals (SQL, stack traces, etc.)
    - DEBUG=true overrides and exposes details for troubleshooting.
    - explicit flag (when provided) has highest priority.
    """
    if explicit is not None:
        return bool(explicit)

    if status_code < 500:
        return True

    try:
        from src.core.config import settings

        return bool(getattr(settings, "DEBUG", False))
    except Exception:
        # Fail-safe: never leak internals if settings cannot be loaded.
        return False


def build_success_body(
    message: str,
    data: Optional[Any] = None,
    correlation_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the success envelope as a plain dict.

    The single definition of the success shape. `create_success_response` wraps
    it for handlers; FastAPI-level code (health, exception handlers) that must
    return a `JSONResponse` uses it directly, so neither path can drift.
    """
    body: Dict[str, Any] = {
        "success": True,
        "message": message,
        "data": _as_data_object(data),
        "timestamp": datetime.utcnow().isoformat(),
    }

    if correlation_id:
        body["correlation_id"] = correlation_id

    return body


def build_error_body(
    message: str,
    error_code: str = "ERROR",
    details: Optional[Dict[str, Any]] = None,
    status_code: int = 400,
    correlation_id: Optional[str] = None,
    include_details: Optional[bool] = None,
) -> Dict[str, Any]:
    """
    Build the error envelope as a plain dict.

    The single definition of the error shape, including the
    `_should_include_error_details` policy — which is why FastAPI's exception
    handlers go through here rather than assembling their own body and
    accidentally returning a 500's internals.
    """
    body: Dict[str, Any] = {
        "success": False,
        "error": error_code,
        "message": message,
        "timestamp": datetime.utcnow().isoformat(),
    }

    if details and _should_include_error_details(status_code, include_details):
        body["details"] = details

    if correlation_id:
        body["correlation_id"] = correlation_id

    return body


def create_success_response(
    message: str,
    data: Optional[Any] = None,
    status_code: int = 200,
    correlation_id: Optional[str] = None,
) -> AbstractResponse:
    """
    Create a standardized success response.

    Args:
        message: Success message
        data: Response payload. Must be (or be coercible to) a JSON object —
            see `_as_data_object`. Always present in the body, as `{}` when the
            endpoint has nothing to return, so clients never branch on its
            absence.
        status_code: HTTP status code (default: 200)
        correlation_id: Optional correlation ID for tracking

    Returns:
        AbstractResponse: Formatted success response
    """
    response_body = build_success_body(message, data, correlation_id)

    headers = {"Content-Type": "application/json"}
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    return AbstractResponse(
        # Ensure non-JSON-native objects (e.g., bson.ObjectId) don't crash success responses.
        body=json.dumps(response_body, default=str),
        status_code=status_code,
        mimetype="application/json",
        headers=headers,
    )


def create_error_response(
    message: str,
    error_code: str = "ERROR",
    details: Optional[Dict[str, Any]] = None,
    status_code: int = 400,
    correlation_id: Optional[str] = None,
    include_details: Optional[bool] = None,
) -> AbstractResponse:
    """
    Create a standardized error response.

    Args:
        message: Error message
        error_code: Error code identifier
        details: Optional error details
        status_code: HTTP status code (default: 400)
        correlation_id: Optional correlation ID for tracking
        include_details: Optional override for details visibility policy

    Returns:
        AbstractResponse: Formatted error response
    """
    response_body = build_error_body(
        message=message,
        error_code=error_code,
        details=details,
        status_code=status_code,
        correlation_id=correlation_id,
        include_details=include_details,
    )

    headers = {"Content-Type": "application/json"}
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    return AbstractResponse(
        body=json.dumps(response_body, default=str),
        status_code=status_code,
        mimetype="application/json",
        headers=headers,
    )


def create_exception_response(
    exc: Exception,
    fallback_message: str,
    fallback_error_code: str = "INTERNAL_ERROR",
    correlation_id: Optional[str] = None,
) -> AbstractResponse:
    """
    Map an exception to its HTTP response.

    An `APIException` already carries the status code and error code that
    describe it (400 validation, 403 authorization, 404 not found, ...), so
    honour them instead of flattening every failure into a 500. Anything else is
    unexpected and becomes a 500 whose internals stay behind the
    `_should_include_error_details` policy.
    """
    from src.core.exceptions import APIException

    if isinstance(exc, APIException):
        return create_error_response(
            message=exc.message,
            error_code=exc.error_code,
            details=exc.details or None,
            status_code=exc.status_code,
            correlation_id=correlation_id,
        )

    return create_error_response(
        message=fallback_message,
        error_code=fallback_error_code,
        details={"error": str(exc)},
        status_code=500,
        correlation_id=correlation_id,
    )


def create_paginated_response(
    message: str,
    items: Iterable[Any],
    page: int,
    page_size: int,
    total_count: int,
    status_code: int = 200,
    correlation_id: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> AbstractResponse:
    """
    Create a standardized paginated response.

    The page goes to `data.items` — never to a top-level array — and
    `pagination` sits beside `data` describing that list.

    Args:
        message: Success message
        items: Records for the current page
        page: Current page number
        page_size: Number of items per page
        total_count: Total number of items across all pages
        status_code: HTTP status code (default: 200)
        correlation_id: Optional correlation ID for tracking
        extra: Optional additional keys merged into `data` alongside `items`,
            for endpoints that return a page *plus* context about it (e.g.
            /chat/load returns a page of messages with the session's metadata).
            `items` itself cannot be overwritten.

    Returns:
        AbstractResponse: Formatted paginated response
    """
    total_pages = (total_count + page_size - 1) // page_size if page_size > 0 else 0

    data: Dict[str, Any] = {ITEMS_KEY: list(items)}
    if extra:
        data.update({k: v for k, v in extra.items() if k != ITEMS_KEY})

    response_body = build_success_body(message, data, correlation_id)
    response_body["pagination"] = {
        "page": page,
        "page_size": page_size,
        "total_count": total_count,
        "total_pages": total_pages,
        "has_next": page < total_pages,
        "has_previous": page > 1,
    }

    headers = {"Content-Type": "application/json"}
    if correlation_id:
        headers["X-Correlation-ID"] = correlation_id

    return AbstractResponse(
        body=json.dumps(response_body, default=str),
        status_code=status_code,
        mimetype="application/json",
        headers=headers,
    )


def create_query_response(
    answer: str,
    sources: Optional[list] = None,
    retrieved_chunks: Optional[list] = None,
    metadata: Optional[Dict[str, Any]] = None,
    status_code: int = 200,
    correlation_id: Optional[str] = None,
    message: str = "Query processed successfully",
) -> AbstractResponse:
    """
    Create a standardized query response for LightRAG queries.

    A thin wrapper over `create_success_response`: the answer and its evidence
    live under `data` like every other payload, so a client reads
    `data.answer` / `data.sources` here exactly as it reads `data` elsewhere.

    Args:
        answer: Generated answer text
        sources: Optional list of source references
        retrieved_chunks: Optional list of retrieved context chunks
        metadata: Optional additional metadata
        status_code: HTTP status code (default: 200)
        correlation_id: Optional correlation ID for tracking
        message: Envelope message

    Returns:
        AbstractResponse: Formatted query response
    """
    data: Dict[str, Any] = {"answer": answer}

    if sources:
        data["sources"] = sources

    if retrieved_chunks:
        data["retrieved_chunks"] = retrieved_chunks

    if metadata:
        data["metadata"] = metadata

    return create_success_response(
        message=message,
        data=data,
        status_code=status_code,
        correlation_id=correlation_id,
    )


def create_batch_response(
    message: str,
    successful: int,
    failed: int,
    total: int,
    details: Optional[list] = None,
    status_code: int = 200,
    correlation_id: Optional[str] = None,
) -> AbstractResponse:
    """
    Create a standardized batch operation response.

    `summary` and the per-operation `details` list live under `data`; note that
    `data.details` is batch *results*, unrelated to an error body's `details`.

    Args:
        message: Success message
        successful: Number of successful operations
        failed: Number of failed operations
        total: Total number of operations
        details: Optional list of operation details
        status_code: HTTP status code (default: 200)
        correlation_id: Optional correlation ID for tracking

    Returns:
        AbstractResponse: Formatted batch response
    """
    data: Dict[str, Any] = {
        "summary": {
            "total": total,
            "successful": successful,
            "failed": failed,
            "success_rate": f"{(successful / total * 100):.1f}%" if total > 0 else "0%",
        },
    }

    if details:
        data["details"] = details

    # `success` describes the request, not the individual operations: the batch
    # ran, and `data.summary.failed` says how it went. It used to be
    # `failed == 0`, which produced a `success: false` body that still carried
    # `data` — a third shape for clients to handle, and inconsistent with
    # delete_files_by_id, which already reports partial failure as a 207 success.
    # A batch that wholly failed should raise, not return here.
    return create_success_response(
        message=message,
        data=data,
        status_code=status_code,
        correlation_id=correlation_id,
    )
