"""Runtime adapters for Azure Functions, AWS Lambda, and GCP Functions.

This module provides a single request-routing core that lets provider-specific
entrypoints call the same handler registry used by the FastAPI app.
"""

import base64
import inspect
import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from src.core.abstractions import AbstractContext, AbstractRequest, AbstractResponse
from src.core.config import settings
from src.core.logging import get_logger
from src.registry import get_handler

logger = get_logger(__name__)

# HTTP route -> handler mapping shared across all serverless runtimes.
ROUTE_TO_HANDLER = {
    ("POST", "/api/v2/kb/index"): "kb_index",
    ("POST", "/api/v2/documents/upload"): "upload_and_index",
    ("POST", "/api/v2/workspaces/index-files"): "index_workspace_files",
    ("GET", "/api/v2/documents/status"): "file_tasks_status",
    ("GET", "/api/v2/workspaces/documents"): "workspace_documents",
    ("DELETE", "/api/v2/files"): "delete_files_by_id",
    ("DELETE", "/api/v2/workspaces/documents/all"): "delete_all_indexed_documents",
    ("POST", "/api/v2/kb/graph"): "fetch_graph",
    ("POST", "/api/v2/kb/graph/mutate"): "mutate_knowledge_graph",
    ("POST", "/api/v2/query-kb"): "query_rag",  # Using optimized query_rag handler
    ("GET", "/api/v2/files/{file_id}/download"): "query_source_download_url",

    # Chat / Conversation session management
    ("POST", "/api/v2/chat/start"): "chat_start_conversation",
    ("GET", "/api/v2/chat/history"): "chat_get_conversation_history",
    ("GET", "/api/v2/chat/load"): "chat_load_conversation",
    ("POST", "/api/v2/chat/session/rename"): "chat_rename_conversation",
    ("DELETE", "/api/v2/chat/session/delete"): "chat_delete_conversation",

    # Chat / Messaging
    ("POST", "/api/v2/chat/message"): "message_gpt",
    ("POST", "/api/v2/chat/message/cancel"): "cancel_chat_message",
}


@dataclass
class ProviderContext(AbstractContext):
    """Minimal provider-agnostic execution context."""

    _request_id: str
    _function_name: str

    @property
    def request_id(self) -> str:
        return self._request_id

    @property
    def function_name(self) -> str:
        return self._function_name


class DictLikeRequest(AbstractRequest):
    """Simple request object backed by plain dictionaries."""

    def __init__(
        self,
        method: str,
        url: str,
        path: str,
        headers: Optional[Dict[str, str]] = None,
        body: Optional[Any] = None,
        query_params: Optional[Dict[str, str]] = None,
        cookies: Optional[Dict[str, str]] = None,
    ):
        self._method = (method or "GET").upper()
        self._url = url or ""
        self._path = path or "/"
        self._headers = {str(k).lower(): str(v) for k, v in (headers or {}).items()}
        self._body = body
        self._query_params = query_params or {}
        self._cookies = cookies or {}

    def get_header(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return self._headers.get(name.lower(), default)

    def get_json(self) -> Dict[str, Any]:
        if self._body is None:
            return {}
        if isinstance(self._body, dict):
            return self._body
        if isinstance(self._body, (bytes, bytearray)):
            raw = self._body.decode("utf-8")
            return json.loads(raw) if raw else {}
        if isinstance(self._body, str):
            return json.loads(self._body) if self._body else {}
        return {}

    def get_query_param(self, name: str, default: Optional[str] = None) -> Optional[str]:
        return self._query_params.get(name, default)

    def get_query_params(self) -> Dict[str, str]:
        return self._query_params

    def get_cookies(self) -> Dict[str, str]:
        return self._cookies

    @property
    def method(self) -> str:
        return self._method

    @property
    def url(self) -> str:
        return self._url

    @property
    def path(self) -> str:
        return self._path


def _normalize_path(raw_path: str) -> str:
    """Normalize path and strip stage prefix when /api/v2 is present."""
    path = raw_path or "/"
    if not path.startswith("/"):
        path = "/" + path

    api_index = path.find("/api/v2/")
    if api_index > 0:
        path = path[api_index:]

    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]

    return path


def _route_to_handler(method: str, path: str) -> Optional[str]:
    """Resolve handler for exact and templated routes."""
    handler = ROUTE_TO_HANDLER.get((method, path))
    if handler:
        return handler

    if method == "GET" and path.startswith("/api/v2/files/") and path.endswith("/download"):
        return ROUTE_TO_HANDLER.get((method, "/api/v2/files/{file_id}/download"))

    return None


def _parse_cookie_header(cookie_header: Optional[str]) -> Dict[str, str]:
    cookies: Dict[str, str] = {}
    if not cookie_header:
        return cookies

    for item in cookie_header.split(";"):
        chunk = item.strip()
        if not chunk or "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        cookies[key.strip()] = value.strip()

    return cookies


async def _build_health_response(correlation_id: str) -> AbstractResponse:
    """
    Health payload in the standard envelope — see `src.common.response_utils`.

    An unhealthy service still answers in the success envelope and signals the
    problem through the 503 status and `data.status`, so this endpoint never
    returns two different body shapes. Mirrors `main.py`'s /health.
    """
    from src.common import create_success_response
    from src.core.health import run_health_checks

    checks, overall_status = await run_health_checks()

    return create_success_response(
        message=overall_status,
        data={
            "status": overall_status,
            "service": "kb-rest-api",
            "version": settings.VERSION,
            "cloud_provider": settings.CLOUD_PROVIDER,
            "storage_provider": settings.STORAGE_PROVIDER or settings.CLOUD_PROVIDER,
            "queue_provider": settings.QUEUE_PROVIDER or settings.CLOUD_PROVIDER,
            "checks": checks,
        },
        status_code=200 if overall_status == "healthy" else 503,
        correlation_id=correlation_id,
    )


async def dispatch_request(req: AbstractRequest, ctx: AbstractContext) -> AbstractResponse:
    """Route request to a registered handler and return AbstractResponse."""
    method = (req.method or "GET").upper()
    path = _normalize_path(req.path)

    if method == "GET" and path == "/health":
        return await _build_health_response(ctx.correlation_id)

    if method == "OPTIONS":
        return AbstractResponse(
            body="",
            status_code=204,
            headers={"X-Correlation-ID": ctx.correlation_id},
            mimetype="text/plain",
        )

    handler_name = _route_to_handler(method, path)
    if not handler_name:
        from src.common import create_error_response

        return create_error_response(
            message=f"No route found for {method} {path}",
            error_code="NOT_FOUND",
            status_code=404,
            correlation_id=ctx.correlation_id,
        )

    handler_module = get_handler(handler_name)
    handler_main = handler_module.main

    if inspect.iscoroutinefunction(handler_main):
        result = await handler_main(req, ctx)
    else:
        result = handler_main(req, ctx)

    if isinstance(result, AbstractResponse):
        response = result
    else:
        response = AbstractResponse(
            body=result,
            status_code=200,
            mimetype="application/json",
        )

    if "X-Correlation-ID" not in response.headers:
        response.headers["X-Correlation-ID"] = ctx.correlation_id

    return response


def abstract_response_to_http_tuple(resp: AbstractResponse) -> Tuple[str, int, Dict[str, str], str]:
    """Convert AbstractResponse to (body, status, headers, mimetype)."""
    body = resp.body
    if isinstance(body, (dict, list)):
        payload = json.dumps(body)
    elif body is None:
        payload = ""
    else:
        payload = str(body)

    headers = dict(resp.headers or {})
    if "Content-Type" not in headers:
        headers["Content-Type"] = resp.mimetype or "application/json"

    return payload, int(resp.status_code), headers, headers["Content-Type"]


def from_azure_request(req: Any) -> Tuple[DictLikeRequest, ProviderContext]:
    """Build abstract request/context from azure.functions.HttpRequest."""
    headers = dict(getattr(req, "headers", {}) or {})
    raw_body = req.get_body()
    try:
        body = req.get_json()
    except Exception:
        body = raw_body.decode("utf-8") if raw_body else ""

    query_params = dict(getattr(req, "params", {}) or {})
    parsed_url = urlparse(req.url)
    if not query_params and parsed_url.query:
        query_params = {k: v[-1] for k, v in parse_qs(parsed_url.query).items()}

    cookies = _parse_cookie_header(headers.get("cookie") or headers.get("Cookie"))
    path = parsed_url.path or "/"

    request = DictLikeRequest(
        method=req.method,
        url=req.url,
        path=path,
        headers=headers,
        body=body,
        query_params=query_params,
        cookies=cookies,
    )

    request_id = (
        headers.get("x-correlation-id")
        or headers.get("X-Correlation-ID")
        or str(uuid.uuid4())
    )
    context = ProviderContext(_request_id=request_id, _function_name="http_trigger")
    return request, context


def from_lambda_event(event: Dict[str, Any], context: Any) -> Tuple[DictLikeRequest, ProviderContext]:
    """Build abstract request/context from API Gateway/Lambda proxy event."""
    headers = event.get("headers") or {}
    method = (
        event.get("requestContext", {}).get("http", {}).get("method")
        or event.get("httpMethod")
        or "GET"
    )
    raw_path = event.get("rawPath") or event.get("path") or "/"
    path = _normalize_path(raw_path)

    query_params = event.get("queryStringParameters") or {}
    raw_query = event.get("rawQueryString")
    if raw_query and not query_params:
        query_params = {k: v[-1] for k, v in parse_qs(raw_query).items()}

    body = event.get("body")
    if body and event.get("isBase64Encoded"):
        body = base64.b64decode(body).decode("utf-8")

    host = headers.get("host", "lambda")
    proto = headers.get("x-forwarded-proto", "https")
    query = f"?{raw_query}" if raw_query else ""
    url = f"{proto}://{host}{raw_path}{query}"

    cookies: Dict[str, str] = {}
    if isinstance(event.get("cookies"), list):
        for item in event["cookies"]:
            cookies.update(_parse_cookie_header(item))
    else:
        cookies = _parse_cookie_header(headers.get("cookie") or headers.get("Cookie"))

    request = DictLikeRequest(
        method=method,
        url=url,
        path=path,
        headers=headers,
        body=body,
        query_params=query_params,
        cookies=cookies,
    )

    request_id = (
        headers.get("x-correlation-id")
        or headers.get("X-Correlation-ID")
        or getattr(context, "aws_request_id", None)
        or str(uuid.uuid4())
    )
    function_name = getattr(context, "function_name", "lambda_handler")
    abstract_context = ProviderContext(_request_id=request_id, _function_name=function_name)
    return request, abstract_context


def from_gcp_request(request: Any) -> Tuple[DictLikeRequest, ProviderContext]:
    """Build abstract request/context from Flask-compatible GCP request."""
    headers = dict(getattr(request, "headers", {}) or {})
    body = request.get_data(as_text=True) if hasattr(request, "get_data") else ""
    json_body = None
    try:
        json_body = request.get_json(silent=True) if hasattr(request, "get_json") else None
    except Exception:
        json_body = None

    query_params = dict(getattr(request, "args", {}) or {})
    cookies = dict(getattr(request, "cookies", {}) or {})

    path = _normalize_path(getattr(request, "path", "/"))
    url = getattr(request, "url", path)

    abstract_request = DictLikeRequest(
        method=getattr(request, "method", "GET"),
        url=url,
        path=path,
        headers=headers,
        body=json_body if json_body is not None else body,
        query_params=query_params,
        cookies=cookies,
    )

    request_id = headers.get("X-Correlation-ID") or headers.get("x-correlation-id") or str(uuid.uuid4())
    ctx = ProviderContext(_request_id=request_id, _function_name="gcp_http")
    return abstract_request, ctx
