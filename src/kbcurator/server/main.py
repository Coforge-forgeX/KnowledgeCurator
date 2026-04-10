
import json
import asyncio
from typing import List, Optional
import os
from agent_search.server.server import mcp
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, StreamingResponse
from agent_search.utils.auth import extract_token_from_headers, verify_jwt_token
from agent_search.utils.request_context import request_var
from agent_search.utils.mongodb_singleton import get_mongodb_client
from agent_search.utils.session_history_manager import SessionHistoryManager
import uvicorn
from agent_search.server import storage_config

# --- Initialize global services (DI singletons) ---
mongo_client = get_mongodb_client()
session = SessionHistoryManager(mongo_client)

storage_config.initialize_storage()

# --- Import tools so they are registered with MCP ---
import agent_search.tools.ingestion_new        # noqa: F401
import agent_search.tools.kb_adapter_tool      # noqa: F401
import agent_search.tools.user_management_system  # noqa: F401
import agent_search.tools.kb_curator_chatbot   # noqa: F401


# ---------------------------
# Middleware
# ---------------------------

class AuthMiddleware(BaseHTTPMiddleware):
    """
    - Sets request context into a ContextVar for downstream tools.
    - Bypasses auth for OPTIONS (CORS preflight).
    - Protects POST /mcp for all tools except the listed public ones.
    """

    PUBLIC_TOOLS: List[str] = [
        "login_user",
        "refresh_jwt_token",
        "query_rag",
        "upload_and_index_tool",
    ]

    async def dispatch(self, request: Request, call_next):
        request_var.set(request)

        if request.method.upper() == "OPTIONS":
            return await call_next(request)

        if request.url.path.startswith("/mcp") and request.method.upper() == "POST":
            try:
                body_bytes = await request.body()
                payload = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}
            except Exception:
                payload = {}

            tool_name = (
                payload.get("name")
                or (payload.get("params") or {}).get("name")
                or payload.get("tool")
                or payload.get("operation")
            )

            if not tool_name:
                return await call_next(request)

            if tool_name in self.PUBLIC_TOOLS:
                return await call_next(request)

            token = extract_token_from_headers(dict(request.headers))
            if not token:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "OAuthError",
                        "message": "Missing authentication token in headers",
                    },
                )

            try:
                claims = verify_jwt_token(token)
                request.state.jwt_claims = claims
            except Exception as e:
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "OAuthError",
                        "message": f"Invalid or expired token: {str(e)}",
                    },
                )

        return await call_next(request)


class SecurityAndCORSMiddleware(BaseHTTPMiddleware):
    """
    - Answers OPTIONS preflight directly with 200 and proper CORS headers.
    - Adds CSP, HSTS, and CORS headers on all responses.
    - Supports dynamic origin reflection using ALLOWED_ORIGINS env var.
    """
    def _parse_allowed_origins(self) -> Optional[List[str]]:
        raw = os.getenv("ALLOWED_ORIGINS", "").strip()
        if not raw:
            return None
        return [o.strip() for o in raw.split(",") if o.strip()]

    def _bool_env(self, name: str, default: bool = False) -> bool:
        val = os.getenv(name, str(default)).strip().lower()
        return val in ("1", "true", "yes", "y", "t")

    async def dispatch(self, request: Request, call_next):
        allowed_origins = self._parse_allowed_origins()
        allow_credentials = self._bool_env("ALLOW_CREDENTIALS", default=False)
        req_origin = request.headers.get("origin")
        ac_req_headers = request.headers.get("access-control-request-headers", "")
        ac_req_method = request.headers.get("access-control-request-method", "GET")

        allow_origin: Optional[str] = None
        vary_origin = False
        if allowed_origins:
            if req_origin and req_origin in allowed_origins:
                allow_origin = req_origin
                vary_origin = True
        else:
            if req_origin and allow_credentials:
                allow_origin = req_origin
                vary_origin = True
            else:
                allow_origin = "*"

        if request.method.upper() == "OPTIONS":
            headers = {}
            if allow_origin:
                headers["Access-Control-Allow-Origin"] = allow_origin
            if vary_origin:
                headers["Vary"] = "Origin"
            if allow_credentials and req_origin:
                headers["Access-Control-Allow-Credentials"] = "true"
            else:
                headers["Access-Control-Allow-Credentials"] = "false"
            headers["Access-Control-Allow-Methods"] = ac_req_method or "GET, POST, OPTIONS"
            headers["Access-Control-Allow-Headers"] = ac_req_headers or "Authorization, Content-Type, x-amz-content-sha256"
            headers["Access-Control-Max-Age"] = "600"
            headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
            headers["Content-Security-Policy"] = (
                "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; "
                "img-src 'self' data: https:; object-src 'none'; frame-ancestors 'none';"
            )
            return PlainTextResponse("ok", status_code=204, headers=headers)

        response = await call_next(request)

        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; font-src 'self'; "
            "img-src 'self' data: https:; object-src 'none'; frame-ancestors 'none';"
        )
        response.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"

        if allow_origin:
            response.headers["Access-Control-Allow-Origin"] = allow_origin
        if vary_origin:
            response.headers["Vary"] = "Origin"
        if allow_credentials and req_origin:
            response.headers["Access-Control-Allow-Credentials"] = "true"
        else:
            response.headers["Access-Control-Allow-Credentials"] = "false"

        response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        response.headers["Access-Control-Expose-Headers"] = (
            "Authorization, Content-Type, Set-Cookie, mcp-protocol-version, x-amz-content-sha256"
        )

        return response


custom_middleware = [
    Middleware(AuthMiddleware),
    Middleware(SecurityAndCORSMiddleware),
]


# ---------------------------
# MCP app + routes
# ---------------------------

base_app = mcp.http_app(
    transport="http",
    path="/mcp",
    middleware=custom_middleware,
    stateless_http=True,
)

async def health_check(request: Request):
    return JSONResponse({"status": "ok"})

base_app.add_route("/health", health_check, methods=["GET", "OPTIONS"])

async def root(request: Request):
    return JSONResponse({"service": "mcp", "status": "ok"})

base_app.add_route("/", root, methods=["GET", "OPTIONS"])


async def mcp_get(request: Request):
    accept = (request.headers.get("accept") or "").lower()

    if "text/event-stream" in accept:
        async def stream():
            try:
                yield b": connected\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    yield b": keep-alive\n\n"
                    await asyncio.sleep(15)
            except asyncio.CancelledError:
                pass

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    return JSONResponse(
        status_code=200,
        content={"endpoint": "/mcp", "methods": ["POST", "GET"], "status": "ok"},
        headers={"Cache-Control": "no-cache"},
    )

base_app.add_route("/mcp", mcp_get, methods=["GET", "OPTIONS"])


# ---------------------------
# Cookie wrapper (refresh token)
# ---------------------------

class CookieWrapperApp:
    """
    Intercepts responses and appends a Set-Cookie for the refresh token,
    if present in request.state.refresh_token.
    """

    def __init__(self, app):
        self.app = app

    def _bool_env(self, name: str, default: bool = False) -> bool:
        val = os.getenv(name, str(default)).strip().lower()
        return val in ("1", "true", "yes", "y", "t")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if scope.get("path", "").startswith("/mcp"):
            await self.app(scope, receive, send)
            return

        response_started = False
        status_code = None
        headers = []
        body_parts = []

        async def send_wrapper(message):
            nonlocal response_started, status_code, headers, body_parts

            if message["type"] == "http.response.start":
                response_started = True
                status_code = message["status"]
                headers = list(message.get("headers", []))

                for k, v in headers:
                    if k.lower() == b"content-type" and b"text/event-stream" in v.lower():
                        await send(message)
                        return
                return

            elif message["type"] == "http.response.body":
                if response_started and any(
                    k.lower() == b"content-type" and b"text/event-stream" in v.lower()
                    for k, v in headers
                ):
                    await send(message)
                    return

                body_parts.append(message.get("body", b""))

                if not message.get("more_body", False):
                    full_body = b"".join(body_parts)

                    try:
                        request = request_var.get()
                        if request and getattr(request.state, "refresh_token", None):
                            pass
                    except Exception:
                        pass

                    await send({
                        "type": "http.response.start",
                        "status": status_code,
                        "headers": headers,
                    })
                    await send({
                        "type": "http.response.body",
                        "body": full_body,
                    })
            else:
                await send(message)

        await self.app(scope, receive, send_wrapper)


http_app = CookieWrapperApp(base_app)


# ---------------------------
# Local run / Azure App Service entry
# ---------------------------

def _get_port() -> int:
    for key in ("WEBSITES_PORT", "PORT"):
        val = os.getenv(key)
        if val and val.isdigit():
            return int(val)
    return 8000


if __name__ == "__main__":
    uvicorn.run(
        "agent_search.server.main:http_app",
        host="0.0.0.0",
        port=_get_port(),
        reload=False,
    )
