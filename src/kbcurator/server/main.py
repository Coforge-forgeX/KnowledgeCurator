from dotenv import load_dotenv
load_dotenv()
from common_adapters.langfuse_instrumentation import setup_langfuse
setup_langfuse()   
import asyncio
import json
import os
from typing import List, Optional

import uvicorn
from kbcurator.server.server import mcp
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import (
    JSONResponse,
    PlainTextResponse,
    StreamingResponse,
)
from kbcurator.utils.auth import extract_token_from_headers
from kbcurator.utils.sso_jwt import verify_token
from kbcurator.utils.mongodb_singleton import get_mongodb_client
from kbcurator.utils.request_context import request_var
from kbcurator.utils.session_history_manager import SessionHistoryManager

# --- Initialize global services (DI singletons) ---
mongo_client = get_mongodb_client()
session = SessionHistoryManager(mongo_client)

# --- Import tools so they are registered with MCP ---
from kbcurator.tools import ingestion_new  # noqa: F401
from kbcurator.tools import kb_adapter_tool  # noqa: F401
from kbcurator.tools import kb_curator_chatbot  # noqa: F401
from kbcurator.tools import user_management_system  # noqa: F401
from kbcurator.tools import sso_login_tool  # noqa: F401
from kbcurator.tools import account_status_tool  # noqa: F401
from kbcurator.tools import llm_router_tool  # noqa: F401
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
        "sso_login_user",
        "refresh_jwt_token",
        "query_rag",
        "upload_and_index_tool",
        "use_llm_provider",
        "query_llm_router_status",
        "test_llm_generation",
    ]

    async def dispatch(self, request: Request, call_next):
        # Make request available to tools via ContextVar
        request_var.set(request)

        # Skip auth for preflight
        if request.method.upper() == "OPTIONS":
            return await call_next(request)

        # Only protect the MCP HTTP endpoint (POST)
        if (
            request.url.path.startswith("/mcp")
            and request.method.upper() == "POST"
        ):
            # Parse body shallowly to infer the tool name without heavy ops
            try:
                body_bytes = await request.body()
                payload = (
                    json.loads(body_bytes.decode("utf-8"))
                    if body_bytes
                    else {}
                )
            except Exception:
                payload = {}

            tool_name = (
                payload.get("name")
                or (payload.get("params") or {}).get("name")
                or payload.get("tool")
                or payload.get("operation")
            )

            # If no tool name, let the request pass (MCP may reject appropriately later)
            if not tool_name:
                return await call_next(request)

            # Allow public tools without JWT
            if tool_name in self.PUBLIC_TOOLS:
                return await call_next(request)

            # Require JWT for any other tool
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
                claims = verify_token(token)
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
        return [self._normalize_origin(o) for o in raw.split(",") if o.strip()]

    def _normalize_origin(self, origin: str) -> str:
        # Browsers send Origin without trailing slash. Normalize env/config values.
        return origin.strip().rstrip("/").lower()

    def _bool_env(self, name: str, default: bool = False) -> bool:
        val = os.getenv(name, str(default)).strip().lower()
        return val in ("1", "true", "yes", "y", "t")

    async def dispatch(self, request: Request, call_next):
        # Handle preflight early; do NOT hit the router
        if request.method.upper() == "OPTIONS":
            response = PlainTextResponse("ok", status_code=200)
        else:
            response = await call_next(request)

        # ---------- Security Headers ----------
        # Content Security Policy (tune as needed)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self'; "
            "font-src 'self'; "
            "img-src 'self' data: https:; "
            "object-src 'none'; "
            "frame-ancestors 'none';"
        )
        # HTTP Strict Transport Security
        response.headers["Strict-Transport-Security"] = (
            "max-age=63072000; includeSubDomains; preload"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # ---------- CORS ----------
        allowed_origins = self._parse_allowed_origins()  # None => not set
        allow_credentials = self._bool_env("ALLOW_CREDENTIALS", default=False)

        request_origin_raw = request.headers.get("origin")

        request_origin = (
            self._normalize_origin(request_origin_raw)
            if request_origin_raw and request_origin_raw != "*"
            else None
        )
        if allowed_origins:
            # Strict allowlist
            if request_origin and (
                request_origin in allowed_origins or "*" in allowed_origins
            ):
                response.headers["Access-Control-Allow-Origin"] = request_origin_raw
                response.headers["Vary"] = "Origin"
                if allow_credentials:
                    response.headers["Access-Control-Allow-Credentials"] = "true"
                else:
                    response.headers["Access-Control-Allow-Credentials"] = "false"
            # else: no ACAO header, browser will block
        else:
            # No explicit allow list set:
            # - If credentials are allowed, reflect the origin (common pattern when you control the app).
            # - If not, allow any origin without credentials.
            if request_origin and allow_credentials:
                response.headers["Access-Control-Allow-Origin"] = (
                    request_origin_raw
                )
                response.headers["Vary"] = "Origin"
                response.headers["Access-Control-Allow-Credentials"] = "true"
            else:
                response.headers["Access-Control-Allow-Origin"] = "*"
                response.headers["Access-Control-Allow-Credentials"] = "false"

        response.headers["Access-Control-Allow-Methods"] = (
            "GET, POST, PUT, DELETE, OPTIONS, PATCH"
        )
        response.headers["Access-Control-Allow-Headers"] = (
            "Authorization, Content-Type, Accept, X-Requested-With, mcp-protocol-version"
        )
        response.headers["Access-Control-Expose-Headers"] = (
            "Authorization, Content-Type, Set-Cookie"
        )
        response.headers["Access-Control-Max-Age"] = (
            "600"  # cache preflight for 10 minutes
        )
        # Remove server header to hide server technology (VAPT requirement)
        if "Server" in response.headers:
            del response.headers["Server"]
        return response


custom_middleware = [
    Middleware(AuthMiddleware),
    Middleware(SecurityAndCORSMiddleware),
]


# ---------------------------
# MCP app + routes
# ---------------------------

# Create the base MCP Starlette app
base_app = mcp.http_app(
    transport="http",
    path="/mcp",
    middleware=custom_middleware,
    stateless_http=True,
)


# Health check (GET + OPTIONS)
async def health_check(request: Request):
    return JSONResponse({"status": "ok"})


base_app.add_route("/health", health_check, methods=["GET", "OPTIONS"])


# Optional root endpoint (GET + OPTIONS)
async def root(request: Request):
    return JSONResponse({"service": "mcp", "status": "ok"})


base_app.add_route("/", root, methods=["GET", "OPTIONS"])


# SSE-aware GET on /mcp to prevent reconnect storms
async def mcp_get(request: Request):
    accept = (request.headers.get("accept") or "").lower()

    if "text/event-stream" in accept:
        # Lightweight SSE stream to keep connection open and avoid reconnect storm
        async def stream():
            try:
                # Open the stream
                yield b": connected\n\n"
                while True:
                    # Stop if client disconnected
                    if await request.is_disconnected():
                        break
                    # Heartbeat every 15s
                    yield b": keep-alive\n\n"
                    await asyncio.sleep(15)
            except asyncio.CancelledError:
                # Server shutting down
                pass

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # Non-SSE callers get a simple JSON
    return JSONResponse(
        status_code=200,
        content={
            "endpoint": "/mcp",
            "methods": ["POST", "GET"],
            "status": "ok",
        },
        headers={"Cache-Control": "no-cache"},
    )


# Replace prior informational GET/OPTIONS with SSE-aware handler
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
                return

            elif message["type"] == "http.response.body":
                body_parts.append(message.get("body", b""))
                if not message.get("more_body", False):
                    full_body = b"".join(body_parts)
                    try:
                        request = request_var.get()
                        if request and getattr(
                            request.state, "refresh_token", None
                        ):
                            # Load cookie config
                            cookie_name = os.getenv(
                                "REFRESH_COOKIE_NAME", "refresh_token"
                            )
                            cookie_value_raw = getattr(
                                request.state, "refresh_token"
                            )
                            max_age = int(
                                getattr(
                                    request.state,
                                    "refresh_token_expires",
                                    86400,
                                )
                            )
                            same_site = os.getenv(
                                "REFRESH_COOKIE_SAMESITE", "None"
                            )  # None|Lax|Strict
                            secure_flag = self._bool_env(
                                "REFRESH_COOKIE_SECURE", default=True
                            )

                            # Build cookie
                            # Note: When sending cookies cross-site, you *must* use SameSite=None; Secure
                            cookie_attrs = [
                                f"{cookie_name}={cookie_value_raw}",
                                "Path=/",
                                f"Max-Age={max_age}",
                                "HttpOnly",
                            ]
                            if secure_flag:
                                cookie_attrs.append("Secure")
                            if same_site:
                                cookie_attrs.append(f"SameSite={same_site}")

                            cookie_header = "; ".join(cookie_attrs)
                            headers.append(
                                (b"set-cookie", cookie_header.encode("utf-8"))
                            )
                    except Exception:
                        # Never break the response flow due to cookie issues
                        pass

                    # Remove server header to hide server technology (VAPT requirement)
                    headers = [
                        (name, value)
                        for name, value in headers
                        if name.lower() != b"server"
                    ]

                    # Now send the actual response start + full body
                    await send(
                        {
                            "type": "http.response.start",
                            "status": status_code,
                            "headers": headers,
                        }
                    )
                    await send(
                        {
                            "type": "http.response.body",
                            "body": full_body,
                        }
                    )
            else:
                await send(message)

        await self.app(scope, receive, send_wrapper)


# Wrap the MCP app with the cookie layer
http_app = CookieWrapperApp(base_app)


# ---------------------------
# Server startup
# ---------------------------
if __name__ == "__main__":
    uvicorn.run(
        "kbcurator.server.main:http_app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["D:/forgex-backend/KnowledgeCurator/KnowledgeCurator/src"],
        log_level="info"
    )