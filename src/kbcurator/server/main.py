from dotenv import load_dotenv
import os


def _load_env_robust() -> None:
    """Load environment variables robustly for local + deployed runs.

    Problem: plain load_dotenv() depends on current working directory.
    In this repo, the server may be launched from /Users/md.4.ali/Documents/PolyRepo
    (or elsewhere), so KnowledgeCurator/.env is not guaranteed to be loaded.

    Fix: look for .env in common locations deterministically and load the first hit.
    """

    # 1) Respect already-provided process env (App Service, docker, shell exports).
    # But for local runs we DO want to load KnowledgeCurator/.env even if a parent
    # repo .env was already loaded earlier.

    candidates = []

    # a) Directory containing this file: .../KnowledgeCurator/src/kbcurator/server
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.abspath(os.path.join(here, "../../../../.env")))  # KnowledgeCurator/.env

    # b) Current working directory .env (some tools run from repo root)
    candidates.append(os.path.abspath(os.path.join(os.getcwd(), ".env")))

    # c) One level up from cwd (monorepo layouts)
    candidates.append(os.path.abspath(os.path.join(os.getcwd(), "../.env")))

    for path in candidates:
        if os.path.exists(path):
            # Override to ensure we don't accidentally keep stale values from an
            # unrelated .env loaded earlier in the process.
            load_dotenv(path, override=True)
            # Also propagate GLOBAL_* into process env even if blank elsewhere.
            return

    # Fall back to default behavior (no-op if no .env)
    load_dotenv(override=True)


_load_env_robust()

# DEBUG/ROBUSTNESS: ensure global workspace env is populated from KnowledgeCurator/.env.
# This guards against cases where other components cleared these keys after startup.
try:
    _kc_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.env"))
    if os.path.exists(_kc_env):
        _vals = {}
        try:
            from dotenv import dotenv_values
            _vals = dotenv_values(_kc_env) or {}
        except Exception:
            _vals = {}
        for _k in ("GLOBAL_WORKSPACE_ID", "GLOBAL_PUBLIC_WORKSPACE_ID"):
            if (not os.getenv(_k)) and _vals.get(_k):
                os.environ[_k] = str(_vals.get(_k))
except Exception:
    pass
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
from kbcurator.tools import sharepoint_agent 
from kbcurator.tools import config
from kbcurator.tools import trustai_tools
from kbcurator.tools import trustai_analytics_tools
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
        "test_llm_generation",
        "admin_configure_llm_provider",
        "admin_list_llm_providers",
        "admin_remove_llm_provider",
        "switch_llm_provider",
        "list_available_llm_providers",
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
            "frame-ancestors 'self' https://login.microsoftonline.com https://*.login.microsoftonline.com;"
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
            "Authorization, Content-Type, Accept, X-Requested-With, mcp-protocol-version, X-Skip-Auth"
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


# Workspace endpoint - validates user access by public_workspace_id
async def get_workspace(request: Request):
    from kbcurator.utils.db import db
    from sqlalchemy import func as sql_func
    from datetime import datetime, date
    from uuid import UUID
    
    def serialize_value(val):
        """Convert non-JSON-serializable types to strings."""
        if val is None:
            return None
        if isinstance(val, (datetime, date)):
            return val.isoformat()
        if isinstance(val, UUID):
            return str(val)
        return val
    
    def serialize_dict(d):
        """Serialize all values in a dict."""
        return {k: serialize_value(v) for k, v in d.items()}
    
    public_workspace_id = request.path_params.get("public_workspace_id")
    if not public_workspace_id:
        return JSONResponse({"error": "public_workspace_id is required"}, status_code=400)
    
    # Authenticate user - extract and verify JWT from headers
    token = extract_token_from_headers(dict(request.headers))
    if not token:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    
    try:
        claims = verify_token(token)
        user_id = claims.get("user_id") or claims.get("sub")
        if not user_id:
            return JSONResponse({"error": "Invalid token: no user_id"}, status_code=401)
    except Exception as e:
        return JSONResponse({"error": f"Invalid token: {str(e)}"}, status_code=401)
    
    session = db.Session()
    try:
        # Find workspace by public_workspace_id
        ws = session.query(db.Workspace).filter(
            db.Workspace.public_workspace_id == public_workspace_id,
            db.Workspace.is_active == True
        ).first()
        
        if not ws:
            return JSONResponse({"error": "Workspace not found"}, status_code=404)
        
        workspace_id = ws.workspace_id

        # Ensure we read global config robustly (dotenv or pydantic settings)
        global_ws_id = None
        try:
            raw_global_id = (os.getenv("GLOBAL_WORKSPACE_ID") or "").strip().strip('"').strip("'")
            global_ws_id = int(raw_global_id) if raw_global_id else None
        except Exception:
            global_ws_id = None

        if global_ws_id is None:
            try:
                from kbcurator.utils.config import settings
                global_ws_id = int(getattr(settings, 'GLOBAL_WORKSPACE_ID', None) or 0) or None
            except Exception:
                global_ws_id = None
        
        # Check if user has access to this workspace
        user_mapping = session.query(db.UserMap).filter(
            db.UserMap.user_id == user_id,
            db.UserMap.workspace_id == workspace_id,
            db.UserMap.is_active == True
        ).first()
        
        if not user_mapping:
            return JSONResponse({"error": "Access denied"}, status_code=403)
        
        # Build workspace info dict
        ws_info = serialize_dict({col: getattr(ws, col) for col in ws.__table__.columns.keys()})
        # Convert public_workspace_id to string
        if ws_info.get('public_workspace_id'):
            ws_info['public_workspace_id'] = str(ws_info['public_workspace_id'])
        
        # Category mapping
        categories = session.query(db.Category).filter(db.Category.is_active == True).all()
        cat_map = {str(c.category_id): c.category_name for c in categories}
        
        # Fetch industry/subindustry mapping and names
        ws_ind_map = session.query(db.WorkspaceIndustrySubIndustryMap).filter(
            db.WorkspaceIndustrySubIndustryMap.workspace_id == workspace_id,
            db.WorkspaceIndustrySubIndustryMap.is_active == True
        ).first()
        industry_id = subindustry_id = intent_id = industry_name = subindustry_name = None
        if ws_ind_map:
            industry_id = getattr(ws_ind_map, 'industry_id', None)
            subindustry_id = getattr(ws_ind_map, 'subindustry_id', None)
            intent_id = getattr(ws_ind_map, 'intent_id', None)
            if industry_id:
                industry_obj = session.query(db.Industry).filter(
                    db.Industry.industry_id == industry_id, db.Industry.is_active == True
                ).first()
                if industry_obj:
                    industry_name = getattr(industry_obj, 'industry_name', None)
            if subindustry_id:
                subindustry_obj = session.query(db.SubIndustry).filter(
                    db.SubIndustry.subindustry_id == subindustry_id, db.SubIndustry.is_active == True
                ).first()
                if subindustry_obj:
                    subindustry_name = getattr(subindustry_obj, 'subindustry_name', None)
        
        # Tools in workspace
        tool_maps = session.query(db.ToolMap).filter(
            db.ToolMap.workspace_id == workspace_id, db.ToolMap.is_active == True
        ).all()
        tool_map_dict = {tm.tool_id: tm for tm in tool_maps}
        tool_ids = list(tool_map_dict.keys())
        tools = []
        if tool_ids:
            tool_query = session.query(db.Tool).filter(db.Tool.tool_id.in_(tool_ids))
            if hasattr(db.Tool, 'is_active'):
                tool_query = tool_query.filter(db.Tool.is_active == True)
            for t in tool_query.all():
                tool_dict = serialize_dict({col: getattr(t, col) for col in t.__table__.columns.keys()})
                # Convert public_tool_id to string
                if tool_dict.get('public_tool_id'):
                    tool_dict['public_tool_id'] = str(tool_dict['public_tool_id'])
                tm = tool_map_dict.get(t.tool_id)
                last_updated_val = getattr(tm, 'last_updated', None) if tm else None
                tool_dict['last_updated'] = str(last_updated_val) if last_updated_val else None
                tool_dict['last_used'] = str(last_updated_val) if last_updated_val else None
                cat_ids = str(tool_dict.get('tool_category', '') or '').split(',')
                tool_dict['tool_category'] = [cat_map.get(cid.strip()) for cid in cat_ids if cid.strip() in cat_map]
                tools.append(tool_dict)
        
        # Agents in workspace
        agent_maps = session.query(db.AgentMap).filter(
            db.AgentMap.workspace_id == workspace_id, db.AgentMap.is_active == True
        ).all()
        agent_map_dict = {am.agent_id: am for am in agent_maps}
        agent_ids = list(agent_map_dict.keys())
        agents = []
        if agent_ids:
            agent_query = session.query(db.Agent).filter(db.Agent.agent_id.in_(agent_ids))
            if hasattr(db.Agent, 'is_active'):
                agent_query = agent_query.filter(db.Agent.is_active == True)
            for a in agent_query.all():
                agent_dict = serialize_dict({col: getattr(a, col) for col in a.__table__.columns.keys()})
                # Convert public_agent_id to string
                if agent_dict.get('public_agent_id'):
                    agent_dict['public_agent_id'] = str(agent_dict['public_agent_id'])
                am = agent_map_dict.get(a.agent_id)
                last_updated_val = getattr(am, 'last_updated', None) if am else None
                agent_dict['last_updated'] = str(last_updated_val) if last_updated_val else None
                agent_dict['last_used'] = str(last_updated_val) if last_updated_val else None
                cat_ids = str(agent_dict.get('agent_category', '') or '').split(',')
                agent_dict['agent_category'] = [cat_map.get(cid.strip()) for cid in cat_ids if cid.strip() in cat_map]
                agent_dict['type'] = 'agent'
                agents.append(agent_dict)
        
        # Users in workspace
        users = []
        user_data_query = (
            session.query(db.User, db.Role, db.UserMap)
            .join(db.UserMap, db.UserMap.user_id == db.User.user_id)
            .outerjoin(db.Role, (db.UserMap.role_id == db.Role.role_id) & (db.Role.is_active == True))
            .filter(db.UserMap.workspace_id == workspace_id, db.UserMap.is_active == True)
        )
        if hasattr(db.User, 'is_active'):
            user_data_query = user_data_query.filter(db.User.is_active == True)
        for user, role, user_map in user_data_query.all():
            user_dict = serialize_dict({col: getattr(user, col) for col in user.__table__.columns.keys()})
            user_dict['role'] = getattr(role, 'role_name', None) if role else None
            user_dict['role_id'] = getattr(user_map, 'role_id', None)
            user_dict['permissions'] = getattr(user_map, 'permissions', None)
            user_dict['can_curate_kb'] = getattr(user_map, 'can_curate_kb', None)
            users.append(user_dict)
        
        # Knowledge bases
        knowledge_bases = []
        if industry_name and subindustry_name:
            industry_obj = session.query(db.Industry).filter(
                sql_func.lower(db.Industry.industry_name) == industry_name.strip().lower(),
                db.Industry.is_active == True
            ).first()
            subindustry_obj = session.query(db.SubIndustry).filter(
                sql_func.lower(db.SubIndustry.subindustry_name) == subindustry_name.strip().lower(),
                db.SubIndustry.is_active == True
            ).first()
            if industry_obj and subindustry_obj:
                kb_query = session.query(db.WorkspaceIndustrySubIndustryMap).filter(
                    db.WorkspaceIndustrySubIndustryMap.industry_id == industry_obj.industry_id,
                    db.WorkspaceIndustrySubIndustryMap.subindustry_id == subindustry_obj.subindustry_id,
                    db.WorkspaceIndustrySubIndustryMap.workspace_id == workspace_id,
                    db.WorkspaceIndustrySubIndustryMap.is_active == True
                )
                for row in kb_query.all():
                    if row.kb_id:
                        kb_obj = session.query(db.KnowledgeBase).filter(
                            db.KnowledgeBase.id == row.kb_id, db.KnowledgeBase.is_active == True
                        ).first()
                        if kb_obj:
                            knowledge_bases.append({
                                'id': getattr(kb_obj, 'id', None),
                                'title': getattr(kb_obj, 'title', None),
                                'description': getattr(kb_obj, 'description', None)
                            })
        
        return JSONResponse({
            "workspace": ws_info,
            "industry": industry_id,
            "industry_name": industry_name,
            "subindustry": subindustry_id,
            "subindustry_name": subindustry_name,
            "intent": intent_id,
            "tools": tools,
            "agents": agents,
            "users": users,
            "knowledge_bases": knowledge_bases
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        session.close()


base_app.add_route("/workspaces/{public_workspace_id}", get_workspace, methods=["GET"])


# Bookmark endpoint - validates and returns workspace/agent/tool details by public IDs
async def get_bookmark(request: Request):
    from kbcurator.utils.db import db
    from sqlalchemy import func as sql_func
    from datetime import datetime, date
    from uuid import UUID
    
    def serialize_value(val):
        """Convert non-JSON-serializable types to strings."""
        if val is None:
            return None
        if isinstance(val, (datetime, date)):
            return val.isoformat()
        if isinstance(val, UUID):
            return str(val)
        return val
    
    def serialize_dict(d):
        """Serialize all values in a dict."""
        return {k: serialize_value(v) for k, v in d.items()}
    
    # Get query params
    public_workspace_id = request.query_params.get("wgid")
    public_agent_id = request.query_params.get("agid")
    public_tool_id = request.query_params.get("tgid")
    
    if not public_workspace_id:
        return JSONResponse({"error": "wgid (public_workspace_id) is required"}, status_code=400)
    
    # Authenticate user
    token = extract_token_from_headers(dict(request.headers))
    if not token:
        return JSONResponse({"error": "Authentication required"}, status_code=401)
    
    try:
        claims = verify_token(token)
        user_id = claims.get("user_id") or claims.get("sub")
        if not user_id:
            return JSONResponse({"error": "Invalid token: no user_id"}, status_code=401)
        try:
            user_id = int(user_id)
        except Exception:
            return JSONResponse({"error": "Invalid token: user_id must be an integer"}, status_code=401)
    except Exception as e:
        return JSONResponse({"error": f"Invalid token: {str(e)}"}, status_code=401)
    
    session = db.Session()
    try:
        # Find workspace by public_workspace_id
        ws = session.query(db.Workspace).filter(
            db.Workspace.public_workspace_id == public_workspace_id,
            db.Workspace.is_active == True
        ).first()
        
        if not ws:
            return JSONResponse({"error": "Workspace not found"}, status_code=404)
        
        workspace_id = ws.workspace_id

        # Resolve global workspace id (for auto-provision access)
        global_ws_id = None
        try:
            raw = (os.getenv("GLOBAL_WORKSPACE_ID") or "").strip().strip('"').strip("'")
            global_ws_id = int(raw) if raw else None
        except Exception:
            global_ws_id = None
        if global_ws_id is None:
            try:
                from kbcurator.utils.config import settings
                if getattr(settings, 'GLOBAL_WORKSPACE_ID', None) is not None:
                    global_ws_id = int(settings.GLOBAL_WORKSPACE_ID)
            except Exception:
                global_ws_id = None

        # Global workspace is intended to be accessible to all authenticated users.
        # If the user is missing a mapping row, create it lazily (idempotent insert).
        
        # Check if user has access to this workspace
        user_mapping = session.query(db.UserMap).filter(
            db.UserMap.user_id == user_id,
            db.UserMap.workspace_id == workspace_id,
            db.UserMap.is_active == True
        ).first()

        if not user_mapping:
            # For the configured global workspace, auto-provision access.
            if global_ws_id is not None and int(workspace_id) == int(global_ws_id):
                try:
                    # Use the shared helper used by SSO/login flows.
                    from kbcurator.utils.auth import _assign_user_to_workspace
                    _assign_user_to_workspace(user_id, int(workspace_id))

                    # Re-check to avoid false positives if insert failed or hit unexpected constraints.
                    user_mapping = session.query(db.UserMap).filter(
                        db.UserMap.user_id == user_id,
                        db.UserMap.workspace_id == workspace_id,
                        db.UserMap.is_active == True
                    ).first()
                except Exception as exc:
                    # Surface the real cause to logs to debug DB constraint/permission issues.
                    logger.exception(
                        "Global workspace auto-mapping failed user_id=%s workspace_id=%s: %s",
                        user_id,
                        workspace_id,
                        exc,
                    )
                    user_mapping = None

            if not user_mapping:
                return JSONResponse({"error": "Access denied to workspace"}, status_code=403)
        
        # Category mapping
        categories = session.query(db.Category).filter(db.Category.is_active == True).all()
        cat_map = {str(c.category_id): c.category_name for c in categories}
        
        result = {
            "workspace": {
                "workspace_id": workspace_id,
                "public_workspace_id": str(ws.public_workspace_id) if ws.public_workspace_id else None,
                "workspace_name": ws.workspace_name,
                "workspace_desc": ws.workspace_desc
            },
            "agent": None,
            "tool": None
        }
        
        # Validate and fetch agent if agid provided
        if public_agent_id:
            agent = session.query(db.Agent).filter(
                db.Agent.public_agent_id == public_agent_id,
                db.Agent.is_active == True
            ).first()
            
            if not agent:
                return JSONResponse({"error": "Agent not found"}, status_code=404)
            
            # Verify agent is mapped to this workspace
            agent_mapping = session.query(db.AgentMap).filter(
                db.AgentMap.workspace_id == workspace_id,
                db.AgentMap.agent_id == agent.agent_id,
                db.AgentMap.is_active == True
            ).first()
            
            if not agent_mapping:
                return JSONResponse({"error": "Agent not mapped to this workspace"}, status_code=403)
            
            agent_dict = serialize_dict({col: getattr(agent, col) for col in agent.__table__.columns.keys()})
            if agent_dict.get('public_agent_id'):
                agent_dict['public_agent_id'] = str(agent_dict['public_agent_id'])
            cat_ids = str(agent_dict.get('agent_category', '') or '').split(',')
            agent_dict['agent_category'] = [cat_map.get(cid.strip()) for cid in cat_ids if cid.strip() in cat_map]
            agent_dict['type'] = 'agent'
            result["agent"] = agent_dict
        
        # Validate and fetch tool if tgid provided
        if public_tool_id:
            tool = session.query(db.Tool).filter(
                db.Tool.public_tool_id == public_tool_id,
                db.Tool.is_active == True
            ).first()
            
            if not tool:
                return JSONResponse({"error": "Tool not found"}, status_code=404)
            
            # Verify tool is mapped to this workspace
            tool_mapping = session.query(db.ToolMap).filter(
                db.ToolMap.workspace_id == workspace_id,
                db.ToolMap.tool_id == tool.tool_id,
                db.ToolMap.is_active == True
            ).first()
            
            if not tool_mapping:
                return JSONResponse({"error": "Tool not mapped to this workspace"}, status_code=403)
            
            tool_dict = serialize_dict({col: getattr(tool, col) for col in tool.__table__.columns.keys()})
            if tool_dict.get('public_tool_id'):
                tool_dict['public_tool_id'] = str(tool_dict['public_tool_id'])
            cat_ids = str(tool_dict.get('tool_category', '') or '').split(',')
            tool_dict['tool_category'] = [cat_map.get(cid.strip()) for cid in cat_ids if cid.strip() in cat_map]
            tool_dict['type'] = 'tool'
            result["tool"] = tool_dict
        
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
    finally:
        session.close()


base_app.add_route("/bookmark/", get_bookmark, methods=["GET"])


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
