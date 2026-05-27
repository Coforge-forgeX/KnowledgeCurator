# import os
# import time
# import uuid
# from typing import Any, Dict, Optional,Tuple
# from functools import lru_cache
# import jwt
# from dotenv import load_dotenv
# import redis
# import sqlite3

# load_dotenv()

# # JWT configuration
# JWT_SECRET = os.getenv("JWT_SECRET", "change-this-in-prod")
# JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
# ACCESS_TOKEN_EXPIRY = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRY_SECONDS", "3600"))  # 60 minutes default
# REFRESH_TOKEN_EXPIRY = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRY_SECONDS", "86400"))  # 24 hours default

# # Redis connection
# redis_client = redis.StrictRedis(host='localhost', port=6379, db=0, decode_responses=True)

# # SQLite database for token revocation
# conn = sqlite3.connect(':memory:', check_same_thread=False)  # Use ':memory:' for in-memory database or a file path for persistent storage
# cursor = conn.cursor()

# # Create a table to store revoked tokens
# cursor.execute('''
# CREATE TABLE IF NOT EXISTS revoked_tokens (
#     jti TEXT PRIMARY KEY,
#     exp INTEGER
# )
# ''')
# conn.commit()

# # Lock for thread safety
# _rev_lock = RLock()

# def is_token_revoked(jti: Optional[str]) -> bool:
#     """Return True if the token JTI is marked revoked."""
#     if not jti:
#         return False
#     _purge_revoked()
#     cursor.execute("SELECT 1 FROM revoked_tokens WHERE jti = ?", (jti,))
#     return cursor.fetchone() is not None

# def revoke_token(token: str) -> Tuple[bool, str]:
#     """
#     Revoke a JWT (access or refresh). Returns (revoked, message).
#     - Decodes ignoring exp to extract jti/exp reliably.
#     - Stores jti until original exp so verification denies it thereafter.
#     - Accepts either raw JWT or Base64URL-wrapped JWT.
#     """
#     try:
#         # Decode the transported token (Base64URL-wrapped or raw JWT)
#         token = maybe_decode_transported_token(token)

#         payload = jwt.decode(
#             token,
#             JWT_SECRET,
#             algorithms=[JWT_ALGORITHM],
#             options={"verify_exp": False},
#         )
#         jti = payload.get("jti")
#         exp = int(payload.get("exp") or (int(time.time()) + ACCESS_TOKEN_EXPIRY))

#         if not jti:
#             return False, "Token missing jti; cannot revoke deterministically"

#         with _rev_lock:
#             cursor.execute("INSERT OR REPLACE INTO revoked_tokens (jti, exp) VALUES (?, ?)", (jti, exp))
#             conn.commit()

#         _cached_jwt_decode.cache_clear()
#         return True, "Token revoked"
#     except jwt.InvalidTokenError as e:
#         _cached_jwt_decode.cache_clear()
#         return False, f"Invalid token: {str(e)}"
#     except Exception as e:
#         _cached_jwt_decode.cache_clear()
#         return False, f"Error revoking token: {str(e)}"

# def _purge_revoked():
#     """Purge expired entries from the SQLite database."""
#     now = int(time.time())
#     cursor.execute("DELETE FROM revoked_tokens WHERE exp < ?", (now,))
#     conn.commit()

# @lru_cache(maxsize=1000)
# def _cached_jwt_decode(token: str, secret: str, algorithm: str) -> Dict:
#     """
#     Cached JWT decode to avoid re-decoding the same token multiple times.
#     Cache is based on token string, secret, and algorithm.
#     """
#     return jwt.decode(token, secret, algorithms=[algorithm])

# def verify_jwt_token(token: str) -> Dict:
#     """
#     Verify a JWT and return its payload. Raises jwt exceptions on failure.
#     Uses LRU cache to avoid re-decoding valid tokens.
#     Also enforces revocation via in-memory denylist.
#     Accepts either raw JWT or Base64URL-wrapped JWT.
#     """
#     try:
#         token = maybe_decode_transported_token(token)

#         payload = _cached_jwt_decode(token, JWT_SECRET, JWT_ALGORITHM)

#         exp = payload.get("exp")
#         if exp and int(time.time()) >= exp:
#             _cached_jwt_decode.cache_clear()
#             raise jwt.ExpiredSignatureError("Token has expired")

#         if is_token_revoked(payload.get("jti")):
#             _cached_jwt_decode.cache_clear()
#             raise jwt.InvalidTokenError("Token has been revoked")

#         return payload

#     except jwt.ExpiredSignatureError:
#         _cached_jwt_decode.cache_clear()
#         raise
#     except Exception:
#         _cached_jwt_decode.cache_clear()
#         raise

# def verify_refresh_token(token: str) -> Dict:
#     """
#     Verify and decode a refresh token specifically.
#     Accepts either raw or Base64URL-wrapped token.
#     """
#     try:
#         token = maybe_decode_transported_token(token)

#         payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

#         if is_token_revoked(payload.get("jti")):
#             raise Exception("Refresh token has been revoked")

#         if payload.get("token_type") != "refresh":
#             raise Exception("Invalid token type. Expected refresh token.")

#         return payload

#     except jwt.ExpiredSignatureError:
#         raise Exception("Refresh token has expired")
#     except jwt.InvalidTokenError:
#         raise Exception("Invalid refresh token")

# def extract_token_from_headers(headers: Dict) -> Optional[str]:
#     """
#     Get token from Authorization: Bearer <token> or 'token' header.
#     Returns None if not present. (Transport-decoding is done in verify_* functions.)
#     """
#     auth = headers.get("authorization") or headers.get("Authorization") or ""
#     if isinstance(auth, str) and auth.lower().startswith("bearer "):
#         return auth.split(" ", 1)[1].strip()
#     alt = headers.get("token") or headers.get("Token")
#     return alt

# auth.py
import os
import time
import uuid
import base64
import ssl
import logging
from typing import Any, Dict, Optional, Tuple
from functools import lru_cache, wraps
from datetime import datetime, timedelta, timezone
import jwt
import redis
import psycopg2
import psycopg2.extras

# Centralized config and enums
from .config import settings

from .constants import Role
from .request_context import request_var


logger = logging.getLogger(__name__)


# Use centralized config
POSTGRESQL_HOST = settings.POSTGRES_HOST
POSTGRESQL_PORT = settings.POSTGRES_PORT
POSTGRESQL_DB = settings.POSTGRES_DB
POSTGRESQL_USER = settings.POSTGRES_USER
POSTGRESQL_PASSWORD = settings.POSTGRES_PASSWORD

JWT_SECRET = settings.JWT_SECRET
JWT_ALGORITHM = settings.JWT_ALGORITHM
ACCESS_TOKEN_EXPIRY = settings.JWT_ACCESS_TOKEN_EXPIRY_SECONDS
REFRESH_TOKEN_EXPIRY = settings.JWT_REFRESH_TOKEN_EXPIRY_SECONDS

JWT_TRANSPORT_ENCODE = settings.JWT_TRANSPORT_ENCODE
JWT_SET_ACCESS_COOKIE = settings.JWT_SET_ACCESS_COOKIE
JWT_RETURN_RAW_ACCESS = settings.JWT_RETURN_RAW_ACCESS

REDIS_HOST = settings.REDIS_HOST
REDIS_PORT = settings.REDIS_PORT
REDIS_PASSWORD = settings.REDIS_PASSWORD
REDIS_SSL = settings.REDIS_SSL

# Initialize Redis client with proper configuration for Azure Redis
try:
    if REDIS_SSL and REDIS_HOST != "localhost":
        # Azure Redis requires SSL
        redis_client = redis.StrictRedis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            db=0,
            decode_responses=True,
            ssl=True,
            ssl_cert_reqs=None,  # Disable cert verification for Azure Redis
            socket_connect_timeout=10,
            socket_timeout=10
        )
    else:
        # Local Redis without SSL
        redis_client = redis.StrictRedis(
            host=REDIS_HOST,
            port=REDIS_PORT,
            password=REDIS_PASSWORD,
            db=0,
            decode_responses=True,
            socket_connect_timeout=5,
            socket_timeout=5
        )
    
    # Test connection
    redis_client.ping()
    print(f"[INFO] Successfully connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
    REDIS_AVAILABLE = True
except Exception as e:
    print(f"[ERROR] Failed to connect to Redis at {REDIS_HOST}:{REDIS_PORT}: {e}")
    print("[WARNING] Token revocation will not work properly without Redis!")
    print("[WARNING] SECURITY RISK: Logout will not properly revoke tokens!")
    redis_client = None
    REDIS_AVAILABLE = False

# Redis key prefix for revoked tokens
REVOKED_TOKEN_PREFIX = "revoked_token:"

# =============================================================================
# Authentication Decorator
# =============================================================================
def require_auth(func):
    """
    Decorator that validates JWT authentication via middleware-injected claims.
    
    Usage:
        @require_auth
        def my_tool_function(...):
            ...
    
    The middleware extracts and validates the JWT, injecting claims into
    request.state.jwt_claims. This decorator checks that claims exist,
    meaning the token was valid and not expired.
    
    If authentication fails, returns an error dict:
        {"error": "Unauthorized: <message>"}
    
    For async functions, use @require_auth_async instead.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        request = request_var.get(None)
        if not request:
            return {"error": "Unauthorized: No request context available"}
        
        claims = getattr(request.state, "jwt_claims", None)
        if not claims:
            return {"error": "Unauthorized: Invalid or expired token"}
        
        return func(*args, **kwargs)
    return wrapper


def require_auth_async(func):
    """
    Async version of require_auth decorator for async tool functions.
    
    Usage:
        @require_auth_async
        async def my_async_tool_function(...):
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        request = request_var.get(None)
        if not request:
            return {"error": "Unauthorized: No request context available"}
        
        claims = getattr(request.state, "jwt_claims", None)
        if not claims:
            return {"error": "Unauthorized: Invalid or expired token"}
        
        return await func(*args, **kwargs)
    return wrapper


def get_current_user():
    """
    Helper to get current user claims from request context.
    Returns (claims, user_id) tuple or (None, None) if not authenticated.
    
    Usage:
        claims, user_id = get_current_user()
        if not claims:
            return {"error": "Unauthorized"}
    """
    request = request_var.get(None)
    if not request:
        return None, None
    
    claims = getattr(request.state, "jwt_claims", None)
    if not claims:
        return None, None
    
    user_id = claims.get("user_id") or claims.get("sub")
    return claims, user_id


def is_token_revoked(jti: Optional[str]) -> bool:
    """
    Return True if the token JTI is marked revoked.
    Uses Redis for shared state across all worker processes.
    """
    if not jti:
        return False
    
    if not REDIS_AVAILABLE or redis_client is None:
        print(f"[WARNING] Redis unavailable - cannot check token revocation for JTI {jti}")
        # Fail open in development, but log the security issue
        return False
    
    try:
        # Check if the JTI exists in Redis
        key = f"{REVOKED_TOKEN_PREFIX}{jti}"
        is_revoked = redis_client.exists(key) > 0
        if is_revoked:
            print(f"[INFO] Token JTI {jti} is revoked")
        return is_revoked
    except Exception as e:
        print(f"[ERROR] Redis check failed for JTI {jti}: {e}")
        # Fail open to avoid blocking valid users if Redis has issues
        return False

def revoke_token(token: str) -> Tuple[bool, str]:
    """
    Revoke a JWT (access or refresh). Returns (revoked, message).
    - Decodes ignoring exp to extract jti/exp reliably.
    - Stores jti in Redis with TTL equal to original token expiry.
    - This ensures revocation works across all worker processes.
    - Accepts either raw JWT or Base64URL-wrapped JWT.
    """
    if not REDIS_AVAILABLE or redis_client is None:
        print("[ERROR] Redis unavailable - cannot revoke token")
        return False, "Token revocation service unavailable (Redis not connected)"
    
    try:
        # Decode the transported token (Base64URL-wrapped or raw JWT)
        token = maybe_decode_transported_token(token)

        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM],
            options={"verify_exp": False},
        )
        jti = payload.get("jti")
        exp = int(payload.get("exp") or (int(time.time()) + ACCESS_TOKEN_EXPIRY))

        if not jti:
            return False, "Token missing jti; cannot revoke deterministically"

        # Calculate TTL (time until token expires naturally)
        now = int(time.time())
        ttl = max(exp - now, 1)  # At least 1 second TTL

        # Store in Redis with expiration
        key = f"{REVOKED_TOKEN_PREFIX}{jti}"
        try:
            redis_client.setex(key, ttl, "revoked")
            print(f"[SUCCESS] ✓ Token JTI {jti} revoked in Redis with TTL {ttl}s (expires at {exp})")
        except Exception as redis_err:
            print(f"[ERROR] Failed to revoke token in Redis: {redis_err}")
            return False, f"Redis error: {str(redis_err)}"

        # Clear JWT decode cache to force re-validation
        _cached_jwt_decode.cache_clear()
        return True, "Token revoked successfully"
        
    except jwt.InvalidTokenError as e:
        _cached_jwt_decode.cache_clear()
        return False, f"Invalid token: {str(e)}"
    except Exception as e:
        _cached_jwt_decode.cache_clear()
        return False, f"Error revoking token: {str(e)}"

# =============================================================================
# Transport helpers
# =============================================================================
def encode_for_transport(token: str) -> str:
    """
    Wrap the JWT in Base64URL encoding for transport (obfuscation, not encryption).
    """
    return base64.urlsafe_b64encode(token.encode("utf-8")).decode("ascii")

def maybe_decode_transported_token(token: str) -> str:
    """
    Accept either a raw JWT or a Base64URL-wrapped JWT.
    If decoding yields a string with 2 dots (header.payload.signature),
    assume it's a wrapped JWT; otherwise return original.
    """
    if not token or not isinstance(token, str):
        return token
    # If it already looks like a JWT (header.payload.signature), keep it
    if token.count(".") == 2:
        return token
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
        if raw.count(".") == 2:
            return raw
    except Exception:
        pass
    return token

# =============================================================================
# Token creation
# =============================================================================
def create_jwt_token(claims: Dict, expires_in: Optional[int] = None) -> Tuple[str, int]:
    """
    Create a signed JWT access token.
    Returns: (token_string, expiry_seconds_from_now)
    """
    now = int(time.time())
    exp = now + int(expires_in or ACCESS_TOKEN_EXPIRY)
    payload = dict(claims)

    if "sub" in payload and payload["sub"] is not None:
        payload["sub"] = str(payload["sub"])  # sub MUST be a string

    payload["jti"] = uuid.uuid4().hex
    payload["iat"] = now
    payload["exp"] = exp
    payload["token_type"] = "access"

    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return token, exp - now

def create_refresh_token(user_id: int) -> Tuple[str, int]:
    """
    Create a JWT refresh token with 24 hour expiry.
    Returns: (refresh_token_string, expiry_seconds)
    """
    now = int(time.time())
    exp = now + REFRESH_TOKEN_EXPIRY
    payload = {
        "user_id": str(user_id),
        "sub": str(user_id),
        "token_type": "refresh",
        "jti": uuid.uuid4().hex,
        "iat": now,
        "exp": exp,
    }
    refresh_token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return refresh_token, REFRESH_TOKEN_EXPIRY

# =============================================================================
# Cached decode (performance optimization)
# =============================================================================
@lru_cache(maxsize=1000)
def _cached_jwt_decode(token: str, secret: str, algorithm: str) -> Dict:
    """Cached JWT decode to avoid re-decoding the same token multiple times."""
    return jwt.decode(token, secret, algorithms=[algorithm])

# =============================================================================
# Verification
# =============================================================================
def verify_jwt_token(token: str) -> Dict:
    """
    Verify a JWT and return its payload. Raises jwt exceptions on failure.
    Uses LRU cache to avoid re-decoding valid tokens.
    Also enforces revocation via in-memory denylist.
    Accepts either raw JWT or Base64URL-wrapped JWT.
    """
    try:
        token = maybe_decode_transported_token(token)

        payload = _cached_jwt_decode(token, JWT_SECRET, JWT_ALGORITHM)

        exp = payload.get("exp")
        if exp and int(time.time()) >= exp:
            _cached_jwt_decode.cache_clear()
            raise jwt.ExpiredSignatureError("Token has expired")

        if is_token_revoked(payload.get("jti")):
            _cached_jwt_decode.cache_clear()
            raise jwt.InvalidTokenError("Token has been revoked")

        return payload

    except jwt.ExpiredSignatureError:
        _cached_jwt_decode.cache_clear()
        raise
    except Exception:
        _cached_jwt_decode.cache_clear()
        raise

def verify_refresh_token(token: str) -> Dict:
    """
    Verify and decode a refresh token specifically.
    Accepts either raw or Base64URL-wrapped token.
    """
    try:
        token = maybe_decode_transported_token(token)

        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])

        if is_token_revoked(payload.get("jti")):
            raise Exception("Refresh token has been revoked")

        if payload.get("token_type") != "refresh":
            raise Exception("Invalid token type. Expected refresh token.")

        return payload

    except jwt.ExpiredSignatureError:
        raise Exception("Refresh token has expired")
    except jwt.InvalidTokenError:
        raise Exception("Invalid refresh token")

# =============================================================================
# Helpers
# =============================================================================
def extract_token_from_headers(headers: Dict) -> Optional[str]:
    """
    Get token from Authorization: Bearer <token> or 'token' header.
    Returns None if not present. (Transport-decoding is done in verify_* functions.)
    """
    auth = headers.get("authorization") or headers.get("Authorization") or ""
    if isinstance(auth, str) and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    alt = headers.get("token") or headers.get("Token")
    return alt


# =============================================================================
# DB helpers (PostgreSQL) — used by sso_login_tool
# =============================================================================
def _get_pg_conn():
    """Open a short-lived PostgreSQL connection."""
    return psycopg2.connect(
        host=POSTGRESQL_HOST,
        port=int(POSTGRESQL_PORT),
        dbname=POSTGRESQL_DB,
        user=POSTGRESQL_USER,
        password=POSTGRESQL_PASSWORD,
        connect_timeout=5,
        sslmode="require",
        cursor_factory=psycopg2.extras.RealDictCursor,
    )


def _fetch_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """
    Fetch active user row from public.users by email_id (case-insensitive).
    This matches the actual table present in the database.
    """
    with _get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM public.users WHERE LOWER(email_id) = %s AND is_active = TRUE",
                (email.strip().lower(),),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def _fetch_user_by_id(user_id: int) -> Optional[Dict[str, Any]]:
    """Fetch active user row from public.users by user_id."""
    with _get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM public.users WHERE user_id = %s AND is_active = TRUE",
                (user_id,),
            )
            row = cur.fetchone()
            return dict(row) if row else None


def _fetch_user_roles(user_id: int) -> list:
    """Fetch user's active roles across all workspaces."""
    with _get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    urm.workspace_id,
                    urm.role_id,
                    rm.role_name AS workflow_stage
                FROM public.workspace_users_mapping urm
                JOIN public.role_master rm ON rm.role_id = urm.role_id
                WHERE urm.user_id = %s AND urm.is_active = TRUE
                ORDER BY urm.workspace_id
                """,
                (user_id,),
            )
            return [dict(r) for r in cur.fetchall()]


def _fetch_user_workspaces(user_id: int) -> list:
    """Fetch all active workspaces the user belongs to."""
    with _get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT wm.workspace_id, wm.workspace_name, wm.workspace_desc
                FROM public.workspace_master wm
                JOIN public.workspace_users_mapping wum
                  ON wum.workspace_id = wm.workspace_id
                WHERE wum.user_id = %s
                  AND wum.is_active = TRUE
                  AND wm.is_active = TRUE
                ORDER BY wm.workspace_id
                """,
                (user_id,),
            )
            return [dict(r) for r in cur.fetchall()]


def _issue_backend_jwt(user: Dict[str, Any]) -> str:
    """
    Issue a backend-signed JWT (HS256, b64url-encoded for transport).
    Includes jti so the token can be individually revoked via Redis.
    """
    now = datetime.now(timezone.utc)
    role_id = user.get('role_id', None)
    payload = {
        "user_id": user["user_id"],
        "sub": str(user["user_id"]),
        "email": user["email_id"],
        "is_admin": True if (role_id == Role.ADMIN.id) else False,
        "role_id": role_id,
        "jti": uuid.uuid4().hex,
        "token_type": "access",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(seconds=ACCESS_TOKEN_EXPIRY)).timestamp()),
    }
    raw_token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return base64.urlsafe_b64encode(raw_token.encode()).decode()


def _serialize_user(user: Dict[str, Any]) -> Dict[str, Any]:
    """Remove sensitive fields and make datetime values JSON-safe."""
    safe = {k: v for k, v in user.items() if k != "password"}
    for k, v in safe.items():
        if hasattr(v, "isoformat"):
            safe[k] = v.isoformat()
    return safe


def _assign_user_to_workspace(user_id: int, workspace_id: int) -> None:
    """
    Insert a workspace_users_mapping row for an existing user.
    Idempotent — silently skips if the mapping already exists.
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    with _get_pg_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.workspace_users_mapping
                    (workspace_id, user_id, is_active, created_date, last_updated)
                VALUES (%s, %s, TRUE, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (workspace_id, user_id, now, now),
            )
            conn.commit()


def _create_user_with_workspace(email: str, workspace_id: int) -> Dict[str, Any]:
    """
    Create a new user row and assign them to workspace_id in a single transaction.
    Returns the newly created user dict.
    Raises on any database error (caller must handle).
    """
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)

    with _get_pg_conn() as conn:
        with conn.cursor() as cur:
            # Create user
            cur.execute(
                """
                INSERT INTO public.users (namespace, email_id, is_active, is_admin, created_date, last_updated, role_id)
                VALUES (%s, %s, TRUE, FALSE, %s, %s, 1)
                RETURNING *
                """,
                ("default", email.strip().lower(), now, now),
            )
            user = dict(cur.fetchone())

            # Assign to dummy workspace — idempotent so re-runs are safe
            cur.execute(
                """
                INSERT INTO public.workspace_users_mapping
                    (workspace_id, user_id, is_active, created_date, last_updated)
                VALUES (%s, %s, TRUE, %s, %s)
                ON CONFLICT DO NOTHING
                """,
                (workspace_id, user["user_id"], now, now),
            )
            conn.commit()

    return user