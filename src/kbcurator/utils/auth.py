# import os
# import time
# import uuid
# from typing import Any, Dict, Optional,Tuple
# from functools import lru_cache
# import jwt
# from dotenv import load_dotenv

# load_dotenv()

# # JWT configuration
# JWT_SECRET = os.getenv("JWT_SECRET", "change-this-in-prod")
# JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
# ACCESS_TOKEN_EXPIRY = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRY_SECONDS", "3600"))  # 60 minutes default
# REFRESH_TOKEN_EXPIRY = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRY_SECONDS", "86400"))  # 24 hours default

# def create_jwt_token(claims: Dict, expires_in: int | None = None) -> Tuple[str, int]:
#     """
#     Create a signed JWT access token.

#     claims: base claims to include in the token (e.g., sub, email, name, roles)
#     expires_in: optional custom expiry in seconds; defaults to ACCESS_TOKEN_EXPIRY
#     """
#     now = int(time.time())
#     exp = now + int(expires_in or ACCESS_TOKEN_EXPIRY)
#     payload = dict(claims)
#     # Ensure RFC-compliant types
#     if "sub" in payload and payload["sub"] is not None:
#         payload["sub"] = str(payload["sub"])  # sub MUST be a string
#     # Provide a unique token identifier to avoid identical tokens when claims/time align
#     payload["jti"] = uuid.uuid4().hex
#     payload["iat"] = now
#     payload["exp"] = exp
#     payload["token_type"] = "access"
#     token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
#     return token, exp - now


# def create_refresh_token(user_id: int) -> Tuple[str, int]:
#     """
#     Create a JWT refresh token with 24 hour expiry.
#     Args:
#         user_id (int): User ID to include in token
#     Returns:
#         tuple: (refresh_token_string, expiry_seconds)
#     """
#     now = int(time.time())
#     exp = now + REFRESH_TOKEN_EXPIRY
#     payload = {
#         'user_id': str(user_id),
#         'sub': str(user_id),
#         'token_type': 'refresh',
#         'jti': uuid.uuid4().hex,
#         'iat': now,
#         'exp': exp
#     }
    
#     refresh_token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
#     return refresh_token, REFRESH_TOKEN_EXPIRY


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
#     """
#     try:
#         payload = _cached_jwt_decode(token, JWT_SECRET, JWT_ALGORITHM)
#         # Check expiry manually since cache doesn't know about time
#         exp = payload.get('exp')
#         if exp and int(time.time()) >= exp:
#             # Token expired, clear from cache and raise
#             _cached_jwt_decode.cache_clear()
#             raise jwt.ExpiredSignatureError("Token has expired")
#         return payload
#     except jwt.ExpiredSignatureError:
#         # Clear cache on expiry to prevent stale tokens
#         _cached_jwt_decode.cache_clear()
#         raise
#     except Exception:
#         # Clear cache on any error
#         _cached_jwt_decode.cache_clear()
#         raise


# def verify_refresh_token(token: str) -> Dict:
#     """
#     Verify and decode a refresh token specifically.
#     Args:
#         token (str): JWT refresh token string
#     Returns:
#         dict: Decoded token payload
#     Raises:
#         Exception: If token is invalid or not a refresh token
#     """
#     try:
#         payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        
#         # Ensure it's a refresh token
#         if payload.get('token_type') != 'refresh':
#             raise Exception("Invalid token type. Expected refresh token.")
        
#         return payload
#     except jwt.ExpiredSignatureError:
#         raise Exception("Refresh token has expired")
#     except jwt.InvalidTokenError:
#         raise Exception("Invalid refresh token")


# def extract_token_from_headers(headers: Dict) -> str | None:
#     """
#     Get token from Authorization: Bearer <token> or 'token' header.
#     Returns None if not present.
#     """
#     auth = headers.get("authorization") or headers.get("Authorization") or ""
#     if auth.lower().startswith("bearer "):
#         return auth.split(" ", 1)[1].strip()
#     alt = headers.get("token") or headers.get("Token")
#     return alt


# auth.py
import os
import time
import uuid
import base64
from typing import Any, Dict, Optional, Tuple
from functools import lru_cache
import jwt
from dotenv import load_dotenv
from threading import RLock

load_dotenv()

# =============================================================================
# JWT configuration (env-driven)
# =============================================================================
JWT_SECRET = os.getenv("JWT_SECRET", "change-this-in-prod")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRY = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRY_SECONDS", "3600"))     # 60 minutes default
REFRESH_TOKEN_EXPIRY = int(os.getenv("JWT_REFRESH_TOKEN_EXPIRY_SECONDS", "86400"))  # 24 hours default

# Transport behavior flags
# - ENCODE: return Base64URL-wrapped token in the JSON response body (default: true)
# - SET_ACCESS_COOKIE: also set access token as HttpOnly cookie via middleware (default: true)
# - RETURN_RAW_ACCESS: include raw token in body (default: false) -> safer default: do NOT return raw JWT
JWT_TRANSPORT_ENCODE = os.getenv("JWT_TRANSPORT_ENCODE", "true").strip().lower() == "true"
JWT_SET_ACCESS_COOKIE = os.getenv("JWT_SET_ACCESS_COOKIE", "true").strip().lower() == "true"
JWT_RETURN_RAW_ACCESS = os.getenv("JWT_RETURN_RAW_ACCESS", "false").strip().lower() == "true"

# =============================================================================
# In-memory revocation store
# =============================================================================
_revoked_jti: Dict[str, int] = {}  # jti -> exp (epoch seconds)
_rev_lock = RLock()

def _purge_revoked() -> None:
    """Purge expired entries from the in-memory revocation set."""
    now = int(time.time())
    with _rev_lock:
        stale = [j for j, exp in _revoked_jti.items() if exp and exp <= now]
        for j in stale:
            _revoked_jti.pop(j, None)

def is_token_revoked(jti: Optional[str]) -> bool:
    """Return True if the token JTI is marked revoked."""
    if not jti:
        return False
    _purge_revoked()
    with _rev_lock:
        return jti in _revoked_jti

def revoke_token(token: str) -> Tuple[bool, str]:
    """
    Revoke a JWT (access or refresh). Returns (revoked, message).
    - Decodes ignoring exp to extract jti/exp reliably.
    - Stores jti until original exp so verification denies it thereafter.
    - Accepts either raw JWT or Base64URL-wrapped JWT.
    """
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

        with _rev_lock:
            _revoked_jti[jti] = exp

        _cached_jwt_decode.cache_clear()
        return True, "Token revoked"
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