"""JWT authentication utilities"""
import base64
import binascii
import inspect
import time
import uuid
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple

import azure.functions as func
import jwt
from passlib.hash import argon2

from .config import settings
from .exceptions import AuthenticationException, AuthorizationException
from .logging import get_logger
from .redis import get_redis_client, is_redis_available

logger = get_logger(__name__)

REVOKED_TOKEN_PREFIX = "revoked_token:"


def create_jwt_token(
    claims: Dict[str, Any], encode_base64: bool = True
) -> Tuple[str, int]:
    """
    Create JWT access token.
    Returns: (token, expiration_seconds)

    Args:
        claims: JWT claims dictionary
        encode_base64: If True, return base64-encoded token; if False, return raw JWT
    """
    expiration_minutes = settings.security.JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    now = datetime.now(timezone.utc)
    exp_time = now + timedelta(minutes=expiration_minutes)

    payload = {
        **claims,
        "token_type": "access",
        "exp": exp_time,
        "iat": now,
        "jti": uuid.uuid4().hex,  # Add JWT ID for revocation support
    }

    token = jwt.encode(
        payload,
        settings.security.JWT_SECRET_KEY,
        algorithm=settings.security.JWT_ALGORITHM,
    )

    if encode_base64:
        token = base64.b64encode(token.encode("utf-8")).decode("utf-8")

    return token, expiration_minutes * 60


def create_refresh_token(user_id: int) -> Tuple[str, int]:
    """
    Create JWT refresh token.
    Returns: (token, expiration_seconds)
    """
    expiration_days = settings.security.JWT_REFRESH_TOKEN_EXPIRE_DAYS
    now = datetime.now(timezone.utc)

    payload = {
        "user_id": user_id,
        "sub": user_id,
        "token_type": "refresh",
        "jti": uuid.uuid4().hex,
        "iat": now,
        "exp": now + timedelta(days=expiration_days),
    }

    token = jwt.encode(
        payload,
        settings.security.JWT_SECRET_KEY,
        algorithm=settings.security.JWT_ALGORITHM,
    )
    return token, expiration_days * 24 * 60 * 60


def decode_and_verify_token(token: str) -> Dict[str, Any]:
    """Decode + verify a JWT. Returns claims dict, or raises AuthenticationException.

    Accepts a raw JWT or a base64-encoded JWT (decoded first via _normalize_token).
    Checks if token has been revoked (if Redis is available).
    """
    token = _normalize_token(token)
    try:
        claims = jwt.decode(
            token,
            settings.security.JWT_SECRET_KEY,
            algorithms=[settings.security.JWT_ALGORITHM],
            # Reject tokens missing the claims every handler relies on, so a
            # bad token fails here at 401 rather than KeyError-ing downstream.
            options={"require": ["exp", "user_id"]},
        )

        # Check if token is revoked (if Redis is available)
        if is_redis_available():
            jti = claims.get("jti")
            if jti:
                redis_client = get_redis_client()
                if redis_client:
                    key = f"{REVOKED_TOKEN_PREFIX}{jti}"
                    if redis_client.exists(key):
                        raise AuthenticationException(message="Token has been revoked")

        return claims
    except jwt.ExpiredSignatureError:
        raise AuthenticationException(message="Token has expired")
    except jwt.InvalidTokenError as e:
        raise AuthenticationException(message=f"Invalid token: {e}")


def decode_refresh_token(token: str) -> Dict[str, Any]:
    """Decode + verify a refresh token with 5-minute grace period.

    Refresh tokens can be used if:
    - Still valid (not expired)
    - Expired within the last 5 minutes (grace period for clock skew)

    Still validates signature (prevents forgery) and revocation status.

    Returns claims dict, or raises AuthenticationException.
    """
    token = _normalize_token(token)
    GRACE_PERIOD_SECONDS = 5 * 60  # 5 minutes

    try:
        # Verify signature but allow expired tokens temporarily to check grace period
        claims = jwt.decode(
            token,
            settings.security.JWT_SECRET_KEY,
            algorithms=[settings.security.JWT_ALGORITHM],
            options={
                "verify_exp": False,  # Temporarily disable to check grace period manually
                "require": ["user_id"],
            },
        )

        # Validate it's actually a refresh token
        if claims.get("token_type") != "refresh":
            raise AuthenticationException(message="Invalid token type")

        # Check expiration with grace period
        exp = claims.get("exp")
        if exp:
            now = datetime.now(timezone.utc).timestamp()
            if now > exp + GRACE_PERIOD_SECONDS:
                raise AuthenticationException(
                    message="Refresh token has expired beyond grace period"
                )

        # Check if token is revoked (if Redis is available)
        if is_redis_available():
            jti = claims.get("jti")
            if jti:
                redis_client = get_redis_client()
                if redis_client:
                    key = f"{REVOKED_TOKEN_PREFIX}{jti}"
                    if redis_client.exists(key):
                        raise AuthenticationException(message="Token has been revoked")

        return claims
    except jwt.InvalidTokenError as e:
        raise AuthenticationException(message=f"Invalid refresh token: {e}")


def verify_password(plain_password: str, hashed_password: Optional[str]) -> bool:
    """Verify password against hash"""
    if hashed_password is None:
        # Default password fallback for legacy users
        return plain_password == settings.security.DEFAULT_PASSWORD

    if hashed_password.startswith("$argon2"):
        try:
            return argon2.verify(plain_password, hashed_password)
        except Exception:
            return False

    # Plain text fallback (legacy)
    return plain_password == hashed_password


def hash_password(plain_password: str) -> str:
    """Hash password with Argon2"""
    return argon2.hash(plain_password)


def extract_bearer_token(req: func.HttpRequest) -> Optional[str]:
    """Return raw token from 'Authorization: Bearer <token>', else None."""
    header = req.headers.get("Authorization") or req.headers.get("authorization")
    if not header:
        return None

    parts = header.split(" ", 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None

    token = parts[1].strip()
    return token or None


def _normalize_token(token: str) -> str:
    """Return a raw JWT.

    A JWT is three base64url segments joined by '.', so a valid token contains
    two dots. If the token has no dots it is base64-encoded (the base64 alphabet
    has no '.'); decode it once to recover the underlying JWT. Anything that does
    not base64-decode to a dotted JWT is returned unchanged for verification to
    reject downstream.
    """
    if token.count(".") == 2:
        return token

    decoders: tuple[Callable[..., bytes], ...] = (
        base64.urlsafe_b64decode,
        base64.b64decode,
    )
    for decoder in decoders:
        try:
            # Restore padding stripped by some encoders.
            padded = token + "=" * (-len(token) % 4)
            decoded = decoder(padded).decode("utf-8")
        except (binascii.Error, ValueError, UnicodeDecodeError):
            continue
        if decoded.count(".") == 2:
            return decoded

    return token


def get_claims(req: func.HttpRequest) -> Dict[str, Any]:
    """Return verified JWT claims attached by require_auth().

    Raises AuthenticationException if no claims present.
    """
    claims = getattr(req, "claims", None)
    if not claims:
        raise AuthenticationException(message="Authentication required")
    return claims


def get_user_id(req: func.HttpRequest) -> int:
    """Authenticated user id from token."""
    return get_claims(req)["user_id"]


def get_email(req: func.HttpRequest) -> str:
    """Authenticated user email from token."""
    return get_claims(req)["email"]


def get_workspace_ids(req: func.HttpRequest) -> list:
    """Workspace ids user has role in, from token's roles array."""
    return [
        r["workspace_id"]
        for r in get_claims(req).get("roles", [])
        if isinstance(r, dict) and "workspace_id" in r
    ]


def require_auth(
    authorize: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Callable:
    """Decorator enforcing valid JWT on Azure Function.

    Stack UNDER @azure_http_decorator so exceptions convert to JSON error:

        @azure_http_decorator()
        @require_auth()
        async def main(req, context):
            user_id = get_user_id(req)
            ...

    Authorization pluggable via authorize callback: receives claims, returns
    True to allow, False/raises AuthorizationException to deny.

    On success, verified claims attached as req.claims (read via get_claims).
    Supports sync + async handlers.
    """

    def _decorator(fn: Callable) -> Callable:
        def _authenticate(args, kwargs) -> None:
            req = kwargs.get("req") if "req" in kwargs else (args[0] if args else None)
            if not isinstance(req, func.HttpRequest):
                raise AuthenticationException(message="Authentication failed")

            token = extract_bearer_token(req)
            if not token:
                raise AuthenticationException(
                    message="Missing or malformed Authorization header"
                )

            claims = decode_and_verify_token(token)
            setattr(req, "claims", claims)

            # Authorization (403) runs only after authentication (401) succeeds
            if authorize is not None and not authorize(claims):
                raise AuthorizationException()

        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            async def _async_wrapped(*args, **kwargs):
                _authenticate(args, kwargs)
                return await fn(*args, **kwargs)

            return _async_wrapped

        @wraps(fn)
        def _wrapped(*args, **kwargs):
            _authenticate(args, kwargs)
            return fn(*args, **kwargs)

        return _wrapped

    return _decorator


def revoke_token(token: str) -> Tuple[bool, str]:
    """
    Revoke a JWT (access or refresh). Returns (revoked, message).
    - Decodes ignoring exp to extract jti/exp reliably.
    - Stores jti in Redis with TTL equal to original token expiry.
    - This ensures revocation works across all worker processes.
    - Accepts either raw JWT or Base64URL-wrapped JWT.
    """
    if not is_redis_available():
        logger.error("Redis unavailable - cannot revoke token")
        return False, "Token revocation service unavailable"

    redis_client = get_redis_client()
    if redis_client is None:
        logger.error("Redis client not initialized")
        return False, "Token revocation service unavailable"

    try:
        # Decode the token (Base64URL-wrapped or raw JWT)
        token = _normalize_token(token)

        payload = jwt.decode(
            token,
            settings.security.JWT_SECRET_KEY,
            algorithms=[settings.security.JWT_ALGORITHM],
            options={"verify_exp": False},  # Decode even if expired
        )
        jti = payload.get("jti")
        exp = int(
            payload.get("exp")
            or (
                int(time.time())
                + settings.security.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60
            )
        )

        if not jti:
            logger.warning("Token missing jti - cannot revoke deterministically")
            return False, "Invalid token format"

        # Calculate TTL (time until token expires naturally)
        now = int(time.time())
        ttl = max(exp - now, 1)  # At least 1 second TTL

        # Store in Redis with expiration
        key = f"{REVOKED_TOKEN_PREFIX}{jti}"
        try:
            redis_client.setex(key, ttl, "revoked")
            logger.info(
                "Token revoked successfully",
                jti=jti,
                ttl_seconds=ttl,
                expires_at=exp,
            )
        except Exception as redis_err:
            logger.error("Failed to revoke token in Redis", error=redis_err, jti=jti)
            return False, "Token revocation failed"

        return True, "Token revoked successfully"

    except jwt.InvalidTokenError as e:
        logger.warning("Invalid token provided for revocation", error=e)
        return False, "Invalid token"
    except Exception as e:
        logger.error("Error revoking token", error=e)
        return False, "Token revocation failed"
