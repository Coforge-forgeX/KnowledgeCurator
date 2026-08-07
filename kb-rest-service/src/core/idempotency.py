"""
Idempotency Service (Redis-based)

Provides idempotency key handling for API endpoints to prevent duplicate processing.
Uses Redis for fast, distributed caching with automatic TTL expiration.

Keys are scoped by workspace_id, user_id, and idempotency_key to allow:
- Same key reuse across different workspaces
- User-specific idempotency within a workspace
- Query/request body verification for duplicate detection

Usage:
    from src.core.idempotency import check_idempotency, store_idempotency_result

    # At the start of an endpoint
    cached_response = await check_idempotency(
        idempotency_key="unique-key",
        workspace_id=workspace_id,
        user_id=user_id,
        endpoint="/api/v2/documents/upload",
        request_body={"files": [...]},
    )
    if cached_response:
        return cached_response  # Return cached response for duplicate request

    # ... process request ...

    # Store the result
    await store_idempotency_result(
        idempotency_key="unique-key",
        workspace_id=workspace_id,
        user_id=user_id,
        endpoint="/api/v2/documents/upload",
        request_body={"files": [...]},
        response_status=202,
        response_body={"success": True, ...},
    )
"""

import hashlib
import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from .abstractions import AbstractResponse
from .logging import get_logger
from .redis import get_redis_client, is_redis_available

logger = get_logger(__name__)

# Default expiration time for idempotency keys (24 hours in seconds)
DEFAULT_EXPIRATION_SECONDS = 60 * 60  # 3600 seconds = 1 hour

# Redis key prefix
IDEMPOTENCY_KEY_PREFIX = "idempotency"


def _compute_request_hash(request_body: Dict[str, Any]) -> str:
    """
    Compute a stable hash of the request body for verification.

    This ensures that if the same idempotency key is used with different
    request bodies, we detect it as a potential error.
    """
    normalized = json.dumps(request_body, sort_keys=True)
    return hashlib.sha256(normalized.encode()).hexdigest()


def _make_redis_key(workspace_id: int, user_id: int, idempotency_key: str) -> str:
    """
    Generate Redis key with namespace, workspace, and user scoping.
    Format: idempotency:{workspace_id}:{user_id}:{idempotency_key}
    """
    return f"{IDEMPOTENCY_KEY_PREFIX}:{workspace_id}:{user_id}:{idempotency_key}"


async def check_idempotency(
    idempotency_key: str,
    workspace_id: int,
    user_id: int,
    endpoint: str,
    request_body: Dict[str, Any],
) -> Optional[AbstractResponse]:
    """
    Check if a request with this idempotency key has been processed before.

    Args:
        idempotency_key: Unique identifier provided by the client
        workspace_id: ID of the workspace (for key scoping)
        user_id: ID of the user making the request
        endpoint: API endpoint path
        request_body: Request payload for hash verification

    Returns:
        AbstractResponse if a matching cached response exists, None otherwise

    Raises:
        ValueError: If the same key is used with different request body
    """
    # Check if Redis is available
    if not is_redis_available():
        logger.warning("Redis not available, idempotency check skipped")
        return None

    try:
        redis_client = get_redis_client()
        if not redis_client:
            logger.warning("Redis client not initialized, idempotency check skipped")
            return None

        request_hash = _compute_request_hash(request_body)
        redis_key = _make_redis_key(workspace_id, user_id, idempotency_key)

        # Get cached data from Redis
        cached_data_json = await redis_client.get(redis_key)

        if not cached_data_json:
            logger.debug(
                "No cached idempotency record found",
                idempotency_key=idempotency_key,
                workspace_id=workspace_id,
                user_id=user_id,
            )
            return None

        # Parse cached data
        cached_data = json.loads(cached_data_json)

        # Verify request hash matches
        if cached_data.get("request_hash") != request_hash:
            logger.error(
                "Idempotency key reused with different request body",
                idempotency_key=idempotency_key,
                workspace_id=workspace_id,
                user_id=user_id,
                endpoint=endpoint,
                stored_hash=cached_data.get("request_hash"),
                current_hash=request_hash,
            )
            raise ValueError(
                f"Idempotency key '{idempotency_key}' was previously used with a different request body"
            )

        logger.info(
            "Returning cached idempotency response",
            idempotency_key=idempotency_key,
            workspace_id=workspace_id,
            user_id=user_id,
            endpoint=endpoint,
            status_code=cached_data.get("response_status"),
        )

        # Return cached response
        return AbstractResponse(
            body=cached_data.get("response_body", {}),
            status_code=cached_data.get("response_status", 200),
            headers={
                "X-Idempotency-Replay": "true",
                "X-Cached-At": cached_data.get("created_at", ""),
            },
            mimetype="application/json",
        )

    except ValueError:
        # Re-raise validation errors
        raise
    except Exception as e:
        logger.error(
            "Error checking idempotency",
            error=e,
            idempotency_key=idempotency_key,
            workspace_id=workspace_id,
            user_id=user_id,
        )
        # Fail open - if idempotency check fails, allow the request to proceed
        return None


async def store_idempotency_result(
    idempotency_key: str,
    workspace_id: int,
    user_id: int,
    endpoint: str,
    request_body: Dict[str, Any],
    response_status: int,
    response_body: Dict[str, Any],
    expiration_seconds: int = DEFAULT_EXPIRATION_SECONDS,
) -> bool:
    """
    Store the result of an API request for idempotency checking.

    Args:
        idempotency_key: Unique identifier provided by the client
        workspace_id: ID of the workspace (for key scoping)
        user_id: ID of the user making the request
        endpoint: API endpoint path
        request_body: Request payload for hash computation
        response_status: HTTP status code of the response
        response_body: Response body to cache
        expiration_seconds: How long to keep the record in seconds (default: 24 hours)

    Returns:
        True if stored successfully, False otherwise
    """
    # Check if Redis is available
    if not is_redis_available():
        logger.warning("Redis not available, idempotency result not stored")
        return False

    try:
        redis_client = get_redis_client()
        if not redis_client:
            logger.warning("Redis client not initialized, idempotency result not stored")
            return False

        request_hash = _compute_request_hash(request_body)
        redis_key = _make_redis_key(workspace_id, user_id, idempotency_key)

        # Prepare data to cache
        cache_data = {
            "idempotency_key": idempotency_key,
            "workspace_id": workspace_id,
            "user_id": user_id,
            "endpoint": endpoint,
            "request_hash": request_hash,
            "response_status": response_status,
            "response_body": response_body,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }

        # Store in Redis with TTL
        await redis_client.set(
            redis_key,
            json.dumps(cache_data),
            ex=expiration_seconds,  # Auto-expire after TTL
        )

        logger.info(
            "Stored idempotency record in Redis",
            idempotency_key=idempotency_key,
            workspace_id=workspace_id,
            user_id=user_id,
            endpoint=endpoint,
            ttl_seconds=expiration_seconds,
        )

        return True

    except Exception as e:
        logger.error(
            "Failed to store idempotency record",
            error=e,
            idempotency_key=idempotency_key,
            workspace_id=workspace_id,
            user_id=user_id,
        )
        # Non-critical failure - the request completed successfully
        # Just log and continue
        return False


async def cleanup_expired_records() -> int:
    """
    Cleanup expired idempotency records.

    Note: With Redis, this is handled automatically via TTL/expiration.
    This function is kept for API compatibility but does nothing.

    Returns:
        0 (Redis handles expiration automatically)
    """
    logger.info("Redis handles idempotency expiration automatically via TTL")
    return 0


async def get_stats() -> Dict[str, Any]:
    """
    Get idempotency cache statistics from Redis.

    Returns:
        Dictionary with cache statistics
    """
    if not is_redis_available():
        return {"available": False, "message": "Redis not available"}

    try:
        redis_client = get_redis_client()
        if not redis_client:
            return {"available": False, "message": "Redis client not initialized"}

        # Count keys matching our prefix
        pattern = f"{IDEMPOTENCY_KEY_PREFIX}:*"
        keys = await redis_client.keys(pattern)
        total_keys = len(keys) if keys else 0

        return {
            "available": True,
            "total_cached_requests": total_keys,
            "key_prefix": IDEMPOTENCY_KEY_PREFIX,
            "default_ttl_seconds": DEFAULT_EXPIRATION_SECONDS,
            "default_ttl_hours": DEFAULT_EXPIRATION_SECONDS / 3600,
        }

    except Exception as e:
        logger.error("Failed to get idempotency stats", error=e)
        return {
            "available": False,
            "error": str(e),
        }
