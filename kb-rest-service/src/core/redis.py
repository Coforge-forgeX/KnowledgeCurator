"""Redis client manager for caching and token revocation"""
import hashlib
import json
from typing import Any, Dict, Optional

from .config import settings
from .logging import get_logger

logger = get_logger(__name__)

# Redis setup
try:
    import redis  # type: ignore[import]

    REDIS_AVAILABLE = True
except ImportError:
    logger.warning("Redis library not installed - Redis features unavailable")
    REDIS_AVAILABLE = False
    redis = None


class RedisClient:
    """Redis connection manager with singleton pattern"""

    def __init__(self):
        self._client: Optional["redis.Redis"] = None
        self._is_connected: bool = False

    def initialize(self) -> bool:
        """
        Initialize Redis client connection.
        Returns True if connected, False otherwise.
        """
        if not REDIS_AVAILABLE:
            logger.info("Redis library not available")
            return False

        if self._client is not None:
            return self._is_connected

        if not settings.database.REDIS_HOST:
            logger.info("Redis not configured (REDIS_HOST not set)")
            return False

        try:
            self._client = redis.Redis(
                host=settings.database.REDIS_HOST,
                port=settings.database.REDIS_PORT,
                password=settings.database.REDIS_PASSWORD,
                db=settings.database.REDIS_DB,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
            )
            # Test connection
            self._client.ping()
            self._is_connected = True
            logger.info(
                "Redis connected successfully",
                redis_host=settings.database.REDIS_HOST,
                redis_port=settings.database.REDIS_PORT,
            )
            return True
        except Exception as e:
            logger.warning("Redis connection unavailable", error=e)
            self._client = None
            self._is_connected = False
            return False

    def close(self) -> None:
        """Close Redis connection and cleanup"""
        if self._client is not None:
            try:
                self._client.close()
            except Exception as e:
                logger.warning("Error closing Redis connection", error=e)
            finally:
                self._client = None
                self._is_connected = False

    @property
    def client(self) -> Optional["redis.Redis"]:
        """
        Get Redis client instance with health check.
        Auto-initializes on first access if not already connected.
        Performs health check and reconnects if needed (serverless-friendly).
        """
        if self._client is None and not self._is_connected:
            self.initialize()
        elif self._client is not None:
            # Health check for serverless environments
            try:
                self._client.ping()
            except Exception as e:
                logger.warning("Redis connection lost, reconnecting", error=e)
                self.close()
                self.initialize()
        return self._client

    @property
    def is_available(self) -> bool:
        """Check if Redis is available and connected"""
        return REDIS_AVAILABLE and self._is_connected

    def get(self, key: str) -> Optional[str]:
        """Get value from Redis, returns None if unavailable or key not found"""
        if not self.is_available or self._client is None:
            return None
        try:
            return self._client.get(key)
        except Exception as e:
            logger.error("Redis GET operation failed", error=e, key=key)
            return None

    def set(self, key: str, value: str, ex: Optional[int] = None) -> bool:
        """Set value in Redis with optional expiration (seconds). Returns success status."""
        if not self.is_available or self._client is None:
            return False
        try:
            self._client.set(key, value, ex=ex)
            return True
        except Exception as e:
            logger.error("Redis SET operation failed", error=e, key=key)
            return False

    def setex(self, key: str, time: int, value: str) -> bool:
        """Set value with expiration time (seconds). Returns success status."""
        if not self.is_available or self._client is None:
            return False
        try:
            self._client.setex(key, time, value)
            return True
        except Exception as e:
            logger.error("Redis SETEX operation failed", error=e, key=key, ttl=time)
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        if not self.is_available or self._client is None:
            return False
        try:
            return bool(self._client.exists(key))
        except Exception as e:
            logger.error("Redis EXISTS operation failed", error=e, key=key)
            return False

    def delete(self, *keys: str) -> int:
        """Delete one or more keys. Returns number of keys deleted."""
        if not self.is_available or self._client is None:
            return 0
        try:
            return self._client.delete(*keys)
        except Exception as e:
            logger.error("Redis DELETE operation failed", error=e, keys=list(keys))
            return 0

    def ttl(self, key: str) -> int:
        """Get time-to-live for a key in seconds. Returns -1 if key has no expiry, -2 if key doesn't exist."""
        if not self.is_available or self._client is None:
            return -2
        try:
            return self._client.ttl(key)
        except Exception as e:
            logger.error("Redis TTL operation failed", error=e, key=key)
            return -2

    # ========================================
    # Query Caching Methods (OPTIMIZED)
    # ========================================

    def _make_query_cache_key(self, workspace_id: int, query: str, mode: str) -> str:
        """Generate cache key for RAG query results"""
        normalized_query = query.strip().lower()
        hash_input = f"{workspace_id}:{normalized_query}:{mode}"
        query_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        return f"query:{workspace_id}:{query_hash}"

    def get_query_result(
        self, workspace_id: int, query: str, mode: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached query result.

        Returns:
            Cached result dict or None if not found
        """
        key = self._make_query_cache_key(workspace_id, query, mode)
        cached = self.get(key)
        if cached:
            try:
                result = json.loads(cached)
                logger.debug("Cache HIT - Query result", workspace_id=workspace_id, mode=mode)
                return result
            except json.JSONDecodeError:
                logger.error("Invalid JSON in cache", key=key)
                self.delete(key)
        return None

    def set_query_result(
        self,
        workspace_id: int,
        query: str,
        mode: str,
        result: Dict[str, Any],
        ttl: int = 3600,
    ) -> bool:
        """
        Cache query result with TTL.

        Args:
            workspace_id: Workspace ID
            query: Query text
            mode: Query mode
            result: Result dict to cache
            ttl: Time-to-live in seconds (default: 1 hour)

        Returns:
            True if cached successfully
        """
        key = self._make_query_cache_key(workspace_id, query, mode)
        try:
            serialized = json.dumps(result)
            success = self.setex(key, ttl, serialized)
            if success:
                logger.debug(
                    "Cache SET - Query result",
                    workspace_id=workspace_id,
                    mode=mode,
                    ttl=ttl,
                    size_kb=round(len(serialized) / 1024, 2),
                )
            return success
        except Exception as e:
            logger.error("Failed to cache query result", error=e, workspace_id=workspace_id)
            return False

    def invalidate_workspace_cache(self, workspace_id: int) -> int:
        """
        Invalidate all cached queries for a workspace.

        Args:
            workspace_id: Workspace ID

        Returns:
            Number of keys deleted
        """
        if not self.is_available or self._client is None:
            return 0

        try:
            pattern = f"query:{workspace_id}:*"
            keys = list(self._client.scan_iter(match=pattern, count=100))
            if keys:
                deleted = self.delete(*keys)
                logger.info(
                    "Cache invalidated for workspace",
                    workspace_id=workspace_id,
                    keys_deleted=deleted,
                )
                return deleted
            return 0
        except Exception as e:
            logger.error("Cache invalidation failed", error=e, workspace_id=workspace_id)
            return 0

    def get_workspace_config(self, workspace_id: int) -> Optional[Dict[str, Any]]:
        """Get cached workspace configuration"""
        key = f"workspace:config:{workspace_id}"
        cached = self.get(key)
        if cached:
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                self.delete(key)
        return None

    def set_workspace_config(
        self, workspace_id: int, config: Dict[str, Any], ttl: int = 300
    ) -> bool:
        """
        Cache workspace configuration.

        Args:
            workspace_id: Workspace ID
            config: Config dict
            ttl: Time-to-live in seconds (default: 5 minutes)
        """
        key = f"workspace:config:{workspace_id}"
        try:
            serialized = json.dumps(config)
            return self.setex(key, ttl, serialized)
        except Exception as e:
            logger.error("Failed to cache workspace config", error=e)
            return False


# Global Redis instance
redis_manager = RedisClient()


# Convenience functions for direct access
def get_redis_client() -> Optional["redis.Redis"]:
    """Get the Redis client instance (lazy-initialized)"""
    return redis_manager.client


def is_redis_available() -> bool:
    """Check if Redis is available"""
    return redis_manager.is_available


# Query caching convenience functions
def get_query_cache(workspace_id: int, query: str, mode: str) -> Optional[Dict[str, Any]]:
    """Get cached query result (convenience wrapper)"""
    return redis_manager.get_query_result(workspace_id, query, mode)


def set_query_cache(
    workspace_id: int, query: str, mode: str, result: Dict[str, Any], ttl: int = 3600
) -> bool:
    """Cache query result (convenience wrapper)"""
    return redis_manager.set_query_result(workspace_id, query, mode, result, ttl)


def invalidate_workspace(workspace_id: int) -> int:
    """Invalidate workspace cache (convenience wrapper)"""
    return redis_manager.invalidate_workspace_cache(workspace_id)
