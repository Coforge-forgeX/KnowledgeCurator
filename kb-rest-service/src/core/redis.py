"""Redis client manager for caching and token revocation"""
from typing import Optional

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


# Global Redis instance
redis_manager = RedisClient()


# Convenience functions for direct access
def get_redis_client() -> Optional["redis.Redis"]:
    """Get the Redis client instance (lazy-initialized)"""
    return redis_manager.client


def is_redis_available() -> bool:
    """Check if Redis is available"""
    return redis_manager.is_available
