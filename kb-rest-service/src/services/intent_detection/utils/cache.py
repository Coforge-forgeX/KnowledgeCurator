"""
Caching utilities for intent detection.

Optimized for serverless: uses in-memory LRU cache with TTL.
For production, can be extended to use Redis or similar.
"""
import hashlib
import logging
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)


class SimpleLRUCache:
    """
    Simple thread-safe LRU cache with TTL support.

    Optimized for serverless cold starts - minimal memory footprint.
    In production, replace with Redis for distributed caching.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        """
        Initialize cache.

        Args:
            max_size: Maximum number of cached items
            ttl_seconds: Time-to-live for cached items (default: 5 minutes)
        """
        self.max_size = max_size
        self.ttl = timedelta(seconds=ttl_seconds)
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._access_order: list[str] = []

    def get(self, key: str) -> Optional[Any]:
        """Get item from cache if not expired"""
        if key not in self._cache:
            return None

        value, timestamp = self._cache[key]

        # Check if expired
        if datetime.utcnow() - timestamp > self.ttl:
            self._evict(key)
            return None

        # Update access order (LRU)
        self._update_access(key)
        return value

    def put(self, key: str, value: Any) -> None:
        """Put item in cache, evicting LRU if full"""
        # Evict oldest if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_lru()

        self._cache[key] = (value, datetime.utcnow())
        self._update_access(key)

    def _update_access(self, key: str) -> None:
        """Update access order for LRU tracking"""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if self._access_order:
            lru_key = self._access_order[0]
            self._evict(lru_key)

    def _evict(self, key: str) -> None:
        """Remove item from cache"""
        self._cache.pop(key, None)
        if key in self._access_order:
            self._access_order.remove(key)

    def clear(self) -> None:
        """Clear all cached items"""
        self._cache.clear()
        self._access_order.clear()

    def stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl_seconds": int(self.ttl.total_seconds()),
        }


# Global cache instance (singleton pattern for serverless optimization)
_global_cache: Optional[SimpleLRUCache] = None


def get_cache(max_size: int = 100, ttl_seconds: int = 300) -> SimpleLRUCache:
    """
    Get or create global cache instance.

    Args:
        max_size: Maximum cache size
        ttl_seconds: Cache TTL in seconds

    Returns:
        Cache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = SimpleLRUCache(max_size=max_size, ttl_seconds=ttl_seconds)
    return _global_cache


def _cache_key(message: str, context: Optional[dict] = None) -> str:
    """
    Generate cache key from message and context.

    Args:
        message: User message
        context: Optional context dict

    Returns:
        Unique cache key (SHA256 hash)
    """
    # Normalize message (lowercase, strip whitespace)
    normalized = message.lower().strip()

    # Include relevant context in cache key
    context_str = ""
    if context:
        # Only include stable context fields (not timestamps, session IDs, etc.)
        stable_keys = {"workspace_id", "mode", "agent_id"}
        context_str = "|".join(
            f"{k}:{context[k]}"
            for k in stable_keys
            if k in context
        )

    cache_input = f"{normalized}|{context_str}"
    return hashlib.sha256(cache_input.encode()).hexdigest()[:16]


def cached_detection(
    enabled: bool = True,
    max_size: int = 100,
    ttl_seconds: int = 300,
) -> Callable:
    """
    Decorator for caching intent detection results.

    Usage:
        @cached_detection(enabled=True, ttl_seconds=300)
        async def detect(self, message, context):
            # ... detection logic

    Args:
        enabled: Whether caching is enabled
        max_size: Maximum cache size
        ttl_seconds: Cache TTL

    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(self, message: str, context: Optional[dict] = None, **kwargs):
            # Skip caching if disabled or detector doesn't support it
            if not enabled or not getattr(self, "supports_caching", lambda: False)():
                return await func(self, message, context, **kwargs)

            # Generate cache key
            key = _cache_key(message, context)
            cache = get_cache(max_size=max_size, ttl_seconds=ttl_seconds)

            # Try to get from cache
            cached_result = cache.get(key)
            if cached_result is not None:
                logger.debug(f"Cache hit for intent detection: {key}")
                return cached_result

            # Cache miss - call actual function
            logger.debug(f"Cache miss for intent detection: {key}")
            result = await func(self, message, context, **kwargs)

            # Store in cache
            cache.put(key, result)

            return result

        return wrapper

    return decorator
