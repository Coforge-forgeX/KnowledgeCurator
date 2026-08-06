"""
MCP Tool Cache

A minimal async cache used for list_tools() results.

Why caching tools matters:
- Deterministic clients often call list_tools() repeatedly (UI refresh, routing, validation).
- FastMCP can notify the client when tool lists change via message notifications
  so you can invalidate the cache when that happens.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

logger = logging.getLogger("core.mcp.cache")


@dataclass
class ToolCache:
    """
    TTL cache for tool metadata.

    Notes:
    - Thread-safe in async context via a lock.
    - invalidate() should be called when you receive notifications/tools/list_changed.
    """

    ttl_s: float = 180.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False)
    _value: Optional[Any] = field(default=None, init=False)
    _expires_at: float = field(default=0.0, init=False)

    def invalidate(self) -> None:
        """Clear cache immediately (e.g., on tool list changed notification)."""
        logger.debug("Tool cache invalidated")
        self._value = None
        self._expires_at = 0.0

    async def get(
        self,
        now_s: float,
        loader: Callable[[], Awaitable[Any]],
    ) -> Any:
        """
        Get cached value or load a fresh one using `loader`.

        Args:
            now_s: Current timestamp in seconds (time.time())
            loader: An async function such as client.list_tools()

        Returns:
            Cached or freshly loaded value
        """
        async with self._lock:
            if self._value is not None and now_s < self._expires_at:
                logger.debug("Tool cache hit")
                return self._value

            logger.debug("Tool cache miss, loading fresh data")
            self._value = await loader()
            self._expires_at = now_s + self.ttl_s
            return self._value

    def is_valid(self) -> bool:
        """Check if cache has a valid, non-expired value."""
        return self._value is not None and time.time() < self._expires_at

    def clear(self) -> None:
        """Alias for invalidate()."""
        self.invalidate()
