"""
MCP Call Interceptors

Composable hooks for cross-cutting concerns:
- Tracing/observability
- Retry logic
- Metrics collection
- Rate limiting
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

logger = logging.getLogger("core.mcp.interceptors")


class CallInterceptor(Protocol):
    """
    Protocol for tool call interceptors.

    Implement before/after/error to add cross-cutting concerns
    like tracing, metrics, or retry logic.
    """

    async def before(self, tool_name: str, args: dict[str, Any]) -> None:
        """Called before tool execution."""
        ...

    async def after(self, tool_name: str, args: dict[str, Any], result: Any) -> None:
        """Called after successful tool execution."""
        ...

    async def error(self, tool_name: str, args: dict[str, Any], exc: BaseException) -> None:
        """Called when tool execution raises an exception."""
        ...


@dataclass
class CompositeInterceptor:
    """
    Composes multiple interceptors into a single chain.

    Calls all interceptors in sequence for each hook.
    """

    interceptors: list[CallInterceptor] = field(default_factory=list)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("core.mcp.interceptors"))

    async def before(self, tool_name: str, args: dict[str, Any]) -> None:
        """Call before hook on all interceptors."""
        self.logger.debug(f"Before interceptors for tool: {tool_name}")
        for interceptor in self.interceptors:
            try:
                await interceptor.before(tool_name, args)
            except Exception as e:
                self.logger.warning(f"Interceptor before hook failed: {e}")

    async def after(self, tool_name: str, args: dict[str, Any], result: Any) -> None:
        """Call after hook on all interceptors."""
        self.logger.debug(f"After interceptors for tool: {tool_name}")
        for interceptor in self.interceptors:
            try:
                await interceptor.after(tool_name, args, result)
            except Exception as e:
                self.logger.warning(f"Interceptor after hook failed: {e}")

    async def error(self, tool_name: str, args: dict[str, Any], exc: BaseException) -> None:
        """Call error hook on all interceptors."""
        self.logger.error(f"Error interceptors for tool: {tool_name}, exception: {exc}")
        for interceptor in self.interceptors:
            try:
                await interceptor.error(tool_name, args, exc)
            except Exception as e:
                self.logger.warning(f"Interceptor error hook failed: {e}")


@dataclass
class LoggingInterceptor:
    """
    Simple logging interceptor for debugging tool calls.
    """

    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("mcp.tools"))
    log_args: bool = True
    log_result: bool = False  # Be careful with sensitive data

    async def before(self, tool_name: str, args: dict[str, Any]) -> None:
        if self.log_args:
            self.logger.info(f"Calling tool '{tool_name}' with args: {args}")
        else:
            self.logger.info(f"Calling tool '{tool_name}'")

    async def after(self, tool_name: str, args: dict[str, Any], result: Any) -> None:
        if self.log_result:
            self.logger.info(f"Tool '{tool_name}' completed with result: {result}")
        else:
            self.logger.info(f"Tool '{tool_name}' completed successfully")

    async def error(self, tool_name: str, args: dict[str, Any], exc: BaseException) -> None:
        self.logger.error(f"Tool '{tool_name}' failed: {exc}")


@dataclass
class MetricsInterceptor:
    """
    Metrics collection interceptor.

    Tracks:
    - Call count per tool
    - Latency per tool
    - Error count per tool
    """

    _call_count: dict[str, int] = field(default_factory=dict, init=False)
    _error_count: dict[str, int] = field(default_factory=dict, init=False)
    _total_latency_ms: dict[str, float] = field(default_factory=dict, init=False)
    _start_times: dict[str, float] = field(default_factory=dict, init=False)

    async def before(self, tool_name: str, args: dict[str, Any]) -> None:
        self._start_times[tool_name] = time.time()
        self._call_count[tool_name] = self._call_count.get(tool_name, 0) + 1

    async def after(self, tool_name: str, args: dict[str, Any], result: Any) -> None:
        if tool_name in self._start_times:
            latency_ms = (time.time() - self._start_times[tool_name]) * 1000
            self._total_latency_ms[tool_name] = (
                self._total_latency_ms.get(tool_name, 0) + latency_ms
            )
            del self._start_times[tool_name]

    async def error(self, tool_name: str, args: dict[str, Any], exc: BaseException) -> None:
        self._error_count[tool_name] = self._error_count.get(tool_name, 0) + 1
        # Still record latency for failed calls
        if tool_name in self._start_times:
            latency_ms = (time.time() - self._start_times[tool_name]) * 1000
            self._total_latency_ms[tool_name] = (
                self._total_latency_ms.get(tool_name, 0) + latency_ms
            )
            del self._start_times[tool_name]

    def get_metrics(self) -> dict[str, Any]:
        """Get collected metrics."""
        return {
            "call_count": dict(self._call_count),
            "error_count": dict(self._error_count),
            "total_latency_ms": dict(self._total_latency_ms),
            "avg_latency_ms": {
                name: self._total_latency_ms.get(name, 0) / count
                for name, count in self._call_count.items()
                if count > 0
            },
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self._call_count.clear()
        self._error_count.clear()
        self._total_latency_ms.clear()
        self._start_times.clear()
