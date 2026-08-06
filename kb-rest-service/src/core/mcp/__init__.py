"""
MCP Client Manager Framework

High-level wrapper around FastMCP Client providing:
- Async context lifecycle management
- Tool caching with TTL + server notification invalidation
- Composable call interceptors for tracing/retry/metrics
- Handler adapters for log/progress/sampling/elicitation
"""

from .client import MCPAppClient
from .config import ClientBuildOptions, MCPConfig, is_mcp_config
from .cache import ToolCache
from .factory import ClientFactory
from .handlers import (
    LogHandler,
    ProgressHandler,
    SamplingHandler,
    ElicitationHandler,
    DefaultLogAdapter,
    DefaultProgressAdapter,
    DefaultSamplingAdapter,
    DefaultElicitationAdapter,
    ToolCacheInvalidationHandler,
    LLMProvider,
)
from .interceptors import CallInterceptor, CompositeInterceptor

__all__ = [
    # Main client
    "MCPAppClient",
    # Configuration
    "ClientBuildOptions",
    "MCPConfig",
    "is_mcp_config",
    # Cache
    "ToolCache",
    # Factory
    "ClientFactory",
    # Handlers
    "LogHandler",
    "ProgressHandler",
    "SamplingHandler",
    "ElicitationHandler",
    "DefaultLogAdapter",
    "DefaultProgressAdapter",
    "DefaultSamplingAdapter",
    "DefaultElicitationAdapter",
    "ToolCacheInvalidationHandler",
    "LLMProvider",
    # Interceptors
    "CallInterceptor",
    "CompositeInterceptor",
]
