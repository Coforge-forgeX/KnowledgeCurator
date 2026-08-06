"""
MCPAppClient - High-level MCP Client Wrapper

A reusable wrapper around FastMCP Client that provides:
- Async context lifecycle management
- Tool caching with TTL + server notification invalidation
- Composable call interceptors for tracing/retry/metrics
- Default handler wiring with sensible defaults
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .cache import ToolCache
from .config import ClientBuildOptions, mcp_settings
from .factory import ClientFactory
from .handlers import (
    DefaultLogAdapter,
    DefaultProgressAdapter,
    DefaultSamplingAdapter,
    DefaultElicitationAdapter,
    ToolCacheInvalidationHandler,
)
from .interceptors import CompositeInterceptor, CallInterceptor

logger = logging.getLogger("core.mcp.client")


@dataclass
class MCPAppClient:
    """
    A reusable wrapper that exposes a stable interface:

    - async context lifecycle:
        FastMCP requires `async with client:` to connect/disconnect cleanly.

    - caching:
        `list_tools()` can be cached and invalidated when the server sends
        tool list change notifications.

    - interception:
        Call interceptors are framework-level (not a FastMCP feature) but help
        standardize tracing/metrics/retries in enterprise usage.

    Example usage:
        async with MCPAppClient(
            source="http://mcp-server/mcp",
            options=ClientBuildOptions(timeout_s=30.0),
        ) as mcp:
            tools = await mcp.list_tools_cached()
            result = await mcp.call_tool("jira_create_issue", {"summary": "Bug fix"})
    """

    source: Any
    options: ClientBuildOptions = field(default_factory=ClientBuildOptions)
    factory: ClientFactory = field(default_factory=ClientFactory)

    # Optional integrations:
    llm_provider: Any = None  # used for sampling handler
    get_user_input: Any = field(default=input)  # used for elicitation handler
    interceptors: list[CallInterceptor] = field(default_factory=list)
    elicitation_handler: Any = None

    # Tool cache with configurable TTL
    tool_cache: ToolCache = field(
        default_factory=lambda: ToolCache(ttl_s=mcp_settings.MCP_CACHE_TTL_S)
    )

    _client: Optional[Any] = field(default=None, init=False)
    _composite: Optional[CompositeInterceptor] = field(default=None, init=False)

    def _build_client(self) -> Any:
        """
        Wire all advanced handlers with sensible defaults.

        - log_handler: structured server logging
        - progress_handler: progress notifications
        - sampling_handler: if llm_provider is provided, handle server sampling requests
        - elicitation_handler: interactive structured user input
        - message_handler: receive list changed notifications to invalidate caches
        """
        log_adapter = DefaultLogAdapter()
        progress_adapter = DefaultProgressAdapter()

        sampling_adapter = (
            DefaultSamplingAdapter(self.llm_provider) if self.llm_provider else None
        )
        elicitation_adapter = (
            self.elicitation_handler or DefaultElicitationAdapter(self.get_user_input)
        )

        async def on_tools_changed():
            logger.info("Tool cache invalidated due to server notification.")
            self.tool_cache.invalidate()

        msg_handler = ToolCacheInvalidationHandler(on_tools_changed=on_tools_changed)

        return self.factory.build(
            self.source,
            options=self.options,
            log_handler=log_adapter,
            progress_handler=progress_adapter,
            sampling_handler=sampling_adapter,
            elicitation_handler=elicitation_adapter,
            message_handler=msg_handler,
        )

    async def __aenter__(self) -> "MCPAppClient":
        """
        Enter connected state.

        This wrapper calls into fastmcp.Client.__aenter__ which performs
        initialization and sets up the active session.
        """
        self._composite = CompositeInterceptor(self.interceptors)
        self._client = self._build_client()
        logger.info("Connecting MCPAppClient...")
        await self._client.__aenter__()
        logger.info("MCPAppClient connected.")
        return self

    async def __aexit__(self, exc_type, exc, tb):
        """
        Exit connected state and close the connection/session.

        FastMCP closes automatically at end of async context manager.
        """
        if self._client is not None:
            logger.info("Disconnecting MCPAppClient...")
            await self._client.__aexit__(exc_type, exc, tb)
            logger.info("MCPAppClient disconnected.")
        self._client = None

    @property
    def client(self) -> Any:
        """
        Access the underlying fastmcp.Client (only valid while connected).
        """
        if not self._client:
            logger.error("Client not connected; use `async with MCPAppClient(...)`")
            raise RuntimeError("Client not connected; use `async with MCPAppClient(...)`")
        return self._client

    async def list_tools_cached(self) -> Any:
        """
        Cached list_tools() with TTL + invalidation.

        list_tools() enumerates server-exposed tools.
        Tool list can change; FastMCP can notify via messages, which we hook
        to invalidate cache.
        """
        now_s = time.time()
        logger.debug("Fetching cached tool list...")
        return await self.tool_cache.get(now_s, self.client.list_tools)

    async def list_tools(self) -> Any:
        """
        Direct list_tools() without caching.
        """
        return await self.client.list_tools()

    async def call_tool(
        self,
        tool_name: str,
        args: dict[str, Any],
        *,
        progress_handler: Any = None,
    ) -> Any:
        """
        Call a tool with optional per-call progress handler override.

        FastMCP supports:
        - calling tools by name with argument dicts
        - receiving progress notifications via a client-level handler
        - overriding the progress handler for a specific call_tool invocation

        Args:
            tool_name: Name of the tool to call
            args: Arguments dict to pass to the tool
            progress_handler: Optional per-call progress handler override

        Returns:
            Tool execution result
        """
        if self._composite:
            await self._composite.before(tool_name, args)
        try:
            logger.info(f"Calling tool '{tool_name}' with args: {args}")
            result = await self.client.call_tool(
                tool_name,
                args,
                progress_handler=progress_handler,
            )
            if self._composite:
                await self._composite.after(tool_name, args, result)
            logger.info(f"Tool '{tool_name}' call completed")
            return result
        except BaseException as e:
            if self._composite:
                await self._composite.error(tool_name, args, e)
            logger.error(f"Error calling tool '{tool_name}': {e}")
            raise

    async def ping(self) -> bool:
        """
        Ping the MCP server to check connectivity.

        Returns:
            True if ping succeeds
        """
        await self.client.ping()
        return True

    async def get_tool(self, tool_name: str) -> Optional[Any]:
        """
        Get a specific tool by name from the cached tool list.

        Args:
            tool_name: Name of the tool to find

        Returns:
            Tool info if found, None otherwise
        """
        tools = await self.list_tools_cached()
        for tool in tools:
            if getattr(tool, "name", None) == tool_name:
                return tool
        return None


# Convenience function for one-shot tool calls
async def call_mcp_tool(
    source: str,
    tool_name: str,
    args: dict[str, Any],
    *,
    options: Optional[ClientBuildOptions] = None,
) -> Any:
    """
    Convenience function for one-shot MCP tool calls.

    Opens a connection, calls the tool, and closes the connection.

    Args:
        source: MCP server URL or config
        tool_name: Name of the tool to call
        args: Arguments dict to pass to the tool
        options: Optional ClientBuildOptions

    Returns:
        Tool execution result
    """
    opts = options or mcp_settings.get_default_options()
    async with MCPAppClient(source=source, options=opts) as mcp:
        return await mcp.call_tool(tool_name, args)
