"""
MCP Handler Adapters

Adapters for FastMCP client advanced callbacks:
- log_handler: structured server logs
- progress_handler: progress notifications + per-call override support
- sampling_handler: server-initiated LLM sampling
- elicitation_handler: server-initiated user input requests
- message_handler: unified notifications including list-changed hooks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Protocol, get_origin, get_args

logger = logging.getLogger("core.mcp.handlers")


# Type aliases for handler signatures
ProgressHandler = Callable[[float, Optional[float], Optional[str]], Awaitable[None]]
LogHandler = Callable[[Any], Awaitable[None]]
SamplingHandler = Callable[[list, Any, Any], Awaitable[str]]
ElicitationHandler = Callable[[str, Optional[type], Any, Any], Awaitable[Any]]


class LLMProvider(Protocol):
    """
    Abstraction over any LLM backend used to fulfill MCP sampling requests.
    """

    async def complete(
        self,
        messages: list,
        params: Any,
        ctx: Any,
    ) -> str:
        ...


@dataclass
class DefaultLogAdapter:
    """
    Forward server-emitted structured logs to Python logging.
    """

    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("mcp.server"))
    level_map: dict[str, int] = field(default=None)

    def __post_init__(self):
        if self.level_map is None:
            self.level_map = {
                "DEBUG": logging.DEBUG,
                "INFO": logging.INFO,
                "WARNING": logging.WARNING,
                "WARN": logging.WARNING,
                "ERROR": logging.ERROR,
                "CRITICAL": logging.CRITICAL,
            }

    async def __call__(self, message: Any) -> None:
        """Handle log message from MCP server."""
        try:
            # Handle both dict-style and object-style messages
            if hasattr(message, "data"):
                msg = message.data.get("msg", str(message))
                extra = message.data.get("extra", {})
                level_str = getattr(message, "level", "INFO").upper()
            elif isinstance(message, dict):
                msg = message.get("msg", str(message))
                extra = message.get("extra", {})
                level_str = message.get("level", "INFO").upper()
            else:
                msg = str(message)
                extra = {}
                level_str = "INFO"

            level = self.level_map.get(level_str, logging.INFO)
            self.logger.log(level, msg, extra={"mcp_extra": extra})
        except Exception as e:
            self.logger.warning(f"Failed to process MCP log message: {e}")


@dataclass
class DefaultProgressAdapter:
    """
    Handle progress updates from long-running operations.
    """

    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("mcp.progress"))
    callback: Optional[Callable[[float, Optional[float], Optional[str]], Awaitable[None]]] = None

    async def __call__(
        self,
        progress: float,
        total: Optional[float],
        message: Optional[str],
    ) -> None:
        """Handle progress notification from MCP server."""
        if total is not None and total != 0:
            pct = (progress / total) * 100
            self.logger.info("progress=%.1f%% message=%s", pct, message or "")
        else:
            self.logger.info("progress=%s message=%s", progress, message or "")

        # Forward to optional callback
        if self.callback:
            try:
                await self.callback(progress, total, message)
            except Exception as e:
                self.logger.warning(f"Progress callback error: {e}")


@dataclass
class DefaultSamplingAdapter:
    """
    Delegate server-initiated sampling requests to a client-side LLM provider.
    """

    llm: LLMProvider

    async def __call__(
        self,
        messages: list,
        params: Any,
        context: Any,
    ) -> str:
        """Handle sampling request from MCP server."""
        return await self.llm.complete(messages, params, context)


@dataclass
class DefaultElicitationAdapter:
    """
    Handle server-initiated user input requests.

    Default implementation uses the provided get_user_input function
    for CLI-based interaction.
    """

    get_user_input: Callable[[str], str] = field(default=input)
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("mcp.elicitation"))

    async def __call__(
        self,
        message: str,
        response_type: Optional[type],
        params: Any,
        context: Any,
    ) -> Any:
        """Handle elicitation request from MCP server."""
        self.logger.info(f"Elicitation request: {message}")

        if response_type is None:
            return None

        # For CLI usage, prompt user
        try:
            raw = self.get_user_input(f"{message}: ")
            return self._convert_response(raw, response_type)
        except Exception as e:
            self.logger.error(f"Elicitation error: {e}")
            return None

    def _convert_response(self, raw: str, response_type: type) -> Any:
        """Convert raw string input to expected type."""
        # Unwrap Optional if present
        actual_type = _unwrap_optional(response_type)

        if actual_type is str:
            return raw
        elif actual_type is int:
            return int(raw)
        elif actual_type is float:
            return float(raw)
        elif actual_type is bool:
            return raw.lower() in ("true", "yes", "1", "y")
        else:
            # For complex types, return raw string
            return raw


def _unwrap_optional(t: Any) -> Any:
    """Unwrap Optional[T] to T when present."""
    origin = get_origin(t)
    if origin is None:
        return t
    if origin is list or origin is dict:
        return t
    args = get_args(t)
    # Optional[T] is Union[T, NoneType]
    if args and type(None) in args:
        non_none = [a for a in args if a is not type(None)]
        return non_none[0] if len(non_none) == 1 else t
    return t


@dataclass
class ToolCacheInvalidationHandler:
    """
    Message handler hooks for list-changed notifications.

    Calls on_tools_changed when server notifies of tool list updates.
    """

    on_tools_changed: Optional[Callable[[], Awaitable[None]]] = None
    on_resources_changed: Optional[Callable[[], Awaitable[None]]] = None
    on_prompts_changed: Optional[Callable[[], Awaitable[None]]] = None
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("mcp.messages"))

    async def on_tool_list_changed(self, notification: Any) -> None:
        """Handle tool list changed notification."""
        self.logger.info("Received tool list changed notification")
        if self.on_tools_changed:
            await self.on_tools_changed()

    async def on_resource_list_changed(self, notification: Any) -> None:
        """Handle resource list changed notification."""
        self.logger.info("Received resource list changed notification")
        if self.on_resources_changed:
            await self.on_resources_changed()

    async def on_prompt_list_changed(self, notification: Any) -> None:
        """Handle prompt list changed notification."""
        self.logger.info("Received prompt list changed notification")
        if self.on_prompts_changed:
            await self.on_prompts_changed()
