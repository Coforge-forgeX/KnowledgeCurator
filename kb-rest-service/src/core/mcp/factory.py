"""
MCP Client Factory

Builds configured FastMCP clients with proper handler wiring.

FastMCP Client construction rules:
- You MUST use the async context manager (async with client) to manage connection lifecycle.
- Transport can be inferred from input:
  - FastMCP instance -> in-memory transport
  - .py or .js script -> stdio transport
  - http/https URL -> HTTP transport
  - config dict w/ mcpServers -> multi-server client
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import ClientBuildOptions, is_mcp_config
from .handlers import LogHandler, ProgressHandler, SamplingHandler, ElicitationHandler

logger = logging.getLogger("core.mcp.factory")


@dataclass
class ClientFactory:
    """
    A thin factory that standardizes creation of fastmcp.Client.

    We keep this intentionally small:
    - It is easier to test.
    - It keeps "policy" (timeouts, handlers, auth) separate from "runtime" usage.
    """

    logger: logging.Logger = field(default_factory=lambda: logging.getLogger("core.mcp.factory"))

    def build(
        self,
        source: Any,
        *,
        options: ClientBuildOptions,
        log_handler: Optional[LogHandler] = None,
        progress_handler: Optional[ProgressHandler] = None,
        sampling_handler: Optional[SamplingHandler] = None,
        elicitation_handler: Optional[ElicitationHandler] = None,
        message_handler: Any = None,
    ) -> Any:
        """
        Build and return a fastmcp.Client.

        Parameters map to FastMCP's Client constructor:
        - timeout / init_timeout: request + init timeouts
        - roots: static or dynamic roots provider
        - auth: OAuth or bearer token for HTTP transports
        - log/progress/sampling/message/elicitation handlers for advanced interaction

        Args:
            source: MCP server URL, script path, FastMCP instance, or config dict
            options: ClientBuildOptions with timeouts, auth, etc.
            log_handler: Handler for server log messages
            progress_handler: Handler for progress notifications
            sampling_handler: Handler for server-initiated LLM sampling
            elicitation_handler: Handler for server-initiated user input
            message_handler: Handler for list-changed notifications

        Returns:
            Configured fastmcp.Client instance
        """
        # Lazy import to avoid import errors when fastmcp is not installed
        try:
            from fastmcp import Client
        except ImportError:
            raise ImportError(
                "fastmcp is required for MCP client functionality. "
                "Install it with: pip install fastmcp"
            )

        kwargs = dict(options.extra or {})

        self.logger.debug(f"Building FastMCP Client with source={source} and options={options}")

        if options.timeout_s is not None:
            kwargs["timeout"] = options.timeout_s
        if options.init_timeout_s is not None:
            kwargs["init_timeout"] = options.init_timeout_s
        if options.roots is not None:
            kwargs["roots"] = options.roots
        if options.auth is not None:
            kwargs["auth"] = options.auth

        # Advanced callbacks (all optional):
        if log_handler is not None:
            kwargs["log_handler"] = log_handler
        if progress_handler is not None:
            kwargs["progress_handler"] = progress_handler
        if sampling_handler is not None:
            kwargs["sampling_handler"] = sampling_handler
        if elicitation_handler is not None:
            kwargs["elicitation_handler"] = elicitation_handler
        if message_handler is not None:
            kwargs["message_handler"] = message_handler

        # Multi-server: config dict with mcpServers is passed directly to Client()
        if is_mcp_config(source):
            self.logger.info("Detected MCP config dict, using multi-server client mode.")
            return Client(source, **kwargs)

        # Otherwise: FastMCP infers the appropriate transport from source
        self.logger.info("Inferring transport for source and building client.")
        return Client(source, **kwargs)

    def build_with_headers(
        self,
        url: str,
        headers: dict[str, str],
        *,
        options: ClientBuildOptions,
        **handler_kwargs,
    ) -> Any:
        """
        Build a client for HTTP transport with custom headers.

        Useful for API gateway authentication (subscription keys, etc.)

        Args:
            url: MCP server HTTP URL
            headers: HTTP headers to include in requests
            options: ClientBuildOptions with timeouts, auth, etc.
            **handler_kwargs: Handler arguments passed to build()

        Returns:
            Configured fastmcp.Client instance
        """
        try:
            from fastmcp.client.transports import StreamableHttpTransport
        except ImportError:
            raise ImportError(
                "fastmcp is required for MCP client functionality. "
                "Install it with: pip install fastmcp"
            )

        source = StreamableHttpTransport(url=url, headers=headers)
        return self.build(source, options=options, **handler_kwargs)
