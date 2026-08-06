"""
MCP Client Configuration

Standardized configuration for building FastMCP clients.
Supports timeouts, auth, roots, and extra kwargs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional


# Convention: MCPConfig has top-level "mcpServers" for multi-server mode
MCPConfig = dict[str, Any]


@dataclass(frozen=True)
class ClientBuildOptions:
    """
    Options for constructing a FastMCP Client in a consistent way.

    Parameters
    ----------
    timeout_s:
        Default request timeout passed to fastmcp.Client.
    init_timeout_s:
        Timeout for initial connection/initialize handshake. Set to 0 to disable.
    roots:
        Either a static list of filesystem roots or a roots handler callback.
        Used to tell servers what local resources the client can access.
    auth:
        For HTTP transports: can be "oauth", a bearer token string, or an auth helper
        object (OAuth/BearerAuth) that implements httpx.Auth.
    extra:
        Escape hatch for any extra keyword args supported by fastmcp.Client.
    """

    timeout_s: Optional[float] = 30.0
    init_timeout_s: Optional[float] = None
    roots: Any = None
    auth: Any = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, prefix: str = "MCP") -> "ClientBuildOptions":
        """
        Load options from environment variables.

        Environment variables:
        - {prefix}_TIMEOUT_S: Request timeout in seconds
        - {prefix}_INIT_TIMEOUT_S: Init timeout in seconds
        """
        timeout_s = float(os.getenv(f"{prefix}_TIMEOUT_S", "30"))
        init_timeout_s_str = os.getenv(f"{prefix}_INIT_TIMEOUT_S", "")
        init_timeout_s = float(init_timeout_s_str) if init_timeout_s_str else None

        return cls(
            timeout_s=timeout_s,
            init_timeout_s=init_timeout_s,
        )


def is_mcp_config(obj: Any) -> bool:
    """
    Detect whether `obj` looks like an MCPConfig dictionary.

    FastMCP supports creating a client from a config dict that includes
    `mcpServers`, enabling multi-server usage with namespaced tool names.
    """
    return isinstance(obj, dict) and "mcpServers" in obj


class MCPSettings:
    """
    MCP-specific settings loaded from environment.
    """

    def __init__(self):
        # MCP Server URLs
        self.MCP_JIRA_URL: str = os.getenv("MCP_JIRA_URL", "")
        self.MCP_ADO_URL: str = os.getenv("MCP_ADO_URL", "")
        self.MCP_DEFAULT_URL: str = os.getenv("MCP_SERVER_URL", "")

        # Timeouts
        self.MCP_TIMEOUT_S: float = float(os.getenv("MCP_TIMEOUT_S", "30"))
        self.MCP_INIT_TIMEOUT_S: Optional[float] = None
        init_timeout_str = os.getenv("MCP_INIT_TIMEOUT_S", "")
        if init_timeout_str:
            self.MCP_INIT_TIMEOUT_S = float(init_timeout_str)

        # Cache TTL
        self.MCP_CACHE_TTL_S: float = float(os.getenv("MCP_CACHE_TTL_S", "180"))

        # Auth (optional)
        self.MCP_AUTH_TOKEN: str = os.getenv("MCP_AUTH_TOKEN", "")
        self.MCP_SUBSCRIPTION_KEY: str = os.getenv("MCP_SUBSCRIPTION_KEY", "")

    def get_default_options(self) -> ClientBuildOptions:
        """Get default ClientBuildOptions from environment."""
        auth = None
        if self.MCP_AUTH_TOKEN:
            auth = self.MCP_AUTH_TOKEN

        return ClientBuildOptions(
            timeout_s=self.MCP_TIMEOUT_S,
            init_timeout_s=self.MCP_INIT_TIMEOUT_S,
            auth=auth,
        )


# Global MCP settings instance
mcp_settings = MCPSettings()
