import asyncio
from typing import Optional
from contextlib import AsyncExitStack
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client


class MCPClient:
    """MCP Client for interacting with an MCP Streamable HTTP server."""

    def __init__(self, server_url: str | None = None, token: str | None = None):
        self.server_url = server_url
        self._token = token
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self._streams_context = None
        self._session_context = None

    async def connect_to_server(self):
        server_url = self.server_url
        headers = None
        if self._token:
            headers = {"Authorization": f"Bearer {self._token}"}
            print("Authorized call.")
        self._streams_context = streamablehttp_client(url=server_url, headers=headers)
        read_stream, write_stream, _ = await self._streams_context.__aenter__()
        self._session_context = ClientSession(read_stream, write_stream)
        self.session = await self._session_context.__aenter__()
        await self.session.initialize()
        response = await self.session.list_tools()
        return response.tools

    async def cleanup(self):
        if self._session_context:
            await self._session_context.__aexit__(None, None, None)
        if self._streams_context:
            await self._streams_context.__aexit__(None, None, None)

    @classmethod
    def from_dict(cls, config):
        return cls(config["host"], config["port"])  # legacy

    def set_token(self, token: str | None):
        """Set or update the bearer token for subsequent connections."""
        self._token = token
