"""Provides the shared MCP server instance used by all cindra tool modules.

Each tool module (e.g., ``configuration_tools``, ``processing_tools``) imports the ``mcp`` instance from this module
and registers tools via the ``@mcp.tool()`` decorator. The ``mcp_server`` module imports all tool modules at module
level to trigger registration before starting the server.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="cindra-mcp", json_response=True)
"""The MCP server instance initialized with JSON response mode for structured output."""
