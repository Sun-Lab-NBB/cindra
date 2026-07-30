"""Provides the shared MCP server instance used by all cindra-mcp data-processing tool modules.

The acquisition, configuration, processing, and results tool modules import the ``mcp`` instance from this module and
register their tools through the ``@mcp.tool()`` decorator. The ``mcp_server`` module imports all four tool modules at
module level to trigger that registration before starting the server.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="cindra-mcp", json_response=True)
"""The MCP server instance initialized with JSON response mode for structured output."""
