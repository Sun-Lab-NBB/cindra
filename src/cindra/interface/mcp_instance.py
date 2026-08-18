"""Provides the shared MCP server instance used by all cindra-mcp data-processing tool modules. The acquisition,
configuration, processing, and results tool modules import the ``mcp`` instance from this module and register their
tools through the ``@mcp.tool()`` decorator.
"""

from __future__ import annotations

from mcp.server import MCPServer

mcp: MCPServer = MCPServer(name="cindra-mcp")
"""The MCP server instance that exposes the data-processing tools to AI agents."""
