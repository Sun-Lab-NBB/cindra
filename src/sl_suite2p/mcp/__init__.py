"""Provides the MCP (Model Context Protocol) server implementation for sl-suite2p.

The MCP server exposes sl-suite2p pipeline functionality as tools that can be accessed by AI assistants.
"""

from .server import mcp, run_server

__all__ = ["mcp", "run_server"]
