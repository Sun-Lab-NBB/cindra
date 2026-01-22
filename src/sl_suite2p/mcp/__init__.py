"""This module provides the MCP (Model Context Protocol) server implementation for sl-suite2p.

The MCP server exposes sl-suite2p pipeline functionality as tools and resources that can be accessed
by AI assistants like Claude.
"""

from .server import server, run_server, create_server

__all__ = ["create_server", "run_server", "server"]
