"""Provides user-facing entry points for the cindra library, including CLI and MCP server interfaces."""

from .mcp_server import mcp, run_server

__all__ = ["mcp", "run_server"]
