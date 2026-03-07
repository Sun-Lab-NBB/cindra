"""Provides user-facing entry points for the cindra library, including CLI and MCP server interfaces."""

from .mcp_server import mcp, run_server
from .gui_mcp_server import gui_mcp, run_gui_server

__all__ = ["gui_mcp", "mcp", "run_gui_server", "run_server"]
