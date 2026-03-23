"""Provides the MCP server entry point for agentic neural imaging data processing.

Imports all tool modules at module level to trigger ``@mcp.tool()`` registration on the shared instance from
``mcp_instance``. The ``run_server`` function starts the server with all registered tools available.
"""

from __future__ import annotations

from typing import Literal

from . import results_tools, processing_tools, acquisition_tools, configuration_tools  # noqa: F401
from .mcp_instance import mcp


def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Starts the MCP server with the specified transport.

    Args:
        transport: The transport type to use ('stdio', 'sse', or 'streamable-http').
    """
    mcp.run(transport=transport)
