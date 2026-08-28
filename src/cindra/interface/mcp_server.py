"""Provides the MCP server entry point for agentic neural imaging data processing. Imports all tool modules at module
level to trigger ``@mcp.tool()`` registration on the shared instance from ``mcp_instance``.
"""

from __future__ import annotations

from typing import Literal

# Imports the modules themselves rather than the names they declare, because importing a module is what runs its
# @mcp.tool() decorators and registers its tools on the shared instance.
from . import results_tools, processing_tools, acquisition_tools, configuration_tools  # noqa: F401
from .mcp_instance import mcp


def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Starts the MCP server with the specified transport.

    Args:
        transport: The transport type to use ('stdio', 'sse', or 'streamable-http').
    """
    if transport == "streamable-http":
        # Frames each response as a single JSON body instead of an event stream. Only the streamable-http transport
        # accepts this flag, so it stays out of the call below.
        mcp.run(transport=transport, json_response=True)
        return

    mcp.run(transport=transport)
