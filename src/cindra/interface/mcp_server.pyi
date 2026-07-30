from typing import Literal

from . import (
    results_tools as results_tools,
    processing_tools as processing_tools,
    acquisition_tools as acquisition_tools,
    configuration_tools as configuration_tools,
)
from .mcp_instance import mcp as mcp

def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None: ...
