from typing import Any, Literal
from threading import RLock
import subprocess
from dataclasses import dataclass

from mcp.server import MCPServer

from ..gui import (
    read_viewer_state as read_viewer_state,
    cleanup_state_file as cleanup_state_file,
    generate_state_path as generate_state_path,
)

_gui_mcp: MCPServer

@dataclass(slots=True)
class _ViewerProcess:
    viewer_id: str
    viewer_type: Literal["roi", "tracking", "registration"]
    output_root: str
    dataset: str | None
    state_path: str
    process: subprocess.Popen[str]

_viewer_registry: dict[str, _ViewerProcess]
_registry_lock: RLock

def run_gui_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None: ...
def launch_viewer_tool(
    viewer_type: Literal["roi", "tracking", "registration"], output_root: str, dataset: str | None = None
) -> dict[str, Any]: ...
def list_viewers_tool() -> dict[str, Any]: ...
def close_viewer_tool(viewer_id: str) -> dict[str, Any]: ...
def query_viewer_state_tool(viewer_id: str) -> dict[str, Any]: ...
def _get_viewer(viewer_id: str) -> _ViewerProcess | None: ...
