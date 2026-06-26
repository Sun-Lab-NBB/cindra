from typing import Any, Literal
import subprocess
from dataclasses import dataclass

from _typeshed import Incomplete

from ..gui.viewer_state import (
    read_viewer_state as read_viewer_state,
    cleanup_state_file as cleanup_state_file,
    generate_state_path as generate_state_path,
)

gui_mcp: Incomplete

@dataclass
class _ViewerProcess:
    viewer_id: str
    viewer_type: str
    recording_path: str
    dataset: str | None
    state_path: str
    process: subprocess.Popen[str]

_viewer_registry: dict[str, _ViewerProcess]

def run_gui_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None: ...
def launch_viewer_tool(
    viewer_type: Literal["roi", "tracking", "registration"], recording_path: str, dataset: str | None = None
) -> dict[str, Any]: ...
def list_viewers_tool() -> dict[str, Any]: ...
def close_viewer_tool(viewer_id: str) -> dict[str, Any]: ...
def query_viewer_state_tool(viewer_id: str) -> dict[str, Any]: ...
def _get_viewer(viewer_id: str) -> _ViewerProcess | None: ...
