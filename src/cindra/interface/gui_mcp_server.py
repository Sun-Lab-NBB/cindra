"""Provides the MCP server for managing GUI viewer subprocesses and querying their display state.

Exposes tools that enable AI agents to launch, list, close, and query the current display state of GUI viewer
windows. All data loading and interpretation is handled by the results tools in the non-GUI MCP server; this server
focuses exclusively on viewer lifecycle management and live display state queries.
"""

from __future__ import annotations

import sys
import uuid
from typing import Any, Literal
from pathlib import Path
import subprocess
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP

from ..gui.viewer_state import read_viewer_state, cleanup_state_file, generate_state_path

gui_mcp = FastMCP(name="cindra-gui-mcp", json_response=True)
"""The GUI MCP server instance initialized with JSON response mode for structured output."""


@dataclass
class _ViewerProcess:
    """Tracks a managed GUI viewer subprocess."""

    viewer_id: str
    """The unique identifier for this viewer instance."""
    viewer_type: str
    """The type of viewer ('roi', 'tracking', or 'registration')."""
    recording_path: str
    """The path to the recording loaded in the viewer."""
    dataset: str | None
    """The multi-recording dataset name, or None for single-recording mode."""
    state_path: str
    """The path to the temporary state file used for cross-process state exchange."""
    process: subprocess.Popen[str]
    """The subprocess.Popen instance for the viewer process."""


_viewer_registry: dict[str, _ViewerProcess] = {}
"""Tracks active viewer subprocesses keyed by viewer_id."""


def run_gui_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Starts the GUI MCP server with the specified transport.

    Args:
        transport: The transport type to use ('stdio', 'sse', or 'streamable-http').
    """
    gui_mcp.run(transport=transport)


def _get_viewer(viewer_id: str) -> _ViewerProcess | None:
    """Returns the viewer process for the given ID, cleaning up dead processes and their state files.

    Args:
        viewer_id: The viewer identifier to look up.

    Returns:
        The _ViewerProcess instance, or None if not found or the process has exited.
    """
    entry = _viewer_registry.get(viewer_id)
    if entry is None:
        return None

    if entry.process.poll() is not None:
        cleanup_state_file(state_path=Path(entry.state_path))
        del _viewer_registry[viewer_id]
        return None

    return entry


@gui_mcp.tool()
def launch_viewer_tool(
    viewer_type: Literal["roi", "tracking", "registration"],
    recording_path: str,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Launches a GUI viewer in a subprocess for the user to interact with.

    Spawns the viewer as a child process using the cindra-gui CLI. The viewer window appears on screen for the user
    to interact with directly. Returns a viewer_id that can be used to check status, query state, or close the
    viewer later.

    Args:
        viewer_type: The type of viewer to launch. 'roi' for ROI inspection, 'tracking' for multi-recording tracking
            quality, 'registration' for registration quality (binary player + PC viewer).
        recording_path: Absolute path to the cindra pipeline output directory for the recording to visualize.
        dataset: Multi-recording dataset name to load on startup. Only used by 'roi' and 'tracking' viewers.
    """
    path = Path(recording_path)
    if not path.exists():
        return {"success": False, "error": f"Unable to launch viewer. Path does not exist: {recording_path}"}

    viewer_id = uuid.uuid4().hex[:12]
    state_path = generate_state_path(viewer_id=viewer_id)

    cindra_gui_exe = str(Path(sys.executable).parent / "cindra-gui")
    cmd = [cindra_gui_exe, viewer_type, "--recording-path", str(path), "--state-file", state_path]
    if dataset is not None and viewer_type in ("roi", "tracking"):
        cmd.extend(["--dataset", dataset])

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)  # noqa: S603
    except OSError as error:
        return {"success": False, "error": f"Unable to launch viewer subprocess. {error}"}

    entry = _ViewerProcess(
        viewer_id=viewer_id,
        viewer_type=viewer_type,
        recording_path=recording_path,
        dataset=dataset,
        state_path=state_path,
        process=process,
    )
    _viewer_registry[viewer_id] = entry

    return {
        "success": True,
        "viewer_id": viewer_id,
        "viewer_type": viewer_type,
        "recording_path": recording_path,
        "dataset": dataset,
    }


@gui_mcp.tool()
def list_viewers_tool() -> dict[str, Any]:
    """Lists all active GUI viewer instances managed by this server.

    Returns viewer IDs, types, recording paths, and alive status for each managed viewer. Dead viewers are
    automatically cleaned up.
    """
    viewers: list[dict[str, Any]] = []
    dead_ids: list[str] = []

    for viewer_id, entry in _viewer_registry.items():
        alive = entry.process.poll() is None
        if not alive:
            dead_ids.append(viewer_id)
        viewers.append(
            {
                "viewer_id": viewer_id,
                "viewer_type": entry.viewer_type,
                "recording_path": entry.recording_path,
                "dataset": entry.dataset,
                "alive": alive,
            }
        )

    for dead_id in dead_ids:
        cleanup_state_file(state_path=Path(_viewer_registry[dead_id].state_path))
        del _viewer_registry[dead_id]

    return {"viewers": viewers, "count": len(viewers)}


@gui_mcp.tool()
def close_viewer_tool(viewer_id: str) -> dict[str, Any]:
    """Closes a GUI viewer and terminates its subprocess.

    Terminates the viewer process, waiting briefly for graceful shutdown before forcing termination. Cleans up the
    state file used for cross-process state exchange.

    Args:
        viewer_id: The unique identifier of the viewer to close, as returned by launch_viewer_tool.
    """
    entry = _get_viewer(viewer_id)
    if entry is None:
        return {"success": False, "error": f"Unable to find viewer with id '{viewer_id}'."}

    entry.process.terminate()
    try:
        entry.process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        entry.process.kill()

    cleanup_state_file(state_path=Path(entry.state_path))
    del _viewer_registry[viewer_id]
    return {"success": True, "viewer_id": viewer_id}


@gui_mcp.tool()
def query_viewer_state_tool(viewer_id: str) -> dict[str, Any]:
    """Queries the current display state of an active GUI viewer.

    Returns the viewer's live display settings including active channel, background view, mask layer, selected ROIs,
    opacity, color mode, and other viewer-type-specific state. The state is updated by the viewer subprocess every
    250 ms when changes are detected.

    Args:
        viewer_id: The unique identifier of the viewer to query, as returned by launch_viewer_tool.
    """
    entry = _get_viewer(viewer_id)
    if entry is None:
        return {"success": False, "error": f"Unable to find viewer with id '{viewer_id}'."}

    state_file = Path(entry.state_path)
    if not state_file.exists():
        return {
            "success": True,
            "viewer_id": viewer_id,
            "state": {"loaded": False},
            "note": "Viewer is starting up. State file has not been written yet.",
        }

    try:
        state = read_viewer_state(state_path=state_file)
    except Exception as error:
        return {"success": False, "error": f"Unable to read viewer state. {error}"}

    return {"success": True, "viewer_id": viewer_id, "state": state}
