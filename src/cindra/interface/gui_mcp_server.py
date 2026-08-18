"""Provides the MCP server for managing GUI viewer subprocesses and querying their display state.

Exposes tools that enable AI agents to launch, list, close, and query the current display state of GUI viewer
windows. All data loading and interpretation is handled by the results tools in the non-GUI MCP server. This server
focuses exclusively on viewer lifecycle management and live display state queries.
"""

from __future__ import annotations

import sys
import uuid
from typing import Any, Literal
from pathlib import Path
from threading import RLock
import subprocess
from dataclasses import dataclass

from mcp.server import MCPServer

from ..gui import read_viewer_state, cleanup_state_file, generate_state_path

gui_mcp: MCPServer = MCPServer(name="cindra-gui-mcp")
"""The GUI MCP server instance that exposes the viewer lifecycle tools to AI agents."""


@dataclass(slots=True)
class _ViewerProcess:
    """Tracks a managed GUI viewer subprocess."""

    viewer_id: str
    """The unique identifier for this viewer instance."""
    viewer_type: Literal["roi", "tracking", "registration"]
    """The kind of viewer this process renders."""
    recording_path: str
    """The path to the recording loaded in the viewer."""
    dataset: str | None
    """The multi-recording dataset name, or None for single-recording mode."""
    state_path: str
    """The path to the temporary state file used for cross-process state exchange."""
    process: subprocess.Popen[str]
    """The running child process hosting the viewer window."""


_viewer_registry: dict[str, _ViewerProcess] = {}
"""Tracks active viewer subprocesses keyed by viewer_id."""

_registry_lock: RLock = RLock()
"""Serializes every read and mutation of the viewer registry.

Notes:
    The MCP server runs each synchronous tool body in one of its own worker threads, so two overlapping viewer calls
    reach the registry at once. The lock is reentrant, because the tools acquire it around compound sequences that
    call the registry helpers, which acquire it in turn.
"""


def run_gui_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Starts the GUI MCP server with the specified transport.

    Args:
        transport: The transport type to use ('stdio', 'sse', or 'streamable-http').
    """
    if transport == "streamable-http":
        # Frames each response as a single JSON body instead of an event stream. Only the streamable-http transport
        # accepts this flag, so it stays out of the call below.
        gui_mcp.run(transport=transport, json_response=True)
        return

    gui_mcp.run(transport=transport)


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

    Note:
        This server only launches and observes viewers. Changing the display, such as selecting an ROI or navigating
        frames, must be done by the user via the GUI controls.

    Args:
        viewer_type: The type of viewer to launch. 'roi' for ROI inspection, 'tracking' for multi-recording tracking
            quality, 'registration' for registration quality (binary player + PC viewer).
        recording_path: Absolute path to the cindra pipeline output directory for the recording to visualize.
        dataset: Multi-recording dataset name to load on startup. Only used by 'roi' and 'tracking' viewers.

    Returns:
        A JSON dictionary containing 'success' flag, and on success 'viewer_id', 'viewer_type', 'recording_path',
        and 'dataset'. On failure, contains an 'error' message.
    """
    path = Path(recording_path)
    if not path.exists():
        return {"success": False, "error": f"Unable to launch viewer. Path does not exist: {recording_path}"}

    # The spawned command accepts a directory alone, and its usage error is written to a stream this server discards,
    # so the requirement is enforced here instead.
    if not path.is_dir():
        return {
            "success": False,
            "error": f"Unable to launch viewer. Path is not a cindra output directory: {recording_path}",
        }

    viewer_id = uuid.uuid4().hex[:12]
    state_path = generate_state_path(viewer_id=viewer_id)

    cindra_gui_executable = str(Path(sys.executable).parent / "cindra-gui")
    command = [cindra_gui_executable, viewer_type, "--recording-path", str(path), "--state-file", state_path]
    if dataset is not None and viewer_type in ("roi", "tracking"):
        command.extend(["--dataset", dataset])

    try:
        # The executable resolves from sys.executable, and no shell is used.
        process = subprocess.Popen(
            args=command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
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
    with _registry_lock:
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

    Returns viewer IDs, types, recording paths, alive status, and live active dataset for each managed viewer. Dead
    viewers are automatically cleaned up. The ``active_dataset`` field reflects the dataset currently displayed by the
    viewer, which may differ from the ``dataset`` value provided at launch if the user switched datasets inside the
    viewer.

    Returns:
        A JSON dictionary containing 'success' flag, 'viewers' list (each with 'viewer_id', 'viewer_type',
        'recording_path', 'dataset', 'active_dataset', and 'alive' flag), and 'count' of listed viewers. A viewer
        whose process has exited is listed once with 'alive' set to False and is then dropped from the registry, so
        'count' can exceed the number of live viewers on that one call.
    """
    viewers: list[dict[str, Any]] = []
    dead_entries: list[_ViewerProcess] = []

    # Holds the lock across the whole sweep, so a viewer another thread launches or closes midway cannot invalidate
    # the iteration or strand a reaped entry.
    with _registry_lock:
        for viewer_id, entry in _viewer_registry.items():
            alive = entry.process.poll() is None
            if not alive:
                dead_entries.append(entry)

            viewer_info: dict[str, Any] = {
                "viewer_id": viewer_id,
                "viewer_type": entry.viewer_type,
                "recording_path": entry.recording_path,
                "dataset": entry.dataset,
                "alive": alive,
                "active_dataset": None,
            }

            if alive:
                state_file = Path(entry.state_path)
                if state_file.exists():
                    try:
                        state = read_viewer_state(state_path=state_file)
                        viewer_info["active_dataset"] = state.get("active_dataset")
                    except Exception:  # noqa: S110 - Best-effort state read. The viewer list should not fail.
                        pass

            viewers.append(viewer_info)

        for dead_entry in dead_entries:
            cleanup_state_file(state_path=Path(dead_entry.state_path))
            _viewer_registry.pop(dead_entry.viewer_id, None)

    return {"success": True, "viewers": viewers, "count": len(viewers)}


@gui_mcp.tool()
def close_viewer_tool(viewer_id: str) -> dict[str, Any]:
    """Closes a GUI viewer and terminates its subprocess.

    Terminates the viewer process, waiting briefly for graceful shutdown before forcing termination. Cleans up the
    state file used for cross-process state exchange.

    Args:
        viewer_id: The unique identifier of the viewer to close, as returned by launch_viewer_tool.

    Returns:
        A JSON dictionary containing 'success' flag and 'viewer_id' on success. On failure, contains an 'error'
        message.
    """
    # Claims the entry under the lock and drops it from the registry before the shutdown wait, so exactly one caller
    # terminates a given viewer and the wait itself blocks no other tool.
    with _registry_lock:
        entry = _get_viewer(viewer_id)
        if entry is None:
            return {"success": False, "error": f"Unable to find viewer with id '{viewer_id}'."}
        del _viewer_registry[viewer_id]

    entry.process.terminate()
    try:
        entry.process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        entry.process.kill()

    cleanup_state_file(state_path=Path(entry.state_path))
    return {"success": True, "viewer_id": viewer_id}


@gui_mcp.tool()
def query_viewer_state_tool(viewer_id: str) -> dict[str, Any]:
    """Queries the current display state of an active GUI viewer.

    The shape of the returned 'state' dictionary depends on viewer_type. The 'roi' and 'tracking' viewers return flat
    display-setting dictionaries. The 'registration' viewer instead returns a nested dictionary keyed by 'binary_player'
    and 'pc_viewer', with no top-level 'loaded' flag. For the exact per-viewer-type key schema, consult the cindra
    visualization skill's "Viewer state reference" section. The state is updated by the viewer subprocess every 250 ms
    when changes are detected.

    Note:
        This tool only observes viewer state and cannot change the display. Selecting an ROI, switching the channel,
        plane, or mask layer, and navigating frames must be done by the user via the GUI controls.

    Args:
        viewer_id: The unique identifier of the viewer to query, as returned by launch_viewer_tool.

    Returns:
        A JSON dictionary containing 'success' flag, 'viewer_id', and 'state' dictionary holding the viewer's current
        display settings in the viewer-type-specific shape described above. While the viewer is still starting up and
        has not written its state file, 'state' is the placeholder {'loaded': False} for every viewer type and an
        extra 'note' key reports the startup delay. On failure, contains an 'error' message.
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


def _get_viewer(viewer_id: str) -> _ViewerProcess | None:
    """Returns the viewer process for the given ID, cleaning up dead processes and their state files.

    Args:
        viewer_id: The viewer identifier to look up.

    Returns:
        The _ViewerProcess instance, or None if not found or the process has exited.
    """
    with _registry_lock:
        entry = _viewer_registry.get(viewer_id)
        if entry is None:
            return None

        if entry.process.poll() is not None:
            cleanup_state_file(state_path=Path(entry.state_path))
            del _viewer_registry[viewer_id]
            return None

        return entry
