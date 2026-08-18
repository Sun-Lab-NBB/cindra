"""Provides cross-process state exchange between GUI viewer subprocesses and the MCP server."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from pathlib import Path
import tempfile

from PySide6 import QtCore
from ataraxis_data_structures import atomic_write

if TYPE_CHECKING:
    from collections.abc import Callable

_POLL_INTERVAL_MILLISECONDS: int = 250
"""The polling interval in milliseconds for the StateWriter to check for state changes."""


def generate_state_path(viewer_id: str) -> str:
    """Generates a temporary file path for viewer state exchange.

    Args:
        viewer_id: The unique identifier for the viewer instance.

    Returns:
        The absolute path string to the temporary state file.
    """
    return str(Path(tempfile.gettempdir()) / f"cindra-gui-{viewer_id}.json")


def read_viewer_state(state_path: Path) -> dict[str, Any]:
    """Reads viewer state from a JSON file.

    Args:
        state_path: The path to the state file.

    Returns:
        The deserialized state dictionary.
    """
    return json.loads(state_path.read_text(encoding="utf-8"))


def cleanup_state_file(state_path: Path) -> None:
    """Removes the state file and any temporary file a killed writer left beside it.

    Args:
        state_path: The path to the state file to clean up.
    """
    state_path.unlink(missing_ok=True)
    for temporary_path in state_path.parent.glob(f".{state_path.name}.*.tmp"):
        temporary_path.unlink(missing_ok=True)


class StateWriter(QtCore.QObject):
    """Polls a viewer's state callback and writes to disk when changes are detected.

    Uses a QTimer to periodically call the ``get_state`` callback. Writes to the state file only when the returned
    dictionary differs from the last written state, minimizing disk I/O during idle periods. Each write publishes
    through a rename, so the MCP server process never reads a half-written snapshot.

    Args:
        state_path: The path to the state file.
        get_state: A callable that returns the current viewer state dictionary.
        parent: Optional Qt parent object for automatic lifetime management.

    Attributes:
        _state_path: Cached path to the state file.
        _get_state: Cached callback that returns the current viewer state.
        _last_state: The most recently written state snapshot, or None before the first write.
        _timer: Timer driving the periodic state polling.
    """

    def __init__(
        self,
        state_path: Path,
        get_state: Callable[[], dict[str, Any]],
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._state_path: Path = state_path
        self._get_state: Callable[[], dict[str, Any]] = get_state
        self._last_state: dict[str, Any] | None = None
        self._timer: QtCore.QTimer = QtCore.QTimer(self)
        self._timer.timeout.connect(self._check_and_write)
        self._timer.start(_POLL_INTERVAL_MILLISECONDS)

        # Writes initial state immediately so the MCP server can query right after launch.
        self._check_and_write()

    def _check_and_write(self) -> None:
        """Compares current state against the last written snapshot and writes to disk if changed."""
        state = self._get_state()
        if state != self._last_state:
            self._last_state = state
            _write_viewer_state(state_path=self._state_path, state=state)


def _write_viewer_state(state_path: Path, state: dict[str, Any]) -> None:
    """Writes viewer state to a JSON file that the MCP server reads concurrently.

    The state reaches its path through a rename, so the MCP server observes either the previous snapshot or the
    complete new one.

    Args:
        state_path: The path to the state file.
        state: The state dictionary to serialize.
    """
    with atomic_write(file_path=state_path) as state_file:
        json.dump(state, state_file)
