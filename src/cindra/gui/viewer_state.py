"""Provides cross-process state exchange between GUI viewer subprocesses and the MCP server."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from pathlib import Path
import tempfile

from PySide6 import QtCore
from filelock import FileLock

if TYPE_CHECKING:
    from collections.abc import Callable

_LOCK_TIMEOUT: float = 10.0
"""The maximum number of seconds to wait when acquiring the state file lock."""

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


def write_viewer_state(state_path: Path, state: dict[str, Any]) -> None:
    """Writes viewer state to a JSON file with cross-platform file locking.

    Acquires an exclusive lock before writing to prevent the MCP server from reading a partially written file.

    Args:
        state_path: The path to the state file.
        state: The state dictionary to serialize.
    """
    lock = FileLock(str(state_path) + ".lock")
    with lock.acquire(timeout=_LOCK_TIMEOUT):
        state_path.write_text(json.dumps(state))


def read_viewer_state(state_path: Path) -> dict[str, Any]:
    """Reads viewer state from a JSON file with cross-platform file locking.

    Acquires a lock before reading to prevent reading a partially written file.

    Args:
        state_path: The path to the state file.

    Returns:
        The deserialized state dictionary.
    """
    lock = FileLock(str(state_path) + ".lock")
    with lock.acquire(timeout=_LOCK_TIMEOUT):
        return json.loads(state_path.read_text())


def cleanup_state_file(state_path: Path) -> None:
    """Removes the state file and its associated lock file.

    Args:
        state_path: The path to the state file to clean up.
    """
    for path in (state_path, state_path.with_name(state_path.name + ".lock")):
        path.unlink(missing_ok=True)


class StateWriter(QtCore.QObject):
    """Polls a viewer's state callback and writes to disk when changes are detected.

    Uses a QTimer to periodically call the ``get_state`` callback. Writes to the state file only when the returned
    dictionary differs from the last written state, minimizing disk I/O during idle periods. All writes use
    ``filelock.FileLock`` to coordinate with the MCP server process.

    Args:
        state_path: The path to the state file.
        get_state: A callable that returns the current viewer state dictionary.
        parent: Optional Qt parent object for automatic lifetime management.
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
            write_viewer_state(state_path=self._state_path, state=state)
