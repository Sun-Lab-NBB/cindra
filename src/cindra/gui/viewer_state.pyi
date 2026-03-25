from typing import Any
from pathlib import Path
from collections.abc import Callable as Callable

from PySide6 import QtCore

_LOCK_TIMEOUT: float
_POLL_INTERVAL_MILLISECONDS: int

def generate_state_path(viewer_id: str) -> str: ...
def write_viewer_state(state_path: Path, state: dict[str, Any]) -> None: ...
def read_viewer_state(state_path: Path) -> dict[str, Any]: ...
def cleanup_state_file(state_path: Path) -> None: ...

class StateWriter(QtCore.QObject):
    _state_path: Path
    _get_state: Callable[[], dict[str, Any]]
    _last_state: dict[str, Any] | None
    _timer: QtCore.QTimer
    def __init__(
        self, state_path: Path, get_state: Callable[[], dict[str, Any]], parent: QtCore.QObject | None = None
    ) -> None: ...
    def _check_and_write(self) -> None: ...
