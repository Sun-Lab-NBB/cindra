from pathlib import Path

from PySide6.QtWidgets import QApplication

from .styles import (
    FONTS as FONTS,
    PC_STYLE as PC_STYLE,
    BINARY_STYLE as BINARY_STYLE,
)
from .pc_viewer import PCViewer as PCViewer
from .roi_viewer import ROIViewer as ROIViewer
from .viewer_state import StateWriter as StateWriter
from .binary_viewer import BinaryPlayer as BinaryPlayer
from .viewer_context import (
    ViewerData as ViewerData,
    SingleRecordingData as SingleRecordingData,
)
from .tracking_viewer import TrackingViewer as TrackingViewer

def run_tracking_viewer(
    recording_path: Path, *, dataset: str | None = None, state_path: Path | None = None
) -> None: ...
def run_registration_viewer(recording_path: Path, *, state_path: Path | None = None) -> None: ...
def run_roi_viewer(recording_path: Path, *, dataset: str | None = None, state_path: Path | None = None) -> None: ...
def _get_or_create_application() -> tuple[QApplication, bool]: ...
