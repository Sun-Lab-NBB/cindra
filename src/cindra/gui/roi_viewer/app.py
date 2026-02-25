"""Provides the application entry point for the standalone read-only ROI viewer GUI."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QApplication

from .viewer import ROIViewer
from .context_data import ROIViewerData

if TYPE_CHECKING:
    from pathlib import Path


def run_roi_viewer(session_path: Path | None = None) -> None:
    """Launches the standalone read-only ROI viewer.

    Creates a QApplication, loads pipeline data from the given session directory (or opens a file
    dialog if no path is provided), shows the ROIViewer window, and enters the event loop.

    Args:
        session_path: Path to a cindra output directory to load on startup. Opens a file dialog
            if None.
    """
    application = QApplication.instance()
    owns_application = application is None
    if owns_application:
        application = QApplication(sys.argv)

    data = ROIViewerData.from_single_day(root_path=session_path) if session_path is not None else None

    window = ROIViewer(data=data)
    window.show()

    if owns_application and application is not None:
        sys.exit(application.exec())
