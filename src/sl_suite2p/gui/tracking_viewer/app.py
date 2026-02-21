"""Provides the application entry point for the multi-day tracking viewer GUI."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QApplication

from .viewer import TrackingViewer
from .context_data import TrackingViewerData

if TYPE_CHECKING:
    from pathlib import Path


def run(session_path: Path | None = None) -> None:
    """Launches the standalone multi-day tracking viewer.

    Creates a QApplication, shows the TrackingViewer window, and enters the event loop.
    If a session path is provided, the viewer loads multi-day tracking data from that
    directory on startup.

    Args:
        session_path: Path to any session's root processed data directory. The loader
            searches recursively for multi-day runtime data and reconstructs the full
            dataset hierarchy.
    """
    application = QApplication.instance()
    owns_application = application is None
    if owns_application:
        application = QApplication(sys.argv)

    window = TrackingViewer()

    if session_path is not None:
        data = TrackingViewerData.from_session(root_path=session_path)
        window.load_data(data=data)

    window.show()

    if owns_application:
        sys.exit(application.exec())
