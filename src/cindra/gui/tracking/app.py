"""Provides the application entry point for the multi-day tracking viewer GUI."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QApplication

from .tracking_viewer import TrackingViewer
from .context_data import TrackingViewerData

if TYPE_CHECKING:
    from pathlib import Path


def run_tracking_viewer(recording_path: Path) -> None:
    """Launches the standalone multi-day tracking viewer.

    Creates a QApplication, shows the TrackingViewer window, and enters the event loop. The viewer loads multi-day
    tracking data from the provided directory on startup.

    Args:
        recording_path: The path to the root data directory for any cindra-processed recording that makes up the
            visualized multi-day dataset. The loader uses that recording's data to search and reconstruct the full
            dataset hierarchy.
    """
    # Reuses the existing QApplication if one is already running (e.g. when embedded in a larger GUI),
    # otherwise creates a new one.
    application = QApplication.instance()
    owns_application = application is None
    if owns_application:
        application = QApplication(sys.argv)

    # Loads recording data upfront so the viewer window receives a fully populated data instance.
    data = TrackingViewerData.from_recording(root_path=recording_path)

    # Creates the viewer window with the loaded data.
    window = TrackingViewer(data=data)

    window.show()

    # Only enters the event loop if this function created the QApplication. When embedded in a larger GUI, the caller
    # is responsible for running the event loop.
    if owns_application and application is not None:
        sys.exit(application.exec())
