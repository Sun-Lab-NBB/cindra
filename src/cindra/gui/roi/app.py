"""Provides the application entry point for the ROI viewer and editor GUI."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from PySide6 import QtGui, QtCore
from PySide6.QtWidgets import QApplication

from .viewer import MainWindow

if TYPE_CHECKING:
    from pathlib import Path

# String path to the application icon file.
_ICON_PATH: str = str(MainWindow.cindra_directory() / "logo" / "logo.png")


def run(session_path: Path | None = None) -> None:
    """Launches the standalone ROI viewer and editor.

    Creates a QApplication, shows the MainWindow, and enters the event loop.
    If a session path is provided, the viewer loads cindra output data from that
    directory on startup.

    Args:
        session_path: Path to a cindra output directory to load on startup.
    """
    application = QApplication.instance()
    owns_application = application is None
    if owns_application:
        application = QApplication(sys.argv)

    app_icon = QtGui.QIcon()
    for size in (16, 24, 32, 48, 64, 256):
        app_icon.addFile(_ICON_PATH, QtCore.QSize(size, size))
    application.setWindowIcon(app_icon)

    _window = MainWindow(session_path=session_path)

    if owns_application:
        sys.exit(application.exec())
