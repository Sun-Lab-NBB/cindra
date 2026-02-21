"""Provides the application entry point for the single-day registration viewer GUI."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QApplication

from .viewer import BinaryPlayer

if TYPE_CHECKING:
    from pathlib import Path


def run(session_path: Path | None = None) -> None:  # noqa: ARG001
    """Launches the standalone single-day registration viewer.

    Creates a QApplication, shows the BinaryPlayer window, and enters the event loop.
    If a session path is provided, the viewer loads registration data from that directory
    on startup.

    Args:
        session_path: Optional path to a suite2p output directory to load on startup.
    """
    application = QApplication.instance()
    owns_application = application is None
    if owns_application:
        application = QApplication(sys.argv)

    window = BinaryPlayer()
    window.show()

    if owns_application:
        sys.exit(application.exec())
