"""Provides the application entry point for the single-day registration viewer GUI."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from PySide6.QtWidgets import QApplication

from .viewer import PCViewer, BinaryPlayer
from .context_data import RegistrationViewerData

if TYPE_CHECKING:
    from pathlib import Path


def run(session_path: Path | None = None) -> None:
    """Launches the standalone single-day registration viewer.

    Creates a QApplication, shows the BinaryPlayer and PCViewer windows, and enters the event
    loop. If a session path is provided, both viewers load registration data from that directory
    on startup and share the same data model for synchronized plane switching.

    Args:
        session_path: Optional path to a suite2p output directory to load on startup.
    """
    application = QApplication.instance()
    owns_application = application is None
    if owns_application:
        application = QApplication(sys.argv)

    data: RegistrationViewerData | None = None
    if session_path is not None:
        data = RegistrationViewerData.from_session(root_path=session_path)

    binary_player = BinaryPlayer(data=data)
    pc_viewer = PCViewer(data=data)

    # Synchronizes plane changes from the binary player to the PC viewer.
    binary_player.plane_changed.connect(lambda _index: pc_viewer.load_data(data=binary_player._data))

    binary_player.show()
    pc_viewer.show()

    if owns_application:
        sys.exit(application.exec())
