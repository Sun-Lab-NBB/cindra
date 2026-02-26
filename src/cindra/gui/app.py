"""Provides application entry points for launching all cindra GUI viewers."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from PySide6 import QtGui, QtCore
from PySide6.QtWidgets import QApplication

from .tracking_viewer import TrackingViewer
from .binary_viewer import BinaryPlayer
from .pc_viewer import PCViewer
from .roi_viewer import ROIViewer
from .roi_editor.viewer import ROIEditor
from .multi_day_context import TrackingViewerData
from .single_day_context import ROIViewerData, RegistrationViewerData

if TYPE_CHECKING:
    from pathlib import Path

_ICON_PATH: str = str(ROIEditor.cindra_directory() / "logo" / "logo.png")
"""The string path to the application icon file."""


def run_tracking_viewer(recording_path: Path) -> None:
    """Launches the standalone multi-day tracking viewer.

    Creates a QApplication, shows the TrackingViewer window, and enters the event loop. The viewer
    loads multi-day tracking data from the provided directory on startup.

    Args:
        recording_path: The path to the root data directory for any cindra-processed recording that
            makes up the visualized multi-day dataset. The loader uses that recording's data to
            search and reconstruct the full dataset hierarchy.
    """
    # Reuses the existing QApplication if one is already running (e.g. when embedded in a larger
    # GUI), otherwise creates a new one.
    application = QApplication.instance()
    owns_application = application is None
    if owns_application:
        application = QApplication(sys.argv)

    # Loads recording data upfront so the viewer window receives a fully populated data instance.
    data = TrackingViewerData.from_recording(root_path=recording_path)

    # Creates the viewer window with the loaded data.
    window = TrackingViewer(data=data)

    window.show()

    # Only enters the event loop if this function created the QApplication. When embedded in a
    # larger GUI, the caller is responsible for running the event loop.
    if owns_application and application is not None:
        sys.exit(application.exec())


def run_registration_viewer(recording_path: Path) -> None:
    """Launches the standalone single-day registration viewer.

    Creates a QApplication, shows the BinaryPlayer and PCViewer windows, and enters the event
    loop. Both viewers load registration data from the provided directory on startup and share the
    same RegistrationViewerData instance for synchronized plane switching.

    Args:
        recording_path: The Path to a cindra-processed recording's root data directory containing
            registration results.
    """
    # Reuses the existing QApplication if one is already running (e.g. when embedded in a larger
    # GUI), otherwise creates a new one.
    application = QApplication.instance()
    owns_application = application is None
    if owns_application:
        application = QApplication(sys.argv)

    # Loads recording data upfront so both viewer windows share the same RegistrationViewerData
    # instance. This ensures plane switches in the binary player are reflected in the PC viewer
    # without reloading from disk.
    data = RegistrationViewerData.from_recording(root_path=recording_path)

    # Creates both viewer windows with the shared RegistrationViewerData instance.
    binary_player = BinaryPlayer(data=data)
    pc_viewer = PCViewer(data=data)

    # When the user switches planes in the binary player, the shared RegistrationViewerData
    # instance is mutated in place (via switch_plane). This signal connection triggers the PC
    # viewer to re-read PC images and metrics from the updated recording data so both windows
    # stay synchronized.
    binary_player.plane_changed.connect(lambda _index: pc_viewer.load_data(data=binary_player.data))

    # Links the two windows so closing either one closes the other. WA_DeleteOnClose ensures the
    # Qt object is destroyed on close, which emits the destroyed signal that triggers close() on
    # the partner window.
    binary_player.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
    pc_viewer.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
    binary_player.destroyed.connect(pc_viewer.close)
    pc_viewer.destroyed.connect(binary_player.close)

    binary_player.show()
    pc_viewer.show()

    # Only enters the event loop if this function created the QApplication. When embedded in a
    # larger GUI, the caller is responsible for running the event loop.
    if owns_application and application is not None:
        sys.exit(application.exec())


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


def run_roi_editor(session_path: Path | None = None) -> None:
    """Launches the standalone ROI viewer and editor.

    Creates a QApplication, shows the ROIEditor window, and enters the event loop. If a session
    path is provided, the editor loads cindra output data from that directory on startup.

    Args:
        session_path: Path to a cindra output directory to load on startup.
    """
    application = QApplication.instance()
    owns_application = application is None
    if owns_application:
        application = QApplication(sys.argv)
    assert isinstance(application, QApplication)

    app_icon = QtGui.QIcon()
    for size in (16, 24, 32, 48, 64, 256):
        app_icon.addFile(_ICON_PATH, QtCore.QSize(size, size))
    application.setWindowIcon(app_icon)

    _window = ROIEditor(session_path=session_path)

    if owns_application:
        sys.exit(application.exec())
