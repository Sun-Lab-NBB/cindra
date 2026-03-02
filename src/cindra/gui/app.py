"""Provides application entry points for launching all cindra GUI viewers."""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6 import QtGui, QtCore
from PySide6.QtWidgets import QApplication
from ataraxis_base_utilities import console

import cindra

from .styles import FONTS, PC_STYLE, BINARY_STYLE
from .pc_viewer import PCViewer
from .roi_viewer import ROIViewer
from .binary_viewer import BinaryPlayer
from .viewer_context import ViewerData, SingleDayData
from .tracking_viewer import TrackingViewer

_ICON_PATH: str = str(Path(cindra.__file__).parent / "logo" / "logo.png")
"""The string path to the application icon file."""


def run_tracking_viewer(recording_path: Path, *, dataset: str | None = None) -> None:
    """Launches the standalone multi-day tracking viewer.

    Creates a QApplication, shows the TrackingViewer window, and enters the event loop. The viewer
    loads multi-day tracking data from the provided directory on startup.

    Args:
        recording_path: The path to the root data directory for any cindra-processed recording that
            makes up the visualized multi-day dataset. The loader uses that recording's data to
            search and reconstruct the full dataset hierarchy.
        dataset: Multi-day dataset name to load. Defaults to the first available dataset.
    """
    # Reuses the existing QApplication if one is already running (e.g. when embedded in a larger
    # GUI), otherwise creates a new one.
    application = QApplication.instance()
    owns_application = application is None
    if owns_application:
        application = QApplication(sys.argv)
        application.setFont(FONTS.small)

    # Loads recording data upfront so the viewer window receives a fully populated data instance.
    data = ViewerData.from_data(root_path=recording_path, dataset=dataset)

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
    same SingleDayData instance for synchronized plane switching.

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
        application.setFont(FONTS.small)

    # Loads recording data upfront so both viewer windows share the same SingleDayData
    # instance. This ensures plane switches in the binary player are reflected in the PC viewer
    # without reloading from disk.
    data = SingleDayData.from_data(root_path=recording_path, view_index=0)

    # Creates both viewer windows with the shared SingleDayData instance.
    binary_player = BinaryPlayer(data=data)
    pc_viewer = PCViewer(data=data)

    # Computes screen-adaptive window positions. On large screens the two viewers sit side by side;
    # on smaller screens they cascade with a visible offset so both title bars remain accessible.
    screen = application.primaryScreen()  # type: ignore[union-attr]
    available = screen.availableGeometry()

    # Top-left corner of the binary viewer, in pixels from the screen's top-left corner.
    binary_viewer_x, binary_viewer_y = 50, 50
    # Default window dimensions from style constants.
    binary_viewer_width = BINARY_STYLE.window_geometry[2]
    binary_viewer_height = BINARY_STYLE.window_geometry[3]
    pc_viewer_width = PC_STYLE.window_geometry[2]
    pc_viewer_height = PC_STYLE.window_geometry[3]
    # Horizontal pixel gap between adjacent windows when placed side by side.
    viewer_gap = 10

    if binary_viewer_x + binary_viewer_width + viewer_gap + pc_viewer_width <= available.width():
        # Side by side: PC viewer sits immediately to the right of the binary viewer.
        pc_viewer_x, pc_viewer_y = binary_viewer_x + binary_viewer_width + viewer_gap, 50
    else:
        # Cascade: offset so both title bars remain accessible on smaller screens.
        pc_viewer_x, pc_viewer_y = binary_viewer_x + 30, binary_viewer_y + 30

    binary_player.setGeometry(binary_viewer_x, binary_viewer_y, binary_viewer_width, binary_viewer_height)
    pc_viewer.setGeometry(pc_viewer_x, pc_viewer_y, pc_viewer_width, pc_viewer_height)

    # When the user switches planes in the binary player, the shared SingleDayData
    # instance is mutated in place (via switch_view). This signal connection triggers the PC
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


def run_roi_viewer(session_path: Path | None = None, *, dataset: str | None = None) -> None:
    """Launches the standalone ROI viewer with right-click reclassification.

    Creates a QApplication, loads pipeline data from the given session directory (or opens a file
    dialog if no path is provided), shows the ROIViewer window, and enters the event loop.

    Args:
        session_path: Path to a cindra output directory to load on startup. Opens a file dialog
            if None.
        dataset: Multi-day dataset name to load. Defaults to the first available dataset.
    """
    application = QApplication.instance()
    owns_application = application is None
    if owns_application:
        application = QApplication(sys.argv)
        application.setFont(FONTS.small)
    if not isinstance(application, QApplication):  # pragma: no cover
        message = "Unable to launch the ROI viewer. Failed to obtain a QApplication instance."
        console.error(message=message, error=RuntimeError)

    app_icon = QtGui.QIcon()
    for size in (16, 24, 32, 48, 64, 256):
        app_icon.addFile(_ICON_PATH, QtCore.QSize(size, size))
    application.setWindowIcon(app_icon)

    data = ViewerData.from_data(root_path=session_path, dataset=dataset) if session_path is not None else None

    window = ROIViewer(data=data)
    window.show()

    if owns_application:
        sys.exit(application.exec())
