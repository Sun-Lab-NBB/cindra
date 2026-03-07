"""Provides application entry points for launching all cindra GUI viewers."""

from __future__ import annotations

import sys
import copy
from typing import TYPE_CHECKING

from PySide6 import QtCore
from PySide6.QtWidgets import QApplication
from ataraxis_base_utilities import LogLevel, console

from .styles import FONTS, PC_STYLE, BINARY_STYLE
from .pc_viewer import PCViewer
from .roi_viewer import ROIViewer
from .binary_viewer import BinaryPlayer
from .viewer_context import ViewerData, SingleRecordingData
from .tracking_viewer import TrackingViewer

if TYPE_CHECKING:
    from pathlib import Path


def run_tracking_viewer(recording_path: Path, *, dataset: str | None = None) -> None:
    """Launches the standalone multi-recording tracking viewer.

    Creates a QApplication, shows the TrackingViewer window, and enters the event loop. The viewer
    loads multi-recording tracking data from the provided directory on startup.

    Args:
        recording_path: The path to the root data directory for any cindra-processed recording that
            makes up the visualized multi-recording dataset. The loader uses that recording's data to
            search and reconstruct the full dataset hierarchy.
        dataset: Multi-recording dataset name to load. Defaults to the first available dataset.
    """
    console.echo(message="Initializing the Tracking GUI...")
    application, owns_application = _get_or_create_application()

    # Loads recording data upfront so the viewer window receives a fully populated data instance.
    # The tracking viewer requires a loaded dataset, so defaults to the first available one.
    data = ViewerData.from_data(root_path=recording_path, dataset=dataset)
    if not data.is_multi_recording and data.available_datasets:
        data.load_dataset(dataset_name=data.available_datasets[0])

    # Creates the viewer window with the loaded data.
    console.echo(message="Initializing Multi-Recording Tracking viewer...")
    window = TrackingViewer(data=data)

    window.show()
    console.echo(message="Tracking viewer: ready.", level=LogLevel.SUCCESS)

    # Only enters the event loop if this function created the QApplication. When embedded in a
    # larger GUI, the caller is responsible for running the event loop.
    if owns_application and application is not None:
        sys.exit(application.exec())


def run_registration_viewer(recording_path: Path) -> None:
    """Launches the standalone single-recording registration viewer.

    Creates a QApplication, shows the BinaryPlayer and PCViewer windows, and enters the event
    loop. The binary viewer always displays the stitched multi-plane movie while the PC viewer
    receives an independent shallow copy of the data with its own plane selector.

    Args:
        recording_path: The Path to a cindra-processed recording's root data directory containing
            registration results.
    """
    console.echo(message="Initializing the Registration GUI...")
    application, owns_application = _get_or_create_application()

    # Loads recording data upfront. The binary viewer uses the combined view (-1) for stitched
    # playback, while the PC viewer gets a shallow copy with its own mutable _view_index set to
    # plane 0 for per-plane PC metrics.
    data = SingleRecordingData.from_data(root_path=recording_path, view_index=-1)
    pc_data = copy.copy(data)
    pc_data.switch_view(view_index=0)

    # Creates both viewer windows with independent SingleRecordingData copies.
    console.echo(message="Initializing Registered Recording viewer...")
    binary_player = BinaryPlayer(data=data)
    console.echo(message="Initializing Registration Quality Metrics viewer...")
    pc_viewer = PCViewer(data=pc_data)

    # Computes screen-adaptive window positions. On large screens the two viewers sit side by side;
    # on smaller screens they cascade with a visible offset so both title bars remain accessible.
    screen = application.primaryScreen()
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

    # Clamps window dimensions to fit within the available screen area.
    binary_viewer_height = min(binary_viewer_height, available.height() - binary_viewer_y)
    binary_viewer_width = min(binary_viewer_width, available.width() - binary_viewer_x)
    pc_viewer_height = min(pc_viewer_height, available.height() - pc_viewer_y)
    pc_viewer_width = min(pc_viewer_width, available.width() - pc_viewer_x)

    binary_player.setGeometry(binary_viewer_x, binary_viewer_y, binary_viewer_width, binary_viewer_height)
    pc_viewer.setGeometry(pc_viewer_x, pc_viewer_y, pc_viewer_width, pc_viewer_height)

    # When the user loads a new recording in the binary player, creates a fresh shallow copy
    # of the new data for the PC viewer so it can manage its own plane state independently.
    def _on_recording_changed() -> None:
        new_pc_data = copy.copy(binary_player.data)
        new_pc_data.switch_view(view_index=0)
        pc_viewer.load_data(data=new_pc_data)

    binary_player.recording_changed.connect(_on_recording_changed)

    # Links the two windows so closing either one closes the other. WA_DeleteOnClose ensures the
    # Qt object is destroyed on close, which emits the destroyed signal that triggers close() on
    # the partner window.
    binary_player.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
    pc_viewer.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose)
    binary_player.destroyed.connect(pc_viewer.close)
    pc_viewer.destroyed.connect(binary_player.close)

    binary_player.show()
    pc_viewer.show()
    console.echo(message="Registration viewers: ready.", level=LogLevel.SUCCESS)

    # Only enters the event loop if this function created the QApplication. When embedded in a
    # larger GUI, the caller is responsible for running the event loop.
    if owns_application and application is not None:
        sys.exit(application.exec())


def run_roi_viewer(recording_path: Path, *, dataset: str | None = None) -> None:
    """Launches the standalone ROI viewer with right-click reclassification.

    Creates a QApplication, loads pipeline data from the given recording directory, shows the ROIViewer window, and
    enters the event loop.

    Args:
        recording_path: Path to a cindra output directory to load on startup.
        dataset: Multi-recording dataset name to load. Stays in single-recording mode if not provided.
    """
    console.echo(message="Initializing the ROI GUI...")
    application, owns_application = _get_or_create_application()

    # Loads recording data upfront so the viewer window receives a fully populated data instance.
    data = ViewerData.from_data(root_path=recording_path, dataset=dataset)

    console.echo(message="Initializing ROI viewer...")
    window = ROIViewer(data=data)
    window.show()
    console.echo(message="ROI viewer: ready.", level=LogLevel.SUCCESS)

    if owns_application:
        sys.exit(application.exec())


def _get_or_create_application() -> tuple[QApplication, bool]:
    """Returns the current QApplication instance, creating one if none exists.

    Returns:
        A tuple of (application, owns_application) where owns_application is True when a new
        QApplication was created by this call.
    """
    application = QApplication.instance()
    owns_application = application is None
    if owns_application:
        application = QApplication(sys.argv)
        application.setFont(FONTS.small)
    return application, owns_application  # type: ignore[return-value]
