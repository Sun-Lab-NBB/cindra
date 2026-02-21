"""Provides registration binary viewer and principal component metrics viewer windows."""

from __future__ import annotations

from typing import IO, TYPE_CHECKING
from pathlib import Path

import numpy as np
from PySide6 import QtGui, QtCore
from tifffile import imread
import pyqtgraph as pg  # type: ignore[import-untyped]
from scipy.ndimage import gaussian_filter1d
from PySide6.QtWidgets import (
    QLabel,
    QStyle,
    QSlider,
    QWidget,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QFileDialog,
    QGridLayout,
    QMainWindow,
    QPushButton,
    QToolButton,
    QButtonGroup,
)
from ataraxis_base_utilities import LogLevel, console

from ..styles import FONT_FAMILY, WHITE_LABEL_STYLESHEET
from .context_data import RegistrationViewerData

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Scatter plot marker point size in pixels.
_SCATTER_POINT_SIZE: int = 10

# Size for media control button icons in pixels.
_ICON_SIZE: int = 30

# Gaussian smoothing sigma for z-position correlation filtering.
_Z_SMOOTHING_SIGMA: int = 2

# Multiplier for real-time playback speed.
_PLAYBACK_SPEED_MULTIPLIER: int = 5

# Number of frames subsampled for dynamic range estimation.
_SUBSAMPLE_FRAME_COUNT: int = 100

# Minimum frame increment for arrow key navigation.
_MIN_FRAME_DELTA: int = 5

# Divisor for computing frame slider step size from total frames.
_FRAME_DELTA_DIVISOR: int = 200

# Default number of principal components.
_DEFAULT_PC_COUNT: int = 50

# Animation interval for PC viewer in milliseconds.
_PC_ANIMATION_INTERVAL_MS: int = 200

# Number of standard deviations below mean for display range lower bound.
_DISPLAY_RANGE_LOW_SIGMA: float = 2.0

# Number of standard deviations above mean for display range upper bound.
_DISPLAY_RANGE_HIGH_SIGMA: float = 5.0

# Z-stack percentile bounds for display range.
_Z_PERCENTILE_LOW: int = 1
_Z_PERCENTILE_HIGH: int = 99

# Width for z-plane input field.
_Z_EDIT_WIDTH: int = 30

# Width for PC number input field.
_PC_EDIT_WIDTH: int = 40

# Font point size for metrics labels.
_METRICS_FONT_SIZE: int = 14


class BinaryPlayer(QMainWindow):
    """Provides a playback window for viewing registered binary imaging data.

    Attributes:
        _data: The registration viewer data model, or None if no session is loaded.
        _loaded: Determines whether registration data has been loaded and is ready for display.
        _z_loaded: Determines whether a z-stack has been loaded.
        _z_on: Determines whether the z-stack side view is currently visible.
        _z_correlation: Cached z-position correlation array, or None if not computed.
        _has_channel_2_binary: Determines whether a channel 2 binary file is available.
        _has_nonrigid: Determines whether nonrigid registration data is available.
        _channel_2_visible: Determines whether the channel 2 overlay is currently displayed.
        _frame_count: Total number of frames in the current plane's binary file.
        _current_frame: Index of the currently displayed frame.
        _frame_delta: Frame step size for arrow key navigation.
        _frame_height: Height of each frame in pixels for the current plane.
        _frame_width: Width of each frame in pixels for the current plane.
        _plane_heights: Per-tile frame heights in pixels.
        _plane_widths: Per-tile frame widths in pixels.
        _plane_y_offsets: Vertical offsets for each tile within the composite frame.
        _plane_x_offsets: Horizontal offsets for each tile within the composite frame.
        _registration_paths: Paths to the registered binary files for each tile.
        _registration_files: Open binary file handles for reading registered frames.
        _registration_file_channel_2: Open binary file handle for channel 2, or None.
        _bytes_per_frame: Number of bytes per frame for each tile.
        _display_range: Low and high bounds for the image display range.
        _aspect_ratio: Aspect ratio for the image display.
        _time_step: Timer interval in milliseconds for playback speed.
        _y_offsets: Per-frame vertical rigid registration offsets.
        _x_offsets: Per-frame horizontal rigid registration offsets.
        _nonrigid_rms: Per-frame RMS nonrigid displacement, or None.
        _image: Current frame image buffer.
        _z_stack: Loaded z-stack volume, or None if not loaded.
        _z_height: Height of z-stack planes in pixels.
        _z_width: Width of z-stack planes in pixels.
        _z_display_range: Percentile-based display range for z-stack images.
        _z_max_positions: Per-frame z-plane indices of maximum correlation, or None.
        _central_widget: Central widget container.
        _layout: Grid layout for arranging all controls and views.
        _graphics_widget: PyQtGraph graphics layout for image and plot views.
        _main_view_box: View box for the primary registered image display.
        _main_image: Image item for the primary frame display.
        _side_view_box: View box for the z-stack side display.
        _side_image: Image item for the z-stack frame display.
        _channel_2_checkbox: Checkbox for toggling channel 2 overlay.
        _z_stack_checkbox: Checkbox for toggling z-stack side view.
        _z_plane_edit: Input field for the current z-plane index.
        _shift_plot: Plot widget for rigid registration X-Y offsets.
        _shift_scatter: Scatter plot overlay indicating the current frame on the shift plot.
        _nonrigid_plot: Plot widget for nonrigid RMS displacement.
        _nonrigid_scatter: Scatter plot overlay indicating the current frame on the nonrigid plot.
        _z_position_plot: Plot widget for the z-position correlation trace.
        _z_position_scatter: Scatter plot overlay indicating the current frame on the z-position plot.
        _movie_label: Label displaying the current session path.
        _frame_label: Label for the frame slider.
        _frame_number_label: Label displaying the current frame number.
        _frame_slider: Horizontal slider for frame navigation.
        _plane_selector: Dropdown for selecting the imaging plane.
        _play_button: Button to start video playback.
        _pause_button: Button to pause video playback.
        _compute_z_button: Button to compute z-position correlations.
        _update_timer: Timer driving frame advancement during playback.
    """

    # Emitted when the user selects a different imaging plane from the plane selector.
    plane_changed = QtCore.Signal(int)

    def __init__(self, data: RegistrationViewerData | None = None) -> None:
        """Initializes the binary player window and all UI components.

        Args:
            data: Pre-loaded registration data to display on startup.
        """
        super().__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(70, 70, 1070, 1070)
        self.setWindowTitle("View registered binary")
        self._central_widget: QWidget = QWidget(self)
        self.setCentralWidget(self._central_widget)
        self._layout: QGridLayout = QGridLayout()
        self._central_widget.setLayout(self._layout)
        self._graphics_widget: pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget()
        self._graphics_widget.move(600, 0)
        self._graphics_widget.resize(1000, 500)
        self._layout.addWidget(self._graphics_widget, 1, 2, 13, 14)

        # State flags and data model.
        self._loaded: bool = False
        self._z_loaded: bool = False
        self._z_on: bool = False
        self._channel_2_visible: bool = False
        self._has_channel_2_binary: bool = False
        self._has_nonrigid: bool = False
        self._z_correlation: np.ndarray | None = None
        self._data: RegistrationViewerData | None = None

        # Frame geometry and binary file handles.
        self._frame_count: int = 0
        self._current_frame: int = 0
        self._frame_delta: int = 10
        self._frame_height: int = 0
        self._frame_width: int = 0
        self._plane_heights: list[int] = []
        self._plane_widths: list[int] = []
        self._plane_y_offsets: list[int] = []
        self._plane_x_offsets: list[int] = []
        self._registration_paths: list[str] = []
        self._registration_files: list[IO[bytes]] = []
        self._registration_file_channel_2: IO[bytes] | None = None
        self._bytes_per_frame: list[int] = []
        self._display_range: np.ndarray = np.zeros((2,), dtype=np.float64)
        self._aspect_ratio: float = 1.0
        self._time_step: float = 0.0
        self._y_offsets: np.ndarray = np.zeros((0,), dtype=np.int32)
        self._x_offsets: np.ndarray = np.zeros((0,), dtype=np.int32)
        self._nonrigid_rms: np.ndarray | None = None
        self._image: np.ndarray | None = None

        # Z-stack data.
        self._z_stack: np.ndarray | None = None
        self._z_height: int = 0
        self._z_width: int = 0
        self._z_display_range: list[float] = []
        self._z_max_positions: np.ndarray | None = None

        # Main image view.
        self._main_view_box: pg.ViewBox = pg.ViewBox(lockAspect=True, invertY=True, name="plot1")
        self._graphics_widget.addItem(self._main_view_box, row=0, col=0)
        self._main_view_box.setMenuEnabled(False)
        self._main_image: pg.ImageItem = pg.ImageItem()
        self._main_view_box.addItem(self._main_image)

        # Side box for z-stack display.
        self._side_view_box: pg.ViewBox = pg.ViewBox(lockAspect=True, invertY=True)
        self._side_view_box.setMenuEnabled(False)
        self._side_image: pg.ImageItem = pg.ImageItem()
        self._side_view_box.addItem(self._side_image)

        # Channel 2 checkbox.
        self._channel_2_checkbox: QCheckBox = QCheckBox("view channel 2")
        self._channel_2_checkbox.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self._channel_2_checkbox.setEnabled(False)
        self._channel_2_checkbox.toggled.connect(self._toggle_channel_2)
        self._layout.addWidget(self._channel_2_checkbox, 0, 5, 1, 1)

        # Z-stack checkbox.
        self._z_stack_checkbox: QCheckBox = QCheckBox("view z-stack")
        self._z_stack_checkbox.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self._z_stack_checkbox.setEnabled(False)
        self._z_stack_checkbox.toggled.connect(self._add_z_stack)
        self._layout.addWidget(self._z_stack_checkbox, 0, 8, 1, 1)

        z_label = QLabel("Z-plane:")
        z_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self._layout.addWidget(z_label, 0, 9, 1, 1)

        self._z_plane_edit: QLineEdit = QLineEdit(self)
        self._z_plane_edit.setValidator(QtGui.QIntValidator(0, 0))
        self._z_plane_edit.setText("0")
        self._z_plane_edit.setFixedWidth(_Z_EDIT_WIDTH)
        self._z_plane_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._layout.addWidget(self._z_plane_edit, 0, 10, 1, 1)

        # Rigid registration offset plot.
        self._shift_plot = self._graphics_widget.addPlot(name="plot_shift", row=1, col=0, colspan=2)
        self._shift_plot.setMouseEnabled(x=True, y=False)
        self._shift_plot.setMenuEnabled(False)
        self._shift_scatter: pg.ScatterPlotItem = pg.ScatterPlotItem()
        self._shift_scatter.setData([0, 0], [0, 0])
        self._shift_plot.addItem(self._shift_scatter)

        # Nonrigid RMS displacement plot.
        self._nonrigid_plot = self._graphics_widget.addPlot(name="plot_nonrigid", row=2, col=0, colspan=2)
        self._nonrigid_plot.setMouseEnabled(x=True, y=False)
        self._nonrigid_plot.setMenuEnabled(False)
        self._nonrigid_scatter: pg.ScatterPlotItem = pg.ScatterPlotItem()
        self._nonrigid_plot.setXLink("plot_shift")

        # Z-position correlation plot.
        self._z_position_plot = self._graphics_widget.addPlot(name="plot_Z", row=3, col=0, colspan=2)
        self._z_position_plot.setMouseEnabled(x=True, y=False)
        self._z_position_plot.setMenuEnabled(False)
        self._z_position_scatter: pg.ScatterPlotItem = pg.ScatterPlotItem()
        self._z_position_plot.setXLink("plot_shift")

        self._graphics_widget.ci.layout.setRowStretchFactor(0, 12)
        self._movie_label: QLabel = QLabel("No session loaded")
        self._movie_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self._movie_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._create_buttons()

        # Plane selector dropdown.
        plane_label = QLabel("Plane:")
        plane_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self._layout.addWidget(plane_label, 6, 0, 1, 1)
        self._plane_selector: QComboBox = QComboBox(self)
        self._plane_selector.setEnabled(False)
        self._plane_selector.currentIndexChanged.connect(self._on_plane_changed)
        self._layout.addWidget(self._plane_selector, 6, 1, 1, 1)

        # Frame slider.
        self._frame_label: QLabel = QLabel("Current frame:")
        self._frame_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self._frame_number_label: QLabel = QLabel("0")
        self._frame_number_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self._frame_slider: QSlider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self._frame_slider.setTickInterval(5)
        self._frame_slider.setTracking(False)
        self._layout.addWidget(QLabel(""), 12, 0, 1, 1)
        self._layout.setRowStretch(12, 1)
        self._layout.addWidget(self._frame_label, 13, 0, 1, 2)
        self._layout.addWidget(self._frame_number_label, 14, 0, 1, 2)
        self._layout.addWidget(self._frame_slider, 13, 2, 14, 13)
        self._layout.addWidget(QLabel(""), 14, 1, 1, 1)
        hint_label = QLabel("(when paused, left/right arrow keys can move slider)")
        hint_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self._layout.addWidget(hint_label, 16, 0, 1, 3)
        self._frame_slider.valueChanged.connect(self._go_to_frame)
        self._layout.addWidget(self._movie_label, 0, 0, 1, 5)
        self._update_frame_slider()
        self._update_buttons()
        self._update_timer: QtCore.QTimer = QtCore.QTimer()
        self._update_timer.timeout.connect(self._next_frame)
        self._graphics_widget.scene().sigMouseClicked.connect(self._plot_clicked)

        if data is not None:
            self.load_data(data=data)

    def load_data(self, data: RegistrationViewerData) -> None:
        """Stores the data model, populates the plane selector, and opens the first plane.

        Args:
            data: The registration viewer data model wrapping all planes.
        """
        self._data = data
        # Populates the plane selector without triggering _on_plane_changed yet.
        self._plane_selector.blockSignals(True)
        self._plane_selector.clear()
        for label in data.plane_labels:
            self._plane_selector.addItem(label)
        self._plane_selector.setCurrentIndex(data.current_plane_index)
        self._plane_selector.blockSignals(False)
        self._plane_selector.setEnabled(data.plane_count > 1)
        self._open_plane()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        """Handles keyboard navigation for frame stepping and playback control."""
        if self._play_button.isEnabled() and event.modifiers() != QtCore.Qt.KeyboardModifier.ShiftModifier:
            if event.key() == QtCore.Qt.Key.Key_Left:
                self._current_frame -= self._frame_delta
                self._current_frame = max(0, min(self._frame_count - 1, self._current_frame))
                self._frame_slider.setValue(self._current_frame)
            elif event.key() == QtCore.Qt.Key.Key_Right:
                self._current_frame += self._frame_delta
                self._current_frame = max(0, min(self._frame_count - 1, self._current_frame))
                self._frame_slider.setValue(self._current_frame)
        if event.modifiers() != QtCore.Qt.KeyboardModifier.ShiftModifier and event.key() == QtCore.Qt.Key.Key_Space:
            if self._play_button.isEnabled():
                self._start_playback()
            else:
                self._pause_playback()

    def _on_plane_changed(self, index: int) -> None:
        """Handles plane selector index changes by switching to the selected plane.

        Args:
            index: Zero-based index of the selected plane.
        """
        if self._data is None or index < 0:
            return
        self._close_binary_files()
        self._data.switch_plane(plane_index=index)
        self._open_plane()
        self.plane_changed.emit(index)

    def _open_plane(self) -> None:
        """Opens the binary file for the current plane and configures frame geometry."""
        if self._data is None:
            return

        registered_path = self._data.registered_binary_path
        if registered_path is None or not registered_path.is_file():
            console.echo(message="No registered binary found for this plane.", level=LogLevel.WARNING)
            return

        self._frame_height = self._data.frame_height
        self._frame_width = self._data.frame_width
        self._plane_heights = [self._data.frame_height]
        self._plane_widths = [self._data.frame_width]
        self._plane_y_offsets = [0]
        self._plane_x_offsets = [0]

        self._registration_paths = [str(registered_path)]
        self._registration_files = [registered_path.open("rb")]
        self._has_channel_2_binary = False

        channel_2_path = self._data.registered_binary_path_channel_2
        if channel_2_path is not None and channel_2_path.is_file():
            self._registration_file_channel_2 = channel_2_path.open("rb")
            self._has_channel_2_binary = True

        self._setup_views()

    def _close_binary_files(self) -> None:
        """Closes all open binary file handles."""
        for handle in self._registration_files:
            handle.close()
        self._registration_files = []
        if self._has_channel_2_binary and self._registration_file_channel_2 is not None:
            self._registration_file_channel_2.close()
            self._registration_file_channel_2 = None
        self._has_channel_2_binary = False

    def _load_session(self) -> None:
        """Opens a file dialog to select a session directory and loads it."""
        directory = QFileDialog.getExistingDirectory(self, "Open session directory")
        if directory:
            try:
                data = RegistrationViewerData.from_session(root_path=Path(directory))
                self.load_data(data=data)
            except Exception as error:
                console.echo(message=f"Unable to load session. {error}", level=LogLevel.ERROR)

    def _toggle_channel_2(self) -> None:
        """Toggles channel 2 display based on checkbox state."""
        if self._loaded:
            self._channel_2_visible = self._channel_2_checkbox.isChecked()
            self._next_frame()

    def _zoom_image(self) -> None:
        """Resets the main and side view zoom to fit the full image extent."""
        self._main_view_box.setRange(yRange=(0, self._frame_height), xRange=(0, self._frame_width))
        if self._z_on:
            self._side_view_box.setRange(yRange=(0, self._z_height), xRange=(0, self._z_width))
            self._side_view_box.setXLink("plot1")
            self._side_view_box.setYLink("plot1")

    def _add_z_stack(self) -> None:
        """Toggles z-stack side view display based on checkbox state."""
        if self._loaded:
            if self._z_stack_checkbox.isChecked():
                self._z_on = True
                self._graphics_widget.addItem(self._side_view_box, row=0, col=1)
            else:
                self._z_on = False
                self._graphics_widget.removeItem(self._side_view_box)
            self._next_frame()

    def _next_frame(self) -> None:
        """Advances to the next frame and updates all display elements."""
        self._current_frame += 1
        if self._current_frame > self._frame_count - 1:
            self._current_frame = 0
            for handle in self._registration_files:
                handle.seek(0, 0)
            if self._has_channel_2_binary and self._registration_file_channel_2 is not None:
                self._registration_file_channel_2.seek(0, 0)
        self._image = np.zeros((self._frame_height, self._frame_width), dtype=np.int16)
        for index in range(len(self._registration_paths)):
            buff = self._registration_files[index].read(self._bytes_per_frame[index])
            frame = np.frombuffer(buff, dtype=np.int16, offset=0).reshape(
                (self._plane_heights[index], self._plane_widths[index])
            )
            self._image[
                self._plane_y_offsets[index] : self._plane_y_offsets[index] + self._plane_heights[index],
                self._plane_x_offsets[index] : self._plane_x_offsets[index] + self._plane_widths[index],
            ] = frame

        if self._has_channel_2_binary and self._channel_2_visible and self._registration_file_channel_2 is not None:
            buff = self._registration_file_channel_2.read(self._bytes_per_frame[0])
            channel_2_frame = np.frombuffer(buff, dtype=np.int16, offset=0).reshape(
                (self._plane_heights[0], self._plane_widths[0])
            )[:, :, np.newaxis]
            self._image = np.concatenate(
                (self._image[:, :, np.newaxis], channel_2_frame, np.zeros_like(channel_2_frame)),
                axis=-1,
            )
        if self._z_loaded and self._z_on and self._z_stack is not None:
            if self._z_max_positions is not None:
                self._z_plane_edit.setText(str(self._z_max_positions[self._current_frame]))
            self._side_image.setImage(self._z_stack[int(self._z_plane_edit.text())], levels=self._z_display_range)

        self._main_image.setImage(self._image, levels=self._display_range)
        self._frame_slider.setValue(self._current_frame)
        self._frame_number_label.setText(str(self._current_frame))
        self._shift_scatter.setData(
            [self._current_frame, self._current_frame],
            [self._y_offsets[self._current_frame], self._x_offsets[self._current_frame]],
            size=_SCATTER_POINT_SIZE,
            brush=pg.mkBrush(255, 0, 0),
        )
        if self._has_nonrigid and self._nonrigid_rms is not None:
            self._nonrigid_scatter.setData(
                [self._current_frame],
                [self._nonrigid_rms[self._current_frame]],
                size=_SCATTER_POINT_SIZE,
                brush=pg.mkBrush(255, 0, 0),
            )
        if self._z_loaded and self._z_on and self._z_max_positions is not None:
            self._z_position_scatter.setData(
                [self._current_frame, self._current_frame],
                [self._z_max_positions[self._current_frame], self._z_max_positions[self._current_frame]],
                size=_SCATTER_POINT_SIZE,
                brush=pg.mkBrush(255, 0, 0),
            )

    def _setup_views(self) -> None:
        """Configures all plot views and display parameters after loading data."""
        if self._data is None:
            return

        self._shift_plot.clear()

        # Computes dynamic range from subsampled frames.
        sample_count = min(self._data.frame_count - 1, _SUBSAMPLE_FRAME_COUNT)
        frames = _subsample_frames(
            frame_count=self._data.frame_count,
            frame_height=self._data.frame_height,
            frame_width=self._data.frame_width,
            sample_count=sample_count,
            registration_path=str(self._data.registered_binary_path),
        )
        self._display_range = frames.mean() + frames.std() * np.array(
            [-_DISPLAY_RANGE_LOW_SIGMA, _DISPLAY_RANGE_HIGH_SIGMA]
        )

        self._movie_label.setText(self._registration_paths[-1])
        self._bytes_per_frame = [
            2 * self._plane_heights[index] * self._plane_widths[index] for index in range(len(self._registration_paths))
        ]

        # Aspect ratio from data model.
        self._aspect_ratio = self._data.aspect_ratio
        self._main_view_box.setAspectLocked(lock=True, ratio=self._aspect_ratio)
        self._side_view_box.setAspectLocked(lock=True, ratio=self._aspect_ratio)

        self._frame_count = self._data.frame_count
        self._time_step = 1.0 / self._data.sampling_rate * 1000 / _PLAYBACK_SPEED_MULTIPLIER
        self._frame_delta = max(_MIN_FRAME_DELTA, int(self._frame_count / _FRAME_DELTA_DIVISOR))
        self._frame_slider.setSingleStep(self._frame_delta)
        if self._frame_count > 0:
            self._update_frame_slider()
            self._update_buttons()

        # Plots registration X-Y offsets.
        rigid_y = self._data.rigid_y_offsets
        rigid_x = self._data.rigid_x_offsets
        if rigid_y is not None and rigid_x is not None:
            self._y_offsets = rigid_y
            self._x_offsets = rigid_x
        else:
            self._y_offsets = np.zeros((self._frame_count,), dtype=np.int32)
            self._x_offsets = np.zeros((self._frame_count,), dtype=np.int32)
        self._shift_plot.plot(self._y_offsets, pen="g")
        self._shift_plot.plot(self._x_offsets, pen="y")
        self._shift_plot.setRange(
            xRange=(0, self._frame_count),
            yRange=(
                min(int(self._y_offsets.min()), int(self._x_offsets.min())),
                max(int(self._y_offsets.max()), int(self._x_offsets.max())),
            ),
            padding=0.0,
        )
        self._shift_plot.setLimits(xMin=0, xMax=self._frame_count)
        self._shift_scatter = pg.ScatterPlotItem()
        self._shift_plot.addItem(self._shift_scatter)
        self._shift_scatter.setData(
            [self._current_frame, self._current_frame],
            [self._y_offsets[self._current_frame], self._x_offsets[self._current_frame]],
            size=_SCATTER_POINT_SIZE,
            brush=pg.mkBrush(255, 0, 0),
        )

        # Plots per-frame nonrigid RMS displacement if available.
        self._nonrigid_plot.clear()
        nonrigid_y = self._data.nonrigid_y_offsets
        nonrigid_x = self._data.nonrigid_x_offsets
        if nonrigid_y is not None and nonrigid_x is not None:
            self._nonrigid_rms = np.sqrt(
                np.mean(nonrigid_y.astype(np.float32) ** 2 + nonrigid_x.astype(np.float32) ** 2, axis=1)
            )
            self._has_nonrigid = True
            self._nonrigid_plot.plot(self._nonrigid_rms, pen=(180, 100, 255))
            self._nonrigid_plot.setRange(
                xRange=(0, self._frame_count),
                yRange=(0, self._nonrigid_rms.max()),
                padding=0.0,
            )
            self._nonrigid_plot.setLimits(xMin=0, xMax=self._frame_count)
            self._nonrigid_plot.setLabel("left", "nonrigid RMS", units="px")
            self._nonrigid_scatter = pg.ScatterPlotItem()
            self._nonrigid_plot.addItem(self._nonrigid_scatter)
        else:
            self._has_nonrigid = False

        self._channel_2_checkbox.setEnabled(self._has_channel_2_binary)

        self._current_frame = -1
        self._loaded = True
        self._next_frame()

    def _plot_clicked(self, event: object) -> None:
        """Handles mouse click events on plots for frame navigation."""
        items = self._graphics_widget.scene().items(event.scenePos())  # type: ignore[attr-defined]
        position_x = 0
        is_time_plot = False
        zoom = False
        choose = False
        if self._loaded:
            for item in items:
                if item in (self._shift_plot, self._nonrigid_plot):
                    view_box = self._shift_plot.vb
                    position = view_box.mapSceneToView(event.scenePos())  # type: ignore[attr-defined]
                    position_x = position.x()
                    is_time_plot = True
                elif item in (self._main_view_box, self._side_view_box):
                    if event.button() == 1 and event.double():  # type: ignore[attr-defined]
                        self._zoom_image()
                if is_time_plot and event.button() == 1:  # type: ignore[attr-defined]
                    if event.double():  # type: ignore[attr-defined]
                        zoom = True
                    else:
                        choose = True
        if zoom:
            self._shift_plot.setRange(xRange=(0, self._frame_count))
            self._nonrigid_plot.setRange(xRange=(0, self._frame_count))
            self._z_position_plot.setRange(xRange=(0, self._frame_count))

        if choose and self._play_button.isEnabled():
            self._current_frame = max(0, min(self._frame_count - 1, round(position_x)))
            self._frame_slider.setValue(self._current_frame)

    def _load_z_stack(self) -> None:
        """Opens a file dialog to load a z-stack TIFF and initializes z-position tracking."""
        file_dialog_result = QFileDialog.getOpenFileName(self, "Open zstack", filter="*.tif")
        z_stack_path = file_dialog_result[0]
        try:
            self._z_stack = imread(z_stack_path)
            self._z_height, self._z_width = self._z_stack.shape[1:]
            self._z_plane_edit.setValidator(QtGui.QIntValidator(0, self._z_stack.shape[0]))
            self._z_display_range = [
                np.percentile(self._z_stack, _Z_PERCENTILE_LOW),
                np.percentile(self._z_stack, _Z_PERCENTILE_HIGH),
            ]

            self._compute_z_button.setEnabled(True)
            self._z_loaded = True
            self._z_stack_checkbox.setEnabled(True)
            self._z_stack_checkbox.setChecked(True)
            self._z_max_positions = np.zeros(self._frame_count, dtype=int)

            # Checks for cached z-correlation data in order of priority:
            # 1. Local instance cache (self._z_correlation)
            # 2. Separate zcorr.npy file from the output directory
            if self._z_correlation is not None and self._z_stack.shape[0] == self._z_correlation.shape[0]:
                self._z_max_positions = np.argmax(
                    gaussian_filter1d(self._z_correlation.T.copy(), sigma=_Z_SMOOTHING_SIGMA, axis=1),
                    axis=1,
                )
                self._plot_z_correlation()
            elif self._data is not None and self._data.output_directory is not None:
                zcorr_path = self._data.output_directory / "zcorr.npy"
                if zcorr_path.exists():
                    self._z_correlation = np.load(zcorr_path)
                    if self._z_stack.shape[0] == self._z_correlation.shape[0]:
                        self._z_max_positions = np.argmax(
                            gaussian_filter1d(self._z_correlation.T.copy(), sigma=_Z_SMOOTHING_SIGMA, axis=1),
                            axis=1,
                        )
                        self._plot_z_correlation()

        except Exception as error:
            console.echo(message=f"Unable to load z-stack TIFF. {error}", level=LogLevel.ERROR)

    def _go_to_frame(self) -> None:
        """Seeks to the frame indicated by the frame slider position."""
        self._current_frame = int(self._frame_slider.value())
        self._jump_to_frame()

    def _jump_to_frame(self) -> None:
        """Seeks all binary file handles to an absolute frame position and displays it."""
        if self._play_button.isEnabled():
            self._current_frame = max(0, min(self._frame_count - 1, self._current_frame))
            self._current_frame = int(self._current_frame)
            for index in range(len(self._registration_files)):
                self._registration_files[index].seek(self._bytes_per_frame[index] * self._current_frame, 0)
            if self._has_channel_2_binary and self._registration_file_channel_2 is not None:
                self._registration_file_channel_2.seek(self._bytes_per_frame[-1] * self._current_frame, 0)
            self._current_frame -= 1
            self._next_frame()

    def _start_playback(self) -> None:
        """Starts video playback by enabling the frame update timer."""
        if self._current_frame < self._frame_count - 1:
            console.echo(message="Playing video...")
            self._play_button.setEnabled(False)
            self._pause_button.setEnabled(True)
            self._frame_slider.setEnabled(False)
            self._update_timer.start(int(self._time_step))

    def _pause_playback(self) -> None:
        """Pauses video playback and re-enables manual frame navigation."""
        self._update_timer.stop()
        self._play_button.setEnabled(True)
        self._pause_button.setEnabled(False)
        self._frame_slider.setEnabled(True)
        console.echo(message="Video paused")

    def _compute_z(self) -> None:
        """Computes z-position correlations for the current session.

        Currently disabled pending refactor to use RuntimeContext instead of legacy ops dictionary.
        """
        console.echo(
            message="Z-position computation is temporarily disabled. The GUI needs to be refactored to use "
            "RuntimeContext instead of the legacy ops dictionary.",
            level=LogLevel.WARNING,
        )

    def _plot_z_correlation(self) -> None:
        """Plots the z-position correlation trace on the z-position plot."""
        if self._z_max_positions is None:
            return
        self._z_position_plot.clear()
        self._z_position_plot.plot(self._z_max_positions, pen="r")
        self._z_position_plot.addItem(self._z_position_scatter)
        self._z_position_plot.setRange(
            xRange=(0, self._frame_count),
            yRange=(self._z_max_positions.min(), self._z_max_positions.max() + 3),
            padding=0.0,
        )
        self._z_position_plot.setLimits(xMin=0, xMax=self._frame_count)
        self._z_position_plot.setXLink("plot_shift")

    def _update_frame_slider(self) -> None:
        """Configures the frame slider range and enables it."""
        self._frame_slider.setMaximum(self._frame_count - 1)
        self._frame_slider.setMinimum(0)
        self._frame_label.setEnabled(True)
        self._frame_slider.setEnabled(True)

    def _update_buttons(self) -> None:
        """Sets the initial enabled state for play and pause buttons."""
        self._play_button.setEnabled(True)
        self._pause_button.setEnabled(False)
        self._pause_button.setChecked(True)

    def _create_buttons(self) -> None:
        """Creates and lays out all control buttons for the player window."""
        icon_size = QtCore.QSize(_ICON_SIZE, _ICON_SIZE)
        load_session_button = QPushButton("Load Session")
        load_session_button.setToolTip("Open a suite2p session directory")
        load_session_button.clicked.connect(self._load_session)

        load_z_button = QPushButton("load z-stack tiff")
        load_z_button.clicked.connect(self._load_z_stack)

        self._compute_z_button: QPushButton = QPushButton("compute z position")
        self._compute_z_button.setEnabled(False)
        self._compute_z_button.clicked.connect(self._compute_z)

        self._play_button: QToolButton = QToolButton()
        self._play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self._play_button.setIconSize(icon_size)
        self._play_button.setToolTip("Play")
        self._play_button.setCheckable(True)
        self._play_button.clicked.connect(self._start_playback)

        self._pause_button: QToolButton = QToolButton()
        self._pause_button.setCheckable(True)
        self._pause_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self._pause_button.setIconSize(icon_size)
        self._pause_button.setToolTip("Pause")
        self._pause_button.clicked.connect(self._pause_playback)

        button_group = QButtonGroup(self)
        button_group.addButton(self._play_button, 0)
        button_group.addButton(self._pause_button, 1)
        button_group.setExclusive(True)

        quit_button = QToolButton()
        quit_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogCloseButton))
        quit_button.setIconSize(icon_size)
        quit_button.setToolTip("Quit")
        quit_button.clicked.connect(self.close)

        self._layout.addWidget(load_session_button, 1, 0, 1, 2)
        self._layout.addWidget(load_z_button, 2, 0, 1, 2)
        self._layout.addWidget(self._compute_z_button, 3, 0, 1, 2)
        self._layout.addWidget(self._play_button, 15, 0, 1, 1)
        self._layout.addWidget(self._pause_button, 15, 1, 1, 1)
        self._play_button.setEnabled(False)
        self._pause_button.setEnabled(False)
        self._pause_button.setChecked(True)


class PCViewer(QMainWindow):
    """Provides a viewer window for principal component registration metrics.

    Attributes:
        _data: The registration viewer data model, or None if no session is loaded.
        _loaded: Determines whether PC data has been loaded and is ready for display.
        _current_frame: Animation toggle state for PC extreme image cycling.
        _pc_count: Number of principal components available.
        _pc_images: PC extreme images array with shape (2, num_pcs, height, width), or None.
        _image_height: Height of PC images in pixels.
        _image_width: Width of PC images in pixels.
        _pc_metrics: Registration shift metrics array with shape (num_pcs, 3), or None.
        _pc_projections: Per-frame PC projection array with shape (num_frames, num_pcs), or None.
        _central_widget: Central widget container.
        _layout: Grid layout for arranging all controls and views.
        _graphics_widget: PyQtGraph graphics layout for image and plot views.
        _metrics_plot: Plot widget for PC shift metrics.
        _difference_view_box: View box for the PC difference image.
        _merged_view_box: View box for the merged PC overlay image.
        _animated_view_box: View box for the animated PC extreme image.
        _difference_image: Image item for the PC difference display.
        _merged_image: Image item for the merged PC overlay display.
        _animated_image: Image item for the animated PC extreme display.
        _projection_plot: Plot widget for the PC time-course projection.
        _pc_edit: Input field for the current principal component number.
        _metric_labels: Labels displaying per-PC registration shift values.
        _title_labels: Labels displaying view titles below the image panels.
        _play_button: Button to start PC animation playback.
        _pause_button: Button to pause PC animation playback.
        _update_timer: Timer driving the PC animation.
        _metrics_scatter: Scatter plot overlay indicating the selected PC on the metrics plot, or None.
        _legend: Legend item for the metrics plot, or None.
    """

    def __init__(self, data: RegistrationViewerData | None = None) -> None:
        """Initializes the PC viewer window and all UI components.

        Args:
            data: Pre-loaded registration data to display on startup.
        """
        super().__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(70, 70, 1300, 800)
        self.setWindowTitle("Metrics for registration")
        self._central_widget: QWidget = QWidget(self)
        self.setCentralWidget(self._central_widget)
        self._layout: QGridLayout = QGridLayout()
        self._central_widget.setLayout(self._layout)

        self._graphics_widget: pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget()
        self._layout.addWidget(self._graphics_widget, 0, 2, 13, 14)

        # State and data.
        self._data: RegistrationViewerData | None = None
        self._loaded: bool = False
        self._current_frame: int = 0
        self._pc_count: int = _DEFAULT_PC_COUNT
        self._pc_images: np.ndarray | None = None
        self._image_height: int = 0
        self._image_width: int = 0
        self._pc_metrics: np.ndarray | None = None
        self._pc_projections: np.ndarray | None = None
        self._metrics_scatter: pg.ScatterPlotItem | None = None
        self._legend: pg.LegendItem | None = None

        # Pixel shift metrics plot.
        self._metrics_plot = self._graphics_widget.addPlot(row=0, col=0)
        self._metrics_plot.setMouseEnabled(x=False, y=False)
        self._metrics_plot.setMenuEnabled(False)

        self._difference_view_box = self._graphics_widget.addViewBox(
            name="plot1", lockAspect=True, row=1, col=0, invertY=True
        )
        self._merged_view_box = self._graphics_widget.addViewBox(lockAspect=True, row=1, col=1, invertY=True)
        self._merged_view_box.setMenuEnabled(False)
        self._merged_view_box.setXLink("plot1")
        self._merged_view_box.setYLink("plot1")
        self._animated_view_box = self._graphics_widget.addViewBox(lockAspect=True, row=1, col=2, invertY=True)
        self._animated_view_box.setMenuEnabled(False)
        self._animated_view_box.setXLink("plot1")
        self._animated_view_box.setYLink("plot1")
        self._difference_image: pg.ImageItem = pg.ImageItem()
        self._merged_image: pg.ImageItem = pg.ImageItem()
        self._animated_image: pg.ImageItem = pg.ImageItem()
        self._difference_view_box.addItem(self._difference_image)
        self._merged_view_box.addItem(self._merged_image)
        self._animated_view_box.addItem(self._animated_image)
        self._graphics_widget.scene().sigMouseClicked.connect(self._plot_clicked)

        self._projection_plot = self._graphics_widget.addPlot(row=0, col=1, colspan=2)
        self._projection_plot.setMouseEnabled(x=False)
        self._projection_plot.setMenuEnabled(False)

        self._pc_edit: QLineEdit = QLineEdit(self)
        self._pc_edit.setText("1")
        self._pc_edit.setFixedWidth(_PC_EDIT_WIDTH)
        self._pc_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._pc_edit.returnPressed.connect(self._plot_frame)
        self._pc_edit.textEdited.connect(self._pause_animation)
        pc_label = QLabel("PC: ")
        bold_font = QtGui.QFont(FONT_FAMILY, pointSize=_METRICS_FONT_SIZE, weight=QtGui.QFont.Weight.Bold)
        big_font = QtGui.QFont(FONT_FAMILY, pointSize=_METRICS_FONT_SIZE)
        pc_label.setFont(bold_font)
        self._pc_edit.setFont(big_font)
        pc_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self._layout.addWidget(QLabel(""), 1, 0, 1, 1)
        self._layout.addWidget(pc_label, 2, 0, 1, 1)
        self._layout.addWidget(self._pc_edit, 2, 1, 1, 1)
        self._metric_labels: list[QLabel] = []
        self._title_labels: list[QLabel] = []
        for index in range(3):
            metric_label = QLabel("")
            metric_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
            self._layout.addWidget(metric_label, 3 + index, 0, 1, 2)
            self._metric_labels.append(metric_label)
            title_label = QLabel("")
            title_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
            self._layout.addWidget(title_label, 12, 4 + index * 4, 1, 2)
            self._title_labels.append(title_label)
        self._layout.addWidget(QLabel(""), 7, 0, 1, 1)
        self._layout.setRowStretch(7, 1)
        self._create_buttons()
        self._pc_edit.setValidator(QtGui.QIntValidator(1, self._pc_count))
        self._update_timer: QtCore.QTimer = QtCore.QTimer()
        self._update_timer.timeout.connect(self._next_frame)

        if data is not None:
            self.load_data(data=data)

    def load_data(self, data: RegistrationViewerData) -> None:
        """Loads principal component data from the registration viewer data model.

        Args:
            data: The registration viewer data model for the current plane.
        """
        self._data = data
        pc_images = data.principal_component_extreme_images
        pc_metrics = data.principal_component_shift_metrics
        pc_projections = data.principal_component_projections

        if pc_images is None or pc_metrics is None:
            console.echo(message="No principal component data available for this plane.", level=LogLevel.WARNING)
            return

        self._pc_images = np.clip(pc_images, np.percentile(pc_images, 1), np.percentile(pc_images, 99))
        self._image_height, self._image_width = self._pc_images.shape[2:]
        self._pc_metrics = pc_metrics
        if pc_projections is not None:
            self._pc_projections = pc_projections
        else:
            self._pc_projections = np.zeros((1, self._pc_images.shape[1]), dtype=np.float32)

        self._loaded = True
        self._pc_count = self._pc_images.shape[1]
        self._pc_edit.setValidator(QtGui.QIntValidator(1, self._pc_count))
        self._plot_frame()
        self._play_button.setEnabled(True)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        """Handles keyboard navigation for PC stepping and animation control."""
        if event.modifiers() != QtCore.Qt.KeyboardModifier.ShiftModifier:
            if event.key() == QtCore.Qt.Key.Key_Left:
                self._pause_animation()
                pc_number = int(self._pc_edit.text())
                pc_number = max(pc_number - 1, 1)
                self._pc_edit.setText(str(pc_number))
                self._plot_frame()
            elif event.key() == QtCore.Qt.Key.Key_Right:
                self._pause_animation()
                pc_number = int(self._pc_edit.text())
                pc_number = min(pc_number + 1, self._pc_count)
                self._pc_edit.setText(str(pc_number))
                self._plot_frame()
            elif event.key() == QtCore.Qt.Key.Key_Space:
                if self._play_button.isEnabled():
                    self._play_button.setChecked(True)
                    self._start_animation()
                else:
                    self._pause_animation()

    def _open_session(self) -> None:
        """Opens a file dialog to select a session directory and loads PC data."""
        directory = QFileDialog.getExistingDirectory(self, "Open session directory")
        if directory:
            try:
                data = RegistrationViewerData.from_session(root_path=Path(directory))
                self.load_data(data=data)
            except Exception as error:
                console.echo(message=f"Unable to load session. {error}", level=LogLevel.ERROR)

    def _start_animation(self) -> None:
        """Starts PC animation playback."""
        if self._loaded:
            self._play_button.setEnabled(False)
            self._pause_button.setEnabled(True)
            self._update_timer.start(_PC_ANIMATION_INTERVAL_MS)

    def _pause_animation(self) -> None:
        """Pauses PC animation playback."""
        self._update_timer.stop()
        self._play_button.setEnabled(True)
        self._pause_button.setChecked(True)
        self._pause_button.setEnabled(False)

    def _next_frame(self) -> None:
        """Advances the PC animation to the next frame, toggling between top and bottom halves."""
        if self._pc_images is None:
            return
        pc_index = int(self._pc_edit.text()) - 1
        pc_high = self._pc_images[1, pc_index, :, :]
        pc_low = self._pc_images[0, pc_index, :, :]
        if self._current_frame == 0:
            self._animated_image.setImage(np.tile(pc_low[:, :, np.newaxis], (1, 1, 3)))
            self._title_labels[2].setText("top")
        else:
            self._animated_image.setImage(np.tile(pc_high[:, :, np.newaxis], (1, 1, 3)))
            self._title_labels[2].setText("bottom")

        self._animated_image.setLevels([pc_low.min(), pc_low.max()])
        self._current_frame = 1 - self._current_frame

    def _plot_frame(self) -> None:
        """Renders all PC visualizations for the currently selected principal component."""
        if not self._loaded or self._pc_images is None or self._pc_metrics is None or self._pc_projections is None:
            return

        self._title_labels[0].setText("difference")
        self._title_labels[1].setText("merged")
        self._title_labels[2].setText("top")
        pc_index = int(self._pc_edit.text()) - 1
        pc_high = self._pc_images[1, pc_index, :, :]
        pc_low = self._pc_images[0, pc_index, :, :]
        diff = pc_high[:, :, np.newaxis] - pc_low[:, :, np.newaxis]
        diff /= np.abs(diff).max() * 2
        diff += 0.5
        self._difference_image.setImage(np.tile(diff * 255, (1, 1, 3)))
        self._difference_image.setLevels([0, 255])
        rgb = np.zeros((self._pc_images.shape[2], self._pc_images.shape[3], 3), dtype=np.float32)
        rgb[:, :, 0] = (pc_high - pc_high.min()) / (pc_high.max() - pc_high.min()) * 255
        rgb[:, :, 1] = np.minimum(1, np.maximum(0, (pc_low - pc_high.min()) / (pc_high.max() - pc_high.min()))) * 255
        rgb[:, :, 2] = (pc_high - pc_high.min()) / (pc_high.max() - pc_high.min()) * 255
        self._merged_image.setImage(rgb)
        if self._current_frame == 0:
            self._animated_image.setImage(np.tile(pc_low[:, :, np.newaxis], (1, 1, 3)))
        else:
            self._animated_image.setImage(np.tile(pc_high[:, :, np.newaxis], (1, 1, 3)))
        self._animated_image.setLevels([pc_low.min(), pc_low.max()])
        self._zoom_plot()
        self._metrics_plot.clear()
        colors = [(200, 200, 255), (255, 100, 100), (100, 50, 200)]
        metric_names = ["rigid", "nonrigid", "nonrigid max"]
        if self._legend is None:
            self._legend = pg.LegendItem((100, 60), offset=(350, 30))
            self._legend.setParentItem(self._metrics_plot)
            draw_legend = True
        else:
            draw_legend = False
        for index in range(3):
            curve = self._metrics_plot.plot(
                np.arange(1, self._pc_count + 1), self._pc_metrics[:, index], pen=colors[index]
            )
            if draw_legend:
                self._legend.addItem(curve, metric_names[index])
            self._metric_labels[index].setText(f"{metric_names[index]}: {self._pc_metrics[pc_index, index]:.3f}")
        self._metrics_scatter = pg.ScatterPlotItem()
        self._metrics_plot.addItem(self._metrics_scatter)
        self._metrics_scatter.setData(
            [pc_index + 1, pc_index + 1, pc_index + 1],
            self._pc_metrics[pc_index, :].tolist(),
            size=_SCATTER_POINT_SIZE,
            brush=pg.mkBrush(255, 255, 255),
        )
        self._metrics_plot.setLabel("left", "pixel shift")
        self._metrics_plot.setLabel("bottom", "PC #")

        self._projection_plot.clear()
        self._projection_plot.plot(self._pc_projections[:, pc_index])
        self._projection_plot.setLabel("left", "magnitude")
        self._projection_plot.setLabel("bottom", "time")
        self.show()
        self._zoom_plot()

    def _zoom_plot(self) -> None:
        """Resets all PC image view ranges to fit the full image extent."""
        self._difference_view_box.setXRange(0, self._image_width)
        self._difference_view_box.setYRange(0, self._image_height)
        self._merged_view_box.setXRange(0, self._image_width)
        self._merged_view_box.setYRange(0, self._image_height)
        self._animated_view_box.setXRange(0, self._image_width)
        self._animated_view_box.setYRange(0, self._image_height)

    def _plot_clicked(self, event: object) -> None:
        """Handles double-click to zoom the PC image plots."""
        if self._loaded:
            items = self._graphics_widget.scene().items(event.scenePos())  # type: ignore[attr-defined]
            for item in items:
                if (
                    item in (self._difference_view_box, self._merged_view_box, self._animated_view_box)
                    and event.button() == 1  # type: ignore[attr-defined]
                    and event.double()  # type: ignore[attr-defined]
                ):
                    self._zoom_plot()

    def _create_buttons(self) -> None:
        """Creates and lays out the open, play, and pause buttons."""
        icon_size = QtCore.QSize(_ICON_SIZE, _ICON_SIZE)
        open_button = QToolButton()
        open_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        open_button.setIconSize(icon_size)
        open_button.setToolTip("Open session directory")
        open_button.clicked.connect(self._open_session)

        self._play_button: QToolButton = QToolButton()
        self._play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self._play_button.setIconSize(icon_size)
        self._play_button.setToolTip("Play")
        self._play_button.setCheckable(True)
        self._play_button.clicked.connect(self._start_animation)

        self._pause_button: QToolButton = QToolButton()
        self._pause_button.setCheckable(True)
        self._pause_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self._pause_button.setIconSize(icon_size)
        self._pause_button.setToolTip("Pause")
        self._pause_button.clicked.connect(self._pause_animation)

        button_group = QButtonGroup(self)
        button_group.addButton(self._play_button, 0)
        button_group.addButton(self._pause_button, 1)
        button_group.setExclusive(True)

        self._layout.addWidget(open_button, 0, 0, 1, 1)
        self._layout.addWidget(self._play_button, 14, 12, 1, 1)
        self._layout.addWidget(self._pause_button, 14, 13, 1, 1)
        self._play_button.setEnabled(False)
        self._pause_button.setEnabled(False)
        self._pause_button.setChecked(True)


def _subsample_frames(
    frame_count: int,
    frame_height: int,
    frame_width: int,
    sample_count: int,
    registration_path: str,
) -> NDArray[np.int16]:
    """Reads evenly-spaced frames from a binary file for dynamic range estimation.

    Args:
        frame_count: Total number of frames in the binary file.
        frame_height: Height of each frame in pixels.
        frame_width: Width of each frame in pixels.
        sample_count: Number of frames to subsample.
        registration_path: Path to the registered binary file.

    Returns:
        Array of subsampled frames with shape (sample_count, height, width).
    """
    frames = np.zeros((sample_count, frame_height, frame_width), dtype=np.int16)
    bytes_per_frame = 2 * frame_height * frame_width
    start_indices = np.linspace(start=0, stop=frame_count, num=1 + sample_count).astype(np.int64)
    with Path(registration_path).open("rb") as binary_file:
        for index in range(sample_count):
            binary_file.seek(bytes_per_frame * start_indices[index], 0)
            buffer = binary_file.read(bytes_per_frame)
            data = np.frombuffer(buffer, dtype=np.int16, offset=0)
            frames[index, :, :] = data.reshape((frame_height, frame_width))
    return frames
