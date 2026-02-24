"""Provides registration binary viewer and principal component metrics viewer windows."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass

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

from .context_data import RegistrationViewerData

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class _BinaryPlayerStyle:
    """Encapsulates visual and behavioral constants for the BinaryPlayer window."""

    white_label_stylesheet: str = "color: white;"
    """Stylesheet for white label text on a dark background."""

    scatter_point_size: int = 10
    """Marker size in pixels for red dot overlays on shift, nonrigid, and z-position plots."""

    icon_size: int = 30
    """Dimension in pixels for media control button icons."""

    playback_speed_multiplier: int = 5
    """Factor by which the real-time frame period is divided to compute the playback timer interval."""

    subsample_frame_count: int = 100
    """Number of evenly-spaced frames subsampled for dynamic range estimation."""

    min_frame_delta: int = 5
    """Minimum frame increment for arrow key navigation."""

    frame_delta_divisor: int = 200
    """Divisor for computing frame slider step size from total frame count."""

    display_range_low_sigma: float = 2.0
    """Standard deviations below mean for display range lower bound."""

    display_range_high_sigma: float = 5.0
    """Standard deviations above mean for display range upper bound."""

    z_percentile_low: int = 1
    """Lower percentile for z-stack display range clipping."""

    z_percentile_high: int = 99
    """Upper percentile for z-stack display range clipping."""

    z_edit_width: int = 30
    """Width in pixels for the z-plane input field."""


@dataclass(frozen=True, slots=True)
class _PCViewerStyle:
    """Encapsulates visual and behavioral constants for the PCViewer window."""

    font_family: str = "Arial"
    """Font family used for metric labels and PC input field."""

    white_label_stylesheet: str = "color: white;"
    """Stylesheet for white label text on a dark background."""

    scatter_point_size: int = 10
    """Marker size in pixels for the selected PC indicator on the metrics plot."""

    icon_size: int = 30
    """Dimension in pixels for media control button icons."""

    animation_interval_ms: int = 200
    """Interval in milliseconds between PC extreme image animation updates."""

    pc_edit_width: int = 40
    """Width in pixels for the principal component number input field."""

    metrics_font_size: int = 14
    """Point size for metric value labels and PC input field font."""


class BinaryPlayer(QMainWindow):
    """Provides a playback window for viewing registered binary imaging data and evaluating the registration's quality.

    Args:
        data: Pre-loaded registration data to display on startup.

    Attributes:
        _style: Frozen style constants for the binary player window.
        _data: The registration viewer data model. Always present; the viewer requires data to function.
        _z_loaded: Determines whether a z-stack has been loaded.
        _z_on: Determines whether the z-stack side view is currently visible.
        _z_correlation: Cached z-position correlation array, or None if not computed.
        _channel_2_visible: Determines whether the channel 2 overlay is currently displayed.
        _current_frame: Index of the currently displayed frame.
        _frame_delta: Frame step size for arrow key navigation.
        _display_range: Low and high bounds for the image display range.
        _time_step: Timer interval in milliseconds for playback speed.
        _image: Current frame image buffer.
        _z_stack: Loaded z-stack volume, or None if not loaded.
        _z_height: Height of z-stack planes in pixels.
        _z_width: Width of z-stack planes in pixels.
        _z_display_range: Percentile-based display range for z-stack images as a 2-element array.
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
        _movie_label: Label displaying the current recording path.
        _frame_label: Label for the frame slider.
        _frame_number_label: Label displaying the current frame number.
        _frame_slider: Horizontal slider for frame navigation.
        _plane_selector: Dropdown for selecting the imaging plane.
        _play_button: Button to start video playback.
        _pause_button: Button to pause video playback.
        _compute_z_button: Button to compute z-position correlations.
        _update_timer: Timer driving frame advancement during playback.
    """

    _style: _BinaryPlayerStyle = _BinaryPlayerStyle()
    """Frozen style constants for the binary player window."""

    # Notifies listeners when the user selects a different imaging plane from the plane selector.
    plane_changed = QtCore.Signal(int)

    def __init__(self, data: RegistrationViewerData) -> None:
        """Initializes the binary player window and all UI components."""
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

        # Initializes state flags and data model.
        self._z_loaded: bool = False
        self._z_on: bool = False
        self._channel_2_visible: bool = False
        self._z_correlation: NDArray[np.float32] | None = None
        self._data: RegistrationViewerData = data

        # Initializes playback state.
        self._current_frame: int = 0
        self._frame_delta: int = 10
        self._display_range: NDArray[np.float32] = np.zeros((2,), dtype=np.float32)
        self._time_step: float = 0.0
        self._image: NDArray[np.int16] | None = None

        # Initializes z-stack data.
        self._z_stack: NDArray[np.int16] | None = None
        self._z_height: int = 0
        self._z_width: int = 0
        self._z_display_range: NDArray[np.float32] = np.zeros((2,), dtype=np.float32)
        self._z_max_positions: NDArray[np.int32] | None = None

        # Configures main image view.
        self._main_view_box: pg.ViewBox = pg.ViewBox(lockAspect=True, invertY=True, name="plot1")
        # noinspection PyUnresolvedReferences,PyArgumentList
        self._graphics_widget.addItem(self._main_view_box, row=0, col=0)
        self._main_view_box.setMenuEnabled(False)
        self._main_image: pg.ImageItem = pg.ImageItem()
        self._main_view_box.addItem(self._main_image)

        # Configures side box for z-stack display.
        self._side_view_box: pg.ViewBox = pg.ViewBox(lockAspect=True, invertY=True)
        self._side_view_box.setMenuEnabled(False)
        self._side_image: pg.ImageItem = pg.ImageItem()
        self._side_view_box.addItem(self._side_image)

        # Configures channel 2 checkbox.
        self._channel_2_checkbox: QCheckBox = QCheckBox("view channel 2")
        self._channel_2_checkbox.setStyleSheet(self._style.white_label_stylesheet)
        self._channel_2_checkbox.setEnabled(False)
        self._channel_2_checkbox.toggled.connect(self._toggle_channel_2)
        self._layout.addWidget(self._channel_2_checkbox, 0, 5, 1, 1)

        # Configures z-stack checkbox.
        self._z_stack_checkbox: QCheckBox = QCheckBox("view z-stack")
        self._z_stack_checkbox.setStyleSheet(self._style.white_label_stylesheet)
        self._z_stack_checkbox.setEnabled(False)
        self._z_stack_checkbox.toggled.connect(self._add_z_stack)
        self._layout.addWidget(self._z_stack_checkbox, 0, 8, 1, 1)

        z_label = QLabel("Z-plane:")
        z_label.setStyleSheet(self._style.white_label_stylesheet)
        self._layout.addWidget(z_label, 0, 9, 1, 1)

        self._z_plane_edit: QLineEdit = QLineEdit(self)
        self._z_plane_edit.setValidator(QtGui.QIntValidator(0, 0))
        self._z_plane_edit.setText("0")
        self._z_plane_edit.setFixedWidth(self._style.z_edit_width)
        self._z_plane_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._layout.addWidget(self._z_plane_edit, 0, 10, 1, 1)

        # Configures rigid registration offset plot.
        # noinspection PyUnresolvedReferences
        self._shift_plot = self._graphics_widget.addPlot(name="plot_shift", row=1, col=0, colspan=2)
        self._shift_plot.setMouseEnabled(x=True, y=False)
        self._shift_plot.setMenuEnabled(False)
        self._shift_scatter: pg.ScatterPlotItem = pg.ScatterPlotItem()
        self._shift_scatter.setData([0, 0], [0, 0])
        self._shift_plot.addItem(self._shift_scatter)

        # Configures nonrigid RMS displacement plot.
        # noinspection PyUnresolvedReferences
        self._nonrigid_plot = self._graphics_widget.addPlot(name="plot_nonrigid", row=2, col=0, colspan=2)
        self._nonrigid_plot.setMouseEnabled(x=True, y=False)
        self._nonrigid_plot.setMenuEnabled(False)
        self._nonrigid_scatter: pg.ScatterPlotItem = pg.ScatterPlotItem()
        self._nonrigid_plot.setXLink("plot_shift")

        # Configures z-position correlation plot.
        # noinspection PyUnresolvedReferences
        self._z_position_plot = self._graphics_widget.addPlot(name="plot_Z", row=3, col=0, colspan=2)
        self._z_position_plot.setMouseEnabled(x=True, y=False)
        self._z_position_plot.setMenuEnabled(False)
        self._z_position_scatter: pg.ScatterPlotItem = pg.ScatterPlotItem()
        self._z_position_plot.setXLink("plot_shift")

        # noinspection PyUnresolvedReferences
        self._graphics_widget.ci.layout.setRowStretchFactor(0, 12)
        self._movie_label: QLabel = QLabel("No recording loaded")
        self._movie_label.setStyleSheet(self._style.white_label_stylesheet)
        self._movie_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self._create_buttons()

        # Configures plane selector dropdown.
        plane_label = QLabel("Plane:")
        plane_label.setStyleSheet(self._style.white_label_stylesheet)
        self._layout.addWidget(plane_label, 6, 0, 1, 1)
        self._plane_selector: QComboBox = QComboBox(self)
        self._plane_selector.setEnabled(False)
        self._plane_selector.currentIndexChanged.connect(self._on_plane_changed)
        self._layout.addWidget(self._plane_selector, 6, 1, 1, 1)

        # Configures frame slider.
        self._frame_label: QLabel = QLabel("Current frame:")
        self._frame_label.setStyleSheet(self._style.white_label_stylesheet)
        self._frame_number_label: QLabel = QLabel("0")
        self._frame_number_label.setStyleSheet(self._style.white_label_stylesheet)
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
        hint_label.setStyleSheet(self._style.white_label_stylesheet)
        self._layout.addWidget(hint_label, 16, 0, 1, 3)
        self._frame_slider.valueChanged.connect(self._go_to_frame)
        self._layout.addWidget(self._movie_label, 0, 0, 1, 5)
        self._update_frame_slider()
        self._update_buttons()
        self._update_timer: QtCore.QTimer = QtCore.QTimer()
        self._update_timer.timeout.connect(self._next_frame)
        # noinspection PyUnresolvedReferences
        self._graphics_widget.scene().sigMouseClicked.connect(self._plot_clicked)

        self.load_data(data=data)

    @property
    def data(self) -> RegistrationViewerData:
        """The registration viewer data model wrapping all planes."""
        return self._data

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
                self._current_frame = max(0, min(self._data.frame_count - 1, self._current_frame))
                self._frame_slider.setValue(self._current_frame)
            elif event.key() == QtCore.Qt.Key.Key_Right:
                self._current_frame += self._frame_delta
                self._current_frame = max(0, min(self._data.frame_count - 1, self._current_frame))
                self._frame_slider.setValue(self._current_frame)
        if event.modifiers() != QtCore.Qt.KeyboardModifier.ShiftModifier and event.key() == QtCore.Qt.Key.Key_Space:
            if self._play_button.isEnabled():
                self._start_playback()
            else:
                self._pause_playback()

    def _create_buttons(self) -> None:
        """Creates and lays out all control buttons for the player window."""
        icon_size = QtCore.QSize(self._style.icon_size, self._style.icon_size)
        load_recording_button = QPushButton("Load Recording")
        load_recording_button.setToolTip("Open a cindra recording directory")
        load_recording_button.clicked.connect(self._load_recording)

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

        self._layout.addWidget(load_recording_button, 1, 0, 1, 2)
        self._layout.addWidget(load_z_button, 2, 0, 1, 2)
        self._layout.addWidget(self._compute_z_button, 3, 0, 1, 2)
        self._layout.addWidget(self._play_button, 15, 0, 1, 1)
        self._layout.addWidget(self._pause_button, 15, 1, 1, 1)
        self._play_button.setEnabled(False)
        self._pause_button.setEnabled(False)
        self._pause_button.setChecked(True)

    def _update_frame_slider(self) -> None:
        """Configures the frame slider range and enables it."""
        self._frame_slider.setMaximum(self._data.frame_count - 1)
        self._frame_slider.setMinimum(0)
        self._frame_label.setEnabled(True)
        self._frame_slider.setEnabled(True)

    def _update_buttons(self) -> None:
        """Sets the initial enabled state for play and pause buttons."""
        self._play_button.setEnabled(True)
        self._pause_button.setEnabled(False)
        self._pause_button.setChecked(True)

    def _load_recording(self) -> None:
        """Opens a file dialog to select a recording directory and loads it."""
        directory = QFileDialog.getExistingDirectory(self, "Open recording directory")
        if directory:
            try:
                data = RegistrationViewerData.from_recording(root_path=Path(directory))
                self.load_data(data=data)
            except Exception as error:
                console.echo(message=f"Unable to load recording. {error}", level=LogLevel.ERROR)

    def _on_plane_changed(self, index: int) -> None:
        """Handles plane selector index changes by switching to the selected plane.

        Args:
            index: Zero-based index of the selected plane.
        """
        if index < 0:
            return
        self._data.switch_plane(plane_index=index)
        self._open_plane()
        self.plane_changed.emit(index)

    def _open_plane(self) -> None:
        """Configures views for the current plane's data."""
        self._setup_views()

    def _setup_views(self) -> None:
        """Configures all plot views and display parameters after loading data."""
        self._shift_plot.clear()

        # Computes dynamic range from subsampled frames.
        frames = self._data.binary_file.subsample_movie(sample_count=self._style.subsample_frame_count)
        frame_mean = np.float32(frames.mean())
        frame_std = np.float32(frames.std())
        self._display_range = frame_mean + frame_std * np.array(
            [-self._style.display_range_low_sigma, self._style.display_range_high_sigma], dtype=np.float32
        )

        self._movie_label.setText(self._data.registered_binary_display_path)

        # Loads aspect ratio from data model.
        self._main_view_box.setAspectLocked(lock=True, ratio=self._data.aspect_ratio)
        self._side_view_box.setAspectLocked(lock=True, ratio=self._data.aspect_ratio)

        frame_count = self._data.frame_count
        self._time_step = 1.0 / self._data.sampling_rate * 1000 / self._style.playback_speed_multiplier
        self._frame_delta = max(self._style.min_frame_delta, int(frame_count / self._style.frame_delta_divisor))
        self._frame_slider.setSingleStep(self._frame_delta)
        if frame_count > 0:
            self._update_frame_slider()
            self._update_buttons()

        # Plots registration X-Y offsets.
        rigid_y = self._data.rigid_y_offsets
        rigid_x = self._data.rigid_x_offsets
        self._shift_plot.plot(rigid_y, pen="g")
        self._shift_plot.plot(rigid_x, pen="y")
        shift_min = min(int(rigid_y.min()), int(rigid_x.min()))
        shift_max = max(int(rigid_y.max()), int(rigid_x.max()))
        # Prevents a zero-height range when all offsets are zero, which causes pyqtgraph to compute
        # infinite scale factors that overflow on cast to integer pixel coordinates.
        if shift_min == shift_max:
            shift_min -= 1
            shift_max += 1
        self._shift_plot.setRange(
            xRange=(0, frame_count),
            yRange=(shift_min, shift_max),
            padding=0.0,
        )
        self._shift_plot.setLimits(xMin=0, xMax=frame_count)
        self._shift_scatter = pg.ScatterPlotItem()
        self._shift_plot.addItem(self._shift_scatter)
        self._shift_scatter.setData(
            [self._current_frame, self._current_frame],
            [int(rigid_y[self._current_frame]), int(rigid_x[self._current_frame])],
            size=self._style.scatter_point_size,
            brush=pg.mkBrush(255, 0, 0),
        )

        # Plots per-frame nonrigid RMS displacement if available.
        self._nonrigid_plot.clear()
        nonrigid_rms = self._data.nonrigid_rms
        if self._data.has_nonrigid and nonrigid_rms is not None:
            self._nonrigid_plot.plot(nonrigid_rms, pen=(180, 100, 255))
            nonrigid_max = float(nonrigid_rms.max())
            # Prevents a zero-height range when all nonrigid displacements are zero.
            if nonrigid_max == 0.0:
                nonrigid_max = 1.0
            self._nonrigid_plot.setRange(
                xRange=(0, frame_count),
                yRange=(0.0, nonrigid_max),
                padding=0.0,
            )
            self._nonrigid_plot.setLimits(xMin=0, xMax=frame_count)
            self._nonrigid_plot.setLabel("left", "nonrigid RMS", units="px")
            self._nonrigid_scatter = pg.ScatterPlotItem()
            self._nonrigid_plot.addItem(self._nonrigid_scatter)

        self._channel_2_checkbox.setEnabled(self._data.has_channel_2)

        self._current_frame = -1
        self._next_frame()

    def _next_frame(self) -> None:
        """Advances to the next frame and updates all display elements."""
        self._current_frame += 1
        frame_count = self._data.frame_count
        if self._current_frame > frame_count - 1:
            self._current_frame = 0

        self._image = np.asarray(self._data.binary_file[self._current_frame])

        if self._data.has_channel_2 and self._channel_2_visible:
            binary_channel_2 = self._data.binary_file_channel_2
            if binary_channel_2 is not None:
                channel_2_frame = np.asarray(binary_channel_2[self._current_frame])[:, :, np.newaxis]
                self._image = np.concatenate(
                    (self._image[:, :, np.newaxis], channel_2_frame, np.zeros_like(channel_2_frame)),
                    axis=-1,
                )

        if self._z_loaded and self._z_on and self._z_stack is not None:
            if self._z_max_positions is not None:
                self._z_plane_edit.setText(str(self._z_max_positions[self._current_frame]))
            z_plane_frame = np.asarray(self._z_stack[int(self._z_plane_edit.text())])
            self._side_image.setImage(z_plane_frame, levels=self._z_display_range)

        self._main_image.setImage(self._image, levels=self._display_range)
        self._frame_slider.setValue(self._current_frame)
        self._frame_number_label.setText(str(self._current_frame))

        rigid_y = self._data.rigid_y_offsets
        rigid_x = self._data.rigid_x_offsets
        self._shift_scatter.setData(
            [self._current_frame, self._current_frame],
            [int(rigid_y[self._current_frame]), int(rigid_x[self._current_frame])],
            size=self._style.scatter_point_size,
            brush=pg.mkBrush(255, 0, 0),
        )

        nonrigid_rms = self._data.nonrigid_rms
        if self._data.has_nonrigid and nonrigid_rms is not None:
            # noinspection PyTypeChecker
            self._nonrigid_scatter.setData(
                [self._current_frame],
                [float(nonrigid_rms[self._current_frame])],
                size=self._style.scatter_point_size,
                brush=pg.mkBrush(255, 0, 0),
            )
        if self._z_loaded and self._z_on and self._z_max_positions is not None:
            # noinspection PyTypeChecker
            z_position = int(self._z_max_positions[self._current_frame])
            self._z_position_scatter.setData(
                [self._current_frame, self._current_frame],
                [z_position, z_position],
                size=self._style.scatter_point_size,
                brush=pg.mkBrush(255, 0, 0),
            )

    def _go_to_frame(self) -> None:
        """Seeks to the frame indicated by the frame slider position."""
        self._current_frame = int(self._frame_slider.value())
        self._jump_to_frame()

    def _jump_to_frame(self) -> None:
        """Jumps to the current frame position and displays it."""
        if self._play_button.isEnabled():
            self._current_frame = max(0, min(self._data.frame_count - 1, self._current_frame))
            self._current_frame = int(self._current_frame)
            self._current_frame -= 1
            self._next_frame()

    def _start_playback(self) -> None:
        """Starts video playback by enabling the frame update timer."""
        if self._current_frame < self._data.frame_count - 1:
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

    def _toggle_channel_2(self) -> None:
        """Toggles channel 2 display based on checkbox state."""
        self._channel_2_visible = self._channel_2_checkbox.isChecked()
        self._next_frame()

    def _add_z_stack(self) -> None:
        """Toggles z-stack side view display based on checkbox state."""
        if self._z_stack_checkbox.isChecked():
            self._z_on = True
            # noinspection PyUnresolvedReferences,PyArgumentList
            self._graphics_widget.addItem(self._side_view_box, row=0, col=1)
        else:
            self._z_on = False
            # noinspection PyUnresolvedReferences
            self._graphics_widget.removeItem(self._side_view_box)
        self._next_frame()

    def _zoom_image(self) -> None:
        """Resets the main and side view zoom to fit the full image extent."""
        self._main_view_box.setRange(
            yRange=(0, self._data.frame_height), xRange=(0, self._data.frame_width)
        )
        if self._z_on:
            self._side_view_box.setRange(yRange=(0, self._z_height), xRange=(0, self._z_width))
            self._side_view_box.setXLink("plot1")
            self._side_view_box.setYLink("plot1")

    def _plot_clicked(self, event: object) -> None:
        """Handles mouse click events on plots for frame navigation."""
        # noinspection PyUnresolvedReferences
        items = self._graphics_widget.scene().items(event.scenePos())  # type: ignore[attr-defined]
        position_x = 0
        is_time_plot = False
        zoom = False
        choose = False
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

        frame_count = self._data.frame_count
        if zoom:
            self._shift_plot.setRange(xRange=(0, frame_count))
            self._nonrigid_plot.setRange(xRange=(0, frame_count))
            self._z_position_plot.setRange(xRange=(0, frame_count))

        if choose and self._play_button.isEnabled():
            self._current_frame = max(0, min(frame_count - 1, round(position_x)))
            self._frame_slider.setValue(self._current_frame)

    def _load_z_stack(self) -> None:
        """Opens a file dialog to load a z-stack TIFF and initializes z-position tracking."""
        file_dialog_result = QFileDialog.getOpenFileName(self, "Open zstack", filter="*.tif")
        z_stack_path = file_dialog_result[0]
        try:
            self._z_stack = imread(z_stack_path)
            self._z_height, self._z_width = self._z_stack.shape[1:]
            self._z_plane_edit.setValidator(QtGui.QIntValidator(0, self._z_stack.shape[0]))
            self._z_display_range = np.array(
                [
                    np.percentile(self._z_stack, self._style.z_percentile_low),
                    np.percentile(self._z_stack, self._style.z_percentile_high),
                ],
                dtype=np.float32,
            )

            self._compute_z_button.setEnabled(True)
            self._z_loaded = True
            self._z_stack_checkbox.setEnabled(True)
            self._z_stack_checkbox.setChecked(True)
            self._z_max_positions = np.zeros(self._data.frame_count, dtype=np.int32)

            # Checks for cached z-correlation data in order of priority:
            # 1. Local instance cache (self._z_correlation)
            # 2. Separate zcorr.npy file from the output directory
            if self._z_correlation is not None and self._z_stack.shape[0] == self._z_correlation.shape[0]:
                self._z_max_positions = np.argmax(
                    gaussian_filter1d(self._z_correlation.T.copy(), sigma=self._data.temporal_smoothing_sigma, axis=1),
                    axis=1,
                ).astype(np.int32)
                self._plot_z_correlation()
            elif self._data.output_directory is not None:
                z_correlation_path = self._data.output_directory / "zcorr.npy"
                if z_correlation_path.exists():
                    self._z_correlation = np.load(z_correlation_path).astype(np.float32)
                    if self._z_stack.shape[0] == self._z_correlation.shape[0]:
                        self._z_max_positions = np.argmax(
                            gaussian_filter1d(
                                self._z_correlation.T.copy(), sigma=self._data.temporal_smoothing_sigma, axis=1
                            ),
                            axis=1,
                        ).astype(np.int32)
                        self._plot_z_correlation()

        except Exception as error:
            console.echo(message=f"Unable to load z-stack TIFF. {error}", level=LogLevel.ERROR)

    def _compute_z(self) -> None:
        """Computes z-position correlations between the loaded z-stack and the registered binary."""
        if self._z_stack is None:
            return

        try:
            self._z_correlation = self._data.compute_z_correlations(
                z_stack=self._z_stack.astype(np.float32),
            )
            self._z_max_positions = np.argmax(
                gaussian_filter1d(self._z_correlation.T.copy(), sigma=self._data.temporal_smoothing_sigma, axis=1),
                axis=1,
            ).astype(np.int32)

            # Persists the correlation array to the output directory for future recordings.
            if self._data.output_directory is not None:
                z_correlation_path = self._data.output_directory / "zcorr.npy"
                np.save(z_correlation_path, self._z_correlation)
                console.echo(message=f"Z-position correlations saved to: {z_correlation_path}")

            self._plot_z_correlation()
        except Exception as error:
            console.echo(message=f"Unable to compute z-position correlations. {error}", level=LogLevel.ERROR)

    def _plot_z_correlation(self) -> None:
        """Plots the z-position correlation trace on the z-position plot."""
        if self._z_max_positions is None:
            return
        frame_count = self._data.frame_count
        self._z_position_plot.clear()
        self._z_position_plot.plot(self._z_max_positions, pen="r")
        self._z_position_plot.addItem(self._z_position_scatter)
        # noinspection PyTypeChecker
        self._z_position_plot.setRange(
            xRange=(0, frame_count),
            yRange=(int(self._z_max_positions.min()), int(self._z_max_positions.max()) + 3),
            padding=0.0,
        )
        self._z_position_plot.setLimits(xMin=0, xMax=frame_count)
        self._z_position_plot.setXLink("plot_shift")


class PCViewer(QMainWindow):
    """Provides a viewer window for principal component registration metrics.

    Args:
        data: Pre-loaded registration data to display on startup.

    Attributes:
        _style: Frozen style constants for the PC viewer window.
        _data: The registration viewer data model. Always present; the viewer requires data to function.
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

    _style: _PCViewerStyle = _PCViewerStyle()
    """Frozen style constants for the PC viewer window."""

    def __init__(self, data: RegistrationViewerData) -> None:
        """Initializes the PC viewer window and all UI components."""
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

        # Initializes state and data.
        self._data: RegistrationViewerData = data
        self._loaded: bool = False
        self._current_frame: int = 0
        self._pc_count: int = data.principal_component_count
        self._pc_images: NDArray[np.float32] | None = None
        self._image_height: int = 0
        self._image_width: int = 0
        self._pc_metrics: NDArray[np.float32] | None = None
        self._pc_projections: NDArray[np.float32] | None = None
        self._metrics_scatter: pg.ScatterPlotItem | None = None
        self._legend: pg.LegendItem | None = None

        # Configures pixel shift metrics plot.
        # noinspection PyUnresolvedReferences
        self._metrics_plot = self._graphics_widget.addPlot(row=0, col=0)
        self._metrics_plot.setMouseEnabled(x=False, y=False)
        self._metrics_plot.setMenuEnabled(False)

        # noinspection PyUnresolvedReferences
        self._difference_view_box = self._graphics_widget.addViewBox(
            name="plot1",
            lockAspect=True,
            row=1,
            col=0,
            invertY=True,
        )
        # noinspection PyUnresolvedReferences
        self._merged_view_box = self._graphics_widget.addViewBox(lockAspect=True, row=1, col=1, invertY=True)
        self._merged_view_box.setMenuEnabled(False)
        self._merged_view_box.setXLink("plot1")
        self._merged_view_box.setYLink("plot1")
        # noinspection PyUnresolvedReferences
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
        # noinspection PyUnresolvedReferences
        self._graphics_widget.scene().sigMouseClicked.connect(self._plot_clicked)

        # noinspection PyUnresolvedReferences
        self._projection_plot = self._graphics_widget.addPlot(row=0, col=1, colspan=2)
        self._projection_plot.setMouseEnabled(x=False)
        self._projection_plot.setMenuEnabled(False)

        self._pc_edit: QLineEdit = QLineEdit(self)
        self._pc_edit.setText("1")
        self._pc_edit.setFixedWidth(self._style.pc_edit_width)
        self._pc_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._pc_edit.returnPressed.connect(self._plot_frame)
        self._pc_edit.textEdited.connect(self._pause_animation)
        pc_label = QLabel("PC: ")
        bold_font = QtGui.QFont(
            self._style.font_family, pointSize=self._style.metrics_font_size, weight=QtGui.QFont.Weight.Bold
        )
        big_font = QtGui.QFont(self._style.font_family, pointSize=self._style.metrics_font_size)
        pc_label.setFont(bold_font)
        self._pc_edit.setFont(big_font)
        pc_label.setStyleSheet(self._style.white_label_stylesheet)
        self._layout.addWidget(QLabel(""), 1, 0, 1, 1)
        self._layout.addWidget(pc_label, 2, 0, 1, 1)
        self._layout.addWidget(self._pc_edit, 2, 1, 1, 1)
        self._metric_labels: list[QLabel] = []
        self._title_labels: list[QLabel] = []
        for index in range(3):
            metric_label = QLabel("")
            metric_label.setStyleSheet(self._style.white_label_stylesheet)
            self._layout.addWidget(metric_label, 3 + index, 0, 1, 2)
            self._metric_labels.append(metric_label)
            title_label = QLabel("")
            title_label.setStyleSheet(self._style.white_label_stylesheet)
            self._layout.addWidget(title_label, 12, 4 + index * 4, 1, 2)
            self._title_labels.append(title_label)
        self._layout.addWidget(QLabel(""), 7, 0, 1, 1)
        self._layout.setRowStretch(7, 1)
        self._create_buttons()
        self._pc_edit.setValidator(QtGui.QIntValidator(1, self._pc_count))
        self._update_timer: QtCore.QTimer = QtCore.QTimer()
        self._update_timer.timeout.connect(self._next_frame)

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

    def _create_buttons(self) -> None:
        """Creates and lays out the open, play, and pause buttons."""
        icon_size = QtCore.QSize(self._style.icon_size, self._style.icon_size)
        open_button = QToolButton()
        open_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogOpenButton))
        open_button.setIconSize(icon_size)
        open_button.setToolTip("Open recording directory")
        open_button.clicked.connect(self._open_recording)

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

    def _open_recording(self) -> None:
        """Opens a file dialog to select a recording directory and loads PC data."""
        directory = QFileDialog.getExistingDirectory(self, "Open recording directory")
        if directory:
            try:
                data = RegistrationViewerData.from_recording(root_path=Path(directory))
                self.load_data(data=data)
            except Exception as error:
                console.echo(message=f"Unable to load recording. {error}", level=LogLevel.ERROR)

    def _start_animation(self) -> None:
        """Starts PC animation playback."""
        if self._loaded:
            self._play_button.setEnabled(False)
            self._pause_button.setEnabled(True)
            self._update_timer.start(self._style.animation_interval_ms)

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
        pc_high = np.asarray(self._pc_images[1, pc_index, :, :])
        pc_low = np.asarray(self._pc_images[0, pc_index, :, :])
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
        pc_high = np.asarray(self._pc_images[1, pc_index, :, :])
        pc_low = np.asarray(self._pc_images[0, pc_index, :, :])
        difference = np.asarray(pc_high[:, :, np.newaxis] - pc_low[:, :, np.newaxis])
        difference /= np.abs(difference).max() * 2
        difference += 0.5
        self._difference_image.setImage(np.tile(difference * 255, (1, 1, 3)))
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
                np.arange(1, self._pc_count + 1, dtype=np.int32), self._pc_metrics[:, index], pen=colors[index]
            )
            if draw_legend:
                self._legend.addItem(curve, metric_names[index])
            self._metric_labels[index].setText(f"{metric_names[index]}: {self._pc_metrics[pc_index, index]:.3f}")
        self._metrics_scatter = pg.ScatterPlotItem()
        self._metrics_plot.addItem(self._metrics_scatter)
        self._metrics_scatter.setData(
            [pc_index + 1, pc_index + 1, pc_index + 1],
            np.asarray(self._pc_metrics[pc_index, :]).tolist(),
            size=self._style.scatter_point_size,
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
            # noinspection PyUnresolvedReferences
            items = self._graphics_widget.scene().items(event.scenePos())  # type: ignore[attr-defined]
            for item in items:
                if (
                    item in (self._difference_view_box, self._merged_view_box, self._animated_view_box)
                    and event.button() == 1  # type: ignore[attr-defined]
                    and event.double()  # type: ignore[attr-defined]
                ):
                    self._zoom_plot()
