"""Provides registration binary viewer and principal component metrics viewer windows."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from PySide6 import QtGui, QtCore
import pyqtgraph as pg  # type: ignore[import-untyped]
from PySide6.QtWidgets import (
    QLabel,
    QStyle,
    QSlider,
    QWidget,
    QComboBox,
    QLineEdit,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
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
    """Marker size in pixels for red dot overlays on shift and nonrigid plots."""

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

    plot_title_size: str = "14pt"
    """Font size for plot titles above the registration offset plots."""

    axis_label_size: str = "12pt"
    """Font size for axis labels on the registration offset plots."""

    legend_label_size: str = "12pt"
    """Font size for legend entries on the registration offset plots."""


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

    plot_title_size: str = "14pt"
    """Font size for plot titles above the metrics and projection plots."""

    axis_label_size: str = "12pt"
    """Font size for axis labels on the metrics and projection plots."""


class BinaryPlayer(QMainWindow):
    """Displays a UI window for viewing registered binary imaging data and evaluating the registration's quality.

    Args:
        data: Pre-loaded registration data to display on startup.

    Attributes:
        _style: Frozen style constants for the binary player window.
        data: The RegistrationViewerData instance that stores the visualized recording's data.
        _channel_2_visible: Determines whether the channel 2 overlay is currently displayed.
        _current_frame: Index of the currently displayed frame.
        _frame_delta: Frame step size for arrow key navigation.
        _display_range: Low and high bounds for the image display range.
        _time_step: Timer interval in milliseconds for playback speed.
        _image: Current frame image buffer.
        _central_widget: Central widget container.
        _layout: Grid layout for arranging all controls and views.
        _graphics_widget: PyQtGraph graphics layout for image and plot views.
        _main_view_box: View box for the primary registered image display.
        _main_image: Image item for the primary frame display.
        _channel_2_button: Button for toggling channel 2 overlay.
        _shift_plot: Plot widget for rigid registration X-Y offsets.
        _shift_scatter: Scatter plot overlay indicating the current frame on the shift plot.
        _nonrigid_plot: Plot widget for nonrigid RMS displacement.
        _nonrigid_scatter: Scatter plot overlay indicating the current frame on the nonrigid plot.
        _movie_label: Label displaying the current recording path.
        _frame_number_label: Label displaying the current frame number.
        _frame_slider: Horizontal slider for frame navigation.
        _plane_selector: Dropdown for selecting the imaging plane.
        _play_button: Button to start video playback.
        _pause_button: Button to pause video playback.
        _update_timer: Timer driving frame advancement during playback.
    """

    _style: _BinaryPlayerStyle = _BinaryPlayerStyle()
    """Frozen style constants for the binary player window."""

    # Notifies listeners when the user selects a different imaging plane from the plane selector.
    plane_changed = QtCore.Signal(int)

    def __init__(self, data: RegistrationViewerData) -> None:
        super().__init__()

        # Adds the main UI window.
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(70, 70, 1070, 1070)
        self.setWindowTitle("Registered Recording")
        self._central_widget: QWidget = QWidget(self)
        self.setCentralWidget(self._central_widget)
        self._layout: QGridLayout = QGridLayout()
        self._central_widget.setLayout(self._layout)
        # Initializes state flags and recording data.
        self._channel_2_visible: bool = False
        self.data: RegistrationViewerData = data

        # Initializes playback state.
        self._current_frame: int = 0
        self._frame_delta: int = 10
        self._display_range: NDArray[np.float32] = np.zeros((2,), dtype=np.float32)
        self._time_step: float = 0.0
        self._image: NDArray[np.int16] | None = None

        # Row 0: Toolbar with recording controls arranged in a horizontal layout. Widgets keep their
        # natural size; only the spacing between them grows when the window is resized.
        toolbar = QHBoxLayout()

        self._movie_label: QLabel = QLabel("Current Path: (none)")
        self._movie_label.setStyleSheet(self._style.white_label_stylesheet)
        toolbar.addWidget(self._movie_label)

        load_recording_button = QPushButton("Load New Recording")
        load_recording_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        load_recording_button.setToolTip("Opens a cindra-processed recording directory.")
        load_recording_button.clicked.connect(self._load_recording)
        toolbar.addWidget(load_recording_button)

        # Groups the plane label and dropdown tightly so there is no gap between them.
        plane_label = QLabel("Plane:")
        plane_label.setStyleSheet(self._style.white_label_stylesheet)
        self._plane_selector: QComboBox = QComboBox(self)
        self._plane_selector.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._plane_selector.setEnabled(False)
        self._plane_selector.currentIndexChanged.connect(self._on_plane_changed)
        toolbar.addWidget(plane_label)
        toolbar.addWidget(self._plane_selector)

        # Channel 2 toggle button. Disabled until a recording with two channels is loaded.
        self._channel_2_button: QPushButton = QPushButton("View Channel 2")
        self._channel_2_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._channel_2_button.setEnabled(False)
        self._channel_2_button.clicked.connect(self._toggle_channel_2)
        toolbar.addWidget(self._channel_2_button)

        # Trailing stretch absorbs extra horizontal space so widgets stay at their natural size.
        toolbar.addStretch()
        self._layout.addLayout(toolbar, 0, 0, 1, 6)

        # Row 1: Graphics widget spanning the full width. Contains the main image view and the
        # registration offset plots.
        self._graphics_widget: pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget()
        self._layout.addWidget(self._graphics_widget, 1, 0, 1, 6)
        self._layout.setRowStretch(1, 1)

        # Configures main image view.
        self._main_view_box: pg.ViewBox = pg.ViewBox(lockAspect=True, invertY=True, name="plot1")
        # noinspection PyUnresolvedReferences,PyArgumentList
        self._graphics_widget.addItem(self._main_view_box, row=0, col=0)
        self._main_view_box.setMenuEnabled(False)
        self._main_image: pg.ImageItem = pg.ImageItem()
        self._main_view_box.addItem(self._main_image)

        # Configures rigid registration offset plot. The bottom axis tick labels are hidden since the
        # x-axis is shared with the nonrigid plot below, which displays the "Frame" label.
        # noinspection PyUnresolvedReferences
        self._shift_plot = self._graphics_widget.addPlot(name="plot_shift", row=1, col=0, colspan=2)
        self._shift_plot.setMouseEnabled(x=True, y=False)
        self._shift_plot.setMenuEnabled(False)
        self._shift_plot.setTitle("Rigid Registration Offsets", size=self._style.plot_title_size, bold=True)
        self._shift_plot.setLabel("left", "Shift (px)", **{"font-size": self._style.axis_label_size})
        self._shift_plot.getAxis("bottom").setStyle(showValues=False)
        self._shift_plot.getAxis("bottom").setHeight(0)
        self._shift_plot.addLegend(horSpacing=20, colCount=2, labelTextSize=self._style.legend_label_size)
        self._shift_scatter: pg.ScatterPlotItem = pg.ScatterPlotItem()
        self._shift_scatter.setData([0, 0], [0, 0])
        self._shift_plot.addItem(self._shift_scatter)

        # Configures nonrigid RMS displacement plot. The x-axis is linked to the rigid plot so both
        # share the same "Frame" range, with the label shown only on this bottom plot.
        # noinspection PyUnresolvedReferences
        self._nonrigid_plot = self._graphics_widget.addPlot(name="plot_nonrigid", row=2, col=0, colspan=2)
        self._nonrigid_plot.setMouseEnabled(x=True, y=False)
        self._nonrigid_plot.setMenuEnabled(False)
        self._nonrigid_plot.setTitle("Nonrigid RMS Displacement", size=self._style.plot_title_size, bold=True)
        self._nonrigid_plot.setLabel("left", "RMS Shift (px)", **{"font-size": self._style.axis_label_size})
        self._nonrigid_plot.setLabel("bottom", "Frame", **{"font-size": self._style.axis_label_size})
        self._nonrigid_scatter: pg.ScatterPlotItem = pg.ScatterPlotItem()
        self._nonrigid_plot.setXLink("plot_shift")

        # noinspection PyUnresolvedReferences
        self._graphics_widget.ci.layout.setRowStretchFactor(0, 12)

        # Row 2: Current frame label.
        self._frame_number_label: QLabel = QLabel("Current frame: 0")
        self._frame_number_label.setStyleSheet(self._style.white_label_stylesheet)
        self._layout.addWidget(self._frame_number_label, 2, 0, 1, 6)

        # Row 3: Playback controls and frame slider.
        self._create_buttons()
        self._frame_slider: QSlider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self._frame_slider.setTickInterval(5)
        self._frame_slider.setTracking(False)
        self._frame_slider.valueChanged.connect(self._go_to_frame)
        self._layout.addWidget(self._frame_slider, 3, 2, 1, 4)

        self._update_frame_slider()
        self._update_buttons()
        self._update_timer: QtCore.QTimer = QtCore.QTimer()
        self._update_timer.timeout.connect(self._next_frame)
        # noinspection PyUnresolvedReferences
        self._graphics_widget.scene().sigMouseClicked.connect(self._plot_clicked)

        self.load_data(data=data)

    def load_data(self, data: RegistrationViewerData) -> None:
        """Caches the input RegistrationViewerData instance and uses it to populate the managed UI window.

        Args:
            data: The RegistrationViewerData instance that stores the visualized recording's data.
        """
        self.data = data

        # Populates the plane selector without triggering _on_plane_changed yet. Signals are blocked so that
        # clearing and re-adding items does not fire redundant plane-switch callbacks.
        self._plane_selector.blockSignals(True)
        self._plane_selector.clear()
        for label in data.plane_labels:
            self._plane_selector.addItem(label)
        self._plane_selector.setCurrentIndex(data.current_plane_index)
        self._plane_selector.blockSignals(False)

        # Only enables the plane selector when multiple planes are available.
        self._plane_selector.setEnabled(data.plane_count > 1)

        # Configures all plot views and display parameters for the initially selected plane.
        self._open_plane()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        """Handles keyboard navigation for frame stepping and playback control.

        Notes:
            Overrides the Qt virtual method. The camelCase name is required to match the parent signature.
        """
        # Left/right arrow keys step through frames when playback is paused. Shift is ignored to avoid
        # conflicting with Qt widget focus shortcuts.
        if self._play_button.isEnabled() and event.modifiers() != QtCore.Qt.KeyboardModifier.ShiftModifier:
            if event.key() == QtCore.Qt.Key.Key_Left:
                self._current_frame -= self._frame_delta
                self._current_frame = max(0, min(self.data.frame_count - 1, self._current_frame))
                self._frame_slider.setValue(self._current_frame)
            elif event.key() == QtCore.Qt.Key.Key_Right:
                self._current_frame += self._frame_delta
                self._current_frame = max(0, min(self.data.frame_count - 1, self._current_frame))
                self._frame_slider.setValue(self._current_frame)

        # Spacebar toggles between play and pause.
        if event.modifiers() != QtCore.Qt.KeyboardModifier.ShiftModifier and event.key() == QtCore.Qt.Key.Key_Space:
            if self._play_button.isEnabled():
                self._start_playback()
            else:
                self._pause_playback()

    def _create_buttons(self) -> None:
        """Creates and lays out playback control buttons for the player window."""
        icon_size = QtCore.QSize(self._style.icon_size, self._style.icon_size)

        # Playback controls. Play and pause are grouped exclusively so only one can be active at a time.
        self._play_button: QToolButton = QToolButton()
        self._play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self._play_button.setIconSize(icon_size)
        self._play_button.setToolTip("Play (Space).")
        self._play_button.setCheckable(True)
        self._play_button.clicked.connect(self._start_playback)

        self._pause_button: QToolButton = QToolButton()
        self._pause_button.setCheckable(True)
        self._pause_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self._pause_button.setIconSize(icon_size)
        self._pause_button.setToolTip("Pause (Space). Use left/right arrow keys to step through frames.")
        self._pause_button.clicked.connect(self._pause_playback)

        button_group = QButtonGroup(self)
        button_group.addButton(self._play_button, 0)
        button_group.addButton(self._pause_button, 1)
        button_group.setExclusive(True)

        # Places play/pause buttons on row 3 alongside the frame slider. Controls start disabled with
        # pause pre-selected, since there is no active playback on startup.
        self._layout.addWidget(self._play_button, 3, 0, 1, 1)
        self._layout.addWidget(self._pause_button, 3, 1, 1, 1)
        self._play_button.setEnabled(False)
        self._pause_button.setEnabled(False)
        self._pause_button.setChecked(True)

    def _update_frame_slider(self) -> None:
        """Configures the frame slider range and enables it."""
        self._frame_slider.setMaximum(self.data.frame_count - 1)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setEnabled(True)

    def _update_buttons(self) -> None:
        """Sets the initial enabled state for play and pause buttons."""
        self._play_button.setEnabled(True)
        self._pause_button.setEnabled(False)
        self._pause_button.setChecked(True)

    def _load_recording(self) -> None:
        """Displays a file dialog that allows users to select a new recording to visualize."""
        # Defaults the file dialog to the parent of the currently loaded recording's output directory,
        # so the user can easily navigate to a sibling recording.
        start_dir = ""
        output = self.data.output_directory
        if output is not None:
            parent = output.parent
            if parent.is_dir():
                start_dir = str(parent)
        directory = QFileDialog.getExistingDirectory(self, "Specify the recording directory to open.", start_dir)
        if directory:
            data = RegistrationViewerData.from_recording(root_path=Path(directory))
            self.load_data(data=data)

    def _on_plane_changed(self, index: int) -> None:
        """Handles plane selector index changes by switching to the selected plane.

        Args:
            index: The index of the recording's plane to switch to.
        """
        if index < 0:
            return
        self.data.switch_plane(plane_index=index)
        self._open_plane()
        self.plane_changed.emit(index)

    def _open_plane(self) -> None:
        """Configures views for the current plane's data."""
        self._setup_views()

    def _setup_views(self) -> None:
        """Configures all plot views and display parameters after loading data."""
        # Temporarily breaks the x-link between plots so that clearing and re-populating one plot
        # does not trigger auto-range propagation that corrupts the other plot's x-axis.
        self._nonrigid_plot.setXLink(None)
        self._shift_plot.clear()
        self._nonrigid_plot.clear()
        self._shift_plot.disableAutoRange()
        self._nonrigid_plot.disableAutoRange()

        # Computes dynamic range from subsampled frames.
        frames = self.data.binary_file.subsample_movie(sample_count=self._style.subsample_frame_count)
        frame_mean = np.float32(frames.mean())
        frame_std = np.float32(frames.std())
        self._display_range = frame_mean + frame_std * np.array(
            [-self._style.display_range_low_sigma, self._style.display_range_high_sigma], dtype=np.float32
        )

        self._movie_label.setText(f"Current Path: {self.data.recording_label}")

        # Loads aspect ratio from recording data.
        self._main_view_box.setAspectLocked(lock=True, ratio=self.data.aspect_ratio)

        # Configures the frame display slider.
        frame_count = self.data.frame_count
        last_frame = frame_count - 1
        self._time_step = 1.0 / self.data.sampling_rate * 1000 / self._style.playback_speed_multiplier
        self._frame_delta = max(self._style.min_frame_delta, int(frame_count / self._style.frame_delta_divisor))
        self._frame_slider.setSingleStep(self._frame_delta)
        if frame_count > 0:
            self._update_frame_slider()
            self._update_buttons()

        # Plots registration X-Y offsets. Explicit x values ensure the frame axis maps correctly.
        rigid_y = self.data.rigid_y_offsets
        rigid_x = self.data.rigid_x_offsets
        x_values = np.arange(frame_count)
        self._shift_plot.plot(x_values, rigid_y, pen="g", name="Y")
        self._shift_plot.plot(x_values, rigid_x, pen="y", name="X")
        shift_min = min(int(rigid_y.min()), int(rigid_x.min()))
        shift_max = max(int(rigid_y.max()), int(rigid_x.max()))
        # Prevents a zero-height range when all offsets are zero, which causes pyqtgraph to compute
        # infinite scale factors that overflow on cast to integer pixel coordinates.
        if shift_min == shift_max:
            shift_min -= 1
            shift_max += 1
        self._shift_plot.setLimits(xMin=0, xMax=last_frame)
        self._shift_plot.setRange(xRange=(0, last_frame), yRange=(shift_min, shift_max), padding=0.0)
        self._shift_scatter = pg.ScatterPlotItem()
        self._shift_plot.addItem(self._shift_scatter)
        self._shift_scatter.setData(
            [0, 0],
            [int(rigid_y[0]), int(rigid_x[0])],
            size=self._style.scatter_point_size,
            brush=pg.mkBrush(255, 0, 0),
        )

        # Plots per-frame nonrigid RMS displacement if available.
        nonrigid_rms = self.data.nonrigid_rms
        if self.data.has_nonrigid and nonrigid_rms is not None:
            self._nonrigid_plot.plot(x_values, nonrigid_rms, pen=(180, 100, 255))
            nonrigid_max = float(nonrigid_rms.max())
            # Prevents a zero-height range when all nonrigid displacements are zero.
            if nonrigid_max == 0.0:
                nonrigid_max = 1.0
            self._nonrigid_plot.setLimits(xMin=0, xMax=last_frame)
            self._nonrigid_plot.setRange(xRange=(0, last_frame), yRange=(0.0, nonrigid_max), padding=0.0)
            self._nonrigid_scatter = pg.ScatterPlotItem()
            self._nonrigid_plot.addItem(self._nonrigid_scatter)

        # Restores the x-link after both plots have their ranges set.
        self._nonrigid_plot.setXLink("plot_shift")

        self._channel_2_button.setEnabled(self.data.has_channel_2)

        self._current_frame = -1
        self._next_frame()

    def _next_frame(self) -> None:
        """Advances to the next frame and updates all display elements."""
        # Advances frame index, wrapping back to zero at the end of the recording.
        self._current_frame += 1
        frame_count = self.data.frame_count
        if self._current_frame > frame_count - 1:
            self._current_frame = 0

        # Reads the current frame from the registered binary.
        self._image = np.asarray(self.data.binary_file[self._current_frame])

        # If channel 2 overlay is active, composites both channels into an RGB image with channel 1 in
        # the red plane and channel 2 in the green plane.
        if self.data.has_channel_2 and self._channel_2_visible:
            binary_channel_2 = self.data.binary_file_channel_2
            if binary_channel_2 is not None:
                channel_2_frame = np.asarray(binary_channel_2[self._current_frame])[:, :, np.newaxis]
                self._image = np.concatenate(
                    (self._image[:, :, np.newaxis], channel_2_frame, np.zeros_like(channel_2_frame)),
                    axis=-1,
                )

        # Updates the main image display and frame navigation controls.
        self._main_image.setImage(self._image, levels=self._display_range)
        self._frame_slider.setValue(self._current_frame)
        self._frame_number_label.setText(f"Current frame: {self._current_frame}")

        # Moves the red dot indicator on the rigid registration offset plot to the current frame.
        rigid_y = self.data.rigid_y_offsets
        rigid_x = self.data.rigid_x_offsets
        self._shift_scatter.setData(
            [self._current_frame, self._current_frame],
            [int(rigid_y[self._current_frame]), int(rigid_x[self._current_frame])],
            size=self._style.scatter_point_size,
            brush=pg.mkBrush(255, 0, 0),
        )

        # Moves the red dot indicator on the nonrigid RMS plot to the current frame, if available.
        nonrigid_rms = self.data.nonrigid_rms
        if self.data.has_nonrigid and nonrigid_rms is not None:
            # noinspection PyTypeChecker
            self._nonrigid_scatter.setData(
                [self._current_frame],
                [float(nonrigid_rms[self._current_frame])],
                size=self._style.scatter_point_size,
                brush=pg.mkBrush(255, 0, 0),
            )

    def _go_to_frame(self) -> None:
        """Seeks to the frame indicated by the frame slider position.

        Notes:
            Serves as a callback target for the position update signal.
        """
        self._current_frame = int(self._frame_slider.value())
        self._jump_to_frame()

    def _jump_to_frame(self) -> None:
        """Jumps to the current frame position and displays it."""
        if self._play_button.isEnabled():
            self._current_frame = max(0, min(self.data.frame_count - 1, self._current_frame))
            self._current_frame = int(self._current_frame)
            self._current_frame -= 1
            self._next_frame()

    def _start_playback(self) -> None:
        """Starts video playback by enabling the frame update timer."""
        if self._current_frame < self.data.frame_count - 1:
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
        """Toggles channel 2 overlay and updates the button label accordingly."""
        self._channel_2_visible = not self._channel_2_visible
        self._channel_2_button.setText("Hide Channel 2" if self._channel_2_visible else "View Channel 2")
        self._next_frame()

    def _zoom_image(self) -> None:
        """Resets the main view zoom to fit the full image extent."""
        self._main_view_box.setRange(yRange=(0, self.data.frame_height), xRange=(0, self.data.frame_width))

    def _plot_clicked(self, event: object) -> None:
        """Handles mouse click events on plots for frame navigation."""
        # Resolves which graphics items lie under the click position.
        # noinspection PyUnresolvedReferences
        items = self._graphics_widget.scene().items(event.scenePos())  # type: ignore[attr-defined]
        position_x = 0
        is_time_plot = False
        zoom = False
        seek_to_frame = False
        for item in items:
            # Click landed on a time-series plot: maps the scene position to a frame index.
            if item in (self._shift_plot, self._nonrigid_plot):
                view_box = self._shift_plot.vb
                position = view_box.mapSceneToView(event.scenePos())  # type: ignore[attr-defined]
                position_x = position.x()
                is_time_plot = True
            # Double-click on the main image resets the zoom to fit the full frame.
            elif item == self._main_view_box:
                if event.button() == 1 and event.double():  # type: ignore[attr-defined]
                    self._zoom_image()
            # For time-series plots, a single click seeks to that frame; a double click resets the
            # x-axis zoom to the full recording range.
            if is_time_plot and event.button() == 1:  # type: ignore[attr-defined]
                if event.double():  # type: ignore[attr-defined]
                    zoom = True
                else:
                    seek_to_frame = True

        # Resets x-axis zoom on all time-series plots when double-clicked.
        frame_count = self.data.frame_count
        if zoom:
            self._shift_plot.setRange(xRange=(0, frame_count - 1))
            self._nonrigid_plot.setRange(xRange=(0, frame_count - 1))

        # Seeks to the clicked frame when playback is paused.
        if seek_to_frame and self._play_button.isEnabled():
            self._current_frame = max(0, min(frame_count - 1, round(position_x)))
            self._frame_slider.setValue(self._current_frame)


class PCViewer(QMainWindow):
    """Displays a UI window for viewing the principal component registration metrics.

    Args:
        data: Pre-loaded registration data to display on startup.

    Attributes:
        _style: Frozen style constants for the PC viewer window.
        data: The RegistrationViewerData instance that stores the visualized recording's data.
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
        _title_labels: Text items anchored inside each image view box, positioned at the bottom center.
        _play_button: Button to start PC animation playback.
        _pause_button: Button to pause PC animation playback.
        _update_timer: Timer driving the PC animation.
        _metrics_scatter: Scatter plot overlay indicating the selected PC on the metrics plot, or None.
        _legend: Legend item for the metrics plot, or None.
        _metrics_y_range: Global y-axis range for the metrics plot, computed across all PCs.
        _projection_y_range: Global y-axis range for the projection plot, computed across all PCs.
    """

    _style: _PCViewerStyle = _PCViewerStyle()
    """Frozen style constants for the PC viewer window."""

    def __init__(self, data: RegistrationViewerData) -> None:
        # Initializes the main viewer window.
        super().__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(70, 70, 1300, 800)
        self.setWindowTitle("Registration Quality Metrics")
        self._central_widget: QWidget = QWidget(self)
        self.setCentralWidget(self._central_widget)
        self._layout: QGridLayout = QGridLayout()
        self._central_widget.setLayout(self._layout)

        # Initializes state and data.
        self.data: RegistrationViewerData = data
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
        self._metrics_y_range: tuple[float, float] = (0.0, 1.0)
        self._projection_y_range: tuple[float, float] = (0.0, 1.0)

        # Row 0: Graphics widget spans the full width. Row stretch gives it all available vertical space.
        self._graphics_widget: pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget()
        self._layout.addWidget(self._graphics_widget, 0, 0, 1, 1)
        self._layout.setRowStretch(0, 1)
        self._layout.setRowStretch(1, 0)

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

        # Title labels anchored inside each image view box so they follow zoom and pan.
        self._title_labels: list[pg.TextItem] = []
        for view_box in (self._difference_view_box, self._merged_view_box, self._animated_view_box):
            label = pg.TextItem("", color="w", anchor=(0.5, 0))
            view_box.addItem(label)
            self._title_labels.append(label)

        # noinspection PyUnresolvedReferences
        self._graphics_widget.scene().sigMouseClicked.connect(self._plot_clicked)

        # noinspection PyUnresolvedReferences
        self._projection_plot = self._graphics_widget.addPlot(row=0, col=1, colspan=2)
        self._projection_plot.setMouseEnabled(x=False)
        self._projection_plot.setMenuEnabled(False)

        # Bottom control panel: PC selector, metric labels, title labels, playback controls.
        self._create_bottom_panel()
        self._pc_edit.setValidator(QtGui.QIntValidator(1, self._pc_count))
        self._update_timer: QtCore.QTimer = QtCore.QTimer()
        self._update_timer.timeout.connect(self._next_frame)

        self.load_data(data=data)

    def load_data(self, data: RegistrationViewerData) -> None:
        """Loads principal component registration data from the RegistrationViewerData instance.

        Args:
            data: The RegistrationViewerData instance that stores the visualized recording's data.
        """
        # Extracts PC arrays from the recording data.
        self.data = data
        pc_images = data.principal_component_extreme_images
        pc_metrics = data.principal_component_shift_metrics
        pc_projections = data.principal_component_projections

        # Aborts if the recording has no PC registration metrics (e.g. registration was skipped).
        if pc_images is None or pc_metrics is None:
            console.echo(message="No principal component data available for this plane.", level=LogLevel.WARNING)
            return

        # Clips extreme pixel values to the 1st-99th percentile range for stable image display.
        self._pc_images = np.clip(pc_images, np.percentile(pc_images, 1), np.percentile(pc_images, 99))
        self._image_height, self._image_width = self._pc_images.shape[2:]
        self._pc_metrics = pc_metrics
        # Falls back to a zero array when the recording has no per-frame PC projections.
        if pc_projections is not None:
            self._pc_projections = pc_projections
        else:
            self._pc_projections = np.zeros((1, self._pc_images.shape[1]), dtype=np.float32)

        # Updates the PC count and constrains the input validator to the available range.
        self._loaded = True
        self._pc_count = self._pc_images.shape[1]
        self._pc_edit.setValidator(QtGui.QIntValidator(1, self._pc_count))

        # Pre-computes global axis ranges so plots stay stable when cycling through PCs.
        self._metrics_y_range = (float(self._pc_metrics.min()), float(self._pc_metrics.max()))
        self._projection_y_range = (float(self._pc_projections.min()), float(self._pc_projections.max()))

        # Renders the first PC and enables playback controls.
        self._plot_frame()
        self._play_button.setEnabled(True)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        """Handles keyboard navigation for PC stepping and animation control.

        Notes:
            Overrides the Qt virtual method. The camelCase name is required to match the parent signature.
        """
        if event.modifiers() != QtCore.Qt.KeyboardModifier.ShiftModifier:
            # Left/right arrow keys step through principal components, pausing animation first.
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
            # Spacebar toggles between play and pause for the PC extreme image animation.
            elif event.key() == QtCore.Qt.Key.Key_Space:
                if self._play_button.isEnabled():
                    self._play_button.setChecked(True)
                    self._start_animation()
                else:
                    self._pause_animation()

    def _create_bottom_panel(self) -> None:
        """Creates the bottom control panel with the PC selector, metric labels, and playback controls.

        Widgets keep their natural size; only the trailing stretch grows when the window is resized.
        Fixed spacing separates each logical group.
        """
        bold_font = QtGui.QFont(
            self._style.font_family, pointSize=self._style.metrics_font_size, weight=QtGui.QFont.Weight.Bold
        )
        big_font = QtGui.QFont(self._style.font_family, pointSize=self._style.metrics_font_size)
        panel = QHBoxLayout()
        group_spacing = 20

        # PC selector: label and input field for the current principal component number.
        pc_label = QLabel("PC:")
        pc_label.setFont(bold_font)
        pc_label.setStyleSheet(self._style.white_label_stylesheet)
        self._pc_edit: QLineEdit = QLineEdit(self)
        self._pc_edit.setText("1")
        self._pc_edit.setFixedWidth(self._style.pc_edit_width)
        self._pc_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._pc_edit.setFont(big_font)
        self._pc_edit.setToolTip("Principal component number (Left/Right arrow keys to step).")
        self._pc_edit.returnPressed.connect(self._plot_frame)
        self._pc_edit.textEdited.connect(self._pause_animation)
        panel.addWidget(pc_label)
        panel.addWidget(self._pc_edit)
        panel.addSpacing(group_spacing)

        # Metric value labels showing per-PC registration shift magnitudes.
        self._metric_labels: list[QLabel] = []
        for _ in range(3):
            metric_label = QLabel("")
            metric_label.setStyleSheet(self._style.white_label_stylesheet)
            panel.addWidget(metric_label)
            self._metric_labels.append(metric_label)
        panel.addSpacing(group_spacing)

        # Playback controls. Play and pause are grouped exclusively so only one can be active at a time.
        icon_size = QtCore.QSize(self._style.icon_size, self._style.icon_size)
        self._play_button: QToolButton = QToolButton()
        self._play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self._play_button.setIconSize(icon_size)
        self._play_button.setToolTip("Play (Space).")
        self._play_button.setCheckable(True)
        self._play_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._play_button.clicked.connect(self._start_animation)

        self._pause_button: QToolButton = QToolButton()
        self._pause_button.setCheckable(True)
        self._pause_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self._pause_button.setIconSize(icon_size)
        self._pause_button.setToolTip("Pause (Space). Use Left/Right arrow keys to step through PCs.")
        self._pause_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._pause_button.clicked.connect(self._pause_animation)

        button_group = QButtonGroup(self)
        button_group.addButton(self._play_button, 0)
        button_group.addButton(self._pause_button, 1)
        button_group.setExclusive(True)

        # Controls start disabled with pause pre-selected, since there is no active playback on startup.
        panel.addWidget(self._play_button)
        panel.addWidget(self._pause_button)

        # Trailing stretch absorbs extra horizontal space so widgets stay at their natural size.
        panel.addStretch()

        self._play_button.setEnabled(False)
        self._pause_button.setEnabled(False)
        self._pause_button.setChecked(True)

        self._layout.addLayout(panel, 1, 0)

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

        # Extracts the high- and low-projection mean images for the currently selected PC.
        pc_index = int(self._pc_edit.text()) - 1
        pc_high = np.asarray(self._pc_images[1, pc_index, :, :])
        pc_low = np.asarray(self._pc_images[0, pc_index, :, :])

        # Alternates the animated view between the top (low-projection) and bottom (high-projection) extremes.
        if self._current_frame == 0:
            self._animated_image.setImage(np.tile(pc_low[:, :, np.newaxis], (1, 1, 3)))
            self._title_labels[2].setText("top")
        else:
            self._animated_image.setImage(np.tile(pc_high[:, :, np.newaxis], (1, 1, 3)))
            self._title_labels[2].setText("bottom")
        # Uses the low-projection range for both frames so brightness stays consistent across toggles.
        self._animated_image.setLevels([pc_low.min(), pc_low.max()])

        # Flips the toggle state for the next timer tick.
        self._current_frame = 1 - self._current_frame

    def _plot_frame(self) -> None:
        """Renders all PC visualizations for the currently selected principal component."""
        if not self._loaded or self._pc_images is None or self._pc_metrics is None or self._pc_projections is None:
            return

        # Extracts the high- and low-projection mean images for the selected PC.
        self._title_labels[0].setText("difference")
        self._title_labels[1].setText("merged")
        self._title_labels[2].setText("top")
        pc_index = int(self._pc_edit.text()) - 1
        pc_high = np.asarray(self._pc_images[1, pc_index, :, :])
        pc_low = np.asarray(self._pc_images[0, pc_index, :, :])

        # Difference image: high minus low, normalized to 0-1 centered at 0.5, then scaled to 0-255.
        difference = np.asarray(pc_high[:, :, np.newaxis] - pc_low[:, :, np.newaxis])
        difference /= np.abs(difference).max() * 2
        difference += 0.5
        self._difference_image.setImage(np.tile(difference * 255, (1, 1, 3)))
        self._difference_image.setLevels([0, 255])

        # Merged image: red/blue channels show the high-projection image, green shows the low-projection image.
        # Regions that differ between top and bottom appear as magenta or green tint.
        rgb = np.zeros((self._pc_images.shape[2], self._pc_images.shape[3], 3), dtype=np.float32)
        rgb[:, :, 0] = (pc_high - pc_high.min()) / (pc_high.max() - pc_high.min()) * 255
        rgb[:, :, 1] = np.minimum(1, np.maximum(0, (pc_low - pc_high.min()) / (pc_high.max() - pc_high.min()))) * 255
        rgb[:, :, 2] = (pc_high - pc_high.min()) / (pc_high.max() - pc_high.min()) * 255
        self._merged_image.setImage(rgb)

        # Animated image: shows whichever extreme the animation toggle is currently on.
        if self._current_frame == 0:
            self._animated_image.setImage(np.tile(pc_low[:, :, np.newaxis], (1, 1, 3)))
        else:
            self._animated_image.setImage(np.tile(pc_high[:, :, np.newaxis], (1, 1, 3)))
        self._animated_image.setLevels([pc_low.min(), pc_low.max()])
        self._zoom_plot()

        # Metrics plot: shows rigid, nonrigid, and nonrigid-max shift magnitudes across all PCs.
        # The legend is created once on the first call and reused on subsequent PC switches.
        self._metrics_plot.clear()
        colors = [(200, 200, 255), (255, 100, 100), (100, 50, 200)]
        metric_names = ["rigid", "nonrigid", "nonrigid max"]
        if self._legend is None:
            self._legend = self._metrics_plot.addLegend()
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

        # White scatter dots mark the selected PC's position on each metric curve.
        self._metrics_scatter = pg.ScatterPlotItem()
        self._metrics_plot.addItem(self._metrics_scatter)
        self._metrics_scatter.setData(
            [pc_index + 1, pc_index + 1, pc_index + 1],
            np.asarray(self._pc_metrics[pc_index, :]).tolist(),
            size=self._style.scatter_point_size,
            brush=pg.mkBrush(255, 255, 255),
        )
        self._metrics_plot.setTitle("PC Extreme Alignment Shifts", size=self._style.plot_title_size, bold=True)
        self._metrics_plot.setLabel("left", "Shift (px)", **{"font-size": self._style.axis_label_size})
        self._metrics_plot.setLabel("bottom", "PC #", **{"font-size": self._style.axis_label_size})
        self._metrics_plot.setXRange(1, self._pc_count)
        self._metrics_plot.setYRange(*self._metrics_y_range)

        # Projection plot: shows the per-frame projection onto the selected PC over time.
        self._projection_plot.clear()
        self._projection_plot.plot(self._pc_projections[:, pc_index])
        self._projection_plot.setTitle("PC Projection Time Course", size=self._style.plot_title_size, bold=True)
        self._projection_plot.setLabel(
            "left", "PC Projection Magnitude (x0.001)", **{"font-size": self._style.axis_label_size}
        )
        self._projection_plot.setLabel("bottom", "sampled frame", **{"font-size": self._style.axis_label_size})
        self._projection_plot.setXRange(0, self._pc_projections.shape[0] - 1)
        self._projection_plot.setYRange(*self._projection_y_range)

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
        # Positions title labels at the bottom center of each image.
        center_x = self._image_width / 2
        for label in self._title_labels:
            label.setPos(center_x, self._image_height)

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
