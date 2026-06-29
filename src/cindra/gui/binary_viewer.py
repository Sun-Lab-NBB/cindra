"""Provides the registration binary viewer window for frame playback and offset visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path

import numpy as np
from PySide6 import QtGui, QtCore
import pyqtgraph as pg  # type: ignore[import-untyped]
from PySide6.QtWidgets import (
    QMenu,
    QLabel,
    QStyle,
    QSlider,
    QWidget,
    QLineEdit,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QToolButton,
)
from ataraxis_base_utilities import LogLevel, console

from .styles import FONTS, STYLE, COLORS, PLOT_STYLE, BINARY_STYLE
from .widgets import configure_plot, add_plot_legend, escape_returns_focus, create_play_pause_group
from .constants import BINARY_CONFIG
from .viewer_context import SingleRecordingData

if TYPE_CHECKING:
    from numpy.typing import NDArray


class BinaryPlayer(QMainWindow):
    """Displays a UI window for viewing registered binary imaging data and evaluating the registration's quality.

    Args:
        data: Pre-loaded registration data to display on startup.

    Attributes:
        data: The SingleRecordingData instance that stores the visualized recording's data.
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
        _offset_plot: Plot widget for rigid registration X-Y offsets.
        _offset_scatter: Scatter plot overlay indicating the current frame on the offset plot.
        _average_rigid_y_offsets: Average rigid Y offsets across all planes.
        _average_rigid_x_offsets: Average rigid X offsets across all planes.

        _step_edit: Input field for the frame step size.
        _frame_number_label: Label displaying the current frame number.
        _frame_slider: Horizontal slider for frame navigation.
        _play_button: Button to start video playback.
        _pause_button: Button to pause video playback.
        _update_timer: Timer driving frame advancement during playback.
    """

    # Notifies listeners when the user loads a new recording via the File menu.
    recording_changed = QtCore.Signal()

    def __init__(self, data: SingleRecordingData) -> None:
        super().__init__()

        # Adds the main UI window.
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(*BINARY_STYLE.window_geometry)
        self.setWindowTitle("Registered Recording")
        self._central_widget: QWidget = QWidget(self)
        self.setCentralWidget(self._central_widget)
        self._layout: QGridLayout = QGridLayout()
        self._central_widget.setLayout(self._layout)
        # Initializes state flags and recording data.
        self._channel_2_visible: bool = False
        self.data: SingleRecordingData = data

        # Initializes playback state.
        self._current_frame: int = 0
        self._frame_delta: int = 100
        self._display_range: NDArray[np.float32] = np.zeros((2,), dtype=np.float32)
        self._time_step: float = 0.0
        self._image: NDArray[np.int16] | None = None
        self._average_rigid_y_offsets: NDArray[np.float32] = np.zeros(1, dtype=np.float32)
        self._average_rigid_x_offsets: NDArray[np.float32] = np.zeros(1, dtype=np.float32)

        # Row 0: Toolbar with file menu and recording controls on a single line.
        toolbar = QHBoxLayout()

        # File menu button with dropdown for loading recordings.
        self._file_button: QPushButton = QPushButton("File")
        self._file_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._file_button.setToolTip("Load a recording for visualization.")
        file_menu = QMenu(self)
        file_menu.setStyleSheet(STYLE.menu)
        load_action = file_menu.addAction("&Load recording")
        load_action.setShortcut("Ctrl+L")
        load_action.triggered.connect(self._load_recording)
        self._file_button.setMenu(file_menu)
        toolbar.addWidget(self._file_button)

        # Channel 2 toggle button. Hidden until a recording with two channels is loaded.
        self._channel_2_button: QPushButton = QPushButton("View Channel 2")
        self._channel_2_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._channel_2_button.setVisible(False)
        self._channel_2_button.setToolTip("Toggle the channel 2 overlay.")
        self._channel_2_button.clicked.connect(self._toggle_channel_2)
        toolbar.addWidget(self._channel_2_button)

        # Hint label for keyboard shortcuts.
        hint_label = QLabel(
            "Hint: Use arrows to navigate recording's frames / adjust frame step size,"
            " use space to toggle recording playback."
        )
        hint_label.setStyleSheet(STYLE.white_label)
        hint_label.setFont(FONTS.small_bold)
        toolbar.addWidget(hint_label)
        toolbar.addStretch()
        self._layout.addLayout(toolbar, 0, 0, 1, 6)

        # Row 1: Graphics widget spanning the full width. Contains the main image view and the
        # registration offset plots.
        self._graphics_widget: pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget()
        self._layout.addWidget(self._graphics_widget, 1, 0, 1, 6)
        self._layout.setRowStretch(1, 1)

        # Configures main image view.
        self._main_view_box: pg.ViewBox = pg.ViewBox(lockAspect=True, invertY=True, name="plot1")
        self._graphics_widget.addItem(self._main_view_box, row=0, col=0)
        self._main_view_box.setMenuEnabled(False)
        self._main_image: pg.ImageItem = pg.ImageItem()
        self._main_view_box.addItem(self._main_image)

        # Configures rigid registration offset plot.
        self._offset_plot = self._graphics_widget.addPlot(name="plot_offset", row=1, col=0, colspan=2)
        configure_plot(
            self._offset_plot,
            title="Rigid Registration Offsets",
            left_label="Offset (px)",
            bottom_label="Frame",
        )

        self._graphics_widget.ci.layout.setRowStretchFactor(0, BINARY_STYLE.image_plot_stretch[0])
        self._graphics_widget.ci.layout.setRowStretchFactor(1, BINARY_STYLE.image_plot_stretch[1])

        # Row 2: Frame navigation step editor and current frame label.
        info_bar = QHBoxLayout()
        bold_font = FONTS.large_bold
        big_font = FONTS.large
        step_label = QLabel("Frame Navigation Step:")
        step_label.setFont(bold_font)
        step_label.setStyleSheet(STYLE.white_label)
        self._step_edit: QLineEdit = QLineEdit(self)
        self._step_edit.setText(str(self._frame_delta))
        self._step_edit.setFixedWidth(STYLE.edit_width)
        self._step_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._step_edit.setFont(big_font)
        self._step_edit.setToolTip("Set the number of frames to skip per navigation step.")
        self._step_edit.setValidator(QtGui.QIntValidator(1, 10000))
        self._step_edit.returnPressed.connect(self._apply_step)
        self._step_edit.returnPressed.connect(self.setFocus)
        self._step_edit.installEventFilter(self)
        info_bar.addWidget(step_label)
        info_bar.addWidget(self._step_edit)
        info_bar.addSpacing(20)
        self._frame_number_label: QLabel = QLabel("Current frame: 0")
        self._frame_number_label.setStyleSheet(STYLE.white_label)
        info_bar.addWidget(self._frame_number_label)
        info_bar.addStretch()
        self._layout.addLayout(info_bar, 2, 0, 1, 6)

        # Row 3: Playback controls and frame slider.
        self._create_buttons()
        self._frame_slider: QSlider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self._frame_slider.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._frame_slider.setToolTip("Seek to a specific frame.")
        self._frame_slider.setTickInterval(BINARY_CONFIG.frame_slider_tick_interval)
        self._frame_slider.setTracking(False)
        self._frame_slider.valueChanged.connect(self._go_to_frame)
        self._layout.addWidget(self._frame_slider, 3, 4, 1, 2)

        self._update_frame_slider()
        self._update_buttons()
        self._update_timer: QtCore.QTimer = QtCore.QTimer()
        self._update_timer.timeout.connect(self._next_frame)
        self._graphics_widget.scene().sigMouseClicked.connect(self._plot_clicked)

        self.load_data(data=data)

    def load_data(self, data: SingleRecordingData) -> None:
        """Caches the input SingleRecordingData instance and uses it to populate the managed UI window.

        Args:
            data: The SingleRecordingData instance that stores the visualized recording's data.
        """
        self.data = data

        # Forces the combined view so the binary viewer always shows the stitched multi-plane movie.
        data.switch_view(view_index=-1)

        # Updates the window title to reflect the loaded recording path.
        self.setWindowTitle(f"Registered Recording — {data.recording_label}")

        # Configures all plot views and display parameters for the stitched display.
        self._setup_views()

    def get_state(self) -> dict[str, Any]:
        """Returns the current display state of the binary player for cross-process state exchange.

        Returns:
            A dictionary containing the current frame, playback status, and channel settings.
        """
        return {
            "current_frame": self._current_frame,
            "frame_count": self.data.frame_count,
            "channel_2_active": self._channel_2_visible,
            "two_channels": self.data.two_channels,
            "playing": self._update_timer.isActive(),
            "frame_step": self._frame_delta,
        }

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

        # Up/down arrow keys adjust the frame step size.
        if event.modifiers() != QtCore.Qt.KeyboardModifier.ShiftModifier:
            if event.key() == QtCore.Qt.Key.Key_Up:
                self._frame_delta = min(self._frame_delta + 1, 10000)
                self._step_edit.setText(str(self._frame_delta))
                self._frame_slider.setSingleStep(self._frame_delta)
            elif event.key() == QtCore.Qt.Key.Key_Down:
                self._frame_delta = max(self._frame_delta - 1, 1)
                self._step_edit.setText(str(self._frame_delta))
                self._frame_slider.setSingleStep(self._frame_delta)

        # Spacebar toggles between play and pause.
        if event.modifiers() != QtCore.Qt.KeyboardModifier.ShiftModifier and event.key() == QtCore.Qt.Key.Key_Space:
            if self._play_button.isEnabled():
                self._start_playback()
            else:
                self._pause_playback()

    def eventFilter(self, source: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        """Returns focus to the main window when Escape is pressed inside an edit field.

        Notes:
            Overrides the Qt virtual method. The camelCase name is required to match the parent signature.
        """
        if escape_returns_focus(self, event):
            return True
        return super().eventFilter(source, event)

    def _create_buttons(self) -> None:
        """Creates and lays out playback control buttons for the player window."""
        icon_size = QtCore.QSize(STYLE.icon_size, STYLE.icon_size)

        self._skip_backward_button: QToolButton = QToolButton()
        self._skip_backward_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward))
        self._skip_backward_button.setIconSize(icon_size)
        self._skip_backward_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._skip_backward_button.setToolTip("Step backward by the current frame delta.")
        self._skip_backward_button.clicked.connect(self._step_backward)

        playback = create_play_pause_group(
            self,
            play_tooltip="Start frame playback.",
            pause_tooltip="Stop frame playback.",
        )
        self._play_button = playback.play_button
        self._pause_button = playback.pause_button
        self._play_button.clicked.connect(self._start_playback)
        self._pause_button.clicked.connect(self._pause_playback)

        self._skip_forward_button: QToolButton = QToolButton()
        self._skip_forward_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward))
        self._skip_forward_button.setIconSize(icon_size)
        self._skip_forward_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._skip_forward_button.setToolTip("Step forward by the current frame delta.")
        self._skip_forward_button.clicked.connect(self._step_forward)

        # Places navigation buttons on row 3 alongside the frame slider.
        self._layout.addWidget(self._skip_backward_button, 3, 0, 1, 1)
        self._layout.addWidget(self._play_button, 3, 1, 1, 1)
        self._layout.addWidget(self._pause_button, 3, 2, 1, 1)
        self._layout.addWidget(self._skip_forward_button, 3, 3, 1, 1)

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

    def _apply_step(self) -> None:
        """Applies the frame step size from the step edit field to the frame delta and slider."""
        text = self._step_edit.text()
        if text:
            self._frame_delta = max(1, int(text))
            self._frame_slider.setSingleStep(self._frame_delta)

    def _load_recording(self) -> None:
        """Displays a file dialog that allows users to select a new recording to visualize."""
        # Defaults the file dialog to the parent of the currently loaded recording's output
        # directory, so the user can easily navigate to a sibling recording.
        start_directory = ""
        output = self.data.output_path
        if output is not None:
            parent = output.parent
            if parent.is_dir():
                start_directory = str(parent)

        directory = QFileDialog.getExistingDirectory(self, "Specify the recording directory to load.", start_directory)
        if not directory:
            return

        recording_path = Path(directory)
        try:
            data = SingleRecordingData.from_data(root_path=recording_path, view_index=-1)
        except Exception:
            console.echo(message="Unable to load recording data.", level=LogLevel.ERROR)
            result = QMessageBox.question(
                self,
                "ERROR",
                "Unable to load recording. Try another directory?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if result == QMessageBox.StandardButton.Yes:
                self._load_recording()
            return

        self.load_data(data=data)
        self.recording_changed.emit()

    def _setup_views(self) -> None:
        """Configures all plot views and display parameters for the stitched multi-plane display."""
        self._offset_plot.clear()
        self._offset_plot.disableAutoRange()

        # Computes dynamic range from subsampled per-plane frames, avoiding zero-border bias from stitched frames.
        # Ravels each plane's samples before concatenation since planes may have different FOV dimensions.
        all_frames = np.concatenate(
            [
                binary.subsample_movie(sample_count=BINARY_CONFIG.subsample_frame_count).ravel()
                for binary in self.data.combined_binary.files
            ]
        )
        frame_mean = np.float32(all_frames.mean())
        frame_std = np.float32(all_frames.std())
        self._display_range = frame_mean + frame_std * np.array(
            [-BINARY_CONFIG.display_range_low_sigma, BINARY_CONFIG.display_range_high_sigma], dtype=np.float32
        )

        # Loads aspect ratio from recording data.
        self._main_view_box.setAspectLocked(lock=True, ratio=self.data.aspect_ratio)

        # Configures the frame display slider.
        frame_count = self.data.frame_count
        last_frame = frame_count - 1
        self._time_step = 1.0 / self.data.sampling_rate * 1000 / BINARY_CONFIG.playback_speed_multiplier
        self._frame_delta = min(BINARY_CONFIG.default_frame_delta, max(1, last_frame))
        self._step_edit.setText(str(self._frame_delta))
        self._frame_slider.setSingleStep(self._frame_delta)
        if frame_count > 0:
            self._update_frame_slider()
            self._update_buttons()

        # Computes average rigid registration offsets across all planes.
        plane_count = self.data.plane_count
        x_values = np.arange(frame_count, dtype=np.int32)
        average_y = np.zeros(frame_count, dtype=np.float32)
        average_x = np.zeros(frame_count, dtype=np.float32)
        for plane_index in range(plane_count):
            rigid_y, rigid_x = self.data.plane_rigid_offsets(plane_index)
            average_y += rigid_y
            average_x += rigid_x
        average_y = np.asarray(average_y / plane_count, dtype=np.float32)
        average_x = np.asarray(average_x / plane_count, dtype=np.float32)
        self._average_rigid_y_offsets = average_y
        self._average_rigid_x_offsets = average_x

        # Plots average Y (gold) and X (green) offset curves with a horizontal legend.
        add_plot_legend(self._offset_plot, column_count=BINARY_STYLE.legend_column_count)
        self._offset_plot.plot(x_values, average_y, pen=pg.mkPen(COLORS.gold), name="Average Y offset")
        self._offset_plot.plot(x_values, average_x, pen=pg.mkPen(COLORS.green), name="Average X offset")
        shift_min = min(float(average_y.min()), float(average_x.min()))
        shift_max = max(float(average_y.max()), float(average_x.max()))
        if shift_min == shift_max:
            shift_min -= 1
            shift_max += 1
        shift_max += (shift_max - shift_min) * PLOT_STYLE.legend_headroom
        self._offset_plot.setLimits(xMin=0, xMax=last_frame)
        self._offset_plot.setRange(xRange=(0, last_frame), yRange=(shift_min, shift_max), padding=0.0)
        self._offset_scatter = pg.ScatterPlotItem()
        self._offset_plot.addItem(self._offset_scatter)
        self._update_offset_scatter(0)

        self._channel_2_button.setVisible(self.data.two_channels)

        self._current_frame = 0
        self._render_frame()

    def _update_offset_scatter(self, frame_index: int) -> None:
        """Updates the scatter overlay on the offset plot to mark the current frame on the average offset curves.

        Args:
            frame_index: The frame index to highlight on the offset plot.
        """
        self._offset_scatter.setData(
            [frame_index, frame_index],
            [float(self._average_rigid_y_offsets[frame_index]), float(self._average_rigid_x_offsets[frame_index])],
            size=PLOT_STYLE.scatter_point_size,
            brush=pg.mkBrush(*COLORS.red),
        )

    def _next_frame(self) -> None:
        """Advances to the next frame by the current step size and updates all display elements."""
        self._current_frame += self._frame_delta
        last_frame = self.data.frame_count - 1
        if self._current_frame > last_frame:
            self._current_frame = 0
        self._render_frame()

    def _render_frame(self) -> None:
        """Reads and displays the frame at ``_current_frame`` and updates all navigation controls."""
        # Reads the current stitched frame combining all planes.
        self._image = np.asarray(self.data.read_stitched_frame(self._current_frame))

        # If channel 2 overlay is active, composites both channels into an RGB image with channel 1 in
        # the red plane and channel 2 in the green plane.
        if self.data.two_channels and self._channel_2_visible:
            channel_2_frame = np.asarray(self.data.read_stitched_frame_channel_2(self._current_frame))[:, :, np.newaxis]
            self._image = np.concatenate(
                (self._image[:, :, np.newaxis], channel_2_frame, np.zeros_like(channel_2_frame)),
                axis=-1,
            )

        # Updates the main image display and frame navigation controls.
        self._main_image.setImage(self._image, levels=self._display_range)
        self._frame_slider.setValue(self._current_frame)
        self._frame_number_label.setText(f"Current frame: {self._current_frame}")

        # Moves the red dot indicators on the rigid registration offset plot to the current frame.
        self._update_offset_scatter(self._current_frame)

    def _go_to_frame(self) -> None:
        """Seeks to the frame indicated by the frame slider position.

        Notes:
            Serves as a callback target for the position update signal.
        """
        self._current_frame = int(self._frame_slider.value())
        self._render_frame()

    def _step_backward(self) -> None:
        """Steps backward by the current frame delta."""
        self._current_frame = max(0, self._current_frame - self._frame_delta)
        self._frame_slider.setValue(self._current_frame)

    def _step_forward(self) -> None:
        """Steps forward by the current frame delta."""
        self._current_frame = min(self.data.frame_count - 1, self._current_frame + self._frame_delta)
        self._frame_slider.setValue(self._current_frame)

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
        self._channel_2_button.setStyleSheet(
            STYLE.button_pressed if self._channel_2_visible else STYLE.button_unpressed
        )
        self._next_frame()

    def _zoom_image(self) -> None:
        """Resets the main view zoom to fit the full image extent."""
        self._main_view_box.setRange(yRange=(0, self.data.frame_height), xRange=(0, self.data.frame_width))

    def _plot_clicked(self, event: object) -> None:
        """Handles mouse click events on plots for frame navigation."""
        # Resolves which graphics items lie under the click position.
        items = self._graphics_widget.scene().items(event.scenePos())  # type: ignore[attr-defined]
        position_x = 0
        is_time_plot = False
        zoom = False
        seek_to_frame = False
        for item in items:
            # Click landed on a time-series plot: maps the scene position to a frame index.
            if item == self._offset_plot:
                view_box = self._offset_plot.vb
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
            self._offset_plot.setRange(xRange=(0, frame_count - 1))

        # Seeks to the clicked frame when playback is paused.
        if seek_to_frame and self._play_button.isEnabled():
            self._current_frame = max(0, min(frame_count - 1, round(position_x)))
            self._frame_slider.setValue(self._current_frame)
