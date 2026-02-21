"""Provides registration binary viewer and principal component metrics viewer windows."""

from pathlib import Path

import numpy as np
from PySide6 import QtGui, QtCore
from tifffile import imread
import pyqtgraph as pg
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
    """Provides a playback window for viewing registered binary imaging data."""

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
        self.cwidget = QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.l0 = QGridLayout()
        self.cwidget.setLayout(self.l0)
        self.win = pg.GraphicsLayoutWidget()
        self.win.move(600, 0)
        self.win.resize(1000, 500)
        self.l0.addWidget(self.win, 1, 2, 13, 14)
        self.loaded = False
        self.zloaded = False
        self.zcorr = None
        self._data: RegistrationViewerData | None = None

        # Main image view.
        self.vmain = pg.ViewBox(lockAspect=True, invertY=True, name="plot1")
        self.win.addItem(self.vmain, row=0, col=0)
        self.vmain.setMenuEnabled(False)
        self.imain = pg.ImageItem()
        self.vmain.addItem(self.imain)

        # Side box for z-stack display.
        self.vside = pg.ViewBox(lockAspect=True, invertY=True)
        self.vside.setMenuEnabled(False)
        self.iside = pg.ImageItem()
        self.vside.addItem(self.iside)

        # Channel 2 checkbox.
        self.channel_2_checkbox = QCheckBox("view channel 2")
        self.channel_2_checkbox.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.channel_2_checkbox.setEnabled(False)
        self.channel_2_checkbox.toggled.connect(self.toggle_channel_2)
        self.l0.addWidget(self.channel_2_checkbox, 0, 5, 1, 1)

        # Z-stack checkbox.
        self.zbox = QCheckBox("view z-stack")
        self.zbox.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.zbox.setEnabled(False)
        self.zbox.toggled.connect(self.add_zstack)
        self.l0.addWidget(self.zbox, 0, 8, 1, 1)

        z_label = QLabel("Z-plane:")
        z_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.l0.addWidget(z_label, 0, 9, 1, 1)

        self.z_edit = QLineEdit(self)
        self.z_edit.setValidator(QtGui.QIntValidator(0, 0))
        self.z_edit.setText("0")
        self.z_edit.setFixedWidth(_Z_EDIT_WIDTH)
        self.z_edit.setAlignment(QtCore.Qt.AlignRight)
        self.l0.addWidget(self.z_edit, 0, 10, 1, 1)

        # Rigid registration offset plot.
        self.p1 = self.win.addPlot(name="plot_shift", row=1, col=0, colspan=2)
        self.p1.setMouseEnabled(x=True, y=False)
        self.p1.setMenuEnabled(False)
        self.scatter1 = pg.ScatterPlotItem()
        self.scatter1.setData([0, 0], [0, 0])
        self.p1.addItem(self.scatter1)

        # Nonrigid RMS displacement plot.
        self.p2 = self.win.addPlot(name="plot_nonrigid", row=2, col=0, colspan=2)
        self.p2.setMouseEnabled(x=True, y=False)
        self.p2.setMenuEnabled(False)
        self.scatter2 = pg.ScatterPlotItem()
        self.p2.setXLink("plot_shift")
        self.has_nonrigid = False

        # Z-position correlation plot.
        self.p3 = self.win.addPlot(name="plot_Z", row=3, col=0, colspan=2)
        self.p3.setMouseEnabled(x=True, y=False)
        self.p3.setMenuEnabled(False)
        self.scatter3 = pg.ScatterPlotItem()
        self.p3.setXLink("plot_shift")

        self.win.ci.layout.setRowStretchFactor(0, 12)
        self.movie_label = QLabel("No session loaded")
        self.movie_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.movie_label.setAlignment(QtCore.Qt.AlignCenter)
        self.nframes = 0
        self.cframe = 0
        self._create_buttons()

        # Plane selector dropdown.
        plane_label = QLabel("Plane:")
        plane_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.l0.addWidget(plane_label, 6, 0, 1, 1)
        self.plane_selector = QComboBox(self)
        self.plane_selector.setEnabled(False)
        self.plane_selector.currentIndexChanged.connect(self._on_plane_changed)
        self.l0.addWidget(self.plane_selector, 6, 1, 1, 1)

        # Frame slider.
        self.frame_label = QLabel("Current frame:")
        self.frame_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.frame_number = QLabel("0")
        self.frame_number.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.frame_slider = QSlider(QtCore.Qt.Horizontal)
        self.frame_slider.setTickInterval(5)
        self.frame_slider.setTracking(False)
        self.frame_delta = 10
        self.l0.addWidget(QLabel(""), 12, 0, 1, 1)
        self.l0.setRowStretch(12, 1)
        self.l0.addWidget(self.frame_label, 13, 0, 1, 2)
        self.l0.addWidget(self.frame_number, 14, 0, 1, 2)
        self.l0.addWidget(self.frame_slider, 13, 2, 14, 13)
        self.l0.addWidget(QLabel(""), 14, 1, 1, 1)
        hint_label = QLabel("(when paused, left/right arrow keys can move slider)")
        hint_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.l0.addWidget(hint_label, 16, 0, 1, 3)
        self.frame_slider.valueChanged.connect(self.go_to_frame)
        self.l0.addWidget(self.movie_label, 0, 0, 1, 5)
        self._update_frame_slider()
        self._update_buttons()
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.next_frame)
        self.cframe = 0
        self.loaded = False
        self.channel_2_visible = False
        self.z_on = False
        self.has_channel_2_binary = False
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)

        if data is not None:
            self.load_data(data=data)

    def load_data(self, data: RegistrationViewerData) -> None:
        """Stores the data model, populates the plane selector, and opens the first plane.

        Args:
            data: The registration viewer data model wrapping all planes.
        """
        self._data = data
        # Populates the plane selector without triggering _on_plane_changed yet.
        self.plane_selector.blockSignals(True)
        self.plane_selector.clear()
        for label in data.plane_labels:
            self.plane_selector.addItem(label)
        self.plane_selector.setCurrentIndex(data.current_plane_index)
        self.plane_selector.blockSignals(False)
        self.plane_selector.setEnabled(data.plane_count > 1)
        self._open_plane()

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

        self.LY = self._data.frame_height
        self.LX = self._data.frame_width
        self.Ly = [self._data.frame_height]
        self.Lx = [self._data.frame_width]
        self.dy = [0]
        self.dx = [0]

        self.reg_loc = [str(registered_path)]
        self.reg_file = [registered_path.open("rb")]
        self.has_channel_2_binary = False

        channel_2_path = self._data.registered_binary_path_channel_2
        if channel_2_path is not None and channel_2_path.is_file():
            self.reg_file_chan2 = channel_2_path.open("rb")
            self.has_channel_2_binary = True

        self._setup_views()

    def _close_binary_files(self) -> None:
        """Closes all open binary file handles."""
        if hasattr(self, "reg_file"):
            for handle in self.reg_file:
                handle.close()
        if self.has_channel_2_binary and hasattr(self, "reg_file_chan2"):
            self.reg_file_chan2.close()
        self.has_channel_2_binary = False

    def _load_session(self) -> None:
        """Opens a file dialog to select a session directory and loads it."""
        directory = QFileDialog.getExistingDirectory(self, "Open session directory")
        if directory:
            try:
                data = RegistrationViewerData.from_session(root_path=Path(directory))
                self.load_data(data=data)
            except Exception as error:
                console.echo(message=f"Failed to load session: {error}", level=LogLevel.ERROR)

    def toggle_channel_2(self) -> None:
        """Toggles channel 2 display based on checkbox state."""
        if self.loaded:
            self.channel_2_visible = self.channel_2_checkbox.isChecked()
            self.next_frame()

    def zoom_image(self) -> None:
        """Resets the main and side view zoom to fit the full image extent."""
        self.vmain.setRange(yRange=(0, self.LY), xRange=(0, self.LX))
        if self.z_on:
            self.vside.setRange(yRange=(0, self.zLy), xRange=(0, self.zLx))
            self.vside.setXLink("plot1")
            self.vside.setYLink("plot1")

    def add_zstack(self) -> None:
        """Toggles z-stack side view display based on checkbox state."""
        if self.loaded:
            if self.zbox.isChecked():
                self.z_on = True
                self.win.addItem(self.vside, row=0, col=1)
            else:
                self.z_on = False
                self.win.removeItem(self.vside)
            self.next_frame()

    def next_frame(self) -> None:
        """Advances to the next frame and updates all display elements."""
        self.cframe += 1
        if self.cframe > self.nframes - 1:
            self.cframe = 0
            for handle in self.reg_file:
                handle.seek(0, 0)
            if self.has_channel_2_binary:
                self.reg_file_chan2.seek(0, 0)
        self.img = np.zeros((self.LY, self.LX), dtype=np.int16)
        for index in range(len(self.reg_loc)):
            buff = self.reg_file[index].read(self.nbytesread[index])
            frame = np.reshape(
                np.frombuffer(buff, dtype=np.int16, offset=0),
                (self.Ly[index], self.Lx[index]),
            )
            self.img[
                self.dy[index] : self.dy[index] + self.Ly[index],
                self.dx[index] : self.dx[index] + self.Lx[index],
            ] = frame

        if self.has_channel_2_binary and self.channel_2_visible:
            buff = self.reg_file_chan2.read(self.nbytesread[0])
            channel_2_frame = np.reshape(
                np.frombuffer(buff, dtype=np.int16, offset=0),
                (self.Ly[0], self.Lx[0]),
            )[:, :, np.newaxis]
            self.img = np.concatenate(
                (self.img[:, :, np.newaxis], channel_2_frame, np.zeros_like(channel_2_frame)),
                axis=-1,
            )
        if self.zloaded and self.z_on:
            if hasattr(self, "zmax"):
                self.z_edit.setText(str(self.zmax[self.cframe]))
            self.iside.setImage(self.zstack[int(self.z_edit.text())], levels=self.zrange)

        self.imain.setImage(self.img, levels=self.srange)
        self.frame_slider.setValue(self.cframe)
        self.frame_number.setText(str(self.cframe))
        self.scatter1.setData(
            [self.cframe, self.cframe],
            [self.yoff[self.cframe], self.xoff[self.cframe]],
            size=_SCATTER_POINT_SIZE,
            brush=pg.mkBrush(255, 0, 0),
        )
        if self.has_nonrigid:
            self.scatter2.setData(
                [self.cframe],
                [self.nonrigid_rms[self.cframe]],
                size=_SCATTER_POINT_SIZE,
                brush=pg.mkBrush(255, 0, 0),
            )
        if self.zloaded and self.z_on:
            self.scatter3.setData(
                [self.cframe, self.cframe],
                [self.zmax[self.cframe], self.zmax[self.cframe]],
                size=_SCATTER_POINT_SIZE,
                brush=pg.mkBrush(255, 0, 0),
            )

    def _setup_views(self) -> None:
        """Configures all plot views and display parameters after loading data."""
        if self._data is None:
            return

        self.p1.clear()

        # Computes dynamic range from subsampled frames.
        sample_count = min(self._data.frame_count - 1, _SUBSAMPLE_FRAME_COUNT)
        frames = _subsample_frames(
            frame_count=self._data.frame_count,
            frame_height=self._data.frame_height,
            frame_width=self._data.frame_width,
            sample_count=sample_count,
            registration_path=str(self._data.registered_binary_path),
        )
        self.srange = frames.mean() + frames.std() * np.array([-_DISPLAY_RANGE_LOW_SIGMA, _DISPLAY_RANGE_HIGH_SIGMA])

        self.movie_label.setText(self.reg_loc[-1])
        self.nbytesread = [2 * self.Ly[index] * self.Lx[index] for index in range(len(self.reg_loc))]

        # Aspect ratio from data model.
        self.xyrat = self._data.aspect_ratio
        self.vmain.setAspectLocked(lock=True, ratio=self.xyrat)
        self.vside.setAspectLocked(lock=True, ratio=self.xyrat)

        self.nframes = self._data.frame_count
        self.time_step = 1.0 / self._data.sampling_rate * 1000 / _PLAYBACK_SPEED_MULTIPLIER
        self.frame_delta = int(np.maximum(_MIN_FRAME_DELTA, self.nframes / _FRAME_DELTA_DIVISOR))
        self.frame_slider.setSingleStep(self.frame_delta)
        if self.nframes > 0:
            self._update_frame_slider()
            self._update_buttons()

        # Plots registration X-Y offsets.
        rigid_y = self._data.rigid_y_offsets
        rigid_x = self._data.rigid_x_offsets
        if rigid_y is not None and rigid_x is not None:
            self.yoff = rigid_y
            self.xoff = rigid_x
        else:
            self.yoff = np.zeros((self.nframes,), dtype=np.int32)
            self.xoff = np.zeros((self.nframes,), dtype=np.int32)
        self.p1.plot(self.yoff, pen="g")
        self.p1.plot(self.xoff, pen="y")
        self.p1.setRange(
            xRange=(0, self.nframes),
            yRange=(np.minimum(self.yoff.min(), self.xoff.min()), np.maximum(self.yoff.max(), self.xoff.max())),
            padding=0.0,
        )
        self.p1.setLimits(xMin=0, xMax=self.nframes)
        self.scatter1 = pg.ScatterPlotItem()
        self.p1.addItem(self.scatter1)
        self.scatter1.setData(
            [self.cframe, self.cframe],
            [self.yoff[self.cframe], self.xoff[self.cframe]],
            size=_SCATTER_POINT_SIZE,
            brush=pg.mkBrush(255, 0, 0),
        )

        # Plots per-frame nonrigid RMS displacement if available.
        self.p2.clear()
        nonrigid_y = self._data.nonrigid_y_offsets
        nonrigid_x = self._data.nonrigid_x_offsets
        if nonrigid_y is not None and nonrigid_x is not None:
            self.nonrigid_rms = np.sqrt(
                np.mean(nonrigid_y.astype(np.float32) ** 2 + nonrigid_x.astype(np.float32) ** 2, axis=1)
            )
            self.has_nonrigid = True
            self.p2.plot(self.nonrigid_rms, pen=(180, 100, 255))
            self.p2.setRange(
                xRange=(0, self.nframes),
                yRange=(0, self.nonrigid_rms.max()),
                padding=0.0,
            )
            self.p2.setLimits(xMin=0, xMax=self.nframes)
            self.p2.setLabel("left", "nonrigid RMS", units="px")
            self.scatter2 = pg.ScatterPlotItem()
            self.p2.addItem(self.scatter2)
        else:
            self.has_nonrigid = False

        self.channel_2_checkbox.setEnabled(self.has_channel_2_binary)

        self.cframe = -1
        self.loaded = True
        self.next_frame()

    def plot_clicked(self, event: object) -> None:
        """Handles mouse click events on plots for frame navigation."""
        items = self.win.scene().items(event.scenePos())
        posx = 0
        is_time_plot = False
        zoom = False
        choose = False
        if self.loaded:
            for item in items:
                if item in (self.p1, self.p2):
                    view_box = self.p1.vb
                    position = view_box.mapSceneToView(event.scenePos())
                    posx = position.x()
                    is_time_plot = True
                elif item in (self.vmain, self.vside):
                    if event.button() == 1 and event.double():
                        self.zoom_image()
                if is_time_plot and event.button() == 1:
                    if event.double():
                        zoom = True
                    else:
                        choose = True
        if zoom:
            self.p1.setRange(xRange=(0, self.nframes))
            self.p2.setRange(xRange=(0, self.nframes))
            self.p3.setRange(xRange=(0, self.nframes))

        if choose and self.playButton.isEnabled():
            self.cframe = np.maximum(0, np.minimum(self.nframes - 1, int(np.round(posx))))
            self.frame_slider.setValue(self.cframe)

    def load_zstack(self) -> None:
        """Opens a file dialog to load a z-stack TIFF and initializes z-position tracking."""
        name = QFileDialog.getOpenFileName(self, "Open zstack", filter="*.tif")
        zstack_path = name[0]
        try:
            self.zstack = imread(zstack_path)
            self.zLy, self.zLx = self.zstack.shape[1:]
            self.z_edit.setValidator(QtGui.QIntValidator(0, self.zstack.shape[0]))
            self.zrange = [
                np.percentile(self.zstack, _Z_PERCENTILE_LOW),
                np.percentile(self.zstack, _Z_PERCENTILE_HIGH),
            ]

            self.compute_z_button.setEnabled(True)
            self.zloaded = True
            self.zbox.setEnabled(True)
            self.zbox.setChecked(True)
            self.zmax = np.zeros(self.nframes, dtype=int)

            # Checks for cached zcorr data in order of priority:
            # 1. Local instance cache (self.zcorr)
            # 2. Separate zcorr.npy file from the output directory
            if self.zcorr is not None and self.zstack.shape[0] == self.zcorr.shape[0]:
                self.zmax = np.argmax(
                    gaussian_filter1d(self.zcorr.T.copy(), _Z_SMOOTHING_SIGMA, axis=1),
                    axis=1,
                )
                self.plot_zcorr()
            elif self._data is not None and self._data.output_directory is not None:
                zcorr_path = self._data.output_directory / "zcorr.npy"
                if zcorr_path.exists():
                    self.zcorr = np.load(zcorr_path)
                    if self.zstack.shape[0] == self.zcorr.shape[0]:
                        self.zmax = np.argmax(
                            gaussian_filter1d(self.zcorr.T.copy(), _Z_SMOOTHING_SIGMA, axis=1),
                            axis=1,
                        )
                        self.plot_zcorr()

        except Exception as error:
            console.echo(message=f"ERROR: {error}", level=LogLevel.ERROR)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        """Handles keyboard navigation for frame stepping and playback control."""
        if self.playButton.isEnabled() and event.modifiers() != QtCore.Qt.ShiftModifier:
            if event.key() == QtCore.Qt.Key_Left:
                self.cframe -= self.frame_delta
                self.cframe = np.maximum(0, np.minimum(self.nframes - 1, self.cframe))
                self.frame_slider.setValue(self.cframe)
            elif event.key() == QtCore.Qt.Key_Right:
                self.cframe += self.frame_delta
                self.cframe = np.maximum(0, np.minimum(self.nframes - 1, self.cframe))
                self.frame_slider.setValue(self.cframe)
        if event.modifiers() != QtCore.Qt.ShiftModifier and event.key() == QtCore.Qt.Key_Space:
            if self.playButton.isEnabled():
                self.start()
            else:
                self.pause()

    def go_to_frame(self) -> None:
        """Seeks to the frame indicated by the frame slider position."""
        self.cframe = int(self.frame_slider.value())
        self.jump_to_frame()

    def jump_to_frame(self) -> None:
        """Seeks all binary file handles to an absolute frame position and displays it."""
        if self.playButton.isEnabled():
            self.cframe = np.maximum(0, np.minimum(self.nframes - 1, self.cframe))
            self.cframe = int(self.cframe)
            for index in range(len(self.reg_file)):
                self.reg_file[index].seek(self.nbytesread[index] * self.cframe, 0)
            if self.has_channel_2_binary:
                self.reg_file_chan2.seek(self.nbytesread[-1] * self.cframe, 0)
            self.cframe -= 1
            self.next_frame()

    def start(self) -> None:
        """Starts video playback by enabling the frame update timer."""
        if self.cframe < self.nframes - 1:
            console.echo(message="Playing video...")
            self.playButton.setEnabled(False)
            self.pauseButton.setEnabled(True)
            self.frame_slider.setEnabled(False)
            self.update_timer.start(self.time_step)

    def pause(self) -> None:
        """Pauses video playback and re-enables manual frame navigation."""
        self.update_timer.stop()
        self.playButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        self.frame_slider.setEnabled(True)
        console.echo(message="Video paused")

    def _compute_z(self) -> None:
        """Computes z-position correlations for the current session.

        Currently disabled pending refactor to use RuntimeContext instead of legacy ops dictionary.
        """
        console.echo(
            message="Z-position computation is temporarily disabled. The GUI needs to be refactored to use "
            "RuntimeContext instead of the legacy ops dictionary.",
            level="WARNING",
        )

    def plot_zcorr(self) -> None:
        """Plots the z-position correlation trace on the z-position plot."""
        self.p3.clear()
        self.p3.plot(self.zmax, pen="r")
        self.p3.addItem(self.scatter3)
        self.p3.setRange(xRange=(0, self.nframes), yRange=(self.zmax.min(), self.zmax.max() + 3), padding=0.0)
        self.p3.setLimits(xMin=0, xMax=self.nframes)
        self.p3.setXLink("plot_shift")

    def _update_frame_slider(self) -> None:
        """Configures the frame slider range and enables it."""
        self.frame_slider.setMaximum(self.nframes - 1)
        self.frame_slider.setMinimum(0)
        self.frame_label.setEnabled(True)
        self.frame_slider.setEnabled(True)

    def _update_buttons(self) -> None:
        """Sets the initial enabled state for play and pause buttons."""
        self.playButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        self.pauseButton.setChecked(True)

    def _create_buttons(self) -> None:
        """Creates and lays out all control buttons for the player window."""
        icon_size = QtCore.QSize(_ICON_SIZE, _ICON_SIZE)
        load_session_button = QPushButton("Load Session")
        load_session_button.setToolTip("Open a suite2p session directory")
        load_session_button.clicked.connect(self._load_session)

        load_z_button = QPushButton("load z-stack tiff")
        load_z_button.clicked.connect(self.load_zstack)

        self.compute_z_button = QPushButton("compute z position")
        self.compute_z_button.setEnabled(False)
        self.compute_z_button.clicked.connect(self._compute_z)

        self.playButton = QToolButton()
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.setIconSize(icon_size)
        self.playButton.setToolTip("Play")
        self.playButton.setCheckable(True)
        self.playButton.clicked.connect(self.start)

        self.pauseButton = QToolButton()
        self.pauseButton.setCheckable(True)
        self.pauseButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.pauseButton.setIconSize(icon_size)
        self.pauseButton.setToolTip("Pause")
        self.pauseButton.clicked.connect(self.pause)

        btns = QButtonGroup(self)
        btns.addButton(self.playButton, 0)
        btns.addButton(self.pauseButton, 1)
        btns.setExclusive(True)

        quit_button = QToolButton()
        quit_button.setIcon(self.style().standardIcon(QStyle.SP_DialogCloseButton))
        quit_button.setIconSize(icon_size)
        quit_button.setToolTip("Quit")
        quit_button.clicked.connect(self.close)

        self.l0.addWidget(load_session_button, 1, 0, 1, 2)
        self.l0.addWidget(load_z_button, 2, 0, 1, 2)
        self.l0.addWidget(self.compute_z_button, 3, 0, 1, 2)
        self.l0.addWidget(self.playButton, 15, 0, 1, 1)
        self.l0.addWidget(self.pauseButton, 15, 1, 1, 1)
        self.playButton.setEnabled(False)
        self.pauseButton.setEnabled(False)
        self.pauseButton.setChecked(True)


def _subsample_frames(
    frame_count: int,
    frame_height: int,
    frame_width: int,
    sample_count: int,
    registration_path: str,
) -> np.ndarray:
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
    start_indices = np.linspace(0, frame_count, 1 + sample_count).astype(np.int64)
    binary_file = Path(registration_path).open("rb")
    for index in range(sample_count):
        binary_file.seek(bytes_per_frame * start_indices[index], 0)
        buffer = binary_file.read(bytes_per_frame)
        data = np.frombuffer(buffer, dtype=np.int16, offset=0)
        frames[index, :, :] = np.reshape(data, (frame_height, frame_width))
    binary_file.close()
    return frames


class PCViewer(QMainWindow):
    """Provides a viewer window for principal component registration metrics."""

    def __init__(self, data: RegistrationViewerData | None = None) -> None:
        """Initializes the PC viewer window and all UI components.

        Args:
            data: Pre-loaded registration data to display on startup.
        """
        super().__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(70, 70, 1300, 800)
        self.setWindowTitle("Metrics for registration")
        self.cwidget = QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.l0 = QGridLayout()
        self.cwidget.setLayout(self.l0)

        self.win = pg.GraphicsLayoutWidget()
        self.l0.addWidget(self.win, 0, 2, 13, 14)

        # Pixel shift metrics plot.
        self.p3 = self.win.addPlot(row=0, col=0)
        self.p3.setMouseEnabled(x=False, y=False)
        self.p3.setMenuEnabled(False)

        self.p0 = self.win.addViewBox(name="plot1", lockAspect=True, row=1, col=0, invertY=True)
        self.p1 = self.win.addViewBox(lockAspect=True, row=1, col=1, invertY=True)
        self.p1.setMenuEnabled(False)
        self.p1.setXLink("plot1")
        self.p1.setYLink("plot1")
        self.p2 = self.win.addViewBox(lockAspect=True, row=1, col=2, invertY=True)
        self.p2.setMenuEnabled(False)
        self.p2.setXLink("plot1")
        self.p2.setYLink("plot1")
        self.img0 = pg.ImageItem()
        self.img1 = pg.ImageItem()
        self.img2 = pg.ImageItem()
        self.p0.addItem(self.img0)
        self.p1.addItem(self.img1)
        self.p2.addItem(self.img2)
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)

        self.p4 = self.win.addPlot(row=0, col=1, colspan=2)
        self.p4.setMouseEnabled(x=False)
        self.p4.setMenuEnabled(False)

        self.pc_edit = QLineEdit(self)
        self.pc_edit.setText("1")
        self.pc_edit.setFixedWidth(_PC_EDIT_WIDTH)
        self.pc_edit.setAlignment(QtCore.Qt.AlignRight)
        self.pc_edit.returnPressed.connect(self.plot_frame)
        self.pc_edit.textEdited.connect(self.pause)
        pc_label = QLabel("PC: ")
        bold_font = QtGui.QFont(FONT_FAMILY, pointSize=_METRICS_FONT_SIZE, weight=QtGui.QFont.Weight.Bold)
        big_font = QtGui.QFont(FONT_FAMILY, pointSize=_METRICS_FONT_SIZE)
        pc_label.setFont(bold_font)
        self.pc_edit.setFont(big_font)
        pc_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.l0.addWidget(QLabel(""), 1, 0, 1, 1)
        self.l0.addWidget(pc_label, 2, 0, 1, 1)
        self.l0.addWidget(self.pc_edit, 2, 1, 1, 1)
        self.nums = []
        self.titles = []
        for j in range(3):
            num_label = QLabel("")
            num_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
            self.l0.addWidget(num_label, 3 + j, 0, 1, 2)
            self.nums.append(num_label)
            title_label = QLabel("")
            title_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
            self.l0.addWidget(title_label, 12, 4 + j * 4, 1, 2)
            self.titles.append(title_label)
        self.loaded = False
        self._data: RegistrationViewerData | None = None
        self.l0.addWidget(QLabel(""), 7, 0, 1, 1)
        self.l0.setRowStretch(7, 1)
        self.cframe = 0
        self._create_buttons()
        self.nPCs = _DEFAULT_PC_COUNT
        self.pc_edit.setValidator(QtGui.QIntValidator(1, self.nPCs))
        self.update_timer = QtCore.QTimer()
        self.update_timer.timeout.connect(self.next_frame)

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

        self.PC = np.clip(pc_images, np.percentile(pc_images, 1), np.percentile(pc_images, 99))
        self.Ly, self.Lx = self.PC.shape[2:]
        self.DX = pc_metrics
        if pc_projections is not None:
            self.tPC = pc_projections
        else:
            self.tPC = np.zeros((1, self.PC.shape[1]), dtype=np.float32)

        self.loaded = True
        self.nPCs = self.PC.shape[1]
        self.pc_edit.setValidator(QtGui.QIntValidator(1, self.nPCs))
        self.plot_frame()
        self.playButton.setEnabled(True)

    def open(self) -> None:
        """Opens a file dialog to select a session directory and loads PC data."""
        directory = QFileDialog.getExistingDirectory(self, "Open session directory")
        if directory:
            try:
                data = RegistrationViewerData.from_session(root_path=Path(directory))
                self.load_data(data=data)
            except Exception as error:
                console.echo(message=f"Failed to load session: {error}", level=LogLevel.ERROR)

    def _create_buttons(self) -> None:
        """Creates and lays out the open, play, and pause buttons."""
        icon_size = QtCore.QSize(_ICON_SIZE, _ICON_SIZE)
        open_button = QToolButton()
        open_button.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        open_button.setIconSize(icon_size)
        open_button.setToolTip("Open session directory")
        open_button.clicked.connect(self.open)

        self.playButton = QToolButton()
        self.playButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.playButton.setIconSize(icon_size)
        self.playButton.setToolTip("Play")
        self.playButton.setCheckable(True)
        self.playButton.clicked.connect(self.start)

        self.pauseButton = QToolButton()
        self.pauseButton.setCheckable(True)
        self.pauseButton.setIcon(self.style().standardIcon(QStyle.SP_MediaPause))
        self.pauseButton.setIconSize(icon_size)
        self.pauseButton.setToolTip("Pause")
        self.pauseButton.clicked.connect(self.pause)

        btns = QButtonGroup(self)
        btns.addButton(self.playButton, 0)
        btns.addButton(self.pauseButton, 1)
        btns.setExclusive(True)

        self.l0.addWidget(open_button, 0, 0, 1, 1)
        self.l0.addWidget(self.playButton, 14, 12, 1, 1)
        self.l0.addWidget(self.pauseButton, 14, 13, 1, 1)
        self.playButton.setEnabled(False)
        self.pauseButton.setEnabled(False)
        self.pauseButton.setChecked(True)

    def start(self) -> None:
        """Starts PC animation playback."""
        if self.loaded:
            self.playButton.setEnabled(False)
            self.pauseButton.setEnabled(True)
            self.update_timer.start(_PC_ANIMATION_INTERVAL_MS)

    def pause(self) -> None:
        """Pauses PC animation playback."""
        self.update_timer.stop()
        self.playButton.setEnabled(True)
        self.pauseButton.setChecked(True)
        self.pauseButton.setEnabled(False)

    def next_frame(self) -> None:
        """Advances the PC animation to the next frame, toggling between top and bottom halves."""
        pc_index = int(self.pc_edit.text()) - 1
        pc1 = self.PC[1, pc_index, :, :]
        pc0 = self.PC[0, pc_index, :, :]
        if self.cframe == 0:
            self.img2.setImage(np.tile(pc0[:, :, np.newaxis], (1, 1, 3)))
            self.titles[2].setText("top")
        else:
            self.img2.setImage(np.tile(pc1[:, :, np.newaxis], (1, 1, 3)))
            self.titles[2].setText("bottom")

        self.img2.setLevels([pc0.min(), pc0.max()])
        self.cframe = 1 - self.cframe

    def plot_frame(self) -> None:
        """Renders all PC visualizations for the currently selected principal component."""
        if self.loaded:
            self.titles[0].setText("difference")
            self.titles[1].setText("merged")
            self.titles[2].setText("top")
            pc_index = int(self.pc_edit.text()) - 1
            pc1 = self.PC[1, pc_index, :, :]
            pc0 = self.PC[0, pc_index, :, :]
            diff = pc1[:, :, np.newaxis] - pc0[:, :, np.newaxis]
            diff /= np.abs(diff).max() * 2
            diff += 0.5
            self.img0.setImage(np.tile(diff * 255, (1, 1, 3)))
            self.img0.setLevels([0, 255])
            rgb = np.zeros((self.PC.shape[2], self.PC.shape[3], 3), dtype=np.float32)
            rgb[:, :, 0] = (pc1 - pc1.min()) / (pc1.max() - pc1.min()) * 255
            rgb[:, :, 1] = np.minimum(1, np.maximum(0, (pc0 - pc1.min()) / (pc1.max() - pc1.min()))) * 255
            rgb[:, :, 2] = (pc1 - pc1.min()) / (pc1.max() - pc1.min()) * 255
            self.img1.setImage(rgb)
            if self.cframe == 0:
                self.img2.setImage(np.tile(pc0[:, :, np.newaxis], (1, 1, 3)))
            else:
                self.img2.setImage(np.tile(pc1[:, :, np.newaxis], (1, 1, 3)))
            self.img2.setLevels([pc0.min(), pc0.max()])
            self.zoom_plot()
            self.p3.clear()
            colors = [(200, 200, 255), (255, 100, 100), (100, 50, 200)]
            metric_names = ["rigid", "nonrigid", "nonrigid max"]
            if not hasattr(self, "leg"):
                self.leg = pg.LegendItem((100, 60), offset=(350, 30))
                self.leg.setParentItem(self.p3)
                draw_legend = True
            else:
                draw_legend = False
            for j in range(3):
                curve = self.p3.plot(np.arange(1, self.nPCs + 1), self.DX[:, j], pen=colors[j])
                if draw_legend:
                    self.leg.addItem(curve, metric_names[j])
                self.nums[j].setText(f"{metric_names[j]}: {self.DX[pc_index, j]:.3f}")
            self.scatter = pg.ScatterPlotItem()
            self.p3.addItem(self.scatter)
            self.scatter.setData(
                [pc_index + 1, pc_index + 1, pc_index + 1],
                self.DX[pc_index, :].tolist(),
                size=_SCATTER_POINT_SIZE,
                brush=pg.mkBrush(255, 255, 255),
            )
            self.p3.setLabel("left", "pixel shift")
            self.p3.setLabel("bottom", "PC #")

            self.p4.clear()
            self.p4.plot(self.tPC[:, pc_index])
            self.p4.setLabel("left", "magnitude")
            self.p4.setLabel("bottom", "time")
            self.show()
            self.zoom_plot()

    def zoom_plot(self) -> None:
        """Resets all PC image view ranges to fit the full image extent."""
        self.p0.setXRange(0, self.Lx)
        self.p0.setYRange(0, self.Ly)
        self.p1.setXRange(0, self.Lx)
        self.p1.setYRange(0, self.Ly)
        self.p2.setXRange(0, self.Lx)
        self.p2.setYRange(0, self.Ly)

    def plot_clicked(self, event: object) -> None:
        """Handles double-click to zoom the PC image plots."""
        if self.loaded:
            items = self.win.scene().items(event.scenePos())
            for item in items:
                if item in (self.p0, self.p1, self.p2) and event.button() == 1 and event.double():
                    self.zoom_plot()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        """Handles keyboard navigation for PC stepping and animation control."""
        if event.modifiers() != QtCore.Qt.ShiftModifier:
            if event.key() == QtCore.Qt.Key_Left:
                self.pause()
                pc_number = int(self.pc_edit.text())
                pc_number = max(pc_number - 1, 1)
                self.pc_edit.setText(str(pc_number))
                self.plot_frame()
            elif event.key() == QtCore.Qt.Key_Right:
                self.pause()
                pc_number = int(self.pc_edit.text())
                pc_number = min(pc_number + 1, self.nPCs)
                self.pc_edit.setText(str(pc_number))
                self.plot_frame()
            elif event.key() == QtCore.Qt.Key_Space:
                if self.playButton.isEnabled():
                    self.playButton.setChecked(True)
                    self.start()
                else:
                    self.pause()
