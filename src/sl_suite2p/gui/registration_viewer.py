"""Provides registration binary viewer and principal component metrics viewer windows."""

import json
from pathlib import Path

import numpy as np
from PySide6 import QtGui, QtCore
from natsort import natsorted
from tifffile import imread
import pyqtgraph as pg
from scipy.ndimage import gaussian_filter1d
from PySide6.QtWidgets import (
    QLabel,
    QStyle,
    QSlider,
    QWidget,
    QCheckBox,
    QLineEdit,
    QFileDialog,
    QGridLayout,
    QMainWindow,
    QPushButton,
    QToolButton,
    QButtonGroup,
)
from ataraxis_base_utilities import LogLevel, console

from ..io import compute_plane_offsets
from .styles import WHITE_LABEL_STYLESHEET, metrics_font, metrics_font_bold
from .roi_geometry import boundary
from .roi_overlays import hsv2rgb

# Scatter plot marker point size in pixels.
_SCATTER_POINT_SIZE: int = 10

# Size for media control button icons in pixels.
_ICON_SIZE: int = 30

# Gaussian smoothing sigma for z-position correlation filtering.
_Z_SMOOTHING_SIGMA: int = 2

# Alpha value for ROI boundary rendering.
_BOUNDARY_ALPHA: int = 200

# Divisor applied to hue values for red cell coloring.
_RED_CELL_HUE_DIVISOR: float = 1.4

# Offset added to hue values for red cell coloring.
_RED_CELL_HUE_OFFSET: float = 0.1

# Multiplier for real-time playback speed.
_PLAYBACK_SPEED_MULTIPLIER: int = 5

# Number of frames subsampled for dynamic range estimation.
_SUBSAMPLE_FRAME_COUNT: int = 100

# Minimum frame increment for arrow key navigation.
_MIN_FRAME_DELTA: int = 5

# Divisor for computing frame slider step size from total frames.
_FRAME_DELTA_DIVISOR: int = 200

# Maximum ROI index for the input validator.
_MAX_ROI_INDEX: int = 10000

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

# Width for ROI index input field.
_ROI_EDIT_WIDTH: int = 45

# Width for PC number input field.
_PC_EDIT_WIDTH: int = 40


class BinaryPlayer(QMainWindow):
    """Provides a playback window for viewing registered binary imaging data."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initializes the binary player window and all UI components."""
        super().__init__(parent)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(70, 70, 1070, 1070)
        self.setWindowTitle("View registered binary")
        self.cwidget = QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.l0 = QGridLayout()
        self.cwidget.setLayout(self.l0)
        self.win = pg.GraphicsLayoutWidget()
        # --- cells image
        self.win = pg.GraphicsLayoutWidget()
        self.win.move(600, 0)
        self.win.resize(1000, 500)
        self.l0.addWidget(self.win, 1, 2, 13, 14)
        self.loaded = False
        self.zloaded = False
        self.zcorr = None

        # A plot area (ViewBox + axes) for displaying the image
        self.vmain = pg.ViewBox(lockAspect=True, invertY=True, name="plot1")
        self.win.addItem(self.vmain, row=0, col=0)
        self.vmain.setMenuEnabled(False)
        self.imain = pg.ImageItem()
        self.vmain.addItem(self.imain)
        self.cellscatter = pg.ScatterPlotItem()
        self.vmain.addItem(self.cellscatter)
        self.maskmain = pg.ImageItem()

        # side box
        self.vside = pg.ViewBox(lockAspect=True, invertY=True)
        self.vside.setMenuEnabled(False)
        self.iside = pg.ImageItem()
        self.vside.addItem(self.iside)
        self.cellscatter_side = pg.ScatterPlotItem()
        self.vside.addItem(self.cellscatter_side)
        self.maskside = pg.ImageItem()

        # view red channel
        self.redbox = QCheckBox("view red channel")
        self.redbox.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.redbox.setEnabled(False)
        self.redbox.toggled.connect(self.add_red)
        self.l0.addWidget(self.redbox, 0, 5, 1, 1)
        # view masks
        self.maskbox = QCheckBox("view masks")
        self.maskbox.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.maskbox.setEnabled(False)
        self.maskbox.toggled.connect(self.add_masks)
        self.l0.addWidget(self.maskbox, 0, 6, 1, 1)
        # view raw binary
        self.rawbox = QCheckBox("view raw binary")
        self.rawbox.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.rawbox.setEnabled(False)
        self.rawbox.toggled.connect(self.add_raw)
        self.l0.addWidget(self.rawbox, 0, 7, 1, 1)
        # view zstack
        self.zbox = QCheckBox("view z-stack")
        self.zbox.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.zbox.setEnabled(False)
        self.zbox.toggled.connect(self.add_zstack)
        self.l0.addWidget(self.zbox, 0, 8, 1, 1)

        zlabel = QLabel("Z-plane:")
        zlabel.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.l0.addWidget(zlabel, 0, 9, 1, 1)

        self.Zedit = QLineEdit(self)
        self.Zedit.setValidator(QtGui.QIntValidator(0, 0))
        self.Zedit.setText("0")
        self.Zedit.setFixedWidth(30)
        self.Zedit.setAlignment(QtCore.Qt.AlignRight)
        self.l0.addWidget(self.Zedit, 0, 10, 1, 1)

        self.p1 = self.win.addPlot(name="plot_shift", row=1, col=0, colspan=2)
        self.p1.setMouseEnabled(x=True, y=False)
        self.p1.setMenuEnabled(False)
        self.scatter1 = pg.ScatterPlotItem()
        self.scatter1.setData([0, 0], [0, 0])
        self.p1.addItem(self.scatter1)

        self.p2 = self.win.addPlot(name="plot_F", row=2, col=0, colspan=2)
        self.p2.setMouseEnabled(x=True, y=False)
        self.p2.setMenuEnabled(False)
        self.scatter2 = pg.ScatterPlotItem()
        self.p2.setXLink("plot_shift")

        self.p3 = self.win.addPlot(name="plot_Z", row=3, col=0, colspan=2)
        self.p3.setMouseEnabled(x=True, y=False)
        self.p3.setMenuEnabled(False)
        self.scatter3 = pg.ScatterPlotItem()
        self.p3.setXLink("plot_shift")

        self.win.ci.layout.setRowStretchFactor(0, 12)
        self.movieLabel = QLabel("No ops chosen")
        self.movieLabel.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.movieLabel.setAlignment(QtCore.Qt.AlignCenter)
        self.nframes = 0
        self.cframe = 0
        self._create_buttons(parent)
        # create ROI chooser
        self.l0.addWidget(QLabel(""), 6, 0, 1, 2)
        qlabel = QLabel(self)
        qlabel.setText("<font color='white'>Selected ROI:</font>")
        self.l0.addWidget(qlabel, 7, 0, 1, 2)
        self.ROIedit = QLineEdit(self)
        self.ROIedit.setValidator(QtGui.QIntValidator(0, 10000))
        self.ROIedit.setText("0")
        self.ROIedit.setFixedWidth(45)
        self.ROIedit.setAlignment(QtCore.Qt.AlignRight)
        self.ROIedit.returnPressed.connect(self.number_chosen)
        self.l0.addWidget(self.ROIedit, 8, 0, 1, 1)
        # create frame slider
        self.frameLabel = QLabel("Current frame:")
        self.frameLabel.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.frameNumber = QLabel("0")
        self.frameNumber.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.frameSlider = QSlider(QtCore.Qt.Horizontal)
        self.frameSlider.setTickInterval(5)
        self.frameSlider.setTracking(False)
        self.frameDelta = 10
        self.l0.addWidget(QLabel(""), 12, 0, 1, 1)
        self.l0.setRowStretch(12, 1)
        self.l0.addWidget(self.frameLabel, 13, 0, 1, 2)
        self.l0.addWidget(self.frameNumber, 14, 0, 1, 2)
        self.l0.addWidget(self.frameSlider, 13, 2, 14, 13)
        self.l0.addWidget(QLabel(""), 14, 1, 1, 1)
        ll = QLabel("(when paused, left/right arrow keys can move slider)")
        ll.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.l0.addWidget(ll, 16, 0, 1, 3)
        self.frameSlider.valueChanged.connect(self.go_to_frame)
        self.l0.addWidget(self.movieLabel, 0, 0, 1, 5)
        self._update_frame_slider()
        self._update_buttons()
        self.updateTimer = QtCore.QTimer()
        self.updateTimer.timeout.connect(self.next_frame)
        self.cframe = 0
        self.loaded = False
        self.Floaded = False
        self.raw_on = False
        self.red_on = False
        self.z_on = False
        self.wraw = False
        self.wred = False
        self.wraw_wred = False
        self.win.scene().sigMouseClicked.connect(self.plot_clicked)
        # if not a combined recording, automatically open binary
        if hasattr(parent, "ops") and Path(parent.ops["save_path"]).name != "combined":
            ops_path = str((Path(parent.basename) / "ops.npy").resolve())
            console.echo(message=f"Opening file: {ops_path}")
            self.Fcell = parent.Fcell
            self.stat = parent.stat
            self.iscell = parent.iscell
            self.Floaded = True
            self._open_file(ops_path, True)

    def add_masks(self) -> None:
        """Toggles mask overlay visibility on the main and side views."""
        if self.loaded:
            if self.maskbox.isChecked():
                self.vmain.addItem(self.maskmain)
                self.vside.addItem(self.maskside)
            else:
                self.vmain.removeItem(self.maskmain)
                self.vside.removeItem(self.maskside)

    def add_red(self) -> None:
        """Toggles red channel display based on checkbox state."""
        if self.loaded:
            if self.redbox.isChecked():
                self.red_on = True
            else:
                self.red_on = False
            self.next_frame()

    def zoom_image(self) -> None:
        """Resets the main and side view zoom to fit the full image extent."""
        self.vmain.setRange(yRange=(0, self.LY), xRange=(0, self.LX))
        if self.raw_on or self.z_on:
            if self.z_on:
                self.vside.setRange(yRange=(0, self.zLy), xRange=(0, self.zLx))
            else:
                self.vside.setRange(yRange=(0, self.LY), xRange=(0, self.LX))
            self.vside.setXLink("plot1")
            self.vside.setYLink("plot1")

    def add_raw(self) -> None:
        """Toggles raw binary side view display based on checkbox state."""
        if self.loaded:
            if self.rawbox.isChecked():
                self.raw_on = True
                self.win.addItem(self.vside, row=0, col=1)
                self.zoom_image()
            else:
                self.raw_on = False
                self.win.removeItem(self.vside)
            self.next_frame()

    def add_zstack(self) -> None:
        """Toggles z-stack side view display based on checkbox state."""
        if self.loaded:
            if self.zbox.isChecked():
                if self.rawbox.isChecked():
                    self.rawbox.setChecked(False)
                    self.add_raw()
                self.z_on = True
                self.win.addItem(self.vside, row=0, col=1)
            else:
                self.z_on = False
                self.win.removeItem(self.vside)
            self.next_frame()

    def next_frame(self) -> None:
        """Advances to the next frame and updates all display elements."""
        # loop after video finishes
        self.cframe += 1
        if self.cframe > self.nframes - 1:
            self.cframe = 0
            if self.LY > 0:
                for n in range(len(self.reg_file)):
                    self.reg_file[n].seek(0, 0)
            else:
                self.reg_file.seek(0, 0)
                if self.wraw:
                    self.reg_file_raw.seek(0, 0)
                if self.wred:
                    self.reg_file_chan2.seek(0, 0)
                if self.wraw_wred:
                    self.reg_file_raw_chan2.seek(0, 0)
        self.img = np.zeros((self.LY, self.LX), dtype=np.int16)
        for n in range(len(self.reg_loc)):
            buff = self.reg_file[n].read(self.nbytesread[n])
            img = np.reshape(np.frombuffer(buff, dtype=np.int16, offset=0), (self.Ly[n], self.Lx[n]))
            self.img[self.dy[n] : self.dy[n] + self.Ly[n], self.dx[n] : self.dx[n] + self.Lx[n]] = img

        if self.wred and self.red_on:
            buff = self.reg_file_chan2.read(self.nbytesread[0])
            imgred = np.reshape(np.frombuffer(buff, dtype=np.int16, offset=0), (self.Ly[0], self.Lx[0]))[
                :, :, np.newaxis
            ]
            self.img = np.concatenate((self.img[:, :, np.newaxis], imgred, np.zeros_like(imgred)), axis=-1)
        if self.wraw and self.raw_on:
            buff = self.reg_file_raw.read(self.nbytesread[0])
            self.imgraw = np.reshape(np.frombuffer(buff, dtype=np.int16, offset=0), (self.Ly[0], self.Lx[0]))
            if self.wraw_wred:
                buff = self.reg_file_raw_chan2.read(self.nbytesread[0])
                imgred_raw = np.reshape(np.frombuffer(buff, dtype=np.int16, offset=0), (self.Ly[0], self.Lx[0]))[
                    :, :, np.newaxis
                ]
                self.imgraw = np.concatenate(
                    (self.imgraw[:, :, np.newaxis], imgred_raw, np.zeros_like(imgred_raw)), axis=-1
                )
            self.iside.setImage(self.imgraw, levels=self.srange)
        if self.zloaded and self.z_on:
            if hasattr(self, "zmax"):
                self.Zedit.setText(str(self.zmax[self.cframe]))
            self.iside.setImage(self.zstack[int(self.Zedit.text())], levels=self.zrange)

        self.imain.setImage(self.img, levels=self.srange)
        self.frameSlider.setValue(self.cframe)
        self.frameNumber.setText(str(self.cframe))
        self.scatter1.setData(
            [self.cframe, self.cframe],
            [self.yoff[self.cframe], self.xoff[self.cframe]],
            size=10,
            brush=pg.mkBrush(255, 0, 0),
        )
        if self.Floaded:
            self.scatter2.setData(
                [self.cframe, self.cframe],
                [self.ft[self.cframe], self.ft[self.cframe]],
                size=10,
                brush=pg.mkBrush(255, 0, 0),
            )
        if self.zloaded and self.z_on:
            self.scatter3.setData(
                [self.cframe, self.cframe],
                [self.zmax[self.cframe], self.zmax[self.cframe]],
                size=10,
                brush=pg.mkBrush(255, 0, 0),
            )

    def make_masks(self) -> None:
        """Generates ROI boundary masks and color overlays for all detected cells."""
        ncells = len(self.stat)
        generator = np.random.default_rng(seed=0)
        allcols = generator.random(ncells)
        if hasattr(self, "redcell"):
            allcols = allcols / 1.4
            allcols = allcols + 0.1
            allcols[self.redcell] = 0
        self.colors = hsv2rgb(allcols)
        self.RGB = -1 * np.ones((self.LY, self.LX, 3), np.int32)
        self.cellpix = -1 * np.ones((self.LY, self.LX), np.int32)
        self.sroi = np.zeros((self.LY, self.LX), np.uint8)

        for n in np.nonzero(self.iscell)[0]:
            ypix = self.stat[n]["y_pixels"].flatten()
            xpix = self.stat[n]["x_pixels"].flatten()
            if not self.ops[0]["allow_overlap"]:
                ypix = ypix[~self.stat[n]["overlap_mask"]]
                xpix = xpix[~self.stat[n]["overlap_mask"]]
            yext, xext = boundary(ypix, xpix)
            if len(yext) > 0:
                goodi = (yext >= 0) & (xext >= 0) & (yext < self.LY) & (xext < self.LX)
                self.stat[n]["yext"] = yext[goodi] + 0.5
                self.stat[n]["xext"] = xext[goodi] + 0.5
                self.sroi[yext[goodi], xext[goodi]] = 200
                self.RGB[yext[goodi], xext[goodi]] = self.colors[n]
            else:
                self.stat[n]["yext"] = yext
                self.stat[n]["xext"] = xext
            self.cellpix[ypix, xpix] = n
        self.mask_bool = self.sroi > 0
        self.allmasks = np.concatenate((self.RGB, self.sroi[:, :, np.newaxis]), axis=-1)
        self.maskmain.setImage(self.allmasks, levels=[0, 255])
        self.maskside.setImage(self.allmasks, levels=[0, 255])

    def plot_trace(self) -> None:
        """Plots the fluorescence trace for the currently selected ROI."""
        self.p2.clear()
        self.ft = self.Fcell[self.ichosen, :]
        self.p2.plot(self.ft, pen=self.colors[self.ichosen])
        self.p2.addItem(self.scatter2)
        self.scatter2.setData([self.cframe], [self.ft[self.cframe]], size=10, brush=pg.mkBrush(255, 0, 0))
        self.p2.setLimits(yMin=self.ft.min(), yMax=self.ft.max())
        self.p2.setRange(xRange=(0, self.nframes), yRange=(self.ft.min(), self.ft.max()), padding=0.0)
        self.p2.setLimits(xMin=0, xMax=self.nframes)

    def open(self) -> None:
        """Opens a file dialog to select and load a single-plane ops file."""
        filename = QFileDialog.getOpenFileName(self, "Open single-plane ops.npy file or single-plane ops.json file")
        # load ops in same folder
        if filename:
            console.echo(message=f"Opening file: {filename[0]}")
            self._open_file(filename[0], False)

    def open_combined(self) -> None:
        """Opens a folder dialog to load multi-plane combined binary data."""
        filename = QFileDialog.getExistingDirectory(
            self, "Load binaries for all planes (choose folder with planeX folders)"
        )
        # load ops in same folder
        if filename:
            console.echo(message=f"Opening combined folder: {filename}")
            self._open_combined(filename)

    def _open_combined(self, save_folder: str) -> None:
        """Opens and loads binary data from a combined multi-plane folder."""
        try:
            save_path = Path(save_folder)
            plane_folders = natsorted(
                [entry for entry in save_path.iterdir() if entry.is_dir() and entry.name[:5] == "plane"],
            )
            ops1 = [np.load(folder / "ops.npy", allow_pickle=True).item() for folder in plane_folders]
            self.LY = 0
            self.LX = 0
            self.reg_loc = []
            self.reg_file = []
            self.Ly = []
            self.Lx = []
            self.dy = []
            self.dx = []
            self.wraw = False
            self.wred = False
            self.wraw_wred = False
            # check that all binaries still exist
            dy, dx = compute_plane_offsets(ops1)
            for ipl, ops in enumerate(ops1):
                registered_path = Path(ops["registered_binary_path"])
                if registered_path.is_file():
                    binary_path = str(registered_path)
                else:
                    binary_path = str((save_path / f"plane{ipl}" / "data.bin").resolve())
                console.echo(
                    message=f"Registration file: {binary_path}, exists: {Path(binary_path).is_file()}",
                )
                self.reg_loc.append(binary_path)
                self.reg_file.append(Path(self.reg_loc[-1]).open("rb"))
                self.Ly.append(ops["frame_height"])
                self.Lx.append(ops["frame_width"])
                self.dy.append(dy[ipl])
                self.dx.append(dx[ipl])
                self.LY = np.maximum(self.LY, self.Ly[-1] + self.dy[-1])
                self.LX = np.maximum(self.LX, self.Lx[-1] + self.dx[-1])
                good = True
            self.Floaded = False

        except Exception as e:
            console.echo(message=f"ERROR: {e}", level=LogLevel.ERROR)
            console.echo(message="Could be incorrect folder or missing binaries", level=LogLevel.WARNING)
            good = False
            try:
                for n in range(len(self.reg_loc)):
                    self.reg_file[n].close()
                console.echo(message="Closed binaries", level=LogLevel.SUCCESS)
            except Exception:
                console.echo(message="Tried to close binaries", level=LogLevel.WARNING)
        if good:
            self.filename = save_folder
            self.ops = ops1
            self.setup_views()

    def _open_file(self, filename: str, fromgui: bool) -> None:
        """Opens a single-plane ops file and its associated binary data."""
        try:
            file_path = Path(filename)
            ext = file_path.suffix
            if ext == ".npy":
                ops = np.load(filename, allow_pickle=True).item()
                parent_dir = file_path.parent
            elif ext == ".json":
                with file_path.open() as f:
                    ops = json.load(f)
                ops["frame_height"] = ops["Lys"] if isinstance(ops["Lys"], int) else ops["Lys"][0]
                ops["frame_width"] = ops["Lxs"] if isinstance(ops["Lxs"], int) else ops["Lxs"][0]
                parent_dir = file_path.parent / "suite2p" / "plane0"
                ops["registered_binary_path"] = str(parent_dir / "data.bin")
                nbytesread = np.int64(2 * ops["frame_height"] * ops["frame_width"])
                ops["frame_count"] = Path(ops["registered_binary_path"]).stat().st_size // nbytesread
            self.LY = ops["frame_height"]
            self.LX = ops["frame_width"]
            self.Ly = [ops["frame_height"]]
            self.Lx = [ops["frame_width"]]
            self.dx = [0]
            self.dy = [0]

            registered_path = Path(ops["registered_binary_path"])
            if registered_path.is_file():
                self.reg_loc = [ops["registered_binary_path"]]
            else:
                self.reg_loc = [str((parent_dir / "data.bin").resolve())]
            self.reg_file = [Path(self.reg_loc[-1]).open("rb")]
            self.wraw = False
            self.wred = False
            self.wraw_wred = False
            if "raw_binary_path" in ops:
                if self.reg_loc == ops["registered_binary_path"]:
                    self.reg_loc_raw = ops["raw_binary_path"]
                else:
                    self.reg_loc_raw = str((file_path.parent / "data_raw.bin").resolve())
                try:
                    self.reg_file_raw = Path(self.reg_loc_raw).open("rb")
                    self.wraw = True
                except Exception:
                    self.wraw = False
            if "registered_binary_path_channel_2" in ops:
                if self.reg_loc == ops["registered_binary_path"]:
                    self.reg_loc_red = ops["registered_binary_path_channel_2"]
                else:
                    self.reg_loc_red = str((file_path.parent / "data_chan2.bin").resolve())
                self.reg_file_chan2 = Path(self.reg_loc_red).open("rb")
                self.wred = True
            if "raw_binary_path_channel_2" in ops:
                if self.reg_loc == ops["registered_binary_path"]:
                    self.reg_loc_raw_chan2 = ops["raw_binary_path_channel_2"]
                else:
                    self.reg_loc_raw_chan2 = str((file_path.parent / "data_raw_chan2.bin").resolve())
                try:
                    self.reg_file_raw_chan2 = Path(self.reg_loc_raw_chan2).open("rb")
                    self.wraw_wred = True
                except Exception:
                    self.wraw_wred = False

            if not fromgui:
                fluorescence_path = file_path.parent / "F.npy"
                if fluorescence_path.is_file():
                    self.Fcell = np.load(fluorescence_path)
                    self.stat = np.load(file_path.parent / "stat.npy", allow_pickle=True)
                    self.iscell = np.load(file_path.parent / "iscell.npy", allow_pickle=True)
                    self.Floaded = True
                else:
                    self.Floaded = False
            else:
                self.Floaded = True
            good = True
            console.echo(message=f"Fluorescence data loaded: {self.Floaded}")
            self.filename = filename
        except Exception as e:
            console.echo(
                message="ERROR: ops.npy incorrect / missing ops['registered_binary_path'] and others",
                level=LogLevel.ERROR,
            )
            console.echo(message=f"Error details: {e}", level=LogLevel.ERROR)
            try:
                for n in range(len(self.reg_loc)):
                    self.reg_file[n].close()
                console.echo(message="Closed binaries", level=LogLevel.SUCCESS)
            except Exception:
                console.echo(message="Tried to close binaries", level=LogLevel.WARNING)
            good = False
        if good:
            self.filename = filename
            self.ops = [ops]
            self.setup_views()

    def setup_views(self) -> None:
        """Configures all plot views and display parameters after loading data."""
        self.p1.clear()
        self.p2.clear()
        self.ichosen = 0
        self.ROIedit.setText("0")
        # get scaling from 100 random frames
        ops = self.ops[-1]
        frames = _subsample_frames(ops, np.minimum(ops["frame_count"] - 1, 100), self.reg_loc[-1])
        self.srange = frames.mean() + frames.std() * np.array([-2, 5])

        self.movieLabel.setText(self.reg_loc[-1])
        self.nbytesread = []
        for n in range(len(self.reg_loc)):
            self.nbytesread.append(2 * self.Ly[n] * self.Lx[n])

        # aspect ratio
        if "aspect_ratio" in ops:
            self.xyrat = ops["aspect_ratio"]
        elif "cell_diameter" in ops and (type(ops["cell_diameter"]) is not int) and (len(ops["cell_diameter"]) > 1):
            self.xyrat = ops["cell_diameter"][0] / ops["cell_diameter"][1]
        else:
            self.xyrat = 1.0
        self.vmain.setAspectLocked(lock=True, ratio=self.xyrat)
        self.vside.setAspectLocked(lock=True, ratio=self.xyrat)

        self.nframes = ops["frame_count"]
        self.time_step = 1.0 / ops["sampling_rate"] * 1000 / 5  # 5x real-time
        self.frameDelta = int(np.maximum(5, self.nframes / 200))
        self.frameSlider.setSingleStep(self.frameDelta)
        self.currentMovieDirectory = QtCore.QFileInfo(self.filename).path()
        if self.nframes > 0:
            self._update_frame_slider()
            self._update_buttons()
        # plot ops X-Y offsets
        if "rigid_y_offsets" in ops:
            self.yoff = ops["rigid_y_offsets"]
            self.xoff = ops["rigid_x_offsets"]
        else:
            self.yoff = np.zeros((ops["frame_count"],))
            self.xoff = np.zeros((ops["frame_count"],))
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
            size=10,
            brush=pg.mkBrush(255, 0, 0),
        )

        if self.wraw:
            self.rawbox.setEnabled(True)
        else:
            self.rawbox.setEnabled(False)
        if self.wred:
            self.redbox.setEnabled(True)
        else:
            self.redbox.setEnabled(False)

        if self.Floaded:
            self.maskbox.setEnabled(True)
            self.make_masks()
            self.cell_chosen()

        self.cframe = -1
        self.loaded = True
        self.next_frame()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        """Handles keyboard navigation for frame stepping and playback control."""
        if self.playButton.isEnabled() and event.modifiers() != QtCore.Qt.ShiftModifier:
            if event.key() == QtCore.Qt.Key_Left:
                self.cframe -= self.frameDelta
                self.cframe = np.maximum(0, np.minimum(self.nframes - 1, self.cframe))
                self.frameSlider.setValue(self.cframe)
            elif event.key() == QtCore.Qt.Key_Right:
                self.cframe += self.frameDelta
                self.cframe = np.maximum(0, np.minimum(self.nframes - 1, self.cframe))
                self.frameSlider.setValue(self.cframe)
        if event.modifiers() != QtCore.Qt.ShiftModifier and event.key() == QtCore.Qt.Key_Space:
            if self.playButton.isEnabled():
                # then play
                self.start()
            else:
                self.pause()

    def number_chosen(self) -> None:
        """Processes the ROI number entered by the user."""
        self.ichosen = int(self.ROIedit.text())
        self.cell_chosen()

    def cell_chosen(self) -> None:
        """Updates the display to highlight and trace the currently selected cell."""
        if self.Floaded:
            self.cell_mask()
            self.ROIedit.setText(str(self.ichosen))
            rgb = np.array(self.colors[self.ichosen])
            self.cellscatter.setData(self.xext, self.yext, pen=pg.mkPen(list(rgb)), brush=pg.mkBrush(list(rgb)), size=3)
            self.cellscatter_side.setData(
                self.xext, self.yext, pen=pg.mkPen(list(rgb)), brush=pg.mkBrush(list(rgb)), size=3
            )

            if self.ichosen >= len(self.stat):
                self.ichosen = len(self.stat) - 1
            self.cell_mask()
            self.ft = self.Fcell[self.ichosen, :]
            self.plot_trace()
            self.p2.setXLink("plot_shift")
            self.jump_to_frame()
            self.show()

    def plot_clicked(self, event: object) -> None:
        """Handles mouse click events on plots for frame navigation and cell selection."""
        items = self.win.scene().items(event.scenePos())
        posx = 0
        posy = 0
        iplot = 0
        zoom = False
        _zoom_image = False
        choose = False
        if self.loaded:
            for x in items:
                if x == self.p1:
                    vb = self.p1.vb
                    pos = vb.mapSceneToView(event.scenePos())
                    posx = pos.x()
                    iplot = 1
                elif x == self.p2 and self.Floaded:
                    vb = self.p1.vb
                    pos = vb.mapSceneToView(event.scenePos())
                    posx = pos.x()
                    iplot = 2
                elif x in (self.vmain, self.vside):
                    if event.button() == 1:
                        if event.double():
                            self.zoom_image()
                        elif self.Floaded:
                            pos = x.mapSceneToView(event.scenePos())
                            posy = int(pos.x())
                            posx = int(pos.y())
                            if (
                                0 <= posy < self.LX
                                and 0 <= posx < self.LY
                                and self.cellpix[posx, posy] > -1
                            ):
                                self.ichosen = self.cellpix[posx, posy]
                                self.cell_chosen()
                if iplot in (1, 2) and event.button() == 1:
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
            self.frameSlider.setValue(self.cframe)

    def load_zstack(self) -> None:
        """Opens a file dialog to load a z-stack TIFF and initializes z-position tracking."""
        name = QFileDialog.getOpenFileName(self, "Open zstack", filter="*.tif")
        self.fname = name[0]
        try:
            self.zstack = imread(self.fname)
            self.zLy, self.zLx = self.zstack.shape[1:]
            self.Zedit.setValidator(QtGui.QIntValidator(0, self.zstack.shape[0]))
            self.zrange = [np.percentile(self.zstack, 1), np.percentile(self.zstack, 99)]

            self.computeZ.setEnabled(True)
            self.zloaded = True
            self.zbox.setEnabled(True)
            self.zbox.setChecked(True)
            self.zmax = np.zeros(self.nframes, "int")

            # Checks for cached zcorr data in order of priority:
            # 1. Local instance cache (self.zcorr)
            # 2. Separate zcorr.npy file (new format)
            # 3. Legacy ops["zcorr"] format
            if self.zcorr is not None and self.zstack.shape[0] == self.zcorr.shape[0]:
                self.zmax = np.argmax(gaussian_filter1d(self.zcorr.T.copy(), 2, axis=1), axis=1)
                self.plot_zcorr()
            else:
                # Attempts to load from separate zcorr.npy file.
                zcorr_path = Path(self.filename).parent / "zcorr.npy"
                if zcorr_path.exists():
                    self.zcorr = np.load(zcorr_path)
                    if self.zstack.shape[0] == self.zcorr.shape[0]:
                        self.zmax = np.argmax(gaussian_filter1d(self.zcorr.T.copy(), 2, axis=1), axis=1)
                        self.plot_zcorr()
                elif "zcorr" in self.ops[0] and self.zstack.shape[0] == self.ops[0]["zcorr"].shape[0]:
                    # Falls back to legacy ops format for backward compatibility.
                    self.zcorr = self.ops[0]["zcorr"]
                    self.zmax = np.argmax(gaussian_filter1d(self.zcorr.T.copy(), 2, axis=1), axis=1)
                    self.plot_zcorr()

        except Exception as e:
            console.echo(message=f"ERROR: {e}", level=LogLevel.ERROR)

    def cell_mask(self) -> None:
        """Extracts the boundary coordinates for the currently selected cell."""
        self.yext = self.stat[self.ichosen]["yext"]
        self.xext = self.stat[self.ichosen]["xext"]

    def go_to_frame(self) -> None:
        """Seeks to the frame indicated by the frame slider position."""
        self.cframe = int(self.frameSlider.value())
        self.jump_to_frame()

    def _update_frame_slider(self) -> None:
        """Configures the frame slider range and enables it."""
        self.frameSlider.setMaximum(self.nframes - 1)
        self.frameSlider.setMinimum(0)
        self.frameLabel.setEnabled(True)
        self.frameSlider.setEnabled(True)

    def _update_buttons(self) -> None:
        """Sets the initial enabled state for play and pause buttons."""
        self.playButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        self.pauseButton.setChecked(True)

    def _create_buttons(self, parent: QWidget | None) -> None:
        """Creates and lays out all control buttons for the player window."""
        icon_size = QtCore.QSize(30, 30)
        open_button = QPushButton("load ops.npy")
        open_button.setToolTip("Open single-plane ops.npy")
        open_button.clicked.connect(self.open)

        open_combined_button = QPushButton("load folder")
        open_combined_button.setToolTip("Choose a folder with planeX folders to load together")
        open_combined_button.clicked.connect(self.open_combined)

        load_z_button = QPushButton("load z-stack tiff")
        load_z_button.clicked.connect(self.load_zstack)

        self.computeZ = QPushButton("compute z position")
        self.computeZ.setEnabled(False)
        self.computeZ.clicked.connect(lambda: self.compute_z(parent))

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

        self.l0.addWidget(open_button, 1, 0, 1, 2)
        self.l0.addWidget(open_combined_button, 2, 0, 1, 2)
        self.l0.addWidget(load_z_button, 3, 0, 1, 2)
        self.l0.addWidget(self.computeZ, 4, 0, 1, 2)
        self.l0.addWidget(self.playButton, 15, 0, 1, 1)
        self.l0.addWidget(self.pauseButton, 15, 1, 1, 1)
        self.playButton.setEnabled(False)
        self.pauseButton.setEnabled(False)
        self.pauseButton.setChecked(True)

    def jump_to_frame(self) -> None:
        """Seeks all binary file handles to an absolute frame position and displays it."""
        if self.playButton.isEnabled():
            self.cframe = np.maximum(0, np.minimum(self.nframes - 1, self.cframe))
            self.cframe = int(self.cframe)
            # seek to absolute position
            for n in range(len(self.reg_file)):
                self.reg_file[n].seek(self.nbytesread[n] * self.cframe, 0)
            if self.wraw:
                self.reg_file_raw.seek(self.nbytesread[-1] * self.cframe, 0)
            if self.wred:
                self.reg_file_chan2.seek(self.nbytesread[-1] * self.cframe, 0)
            if self.wraw_wred:
                self.reg_file_raw_chan2.seek(self.nbytesread[-1] * self.cframe, 0)
            self.cframe -= 1
            self.next_frame()

    def start(self) -> None:
        """Starts video playback by enabling the frame update timer."""
        if self.cframe < self.nframes - 1:
            console.echo(message="Playing video...")
            self.playButton.setEnabled(False)
            self.pauseButton.setEnabled(True)
            self.frameSlider.setEnabled(False)
            self.updateTimer.start(self.time_step)

    def pause(self) -> None:
        """Pauses video playback and re-enables manual frame navigation."""
        self.updateTimer.stop()
        self.playButton.setEnabled(True)
        self.pauseButton.setEnabled(False)
        self.frameSlider.setEnabled(True)
        console.echo(message="Video paused")

    def compute_z(self, parent: QWidget | None) -> None:  # noqa: ARG002
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


def _subsample_frames(ops: dict, sample_count: int, registration_path: str) -> np.ndarray:
    """Reads evenly-spaced frames from a binary file for dynamic range estimation.

    Args:
        ops: Operations dictionary containing frame dimensions and count.
        sample_count: Number of frames to subsample.
        registration_path: Path to the registered binary file.

    Returns:
        Array of subsampled frames with shape (sample_count, height, width).
    """
    frame_count = ops["frame_count"]
    height = ops["frame_height"]
    width = ops["frame_width"]
    frames = np.zeros((sample_count, height, width), dtype=np.int16)
    bytes_per_frame = 2 * height * width
    start_indices = np.linspace(0, frame_count, 1 + sample_count).astype(np.int64)
    binary_file = Path(registration_path).open("rb")
    for index in range(sample_count):
        binary_file.seek(bytes_per_frame * start_indices[index], 0)
        buffer = binary_file.read(bytes_per_frame)
        data = np.frombuffer(buffer, dtype=np.int16, offset=0)
        frames[index, :, :] = np.reshape(data, (height, width))
    binary_file.close()
    return frames


class PCViewer(QMainWindow):
    """Provides a viewer window for principal component registration metrics."""

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initializes the PC viewer window and all UI components."""
        super().__init__(parent)
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(70, 70, 1300, 800)
        self.setWindowTitle("Metrics for registration")
        self.cwidget = QWidget(self)
        self.setCentralWidget(self.cwidget)
        self.l0 = QGridLayout()
        self.cwidget.setLayout(self.l0)

        self.win = pg.GraphicsLayoutWidget()
        # --- cells image
        self.win = pg.GraphicsLayoutWidget()
        self.l0.addWidget(self.win, 0, 2, 13, 14)
        # A plot area (ViewBox + axes) for displaying the image
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

        self.PCedit = QLineEdit(self)
        self.PCedit.setText("1")
        self.PCedit.setFixedWidth(40)
        self.PCedit.setAlignment(QtCore.Qt.AlignRight)
        self.PCedit.returnPressed.connect(self.plot_frame)
        self.PCedit.textEdited.connect(self.pause)
        qlabel = QLabel("PC: ")
        boldfont = metrics_font_bold()
        bigfont = metrics_font()
        qlabel.setFont(boldfont)
        self.PCedit.setFont(bigfont)
        qlabel.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.l0.addWidget(QLabel(""), 1, 0, 1, 1)
        self.l0.addWidget(qlabel, 2, 0, 1, 1)
        self.l0.addWidget(self.PCedit, 2, 1, 1, 1)
        self.nums = []
        self.titles = []
        for j in range(3):
            num1 = QLabel("")
            num1.setStyleSheet(WHITE_LABEL_STYLESHEET)
            self.l0.addWidget(num1, 3 + j, 0, 1, 2)
            self.nums.append(num1)
            t1 = QLabel("")
            t1.setStyleSheet(WHITE_LABEL_STYLESHEET)
            self.l0.addWidget(t1, 12, 4 + j * 4, 1, 2)
            self.titles.append(t1)
        self.loaded = False
        self.wraw = False
        self.wred = False
        self.wraw_wred = False
        self.l0.addWidget(QLabel(""), 7, 0, 1, 1)
        self.l0.setRowStretch(7, 1)
        self.cframe = 0
        self._create_buttons()
        self.nPCs = 50
        self.PCedit.setValidator(QtGui.QIntValidator(1, self.nPCs))
        # play button
        self.updateTimer = QtCore.QTimer()
        self.updateTimer.timeout.connect(self.next_frame)
        # if not a combined recording, automatically open binary
        if hasattr(parent, "ops") and Path(parent.ops["save_path"]).name != "combined":
            ops_path = str((Path(parent.basename) / "ops.npy").resolve())
            console.echo(message=f"Opening ops file: {ops_path}")
            self._open_file(ops_path)

    def _create_buttons(self) -> None:
        """Creates and lays out the open, play, and pause buttons."""
        icon_size = QtCore.QSize(30, 30)
        open_button = QToolButton()
        open_button.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        open_button.setIconSize(icon_size)
        open_button.setToolTip("Open ops file")
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
            self.updateTimer.start(200)

    def pause(self) -> None:
        """Pauses PC animation playback."""
        self.updateTimer.stop()
        self.playButton.setEnabled(True)
        self.pauseButton.setChecked(True)
        self.pauseButton.setEnabled(False)

    def open(self) -> None:
        """Opens a file dialog to select and load a single-plane ops file."""
        filename = QFileDialog.getOpenFileName(self, "Open single-plane ops.npy file", filter="ops*.npy")
        # load ops in same folder
        if filename:
            console.echo(message=f"Opening ops file: {filename[0]}")
            self._open_file(filename[0])

    def _open_file(self, filename: str) -> None:
        """Loads principal component data from the specified ops file."""
        try:
            ops = np.load(filename, allow_pickle=True).item()
            self.PC = ops["regPC"]
            self.PC = np.clip(self.PC, np.percentile(self.PC, 1), np.percentile(self.PC, 99))

            self.Ly, self.Lx = self.PC.shape[2:]
            self.DX = ops["regDX"]
            if "tPC" in ops:
                self.tPC = ops["tPC"]
            else:
                self.tPC = np.zeros((1, self.PC.shape[1]))
            good = True
        except Exception as e:
            console.echo(
                message="ERROR: ops.npy incorrect / missing ops['regPC'] and ops['regDX']",
                level=LogLevel.ERROR,
            )
            console.echo(message=f"Error details: {e}", level=LogLevel.ERROR)
            good = False
        if good:
            self.loaded = True
            self.nPCs = self.PC.shape[1]
            self.PCedit.setValidator(QtGui.QIntValidator(1, self.nPCs))
            self.plot_frame()
            self.playButton.setEnabled(True)

    def next_frame(self) -> None:
        """Advances the PC animation to the next frame, toggling between top and bottom halves."""
        pc_index = int(self.PCedit.text()) - 1
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
            pc_index = int(self.PCedit.text()) - 1
            pc1 = self.PC[1, pc_index, :, :]
            pc0 = self.PC[0, pc_index, :, :]
            diff = pc1[:, :, np.newaxis] - pc0[:, :, np.newaxis]
            diff /= np.abs(diff).max() * 2
            diff += 0.5
            self.img0.setImage(np.tile(diff * 255, (1, 1, 3)))
            self.img0.setLevels([0, 255])
            rgb = np.zeros((self.PC.shape[2], self.PC.shape[3], 3), np.float32)
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
            p = [(200, 200, 255), (255, 100, 100), (100, 50, 200)]
            ptitle = ["rigid", "nonrigid", "nonrigid max"]
            if not hasattr(self, "leg"):
                self.leg = pg.LegendItem((100, 60), offset=(350, 30))
                self.leg.setParentItem(self.p3)
                draw_legend = True
            else:
                draw_legend = False
            for j in range(3):
                cj = self.p3.plot(np.arange(1, self.nPCs + 1), self.DX[:, j], pen=p[j])
                if draw_legend:
                    self.leg.addItem(cj, ptitle[j])
                self.nums[j].setText(f"{ptitle[j]}: {self.DX[pc_index, j]:.3f}")
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
            for x in items:
                if x in (self.p0, self.p1, self.p2) and event.button() == 1 and event.double():
                    self.zoom_plot()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        """Handles keyboard navigation for PC stepping and animation control."""
        if event.modifiers() != QtCore.Qt.ShiftModifier:
            if event.key() == QtCore.Qt.Key_Left:
                self.pause()
                ipc = int(self.PCedit.text())
                ipc = max(ipc - 1, 1)
                self.PCedit.setText(str(ipc))
                self.plot_frame()
            elif event.key() == QtCore.Qt.Key_Right:
                self.pause()
                ipc = int(self.PCedit.text())
                ipc = min(ipc + 1, self.nPCs)
                self.PCedit.setText(str(ipc))
                self.plot_frame()
            elif event.key() == QtCore.Qt.Key_Space:
                if self.playButton.isEnabled():
                    # then play
                    self.playButton.setChecked(True)
                    self.start()
                else:
                    self.pause()
