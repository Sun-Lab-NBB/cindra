"""Provides the rastermap visualization window and supporting slider widgets."""

from __future__ import annotations

import sys
import time
from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtGui, QtCore
import pyqtgraph as pg
from matplotlib import cm
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter1d
from PySide6.QtWidgets import (
    QLabel,
    QStyle,
    QSlider,
    QWidget,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QGridLayout,
    QMainWindow,
    QPushButton,
    QApplication,
    QStyleOptionSlider,
)
from rastermap.rastermap import Rastermap
from ataraxis_base_utilities import LogLevel, console

from .styles import WHITE_LABEL_STYLESHEET, RANGE_SLIDER_STYLESHEET, BUTTON_UNPRESSED_STYLESHEET
from .roi_overlays import rastermap_masks

if TYPE_CHECKING:
    from .main_window import MainWindow

# Bin size divisor for computing the default temporal bin.
_BIN_SIZE_DIVISOR: int = 10

# Z-score clipping bounds for spike visualization.
_ZSCORE_LOWER_BOUND: float = -4.0
_ZSCORE_UPPER_BOUND: float = 8.0
_ZSCORE_NORMALIZATION: float = 12.0

# Default initial saturation levels (fraction of range).
_DEFAULT_SATURATION_LOW: float = 0.3
_DEFAULT_SATURATION_HIGH: float = 0.7

# Slider range as percentage.
_SLIDER_MIN: int = 0
_SLIDER_MAX: int = 100
_SLIDER_LOW_DEFAULT: int = 30
_SLIDER_HIGH_DEFAULT: int = 70
_SLIDER_PERCENTAGE: float = 100.0

# ROI pen width.
_ROI_PEN_WIDTH: int = 3

# ROI handle size.
_ROI_HANDLE_SIZE: int = 10

# ROI z-value to ensure drawing above images.
_ROI_Z_VALUE: int = 10

# Maximum number of cells that can be selected for GUI display.
_MAX_SELECTED_CELLS: int = 5000

# Direction index for left arrow key (used in navigation logic).
_DIRECTION_LEFT: int = 2

# Movement fraction for arrow key navigation.
_MOVEMENT_FRACTION: float = 0.05

# Gaussian smoothing fraction for neural sorting.
_SMOOTHING_FRACTION: float = 0.005

# Minimum and maximum smoothing sigma.
_MIN_SMOOTHING_SIGMA: int = 1
_MAX_SMOOTHING_SIGMA: int = 8

# Neuropil coefficient for activity mode 2.
_NEUROPIL_COEFFICIENT: float = 0.7

# Colormap truncation offset (removes last 3 entries from matplotlib LUT).
_COLORMAP_TRUNCATION: int = 3

# Number of activity modes that require fluorescence subtraction.
_ACTIVITY_MODE_COUNT: int = 2


class _VerticalLabel(QWidget):
    """Widget that renders text rotated 90 degrees for vertical labeling.

    Args:
        text: The label text to display vertically.
    """

    def __init__(self, text: str | None = None) -> None:
        super().__init__()
        self._text = text

    def paintEvent(self, event: object) -> None:  # noqa: N802, ARG002
        """Renders the rotated text."""
        painter = QtGui.QPainter(self)
        painter.setPen(QtCore.Qt.white)
        painter.translate(0, 0)
        painter.rotate(90)
        if self._text:
            painter.drawText(0, 0, self._text)
        painter.end()


class _RangeSlider(QSlider):
    """Dual-handle range slider for selecting a value range.

    Provides two slider handles (low and high) that define a range within the
    slider's minimum and maximum bounds. The low handle cannot exceed the high
    handle and vice versa.

    Args:
        parent: The parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__()

        self._low = self.minimum()
        self._high = self.maximum()

        self._pressed_control = QStyle.SC_None
        self._hover_control = QStyle.SC_None
        self._click_offset = 0

        self.setOrientation(QtCore.Qt.Vertical)
        self.setTickPosition(QSlider.TicksRight)
        self.setStyleSheet(RANGE_SLIDER_STYLESHEET)
        self._active_slider = 0
        self._parent = parent

    def level_change(self) -> None:
        """Updates the saturation range from the current handle positions."""
        self.saturation = [self._low, self._high]

    def low(self) -> int:
        """Returns the current low handle value."""
        return self._low

    def set_low(self, low: int) -> None:
        """Sets the low handle value.

        Args:
            low: The new low value.
        """
        self._low = low
        self.update()

    def high(self) -> int:
        """Returns the current high handle value."""
        return self._high

    def set_high(self, high: int) -> None:
        """Sets the high handle value.

        Args:
            high: The new high value.
        """
        self._high = high
        self.update()

    def paintEvent(self, event: object) -> None:  # noqa: N802, ARG002
        """Renders both slider handles."""
        painter = QtGui.QPainter(self)
        style = QApplication.style()
        for _index, value in enumerate([self._low, self._high]):
            option = QStyleOptionSlider()
            self.initStyleOption(option)
            option.subControls = QStyle.SC_SliderHandle
            if self.tickPosition() != self.NoTicks:
                option.subControls |= QStyle.SC_SliderTickmarks
            if self._pressed_control:
                option.activeSubControls = self._pressed_control
                option.state |= QStyle.State_Sunken
            else:
                option.activeSubControls = self._hover_control
            option.sliderPosition = value
            option.sliderValue = value
            style.drawComplexControl(QStyle.CC_Slider, option, painter, self)

    def mousePressEvent(self, event: object) -> None:  # noqa: N802
        """Handles mouse press to begin slider drag."""
        event.accept()
        style = QApplication.style()
        button = event.button()
        if button:
            option = QStyleOptionSlider()
            self.initStyleOption(option)
            self._active_slider = -1
            for handle_index, value in enumerate([self._low, self._high]):
                option.sliderPosition = value
                hit = style.hitTestComplexControl(style.CC_Slider, option, event.pos(), self)
                if hit == style.SC_SliderHandle:
                    self._active_slider = handle_index
                    self._pressed_control = hit
                    self.triggerAction(self.SliderMove)
                    self.setRepeatAction(self.SliderNoAction)
                    self.setSliderDown(True)
                    break
            if self._active_slider < 0:
                self._pressed_control = QStyle.SC_SliderHandle
                self._click_offset = self._pixel_position_to_range_value(self._pick(event.pos()))
                self.triggerAction(self.SliderMove)
                self.setRepeatAction(self.SliderNoAction)
        else:
            event.ignore()

    def mouseMoveEvent(self, event: object) -> None:  # noqa: N802
        """Handles mouse drag to move slider handles."""
        if self._pressed_control != QStyle.SC_SliderHandle:
            event.ignore()
            return
        event.accept()
        new_position = self._pixel_position_to_range_value(self._pick(event.pos()))
        if self._active_slider < 0:
            offset = new_position - self._click_offset
            self._high += offset
            self._low += offset
            if self._low < self.minimum():
                diff = self.minimum() - self._low
                self._low += diff
                self._high += diff
            if self._high > self.maximum():
                diff = self.maximum() - self._high
                self._low += diff
                self._high += diff
        elif self._active_slider == 0:
            if new_position >= self._high:
                new_position = self._high - 1
            self._low = new_position
        else:
            if new_position <= self._low:
                new_position = self._low + 1
            self._high = new_position
        self._click_offset = new_position
        self.update()

    def mouseReleaseEvent(self, event: object) -> None:  # noqa: N802, ARG002
        """Handles mouse release to finalize slider position."""
        self.level_change()

    def _pick(self, point: object) -> int:
        """Extracts the relevant coordinate from a point based on orientation.

        Args:
            point: The mouse position point.

        Returns:
            The x or y coordinate depending on slider orientation.
        """
        if self.orientation() == QtCore.Qt.Horizontal:
            return point.x()
        return point.y()

    def _pixel_position_to_range_value(self, position: int) -> int:
        """Converts a pixel position to a slider range value.

        Args:
            position: The pixel position along the slider track.

        Returns:
            The corresponding slider value.
        """
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        style = QApplication.style()

        groove_rect = style.subControlRect(style.CC_Slider, option, style.SC_SliderGroove, self)
        handle_rect = style.subControlRect(style.CC_Slider, option, style.SC_SliderHandle, self)

        if self.orientation() == QtCore.Qt.Horizontal:
            slider_length = handle_rect.width()
            slider_min = groove_rect.x()
            slider_max = groove_rect.right() - slider_length + 1
        else:
            slider_length = handle_rect.height()
            slider_min = groove_rect.y()
            slider_max = groove_rect.bottom() - slider_length + 1

        return style.sliderValueFromPosition(
            self.minimum(), self.maximum(), position - slider_min,
            slider_max - slider_min, option.upsideDown,
        )


class _SaturationSlider(_RangeSlider):
    """Saturation range slider for controlling image display levels.

    Args:
        parent: The visualization window that owns this slider.
    """

    def __init__(self, parent: VisWindow | None = None) -> None:
        super().__init__(parent=parent)
        self._vis_parent = parent
        self.setMinimum(_SLIDER_MIN)
        self.setMaximum(_SLIDER_MAX)
        self.set_low(_SLIDER_LOW_DEFAULT)
        self.set_high(_SLIDER_HIGH_DEFAULT)

    def level_change(self) -> None:
        """Updates image display levels from the slider positions."""
        self._vis_parent.sat[0] = float(self._low) / _SLIDER_PERCENTAGE
        self._vis_parent.sat[1] = float(self._high) / _SLIDER_PERCENTAGE
        self._vis_parent.img.setLevels([self._vis_parent.sat[0], self._vis_parent.sat[1]])
        self._vis_parent.imgROI.setLevels([self._vis_parent.sat[0], self._vis_parent.sat[1]])
        self._vis_parent._plot_widget.show()


class _NeuronSlider(_RangeSlider):
    """Neuron range slider for controlling neuron display range.

    Args:
        parent: The visualization window that owns this slider.
    """

    def __init__(self, parent: VisWindow | None = None) -> None:
        super().__init__(parent=parent)
        self._vis_parent = parent
        self.setMinimum(_SLIDER_MIN)
        self.setMaximum(_SLIDER_MAX)
        self.set_low(_SLIDER_LOW_DEFAULT)
        self.set_high(_SLIDER_HIGH_DEFAULT)

    def level_change(self) -> None:
        """Updates image display levels from the slider positions."""
        self._vis_parent.sat[0] = float(self._low) / _SLIDER_PERCENTAGE
        self._vis_parent.sat[1] = float(self._high) / _SLIDER_PERCENTAGE
        self._vis_parent.img.setLevels([self._vis_parent.sat[0], self._vis_parent.sat[1]])
        self._vis_parent.imgROI.setLevels([self._vis_parent.sat[0], self._vis_parent.sat[1]])
        self._vis_parent._plot_widget.show()


class VisWindow(QMainWindow):
    """Rastermap visualization window for population activity analysis.

    Displays a sorted neural activity matrix with interactive ROI selection,
    time-range zooming, and rastermap-based dimensionality reduction sorting.
    Supports integration with the main GUI for cell selection updates.

    Args:
        parent: The main GUI window containing session data.
    """

    def __init__(self, parent: MainWindow | None = None) -> None:
        super().__init__(parent)
        self._parent = parent
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(70, 70, 1100, 900)
        self.setWindowTitle("Visualize data")
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self._grid_layout = QGridLayout()
        central_widget.setLayout(self._grid_layout)
        self._plot_widget = pg.GraphicsLayoutWidget()
        self._plot_widget.move(600, 0)
        self._plot_widget.resize(1000, 500)
        self._grid_layout.addWidget(self._plot_widget, 0, 0, 14, 14)
        layout = self._plot_widget.ci.layout

        # Full view panel.
        self._overview_view = self._plot_widget.addViewBox(row=0, col=0)
        self._overview_view.setMouseEnabled(x=False, y=False)
        self._overview_view.setMenuEnabled(False)
        self._full_plot = self._plot_widget.addPlot(title="FULL VIEW", row=0, col=1)
        self._full_plot.setMouseEnabled(x=False, y=False)
        self.img = pg.ImageItem(autoDownsample=True)
        self._full_plot.addItem(self.img)

        # Computes activity data based on parent's activity mode.
        if len(self._parent.imerge) == 1:
            is_cell = self._parent.iscell[self._parent.imerge[0]]
            self._cells = np.array((self._parent.iscell == is_cell).nonzero()).flatten()
        else:
            self._cells = np.array(self._parent.imerge).flatten()

        activity_mode = self._parent.activityMode
        if activity_mode == 0:
            spike_data = self._parent.Fcell[self._cells, :]
        elif activity_mode == 1:
            spike_data = self._parent.Fneu[self._cells, :]
        elif activity_mode == _ACTIVITY_MODE_COUNT:
            spike_data = (
                self._parent.Fcell[self._cells, :]
                - _NEUROPIL_COEFFICIENT * self._parent.Fneu[self._cells, :]
            )
        else:
            spike_data = self._parent.Spks[self._cells, :]
        spike_data = np.squeeze(spike_data)
        spike_data = zscore(spike_data, axis=1)
        self._spike_matrix = np.maximum(
            _ZSCORE_LOWER_BOUND, np.minimum(_ZSCORE_UPPER_BOUND, spike_data),
        ) + abs(_ZSCORE_LOWER_BOUND)
        self._spike_matrix /= _ZSCORE_NORMALIZATION
        self._time_sort = np.arange(0, spike_data.shape[1]).astype(np.int32)
        self._bin_size = int(np.maximum(1, int(self._parent.ops["sampling_rate"] / _BIN_SIZE_DIVISOR)))

        # Configures axes for the full view panel.
        self._full_plot.setXRange(0, spike_data.shape[1])
        self._full_plot.setYRange(0, spike_data.shape[0])
        self._full_plot.setLimits(
            xMin=-10, xMax=spike_data.shape[1] + 10,
            yMin=-10, yMax=spike_data.shape[0] + 10,
        )
        self._full_plot.setLabel("left", "neurons")
        self._full_plot.setLabel("bottom", "time")

        # Zoom view panel.
        frame_count = spike_data.shape[1]
        neuron_count = spike_data.shape[0]
        self._selected = np.arange(0, neuron_count, 1, int)
        self._zoom_plot = self._plot_widget.addPlot(title="ZOOM IN", row=1, col=0, colspan=2)
        self.imgROI = pg.ImageItem(autoDownsample=True)
        self._zoom_plot.addItem(self.imgROI)
        self._zoom_plot.setMouseEnabled(x=False, y=False)
        self._zoom_plot.hideAxis("bottom")
        self._behavior_loaded = self._parent.bloaded
        self._trace_plot = self._plot_widget.addPlot(title="", row=2, col=0, colspan=2)
        self._trace_plot.setMouseEnabled(x=False, y=False)
        self._trace_plot.setLabel("bottom", "time")

        # Configures viridis colormap for activity display.
        colormap = cm.get_cmap("gray_r")
        colormap._init()
        lut = (colormap._lut * 255).view(np.ndarray)
        lut = lut[:-_COLORMAP_TRUNCATION, :]
        self.img.setLookupTable(lut)
        self.imgROI.setLookupTable(lut)
        layout.setColumnStretchFactor(1, 3)
        layout.setRowStretchFactor(1, 3)

        # Saturation slider.
        self.sat = [_DEFAULT_SATURATION_LOW, _DEFAULT_SATURATION_HIGH]
        slider = _SaturationSlider(self)
        slider.setTickPosition(QSlider.TicksBelow)
        self._grid_layout.addWidget(slider, 0, 2, 5, 1)
        saturation_label = _VerticalLabel(text="saturation")
        saturation_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.img.setLevels([self.sat[0], self.sat[1]])
        self.imgROI.setLevels([self.sat[0], self.sat[1]])
        self._grid_layout.addWidget(saturation_label, 2, 3, 3, 2)
        self._sort_indices = np.arange(0, self._cells.size).astype(np.int32)

        # Main time-range selection ROI.
        red_pen = pg.mkPen(pg.mkColor(255, 0, 0), width=_ROI_PEN_WIDTH, style=QtCore.Qt.SolidLine)
        self._time_roi = pg.RectROI(
            [frame_count * 0.25, -1],
            [frame_count * 0.25, neuron_count + 1],
            maxBounds=QtCore.QRectF(-1.0, -1.0, frame_count + 1, neuron_count + 1),
            pen=red_pen,
        )
        self._x_range = np.arange(frame_count * 0.25, frame_count * 0.5, 1, int)
        self._time_roi.handleSize = _ROI_HANDLE_SIZE
        self._time_roi.handlePen = red_pen
        self._time_roi.handles = []
        self._time_roi.addScaleHandle([1, 0.5], [0.0, 0.5])
        self._time_roi.addScaleHandle([0.0, 0.5], [1.0, 0.5])
        self._time_roi.sigRegionChangeFinished.connect(self._on_time_roi_changed)
        self._full_plot.addItem(self._time_roi)
        self._time_roi.setZValue(_ROI_Z_VALUE)

        # Neuron range selection ROI.
        self._neuron_roi = pg.RectROI(
            [-1, neuron_count * 0.4],
            [frame_count * 0.25, neuron_count * 0.2],
            maxBounds=QtCore.QRectF(-1, -1.0, frame_count * 0.25, neuron_count + 1),
            pen=red_pen,
        )
        self._selected = np.arange(neuron_count * 0.4, neuron_count * 0.6, 1, int)
        self._neuron_roi.handleSize = _ROI_HANDLE_SIZE
        self._neuron_roi.handlePen = red_pen
        self._neuron_roi.handles = []
        self._neuron_roi.addScaleHandle([0.5, 1], [0.5, 0])
        self._neuron_roi.addScaleHandle([0.5, 0], [0.5, 1])
        self._neuron_roi.sigRegionChangeFinished.connect(self._on_neuron_roi_changed)
        self._zoom_plot.addItem(self._neuron_roi)
        self._neuron_roi.setZValue(_ROI_Z_VALUE)

        # Threshold selection ROI on the trace plot.
        green_pen = pg.mkPen(pg.mkColor(0, 255, 0), width=_ROI_PEN_WIDTH, style=QtCore.Qt.SolidLine)
        self._threshold_roi = pg.RectROI(
            [-0.5, 0],
            [frame_count * 0.25, 1],
            maxBounds=QtCore.QRectF(-1.0, -10.0, frame_count * 0.25, 10),
            pen=green_pen,
        )
        self._threshold_roi.handleSize = _ROI_HANDLE_SIZE
        self._threshold_roi.handlePen = green_pen
        self._threshold_roi.handles = []
        self._threshold_roi.addScaleHandle([0.5, 1], [0.5, 0])
        self._threshold_roi.addScaleHandle([0.5, 0], [0.5, 1])
        self._threshold_roi.sigRegionChangeFinished.connect(self._on_threshold_roi_changed)
        self._threshold_position = -0.5
        self._threshold_size = 1
        self._sorted_spike_matrix = self._spike_matrix
        self._plot_traces()

        self._neural_sorting(2)

        # Control buttons.
        self._compute_button = QPushButton("compute rastermap + PCs")
        self._compute_button.clicked.connect(self._compute_map)
        self._grid_layout.addWidget(self._compute_button, 0, 0, 1, 2)
        self._sorting_combo = QComboBox(self)
        self._grid_layout.addWidget(self._sorting_combo, 1, 0, 1, 2)
        self._grid_layout.addWidget(QLabel("PC 1:"), 2, 0, 1, 2)
        self._select_button = QPushButton("show selected cells in GUI")
        self._select_button.clicked.connect(self._select_cells)
        self._select_button.setEnabled(True)
        self._grid_layout.addWidget(self._select_button, 3, 0, 1, 2)
        self._time_sort_checkbox = QCheckBox("&Time sort")
        self._time_sort_checkbox.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self._time_sort_checkbox.stateChanged.connect(self._sort_time)
        self._grid_layout.addWidget(self._time_sort_checkbox, 4, 0, 1, 2)
        self._grid_layout.addWidget(QLabel(""), 5, 0, 1, 1)
        self._grid_layout.setRowStretch(6, 1)
        self._raster_computed = False

        self._process = QtCore.QProcess(self)
        self._process.readyReadStandardOutput.connect(self._stdout_write)
        self._process.readyReadStandardError.connect(self._stderr_write)
        self._process.finished.connect(self._finished)

        self._plot_widget.show()
        self._plot_widget.scene().sigMouseClicked.connect(self._plot_clicked)
        self.show()

        # State for rastermap computation.
        self._model: Rastermap | None = None
        self._sort_indices_1: np.ndarray | None = None
        self._svd_u: np.ndarray | None = None
        self._svd_v: np.ndarray | None = None
        self._error: bool = False
        self._finish: bool = True
        self._computation_start: float = 0.0
        self._pc_edit: QLineEdit | None = None

    def _plot_clicked(self, event: object) -> None:
        """Handles double-click to reset the time ROI to full range.

        Args:
            event: The mouse click event.
        """
        items = self._plot_widget.scene().items(event.scenePos())
        for item in items:
            if item == self._full_plot and event.button() == 1 and event.double():
                self._time_roi.setPos([-1, -1])
                self._time_roi.setSize([
                    self._spike_matrix.shape[1] + 1,
                    self._spike_matrix.shape[0] + 1,
                ])

    def keyPressEvent(self, event: object) -> None:  # noqa: N802
        """Handles arrow key navigation for time and neuron ROIs."""
        direction = -1
        move = False
        neuron_count, frame_count = self._spike_matrix.shape
        if event.modifiers() != QtCore.Qt.ShiftModifier:
            if event.key() == QtCore.Qt.Key_Down:
                direction = 0
            elif event.key() == QtCore.Qt.Key_Up:
                direction = 1
            elif event.key() == QtCore.Qt.Key_Left:
                direction = 2
            elif event.key() == QtCore.Qt.Key_Right:
                direction = 3
            if direction in (2, 3):
                x_range, _ = self._roi_range(roi=self._time_roi)
                if x_range.size < frame_count:
                    if direction == _DIRECTION_LEFT:
                        move = True
                        x_range = x_range - np.minimum(x_range.min() + 1, frame_count * _MOVEMENT_FRACTION)
                    else:
                        move = True
                        x_range = x_range + np.minimum(
                            frame_count - x_range.max() - 1, frame_count * _MOVEMENT_FRACTION,
                        )
                    if move:
                        self._time_roi.setPos([x_range.min() - 1, -1])
                        self._time_roi.setSize([x_range.size + 1, neuron_count + 1])
            if direction in (0, 1):
                _, y_range = self._roi_range(roi=self._neuron_roi)
                if y_range.size < neuron_count:
                    if direction == 0:
                        move = True
                        y_range = y_range - np.minimum(y_range.min(), neuron_count * _MOVEMENT_FRACTION)
                    else:
                        move = True
                        y_range = y_range + np.minimum(
                            neuron_count - y_range.max() - 1, neuron_count * _MOVEMENT_FRACTION,
                        )
                    if move:
                        self._neuron_roi.setPos([-1, y_range.min()])
                        self._neuron_roi.setSize([self._x_range.size + 1, y_range.size])
        else:
            if event.key() == QtCore.Qt.Key_Down:
                direction = 0
            elif event.key() == QtCore.Qt.Key_Up:
                direction = 1
            elif event.key() == QtCore.Qt.Key_Left:
                direction = 2
            elif event.key() == QtCore.Qt.Key_Right:
                direction = 3
            if direction in (2, 3):
                x_range, _ = self._roi_range(roi=self._time_roi)
                step = frame_count * _MOVEMENT_FRACTION / (frame_count / x_range.size)
                if direction == _DIRECTION_LEFT:
                    if x_range.size > step:
                        move = True
                        new_size = x_range.size - step
                        x_range = x_range.min() + np.arange(0, new_size).astype(np.int32)
                elif x_range.size < frame_count - step + 1:
                    move = True
                    new_size = x_range.size + step
                    x_range = x_range.min() + np.arange(0, new_size).astype(np.int32)
                if move:
                    self._time_roi.setPos([x_range.min() - 1, -1])
                    self._time_roi.setSize([x_range.size + 1, neuron_count + 1])
            elif direction >= 0:
                _, y_range = self._roi_range(roi=self._neuron_roi)
                step = neuron_count * _MOVEMENT_FRACTION / (neuron_count / y_range.size)
                if direction == 0:
                    if y_range.size > step:
                        move = True
                        new_size = y_range.size - step
                        y_range = y_range.min() + np.arange(0, new_size).astype(np.int32)
                elif y_range.size < neuron_count - step + 1:
                    move = True
                    new_size = y_range.size + step
                    y_range = y_range.min() + np.arange(0, new_size).astype(np.int32)
                if move:
                    self._neuron_roi.setPos([-1, y_range.min()])
                    self._neuron_roi.setSize([self._x_range.size + 1, y_range.size])

    def _roi_range(self, roi: object) -> tuple[np.ndarray, np.ndarray]:
        """Computes the integer pixel ranges covered by an ROI.

        Args:
            roi: The pyqtgraph RectROI to extract ranges from.

        Returns:
            Tuple of (x_range, y_range) integer arrays clipped to data bounds.
        """
        position = roi.pos()
        position_y = position.y()
        position_x = position.x()
        size_x, size_y = roi.size()
        x_range = (np.arange(0, int(size_x)) + int(position_x)).astype(np.int32)
        y_range = (np.arange(0, int(size_y)) + int(position_y)).astype(np.int32)
        x_range = x_range[x_range >= 0]
        x_range = x_range[x_range < self._spike_matrix.shape[1]]
        y_range = y_range[y_range >= 0]
        y_range = y_range[y_range < self._spike_matrix.shape[0]]
        return x_range, y_range

    def _plot_traces(self) -> None:
        """Renders the averaged trace for the selected neurons and time range."""
        average = self._sorted_spike_matrix[np.ix_(self._selected, self._x_range)].mean(axis=0)
        average -= average.min()
        average /= average.max()
        self._trace_plot.clear()
        self._trace_plot.plot(self._x_range, average, pen=(255, 0, 0))
        if self._behavior_loaded:
            self._trace_plot.plot(self._parent.beh_time, self._parent.beh, pen="w")
        self._trace_plot.setXRange(self._x_range[0], self._x_range[-1])
        self._trace_plot.addItem(self._threshold_roi)
        self._threshold_roi.setZValue(_ROI_Z_VALUE)

    def _on_neuron_roi_changed(self) -> None:
        """Handles neuron ROI position change to update selected neurons."""
        _, y_range = self._roi_range(roi=self._neuron_roi)
        self._selected = y_range.astype("int")
        self._plot_traces()

    def _on_threshold_roi_changed(self) -> None:
        """Handles threshold ROI position change."""
        position = self._threshold_roi.pos()
        _, size_y = self._threshold_roi.size()
        self._threshold_position = position.y()
        self._threshold_size = size_y

    def _on_time_roi_changed(self) -> None:
        """Handles time ROI position change to update the zoom view."""
        x_range, _ = self._roi_range(roi=self._time_roi)
        self._x_range = x_range
        self.imgROI.setImage(self._sorted_spike_matrix[:, self._x_range])
        self._zoom_plot.setXRange(0, self._x_range.size)

        self._plot_traces()

        # Resets dependent ROIs.
        self._neuron_roi.maxBounds = QtCore.QRectF(
            -1, -1.0, x_range.size + 1, self._spike_matrix.shape[0] + 1,
        )
        self._neuron_roi.setSize([x_range.size + 1, self._selected.size])
        self._neuron_roi.setZValue(_ROI_Z_VALUE)

        self._threshold_roi.maxBounds = QtCore.QRectF(
            self._x_range[0] - 1, -5.0, self._x_range[1] + 1, 10,
        )
        self._threshold_roi.setPos([self._x_range[0] - 1, self._threshold_position])
        self._threshold_roi.setSize([x_range.size + 1, self._threshold_size])
        self._threshold_roi.setZValue(_ROI_Z_VALUE)

        self.imgROI.setLevels([self.sat[0], self.sat[1]])

    def _setup_pc_controls(self, plot: bool) -> None:
        """Creates PC selection controls and computes SVD.

        Args:
            plot: Whether to trigger a neural sorting plot after setup.
        """
        self._pc_edit = QLineEdit(self)
        self._pc_edit.setValidator(
            QtGui.QIntValidator(1, np.minimum(self._spike_matrix.shape[0], self._spike_matrix.shape[1])),
        )
        self._pc_edit.setText("1")
        self._pc_edit.setFixedWidth(60)
        self._pc_edit.setAlignment(QtCore.Qt.AlignRight)
        pc_label = QLabel("PC: ")
        pc_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self._grid_layout.addWidget(pc_label, 3, 0, 1, 1)
        self._grid_layout.addWidget(self._pc_edit, 3, 1, 1, 1)
        self._sorting_combo.addItem("PC")
        self._pc_edit.returnPressed.connect(self._on_pc_return)
        self._compute_svd(bin_size=self._bin_size)
        self._sorting_combo.currentIndexChanged.connect(self._neural_sorting)
        if plot:
            self._neural_sorting(0)

    def _on_pc_return(self) -> None:
        """Handles PC edit field return press to resort by the selected PC."""
        self._sorting_combo.setCurrentIndex(0)
        self._neural_sorting(0)

    def _activate(self) -> None:
        """Activates rastermap controls after successful computation."""
        self._pc_edit = QLineEdit(self)
        self._pc_edit.setValidator(
            QtGui.QIntValidator(1, np.minimum(self._spike_matrix.shape[0], self._spike_matrix.shape[1])),
        )
        self._pc_edit.setText("1")
        self._pc_edit.setFixedWidth(60)
        self._pc_edit.setAlignment(QtCore.Qt.AlignRight)
        pc_label = QLabel("PC: ")
        pc_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self._grid_layout.addWidget(pc_label, 2, 0, 1, 1)
        self._grid_layout.addWidget(self._pc_edit, 2, 1, 1, 1)
        self._sorting_combo.addItem("PC")
        self._pc_edit.returnPressed.connect(self._on_pc_return)

        self._sort_indices_1 = np.argsort(self._model.embedding[:, 0])
        self._svd_u = self._model.Usv
        self._svd_v = self._model.Vsv
        self._sorting_combo.addItem("rastermap")

        self._raster_computed = True
        cell_count = len(self._parent.stat)
        self._parent.isort = -1 * np.ones((cell_count,), dtype=np.int64)
        selected_count = len(self._cells)
        sorting_map = np.zeros(selected_count)
        sorting_map[self._sort_indices_1] = np.arange(selected_count).astype("int")
        self._parent.isort[self._cells] = sorting_map

        rastermap_masks(self._parent)
        last_button_index = len(self._parent.color_names) - 1
        self._parent.colorbtns.button(last_button_index).setEnabled(True)
        self._parent.colorbtns.button(last_button_index).setStyleSheet(BUTTON_UNPRESSED_STYLESHEET)
        self._parent.rastermap = True

        self._sorting_combo.setCurrentIndex(1)
        self._sorting_combo.currentIndexChanged.connect(self._neural_sorting)
        self._neural_sorting(1)
        self._compute_button.setEnabled(False)
        self._time_sort_checkbox.setChecked(False)

    def _compute_map(self) -> None:
        """Runs rastermap dimensionality reduction on the spike matrix."""
        self._error = False
        self._finish = True
        self._compute_button.setEnabled(False)
        self._computation_start = time.time()
        try:
            self._model = Rastermap()
            self._model.fit(self._spike_matrix)
            self._activate()
        except Exception as exception:
            console.echo(
                message="Rastermap issue: Interrupted by error (not finished)",
                level=LogLevel.ERROR,
            )
            console.echo(message=f"Error details: {exception}", level=LogLevel.ERROR)

    def _finished(self) -> None:
        """Handles process completion after rastermap computation."""
        if self._finish and not self._error:
            console.echo(
                message=f"Raster map computed in {time.time() - self._computation_start:.2f} s",
                level=LogLevel.SUCCESS,
            )
            self._activate()
        else:
            sys.stdout.write("Interrupted by error (not finished)\n")

    def _stdout_write(self) -> None:
        """Writes process stdout to the console."""
        output = str(self._process.readAllStandardOutput(), "utf-8")
        sys.stdout.write(output)

    def _stderr_write(self) -> None:
        """Writes process stderr to the console and flags the error."""
        sys.stdout.write(">>>ERROR<<<\n")
        output = str(self._process.readAllStandardError(), "utf-8")
        sys.stdout.write(output)
        self._error = True
        self._finish = False

    def _select_cells(self) -> None:
        """Sends the currently selected neurons back to the main GUI."""
        self._parent.imerge = []
        if self._selected.size < _MAX_SELECTED_CELLS:
            for neuron_index in self._selected:
                self._parent.imerge.append(self._cells[self._sort_indices[neuron_index]])
            self._parent.ichosen = self._parent.imerge[0]
            self._parent.update_plot()
        else:
            console.echo(
                message=f"Too many cells selected (maximum: {_MAX_SELECTED_CELLS})",
                level=LogLevel.WARNING,
            )

    def _sort_time(self) -> None:
        """Toggles time-domain sorting of the spike matrix columns."""
        if self._raster_computed:
            if self._time_sort_checkbox.isChecked():
                if not hasattr(self, "_sort_indices_2"):
                    self._model = Rastermap()
                    self._model.fit(
                        self._spike_matrix.T, Usv=self._svd_v, Vsv=self._svd_u,
                    )
                    self._sort_indices_2 = np.argsort(self._model.embedding[:, 0])
                self._time_sort = self._sort_indices_2.astype(np.int32)
            else:
                self._time_sort = np.arange(0, self._spike_matrix.shape[1]).astype(np.int32)
            self._neural_sorting(self._sorting_combo.currentIndex())

    def _neural_sorting(self, sort_index: int) -> None:
        """Sorts the spike matrix by the selected method and updates the display.

        Args:
            sort_index: Index of the sorting method (0=PC, 1=rastermap, 2+=none).
        """
        if sort_index == 0:
            self._sort_indices = np.argsort(self._svd_u[:, int(self._pc_edit.text()) - 1])
        elif sort_index == 1:
            self._sort_indices = self._sort_indices_1
        if sort_index < _ACTIVITY_MODE_COUNT:
            smoothing_sigma = np.minimum(
                _MAX_SMOOTHING_SIGMA,
                np.maximum(_MIN_SMOOTHING_SIGMA, int(self._spike_matrix.shape[0] * _SMOOTHING_FRACTION)),
            )
            self._sorted_spike_matrix = gaussian_filter1d(
                self._spike_matrix[np.ix_(self._sort_indices, self._time_sort)].T,
                smoothing_sigma,
                axis=1,
            )
            self._sorted_spike_matrix = self._sorted_spike_matrix.T
        else:
            self._sorted_spike_matrix = self._spike_matrix
        self._sorted_spike_matrix = zscore(self._sorted_spike_matrix, axis=1)
        self._sorted_spike_matrix = np.minimum(_ZSCORE_UPPER_BOUND, self._sorted_spike_matrix)
        self._sorted_spike_matrix = np.maximum(
            _ZSCORE_LOWER_BOUND, self._sorted_spike_matrix,
        ) + abs(_ZSCORE_LOWER_BOUND)
        self._sorted_spike_matrix /= _ZSCORE_NORMALIZATION
        self.img.setImage(self._sorted_spike_matrix)
        self._on_time_roi_changed()
