"""Provides the principal component metrics viewer window for registration quality evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
from PySide6 import QtGui, QtCore
import pyqtgraph as pg  # type: ignore[import-untyped]
from PySide6.QtWidgets import (
    QLabel,
    QStyle,
    QWidget,
    QLineEdit,
    QGridLayout,
    QHBoxLayout,
    QMainWindow,
    QToolButton,
    QButtonGroup,
)
from ataraxis_base_utilities import LogLevel, console

from .single_day_context import RegistrationViewerData

if TYPE_CHECKING:
    from numpy.typing import NDArray


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

    legend_label_size: str = "12pt"
    """Font size for legend entry labels on the metrics plot."""

    legend_headroom: float = 0.25
    """Fraction of the y-axis data range added as top padding so legends never overlap traces."""

    axis_fixed_width: int = 60
    """Fixed pixel width for the y-axis so the plot area stays stable when tick label digit counts change."""

    title_gutter_fraction: float = 0.08
    """Fraction of image height added as black space below the image for title labels."""


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
        _projection_y_range: Per-plane y-axis range for the projection plot, computed across all PCs.
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

        # Configures pixel shift metrics plot. Top content margin provides space for the legend row.
        # noinspection PyUnresolvedReferences
        self._metrics_plot = self._graphics_widget.addPlot(row=0, col=0)
        self._metrics_plot.setMouseEnabled(x=False, y=False)
        self._metrics_plot.setMenuEnabled(False)
        self._metrics_plot.getAxis("left").setWidth(self._style.axis_fixed_width)

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

        # Title labels anchored inside each image ViewBox. Their position is set in _zoom_plot() and
        # they serve as content anchors that stabilize the ViewBox bounds during animation.
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
        self._projection_plot.getAxis("left").setWidth(self._style.axis_fixed_width)

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

        # Pre-computes per-plane y-ranges so axes stay stable when cycling through PCs.
        metrics_min, metrics_max = float(self._pc_metrics.min()), float(self._pc_metrics.max())
        metrics_max += (metrics_max - metrics_min) * self._style.legend_headroom
        self._metrics_y_range = (metrics_min, metrics_max)
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
        bold_font = QtGui.QFont(self._style.font_family, self._style.metrics_font_size, QtGui.QFont.Weight.Bold.value)
        big_font = QtGui.QFont(self._style.font_family, self._style.metrics_font_size)
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
        # The legend is recreated on every update because clear() removes it from the plot.
        self._metrics_plot.clear()
        self._metrics_plot.disableAutoRange()
        colors = [(200, 200, 255), (255, 100, 100), (100, 50, 200)]
        metric_names = ["rigid", "nonrigid", "nonrigid max"]
        self._legend = self._metrics_plot.addLegend(
            horSpacing=20, colCount=3, offset=(-10, 1), labelTextSize=self._style.legend_label_size
        )
        for index in range(3):
            curve = self._metrics_plot.plot(
                np.arange(1, self._pc_count + 1, dtype=np.int32), self._pc_metrics[:, index], pen=colors[index]
            )
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
        self._metrics_plot.setTitle("PC Registration Shifts", size=self._style.plot_title_size, bold=True)
        self._metrics_plot.setLabel("left", "Shift (px)", **{"font-size": self._style.axis_label_size})
        self._metrics_plot.setLabel("bottom", "PC #", **{"font-size": self._style.axis_label_size})
        self._metrics_plot.setXRange(1, self._pc_count, padding=0.0)
        self._metrics_plot.setYRange(*self._metrics_y_range, padding=0.0)

        # Projection plot: shows the per-frame projection onto the selected PC over time.
        self._projection_plot.clear()
        self._projection_plot.plot(self._pc_projections[:, pc_index])
        self._projection_plot.setTitle("PC Projection Weight", size=self._style.plot_title_size, bold=True)
        self._projection_plot.setLabel("left", "Magnitude", **{"font-size": self._style.axis_label_size})
        self._projection_plot.setLabel("bottom", "Sampled Frame", **{"font-size": self._style.axis_label_size})
        self._projection_plot.setXRange(0, self._pc_projections.shape[0] - 1)
        self._projection_plot.setYRange(*self._projection_y_range)

        self.show()
        self._zoom_plot()

    def _zoom_plot(self) -> None:
        """Resets all PC image view ranges to fit the full image extent plus a title gutter."""
        gutter = self._image_height * self._style.title_gutter_fraction
        y_max = self._image_height + gutter
        self._difference_view_box.setXRange(0, self._image_width)
        self._difference_view_box.setYRange(0, y_max)
        self._merged_view_box.setXRange(0, self._image_width)
        self._merged_view_box.setYRange(0, y_max)
        self._animated_view_box.setXRange(0, self._image_width)
        self._animated_view_box.setYRange(0, y_max)

        # Positions title labels in the gutter below the image.
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
