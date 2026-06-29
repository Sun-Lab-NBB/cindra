"""Provides the principal component metrics viewer window for registration quality evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6 import QtGui, QtCore
import pyqtgraph as pg  # type: ignore[import-untyped]
from PySide6.QtWidgets import (
    QLabel,
    QWidget,
    QComboBox,
    QLineEdit,
    QGridLayout,
    QHBoxLayout,
    QMainWindow,
)
from ataraxis_base_utilities import LogLevel, console

from .styles import FONTS, STYLE, COLORS, PC_STYLE, PLOT_STYLE
from .widgets import configure_plot, add_plot_legend, escape_returns_focus, create_play_pause_group
from .constants import PC_CONFIG, COMMON_CONFIG

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .viewer_context import SingleRecordingData


class PCViewer(QMainWindow):
    """Displays a UI window for viewing the principal component registration metrics.

    Args:
        data: Pre-loaded registration data to display on startup.

    Attributes:
        data: The SingleRecordingData instance that stores the visualized recording's data.
        _loaded: Determines whether PC data has been loaded and is ready for display.
        _current_frame: Animation toggle state for PC extreme image cycling.
        _pc_count: Number of principal components available.
        _pc_images: PC extreme images array with shape (2, num_pcs, height, width), or None.
        _image_height: Height of PC images in pixels.
        _image_width: Width of PC images in pixels.
        _pc_metrics: Registration offset metrics array with shape (num_pcs, 3), or None.
        _pc_projections: Per-frame PC projection array with shape (num_frames, num_pcs), or None.
        _central_widget: Central widget container.
        _layout: Grid layout for arranging all controls and views.
        _graphics_widget: PyQtGraph graphics layout for image and plot views.
        _metrics_plot: Plot widget for PC offset metrics.
        _difference_view_box: View box for the PC difference image.
        _merged_view_box: View box for the merged PC overlay image.
        _animated_view_box: View box for the animated PC extreme image.
        _difference_image: Image item for the PC difference display.
        _merged_image: Image item for the merged PC overlay display.
        _animated_image: Image item for the animated PC extreme display.
        _projection_plot: Plot widget for the PC time-course projection.
        _plane_selector: Dropdown for selecting the imaging plane.
        _pc_edit: Input field for the current principal component number.
        _metric_labels: Labels displaying per-PC registration offset values.
        _title_labels: Text items anchored inside each image view box, positioned at the bottom center.
        _play_button: Button to start PC animation playback.
        _pause_button: Button to pause PC animation playback.
        _update_timer: Timer driving the PC animation.
        _metrics_scatter: Scatter plot overlay indicating the selected PC on the metrics plot, or None.
        _legend: Legend item for the metrics plot, or None.
        _metrics_y_range: Global y-axis range for the metrics plot, computed across all PCs.
        _projection_y_range: Per-plane y-axis range for the projection plot, computed across all PCs.
    """

    def __init__(self, data: SingleRecordingData) -> None:
        # Initializes the main viewer window.
        super().__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")
        self.setGeometry(*PC_STYLE.window_geometry)
        self.setWindowTitle("Registration Quality Metrics")
        self._central_widget: QWidget = QWidget(self)
        self.setCentralWidget(self._central_widget)
        self._layout: QGridLayout = QGridLayout()
        self._central_widget.setLayout(self._layout)

        # Initializes state and data.
        self.data: SingleRecordingData = data
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

        # Row 0: Toolbar with plane selector.
        toolbar = QHBoxLayout()
        plane_label = QLabel("Plane:")
        plane_label.setStyleSheet(STYLE.white_label)
        self._plane_selector: QComboBox = QComboBox(self)
        self._plane_selector.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._plane_selector.setEnabled(False)
        self._plane_selector.setToolTip("Select the imaging plane.")
        self._plane_selector.currentIndexChanged.connect(self._on_plane_changed)
        toolbar.addWidget(plane_label)
        toolbar.addWidget(self._plane_selector)
        hint_label = QLabel(
            "Hint: Use arrows to navigate planes / PCs, use space to toggle top / bottom PC image cycling."
        )
        hint_label.setStyleSheet(STYLE.white_label)
        hint_label.setFont(FONTS.small_bold)
        toolbar.addWidget(hint_label)
        toolbar.addStretch()
        self._layout.addLayout(toolbar, 0, 0, 1, 1)

        # Row 1: Graphics widget spans the full width. Row stretch gives it all available vertical space.
        self._graphics_widget: pg.GraphicsLayoutWidget = pg.GraphicsLayoutWidget()
        self._layout.addWidget(self._graphics_widget, 1, 0, 1, 1)
        self._layout.setRowStretch(1, 1)
        self._layout.setRowStretch(2, 0)

        # Configures pixel offset metrics plot. Top content margin provides space for the legend row.
        self._metrics_plot = self._graphics_widget.addPlot(row=0, col=0)
        configure_plot(
            self._metrics_plot,
            mouse_x=False,
            mouse_y=False,
            title="PC Registration Offsets",
            left_label="Offset (px)",
            bottom_label="PC #",
        )

        self._difference_view_box = self._graphics_widget.addViewBox(
            name="plot1",
            lockAspect=True,
            row=1,
            col=0,
            invertY=True,
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

        # Title labels anchored inside each image ViewBox. Their position is set in _zoom_plot() and
        # they serve as content anchors that stabilize the ViewBox bounds during animation.
        self._title_labels: list[pg.TextItem] = []
        for view_box in (self._difference_view_box, self._merged_view_box, self._animated_view_box):
            label = pg.TextItem("", color=COLORS.white, anchor=(0.5, 0))
            label.setFont(FONTS.small)
            view_box.addItem(label)
            self._title_labels.append(label)

        self._graphics_widget.scene().sigMouseClicked.connect(self._plot_clicked)

        self._projection_plot = self._graphics_widget.addPlot(row=0, col=1, colspan=2)
        configure_plot(
            self._projection_plot,
            mouse_x=False,
            mouse_y=True,
            title="PC Projection Weight",
            left_label="Magnitude",
            bottom_label="Sampled Frame",
        )

        # Bottom control panel: PC selector, metric labels, title labels, playback controls.
        self._create_bottom_panel()
        self._pc_edit.setValidator(QtGui.QIntValidator(1, self._pc_count))
        self._update_timer: QtCore.QTimer = QtCore.QTimer()
        self._update_timer.timeout.connect(self._next_frame)

        self.load_data(data=data)

    def load_data(self, data: SingleRecordingData) -> None:
        """Loads principal component registration data from the SingleRecordingData instance.

        Populates the plane selector from the recording's view labels and switches to the appropriate plane before
        loading PC data.

        Args:
            data: The SingleRecordingData instance that stores the visualized recording's data.
        """
        self.data = data

        # Populates the plane selector without triggering _on_plane_changed yet.
        self._plane_selector.blockSignals(True)
        self._plane_selector.clear()
        for label in data.view_labels[1:]:
            self._plane_selector.addItem(label)
        self._plane_selector.setCurrentIndex(max(0, data.view_index))
        self._plane_selector.blockSignals(False)
        self._plane_selector.setEnabled(data.plane_count > 1)

        # Ensures the data points at a valid per-plane view for PC metric access.
        data.switch_view(view_index=max(0, data.view_index))

        self._reload_pc_data()

    def get_state(self) -> dict[str, Any]:
        """Returns the current display state of the PC viewer for cross-process state exchange.

        Returns:
            A dictionary containing the current plane, principal component, and animation status.
        """
        current_pc = int(self._pc_edit.text()) if self._pc_edit.text() else 1
        return {
            "current_plane": self._plane_selector.currentIndex(),
            "current_plane_label": self._plane_selector.currentText(),
            "plane_count": self.data.plane_count,
            "current_pc": current_pc,
            "pc_count": self._pc_count,
            "playing": self._update_timer.isActive(),
            "loaded": self._loaded,
        }

    def _on_plane_changed(self, index: int) -> None:
        """Handles plane selector index changes by switching to the selected plane.

        Args:
            index: The index of the recording's plane to switch to.
        """
        if index < 0:
            return
        self.data.switch_view(view_index=index)
        self._reload_pc_data()

    def _reload_pc_data(self) -> None:
        """Loads and renders PC registration data for the currently selected plane."""
        # Updates the window title to reflect the loaded recording path.
        self.setWindowTitle(f"Registration Quality Metrics — {self.data.recording_label}")
        pc_images = self.data.principal_component_extreme_images
        pc_metrics = self.data.principal_component_shift_metrics
        pc_projections = self.data.principal_component_projections

        # Aborts if the recording has no PC registration metrics (e.g. registration was skipped).
        if pc_images is None or pc_metrics is None:
            console.echo(message="No principal component data available for this plane.", level=LogLevel.WARNING)
            return

        # Clips extreme pixel values to the 1st-99th percentile range for stable image display.
        self._pc_images = np.clip(
            pc_images,
            np.percentile(pc_images, COMMON_CONFIG.lower_percentile),
            np.percentile(pc_images, COMMON_CONFIG.upper_percentile),
        )
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
        metrics_max += (metrics_max - metrics_min) * PLOT_STYLE.legend_headroom
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
            # Up/down arrow keys cycle through imaging planes.
            elif event.key() == QtCore.Qt.Key.Key_Up:
                index = self._plane_selector.currentIndex()
                if index > 0:
                    self._plane_selector.setCurrentIndex(index - 1)
            elif event.key() == QtCore.Qt.Key.Key_Down:
                index = self._plane_selector.currentIndex()
                if index < self._plane_selector.count() - 1:
                    self._plane_selector.setCurrentIndex(index + 1)
            # Spacebar toggles between play and pause for the PC extreme image animation.
            elif event.key() == QtCore.Qt.Key.Key_Space:
                if self._play_button.isEnabled():
                    self._play_button.setChecked(True)
                    self._start_animation()
                else:
                    self._pause_animation()

    def eventFilter(self, source: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        """Returns focus to the main window when Escape is pressed inside an edit field.

        Notes:
            Overrides the Qt virtual method. The camelCase name is required to match the parent signature.
        """
        if escape_returns_focus(self, event):
            return True
        return super().eventFilter(source, event)

    def _create_bottom_panel(self) -> None:
        """Creates the bottom control panel with the PC selector, metric labels, and playback controls.

        Widgets keep their natural size; only the trailing stretch grows when the window is resized.
        Fixed spacing separates each logical group.
        """
        bold_font = FONTS.large_bold
        big_font = FONTS.large
        panel = QHBoxLayout()
        group_spacing = PC_STYLE.group_spacing

        # PC selector: label and input field for the current principal component number.
        pc_label = QLabel("PC:")
        pc_label.setFont(bold_font)
        pc_label.setStyleSheet(STYLE.white_label)
        self._pc_edit: QLineEdit = QLineEdit(self)
        self._pc_edit.setText("1")
        self._pc_edit.setFixedWidth(STYLE.edit_width)
        self._pc_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._pc_edit.setFont(big_font)
        self._pc_edit.setToolTip("Principal component number.")
        self._pc_edit.returnPressed.connect(self._plot_frame)
        self._pc_edit.returnPressed.connect(self.setFocus)
        self._pc_edit.textEdited.connect(self._pause_animation)
        self._pc_edit.installEventFilter(self)
        panel.addWidget(pc_label)
        panel.addWidget(self._pc_edit)
        panel.addSpacing(group_spacing)

        # Metric value labels showing per-PC registration offset magnitudes.
        self._metric_labels: list[QLabel] = []
        for _ in range(3):
            metric_label = QLabel("")
            metric_label.setStyleSheet(STYLE.white_label)
            panel.addWidget(metric_label)
            self._metric_labels.append(metric_label)
        panel.addSpacing(group_spacing)

        # Playback controls.
        playback = create_play_pause_group(
            self,
            play_tooltip="Start automatic PC cycling.",
            pause_tooltip="Stop automatic PC cycling.",
            no_focus=True,
        )
        self._play_button = playback.play_button
        self._pause_button = playback.pause_button
        self._play_button.clicked.connect(self._start_animation)
        self._pause_button.clicked.connect(self._pause_animation)

        panel.addWidget(self._play_button)
        panel.addWidget(self._pause_button)

        # Trailing stretch absorbs extra horizontal space so widgets stay at their natural size.
        panel.addStretch()

        self._layout.addLayout(panel, 2, 0)

    def _start_animation(self) -> None:
        """Starts PC animation playback."""
        if self._loaded:
            self._play_button.setEnabled(False)
            self._pause_button.setEnabled(True)
            self._update_timer.start(PC_CONFIG.animation_interval_milliseconds)

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

        # Metrics plot: shows rigid, nonrigid, and nonrigid-max offset magnitudes across all PCs.
        # The legend is recreated on every update because clear() removes it from the plot.
        self._metrics_plot.clear()
        self._metrics_plot.disableAutoRange()
        colors = (COLORS.cyan, COLORS.red, COLORS.magenta)
        metric_names = ["rigid", "nonrigid", "nonrigid max"]
        self._legend = add_plot_legend(self._metrics_plot, column_count=PC_STYLE.legend_column_count)
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
            size=PLOT_STYLE.scatter_point_size,
            brush=pg.mkBrush(*COLORS.white),
        )
        self._metrics_plot.setXRange(1, self._pc_count, padding=0.0)
        self._metrics_plot.setYRange(*self._metrics_y_range, padding=0.0)

        # Projection plot: shows the per-frame projection onto the selected PC over time.
        self._projection_plot.clear()
        self._projection_plot.plot(self._pc_projections[:, pc_index])
        self._projection_plot.setXRange(0, self._pc_projections.shape[0] - 1)
        self._projection_plot.setYRange(*self._projection_y_range)

        self.show()
        self._zoom_plot()
        # Defers a second zoom call so the ViewBoxes recalculate after Qt finalizes the window geometry.
        QtCore.QTimer.singleShot(0, self._zoom_plot)

    def _zoom_plot(self) -> None:
        """Resets all PC image view ranges to fit the full image extent plus a title gutter."""
        gutter = self._image_height * PC_STYLE.title_gutter_fraction
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
            items = self._graphics_widget.scene().items(event.scenePos())  # type: ignore[attr-defined]
            for item in items:
                if (
                    item in (self._difference_view_box, self._merged_view_box, self._animated_view_box)
                    and event.button() == 1  # type: ignore[attr-defined]
                    and event.double()  # type: ignore[attr-defined]
                ):
                    self._zoom_plot()
