"""Provides the multi-day tracking quality viewer window."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
from PySide6 import QtGui, QtCore
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QLabel,
    QSlider,
    QWidget,
    QComboBox,
    QGroupBox,
    QStatusBar,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QButtonGroup,
)

from .context_data import MaskLayer, BackgroundImage, CoordinateSpace, TrackingViewerData

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...dataclasses import ROIStatistics


@dataclass(frozen=True, slots=True)
class _TrackingViewerStyle:
    """Encapsulates visual and behavioral constants for the TrackingViewer window."""

    cycle_interval: int = 500
    """The millisecond interval for auto-cycling between recordings."""

    default_mask_opacity: int = 127
    """The default mask overlay opacity (0-255 uint8 range)."""

    channel_1_mask_color: tuple[int, int, int] = (0, 255, 255)
    """The mask outline color for channel 1 ROIs (cyan)."""

    channel_2_mask_color: tuple[int, int, int] = (255, 80, 80)
    """The mask outline color for channel 2 ROIs (red)."""

    lower_percentile: float = 1.0
    """The lower percentile value for normalizing background images."""

    upper_percentile: float = 99.0
    """The upper percentile value for normalizing background images."""


class TrackingViewer(QMainWindow):
    """Displays a UI window for viewing the quality of across-day ROI tracking.

    Displays background images with ROI mask overlays for each recording in a multi-day dataset. Supports manual and
    automatic recording cycling, coordinate space switching, mask layer selection, channel toggling, and mask opacity
    control.

    Args:
        data: The preloaded tracking data to display on startup.
    """

    _style: _TrackingViewerStyle = _TrackingViewerStyle()
    """Frozen style constants for the tracking viewer window."""

    def __init__(self, data: TrackingViewerData) -> None:
        super().__init__()
        self.setWindowTitle("Multi-Day Tracking Viewer")
        self.resize(1200, 800)

        self.data: TrackingViewerData = data
        self._auto_cycle_timer: QtCore.QTimer = QtCore.QTimer(self)
        self._auto_cycle_timer.timeout.connect(self._advance_recording)

        # Builds the UI layout.
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Image display panel (pyqtgraph).
        self._graphics_widget = pg.GraphicsLayoutWidget()
        # noinspection PyUnresolvedReferences
        self._view_box: pg.ViewBox = self._graphics_widget.addViewBox(row=0, col=0)
        self._view_box.setAspectLocked(True)
        self._view_box.invertY(True)
        self._image_item: pg.ImageItem = pg.ImageItem()
        self._view_box.addItem(self._image_item)
        main_layout.addWidget(self._graphics_widget, stretch=3)

        # Control panel (right sidebar).
        control_panel = self._build_control_panel()
        main_layout.addWidget(control_panel, stretch=1)

        # Status bar.
        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)

        # Populates the UI with the initial data.
        self.load_data(data=data)

    def load_data(self, data: TrackingViewerData) -> None:
        """Caches the input TrackingViewerData instance and uses it to populate the managed UI window.

        Args:
            data: The TrackingViewerData instance that stores the visualized dataset's data.
        """
        self.data = data

        # Populates the recording selector.
        self._recording_combo.blockSignals(True)
        self._recording_combo.clear()
        for index, recording_id in enumerate(data.recording_ids):
            self._recording_combo.addItem(f"{index}: {recording_id}", userData=index)
        self._recording_combo.setCurrentIndex(0)
        self._recording_combo.blockSignals(False)

        # Enables channel 2 toggle if available.
        self._channel_2_checkbox.setEnabled(data.has_channel_2)

        self._refresh_display()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        """Handles keyboard navigation for recording stepping and auto-cycle control.

        Notes:
            Overrides the Qt virtual method. The camelCase name is required to match the parent signature.
        """
        # Left/right arrow keys step through recordings when auto-cycling is stopped.
        if self._start_button.isEnabled():
            if event.key() == QtCore.Qt.Key.Key_Left:
                self._previous_recording()
            elif event.key() == QtCore.Qt.Key.Key_Right:
                self._next_recording()

        # Spacebar toggles between start and stop.
        if event.key() == QtCore.Qt.Key.Key_Space:
            if self._start_button.isEnabled():
                self._start_cycling()
            else:
                self._stop_cycling()

    def _build_control_panel(self) -> QWidget:
        """Constructs the right-side control panel with all viewer controls.

        Returns:
            The assembled control panel widget.
        """
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        # Recording navigation group.
        recording_group = QGroupBox("Recording Navigation")
        recording_layout = QVBoxLayout(recording_group)

        self._recording_combo = QComboBox()
        self._recording_combo.currentIndexChanged.connect(self._on_recording_selected)
        recording_layout.addWidget(self._recording_combo)

        # Start and stop are grouped exclusively so only one can be active at a time.
        navigation_row = QHBoxLayout()
        self._start_button = QPushButton("Start")
        self._start_button.setCheckable(True)
        self._start_button.setToolTip("Start auto-cycling (Space).")
        self._start_button.clicked.connect(self._start_cycling)

        self._stop_button = QPushButton("Stop")
        self._stop_button.setCheckable(True)
        self._stop_button.setToolTip("Stop auto-cycling (Space). Use Left/Right arrow keys to step through recordings.")
        self._stop_button.clicked.connect(self._stop_cycling)

        button_group = QButtonGroup(self)
        button_group.addButton(self._start_button, 0)
        button_group.addButton(self._stop_button, 1)
        button_group.setExclusive(True)

        navigation_row.addWidget(self._start_button)
        navigation_row.addWidget(self._stop_button)

        # Controls start with stop pre-selected, since there is no active cycling on startup.
        self._start_button.setEnabled(True)
        self._stop_button.setEnabled(False)
        self._stop_button.setChecked(True)
        recording_layout.addLayout(navigation_row)

        layout.addWidget(recording_group)

        # Background image group.
        background_group = QGroupBox("Background Image")
        background_layout = QVBoxLayout(background_group)

        self._background_combo = QComboBox()
        self._background_combo.addItem("Mean Image", userData=BackgroundImage.MEAN)
        self._background_combo.addItem("Enhanced Mean", userData=BackgroundImage.ENHANCED_MEAN)
        self._background_combo.addItem("Max Projection", userData=BackgroundImage.MAX_PROJECTION)
        self._background_combo.addItem("Correlation Map", userData=BackgroundImage.CORRELATION_MAP)
        self._background_combo.currentIndexChanged.connect(self._refresh_display)
        background_layout.addWidget(self._background_combo)

        layout.addWidget(background_group)

        # Coordinate space group.
        space_group = QGroupBox("Coordinate Space")
        space_layout = QVBoxLayout(space_group)

        self._space_combo = QComboBox()
        self._space_combo.addItem("Native", userData=CoordinateSpace.NATIVE)
        self._space_combo.addItem("Transformed", userData=CoordinateSpace.TRANSFORMED)
        self._space_combo.currentIndexChanged.connect(self._refresh_display)
        space_layout.addWidget(self._space_combo)

        layout.addWidget(space_group)

        # Mask layer group.
        mask_group = QGroupBox("Mask Layer")
        mask_layout = QVBoxLayout(mask_group)

        self._mask_combo = QComboBox()
        self._mask_combo.addItem("Original", userData=MaskLayer.ORIGINAL)
        self._mask_combo.addItem("Deformed", userData=MaskLayer.DEFORMED)
        self._mask_combo.addItem("Template", userData=MaskLayer.TEMPLATE)
        self._mask_combo.currentIndexChanged.connect(self._refresh_display)
        mask_layout.addWidget(self._mask_combo)

        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Opacity:"))
        self._opacity_slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 255)
        self._opacity_slider.setValue(self._style.default_mask_opacity)
        self._opacity_slider.valueChanged.connect(self._refresh_display)
        opacity_row.addWidget(self._opacity_slider)
        mask_layout.addLayout(opacity_row)

        layout.addWidget(mask_group)

        # Channel group.
        channel_group = QGroupBox("Channel")
        channel_layout = QVBoxLayout(channel_group)

        self._channel_2_checkbox = QPushButton("Channel 2")
        self._channel_2_checkbox.setCheckable(True)
        self._channel_2_checkbox.setEnabled(False)
        self._channel_2_checkbox.toggled.connect(self._refresh_display)
        channel_layout.addWidget(self._channel_2_checkbox)

        layout.addWidget(channel_group)

        layout.addStretch()
        return panel

    def _refresh_display(self) -> None:
        """Redraws the image panel with the current background image and mask overlay."""
        coordinate_space = self._space_combo.currentData()
        background_type = self._background_combo.currentData()
        mask_layer = self._mask_combo.currentData()
        channel_2 = self._channel_2_checkbox.isChecked()
        opacity = self._opacity_slider.value()

        # Retrieves and normalizes the background image.
        background = self.data.background_image(
            image_type=background_type,
            coordinate_space=coordinate_space,
            channel_2=channel_2,
        )
        display_image = self._normalize_image(image=background)

        # Retrieves the mask set for the current layer and channel.
        masks = self.data.masks_for_layer(layer=mask_layer, channel_2=channel_2)

        # Composites the mask overlay onto the background.
        if masks:
            mask_color = self._style.channel_2_mask_color if channel_2 else self._style.channel_1_mask_color
            display_image = self._overlay_masks(
                background=display_image,
                masks=masks,
                color=mask_color,
                opacity=opacity,
            )

        self._image_item.setImage(display_image)

        # Updates the status bar.
        recording_id = self.data.current_recording_id
        mask_count = len(masks) if masks else 0
        self._status_bar.showMessage(
            f"Recording: {recording_id}  |  Masks: {mask_count}  |  "
            f"Size: {self.data.frame_height} x {self.data.frame_width}"
        )

    def _on_recording_selected(self, index: int) -> None:
        """Handles recording combo box selection changes.

        Args:
            index: The newly selected combo box index.
        """
        if index < 0:
            return
        self.data.switch_recording(recording_index=index)
        self._refresh_display()

    def _previous_recording(self) -> None:
        """Navigates to the previous recording, wrapping around to the last."""
        new_index = (self.data.current_recording_index - 1) % self.data.recording_count
        self._recording_combo.setCurrentIndex(new_index)

    def _next_recording(self) -> None:
        """Navigates to the next recording, wrapping around to the first."""
        new_index = (self.data.current_recording_index + 1) % self.data.recording_count
        self._recording_combo.setCurrentIndex(new_index)

    def _advance_recording(self) -> None:
        """Advances to the next recording during auto-cycling."""
        self._next_recording()

    def _start_cycling(self) -> None:
        """Starts automatic recording cycling."""
        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(True)
        self._auto_cycle_timer.start(self._style.cycle_interval)

    def _stop_cycling(self) -> None:
        """Stops automatic recording cycling and re-enables manual navigation."""
        self._auto_cycle_timer.stop()
        self._start_button.setEnabled(True)
        self._stop_button.setEnabled(False)

    def _normalize_image(self, image: NDArray[np.float32] | None) -> NDArray[np.uint8]:
        """Normalizes a float32 image to an uint8 RGB array using percentile clipping.

        Args:
            image: The input float32 image, or None for a black fallback.

        Returns:
            Normalized RGB image of shape (height, width, 3) with uint8 values.
        """
        frame_height = self.data.frame_height
        frame_width = self.data.frame_width

        if image is None:
            return np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        percentile_low = np.percentile(image, self._style.lower_percentile)
        percentile_high = np.percentile(image, self._style.upper_percentile)

        if percentile_high <= percentile_low:
            return np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        normalized = (image - percentile_low) / (percentile_high - percentile_low)
        normalized = np.clip(normalized, 0.0, 1.0)
        grayscale = (normalized * 255).astype(np.uint8)

        return np.stack([grayscale, grayscale, grayscale], axis=-1)

    def _overlay_masks(
        self,
        background: NDArray[np.uint8],
        masks: list[ROIStatistics],
        color: tuple[int, int, int],
        opacity: int,
    ) -> NDArray[np.uint8]:
        """Composites ROI mask outlines onto a background RGB image.

        Renders each ROI as a filled semi-transparent region with the specified color and opacity. Pixels outside the
        frame bounds are clipped.

        Args:
            background: The background RGB image of shape (height, width, 3).
            masks: The list of ROIStatistics to overlay.
            color: The RGB color tuple for mask pixels.
            opacity: The alpha value (0-255) for mask blending.

        Returns:
            The composited RGB image with mask overlays.
        """
        frame_height = self.data.frame_height
        frame_width = self.data.frame_width
        result = background.copy()
        alpha = opacity / 255.0

        for roi in masks:
            y_pixels = roi.y_pixels
            x_pixels = roi.x_pixels

            # Clips pixels to frame bounds.
            valid = (y_pixels >= 0) & (y_pixels < frame_height) & (x_pixels >= 0) & (x_pixels < frame_width)
            y_valid = y_pixels[valid]
            x_valid = x_pixels[valid]

            if len(y_valid) == 0:
                continue

            # Blends the mask color with the background.
            for channel_index in range(3):
                result[y_valid, x_valid, channel_index] = (
                    alpha * color[channel_index] + (1.0 - alpha) * result[y_valid, x_valid, channel_index]
                ).astype(np.uint8)

        return result
