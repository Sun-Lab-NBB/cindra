"""Provides the main window for the multi-day tracking quality viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtGui, QtCore
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QLabel,
    QSlider,
    QWidget,
    QSpinBox,
    QComboBox,
    QGroupBox,
    QStatusBar,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from .context_data import MaskLayer, BackgroundImage, CoordinateSpace, TrackingViewerData

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ...dataclasses.single_day_data import ROIStatistics

# Millisecond interval for auto-cycling between sessions.
_DEFAULT_CYCLE_INTERVAL: int = 2000

# Minimum auto-cycle interval in milliseconds.
_MINIMUM_CYCLE_INTERVAL: int = 200

# Maximum auto-cycle interval in milliseconds.
_MAXIMUM_CYCLE_INTERVAL: int = 10000

# Step size for the auto-cycle interval spin box.
_CYCLE_INTERVAL_STEP: int = 200

# Default mask overlay opacity (0-255 uint8 range).
_DEFAULT_MASK_OPACITY: int = 127

# Mask outline color for channel 1 ROIs (cyan).
_CHANNEL_1_MASK_COLOR: tuple[int, int, int] = (0, 255, 255)

# Mask outline color for channel 2 ROIs (red).
_CHANNEL_2_MASK_COLOR: tuple[int, int, int] = (255, 80, 80)

# Percentile values for normalizing background images.
_LOWER_PERCENTILE: float = 1.0
_UPPER_PERCENTILE: float = 99.0


class TrackingViewer(QMainWindow):
    """Multi-day cell tracking quality viewer.

    Displays background images with ROI mask overlays for each session in a multi-day
    dataset. Supports manual and automatic session cycling, coordinate space switching,
    mask layer selection, channel toggling, and mask opacity control.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Multi-Day Tracking Viewer")
        self.resize(1200, 800)

        self._data: TrackingViewerData | None = None
        self._auto_cycle_timer: QtCore.QTimer = QtCore.QTimer(self)
        self._auto_cycle_timer.timeout.connect(self._advance_session)

        # Builds the UI layout.
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Image display panel (pyqtgraph).
        self._graphics_widget = pg.GraphicsLayoutWidget()
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

    def load_data(self, data: TrackingViewerData) -> None:
        """Loads a multi-day dataset into the viewer.

        Args:
            data: The tracking viewer data model to display.
        """
        self._data = data

        # Populates the session selector.
        self._session_combo.blockSignals(True)
        self._session_combo.clear()
        for index, session_id in enumerate(data.session_ids):
            self._session_combo.addItem(f"{index}: {session_id}", userData=index)
        self._session_combo.setCurrentIndex(0)
        self._session_combo.blockSignals(False)

        # Enables channel 2 toggle if available.
        self._channel_2_checkbox.setEnabled(data.has_channel_2)

        self._refresh_display()

    def _build_control_panel(self) -> QWidget:
        """Constructs the right-side control panel with all viewer controls.

        Returns:
            The assembled control panel widget.
        """
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)

        # Session navigation group.
        session_group = QGroupBox("Session Navigation")
        session_layout = QVBoxLayout(session_group)

        self._session_combo = QComboBox()
        self._session_combo.currentIndexChanged.connect(self._on_session_selected)
        session_layout.addWidget(self._session_combo)

        navigation_row = QHBoxLayout()
        self._previous_button = QPushButton("<")
        self._previous_button.clicked.connect(self._previous_session)
        navigation_row.addWidget(self._previous_button)

        self._next_button = QPushButton(">")
        self._next_button.clicked.connect(self._next_session)
        navigation_row.addWidget(self._next_button)

        self._cycle_button = QPushButton("Auto Cycle")
        self._cycle_button.setCheckable(True)
        self._cycle_button.toggled.connect(self._toggle_auto_cycle)
        navigation_row.addWidget(self._cycle_button)
        session_layout.addLayout(navigation_row)

        interval_row = QHBoxLayout()
        interval_row.addWidget(QLabel("Interval (ms):"))
        self._interval_spin = QSpinBox()
        self._interval_spin.setRange(_MINIMUM_CYCLE_INTERVAL, _MAXIMUM_CYCLE_INTERVAL)
        self._interval_spin.setSingleStep(_CYCLE_INTERVAL_STEP)
        self._interval_spin.setValue(_DEFAULT_CYCLE_INTERVAL)
        self._interval_spin.valueChanged.connect(self._on_interval_changed)
        interval_row.addWidget(self._interval_spin)
        session_layout.addLayout(interval_row)

        layout.addWidget(session_group)

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
        self._opacity_slider = QSlider(QtCore.Qt.Horizontal)
        self._opacity_slider.setRange(0, 255)
        self._opacity_slider.setValue(_DEFAULT_MASK_OPACITY)
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
        if self._data is None:
            return

        coordinate_space = self._space_combo.currentData()
        background_type = self._background_combo.currentData()
        mask_layer = self._mask_combo.currentData()
        channel_2 = self._channel_2_checkbox.isChecked()
        opacity = self._opacity_slider.value()

        # Retrieves and normalizes the background image.
        background = self._data.background_image(
            image_type=background_type,
            coordinate_space=coordinate_space,
            channel_2=channel_2,
        )
        display_image = _normalize_image(
            image=background,
            frame_height=self._data.frame_height,
            frame_width=self._data.frame_width,
        )

        # Retrieves the mask set for the current layer and channel.
        masks = self._data.masks_for_layer(layer=mask_layer, channel_2=channel_2)

        # Composites the mask overlay onto the background.
        if masks:
            mask_color = _CHANNEL_2_MASK_COLOR if channel_2 else _CHANNEL_1_MASK_COLOR
            display_image = _overlay_masks(
                background=display_image,
                masks=masks,
                color=mask_color,
                opacity=opacity,
                frame_height=self._data.frame_height,
                frame_width=self._data.frame_width,
            )

        self._image_item.setImage(display_image)

        # Updates the status bar.
        session_id = self._data.current_session_id
        mask_count = len(masks) if masks else 0
        self._status_bar.showMessage(
            f"Session: {session_id}  |  Masks: {mask_count}  |  "
            f"Size: {self._data.frame_height} x {self._data.frame_width}"
        )

    def _on_session_selected(self, index: int) -> None:
        """Handles session combo box selection changes.

        Args:
            index: The newly selected combo box index.
        """
        if self._data is None or index < 0:
            return
        self._data.switch_session(session_index=index)
        self._refresh_display()

    def _previous_session(self) -> None:
        """Navigates to the previous session, wrapping around to the last."""
        if self._data is None:
            return
        new_index = (self._data.current_session_index - 1) % self._data.session_count
        self._session_combo.setCurrentIndex(new_index)

    def _next_session(self) -> None:
        """Navigates to the next session, wrapping around to the first."""
        if self._data is None:
            return
        new_index = (self._data.current_session_index + 1) % self._data.session_count
        self._session_combo.setCurrentIndex(new_index)

    def _advance_session(self) -> None:
        """Advances to the next session during auto-cycling."""
        self._next_session()

    def _toggle_auto_cycle(self, enabled: bool) -> None:
        """Starts or stops automatic session cycling.

        Args:
            enabled: True to start cycling, False to stop.
        """
        if enabled:
            self._auto_cycle_timer.start(self._interval_spin.value())
            self._cycle_button.setText("Stop Cycle")
        else:
            self._auto_cycle_timer.stop()
            self._cycle_button.setText("Auto Cycle")

    def _on_interval_changed(self, interval: int) -> None:
        """Updates the auto-cycle timer interval.

        Args:
            interval: New interval in milliseconds.
        """
        if self._auto_cycle_timer.isActive():
            self._auto_cycle_timer.setInterval(interval)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        """Handles keyboard shortcuts for session navigation.

        Left/Right arrows navigate between sessions. Space toggles auto-cycling.
        """
        key = event.key()
        if key == QtCore.Qt.Key_Left:
            self._previous_session()
        elif key == QtCore.Qt.Key_Right:
            self._next_session()
        elif key == QtCore.Qt.Key_Space:
            self._cycle_button.toggle()
        else:
            super().keyPressEvent(event)


def _normalize_image(
    image: NDArray[np.float32] | None,
    frame_height: int,
    frame_width: int,
) -> NDArray[np.uint8]:
    """Normalizes a float32 image to a uint8 RGB array using percentile clipping.

    Args:
        image: Input float32 image, or None for a black fallback.
        frame_height: Height for the fallback image.
        frame_width: Width for the fallback image.

    Returns:
        Normalized RGB image of shape (height, width, 3) with uint8 values.
    """
    if image is None:
        return np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    percentile_low = np.percentile(image, _LOWER_PERCENTILE)
    percentile_high = np.percentile(image, _UPPER_PERCENTILE)

    if percentile_high <= percentile_low:
        return np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

    normalized = (image - percentile_low) / (percentile_high - percentile_low)
    normalized = np.clip(normalized, 0.0, 1.0)
    grayscale = (normalized * 255).astype(np.uint8)

    return np.stack([grayscale, grayscale, grayscale], axis=-1)


def _overlay_masks(
    background: NDArray[np.uint8],
    masks: list[ROIStatistics],
    color: tuple[int, int, int],
    opacity: int,
    frame_height: int,
    frame_width: int,
) -> NDArray[np.uint8]:
    """Composites ROI mask outlines onto a background RGB image.

    Renders each ROI as a filled semi-transparent region with the specified color
    and opacity. Pixels outside the frame bounds are clipped.

    Args:
        background: The background RGB image of shape (height, width, 3).
        masks: The list of ROIStatistics to overlay.
        color: The RGB color tuple for mask pixels.
        opacity: The alpha value (0-255) for mask blending.
        frame_height: The image height for bounds checking.
        frame_width: The image width for bounds checking.

    Returns:
        The composited RGB image with mask overlays.
    """
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
