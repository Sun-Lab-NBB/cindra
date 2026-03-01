"""Provides the multi-day tracking viewer window."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtGui, QtCore
import pyqtgraph as pg  # type: ignore[import-untyped]
from PySide6.QtWidgets import (
    QLabel,
    QSlider,
    QWidget,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLineEdit,
    QStatusBar,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QButtonGroup,
)
from matplotlib.colors import hsv_to_rgb

from .styles import STYLE, TRACKING_STYLE
from .widgets import TraceBox, plot_trace
from .constants import CONFIG, TRACKING_CONFIG, MaskLayer, BackgroundView, CoordinateSpace

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .multi_day_context import TrackingViewerData


class TrackingViewer(QMainWindow):
    """Displays a UI window for viewing the quality of across-day ROI tracking.

    Displays background images with ROI mask overlays for each recording in a multi-day dataset. Supports manual and
    automatic recording cycling, coordinate space switching, mask layer selection, channel toggling, and mask opacity
    control.

    Args:
        data: The preloaded tracking data to display on startup.
    """

    def __init__(self, data: TrackingViewerData) -> None:
        super().__init__()
        self.setWindowTitle("Multi-Day ROI Tracking")
        self.setGeometry(*TRACKING_STYLE.window_geometry)

        self.data: TrackingViewerData = data
        self._auto_cycle_timer: QtCore.QTimer = QtCore.QTimer(self)
        self._auto_cycle_timer.timeout.connect(self._advance_recording)

        # Configures pyqtgraph to use row-major axis order so images display with the correct orientation.
        pg.setConfigOptions(imageAxisOrder="row-major")

        # Display cache fields. Populated by _refresh_display and reused by _composite_and_display for fast opacity
        # updates without recomputing the background or mask coordinates.
        self._cached_background: NDArray[np.uint8] | None = None
        self._cached_mask_y: NDArray[np.int32] | None = None
        self._cached_mask_x: NDArray[np.int32] | None = None
        self._cached_mask_colors: NDArray[np.uint8] | None = None
        self._cached_mask_roi_indices: NDArray[np.int32] | None = None
        self._cached_roi_map: NDArray[np.int32] | None = None
        self._cached_mask_count: int = 0
        self._selected_rois: set[int] | None = None

        # Tracks the mask layer group and recording index that were active when the selection was last valid. Used by
        # _refresh_display to decide whether a recording or layer switch invalidates the current ROI selection.
        self._selection_was_template: bool = False
        self._selection_recording_index: int = -1

        # Trace display state.
        self._trace_activity_mode: int = CONFIG.default_activity_mode
        self._trace_deconvolved_visible: bool = True
        self._trace_neuropil_visible: bool = True
        self._trace_raw_visible: bool = True
        self._trace_scale_factor: float = CONFIG.default_scale_factor
        self._trace_all_recordings: bool = False

        # Builds the UI layout.
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Image + trace display panel (pyqtgraph).
        self._graphics_widget = pg.GraphicsLayoutWidget()
        # noinspection PyUnresolvedReferences
        self._view_box: pg.ViewBox = self._graphics_widget.addViewBox(row=0, col=0)
        self._view_box.setAspectLocked(True)
        self._view_box.invertY(True)
        self._image_item: pg.ImageItem = pg.ImageItem()
        self._view_box.addItem(self._image_item)
        self._graphics_widget.ci.layout.setRowStretchFactor(0, 2)

        # Trace panel below the image.
        self._trace_box = TraceBox()
        self._trace_box.setMouseEnabled(x=True, y=False)
        self._trace_box.enableAutoRange(x=True, y=True)
        self._graphics_widget.addItem(self._trace_box, row=1, col=0)
        self._graphics_widget.ci.layout.setRowStretchFactor(1, 1)

        main_layout.addWidget(self._graphics_widget, stretch=3)
        # noinspection PyUnresolvedReferences
        self._graphics_widget.scene().sigMouseClicked.connect(self._on_image_clicked)

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
        self._background_combo.addItem("Mean Image", userData=BackgroundView.MEAN_IMAGE)
        self._background_combo.addItem("Enhanced Mean", userData=BackgroundView.ENHANCED_MEAN_IMAGE)
        self._background_combo.addItem("Max Projection", userData=BackgroundView.MAXIMUM_PROJECTION)
        self._background_combo.addItem("Correlation Map", userData=BackgroundView.CORRELATION_MAP)
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
        self._mask_combo.addItem("Tracked", userData=MaskLayer.TRACKED)
        self._mask_combo.currentIndexChanged.connect(self._refresh_display)
        mask_layout.addWidget(self._mask_combo)

        opacity_row = QHBoxLayout()
        opacity_row.addWidget(QLabel("Opacity:"))
        self._opacity_slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(0, 255)
        self._opacity_slider.setValue(TRACKING_STYLE.default_mask_opacity)
        self._opacity_slider.setToolTip("Adjust mask opacity. Use the mouse wheel over the image to change quickly.")
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
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

        # ROI selection group.
        roi_group = QGroupBox("ROI Selection")
        roi_layout = QVBoxLayout(roi_group)

        roi_group.setToolTip("Click an ROI to select it. Ctrl-click or Shift-click to toggle individual ROIs.")

        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("ROI:"))
        self._roi_edit = QLineEdit()
        self._roi_edit.setFixedWidth(STYLE.roi_edit_width)
        self._roi_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._roi_edit.setReadOnly(True)
        self._roi_edit.setToolTip("Displays the index of the last clicked ROI.")
        input_row.addWidget(self._roi_edit)
        roi_layout.addLayout(input_row)

        button_row = QHBoxLayout()
        all_button = QPushButton("All")
        all_button.setToolTip("Show all ROIs.")
        all_button.clicked.connect(self._select_all_rois)
        button_row.addWidget(all_button)
        none_button = QPushButton("None")
        none_button.setToolTip("Hide all ROIs.")
        none_button.clicked.connect(self._deselect_all_rois)
        button_row.addWidget(none_button)
        roi_layout.addLayout(button_row)

        layout.addWidget(roi_group)

        # Trace display group.
        trace_group = QGroupBox("Trace Display")
        trace_layout = QVBoxLayout(trace_group)

        activity_row = QHBoxLayout()
        activity_row.addWidget(QLabel("Activity:"))
        self._trace_activity_combo = QComboBox()
        self._trace_activity_combo.addItems(CONFIG.activity_mode_labels)
        self._trace_activity_combo.setCurrentIndex(CONFIG.default_activity_mode)
        self._trace_activity_combo.currentIndexChanged.connect(self._on_trace_activity_changed)
        activity_row.addWidget(self._trace_activity_combo)
        trace_layout.addLayout(activity_row)

        checkbox_row = QHBoxLayout()
        self._trace_deconvolved_checkbox = QCheckBox("deconv")
        self._trace_deconvolved_checkbox.setChecked(True)
        self._trace_deconvolved_checkbox.toggled.connect(self._on_trace_visibility_changed)
        checkbox_row.addWidget(self._trace_deconvolved_checkbox)

        self._trace_neuropil_checkbox = QCheckBox("neuropil")
        self._trace_neuropil_checkbox.setChecked(True)
        self._trace_neuropil_checkbox.toggled.connect(self._on_trace_visibility_changed)
        checkbox_row.addWidget(self._trace_neuropil_checkbox)

        self._trace_raw_checkbox = QCheckBox("raw")
        self._trace_raw_checkbox.setChecked(True)
        self._trace_raw_checkbox.toggled.connect(self._on_trace_visibility_changed)
        checkbox_row.addWidget(self._trace_raw_checkbox)
        trace_layout.addLayout(checkbox_row)

        scale_row = QHBoxLayout()
        scale_up = QPushButton("+")
        scale_up.setMaximumWidth(STYLE.square_button_width)
        scale_up.clicked.connect(lambda: self._adjust_trace_scale(CONFIG.scale_step))
        scale_row.addWidget(scale_up)
        scale_down = QPushButton("-")
        scale_down.setMaximumWidth(STYLE.square_button_width)
        scale_down.clicked.connect(lambda: self._adjust_trace_scale(-CONFIG.scale_step))
        scale_row.addWidget(scale_down)

        self._trace_all_recordings_button = QPushButton("All recordings")
        self._trace_all_recordings_button.setCheckable(True)
        self._trace_all_recordings_button.setToolTip(
            "Show traces from all recordings stacked vertically for the selected ROI."
        )
        self._trace_all_recordings_button.toggled.connect(self._on_all_recordings_toggled)
        scale_row.addWidget(self._trace_all_recordings_button)
        trace_layout.addLayout(scale_row)

        layout.addWidget(trace_group)

        layout.addStretch()

        # Prevents control panel widgets from capturing keyboard focus so spacebar and arrow keys always reach the
        # main window's keyPressEvent.
        for child in panel.findChildren(QWidget):
            child.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        return panel

    def _refresh_display(self) -> None:
        """Rebuilds the cached background and mask coordinates, then composites and displays the result.

        Caches the normalized background image and pre-collected mask pixel coordinates so that opacity-only changes
        can skip the expensive recomputation and go directly through ``_composite_and_display``.
        """
        coordinate_space = self._space_combo.currentData()
        background_type = self._background_combo.currentData()
        mask_layer = self._mask_combo.currentData()
        channel_2 = self._channel_2_checkbox.isChecked()

        # Retrieves and normalizes the background image into the cache.
        background = self.data.background_image(
            image_type=background_type,
            coordinate_space=coordinate_space,
            channel_2=channel_2,
        )
        self._cached_background = self._normalize_image(image=background)

        # Pre-collects all valid mask pixel coordinates and per-ROI colors into the cache.
        masks = self.data.masks_for_layer(layer=mask_layer, channel_2=channel_2)
        self._cached_mask_count = len(masks) if masks else 0
        self._cached_mask_y = None
        self._cached_mask_x = None
        self._cached_mask_colors = None
        self._cached_mask_roi_indices = None
        self._cached_roi_map = None

        # Determines whether the ROI identity set has changed, requiring a selection reset. Template and Tracked
        # layers share the same ROI identity set (template-derived) so the selection persists across recording
        # switches and Template/Tracked toggles. Original and Deformed share a separate identity set (single-day
        # extraction) so the selection persists only within the same session.
        current_is_template_group = mask_layer in (MaskLayer.TEMPLATE, MaskLayer.TRACKED)
        recording_index = self.data.current_recording_index
        layer_group_changed = current_is_template_group != self._selection_was_template
        session_changed = not current_is_template_group and recording_index != self._selection_recording_index
        if layer_group_changed or session_changed:
            self._selected_rois = None
            self._roi_edit.clear()
        elif self._selected_rois is not None and self._cached_mask_count > 0:
            # Clamps the selection to valid indices in case the mask count differs.
            self._selected_rois = {i for i in self._selected_rois if i < self._cached_mask_count}
        self._selection_was_template = current_is_template_group
        self._selection_recording_index = recording_index

        if masks:
            frame_height = self.data.frame_height
            frame_width = self.data.frame_width

            # Generates deterministic per-ROI colors using random HSV hues with full saturation and value.
            # Original and Deformed layers use the Original mask count as the palette reference so both layers
            # share identical colors within a session (they represent the same ROIs in different coordinate
            # spaces). Template layers use their own count directly, which is identical across all sessions,
            # ensuring consistent colors when switching recordings.
            # Template and Tracked layers share a color palette (same ROI identity set). Original and Deformed
            # layers share a separate palette (same single-day ROIs in different coordinate spaces).
            if mask_layer in (MaskLayer.TEMPLATE, MaskLayer.TRACKED):
                template_masks = self.data.masks_for_layer(layer=MaskLayer.TEMPLATE, channel_2=channel_2)
                color_count = len(template_masks) if template_masks else len(masks)
            else:
                original_masks = self.data.masks_for_layer(layer=MaskLayer.ORIGINAL, channel_2=channel_2)
                color_count = len(original_masks) if original_masks else len(masks)

            rng = np.random.default_rng(seed=CONFIG.random_color_seed)
            hues = rng.random(color_count)
            hsv = np.stack([hues, np.ones_like(hues), np.ones_like(hues)], axis=-1)
            roi_colors = (255.0 * hsv_to_rgb(hsv)).astype(np.uint8)

            # Builds the ROI ownership map for O(1) click-to-ROI lookup. Initialized to -1 (no ROI).
            roi_map = np.full(shape=(frame_height, frame_width), fill_value=-1, dtype=np.int32)

            all_y: list[NDArray[np.int32]] = []
            all_x: list[NDArray[np.int32]] = []
            pixel_counts: list[int] = []
            valid_roi_colors: list[NDArray[np.uint8]] = []
            valid_roi_indices: list[int] = []
            for roi_index, item in enumerate(masks):
                roi = item.mask if hasattr(item, "mask") else item
                valid = (
                    (roi.y_pixels >= 0)
                    & (roi.y_pixels < frame_height)
                    & (roi.x_pixels >= 0)
                    & (roi.x_pixels < frame_width)
                )
                y_valid = roi.y_pixels[valid].astype(np.int32)
                x_valid = roi.x_pixels[valid].astype(np.int32)
                if len(y_valid) > 0:
                    all_y.append(y_valid)
                    all_x.append(x_valid)
                    pixel_counts.append(len(y_valid))
                    valid_roi_colors.append(roi_colors[roi_index])
                    valid_roi_indices.append(roi_index)
                    roi_map[y_valid, x_valid] = roi_index
            if all_y:
                self._cached_mask_y = np.concatenate(all_y)
                self._cached_mask_x = np.concatenate(all_x)
                self._cached_mask_colors = np.repeat(
                    np.array(valid_roi_colors, dtype=np.uint8), repeats=pixel_counts, axis=0
                )
                self._cached_mask_roi_indices = np.repeat(
                    np.array(valid_roi_indices, dtype=np.int32), repeats=pixel_counts
                )
            self._cached_roi_map = roi_map

        self._composite_and_display()

    def _composite_and_display(self) -> None:
        """Blends cached mask coordinates onto the cached background at the current opacity and updates the display."""
        if self._cached_background is None:
            return

        opacity = self._opacity_slider.value()
        display_image = self._cached_background.copy()

        # Vectorized alpha-blend of all mask pixels in a single operation using per-ROI colors.
        if (
            self._cached_mask_y is not None
            and self._cached_mask_x is not None
            and self._cached_mask_colors is not None
            and len(self._cached_mask_y) > 0
        ):
            # Filters to only selected ROIs when a subset is active.
            if self._selected_rois is not None and self._cached_mask_roi_indices is not None:
                selected_mask = np.isin(self._cached_mask_roi_indices, list(self._selected_rois))
                mask_y = self._cached_mask_y[selected_mask]
                mask_x = self._cached_mask_x[selected_mask]
                mask_colors = self._cached_mask_colors[selected_mask]
            else:
                mask_y = self._cached_mask_y
                mask_x = self._cached_mask_x
                mask_colors = self._cached_mask_colors

            if len(mask_y) > 0:
                alpha = opacity / 255.0
                display_image[mask_y, mask_x] = (
                    alpha * mask_colors.astype(np.float32)
                    + (1.0 - alpha) * display_image[mask_y, mask_x].astype(np.float32)
                ).astype(np.uint8)

        self._image_item.setImage(display_image)

        # Updates the status bar with selection info when a subset is active.
        recording_id = self.data.current_recording_id
        if self._selected_rois is not None:
            selection_text = f"Selected: {len(self._selected_rois)} / {self._cached_mask_count}"
        else:
            selection_text = f"Masks: {self._cached_mask_count}"
        self._status_bar.showMessage(
            f"Recording: {recording_id}  |  {selection_text}  |  "
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
        self._refresh_traces()

    def _on_opacity_changed(self) -> None:
        """Handles opacity slider changes by re-compositing from cached data.

        Skips the expensive background normalization and mask coordinate collection that ``_refresh_display`` performs,
        since only the alpha blend value has changed.
        """
        self._composite_and_display()

    def _on_image_clicked(self, event: object) -> None:
        """Handles mouse clicks on the image to select or toggle ROIs.

        Plain click selects a single ROI. Ctrl-click or Shift-click toggles an ROI in/out of the current selection.
        Clicking empty space (no ROI) is ignored.

        Args:
            event: The pyqtgraph mouse click event from the scene signal.
        """
        if self._cached_roi_map is None:
            return

        # Maps click position from scene coordinates to image pixel coordinates.
        scene_position = event.scenePos()  # type: ignore[attr-defined]
        view_position = self._view_box.mapSceneToView(scene_position)
        click_x = int(view_position.x())
        click_y = int(view_position.y())

        # Bounds-checks against frame dimensions.
        if click_y < 0 or click_y >= self.data.frame_height or click_x < 0 or click_x >= self.data.frame_width:
            return

        # noinspection PyTypeChecker
        roi_index = int(self._cached_roi_map[click_y, click_x])
        if roi_index < 0:
            return

        # Determines whether to toggle (Ctrl/Shift held) or replace selection (plain click).
        modifiers = event.modifiers()  # type: ignore[attr-defined]
        # noinspection PyTypeChecker
        is_toggle = bool(
            modifiers & (QtCore.Qt.KeyboardModifier.ControlModifier | QtCore.Qt.KeyboardModifier.ShiftModifier)
        )

        if is_toggle:
            # Converts from "all visible" to an explicit set containing every ROI, then toggles the clicked one.
            if self._selected_rois is None:
                self._selected_rois = set(range(self._cached_mask_count))
            if roi_index in self._selected_rois:
                self._selected_rois.discard(roi_index)
            else:
                self._selected_rois.add(roi_index)
        else:
            self._selected_rois = {roi_index}

        self._roi_edit.setText(str(roi_index))
        self._composite_and_display()
        self._refresh_traces()

    def _select_all_rois(self) -> None:
        """Resets the selection to show all ROIs."""
        self._selected_rois = None
        self._roi_edit.clear()
        self._composite_and_display()
        self._refresh_traces()

    def _deselect_all_rois(self) -> None:
        """Clears the selection so no ROIs are visible."""
        self._selected_rois = set()
        self._roi_edit.clear()
        self._composite_and_display()
        self._refresh_traces()

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
        self._auto_cycle_timer.start(TRACKING_CONFIG.cycle_interval)

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

        percentile_low = np.percentile(image, TRACKING_CONFIG.lower_percentile)
        percentile_high = np.percentile(image, TRACKING_CONFIG.upper_percentile)

        if percentile_high <= percentile_low:
            return np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        normalized = (image - percentile_low) / (percentile_high - percentile_low)
        normalized = np.clip(normalized, 0.0, 1.0)
        grayscale = (normalized * 255).astype(np.uint8)

        return np.stack([grayscale, grayscale, grayscale], axis=-1)

    def _on_trace_activity_changed(self, index: int) -> None:
        """Handles trace activity mode dropdown changes.

        Args:
            index: The activity mode index selected.
        """
        self._trace_activity_mode = index
        self._refresh_traces()

    def _on_trace_visibility_changed(self) -> None:
        """Handles trace visibility checkbox toggles."""
        self._trace_deconvolved_visible = self._trace_deconvolved_checkbox.isChecked()
        self._trace_neuropil_visible = self._trace_neuropil_checkbox.isChecked()
        self._trace_raw_visible = self._trace_raw_checkbox.isChecked()
        self._refresh_traces()

    def _adjust_trace_scale(self, delta: float) -> None:
        """Adjusts the vertical scale factor for multi-trace stacking.

        Args:
            delta: The amount to change the scale factor by.
        """
        self._trace_scale_factor = max(CONFIG.min_scale, min(CONFIG.max_scale, self._trace_scale_factor + delta))
        self._refresh_traces()

    def _on_all_recordings_toggled(self, checked: bool) -> None:
        """Handles the 'all recordings' toggle button.

        Args:
            checked: True when stacked all-recordings view is enabled.
        """
        self._trace_all_recordings = checked
        self._refresh_traces()

    def _refresh_traces(self) -> None:
        """Refreshes the trace panel for the currently selected ROIs.

        In single-recording mode, plots traces from the current recording's extraction data.
        In all-recordings mode, stacks traces from every recording for the first selected ROI.
        """
        if self._trace_all_recordings and self._selected_rois is not None and len(self._selected_rois) == 1:
            self._refresh_all_recording_traces()
            return

        if not self.data.has_traces:
            self._trace_box.clear()
            return

        cell_fluorescence = self.data.cell_fluorescence
        neuropil_fluorescence = self.data.neuropil_fluorescence
        spikes = self.data.spikes

        if cell_fluorescence is None or neuropil_fluorescence is None or spikes is None:
            self._trace_box.clear()
            return

        # Determines which ROIs to plot.
        merge_indices = sorted(self._selected_rois) if self._selected_rois is not None else []

        if not merge_indices:
            self._trace_box.clear()
            return

        # Guards against out-of-range indices.
        roi_count = cell_fluorescence.shape[0]
        merge_indices = [i for i in merge_indices if i < roi_count]
        if not merge_indices:
            self._trace_box.clear()
            return

        frame_indices = np.arange(cell_fluorescence.shape[1], dtype=np.int32)
        plot_trace(
            trace_box=self._trace_box,
            cell_fluorescence=cell_fluorescence,
            neuropil_fluorescence=neuropil_fluorescence,
            spikes=spikes,
            frame_indices=frame_indices,
            merge_indices=merge_indices,
            activity_mode=self._trace_activity_mode,
            traces_visible=self._trace_raw_visible,
            neuropil_visible=self._trace_neuropil_visible,
            deconvolved_visible=self._trace_deconvolved_visible,
            scale_factor=self._trace_scale_factor,
        )

    def _refresh_all_recording_traces(self) -> None:
        """Plots traces from all recordings stacked vertically for the first selected ROI.

        Iterates over every recording in the dataset, extracts the selected ROI's trace from each,
        and plots them stacked with recording-index labels on the y-axis.
        """
        self._trace_box.clear()
        if self._selected_rois is None or len(self._selected_rois) == 0:
            return

        roi_index = next(iter(self._selected_rois))
        axis = self._trace_box.getAxis("left")
        tick_labels: list[tuple[float, str]] = []
        k_space = 1.0 / self._trace_scale_factor
        max_frames = 0
        y_maximum = 0.0
        stack_position = self.data.recording_count - 1

        for recording_index in range(self.data.recording_count):
            cell_fluorescence = self.data.cell_fluorescence_for_recording(recording_index)
            neuropil_fluorescence = self.data.neuropil_fluorescence_for_recording(recording_index)
            spikes = self.data.spikes_for_recording(recording_index)

            if cell_fluorescence is None or cell_fluorescence.size == 0:
                stack_position -= 1
                continue
            if roi_index >= cell_fluorescence.shape[0]:
                stack_position -= 1
                continue

            # Selects trace based on activity mode.
            if self._trace_activity_mode == 0:
                trace = cell_fluorescence[roi_index, :]
            elif self._trace_activity_mode == 1:
                trace = (
                    neuropil_fluorescence[roi_index, :]
                    if neuropil_fluorescence is not None
                    else cell_fluorescence[roi_index, :]
                )
            elif self._trace_activity_mode == CONFIG.activity_mode_subtracted:
                nf = (
                    neuropil_fluorescence[roi_index, :]
                    if neuropil_fluorescence is not None
                    else np.zeros_like(cell_fluorescence[roi_index, :])
                )
                trace = cell_fluorescence[roi_index, :] - CONFIG.neuropil_coefficient * nf
            else:
                trace = spikes[roi_index, :] if spikes is not None else cell_fluorescence[roi_index, :]

            frame_indices = np.arange(len(trace), dtype=np.int32)
            max_frames = max(max_frames, len(trace))

            # Normalizes trace to [0, 1].
            trace_max = float(trace.max())
            trace_min = float(trace.min())
            if trace_max > trace_min:
                normalized = (trace - trace_min) / (trace_max - trace_min)
            else:
                normalized = np.zeros_like(trace)

            # Generates a color for this recording.
            hue = recording_index / max(self.data.recording_count, 1)
            hsv = np.array([[hue, 1.0, 1.0]])
            rgb = (255.0 * hsv_to_rgb(hsv)).astype(np.uint8)[0]
            pen_color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

            self._trace_box.plot(frame_indices, normalized + stack_position * k_space, pen=pen_color)
            tick_labels.append((stack_position * k_space + float(normalized.mean()), str(recording_index)))
            y_maximum = max(y_maximum, stack_position * k_space + 1)
            stack_position -= 1

        axis.setTicks([tick_labels])
        self._trace_box.update_range(
            frame_count=max_frames,
            y_minimum=0.0,
            y_maximum=y_maximum if y_maximum > 0 else 1.0,
        )
