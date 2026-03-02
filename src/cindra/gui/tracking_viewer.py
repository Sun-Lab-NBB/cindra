"""Provides the multi-day tracking viewer window."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtGui, QtCore
import pyqtgraph as pg  # type: ignore[import-untyped]
from PySide6.QtWidgets import (
    QLabel,
    QStyle,
    QSlider,
    QWidget,
    QComboBox,
    QGroupBox,
    QLineEdit,
    QStatusBar,
    QHBoxLayout,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QButtonGroup,
)
from matplotlib.colors import hsv_to_rgb

from .styles import STYLE, TRACKING_STYLE
from .constants import COMMON_CONFIG, ROI_CONFIG, TRACKING_CONFIG, MaskLayer, BackgroundView, CoordinateSpace
from .viewer_context import EMPTY

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ..dataclasses import ROIMask, ROIStatistics
    from .viewer_context import ViewerData, MultiDayData


class TrackingViewer(QMainWindow):
    """Displays a UI window for viewing the quality of across-day ROI tracking.

    Displays background images with ROI mask overlays for each recording in a multi-day dataset. Supports manual and
    automatic recording cycling, coordinate space switching, mask layer selection, channel toggling, and mask opacity
    control.

    Args:
        data: The preloaded tracking data to display on startup.
    """

    def __init__(self, data: ViewerData) -> None:
        super().__init__()
        self.setWindowTitle("Multi-Day ROI Tracking")
        self.setGeometry(*TRACKING_STYLE.window_geometry)

        self.data: ViewerData = data
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

    def load_data(self, data: ViewerData) -> None:
        """Caches the input ViewerData instance and uses it to populate the managed UI window.

        Args:
            data: The ViewerData instance that stores the visualized dataset's data.
        """
        self.data = data

        # Populates the dataset selector.
        self._dataset_combo.blockSignals(True)
        self._dataset_combo.clear()
        for name in data.available_datasets:
            self._dataset_combo.addItem(name, userData=name)
        # Selects the active dataset in the combo box.
        active = data.active_dataset_name
        for i in range(self._dataset_combo.count()):
            if self._dataset_combo.itemData(i) == active:
                self._dataset_combo.setCurrentIndex(i)
                break
        self._dataset_combo.blockSignals(False)

        # Populates the recording selector.
        self._recording_combo.blockSignals(True)
        self._recording_combo.clear()
        for index, recording_id in enumerate(data.recording_ids):
            self._recording_combo.addItem(f"{index}: {recording_id}", userData=index)
        self._recording_combo.setCurrentIndex(0)
        self._recording_combo.blockSignals(False)

        # Enables channel 2 toggle if available.
        has_channel_2 = data.current_recording.has_channel_2
        self._channel_2_checkbox.setEnabled(has_channel_2)
        self._channel_2_checkbox.setStyleSheet(STYLE.button_unpressed if has_channel_2 else STYLE.button_inactive)

        self._refresh_display()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        """Handles keyboard navigation for recording stepping and auto-cycle control.

        Notes:
            Overrides the Qt virtual method. The camelCase name is required to match the parent signature.
        """
        # Left/right arrow keys step through recordings when auto-cycling is stopped.
        if self._play_button.isEnabled():
            if event.key() == QtCore.Qt.Key.Key_Left:
                self._previous_recording()
            elif event.key() == QtCore.Qt.Key.Key_Right:
                self._next_recording()

        # Spacebar toggles between play and pause.
        if event.key() == QtCore.Qt.Key.Key_Space:
            if self._play_button.isEnabled():
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

        # Dataset selector group. Lists all discovered multi-day datasets for runtime switching.
        dataset_group = QGroupBox("Dataset")
        dataset_group.setStyleSheet(STYLE.group_box)
        dataset_layout = QVBoxLayout(dataset_group)
        self._dataset_combo = QComboBox()
        self._dataset_combo.currentIndexChanged.connect(self._on_dataset_selected)
        dataset_layout.addWidget(self._dataset_combo)
        layout.addWidget(dataset_group)

        # Recording navigation group.
        recording_group = QGroupBox("Recording Navigation")
        recording_group.setStyleSheet(STYLE.group_box)
        recording_layout = QVBoxLayout(recording_group)

        self._recording_combo = QComboBox()
        self._recording_combo.currentIndexChanged.connect(self._on_recording_selected)
        recording_layout.addWidget(self._recording_combo)

        # Playback controls. Play and pause are grouped exclusively so only one can be active at a time.
        navigation_row = QHBoxLayout()
        icon_size = QtCore.QSize(STYLE.icon_size, STYLE.icon_size)

        self._skip_backward_button: QToolButton = QToolButton()
        self._skip_backward_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward))
        self._skip_backward_button.setIconSize(icon_size)
        self._skip_backward_button.setToolTip("Previous recording (Left arrow).")
        self._skip_backward_button.clicked.connect(self._previous_recording)

        self._play_button: QToolButton = QToolButton()
        self._play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self._play_button.setIconSize(icon_size)
        self._play_button.setCheckable(True)
        self._play_button.setToolTip("Start auto-cycling (Space).")
        self._play_button.clicked.connect(self._start_cycling)

        self._pause_button: QToolButton = QToolButton()
        self._pause_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        self._pause_button.setIconSize(icon_size)
        self._pause_button.setCheckable(True)
        self._pause_button.setToolTip(
            "Stop auto-cycling (Space). Use Left/Right arrow keys to step through recordings."
        )
        self._pause_button.clicked.connect(self._stop_cycling)

        self._skip_forward_button: QToolButton = QToolButton()
        self._skip_forward_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward))
        self._skip_forward_button.setIconSize(icon_size)
        self._skip_forward_button.setToolTip("Next recording (Right arrow).")
        self._skip_forward_button.clicked.connect(self._next_recording)

        button_group = QButtonGroup(self)
        button_group.addButton(self._play_button, 0)
        button_group.addButton(self._pause_button, 1)
        button_group.setExclusive(True)

        navigation_row.addWidget(self._skip_backward_button)
        navigation_row.addWidget(self._play_button)
        navigation_row.addWidget(self._pause_button)
        navigation_row.addWidget(self._skip_forward_button)

        # Controls start with pause pre-selected, since there is no active cycling on startup.
        self._play_button.setEnabled(True)
        self._pause_button.setEnabled(False)
        self._pause_button.setChecked(True)
        recording_layout.addLayout(navigation_row)

        layout.addWidget(recording_group)

        # Background image group.
        background_group = QGroupBox("Background Image")
        background_group.setStyleSheet(STYLE.group_box)
        background_layout = QVBoxLayout(background_group)

        self._background_combo = QComboBox()
        self._background_combo.addItem("ROIs Only", userData=BackgroundView.ROIS_ONLY)
        self._background_combo.addItem("Mean Image", userData=BackgroundView.MEAN_IMAGE)
        self._background_combo.addItem("Enhanced Mean", userData=BackgroundView.ENHANCED_MEAN_IMAGE)
        self._background_combo.addItem("Max Projection", userData=BackgroundView.MAXIMUM_PROJECTION)
        self._background_combo.addItem("Correlation Map", userData=BackgroundView.CORRELATION_MAP)
        self._background_combo.currentIndexChanged.connect(self._refresh_display)
        background_layout.addWidget(self._background_combo)

        layout.addWidget(background_group)

        # Coordinate space group.
        space_group = QGroupBox("Coordinate Space")
        space_group.setStyleSheet(STYLE.group_box)
        space_layout = QVBoxLayout(space_group)

        self._space_combo = QComboBox()
        self._space_combo.addItem("Native", userData=CoordinateSpace.NATIVE)
        self._space_combo.addItem("Transformed", userData=CoordinateSpace.TRANSFORMED)
        self._space_combo.currentIndexChanged.connect(self._refresh_display)
        space_layout.addWidget(self._space_combo)

        layout.addWidget(space_group)

        # Mask layer group.
        mask_group = QGroupBox("Mask Layer")
        mask_group.setStyleSheet(STYLE.group_box)
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
        self._opacity_slider.setValue(STYLE.default_mask_opacity)
        self._opacity_slider.setToolTip("Adjust mask opacity. Use the mouse wheel over the image to change quickly.")
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_row.addWidget(self._opacity_slider)
        mask_layout.addLayout(opacity_row)

        layout.addWidget(mask_group)

        # Channel group.
        channel_group = QGroupBox("Channel")
        channel_group.setStyleSheet(STYLE.group_box)
        channel_layout = QVBoxLayout(channel_group)

        self._channel_2_checkbox = QPushButton("Channel 2")
        self._channel_2_checkbox.setCheckable(True)
        self._channel_2_checkbox.setEnabled(False)
        self._channel_2_checkbox.setStyleSheet(STYLE.button_inactive)
        self._channel_2_checkbox.toggled.connect(self._on_channel_2_toggled)
        channel_layout.addWidget(self._channel_2_checkbox)

        layout.addWidget(channel_group)

        # ROI selection group.
        roi_group = QGroupBox("ROI Selection")
        roi_group.setStyleSheet(STYLE.group_box)
        roi_layout = QVBoxLayout(roi_group)

        roi_group.setToolTip("Click an ROI to select it. Ctrl-click or Shift-click to toggle individual ROIs.")

        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("ROI:"))
        self._roi_edit = QLineEdit()
        self._roi_edit.setFixedWidth(STYLE.edit_width)
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
        rec = self.data.current_recording
        background = self._resolve_background(rec, background_type, coordinate_space, channel_2)
        self._cached_background = self._normalize_image(image=background)

        # Pre-collects all valid mask pixel coordinates and per-ROI colors into the cache.
        masks = self._resolve_masks(rec, mask_layer, channel_2)
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
            frame_height = self.data.single_day.frame_height
            frame_width = self.data.single_day.frame_width

            # Generates deterministic per-ROI colors using random HSV hues with full saturation and value.
            # Original and Deformed layers use the Original mask count as the palette reference so both layers
            # share identical colors within a session (they represent the same ROIs in different coordinate
            # spaces). Template layers use their own count directly, which is identical across all sessions,
            # ensuring consistent colors when switching recordings.
            # Template and Tracked layers share a color palette (same ROI identity set). Original and Deformed
            # layers share a separate palette (same single-day ROIs in different coordinate spaces).
            if mask_layer in (MaskLayer.TEMPLATE, MaskLayer.TRACKED):
                template_masks = self._resolve_masks(rec, MaskLayer.TEMPLATE, channel_2)
                color_count = len(template_masks) if template_masks else len(masks)
            else:
                original_masks = self._resolve_masks(rec, MaskLayer.ORIGINAL, channel_2)
                color_count = len(original_masks) if original_masks else len(masks)

            rng = np.random.default_rng(seed=ROI_CONFIG.random_color_seed)
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
            f"Size: {self.data.single_day.frame_height} x {self.data.single_day.frame_width}"
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

    def _on_dataset_selected(self, index: int) -> None:
        """Handles dataset combo box selection changes by loading the selected dataset.

        Args:
            index: The newly selected combo box index.
        """
        if index < 0:
            return
        dataset_name = self._dataset_combo.itemData(index)
        if dataset_name and dataset_name != self.data.active_dataset_name:
            self.data.load_dataset(dataset_name=dataset_name)
            self.load_data(data=self.data)

    def _on_channel_2_toggled(self, checked: bool) -> None:
        """Updates the channel 2 button style and refreshes the display.

        Args:
            checked: Determines whether channel 2 is toggled on.
        """
        self._channel_2_checkbox.setStyleSheet(STYLE.button_pressed if checked else STYLE.button_unpressed)
        self._refresh_display()

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
        sd = self.data.single_day
        if click_y < 0 or click_y >= sd.frame_height or click_x < 0 or click_x >= sd.frame_width:
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

    def _select_all_rois(self) -> None:
        """Resets the selection to show all ROIs."""
        self._selected_rois = None
        self._roi_edit.clear()
        self._composite_and_display()

    def _deselect_all_rois(self) -> None:
        """Clears the selection so no ROIs are visible."""
        self._selected_rois = set()
        self._roi_edit.clear()
        self._composite_and_display()

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
        self._play_button.setEnabled(False)
        self._pause_button.setEnabled(True)
        self._skip_backward_button.setEnabled(False)
        self._skip_forward_button.setEnabled(False)
        self._auto_cycle_timer.start(TRACKING_CONFIG.cycle_interval)

    def _stop_cycling(self) -> None:
        """Stops automatic recording cycling and re-enables manual navigation."""
        self._auto_cycle_timer.stop()
        self._play_button.setEnabled(True)
        self._pause_button.setEnabled(False)
        self._skip_backward_button.setEnabled(True)
        self._skip_forward_button.setEnabled(True)

    def _normalize_image(self, image: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Normalizes a float32 image to an uint8 RGB array using percentile clipping.

        Args:
            image: The input float32 image. A size-0 array produces a black fallback.

        Returns:
            Normalized RGB image of shape (height, width, 3) with uint8 values.
        """
        frame_height = self.data.single_day.frame_height
        frame_width = self.data.single_day.frame_width

        if image.size == 0:
            return np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        percentile_low = np.percentile(image, COMMON_CONFIG.lower_percentile)
        percentile_high = np.percentile(image, COMMON_CONFIG.upper_percentile)

        if percentile_high <= percentile_low:
            return np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        normalized = (image - percentile_low) / (percentile_high - percentile_low)
        normalized = np.clip(normalized, 0.0, 1.0)
        grayscale = (normalized * 255).astype(np.uint8)

        return np.stack([grayscale, grayscale, grayscale], axis=-1)

    @staticmethod
    def _resolve_background(
        rec: MultiDayData,
        image_type: BackgroundView,
        coordinate_space: CoordinateSpace,
        channel_2: bool,
    ) -> NDArray[np.float32]:
        """Resolves the background image from a recording based on the active view settings.

        Args:
            rec: The multi-day recording to read from.
            image_type: The selected background image type.
            coordinate_space: Native or transformed coordinate space.
            channel_2: Determines whether to return the channel 2 variant.

        Returns:
            The resolved background image. Channel 2 variants return an empty array when single-channel.
        """
        if image_type == BackgroundView.ROIS_ONLY:
            return EMPTY

        if coordinate_space == CoordinateSpace.TRANSFORMED:
            if image_type == BackgroundView.MEAN_IMAGE:
                return rec.transformed_mean_image_channel_2 if channel_2 else rec.transformed_mean_image
            if image_type == BackgroundView.ENHANCED_MEAN_IMAGE:
                return (
                    rec.transformed_enhanced_mean_image_channel_2 if channel_2 else rec.transformed_enhanced_mean_image
                )
            if image_type == BackgroundView.MAXIMUM_PROJECTION:
                return rec.transformed_maximum_projection_channel_2 if channel_2 else rec.transformed_maximum_projection
            return rec.transformed_mean_image_channel_2 if channel_2 else rec.transformed_mean_image

        if image_type == BackgroundView.MEAN_IMAGE:
            return rec.mean_image_channel_2 if channel_2 else rec.mean_image
        if image_type == BackgroundView.ENHANCED_MEAN_IMAGE:
            return rec.enhanced_mean_image_channel_2 if channel_2 else rec.enhanced_mean_image
        if image_type == BackgroundView.MAXIMUM_PROJECTION:
            return rec.maximum_projection_channel_2 if channel_2 else rec.maximum_projection
        if image_type == BackgroundView.CORRELATION_MAP:
            return rec.correlation_map_channel_2 if channel_2 else rec.correlation_map
        return rec.mean_image_channel_2 if channel_2 else rec.mean_image

    @staticmethod
    def _resolve_masks(
        rec: MultiDayData,
        layer: MaskLayer,
        channel_2: bool,
    ) -> Sequence[ROIMask | ROIStatistics]:
        """Resolves the mask list from a recording based on the active mask layer and channel.

        Args:
            rec: The multi-day recording to read from.
            layer: The selected mask layer.
            channel_2: Determines whether to return the channel 2 variant.

        Returns:
            The resolved mask list. Channel 2 variants return an empty list when single-channel.
        """
        if layer == MaskLayer.ORIGINAL:
            return rec.original_masks_channel_2 if channel_2 else rec.original_masks
        if layer == MaskLayer.DEFORMED:
            return rec.deformed_masks_channel_2 if channel_2 else rec.deformed_masks
        if layer == MaskLayer.TEMPLATE:
            return rec.template_masks_channel_2 if channel_2 else rec.template_masks
        if layer == MaskLayer.TRACKED:
            return rec.tracked_masks_channel_2 if channel_2 else rec.tracked_masks
        return []
