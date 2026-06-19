"""Provides the multi-recording tracking viewer window."""

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
    QComboBox,
    QGroupBox,
    QLineEdit,
    QStatusBar,
    QFileDialog,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
)
from matplotlib.colors import hsv_to_rgb
from ataraxis_base_utilities import LogLevel, console

from .styles import FONTS, STYLE, TRACKING_STYLE
from .widgets import escape_returns_focus, create_play_pause_group
from .overlays import normalize_percentile
from .constants import (
    ROI_CONFIG,
    TRACKING_CONFIG,
    MaskLayer,
    BackgroundView,
    CoordinateSpace,
    BackgroundViewLabel,
)
from .viewer_context import EMPTY, ViewerData

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    from ..dataclasses import ROIMask, ROIStatistics
    from .viewer_context import MultiRecordingData


class TrackingViewer(QMainWindow):
    """Displays background images with ROI mask overlays for each recording in a multi-recording dataset, supporting
    manual and automatic recording cycling, coordinate space switching, mask layer selection, channel toggling, and mask
    opacity control.

    Args:
        data: The preloaded tracking data to display on startup.

    Attributes:
        data: The ViewerData instance that stores the visualized dataset's data.
        _auto_cycle_timer: Timer driving automatic recording cycling.
        _cached_background: Normalized background image cache, or None.
        _cached_mask_y: Cached Y pixel coordinates for all valid mask pixels, or None.
        _cached_mask_x: Cached X pixel coordinates for all valid mask pixels, or None.
        _cached_mask_colors: Cached per-pixel RGB colors for all valid mask pixels, or None.
        _cached_mask_roi_indices: Cached per-pixel ROI index for all valid mask pixels, or None.
        _cached_roi_map: ROI ownership map for O(1) click-to-ROI lookup, or None.
        _cached_mask_count: Number of masks in the current layer.
        _selected_rois: Set of selected ROI indices, or None when all ROIs are visible.
        _selection_was_template: Determines whether the last valid selection used a template-group layer.
        _selection_recording_index: Recording index when the selection was last valid.
        _file_button: File menu button with dropdown for loading recordings.
        _graphics_widget: PyQtGraph graphics layout for image display.
        _view_box: View box for the primary image display.
        _image_item: Image item for the composited background and mask display.
        _status_bar: Status bar displaying recording info and selection state.
        _dataset_combo: Dropdown for selecting the active multi-recording dataset.
        _recording_combo: Dropdown for selecting the active recording.
        _skip_backward_button: Button for navigating to the previous recording.
        _play_button: Button to start automatic recording cycling.
        _pause_button: Button to stop automatic recording cycling.
        _skip_forward_button: Button for navigating to the next recording.
        _background_combo: Dropdown for selecting the background image type.
        _space_combo: Dropdown for selecting the coordinate space.
        _mask_combo: Dropdown for selecting the mask layer.
        _opacity_slider: Slider for adjusting mask overlay opacity.
        _channel_2_checkbox: Toggle button for channel 2 overlay display.
        _roi_edit: Read-only input field displaying the index of the last clicked ROI.
    """

    def __init__(self, data: ViewerData) -> None:
        super().__init__()
        self.setWindowTitle("Multi-Recording ROI Tracking")
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
        outer_layout = QVBoxLayout(central_widget)

        # Toolbar row with file menu button.
        toolbar = QHBoxLayout()
        self._file_button: QPushButton = QPushButton("File")
        self._file_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._file_button.setToolTip("Load a multi-recording dataset for visualization.")
        file_menu = QMenu(self)
        file_menu.setStyleSheet(STYLE.menu)
        load_action = file_menu.addAction("&Load dataset")
        load_action.setShortcut("Ctrl+L")
        load_action.triggered.connect(self._load_dataset)
        self._file_button.setMenu(file_menu)
        toolbar.addWidget(self._file_button)
        hint_label = QLabel(
            "Hint: Use arrows to navigate recordings and mask layers, use space to toggle auto-cycling."
        )
        hint_label.setStyleSheet(STYLE.white_label)
        hint_label.setFont(FONTS.small_bold)
        toolbar.addWidget(hint_label)
        toolbar.addStretch()
        outer_layout.addLayout(toolbar)

        # Main content row: image panel + control panel sidebar.
        main_layout = QHBoxLayout()

        # Image + trace display panel (pyqtgraph).
        self._graphics_widget = pg.GraphicsLayoutWidget()
        self._view_box: pg.ViewBox = self._graphics_widget.addViewBox(row=0, col=0)
        self._view_box.setAspectLocked(True)
        self._view_box.invertY(True)
        self._image_item: pg.ImageItem = pg.ImageItem()
        self._view_box.addItem(self._image_item)
        main_layout.addWidget(self._graphics_widget, stretch=3)
        self._graphics_widget.scene().sigMouseClicked.connect(self._on_image_clicked)

        # Control panel (right sidebar).
        control_panel = self._build_control_panel()
        main_layout.addWidget(control_panel, stretch=0)

        outer_layout.addLayout(main_layout, stretch=1)

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
        for index in range(self._dataset_combo.count()):
            if self._dataset_combo.itemData(index) == active:
                self._dataset_combo.setCurrentIndex(index)
                break
        self._dataset_combo.blockSignals(False)

        # Populates the recording selector.
        self._recording_combo.blockSignals(True)
        self._recording_combo.clear()
        for index, recording_id in enumerate(data.recording_ids):
            self._recording_combo.addItem(f"{index}: {recording_id}", userData=index)
        self._recording_combo.setCurrentIndex(0)
        self._recording_combo.blockSignals(False)

        # Shows channel 2 group only if channel 2 data exists.
        self._channel_group.setVisible(data.current_recording.has_channel_2)

        # Updates the window title to reflect the active dataset.
        self.setWindowTitle(f"Multi-Recording ROI Tracking — {data.dataset_name}")

        self._refresh_display()

    def get_state(self) -> dict[str, Any]:
        """Returns the current display state of the tracking viewer for cross-process state exchange.

        Returns:
            A dictionary containing the viewer type, active display settings, and selection state.
        """
        if not self.data.is_multi_recording:
            return {"viewer_type": "tracking", "loaded": False}

        return {
            "viewer_type": "tracking",
            "loaded": True,
            "active_dataset": self._dataset_combo.currentText(),
            "available_datasets": list(self.data.available_datasets),
            "current_recording_index": self._recording_combo.currentIndex(),
            "current_recording_id": self.data.current_recording_id,
            "recording_count": self._recording_combo.count(),
            "background_view": BackgroundView(self._background_combo.currentData()).name.lower(),
            "coordinate_space": CoordinateSpace(self._space_combo.currentData()).name.lower(),
            "mask_layer": MaskLayer(self._mask_combo.currentData()).name.lower(),
            "channel_2_active": self._channel_2_checkbox.isChecked(),
            "opacity": self._opacity_slider.value(),
            "selected_roi_indices": sorted(self._selected_rois) if self._selected_rois is not None else None,
            "mask_count": self._cached_mask_count,
            "auto_cycling": self._auto_cycle_timer.isActive(),
        }

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        """Handles keyboard navigation for recording stepping, opacity controls, and auto-cycle toggling.

        Notes:
            Overrides the Qt virtual method. The camelCase name is required to match the parent signature.
        """
        # Left/right arrow keys step through recordings when auto-cycling is stopped.
        if self._play_button.isEnabled():
            if event.key() == QtCore.Qt.Key.Key_Left:
                self._previous_recording()
            elif event.key() == QtCore.Qt.Key.Key_Right:
                self._next_recording()

        # Up/down arrow keys cycle through mask layers.
        if event.key() == QtCore.Qt.Key.Key_Up:
            index = self._mask_combo.currentIndex()
            if index > 0:
                self._mask_combo.setCurrentIndex(index - 1)
        elif event.key() == QtCore.Qt.Key.Key_Down:
            index = self._mask_combo.currentIndex()
            if index < self._mask_combo.count() - 1:
                self._mask_combo.setCurrentIndex(index + 1)

        # Spacebar toggles between play and pause.
        if event.key() == QtCore.Qt.Key.Key_Space:
            if self._play_button.isEnabled():
                self._start_cycling()
            else:
                self._stop_cycling()

    def eventFilter(self, source: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        """Returns focus to the main window when Escape is pressed inside an edit field.

        Notes:
            Overrides the Qt virtual method. The camelCase name is required to match the parent signature.
        """
        if escape_returns_focus(self, event):
            return True
        return super().eventFilter(source, event)

    def _load_dataset(self) -> None:
        """Displays a file dialog that allows users to select a new multi-recording dataset to visualize."""
        # Defaults the file dialog to the parent of the currently loaded recording's output
        # directory, so the user can easily navigate to a sibling recording.
        start_directory = ""
        output = self.data.single_recording.output_path
        parent = output.parent
        if parent.is_dir():
            start_directory = str(parent)

        directory = QFileDialog.getExistingDirectory(
            self, "Select any recording directory from the dataset", start_directory
        )
        if not directory:
            return

        recording_path = Path(directory)
        console.echo(message=f"Loading dataset from recording: {recording_path}.")

        try:
            data = ViewerData.from_data(root_path=recording_path)
            if not data.is_multi_recording and data.available_datasets:
                data.load_dataset(dataset_name=data.available_datasets[0])
        except Exception:
            console.echo(message="Unable to load dataset.", level=LogLevel.ERROR)
            result = QMessageBox.question(
                self,
                "ERROR",
                "Unable to load dataset. Try another directory?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if result == QMessageBox.StandardButton.Yes:
                self._load_dataset()
            return

        self.load_data(data=data)

    def _build_control_panel(self) -> QWidget:
        """Constructs the right-side control panel with all viewer controls.

        Returns:
            The assembled control panel widget.
        """
        panel = QWidget(self)
        layout = QVBoxLayout(panel)
        panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)

        # Dataset selector group. Lists all discovered multi-recording datasets for runtime switching.
        dataset_group = QGroupBox("Dataset")
        dataset_group.setStyleSheet(STYLE.group_box)
        dataset_layout = QVBoxLayout(dataset_group)
        self._dataset_combo = QComboBox()
        self._dataset_combo.setToolTip("Select the active multi-recording dataset.")
        self._dataset_combo.currentIndexChanged.connect(self._on_dataset_selected)
        dataset_layout.addWidget(self._dataset_combo)
        layout.addWidget(dataset_group)

        # Recording navigation group.
        recording_group = QGroupBox("Recording Navigation")
        recording_group.setStyleSheet(STYLE.group_box)
        recording_layout = QVBoxLayout(recording_group)

        self._recording_combo = QComboBox()
        self._recording_combo.setToolTip("Select the active recording.")
        self._recording_combo.currentIndexChanged.connect(self._on_recording_selected)
        recording_layout.addWidget(self._recording_combo)

        # Playback controls. Play and pause are grouped exclusively so only one can be active at a time.
        navigation_row = QHBoxLayout()
        icon_size = QtCore.QSize(STYLE.icon_size, STYLE.icon_size)

        self._skip_backward_button: QToolButton = QToolButton()
        self._skip_backward_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward))
        self._skip_backward_button.setIconSize(icon_size)
        self._skip_backward_button.setToolTip("Go to the previous recording.")
        self._skip_backward_button.clicked.connect(self._previous_recording)

        playback = create_play_pause_group(
            self,
            play_tooltip="Start automatic recording cycling.",
            pause_tooltip="Stop automatic recording cycling.",
        )
        self._play_button = playback.play_button
        self._pause_button = playback.pause_button
        self._play_button.clicked.connect(self._start_cycling)
        self._pause_button.clicked.connect(self._stop_cycling)

        self._skip_forward_button: QToolButton = QToolButton()
        self._skip_forward_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward))
        self._skip_forward_button.setIconSize(icon_size)
        self._skip_forward_button.setToolTip("Go to the next recording.")
        self._skip_forward_button.clicked.connect(self._next_recording)

        navigation_row.addWidget(self._skip_backward_button)
        navigation_row.addWidget(self._play_button)
        navigation_row.addWidget(self._pause_button)
        navigation_row.addWidget(self._skip_forward_button)

        # Overrides the default disabled state: tracking viewer starts ready to auto-cycle.
        self._play_button.setEnabled(True)
        recording_layout.addLayout(navigation_row)

        layout.addWidget(recording_group)

        # Background image group.
        background_group = QGroupBox("Background Image")
        background_group.setStyleSheet(STYLE.group_box)
        background_layout = QVBoxLayout(background_group)

        self._background_combo = QComboBox()
        self._background_combo.setToolTip(
            "Select the background image displayed behind the ROI overlay. 'ROIs' shows masks on a black "
            "background. Other options show the corresponding detection image."
        )
        self._background_combo.addItem(BackgroundViewLabel.ROIS_ONLY, userData=BackgroundView.ROIS_ONLY)
        self._background_combo.addItem(BackgroundViewLabel.MEAN_IMAGE, userData=BackgroundView.MEAN_IMAGE)
        self._background_combo.addItem(
            BackgroundViewLabel.ENHANCED_MEAN_IMAGE, userData=BackgroundView.ENHANCED_MEAN_IMAGE
        )
        self._background_combo.addItem(
            BackgroundViewLabel.MAXIMUM_PROJECTION, userData=BackgroundView.MAXIMUM_PROJECTION
        )
        self._background_combo.addItem(BackgroundViewLabel.CORRELATION_MAP, userData=BackgroundView.CORRELATION_MAP)
        self._background_combo.currentIndexChanged.connect(self._refresh_display)
        background_layout.addWidget(self._background_combo)

        layout.addWidget(background_group)

        # Coordinate space group.
        space_group = QGroupBox("Coordinate Space")
        space_group.setStyleSheet(STYLE.group_box)
        space_layout = QVBoxLayout(space_group)

        self._space_combo = QComboBox()
        self._space_combo.setToolTip(
            "Select the coordinate space. 'Native' shows masks in the recording's original coordinates. "
            "'Transformed' shows masks in the template coordinate space."
        )
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
        self._mask_combo.setToolTip("Select the mask layer to display.")
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
        self._opacity_slider.setToolTip("Adjust mask opacity.")
        self._opacity_slider.valueChanged.connect(self._on_opacity_changed)
        opacity_row.addWidget(self._opacity_slider)
        mask_layout.addLayout(opacity_row)

        layout.addWidget(mask_group)

        # Channel group. Hidden when channel 2 data does not exist.
        self._channel_group = QGroupBox("Channel")
        self._channel_group.setStyleSheet(STYLE.group_box)
        channel_layout = QVBoxLayout(self._channel_group)

        self._channel_2_checkbox = QPushButton("Channel 2")
        self._channel_2_checkbox.setCheckable(True)
        self._channel_2_checkbox.setToolTip(
            "Toggle display between channel 1 and channel 2 data. When active, background images and ROI masks "
            "switch to the channel 2 variants."
        )
        self._channel_2_checkbox.toggled.connect(self._on_channel_2_toggled)
        channel_layout.addWidget(self._channel_2_checkbox)

        self._channel_group.setVisible(False)
        layout.addWidget(self._channel_group)

        # ROI selection group.
        roi_group = QGroupBox("ROI Selection")
        roi_group.setStyleSheet(STYLE.group_box)
        roi_layout = QVBoxLayout(roi_group)

        roi_group.setToolTip("Click an ROI to select it. Ctrl-click or Shift-click to toggle individual ROIs.")

        input_row = QHBoxLayout()
        input_row.addWidget(QLabel("ROI:"))
        self._roi_edit = QLineEdit()
        self._roi_edit.setFixedWidth(STYLE.edit_width)
        self._roi_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self._roi_edit.setToolTip("Enter an ROI index to select it.")
        self._roi_edit.returnPressed.connect(self._on_roi_entered)
        self._roi_edit.returnPressed.connect(self.setFocus)
        self._roi_edit.installEventFilter(self)
        input_row.addWidget(self._roi_edit)
        input_row.addStretch()
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
        # main window's keyPressEvent. The ROI edit field is re-enabled so users can type ROI indices.
        for child in panel.findChildren(QWidget):
            child.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._roi_edit.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

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
        recording = self.data.current_recording
        background = self._resolve_background(recording, background_type, coordinate_space, channel_2)
        self._cached_background = self._normalize_image(image=background)

        # Pre-collects all valid mask pixel coordinates and per-ROI colors into the cache.
        masks = self._resolve_masks(recording, mask_layer, channel_2)
        self._cached_mask_count = len(masks) if masks else 0
        self._roi_edit.setValidator(QtGui.QIntValidator(0, max(0, self._cached_mask_count - 1)))
        self._cached_mask_y = None
        self._cached_mask_x = None
        self._cached_mask_colors = None
        self._cached_mask_roi_indices = None
        self._cached_roi_map = None

        # Determines whether the ROI identity set has changed, requiring a selection reset. Template and Tracked
        # layers share the same ROI identity set (template-derived) so the selection persists across recording
        # switches and Template/Tracked toggles. Original and Deformed share a separate identity set (single-recording
        # extraction) so the selection persists only within the same recording.
        current_is_template_group = mask_layer in (MaskLayer.TEMPLATE, MaskLayer.TRACKED)
        recording_index = self.data.current_recording_index
        layer_group_changed = current_is_template_group != self._selection_was_template
        recording_changed = not current_is_template_group and recording_index != self._selection_recording_index
        if layer_group_changed or recording_changed:
            self._selected_rois = None
            self._roi_edit.clear()
        elif self._selected_rois is not None and self._cached_mask_count > 0:
            # Clamps the selection to valid indices in case the mask count differs.
            self._selected_rois = {
                roi_index for roi_index in self._selected_rois if roi_index < self._cached_mask_count
            }
        self._selection_was_template = current_is_template_group
        self._selection_recording_index = recording_index

        if masks:
            frame_height = self.data.single_recording.frame_height
            frame_width = self.data.single_recording.frame_width

            # Generates deterministic per-ROI colors using random HSV hues with full saturation and value.
            # Original and Deformed layers use the Original mask count as the palette reference so both layers
            # share identical colors within a recording (they represent the same ROIs in different coordinate
            # spaces). Template layers use their own count directly, which is identical across all recordings,
            # ensuring consistent colors when switching recordings.
            # Template and Tracked layers share a color palette (same ROI identity set). Original and Deformed
            # layers share a separate palette (same single-recording ROIs in different coordinate spaces).
            if mask_layer in (MaskLayer.TEMPLATE, MaskLayer.TRACKED):
                template_masks = self._resolve_masks(recording, MaskLayer.TEMPLATE, channel_2)
                color_count = len(template_masks) if template_masks else len(masks)
            else:
                original_masks = self._resolve_masks(recording, MaskLayer.ORIGINAL, channel_2)
                color_count = len(original_masks) if original_masks else len(masks)

            rng = np.random.default_rng(seed=ROI_CONFIG.random_color_seed)
            hues = rng.random(color_count).astype(np.float32)
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
            f"Size: {self.data.single_recording.frame_height} x {self.data.single_recording.frame_width}"
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
        single_recording = self.data.single_recording
        if (
            click_y < 0
            or click_y >= single_recording.frame_height
            or click_x < 0
            or click_x >= single_recording.frame_width
        ):
            return

        roi_index = int(self._cached_roi_map[click_y, click_x])
        if roi_index < 0:
            return

        # Determines whether to toggle (Ctrl/Shift held) or replace selection (plain click).
        modifiers = event.modifiers()  # type: ignore[attr-defined]
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

    def _on_roi_entered(self) -> None:
        """Selects the ROI whose index was typed into the ROI edit field."""
        text = self._roi_edit.text().strip()
        if not text:
            return
        try:
            roi_index = int(text)
        except ValueError:
            return
        if 0 <= roi_index < self._cached_mask_count:
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
        normalized = normalize_percentile(
            image=image,
            frame_height=self.data.single_recording.frame_height,
            frame_width=self.data.single_recording.frame_width,
        )
        grayscale = (normalized * 255).astype(np.uint8)
        return np.stack([grayscale, grayscale, grayscale], axis=-1)

    @staticmethod
    def _resolve_background(
        recording: MultiRecordingData,
        image_type: BackgroundView,
        coordinate_space: CoordinateSpace,
        channel_2: bool,
    ) -> NDArray[np.float32]:
        """Resolves the background image from a recording based on the active view settings.

        Args:
            recording: The multi-recording recording to read from.
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
                return recording.transformed_mean_image_channel_2 if channel_2 else recording.transformed_mean_image
            if image_type == BackgroundView.ENHANCED_MEAN_IMAGE:
                return (
                    recording.transformed_enhanced_mean_image_channel_2
                    if channel_2
                    else recording.transformed_enhanced_mean_image
                )
            if image_type == BackgroundView.MAXIMUM_PROJECTION:
                return (
                    recording.transformed_maximum_projection_channel_2
                    if channel_2
                    else recording.transformed_maximum_projection
                )
            return recording.transformed_mean_image_channel_2 if channel_2 else recording.transformed_mean_image

        if image_type == BackgroundView.MEAN_IMAGE:
            return recording.mean_image_channel_2 if channel_2 else recording.mean_image
        if image_type == BackgroundView.ENHANCED_MEAN_IMAGE:
            return recording.enhanced_mean_image_channel_2 if channel_2 else recording.enhanced_mean_image
        if image_type == BackgroundView.MAXIMUM_PROJECTION:
            return recording.maximum_projection_channel_2 if channel_2 else recording.maximum_projection
        if image_type == BackgroundView.CORRELATION_MAP:
            return recording.correlation_map_channel_2 if channel_2 else recording.correlation_map
        return recording.mean_image_channel_2 if channel_2 else recording.mean_image

    @staticmethod
    def _resolve_masks(
        recording: MultiRecordingData,
        layer: MaskLayer,
        channel_2: bool,
    ) -> Sequence[ROIMask | ROIStatistics]:
        """Resolves the mask list from a recording based on the active mask layer and channel.

        Args:
            recording: The multi-recording recording to read from.
            layer: The selected mask layer.
            channel_2: Determines whether to return the channel 2 variant.

        Returns:
            The resolved mask list. Channel 2 variants return an empty list when single-channel.
        """
        if layer == MaskLayer.ORIGINAL:
            return recording.original_masks_channel_2 if channel_2 else recording.original_masks
        if layer == MaskLayer.DEFORMED:
            return recording.deformed_masks_channel_2 if channel_2 else recording.deformed_masks
        if layer == MaskLayer.TEMPLATE:
            return recording.template_masks_channel_2 if channel_2 else recording.template_masks
        if layer == MaskLayer.TRACKED:
            return recording.tracked_masks_channel_2 if channel_2 else recording.tracked_masks
        return []
