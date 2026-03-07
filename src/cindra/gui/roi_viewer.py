"""Provides the ROI viewer window for inspecting and reclassifying single-recording pipeline results."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
from PySide6 import QtGui, QtCore
import pyqtgraph as pg  # type: ignore[import-untyped]
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QMenu,
    QLabel,
    QSlider,
    QWidget,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLineEdit,
    QSplitter,
    QStatusBar,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)
from matplotlib.colors import hsv_to_rgb
from ataraxis_base_utilities import LogLevel, console

from .styles import FONTS, STYLE, COLORS, ROI_STYLE
from .widgets import ViewBox, TraceBox, plot_trace, add_plot_legend
from .overlays import (
    flip_rois,
    draw_masks,
    build_views,
    display_masks,
    display_views,
    draw_colorbar,
    compute_colors,
    render_colorbar,
    update_colormap,
    initialize_roi_maps,
    update_correlation_masks,
    recompute_binary_classification,
)
from .constants import (
    ROI_CONFIG,
    Colormap,
    TraceMode,
    ROIColorMode,
    BackgroundView,
    TraceModeLabel,
    ROIColorModeLabel,
    BackgroundViewLabel,
)
from .data_models import (
    ColorArrays,
    ROIIndexMaps,
    ColorControls,
    TraceControls,
    ColorbarWidgets,
    ClassifierControls,
)
from .viewer_context import EMPTY, ViewerData
from ..classification import Classifier

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..dataclasses import ROIStatistics

# Statistics attribute names displayed in the info bar. Centroid is handled separately via roi.mask.centroid.
_STATISTICS_TO_SHOW: tuple[str, ...] = (
    "pixel_count",
    "skewness",
    "compactness",
    "footprint",
    "aspect_ratio",
)


class ROIViewer(QMainWindow):
    """Displays a UI window for inspecting and reclassifying single-recording pipeline results.

    Displays ROI overlays, background images, and fluorescence traces. Supports left-click ROI selection,
    shift/ctrl multi-select, right-click cell/non-cell reclassification (active only in cell/non-cell color mode) with
    auto-save, keyboard shortcuts for view/color switching, and double-click zoom-to-fit.

    Args:
        data: The preloaded viewer data to display on startup.

    Attributes:
        _roi_color_mode: Active ROI color statistic index.
        _background_view: Active background image type index.
        _roi_colormap: Active matplotlib colormap name for ROI coloring.
        _selected_roi_index: Index of the most recently selected ROI.
        _selected_roi_indices: List of currently selected ROI indices.
        _temporal_bin_size: Number of frames per temporal bin for activity correlation.
        _recording_loaded: Determines whether a recording has been fully loaded and initialized.
        _colocalization_threshold: Probability threshold for channel 2 colocalization display.
        _last_reclassified_index: Index of the most recently reclassified ROI, or -1 if none.
        _classify_mode: Determines whether classifier mode is active (clicks flip ROI labels instead of selecting).
        _pre_classify_color_mode: Stores the color mode before classify was toggled on, for restoration on toggle off.
        _all_recordings_visible: Determines whether the stacked all-recordings trace view is active.
        _context_data: The ViewerData instance that stores the visualized recording's data.
        _color_arrays: Precomputed per-ROI color arrays for each color mode, or None.
        _roi_maps: Precomputed ROI index maps for click-to-ROI and label lookup, or None.
        _colorbar_widgets: PyQtGraph colorbar image, labels, and container widget, or None.
        _colorbar_image: Rendered colorbar image array, or None.
        _views: Precomputed background view image stack, or None.
        _roi_statistics: List of ROIStatistics instances for the current recording.
        _cell_classification: Cell classification probability array with shape (roi_count, 2).
        _cell_colocalization: Channel 2 colocalization probability array with shape (roi_count, 2).
        _two_channels: Determines whether the current recording has channel 2 data.
        _cell_fluorescence: Raw cell fluorescence traces array, or an empty array.
        _neuropil_fluorescence: Neuropil fluorescence traces array, or an empty array.
        _subtracted_fluorescence: Neuropil-subtracted fluorescence traces array, or an empty array.
        _spikes: Deconvolved spike rate traces array, or an empty array.
        _frame_count: Number of frames in the current recording.
        _roi_count: Number of detected ROIs in the current recording.
        _binned_fluorescence: Temporally binned fluorescence array for correlation coloring, or None.
        _fluorescence_standard_deviation: Per-ROI standard deviation of binned fluorescence, or None.
        _frame_indices: Frame index array for trace x-axis, or None.
        _graphics_splitter: Vertical splitter separating the image and trace panels.
        _image_widget: PyQtGraph graphics layout for the ROI image panel.
        _trace_widget: PyQtGraph graphics layout for the fluorescence trace panel.
        _status_bar: Status bar displaying recording info and selection state.
        _view_box: View box for the primary ROI image display.
        _background: Image item for the background image display.
        _overlay: Image item for the ROI mask overlay display.
        _trace_box: Trace plot box for fluorescence trace display.
        _roi_source_group: Group box for the multi-recording ROI source selector.
        _roi_source_combo: Dropdown for selecting single-recording or multi-recording ROI source.
        _roi_selection_group: Group box for ROI selection controls.
        _colors_group: Group box for Mask Colors controls.
        _trace_group: Group box for Trace Display controls.
        _top_button: Button for selecting top-n ranked ROIs by the active color statistic.
        _bottom_button: Button for selecting bottom-n ranked ROIs by the active color statistic.
        _ranked_count_edit: Input field for the number of top/bottom ROIs to select.
        _background_combo: Dropdown for selecting the background image type.
        _channel_2_button: Checkable button for toggling channel 2 overlay display.
        _color_controls: ROI color mode dropdown, colormap chooser, threshold, and temporal bins controls.
        _trace_controls: Trace display mode, visibility checkboxes, and max plotted controls.
        _classifier_controls: Classifier builder panel with Classify toggle, New, and Add to Existing buttons.
        _roi_index_edit: Input field for jumping to a specific ROI by number in the ROI info bar.
        _all_recordings_button: Toggle button for stacked all-recordings trace display.
    """

    def __init__(self, data: ViewerData) -> None:
        super().__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")

        # Display state fields.
        self._roi_color_mode: int = ROIColorMode.RANDOM
        self._background_view: int = BackgroundView.ROIS_ONLY
        self._roi_colormap: str = Colormap.HSV
        self._selected_roi_index: int = 0
        self._selected_roi_indices: list[int] = [0]
        self._temporal_bin_size: int = 1
        self._recording_loaded: bool = False
        self._colocalization_threshold: float = ROI_CONFIG.default_channel_2_threshold
        self._last_reclassified_index: int = -1
        self._classify_mode: bool = False
        self._pre_classify_color_mode: int = ROIColorMode.RANDOM
        self._saved_opacity: int = STYLE.default_mask_opacity

        # Multi-recording state. Persists across _reset_state calls.
        self._all_recordings_visible: bool = False

        # Core data objects.
        self._context_data: ViewerData | None = None
        self._color_arrays: ColorArrays | None = None
        self._roi_maps: ROIIndexMaps | None = None
        self._colorbar_widgets: ColorbarWidgets | None = None
        self._colorbar_image: NDArray[np.uint8] | None = None
        self._views: NDArray[np.uint8] | None = None

        # Mode-dependent data cache. Populated by _initialize_gui from either single_recording or current_recording.
        self._roi_statistics: list[ROIStatistics] = []
        self._cell_classification: NDArray[np.float32] = EMPTY
        self._cell_colocalization: NDArray[np.float32] = EMPTY
        self._two_channels: bool = False
        self._cell_fluorescence: NDArray[np.float32] = EMPTY
        self._neuropil_fluorescence: NDArray[np.float32] = EMPTY
        self._subtracted_fluorescence: NDArray[np.float32] = EMPTY
        self._spikes: NDArray[np.float32] = EMPTY
        self._frame_count: int = 0
        self._roi_count: int = 0

        # Binned activity state used by correlation coloring.
        self._binned_fluorescence: NDArray[np.float32] | None = None
        self._fluorescence_standard_deviation: NDArray[np.float32] | None = None
        self._frame_indices: NDArray[np.int32] | None = None

        # Window geometry and title.
        self.setGeometry(*ROI_STYLE.window_geometry)
        self.setWindowTitle("ROI Viewer")

        # Main widget layout: toolbar, graphics | control panel, ROI info bar at the bottom.
        central_widget = QWidget(self)
        outer_layout = QVBoxLayout(central_widget)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)
        self.setCentralWidget(central_widget)

        # Toolbar row with File menu button.
        self._build_toolbar(outer_layout)

        content_widget = QWidget()
        main_layout = QHBoxLayout(content_widget)
        outer_layout.addWidget(content_widget, stretch=1)

        # Left: graphics panel split vertically between image and trace with equal initial allocation.
        self._graphics_splitter = QSplitter(QtCore.Qt.Orientation.Vertical)
        self._image_widget = pg.GraphicsLayoutWidget()
        self._trace_widget = pg.GraphicsLayoutWidget()
        self._graphics_splitter.addWidget(self._image_widget)
        self._graphics_splitter.addWidget(self._trace_widget)
        self._graphics_splitter.setStretchFactor(0, 1)
        self._graphics_splitter.setStretchFactor(1, 1)
        main_layout.addWidget(self._graphics_splitter, stretch=3)

        # Right: control panel.
        control_panel = self._build_control_panel()
        main_layout.addWidget(control_panel, stretch=0)

        # Selected ROI info bar, placed between the main content and the status bar.
        self._build_roi_info_bar(outer_layout)

        # Status bar.
        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)

        # Builds graphics panels.
        self._build_graphics()

        # Prevents control panel widgets from capturing keyboard focus so spacebar and arrow keys always reach the
        # main window's keyPressEvent. Edit fields are re-enabled so users can click to type values.
        for widget in control_panel.findChildren(QWidget):
            widget.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        for edit_field in (
            self._roi_index_edit,
            self._ranked_count_edit,
            self._color_controls.threshold_edit,
            self._color_controls.binning_edit,
            self._trace_controls.maximum_trace_count_edit,
        ):
            edit_field.setFocusPolicy(QtCore.Qt.FocusPolicy.ClickFocus)

        # Accepts drag-and-drop of directories.
        self.setAcceptDrops(True)

        # Populates the UI with the startup data provided by the caller.
        self.load_data(data=data)

        self.show()
        self._image_widget.show()
        self._trace_widget.show()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        """Handles keyboard controls for mask opacity, color mode, and colormap cycling.

        Space toggles opacity between zero and the previous slider value. Left/right arrow keys cycle through enabled
        color modes. Up/down arrow keys cycle through colormaps.

        Notes:
            Overrides the Qt virtual method. The camelCase name is required to match the parent signature.
        """
        slider = self._color_controls.opacity_slider
        if event.key() == QtCore.Qt.Key.Key_Space:
            if slider.value() > 0:
                self._saved_opacity = slider.value()
                slider.setValue(0)
            else:
                slider.setValue(self._saved_opacity)
        elif event.key() in (QtCore.Qt.Key.Key_Up, QtCore.Qt.Key.Key_Down):
            chooser = self._color_controls.colormap_chooser
            count = chooser.count()
            if count > 0:
                step = -1 if event.key() == QtCore.Qt.Key.Key_Up else 1
                chooser.setCurrentIndex((chooser.currentIndex() + step) % count)
                self._on_color_changed(self._roi_color_mode)
        elif event.key() in (QtCore.Qt.Key.Key_Left, QtCore.Qt.Key.Key_Right):
            combo = self._color_controls.color_combo
            if not combo.isEnabled():
                return
            count = combo.count()
            step = 1 if event.key() == QtCore.Qt.Key.Key_Right else -1
            model = combo.model()
            current = combo.currentIndex()
            for _ in range(count):
                current = (current + step) % count
                item = model.item(current) if isinstance(model, QStandardItemModel) else None
                if item is None or item.isEnabled():
                    combo.setCurrentIndex(current)
                    self._on_color_changed(current)
                    break

    def eventFilter(self, source: QtCore.QObject, event: QtCore.QEvent) -> bool:  # noqa: N802
        """Returns focus to the main window when Escape is pressed inside an edit field.

        Notes:
            Overrides the Qt virtual method. The camelCase name is required to match the parent signature.
        """
        if (
            event.type() == QtCore.QEvent.Type.KeyPress
            and isinstance(event, QtGui.QKeyEvent)
            and event.key() == QtCore.Qt.Key.Key_Escape
        ):
            self.setFocus()
            return True
        return super().eventFilter(source, event)

    def load_data(self, data: ViewerData) -> None:
        """Caches the input ViewerData instance and uses it to populate the managed UI window.

        Args:
            data: The ViewerData instance that stores the visualized recording's data.
        """
        self._context_data = data

        # Populates the ROI Source combo with "Original" + discovered datasets.
        self._roi_source_combo.blockSignals(True)
        self._roi_source_combo.clear()
        self._roi_source_combo.addItem("Original")
        for name in data.available_datasets:
            self._roi_source_combo.addItem(f"{name} Dataset")

        # Always starts in "Original" (single-recording) view.
        self._roi_source_combo.setCurrentIndex(0)
        self._roi_source_combo.blockSignals(False)
        self._roi_source_group.setVisible(bool(data.available_datasets))

        self._reset_state()
        self._initialize_gui()

    @property
    def _is_multi_recording(self) -> bool:
        """Returns True when the viewer is displaying multi-recording tracked ROI data."""
        return self._context_data is not None and self._context_data.is_multi_recording

    def _build_toolbar(self, parent_layout: QVBoxLayout) -> None:
        """Builds the toolbar row with the File menu button.

        Args:
            parent_layout: The outer vertical layout to prepend the toolbar to.
        """
        toolbar = QHBoxLayout()

        # File menu button with dropdown for loading recordings.
        file_button = QPushButton("File")
        file_button.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        file_button.setToolTip("Load a recording for visualization.")
        file_menu = QMenu(self)
        file_menu.setStyleSheet(STYLE.menu)
        load_action = file_menu.addAction("&Load recording")
        load_action.setShortcut("Ctrl+L")
        load_action.triggered.connect(self._load_recording)
        file_button.setMenu(file_menu)
        toolbar.addWidget(file_button)
        hint_label = QLabel("Hint: Use arrows to change color mode and colormap, use space to toggle masks.")
        hint_label.setStyleSheet(STYLE.white_label)
        hint_label.setFont(FONTS.small_bold)
        toolbar.addWidget(hint_label)
        toolbar.addStretch()
        parent_layout.addLayout(toolbar)

    def _build_control_panel(self) -> QWidget:
        """Constructs the right-side control panel with all grouped viewer controls.

        Returns:
            The control panel widget containing all grouped controls.
        """
        panel = QWidget()
        panel.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
        layout = QVBoxLayout(panel)

        # ROI Source selector (Original / Dataset: ...). Hidden when no multi-recording datasets exist.
        self._roi_source_group = QGroupBox("ROI Source")
        self._roi_source_group.setStyleSheet(STYLE.group_box)
        roi_source_layout = QVBoxLayout(self._roi_source_group)
        self._roi_source_combo: QComboBox = QComboBox(self)
        self._roi_source_combo.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._roi_source_combo.setToolTip(
            "Select the ROI source. 'Original' uses single-recording detected ROIs. Dataset entries use "
            "tracked ROIs from the multi-recording pipeline projected back to this recording's native "
            "coordinate space."
        )
        self._roi_source_combo.activated.connect(self._on_dataset_source_changed)
        roi_source_layout.addWidget(self._roi_source_combo)
        self._roi_source_group.setVisible(False)
        layout.addWidget(self._roi_source_group)

        # ROI selection controls.
        self._roi_selection_group = QGroupBox("ROI Selection")
        roi_group = self._roi_selection_group
        roi_group.setStyleSheet(STYLE.group_box)
        roi_group.setToolTip("Click an ROI to select it. Ctrl-click or Shift-click to toggle individual ROIs.")
        roi_layout = QVBoxLayout(roi_group)

        # Editable fields row: ROI index selector and ranked selection count.
        fields_row = QHBoxLayout()
        fields_row.addWidget(QLabel("ROI:"))
        self._roi_index_edit = QLineEdit()
        self._roi_index_edit.setFixedWidth(STYLE.edit_width)
        self._roi_index_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self._roi_index_edit.setToolTip("Enter an ROI index to select it.")
        self._roi_index_edit.returnPressed.connect(self._on_number_chosen)
        self._roi_index_edit.returnPressed.connect(self.setFocus)
        self._roi_index_edit.installEventFilter(self)
        fields_row.addWidget(self._roi_index_edit)
        fields_row.addWidget(QLabel("Autoselection #:"))
        self._ranked_count_edit = QLineEdit()
        self._ranked_count_edit.setValidator(QtGui.QIntValidator(1, ROI_CONFIG.top_selection_count))
        self._ranked_count_edit.setText(str(ROI_CONFIG.top_selection_count))
        self._ranked_count_edit.setFixedWidth(STYLE.edit_width)
        self._ranked_count_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft)
        self._ranked_count_edit.setToolTip("Set the number of top/bottom ROIs to auto-select.")
        self._ranked_count_edit.returnPressed.connect(self.setFocus)
        self._ranked_count_edit.installEventFilter(self)
        fields_row.addWidget(self._ranked_count_edit)
        fields_row.addStretch()
        roi_layout.addLayout(fields_row)

        # Selection buttons: All/None on the first row, Top/Bottom on the second.
        all_button = QPushButton("All")
        all_button.setToolTip("Select all ROIs.")
        all_button.clicked.connect(self._select_all_rois)
        none_button = QPushButton("None")
        none_button.setToolTip("Deselect all ROIs.")
        none_button.clicked.connect(self._deselect_all_rois)
        self._top_button = QPushButton("Top")
        self._top_button.setToolTip("Select the top-n ROIs ranked by the active color statistic.")
        self._top_button.clicked.connect(lambda: self._on_ranked_selection(top=True))
        self._bottom_button = QPushButton("Bottom")
        self._bottom_button.setToolTip("Select the bottom-n ROIs ranked by the active color statistic.")
        self._bottom_button.clicked.connect(lambda: self._on_ranked_selection(top=False))

        button_grid = QGridLayout()
        button_grid.addWidget(all_button, 0, 0)
        button_grid.addWidget(none_button, 0, 1)
        button_grid.addWidget(self._top_button, 1, 0)
        button_grid.addWidget(self._bottom_button, 1, 1)
        roi_layout.addLayout(button_grid)

        layout.addWidget(roi_group)

        # Background image controls.
        background_box, self._background_combo = self._create_view_controls()
        layout.addWidget(background_box)

        # Channel toggle.
        channel_group = QGroupBox("Channel")
        channel_group.setStyleSheet(STYLE.group_box)
        channel_layout = QVBoxLayout(channel_group)

        self._channel_2_button = QPushButton("Channel 2")
        self._channel_2_button.setCheckable(True)
        self._channel_2_button.setEnabled(False)
        self._channel_2_button.setToolTip(
            "Toggle display between channel 1 and channel 2 data. When active, background images, ROI masks, and "
            "fluorescence traces switch to the channel 2 variants."
        )
        self._channel_2_button.toggled.connect(self._on_channel_2_toggled)
        channel_layout.addWidget(self._channel_2_button)

        layout.addWidget(channel_group)

        # ROI color controls and colorbar.
        self._colors_group, self._color_controls = self._create_color_controls()
        colors_box = self._colors_group
        self._colorbar_widgets = self._create_colorbar()
        colors_box.layout().addWidget(self._colorbar_widgets.widget)  # type: ignore[union-attr]
        layout.addWidget(colors_box)

        # Trace display controls.
        self._trace_group, self._trace_controls = self._create_trace_controls()
        layout.addWidget(self._trace_group)

        # Classifier builder controls.
        classifier_box, self._classifier_controls = self._create_classifier_controls()
        layout.addWidget(classifier_box)

        layout.addStretch()
        return panel

    def _build_roi_info_bar(self, parent_layout: QVBoxLayout) -> None:
        """Builds the horizontal ROI info bar displayed between the main content and the status bar.

        Args:
            parent_layout: The outer vertical layout to append the info bar to.
        """
        info_bar = QWidget()
        info_bar_layout = QHBoxLayout(info_bar)
        info_bar_layout.setContentsMargins(8, 2, 8, 2)

        self._info_label = QLabel()
        self._info_label.setStyleSheet(STYLE.white_label)
        info_bar_layout.addWidget(self._info_label)
        info_bar_layout.addStretch()
        parent_layout.addWidget(info_bar)

    def _create_view_controls(self) -> tuple[QGroupBox, QComboBox]:
        """Creates the background image dropdown.

        Returns:
            A tuple of the group box and the background view combo box.
        """
        group_box = QGroupBox("Background Image")
        group_box.setStyleSheet(STYLE.group_box)
        layout = QVBoxLayout(group_box)

        view_combo = QComboBox(self)
        view_combo.addItems(list(BackgroundViewLabel))
        view_combo.setFont(FONTS.small_bold)
        view_combo.setEnabled(False)
        view_combo.setToolTip(
            "Select the background image displayed behind the ROI overlay. 'ROIs' shows masks on a black "
            "background. Other options show the corresponding single-recording detection image."
        )
        view_combo.activated.connect(self._on_view_changed)
        layout.addWidget(view_combo)

        return group_box, view_combo

    def _create_color_controls(self) -> tuple[QGroupBox, ColorControls]:
        """Creates the mask color controls including opacity, color mode, colormap, and classification toggles.

        Returns:
            A tuple of the group box and the populated ColorControls dataclass.
        """
        group_box = QGroupBox("Mask Colors")
        group_box.setStyleSheet(STYLE.group_box)
        layout = QVBoxLayout(group_box)

        # Opacity label and slider on one row.
        opacity_row = QHBoxLayout()
        opacity_label = QLabel("Opacity:")
        opacity_label.setStyleSheet(STYLE.white_label)
        opacity_row.addWidget(opacity_label)
        opacity_slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        opacity_slider.setRange(0, 255)
        opacity_slider.setValue(STYLE.default_mask_opacity)
        opacity_slider.setToolTip("Adjust ROI mask opacity.")
        opacity_slider.valueChanged.connect(self._update_plot)
        opacity_row.addWidget(opacity_slider)
        layout.addLayout(opacity_row)

        # Color mode dropdown. Selects which statistic drives ROI coloring.
        color_row = QHBoxLayout()
        color_label = QLabel("Color By:")
        color_label.setStyleSheet(STYLE.white_label)
        color_row.addWidget(color_label)
        color_combo = QComboBox(self)
        color_combo.addItems(list(ROIColorModeLabel))
        color_combo.setFont(FONTS.small_bold)
        color_combo.setEnabled(False)
        color_combo.setToolTip(
            "Select the statistic used to color ROI masks. Each mode maps a per-ROI value to the active colormap."
        )
        color_combo.activated.connect(self._on_color_changed)
        color_row.addWidget(color_combo)
        color_row.addStretch()
        layout.addLayout(color_row)

        # Colormap chooser. Determines the color gradient applied to statistic values.
        colormap_row = QHBoxLayout()
        colormap_label = QLabel("Colormap:")
        colormap_label.setStyleSheet(STYLE.white_label)
        colormap_row.addWidget(colormap_label)
        colormap_chooser = QComboBox()
        colormap_chooser.addItems([cm.value for cm in Colormap])
        colormap_chooser.setCurrentIndex(0)
        colormap_chooser.setFont(FONTS.small_bold)
        colormap_chooser.setToolTip(
            "Select the color gradient applied when mapping ROI statistic values to overlay colors."
        )
        colormap_row.addWidget(colormap_chooser)
        colormap_row.addStretch()
        layout.addLayout(colormap_row)

        # Classifier threshold and temporal bin count on one row.
        params_row = QHBoxLayout()
        threshold_label = QLabel("Threshold:")
        threshold_label.setStyleSheet(STYLE.white_label)
        params_row.addWidget(threshold_label)
        threshold_edit = QLineEdit(self)
        threshold_edit.setText("0.5")
        threshold_edit.setFixedWidth(STYLE.edit_width)
        threshold_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        threshold_edit.setEnabled(False)
        threshold_edit.setToolTip("Classifier probability threshold for the Cell Classification color mode.")
        threshold_edit.returnPressed.connect(self._on_threshold_changed)
        threshold_edit.returnPressed.connect(self.setFocus)
        threshold_edit.installEventFilter(self)
        params_row.addWidget(threshold_edit)
        binning_label = QLabel("Bin Size:")
        binning_label.setStyleSheet(STYLE.white_label)
        params_row.addWidget(binning_label)
        binning_edit = QLineEdit(self)
        binning_edit.setFixedWidth(STYLE.edit_width)
        binning_edit.setValidator(QtGui.QIntValidator(0, 500))
        binning_edit.setText("1")
        binning_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        binning_edit.setEnabled(False)
        binning_edit.setToolTip("Number of frames averaged per temporal bin when computing activity correlations.")
        binning_edit.returnPressed.connect(self._recompute_binned_fluorescence)
        binning_edit.returnPressed.connect(self.setFocus)
        binning_edit.installEventFilter(self)
        params_row.addWidget(binning_edit)
        params_row.addStretch()
        layout.addLayout(params_row)

        # Connects the colormap chooser so that switching colormaps repaints the ROIs using the
        # currently active color mode without recalculating the statistic values.
        colormap_chooser.activated.connect(lambda: self._on_color_changed(self._roi_color_mode))

        controls = ColorControls(
            color_combo=color_combo,
            colormap_chooser=colormap_chooser,
            threshold_edit=threshold_edit,
            binning_edit=binning_edit,
            opacity_slider=opacity_slider,
        )
        return group_box, controls

    def _create_colorbar(self) -> ColorbarWidgets:
        """Creates the colorbar widget displaying the current color mapping.

        Returns:
            The populated ColorbarWidgets dataclass.
        """
        # Container widget. Uses a two-row grid: the color gradient image on top and numeric
        # boundary labels on the bottom. Row 0 gets a higher stretch factor so the gradient
        # occupies most of the vertical space.
        colorbar_widget = pg.GraphicsLayoutWidget(self)
        colorbar_widget.setMaximumHeight(ROI_STYLE.colorbar_max_height)
        colorbar_widget.setMinimumWidth(0)
        colorbar_widget.ci.layout.setRowStretchFactor(0, 2)
        colorbar_widget.ci.layout.setContentsMargins(0, 0, 0, 0)

        # Color gradient image. Renders a 1D colormap strip that is updated whenever the active
        # color mode or colormap changes. The view box spans all three label columns.
        image = pg.ImageItem()
        # noinspection PyUnresolvedReferences
        colorbar_view = colorbar_widget.addViewBox(row=0, col=0, colspan=3)
        colorbar_view.setMenuEnabled(False)
        colorbar_view.addItem(image)

        # Boundary labels. Displays the minimum, midpoint, and maximum values of the current
        # color range. Updated dynamically when the color mode changes to reflect actual data
        # bounds.
        colorbar_font = FONTS.small
        # noinspection PyUnresolvedReferences
        label_0 = colorbar_widget.addLabel("0.0", color=list(COLORS.white), row=1, col=0)
        label_0.setFont(colorbar_font)
        # noinspection PyUnresolvedReferences
        label_half = colorbar_widget.addLabel("0.5", color=list(COLORS.white), row=1, col=1)
        label_half.setFont(colorbar_font)
        # noinspection PyUnresolvedReferences
        label_1 = colorbar_widget.addLabel("1.0", color=list(COLORS.white), row=1, col=2)
        label_1.setFont(colorbar_font)
        labels = [label_0, label_half, label_1]
        return ColorbarWidgets(image=image, labels=labels, widget=colorbar_widget)

    def _create_classifier_controls(self) -> tuple[QGroupBox, ClassifierControls]:
        """Creates the classifier builder panel with New and Add to Existing buttons.

        Returns:
            A tuple of the group box and the populated ClassifierControls dataclass.
        """
        group_box = QGroupBox("Classifier")
        group_box.setStyleSheet(STYLE.group_box)
        layout = QGridLayout(group_box)

        # Classify mode toggle. When checked, clicks flip ROI cell/non-cell labels instead of selecting
        # ROIs for trace plotting. Spans both columns and two rows for visual prominence.
        classify_button = QPushButton("Classify", self)
        classify_button.setCheckable(True)
        classify_button.setFont(FONTS.small_bold)
        classify_button.setEnabled(False)
        classify_button.setToolTip(
            "Toggle classifier mode. When active, clicking an ROI flips its cell/non-cell label instead of "
            "selecting it for trace plotting."
        )
        classify_button.toggled.connect(self._on_classify_toggled)
        layout.addWidget(classify_button, 0, 0, 2, 2)

        # Trains a new classifier from scratch using the current cell/non-cell labels.
        new_button = QPushButton("New", self)
        new_button.setFont(FONTS.small_bold)
        new_button.setToolTip("Create a new classifier training dataset from the current cell/non-cell labels.")
        new_button.clicked.connect(self._on_classifier_new)
        layout.addWidget(new_button, 2, 0, 1, 1)

        # Appends the current recording's labels to an existing classifier and retrains it.
        add_button = QPushButton("Add to Existing", self)
        add_button.setFont(FONTS.small_bold)
        add_button.setToolTip(
            "Append this recording's cell/non-cell labels to an existing classifier training dataset and retrain."
        )
        add_button.clicked.connect(self._on_classifier_add_to_existing)
        layout.addWidget(add_button, 2, 1, 1, 1)

        # Status feedback label. Displays classifier training progress or error messages.
        # Spans both columns and wraps long text to stay within the panel width.
        status_label = QLabel("")
        status_label.setStyleSheet(STYLE.white_label)
        status_label.setFont(FONTS.small_bold)
        status_label.setWordWrap(True)
        layout.addWidget(status_label, 3, 0, 1, 2)

        controls = ClassifierControls(
            classify_button=classify_button,
            new_button=new_button,
            add_button=add_button,
            status_label=status_label,
        )
        return group_box, controls

    def _create_trace_controls(self) -> tuple[QGroupBox, TraceControls]:
        """Creates the trace panel controls inside a group box.

        Returns:
            A tuple of the group box and the populated TraceControls dataclass.
        """
        group_box = QGroupBox("Trace Display")
        group_box.setStyleSheet(STYLE.group_box)
        layout = QGridLayout(group_box)

        # Trace mode checkboxes in a 2x2 grid. Each toggles visibility of a trace type. In multi-recording
        # mode, only one can be active at a time; in single-recording mode, multiple can be checked.
        fluorescence_checkbox = QCheckBox(TraceModeLabel.RAW_FLUORESCENCE)
        fluorescence_checkbox.setStyleSheet(STYLE.white_label)
        fluorescence_checkbox.setToolTip("Toggle the raw fluorescence trace in the trace panel.")
        layout.addWidget(fluorescence_checkbox, 0, 0, 1, 1)

        neuropil_checkbox = QCheckBox(TraceModeLabel.NEUROPIL)
        neuropil_checkbox.setStyleSheet(STYLE.white_label)
        neuropil_checkbox.setToolTip("Toggle the neuropil fluorescence trace in the trace panel.")
        layout.addWidget(neuropil_checkbox, 0, 1, 1, 1)

        corrected_checkbox = QCheckBox(TraceModeLabel.NEUROPIL_CORRECTED)
        corrected_checkbox.setStyleSheet(STYLE.white_label)
        corrected_checkbox.setChecked(True)
        corrected_checkbox.setToolTip("Toggle the neuropil-corrected fluorescence trace in the trace panel.")
        layout.addWidget(corrected_checkbox, 1, 0, 1, 1)

        spikes_checkbox = QCheckBox(TraceModeLabel.DECONVOLVED)
        spikes_checkbox.setStyleSheet(STYLE.white_label)
        spikes_checkbox.setToolTip("Toggle the deconvolved spike trace in the trace panel.")
        layout.addWidget(spikes_checkbox, 1, 1, 1, 1)

        # Maximum trace count. Caps how many ROI traces are drawn simultaneously in the trace
        # panel. Pressing Enter or Escape returns focus to the main window.
        trace_count_row = QHBoxLayout()
        maximum_trace_count_label = QLabel("Maximum Traces:")
        maximum_trace_count_label.setStyleSheet(STYLE.white_label)
        trace_count_row.addWidget(maximum_trace_count_label)
        maximum_trace_count_edit = QLineEdit(self)
        maximum_trace_count_edit.setValidator(QtGui.QIntValidator(0, ROI_CONFIG.plotted_trace_count))
        maximum_trace_count_edit.setText(str(ROI_CONFIG.plotted_trace_count))
        maximum_trace_count_edit.setFixedWidth(STYLE.edit_width)
        maximum_trace_count_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        maximum_trace_count_edit.setToolTip(
            "Limit the maximum number of traces in multi-ROI or multi-recording display modes."
        )
        maximum_trace_count_edit.installEventFilter(self)
        trace_count_row.addWidget(maximum_trace_count_edit)
        trace_count_row.addStretch()
        layout.addLayout(trace_count_row, 2, 0, 1, 2)

        # All Recordings toggle. When active, stacks traces from every recording in the multi-recording
        # dataset for the selected ROI. Hidden until multi-recording data is loaded.
        self._all_recordings_button = QPushButton("All Recordings")
        self._all_recordings_button.setCheckable(True)
        self._all_recordings_button.setToolTip(
            "Show traces from all recordings stacked vertically for the selected ROI."
        )
        self._all_recordings_button.setVisible(False)
        self._all_recordings_button.toggled.connect(self._on_all_recordings_toggled)
        layout.addWidget(self._all_recordings_button, 3, 0, 1, 2)

        controls = TraceControls(
            fluorescence_checkbox=fluorescence_checkbox,
            neuropil_checkbox=neuropil_checkbox,
            corrected_checkbox=corrected_checkbox,
            spikes_checkbox=spikes_checkbox,
            maximum_trace_count_edit=maximum_trace_count_edit,
        )

        # Connects signals after constructing the dataclass so all widget references are stable.
        maximum_trace_count_edit.returnPressed.connect(self._refresh_traces)
        maximum_trace_count_edit.returnPressed.connect(self.setFocus)
        fluorescence_checkbox.toggled.connect(lambda: self._on_trace_toggle(fluorescence_checkbox))
        neuropil_checkbox.toggled.connect(lambda: self._on_trace_toggle(neuropil_checkbox))
        corrected_checkbox.toggled.connect(lambda: self._on_trace_toggle(corrected_checkbox))
        spikes_checkbox.toggled.connect(lambda: self._on_trace_toggle(spikes_checkbox))

        return group_box, controls

    def _build_graphics(self) -> None:
        """Creates the main plotting area with image and trace panels in a vertical splitter."""
        # Adds the image view box with background and overlay layers. Locks the aspect ratio so the
        # image maintains its native proportions regardless of panel size.
        self._view_box = ViewBox(name="plot1", border=list(COLORS.gray), invert_y=True)
        self._view_box.setAspectLocked(lock=True)
        self._image_widget.addItem(self._view_box, 0, 0)
        self._view_box.setMenuEnabled(False)
        self._view_box.scene().contextMenuItem = self._view_box
        self._background = pg.ImageItem(viewbox=self._view_box, parent=self)
        self._background.autoDownsample = False
        self._overlay = pg.ImageItem(viewbox=self._view_box, parent=self)
        self._overlay.autoDownsample = False
        self._view_box.addItem(self._background)
        self._view_box.addItem(self._overlay)
        self._background.setLevels([0, 255])
        self._overlay.setLevels([0, 255])

        # Connects custom click and zoom handlers for ROI selection.
        self._view_box.set_click_handler(self._handle_click)
        self._view_box.set_zoom_handler(self._zoom_plot)

        # Adds the fluorescence trace panel in its own graphics widget below the image.
        self._trace_box = TraceBox()
        # noinspection PyArgumentList
        self._trace_widget.addItem(self._trace_box, row=0, col=0)

    def _load_recording(self) -> None:
        """Displays a file dialog that allows users to select a new recording to visualize."""
        # Defaults the file dialog to the parent of the currently loaded recording's output
        # directory, so the user can easily navigate to a sibling recording.
        start_directory = ""
        if self._context_data is not None:
            output = self._context_data.single_recording.output_path
            parent = output.parent
            if parent.is_dir():
                start_directory = str(parent)

        directory = QFileDialog.getExistingDirectory(self, "Specify the recording directory to load.", start_directory)
        if not directory:
            return

        recording_path = Path(directory)
        console.echo(message=f"Loading recording: {recording_path}.")

        try:
            context_data = ViewerData.from_data(root_path=recording_path)
        except Exception:
            console.echo(message="Unable to load recording data.", level=LogLevel.ERROR)
            result = QMessageBox.question(
                self,
                "ERROR",
                "Unable to load recording. Try another directory?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if result == QMessageBox.StandardButton.Yes:
                self._load_recording()
            return

        self.load_data(data=context_data)

    def _reset_state(self) -> None:
        """Resets all display state to defaults before loading new data.

        Does NOT reset ``_context_data`` or the ROI Source dropdown, which persist across state resets.
        """
        self._roi_color_mode = ROIColorMode.RANDOM
        self._background_view = BackgroundView.ROIS_ONLY
        self._color_controls.opacity_slider.setValue(STYLE.default_mask_opacity)
        self._roi_colormap = Colormap.HSV
        self._selected_roi_index = 0
        self._selected_roi_indices = [0]
        self._temporal_bin_size = 1
        self._recording_loaded = False
        self._colocalization_threshold = ROI_CONFIG.default_channel_2_threshold
        self._last_reclassified_index = -1
        self._classify_mode = False
        self._pre_classify_color_mode = ROIColorMode.RANDOM
        self._classifier_controls.classify_button.setChecked(False)
        self._all_recordings_visible = False

    def _initialize_gui(self) -> None:
        """Initializes all GUI components after loading context data."""
        context = self._context_data
        if context is None:
            return

        single_recording = context.single_recording
        is_multi_recording = self._is_multi_recording

        # Resolves mode-dependent data from the appropriate source. Multi-recording mode pulls from the current
        # recording's tracked masks, while single-recording mode pulls directly from the SingleRecordingData.
        if is_multi_recording:
            recording = context.current_recording
            self._roi_statistics = list(recording.tracked_masks)
            roi_count = len(self._roi_statistics)
            self._cell_classification = (
                np.column_stack([np.ones(roi_count, dtype=np.float32), np.ones(roi_count, dtype=np.float32)])
                if roi_count > 0
                else np.empty((0, 2), dtype=np.float32)
            )
            self._cell_colocalization = np.zeros((roi_count, 2), dtype=np.float32)
            self._two_channels = recording.has_channel_2
            self._cell_fluorescence = recording.cell_fluorescence
            self._neuropil_fluorescence = recording.neuropil_fluorescence
            self._subtracted_fluorescence = recording.subtracted_fluorescence
            self._spikes = recording.spikes
            self._frame_count = int(self._cell_fluorescence.shape[1]) if self._cell_fluorescence.size > 0 else 0
            self._roi_count = roi_count
        else:
            self._roi_statistics = single_recording.roi_statistics
            self._cell_classification = single_recording.cell_classification
            self._cell_colocalization = single_recording.cell_colocalization
            self._two_channels = single_recording.two_channels
            self._cell_fluorescence = single_recording.cell_fluorescence
            self._neuropil_fluorescence = single_recording.neuropil_fluorescence
            self._subtracted_fluorescence = single_recording.subtracted_fluorescence
            self._spikes = single_recording.spikes
            self._frame_count = single_recording.frame_count
            self._roi_count = single_recording.roi_count

        # Resets display controls.
        self.setWindowTitle(f"ROI Viewer — {single_recording.recording_label}")

        # Computes default bin size from tau and sampling rate.
        self._temporal_bin_size = max(
            1, int(single_recording.tau * single_recording.sampling_rate / ROI_CONFIG.bin_size_divisor)
        )
        self._color_controls.binning_edit.setText(str(self._temporal_bin_size))
        self._colocalization_threshold = ROI_CONFIG.default_channel_2_threshold

        # Enables interactive controls.
        self._enable_controls()

        # Multi-recording mode gating: disables inapplicable controls.
        if is_multi_recording:
            # Disables selection modes (top-n, bottom-n).
            self._top_button.setEnabled(False)
            self._bottom_button.setEnabled(False)

            # Disables inapplicable color modes: cell probability, correlations, cell classification.
            color_model = self._color_controls.color_combo.model()
            if isinstance(color_model, QStandardItemModel):
                for disabled_index in (
                    ROIColorMode.CELL_PROBABILITY,
                    ROIColorMode.CORRELATIONS,
                    ROIColorMode.CELL_CLASSIFICATION,
                ):
                    item = color_model.item(disabled_index)
                    if item is not None:
                        item.setEnabled(False)

            # Shows the "All Recordings" toggle.
            self._all_recordings_button.setVisible(True)
            self._all_recordings_button.setChecked(False)

            # Disables classifier controls (requires single-recording classification state).
            self._classifier_controls.classify_button.setEnabled(False)
            self._classifier_controls.new_button.setEnabled(False)
            self._classifier_controls.add_button.setEnabled(False)
        else:
            self._all_recordings_button.setVisible(False)

            # Enables classifier controls for single-recording recordings.
            self._classifier_controls.classify_button.setEnabled(True)
            self._classifier_controls.new_button.setEnabled(True)
            self._classifier_controls.add_button.setEnabled(True)

        # Resets channel 2 toggle state.
        self._channel_2_button.setChecked(False)

        # Builds background views from detection images.
        self._views = build_views(
            frame_height=single_recording.frame_height,
            frame_width=single_recording.frame_width,
            mean_image=single_recording.mean_image,
            enhanced_mean_image=single_recording.enhanced_mean_image,
            correlation_map=single_recording.correlation_map,
            maximum_projection=single_recording.maximum_projection,
            corrected_structural_mean_image=single_recording.corrected_structural_mean_image,
            channel_2=False,
            channel_2_mean_image=single_recording.mean_image_channel_2,
            channel_2_enhanced_mean_image=single_recording.enhanced_mean_image_channel_2,
            channel_2_correlation_map=single_recording.correlation_map_channel_2,
            channel_2_maximum_projection=single_recording.maximum_projection_channel_2,
        )

        # Computes color statistics and builds ROI index maps.
        self._color_arrays = compute_colors(
            roi_statistics=self._roi_statistics,
            frame_height=single_recording.frame_height,
            frame_width=single_recording.frame_width,
            cell_classification=self._cell_classification,
            cell_colocalization=self._cell_colocalization,
            roi_colormap=self._roi_colormap,
            colocalization_threshold=self._colocalization_threshold,
            classifier_threshold=float(self._color_controls.threshold_edit.text() or "0.5"),
            two_channels=self._two_channels,
        )
        self._roi_maps = initialize_roi_maps(
            roi_statistics=self._roi_statistics,
            frame_height=single_recording.frame_height,
            frame_width=single_recording.frame_width,
            color_arrays=self._color_arrays,
        )

        # Selects the first classified cell as the initial selection.
        first_cell = int(np.nonzero(self._cell_classification[:, 0])[0][0]) if self._roi_count > 0 else 0
        self._selected_roi_index = first_cell
        self._selected_roi_indices = [first_cell]
        self._update_selected_roi_statistics()
        self._recompute_binned_fluorescence()

        # Draws the colorbar and initial mask overlays.
        self._colorbar_image = draw_colorbar(colormap=self._roi_colormap)
        if self._colorbar_widgets is None or self._colorbar_image is None:
            return
        render_colorbar(
            roi_color_mode=self._roi_color_mode,
            color_arrays=self._color_arrays,
            colorbar_widgets=self._colorbar_widgets,
            colorbar_image=self._colorbar_image,
        )

        mask = draw_masks(
            roi_statistics=self._roi_statistics,
            frame_height=single_recording.frame_height,
            frame_width=single_recording.frame_width,
            color_arrays=self._color_arrays,
            roi_maps=self._roi_maps,
            roi_color_mode=self._roi_color_mode,
            background_view=self._background_view,
            selected_roi_indices=self._selected_roi_indices,
            roi_opacity=self._color_controls.opacity_slider.value(),
        )
        display_masks(overlay_item=self._overlay, mask=mask)

        # Initializes plot ranges.
        self._view_box.setXRange(0, single_recording.frame_width)
        self._view_box.setYRange(0, single_recording.frame_height)
        self._trace_box.getViewBox().setLimits(xMin=0, xMax=self._frame_count)
        self._frame_indices = np.arange(0, self._frame_count, dtype=np.int32)

        display_views(
            view=self._background,
            views=self._views,
            view_index=self._background_view,
        )
        plot_trace(
            trace_box=self._trace_box,
            cell_fluorescence=self._cell_fluorescence,
            neuropil_fluorescence=self._neuropil_fluorescence,
            subtracted_fluorescence=self._subtracted_fluorescence,
            spikes=self._spikes,
            frame_indices=self._frame_indices,
            selected_indices=self._selected_roi_indices,
            activity_mode=self._active_trace_mode,
            fluorescence_visible=self._trace_controls.fluorescence_checkbox.isChecked(),
            neuropil_visible=self._trace_controls.neuropil_checkbox.isChecked(),
            corrected_visible=self._trace_controls.corrected_checkbox.isChecked(),
            spikes_visible=self._trace_controls.spikes_checkbox.isChecked(),
        )

        # Sets aspect ratio on the image panel.
        self._view_box.setAspectLocked(lock=True, ratio=single_recording.aspect_ratio)

        self._recording_loaded = True

        # Triggers initial full redraw.
        self._update_plot()
        self.show()

    def _enable_controls(self) -> None:
        """Enables all view, color, and selection dropdowns after data loading."""
        if self._context_data is None:
            return
        single_recording = self._context_data.single_recording

        # Enables view dropdown and sets initial selection.
        self._background_combo.setEnabled(True)
        self._background_combo.setCurrentIndex(0)

        # Disables corrected structural view item if not available.
        view_model = self._background_combo.model()
        if isinstance(view_model, QStandardItemModel):
            structural_item = view_model.item(BackgroundView.CORRECTED_STRUCTURAL)
            if structural_item is not None:
                if single_recording.corrected_structural_mean_image.size == 0:
                    structural_item.setEnabled(False)
                else:
                    structural_item.setEnabled(True)

        # Enables channel 2 toggle if channel 2 data exists.
        has_channel_2 = self._two_channels
        self._channel_2_button.setEnabled(has_channel_2)

        # Enables color dropdown and classify mode toggle.
        self._color_controls.color_combo.setEnabled(True)
        self._color_controls.color_combo.setCurrentIndex(0)
        self._classifier_controls.classify_button.setEnabled(True)

        # Re-enables all color mode items, then disables channel 2 mode if not available.
        color_model = self._color_controls.color_combo.model()
        if isinstance(color_model, QStandardItemModel):
            for item_index in range(color_model.rowCount()):
                item = color_model.item(item_index)
                if item is not None:
                    item.setEnabled(True)
            channel_2_item = color_model.item(ROIColorMode.COLOCALIZATION_PROBABILITY)
            if channel_2_item is not None:
                channel_2_item.setEnabled(self._two_channels)

        # Enables ranked selection buttons.
        self._top_button.setEnabled(True)
        self._bottom_button.setEnabled(True)

    def _on_view_changed(self, index: int) -> None:
        """Handles background view dropdown changes.

        Args:
            index: The background view index selected.
        """
        self._background_view = BackgroundView(index)
        self._update_plot()

    def _on_color_changed(self, index: int) -> None:
        """Handles ROI color mode dropdown changes.

        Enables the classifier threshold field for the cell classification mode when classify mode is off, and the
        temporal bins field for the activity correlation mode.

        Args:
            index: The color mode index selected.
        """
        self._roi_color_mode = ROIColorMode(index)

        # Enables mode-specific parameter fields and disables the rest. Threshold is only available in cell
        # classification mode when classify mode is off (threshold-based coloring from probabilities).
        uses_threshold = self._roi_color_mode == ROIColorMode.CELL_CLASSIFICATION and not self._classify_mode
        self._color_controls.threshold_edit.setEnabled(uses_threshold)
        uses_binning = self._roi_color_mode == ROIColorMode.CORRELATIONS
        self._color_controls.binning_edit.setEnabled(uses_binning)

        # Defaults to ROI 0 when switching to correlation mode with no selection.
        if uses_binning and not self._selected_roi_indices:
            self._selected_roi_indices = [0]

        if self._context_data is not None and self._color_arrays is not None and self._roi_maps is not None:
            colormap = self._color_controls.colormap_chooser.currentText()
            if colormap != self._roi_colormap:
                self._roi_colormap = colormap
                self._colorbar_image = update_colormap(
                    color_arrays=self._color_arrays,
                    roi_maps=self._roi_maps,
                    colormap=colormap,
                )
        self._update_plot()

    @property
    def _active_trace_mode(self) -> int:
        """Returns the primary trace mode derived from the checked trace checkboxes, using priority order spikes >
        corrected > neuropil > fluorescence and falling back to corrected if nothing is checked.
        """
        trace_controls = self._trace_controls
        if trace_controls.spikes_checkbox.isChecked():
            return TraceMode.DECONVOLVED
        if trace_controls.corrected_checkbox.isChecked():
            return TraceMode.NEUROPIL_CORRECTED
        if trace_controls.neuropil_checkbox.isChecked():
            return TraceMode.NEUROPIL
        if trace_controls.fluorescence_checkbox.isChecked():
            return TraceMode.RAW_FLUORESCENCE
        return TraceMode.NEUROPIL_CORRECTED

    def _recompute_binned_fluorescence(self) -> None:
        """Recomputes temporally binned fluorescence for correlation coloring using the active trace mode."""
        if not self._recording_loaded or self._context_data is None:
            return
        self._temporal_bin_size = max(1, int(self._color_controls.binning_edit.text()))
        bin_count = int(np.floor(float(self._frame_count) / float(self._temporal_bin_size)))

        mode = self._active_trace_mode
        if mode == TraceMode.RAW_FLUORESCENCE:
            fluorescence = self._cell_fluorescence
        elif mode == TraceMode.NEUROPIL:
            fluorescence = self._neuropil_fluorescence
        elif mode == TraceMode.NEUROPIL_CORRECTED:
            fluorescence = self._subtracted_fluorescence
        else:
            fluorescence = self._spikes

        roi_count = len(self._roi_statistics)
        bin_size = self._temporal_bin_size
        self._binned_fluorescence = (
            fluorescence[:, : bin_count * bin_size].reshape((roi_count, bin_count, bin_size)).mean(axis=2)
        )
        self._binned_fluorescence -= self._binned_fluorescence.mean(axis=1)[:, np.newaxis]
        self._fluorescence_standard_deviation = (self._binned_fluorescence**2).mean(axis=1) ** 0.5
        self._update_plot()

    def _on_channel_2_toggled(self, checked: bool) -> None:
        """Updates the channel 2 button style and refreshes the background view stack.

        Args:
            checked: Determines whether channel 2 is toggled on.
        """
        if self._context_data is None:
            return
        self._channel_2_button.setStyleSheet(STYLE.button_pressed if checked else STYLE.button_unpressed)
        single_recording = self._context_data.single_recording
        self._views = build_views(
            frame_height=single_recording.frame_height,
            frame_width=single_recording.frame_width,
            mean_image=single_recording.mean_image,
            enhanced_mean_image=single_recording.enhanced_mean_image,
            correlation_map=single_recording.correlation_map,
            maximum_projection=single_recording.maximum_projection,
            corrected_structural_mean_image=single_recording.corrected_structural_mean_image,
            channel_2=checked,
            channel_2_mean_image=single_recording.mean_image_channel_2,
            channel_2_enhanced_mean_image=single_recording.enhanced_mean_image_channel_2,
            channel_2_correlation_map=single_recording.correlation_map_channel_2,
            channel_2_maximum_projection=single_recording.maximum_projection_channel_2,
        )
        self._update_plot()

    def _on_classify_toggled(self, checked: bool) -> None:
        """Toggles classifier mode where clicks flip ROI cell/non-cell labels instead of selecting ROIs.

        When toggled on, stores the current color mode, forces CELL_CLASSIFICATION with original labels, and disables
        ROI Source, ROI Selection, Mask Colors, and Trace Display controls. When toggled off, restores the previous
        color mode, recolorizes from the probability threshold, and re-enables all disabled sections.

        Args:
            checked: Determines whether classifier mode is active.
        """
        self._classify_mode = checked
        self._classifier_controls.classify_button.setStyleSheet(
            STYLE.button_pressed if checked else STYLE.button_unpressed
        )

        controls = self._color_controls
        if checked:
            # Stores the current color mode and forces cell classification with original labels.
            self._pre_classify_color_mode = self._roi_color_mode
            self._roi_color_mode = ROIColorMode.CELL_CLASSIFICATION
            controls.color_combo.blockSignals(True)
            controls.color_combo.setCurrentIndex(ROIColorMode.CELL_CLASSIFICATION)
            controls.color_combo.blockSignals(False)

            if self._color_arrays is not None and self._roi_maps is not None:
                recompute_binary_classification(
                    cell_classification=self._cell_classification,
                    color_arrays=self._color_arrays,
                    roi_maps=self._roi_maps,
                    colormap=self._roi_colormap,
                )

            # Disables ROI Source, ROI Selection, Trace Display, and most Mask Colors controls.
            # The opacity slider and colormap chooser stay enabled during classify mode.
            self._roi_source_group.setEnabled(False)
            self._roi_selection_group.setEnabled(False)
            controls.color_combo.setEnabled(False)
            controls.threshold_edit.setEnabled(False)
            controls.binning_edit.setEnabled(False)
            self._trace_group.setEnabled(False)
        else:
            # Restores previous color mode and recolorizes from the probability threshold.
            restored = self._pre_classify_color_mode
            self._roi_color_mode = restored
            controls.color_combo.blockSignals(True)
            controls.color_combo.setCurrentIndex(restored)
            controls.color_combo.blockSignals(False)

            if self._color_arrays is not None and self._roi_maps is not None:
                threshold = float(controls.threshold_edit.text() or "0.5")
                recompute_binary_classification(
                    cell_classification=self._cell_classification,
                    color_arrays=self._color_arrays,
                    roi_maps=self._roi_maps,
                    colormap=self._roi_colormap,
                    threshold=threshold,
                )

            # Re-enables all disabled sections and applies mode-specific field gating.
            self._roi_source_group.setEnabled(True)
            self._roi_selection_group.setEnabled(True)
            controls.color_combo.setEnabled(True)
            controls.threshold_edit.setEnabled(restored == ROIColorMode.CELL_CLASSIFICATION)
            controls.binning_edit.setEnabled(restored == ROIColorMode.CORRELATIONS)
            self._trace_group.setEnabled(True)

        self._update_plot()

    def _on_threshold_changed(self) -> None:
        """Recomputes binary classification colors from the updated threshold and redraws."""
        if self._color_arrays is None or self._roi_maps is None:
            return
        threshold = float(self._color_controls.threshold_edit.text() or "0.5")
        recompute_binary_classification(
            cell_classification=self._cell_classification,
            color_arrays=self._color_arrays,
            roi_maps=self._roi_maps,
            colormap=self._roi_colormap,
            threshold=threshold,
        )
        self._update_plot()

    def _on_number_chosen(self) -> None:
        """Jumps to the ROI number entered in the ROI edit field."""
        if self._recording_loaded and self._context_data is not None:
            self._selected_roi_index = int(self._roi_index_edit.text())
            roi_count = len(self._roi_statistics)
            if self._selected_roi_index >= roi_count:
                self._selected_roi_index = roi_count - 1
            self._selected_roi_indices = [self._selected_roi_index]
            self._update_plot()

    def _enforce_exclusive_trace(self) -> None:
        """Ensures only one trace checkbox is active for multi-ROI or multi-recording modes.

        Resets to Corrected only when multiple checkboxes are checked. When exactly one is already
        checked, preserves the current selection.
        """
        trace_controls = self._trace_controls
        checked_count = sum(
            checkbox.isChecked()
            for checkbox in (
                trace_controls.fluorescence_checkbox,
                trace_controls.neuropil_checkbox,
                trace_controls.corrected_checkbox,
                trace_controls.spikes_checkbox,
            )
        )
        if checked_count <= 1:
            return

        for checkbox in (
            trace_controls.fluorescence_checkbox,
            trace_controls.neuropil_checkbox,
            trace_controls.spikes_checkbox,
        ):
            checkbox.blockSignals(True)
            checkbox.setChecked(False)
            checkbox.blockSignals(False)
        trace_controls.corrected_checkbox.blockSignals(True)
        trace_controls.corrected_checkbox.setChecked(True)
        trace_controls.corrected_checkbox.blockSignals(False)

    def _on_trace_toggle(self, toggled: QCheckBox) -> None:
        """Handles trace visibility checkbox toggles.

        Enforces mutual exclusivity when multiple ROIs are selected or when All Recordings mode
        is active, so stacked traces use a single trace type. In single-ROI mode, allows
        overlaying multiple trace types simultaneously.

        Args:
            toggled: The checkbox that was toggled.
        """
        is_multi_roi = len(self._selected_roi_indices) > 1
        is_all_recordings = self._is_multi_recording and self._all_recordings_visible

        # Unchecks all other checkboxes when one is checked in multi-ROI or All Recordings mode.
        if toggled.isChecked() and (is_multi_roi or is_all_recordings):
            trace_controls = self._trace_controls
            for checkbox in (
                trace_controls.fluorescence_checkbox,
                trace_controls.neuropil_checkbox,
                trace_controls.corrected_checkbox,
                trace_controls.spikes_checkbox,
            ):
                if checkbox is not toggled:
                    checkbox.blockSignals(True)
                    checkbox.setChecked(False)
                    checkbox.blockSignals(False)

        # Recomputes correlation binning only when the correlation color mode is active.
        if self._roi_color_mode == ROIColorMode.CORRELATIONS:
            self._recompute_binned_fluorescence()
        self._refresh_traces()

    def _refresh_traces(self) -> None:
        """Refreshes the trace panel without redrawing image panels."""
        if self._context_data is None or self._color_arrays is None or self._cell_fluorescence.size == 0:
            return

        # In multi-recording mode with "All Recordings" enabled and exactly one ROI selected, shows stacked traces.
        if self._is_multi_recording and self._all_recordings_visible and len(self._selected_roi_indices) == 1:
            self._refresh_all_recording_traces()
            return

        # Derives frame indices from the actual fluorescence array to avoid size mismatches
        # when recordings have different frame counts.
        frame_count = self._cell_fluorescence.shape[1]
        frame_indices = np.arange(frame_count, dtype=np.int32)

        plot_trace(
            trace_box=self._trace_box,
            cell_fluorescence=self._cell_fluorescence,
            neuropil_fluorescence=self._neuropil_fluorescence,
            subtracted_fluorescence=self._subtracted_fluorescence,
            spikes=self._spikes,
            frame_indices=frame_indices,
            selected_indices=self._selected_roi_indices,
            activity_mode=self._active_trace_mode,
            roi_colors=self._color_arrays.colors[self._roi_color_mode],
            fluorescence_visible=self._trace_controls.fluorescence_checkbox.isChecked(),
            neuropil_visible=self._trace_controls.neuropil_checkbox.isChecked(),
            corrected_visible=self._trace_controls.corrected_checkbox.isChecked(),
            spikes_visible=self._trace_controls.spikes_checkbox.isChecked(),
            maximum_trace_count=int(
                self._trace_controls.maximum_trace_count_edit.text() or str(ROI_CONFIG.plotted_trace_count)
            ),
        )

    def _update_plot(self) -> None:
        """Redraws all plot panels including masks, traces, and colorbar."""
        if self._context_data is None or self._color_arrays is None or self._roi_maps is None:
            return
        if self._views is None or self._colorbar_widgets is None or self._colorbar_image is None:
            return

        # Updates correlation masks if the correlation color mode is active.
        if (
            self._roi_color_mode == ROIColorMode.CORRELATIONS
            and self._binned_fluorescence is not None
            and self._fluorescence_standard_deviation is not None
        ):
            update_correlation_masks(
                color_arrays=self._color_arrays,
                roi_maps=self._roi_maps,
                binned_fluorescence=self._binned_fluorescence,
                fluorescence_standard_deviation=self._fluorescence_standard_deviation,
                selected_indices=self._selected_roi_indices,
                colormap=self._roi_colormap,
            )

        # Renders the colorbar for the active color mode.
        render_colorbar(
            roi_color_mode=self._roi_color_mode,
            color_arrays=self._color_arrays,
            colorbar_widgets=self._colorbar_widgets,
            colorbar_image=self._colorbar_image,
        )
        self._update_selected_roi_statistics()

        # Renders background and mask overlay images.
        display_views(
            view=self._background,
            views=self._views,
            view_index=self._background_view,
        )
        mask = draw_masks(
            roi_statistics=self._roi_statistics,
            frame_height=self._context_data.single_recording.frame_height,
            frame_width=self._context_data.single_recording.frame_width,
            color_arrays=self._color_arrays,
            roi_maps=self._roi_maps,
            roi_color_mode=self._roi_color_mode,
            background_view=self._background_view,
            selected_roi_indices=self._selected_roi_indices,
            roi_opacity=self._color_controls.opacity_slider.value(),
        )
        display_masks(overlay_item=self._overlay, mask=mask)

        self._refresh_traces()
        self._view_box.show()
        self._image_widget.show()
        self._trace_widget.show()
        self.show()

    def _update_selected_roi_statistics(self) -> None:
        """Updates the info bar with ROI count, selection state, and statistics for the selected ROI."""
        if self._context_data is None:
            return

        roi_index = self._selected_roi_index
        self._roi_index_edit.setText(str(roi_index))
        selected_count = len(self._selected_roi_indices)
        single_recording = self._context_data.single_recording

        # Builds the selection segment.
        if selected_count == self._roi_count:
            selection = f"ROIs: {self._roi_count}"
        else:
            selection = f"Selected {selected_count} of {self._roi_count}"

        # Builds the statistics segment from the primary selected ROI.
        roi = self._roi_statistics[roi_index]
        centroid = roi.mask.centroid
        stats_parts = [f"centroid: [{centroid[0]}, {centroid[1]}]"]
        for statistic_name in _STATISTICS_TO_SHOW:
            value = getattr(roi, statistic_name, None)
            if value is None:
                continue
            if isinstance(value, int):
                stats_parts.append(f"{statistic_name}: {value}")
            else:
                stats_parts.append(f"{statistic_name}: {value:.2f}")
        stats = "  ".join(stats_parts)

        size = f"Size: {single_recording.frame_height} x {single_recording.frame_width}"
        self._info_label.setText(f"{selection}  |  {stats}  |  {size}")

    def _select_all_rois(self) -> None:
        """Selects all ROIs and refreshes the display."""
        if not self._recording_loaded:
            return
        self._selected_roi_indices = list(range(self._roi_count))
        if self._selected_roi_indices:
            self._selected_roi_index = self._selected_roi_indices[0]
            if len(self._selected_roi_indices) > 1:
                self._enforce_exclusive_trace()
        self._update_plot()

    def _deselect_all_rois(self) -> None:
        """Clears the ROI selection and refreshes the display."""
        if not self._recording_loaded:
            return
        self._selected_roi_indices = []
        self._update_plot()

    def _zoom_plot(self) -> None:
        """Resets the view range for the image panel."""
        self._view_box.autoRange()

    def _flip_plot(self) -> None:
        """Flips the selected ROIs between cell and non-cell classification.

        Classification writes go directly through the r+ memory-mapped file, so no explicit save is needed.
        """
        if self._context_data is None or self._color_arrays is None or self._roi_maps is None:
            return
        flip_rois(
            roi_statistics=self._roi_statistics,
            cell_classification=self._cell_classification,
            color_arrays=self._color_arrays,
            roi_maps=self._roi_maps,
            selected_roi_indices=self._selected_roi_indices,
            colormap=self._roi_colormap,
        )
        self._last_reclassified_index = self._selected_roi_index
        self._update_plot()

    def _handle_click(self, click_x: int, click_y: int, is_right_button: bool, is_multi_select: bool) -> bool:
        """Handles mouse clicks on the image panel.

        Left-click chooses a cell. Shift/ctrl-click adds or removes from the merge selection. Right-click reclassifies
        the clicked ROI when the cell/non-cell color mode is active.

        Args:
            click_x: Column coordinate of the click.
            click_y: Row coordinate of the click.
            is_right_button: Determines whether the click was a right-click.
            is_multi_select: Determines whether shift or ctrl was held during the click.

        Returns:
            True if the click was consumed, False to allow the default context menu.
        """
        if not self._recording_loaded or self._roi_maps is None or self._context_data is None:
            return False

        single_recording = self._context_data.single_recording
        if (
            click_y < 0
            or click_y >= single_recording.frame_height
            or click_x < 0
            or click_x >= single_recording.frame_width
        ):
            return False

        chosen_index = int(self._roi_maps.roi_indices[0, click_y, click_x])
        if chosen_index < 0:
            return False

        # When the classifier toggle is active, any click (left or right) flips the clicked ROI's label
        # without changing the current selection.
        if self._classify_mode and not self._is_multi_recording:
            if self._context_data is not None and self._color_arrays is not None and self._roi_maps is not None:
                flip_rois(
                    roi_statistics=self._roi_statistics,
                    cell_classification=self._cell_classification,
                    color_arrays=self._color_arrays,
                    roi_maps=self._roi_maps,
                    selected_roi_indices=[chosen_index],
                    colormap=self._roi_colormap,
                )
                self._update_plot()
            return True

        if is_right_button:
            return False

        # Multi-recording mode restricts selection to a single ROI.
        if self._is_multi_recording:
            self._selected_roi_indices = [chosen_index]
            self._selected_roi_index = chosen_index
        else:
            was_single = len(self._selected_roi_indices) <= 1
            merged = False
            if is_multi_select:
                if chosen_index not in self._selected_roi_indices:
                    self._selected_roi_indices.append(chosen_index)
                    self._selected_roi_index = chosen_index
                    merged = True
                elif len(self._selected_roi_indices) > 1:
                    self._selected_roi_indices.remove(chosen_index)
                    self._selected_roi_index = self._selected_roi_indices[0]
                    merged = True
            if not merged:
                self._selected_roi_indices = [chosen_index]
                self._selected_roi_index = chosen_index

            # Resets trace checkboxes when transitioning from single-ROI to multi-ROI selection.
            if was_single and len(self._selected_roi_indices) > 1:
                self._enforce_exclusive_trace()

        self._update_plot()
        return True

    def _on_ranked_selection(self, *, top: bool) -> None:
        """Selects the top-n or bottom-n ROIs ranked by the active color statistic.

        Args:
            top: Determines whether to select the highest-ranked ROIs. When False, selects the lowest-ranked.
        """
        if self._color_arrays is None:
            return
        count = int(self._ranked_count_edit.text() or str(ROI_CONFIG.top_selection_count))
        count = min(count, ROI_CONFIG.top_selection_count)
        values = self._color_arrays.normalized_statistics[self._roi_color_mode]
        ranked = np.argsort(values)
        selected = ranked[-count:][::-1] if top else ranked[:count]
        self._selected_roi_indices = selected.tolist()
        if self._selected_roi_indices:
            self._selected_roi_index = self._selected_roi_indices[0]
            if len(self._selected_roi_indices) > 1:
                self._enforce_exclusive_trace()
            self._update_plot()

    def _on_dataset_source_changed(self, index: int) -> None:
        """Handles ROI Source dropdown changes to switch between single-recording and multi-recording data.

        Args:
            index: The selected combo box index. 0 = Original (single-recording), 1+ = multi-recording datasets.
        """
        if self._context_data is None:
            return

        is_currently_multi_recording = self._context_data.is_multi_recording

        if index == 0:
            if not is_currently_multi_recording:
                return
            self._context_data.unload_dataset()
        elif index > 0:
            available = self._context_data.available_datasets
            dataset_index = index - 1
            if dataset_index >= len(available):
                return
            target_name = available[dataset_index]
            if is_currently_multi_recording and self._context_data.active_dataset_name == target_name:
                return
            self._context_data.load_dataset(dataset_name=target_name)
        else:
            return

        self._reset_state()
        self._initialize_gui()

    def _on_all_recordings_toggled(self, checked: bool) -> None:
        """Handles the All Recordings toggle button for multi-recording stacked trace display.

        Args:
            checked: Determines whether the stacked all-recordings view is enabled.
        """
        self._all_recordings_visible = checked
        self._all_recordings_button.setStyleSheet(STYLE.button_pressed if checked else STYLE.button_unpressed)
        if checked:
            self._enforce_exclusive_trace()
        self._refresh_traces()

    def _refresh_all_recording_traces(self) -> None:
        """Plots traces from all recordings stacked vertically for the selected ROI.

        Iterates over every recording in the multi-recording dataset, extracts the selected ROI's trace from each,
        and plots them stacked with recording-index labels on the y-axis.
        """
        if self._context_data is None or not self._context_data.is_multi_recording:
            return

        self._trace_box.clear()
        if self._trace_box.legend is not None:
            self._trace_box.legend.scene().removeItem(self._trace_box.legend)
            self._trace_box.legend = None
        if not self._selected_roi_indices:
            return

        self._trace_box.setLabel("left", "Recording", **{"font-size": FONTS.label_size})
        add_plot_legend(self._trace_box, column_count=1)

        roi_index = self._selected_roi_indices[0]
        axis = self._trace_box.getAxis("left")
        tick_labels: list[tuple[float, str]] = []
        trace_spacing = 1.0 / ROI_CONFIG.default_scale_factor
        max_frames = 0
        y_maximum = 0.0
        maximum_trace_count = int(
            self._trace_controls.maximum_trace_count_edit.text() or str(ROI_CONFIG.plotted_trace_count)
        )
        recording_count = min(self._context_data.recording_count, maximum_trace_count)
        stack_position = recording_count - 1
        active_mode = self._active_trace_mode
        plotted_count = 0
        raw_traces: list[NDArray[np.float32]] = []

        for recording_index in range(recording_count):
            recording = self._context_data.recording(recording_index)
            cell_fluorescence = recording.cell_fluorescence
            neuropil_fluorescence = recording.neuropil_fluorescence
            subtracted_fluorescence = recording.subtracted_fluorescence
            spikes = recording.spikes

            if cell_fluorescence.size == 0:
                stack_position -= 1
                continue
            if roi_index >= cell_fluorescence.shape[0]:
                stack_position -= 1
                continue

            # Selects trace based on the active trace mode.
            if active_mode == TraceMode.RAW_FLUORESCENCE:
                trace = cell_fluorescence[roi_index, :]
            elif active_mode == TraceMode.NEUROPIL:
                trace = neuropil_fluorescence[roi_index, :]
            elif active_mode == TraceMode.NEUROPIL_CORRECTED:
                trace = subtracted_fluorescence[roi_index, :]
            else:
                trace = spikes[roi_index, :]

            frame_indices = np.arange(len(trace), dtype=np.int32)
            max_frames = max(max_frames, len(trace))
            raw_traces.append(trace.astype(np.float32).ravel())

            # Normalizes trace to [0, 1] range for stacked display.
            trace_max = float(trace.max())
            trace_min = float(trace.min())
            if trace_max > trace_min:
                normalized = (trace - trace_min) / (trace_max - trace_min)
            else:
                normalized = np.zeros_like(trace)

            # Generates a deterministic color for this recording using HSV hues.
            hue = recording_index / max(self._context_data.recording_count, 1)
            hsv = np.array([[hue, 1.0, 1.0]], dtype=np.float32)
            rgb = (255.0 * hsv_to_rgb(hsv)).astype(np.uint8)[0]
            pen_color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

            self._trace_box.plot(frame_indices, normalized + stack_position * trace_spacing, pen=pen_color)
            tick_labels.append((stack_position * trace_spacing + float(normalized.mean()), str(recording_index)))
            y_maximum = max(y_maximum, stack_position * trace_spacing + 1)
            stack_position -= 1
            plotted_count += 1

        # Plots a combined average trace at the bottom. Pads shorter recordings with NaN so the
        # average uses all available data at each frame. Frames with fewer than 5 contributing
        # recordings are excluded.
        y_minimum = 0.0
        minimum_contributors = 5
        if plotted_count > ROI_CONFIG.average_threshold and raw_traces:
            padded = np.full((len(raw_traces), max_frames), np.nan, dtype=np.float32)
            for trace_index, raw_trace in enumerate(raw_traces):
                padded[trace_index, : len(raw_trace)] = raw_trace
            contributor_count = np.sum(~np.isnan(padded), axis=0)
            valid_mask = contributor_count >= minimum_contributors
            if np.any(valid_mask):
                average = np.nanmean(padded, axis=0)
                average[~valid_mask] = np.nan
                average_min = float(np.nanmin(average[valid_mask]))
                average -= average_min
                average_max = float(np.nanmax(average[valid_mask]))
                if average_max > 0:
                    average /= average_max
                average_scale = plotted_count / ROI_CONFIG.average_scale_divisor + 1
                average_frames = np.arange(max_frames, dtype=np.int32)
                # Splits at NaN boundaries to avoid connecting gaps with lines.
                segments = np.split(
                    np.arange(max_frames, dtype=np.int32),
                    np.where(~valid_mask)[0],
                )
                for segment in segments:
                    if len(segment) < 2:  # noqa: PLR2004
                        continue
                    segment_frames = average_frames[segment]
                    segment_values = -1 * average_scale + average[segment] * average_scale
                    name = "Average" if segment is segments[0] else None
                    self._trace_box.plot(segment_frames, segment_values, pen=COLORS.silver, name=name)
                y_minimum = -1 * average_scale

        axis.setTicks([tick_labels])
        effective_y_maximum = y_maximum if y_maximum > 0 else 1.0
        self._trace_box.update_range(
            frame_count=max_frames,
            y_minimum=y_minimum,
            y_maximum=effective_y_maximum,
        )
        # Rescales both axes to fit the new data range.
        self._trace_box.getViewBox().autoRange()

    def _extract_classifier_data(
        self,
    ) -> tuple[NDArray[np.bool_], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]] | None:
        """Extracts training labels and classification features from the current recording state.

        Returns:
            A tuple of (training_labels, normalized_pixel_count, compactness, skewness) arrays, or None if no recording
            is loaded or no ROIs exist.
        """
        if not self._recording_loaded or not self._roi_statistics:
            return None

        training_labels = self._cell_classification[:, 0].astype(np.bool_)
        roi_count = len(self._roi_statistics)
        normalized_pixel_count = np.empty(roi_count, dtype=np.float32)
        compactness = np.empty(roi_count, dtype=np.float32)
        skewness = np.empty(roi_count, dtype=np.float32)

        for index, roi in enumerate(self._roi_statistics):
            normalized_pixel_count[index] = roi.normalized_pixel_count
            compactness[index] = roi.compactness
            skewness[index] = np.nan if roi.skewness is None else roi.skewness

        return training_labels, normalized_pixel_count, compactness, skewness

    def _on_classifier_new(self) -> None:
        """Handles the New classifier button by saving current labels and features as a new training dataset."""
        if self._is_multi_recording:
            return

        result = self._extract_classifier_data()
        if result is None:
            return
        training_labels, normalized_pixel_count, compactness, skewness = result

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Classifier Dataset", "", "NumPy files (*.npz)")
        if not file_path:
            return

        Classifier.create_training_dataset(
            file_path=Path(file_path),
            training_labels=training_labels,
            normalized_pixel_count=normalized_pixel_count,
            compactness=compactness,
            skewness=skewness,
        )

        sample_count = len(training_labels)
        filename = Path(file_path).name
        self._classifier_controls.status_label.setText(f"Saved {sample_count} samples to {filename}.")

    def _on_classifier_add_to_existing(self) -> None:
        """Handles the Add to Existing button by merging current recording data into an existing training dataset."""
        if self._is_multi_recording:
            return

        result = self._extract_classifier_data()
        if result is None:
            return
        new_training_labels, new_normalized_pixel_count, new_compactness, new_skewness = result

        source_path, _ = QFileDialog.getOpenFileName(
            self, "Load Existing Classifier Dataset", "", "NumPy files (*.npz)"
        )
        if not source_path:
            return

        try:
            data = np.load(source_path, allow_pickle=False)
            existing_training_labels = data["training_labels"].astype(np.bool_)
            existing_normalized_pixel_count = data["normalized_pixel_count"].astype(np.float32)
            existing_compactness = data["compactness"].astype(np.float32)
            existing_skewness = data["skewness"].astype(np.float32)
        except (KeyError, ValueError, FileNotFoundError) as error:
            self._classifier_controls.status_label.setText(f"Error loading file: {error}")
            return

        merged_training_labels = np.concatenate([existing_training_labels, new_training_labels])
        merged_normalized_pixel_count = np.concatenate([existing_normalized_pixel_count, new_normalized_pixel_count])
        merged_compactness = np.concatenate([existing_compactness, new_compactness])
        merged_skewness = np.concatenate([existing_skewness, new_skewness])

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Merged Dataset", source_path, "NumPy files (*.npz)")
        if not save_path:
            return

        Classifier.create_training_dataset(
            file_path=Path(save_path),
            training_labels=merged_training_labels,
            normalized_pixel_count=merged_normalized_pixel_count,
            compactness=merged_compactness,
            skewness=merged_skewness,
        )

        new_sample_count = len(new_training_labels)
        total_sample_count = len(merged_training_labels)
        filename = Path(save_path).name
        self._classifier_controls.status_label.setText(
            f"Added {new_sample_count} samples ({total_sample_count} total) to {filename}."
        )
