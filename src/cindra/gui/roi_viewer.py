"""Provides the ROI viewer window for inspecting and reclassifying single-day pipeline results."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
from contextlib import suppress

import numpy as np
from PySide6 import QtGui, QtCore
import pyqtgraph as pg  # type: ignore[import-untyped]
from PySide6.QtGui import QStandardItemModel
from PySide6.QtWidgets import (
    QLabel,
    QSlider,
    QWidget,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLineEdit,
    QStatusBar,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)
from matplotlib.colors import hsv_to_rgb
from ataraxis_base_utilities import LogLevel, console

from .styles import FONTS, STYLE, COLORS, ROI_STYLE
from .widgets import ViewBox, TraceBox, plot_trace
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
    ViewControls,
    ColorControls,
    TraceControls,
    ColorbarWidgets,
    SelectionControls,
    ClassifierControls,
)
from .viewer_context import EMPTY, ViewerData
from ..classification import Classifier

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..dataclasses import ROIStatistics

# Statistics attribute names displayed in the Selected ROI panel.
_STATISTICS_TO_SHOW: tuple[str, ...] = (
    "centroid",
    "pixel_count",
    "skewness",
    "compactness",
    "footprint",
    "aspect_ratio",
)


class ROIViewer(QMainWindow):
    """Displays a UI window for inspecting and reclassifying single-day pipeline results.

    Displays ROI overlays, background images, and fluorescence traces. Supports left-click ROI selection,
    shift/ctrl multi-select, right-click cell/non-cell reclassification (active only in cell/non-cell color mode) with
    auto-save, keyboard shortcuts for view/color switching, and double-click zoom-to-fit.

    Args:
        data: The preloaded viewer data to display on startup.

    Attributes:
        _rois_visible: Determines whether ROI overlays are currently displayed.
        _roi_color_mode: Active ROI color statistic index.
        _background_view: Active background image type index.
        _roi_colormap: Active matplotlib colormap name for ROI coloring.
        _selected_roi_index: Index of the most recently selected ROI.
        _selected_roi_indices: List of currently selected ROI indices.
        _trace_mode: Active fluorescence trace type index.
        _temporal_bin_size: Number of frames per temporal bin for activity correlation.
        _auto_zoom_to_roi: Determines whether the image panel auto-zooms to the selected ROI.
        _roi_labels_visible: Determines whether ROI number text labels are shown on the image panel.
        _session_loaded: Determines whether a session has been fully loaded and initialized.
        _colocalization_threshold: Probability threshold for channel 2 colocalization display.
        _last_reclassified_index: Index of the most recently reclassified ROI, or -1 if none.
        _classification_label_mode: Determines whether the cell/non-cell binary label view is active.
        _all_recordings_visible: Determines whether the stacked all-recordings trace view is active.
        _context_data: The ViewerData instance that stores the visualized session's data.
        _color_arrays: Precomputed per-ROI color arrays for each color mode, or None.
        _roi_maps: Precomputed ROI index maps for click-to-ROI and label lookup, or None.
        _colorbar_widgets: PyQtGraph colorbar image, labels, and container widget, or None.
        _colorbar_image: Rendered colorbar image array, or None.
        _views: Precomputed background view image stack, or None.
        _roi_statistics: List of ROIStatistics instances for the current session or recording.
        _cell_classification: Cell classification probability array with shape (cell_count, 2).
        _cell_colocalization: Channel 2 colocalization probability array with shape (cell_count, 2).
        _two_channels: Determines whether the current session has channel 2 data.
        _cell_fluorescence: Raw cell fluorescence traces array, or an empty array.
        _neuropil_fluorescence: Neuropil fluorescence traces array, or an empty array.
        _subtracted_fluorescence: Neuropil-subtracted fluorescence traces array, or an empty array.
        _spikes: Deconvolved spike rate traces array, or an empty array.
        _frame_count: Number of frames in the current session.
        _cell_count: Number of detected cells in the current session.
        _binned_fluorescence: Temporally binned fluorescence array for correlation coloring, or None.
        _fluorescence_standard_deviation: Per-cell standard deviation of binned fluorescence, or None.
        _frame_indices: Frame index array for trace x-axis, or None.
        _graphics_widget: PyQtGraph graphics layout for image and trace panels.
        _status_bar: Status bar displaying session info and selection state.
        _view_box: View box for the primary ROI image display.
        _background: Image item for the background image display.
        _overlay: Image item for the ROI mask overlay display.
        _trace_box: Trace plot box for fluorescence trace display.
        _roi_source_group: Group box for the multi-day ROI source selector.
        _roi_source_combo: Dropdown for selecting single-day or multi-day ROI source.
        _view_selector: Dropdown for selecting the combined or per-plane view.
        _roi_visibility_checkbox: Checkbox for toggling ROI overlay visibility.
        _roi_labels_checkbox: Checkbox for toggling ROI number labels.
        _selection_controls: Cell selection dropdown and top-n input controls.
        _view_controls: Background view dropdown, channel 2 toggle, and opacity slider controls.
        _color_controls: ROI color mode dropdown, colormap chooser, and classifier threshold controls.
        _trace_controls: Trace display mode, visibility checkboxes, and max plotted controls.
        _classifier_controls: Classifier builder panel with New and Add to Existing buttons.
        _roi_index_edit: Input field for jumping to a specific ROI by number.
        _roi_statistic_labels: Labels displaying per-ROI statistics in the Selected ROI panel.
        _zoom_to_cell_checkbox: Checkbox for toggling auto-zoom-to-cell behavior.
        _all_recordings_button: Toggle button for stacked all-recordings trace display.
    """

    def __init__(self, data: ViewerData) -> None:
        super().__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")

        # Display state fields.
        self._rois_visible: bool = True
        self._roi_color_mode: int = ROIColorMode.RANDOM
        self._background_view: int = BackgroundView.ROIS_ONLY
        self._roi_colormap: str = Colormap.HSV
        self._selected_roi_index: int = 0
        self._selected_roi_indices: list[int] = [0]
        self._trace_mode: int = TraceMode.NEUROPIL_CORRECTED
        self._temporal_bin_size: int = 1
        self._auto_zoom_to_roi: bool = False
        self._roi_labels_visible: bool = False
        self._session_loaded: bool = False
        self._colocalization_threshold: float = ROI_CONFIG.default_channel_2_threshold
        self._last_reclassified_index: int = -1
        self._classification_label_mode: bool = False

        # Multi-day state. Persists across _reset_state calls.
        self._all_recordings_visible: bool = False

        # Core data objects.
        self._context_data: ViewerData | None = None
        self._color_arrays: ColorArrays | None = None
        self._roi_maps: ROIIndexMaps | None = None
        self._colorbar_widgets: ColorbarWidgets | None = None
        self._colorbar_image: NDArray[np.uint8] | None = None
        self._views: NDArray[np.uint8] | None = None

        # Mode-dependent data cache. Populated by _initialize_gui from either single_day or current_recording.
        self._roi_statistics: list[ROIStatistics] = []
        self._cell_classification: NDArray[np.float32] = EMPTY
        self._cell_colocalization: NDArray[np.float32] = EMPTY
        self._two_channels: bool = False
        self._cell_fluorescence: NDArray[np.float32] = EMPTY
        self._neuropil_fluorescence: NDArray[np.float32] = EMPTY
        self._subtracted_fluorescence: NDArray[np.float32] = EMPTY
        self._spikes: NDArray[np.float32] = EMPTY
        self._frame_count: int = 0
        self._cell_count: int = 0

        # Binned activity state used by correlation coloring.
        self._binned_fluorescence: NDArray[np.float32] | None = None
        self._fluorescence_standard_deviation: NDArray[np.float32] | None = None
        self._frame_indices: NDArray[np.int32] | None = None

        # Window geometry and title.
        self.setGeometry(*ROI_STYLE.window_geometry)
        self.setWindowTitle("ROI Viewer")

        self.setStyleSheet(STYLE.main_window)

        # File-only menu bar.
        self._build_menus()

        # Main widget layout: graphics | control panel.
        central_widget = QWidget(self)
        main_layout = QHBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # Left: graphics panel.
        self._graphics_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self._graphics_widget, stretch=3)

        # Right: control panel.
        control_panel = self._build_control_panel()
        main_layout.addWidget(control_panel, stretch=1)

        # Status bar.
        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)

        # Builds graphics panels.
        self._build_graphics()

        # Prevents control panel widgets from capturing keyboard focus so spacebar and arrow keys always reach the
        # main window's keyPressEvent.
        for widget in control_panel.findChildren(QWidget):
            widget.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        # Accepts drag-and-drop of directories.
        self.setAcceptDrops(True)

        # Populates the UI with the startup data provided by the caller.
        self.load_data(data=data)

        self.show()
        self._graphics_widget.show()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        """Handles keyboard shortcuts for ROI visibility toggling.

        Notes:
            Overrides the Qt virtual method. The camelCase name is required to match the parent signature.
        """
        if event.key() == QtCore.Qt.Key.Key_Space:
            self._roi_visibility_checkbox.toggle()

    def load_data(self, data: ViewerData) -> None:
        """Caches the input ViewerData instance and uses it to populate the managed UI window.

        Args:
            data: The ViewerData instance that stores the visualized session's data.
        """
        self._context_data = data

        # Populates the ROI Source combo with "Original" + discovered datasets.
        self._roi_source_combo.blockSignals(True)
        self._roi_source_combo.clear()
        self._roi_source_combo.addItem("Original")
        for name in data.available_datasets:
            self._roi_source_combo.addItem(f"Dataset: {name}")

        # Selects the active dataset if one is loaded, otherwise selects "Original".
        if data.is_multi_day:
            for index in range(1, self._roi_source_combo.count()):
                item_text = self._roi_source_combo.itemText(index)
                if item_text == f"Dataset: {data.active_dataset_name}":
                    self._roi_source_combo.setCurrentIndex(index)
                    break
        else:
            self._roi_source_combo.setCurrentIndex(0)
        self._roi_source_combo.blockSignals(False)
        self._roi_source_group.setVisible(bool(data.available_datasets))

        self._reset_state()
        self._initialize_gui()

    @property
    def _is_multi_day(self) -> bool:
        """Returns True when the viewer is displaying multi-day tracked ROI data."""
        return self._context_data is not None and self._context_data.is_multi_day

    def _build_menus(self) -> None:
        """Builds the File-only menu bar for the viewer."""
        file_menu = self.menuBar().addMenu("&File")

        load_action = file_menu.addAction("&Load recording")
        load_action.setShortcut("Ctrl+L")
        load_action.triggered.connect(self._load_recording)

    def _build_control_panel(self) -> QWidget:
        """Constructs the right-side control panel with all grouped viewer controls.

        Returns:
            The control panel widget containing all grouped controls.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # ROI Source selector (Original / Dataset: ...). Hidden when no multi-day datasets exist.
        self._roi_source_group = QGroupBox("ROI Source")
        self._roi_source_group.setStyleSheet(STYLE.group_box)
        roi_source_layout = QHBoxLayout(self._roi_source_group)
        roi_source_label = QLabel("Source:")
        roi_source_label.setStyleSheet(STYLE.white_label)
        roi_source_layout.addWidget(roi_source_label)
        self._roi_source_combo: QComboBox = QComboBox(self)
        self._roi_source_combo.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._roi_source_combo.activated.connect(self._on_dataset_source_changed)
        roi_source_layout.addWidget(self._roi_source_combo)
        roi_source_layout.addStretch()
        self._roi_source_group.setVisible(False)
        layout.addWidget(self._roi_source_group)

        # View selector (Combined / Plane 0 / Plane 1 / ...).
        view_box = QGroupBox("View")
        view_box.setStyleSheet(STYLE.group_box)
        view_layout = QHBoxLayout(view_box)
        view_label = QLabel("View:")
        view_label.setStyleSheet(STYLE.white_label)
        view_layout.addWidget(view_label)
        self._view_selector: QComboBox = QComboBox(self)
        self._view_selector.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self._view_selector.setEnabled(False)
        self._view_selector.currentIndexChanged.connect(self._on_view_selector_changed)
        view_layout.addWidget(self._view_selector)
        view_layout.addStretch()
        layout.addWidget(view_box)

        # ROI visibility toggles.
        visibility_box = QGroupBox("ROI Visibility")
        visibility_box.setStyleSheet(STYLE.group_box)
        visibility_layout = QVBoxLayout(visibility_box)
        self._roi_visibility_checkbox = QCheckBox("ROIs On")
        self._roi_visibility_checkbox.setStyleSheet(STYLE.white_label)
        self._roi_visibility_checkbox.toggle()
        self._roi_visibility_checkbox.stateChanged.connect(self._toggle_rois)
        visibility_layout.addWidget(self._roi_visibility_checkbox)
        self._roi_labels_checkbox = QCheckBox("add ROI # to plot")
        self._roi_labels_checkbox.setStyleSheet(STYLE.white_label)
        self._roi_labels_checkbox.stateChanged.connect(self._toggle_roi_labels)
        self._roi_labels_checkbox.setEnabled(False)
        visibility_layout.addWidget(self._roi_labels_checkbox)
        layout.addWidget(visibility_box)

        # Cell selection controls.
        selection_box, self._selection_controls = self._create_selection_buttons()
        layout.addWidget(selection_box)

        # Background view controls.
        background_box, self._view_controls = self._create_view_controls()
        layout.addWidget(background_box)

        # ROI color controls and colorbar.
        colors_box, self._color_controls = self._create_color_controls()
        self._colorbar_widgets = self._create_colorbar()
        colors_layout = colors_box.layout()
        if colors_layout is not None:
            colors_layout.addWidget(self._colorbar_widgets.widget)
        layout.addWidget(colors_box)

        # Selected ROI panel with index edit and statistics labels.
        roi_box = QGroupBox("Selected ROI")
        roi_box.setStyleSheet(STYLE.group_box)
        roi_layout = QVBoxLayout(roi_box)
        self._roi_index_edit = QLineEdit(self)
        self._roi_index_edit.setValidator(QtGui.QIntValidator(0, 10000))
        self._roi_index_edit.setText("0")
        self._roi_index_edit.setFixedWidth(STYLE.edit_width)
        self._roi_index_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._roi_index_edit.returnPressed.connect(self._on_number_chosen)
        roi_layout.addWidget(self._roi_index_edit)
        self._roi_statistic_labels: list[QLabel] = []
        for statistic_name in _STATISTICS_TO_SHOW:
            statistic_label = QLabel(statistic_name)
            statistic_label.setStyleSheet(STYLE.white_label)
            statistic_label.resize(statistic_label.minimumSizeHint())
            roi_layout.addWidget(statistic_label)
            self._roi_statistic_labels.append(statistic_label)
        layout.addWidget(roi_box)

        # Trace display controls.
        trace_box, self._trace_controls = self._create_trace_controls()
        self._zoom_to_cell_checkbox = QCheckBox("zoom to cell")
        self._zoom_to_cell_checkbox.setStyleSheet(STYLE.white_label)
        self._zoom_to_cell_checkbox.stateChanged.connect(self._on_zoom_cell_toggled)
        trace_layout = trace_box.layout()
        if trace_layout is not None:
            trace_layout.addWidget(self._zoom_to_cell_checkbox)
        layout.addWidget(trace_box)

        # Classifier builder controls.
        classifier_box, self._classifier_controls = self._create_classifier_controls()
        layout.addWidget(classifier_box)

        layout.addStretch()
        return panel

    def _create_selection_buttons(self) -> tuple[QGroupBox, SelectionControls]:
        """Creates the cell selection dropdown and top-n input field.

        Returns:
            A tuple of the group box and the populated SelectionControls dataclass.
        """
        group_box = QGroupBox("Cell Selection")
        group_box.setStyleSheet(STYLE.group_box)
        layout = QGridLayout(group_box)

        selection_combo = QComboBox(self)
        selection_combo.addItems(["select top n", "select bottom n"])
        selection_combo.setFont(FONTS.small_bold)
        selection_combo.setEnabled(False)
        selection_combo.activated.connect(lambda _: self._on_top_bottom_selection())
        layout.addWidget(selection_combo, 0, 0, 1, 1)

        count_label = QLabel("n=")
        count_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        count_label.setStyleSheet(STYLE.white_label)
        count_label.setFont(FONTS.small_bold)
        layout.addWidget(count_label, 0, 1, 1, 1)

        top_count_edit = QLineEdit(self)
        top_count_edit.setValidator(QtGui.QIntValidator(0, ROI_CONFIG.top_selection_count))
        top_count_edit.setText(str(ROI_CONFIG.top_selection_count))
        top_count_edit.setFixedWidth(STYLE.edit_width)
        top_count_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        top_count_edit.returnPressed.connect(self._on_top_bottom_selection)
        layout.addWidget(top_count_edit, 0, 2, 1, 1)

        controls = SelectionControls(selection_combo=selection_combo, top_count_edit=top_count_edit)
        return group_box, controls

    def _create_view_controls(self) -> tuple[QGroupBox, ViewControls]:
        """Creates the background view dropdown, channel 2 toggle, and opacity slider.

        Returns:
            A tuple of the group box and the populated ViewControls dataclass.
        """
        group_box = QGroupBox("Background")
        group_box.setStyleSheet(STYLE.group_box)
        layout = QGridLayout(group_box)

        view_combo = QComboBox(self)
        view_combo.addItems(list(BackgroundViewLabel))
        view_combo.setFont(FONTS.small_bold)
        view_combo.setEnabled(False)
        view_combo.activated.connect(self._on_view_changed)
        layout.addWidget(view_combo, 0, 0, 1, 1)

        channel_2_button = QPushButton("Channel 2", self)
        channel_2_button.setCheckable(True)
        channel_2_button.setFont(FONTS.small_bold)
        channel_2_button.setStyleSheet(STYLE.button_inactive)
        channel_2_button.setEnabled(False)
        channel_2_button.toggled.connect(self._on_channel_2_toggled)
        layout.addWidget(channel_2_button, 1, 0, 1, 1)

        opacity_label = QLabel("Opacity:")
        opacity_label.setStyleSheet(STYLE.white_label)
        layout.addWidget(opacity_label, 0, 1, 1, 1)

        opacity_slider = QSlider(QtCore.Qt.Orientation.Horizontal)
        opacity_slider.setRange(0, 255)
        opacity_slider.setValue(STYLE.default_mask_opacity)
        opacity_slider.setToolTip("Adjust ROI mask opacity.")
        opacity_slider.valueChanged.connect(self._update_plot)
        layout.addWidget(opacity_slider, 1, 1, 1, 1)

        controls = ViewControls(view_combo=view_combo, channel_2_button=channel_2_button, opacity_slider=opacity_slider)
        return group_box, controls

    def _create_color_controls(self) -> tuple[QGroupBox, ColorControls]:
        """Creates the color statistic dropdown and associated controls.

        Returns:
            A tuple of the group box and the populated ColorControls dataclass.
        """
        group_box = QGroupBox("ROI Colors")
        group_box.setStyleSheet(STYLE.group_box)
        layout = QGridLayout(group_box)

        # Color mode dropdown. Selects which statistic drives ROI coloring (e.g. classifier,
        # correlation, skewness). Starts disabled until session data is loaded.
        color_combo = QComboBox(self)
        color_combo.addItems(list(ROIColorModeLabel))
        color_combo.setFont(FONTS.small_bold)
        color_combo.setEnabled(False)
        color_combo.activated.connect(self._on_color_changed)
        layout.addWidget(color_combo, 0, 0, 1, 1)

        # Colormap chooser. Determines the color gradient applied to statistic values.
        # Changing the colormap triggers a recolor without recalculating the underlying statistic.
        colormap_chooser = QComboBox()
        colormap_chooser.addItems([cm.value for cm in Colormap])
        colormap_chooser.setCurrentIndex(0)
        colormap_chooser.setFont(FONTS.small_bold)
        colormap_chooser.setFixedWidth(ROI_STYLE.color_edit_width)
        layout.addWidget(colormap_chooser, 0, 1, 1, 1)

        # Classifier probability threshold. ROIs with classifier probability below this value are
        # colored as non-cells when the classifier color mode is active.
        classifier_label = QLabel("cell prob=")
        classifier_label.setStyleSheet(STYLE.white_label)
        layout.addWidget(classifier_label, 1, 0, 1, 1)

        classifier_edit = QLineEdit(self)
        classifier_edit.setText("0.5")
        classifier_edit.setFixedWidth(ROI_STYLE.color_edit_width)
        classifier_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(classifier_edit, 1, 1, 1, 1)
        classifier_edit.returnPressed.connect(self._update_plot)

        # Temporal binning factor. Groups consecutive frames when computing activity-based color
        # statistics. Pressing Enter recalculates the activity metric with the new bin size.
        bin_label = QLabel("bin=")
        bin_label.setStyleSheet(STYLE.white_label)
        layout.addWidget(bin_label, 2, 0, 1, 1)

        bin_edit = QLineEdit(self)
        bin_edit.setValidator(QtGui.QIntValidator(0, 500))
        bin_edit.setText("1")
        bin_edit.setFixedWidth(ROI_STYLE.color_edit_width)
        bin_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(bin_edit, 2, 1, 1, 1)
        bin_edit.returnPressed.connect(
            lambda: self._on_activity_changed(self._trace_controls.activity_combo.currentIndex())
        )

        # Connects the colormap chooser so that switching colormaps repaints the ROIs using the
        # currently active color mode without recalculating the statistic values.
        colormap_chooser.activated.connect(lambda: self._on_color_changed(self._roi_color_mode))

        # Cell / Non-Cell label toggle. When checked, overlays classification text labels on each
        # ROI. Starts disabled and inactive until session data is loaded.
        classification_label_button = QPushButton("Cell / Non-Cell", self)
        classification_label_button.setCheckable(True)
        classification_label_button.setFont(FONTS.small_bold)
        classification_label_button.setStyleSheet(STYLE.button_inactive)
        classification_label_button.setEnabled(False)
        classification_label_button.toggled.connect(self._on_classification_label_toggled)
        layout.addWidget(classification_label_button, 3, 0, 1, 2)

        controls = ColorControls(
            color_combo=color_combo,
            colormap_chooser=colormap_chooser,
            classifier_edit=classifier_edit,
            binning_edit=bin_edit,
            classification_label_button=classification_label_button,
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
        colorbar_widget.setMaximumWidth(ROI_STYLE.colorbar_max_width)
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

        # Trains a new classifier from scratch using the current cell/non-cell labels.
        new_button = QPushButton("New", self)
        new_button.setFont(FONTS.small_bold)
        new_button.setStyleSheet(STYLE.button_unpressed)
        new_button.clicked.connect(self._on_classifier_new)
        layout.addWidget(new_button, 0, 0, 1, 1)

        # Appends the current session's labels to an existing classifier and retrains it.
        add_button = QPushButton("Add to Existing", self)
        add_button.setFont(FONTS.small_bold)
        add_button.setStyleSheet(STYLE.button_unpressed)
        add_button.clicked.connect(self._on_classifier_add_to_existing)
        layout.addWidget(add_button, 0, 1, 1, 1)

        # Status feedback label. Displays classifier training progress or error messages.
        # Spans both columns and wraps long text to stay within the panel width.
        status_label = QLabel("")
        status_label.setStyleSheet(STYLE.white_label)
        status_label.setFont(FONTS.small_bold)
        status_label.setWordWrap(True)
        layout.addWidget(status_label, 1, 0, 1, 2)

        controls = ClassifierControls(new_button=new_button, add_button=add_button, status_label=status_label)
        return group_box, controls

    def _create_trace_controls(self) -> tuple[QGroupBox, TraceControls]:
        """Creates the trace panel controls inside a group box.

        Returns:
            A tuple of the group box and the populated TraceControls dataclass.
        """
        group_box = QGroupBox("Trace Display")
        group_box.setStyleSheet(STYLE.group_box)
        layout = QGridLayout(group_box)

        # Activity mode selector. Determines which fluorescence signal is used for trace display
        # and activity-based ROI coloring. Defaults to the deconvolved trace.
        activity_label = QLabel("Activity mode:")
        activity_label.setStyleSheet(STYLE.white_label)
        layout.addWidget(activity_label, 0, 0, 1, 1)

        activity_combo = QComboBox(self)
        layout.addWidget(activity_combo, 1, 0, 1, 1)
        activity_combo.addItems(list(TraceModeLabel))
        activity_combo.setCurrentIndex(TraceMode.DECONVOLVED)
        activity_combo.currentIndexChanged.connect(self._on_activity_changed)

        # Maximum plotted traces. Caps how many ROI traces are drawn simultaneously in the trace
        # panel. Pressing Enter refreshes the visible traces with the new limit.
        max_plotted_label = QLabel("max # plotted:")
        max_plotted_label.setStyleSheet(STYLE.white_label)
        layout.addWidget(max_plotted_label, 2, 0, 1, 1)

        max_plotted_edit = QLineEdit(self)
        max_plotted_edit.setValidator(QtGui.QIntValidator(0, ROI_CONFIG.plotted_trace_count))
        max_plotted_edit.setText(str(ROI_CONFIG.plotted_trace_count))
        max_plotted_edit.setFixedWidth(STYLE.edit_width)
        max_plotted_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(max_plotted_edit, 3, 0, 1, 1)

        # Trace visibility checkboxes. Each toggles an individual trace layer in the trace panel.
        # Bracket labels indicate keyboard shortcuts: [N] deconvolved, [B] neuropil, [V] raw
        # fluorescence. All start checked so every trace layer is visible by default.
        deconvolved_checkbox = QCheckBox("deconvolved [N]")
        deconvolved_checkbox.setStyleSheet(STYLE.white_label)
        deconvolved_checkbox.toggle()
        layout.addWidget(deconvolved_checkbox, 3, 1, 1, 1)

        neuropil_checkbox = QCheckBox("neuropil [B]")
        neuropil_checkbox.setStyleSheet(STYLE.white_label)
        neuropil_checkbox.toggle()
        layout.addWidget(neuropil_checkbox, 3, 2, 1, 1)

        traces_checkbox = QCheckBox("fluorescence [V]")
        traces_checkbox.setStyleSheet(STYLE.white_label)
        traces_checkbox.toggle()
        layout.addWidget(traces_checkbox, 3, 3, 1, 1)

        # All Recordings toggle. When active, stacks traces from every recording in the multi-day
        # dataset for the selected ROI. Hidden until multi-day data is loaded.
        self._all_recordings_button = QPushButton("All Recordings")
        self._all_recordings_button.setCheckable(True)
        self._all_recordings_button.setStyleSheet(STYLE.button_unpressed)
        self._all_recordings_button.setToolTip(
            "Show traces from all recordings stacked vertically for the selected ROI."
        )
        self._all_recordings_button.setVisible(False)
        self._all_recordings_button.toggled.connect(self._on_all_recordings_toggled)
        layout.addWidget(self._all_recordings_button, 4, 0, 1, 2)

        controls = TraceControls(
            activity_combo=activity_combo,
            deconvolved_checkbox=deconvolved_checkbox,
            neuropil_checkbox=neuropil_checkbox,
            traces_checkbox=traces_checkbox,
            max_plotted_edit=max_plotted_edit,
        )

        # Connects signals after constructing the dataclass so all widget references are stable.
        max_plotted_edit.returnPressed.connect(self._refresh_traces)
        deconvolved_checkbox.toggled.connect(lambda: self._on_trace_toggle("deconvolved"))
        neuropil_checkbox.toggled.connect(lambda: self._on_trace_toggle("neuropil"))
        traces_checkbox.toggled.connect(lambda: self._on_trace_toggle("traces"))

        return group_box, controls

    def _build_graphics(self) -> None:
        """Creates the main plotting area with image and trace panels."""
        # Adds the image view box with background and overlay layers.
        self._view_box = ViewBox(name="plot1", border=list(COLORS.gray), invert_y=True)
        self._graphics_widget.addItem(self._view_box, 0, 0)
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

        # Adds the fluorescence trace panel below the image view.
        self._trace_box = TraceBox()
        # noinspection PyUnresolvedReferences
        self._trace_box.enableAutoRange(x=True, y=True)
        # noinspection PyArgumentList
        self._graphics_widget.addItem(self._trace_box, row=1, col=0)
        self._graphics_widget.ci.layout.setRowStretchFactor(0, 2)
        graphics_layout = self._graphics_widget.ci.layout
        graphics_layout.setColumnMinimumWidth(0, 1)
        graphics_layout.setHorizontalSpacing(20)

    def _load_recording(self) -> None:
        """Displays a file dialog that allows users to select a new recording to visualize."""
        # Defaults the file dialog to the parent of the currently loaded recording's output
        # directory, so the user can easily navigate to a sibling recording.
        start_directory = ""
        if self._context_data is not None:
            output = self._context_data.single_day.output_path
            parent = output.parent
            if parent.is_dir():
                start_directory = str(parent)

        directory = QFileDialog.getExistingDirectory(self, "Specify the recording directory to load.", start_directory)
        if not directory:
            return

        recording_path = Path(directory)
        console.echo(message=f"Loading recording: {recording_path}")

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
        self._rois_visible = True
        self._roi_color_mode = ROIColorMode.RANDOM
        self._background_view = BackgroundView.ROIS_ONLY
        self._view_controls.opacity_slider.setValue(STYLE.default_mask_opacity)
        self._roi_colormap = Colormap.HSV
        self._selected_roi_index = 0
        self._selected_roi_indices = [0]
        self._trace_mode = TraceMode.NEUROPIL_CORRECTED
        self._temporal_bin_size = 1
        self._auto_zoom_to_roi = False
        self._roi_labels_visible = False
        self._session_loaded = False
        self._colocalization_threshold = ROI_CONFIG.default_channel_2_threshold
        self._last_reclassified_index = -1
        self._classification_label_mode = False
        self._color_controls.classification_label_button.setChecked(False)
        self._all_recordings_visible = False

    def _initialize_gui(self) -> None:
        """Initializes all GUI components after loading context data."""
        context = self._context_data
        if context is None:
            return

        single_day = context.single_day
        is_multi_day = self._is_multi_day

        # Resolves mode-dependent data from the appropriate source. Multi-day mode pulls from the current recording's
        # tracked masks, while single-day mode pulls directly from the SingleDayData.
        if is_multi_day:
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
            self._cell_count = roi_count
        else:
            self._roi_statistics = single_day.roi_statistics
            self._cell_classification = single_day.cell_classification
            self._cell_colocalization = single_day.cell_colocalization
            self._two_channels = single_day.two_channels
            self._cell_fluorescence = single_day.cell_fluorescence
            self._neuropil_fluorescence = single_day.neuropil_fluorescence
            self._subtracted_fluorescence = single_day.subtracted_fluorescence
            self._spikes = single_day.spikes
            self._frame_count = single_day.frame_count
            self._cell_count = single_day.cell_count

        # Populates the view selector without triggering _on_view_selector_changed. Signals are blocked so that
        # clearing and re-adding items does not fire redundant view-switch callbacks.
        self._view_selector.blockSignals(True)
        self._view_selector.clear()
        if is_multi_day:
            self._view_selector.addItem("Combined")
            self._view_selector.setCurrentIndex(0)
        else:
            for label in single_day.view_labels:
                self._view_selector.addItem(label)
            self._view_selector.setCurrentIndex(single_day.view_index + 1)
        self._view_selector.blockSignals(False)
        self._view_selector.setEnabled(not is_multi_day and len(single_day.view_labels) > 1)

        # Resets display controls.
        self._roi_visibility_checkbox.setChecked(True)
        if self._roi_labels_checkbox.isChecked():
            self._toggle_roi_labels(False)
        self._roi_labels_checkbox.setChecked(False)
        self._roi_labels_checkbox.setEnabled(True)

        self.setWindowTitle(f"ROI Viewer — {single_day.recording_label}")

        # Computes default bin size from tau and sampling rate.
        self._temporal_bin_size = max(1, int(single_day.tau * single_day.sampling_rate / ROI_CONFIG.bin_size_divisor))
        self._color_controls.binning_edit.setText(str(self._temporal_bin_size))
        self._colocalization_threshold = ROI_CONFIG.default_channel_2_threshold

        # Enables interactive controls.
        self._enable_controls()

        # Multi-day mode gating: disables inapplicable controls.
        if is_multi_day:
            # Disables selection modes (top-n, bottom-n).
            self._selection_controls.selection_combo.setEnabled(False)

            # Disables inapplicable color modes: cell classification, correlations.
            color_model = self._color_controls.color_combo.model()
            if isinstance(color_model, QStandardItemModel):
                for disabled_index in (
                    ROIColorMode.CELL_CLASSIFICATION,
                    ROIColorMode.CORRELATIONS,
                ):
                    item = color_model.item(disabled_index)
                    if item is not None:
                        item.setEnabled(False)

            # Shows the "All Recordings" toggle.
            self._all_recordings_button.setVisible(True)
            self._all_recordings_button.setChecked(False)

            # Disables classifier builder (requires single-day classification state).
            self._classifier_controls.new_button.setEnabled(False)
            self._classifier_controls.new_button.setStyleSheet(STYLE.button_inactive)
            self._classifier_controls.add_button.setEnabled(False)
            self._classifier_controls.add_button.setStyleSheet(STYLE.button_inactive)
        else:
            self._all_recordings_button.setVisible(False)

            # Enables classifier builder for single-day sessions.
            self._classifier_controls.new_button.setEnabled(True)
            self._classifier_controls.new_button.setStyleSheet(STYLE.button_unpressed)
            self._classifier_controls.add_button.setEnabled(True)
            self._classifier_controls.add_button.setStyleSheet(STYLE.button_unpressed)

        # Resets channel 2 toggle state.
        self._view_controls.channel_2_button.setChecked(False)

        # Builds background views from detection images.
        self._views = build_views(
            frame_height=single_day.frame_height,
            frame_width=single_day.frame_width,
            mean_image=single_day.mean_image,
            enhanced_mean_image=single_day.enhanced_mean_image,
            correlation_map=single_day.correlation_map,
            maximum_projection=single_day.maximum_projection,
            corrected_structural_mean_image=single_day.corrected_structural_mean_image,
            channel_2=False,
            channel_2_mean_image=single_day.mean_image_channel_2,
            channel_2_enhanced_mean_image=single_day.enhanced_mean_image_channel_2,
            channel_2_correlation_map=single_day.correlation_map_channel_2,
            channel_2_maximum_projection=single_day.maximum_projection_channel_2,
            valid_y_range=single_day.valid_y_range,
            valid_x_range=single_day.valid_x_range,
        )

        # Computes color statistics and builds ROI index maps.
        self._color_arrays = compute_colors(
            roi_statistics=self._roi_statistics,
            frame_height=single_day.frame_height,
            frame_width=single_day.frame_width,
            cell_classification=self._cell_classification,
            cell_colocalization=self._cell_colocalization,
            roi_colormap=self._roi_colormap,
            colocalization_threshold=self._colocalization_threshold,
            two_channels=self._two_channels,
        )
        self._roi_maps = initialize_roi_maps(
            roi_statistics=self._roi_statistics,
            frame_height=single_day.frame_height,
            frame_width=single_day.frame_width,
            color_arrays=self._color_arrays,
        )

        # Selects the first classified cell as the initial selection.
        first_cell = int(np.nonzero(self._cell_classification[:, 1])[0][0]) if self._cell_count > 0 else 0
        self._selected_roi_index = first_cell
        self._selected_roi_indices = [first_cell]
        self._update_selected_roi_statistics()
        self._trace_controls.activity_combo.setCurrentIndex(TraceMode.DECONVOLVED)

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
            frame_height=single_day.frame_height,
            frame_width=single_day.frame_width,
            color_arrays=self._color_arrays,
            roi_maps=self._roi_maps,
            roi_color_mode=self._roi_color_mode,
            background_view=self._background_view,
            selected_roi_indices=self._selected_roi_indices,
            roi_opacity=self._view_controls.opacity_slider.value(),
            classification_label_mode=self._classification_label_mode,
        )
        display_masks(overlay_item=self._overlay, mask=mask)

        # Initializes plot ranges.
        self._view_box.setXRange(0, single_day.frame_width)
        self._view_box.setYRange(0, single_day.frame_height)
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
            activity_mode=self._trace_mode,
        )

        # Sets aspect ratio on the image panel.
        self._view_box.setAspectLocked(lock=True, ratio=single_day.aspect_ratio)

        self._session_loaded = True

        # Computes binned activity and triggers initial full redraw.
        self._on_activity_changed(TraceMode.DECONVOLVED)
        self.show()

    def _enable_controls(self) -> None:
        """Enables all view, color, and selection dropdowns after data loading."""
        if self._context_data is None:
            return
        single_day = self._context_data.single_day

        # Enables view dropdown and sets initial selection.
        self._view_controls.view_combo.setEnabled(True)
        self._view_controls.view_combo.setCurrentIndex(0)

        # Disables corrected structural view item if not available.
        view_model = self._view_controls.view_combo.model()
        if isinstance(view_model, QStandardItemModel):
            structural_item = view_model.item(BackgroundView.CORRECTED_STRUCTURAL)
            if structural_item is not None:
                if single_day.corrected_structural_mean_image.size == 0:
                    structural_item.setEnabled(False)
                else:
                    structural_item.setEnabled(True)

        # Enables channel 2 toggle if channel 2 data exists.
        if self._two_channels:
            self._view_controls.channel_2_button.setEnabled(True)
            self._view_controls.channel_2_button.setStyleSheet(STYLE.button_unpressed)
        else:
            self._view_controls.channel_2_button.setEnabled(False)
            self._view_controls.channel_2_button.setStyleSheet(STYLE.button_inactive)

        # Enables color dropdown and classification label toggle.
        self._color_controls.color_combo.setEnabled(True)
        self._color_controls.color_combo.setCurrentIndex(0)
        self._color_controls.classification_label_button.setEnabled(True)
        self._color_controls.classification_label_button.setStyleSheet(STYLE.button_unpressed)

        # Disables channel 2 color mode if not available.
        color_model = self._color_controls.color_combo.model()
        if isinstance(color_model, QStandardItemModel):
            channel_2_item = color_model.item(ROIColorMode.COLOCALIZATION_PROBABILITY)
            if channel_2_item is not None:
                channel_2_item.setEnabled(self._two_channels)

        # Enables selection dropdown.
        self._selection_controls.selection_combo.setEnabled(True)
        self._selection_controls.selection_combo.setCurrentIndex(0)

    def _on_view_changed(self, index: int) -> None:
        """Handles background view dropdown changes.

        Args:
            index: The background view index selected.
        """
        self._background_view = BackgroundView(index)
        self._update_plot()

    def _on_color_changed(self, index: int) -> None:
        """Handles ROI color mode dropdown changes.

        Args:
            index: The color mode index selected.
        """
        self._roi_color_mode = ROIColorMode(index)
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

    def _on_activity_changed(self, index: int) -> None:
        """Changes the activity mode used for multi-neuron display and correlation.

        Args:
            index: The activity mode index to switch to.
        """
        self._trace_mode = TraceMode(index)
        if self._session_loaded and self._context_data is not None:
            self._temporal_bin_size = max(1, int(self._color_controls.binning_edit.text()))
            bin_count = int(np.floor(float(self._frame_count) / float(self._temporal_bin_size)))

            # Selects the fluorescence array matching the active trace mode.
            if index == TraceMode.RAW_FLUORESCENCE:
                fluorescence = self._cell_fluorescence
            elif index == TraceMode.NEUROPIL:
                fluorescence = self._neuropil_fluorescence
            elif index == TraceMode.NEUROPIL_CORRECTED:
                fluorescence = self._subtracted_fluorescence
            else:
                fluorescence = self._spikes

            # Computes temporally binned and mean-subtracted fluorescence for correlation coloring.
            cell_count = len(self._roi_statistics)
            bin_size = self._temporal_bin_size
            self._binned_fluorescence = (
                fluorescence[:, : bin_count * bin_size].reshape((cell_count, bin_count, bin_size)).mean(axis=2)
            )
            self._binned_fluorescence -= self._binned_fluorescence.mean(axis=1)[:, np.newaxis]
            self._fluorescence_standard_deviation = (self._binned_fluorescence**2).mean(axis=1) ** 0.5
            self._frame_indices = np.arange(0, self._frame_count, dtype=np.int32)
            self._update_plot()

    def _on_channel_2_toggled(self, checked: bool) -> None:
        """Rebuilds the view stack when the channel 2 toggle changes.

        Args:
            checked: Determines whether channel 2 is toggled on.
        """
        if self._context_data is None:
            return
        self._view_controls.channel_2_button.setStyleSheet(STYLE.button_pressed if checked else STYLE.button_unpressed)
        single_day = self._context_data.single_day
        self._views = build_views(
            frame_height=single_day.frame_height,
            frame_width=single_day.frame_width,
            mean_image=single_day.mean_image,
            enhanced_mean_image=single_day.enhanced_mean_image,
            correlation_map=single_day.correlation_map,
            maximum_projection=single_day.maximum_projection,
            corrected_structural_mean_image=single_day.corrected_structural_mean_image,
            channel_2=checked,
            channel_2_mean_image=single_day.mean_image_channel_2,
            channel_2_enhanced_mean_image=single_day.enhanced_mean_image_channel_2,
            channel_2_correlation_map=single_day.correlation_map_channel_2,
            channel_2_maximum_projection=single_day.maximum_projection_channel_2,
            valid_y_range=single_day.valid_y_range,
            valid_x_range=single_day.valid_x_range,
        )
        self._update_plot()

    def _on_classification_label_toggled(self, checked: bool) -> None:
        """Switches between probability gradient and binary cell/non-cell label views.

        Args:
            checked: Determines whether the label view toggle is pressed.
        """
        self._classification_label_mode = checked
        self._color_controls.classification_label_button.setStyleSheet(
            STYLE.button_pressed if checked else STYLE.button_unpressed
        )
        self._update_plot()

    def _on_number_chosen(self) -> None:
        """Jumps to the ROI number entered in the ROI edit field."""
        if self._session_loaded and self._context_data is not None:
            self._selected_roi_index = int(self._roi_index_edit.text())
            roi_count = len(self._roi_statistics)
            if self._selected_roi_index >= roi_count:
                self._selected_roi_index = roi_count - 1
            self._selected_roi_indices = [self._selected_roi_index]
            self._update_plot()

    def _on_zoom_cell_toggled(self, state: int) -> None:
        """Toggles zoom-to-cell behavior based on checkbox state.

        Args:
            state: The Qt checkbox state value.
        """
        if not self._session_loaded:
            return
        self._auto_zoom_to_roi = QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked
        self._update_plot()

    def _on_view_selector_changed(self, combo_index: int) -> None:
        """Handles view selector dropdown changes by switching to the selected view.

        Maps the combo box index to a view index (accounting for the combined view offset) and reloads all GUI
        components for the new view.

        Args:
            combo_index: The index selected in the view selector combo box.
        """
        if combo_index < 0 or self._context_data is None:
            return

        # Maps combo index to view_index: combo 0 maps to view_index -1 (combined), combo 1+ maps to planes.
        view_index = combo_index - 1

        self._context_data.single_day.switch_view(view_index=view_index)
        self._reset_state()
        self._initialize_gui()

    def _on_trace_toggle(self, which: str) -> None:
        """Handles trace visibility checkbox toggles.

        Args:
            which: The trace type to toggle ("deconvolved", "neuropil", or "traces").
        """
        trace_controls = self._trace_controls
        if which == "deconvolved":
            trace_controls.deconvolved_visible = trace_controls.deconvolved_checkbox.isChecked()
        elif which == "neuropil":
            trace_controls.neuropil_visible = trace_controls.neuropil_checkbox.isChecked()
        elif which == "traces":
            trace_controls.traces_visible = trace_controls.traces_checkbox.isChecked()
        self._refresh_traces()

    def _refresh_traces(self) -> None:
        """Refreshes the trace panel without redrawing image panels."""
        if self._context_data is None or self._color_arrays is None or self._frame_indices is None:
            return

        # In multi-day mode with "All Recordings" enabled and exactly one ROI selected, shows stacked traces.
        if self._is_multi_day and self._all_recordings_visible and len(self._selected_roi_indices) == 1:
            self._refresh_all_recording_traces()
            return

        plot_trace(
            trace_box=self._trace_box,
            cell_fluorescence=self._cell_fluorescence,
            neuropil_fluorescence=self._neuropil_fluorescence,
            subtracted_fluorescence=self._subtracted_fluorescence,
            spikes=self._spikes,
            frame_indices=self._frame_indices,
            selected_indices=self._selected_roi_indices,
            activity_mode=self._trace_mode,
            roi_colors=self._color_arrays.colors[self._roi_color_mode],
            traces_visible=self._trace_controls.traces_visible,
            neuropil_visible=self._trace_controls.neuropil_visible,
            deconvolved_visible=self._trace_controls.deconvolved_visible,
            scale_factor=ROI_CONFIG.default_scale_factor,
            max_plotted=int(self._trace_controls.max_plotted_edit.text() or str(ROI_CONFIG.plotted_trace_count)),
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
            frame_height=self._context_data.single_day.frame_height,
            frame_width=self._context_data.single_day.frame_width,
            color_arrays=self._color_arrays,
            roi_maps=self._roi_maps,
            roi_color_mode=self._roi_color_mode,
            background_view=self._background_view,
            selected_roi_indices=self._selected_roi_indices,
            roi_opacity=self._view_controls.opacity_slider.value(),
            classification_label_mode=self._classification_label_mode,
        )
        display_masks(overlay_item=self._overlay, mask=mask)

        # Refreshes traces and applies zoom if enabled.
        self._refresh_traces()
        if self._auto_zoom_to_roi:
            self._zoom_to_cell()
        self._view_box.show()
        self._graphics_widget.show()
        self.show()

        # Updates the status bar with session and selection info.
        single_day = self._context_data.single_day
        self._status_bar.showMessage(
            f"Session: {single_day.output_path}  |  ROI: {self._selected_roi_index}  |  "
            f"Cells: {self._cell_count}  |  Size: {single_day.frame_height} x {single_day.frame_width}"
        )

    def _update_selected_roi_statistics(self) -> None:
        """Updates the ROI statistics labels for the currently selected cell."""
        if self._context_data is None:
            return
        roi_index = self._selected_roi_index
        self._roi_index_edit.setText(str(roi_index))
        roi = self._roi_statistics[roi_index]
        for label_index, statistic_name in enumerate(_STATISTICS_TO_SHOW):
            value = getattr(roi, statistic_name, None)
            if value is None:
                continue
            if isinstance(value, tuple):
                self._roi_statistic_labels[label_index].setText(f"{statistic_name}: [{value[0]:d}, {value[1]:d}]")
            elif isinstance(value, int):
                self._roi_statistic_labels[label_index].setText(f"{statistic_name}: {value:d}")
            else:
                self._roi_statistic_labels[label_index].setText(f"{statistic_name}: {value:2.2f}")

    def _toggle_rois(self, state: int) -> None:
        """Toggles ROI overlay visibility on the image panel.

        Args:
            state: The Qt checkbox state value.
        """
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self._rois_visible = True
            self._view_box.addItem(self._overlay)
        else:
            self._rois_visible = False
            self._view_box.removeItem(self._overlay)
        self._graphics_widget.show()
        self.show()

    def _toggle_roi_labels(self, state: int) -> None:
        """Toggles ROI number text labels on the image panel.

        Args:
            state: The Qt checkbox state value.
        """
        if self._roi_maps is None or self._context_data is None:
            return

        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            for label in self._roi_maps.text_labels:
                self._view_box.addItem(label)
            self._roi_labels_visible = True
        else:
            for label in self._roi_maps.text_labels:
                with suppress(Exception):
                    self._view_box.removeItem(label)
            self._roi_labels_visible = False

    def _zoom_to_cell(self) -> None:
        """Zooms the image panel to center on the currently selected cell."""
        if self._context_data is None:
            return
        single_day = self._context_data.single_day
        zoom_range = (
            ROI_CONFIG.zoom_to_cell_fraction * np.array([single_day.frame_height, single_day.frame_width]).max()
        )
        roi_statistics = self._roi_statistics

        if len(self._selected_roi_indices) > 1:
            # Collects all pixel coordinates from the selected ROIs to compute a bounding box.
            all_pixels = np.zeros((0, 2))
            for roi_index in self._selected_roi_indices:
                all_pixels = np.append(
                    all_pixels,
                    np.concatenate(
                        (
                            roi_statistics[roi_index].mask.y_pixels.flatten()[:, np.newaxis],
                            roi_statistics[roi_index].mask.x_pixels.flatten()[:, np.newaxis],
                        ),
                        axis=1,
                    ),
                    axis=0,
                )
            minimum_bounds = all_pixels.min(axis=0)
            maximum_bounds = all_pixels.max(axis=0)
            centroid = all_pixels.mean(axis=0)

            # Expands the bounding box to at least the zoom range around the centroid.
            minimum_bounds[0] = min(centroid[0] - zoom_range, minimum_bounds[0])
            minimum_bounds[1] = min(centroid[1] - zoom_range, minimum_bounds[1])
            maximum_bounds[0] = max(centroid[0] + zoom_range, maximum_bounds[0])
            maximum_bounds[1] = max(centroid[1] + zoom_range, maximum_bounds[1])
        else:
            centroid = np.array(roi_statistics[self._selected_roi_index].mask.centroid)
            minimum_bounds = centroid - zoom_range
            maximum_bounds = centroid + zoom_range

        self._view_box.setYRange(minimum_bounds[0], maximum_bounds[0])
        self._view_box.setXRange(minimum_bounds[1], maximum_bounds[1])

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
        if not self._session_loaded or self._roi_maps is None or self._context_data is None:
            return False

        single_day = self._context_data.single_day
        if click_y < 0 or click_y >= single_day.frame_height or click_x < 0 or click_x >= single_day.frame_width:
            return False

        chosen_index = int(self._roi_maps.roi_indices[0, click_y, click_x])
        if chosen_index < 0:
            return False

        if is_right_button:
            # Reclassification is disabled in multi-day mode.
            if self._is_multi_day:
                return False
            # Reclassification is only available in cell classification label mode.
            if self._roi_color_mode != ROIColorMode.CELL_CLASSIFICATION or not self._classification_label_mode:
                return False
            if chosen_index not in self._selected_roi_indices:
                self._selected_roi_indices = [chosen_index]
                self._selected_roi_index = chosen_index
            self._flip_plot()
            return True

        # Multi-day mode restricts selection to a single ROI.
        if self._is_multi_day:
            self._selected_roi_indices = [chosen_index]
            self._selected_roi_index = chosen_index
        else:
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

        self._update_plot()
        return True

    def _on_top_bottom_selection(self) -> None:
        """Selects the top-n or bottom-n ROIs ranked by the active color statistic."""
        if self._color_arrays is None:
            return
        count = int(self._selection_controls.top_count_edit.text() or str(ROI_CONFIG.top_selection_count))
        count = min(count, ROI_CONFIG.top_selection_count)
        values = self._color_arrays.normalized_statistics[self._roi_color_mode]
        ranked = np.argsort(values)
        # Index 0 = "select top n", index 1 = "select bottom n".
        if self._selection_controls.selection_combo.currentIndex() == 0:
            selected = ranked[-count:][::-1]
        else:
            selected = ranked[:count]
        self._selected_roi_indices = selected.tolist()
        if self._selected_roi_indices:
            self._selected_roi_index = self._selected_roi_indices[0]
            self._update_plot()

    def _on_dataset_source_changed(self, index: int) -> None:
        """Handles ROI Source dropdown changes to switch between single-day and multi-day data.

        Args:
            index: The selected combo box index. 0 = Original (single-day), 1+ = multi-day datasets.
        """
        if self._context_data is None:
            return

        if index == 0:
            self._context_data.unload_dataset()
        elif index > 0:
            available = self._context_data.available_datasets
            dataset_index = index - 1
            if dataset_index < len(available):
                self._context_data.load_dataset(dataset_name=available[dataset_index])
            else:
                return
        else:
            return

        self._reset_state()
        self._initialize_gui()

    def _on_all_recordings_toggled(self, checked: bool) -> None:
        """Handles the All Recordings toggle button for multi-day stacked trace display.

        Args:
            checked: Determines whether the stacked all-recordings view is enabled.
        """
        self._all_recordings_visible = checked
        self._all_recordings_button.setStyleSheet(STYLE.button_pressed if checked else STYLE.button_unpressed)
        self._refresh_traces()

    def _refresh_all_recording_traces(self) -> None:
        """Plots traces from all recordings stacked vertically for the selected ROI.

        Iterates over every recording in the multi-day dataset, extracts the selected ROI's trace from each, and plots
        them stacked with recording-index labels on the y-axis.
        """
        if self._context_data is None or not self._context_data.is_multi_day:
            return

        self._trace_box.clear()
        if not self._selected_roi_indices:
            return

        roi_index = self._selected_roi_indices[0]
        axis = self._trace_box.getAxis("left")
        tick_labels: list[tuple[float, str]] = []
        trace_spacing = 1.0 / ROI_CONFIG.default_scale_factor
        max_frames = 0
        y_maximum = 0.0
        stack_position = self._context_data.recording_count - 1

        for recording_index in range(self._context_data.recording_count):
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
            if self._trace_mode == TraceMode.RAW_FLUORESCENCE:
                trace = cell_fluorescence[roi_index, :]
            elif self._trace_mode == TraceMode.NEUROPIL:
                trace = neuropil_fluorescence[roi_index, :]
            elif self._trace_mode == TraceMode.NEUROPIL_CORRECTED:
                trace = subtracted_fluorescence[roi_index, :]
            else:
                trace = spikes[roi_index, :]

            frame_indices = np.arange(len(trace), dtype=np.int32)
            max_frames = max(max_frames, len(trace))

            # Normalizes trace to [0, 1] range for stacked display.
            trace_max = float(trace.max())
            trace_min = float(trace.min())
            if trace_max > trace_min:
                normalized = (trace - trace_min) / (trace_max - trace_min)
            else:
                normalized = np.zeros_like(trace)

            # Generates a deterministic color for this recording using HSV hues.
            hue = recording_index / max(self._context_data.recording_count, 1)
            hsv = np.array([[hue, 1.0, 1.0]])
            rgb = (255.0 * hsv_to_rgb(hsv)).astype(np.uint8)[0]
            pen_color = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

            self._trace_box.plot(frame_indices, normalized + stack_position * trace_spacing, pen=pen_color)
            tick_labels.append((stack_position * trace_spacing + float(normalized.mean()), str(recording_index)))
            y_maximum = max(y_maximum, stack_position * trace_spacing + 1)
            stack_position -= 1

        axis.setTicks([tick_labels])
        self._trace_box.update_range(
            frame_count=max_frames,
            y_minimum=0.0,
            y_maximum=y_maximum if y_maximum > 0 else 1.0,
        )

    def _extract_classifier_data(
        self,
    ) -> tuple[NDArray[np.bool_], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]] | None:
        """Extracts training labels and classification features from the current session state.

        Returns:
            A tuple of (training_labels, normalized_pixel_count, compactness, skewness) arrays, or None if no session
            is loaded or no ROIs exist.
        """
        if not self._session_loaded or not self._roi_statistics:
            return None

        training_labels = (self._cell_classification[:, 1] > ROI_CONFIG.default_classifier_threshold).astype(np.bool_)
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
        if self._is_multi_day:
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
        """Handles the Add to Existing button by merging current session data into an existing training dataset."""
        if self._is_multi_day:
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
