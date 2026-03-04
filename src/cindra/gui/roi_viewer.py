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


class ROIViewer(QMainWindow):
    """Single-panel ROI viewer for single-day pipeline outputs with right-click reclassification.

    Displays ROI overlays, background images, and fluorescence traces. Supports left-click ROI
    selection, shift/ctrl multi-select, right-click cell/non-cell reclassification (active only
    in cell/non-cell color mode) with auto-save, keyboard shortcuts for view/color switching,
    and double-click zoom-to-fit.

    Args:
        data: Pre-loaded viewer data. If None the viewer starts empty and the user can load
            a session via the File menu or drag-and-drop.
    """

    def __init__(self, data: ViewerData | None = None) -> None:
        super().__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")

        # Display state (replaces ViewState — lives directly on the window).
        self.rois_visible: bool = True
        self.roi_color_mode: int = ROIColorMode.RANDOM
        self.background_view: int = BackgroundView.ROIS_ONLY
        self.roi_colormap: str = Colormap.HSV
        self.selected_roi_index: int = 0
        self.selected_roi_indices: list[int] = [0]
        self.trace_mode: int = TraceMode.NEUROPIL_CORRECTED
        self.temporal_bin_size: int = 1
        self.auto_zoom_to_roi: bool = False
        self.roi_labels_visible: bool = False
        self.session_loaded: bool = False
        self.colocalization_threshold: float = ROI_CONFIG.default_channel_2_threshold
        self.last_reclassified_index: int = -1
        self.classification_label_mode: bool = False

        # Multi-day state. Persists across _reset_state calls.
        self._all_recordings_visible: bool = False

        # Core data objects.
        self.context_data: ViewerData | None = None
        self.color_arrays: ColorArrays | None = None
        self.roi_maps: ROIIndexMaps | None = None
        self.colorbar_widgets: ColorbarWidgets | None = None
        self.colorbar_image: NDArray[np.uint8] | None = None
        self.views: NDArray[np.uint8] | None = None

        # Mode-dependent data cache. Populated by _initialize_gui() from either single_day or current_recording.
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

        # Binned activity state (used by correlation coloring).
        self.Fbin: NDArray[np.float32] | None = None
        self.Fstd: NDArray[np.float32] | None = None
        self.frame_indices: NDArray[np.intp] | None = None

        # Window geometry and title.
        self.setGeometry(*ROI_STYLE.window_geometry)
        self.setWindowTitle("cindra ROI Viewer (load session directory)")

        self.setStyleSheet(STYLE.main_window)

        # File-only menu bar.
        self._build_menus()

        # Main widget layout: graphics | control panel.
        central_widget = QWidget(self)
        main_layout = QHBoxLayout(central_widget)
        self.setCentralWidget(central_widget)

        # Left: graphics (stretch=3).
        self._graphics_widget = pg.GraphicsLayoutWidget()
        main_layout.addWidget(self._graphics_widget, stretch=3)

        # Right: control panel (stretch=1).
        control_panel = self._build_control_panel()
        main_layout.addWidget(control_panel, stretch=1)

        # Status bar.
        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)

        # Builds graphics panels.
        self._build_graphics()

        # Applies NoFocus policy to all buttons in the control panel.
        for widget in control_panel.findChildren(QWidget):
            widget.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        # Accepts drag-and-drop of directories.
        self.setAcceptDrops(True)

        # Loads data if provided.
        if data is not None:
            self.context_data = data
            self._initialize_gui()

        self.show()
        self._graphics_widget.show()

    @property
    def _is_multi_day(self) -> bool:
        """Returns True when the viewer is displaying multi-day tracked ROI data."""
        return self.context_data is not None and self.context_data.is_multi_day

    def _build_menus(self) -> None:
        """Builds the File-only menu bar for the read-only viewer."""
        file_menu = self.menuBar().addMenu("&File")

        load_action = file_menu.addAction("&Load session")
        load_action.setShortcut("Ctrl+L")
        load_action.triggered.connect(self._load_session)

    def _build_control_panel(self) -> QWidget:
        """Builds the right-side control panel with QGroupBox sections.

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

        # 0. View selector (Combined / Plane 0 / Plane 1 / ...).
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

        # 1. ROI Visibility.
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
        self._roi_labels_checkbox.stateChanged.connect(self._roi_text)
        self._roi_labels_checkbox.setEnabled(False)
        visibility_layout.addWidget(self._roi_labels_checkbox)
        layout.addWidget(visibility_box)

        # 2. Cell Selection.
        selection_box, self._selection_controls = self._create_selection_buttons()
        layout.addWidget(selection_box)

        # 3. Background Views.
        background_box, self._view_controls = self._create_view_controls()
        layout.addWidget(background_box)

        # 4. ROI Colors + colorbar.
        colors_box, self._color_controls = self._create_color_controls()
        self.colorbar_widgets = self._create_colorbar()
        colors_layout = colors_box.layout()
        if colors_layout is not None:
            colors_layout.addWidget(self.colorbar_widgets.widget)
        layout.addWidget(colors_box)

        # 5. Selected ROI — ROI index edit + stat labels.
        roi_box = QGroupBox("Selected ROI")
        roi_box.setStyleSheet(STYLE.group_box)
        roi_layout = QVBoxLayout(roi_box)
        self._stats_to_show = [
            "centroid",
            "pixel_count",
            "skewness",
            "compactness",
            "footprint",
            "aspect_ratio",
        ]
        self._roi_index_edit = QLineEdit(self)
        self._roi_index_edit.setValidator(QtGui.QIntValidator(0, 10000))
        self._roi_index_edit.setText("0")
        self._roi_index_edit.setFixedWidth(STYLE.edit_width)
        self._roi_index_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._roi_index_edit.returnPressed.connect(self._on_number_chosen)
        roi_layout.addWidget(self._roi_index_edit)
        self._roi_stat_labels: list[QLabel] = []
        for k in range(len(self._stats_to_show)):
            stat_label = QLabel(self._stats_to_show[k])
            stat_label.setStyleSheet(STYLE.white_label)
            stat_label.resize(stat_label.minimumSizeHint())
            roi_layout.addWidget(stat_label)
            self._roi_stat_labels.append(stat_label)
        layout.addWidget(roi_box)

        # 6. Trace Display.
        trace_box, self._trace_controls = self._create_trace_controls()
        self._zoom_to_cell_checkbox = QCheckBox("zoom to cell")
        self._zoom_to_cell_checkbox.setStyleSheet(STYLE.white_label)
        self._zoom_to_cell_checkbox.stateChanged.connect(self._on_zoom_cell_toggled)
        trace_layout = trace_box.layout()
        if trace_layout is not None:
            trace_layout.addWidget(self._zoom_to_cell_checkbox)
        layout.addWidget(trace_box)

        # 7. Classifier Builder.
        classifier_box, self._classifier_controls = self._create_classifier_controls()
        layout.addWidget(classifier_box)

        layout.addStretch()
        return panel

    def _create_selection_buttons(self) -> tuple[QGroupBox, SelectionControls]:
        """Creates the cell selection dropdown and top-n input field."""
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
        """Creates background view dropdown, channel 2 toggle, and saturation slider."""
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
        opacity_slider.valueChanged.connect(self.update_plot)
        layout.addWidget(opacity_slider, 1, 1, 1, 1)

        controls = ViewControls(view_combo=view_combo, channel_2_button=channel_2_button, opacity_slider=opacity_slider)
        return group_box, controls

    def _create_color_controls(self) -> tuple[QGroupBox, ColorControls]:
        """Creates color statistic dropdown and associated controls."""
        group_box = QGroupBox("ROI Colors")
        group_box.setStyleSheet(STYLE.group_box)
        layout = QGridLayout(group_box)

        color_combo = QComboBox(self)
        color_combo.addItems(list(ROIColorModeLabel))
        color_combo.setFont(FONTS.small_bold)
        color_combo.setEnabled(False)
        color_combo.activated.connect(self._on_color_changed)
        layout.addWidget(color_combo, 0, 0, 1, 1)

        colormap_chooser = QComboBox()
        colormap_chooser.addItems([cm.value for cm in Colormap])
        colormap_chooser.setCurrentIndex(0)
        colormap_chooser.setFont(FONTS.small_bold)
        colormap_chooser.setFixedWidth(ROI_STYLE.color_edit_width)
        layout.addWidget(colormap_chooser, 0, 1, 1, 1)

        classifier_label = QLabel("cell prob=")
        classifier_label.setStyleSheet(STYLE.white_label)
        layout.addWidget(classifier_label, 1, 0, 1, 1)

        classifier_edit = QLineEdit(self)
        classifier_edit.setText("0.5")
        classifier_edit.setFixedWidth(ROI_STYLE.color_edit_width)
        classifier_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(classifier_edit, 1, 1, 1, 1)
        classifier_edit.returnPressed.connect(self.update_plot)

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

        colormap_chooser.activated.connect(lambda: self._on_color_changed(self.roi_color_mode))

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
        """Creates the colorbar widget displaying the current color mapping."""
        colorbar_widget = pg.GraphicsLayoutWidget(self)
        colorbar_widget.setMaximumHeight(ROI_STYLE.colorbar_max_height)
        colorbar_widget.setMaximumWidth(ROI_STYLE.colorbar_max_width)
        colorbar_widget.ci.layout.setRowStretchFactor(0, 2)
        colorbar_widget.ci.layout.setContentsMargins(0, 0, 0, 0)

        image = pg.ImageItem()
        colorbar_view = colorbar_widget.addViewBox(row=0, col=0, colspan=3)
        colorbar_view.setMenuEnabled(False)
        colorbar_view.addItem(image)

        colorbar_font = FONTS.small
        label_0 = colorbar_widget.addLabel("0.0", color=list(COLORS.white), row=1, col=0)
        label_0.setFont(colorbar_font)
        label_half = colorbar_widget.addLabel("0.5", color=list(COLORS.white), row=1, col=1)
        label_half.setFont(colorbar_font)
        label_1 = colorbar_widget.addLabel("1.0", color=list(COLORS.white), row=1, col=2)
        label_1.setFont(colorbar_font)
        labels = [label_0, label_half, label_1]
        return ColorbarWidgets(image=image, labels=labels, widget=colorbar_widget)

    def _create_classifier_controls(self) -> tuple[QGroupBox, ClassifierControls]:
        """Creates the classifier builder panel with New and Add to Existing buttons."""
        group_box = QGroupBox("Classifier")
        group_box.setStyleSheet(STYLE.group_box)
        layout = QGridLayout(group_box)

        new_button = QPushButton("New", self)
        new_button.setFont(FONTS.small_bold)
        new_button.setStyleSheet(STYLE.button_unpressed)
        new_button.clicked.connect(self._on_classifier_new)
        layout.addWidget(new_button, 0, 0, 1, 1)

        add_button = QPushButton("Add to Existing", self)
        add_button.setFont(FONTS.small_bold)
        add_button.setStyleSheet(STYLE.button_unpressed)
        add_button.clicked.connect(self._on_classifier_add_to_existing)
        layout.addWidget(add_button, 0, 1, 1, 1)

        status_label = QLabel("")
        status_label.setStyleSheet(STYLE.white_label)
        status_label.setFont(FONTS.small_bold)
        status_label.setWordWrap(True)
        layout.addWidget(status_label, 1, 0, 1, 2)

        controls = ClassifierControls(new_button=new_button, add_button=add_button, status_label=status_label)
        return group_box, controls

    def _create_trace_controls(self) -> tuple[QGroupBox, TraceControls]:
        """Creates trace panel controls inside a group box."""
        group_box = QGroupBox("Trace Display")
        group_box.setStyleSheet(STYLE.group_box)
        layout = QGridLayout(group_box)

        activity_label = QLabel("Activity mode:")
        activity_label.setStyleSheet(STYLE.white_label)
        layout.addWidget(activity_label, 0, 0, 1, 1)

        activity_combo = QComboBox(self)
        layout.addWidget(activity_combo, 1, 0, 1, 1)
        activity_combo.addItems(list(TraceModeLabel))
        activity_combo.setCurrentIndex(TraceMode.DECONVOLVED)
        activity_combo.currentIndexChanged.connect(self._on_activity_changed)

        max_plotted_label = QLabel("max # plotted:")
        max_plotted_label.setStyleSheet(STYLE.white_label)
        layout.addWidget(max_plotted_label, 2, 0, 1, 1)

        max_plotted_edit = QLineEdit(self)
        max_plotted_edit.setValidator(QtGui.QIntValidator(0, ROI_CONFIG.plotted_trace_count))
        max_plotted_edit.setText(str(ROI_CONFIG.plotted_trace_count))
        max_plotted_edit.setFixedWidth(STYLE.edit_width)
        max_plotted_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(max_plotted_edit, 3, 0, 1, 1)

        deconvolved_checkbox = QCheckBox("deconv [N]")
        deconvolved_checkbox.setStyleSheet(STYLE.white_label)
        deconvolved_checkbox.toggle()
        layout.addWidget(deconvolved_checkbox, 3, 1, 1, 1)

        neuropil_checkbox = QCheckBox("neuropil [B]")
        neuropil_checkbox.setStyleSheet(STYLE.white_label)
        neuropil_checkbox.toggle()
        layout.addWidget(neuropil_checkbox, 3, 2, 1, 1)

        traces_checkbox = QCheckBox("raw fluor [V]")
        traces_checkbox.setStyleSheet(STYLE.white_label)
        traces_checkbox.toggle()
        layout.addWidget(traces_checkbox, 3, 3, 1, 1)

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

        max_plotted_edit.returnPressed.connect(self._refresh_traces)
        deconvolved_checkbox.toggled.connect(lambda: self._on_trace_toggle("deconvolved"))
        neuropil_checkbox.toggled.connect(lambda: self._on_trace_toggle("neuropil"))
        traces_checkbox.toggled.connect(lambda: self._on_trace_toggle("traces"))

        return group_box, controls

    def _build_graphics(self) -> None:
        """Creates the main plotting area with image and trace panels."""
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

        self._view_box.set_click_handler(self._handle_click)
        self._view_box.set_zoom_handler(self._zoom_plot)

        self._trace_box = TraceBox()
        self._trace_box.enableAutoRange(x=True, y=True)
        self._graphics_widget.addItem(self._trace_box, row=1, col=0)
        self._graphics_widget.ci.layout.setRowStretchFactor(0, 2)
        gl = self._graphics_widget.ci.layout
        gl.setColumnMinimumWidth(0, 1)
        gl.setHorizontalSpacing(20)

    def _load_session(self, session_path: Path | None = None) -> None:
        """Loads a pipeline output directory into the viewer.

        Args:
            session_path: Path to the cindra output directory. If None, opens a dialog.
        """
        if session_path is None:
            name = QFileDialog.getExistingDirectory(parent=self, caption="Open cindra output directory")
            if not name:
                return
            session_path = Path(name)

        console.echo(message=f"Loading session: {session_path}")

        try:
            context_data = ViewerData.from_data(root_path=session_path)
        except Exception:
            console.echo(message="Failed to load session data.", level=LogLevel.ERROR)
            result = QMessageBox.question(
                self,
                "ERROR",
                "Failed to load session. Try another directory?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if result == QMessageBox.StandardButton.Yes:
                self._load_session()
            return

        self.context_data = context_data

        # Populates the ROI Source combo with "Original" + discovered datasets.
        self._roi_source_combo.blockSignals(True)
        self._roi_source_combo.clear()
        self._roi_source_combo.addItem("Original")
        for name in context_data.available_datasets:
            self._roi_source_combo.addItem(f"Dataset: {name}")
        # Selects the active dataset if one is loaded, otherwise selects "Original".
        if context_data.is_multi_day:
            for i in range(1, self._roi_source_combo.count()):
                item_text = self._roi_source_combo.itemText(i)
                if item_text == f"Dataset: {context_data.active_dataset_name}":
                    self._roi_source_combo.setCurrentIndex(i)
                    break
        else:
            self._roi_source_combo.setCurrentIndex(0)
        self._roi_source_combo.blockSignals(False)
        self._roi_source_group.setVisible(bool(context_data.available_datasets))

        self._reset_state()
        self._initialize_gui()

    def _reset_state(self) -> None:
        """Resets all display state to defaults before loading new data.

        Does NOT reset ``context_data`` or the ROI Source dropdown, which persist across state resets.
        """
        self.rois_visible = True
        self.roi_color_mode = ROIColorMode.RANDOM
        self.background_view = BackgroundView.ROIS_ONLY
        self._view_controls.opacity_slider.setValue(STYLE.default_mask_opacity)
        self.roi_colormap = Colormap.HSV
        self.selected_roi_index = 0
        self.selected_roi_indices = [0]
        self.trace_mode = TraceMode.NEUROPIL_CORRECTED
        self.temporal_bin_size = 1
        self.auto_zoom_to_roi = False
        self.roi_labels_visible = False
        self.session_loaded = False
        self.colocalization_threshold = ROI_CONFIG.default_channel_2_threshold
        self.last_reclassified_index = -1
        self.classification_label_mode = False
        self._color_controls.classification_label_button.setChecked(False)
        self._all_recordings_visible = False

    def _initialize_gui(self) -> None:
        """Initializes all GUI components after loading context data."""
        context = self.context_data
        if context is None:
            return

        sd = context.single_day
        is_multi_day = self._is_multi_day

        # Resolves mode-dependent data from the appropriate source.
        if is_multi_day:
            rec = context.current_recording
            self._roi_statistics = list(rec.tracked_masks)
            n = len(self._roi_statistics)
            self._cell_classification = (
                np.column_stack([np.ones(n, dtype=np.float32), np.ones(n, dtype=np.float32)])
                if n > 0
                else np.empty((0, 2), dtype=np.float32)
            )
            self._cell_colocalization = np.zeros((n, 2), dtype=np.float32)
            self._two_channels = rec.has_channel_2
            self._cell_fluorescence = rec.cell_fluorescence
            self._neuropil_fluorescence = rec.neuropil_fluorescence
            self._subtracted_fluorescence = rec.subtracted_fluorescence
            self._spikes = rec.spikes
            self._frame_count = int(self._cell_fluorescence.shape[1]) if self._cell_fluorescence.size > 0 else 0
            self._cell_count = n
        else:
            self._roi_statistics = sd.roi_statistics
            self._cell_classification = sd.cell_classification
            self._cell_colocalization = sd.cell_colocalization
            self._two_channels = sd.two_channels
            self._cell_fluorescence = sd.cell_fluorescence
            self._neuropil_fluorescence = sd.neuropil_fluorescence
            self._subtracted_fluorescence = sd.subtracted_fluorescence
            self._spikes = sd.spikes
            self._frame_count = sd.frame_count
            self._cell_count = sd.cell_count

        # Populates the view selector without triggering _on_view_selector_changed. Signals are
        # blocked so that clearing and re-adding items does not fire redundant view-switch callbacks.
        self._view_selector.blockSignals(True)
        self._view_selector.clear()
        if is_multi_day:
            self._view_selector.addItem("Combined")
            self._view_selector.setCurrentIndex(0)
        else:
            for label in sd.view_labels:
                self._view_selector.addItem(label)
            self._view_selector.setCurrentIndex(sd.view_index + 1)
        self._view_selector.blockSignals(False)
        self._view_selector.setEnabled(not is_multi_day and len(sd.view_labels) > 1)

        # Resets display controls.
        self._roi_visibility_checkbox.setChecked(True)
        if self._roi_labels_checkbox.isChecked():
            self._roi_text(False)
        self._roi_labels_checkbox.setChecked(False)
        self._roi_labels_checkbox.setEnabled(True)

        session_title = str(sd.output_path)
        self.setWindowTitle(f"cindra ROI Viewer — {session_title}")

        # Computes default bin size from tau and sampling rate.
        self.temporal_bin_size = max(1, int(sd.tau * sd.sampling_rate / ROI_CONFIG.bin_size_divisor))
        self._color_controls.binning_edit.setText(str(self.temporal_bin_size))
        self.colocalization_threshold = ROI_CONFIG.default_channel_2_threshold

        # Enables buttons.
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
        self.views = build_views(
            frame_height=sd.frame_height,
            frame_width=sd.frame_width,
            mean_image=sd.mean_image,
            enhanced_mean_image=sd.enhanced_mean_image,
            correlation_map=sd.correlation_map,
            maximum_projection=sd.maximum_projection,
            corrected_structural_mean_image=sd.corrected_structural_mean_image,
            channel_2=False,
            channel_2_mean_image=sd.mean_image_channel_2,
            channel_2_enhanced_mean_image=sd.enhanced_mean_image_channel_2,
            channel_2_correlation_map=sd.correlation_map_channel_2,
            channel_2_maximum_projection=sd.maximum_projection_channel_2,
            valid_y_range=sd.valid_y_range,
            valid_x_range=sd.valid_x_range,
        )

        # Computes color statistics and builds ROI index maps.
        self.color_arrays = compute_colors(
            roi_statistics=self._roi_statistics,
            frame_height=sd.frame_height,
            frame_width=sd.frame_width,
            cell_classification=self._cell_classification,
            cell_colocalization=self._cell_colocalization,
            roi_colormap=self.roi_colormap,
            colocalization_threshold=self.colocalization_threshold,
            two_channels=self._two_channels,
        )
        self.roi_maps = initialize_roi_maps(
            roi_statistics=self._roi_statistics,
            frame_height=sd.frame_height,
            frame_width=sd.frame_width,
            color_arrays=self.color_arrays,
        )

        # Selects the first classified cell as the initial selection.
        first_cell = int(np.nonzero(self._cell_classification[:, 1])[0][0]) if self._cell_count > 0 else 0
        self.selected_roi_index = first_cell
        self.selected_roi_indices = [first_cell]
        self._ichosen_stats()
        self._trace_controls.activity_combo.setCurrentIndex(TraceMode.DECONVOLVED)

        # Draws the colorbar and initial mask overlays.
        self.colorbar_image = draw_colorbar(colormap=self.roi_colormap)
        if self.colorbar_widgets is None or self.colorbar_image is None:
            return
        render_colorbar(
            roi_color_mode=self.roi_color_mode,
            color_arrays=self.color_arrays,
            colorbar_widgets=self.colorbar_widgets,
            colorbar_image=self.colorbar_image,
        )

        mask = draw_masks(
            roi_statistics=self._roi_statistics,
            frame_height=sd.frame_height,
            frame_width=sd.frame_width,
            color_arrays=self.color_arrays,
            roi_maps=self.roi_maps,
            roi_color_mode=self.roi_color_mode,
            background_view=self.background_view,
            selected_roi_indices=self.selected_roi_indices,
            roi_opacity=self._view_controls.opacity_slider.value(),
            classification_label_mode=self.classification_label_mode,
        )
        display_masks(overlay_item=self._overlay, mask=mask)

        # Initializes plot ranges.
        self._view_box.setXRange(0, sd.frame_width)
        self._view_box.setYRange(0, sd.frame_height)
        self._trace_box.getViewBox().setLimits(xMin=0, xMax=self._frame_count)
        self.frame_indices = np.arange(0, self._frame_count, dtype=np.int32)

        display_views(
            view=self._background,
            views=self.views,
            view_index=self.background_view,
        )
        plot_trace(
            trace_box=self._trace_box,
            cell_fluorescence=self._cell_fluorescence,
            neuropil_fluorescence=self._neuropil_fluorescence,
            subtracted_fluorescence=self._subtracted_fluorescence,
            spikes=self._spikes,
            frame_indices=self.frame_indices,
            selected_indices=self.selected_roi_indices,
            activity_mode=self.trace_mode,
        )

        # Sets aspect ratio on the image panel.
        self._view_box.setAspectLocked(lock=True, ratio=sd.aspect_ratio)

        self.session_loaded = True

        # Computes binned activity and triggers initial full redraw.
        self._on_activity_changed(TraceMode.DECONVOLVED)
        self.show()

    def _enable_controls(self) -> None:
        """Enables all view, color, and selection dropdowns after data loading."""
        if self.context_data is None:
            return
        sd = self.context_data.single_day

        # Enables view dropdown and sets initial selection.
        self._view_controls.view_combo.setEnabled(True)
        self._view_controls.view_combo.setCurrentIndex(0)

        # Disables corrected structural view item if not available.
        view_model = self._view_controls.view_combo.model()
        if isinstance(view_model, QStandardItemModel):
            structural_item = view_model.item(BackgroundView.CORRECTED_STRUCTURAL)
            if structural_item is not None:
                if sd.corrected_structural_mean_image.size == 0:
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
            ch2_item = color_model.item(ROIColorMode.COLOCALIZATION_PROBABILITY)
            if ch2_item is not None:
                ch2_item.setEnabled(self._two_channels)

        # Enables selection dropdown.
        self._selection_controls.selection_combo.setEnabled(True)
        self._selection_controls.selection_combo.setCurrentIndex(0)

    def _on_view_changed(self, index: int) -> None:
        """Handles background view dropdown changes.

        Args:
            index: The background view index selected.
        """
        self.background_view = BackgroundView(index)
        self.update_plot()

    def _on_color_changed(self, index: int) -> None:
        """Handles ROI color mode dropdown changes.

        Args:
            index: The color mode index selected.
        """
        self.roi_color_mode = ROIColorMode(index)
        if self.context_data is not None and self.color_arrays is not None and self.roi_maps is not None:
            colormap = self._color_controls.colormap_chooser.currentText()
            if colormap != self.roi_colormap:
                self.roi_colormap = colormap
                self.colorbar_image = update_colormap(
                    color_arrays=self.color_arrays,
                    roi_maps=self.roi_maps,
                    colormap=colormap,
                )
        self.update_plot()

    def _on_activity_changed(self, i: int) -> None:
        """Changes the activity mode used for multi-neuron display and correlation.

        Args:
            i: The activity mode index to switch to.
        """
        self.trace_mode = TraceMode(i)
        if self.session_loaded and self.context_data is not None:
            self.temporal_bin_size = max(1, int(self._color_controls.binning_edit.text()))
            nb = int(np.floor(float(self._frame_count) / float(self.temporal_bin_size)))
            if i == 0:
                f = self._cell_fluorescence
            elif i == 1:
                f = self._neuropil_fluorescence
            elif i == TraceMode.NEUROPIL_CORRECTED:
                f = self._subtracted_fluorescence
            else:
                f = self._spikes
            ncells = len(self._roi_statistics)
            bin_size = self.temporal_bin_size
            self.Fbin = f[:, : nb * bin_size].reshape((ncells, nb, bin_size)).mean(axis=2)
            self.Fbin -= self.Fbin.mean(axis=1)[:, np.newaxis]
            self.Fstd = (self.Fbin**2).mean(axis=1) ** 0.5
            self.frame_indices = np.arange(0, self._frame_count, dtype=np.int32)
            self.update_plot()

    def _on_channel_2_toggled(self, checked: bool) -> None:
        """Rebuilds the view stack when the channel 2 toggle changes.

        Args:
            checked: True when channel 2 is toggled on.
        """
        if self.context_data is None:
            return
        self._view_controls.channel_2_button.setStyleSheet(STYLE.button_pressed if checked else STYLE.button_unpressed)
        sd = self.context_data.single_day
        self.views = build_views(
            frame_height=sd.frame_height,
            frame_width=sd.frame_width,
            mean_image=sd.mean_image,
            enhanced_mean_image=sd.enhanced_mean_image,
            correlation_map=sd.correlation_map,
            maximum_projection=sd.maximum_projection,
            corrected_structural_mean_image=sd.corrected_structural_mean_image,
            channel_2=checked,
            channel_2_mean_image=sd.mean_image_channel_2,
            channel_2_enhanced_mean_image=sd.enhanced_mean_image_channel_2,
            channel_2_correlation_map=sd.correlation_map_channel_2,
            channel_2_maximum_projection=sd.maximum_projection_channel_2,
            valid_y_range=sd.valid_y_range,
            valid_x_range=sd.valid_x_range,
        )
        self.update_plot()

    def _on_classification_label_toggled(self, checked: bool) -> None:
        """Switches between probability gradient and binary cell/non-cell label views.

        Args:
            checked: True when the label view toggle is pressed.
        """
        self.classification_label_mode = checked
        self._color_controls.classification_label_button.setStyleSheet(
            STYLE.button_pressed if checked else STYLE.button_unpressed
        )
        self.update_plot()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        """Handles keyboard shortcuts.

        Notes:
            Overrides the Qt virtual method. The camelCase name is required to match the parent signature.
        """
        if event.key() == QtCore.Qt.Key.Key_Space:
            self._roi_visibility_checkbox.toggle()

    def _on_number_chosen(self) -> None:
        """Jumps to the ROI number entered in the ROI edit field."""
        if self.session_loaded and self.context_data is not None:
            self.selected_roi_index = int(self._roi_index_edit.text())
            roi_count = len(self._roi_statistics)
            if self.selected_roi_index >= roi_count:
                self.selected_roi_index = roi_count - 1
            self.selected_roi_indices = [self.selected_roi_index]
            self.update_plot()

    def _on_zoom_cell_toggled(self, state: int) -> None:
        """Toggles zoom-to-cell behavior based on checkbox state."""
        if not self.session_loaded:
            return
        self.auto_zoom_to_roi = QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked
        self.update_plot()

    def _on_view_selector_changed(self, combo_index: int) -> None:
        """Handles view selector dropdown changes by switching to the selected view.

        Maps the combo box index to a view index (accounting for the combined view offset) and
        reloads all GUI components for the new view.

        Args:
            combo_index: The index selected in the view selector combo box.
        """
        if combo_index < 0 or self.context_data is None:
            return

        # Maps combo index to view_index: combo 0 maps to view_index -1 (combined), combo 1+ maps to planes.
        view_index = combo_index - 1

        self.context_data.single_day.switch_view(view_index=view_index)
        self._reset_state()
        self._initialize_gui()

    def _on_trace_toggle(self, which: str) -> None:
        """Handles trace visibility checkbox toggles."""
        tc = self._trace_controls
        if which == "deconvolved":
            tc.deconvolved_visible = tc.deconvolved_checkbox.isChecked()
        elif which == "neuropil":
            tc.neuropil_visible = tc.neuropil_checkbox.isChecked()
        elif which == "traces":
            tc.traces_visible = tc.traces_checkbox.isChecked()
        self._refresh_traces()

    def _refresh_traces(self) -> None:
        """Refreshes the trace panel without redrawing image panels."""
        if self.context_data is None or self.color_arrays is None or self.frame_indices is None:
            return

        # In multi-day mode with "All Recordings" enabled and exactly one ROI selected, show stacked traces.
        if self._is_multi_day and self._all_recordings_visible and len(self.selected_roi_indices) == 1:
            self._refresh_all_recording_traces()
            return

        plot_trace(
            trace_box=self._trace_box,
            cell_fluorescence=self._cell_fluorescence,
            neuropil_fluorescence=self._neuropil_fluorescence,
            subtracted_fluorescence=self._subtracted_fluorescence,
            spikes=self._spikes,
            frame_indices=self.frame_indices,
            selected_indices=self.selected_roi_indices,
            activity_mode=self.trace_mode,
            roi_colors=self.color_arrays.colors[self.roi_color_mode],
            traces_visible=self._trace_controls.traces_visible,
            neuropil_visible=self._trace_controls.neuropil_visible,
            deconvolved_visible=self._trace_controls.deconvolved_visible,
            scale_factor=ROI_CONFIG.default_scale_factor,
            max_plotted=int(self._trace_controls.max_plotted_edit.text() or str(ROI_CONFIG.plotted_trace_count)),
        )

    def update_plot(self) -> None:
        """Redraws all plot panels including masks, traces, and colorbar."""
        if self.context_data is None or self.color_arrays is None or self.roi_maps is None:
            return
        if self.views is None or self.colorbar_widgets is None or self.colorbar_image is None:
            return
        if self.roi_color_mode == ROIColorMode.CORRELATIONS and self.Fbin is not None and self.Fstd is not None:
            update_correlation_masks(
                color_arrays=self.color_arrays,
                roi_maps=self.roi_maps,
                binned_fluorescence=self.Fbin,
                fluorescence_standard_deviation=self.Fstd,
                selected_indices=self.selected_roi_indices,
                colormap=self.roi_colormap,
            )
        render_colorbar(
            roi_color_mode=self.roi_color_mode,
            color_arrays=self.color_arrays,
            colorbar_widgets=self.colorbar_widgets,
            colorbar_image=self.colorbar_image,
        )
        self._ichosen_stats()
        display_views(
            view=self._background,
            views=self.views,
            view_index=self.background_view,
        )
        mask = draw_masks(
            roi_statistics=self._roi_statistics,
            frame_height=self.context_data.single_day.frame_height,
            frame_width=self.context_data.single_day.frame_width,
            color_arrays=self.color_arrays,
            roi_maps=self.roi_maps,
            roi_color_mode=self.roi_color_mode,
            background_view=self.background_view,
            selected_roi_indices=self.selected_roi_indices,
            roi_opacity=self._view_controls.opacity_slider.value(),
            classification_label_mode=self.classification_label_mode,
        )
        display_masks(overlay_item=self._overlay, mask=mask)
        self._refresh_traces()
        if self.auto_zoom_to_roi:
            self._zoom_to_cell()
        self._view_box.show()
        self._graphics_widget.show()
        self.show()

        # Updates status bar.
        roi_index = self.selected_roi_index
        cell_count = self._cell_count
        sd = self.context_data.single_day
        height = sd.frame_height
        width = sd.frame_width
        session_name = str(sd.output_path)
        self._status_bar.showMessage(
            f"Session: {session_name}  |  ROI: {roi_index}  |  Cells: {cell_count}  |  Size: {height} x {width}"
        )

    def _ichosen_stats(self) -> None:
        """Updates the ROI statistics labels for the currently selected cell."""
        if self.context_data is None:
            return
        n = self.selected_roi_index
        self._roi_index_edit.setText(str(n))
        roi = self._roi_statistics[n]
        for k in range(len(self._stats_to_show)):
            key = self._stats_to_show[k]
            ival = getattr(roi, key, None)
            if ival is None:
                continue
            if isinstance(ival, tuple):
                self._roi_stat_labels[k].setText(f"{key}: [{ival[0]:d}, {ival[1]:d}]")
            elif isinstance(ival, int):
                self._roi_stat_labels[k].setText(f"{key}: {ival:d}")
            else:
                self._roi_stat_labels[k].setText(f"{key}: {ival:2.2f}")

    def _toggle_rois(self, state: int) -> None:
        """Toggles ROI overlay visibility on the image panel."""
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.rois_visible = True
            self._view_box.addItem(self._overlay)
        else:
            self.rois_visible = False
            self._view_box.removeItem(self._overlay)
        self._graphics_widget.show()
        self.show()

    def _roi_text(self, state: int) -> None:
        """Toggles ROI number text labels on the image panel."""
        if self.roi_maps is None or self.context_data is None:
            return

        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            for label in self.roi_maps.text_labels:
                self._view_box.addItem(label)
            self.roi_labels_visible = True
        else:
            for label in self.roi_maps.text_labels:
                with suppress(Exception):
                    self._view_box.removeItem(label)
            self.roi_labels_visible = False

    def _zoom_to_cell(self) -> None:
        """Zooms the image panel to center on the currently selected cell."""
        if self.context_data is None:
            return
        sd = self.context_data.single_day
        irange = ROI_CONFIG.zoom_to_cell_fraction * np.array([sd.frame_height, sd.frame_width]).max()
        roi_statistics = self._roi_statistics
        if len(self.selected_roi_indices) > 1:
            apix = np.zeros((0, 2))
            for k in self.selected_roi_indices:
                apix = np.append(
                    apix,
                    np.concatenate(
                        (
                            roi_statistics[k].mask.y_pixels.flatten()[:, np.newaxis],
                            roi_statistics[k].mask.x_pixels.flatten()[:, np.newaxis],
                        ),
                        axis=1,
                    ),
                    axis=0,
                )
            imin = apix.min(axis=0)
            imax = apix.max(axis=0)
            icent = apix.mean(axis=0)
            imin[0] = min(icent[0] - irange, imin[0])
            imin[1] = min(icent[1] - irange, imin[1])
            imax[0] = max(icent[0] + irange, imax[0])
            imax[1] = max(icent[1] + irange, imax[1])
        else:
            icent = np.array(roi_statistics[self.selected_roi_index].mask.centroid)
            imin = icent - irange
            imax = icent + irange
        self._view_box.setYRange(imin[0], imax[0])
        self._view_box.setXRange(imin[1], imax[1])

    def _zoom_plot(self) -> None:
        """Resets the view range for the image panel."""
        self._view_box.autoRange()

    def _flip_plot(self) -> None:
        """Flips the selected ROIs between cell and non-cell classification.

        Classification writes go directly through the r+ memory-mapped file, so no explicit save
        is needed.
        """
        if self.context_data is None or self.color_arrays is None or self.roi_maps is None:
            return
        flip_rois(
            roi_statistics=self._roi_statistics,
            cell_classification=self._cell_classification,
            color_arrays=self.color_arrays,
            roi_maps=self.roi_maps,
            selected_roi_indices=self.selected_roi_indices,
        )
        self.last_reclassified_index = self.selected_roi_index
        self.update_plot()

    def _handle_click(self, click_x: int, click_y: int, is_right_button: bool, is_multi_select: bool) -> bool:
        """Handles mouse clicks on the image panel.

        Left-click chooses a cell. Shift/ctrl-click adds or removes from the merge selection.
        Right-click reclassifies the clicked ROI when the cell/non-cell color mode is active.

        Args:
            click_x: Column coordinate of the click.
            click_y: Row coordinate of the click.
            is_right_button: Determines whether the click was a right-click.
            is_multi_select: Determines whether shift or ctrl was held during the click.

        Returns:
            True if the click was consumed, False to allow the default context menu.
        """
        if not self.session_loaded or self.roi_maps is None or self.context_data is None:
            return False

        sd = self.context_data.single_day
        if click_y < 0 or click_y >= sd.frame_height or click_x < 0 or click_x >= sd.frame_width:
            return False

        ichosen = int(self.roi_maps.roi_indices[0, click_y, click_x])
        if ichosen < 0:
            return False

        if is_right_button:
            # Reclassification is disabled in multi-day mode.
            if self._is_multi_day:
                return False
            # Reclassification is only available in cell classification label mode.
            if self.roi_color_mode != ROIColorMode.CELL_CLASSIFICATION or not self.classification_label_mode:
                return False
            if ichosen not in self.selected_roi_indices:
                self.selected_roi_indices = [ichosen]
                self.selected_roi_index = ichosen
            self._flip_plot()
            return True

        # Multi-day mode restricts selection to a single ROI.
        if self._is_multi_day:
            self.selected_roi_indices = [ichosen]
            self.selected_roi_index = ichosen
        else:
            merged = False
            if is_multi_select:
                if ichosen not in self.selected_roi_indices:
                    self.selected_roi_indices.append(ichosen)
                    self.selected_roi_index = ichosen
                    merged = True
                elif len(self.selected_roi_indices) > 1:
                    self.selected_roi_indices.remove(ichosen)
                    self.selected_roi_index = self.selected_roi_indices[0]
                    merged = True
            if not merged:
                self.selected_roi_indices = [ichosen]
                self.selected_roi_index = ichosen

        self.update_plot()
        return True

    def _on_top_bottom_selection(self) -> None:
        """Selects the top-n or bottom-n ROIs ranked by the active color statistic."""
        if self.color_arrays is None:
            return
        count = int(self._selection_controls.top_count_edit.text() or str(ROI_CONFIG.top_selection_count))
        count = min(count, ROI_CONFIG.top_selection_count)
        values = self.color_arrays.normalized_statistics[self.roi_color_mode]
        ranked = np.argsort(values)
        # Index 0 = "select top n", index 1 = "select bottom n".
        if self._selection_controls.selection_combo.currentIndex() == 0:
            selected = ranked[-count:][::-1]
        else:
            selected = ranked[:count]
        self.selected_roi_indices = selected.tolist()
        if self.selected_roi_indices:
            self.selected_roi_index = self.selected_roi_indices[0]
            self.update_plot()

    def _on_dataset_source_changed(self, index: int) -> None:
        """Handles ROI Source dropdown changes to switch between single-day and multi-day data.

        Args:
            index: The selected combo box index. 0 = Original (single-day), 1+ = multi-day datasets.
        """
        if self.context_data is None:
            return

        if index == 0:
            self.context_data.unload_dataset()
        elif index > 0:
            available = self.context_data.available_datasets
            dataset_index = index - 1
            if dataset_index < len(available):
                self.context_data.load_dataset(dataset_name=available[dataset_index])
            else:
                return
        else:
            return

        self._reset_state()
        self._initialize_gui()

    def _on_all_recordings_toggled(self, checked: bool) -> None:
        """Handles the 'All Recordings' toggle button for multi-day stacked trace display.

        Args:
            checked: True when stacked all-recordings view is enabled.
        """
        self._all_recordings_visible = checked
        self._all_recordings_button.setStyleSheet(STYLE.button_pressed if checked else STYLE.button_unpressed)
        self._refresh_traces()

    def _refresh_all_recording_traces(self) -> None:
        """Plots traces from all recordings stacked vertically for the selected ROI.

        Iterates over every recording in the multi-day dataset, extracts the selected ROI's trace from each, and plots
        them stacked with recording-index labels on the y-axis.
        """
        if self.context_data is None or not self.context_data.is_multi_day:
            return

        self._trace_box.clear()
        if not self.selected_roi_indices:
            return

        roi_index = self.selected_roi_indices[0]
        axis = self._trace_box.getAxis("left")
        tick_labels: list[tuple[float, str]] = []
        trace_spacing = 1.0 / ROI_CONFIG.default_scale_factor
        max_frames = 0
        y_maximum = 0.0
        stack_position = self.context_data.recording_count - 1

        for recording_index in range(self.context_data.recording_count):
            rec = self.context_data.recording(recording_index)
            cell_fluorescence = rec.cell_fluorescence
            neuropil_fluorescence = rec.neuropil_fluorescence
            subtracted_fluorescence = rec.subtracted_fluorescence
            spikes = rec.spikes

            if cell_fluorescence.size == 0:
                stack_position -= 1
                continue
            if roi_index >= cell_fluorescence.shape[0]:
                stack_position -= 1
                continue

            # Selects trace based on activity mode.
            if self.trace_mode == 0:
                trace = cell_fluorescence[roi_index, :]
            elif self.trace_mode == 1:
                trace = neuropil_fluorescence[roi_index, :]
            elif self.trace_mode == TraceMode.NEUROPIL_CORRECTED:
                trace = subtracted_fluorescence[roi_index, :]
            else:
                trace = spikes[roi_index, :]

            frame_indices = np.arange(len(trace), dtype=np.int32)
            max_frames = max(max_frames, len(trace))

            # Normalizes trace to [0, 1].
            trace_max = float(trace.max())
            trace_min = float(trace.min())
            if trace_max > trace_min:
                normalized = (trace - trace_min) / (trace_max - trace_min)
            else:
                normalized = np.zeros_like(trace)

            # Generates a color for this recording using HSV hues.
            hue = recording_index / max(self.context_data.recording_count, 1)
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
        if not self.session_loaded or not self._roi_statistics:
            return None

        training_labels = (self._cell_classification[:, 1] > ROI_CONFIG.default_classifier_threshold).astype(np.bool_)
        n = len(self._roi_statistics)
        normalized_pixel_count = np.empty(n, dtype=np.float32)
        compactness = np.empty(n, dtype=np.float32)
        skewness = np.empty(n, dtype=np.float32)

        for i, roi in enumerate(self._roi_statistics):
            normalized_pixel_count[i] = roi.normalized_pixel_count
            compactness[i] = roi.compactness
            skewness[i] = np.nan if roi.skewness is None else roi.skewness

        return training_labels, normalized_pixel_count, compactness, skewness

    def _on_classifier_new(self) -> None:
        """Handles the 'New' classifier button: saves current labels and features as a new training dataset."""
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

        n = len(training_labels)
        filename = Path(file_path).name
        self._classifier_controls.status_label.setText(f"Saved {n} samples to {filename}")

    def _on_classifier_add_to_existing(self) -> None:
        """Handles the 'Add to Existing' button: merges current session data into an existing training dataset."""
        if self._is_multi_day:
            return

        result = self._extract_classifier_data()
        if result is None:
            return
        new_labels, new_npc, new_comp, new_skew = result

        source_path, _ = QFileDialog.getOpenFileName(
            self, "Open Existing Classifier Dataset", "", "NumPy files (*.npz)"
        )
        if not source_path:
            return

        try:
            data = np.load(source_path, allow_pickle=False)
            existing_labels = data["training_labels"].astype(np.bool_)
            existing_npc = data["normalized_pixel_count"].astype(np.float32)
            existing_comp = data["compactness"].astype(np.float32)
            existing_skew = data["skewness"].astype(np.float32)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            self._classifier_controls.status_label.setText(f"Error loading file: {exc}")
            return

        merged_labels = np.concatenate([existing_labels, new_labels])
        merged_npc = np.concatenate([existing_npc, new_npc])
        merged_comp = np.concatenate([existing_comp, new_comp])
        merged_skew = np.concatenate([existing_skew, new_skew])

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Merged Dataset", source_path, "NumPy files (*.npz)")
        if not save_path:
            return

        Classifier.create_training_dataset(
            file_path=Path(save_path),
            training_labels=merged_labels,
            normalized_pixel_count=merged_npc,
            compactness=merged_comp,
            skewness=merged_skew,
        )

        n_new = len(new_labels)
        n_total = len(merged_labels)
        filename = Path(save_path).name
        self._classifier_controls.status_label.setText(f"Added {n_new} samples ({n_total} total) to {filename}")
