"""Provides the ROI viewer window for inspecting and reclassifying single-day pipeline results."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path
from contextlib import suppress

import numpy as np
from PySide6 import QtGui, QtCore
import pyqtgraph as pg  # type: ignore[import-untyped]
from PySide6.QtWidgets import (
    QLabel,
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
    QButtonGroup,
    QSlider,
)
from ataraxis_base_utilities import LogLevel, console

from .constants import (
    STYLE,
    CONFIG,
    BackgroundView,
    ROIColorMode,
    ROIToolPanel,
    TraceMode,
)
from .single_day_context import ROIViewerData
from .data_models import (
    CellToggleControls,
    ColorArrays,
    ColorbarWidgets,
    ColorControls,
    QuadrantControls,
    ROIIndexMaps,
    SelectionControls,
    TraceControls,
    ViewControls,
)
from .overlays import (
    build_views,
    compute_colors,
    display_masks,
    display_views,
    draw_colorbar,
    draw_masks,
    flip_rois,
    init_roi_maps,
    render_colorbar,
    update_colormap,
    update_correlation_masks,
)
from .widgets import RangeSlider, TraceBox, ViewBox, plot_trace

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ROIViewer(QMainWindow):
    """Two-panel ROI viewer for single-day pipeline outputs with right-click reclassification.

    Displays ROI overlays, background images, and fluorescence traces. Supports left-click ROI
    selection, shift/ctrl multi-select, right-click cell/non-cell reclassification with auto-save,
    keyboard shortcuts for view/color switching, and quadrant zoom navigation.

    Args:
        data: Pre-loaded viewer data. If None the viewer starts empty and the user can load
            a session via the File menu or drag-and-drop.
    """

    def __init__(self, data: ROIViewerData | None = None) -> None:
        super().__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")

        # Display state (replaces ViewState — lives directly on the window).
        self.rois_visible: bool = True
        self.roi_color_mode: int = ROIColorMode.RANDOM
        self.background_view: int = BackgroundView.ROIS_ONLY
        self.roi_opacity: list[int] = [127, 255]
        self.background_saturation: list[int] = [0, 255]
        self.roi_colormap: str = "hsv"
        self.selected_roi_index: int = 0
        self.merge_roi_indices: list[int] = [0]
        self.roi_tool_active: bool = False
        self.roi_tool_panel: int = ROIToolPanel.CELLS
        self.trace_mode: int = TraceMode.NEUROPIL_CORRECTED
        self.temporal_bin_size: int = 1
        self.auto_zoom_to_roi: bool = False
        self.roi_labels_visible: bool = False
        self.session_loaded: bool = False
        self.colocalization_threshold: float = 0.6
        self.last_reclassified_index: int = -1

        # Core data objects.
        self.context_data: ROIViewerData | None = None
        self.color_arrays: ColorArrays | None = None
        self.roi_maps: ROIIndexMaps | None = None
        self.colorbar_widgets: ColorbarWidgets | None = None
        self.colorbar_image: NDArray[np.uint8] | None = None
        self.views: NDArray[np.uint8] | None = None

        # Binned activity state (used by correlation coloring).
        self.Fbin: NDArray[np.float32] | None = None
        self.Fstd: NDArray[np.float32] | None = None
        self.frame_indices: NDArray[np.intp] | None = None

        # Window geometry and title.
        self.setGeometry(50, 50, 1500, 800)
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

    def _build_menus(self) -> None:
        """Builds the File-only menu bar for the read-only viewer."""
        file_menu = self.menuBar().addMenu("&File")

        load_action = file_menu.addAction("&Load processed data")
        load_action.setShortcut("Ctrl+L")
        load_action.triggered.connect(self._on_load_dialog)

        load_folder_action = file_menu.addAction("Load &folder with planeX folders")
        load_folder_action.setShortcut("Ctrl+F")
        load_folder_action.triggered.connect(self._on_load_dialog)

        export_action = file_menu.addAction("&Export as image (svg)")
        export_action.triggered.connect(self._on_export_fig)

    def _on_load_dialog(self) -> None:
        """Opens a directory dialog and loads the selected session."""
        self._load_session()

    def _on_export_fig(self) -> None:
        """Opens the pyqtgraph export dialog for the current plot."""
        self._graphics_widget.scene().contextMenuItem = self._cells_view_box
        self._graphics_widget.scene().showExportDialog()

    def _build_control_panel(self) -> QWidget:
        """Builds the right-side control panel with QGroupBox sections.

        Returns:
            The control panel widget containing all grouped controls.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 1. ROI Visibility.
        visibility_box = QGroupBox("ROI Visibility")
        visibility_box.setStyleSheet("QGroupBox { color: white; }")
        visibility_layout = QVBoxLayout(visibility_box)
        self._roi_visibility_checkbox = QCheckBox("ROIs On [space bar]")
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

        # 3. View Toggle.
        toggle_box, self._cell_toggle_controls = self._create_cell_toggle_buttons()
        layout.addWidget(toggle_box)

        # 4. Background Views.
        background_box, self._view_controls = self._create_view_controls()
        layout.addWidget(background_box)

        # 5. ROI Colors + colorbar.
        colors_box, self._color_controls = self._create_color_controls()
        self.colorbar_widgets = self._create_colorbar()
        colors_layout = colors_box.layout()
        assert colors_layout is not None
        colors_layout.addWidget(self.colorbar_widgets.widget)
        layout.addWidget(colors_box)

        # 6. Selected ROI — ROI index edit + stat labels.
        roi_box = QGroupBox("Selected ROI")
        roi_box.setStyleSheet("QGroupBox { color: white; }")
        roi_layout = QVBoxLayout(roi_box)
        self._stats_to_show = [
            "centroid",
            "pixel_count",
            "skewness",
            "compactness",
            "footprint",
            "aspect_ratio",
        ]
        lilfont = STYLE.label_font()
        self._roi_index_edit = QLineEdit(self)
        self._roi_index_edit.setValidator(QtGui.QIntValidator(0, 10000))
        self._roi_index_edit.setText("0")
        self._roi_index_edit.setFixedWidth(STYLE.roi_edit_width)
        self._roi_index_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._roi_index_edit.returnPressed.connect(self._on_number_chosen)
        roi_layout.addWidget(self._roi_index_edit)
        self._roi_stat_labels: list[QLabel] = []
        for k in range(len(self._stats_to_show)):
            stat_label = QLabel(self._stats_to_show[k])
            stat_label.setFont(lilfont)
            stat_label.setStyleSheet(STYLE.white_label)
            stat_label.resize(stat_label.minimumSizeHint())
            roi_layout.addWidget(stat_label)
            self._roi_stat_labels.append(stat_label)
        layout.addWidget(roi_box)

        # 7. Trace Display.
        trace_box, self._trace_controls = self._create_trace_controls()
        self._zoom_to_cell_checkbox = QCheckBox("zoom to cell")
        self._zoom_to_cell_checkbox.setStyleSheet(STYLE.white_label)
        self._zoom_to_cell_checkbox.stateChanged.connect(self._on_zoom_cell_toggled)
        trace_layout = trace_box.layout()
        assert trace_layout is not None
        trace_layout.addWidget(self._zoom_to_cell_checkbox)
        layout.addWidget(trace_box)

        # 8. Navigation.
        nav_box, self._quadrant_controls = self._create_quadrant_buttons()
        layout.addWidget(nav_box)

        layout.addStretch()
        return panel

    def _create_selection_buttons(self) -> tuple[QGroupBox, SelectionControls]:
        """Creates the cell selection buttons and top-n input field."""
        group_box = QGroupBox("Cell Selection")
        group_box.setStyleSheet("QGroupBox { color: white; }")
        layout = QGridLayout(group_box)

        selection_buttons = QButtonGroup()
        labels = [" draw selection", " select top n", " select bottom n"]
        for button_index in range(3):
            button = QPushButton(labels[button_index], self)
            button.setCheckable(True)
            button.setStyleSheet(STYLE.button_inactive)
            button.setFont(STYLE.label_font_bold())
            button.resize(button.minimumSizeHint())
            selection_buttons.addButton(button, button_index)
            layout.addWidget(button, button_index, 0, 1, 1)
            button.setEnabled(False)
            button.clicked.connect(lambda _checked, idx=button_index: self._on_selection_button(idx))
        selection_buttons.setExclusive(True)

        count_label = QLabel("n=")
        count_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        count_label.setStyleSheet(STYLE.white_label)
        count_label.setFont(STYLE.label_font_bold())
        layout.addWidget(count_label, 1, 1, 1, 1)

        top_count_edit = QLineEdit(self)
        top_count_edit.setValidator(QtGui.QIntValidator(0, CONFIG.max_top_n))
        top_count_edit.setText(str(CONFIG.default_top_n))
        top_count_edit.setFixedWidth(STYLE.small_edit_width)
        top_count_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        top_count_edit.returnPressed.connect(self._on_roi_selection)
        layout.addWidget(top_count_edit, 2, 1, 1, 1)

        controls = SelectionControls(selection_buttons=selection_buttons, top_count_edit=top_count_edit)
        return group_box, controls

    def _create_cell_toggle_buttons(self) -> tuple[QGroupBox, CellToggleControls]:
        """Creates the cell / not-cell / both size-toggle buttons and ROI count labels."""
        group_box = QGroupBox("View Toggle")
        group_box.setStyleSheet("QGroupBox { color: white; }")
        layout = QGridLayout(group_box)

        cell_count_label = QLabel("")
        cell_count_label.setStyleSheet(STYLE.white_label)
        cell_count_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(cell_count_label, 0, 0, 1, 1)

        noncell_count_label = QLabel("")
        noncell_count_label.setStyleSheet(STYLE.white_label)
        layout.addWidget(noncell_count_label, 0, 2, 1, 1)

        size_buttons = QButtonGroup(self)
        labels = [" cells", " both", " not cells"]
        for button_index, label_text in enumerate(labels):
            button = QPushButton(label_text, self)
            button.setCheckable(True)
            button.setStyleSheet(STYLE.button_inactive)
            button.setFont(STYLE.label_font_bold())
            button.resize(button.minimumSizeHint())
            size_buttons.addButton(button, button_index)
            layout.addWidget(button, 1, button_index, 1, 1)
            button.setEnabled(button_index == CONFIG.view_both)
            button.clicked.connect(lambda _checked: self.update_plot())
        size_buttons.setExclusive(True)

        return group_box, CellToggleControls(
            size_buttons=size_buttons,
            cell_count_label=cell_count_label,
            noncell_count_label=noncell_count_label,
        )

    def _create_view_controls(self) -> tuple[QGroupBox, ViewControls]:
        """Creates background view selection controls inside a group box."""
        group_box = QGroupBox("Background")
        group_box.setStyleSheet("QGroupBox { color: white; }")
        layout = QGridLayout(group_box)

        view_buttons = QButtonGroup(self)
        for button_index, name in enumerate(CONFIG.view_names):
            button = QPushButton("&" + name, self)
            button.setCheckable(True)
            button.setStyleSheet(STYLE.button_inactive)
            button.setFont(STYLE.label_font_bold())
            button.resize(button.minimumSizeHint())
            view_buttons.addButton(button, button_index)
            layout.addWidget(button, button_index, 0, 1, 1)
            if button_index == 0:
                saturation_label = QLabel("sat: ")
                saturation_label.setStyleSheet(STYLE.white_label)
                layout.addWidget(saturation_label, button_index, 1, 1, 1)
            button.setEnabled(False)
            button.clicked.connect(lambda _checked, idx=button_index: self._on_view_changed(idx))
        view_buttons.setExclusive(True)

        range_slider = RangeSlider(owner=self, on_release=self._on_saturation_changed)
        range_slider.setMinimum(0)
        range_slider.setMaximum(255)
        range_slider.setLow(0)
        range_slider.setHigh(255)
        range_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        layout.addWidget(range_slider, 1, 1, len(CONFIG.view_names) - 2, 1)

        controls = ViewControls(view_buttons=view_buttons, range_slider=range_slider)
        return group_box, controls

    def _create_color_controls(self) -> tuple[QGroupBox, ColorControls]:
        """Creates color statistic selection buttons and their associated controls."""
        group_box = QGroupBox("ROI Colors")
        group_box.setStyleSheet("QGroupBox { color: white; }")
        layout = QGridLayout(group_box)

        color_buttons = QButtonGroup(self)

        colormap_chooser = QComboBox()
        colormap_chooser.addItems(CONFIG.colormaps)
        colormap_chooser.setCurrentIndex(0)
        colormap_chooser.setFont(STYLE.label_font())
        colormap_chooser.setFixedWidth(STYLE.color_edit_width)
        layout.addWidget(colormap_chooser, 0, 1, 1, 1)

        for button_index, name in enumerate(CONFIG.color_names):
            button = QPushButton("&" + name, self)
            button.setCheckable(True)
            button.setStyleSheet(STYLE.button_inactive)
            button.setFont(STYLE.label_font_bold())
            button.resize(button.minimumSizeHint())
            color_buttons.addButton(button, button_index)
            if CONFIG.color_narrow_range_start <= button_index < CONFIG.color_narrow_range_end:
                layout.addWidget(button, button_index, 0, 1, 1)
            else:
                layout.addWidget(button, button_index, 0, 1, 2)
            button.setEnabled(False)
            button.clicked.connect(lambda _checked, idx=button_index: self._on_color_changed(idx))

        classifier_edit = QLineEdit(self)
        classifier_edit.setText("0.5")
        classifier_edit.setFixedWidth(STYLE.color_edit_width)
        classifier_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(classifier_edit, len(CONFIG.color_names) - 3, 1, 1, 1)
        classifier_edit.returnPressed.connect(self.update_plot)

        bin_edit = QLineEdit(self)
        bin_edit.setValidator(QtGui.QIntValidator(0, 500))
        bin_edit.setText("1")
        bin_edit.setFixedWidth(STYLE.color_edit_width)
        bin_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(bin_edit, len(CONFIG.color_names) - 2, 1, 1, 1)
        bin_edit.returnPressed.connect(
            lambda: self._on_activity_changed(self._trace_controls.activity_combo.currentIndex())
        )

        colormap_chooser.activated.connect(lambda: self._on_color_changed(self.roi_color_mode))

        controls = ColorControls(
            color_buttons=color_buttons,
            colormap_chooser=colormap_chooser,
            classifier_edit=classifier_edit,
            bin_edit=bin_edit,
        )
        return group_box, controls

    def _create_colorbar(self) -> ColorbarWidgets:
        """Creates the colorbar widget displaying the current color mapping."""
        colorbar_widget = pg.GraphicsLayoutWidget(self)
        colorbar_widget.setMaximumHeight(STYLE.colorbar_max_height)
        colorbar_widget.setMaximumWidth(STYLE.colorbar_max_width)
        colorbar_widget.ci.layout.setRowStretchFactor(0, 2)
        colorbar_widget.ci.layout.setContentsMargins(0, 0, 0, 0)

        image = pg.ImageItem()
        colorbar_view = colorbar_widget.addViewBox(row=0, col=0, colspan=3)
        colorbar_view.setMenuEnabled(False)
        colorbar_view.addItem(image)

        labels = [
            colorbar_widget.addLabel("0.0", color=[255, 255, 255], row=1, col=0),
            colorbar_widget.addLabel("0.5", color=[255, 255, 255], row=1, col=1),
            colorbar_widget.addLabel("1.0", color=[255, 255, 255], row=1, col=2),
        ]
        return ColorbarWidgets(image=image, labels=labels, widget=colorbar_widget)

    def _create_trace_controls(self) -> tuple[QGroupBox, TraceControls]:
        """Creates trace panel controls inside a group box."""
        group_box = QGroupBox("Trace Display")
        group_box.setStyleSheet("QGroupBox { color: white; }")
        layout = QGridLayout(group_box)

        activity_label = QLabel("Activity mode:")
        activity_label.setStyleSheet(STYLE.white_label)
        layout.addWidget(activity_label, 0, 0, 1, 1)

        activity_combo = QComboBox(self)
        activity_combo.setFixedWidth(STYLE.combo_box_width)
        layout.addWidget(activity_combo, 1, 0, 1, 1)
        activity_combo.addItem("F")
        activity_combo.addItem("Fneu")
        activity_combo.addItem("F - 0.7*Fneu")
        activity_combo.addItem("deconvolved")
        activity_combo.setCurrentIndex(CONFIG.default_activity_mode)
        activity_combo.currentIndexChanged.connect(self._on_activity_changed)

        arrow_up = QPushButton(" \u25b2")
        arrow_down = QPushButton(" \u25bc")
        arrow_buttons = [arrow_up, arrow_down]
        for button_index, button in enumerate(arrow_buttons):
            button.setMaximumWidth(STYLE.square_button_max_width)
            button.setFont(STYLE.arrow_button_font())
            button.setStyleSheet(STYLE.button_unpressed)
            layout.addWidget(button, button_index, 1, 1, 1, QtCore.Qt.AlignmentFlag.AlignRight)

        scale_up = QPushButton(" +")
        scale_down = QPushButton(" -")
        scale_buttons = [scale_up, scale_down]
        for button_index, button in enumerate(scale_buttons):
            button.setMaximumWidth(STYLE.square_button_max_width)
            button.setFont(STYLE.arrow_button_font())
            button.setStyleSheet(STYLE.button_unpressed)
            layout.addWidget(button, button_index, 2, 1, 1)

        max_plotted_label = QLabel("max # plotted:")
        max_plotted_label.setStyleSheet(STYLE.white_label)
        layout.addWidget(max_plotted_label, 2, 0, 1, 1)

        max_plotted_edit = QLineEdit(self)
        max_plotted_edit.setValidator(QtGui.QIntValidator(0, CONFIG.max_plotted_count))
        max_plotted_edit.setText(str(CONFIG.default_plotted_count))
        max_plotted_edit.setFixedWidth(STYLE.small_edit_width)
        max_plotted_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(max_plotted_edit, 3, 0, 1, 1)

        deconvolved_checkbox = QCheckBox("deconv [N]")
        deconvolved_checkbox.setStyleSheet(STYLE.white_label)
        deconvolved_checkbox.toggle()
        layout.addWidget(deconvolved_checkbox, 3, 1, 1, 1)

        neuropil_checkbox = QCheckBox("neuropil [B]")
        neuropil_checkbox.setStyleSheet(STYLE.red_label)
        neuropil_checkbox.toggle()
        layout.addWidget(neuropil_checkbox, 3, 2, 1, 1)

        traces_checkbox = QCheckBox("raw fluor [V]")
        traces_checkbox.setStyleSheet(STYLE.cyan_label)
        traces_checkbox.toggle()
        layout.addWidget(traces_checkbox, 3, 3, 1, 1)

        controls = TraceControls(
            activity_combo=activity_combo,
            deconvolved_checkbox=deconvolved_checkbox,
            neuropil_checkbox=neuropil_checkbox,
            traces_checkbox=traces_checkbox,
            max_plotted_edit=max_plotted_edit,
            arrow_buttons=arrow_buttons,
            scale_buttons=scale_buttons,
        )

        arrow_up.clicked.connect(lambda: self._adjust_trace_level(1))
        arrow_down.clicked.connect(lambda: self._adjust_trace_level(-1))
        scale_up.clicked.connect(lambda: self._adjust_scale(CONFIG.scale_step))
        scale_down.clicked.connect(lambda: self._adjust_scale(-CONFIG.scale_step))
        max_plotted_edit.returnPressed.connect(self._refresh_traces)
        deconvolved_checkbox.toggled.connect(lambda: self._on_trace_toggle("deconvolved"))
        neuropil_checkbox.toggled.connect(lambda: self._on_trace_toggle("neuropil"))
        traces_checkbox.toggled.connect(lambda: self._on_trace_toggle("traces"))

        return group_box, controls

    def _create_quadrant_buttons(self) -> tuple[QGroupBox, QuadrantControls]:
        """Creates the 3x3 quadrant zoom navigation buttons."""
        group_box = QGroupBox("Navigation")
        group_box.setStyleSheet("QGroupBox { color: white; }")
        layout = QGridLayout(group_box)

        quadrant_buttons = QButtonGroup(self)
        for button_index in range(9):
            button = QPushButton(" " + str(button_index + 1), self)
            button.setCheckable(True)
            button.setStyleSheet(STYLE.button_inactive)
            button.setFont(STYLE.label_font_bold())
            button.resize(button.minimumSizeHint())
            button.setMaximumWidth(STYLE.small_edit_width)
            quadrant_buttons.addButton(button, button_index)
            row = button_index // STYLE.quadrant_columns
            col = button_index % STYLE.quadrant_columns
            layout.addWidget(button, row, col, 1, 1)
            button.setEnabled(False)
            button.clicked.connect(lambda _checked, idx=button_index: self._on_quadrant(idx))
        quadrant_buttons.setExclusive(True)

        return group_box, QuadrantControls(quadrant_buttons=quadrant_buttons)

    def _build_graphics(self) -> None:
        """Creates the main plotting area with cells, non-cells, and trace panels."""
        self._cells_view_box = ViewBox(panel=ROIToolPanel.CELLS, name="plot1", border=[100, 100, 100], invert_y=True)
        self._graphics_widget.addItem(self._cells_view_box, 0, 0)
        self._cells_view_box.setMenuEnabled(False)
        self._cells_view_box.scene().contextMenuItem = self._cells_view_box
        self._cells_background = pg.ImageItem(viewbox=self._cells_view_box, parent=self)
        self._cells_background.autoDownsample = False
        self._cells_overlay = pg.ImageItem(viewbox=self._cells_view_box, parent=self)
        self._cells_overlay.autoDownsample = False
        self._cells_view_box.addItem(self._cells_background)
        self._cells_view_box.addItem(self._cells_overlay)
        self._cells_background.setLevels([0, 255])
        self._cells_overlay.setLevels([0, 255])

        self._noncells_view_box = ViewBox(
            panel=ROIToolPanel.NON_CELLS, name="plot2", border=[100, 100, 100], invert_y=True
        )
        self._graphics_widget.addItem(self._noncells_view_box, 0, 1)
        self._noncells_view_box.setMenuEnabled(False)
        self._noncells_view_box.scene().contextMenuItem = self._noncells_view_box
        self._noncells_background = pg.ImageItem(viewbox=self._cells_view_box, parent=self)
        self._noncells_background.autoDownsample = False
        self._noncells_overlay = pg.ImageItem(viewbox=self._cells_view_box, parent=self)
        self._noncells_overlay.autoDownsample = False
        self._noncells_view_box.addItem(self._noncells_background)
        self._noncells_view_box.addItem(self._noncells_overlay)
        self._noncells_background.setLevels([0, 255])
        self._noncells_overlay.setLevels([0, 255])

        self._noncells_view_box.setXLink("plot1")
        self._noncells_view_box.setYLink("plot1")

        self._cells_view_box.set_click_handler(self._handle_click)
        self._noncells_view_box.set_click_handler(self._handle_click)
        self._cells_view_box.set_zoom_handler(lambda: self._zoom_plot(CONFIG.cells_plot))
        self._noncells_view_box.set_zoom_handler(lambda: self._zoom_plot(CONFIG.noncells_plot))

        self._trace_box = TraceBox()
        self._trace_box.setMouseEnabled(x=True, y=False)
        self._trace_box.enableAutoRange(x=True, y=True)
        self._graphics_widget.addItem(self._trace_box, row=1, col=0, colspan=2)
        self._graphics_widget.ci.layout.setRowStretchFactor(0, 2)
        gl = self._graphics_widget.ci.layout
        gl.setColumnMinimumWidth(0, 1)
        gl.setColumnMinimumWidth(1, 1)
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
            context_data = ROIViewerData.from_single_day(root_path=session_path, mutable=True)
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
        self._reset_state()
        self._initialize_gui()

    def _reset_state(self) -> None:
        """Resets all display state to defaults before loading new data."""
        self.rois_visible = True
        self.roi_color_mode = ROIColorMode.RANDOM
        self.background_view = BackgroundView.ROIS_ONLY
        self.roi_opacity = [127, 255]
        self.background_saturation = [0, 255]
        self.roi_colormap = "hsv"
        self.selected_roi_index = 0
        self.merge_roi_indices = [0]
        self.roi_tool_active = False
        self.roi_tool_panel = ROIToolPanel.CELLS
        self.trace_mode = TraceMode.NEUROPIL_CORRECTED
        self.temporal_bin_size = 1
        self.auto_zoom_to_roi = False
        self.roi_labels_visible = False
        self.session_loaded = False
        self.colocalization_threshold = 0.6
        self.last_reclassified_index = -1

    def _initialize_gui(self) -> None:
        """Initializes all GUI components after loading context data."""
        context = self.context_data
        if context is None:
            return

        # Resets display controls.
        self._roi_visibility_checkbox.setChecked(True)
        if self._roi_labels_checkbox.isChecked():
            self._roi_text(False)
        self._roi_labels_checkbox.setChecked(False)
        self._roi_labels_checkbox.setEnabled(True)
        self._roi_remove()

        session_title = str(context.output_path) if context.output_path is not None else "unknown session"
        self.setWindowTitle(f"cindra ROI Viewer — {session_title}")

        # Computes default bin size from tau and sampling rate.
        self.temporal_bin_size = max(1, int(context.tau * context.sampling_rate / CONFIG.bin_size_divisor))
        self._color_controls.bin_edit.setText(str(self.temporal_bin_size))
        self.colocalization_threshold = CONFIG.default_channel_2_threshold

        # Enables buttons.
        self._enable_controls()

        # Builds background views from detection images.
        self.views = build_views(
            frame_height=context.frame_height,
            frame_width=context.frame_width,
            mean_image=context.mean_image,
            enhanced_mean_image=context.enhanced_mean_image,
            correlation_map=context.correlation_map,
            maximum_projection=context.maximum_projection,
            corrected_channel_2_image=context.corrected_structural_mean_image,
            channel_2_mean_image=context.mean_image_channel_2,
            valid_y_range=context.valid_y_range,
            valid_x_range=context.valid_x_range,
        )

        # Computes color statistics and builds ROI index maps.
        self.color_arrays = compute_colors(
            context=context,
            roi_colormap=self.roi_colormap,
            colocalization_threshold=self.colocalization_threshold,
        )
        self.roi_maps = init_roi_maps(context=context, color_arrays=self.color_arrays)

        # Selects the first classified cell as the initial selection.
        first_cell = int(np.nonzero(context.cell_classification_labels)[0][0]) if context.cell_count > 0 else 0
        self.selected_roi_index = first_cell
        self.merge_roi_indices = [first_cell]
        self._ichosen_stats()
        self._trace_controls.activity_combo.setCurrentIndex(CONFIG.default_activity_mode)

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

        masks = draw_masks(
            context=context,
            color_arrays=self.color_arrays,
            roi_maps=self.roi_maps,
            roi_color_mode=self.roi_color_mode,
            background_view=self.background_view,
            selected_roi_index=self.selected_roi_index,
            merge_roi_indices=self.merge_roi_indices,
            roi_opacity=self.roi_opacity,
        )
        display_masks(color1=self._cells_overlay, color2=self._noncells_overlay, masks=masks)

        # Updates cell count labels.
        self._cell_toggle_controls.cell_count_label.setText(f"{int(context.cell_count)}")
        self._cell_toggle_controls.noncell_count_label.setText(f"{int(context.roi_count - context.cell_count)}")

        # Initializes plot ranges.
        self._cells_view_box.setXRange(0, context.frame_width)
        self._cells_view_box.setYRange(0, context.frame_height)
        self._noncells_view_box.setXRange(0, context.frame_width)
        self._noncells_view_box.setYRange(0, context.frame_height)
        self._trace_box.getViewBox().setLimits(xMin=0, xMax=context.frame_count)
        self.frame_indices = np.arange(0, context.frame_count, dtype=np.int32)

        display_views(
            view1=self._cells_background,
            view2=self._noncells_background,
            views=self.views,
            view_index=self.background_view,
            saturation=self.background_saturation,
        )
        plot_trace(
            trace_box=self._trace_box,
            cell_fluorescence=context.cell_fluorescence,
            neuropil_fluorescence=context.neuropil_fluorescence,
            spikes=context.spikes,
            frame_indices=self.frame_indices,
            merge_indices=self.merge_roi_indices,
            activity_mode=self.trace_mode,
        )

        # Sets aspect ratio on both panels.
        self._cells_view_box.setAspectLocked(lock=True, ratio=context.aspect_ratio)
        self._noncells_view_box.setAspectLocked(lock=True, ratio=context.aspect_ratio)

        self.session_loaded = True

        # Computes binned activity and triggers initial full redraw.
        self._on_activity_changed(CONFIG.default_activity_mode)
        self.show()

    def _enable_controls(self) -> None:
        """Enables all view, color, and selection buttons after data loading."""
        if self.context_data is None:
            return
        context = self.context_data

        # Enables quadrant buttons.
        for b in range(9):
            self._quadrant_controls.quadrant_buttons.button(b).setEnabled(True)
            self._quadrant_controls.quadrant_buttons.button(b).setStyleSheet(STYLE.button_unpressed)

        # Enables view buttons.
        for b in range(len(self._view_controls.view_names)):
            self._view_controls.view_buttons.button(b).setEnabled(True)
            self._view_controls.view_buttons.button(b).setStyleSheet(STYLE.button_unpressed)
            if b == 0:
                self._view_controls.view_buttons.button(b).setChecked(True)
                self._view_controls.view_buttons.button(b).setStyleSheet(STYLE.button_pressed)

        # Disables channel 2 views if no channel 2 data is available.
        if context.corrected_structural_mean_image is None:
            self._view_controls.view_buttons.button(5).setEnabled(False)
            self._view_controls.view_buttons.button(5).setStyleSheet(STYLE.button_inactive)
            if context.mean_image_channel_2 is None:
                self._view_controls.view_buttons.button(6).setEnabled(False)
                self._view_controls.view_buttons.button(6).setStyleSheet(STYLE.button_inactive)

        # Enables color mode buttons.
        color_button_count = len(self._color_controls.color_buttons.buttons())
        for b in range(color_button_count):
            if b == CONFIG.color_channel_2:
                if context.has_channel_2:
                    self._color_controls.color_buttons.button(b).setEnabled(True)
                    self._color_controls.color_buttons.button(b).setStyleSheet(STYLE.button_unpressed)
            elif b == 0:
                self._color_controls.color_buttons.button(b).setEnabled(True)
                self._color_controls.color_buttons.button(b).setChecked(True)
                self._color_controls.color_buttons.button(b).setStyleSheet(STYLE.button_pressed)
            elif b < CONFIG.color_stat_count:
                self._color_controls.color_buttons.button(b).setEnabled(True)
                self._color_controls.color_buttons.button(b).setStyleSheet(STYLE.button_unpressed)

        # Enables size toggle buttons.
        for button_index, btn in enumerate(self._cell_toggle_controls.size_buttons.buttons()):
            btn.setStyleSheet(STYLE.button_unpressed)
            btn.setEnabled(True)
            if button_index == 0:
                btn.setChecked(True)
                btn.setStyleSheet(STYLE.button_pressed)

        # Enables selection buttons (draw enabled, top/bottom disabled until data analyzed).
        for b in range(3):
            if b == 0:
                self._selection_controls.selection_buttons.button(b).setEnabled(True)
                self._selection_controls.selection_buttons.button(b).setStyleSheet(STYLE.button_unpressed)
            else:
                self._selection_controls.selection_buttons.button(b).setEnabled(False)
                self._selection_controls.selection_buttons.button(b).setStyleSheet(STYLE.button_inactive)

    def _on_view_changed(self, index: int) -> None:
        """Handles background view mode changes.

        Args:
            index: The background view index selected.
        """
        # Updates button styles.
        for i in range(len(CONFIG.view_names)):
            btn = self._view_controls.view_buttons.button(i)
            if btn is not None and btn.isEnabled():
                btn.setStyleSheet(STYLE.button_unpressed)
        btn = self._view_controls.view_buttons.button(index)
        if btn is not None:
            btn.setChecked(True)
            btn.setStyleSheet(STYLE.button_pressed)

        self.background_view = BackgroundView(index)
        self.update_plot()

    def _on_color_changed(self, index: int) -> None:
        """Handles ROI color mode changes.

        Args:
            index: The color mode index selected.
        """
        # Updates button styles.
        for i in range(len(CONFIG.color_names)):
            btn = self._color_controls.color_buttons.button(i)
            if btn is not None and btn.isEnabled():
                btn.setStyleSheet(STYLE.button_unpressed)
        btn = self._color_controls.color_buttons.button(index)
        if btn is not None:
            btn.setChecked(True)
            btn.setStyleSheet(STYLE.button_pressed)

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
            self.temporal_bin_size = max(1, int(self._color_controls.bin_edit.text()))
            nb = int(np.floor(float(self.context_data.frame_count) / float(self.temporal_bin_size)))
            if i == 0:
                f = self.context_data.cell_fluorescence
            elif i == 1:
                f = self.context_data.neuropil_fluorescence
            elif i == CONFIG.activity_mode_subtracted:
                f = self.context_data.cell_fluorescence - 0.7 * self.context_data.neuropil_fluorescence
            else:
                f = self.context_data.spikes
            ncells = self.context_data.roi_count
            bin_size = self.temporal_bin_size
            self.Fbin = f[:, : nb * bin_size].reshape((ncells, nb, bin_size)).mean(axis=2)
            self.Fbin -= self.Fbin.mean(axis=1)[:, np.newaxis]
            self.Fstd = (self.Fbin**2).mean(axis=1) ** 0.5
            self.frame_indices = np.arange(0, self.context_data.frame_count, dtype=np.int32)
            self.update_plot()

    def _on_saturation_changed(self) -> None:
        """Handles saturation range slider changes."""
        self.background_saturation = self._view_controls.range_slider.saturation_values()
        self.update_plot()

    def _on_selection_button(self, index: int) -> None:
        """Handles selection button presses."""
        self._on_roi_selection()

    def _on_quadrant(self, index: int) -> None:
        """Handles quadrant zoom button presses.

        Args:
            index: The quadrant button index (0-8).
        """
        if self.context_data is None:
            return
        qb = self._quadrant_controls.quadrant_buttons
        for i in range(9):
            if qb.button(i).isEnabled():
                qb.button(i).setStyleSheet(STYLE.button_unpressed)
        qb.button(index).setStyleSheet(STYLE.button_pressed)

        x_column = index % STYLE.quadrant_columns
        y_row = index // STYLE.quadrant_columns
        fw = self.context_data.frame_width
        fh = self.context_data.frame_height
        x_lo = (x_column - CONFIG.quadrant_zoom_margin) * fw / STYLE.quadrant_columns
        x_hi = (x_column + 1 + CONFIG.quadrant_zoom_margin) * fw / STYLE.quadrant_columns
        y_lo = (y_row - CONFIG.quadrant_zoom_margin) * fh / STYLE.quadrant_columns
        y_hi = (y_row + 1 + CONFIG.quadrant_zoom_margin) * fh / STYLE.quadrant_columns
        self._cells_view_box.setXRange(x_lo, x_hi)
        self._cells_view_box.setYRange(y_lo, y_hi)
        self._noncells_view_box.setXRange(x_lo, x_hi)
        self._noncells_view_box.setYRange(y_lo, y_hi)

    def _on_number_chosen(self) -> None:
        """Jumps to the ROI number entered in the ROI edit field."""
        if self.session_loaded and self.context_data is not None:
            self.selected_roi_index = int(self._roi_index_edit.text())
            if self.selected_roi_index >= self.context_data.roi_count:
                self.selected_roi_index = self.context_data.roi_count - 1
            self.merge_roi_indices = [self.selected_roi_index]
            self.update_plot()
            self.show()

    def _on_zoom_cell_toggled(self, state: int) -> None:
        """Toggles zoom-to-cell behavior based on checkbox state."""
        if not self.session_loaded:
            return
        self.auto_zoom_to_roi = QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked
        self.update_plot()

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

    def _adjust_trace_level(self, delta: int) -> None:
        """Adjusts the trace panel row stretch factor."""
        tc = self._trace_controls
        tc.trace_level = max(CONFIG.min_trace_level, min(CONFIG.max_trace_level, tc.trace_level + delta))
        self._refresh_traces()

    def _adjust_scale(self, delta: float) -> None:
        """Adjusts the vertical scale factor for multi-trace stacking."""
        tc = self._trace_controls
        tc.scale_factor = max(CONFIG.min_scale, min(CONFIG.max_scale, tc.scale_factor + delta))
        self._refresh_traces()

    def _refresh_traces(self) -> None:
        """Refreshes the trace panel without redrawing image panels."""
        if self.context_data is None or self.color_arrays is None or self.frame_indices is None:
            return
        plot_trace(
            trace_box=self._trace_box,
            cell_fluorescence=self.context_data.cell_fluorescence,
            neuropil_fluorescence=self.context_data.neuropil_fluorescence,
            spikes=self.context_data.spikes,
            frame_indices=self.frame_indices,
            merge_indices=self.merge_roi_indices,
            activity_mode=self.trace_mode,
            roi_colors=self.color_arrays.cols[self.roi_color_mode],
            traces_visible=self._trace_controls.traces_visible,
            neuropil_visible=self._trace_controls.neuropil_visible,
            deconvolved_visible=self._trace_controls.deconvolved_visible,
            scale_factor=self._trace_controls.scale_factor,
            max_plotted=int(self._trace_controls.max_plotted_edit.text() or "40"),
        )

    def update_plot(self) -> None:
        """Redraws all plot panels including masks, traces, and colorbar."""
        if self.context_data is None or self.color_arrays is None or self.roi_maps is None:
            return
        if self.views is None or self.colorbar_widgets is None or self.colorbar_image is None:
            return
        if self.roi_color_mode == CONFIG.color_correlation and self.Fbin is not None:
            assert self.Fstd is not None
            update_correlation_masks(
                color_arrays=self.color_arrays,
                roi_maps=self.roi_maps,
                binned_fluorescence=self.Fbin,
                fluorescence_std=self.Fstd,
                merge_indices=self.merge_roi_indices,
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
            view1=self._cells_background,
            view2=self._noncells_background,
            views=self.views,
            view_index=self.background_view,
            saturation=self.background_saturation,
        )
        masks = draw_masks(
            context=self.context_data,
            color_arrays=self.color_arrays,
            roi_maps=self.roi_maps,
            roi_color_mode=self.roi_color_mode,
            background_view=self.background_view,
            selected_roi_index=self.selected_roi_index,
            merge_roi_indices=self.merge_roi_indices,
            roi_opacity=self.roi_opacity,
        )
        display_masks(color1=self._cells_overlay, color2=self._noncells_overlay, masks=masks)
        assert self.frame_indices is not None
        plot_trace(
            trace_box=self._trace_box,
            cell_fluorescence=self.context_data.cell_fluorescence,
            neuropil_fluorescence=self.context_data.neuropil_fluorescence,
            spikes=self.context_data.spikes,
            frame_indices=self.frame_indices,
            merge_indices=self.merge_roi_indices,
            activity_mode=self.trace_mode,
            roi_colors=self.color_arrays.cols[self.roi_color_mode],
            traces_visible=self._trace_controls.traces_visible,
            neuropil_visible=self._trace_controls.neuropil_visible,
            deconvolved_visible=self._trace_controls.deconvolved_visible,
            scale_factor=self._trace_controls.scale_factor,
            max_plotted=int(self._trace_controls.max_plotted_edit.text() or "40"),
        )
        if self.auto_zoom_to_roi:
            self._zoom_to_cell()
        self._cells_view_box.show()
        self._noncells_view_box.show()
        self._graphics_widget.show()
        self.show()

        # Updates status bar.
        roi_index = self.selected_roi_index
        cell_count = int(self.context_data.cell_count)
        height = self.context_data.frame_height
        width = self.context_data.frame_width
        session_name = str(self.context_data.output_path) if self.context_data.output_path is not None else "unknown"
        self._status_bar.showMessage(
            f"Session: {session_name}  |  ROI: {roi_index}  |  Cells: {cell_count}  |  Size: {height} x {width}"
        )

    def _ichosen_stats(self) -> None:
        """Updates the ROI statistics labels for the currently selected cell."""
        if self.context_data is None:
            return
        n = self.selected_roi_index
        self._roi_index_edit.setText(str(n))
        roi = self.context_data.roi_statistics[n]
        for k in range(len(self._stats_to_show)):
            key = self._stats_to_show[k]
            ival = getattr(roi, key, None)
            if ival is None:
                continue
            if k + 1 == CONFIG.centroid_stat_index:
                self._roi_stat_labels[k].setText(f"{key}: [{ival[0]:d}, {ival[1]:d}]")
            elif k + 1 == CONFIG.pixel_count_stat_index:
                self._roi_stat_labels[k].setText(f"{key}: {ival:d}")
            else:
                self._roi_stat_labels[k].setText(f"{key}: {ival:2.2f}")

    def _toggle_rois(self, state: int) -> None:
        """Toggles ROI overlay visibility on both image panels."""
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.rois_visible = True
            self._cells_view_box.addItem(self._cells_overlay)
            self._noncells_view_box.addItem(self._noncells_overlay)
        else:
            self.rois_visible = False
            self._cells_view_box.removeItem(self._cells_overlay)
            self._noncells_view_box.removeItem(self._noncells_overlay)
        self._graphics_widget.show()
        self.show()

    def _roi_text(self, state: int) -> None:
        """Toggles ROI number text labels on the image panels."""
        if self.roi_maps is None or self.context_data is None:
            return

        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            for n in range(len(self.roi_maps.text_labels)):
                if self.context_data.cell_classification_labels[n] == 1:
                    self._cells_view_box.addItem(self.roi_maps.text_labels[n])
                else:
                    self._noncells_view_box.addItem(self.roi_maps.text_labels[n])
            self.roi_labels_visible = True
        else:
            for n in range(len(self.roi_maps.text_labels)):
                if self.context_data.cell_classification_labels[n] == 1:
                    with suppress(Exception):
                        self._cells_view_box.removeItem(self.roi_maps.text_labels[n])
                else:
                    with suppress(Exception):
                        self._noncells_view_box.removeItem(self.roi_maps.text_labels[n])
            self.roi_labels_visible = False

    def _zoom_to_cell(self) -> None:
        """Zooms both image panels to center on the currently selected cell."""
        if self.context_data is None:
            return
        irange = 0.1 * np.array([self.context_data.frame_height, self.context_data.frame_width]).max()
        roi_statistics = self.context_data.roi_statistics
        if len(self.merge_roi_indices) > 1:
            apix = np.zeros((0, 2))
            for k in self.merge_roi_indices:
                apix = np.append(
                    apix,
                    np.concatenate(
                        (
                            roi_statistics[k].y_pixels.flatten()[:, np.newaxis],
                            roi_statistics[k].x_pixels.flatten()[:, np.newaxis],
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
            icent = np.array(roi_statistics[self.selected_roi_index].centroid)
            imin = icent - irange
            imax = icent + irange
        self._cells_view_box.setYRange(imin[0], imax[0])
        self._cells_view_box.setXRange(imin[1], imax[1])
        self._noncells_view_box.setYRange(imin[0], imax[0])
        self._noncells_view_box.setXRange(imin[1], imax[1])
        self._graphics_widget.show()
        self.show()

    def _zoom_plot(self, panel: int) -> None:
        """Resets the view range for the specified panel."""
        if panel == CONFIG.cells_plot:
            self._cells_view_box.autoRange()
            self._noncells_view_box.autoRange()
        elif panel == CONFIG.noncells_plot:
            self._noncells_view_box.autoRange()

    def save_cell_classification(self) -> None:
        """Saves the current cell classification labels to cell_classification.npy."""
        if self.context_data is None:
            return
        context = self.context_data
        output_path = context.output_path
        if output_path is None:
            return
        np.save(
            str(output_path / "cell_classification.npy"),
            np.concatenate(
                (
                    np.expand_dims(context.cell_classification_labels, axis=1),
                    np.expand_dims(context.cell_classification_probabilities, axis=1),
                ),
                axis=1,
            ),
        )
        self._cell_toggle_controls.cell_count_label.setText(
            f"{int(context.cell_classification_labels.sum())}"
        )
        self._cell_toggle_controls.noncell_count_label.setText(
            f"{int(context.cell_classification_labels.size - context.cell_classification_labels.sum())}"
        )

    def _flip_plot(self) -> None:
        """Flips the selected ROIs between the cell and non-cell panels and saves the result."""
        if self.context_data is None or self.color_arrays is None or self.roi_maps is None:
            return
        last_reclassified = [self.last_reclassified_index]
        flip_rois(
            context=self.context_data,
            color_arrays=self.color_arrays,
            roi_maps=self.roi_maps,
            selected_roi_index=self.selected_roi_index,
            merge_roi_indices=self.merge_roi_indices,
            last_reclassified_index_out=last_reclassified,
        )
        self.last_reclassified_index = last_reclassified[0]
        self.save_cell_classification()
        self.update_plot()

    def _handle_click(
        self, click_x: int, click_y: int, panel: ROIToolPanel, is_right: bool, is_multi: bool
    ) -> bool:
        """Handles mouse clicks on image panels.

        Left-click chooses a cell. Shift/ctrl-click adds or removes from the merge selection.
        Right-click flips the clicked ROI between cell and non-cell panels.

        Args:
            click_x: Column coordinate of the click.
            click_y: Row coordinate of the click.
            panel: Identifies the image panel that received the click.
            is_right: Determines whether the click was a right-click.
            is_multi: Determines whether shift or ctrl was held during the click.

        Returns:
            True if the click was consumed, False to allow the default context menu.
        """
        if not self.session_loaded or self.roi_maps is None or self.context_data is None:
            return False

        if (
            click_y < 0
            or click_y >= self.context_data.frame_height
            or click_x < 0
            or click_x >= self.context_data.frame_width
        ):
            return False

        ichosen = int(self.roi_maps.iroi[panel, 0, click_y, click_x])
        if ichosen < 0:
            return False

        if is_right:
            # Sets selection to the clicked ROI (if not already selected) and flips it.
            if ichosen not in self.merge_roi_indices:
                self.merge_roi_indices = [ichosen]
                self.selected_roi_index = ichosen
            self._flip_plot()
            return True

        merged = False
        if is_multi and (
            self.context_data.cell_classification_labels[self.merge_roi_indices[0]]
            == self.context_data.cell_classification_labels[ichosen]
        ):
            if ichosen not in self.merge_roi_indices:
                self.merge_roi_indices.append(ichosen)
                self.selected_roi_index = ichosen
                merged = True
            elif len(self.merge_roi_indices) > 1:
                self.merge_roi_indices.remove(ichosen)
                self.selected_roi_index = self.merge_roi_indices[0]
                merged = True
        if not merged:
            self.merge_roi_indices = [ichosen]
            self.selected_roi_index = ichosen

        if self.roi_tool_active:
            self._roi_remove()
        if not self._cell_toggle_controls.size_buttons.button(1).isChecked():
            for btn in self._selection_controls.selection_buttons.buttons():
                if btn.isChecked():
                    btn.setStyleSheet(STYLE.button_unpressed)
        self.update_plot()
        return True

    def _on_roi_selection(self) -> None:
        """Draws a rectangular ROI selection on the active image panel."""
        draw = False
        if self._cell_toggle_controls.size_buttons.button(0).isChecked():
            wplot = 0
            view = self._cells_view_box.viewRange()
            draw = True
        elif self._cell_toggle_controls.size_buttons.button(2).isChecked():
            wplot = 1
            view = self._noncells_view_box.viewRange()
            draw = True
        if draw:
            self._roi_remove()
            self._selection_controls.selection_buttons.button(0).setStyleSheet(STYLE.button_pressed)
            self.roi_tool_panel = ROIToolPanel(wplot)
            imx = (view[0][1] + view[0][0]) / 2
            imy = (view[1][1] + view[1][0]) / 2
            dx = (view[0][1] - view[0][0]) / 4
            dy = (view[1][1] - view[1][0]) / 4
            dx = np.minimum(dx, 300)
            dy = np.minimum(dy, 300)
            imx = imx - dx / 2
            imy = imy - dy / 2
            self._active_roi_selection = pg.RectROI([imx, imy], [dx, dy], pen="w", sideScalers=True)
            if wplot == 0:
                self._cells_view_box.addItem(self._active_roi_selection)
            else:
                self._noncells_view_box.addItem(self._active_roi_selection)
            self._roi_position()
            self._active_roi_selection.sigRegionChangeFinished.connect(self._roi_position)
            self.roi_tool_active = True

    def _roi_remove(self) -> None:
        """Removes the current rectangular ROI selection and resets button styles."""
        if self.roi_tool_active:
            if self.roi_tool_panel == 0:
                self._cells_view_box.removeItem(self._active_roi_selection)
            else:
                self._noncells_view_box.removeItem(self._active_roi_selection)
            self.roi_tool_active = False
        if self._cell_toggle_controls.size_buttons.button(1).isChecked():
            self._selection_controls.selection_buttons.button(0).setStyleSheet(STYLE.button_inactive)
            self._selection_controls.selection_buttons.button(0).setEnabled(False)
        else:
            self._selection_controls.selection_buttons.button(0).setStyleSheet(STYLE.button_unpressed)

    def _roi_position(self) -> None:
        """Computes the pixel region covered by the ROI and selects contained cells."""
        if self.context_data is None:
            return
        pos0 = self._active_roi_selection.getSceneHandlePositions()
        pos = (
            self._cells_view_box.mapSceneToView(pos0[0][1])
            if self.roi_tool_panel == 0
            else self._noncells_view_box.mapSceneToView(pos0[0][1])
        )
        posy = pos.y()
        posx = pos.x()
        sizex, sizey = self._active_roi_selection.size()
        xrange = (np.arange(-1 * int(sizex), 1) + int(posx)).astype(np.int32)
        yrange = (np.arange(-1 * int(sizey), 1) + int(posy)).astype(np.int32)
        xrange = xrange[xrange >= 0]
        xrange = xrange[xrange < self.context_data.frame_width]
        yrange = yrange[yrange >= 0]
        yrange = yrange[yrange < self.context_data.frame_height]
        ypix, xpix = np.meshgrid(yrange, xrange)
        self._select_cells(ypix, xpix)

    def _select_cells(self, ypix: np.ndarray, xpix: np.ndarray) -> None:
        """Selects cells whose pixels overlap the given coordinate arrays."""
        if self.roi_maps is None or self.context_data is None:
            return
        i = self.roi_tool_panel
        roi_indices = self.roi_maps.iroi[i, 0, ypix, xpix]
        icells = np.unique(roi_indices[roi_indices >= 0])
        self.merge_roi_indices = []
        for n in icells:
            pixel_count = self.context_data.roi_statistics[n].pixel_count
            if (self.roi_maps.iroi[i, :, ypix, xpix] == n).sum() > 0.6 * pixel_count:
                self.merge_roi_indices.append(n)
        if self.merge_roi_indices:
            self.selected_roi_index = self.merge_roi_indices[0]
            self.update_plot()
            self.show()

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # noqa: N802
        """Accepts drag events that contain file URLs."""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # noqa: N802
        """Handles dropped directories by loading session data."""
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        console.echo(message=f"Files dropped: {files}")
        dropped_path = Path(files[0])
        if dropped_path.is_dir():
            self._load_session(session_path=dropped_path)
        else:
            console.echo(
                message=f"Invalid drop target '{dropped_path}'. Drop a cindra output directory.",
                level=LogLevel.ERROR,
            )
