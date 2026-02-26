"""Provides the main application window for the ROI viewer and editor GUI."""

from __future__ import annotations

import shutil
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
    QHBoxLayout,
    QMainWindow,
    QVBoxLayout,
    QGridLayout,
    QPushButton,
    QButtonGroup,
    QSlider,
)
from ataraxis_base_utilities import LogLevel, console

import cindra

from . import (
    menu_bar,
    merge_dialog,
    context_loader,
    classifier_panel,
)
from .. import overlays
from ..widgets import (
    TraceBox,
    ViewBox,
    RangeSlider,
    plot_trace,
    apply_quadrant_zoom,
    _ColorButton,
    _ViewButton,
    _QuadButton,
    _SizeButton,
    _SelectionButton,
)
from ..constants import STYLE, CONFIG, TraceMode, ROIColorMode, ROIToolPanel, BackgroundView
from ..data_models import (
    ColorControls,
    ViewControls,
    TraceControls,
    ColorArrays,
    ROIIndexMaps,
    ColorbarWidgets,
    SelectionControls,
    CellToggleControls,
    QuadrantControls,
    ClassifierControls,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PySide6.QtGui import QAction, QKeyEvent, QDropEvent, QDragEnterEvent
    from PySide6.QtWidgets import QMenu

    from ..single_day_context import ROIViewerData

_CINDRA_DIR: Path = Path(cindra.__file__).parent
"""The path to the root of the cindra package directory."""

_ICON_PATH: str = str(_CINDRA_DIR / "logo" / "logo.png")
"""The string path to the application icon file."""


class ROIEditor(QMainWindow):
    """Provides the main application window for the cindra graphical interface."""

    def __init__(self, session_path: Path | None = None) -> None:
        """Initializes the main window, menus, buttons, and graphics panels.

        Args:
            session_path: Optional path to a cindra output directory to load on startup.
        """
        super().__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")

        # Core state objects.
        self.context_data: ROIViewerData | None = None
        self.color_arrays: ColorArrays | None = None
        self.roi_maps: ROIIndexMaps | None = None
        self.colorbar_widgets: ColorbarWidgets | None = None
        self.colorbar_image: NDArray[np.uint8] | None = None
        self.views: NDArray[np.uint8] | None = None

        # Display state (replaces the old ViewState dataclass).
        self.rois_visible: bool = True
        self.roi_color_mode: ROIColorMode = ROIColorMode.RANDOM
        self.background_view: BackgroundView = BackgroundView.ROIS_ONLY
        self.roi_opacity: list[int] = [127, 255]
        self.background_saturation: list[int] = [0, 255]
        self.roi_colormap: str = "hsv"
        self.selected_roi_index: int = 0
        self.merge_roi_indices: list[int] = [0]
        self.last_reclassified_index: int = 0
        self.roi_tool_active: bool = False
        self.roi_tool_panel: ROIToolPanel = ROIToolPanel.CELLS
        self.trace_mode: int = TraceMode.NEUROPIL_CORRECTED
        self.temporal_bin_size: int = 1
        self.fluorescence_visible: bool = True
        self.neuropil_visible: bool = True
        self.deconvolved_visible: bool = True
        self.auto_zoom_to_roi: bool = False
        self.roi_labels_visible: bool = False
        self.session_loaded: bool = False
        self.colocalization_threshold: float = 0.6

        # Computed binned activity state (used by MergeWindow correlation computation).
        self.Fbin: NDArray[np.float32] | None = None
        self.Fstd: NDArray[np.float32] | None = None
        self.frame_indices: NDArray | None = None

        self.setGeometry(50, 50, 1500, 800)
        self.setWindowTitle("cindra (run pipeline or load session directory)")

        app_icon = QtGui.QIcon()
        app_icon.addFile(_ICON_PATH, QtCore.QSize(16, 16))
        app_icon.addFile(_ICON_PATH, QtCore.QSize(24, 24))
        app_icon.addFile(_ICON_PATH, QtCore.QSize(32, 32))
        app_icon.addFile(_ICON_PATH, QtCore.QSize(48, 48))
        app_icon.addFile(_ICON_PATH, QtCore.QSize(64, 64))
        app_icon.addFile(_ICON_PATH, QtCore.QSize(256, 256))
        self.setWindowIcon(app_icon)
        self.setStyleSheet(STYLE.main_window)

        # Classifier file initialization.
        user_dir = Path.home() / ".cindra"
        user_dir.mkdir(exist_ok=True)
        class_dir = user_dir / "classifiers"
        class_dir.mkdir(exist_ok=True)
        self.classuser = str(class_dir / "classifier_user.npz")
        self.classorig = str(_CINDRA_DIR / "classification" / "classifier.npz")
        if not Path(self.classuser).is_file():
            shutil.copy(self.classorig, self.classuser)
        self.classfile = self.classuser

        menu_bar.mainmenu(self)
        menu_bar.classifier(self)
        menu_bar.mergebar(self)
        menu_bar.plugins(self)

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

        # Initializes merge tracking.
        self.merged: list[object] = []

        # Menu bar actions (set dynamically by menu_bar module).
        self.manual: QAction
        self.saveMerge: QAction
        self.sugMerge: QAction
        self.loadClass: QAction
        self.loadTrain: QAction
        self.loadUClass: QAction
        self.loadSClass: QAction
        self.saveDefault: QAction
        self.resetDefault: QAction
        self.loadMenu: QMenu
        self.trainfiles: list[str]
        self.statlabels: object
        self.plugins: dict[str, object]

        # Classifier model (set by classifier_panel when a classifier is loaded).
        self.model: object

        # Applies NoFocus policy to all buttons in the control panel.
        for widget in control_panel.findChildren(QWidget):
            widget.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        if session_path is not None:
            context_loader.load_session(parent=self, session_path=session_path)
        self.setAcceptDrops(True)
        self.show()
        self._graphics_widget.show()

    def _build_control_panel(self) -> QWidget:
        """Builds the right-side control panel with QGroupBox sections.

        Returns:
            The control panel widget containing all grouped controls.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 1. ROI Visibility — inline checkboxes.
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

        # 4. Background.
        background_box, self._view_controls = self._create_view_controls()
        layout.addWidget(background_box)

        # 5. ROI Colors + colorbar.
        colors_box, self._color_controls = self._create_color_controls()
        self.colorbar_widgets = self._create_colorbar()
        colors_layout = colors_box.layout()
        assert colors_layout is not None
        colors_layout.addWidget(self.colorbar_widgets.widget)
        layout.addWidget(colors_box)

        # 6. Classifier.
        classifier_box, self._classifier_controls = self._create_classifier_controls()
        self._classifier_controls.add_to_class_button.clicked.connect(
            lambda: classifier_panel._add_to(self)
        )
        layout.addWidget(classifier_box)

        # 7. Selected ROI — inline: ROI index edit + stat labels.
        roi_box = QGroupBox("Selected ROI")
        roi_box.setStyleSheet("QGroupBox { color: white; }")
        roi_layout = QVBoxLayout(roi_box)
        self.stats_to_show = [
            "centroid",
            "pixel_count",
            "skewness",
            "compactness",
            "footprint",
            "aspect_ratio",
        ]
        label_font = STYLE.label_font()
        self._roi_index_edit = QLineEdit(self)
        self._roi_index_edit.setValidator(QtGui.QIntValidator(0, 10000))
        self._roi_index_edit.setText("0")
        self._roi_index_edit.setFixedWidth(STYLE.roi_edit_width)
        self._roi_index_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        self._roi_index_edit.returnPressed.connect(self.number_chosen)
        roi_layout.addWidget(self._roi_index_edit)
        self._roi_stat_labels: list[QLabel] = []
        for k in range(len(self.stats_to_show)):
            stat_label = QLabel(self.stats_to_show[k])
            stat_label.setFont(label_font)
            stat_label.setStyleSheet(STYLE.white_label)
            stat_label.resize(stat_label.minimumSizeHint())
            roi_layout.addWidget(stat_label)
            self._roi_stat_labels.append(stat_label)
        layout.addWidget(roi_box)

        # 8. Trace Display.
        trace_box, self._trace_controls = self._create_trace_controls()
        self._zoom_to_cell_checkbox = QCheckBox("zoom to cell")
        self._zoom_to_cell_checkbox.setStyleSheet(STYLE.white_label)
        self._zoom_to_cell_checkbox.stateChanged.connect(self._zoom_cell)
        trace_layout = trace_box.layout()
        assert trace_layout is not None
        trace_layout.addWidget(self._zoom_to_cell_checkbox)
        layout.addWidget(trace_box)

        # 9. Navigation.
        nav_box, self._quadrant_controls = self._create_quadrant_buttons()
        layout.addWidget(nav_box)

        layout.addStretch()
        return panel

    def _create_selection_buttons(self) -> tuple[QGroupBox, SelectionControls]:
        """Creates the cell selection buttons and the top-n input field.

        Returns:
            Tuple of (group box, selection controls container).
        """
        group_box = QGroupBox("Cell Selection")
        group_box.setStyleSheet("QGroupBox { color: white; }")
        grid_layout = QGridLayout(group_box)

        selection_button_group = QButtonGroup()

        labels = [" draw selection", " select top n", " select bottom n"]
        for button_index in range(3):
            button = _SelectionButton(
                button_id=button_index,
                text=labels[button_index],
                owner=self,
                button_group=selection_button_group,
                on_press=self._on_roi_selection,
            )
            selection_button_group.addButton(button, button_index)
            grid_layout.addWidget(button, button_index, 0, 1, 1)
            button.setEnabled(False)
        selection_button_group.setExclusive(True)

        count_label = QLabel("n=")
        count_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        count_label.setStyleSheet(STYLE.white_label)
        count_label.setFont(STYLE.label_font_bold())
        grid_layout.addWidget(count_label, 1, 1, 1, 1)

        controls = SelectionControls(
            selection_buttons=selection_button_group,
            top_count_edit=QLineEdit(self),
        )

        controls.top_count_edit.setValidator(QtGui.QIntValidator(0, CONFIG.max_top_n))
        controls.top_count_edit.setText(str(CONFIG.default_top_n))
        controls.top_count_edit.setFixedWidth(STYLE.small_edit_width)
        controls.top_count_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        controls.top_count_edit.returnPressed.connect(self._on_roi_selection)
        grid_layout.addWidget(controls.top_count_edit, 2, 1, 1, 1)

        return group_box, controls

    def _create_cell_toggle_buttons(self) -> tuple[QGroupBox, CellToggleControls]:
        """Creates the cell / not-cell / both size-toggle buttons and ROI count labels.

        Returns:
            Tuple of (group box, cell toggle controls container).
        """
        group_box = QGroupBox("View Toggle")
        group_box.setStyleSheet("QGroupBox { color: white; }")
        grid_layout = QGridLayout(group_box)

        cell_count_label = QLabel("")
        cell_count_label.setStyleSheet(STYLE.white_label)
        cell_count_label.setAlignment(
            QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        grid_layout.addWidget(cell_count_label, 0, 0, 1, 1)

        noncell_count_label = QLabel("")
        noncell_count_label.setStyleSheet(STYLE.white_label)
        grid_layout.addWidget(noncell_count_label, 0, 2, 1, 1)

        size_button_group = QButtonGroup(self)
        labels = [" cells", " both", " not cells"]
        for button_index, label_text in enumerate(labels):
            button = _SizeButton(
                button_id=button_index,
                text=label_text,
                owner=self,
                button_group=size_button_group,
                on_press=self.update_plot,
            )
            size_button_group.addButton(button, button_index)
            grid_layout.addWidget(button, 1, button_index, 1, 1)
            button.setEnabled(button_index == CONFIG.view_both)
        size_button_group.setExclusive(True)

        return group_box, CellToggleControls(
            size_buttons=size_button_group,
            cell_count_label=cell_count_label,
            noncell_count_label=noncell_count_label,
        )

    def _create_view_controls(self) -> tuple[QGroupBox, ViewControls]:
        """Creates background view selection controls inside a group box.

        Returns:
            Tuple of (group box, view controls container).
        """
        group_box = QGroupBox("Background")
        group_box.setStyleSheet("QGroupBox { color: white; }")
        grid_layout = QGridLayout(group_box)

        view_button_group = QButtonGroup(self)

        for button_index, name in enumerate(CONFIG.view_names):
            button = _ViewButton(
                button_id=button_index,
                text="&" + name,
                owner=self,
                button_group=view_button_group,
                on_press=lambda idx=button_index: self._on_view_mode_changed(idx),
            )
            view_button_group.addButton(button, button_index)
            grid_layout.addWidget(button, button_index, 0, 1, 1)

            if button_index == 0:
                saturation_label = QLabel("sat: ")
                saturation_label.setStyleSheet(STYLE.white_label)
                grid_layout.addWidget(saturation_label, button_index, 1, 1, 1)

            button.setEnabled(False)

        view_button_group.setExclusive(True)

        range_slider = RangeSlider(
            owner=self,
            on_release=self.update_plot,
        )
        range_slider.setMinimum(0)
        range_slider.setMaximum(255)
        range_slider.setLow(0)
        range_slider.setHigh(255)
        range_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        grid_layout.addWidget(range_slider, 1, 1, len(CONFIG.view_names) - 2, 1)

        controls = ViewControls(
            view_buttons=view_button_group,
            range_slider=range_slider,
        )
        return group_box, controls

    def _create_color_controls(self) -> tuple[QGroupBox, ColorControls]:
        """Creates color statistic selection buttons and their associated controls.

        Returns:
            Tuple of (group box, color controls container).
        """
        group_box = QGroupBox("ROI Colors")
        group_box.setStyleSheet("QGroupBox { color: white; }")
        grid_layout = QGridLayout(group_box)

        color_button_group = QButtonGroup(self)

        colormap_chooser = QComboBox()
        colormap_chooser.addItems(CONFIG.colormaps)
        colormap_chooser.setCurrentIndex(0)
        colormap_chooser.setFont(STYLE.label_font())
        colormap_chooser.setFixedWidth(STYLE.color_edit_width)
        grid_layout.addWidget(colormap_chooser, 0, 1, 1, 1)

        for button_index, name in enumerate(CONFIG.color_names):
            button = _ColorButton(
                button_id=button_index,
                text="&" + name,
                owner=self,
                button_group=color_button_group,
                on_press=lambda idx=button_index: self._on_color_mode_changed(idx),
            )
            color_button_group.addButton(button, button_index)

            if CONFIG.color_narrow_range_start <= button_index < CONFIG.color_narrow_range_end:
                grid_layout.addWidget(button, button_index, 0, 1, 1)
            else:
                grid_layout.addWidget(button, button_index, 0, 1, 2)

            button.setEnabled(False)

        # Channel 2 probability threshold edit.
        channel_2_edit = QLineEdit(self)
        channel_2_edit.setText("0.6")
        channel_2_edit.setFixedWidth(STYLE.color_edit_width)
        channel_2_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(channel_2_edit, len(CONFIG.color_names) - 4, 1, 1, 1)

        # Classifier probability edit.
        classifier_edit = QLineEdit(self)
        classifier_edit.setText("0.5")
        classifier_edit.setFixedWidth(STYLE.color_edit_width)
        classifier_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(classifier_edit, len(CONFIG.color_names) - 3, 1, 1, 1)

        # Binning size edit.
        bin_edit = QLineEdit(self)
        bin_edit.setValidator(QtGui.QIntValidator(0, 500))
        bin_edit.setText("1")
        bin_edit.setFixedWidth(STYLE.color_edit_width)
        bin_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(bin_edit, len(CONFIG.color_names) - 2, 1, 1, 1)

        controls = ColorControls(
            color_buttons=color_button_group,
            colormap_chooser=colormap_chooser,
            channel_2_edit=channel_2_edit,
            classifier_edit=classifier_edit,
            bin_edit=bin_edit,
        )

        # Connects callbacks directly instead of through the signal bus.
        colormap_chooser.activated.connect(self._on_color_mode_changed)
        channel_2_edit.returnPressed.connect(self.update_plot)
        classifier_edit.returnPressed.connect(self.update_plot)
        bin_edit.returnPressed.connect(self.mode_change)

        return group_box, controls

    def _create_colorbar(self) -> ColorbarWidgets:
        """Creates the colorbar widget displaying the current color mapping.

        Returns:
            Colorbar widgets container.
        """
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

    def _create_classifier_controls(self) -> tuple[QGroupBox, ClassifierControls]:
        """Creates the classifier section inside a group box.

        Returns:
            Tuple of (group box, classifier controls container).
        """
        group_box = QGroupBox("Classifier")
        group_box.setStyleSheet("QGroupBox { color: white; }")
        box_layout = QVBoxLayout(group_box)

        classifier_label = QLabel(
            "<font color='white'>not loaded (using prob from cell_classification.npy)</font>"
        )
        classifier_label.setFont(STYLE.label_font())
        box_layout.addWidget(classifier_label)

        add_to_class_button = QPushButton(" add current data to classifier")
        add_to_class_button.setFont(STYLE.label_font_bold())
        add_to_class_button.setStyleSheet(STYLE.button_inactive)
        box_layout.addWidget(add_to_class_button)

        return group_box, ClassifierControls(
            classifier_label=classifier_label,
            add_to_class_button=add_to_class_button,
        )

    def _create_trace_controls(self) -> tuple[QGroupBox, TraceControls]:
        """Creates trace panel controls inside a group box.

        Returns:
            Tuple of (group box, trace controls container).
        """
        group_box = QGroupBox("Trace Display")
        group_box.setStyleSheet("QGroupBox { color: white; }")
        grid_layout = QGridLayout(group_box)

        # Activity mode label and combo box.
        activity_label = QLabel("Activity mode:")
        activity_label.setStyleSheet(STYLE.white_label)
        grid_layout.addWidget(activity_label, 0, 0, 1, 1)

        activity_combo = QComboBox(self)
        activity_combo.setFixedWidth(STYLE.combo_box_width)
        grid_layout.addWidget(activity_combo, 1, 0, 1, 1)
        activity_combo.addItem("F")
        activity_combo.addItem("Fneu")
        activity_combo.addItem("F - 0.7*Fneu")
        activity_combo.addItem("deconvolved")
        activity_combo.setCurrentIndex(CONFIG.default_activity_mode)
        activity_combo.currentIndexChanged.connect(self.mode_change)

        # Trace resize arrow buttons (up/down).
        arrow_up = QPushButton(" \u25b2")
        arrow_down = QPushButton(" \u25bc")
        arrow_buttons = [arrow_up, arrow_down]

        for button_index, button in enumerate(arrow_buttons):
            button.setMaximumWidth(STYLE.square_button_max_width)
            button.setFont(STYLE.arrow_button_font())
            button.setStyleSheet(STYLE.button_unpressed)
            grid_layout.addWidget(
                button, button_index, 1, 1, 1, QtCore.Qt.AlignmentFlag.AlignRight
            )

        # Scale adjustment buttons (+/-).
        scale_up = QPushButton(" +")
        scale_down = QPushButton(" -")
        scale_buttons = [scale_up, scale_down]

        for button_index, button in enumerate(scale_buttons):
            button.setMaximumWidth(STYLE.square_button_max_width)
            button.setFont(STYLE.arrow_button_font())
            button.setStyleSheet(STYLE.button_unpressed)
            grid_layout.addWidget(button, button_index, 2, 1, 1)

        # Max plotted count label and input.
        max_plotted_label = QLabel("max # plotted:")
        max_plotted_label.setStyleSheet(STYLE.white_label)
        grid_layout.addWidget(max_plotted_label, 2, 0, 1, 1)

        max_plotted_edit = QLineEdit(self)
        max_plotted_edit.setValidator(QtGui.QIntValidator(0, CONFIG.max_plotted_count))
        max_plotted_edit.setText(str(CONFIG.default_plotted_count))
        max_plotted_edit.setFixedWidth(STYLE.small_edit_width)
        max_plotted_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        grid_layout.addWidget(max_plotted_edit, 3, 0, 1, 1)

        # Trace visibility checkboxes.
        deconvolved_checkbox = QCheckBox("deconv [N]")
        deconvolved_checkbox.setStyleSheet(STYLE.white_label)
        deconvolved_checkbox.toggle()
        grid_layout.addWidget(deconvolved_checkbox, 3, 1, 1, 1)

        neuropil_checkbox = QCheckBox("neuropil [B]")
        neuropil_checkbox.setStyleSheet(STYLE.red_label)
        neuropil_checkbox.toggle()
        grid_layout.addWidget(neuropil_checkbox, 3, 2, 1, 1)

        traces_checkbox = QCheckBox("raw fluor [V]")
        traces_checkbox.setStyleSheet(STYLE.cyan_label)
        traces_checkbox.toggle()
        grid_layout.addWidget(traces_checkbox, 3, 3, 1, 1)

        # Assembles the controls container.
        controls = TraceControls(
            activity_combo=activity_combo,
            deconvolved_checkbox=deconvolved_checkbox,
            neuropil_checkbox=neuropil_checkbox,
            traces_checkbox=traces_checkbox,
            max_plotted_edit=max_plotted_edit,
            arrow_buttons=arrow_buttons,
            scale_buttons=scale_buttons,
        )

        # Connects callbacks directly.
        arrow_up.clicked.connect(lambda: self._expand_trace(controls=controls))
        arrow_down.clicked.connect(lambda: self._collapse_trace(controls=controls))
        scale_up.clicked.connect(lambda: self._expand_scale(controls=controls))
        scale_down.clicked.connect(lambda: self._collapse_scale(controls=controls))
        max_plotted_edit.returnPressed.connect(self._on_trace_update)
        deconvolved_checkbox.toggled.connect(
            lambda: self._on_deconvolved_toggle(controls=controls)
        )
        neuropil_checkbox.toggled.connect(
            lambda: self._on_neuropil_toggle(controls=controls)
        )
        traces_checkbox.toggled.connect(
            lambda: self._on_traces_toggle(controls=controls)
        )

        return group_box, controls

    def _create_quadrant_buttons(self) -> tuple[QGroupBox, QuadrantControls]:
        """Creates the 3x3 quadrant zoom navigation buttons.

        Returns:
            Tuple of (group box, quadrant controls container).
        """
        group_box = QGroupBox("Navigation")
        group_box.setStyleSheet("QGroupBox { color: white; }")
        grid_layout = QGridLayout(group_box)

        quadrant_button_group = QButtonGroup(self)
        columns = STYLE.quadrant_columns
        for button_index in range(9):
            button = _QuadButton(
                button_id=button_index,
                text=" " + str(button_index + 1),
                owner=self,
                button_group=quadrant_button_group,
                on_press=self.update_plot,
            )
            quadrant_button_group.addButton(button, button_index)
            row = button_index // columns
            col = button_index % columns
            grid_layout.addWidget(button, row, col, 1, 1)
            button.setEnabled(False)
        quadrant_button_group.setExclusive(True)

        return group_box, QuadrantControls(quadrant_buttons=quadrant_button_group)

    def _build_graphics(self) -> None:
        """Creates the main plotting area with cells, non-cells, and trace panels."""
        # Cells image panel.
        self._cells_view_box = ViewBox(
            panel=ROIToolPanel.CELLS,
            name="plot1",
            border=[100, 100, 100],
            invert_y=True,
        )
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

        # Non-cells image panel.
        self._noncells_view_box = ViewBox(
            panel=ROIToolPanel.NON_CELLS,
            name="plot2",
            border=[100, 100, 100],
            invert_y=True,
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

        # Links the two view panels.
        self._noncells_view_box.setXLink("plot1")
        self._noncells_view_box.setYLink("plot1")

        # Installs click and zoom handlers via the ViewBox callback API.
        self._cells_view_box.set_click_handler(self._handle_click)
        self._noncells_view_box.set_click_handler(self._handle_click)
        self._cells_view_box.set_zoom_handler(lambda: self._zoom_plot(CONFIG.cells_plot))
        self._noncells_view_box.set_zoom_handler(lambda: self._zoom_plot(CONFIG.noncells_plot))

        # Fluorescence trace plot.
        self._trace_box = TraceBox()
        self._trace_box.setMouseEnabled(x=True, y=False)
        self._trace_box.enableAutoRange(x=True, y=True)
        self._graphics_widget.addItem(self._trace_box, row=1, col=0, colspan=2)
        self._graphics_widget.ci.layout.setRowStretchFactor(0, 2)
        ci_layout = self._graphics_widget.ci.layout
        ci_layout.setColumnMinimumWidth(0, 1)
        ci_layout.setColumnMinimumWidth(1, 1)
        ci_layout.setHorizontalSpacing(20)

    # --- View / color mode change handlers ---

    def _on_view_mode_changed(self, index: int) -> None:
        """Handles background view mode changes.

        Args:
            index: The background view index selected.
        """
        self.background_view = BackgroundView(index)
        self.update_plot()

    def _on_color_mode_changed(self, index: int | None = None) -> None:
        """Handles ROI color mode changes.

        Args:
            index: The color mode index selected. When None, uses the current color button.
        """
        if index is not None:
            self.roi_color_mode = ROIColorMode(index)
        if self.context_data is not None and self.color_arrays is not None and self.roi_maps is not None:
            colormap = self._color_controls.colormap_chooser.currentText()
            if colormap != self.roi_colormap:
                self.roi_colormap = colormap
                self.colorbar_image = overlays.update_colormap(
                    color_arrays=self.color_arrays,
                    roi_maps=self.roi_maps,
                    colormap=colormap,
                )
            if (
                self.context_data.has_channel_2
                and abs(
                    float(self._color_controls.channel_2_edit.text()) - self.colocalization_threshold
                )
                > CONFIG.channel_2_threshold_epsilon
            ):
                self.colocalization_threshold = float(self._color_controls.channel_2_edit.text())
                overlays.update_chan2_colors(
                    context=self.context_data,
                    colocalization_threshold=self.colocalization_threshold,
                    color_arrays=self.color_arrays,
                    roi_maps=self.roi_maps,
                )
        self.update_plot()

    # --- Trace control handlers ---

    def _on_trace_update(self) -> None:
        """Handles trace-only update requests."""
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

    def _expand_scale(self, controls: TraceControls) -> None:
        """Increases the vertical scale factor for multi-trace stacking.

        Args:
            controls: The trace controls container.
        """
        controls.scale_factor = min(CONFIG.max_scale, controls.scale_factor + CONFIG.scale_step)
        self._on_trace_update()

    def _collapse_scale(self, controls: TraceControls) -> None:
        """Decreases the vertical scale factor for multi-trace stacking.

        Args:
            controls: The trace controls container.
        """
        controls.scale_factor = max(CONFIG.min_scale, controls.scale_factor - CONFIG.scale_step)
        self._on_trace_update()

    def _expand_trace(self, controls: TraceControls) -> None:
        """Increases the trace panel row stretch factor.

        Args:
            controls: The trace controls container.
        """
        controls.trace_level = min(CONFIG.max_trace_level, controls.trace_level + 1)
        self._on_trace_update()

    def _collapse_trace(self, controls: TraceControls) -> None:
        """Decreases the trace panel row stretch factor.

        Args:
            controls: The trace controls container.
        """
        controls.trace_level = max(CONFIG.min_trace_level, controls.trace_level - 1)
        self._on_trace_update()

    def _on_deconvolved_toggle(self, controls: TraceControls) -> None:
        """Handles the deconvolved trace visibility checkbox toggle.

        Args:
            controls: The trace controls container.
        """
        controls.deconvolved_visible = controls.deconvolved_checkbox.isChecked()
        self._on_trace_update()

    def _on_neuropil_toggle(self, controls: TraceControls) -> None:
        """Handles the neuropil trace visibility checkbox toggle.

        Args:
            controls: The trace controls container.
        """
        controls.neuropil_visible = controls.neuropil_checkbox.isChecked()
        self._on_trace_update()

    def _on_traces_toggle(self, controls: TraceControls) -> None:
        """Handles the raw fluorescence trace visibility checkbox toggle.

        Args:
            controls: The trace controls container.
        """
        controls.traces_visible = controls.traces_checkbox.isChecked()
        self._on_trace_update()

    # --- Drag and drop ---

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802
        """Accepts drag events that contain file URLs."""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802
        """Handles dropped directories by loading session data."""
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        console.echo(message=f"Files dropped: {files}")
        dropped_path = Path(files[0])
        if dropped_path.is_dir():
            context_loader.load_session(parent=self, session_path=dropped_path)
        else:
            console.echo(
                message=f"Invalid drop target '{dropped_path}'. Drop a cindra output directory.",
                level=LogLevel.ERROR,
            )

    # --- ROI text and zoom toggles ---

    def _roi_text(self, state: int) -> None:
        """Toggles ROI number text labels on the image panels.

        Args:
            state: Qt checkbox state value.
        """
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

    def _zoom_cell(self, state: int) -> None:
        """Toggles zoom-to-cell behavior based on checkbox state.

        Args:
            state: Qt checkbox state value.
        """
        if not self.session_loaded:
            return
        self.auto_zoom_to_roi = QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked
        self.update_plot()

    # --- Keyboard shortcuts ---

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        """Handles keyboard shortcuts for view switching, ROI navigation, and toggles.

        Args:
            event: The key press event from Qt.
        """
        if not self.session_loaded:
            return
        if event.modifiers() in {
            QtCore.Qt.KeyboardModifier.ControlModifier,
            QtCore.Qt.KeyboardModifier.ShiftModifier,
        }:
            return
        if event.key() == QtCore.Qt.Key.Key_Return:
            if (
                event.modifiers() == QtCore.Qt.KeyboardModifier.AltModifier
                and len(self.merge_roi_indices) > 1
            ):
                merge_dialog.do_merge(self)
        elif event.key() == QtCore.Qt.Key.Key_Escape:
            self._zoom_plot(CONFIG.cells_plot)
            self._trace_box.autoRange()
            self.show()
        elif event.key() == QtCore.Qt.Key.Key_Delete:
            self._roi_remove()
        elif event.key() == QtCore.Qt.Key.Key_Q:
            self._view_controls.view_buttons.button(0).setChecked(True)
            self._on_view_mode_changed(0)
        elif event.key() == QtCore.Qt.Key.Key_W:
            self._view_controls.view_buttons.button(1).setChecked(True)
            self._on_view_mode_changed(1)
        elif event.key() == QtCore.Qt.Key.Key_E:
            self._view_controls.view_buttons.button(2).setChecked(True)
            self._on_view_mode_changed(2)
        elif event.key() == QtCore.Qt.Key.Key_R:
            self._view_controls.view_buttons.button(3).setChecked(True)
            self._on_view_mode_changed(3)
        elif event.key() == QtCore.Qt.Key.Key_T:
            self._view_controls.view_buttons.button(4).setChecked(True)
            self._on_view_mode_changed(4)
        elif event.key() == QtCore.Qt.Key.Key_U:
            if self.context_data is not None and self.context_data.mean_image_channel_2 is not None:
                self._view_controls.view_buttons.button(6).setChecked(True)
                self._on_view_mode_changed(6)
        elif event.key() == QtCore.Qt.Key.Key_Y:
            if (
                self.context_data is not None
                and self.context_data.corrected_structural_mean_image is not None
            ):
                self._view_controls.view_buttons.button(5).setChecked(True)
                self._on_view_mode_changed(5)
        elif event.key() == QtCore.Qt.Key.Key_Space:
            self._roi_visibility_checkbox.toggle()
        elif event.key() == QtCore.Qt.Key.Key_N:
            self._trace_controls.deconvolved_checkbox.toggle()
        elif event.key() == QtCore.Qt.Key.Key_B:
            self._trace_controls.neuropil_checkbox.toggle()
        elif event.key() == QtCore.Qt.Key.Key_V:
            self._trace_controls.traces_checkbox.toggle()
        elif event.key() == QtCore.Qt.Key.Key_A:
            self._color_controls.color_buttons.button(0).setChecked(True)
            self._on_color_mode_changed(0)
        elif event.key() == QtCore.Qt.Key.Key_S:
            self._color_controls.color_buttons.button(1).setChecked(True)
            self._on_color_mode_changed(1)
        elif event.key() == QtCore.Qt.Key.Key_D:
            self._color_controls.color_buttons.button(2).setChecked(True)
            self._on_color_mode_changed(2)
        elif event.key() == QtCore.Qt.Key.Key_F:
            self._color_controls.color_buttons.button(3).setChecked(True)
            self._on_color_mode_changed(3)
        elif event.key() == QtCore.Qt.Key.Key_G:
            self._color_controls.color_buttons.button(4).setChecked(True)
            self._on_color_mode_changed(4)
        elif event.key() == QtCore.Qt.Key.Key_H:
            if self.context_data is not None and self.context_data.has_channel_2:
                self._color_controls.color_buttons.button(5).setChecked(True)
                self._on_color_mode_changed(5)
        elif event.key() == QtCore.Qt.Key.Key_J:
            self._color_controls.color_buttons.button(6).setChecked(True)
            self._on_color_mode_changed(6)
        elif event.key() == QtCore.Qt.Key.Key_K:
            self._color_controls.color_buttons.button(7).setChecked(True)
            self._on_color_mode_changed(7)
        elif event.key() == QtCore.Qt.Key.Key_Left:
            if self.context_data is None:
                return
            ctype = self.context_data.cell_classification_labels[self.selected_roi_index]
            roi_count = self.context_data.roi_count
            while True:
                self.selected_roi_index = (self.selected_roi_index - 1) % roi_count
                if self.context_data.cell_classification_labels[self.selected_roi_index] is ctype:
                    break
            self.merge_roi_indices = [self.selected_roi_index]
            self._roi_remove()
            self.update_plot()
        elif event.key() == QtCore.Qt.Key.Key_Right:
            if self.context_data is None:
                return
            self._roi_remove()
            ctype = self.context_data.cell_classification_labels[self.selected_roi_index]
            roi_count = self.context_data.roi_count
            while True:
                self.selected_roi_index = (self.selected_roi_index + 1) % roi_count
                if self.context_data.cell_classification_labels[self.selected_roi_index] is ctype:
                    break
            self.merge_roi_indices = [self.selected_roi_index]
            self.update_plot()
            self.show()
        elif event.key() == QtCore.Qt.Key.Key_Up:
            self._flip_plot()
            self._roi_remove()

    # --- Main plot update ---

    def update_plot(self) -> None:
        """Redraws all plot panels including masks, traces, and colorbar."""
        if self.context_data is None or self.color_arrays is None or self.roi_maps is None:
            return
        if self.views is None or self.colorbar_widgets is None or self.colorbar_image is None:
            return
        if self.roi_color_mode == CONFIG.color_correlation and self.Fbin is not None:
            assert self.Fstd is not None
            overlays.update_correlation_masks(
                color_arrays=self.color_arrays,
                roi_maps=self.roi_maps,
                binned_fluorescence=self.Fbin,
                fluorescence_std=self.Fstd,
                merge_indices=self.merge_roi_indices,
                colormap=self.roi_colormap,
            )
        overlays.render_colorbar(
            roi_color_mode=self.roi_color_mode,
            color_arrays=self.color_arrays,
            colorbar_widgets=self.colorbar_widgets,
            colorbar_image=self.colorbar_image,
        )
        self._ichosen_stats()
        overlays.display_views(
            view1=self._cells_background,
            view2=self._noncells_background,
            views=self.views,
            view_index=self.background_view,
            saturation=self.background_saturation,
        )
        masks = overlays.draw_masks(
            context=self.context_data,
            color_arrays=self.color_arrays,
            roi_maps=self.roi_maps,
            roi_color_mode=self.roi_color_mode,
            background_view=self.background_view,
            roi_opacity=self.roi_opacity,
            selected_roi_index=self.selected_roi_index,
            merge_roi_indices=self.merge_roi_indices,
        )
        overlays.display_masks(
            color1=self._cells_overlay,
            color2=self._noncells_overlay,
            masks=masks,
        )
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
        if self.context_data is not None:
            roi_index = self.selected_roi_index
            cell_count = int(self.context_data.cell_count)
            height = self.context_data.frame_height
            width = self.context_data.frame_width
            session_name = (
                str(self.context_data.output_path)
                if self.context_data.output_path is not None
                else "unknown"
            )
            self._status_bar.showMessage(
                f"Session: {session_name}  |  ROI: {roi_index}  |"
                f"  Cells: {cell_count}  |  Size: {height} x {width}"
            )

    def mode_change(self, i: int | None = None) -> None:
        """Changes the activity mode used for multi-neuron display and correlation.

        Activity modes: 0=F, 1=Fneu, 2=F-0.7*Fneu (default), 3=spks.

        Args:
            i: The activity mode index to switch to. When None, uses the current combo box index.
        """
        if i is None:
            i = self._trace_controls.activity_combo.currentIndex()
        self.trace_mode = TraceMode(i)
        if self.session_loaded and self.context_data is not None:
            self.temporal_bin_size = max(1, int(self._color_controls.bin_edit.text()))
            nb = int(
                np.floor(
                    float(self.context_data.frame_count) / float(self.temporal_bin_size)
                )
            )
            if i == 0:
                f = self.context_data.cell_fluorescence
            elif i == 1:
                f = self.context_data.neuropil_fluorescence
            elif i == CONFIG.activity_mode_subtracted:
                f = (
                    self.context_data.cell_fluorescence
                    - CONFIG.neuropil_coefficient * self.context_data.neuropil_fluorescence
                )
            else:
                f = self.context_data.spikes
            ncells = self.context_data.roi_count
            bin_size = self.temporal_bin_size
            self.Fbin = f[:, : nb * bin_size].reshape((ncells, nb, bin_size)).mean(axis=2)
            self.Fbin -= self.Fbin.mean(axis=1)[:, np.newaxis]
            self.Fstd = (self.Fbin**2).mean(axis=1) ** 0.5
            self.frame_indices = np.arange(0, self.context_data.frame_count, dtype=np.int32)
            self.update_plot()

    def top_number_chosen(self) -> None:
        """Updates the top-N ROI count and refreshes the selection if applicable."""
        self._selection_controls.top_count = int(self._selection_controls.top_count_edit.text())
        if self.session_loaded and not self._cell_toggle_controls.size_buttons.button(1).isChecked():
            for b in [1, 2]:
                if self._selection_controls.selection_buttons.button(b).isChecked():
                    self._on_roi_selection()
                    self.show()

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
            self._active_roi_selection = pg.RectROI(
                [imx, imy], [dx, dy], pen="w", sideScalers=True
            )
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
        """Selects cells whose pixels overlap the given coordinate arrays.

        Args:
            ypix: 2D array of y-pixel coordinates from the ROI region.
            xpix: 2D array of x-pixel coordinates from the ROI region.
        """
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

    def number_chosen(self) -> None:
        """Jumps to the ROI number entered in the ROI edit field."""
        if self.session_loaded and self.context_data is not None:
            self.selected_roi_index = int(self._roi_index_edit.text())
            if self.selected_roi_index >= self.context_data.roi_count:
                self.selected_roi_index = self.context_data.roi_count - 1
            self.merge_roi_indices = [self.selected_roi_index]
            self.update_plot()
            self.show()

    def _toggle_rois(self, state: int) -> None:
        """Toggles ROI overlay visibility on both image panels.

        Args:
            state: Qt checkbox state value.
        """
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

    def _handle_click(
        self,
        click_x: int,
        click_y: int,
        panel_index: int,
        is_right: bool,
        is_multi: bool,
    ) -> bool:
        """Handles mouse clicks on image panels via the ViewBox callback API.

        Left-click chooses a cell. Right-click flips a cell to the other view.
        Shift/ctrl-click adds or removes from the merge selection.

        Args:
            click_x: Column coordinate of the click.
            click_y: Row coordinate of the click.
            panel_index: Panel index (0=cells, 1=non-cells).
            is_right: Determines whether the right mouse button was clicked.
            is_multi: Determines whether shift or ctrl was held during the click.

        Returns:
            True if the click was handled (hit an ROI), False otherwise.
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

        ichosen = int(self.roi_maps.iroi[panel_index, 0, click_y, click_x])
        if ichosen < 0:
            return False

        if is_right:
            if ichosen not in self.merge_roi_indices:
                self.merge_roi_indices = [ichosen]
                self.selected_roi_index = ichosen
            self._flip_plot()
        else:
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

    def _ichosen_stats(self) -> None:
        """Updates the ROI statistics labels for the currently selected cell."""
        if self.context_data is None:
            return
        n = self.selected_roi_index
        self._roi_index_edit.setText(str(n))
        roi = self.context_data.roi_statistics[n]
        for k in range(len(self.stats_to_show)):
            key = self.stats_to_show[k]
            ival = getattr(roi, key, None)
            if ival is None:
                continue
            if k + 1 == CONFIG.centroid_stat_index:
                self._roi_stat_labels[k].setText(f"{key}: [{ival[0]:d}, {ival[1]:d}]")
            elif k + 1 == CONFIG.pixel_count_stat_index:
                self._roi_stat_labels[k].setText(f"{key}: {ival:d}")
            else:
                self._roi_stat_labels[k].setText(f"{key}: {ival:2.2f}")

    def _zoom_to_cell(self) -> None:
        """Zooms both image panels to center on the currently selected cell."""
        if self.context_data is None:
            return
        irange = 0.1 * np.array(
            [self.context_data.frame_height, self.context_data.frame_width]
        ).max()
        roi_statistics = self.context_data.roi_statistics
        if len(self.merge_roi_indices) > 1:
            apix = np.zeros((0, 2))
            for _i, k in enumerate(self.merge_roi_indices):
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

    def _flip_plot(self) -> None:
        """Flips the selected ROIs between the cell and non-cell panels."""
        if self.context_data is None or self.color_arrays is None or self.roi_maps is None:
            return
        last_reclassified = [self.last_reclassified_index]
        overlays.flip_rois(
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

    def _zoom_plot(self, panel: int) -> None:
        """Resets the view range for the specified panel.

        Args:
            panel: Panel index (0=cells, 1=non-cells).
        """
        if panel == CONFIG.cells_plot:
            self._cells_view_box.autoRange()
            self._noncells_view_box.autoRange()
        elif panel == CONFIG.noncells_plot:
            self._noncells_view_box.autoRange()

    def save_cell_classification(self) -> None:
        """Saves the current cell classification labels to disk."""
        context_loader.save_cell_classification(self)

    def save_cell_colocalization(self) -> None:
        """Saves the current cell colocalization labels to disk."""
        context_loader.save_cell_colocalization(self)

    def apply_merge(self) -> None:
        """Applies the pending ROI merge operation."""
        merge_dialog.apply(self)

    @staticmethod
    def cindra_directory() -> Path:
        """Returns the root path of the cindra package directory."""
        return _CINDRA_DIR
