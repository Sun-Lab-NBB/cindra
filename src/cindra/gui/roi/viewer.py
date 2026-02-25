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
    QGroupBox,
    QLineEdit,
    QStatusBar,
    QHBoxLayout,
    QMainWindow,
    QVBoxLayout,
)
from ataraxis_base_utilities import LogLevel, console

import cindra

from . import (
    menu_bar,
    trace_panel,
    merge_dialog,
    plot_widgets,
    roi_overlays,
    context_loader,
    background_views,
    classifier_panel,
    selection_buttons,
)
from .styles import STYLE, label_font, _ROIViewerStyle
from .signals import GUISignals
from .view_state import TraceMode, ViewState, ROIColorMode, ROIToolPanel, BackgroundView

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PySide6.QtGui import QAction, QKeyEvent, QDropEvent, QDragEnterEvent
    from PySide6.QtWidgets import QMenu

    from .context_data import ContextData
    from .roi_overlays import ColorArrays, ROIIndexMaps, ColorbarWidgets

# Path to the root of the cindra package directory.
_CINDRA_DIR: Path = Path(cindra.__file__).parent

# String path to the application icon file.
_ICON_PATH: str = str(_CINDRA_DIR / "logo" / "logo.png")

# Color index for correlation-based coloring mode.
_CORRELATION_COLOR: int = 7

# Stat display index for the centroid stat (formatted as coordinate pair).
_CENTROID_STAT_INDEX: int = 1

# Stat display index for the pixel-count stat (formatted as integer).
_PIXEL_COUNT_STAT_INDEX: int = 2

# Activity mode index for neuropil-corrected fluorescence (F - 0.7 * Fneu).
_NEUROPIL_CORRECTED_MODE: int = 2

# View plot index for the cells image panel.
_CELLS_PLOT: int = 0

# View plot index for the non-cells image panel.
_NONCELLS_PLOT: int = 1

# Epsilon for comparing floating-point threshold values.
_THRESHOLD_EPSILON: float = 1e-3


class MainWindow(QMainWindow):
    """Provides the main application window for the cindra graphical interface."""

    _style: _ROIViewerStyle = _ROIViewerStyle()
    """Frozen style constants for the ROI viewer window."""

    def __init__(self, session_path: Path | None = None) -> None:
        """Initializes the main window, menus, buttons, and graphics panels.

        Args:
            session_path: Optional path to a cindra output directory to load on startup.
        """
        super().__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")
        self._signals = GUISignals()

        # Core state objects.
        self.view_state: ViewState = ViewState()
        self.context_data: ContextData | None = None
        self.color_arrays: ColorArrays | None = None
        self.roi_maps: ROIIndexMaps | None = None
        self.colorbar_widgets: ColorbarWidgets | None = None
        self.colorbar_image: NDArray[np.uint8] | None = None
        self.views: NDArray[np.uint8] | None = None

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

        # Build graphics panels.
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

        # Wire signal bus.
        self._signals.view_mode_changed.connect(self._on_view_mode_changed)
        self._signals.color_mode_changed.connect(self._on_color_mode_changed)
        self._signals.activity_mode_changed.connect(self.mode_change)
        self._signals.plot_needs_update.connect(self.update_plot)
        self._signals.trace_needs_update.connect(self._on_trace_update)
        self._signals.roi_selection_changed.connect(self._on_roi_selection)

        # Apply NoFocus policy to all buttons in the control panel.
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
        selection_box, self._selection_controls = selection_buttons.create_selection_buttons(
            owner=self,
            signals=self._signals,
        )
        layout.addWidget(selection_box)

        # 3. View Toggle.
        toggle_box, self._cell_toggle_controls = selection_buttons.create_cell_toggle_buttons(
            owner=self,
            signals=self._signals,
        )
        layout.addWidget(toggle_box)

        # 4. Background.
        background_box, self._view_controls = background_views.create_view_controls(
            owner=self,
            signals=self._signals,
        )
        layout.addWidget(background_box)

        # 5. ROI Colors + colorbar.
        colors_box, self._color_controls = roi_overlays.build_color_controls(
            owner=self,
            signals=self._signals,
        )
        self.colorbar_widgets = roi_overlays.build_colorbar(owner=self)
        # Add the colorbar widget into the colors group box layout.
        colors_layout = colors_box.layout()
        assert colors_layout is not None
        colors_layout.addWidget(self.colorbar_widgets.widget)
        layout.addWidget(colors_box)

        # 6. Classifier.
        classifier_box, self._classifier_controls = classifier_panel.create_classifier_controls(
            owner=self,
            signals=self._signals,
        )
        self._classifier_controls.add_to_class_button.clicked.connect(lambda: classifier_panel._add_to(self))
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
        lilfont = label_font()
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
            stat_label.setFont(lilfont)
            stat_label.setStyleSheet(STYLE.white_label)
            stat_label.resize(stat_label.minimumSizeHint())
            roi_layout.addWidget(stat_label)
            self._roi_stat_labels.append(stat_label)
        layout.addWidget(roi_box)

        # 8. Trace Display.
        trace_box, self._trace_controls = trace_panel.create_trace_controls(
            owner=self,
            signals=self._signals,
        )
        # Zoom to cell checkbox.
        self._zoom_to_cell_checkbox = QCheckBox("zoom to cell")
        self._zoom_to_cell_checkbox.setStyleSheet(STYLE.white_label)
        self._zoom_to_cell_checkbox.stateChanged.connect(self._zoom_cell)
        trace_layout = trace_box.layout()
        assert trace_layout is not None
        trace_layout.addWidget(self._zoom_to_cell_checkbox)
        layout.addWidget(trace_box)

        # 9. Navigation.
        nav_box, self._quadrant_controls = selection_buttons.create_quadrant_buttons(
            owner=self,
            signals=self._signals,
        )
        layout.addWidget(nav_box)

        layout.addStretch()
        return panel

    def _build_graphics(self) -> None:
        """Creates the main plotting area with cells, non-cells, and trace panels."""
        # Cells image panel.
        self._cells_view_box = plot_widgets.ViewBox(
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
        self._noncells_view_box = plot_widgets.ViewBox(
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

        # Installs click and zoom handlers via the modernized ViewBox callback API.
        self._cells_view_box.set_click_handler(self._handle_click)
        self._noncells_view_box.set_click_handler(self._handle_click)
        self._cells_view_box.set_zoom_handler(lambda: self._zoom_plot(_CELLS_PLOT))
        self._noncells_view_box.set_zoom_handler(lambda: self._zoom_plot(_NONCELLS_PLOT))

        # Fluorescence trace plot.
        self._trace_box = plot_widgets.TraceBox()
        self._trace_box.setMouseEnabled(x=True, y=False)
        self._trace_box.enableAutoRange(x=True, y=True)
        self._graphics_widget.addItem(self._trace_box, row=1, col=0, colspan=2)
        self._graphics_widget.ci.layout.setRowStretchFactor(0, 2)
        layout = self._graphics_widget.ci.layout
        layout.setColumnMinimumWidth(0, 1)
        layout.setColumnMinimumWidth(1, 1)
        layout.setHorizontalSpacing(20)

    def _on_view_mode_changed(self, index: int) -> None:
        """Handles background view mode changes from the signal bus.

        Args:
            index: The background view index selected.
        """
        self.view_state.background_view = BackgroundView(index)
        self.update_plot()

    def _on_color_mode_changed(self, index: int) -> None:
        """Handles ROI color mode changes from the signal bus.

        Args:
            index: The color mode index selected.
        """
        self.view_state.roi_color_mode = ROIColorMode(index)
        if self.context_data is not None and self.color_arrays is not None and self.roi_maps is not None:
            colormap = self._color_controls.colormap_chooser.currentText()
            if colormap != self.view_state.roi_colormap:
                self.view_state.roi_colormap = colormap
                self.colorbar_image = roi_overlays.update_colormap(
                    color_arrays=self.color_arrays,
                    roi_maps=self.roi_maps,
                    colormap=colormap,
                )
            if (
                self.context_data.has_channel_2
                and abs(float(self._color_controls.channel_2_edit.text()) - self.view_state.colocalization_threshold)
                > _THRESHOLD_EPSILON
            ):
                self.view_state.colocalization_threshold = float(self._color_controls.channel_2_edit.text())
                roi_overlays.update_chan2_colors(
                    context=self.context_data,
                    state=self.view_state,
                    color_arrays=self.color_arrays,
                    roi_maps=self.roi_maps,
                )
        self.update_plot()

    def _on_trace_update(self) -> None:
        """Handles trace-only update requests from the signal bus."""
        if self.context_data is None or self.color_arrays is None or self.frame_indices is None:
            return
        trace_panel.plot_trace(
            trace_box=self._trace_box,
            cell_fluorescence=self.context_data.cell_fluorescence,
            neuropil_fluorescence=self.context_data.neuropil_fluorescence,
            spikes=self.context_data.spikes,
            frame_indices=self.frame_indices,
            merge_indices=self.view_state.merge_roi_indices,
            activity_mode=self.view_state.trace_mode,
            roi_colors=self.color_arrays.cols[self.view_state.roi_color_mode],
            traces_visible=self._trace_controls.traces_visible,
            neuropil_visible=self._trace_controls.neuropil_visible,
            deconvolved_visible=self._trace_controls.deconvolved_visible,
            scale_factor=self._trace_controls.scale_factor,
            max_plotted=int(self._trace_controls.max_plotted_edit.text() or "40"),
        )

    def _on_roi_selection(self) -> None:
        """Handles ROI selection changes from the signal bus."""
        self._roi_selection()

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
            self.view_state.roi_labels_visible = True
        else:
            for n in range(len(self.roi_maps.text_labels)):
                if self.context_data.cell_classification_labels[n] == 1:
                    with suppress(Exception):
                        self._cells_view_box.removeItem(self.roi_maps.text_labels[n])
                else:
                    with suppress(Exception):
                        self._noncells_view_box.removeItem(self.roi_maps.text_labels[n])
            self.view_state.roi_labels_visible = False

    def _zoom_cell(self, state: int) -> None:
        """Toggles zoom-to-cell behavior based on checkbox state.

        Args:
            state: Qt checkbox state value.
        """
        if not self.view_state.session_loaded:
            return
        self.view_state.auto_zoom_to_roi = QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked
        self.update_plot()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        """Handles keyboard shortcuts for view switching, ROI navigation, and toggles.

        Args:
            event: The key press event from Qt.
        """
        if not self.view_state.session_loaded:
            return
        if event.modifiers() in {QtCore.Qt.KeyboardModifier.ControlModifier, QtCore.Qt.KeyboardModifier.ShiftModifier}:
            return
        if event.key() == QtCore.Qt.Key.Key_Return:
            if (
                event.modifiers() == QtCore.Qt.KeyboardModifier.AltModifier
                and len(self.view_state.merge_roi_indices) > 1
            ):
                merge_dialog.do_merge(self)
        elif event.key() == QtCore.Qt.Key.Key_Escape:
            self._zoom_plot(_CELLS_PLOT)
            self._trace_box.autoRange()
            self.show()
        elif event.key() == QtCore.Qt.Key.Key_Delete:
            self._roi_remove()
        elif event.key() == QtCore.Qt.Key.Key_Q:
            self._view_controls.view_buttons.button(0).setChecked(True)
            self._signals.view_mode_changed.emit(0)
        elif event.key() == QtCore.Qt.Key.Key_W:
            self._view_controls.view_buttons.button(1).setChecked(True)
            self._signals.view_mode_changed.emit(1)
        elif event.key() == QtCore.Qt.Key.Key_E:
            self._view_controls.view_buttons.button(2).setChecked(True)
            self._signals.view_mode_changed.emit(2)
        elif event.key() == QtCore.Qt.Key.Key_R:
            self._view_controls.view_buttons.button(3).setChecked(True)
            self._signals.view_mode_changed.emit(3)
        elif event.key() == QtCore.Qt.Key.Key_T:
            self._view_controls.view_buttons.button(4).setChecked(True)
            self._signals.view_mode_changed.emit(4)
        elif event.key() == QtCore.Qt.Key.Key_U:
            if self.context_data is not None and self.context_data.mean_image_channel_2 is not None:
                self._view_controls.view_buttons.button(6).setChecked(True)
                self._signals.view_mode_changed.emit(6)
        elif event.key() == QtCore.Qt.Key.Key_Y:
            if self.context_data is not None and self.context_data.corrected_structural_mean_image is not None:
                self._view_controls.view_buttons.button(5).setChecked(True)
                self._signals.view_mode_changed.emit(5)
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
            self._signals.color_mode_changed.emit(0)
        elif event.key() == QtCore.Qt.Key.Key_S:
            self._color_controls.color_buttons.button(1).setChecked(True)
            self._signals.color_mode_changed.emit(1)
        elif event.key() == QtCore.Qt.Key.Key_D:
            self._color_controls.color_buttons.button(2).setChecked(True)
            self._signals.color_mode_changed.emit(2)
        elif event.key() == QtCore.Qt.Key.Key_F:
            self._color_controls.color_buttons.button(3).setChecked(True)
            self._signals.color_mode_changed.emit(3)
        elif event.key() == QtCore.Qt.Key.Key_G:
            self._color_controls.color_buttons.button(4).setChecked(True)
            self._signals.color_mode_changed.emit(4)
        elif event.key() == QtCore.Qt.Key.Key_H:
            if self.context_data is not None and self.context_data.has_channel_2:
                self._color_controls.color_buttons.button(5).setChecked(True)
                self._signals.color_mode_changed.emit(5)
        elif event.key() == QtCore.Qt.Key.Key_J:
            self._color_controls.color_buttons.button(6).setChecked(True)
            self._signals.color_mode_changed.emit(6)
        elif event.key() == QtCore.Qt.Key.Key_K:
            self._color_controls.color_buttons.button(7).setChecked(True)
            self._signals.color_mode_changed.emit(7)
        elif event.key() == QtCore.Qt.Key.Key_Left:
            if self.context_data is None:
                return
            ctype = self.context_data.cell_classification_labels[self.view_state.selected_roi_index]
            roi_count = self.context_data.roi_count
            while True:
                self.view_state.selected_roi_index = (self.view_state.selected_roi_index - 1) % roi_count
                if self.context_data.cell_classification_labels[self.view_state.selected_roi_index] is ctype:
                    break
            self.view_state.merge_roi_indices = [self.view_state.selected_roi_index]
            self._roi_remove()
            self.update_plot()
        elif event.key() == QtCore.Qt.Key.Key_Right:
            if self.context_data is None:
                return
            self._roi_remove()
            ctype = self.context_data.cell_classification_labels[self.view_state.selected_roi_index]
            roi_count = self.context_data.roi_count
            while True:
                self.view_state.selected_roi_index = (self.view_state.selected_roi_index + 1) % roi_count
                if self.context_data.cell_classification_labels[self.view_state.selected_roi_index] is ctype:
                    break
            self.view_state.merge_roi_indices = [self.view_state.selected_roi_index]
            self.update_plot()
            self.show()
        elif event.key() == QtCore.Qt.Key.Key_Up:
            self._flip_plot()
            self._roi_remove()

    def update_plot(self) -> None:
        """Redraws all plot panels including masks, traces, and colorbar."""
        if self.context_data is None or self.color_arrays is None or self.roi_maps is None:
            return
        if self.views is None or self.colorbar_widgets is None or self.colorbar_image is None:
            return
        if self.view_state.roi_color_mode == _CORRELATION_COLOR and self.Fbin is not None:
            assert self.Fstd is not None
            roi_overlays.update_correlation_masks(
                color_arrays=self.color_arrays,
                roi_maps=self.roi_maps,
                binned_fluorescence=self.Fbin,
                fluorescence_std=self.Fstd,
                merge_indices=self.view_state.merge_roi_indices,
                colormap=self.view_state.roi_colormap,
            )
        roi_overlays.render_colorbar(
            state=self.view_state,
            color_arrays=self.color_arrays,
            colorbar_widgets=self.colorbar_widgets,
            colorbar_image=self.colorbar_image,
        )
        self._ichosen_stats()
        background_views.display_views(
            view1=self._cells_background,
            view2=self._noncells_background,
            views=self.views,
            view_index=self.view_state.background_view,
            saturation=self.view_state.background_saturation,
        )
        masks = roi_overlays.draw_masks(
            context=self.context_data,
            state=self.view_state,
            color_arrays=self.color_arrays,
            roi_maps=self.roi_maps,
        )
        roi_overlays.display_masks(
            color1=self._cells_overlay,
            color2=self._noncells_overlay,
            masks=masks,
        )
        assert self.frame_indices is not None
        trace_panel.plot_trace(
            trace_box=self._trace_box,
            cell_fluorescence=self.context_data.cell_fluorescence,
            neuropil_fluorescence=self.context_data.neuropil_fluorescence,
            spikes=self.context_data.spikes,
            frame_indices=self.frame_indices,
            merge_indices=self.view_state.merge_roi_indices,
            activity_mode=self.view_state.trace_mode,
            roi_colors=self.color_arrays.cols[self.view_state.roi_color_mode],
            traces_visible=self._trace_controls.traces_visible,
            neuropil_visible=self._trace_controls.neuropil_visible,
            deconvolved_visible=self._trace_controls.deconvolved_visible,
            scale_factor=self._trace_controls.scale_factor,
            max_plotted=int(self._trace_controls.max_plotted_edit.text() or "40"),
        )
        if self.view_state.auto_zoom_to_roi:
            self._zoom_to_cell()
        self._cells_view_box.show()
        self._noncells_view_box.show()
        self._graphics_widget.show()
        self.show()

        # Update status bar.
        if self.context_data is not None:
            roi_index = self.view_state.selected_roi_index
            cell_count = int(self.context_data.cell_count)
            height = self.context_data.frame_height
            width = self.context_data.frame_width
            session_name = str(self.context_data.save_path) if self.context_data.save_path is not None else "unknown"
            self._status_bar.showMessage(
                f"Session: {session_name}  |  ROI: {roi_index}  |  Cells: {cell_count}  |  Size: {height} x {width}"
            )

    def mode_change(self, i: int) -> None:
        """Changes the activity mode used for multi-neuron display and correlation.

        Activity modes: 0=F, 1=Fneu, 2=F-0.7*Fneu (default), 3=spks.

        Args:
            i: The activity mode index to switch to.
        """
        self.view_state.trace_mode = TraceMode(i)
        if self.view_state.session_loaded and self.context_data is not None:
            self.view_state.temporal_bin_size = max(1, int(self._color_controls.bin_edit.text()))
            nb = int(np.floor(float(self.context_data.frame_count) / float(self.view_state.temporal_bin_size)))
            if i == 0:
                f = self.context_data.cell_fluorescence
            elif i == 1:
                f = self.context_data.neuropil_fluorescence
            elif i == _NEUROPIL_CORRECTED_MODE:
                f = self.context_data.cell_fluorescence - 0.7 * self.context_data.neuropil_fluorescence
            else:
                f = self.context_data.spikes
            ncells = self.context_data.roi_count
            bin_size = self.view_state.temporal_bin_size
            self.Fbin = f[:, : nb * bin_size].reshape((ncells, nb, bin_size)).mean(axis=2)
            self.Fbin -= self.Fbin.mean(axis=1)[:, np.newaxis]
            self.Fstd = (self.Fbin**2).mean(axis=1) ** 0.5
            self.frame_indices = np.arange(0, self.context_data.frame_count, dtype=np.int32)
            self.update_plot()

    def top_number_chosen(self) -> None:
        """Updates the top-N ROI count and refreshes the selection if applicable."""
        self._selection_controls.top_count = int(self._selection_controls.top_count_edit.text())
        if self.view_state.session_loaded and not self._cell_toggle_controls.size_buttons.button(1).isChecked():
            for b in [1, 2]:
                if self._selection_controls.selection_buttons.button(b).isChecked():
                    self._signals.roi_selection_changed.emit()
                    self.show()

    def _roi_selection(self) -> None:
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
            self.view_state.roi_tool_panel = ROIToolPanel(wplot)
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
            self.view_state.roi_tool_active = True

    def _roi_remove(self) -> None:
        """Removes the current rectangular ROI selection and resets button styles."""
        if self.view_state.roi_tool_active:
            if self.view_state.roi_tool_panel == 0:
                self._cells_view_box.removeItem(self._active_roi_selection)
            else:
                self._noncells_view_box.removeItem(self._active_roi_selection)
            self.view_state.roi_tool_active = False
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
            if self.view_state.roi_tool_panel == 0
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
        i = self.view_state.roi_tool_panel
        roi_indices = self.roi_maps.iroi[i, 0, ypix, xpix]
        icells = np.unique(roi_indices[roi_indices >= 0])
        self.view_state.merge_roi_indices = []
        for n in icells:
            pixel_count = self.context_data.roi_statistics[n].pixel_count
            if (self.roi_maps.iroi[i, :, ypix, xpix] == n).sum() > 0.6 * pixel_count:
                self.view_state.merge_roi_indices.append(n)
        if len(self.view_state.merge_roi_indices) > 0:
            self.view_state.selected_roi_index = self.view_state.merge_roi_indices[0]
            self.update_plot()
            self.show()

    def number_chosen(self) -> None:
        """Jumps to the ROI number entered in the ROI edit field."""
        if self.view_state.session_loaded and self.context_data is not None:
            self.view_state.selected_roi_index = int(self._roi_index_edit.text())
            if self.view_state.selected_roi_index >= self.context_data.roi_count:
                self.view_state.selected_roi_index = self.context_data.roi_count - 1
            self.view_state.merge_roi_indices = [self.view_state.selected_roi_index]
            self.update_plot()
            self.show()

    def _toggle_rois(self, state: int) -> None:
        """Toggles ROI overlay visibility on both image panels.

        Args:
            state: Qt checkbox state value.
        """
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.view_state.rois_visible = True
            self._cells_view_box.addItem(self._cells_overlay)
            self._noncells_view_box.addItem(self._noncells_overlay)
        else:
            self.view_state.rois_visible = False
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
        if not self.view_state.session_loaded or self.roi_maps is None or self.context_data is None:
            return False

        # Bounds-checks the click coordinates.
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
            if ichosen not in self.view_state.merge_roi_indices:
                self.view_state.merge_roi_indices = [ichosen]
                self.view_state.selected_roi_index = ichosen
            self._flip_plot()
        else:
            merged = False
            if is_multi and (
                self.context_data.cell_classification_labels[self.view_state.merge_roi_indices[0]]
                == self.context_data.cell_classification_labels[ichosen]
            ):
                if ichosen not in self.view_state.merge_roi_indices:
                    self.view_state.merge_roi_indices.append(ichosen)
                    self.view_state.selected_roi_index = ichosen
                    merged = True
                elif len(self.view_state.merge_roi_indices) > 1:
                    self.view_state.merge_roi_indices.remove(ichosen)
                    self.view_state.selected_roi_index = self.view_state.merge_roi_indices[0]
                    merged = True
            if not merged:
                self.view_state.merge_roi_indices = [ichosen]
                self.view_state.selected_roi_index = ichosen

        if self.view_state.roi_tool_active:
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
        n = self.view_state.selected_roi_index
        self._roi_index_edit.setText(str(n))
        roi = self.context_data.roi_statistics[n]
        for k in range(len(self.stats_to_show)):
            key = self.stats_to_show[k]
            ival = getattr(roi, key, None)
            if ival is None:
                continue
            if k + 1 == _CENTROID_STAT_INDEX:
                self._roi_stat_labels[k].setText(f"{key}: [{ival[0]:d}, {ival[1]:d}]")
            elif k + 1 == _PIXEL_COUNT_STAT_INDEX:
                self._roi_stat_labels[k].setText(f"{key}: {ival:d}")
            else:
                self._roi_stat_labels[k].setText(f"{key}: {ival:2.2f}")

    def _zoom_to_cell(self) -> None:
        """Zooms both image panels to center on the currently selected cell."""
        if self.context_data is None:
            return
        irange = 0.1 * np.array([self.context_data.frame_height, self.context_data.frame_width]).max()
        roi_statistics = self.context_data.roi_statistics
        if len(self.view_state.merge_roi_indices) > 1:
            apix = np.zeros((0, 2))
            for _i, k in enumerate(self.view_state.merge_roi_indices):
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
            icent = np.array(roi_statistics[self.view_state.selected_roi_index].centroid)
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
        roi_overlays.flip_rois(
            context=self.context_data,
            state=self.view_state,
            color_arrays=self.color_arrays,
            roi_maps=self.roi_maps,
        )
        self.save_cell_classification()
        self.update_plot()

    def _zoom_plot(self, panel: int) -> None:
        """Resets the view range for the specified panel.

        Args:
            panel: Panel index (0=cells, 1=non-cells).
        """
        if panel == _CELLS_PLOT:
            self._cells_view_box.autoRange()
            self._noncells_view_box.autoRange()
        elif panel == _NONCELLS_PLOT:
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
