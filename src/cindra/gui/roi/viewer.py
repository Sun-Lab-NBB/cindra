"""Provides the main application window for the ROI viewer and editor GUI."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING
from pathlib import Path
from contextlib import suppress

import numpy as np
from PySide6 import QtGui, QtCore
import pyqtgraph as pg
from PySide6.QtWidgets import (
    QLabel,
    QWidget,
    QCheckBox,
    QLineEdit,
    QGridLayout,
    QMainWindow,
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
from ..styles import (
    MAIN_WINDOW_STYLESHEET,
    WHITE_LABEL_STYLESHEET,
    BUTTON_PRESSED_STYLESHEET,
    BUTTON_INACTIVE_STYLESHEET,
    BUTTON_UNPRESSED_STYLESHEET,
    label_font,
    header_font,
)
from .signals import GUISignals
from .view_state import ViewState, ROIToolPanel

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PySide6.QtGui import QKeyEvent, QDropEvent, QDragEnterEvent

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


class MainWindow(QMainWindow):
    """Provides the main application window for the cindra graphical interface."""

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
        self.setStyleSheet(MAIN_WINDOW_STYLESHEET)

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

        self.boldfont = header_font()

        # Main widget layout.
        cwidget = QWidget()
        self.l0 = QGridLayout()
        cwidget.setLayout(self.l0)
        self.setCentralWidget(cwidget)

        b0 = self.make_buttons()
        self.make_graphics(b0)
        # Draws quadrant buttons last so they appear on top of the plot.
        self._quadrant_controls = selection_buttons.create_quadrant_buttons(
            owner=self,
            layout=self.l0,
            signals=self._signals,
        )
        self.quadbtns = self._quadrant_controls.quadrant_buttons

        # Initializes merge tracking.
        self.merged: list[object] = []

        if session_path is not None:
            context_loader.load_session(parent=self, session_path=session_path)
        self.setAcceptDrops(True)
        self.show()
        self.win.show()

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

    def make_buttons(self, b0: int = 0) -> int:
        """Creates all sidebar control buttons and ROI stat labels.

        Args:
            b0: Starting row index in the grid layout.

        Returns:
            The next available row index after the last widget placed.
        """
        # ROI checkbox.
        self.l0.setVerticalSpacing(4)
        self.checkBox = QCheckBox("ROIs On [space bar]")
        self.checkBox.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.checkBox.toggle()
        self.checkBox.stateChanged.connect(self._toggle_rois)
        self.l0.addWidget(self.checkBox, 0, 0, 1, 2)

        self._selection_controls = selection_buttons.create_selection_buttons(
            owner=self,
            layout=self.l0,
            signals=self._signals,
        )
        self._cell_toggle_controls = selection_buttons.create_cell_toggle_buttons(
            owner=self,
            layout=self.l0,
            signals=self._signals,
        )
        # Backward-compat aliases for button groups used by context_loader and other modules.
        self.topbtns = self._selection_controls.selection_buttons
        self.topedit = self._selection_controls.top_count_edit
        self.ntop = self._selection_controls.top_count
        self.sizebtns = self._cell_toggle_controls.size_buttons
        self.lcell0 = self._cell_toggle_controls.cell_count_label
        self.lcell1 = self._cell_toggle_controls.noncell_count_label

        self._view_controls, b0 = background_views.create_view_controls(
            owner=self,
            layout=self.l0,
            row=1,
            signals=self._signals,
        )
        self.viewbtns = self._view_controls.view_buttons
        self.view_names = self._view_controls.view_names

        self._color_controls, b0 = roi_overlays.build_color_controls(
            owner=self,
            layout=self.l0,
            row=b0,
            signals=self._signals,
        )
        self.colorbtns = self._color_controls.color_buttons
        self.binedit = self._color_controls.bin_edit
        self.chan2edit = self._color_controls.channel_2_edit
        self.probedit = self._color_controls.classifier_edit

        self.colorbar_widgets = roi_overlays.build_colorbar(
            owner=self,
            layout=self.l0,
            row=b0,
        )
        b0 += 1
        b0 = classifier_panel.make_buttons(self, b0)
        b0 += 1

        # Cell stats / ROI selection.
        self.stats_to_show = [
            "centroid",
            "pixel_count",
            "skewness",
            "compactness",
            "footprint",
            "aspect_ratio",
        ]
        lilfont = label_font()
        qlabel = QLabel(self)
        qlabel.setFont(self.boldfont)
        qlabel.setText("<font color='white'>Selected ROI:</font>")
        self.l0.addWidget(qlabel, b0, 0, 1, 1)
        self.ROIedit = QLineEdit(self)
        self.ROIedit.setValidator(QtGui.QIntValidator(0, 10000))
        self.ROIedit.setText("0")
        self.ROIedit.setFixedWidth(45)
        self.ROIedit.setAlignment(QtCore.Qt.AlignRight)
        self.ROIedit.returnPressed.connect(self.number_chosen)
        self.l0.addWidget(self.ROIedit, b0, 1, 1, 1)
        b0 += 1
        self.ROIstats: list[QLabel] = []
        self.ROIstats.append(qlabel)
        for k in range(1, len(self.stats_to_show) + 1):
            llabel = QLabel(self.stats_to_show[k - 1])
            self.ROIstats.append(llabel)
            self.ROIstats[k].setFont(lilfont)
            self.ROIstats[k].setStyleSheet(WHITE_LABEL_STYLESHEET)
            self.ROIstats[k].resize(self.ROIstats[k].minimumSizeHint())
            self.l0.addWidget(self.ROIstats[k], b0, 0, 1, 2)
            b0 += 1
        self.l0.addWidget(QLabel(""), b0, 0, 1, 2)
        self.l0.setRowStretch(b0, 1)
        b0 += 2
        self._trace_controls, b0 = trace_panel.create_trace_controls(
            owner=self,
            layout=self.l0,
            row=b0,
            signals=self._signals,
        )
        # Backward-compat aliases for modules that reference parent.comboBox, checkboxes, etc.
        self.comboBox = self._trace_controls.activity_combo
        self.checkBoxd = self._trace_controls.deconvolved_checkbox
        self.checkBoxn = self._trace_controls.neuropil_checkbox
        self.checkBoxt = self._trace_controls.traces_checkbox
        self.ncedit = self._trace_controls.max_plotted_edit

        # Zoom to cell checkbox.
        self.l0.setVerticalSpacing(4)
        self.checkBoxz = QCheckBox("zoom to cell")
        self.checkBoxz.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.checkBoxz.stateChanged.connect(self._zoom_cell)
        self.l0.addWidget(self.checkBoxz, b0, 15, 1, 2)

        self.checkBoxN = QCheckBox("add ROI # to plot")
        self.checkBoxN.setStyleSheet(WHITE_LABEL_STYLESHEET)
        self.checkBoxN.stateChanged.connect(self._roi_text)
        self.checkBoxN.setEnabled(False)
        self.l0.addWidget(self.checkBoxN, b0, 18, 1, 2)

        return b0

    def _roi_text(self, state: int) -> None:
        """Toggles ROI number text labels on the image panels.

        Args:
            state: Qt checkbox state value.
        """
        if self.roi_maps is None or self.context_data is None:
            return
        if QtCore.Qt.CheckState(state) == QtCore.Qt.Checked:
            for n in range(len(self.roi_maps.text_labels)):
                if self.context_data.cell_classification_labels[n] == 1:
                    self.p1.addItem(self.roi_maps.text_labels[n])
                else:
                    self.p2.addItem(self.roi_maps.text_labels[n])
            self.view_state.roi_labels_visible = True
        else:
            for n in range(len(self.roi_maps.text_labels)):
                if self.context_data.cell_classification_labels[n] == 1:
                    with suppress(Exception):
                        self.p1.removeItem(self.roi_maps.text_labels[n])
                else:
                    with suppress(Exception):
                        self.p2.removeItem(self.roi_maps.text_labels[n])
            self.view_state.roi_labels_visible = False

    def _zoom_cell(self, state: int) -> None:
        """Toggles zoom-to-cell behavior based on checkbox state.

        Args:
            state: Qt checkbox state value.
        """
        if not self.view_state.session_loaded:
            return
        self.view_state.auto_zoom_to_roi = QtCore.Qt.CheckState(state) == QtCore.Qt.Checked
        self.update_plot()

    def make_graphics(self, b0: int) -> None:
        """Creates the main plotting area with cells, non-cells, and trace panels.

        Args:
            b0: Row span for the graphics widget in the grid layout.
        """
        self.win = pg.GraphicsLayoutWidget()
        self.win.move(600, 0)
        self.win.resize(1000, 500)
        self.l0.addWidget(self.win, 1, 2, b0 - 1, 30)
        # Cells image panel.
        self.p1 = plot_widgets.ViewBox(
            panel=ROIToolPanel.CELLS,
            name="plot1",
            border=[100, 100, 100],
            invert_y=True,
        )
        self.win.addItem(self.p1, 0, 0)
        self.p1.setMenuEnabled(False)
        self.p1.scene().contextMenuItem = self.p1
        self.view1 = pg.ImageItem(viewbox=self.p1, parent=self)
        self.view1.autoDownsample = False
        self.color1 = pg.ImageItem(viewbox=self.p1, parent=self)
        self.color1.autoDownsample = False
        self.p1.addItem(self.view1)
        self.p1.addItem(self.color1)
        self.view1.setLevels([0, 255])
        self.color1.setLevels([0, 255])
        # Non-cells image panel.
        self.p2 = plot_widgets.ViewBox(
            panel=ROIToolPanel.NON_CELLS,
            name="plot2",
            border=[100, 100, 100],
            invert_y=True,
        )
        self.win.addItem(self.p2, 0, 1)
        self.p2.setMenuEnabled(False)
        self.p2.scene().contextMenuItem = self.p2
        self.view2 = pg.ImageItem(viewbox=self.p1, parent=self)
        self.view2.autoDownsample = False
        self.color2 = pg.ImageItem(viewbox=self.p1, parent=self)
        self.color2.autoDownsample = False
        self.p2.addItem(self.view2)
        self.p2.addItem(self.color2)
        self.view2.setLevels([0, 255])
        self.color2.setLevels([0, 255])

        # Links the two view panels.
        self.p2.setXLink("plot1")
        self.p2.setYLink("plot1")

        # Installs click and zoom handlers via the modernized ViewBox callback API.
        self.p1.set_click_handler(self._handle_click)
        self.p2.set_click_handler(self._handle_click)
        self.p1.set_zoom_handler(lambda: self._zoom_plot(_CELLS_PLOT))
        self.p2.set_zoom_handler(lambda: self._zoom_plot(_NONCELLS_PLOT))

        # Fluorescence trace plot.
        self.p3 = plot_widgets.TraceBox()
        self.p3.setMouseEnabled(x=True, y=False)
        self.p3.enableAutoRange(x=True, y=True)
        self.win.addItem(self.p3, row=1, col=0, colspan=2)
        self.win.ci.layout.setRowStretchFactor(0, 2)
        layout = self.win.ci.layout
        layout.setColumnMinimumWidth(0, 1)
        layout.setColumnMinimumWidth(1, 1)
        layout.setHorizontalSpacing(20)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        """Handles keyboard shortcuts for view switching, ROI navigation, and toggles.

        Args:
            event: The key press event from Qt.
        """
        if not self.view_state.session_loaded:
            return
        if event.modifiers() in {QtCore.Qt.ControlModifier, QtCore.Qt.ShiftModifier}:
            return
        if event.key() == QtCore.Qt.Key_Return:
            if event.modifiers() == QtCore.Qt.AltModifier and len(self.view_state.merge_roi_indices) > 1:
                merge_dialog.do_merge(self)
        elif event.key() == QtCore.Qt.Key_Escape:
            self._zoom_plot(_CELLS_PLOT)
            self.p3.autoRange()
            self.show()
        elif event.key() == QtCore.Qt.Key_Delete:
            self._roi_remove()
        elif event.key() == QtCore.Qt.Key_Q:
            self.viewbtns.button(0).setChecked(True)
            self._signals.view_mode_changed.emit(0)
        elif event.key() == QtCore.Qt.Key_W:
            self.viewbtns.button(1).setChecked(True)
            self._signals.view_mode_changed.emit(1)
        elif event.key() == QtCore.Qt.Key_E:
            self.viewbtns.button(2).setChecked(True)
            self._signals.view_mode_changed.emit(2)
        elif event.key() == QtCore.Qt.Key_R:
            self.viewbtns.button(3).setChecked(True)
            self._signals.view_mode_changed.emit(3)
        elif event.key() == QtCore.Qt.Key_T:
            self.viewbtns.button(4).setChecked(True)
            self._signals.view_mode_changed.emit(4)
        elif event.key() == QtCore.Qt.Key_U:
            if self.context_data is not None and self.context_data.mean_image_channel_2 is not None:
                self.viewbtns.button(6).setChecked(True)
                self._signals.view_mode_changed.emit(6)
        elif event.key() == QtCore.Qt.Key_Y:
            if self.context_data is not None and self.context_data.corrected_structural_mean_image is not None:
                self.viewbtns.button(5).setChecked(True)
                self._signals.view_mode_changed.emit(5)
        elif event.key() == QtCore.Qt.Key_Space:
            self.checkBox.toggle()
        elif event.key() == QtCore.Qt.Key_N:
            self.checkBoxd.toggle()
        elif event.key() == QtCore.Qt.Key_B:
            self.checkBoxn.toggle()
        elif event.key() == QtCore.Qt.Key_V:
            self.checkBoxt.toggle()
        elif event.key() == QtCore.Qt.Key_A:
            self.colorbtns.button(0).setChecked(True)
            self._signals.color_mode_changed.emit(0)
        elif event.key() == QtCore.Qt.Key_S:
            self.colorbtns.button(1).setChecked(True)
            self._signals.color_mode_changed.emit(1)
        elif event.key() == QtCore.Qt.Key_D:
            self.colorbtns.button(2).setChecked(True)
            self._signals.color_mode_changed.emit(2)
        elif event.key() == QtCore.Qt.Key_F:
            self.colorbtns.button(3).setChecked(True)
            self._signals.color_mode_changed.emit(3)
        elif event.key() == QtCore.Qt.Key_G:
            self.colorbtns.button(4).setChecked(True)
            self._signals.color_mode_changed.emit(4)
        elif event.key() == QtCore.Qt.Key_H:
            if self.context_data is not None and self.context_data.has_channel_2:
                self.colorbtns.button(5).setChecked(True)
                self._signals.color_mode_changed.emit(5)
        elif event.key() == QtCore.Qt.Key_J:
            self.colorbtns.button(6).setChecked(True)
            self._signals.color_mode_changed.emit(6)
        elif event.key() == QtCore.Qt.Key_K:
            self.colorbtns.button(7).setChecked(True)
            self._signals.color_mode_changed.emit(7)
        elif event.key() == QtCore.Qt.Key_Left:
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
        elif event.key() == QtCore.Qt.Key_Right:
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
        elif event.key() == QtCore.Qt.Key_Up:
            self._flip_plot()
            self._roi_remove()

    def update_plot(self) -> None:
        """Redraws all plot panels including masks, traces, and colorbar."""
        if self.context_data is None or self.color_arrays is None or self.roi_maps is None:
            return
        if self.views is None or self.colorbar_widgets is None or self.colorbar_image is None:
            return
        if self.view_state.roi_color_mode == _CORRELATION_COLOR and self.Fbin is not None:
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
            view1=self.view1,
            view2=self.view2,
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
            color1=self.color1,
            color2=self.color2,
            masks=masks,
        )
        trace_panel.plot_trace(
            trace_box=self.p3,
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
        self.p1.show()
        self.p2.show()
        self.win.show()
        self.show()

    def mode_change(self, i: int) -> None:
        """Changes the activity mode used for multi-neuron display and correlation.

        Activity modes: 0=F, 1=Fneu, 2=F-0.7*Fneu (default), 3=spks.

        Args:
            i: The activity mode index to switch to.
        """
        self.view_state.trace_mode = i
        if self.view_state.session_loaded and self.context_data is not None:
            self.view_state.temporal_bin_size = max(1, int(self.binedit.text()))
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
        self.ntop = int(self.topedit.text())
        if self.view_state.session_loaded and not self.sizebtns.button(1).isChecked():
            for b in [1, 2]:
                if self.topbtns.button(b).isChecked():
                    self._signals.roi_selection_changed.emit()
                    self.show()

    def _roi_selection(self) -> None:
        """Draws a rectangular ROI selection on the active image panel."""
        draw = False
        if self.sizebtns.button(0).isChecked():
            wplot = 0
            view = self.p1.viewRange()
            draw = True
        elif self.sizebtns.button(2).isChecked():
            wplot = 1
            view = self.p2.viewRange()
            draw = True
        if draw:
            self._roi_remove()
            self.topbtns.button(0).setStyleSheet(BUTTON_PRESSED_STYLESHEET)
            self.view_state.roi_tool_panel = wplot
            imx = (view[0][1] + view[0][0]) / 2
            imy = (view[1][1] + view[1][0]) / 2
            dx = (view[0][1] - view[0][0]) / 4
            dy = (view[1][1] - view[1][0]) / 4
            dx = np.minimum(dx, 300)
            dy = np.minimum(dy, 300)
            imx = imx - dx / 2
            imy = imy - dy / 2
            self.ROI = pg.RectROI([imx, imy], [dx, dy], pen="w", sideScalers=True)
            if wplot == 0:
                self.p1.addItem(self.ROI)
            else:
                self.p2.addItem(self.ROI)
            self._roi_position()
            self.ROI.sigRegionChangeFinished.connect(self._roi_position)
            self.view_state.roi_tool_active = True

    def _roi_remove(self) -> None:
        """Removes the current rectangular ROI selection and resets button styles."""
        if self.view_state.roi_tool_active:
            if self.view_state.roi_tool_panel == 0:
                self.p1.removeItem(self.ROI)
            else:
                self.p2.removeItem(self.ROI)
            self.view_state.roi_tool_active = False
        if self.sizebtns.button(1).isChecked():
            self.topbtns.button(0).setStyleSheet(BUTTON_INACTIVE_STYLESHEET)
            self.topbtns.button(0).setEnabled(False)
        else:
            self.topbtns.button(0).setStyleSheet(BUTTON_UNPRESSED_STYLESHEET)

    def _roi_position(self) -> None:
        """Computes the pixel region covered by the ROI and selects contained cells."""
        if self.context_data is None:
            return
        pos0 = self.ROI.getSceneHandlePositions()
        pos = (
            self.p1.mapSceneToView(pos0[0][1])
            if self.view_state.roi_tool_panel == 0
            else self.p2.mapSceneToView(pos0[0][1])
        )
        posy = pos.y()
        posx = pos.x()
        sizex, sizey = self.ROI.size()
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
            self.view_state.selected_roi_index = int(self.ROIedit.text())
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
        if QtCore.Qt.CheckState(state) == QtCore.Qt.Checked:
            self.view_state.rois_visible = True
            self.p1.addItem(self.color1)
            self.p2.addItem(self.color2)
        else:
            self.view_state.rois_visible = False
            self.p1.removeItem(self.color1)
            self.p2.removeItem(self.color2)
        self.win.show()
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
        if not self.sizebtns.button(1).isChecked():
            for btn in self.topbtns.buttons():
                if btn.isChecked():
                    btn.setStyleSheet(BUTTON_UNPRESSED_STYLESHEET)
        self.update_plot()
        return True

    def _ichosen_stats(self) -> None:
        """Updates the ROI statistics labels for the currently selected cell."""
        if self.context_data is None:
            return
        n = self.view_state.selected_roi_index
        self.ROIedit.setText(str(n))
        roi = self.context_data.roi_statistics[n]
        for k in range(1, len(self.stats_to_show) + 1):
            key = self.stats_to_show[k - 1]
            ival = getattr(roi, key, None)
            if ival is None:
                continue
            if k == _CENTROID_STAT_INDEX:
                self.ROIstats[k].setText(f"{key}: [{ival[0]:d}, {ival[1]:d}]")
            elif k == _PIXEL_COUNT_STAT_INDEX:
                self.ROIstats[k].setText(f"{key}: {ival:d}")
            else:
                self.ROIstats[k].setText(f"{key}: {ival:2.2f}")

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
        self.p1.setYRange(imin[0], imax[0])
        self.p1.setXRange(imin[1], imax[1])
        self.p2.setYRange(imin[0], imax[0])
        self.p2.setXRange(imin[1], imax[1])
        self.win.show()
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
            self.p1.autoRange()
            self.p2.autoRange()
        elif panel == _NONCELLS_PLOT:
            self.p2.autoRange()

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
