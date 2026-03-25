from typing import Any

import numpy as np
from PySide6 import QtGui, QtCore
from _typeshed import Incomplete
import pyqtgraph as pg
from numpy.typing import NDArray as NDArray
from PySide6.QtWidgets import QLabel, QWidget, QComboBox, QLineEdit, QGridLayout, QMainWindow

from .styles import (
    FONTS as FONTS,
    STYLE as STYLE,
    COLORS as COLORS,
    PC_STYLE as PC_STYLE,
    PLOT_STYLE as PLOT_STYLE,
)
from .widgets import (
    configure_plot as configure_plot,
    add_plot_legend as add_plot_legend,
    escape_returns_focus as escape_returns_focus,
    create_play_pause_group as create_play_pause_group,
)
from .constants import (
    PC_CONFIG as PC_CONFIG,
    COMMON_CONFIG as COMMON_CONFIG,
)
from .viewer_context import SingleRecordingData as SingleRecordingData

class PCViewer(QMainWindow):
    _central_widget: QWidget
    _layout: QGridLayout
    data: SingleRecordingData
    _loaded: bool
    _current_frame: int
    _pc_count: int
    _pc_images: NDArray[np.float32] | None
    _image_height: int
    _image_width: int
    _pc_metrics: NDArray[np.float32] | None
    _pc_projections: NDArray[np.float32] | None
    _metrics_scatter: pg.ScatterPlotItem | None
    _legend: pg.LegendItem | None
    _metrics_y_range: tuple[float, float]
    _projection_y_range: tuple[float, float]
    _plane_selector: QComboBox
    _graphics_widget: pg.GraphicsLayoutWidget
    _metrics_plot: Incomplete
    _difference_view_box: Incomplete
    _merged_view_box: Incomplete
    _animated_view_box: Incomplete
    _difference_image: pg.ImageItem
    _merged_image: pg.ImageItem
    _animated_image: pg.ImageItem
    _title_labels: list[pg.TextItem]
    _projection_plot: Incomplete
    _update_timer: QtCore.QTimer
    def __init__(self, data: SingleRecordingData) -> None: ...
    def load_data(self, data: SingleRecordingData) -> None: ...
    def get_state(self) -> dict[str, Any]: ...
    def _on_plane_changed(self, index: int) -> None: ...
    def _reload_pc_data(self) -> None: ...
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None: ...
    def eventFilter(self, source: QtCore.QObject, event: QtCore.QEvent) -> bool: ...
    _pc_edit: QLineEdit
    _metric_labels: list[QLabel]
    _play_button: Incomplete
    _pause_button: Incomplete
    def _create_bottom_panel(self) -> None: ...
    def _start_animation(self) -> None: ...
    def _pause_animation(self) -> None: ...
    def _next_frame(self) -> None: ...
    def _plot_frame(self) -> None: ...
    def _zoom_plot(self) -> None: ...
    def _plot_clicked(self, event: object) -> None: ...
