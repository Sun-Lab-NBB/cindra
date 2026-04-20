from dataclasses import dataclass
from collections.abc import Callable

import numpy as np
from PySide6 import QtCore
from _typeshed import Incomplete
import pyqtgraph as pg
from numpy.typing import NDArray as NDArray
from PySide6.QtWidgets import QWidget, QToolButton, QButtonGroup
from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent as MouseClickEvent

from .styles import (
    FONTS as FONTS,
    STYLE as STYLE,
    COLORS as COLORS,
    PLOT_STYLE as PLOT_STYLE,
)
from .constants import (
    ROI_CONFIG as ROI_CONFIG,
    TraceMode as TraceMode,
)

type _ClickHandler = Callable[[int, int, bool, bool], bool]
type _ZoomHandler = Callable[[], None]

def escape_returns_focus(window: QWidget, event: QtCore.QEvent) -> bool: ...

@dataclass(frozen=True)
class PlayPauseGroup:
    play_button: QToolButton
    pause_button: QToolButton
    button_group: QButtonGroup

def configure_plot(
    plot: pg.PlotItem,
    *,
    title: str | None = None,
    left_label: str | None = None,
    bottom_label: str | None = None,
    mouse_x: bool = True,
    mouse_y: bool = False,
) -> None: ...
def add_plot_legend(plot: pg.PlotItem, *, column_count: int) -> pg.LegendItem: ...

class TraceBox(pg.PlotItem):
    _frame_count: int
    _y_minimum: float
    _y_maximum: float
    def __init__(self) -> None: ...
    def update_range(self, frame_count: int, y_minimum: float, y_maximum: float) -> None: ...
    def mouseDoubleClickEvent(self, event: object) -> None: ...

class ViewBox(pg.ViewBox):
    border: Incomplete
    menu: Incomplete
    name: Incomplete
    _click_handler: _ClickHandler | None
    _zoom_handler: _ZoomHandler | None
    def __init__(
        self, *, border: object = None, invert_y: bool = False, enable_menu: bool = True, name: str | None = None
    ) -> None: ...
    def set_click_handler(self, handler: _ClickHandler) -> None: ...
    def set_zoom_handler(self, handler: _ZoomHandler) -> None: ...
    def mouseDoubleClickEvent(self, event: object) -> None: ...
    def mouseClickEvent(self, event: MouseClickEvent) -> None: ...

def plot_trace(
    trace_box: TraceBox,
    *,
    cell_fluorescence: NDArray[np.float32],
    neuropil_fluorescence: NDArray[np.float32],
    subtracted_fluorescence: NDArray[np.float32],
    spikes: NDArray[np.float32],
    frame_indices: NDArray[np.int32],
    selected_indices: list[int],
    activity_mode: int,
    roi_colors: NDArray[np.uint8] | None = None,
    fluorescence_visible: bool = True,
    neuropil_visible: bool = True,
    corrected_visible: bool = True,
    spikes_visible: bool = True,
    scale_factor: float = ...,
    maximum_trace_count: int = ...,
) -> tuple[float, float]: ...
def _plot_single_trace(
    trace_box: pg.PlotItem,
    axis: pg.AxisItem,
    cell_fluorescence: NDArray[np.float32],
    neuropil_fluorescence: NDArray[np.float32],
    subtracted_fluorescence: NDArray[np.float32],
    spikes: NDArray[np.float32],
    frame_indices: NDArray[np.int32],
    roi_index: int,
    fluorescence_visible: bool,
    neuropil_visible: bool,
    corrected_visible: bool,
    spikes_visible: bool,
) -> tuple[float, float]: ...
def _plot_multi_trace(
    trace_box: pg.PlotItem,
    axis: pg.AxisItem,
    cell_fluorescence: NDArray[np.float32],
    neuropil_fluorescence: NDArray[np.float32],
    subtracted_fluorescence: NDArray[np.float32],
    spikes: NDArray[np.float32],
    frame_indices: NDArray[np.int32],
    selected_indices: list[int],
    activity_mode: int,
    roi_colors: NDArray[np.uint8] | None,
    scale_factor: float,
    maximum_trace_count: int,
) -> tuple[float, float]: ...
def create_play_pause_group(
    parent: QWidget, *, play_tooltip: str, pause_tooltip: str, no_focus: bool = False
) -> PlayPauseGroup: ...
