"""Provides custom pyqtgraph plot widgets for the main GUI viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtCore
import pyqtgraph as pg
from pyqtgraph import functions as fn
from pyqtgraph.graphicsItems.ViewBox.ViewBoxMenu import ViewBoxMenu

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


class TraceBox(pg.PlotItem):
    """Trace plot widget for displaying fluorescence time series.

    Extends PlotItem with double-click zoom-to-fit and stored trace range values.
    The range values are updated after each call to ``plot_trace`` via ``update_range``.

    Args:
        owner: The parent QWidget that owns this plot item.
    """

    def __init__(self, *, owner: object | None = None) -> None:  # noqa: ARG002
        super().__init__()
        self._frame_count: int = 0
        self._y_minimum: float = 0.0
        self._y_maximum: float = 0.0

    def update_range(self, frame_count: int, y_minimum: float, y_maximum: float) -> None:
        """Updates the stored trace range for zoom-to-fit.

        Args:
            frame_count: Total number of frames in the trace data.
            y_minimum: Minimum y-axis value.
            y_maximum: Maximum y-axis value.
        """
        self._frame_count = frame_count
        self._y_minimum = y_minimum
        self._y_maximum = y_maximum

    def mouseDoubleClickEvent(self, ev: object) -> None:  # noqa: N802, ARG002
        """Zooms the trace plot to fit the full data range."""
        self.setXRange(0, self._frame_count)
        self.setYRange(self._y_minimum, self._y_maximum)


class ViewBox(pg.ViewBox):
    """Image panel view box with ROI click handling via callbacks.

    Extends ViewBox with click-to-select and right-click-to-flip ROI behavior.
    Handles shift/ctrl-click for multi-ROI merge selection. All click logic is
    delegated to the orchestrator via an installed click handler.

    Args:
        panel_index: Zero-based panel identifier (0 for cells, 1 for non-cells).
        border: Border pen specification forwarded to ``fn.mkPen``.
        invert_y: Determines whether to invert the Y axis.
        enable_menu: Determines whether the context menu is enabled.
        name: Unique name for linking views.
    """

    def __init__(
        self,
        *,
        panel_index: int = 0,
        border: object = None,
        invert_y: bool = False,
        enable_menu: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__()
        self.border = fn.mkPen(border)
        if enable_menu:
            self.menu = ViewBoxMenu(self)
        self.name = name
        self._panel_index: int = panel_index

        # Configures view state.
        self.state["enableMenu"] = enable_menu
        self.state["yInverted"] = invert_y

        # Callbacks installed by the orchestrator after construction.
        self._click_handler: Callable[[int, int, int, bool, bool], bool] | None = None
        self._zoom_handler: Callable[[], None] | None = None

    def set_click_handler(
        self,
        handler: Callable[[int, int, int, bool, bool], bool],
    ) -> None:
        """Installs the click handler called when the user clicks in this view.

        Args:
            handler: Callback with signature (click_x, click_y, panel_index,
                is_right_button, is_multi_select). Returns True if the click
                was handled, False if the click was on empty space.
        """
        self._click_handler = handler

    def set_zoom_handler(self, handler: Callable[[], None]) -> None:
        """Installs the handler called on double-click zoom-to-fit.

        Args:
            handler: Callback that resets view ranges to the full field of view.
        """
        self._zoom_handler = handler

    def mouseDoubleClickEvent(self, ev: object) -> None:  # noqa: N802, ARG002
        """Zooms the image view to fit the full field of view."""
        if self._zoom_handler is not None:
            self._zoom_handler()

    def mouseClickEvent(self, ev) -> None:  # noqa: N802, ANN001
        """Delegates cell selection and flip clicks to the installed handler.

        Left-click selects a cell. Shift/ctrl-click toggles multi-selection.
        Right-click flips the cell between cell and non-cell panels.
        Unhandled right-clicks raise the default context menu.
        """
        if self._click_handler is None:
            return

        position = self.mapSceneToView(ev.scenePos())
        click_x = int(position.x())
        click_y = int(position.y())
        is_right = ev.button() == QtCore.Qt.RightButton
        is_multi = ev.modifiers() in (QtCore.Qt.ShiftModifier, QtCore.Qt.ControlModifier)

        handled = self._click_handler(click_x, click_y, self._panel_index, is_right, is_multi)
        if not handled and is_right and self.menuEnabled():
            self.raiseContextMenu(ev)


def initialize_ranges(
    cells_view: ViewBox,
    noncells_view: ViewBox,
    trace_box: TraceBox,
    frame_width: int,
    frame_height: int,
    frame_count: int,
) -> NDArray:
    """Initializes plot ranges for all panels after data loading.

    Sets the image panel ranges to the full field of view and configures the trace
    panel x-axis limits.

    Args:
        cells_view: The cell panel view box.
        noncells_view: The non-cell panel view box.
        trace_box: The trace display panel.
        frame_width: Width of the field of view in pixels.
        frame_height: Height of the field of view in pixels.
        frame_count: Total number of frames in the recording.

    Returns:
        Time range array of shape (frame_count,).
    """
    cells_view.setXRange(0, frame_width)
    cells_view.setYRange(0, frame_height)
    noncells_view.setXRange(0, frame_width)
    noncells_view.setYRange(0, frame_height)
    trace_box.setLimits(xMin=0, xMax=frame_count)
    return np.arange(0, frame_count)
