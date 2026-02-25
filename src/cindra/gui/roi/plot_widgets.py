"""Provides custom pyqtgraph plot widgets for rendering fluorescent traces and ROI panels in the main GUI viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtCore
import pyqtgraph as pg  # type: ignore[import-untyped]
from pyqtgraph import functions as fn
from pyqtgraph.graphicsItems.ViewBox.ViewBoxMenu import ViewBoxMenu  # type: ignore[import-untyped]

from .view_state import ROIToolPanel

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent  # type: ignore[import-untyped]

type _ClickHandler = Callable[[int, int, ROIToolPanel, bool, bool], bool]
"""The callback type for click events dispatched by a ViewBox to the orchestrator."""

type _ZoomHandler = Callable[[], None]
"""The callback type for double-click zoom-to-fit events dispatched by a ViewBox to the orchestrator."""


class TraceBox(pg.PlotItem):
    """Displays fluorescence time series with support for custom mouse interactions.

    Extends pyqtgraph's PlotItem class with stored trace range values that are updated after each call to
    ``plot_trace`` via ``update_range``. Double-clicking the plot resets the view to the
    full data range.

    Attributes:
        _frame_count: Total number of frames in the current trace data.
        _y_minimum: Minimum y-axis value for zoom-to-fit.
        _y_maximum: Maximum y-axis value for zoom-to-fit.
    """

    def __init__(self) -> None:
        super().__init__()
        self._frame_count: int = 0
        self._y_minimum: float = 0.0
        self._y_maximum: float = 0.0

    def update_range(self, frame_count: int, y_minimum: float, y_maximum: float) -> None:
        """Updates the axis limits used by double-click zoom-to-fit interaction.

        This determines the behavior of the zoom-to-fit user-triggered interface transformation.

        Args:
            frame_count: Total number of frames in the trace data.
            y_minimum: Minimum y-axis value.
            y_maximum: Maximum y-axis value.
        """
        self._frame_count = frame_count
        self._y_minimum = y_minimum
        self._y_maximum = y_maximum

    def mouseDoubleClickEvent(self, ev: object) -> None:  # noqa: N802, ARG002
        """Zooms the managed trace plot to fit the full data range.

        Notes:
            Overrides the pyqtgraph/Qt virtual method. The camelCase name and unused ``ev``
            parameter are required to match the parent signature.
        """
        view_box = self.getViewBox()
        view_box.setXRange(0, self._frame_count)
        view_box.setYRange(self._y_minimum, self._y_maximum)


class ViewBox(pg.ViewBox):
    """Displays field-of-view images with support for custom keyboard and mouse interactions.

    Extends pyqtgraph's ViewBox class with left-click ROI selection, right-click ROI reclassification (cell / non-cell),
    shift/ctrl-click multi-ROI merge selection, and double-click zoom-to-fit functionality. All click logic is
    delegated to the orchestrator via installed callback handlers.

    Args:
        panel: Identifies which image panel this view box belongs to. Typically, this is either the 'cells' or
            'non-cells' panel.
        border: The panel border frame pen specification forwarded to ``fn.mkPen``. This determines the appearance of
            the rendered panel's border.
        invert_y: Determines whether to invert the Y axis.
        enable_menu: Determines whether the context menu is enabled.
        name: The unique name for the managed panel used by pyqtgraph's view-linking system.

    Attributes:
        _panel: Cached panel identifier for click handler delegation.
        _click_handler: Callback installed by the orchestrator for click events.
        _zoom_handler: Callback installed by the orchestrator for double-click zoom.
    """

    def __init__(
        self,
        *,
        panel: ROIToolPanel = ROIToolPanel.CELLS,
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
        self._panel: ROIToolPanel = panel

        # Configures view state.
        self.state["enableMenu"] = enable_menu
        self.state["yInverted"] = invert_y

        # Callbacks installed by the orchestrator after construction.
        self._click_handler: _ClickHandler | None = None
        self._zoom_handler: _ZoomHandler | None = None

    def set_click_handler(self, handler: _ClickHandler) -> None:
        """Configures the instance to use the provided click handler when the user clicks in this panel.

        Args:
            handler: The callback instance to be invoked on each mouse click in this panel.
        """
        self._click_handler = handler

    def set_zoom_handler(self, handler: _ZoomHandler) -> None:
        """Configures the instance to use the provided click handler on double-click zoom-to-fit user interactions.

        Args:
            handler: The callback instance to be invoked on double-click to reset the view range.
        """
        self._zoom_handler = handler

    def mouseDoubleClickEvent(self, ev: object) -> None:  # noqa: N802, ARG002
        """Zooms the image view to fit the full field of view.

        Notes:
            Overrides the pyqtgraph/Qt virtual method. The camelCase name and unused ``ev``
            parameter are required to match the parent signature.
        """
        # Silent handling is fine since the handler is always configured before this is called during valid class use
        # patterns.
        if self._zoom_handler is not None:
            self._zoom_handler()

    def mouseClickEvent(self, ev: MouseClickEvent) -> None:  # noqa: N802
        """Dispatches mouse click events to the installed click handler.

        Left-click selects the targeted ROI. Shift/ctrl-click toggles multi-ROI merge selection.
        Right-click reclassifies the ROI between the cell and non-cell panels.
        Unhandled right-clicks raise the default context menu.

        Notes:
            Overrides the pyqtgraph/Qt virtual method. The camelCase name is required to match
            the parent signature.
        """
        if self._click_handler is None:
            return

        # Converts the scene-space click position to image-space pixel coordinates.
        position = self.mapSceneToView(ev.scenePos())
        click_x = int(position.x())
        click_y = int(position.y())

        # Extracts modifier state for the click handler.
        is_right = ev.button() == QtCore.Qt.MouseButton.RightButton
        is_multi = ev.modifiers() in (
            QtCore.Qt.KeyboardModifier.ShiftModifier,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )

        # Falls back to the default context menu if the click handler did not consume the event.
        handled = self._click_handler(click_x, click_y, self._panel, is_right, is_multi)
        if not handled and is_right and self.menuEnabled():
            self.raiseContextMenu(ev)


def initialize_ranges(
    cells_view: ViewBox,
    noncells_view: ViewBox,
    trace_box: TraceBox,
    frame_width: int,
    frame_height: int,
    frame_count: int,
) -> NDArray[np.int32]:
    """Initializes plot ranges for all panels after the UI loads the visualized runtime context data.

    Sets both cell and non-cell image panel view ranges to span the full field of view, constrains the trace
    panel x-axis to the recording length, and returns a frame indices array for trace plotting.

    Notes:
        Called once per session load from ``context_loader`` after pipeline arrays and metadata
        have been populated into ``ContextData``. The returned frame indices array is stored by
        the caller and reused across all subsequent ``plot_trace`` calls for the session.

    Args:
        cells_view: The cell image panel view box.
        noncells_view: The non-cell image panel view box.
        trace_box: The fluorescence trace display panel.
        frame_width: Width of the field of view in pixels.
        frame_height: Height of the field of view in pixels.
        frame_count: Total number of frames in the visualized recording.

    Returns:
        The monotonically increasing frame number array of shape (frame_count,) used as the
        x-axis for trace plotting.
    """
    cells_view.setXRange(0, frame_width)
    cells_view.setYRange(0, frame_height)
    noncells_view.setXRange(0, frame_width)
    noncells_view.setYRange(0, frame_height)
    trace_box.getViewBox().setLimits(xMin=0, xMax=frame_count)
    return np.arange(0, frame_count, dtype=np.int32)
