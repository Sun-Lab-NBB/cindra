"""Provides custom Qt widgets, trace plotting helpers, and quadrant zoom for all GUI applications."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6 import QtCore
import pyqtgraph as pg  # type: ignore[import-untyped]
from pyqtgraph import functions as fn
from pyqtgraph.graphicsItems.ViewBox.ViewBoxMenu import ViewBoxMenu  # type: ignore[import-untyped]

from .styles import FONTS, COLORS, PLOT_STYLE
from .constants import ROI_CONFIG, TraceMode

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent  # type: ignore[import-untyped]

type _ClickHandler = Callable[[int, int, bool, bool], bool]
"""The callback type for click events dispatched by a ViewBox to the orchestrator.

Signature: (click_x, click_y, is_right_button, is_multi_select) -> handled.
"""

type _ZoomHandler = Callable[[], None]
"""The callback type for double-click zoom-to-fit events dispatched by a ViewBox to the orchestrator."""


def configure_plot(
    plot: pg.PlotItem,
    *,
    title: str | None = None,
    left_label: str | None = None,
    bottom_label: str | None = None,
    mouse_x: bool = True,
    mouse_y: bool = False,
) -> None:
    """Applies the shared pyqtgraph plot configuration backbone.

    Disables the context menu, sets mouse interaction axes, fixes axis widths to prevent layout
    shifts, and optionally sets the plot title and axis labels.

    Args:
        plot: The pyqtgraph PlotItem to configure.
        title: Optional plot title text.
        left_label: Optional label for the left (y) axis.
        bottom_label: Optional label for the bottom (x) axis.
        mouse_x: Determines whether horizontal mouse interaction is enabled.
        mouse_y: Determines whether vertical mouse interaction is enabled.
    """
    plot.setMenuEnabled(False)
    # noinspection PyUnresolvedReferences
    plot.setMouseEnabled(x=mouse_x, y=mouse_y)
    plot.getAxis("left").setWidth(PLOT_STYLE.axis_fixed_width)
    plot.getAxis("bottom").setHeight(PLOT_STYLE.axis_fixed_width)
    if title:
        # noinspection PyTypeChecker
        plot.setTitle(title, size=FONTS.plot_title_size, bold=True)
    if left_label:
        plot.setLabel("left", left_label, **{"font-size": FONTS.label_size})
    if bottom_label:
        plot.setLabel("bottom", bottom_label, **{"font-size": FONTS.label_size})


def add_plot_legend(plot: pg.PlotItem, *, column_count: int) -> pg.LegendItem:
    """Adds a standardized legend to a pyqtgraph plot.

    Args:
        plot: The pyqtgraph PlotItem to add a legend to.
        column_count: Number of columns in the legend layout.

    Returns:
        The created LegendItem instance.
    """
    return plot.addLegend(
        horSpacing=PLOT_STYLE.legend_horizontal_spacing,
        colCount=column_count,
        offset=PLOT_STYLE.legend_offset,
        labelTextSize=FONTS.label_size,
    )


class TraceBox(pg.PlotItem):
    """Displays fluorescence time series with support for custom mouse interactions.

    Extends pyqtgraph's PlotItem class with stored trace range values that are updated after each
    call to ``plot_trace`` via ``update_range``. Double-clicking the plot resets the view to the
    full data range.

    Attributes:
        _frame_count: Total number of frames in the current trace data.
        _y_minimum: Minimum y-axis value for zoom-to-fit.
        _y_maximum: Maximum y-axis value for zoom-to-fit.
    """

    def __init__(self) -> None:
        super().__init__()
        configure_plot(self)
        self._frame_count: int = 0
        self._y_minimum: float = 0.0
        self._y_maximum: float = 0.0

    def update_range(self, frame_count: int, y_minimum: float, y_maximum: float) -> None:
        """Updates the axis limits used by double-click zoom-to-fit interaction.

        Args:
            frame_count: Total number of frames in the trace data.
            y_minimum: Minimum y-axis value.
            y_maximum: Maximum y-axis value.
        """
        self._frame_count = frame_count
        self._y_minimum = y_minimum
        self._y_maximum = y_maximum

    def mouseDoubleClickEvent(self, event: object) -> None:  # noqa: N802, ARG002
        """Zooms the managed trace plot to fit the full data range.

        Notes:
            Overrides the pyqtgraph/Qt virtual method. The camelCase name and unused ``event``
            parameter are required to match the parent signature.
        """
        view_box = self.getViewBox()
        view_box.setXRange(0, self._frame_count)
        view_box.setYRange(self._y_minimum, self._y_maximum)


class ViewBox(pg.ViewBox):
    """Displays field-of-view images with support for custom keyboard and mouse interactions.

    Extends pyqtgraph's ViewBox class with left-click ROI selection, right-click ROI
    reclassification (cell / non-cell), shift/ctrl-click multi-ROI merge selection, and
    double-click zoom-to-fit functionality. All click logic is delegated to the orchestrator via
    installed callback handlers.

    Args:
        border: The panel border frame pen specification forwarded to ``fn.mkPen``.
        invert_y: Determines whether to invert the Y axis.
        enable_menu: Determines whether the context menu is enabled.
        name: The unique name for the managed panel used by pyqtgraph's view-linking system.

    Attributes:
        _click_handler: Callback installed by the orchestrator for click events.
        _zoom_handler: Callback installed by the orchestrator for double-click zoom.
    """

    def __init__(
        self,
        *,
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

        # Configures view state.
        self.state["enableMenu"] = enable_menu
        self.state["yInverted"] = invert_y

        # Callbacks installed by the orchestrator after construction.
        self._click_handler: _ClickHandler | None = None
        self._zoom_handler: _ZoomHandler | None = None

    def set_click_handler(self, handler: _ClickHandler) -> None:
        """Configures the instance to use the provided click handler when the user clicks.

        Args:
            handler: The callback instance to be invoked on each mouse click in this panel.
        """
        self._click_handler = handler

    def set_zoom_handler(self, handler: _ZoomHandler) -> None:
        """Configures the instance to use the provided zoom handler on double-click zoom-to-fit.

        Args:
            handler: The callback instance to be invoked on double-click to reset the view range.
        """
        self._zoom_handler = handler

    def mouseDoubleClickEvent(self, event: object) -> None:  # noqa: N802, ARG002
        """Zooms the image view to fit the full field of view.

        Notes:
            Overrides the pyqtgraph/Qt virtual method. The camelCase name and unused ``event``
            parameter are required to match the parent signature.
        """
        if self._zoom_handler is not None:
            self._zoom_handler()

    def mouseClickEvent(self, event: MouseClickEvent) -> None:  # noqa: N802
        """Dispatches mouse click events to the installed click handler.

        Left-click selects the targeted ROI. Shift/ctrl-click toggles multi-ROI merge selection.
        Right-click reclassifies the ROI between the cell and non-cell panels. Unhandled
        right-clicks raise the default context menu.

        Notes:
            Overrides the pyqtgraph/Qt virtual method. The camelCase name is required to match
            the parent signature.
        """
        if self._click_handler is None:
            return

        # Converts the scene-space click position to image-space pixel coordinates.
        position = self.mapSceneToView(event.scenePos())
        click_x = int(position.x())
        click_y = int(position.y())

        # Extracts modifier state for the click handler.
        is_right_button = event.button() == QtCore.Qt.MouseButton.RightButton
        is_multi_select = event.modifiers() in (
            QtCore.Qt.KeyboardModifier.ShiftModifier,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )

        # Falls back to the default context menu if the click handler did not consume the event.
        handled = self._click_handler(click_x, click_y, is_right_button, is_multi_select)
        if not handled and is_right_button and self.menuEnabled():
            self.raiseContextMenu(event)


def plot_trace(
    trace_box: TraceBox,
    *,
    cell_fluorescence: NDArray[np.float32],
    neuropil_fluorescence: NDArray[np.float32],
    subtracted_fluorescence: NDArray[np.float32],
    spikes: NDArray[np.float32],
    frame_indices: NDArray[np.int32],
    merge_indices: list[int],
    activity_mode: int,
    roi_colors: NDArray[np.uint8] | None = None,
    traces_visible: bool = True,
    neuropil_visible: bool = True,
    deconvolved_visible: bool = True,
    scale_factor: float = ROI_CONFIG.default_scale_factor,
    max_plotted: int = ROI_CONFIG.plotted_trace_count,
) -> tuple[float, float]:
    """Draws fluorescence traces for the selected ROIs.

    For a single selected ROI, displays the raw fluorescence, neuropil, and deconvolved
    traces on the same axes. For multiple selected ROIs, stacks normalized traces vertically
    with per-ROI coloring and an optional averaged summary at the bottom.

    Args:
        trace_box: The pyqtgraph PlotItem to draw traces on.
        cell_fluorescence: Cell fluorescence array with shape (roi_count, frame_count).
        neuropil_fluorescence: Neuropil fluorescence array with shape (roi_count, frame_count).
        subtracted_fluorescence: Pre-computed baseline-and-neuropil-subtracted fluorescence with shape
            (roi_count, frame_count).
        spikes: Deconvolved spike array with shape (roi_count, frame_count).
        frame_indices: Time axis array with shape (frame_count,).
        merge_indices: Indices of the selected ROIs to display.
        activity_mode: Trace type index (0=Fluorescence, 1=Neuropil, 2=Neuropil Subtracted, 3=Spikes).
        roi_colors: Per-ROI RGB colors with shape (roi_count, 3) for multi-trace coloring.
        traces_visible: Determines whether the raw fluorescence trace is drawn.
        neuropil_visible: Determines whether the neuropil trace is drawn.
        deconvolved_visible: Determines whether the deconvolved spike trace is drawn.
        scale_factor: Vertical spacing factor for stacked multi-trace display.
        max_plotted: Maximum number of traces to plot in multi-ROI mode.

    Returns:
        Tuple of (y_minimum, y_maximum) defining the plotted y-axis range.
    """
    trace_box.clear()
    axis = trace_box.getAxis("left")

    if len(merge_indices) == 1:
        y_minimum, y_maximum = _plot_single_trace(
            trace_box=trace_box,
            axis=axis,
            cell_fluorescence=cell_fluorescence,
            neuropil_fluorescence=neuropil_fluorescence,
            spikes=spikes,
            frame_indices=frame_indices,
            roi_index=merge_indices[0],
            traces_visible=traces_visible,
            neuropil_visible=neuropil_visible,
            deconvolved_visible=deconvolved_visible,
        )
    else:
        y_minimum, y_maximum = _plot_multi_trace(
            trace_box=trace_box,
            axis=axis,
            cell_fluorescence=cell_fluorescence,
            neuropil_fluorescence=neuropil_fluorescence,
            subtracted_fluorescence=subtracted_fluorescence,
            spikes=spikes,
            frame_indices=frame_indices,
            merge_indices=merge_indices,
            activity_mode=activity_mode,
            roi_colors=roi_colors,
            scale_factor=scale_factor,
            max_plotted=max_plotted,
        )

    trace_box.update_range(
        frame_count=len(frame_indices),
        y_minimum=y_minimum,
        y_maximum=y_maximum,
    )
    # noinspection PyUnresolvedReferences
    trace_box.setYRange(y_minimum, y_maximum)
    return y_minimum, y_maximum


def _plot_single_trace(
    trace_box: pg.PlotItem,
    axis: pg.AxisItem,
    cell_fluorescence: NDArray[np.float32],
    neuropil_fluorescence: NDArray[np.float32],
    spikes: NDArray[np.float32],
    frame_indices: NDArray[np.int32],
    roi_index: int,
    traces_visible: bool,
    neuropil_visible: bool,
    deconvolved_visible: bool,
) -> tuple[float, float]:
    """Plots traces for a single selected ROI.

    Args:
        trace_box: The plot item to draw on.
        axis: The left y-axis for tick configuration.
        cell_fluorescence: Cell fluorescence array with shape (roi_count, frame_count).
        neuropil_fluorescence: Neuropil fluorescence array with shape (roi_count, frame_count).
        spikes: Deconvolved spike array with shape (roi_count, frame_count).
        frame_indices: Time axis array.
        roi_index: Index of the ROI to plot.
        traces_visible: Determines whether the raw fluorescence trace is drawn.
        neuropil_visible: Determines whether the neuropil trace is drawn.
        deconvolved_visible: Determines whether the deconvolved spike trace is drawn.

    Returns:
        Tuple of (y_minimum, y_maximum) for the plotted range.
    """
    fluorescence = cell_fluorescence[roi_index, :]
    neuropil = neuropil_fluorescence[roi_index, :]
    spike_trace = spikes[roi_index, :].copy()

    if np.ptp(neuropil) == 0:
        y_maximum = float(fluorescence.max())
        y_minimum = float(fluorescence.min())
    else:
        y_maximum = float(max(fluorescence.max(), neuropil.max()))
        y_minimum = float(min(fluorescence.min(), neuropil.min()))

    # Normalizes spike trace to fill the y-range.
    spike_maximum = spike_trace.max()
    if spike_maximum > 0:
        spike_trace /= spike_maximum
    spike_trace *= y_maximum - y_minimum

    if traces_visible:
        trace_box.plot(frame_indices, fluorescence, pen=COLORS.cyan)
    if neuropil_visible:
        trace_box.plot(frame_indices, neuropil, pen=COLORS.red)
    if deconvolved_visible:
        trace_box.plot(frame_indices, spike_trace + y_minimum, pen=COLORS.silver)

    axis.setTicks(None)
    return y_minimum, y_maximum


def _plot_multi_trace(
    trace_box: pg.PlotItem,
    axis: pg.AxisItem,
    cell_fluorescence: NDArray[np.float32],
    neuropil_fluorescence: NDArray[np.float32],
    subtracted_fluorescence: NDArray[np.float32],
    spikes: NDArray[np.float32],
    frame_indices: NDArray[np.int32],
    merge_indices: list[int],
    activity_mode: int,
    roi_colors: NDArray[np.uint8] | None,
    scale_factor: float,
    max_plotted: int,
) -> tuple[float, float]:
    """Plots stacked traces for multiple selected ROIs.

    Args:
        trace_box: The plot item to draw on.
        axis: The left y-axis for tick configuration.
        cell_fluorescence: Cell fluorescence array with shape (roi_count, frame_count).
        neuropil_fluorescence: Neuropil fluorescence array with shape (roi_count, frame_count).
        subtracted_fluorescence: Pre-computed baseline-and-neuropil-subtracted fluorescence with shape
            (roi_count, frame_count).
        spikes: Deconvolved spike array with shape (roi_count, frame_count).
        frame_indices: Time axis array.
        merge_indices: Indices of selected ROIs.
        activity_mode: Trace type index (0=Fluorescence, 1=Neuropil, 2=Neuropil Subtracted, 3=Spikes).
        roi_colors: Per-ROI RGB colors with shape (roi_count, 3).
        scale_factor: Vertical spacing factor for trace stacking.
        max_plotted: Maximum number of traces to display.

    Returns:
        Tuple of (y_minimum, y_maximum) for the plotted range.
    """
    selected = merge_indices[: min(len(merge_indices), max_plotted)]
    trace_spacing = 1.0 / scale_factor
    tick_labels: list[tuple[float, str]] = []
    stack_position = len(selected) - 1
    average = np.zeros((cell_fluorescence.shape[1],), dtype=np.float64)

    for index in selected[::-1]:
        # Selects trace based on activity mode.
        if activity_mode == 0:
            trace = cell_fluorescence[index, :]
        elif activity_mode == 1:
            trace = neuropil_fluorescence[index, :]
        elif activity_mode == TraceMode.NEUROPIL_CORRECTED:
            trace = subtracted_fluorescence[index, :]
        else:
            trace = spikes[index, :]

        average += trace.flatten()
        trace_max = float(trace.max())
        trace_min = float(trace.min())

        # Normalizes trace to [0, 1] range.
        if trace_max > trace_min:  # noqa: SIM108
            normalized = (trace - trace_min) / (trace_max - trace_min)
        else:
            normalized = np.zeros_like(trace)

        # Determines pen color for this ROI.
        pen_color = roi_colors[index, :] if roi_colors is not None else COLORS.white

        trace_box.plot(frame_indices, normalized + stack_position * trace_spacing, pen=pen_color)
        tick_labels.append((stack_position * trace_spacing + float(normalized.mean()), str(index)))
        stack_position -= 1

    # Computes average trace scale.
    average_scale = len(selected) / ROI_CONFIG.average_scale_divisor + 1
    average -= average.min()
    average_max = average.max()
    if average_max > 0:
        average /= average_max

    y_minimum = 0.0
    average_pen = COLORS.silver

    # Plots average trace at bottom when enough cells are selected.
    if len(selected) > ROI_CONFIG.average_threshold:
        trace_box.plot(
            frame_indices,
            -1 * average_scale + average * average_scale,
            pen=average_pen,
        )
        y_minimum = -1 * average_scale

    y_maximum = (len(selected) - 1) * trace_spacing + 1
    axis.setTicks([tick_labels])
    return y_minimum, y_maximum
