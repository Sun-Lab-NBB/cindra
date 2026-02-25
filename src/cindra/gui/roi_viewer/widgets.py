"""Provides custom Qt widgets and trace plotting helpers for the ROI viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import Callable

import numpy as np
from PySide6 import QtGui, QtCore
import pyqtgraph as pg  # type: ignore[import-untyped]
from pyqtgraph import functions as fn
from PySide6.QtGui import QPainter, QMouseEvent, QPaintEvent
from PySide6.QtWidgets import (
    QStyle,
    QSlider,
    QWidget,
    QApplication,
    QStyleOptionSlider,
)
from pyqtgraph.graphicsItems.ViewBox.ViewBoxMenu import ViewBoxMenu  # type: ignore[import-untyped]

from .constants import STYLE, CONFIG

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent  # type: ignore[import-untyped]


class RangeSlider(QSlider):
    """Dual-handle range slider for controlling image saturation levels.

    Provides two independently movable slider handles that define a low-high range.
    Dragging between the handles moves both together. Releasing the mouse invokes the
    optional ``on_release`` callback so the viewer can refresh its display.

    Args:
        owner: The parent QWidget.
        on_release: Optional callback invoked when the user finishes dragging.
    """

    def __init__(self, owner: QWidget | None = None, on_release: Callable[[], None] | None = None) -> None:
        super().__init__(owner)

        self._low: int = self.minimum()
        self._high: int = self.maximum()
        self._pressed_control = QStyle.SubControl.SC_None
        self._hover_control = QStyle.SubControl.SC_None
        self._click_offset: int = 0

        self.setOrientation(QtCore.Qt.Orientation.Vertical)
        self.setTickPosition(QSlider.TickPosition.TicksRight)
        self.setStyleSheet(STYLE.range_slider)

        # 0 for the low handle, 1 for the high handle, -1 for both.
        self._active_slider: int = 0
        self._on_release: Callable[[], None] | None = on_release

    def low(self) -> int:
        """Returns the current low handle value."""
        return self._low

    def setLow(self, low: int) -> None:  # noqa: N802
        """Sets the low handle value.

        Args:
            low: New low handle position.
        """
        self._low = low
        self.update()

    def high(self) -> int:
        """Returns the current high handle value."""
        return self._high

    def setHigh(self, high: int) -> None:  # noqa: N802
        """Sets the high handle value.

        Args:
            high: New high handle position.
        """
        self._high = high
        self.update()

    def saturation_values(self) -> list[int]:
        """Returns the current [low, high] saturation range."""
        return [self._low, self._high]

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802, ARG002
        """Paints both slider handles on the slider track."""
        painter = QPainter(self)
        style = QApplication.style()

        for _handle_index, value in enumerate([self._low, self._high]):
            option = QStyleOptionSlider()
            self.initStyleOption(option)
            option.subControls = QStyle.SubControl.SC_SliderHandle

            if self.tickPosition() != QSlider.TickPosition.NoTicks:
                option.subControls |= QStyle.SubControl.SC_SliderTickmarks

            if self._pressed_control:
                option.activeSubControls = self._pressed_control
                option.state |= QStyle.StateFlag.State_Sunken
            else:
                option.activeSubControls = self._hover_control

            option.sliderPosition = value
            option.sliderValue = value
            style.drawComplexControl(QStyle.ComplexControl.CC_Slider, option, painter, self)

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Handles mouse press to select and begin dragging a slider handle."""
        event.accept()
        style = QApplication.style()
        button = event.button()

        if button:
            option = QStyleOptionSlider()
            self.initStyleOption(option)
            self._active_slider = -1

            for handle_index, value in enumerate([self._low, self._high]):
                option.sliderPosition = value
                hit = style.hitTestComplexControl(QStyle.ComplexControl.CC_Slider, option, event.pos(), self)
                if hit == QStyle.SubControl.SC_SliderHandle:
                    self._active_slider = handle_index
                    self._pressed_control = hit
                    self.triggerAction(QSlider.SliderAction.SliderMove)
                    self.setRepeatAction(QSlider.SliderAction.SliderNoAction)
                    self.setSliderDown(True)
                    break

            if self._active_slider < 0:
                self._pressed_control = QStyle.SubControl.SC_SliderHandle
                self._click_offset = self._pixel_position_to_value(self._pick_coordinate(event.pos()))
                self.triggerAction(QSlider.SliderAction.SliderMove)
                self.setRepeatAction(QSlider.SliderAction.SliderNoAction)
        else:
            event.ignore()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Handles mouse drag to move the active slider handle."""
        if self._pressed_control != QStyle.SubControl.SC_SliderHandle:
            event.ignore()
            return

        event.accept()
        new_position = self._pixel_position_to_value(self._pick_coordinate(event.pos()))
        option = QStyleOptionSlider()
        self.initStyleOption(option)

        if self._active_slider < 0:
            # Moves both handles together.
            offset = new_position - self._click_offset
            self._high += offset
            self._low += offset
            if self._low < self.minimum():
                difference = self.minimum() - self._low
                self._low += difference
                self._high += difference
            if self._high > self.maximum():
                difference = self.maximum() - self._high
                self._low += difference
                self._high += difference
        elif self._active_slider == 0:
            if new_position >= self._high:
                new_position = self._high - 1
            self._low = new_position
        else:
            if new_position <= self._low:
                new_position = self._low + 1
            self._high = new_position

        self._click_offset = new_position
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802, ARG002
        """Invokes the on_release callback after the slider handles are released."""
        if self._on_release is not None:
            self._on_release()

    def _pick_coordinate(self, point: QtCore.QPoint) -> int:
        """Extracts the relevant coordinate from a point based on slider orientation.

        Args:
            point: The mouse position.

        Returns:
            The x or y coordinate depending on the slider orientation.
        """
        if self.orientation() == QtCore.Qt.Orientation.Horizontal:
            return point.x()
        return point.y()

    def _pixel_position_to_value(self, pixel_position: int) -> int:
        """Converts a pixel position to a slider value.

        Args:
            pixel_position: Pixel coordinate along the slider axis.

        Returns:
            The corresponding slider value.
        """
        option = QStyleOptionSlider()
        self.initStyleOption(option)
        style = QApplication.style()

        groove_rect = style.subControlRect(
            QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderGroove, self
        )
        handle_rect = style.subControlRect(
            QStyle.ComplexControl.CC_Slider, option, QStyle.SubControl.SC_SliderHandle, self
        )

        if self.orientation() == QtCore.Qt.Orientation.Horizontal:
            slider_length = handle_rect.width()
            slider_min = groove_rect.x()
            slider_max = groove_rect.right() - slider_length + 1
        else:
            slider_length = handle_rect.height()
            slider_min = groove_rect.y()
            slider_max = groove_rect.bottom() - slider_length + 1

        return style.sliderValueFromPosition(
            self.minimum(),
            self.maximum(),
            pixel_position - slider_min,
            slider_max - slider_min,
            option.upsideDown,
        )


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


type _ClickHandler = Callable[[int, int, int, bool], bool]
"""The callback type for click events dispatched by a ViewBox to the orchestrator.

Signature: (click_x, click_y, panel_index, is_multi) -> handled.
"""

type _ZoomHandler = Callable[[], None]
"""The callback type for double-click zoom-to-fit events dispatched by a ViewBox to the orchestrator."""


class ViewBox(pg.ViewBox):
    """Displays field-of-view images with left-click ROI selection for the read-only viewer.

    Extends pyqtgraph's ViewBox class with left-click ROI selection, shift/ctrl-click multi-ROI
    selection, and double-click zoom-to-fit functionality. All click logic is delegated to the
    orchestrator via installed callback handlers. Right-click reclassification is excluded
    because this is a read-only viewer.

    Args:
        panel: Identifies which image panel this view box belongs to.
        border: The panel border frame pen specification forwarded to ``fn.mkPen``.
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
        panel: int = 0,
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
        self._panel: int = panel

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
        if self._zoom_handler is not None:
            self._zoom_handler()

    def mouseClickEvent(self, ev: MouseClickEvent) -> None:  # noqa: N802
        """Dispatches left-click events to the installed click handler for ROI selection.

        Left-click selects the targeted ROI. Shift/ctrl-click toggles multi-ROI selection.
        Right-click is not handled (read-only viewer); unhandled right-clicks raise the default
        context menu.

        Notes:
            Overrides the pyqtgraph/Qt virtual method. The camelCase name is required to match
            the parent signature.
        """
        if self._click_handler is None:
            return

        # Only handles left-click in the read-only viewer.
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            if ev.button() == QtCore.Qt.MouseButton.RightButton and self.menuEnabled():
                self.raiseContextMenu(ev)
            return

        # Converts the scene-space click position to image-space pixel coordinates.
        position = self.mapSceneToView(ev.scenePos())
        click_x = int(position.x())
        click_y = int(position.y())

        # Extracts modifier state for the click handler.
        is_multi = ev.modifiers() in (
            QtCore.Qt.KeyboardModifier.ShiftModifier,
            QtCore.Qt.KeyboardModifier.ControlModifier,
        )

        self._click_handler(click_x, click_y, self._panel, is_multi)


def plot_trace(
    trace_box: TraceBox,
    *,
    cell_fluorescence: NDArray[np.float32],
    neuropil_fluorescence: NDArray[np.float32],
    spikes: NDArray[np.float32],
    frame_indices: NDArray,
    merge_indices: list[int],
    activity_mode: int,
    roi_colors: NDArray | None = None,
    traces_visible: bool = True,
    neuropil_visible: bool = True,
    deconvolved_visible: bool = True,
    scale_factor: float = CONFIG.default_scale_factor,
    max_plotted: int = CONFIG.default_plotted_count,
) -> tuple[float, float]:
    """Draws fluorescence traces for the selected ROIs.

    For a single selected ROI, displays the raw fluorescence, neuropil, and deconvolved
    traces on the same axes. For multiple selected ROIs, stacks normalized traces
    vertically with per-ROI coloring and an optional averaged summary at the bottom.

    Args:
        trace_box: The pyqtgraph PlotItem to draw traces on.
        cell_fluorescence: Cell fluorescence array with shape (roi_count, frame_count).
        neuropil_fluorescence: Neuropil fluorescence array with shape (roi_count, frame_count).
        spikes: Deconvolved spike array with shape (roi_count, frame_count).
        frame_indices: Time axis array with shape (frame_count,).
        merge_indices: Indices of the selected ROIs to display.
        activity_mode: Trace type index (0=F, 1=Fneu, 2=F-0.7*Fneu, 3=spks).
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
    trace_box.setYRange(y_minimum, y_maximum)
    return y_minimum, y_maximum


def _plot_single_trace(
    trace_box: pg.PlotItem,
    axis: pg.AxisItem,
    cell_fluorescence: NDArray[np.float32],
    neuropil_fluorescence: NDArray[np.float32],
    spikes: NDArray[np.float32],
    frame_indices: NDArray,
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
        trace_box.plot(frame_indices, fluorescence, pen="c")
    if neuropil_visible:
        trace_box.plot(frame_indices, neuropil, pen="r")
    if deconvolved_visible:
        trace_box.plot(
            frame_indices,
            spike_trace + y_minimum,
            pen=(255, 255, 255, STYLE.deconvolved_alpha),
        )

    axis.setTicks(None)
    return y_minimum, y_maximum


def _plot_multi_trace(
    trace_box: pg.PlotItem,
    axis: pg.AxisItem,
    cell_fluorescence: NDArray[np.float32],
    neuropil_fluorescence: NDArray[np.float32],
    spikes: NDArray[np.float32],
    frame_indices: NDArray,
    merge_indices: list[int],
    activity_mode: int,
    roi_colors: NDArray | None,
    scale_factor: float,
    max_plotted: int,
) -> tuple[float, float]:
    """Plots stacked traces for multiple selected ROIs.

    Args:
        trace_box: The plot item to draw on.
        axis: The left y-axis for tick configuration.
        cell_fluorescence: Cell fluorescence array with shape (roi_count, frame_count).
        neuropil_fluorescence: Neuropil fluorescence array with shape (roi_count, frame_count).
        spikes: Deconvolved spike array with shape (roi_count, frame_count).
        frame_indices: Time axis array.
        merge_indices: Indices of selected ROIs.
        activity_mode: Trace type index (0=F, 1=Fneu, 2=F-0.7*Fneu, 3=spks).
        roi_colors: Per-ROI RGB colors with shape (roi_count, 3).
        scale_factor: Vertical spacing factor for trace stacking.
        max_plotted: Maximum number of traces to display.

    Returns:
        Tuple of (y_minimum, y_maximum) for the plotted range.
    """
    selected = merge_indices[: min(len(merge_indices), max_plotted)]
    k_space = 1.0 / scale_factor
    tick_labels: list[tuple[float, str]] = []
    stack_position = len(selected) - 1
    average = np.zeros((cell_fluorescence.shape[1],))

    for index in selected[::-1]:
        # Selects trace based on activity mode.
        if activity_mode == 0:
            trace = cell_fluorescence[index, :]
        elif activity_mode == 1:
            trace = neuropil_fluorescence[index, :]
        elif activity_mode == CONFIG.activity_mode_subtracted:
            trace = cell_fluorescence[index, :] - CONFIG.neuropil_coefficient * neuropil_fluorescence[index, :]
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
        pen_color = roi_colors[index, :] if roi_colors is not None else (255, 255, 255)

        trace_box.plot(frame_indices, normalized + stack_position * k_space, pen=pen_color)
        tick_labels.append((stack_position * k_space + float(normalized.mean()), str(index)))
        stack_position -= 1

    # Computes average trace scale.
    average_scale = len(selected) / CONFIG.average_scale_divisor + 1
    average -= average.min()
    average_max = average.max()
    if average_max > 0:
        average /= average_max

    y_minimum = 0.0
    average_pen = (STYLE.average_gray, STYLE.average_gray, STYLE.average_gray)

    # Plots average trace at bottom when enough cells are selected.
    if len(selected) > CONFIG.average_threshold:
        trace_box.plot(
            frame_indices,
            -1 * average_scale + average * average_scale,
            pen=average_pen,
        )
        y_minimum = -1 * average_scale

    y_maximum = (len(selected) - 1) * k_space + 1
    axis.setTicks([tick_labels])
    return y_minimum, y_maximum
