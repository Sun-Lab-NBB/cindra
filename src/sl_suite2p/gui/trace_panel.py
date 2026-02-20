"""Provides the fluorescence trace display panel and its controls."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

import numpy as np
from PySide6 import QtGui, QtCore
from PySide6.QtWidgets import QLabel, QCheckBox, QComboBox, QLineEdit, QPushButton

from .styles import (
    COMBO_BOX_WIDTH,
    SMALL_EDIT_WIDTH,
    RED_LABEL_STYLESHEET,
    CYAN_LABEL_STYLESHEET,
    WHITE_LABEL_STYLESHEET,
    SQUARE_BUTTON_MAX_WIDTH,
    BUTTON_UNPRESSED_STYLESHEET,
    arrow_button_font,
)

if TYPE_CHECKING:
    import pyqtgraph as pg
    from numpy.typing import NDArray
    from PySide6.QtWidgets import QWidget, QGridLayout

    from .signals import GUISignals
    from .plot_widgets import TraceBox


# Default activity mode index (deconvolved).
_DEFAULT_ACTIVITY_MODE: int = 3

# Maximum number of traces that can be plotted simultaneously.
_MAX_PLOTTED_COUNT: int = 400

# Default number of traces plotted.
_DEFAULT_PLOTTED_COUNT: int = 40

# Default vertical scale factor for multi-trace stacking.
_DEFAULT_SCALE_FACTOR: float = 2.0

# Scale factor adjustment step per button press.
_SCALE_STEP: float = 0.5

# Minimum allowed scale factor.
_MIN_SCALE: float = 0.5

# Maximum allowed scale factor.
_MAX_SCALE: float = 10.0

# Default trace panel row stretch level.
_DEFAULT_TRACE_LEVEL: int = 1

# Minimum trace panel stretch level.
_MIN_TRACE_LEVEL: int = 1

# Maximum trace panel stretch level.
_MAX_TRACE_LEVEL: int = 5

# Activity mode index for neuropil-subtracted fluorescence (F - 0.7*Fneu).
_ACTIVITY_MODE_SUBTRACTED: int = 2

# Neuropil subtraction coefficient for the F - 0.7*Fneu activity mode.
_NEUROPIL_COEFFICIENT: float = 0.7

# Alpha value for the deconvolved trace pen.
_DECONVOLVED_ALPHA: int = 150

# Gray intensity for the average trace pen.
_AVERAGE_GRAY: int = 140

# Minimum number of selected cells before the average trace is displayed.
_AVERAGE_THRESHOLD: int = 5

# Ratio of selected cells to determine behavior/average trace vertical scale.
_BEHAVIOR_SCALE_DIVISOR: float = 25.0


@dataclass
class TraceControls:
    """Holds references to trace panel widgets and their mutable state.

    Replaces the scattered ``parent.comboBox``, ``parent.checkBoxd``, ``parent.sc``,
    and similar attributes that were previously stored directly on the MainWindow instance.

    Attributes:
        activity_combo: Combo box for selecting the activity mode.
        deconvolved_checkbox: Checkbox toggling deconvolved spike trace visibility.
        neuropil_checkbox: Checkbox toggling neuropil fluorescence trace visibility.
        traces_checkbox: Checkbox toggling raw fluorescence trace visibility.
        max_plotted_edit: Text input for the maximum number of plotted traces.
        arrow_buttons: Up/down buttons for resizing the trace panel.
        scale_buttons: +/- buttons for adjusting multi-trace vertical scale.
        scale_factor: Current vertical scale factor for multi-trace stacking.
        trace_level: Current row stretch factor for the trace panel.
        deconvolved_visible: Determines whether the deconvolved trace is drawn.
        neuropil_visible: Determines whether the neuropil trace is drawn.
        traces_visible: Determines whether the raw fluorescence trace is drawn.
    """

    activity_combo: QComboBox
    deconvolved_checkbox: QCheckBox
    neuropil_checkbox: QCheckBox
    traces_checkbox: QCheckBox
    max_plotted_edit: QLineEdit
    arrow_buttons: list[QPushButton] = field(default_factory=list)
    scale_buttons: list[QPushButton] = field(default_factory=list)
    scale_factor: float = _DEFAULT_SCALE_FACTOR
    trace_level: int = _DEFAULT_TRACE_LEVEL
    deconvolved_visible: bool = True
    neuropil_visible: bool = True
    traces_visible: bool = True


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
    scale_factor: float = _DEFAULT_SCALE_FACTOR,
    max_plotted: int = _DEFAULT_PLOTTED_COUNT,
    behavior: NDArray[np.float32] | None = None,
    behavior_time: NDArray[np.float32] | None = None,
    behavior_loaded: bool = False,
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
        behavior: Behavioral trace array, or None if not loaded.
        behavior_time: Time axis for the behavioral trace, or None if not loaded.
        behavior_loaded: Determines whether behavioral data has been loaded.

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
            behavior=behavior,
            behavior_time=behavior_time,
            behavior_loaded=behavior_loaded,
        )

    trace_box.update_range(
        frame_count=len(frame_indices),
        y_minimum=y_minimum,
        y_maximum=y_maximum,
    )
    trace_box.setYRange(y_minimum, y_maximum)
    return y_minimum, y_maximum


def create_trace_controls(
    owner: QWidget,
    layout: QGridLayout,
    row: int,
    signals: GUISignals,
) -> tuple[TraceControls, int]:
    """Creates trace panel controls and adds them to the layout.

    Builds the activity mode selector, trace resize buttons, scale buttons,
    max-plotted input, and trace visibility checkboxes. Connects all widget
    callbacks to the signal bus.

    Args:
        owner: The parent widget for ownership of created widgets.
        layout: The grid layout to add widgets to.
        row: Starting row index in the layout.
        signals: The central signal bus for GUI events.

    Returns:
        Tuple of (trace controls container, next available row index).
    """
    # Activity mode label and combo box.
    activity_label = QLabel("Activity mode:")
    activity_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
    layout.addWidget(activity_label, row, 0, 1, 1)

    activity_combo = QComboBox(owner)
    activity_combo.setFixedWidth(COMBO_BOX_WIDTH)
    layout.addWidget(activity_combo, row + 1, 0, 1, 1)
    activity_combo.addItem("F")
    activity_combo.addItem("Fneu")
    activity_combo.addItem("F - 0.7*Fneu")
    activity_combo.addItem("deconvolved")
    activity_combo.setCurrentIndex(_DEFAULT_ACTIVITY_MODE)
    activity_combo.currentIndexChanged.connect(signals.activity_mode_changed.emit)

    # Trace resize arrow buttons (up/down).
    arrow_up = QPushButton(" \u25b2")
    arrow_down = QPushButton(" \u25bc")
    arrow_buttons = [arrow_up, arrow_down]

    for button_index, button in enumerate(arrow_buttons):
        button.setMaximumWidth(SQUARE_BUTTON_MAX_WIDTH)
        button.setFont(arrow_button_font())
        button.setStyleSheet(BUTTON_UNPRESSED_STYLESHEET)
        layout.addWidget(button, row + button_index, 1, 1, 1, QtCore.Qt.AlignRight)

    # Scale adjustment buttons (+/-).
    scale_up = QPushButton(" +")
    scale_down = QPushButton(" -")
    scale_buttons = [scale_up, scale_down]

    for button_index, button in enumerate(scale_buttons):
        button.setMaximumWidth(SQUARE_BUTTON_MAX_WIDTH)
        button.setFont(arrow_button_font())
        button.setStyleSheet(BUTTON_UNPRESSED_STYLESHEET)
        layout.addWidget(button, row + button_index, 1, 1, 1)

    # Max plotted count label and input.
    max_plotted_label = QLabel("max # plotted:")
    max_plotted_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
    layout.addWidget(max_plotted_label, row + 2, 0, 1, 1)
    row += 3

    max_plotted_edit = QLineEdit(owner)
    max_plotted_edit.setValidator(QtGui.QIntValidator(0, _MAX_PLOTTED_COUNT))
    max_plotted_edit.setText(str(_DEFAULT_PLOTTED_COUNT))
    max_plotted_edit.setFixedWidth(SMALL_EDIT_WIDTH)
    max_plotted_edit.setAlignment(QtCore.Qt.AlignRight)
    layout.addWidget(max_plotted_edit, row, 0, 1, 1)

    # Trace visibility checkboxes.
    layout.setVerticalSpacing(4)

    deconvolved_checkbox = QCheckBox("deconv [N]")
    deconvolved_checkbox.setStyleSheet(WHITE_LABEL_STYLESHEET)
    deconvolved_checkbox.toggle()
    layout.addWidget(deconvolved_checkbox, row, 3, 1, 2)

    neuropil_checkbox = QCheckBox("neuropil [B]")
    neuropil_checkbox.setStyleSheet(RED_LABEL_STYLESHEET)
    neuropil_checkbox.toggle()
    layout.addWidget(neuropil_checkbox, row, 5, 1, 2)

    traces_checkbox = QCheckBox("raw fluor [V]")
    traces_checkbox.setStyleSheet(CYAN_LABEL_STYLESHEET)
    traces_checkbox.toggle()
    layout.addWidget(traces_checkbox, row, 7, 1, 2)

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

    # Connects callbacks using closures that capture the controls and signals instances.
    arrow_up.clicked.connect(lambda: _expand_trace(controls=controls, signals=signals))
    arrow_down.clicked.connect(lambda: _collapse_trace(controls=controls, signals=signals))
    scale_up.clicked.connect(lambda: _expand_scale(controls=controls, signals=signals))
    scale_down.clicked.connect(lambda: _collapse_scale(controls=controls, signals=signals))
    max_plotted_edit.returnPressed.connect(signals.trace_needs_update.emit)
    deconvolved_checkbox.toggled.connect(lambda: _on_deconvolved_toggle(controls=controls, signals=signals))
    neuropil_checkbox.toggled.connect(lambda: _on_neuropil_toggle(controls=controls, signals=signals))
    traces_checkbox.toggled.connect(lambda: _on_traces_toggle(controls=controls, signals=signals))

    return controls, row


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
            pen=(255, 255, 255, _DECONVOLVED_ALPHA),
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
    behavior: NDArray[np.float32] | None,
    behavior_time: NDArray[np.float32] | None,
    behavior_loaded: bool,
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
        behavior: Behavioral trace array, or None.
        behavior_time: Time axis for the behavioral trace, or None.
        behavior_loaded: Determines whether behavioral data is available.

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
        elif activity_mode == _ACTIVITY_MODE_SUBTRACTED:
            trace = cell_fluorescence[index, :] - _NEUROPIL_COEFFICIENT * neuropil_fluorescence[index, :]
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

    # Computes average trace and behavior scale.
    behavior_scale = len(selected) / _BEHAVIOR_SCALE_DIVISOR + 1
    average -= average.min()
    average_max = average.max()
    if average_max > 0:
        average /= average_max

    y_minimum = 0.0
    average_pen = (_AVERAGE_GRAY, _AVERAGE_GRAY, _AVERAGE_GRAY)

    # Plots average trace at bottom when enough cells are selected.
    if len(selected) > _AVERAGE_THRESHOLD:
        trace_box.plot(
            frame_indices,
            -1 * behavior_scale + average * behavior_scale,
            pen=average_pen,
        )
        y_minimum = -1 * behavior_scale

    # Overlays behavioral trace when loaded.
    if behavior_loaded and behavior is not None and behavior_time is not None:
        trace_box.plot(
            frame_indices,
            -1 * behavior_scale + average * behavior_scale,
            pen=average_pen,
        )
        trace_box.plot(
            behavior_time,
            -1 * behavior_scale + behavior * behavior_scale,
            pen="w",
        )
        y_minimum = -1 * behavior_scale

    y_maximum = (len(selected) - 1) * k_space + 1
    axis.setTicks([tick_labels])
    return y_minimum, y_maximum


def _expand_scale(controls: TraceControls, signals: GUISignals) -> None:
    """Increases the vertical scale factor for multi-trace stacking.

    Args:
        controls: The trace controls container.
        signals: The central signal bus for GUI events.
    """
    controls.scale_factor = min(_MAX_SCALE, controls.scale_factor + _SCALE_STEP)
    signals.trace_needs_update.emit()


def _collapse_scale(controls: TraceControls, signals: GUISignals) -> None:
    """Decreases the vertical scale factor for multi-trace stacking.

    Args:
        controls: The trace controls container.
        signals: The central signal bus for GUI events.
    """
    controls.scale_factor = max(_MIN_SCALE, controls.scale_factor - _SCALE_STEP)
    signals.trace_needs_update.emit()


def _expand_trace(controls: TraceControls, signals: GUISignals) -> None:
    """Increases the trace panel row stretch factor.

    Args:
        controls: The trace controls container.
        signals: The central signal bus for GUI events.
    """
    controls.trace_level = min(_MAX_TRACE_LEVEL, controls.trace_level + 1)
    signals.trace_needs_update.emit()


def _collapse_trace(controls: TraceControls, signals: GUISignals) -> None:
    """Decreases the trace panel row stretch factor.

    Args:
        controls: The trace controls container.
        signals: The central signal bus for GUI events.
    """
    controls.trace_level = max(_MIN_TRACE_LEVEL, controls.trace_level - 1)
    signals.trace_needs_update.emit()


def _on_deconvolved_toggle(controls: TraceControls, signals: GUISignals) -> None:
    """Handles the deconvolved trace visibility checkbox toggle.

    Args:
        controls: The trace controls container.
        signals: The central signal bus for GUI events.
    """
    controls.deconvolved_visible = controls.deconvolved_checkbox.isChecked()
    signals.trace_needs_update.emit()


def _on_neuropil_toggle(controls: TraceControls, signals: GUISignals) -> None:
    """Handles the neuropil trace visibility checkbox toggle.

    Args:
        controls: The trace controls container.
        signals: The central signal bus for GUI events.
    """
    controls.neuropil_visible = controls.neuropil_checkbox.isChecked()
    signals.trace_needs_update.emit()


def _on_traces_toggle(controls: TraceControls, signals: GUISignals) -> None:
    """Handles the raw fluorescence trace visibility checkbox toggle.

    Args:
        controls: The trace controls container.
        signals: The central signal bus for GUI events.
    """
    controls.traces_visible = controls.traces_checkbox.isChecked()
    signals.trace_needs_update.emit()
