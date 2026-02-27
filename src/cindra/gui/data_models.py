"""Provides widget-reference and rendering-state dataclasses shared by the ROI viewer and editor."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

import numpy as np

from .constants import CONFIG

if TYPE_CHECKING:
    import pyqtgraph as pg  # type: ignore[import-untyped]
    from numpy.typing import NDArray
    from PySide6.QtWidgets import QLabel, QCheckBox, QComboBox, QLineEdit, QPushButton, QButtonGroup

    from .widgets import RangeSlider


@dataclass
class ColorControls:
    """Holds references to color statistic control widgets.

    Attributes:
        color_buttons: Button group for selecting the active color statistic.
        colormap_chooser: Dropdown for selecting the active colormap.
        classifier_edit: Text input for the classifier probability threshold.
        bin_edit: Text input for the binning size.
    """

    color_buttons: QButtonGroup
    colormap_chooser: QComboBox
    classifier_edit: QLineEdit
    bin_edit: QLineEdit


@dataclass
class ColorbarWidgets:
    """Holds references to the colorbar display widgets.

    Attributes:
        image: The pyqtgraph image item displaying the colorbar gradient.
        labels: The three label items showing the low, mid, and high colorbar values.
        widget: The pyqtgraph GraphicsLayoutWidget containing the colorbar.
    """

    image: pg.ImageItem
    labels: list[pg.LabelItem]
    widget: pg.GraphicsLayoutWidget


@dataclass
class ColorArrays:
    """Holds all computed color data for ROI overlay rendering.

    Attributes:
        cols: Per-statistic RGB colors with shape (color_count, roi_count, 3).
        istat: Per-statistic normalized values with shape (color_count, roi_count).
        colorbar: Per-statistic colorbar range values as [low, mid, high] lists.
        rgb: RGBA overlay arrays with shape (color_count, height, width, 4).
        random_hues: Per-ROI random hue values with shape (roi_count,).
    """

    cols: NDArray[np.uint8]
    istat: NDArray[np.float32]
    colorbar: list[list[float]]
    rgb: NDArray[np.uint8]
    random_hues: NDArray[np.float64]


@dataclass
class ROIIndexMaps:
    """Holds the multi-layer ROI index and weight maps for overlay rendering.

    Attributes:
        sroi: Boolean presence map with shape (height, width).
        lam: Weight layers with shape (3, height, width).
        iroi: ROI index layers with shape (3, height, width).
        lam_mean: Mean weight across all ROI pixels.
        lam_norm: Normalized weights with shape (height, width).
        text_labels: Per-ROI text label items for centroid display.
    """

    sroi: NDArray[np.bool_]
    lam: NDArray[np.float32]
    iroi: NDArray[np.int32]
    lam_mean: float
    lam_norm: NDArray[np.float32]
    text_labels: list[pg.TextItem] = field(default_factory=list)


@dataclass
class ViewControls:
    """Holds references to background view panel widgets.

    Attributes:
        view_buttons: Button group for mutually exclusive view selection.
        range_slider: Dual-handle slider controlling image saturation levels.
        view_names: Display names for each view mode.
    """

    view_buttons: QButtonGroup
    range_slider: RangeSlider
    view_names: list[str] = field(default_factory=lambda: list(CONFIG.view_names))


@dataclass
class TraceControls:
    """Holds references to trace panel widgets and their mutable state.

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
    scale_factor: float = CONFIG.default_scale_factor
    trace_level: int = CONFIG.default_trace_level
    deconvolved_visible: bool = True
    neuropil_visible: bool = True
    traces_visible: bool = True


@dataclass
class SelectionControls:
    """Holds references to cell selection widgets and their mutable state.

    Attributes:
        selection_buttons: Button group with draw/top-n/bottom-n selection modes.
        top_count_edit: Text input for the number of top/bottom cells to select.
        top_count: Current top-n/bottom-n count value.
    """

    selection_buttons: QButtonGroup
    top_count_edit: QLineEdit
    top_count: int = CONFIG.default_top_n


@dataclass
class ClassifierControls:
    """Holds references to classifier section widgets.

    Attributes:
        classifier_label: Label displaying the current classifier name or status.
        add_to_class_button: Button for adding the current session data to the classifier.
    """

    classifier_label: QLabel
    add_to_class_button: QPushButton
