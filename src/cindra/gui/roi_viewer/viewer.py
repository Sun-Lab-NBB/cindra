"""Provides the self-contained read-only ROI viewer window for inspecting single-day pipeline results.

This module merges view-state enums, style constants, dataclasses, background-view helpers,
ROI-overlay color/mask functions, trace-plot helpers, and custom Qt widgets into a single
file.  The ``ROIViewer`` class (defined at the bottom of this file) orchestrates all
rendering and user interaction; every other symbol is a private helper or public dataclass
consumed exclusively within this module.

Design constraints
------------------
* **Self-contained** -- zero imports from other ``gui.*`` packages.
* **Read-only** -- no ROI mutation functions (flip, add, remove, redraw, merge).
* **No signal bus** -- callbacks are wired directly; ``GUISignals`` is eliminated.
"""

from __future__ import annotations

from enum import IntEnum
from typing import TYPE_CHECKING
from pathlib import Path
from contextlib import suppress
from dataclasses import field, dataclass
from collections.abc import Callable

import cindra
import numpy as np
from PySide6 import QtGui, QtCore
import pyqtgraph as pg  # type: ignore[import-untyped]
from pyqtgraph import functions as fn
from PySide6.QtGui import QPainter, QMouseEvent, QPaintEvent
import matplotlib.cm
from PySide6.QtWidgets import (
    QLabel,
    QStyle,
    QSlider,
    QWidget,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QLineEdit,
    QStatusBar,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QApplication,
    QButtonGroup,
    QStyleOptionSlider,
)
from matplotlib.colors import hsv_to_rgb
from ataraxis_base_utilities import LogLevel, console
from pyqtgraph.graphicsItems.ViewBox.ViewBoxMenu import ViewBoxMenu  # type: ignore[import-untyped]

from .context_data import ROIViewerData

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pyqtgraph.GraphicsScene.mouseEvents import MouseClickEvent  # type: ignore[import-untyped]


# ============================================================ #
#  Module-level constants (merged from all original modules)    #
# ============================================================ #

# --- roi_overlays constants ---

_COLOR_EDIT_WIDTH: int = 65
"""The width for color panel edit fields and the colormap combo box."""

_OVERLAP_LAYERS: int = 3
"""The number of overlap layers stored in the ROI index map."""

_LAMBDA_NORM_SCALE: float = 0.75
"""The ROI weight normalization scale factor."""

_LAMBDA_THRESHOLD: float = 1e-10
"""The minimum lambda value threshold for computing the mean weight."""

_COLOR_STAT_COUNT: int = 8
"""The number of color statistics (random, skew, compact, footprint, aspect, chan2, class, corr)."""

_FIXED_COLORBAR_RANGE: list[float] = [0.0, 0.5, 1.0]
"""The fixed colorbar range for statistics without computed percentiles."""

_LOWER_PERCENTILE: float = 2.0
"""The lower percentile value for computing istat normalization bounds."""

_UPPER_PERCENTILE: float = 98.0
"""The upper percentile value for computing istat normalization bounds."""

_CHAN2_COLOR_DIVISOR: float = 1.4
"""The random color adjustment divisor for channel 2 data."""

_CHAN2_COLOR_OFFSET: float = 0.1
"""The random color adjustment offset for channel 2 data."""

_HSV_DIVISOR: float = 1.4
"""The HSV transform normalization divisor."""

_HSV_OFFSET: float = 0.4
"""The HSV transform normalization offset."""

_COLORBAR_SAMPLE_COUNT: int = 101
"""The number of samples for the colorbar gradient."""

_COLORBAR_ROW_COUNT: int = 20
"""The number of rows in the colorbar image."""

_COLORMAPS: list[str] = [
    "hsv",
    "viridis",
    "plasma",
    "inferno",
    "magma",
    "cividis",
    "viridis_r",
    "plasma_r",
    "inferno_r",
    "magma_r",
    "cividis_r",
]
"""The available colormaps for the colormap chooser."""

_COLOR_NAMES: list[str] = [
    "A: random",
    "S: skew",
    "D: compact",
    "F: footprint",
    "G: aspect_ratio",
    "H: chan2_prob",
    "J: classifier, cell prob=",
    "K: correlations, bin=",
]
"""The color statistic names displayed on the color buttons."""

_COLOR_SHORT_NAMES: list[str] = [name[3:] for name in _COLOR_NAMES]
"""The short names extracted from the color buttons (after the keyboard shortcut prefix)."""

_COLOR_NARROW_RANGE_START: int = 5
"""The starting index of the color buttons that require the adjacent edit field column."""

_COLOR_NARROW_RANGE_END: int = 8
"""The ending index of the color buttons that require the adjacent edit field column."""

_COLOR_CHAN2: int = 5
"""The channel 2 color index."""

_COLOR_CLASSIFIER: int = 6
"""The classifier probability color index."""

_COLOR_CORRELATION: int = 7
"""The correlation color index."""

_CHAN2_THRESHOLD_EPSILON: float = 1e-3
"""The minimum change in channel 2 threshold to trigger a recoloring update."""

_ROI_TEXT_SIZE: int = 8
"""The font size for ROI text labels."""

_ROI_TEXT_COLOR: tuple[int, int, int] = (180, 180, 180)
"""The color for ROI text labels."""

_STAT_FIELD_MAP: dict[str, str] = {
    "skew": "skewness",
    "compact": "compactness",
    "footprint": "footprint",
    "aspect_ratio": "aspect_ratio",
    "chan2_prob": "colocalization_probability",
}
"""The mapping from color statistic display names to ROIStatistics attribute names."""

# --- background_views constants ---

_VIEW_COUNT: int = 7
"""The number of background view types available."""

_VIEW_ROIS: int = 0
"""The index for the ROI overlay view (no background image)."""

_VIEW_MEAN: int = 1
"""The index for the mean image view."""

_VIEW_ENHANCED: int = 2
"""The index for the enhanced mean image view."""

_VIEW_CORRELATION: int = 3
"""The index for the correlation map view."""

_VIEW_MAX_PROJECTION: int = 4
"""The index for the maximum projection view."""

_VIEW_CHAN2_CORRECTED: int = 5
"""The index for the corrected channel 2 mean image view."""

_VIEW_CHAN2_RAW: int = 6
"""The index for the raw channel 2 mean image view."""

_VIEW_NAMES: list[str] = [
    "Q: ROIs",
    "W: mean img",
    "E: mean img (enhanced)",
    "R: correlation map",
    "T: max projection",
    "Y: mean img chan2, corr",
    "U: mean img chan2",
]
"""The names displayed on view selection buttons, with keyboard shortcut prefixes."""

# --- trace_panel constants ---

_DEFAULT_ACTIVITY_MODE: int = 3
"""The default activity mode index (deconvolved)."""

_MAX_PLOTTED_COUNT: int = 400
"""The maximum number of traces that can be plotted simultaneously."""

_DEFAULT_PLOTTED_COUNT: int = 40
"""The default number of traces plotted."""

_DEFAULT_SCALE_FACTOR: float = 2.0
"""The default vertical scale factor for multi-trace stacking."""

_SCALE_STEP: float = 0.5
"""The scale factor adjustment step per button press."""

_MIN_SCALE: float = 0.5
"""The minimum allowed scale factor."""

_MAX_SCALE: float = 10.0
"""The maximum allowed scale factor."""

_DEFAULT_TRACE_LEVEL: int = 1
"""The default trace panel row stretch level."""

_MIN_TRACE_LEVEL: int = 1
"""The minimum trace panel stretch level."""

_MAX_TRACE_LEVEL: int = 5
"""The maximum trace panel stretch level."""

_ACTIVITY_MODE_SUBTRACTED: int = 2
"""The activity mode index for neuropil-subtracted fluorescence (F - 0.7*Fneu)."""

_NEUROPIL_COEFFICIENT: float = 0.7
"""The neuropil subtraction coefficient for the F - 0.7*Fneu activity mode."""

_DECONVOLVED_ALPHA: int = 150
"""The alpha value for the deconvolved trace pen."""

_AVERAGE_GRAY: int = 140
"""The gray intensity for the average trace pen."""

_AVERAGE_THRESHOLD: int = 5
"""The minimum number of selected cells before the average trace is displayed."""

_AVERAGE_SCALE_DIVISOR: float = 25.0
"""The ratio of selected cells to determine average trace vertical scale."""

# --- selection_buttons constants ---

_STRETCH_FACTOR: int = 100
"""The layout stretch factor applied when toggling single-panel vs dual-panel view."""

_MAX_TOP_N: int = 500
"""The maximum number of cells allowed in the top-n / bottom-n selection input."""

_DEFAULT_TOP_N: int = 40
"""The default number of cells selected by top-n / bottom-n."""

_QUADRANT_ZOOM_MARGIN: float = 0.15
"""The margin fraction added to quadrant zoom ranges to provide padding."""

_QUADRANT_COLUMNS: int = 3
"""The number of columns in the quadrant grid."""

_VIEW_CELLS_ONLY: int = 0
"""The view mode index for displaying cells only."""

_VIEW_BOTH: int = 1
"""The view mode index for displaying both cells and non-cells."""

_VIEW_NONCELLS_ONLY: int = 2
"""The view mode index for displaying non-cells only."""

# --- viewer constants ---

_CELLS_PLOT: int = 0
"""The panel index for the cells image panel."""

_NONCELLS_PLOT: int = 1
"""The panel index for the non-cells image panel."""

_CENTROID_STAT_INDEX: int = 1
"""The 1-based stat index for the centroid field (used to display ROI position)."""

_PIXEL_COUNT_STAT_INDEX: int = 2
"""The 1-based stat index for the pixel count field."""

# --- context_loader constants ---

_DEFAULT_CHANNEL_2_THRESHOLD: float = 0.6
"""The default colocalization threshold for channel 2 data."""

_BIN_SIZE_DIVISOR: int = 2
"""The divisor for computing the temporal bin size from tau * sampling_rate."""


# ============================================================ #
#  Enums                                                        #
# ============================================================ #

class ROIColorMode(IntEnum):
    """Selects the statistic used to color ROI overlays in the image panels."""

    RANDOM = 0
    """Assigns each ROI a random color from the active colormap."""

    SKEWNESS = 1
    """Colors ROIs by the skewness of their spatial footprint pixel distribution."""

    COMPACTNESS = 2
    """Colors ROIs by the compactness (circularity) of their spatial footprint."""

    FOOTPRINT = 3
    """Colors ROIs by their total spatial footprint area in pixels."""

    ASPECT_RATIO = 4
    """Colors ROIs by the aspect ratio of their bounding ellipse."""

    COLOCALIZATION_PROBABILITY = 5
    """Colors ROIs by their channel 2 colocalization probability."""

    CLASSIFIER_PROBABILITY = 6
    """Colors ROIs by the trained classifier's cell-probability estimate."""

    CORRELATIONS = 7
    """Colors ROIs by pairwise activity correlation with the selected ROI."""


class BackgroundView(IntEnum):
    """Selects the background image displayed behind ROI overlays in the image panels."""

    ROIS_ONLY = 0
    """Displays a blank background with ROI overlays only."""

    # Channel 1 reference images.
    MEAN_IMAGE = 1
    """Displays the temporal mean of all registered channel 1 frames."""

    ENHANCED_MEAN_IMAGE = 2
    """Displays the high-pass filtered channel 1 mean image used for cell boundary detection."""

    CORRELATION_MAP = 3
    """Displays the pixel-wise activity correlation map computed during channel 1 detection."""

    MAXIMUM_PROJECTION = 4
    """Displays the maximum intensity projection across all channel 1 frames."""

    # Channel 2 reference images.
    MEAN_IMAGE_CHANNEL_2 = 5
    """Displays the temporal mean of all registered channel 2 frames."""

    ENHANCED_MEAN_IMAGE_CHANNEL_2 = 6
    """Displays the high-pass filtered channel 2 mean image used for cell boundary detection."""

    CORRELATION_MAP_CHANNEL_2 = 7
    """Displays the pixel-wise activity correlation map computed during channel 2 detection."""

    MAXIMUM_PROJECTION_CHANNEL_2 = 8
    """Displays the maximum intensity projection across all channel 2 frames."""

    # Structural reference images.
    CORRECTED_STRUCTURAL_MEAN_IMAGE = 9
    """Displays the bleed-through-corrected structural channel mean image computed during
    functional-to-structural channel colocalization."""


class TraceMode(IntEnum):
    """Selects the fluorescence trace type displayed in the trace panel."""

    RAW_FLUORESCENCE = 0
    """Displays the raw cell_fluorescence trace."""

    NEUROPIL = 1
    """Displays the neuropil_fluorescence trace."""

    NEUROPIL_CORRECTED = 2
    """Displays the neuropil-corrected trace (cell_fluorescence - neuropil_coefficient *
    neuropil_fluorescence)."""

    DECONVOLVED = 3
    """Displays the deconvolved spikes trace."""


class ROIToolPanel(IntEnum):
    """Identifies which image panel hosts the rectangular ROI selection tool."""

    CELLS = 0
    """The ROI selection tool is active on the cell image panel."""

    NON_CELLS = 1
    """The ROI selection tool is active on the non-cell image panel."""


# ============================================================ #
#  Style dataclass + font factories                             #
# ============================================================ #

@dataclass(frozen=True, slots=True)
class _Style:
    """Encapsulates visual and dimensional constants for the ROI viewer window."""

    main_window: str = "QMainWindow {background: 'black';}"
    """Stylesheet applied to the main window background."""
    button_pressed: str = "QPushButton {Text-align: left; background-color: rgb(100,50,100); color:white;}"
    """Stylesheet for a button in the pressed (active/selected) state."""
    button_unpressed: str = "QPushButton {Text-align: left; background-color: rgb(50,50,50); color:white;}"
    """Stylesheet for a button in the unpressed (enabled, not selected) state."""
    button_inactive: str = "QPushButton {Text-align: left; background-color: rgb(50,50,50); color:gray;}"
    """Stylesheet for a button in the inactive (disabled/grayed-out) state."""
    white_label: str = "color: white;"
    """Stylesheet for white label text on a dark background."""
    red_label: str = "color: red;"
    """Stylesheet for red label text (used for neuropil trace indicators)."""
    cyan_label: str = "color: cyan;"
    """Stylesheet for cyan label text (used for raw fluorescence trace indicators)."""
    range_slider: str = (
        "QSlider::handle:horizontal {"
        "background-color: white;"
        "border: 1px solid #5c5c5c;"
        "border-radius: 0px;"
        "border-color: black;"
        "height: 8px;"
        "width: 6px;"
        "margin: -8px 2;"
        "}"
    )
    """Stylesheet for the dual-handle range slider used in saturation controls."""
    small_edit_width: int = 35
    """Width for small input fields and quadrant buttons (top-n count, diameter, max plotted count)."""
    roi_edit_width: int = 45
    """Width for ROI index input fields."""
    medium_edit_width: int = 60
    """Width for medium-sized widgets in the ROI editor (add-ROI button, diameter label)."""
    parameter_edit_width: int = 90
    """Width for parameter input fields in the merge dialog."""
    combo_box_width: int = 100
    """Width for the activity mode combo box in the trace panel."""
    square_button_max_width: int = 22
    """Maximum width for small square buttons (trace arrows, scale buttons)."""
    colorbar_max_height: int = 60
    """Maximum height for the colorbar widget."""
    colorbar_max_width: int = 150
    """Maximum width for the colorbar widget."""
    font_family: str = "Arial"
    """Standard font family used throughout the GUI."""
    alternative_font_family: str = "Times"
    """Alternative font family used for colorbar and merge dialog labels."""


STYLE: _Style = _Style()
"""Module-level singleton providing all ROI viewer style constants."""


def label_font() -> QtGui.QFont:
    """Creates the standard label font (Arial 8pt)."""
    # noinspection PyArgumentList
    return QtGui.QFont(family=STYLE.font_family, pointSize=8)


def label_font_bold() -> QtGui.QFont:
    """Creates the standard bold label font (Arial 8pt bold)."""
    # noinspection PyArgumentList
    return QtGui.QFont(family=STYLE.font_family, pointSize=8, weight=QtGui.QFont.Weight.Bold)


def header_font() -> QtGui.QFont:
    """Creates the section header font (Arial 10pt bold)."""
    # noinspection PyArgumentList
    return QtGui.QFont(family=STYLE.font_family, pointSize=10, weight=QtGui.QFont.Weight.Bold)


def arrow_button_font() -> QtGui.QFont:
    """Creates the font for trace expand/collapse arrow buttons (Arial 11pt bold)."""
    # noinspection PyArgumentList
    return QtGui.QFont(family=STYLE.font_family, pointSize=11, weight=QtGui.QFont.Weight.Bold)


def colorbar_font() -> QtGui.QFont:
    """Creates the font for colorbar tick labels (Times 8pt bold)."""
    # noinspection PyArgumentList
    return QtGui.QFont(family=STYLE.alternative_font_family, pointSize=8, weight=QtGui.QFont.Weight.Bold)


def mergelabel_font() -> QtGui.QFont:
    """Creates the font for merge dialog parameter labels (Times bold)."""
    # noinspection PyArgumentList
    return QtGui.QFont(family=STYLE.alternative_font_family, weight=QtGui.QFont.Weight.Bold)


# ============================================================ #
#  Data / control dataclasses                                   #
# ============================================================ #

@dataclass
class ColorControls:
    """Holds references to color statistic control widgets.

    Attributes:
        color_buttons: Button group for selecting the active color statistic.
        colormap_chooser: Dropdown for selecting the active colormap.
        channel_2_edit: Text input for the channel 2 probability threshold.
        classifier_edit: Text input for the classifier probability threshold.
        bin_edit: Text input for the binning size.
    """

    color_buttons: QButtonGroup
    colormap_chooser: QComboBox
    channel_2_edit: QLineEdit
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
        rgb: RGBA overlay arrays with shape (2, color_count, height, width, 4).
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
        sroi: Boolean presence map with shape (2, height, width).
        lam: Weight layers with shape (2, 3, height, width).
        iroi: ROI index layers with shape (2, 3, height, width).
        lam_mean: Mean weight across all ROI pixels.
        lam_norm: Normalized weights with shape (2, height, width).
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
    view_names: list[str] = field(default_factory=lambda: list(_VIEW_NAMES))


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
    scale_factor: float = _DEFAULT_SCALE_FACTOR
    trace_level: int = _DEFAULT_TRACE_LEVEL
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
    top_count: int = _DEFAULT_TOP_N


@dataclass
class CellToggleControls:
    """Holds references to cell/non-cell/both toggle widgets and ROI count labels.

    Attributes:
        size_buttons: Button group with cells/both/non-cells toggle modes.
        cell_count_label: Label showing the number of classified cells.
        noncell_count_label: Label showing the number of non-cells.
    """

    size_buttons: QButtonGroup
    cell_count_label: QLabel
    noncell_count_label: QLabel


@dataclass
class QuadrantControls:
    """Holds references to quadrant zoom navigation widgets.

    Attributes:
        quadrant_buttons: Button group with the 3x3 quadrant zoom buttons.
    """

    quadrant_buttons: QButtonGroup


# ============================================================ #
#  Background view helpers                                      #
# ============================================================ #

def build_views(
    frame_height: int,
    frame_width: int,
    *,
    mean_image: NDArray[np.float32] | None = None,
    enhanced_mean_image: NDArray[np.float32] | None = None,
    correlation_map: NDArray[np.float32] | None = None,
    maximum_projection: NDArray[np.float32] | None = None,
    corrected_channel_2_image: NDArray[np.float32] | None = None,
    channel_2_mean_image: NDArray[np.float32] | None = None,
    valid_y_range: tuple[int, int] | None = None,
    valid_x_range: tuple[int, int] | None = None,
) -> NDArray[np.uint8]:
    """Builds the background view stack from detection images.

    Creates a stack of 7 RGB background images, each normalized to [0, 255] uint8 range.
    Views are indexed as: 0=ROIs (black), 1=mean, 2=enhanced mean, 3=correlation map,
    4=max projection, 5=corrected channel 2, 6=raw channel 2.

    Args:
        frame_height: Height of the field of view in pixels.
        frame_width: Width of the field of view in pixels.
        mean_image: Mean fluorescence image.
        enhanced_mean_image: Contrast-enhanced mean image.
        correlation_map: Pixel correlation map.
        maximum_projection: Maximum intensity projection.
        corrected_channel_2_image: Corrected channel 2 mean image.
        channel_2_mean_image: Raw channel 2 mean image.
        valid_y_range: Tuple of (start, end) row indices for the valid image region.
        valid_x_range: Tuple of (start, end) column indices for the valid image region.

    Returns:
        Array of shape (7, frame_height, frame_width, 3) containing uint8 RGB views.
    """
    views = np.zeros((_VIEW_COUNT, frame_height, frame_width, 3), dtype=np.float32)

    for view_index in range(_VIEW_COUNT):
        image = _build_single_view(
            view_index=view_index,
            frame_height=frame_height,
            frame_width=frame_width,
            mean_image=mean_image,
            enhanced_mean_image=enhanced_mean_image,
            correlation_map=correlation_map,
            maximum_projection=maximum_projection,
            corrected_channel_2_image=corrected_channel_2_image,
            channel_2_mean_image=channel_2_mean_image,
            valid_y_range=valid_y_range,
            valid_x_range=valid_x_range,
        )
        image_uint8 = (image * 255).astype(np.uint8)
        views[view_index] = np.tile(image_uint8[:, :, np.newaxis], (1, 1, 3))

    return views.astype(np.uint8)


def display_views(
    view1: pg.ImageItem,
    view2: pg.ImageItem,
    views: NDArray[np.uint8],
    view_index: int,
    saturation: list[int],
) -> None:
    """Displays the selected background view on both image panels.

    Args:
        view1: The cell panel background image item.
        view2: The non-cell panel background image item.
        views: The full view stack of shape (7, height, width, 3).
        view_index: Index of the view to display (0-6).
        saturation: Two-element list of [low, high] saturation levels.
    """
    view1.setImage(views[view_index], levels=saturation)
    view2.setImage(views[view_index], levels=saturation)
    view1.show()
    view2.show()


def _build_single_view(
    view_index: int,
    frame_height: int,
    frame_width: int,
    mean_image: NDArray[np.float32] | None,
    enhanced_mean_image: NDArray[np.float32] | None,
    correlation_map: NDArray[np.float32] | None,
    maximum_projection: NDArray[np.float32] | None,
    corrected_channel_2_image: NDArray[np.float32] | None,
    channel_2_mean_image: NDArray[np.float32] | None,
    valid_y_range: tuple[int, int] | None,
    valid_x_range: tuple[int, int] | None,
) -> NDArray[np.float32]:
    """Builds a single background view image normalized to [0, 1].

    Args:
        view_index: Index of the view to build (0-6).
        frame_height: Height of the field of view in pixels.
        frame_width: Width of the field of view in pixels.
        mean_image: Mean fluorescence image.
        enhanced_mean_image: Contrast-enhanced mean image.
        correlation_map: Pixel correlation map.
        maximum_projection: Maximum intensity projection.
        corrected_channel_2_image: Corrected channel 2 mean image.
        channel_2_mean_image: Raw channel 2 mean image.
        valid_y_range: Tuple of (start, end) row indices for the valid image region.
        valid_x_range: Tuple of (start, end) column indices for the valid image region.

    Returns:
        Normalized image of shape (frame_height, frame_width) with values in [0, 1].
    """
    if view_index == _VIEW_ROIS:
        return np.zeros((frame_height, frame_width), dtype=np.float32)

    if view_index == _VIEW_MEAN:
        return _normalize_percentile(
            image=mean_image,
            frame_height=frame_height,
            frame_width=frame_width,
        )

    if view_index == _VIEW_ENHANCED:
        return _normalize_percentile(
            image=enhanced_mean_image,
            frame_height=frame_height,
            frame_width=frame_width,
        )

    if view_index == _VIEW_CORRELATION:
        return _place_in_valid_region(
            image=correlation_map,
            frame_height=frame_height,
            frame_width=frame_width,
            valid_y_range=valid_y_range,
            valid_x_range=valid_x_range,
        )

    if view_index == _VIEW_MAX_PROJECTION:
        return _place_in_valid_region(
            image=maximum_projection,
            frame_height=frame_height,
            frame_width=frame_width,
            valid_y_range=valid_y_range,
            valid_x_range=valid_x_range,
            warn_on_error=True,
        )

    if view_index == _VIEW_CHAN2_CORRECTED:
        return _normalize_percentile(
            image=corrected_channel_2_image,
            frame_height=frame_height,
            frame_width=frame_width,
        )

    if view_index == _VIEW_CHAN2_RAW:
        return _normalize_percentile(
            image=channel_2_mean_image,
            frame_height=frame_height,
            frame_width=frame_width,
        )

    return np.zeros((frame_height, frame_width), dtype=np.float32)


def _normalize_percentile(
    image: NDArray[np.float32] | None,
    frame_height: int,
    frame_width: int,
) -> NDArray[np.float32]:
    """Normalizes an image to [0, 1] using 1st and 99th percentile clipping.

    Args:
        image: Input image to normalize, or None.
        frame_height: Height for the fallback zero image.
        frame_width: Width for the fallback zero image.

    Returns:
        Normalized image with values clipped to [0, 1].
    """
    if image is None:
        return np.zeros((frame_height, frame_width), dtype=np.float32)

    percentile_1 = np.percentile(image, 1)
    percentile_99 = np.percentile(image, 99)

    if percentile_99 <= percentile_1:
        return np.zeros((frame_height, frame_width), dtype=np.float32)

    normalized = (image - percentile_1) / (percentile_99 - percentile_1)
    return np.clip(normalized, 0, 1).astype(np.float32)


def _place_in_valid_region(
    image: NDArray[np.float32] | None,
    frame_height: int,
    frame_width: int,
    valid_y_range: tuple[int, int] | None,
    valid_x_range: tuple[int, int] | None,
    warn_on_error: bool = False,
) -> NDArray[np.float32]:
    """Normalizes and places an image into the valid subregion of the full frame.

    Args:
        image: Input image to normalize and place.
        frame_height: Height of the full frame.
        frame_width: Width of the full frame.
        valid_y_range: Row range (start, end) for the valid subregion.
        valid_x_range: Column range (start, end) for the valid subregion.
        warn_on_error: Determines whether to log a warning on placement failure.

    Returns:
        Full-frame image with the normalized data placed in the valid region.
    """
    if image is None:
        return 0.5 * np.ones((frame_height, frame_width), dtype=np.float32)

    # Normalizes the image using percentile clipping.
    percentile_1 = np.percentile(image, 1)
    percentile_99 = np.percentile(image, 99)

    if percentile_99 <= percentile_1:
        return np.zeros((frame_height, frame_width), dtype=np.float32)

    normalized = (image - percentile_1) / (percentile_99 - percentile_1)

    # Places in the valid subregion.
    output = percentile_1 * np.ones((frame_height, frame_width), dtype=np.float32)
    if valid_y_range is not None and valid_x_range is not None:
        try:
            output[valid_y_range[0] : valid_y_range[1], valid_x_range[0] : valid_x_range[1]] = normalized
        except (ValueError, IndexError):
            if warn_on_error:
                console.echo(
                    message="Max projection not in combined view",
                    level=LogLevel.WARNING,
                )
    else:
        output = normalized

    return np.clip(output, 0, 1).astype(np.float32)


# ============================================================ #
#  Color / overlay read-only helpers                            #
# ============================================================ #

def compute_colors(
    context: ROIViewerData,
    roi_colormap: str,
    colocalization_threshold: float,
    has_channel_2: bool,
) -> ColorArrays:
    """Computes color statistics and RGB color arrays for all ROIs.

    Initializes per-statistic color arrays and normalization values. The first color
    channel uses random HSV coloring; subsequent channels use computed statistics (skew,
    compact, footprint, aspect ratio, etc.).

    Args:
        context: The loaded data context.
        roi_colormap: Name of the matplotlib colormap applied when mapping ROI statistics to overlay colors.
        colocalization_threshold: Display threshold applied to cell_colocalization_probabilities.
        has_channel_2: Determines whether channel 2 data is available.

    Returns:
        Computed color arrays for all ROIs.
    """
    roi_statistics = context.roi_statistics
    cell_count = len(roi_statistics)
    color_count = len(_COLOR_SHORT_NAMES)
    colorbar: list[list[float]] = []

    cols = np.zeros((color_count, cell_count, 3), dtype=np.uint8)
    istat = np.zeros((color_count, cell_count), dtype=np.float32)

    # Generates random colors, adjusting for channel 2 data if present.
    np.random.seed(seed=0)  # noqa: NPY002
    random_colors = np.random.random((cell_count,))  # noqa: NPY002
    if has_channel_2:
        random_colors = random_colors / _CHAN2_COLOR_DIVISOR + _CHAN2_COLOR_OFFSET
        is_channel_2 = context.cell_colocalization_probabilities > colocalization_threshold
        console.echo(message=f"Number of channel 2 cells: {int(is_channel_2.sum())}")
        random_hues = random_colors.copy()
        random_colors[is_channel_2] = 0
    else:
        random_hues = random_colors.copy()

    istat[0] = random_hues
    cols[0] = hsv2rgb(random_colors)

    # Computes color arrays for each statistic.
    for stat_index, name in enumerate(_COLOR_SHORT_NAMES[:-3]):
        if stat_index > 0:
            stat_values = np.zeros((cell_count, 1))
            if stat_index < color_count - 2:
                field_name = _STAT_FIELD_MAP.get(name)
                if field_name is not None:
                    for roi_index in range(cell_count):
                        value = getattr(roi_statistics[roi_index], field_name, None)
                        if value is not None:
                            stat_values[roi_index] = value

                stat_low = np.percentile(stat_values, _LOWER_PERCENTILE)
                stat_high = np.percentile(stat_values, _UPPER_PERCENTILE)
                colorbar.append([float(stat_low), float((stat_high - stat_low) / 2 + stat_low), float(stat_high)])
                stat_values = stat_values - stat_low
                stat_values = stat_values / (stat_high - stat_low)
                stat_values = np.maximum(0, np.minimum(1, stat_values))
            else:
                stat_values = np.expand_dims(context.cell_classification_probabilities, axis=1)
                colorbar.append(list(_FIXED_COLORBAR_RANGE))

            color = istat_transform(stat_values.astype(np.float32), roi_colormap)
            cols[stat_index] = color
            istat[stat_index] = stat_values.flatten()
        else:
            colorbar.append(list(_FIXED_COLORBAR_RANGE))

    # Appends a fixed range for the correlation color channel.
    colorbar.append(list(_FIXED_COLORBAR_RANGE))

    # Creates a placeholder RGB array (populated by init_roi_maps via rgb_masks).
    rgb = np.zeros((2, color_count, context.frame_height, context.frame_width, 4), dtype=np.uint8)

    return ColorArrays(
        cols=cols,
        istat=istat,
        colorbar=colorbar,
        rgb=rgb,
        random_hues=random_hues,
    )


def init_roi_maps(
    context: ROIViewerData,
    color_arrays: ColorArrays,
) -> ROIIndexMaps:
    """Initializes ROI index maps, weight layers, and RGB overlay arrays.

    Creates the multi-layer ROI index map that tracks cell/non-cell panel assignments
    and pixel overlap ordering. Also generates per-ROI text labels at centroids and
    populates the RGB overlay in color_arrays.

    Args:
        context: The loaded data context.
        color_arrays: The computed color arrays (rgb field is populated in place).

    Returns:
        Initialized ROI index maps.
    """
    roi_statistics = context.roi_statistics
    classification_labels = context.cell_classification_labels
    cell_count = len(roi_statistics)
    frame_height = context.frame_height
    frame_width = context.frame_width

    sroi = np.zeros((2, frame_height, frame_width), dtype=bool)
    lambda_all = np.zeros((frame_height, frame_width), dtype=np.float32)
    lam = np.zeros((2, _OVERLAP_LAYERS, frame_height, frame_width), dtype=np.float32)
    iroi = -1 * np.ones((2, _OVERLAP_LAYERS, frame_height, frame_width), dtype=np.int32)

    # Ignores cells that are part of a merge group.
    is_ignored = np.zeros(cell_count, dtype=bool)
    text_labels: list[pg.TextItem] = []

    for roi_index in np.arange(cell_count - 1, -1, -1, dtype=np.int32):
        y_pixels = roi_statistics[roi_index].y_pixels
        if y_pixels is not None and not is_ignored[roi_index]:
            merged = roi_statistics[roi_index].merged_roi_indices
            if merged is not None:
                for merged_index in merged:
                    is_ignored[merged_index] = True
                    console.echo(message=f"ROI {merged_index} in merged ROI")

            x_pixels = roi_statistics[roi_index].x_pixels
            weights = roi_statistics[roi_index].pixel_weights
            weights = weights / weights.sum()
            panel = int(1 - classification_labels[roi_index])

            # Pushes down existing layers and adds cell on top.
            iroi[panel, 2, y_pixels, x_pixels] = iroi[panel, 1, y_pixels, x_pixels]
            iroi[panel, 1, y_pixels, x_pixels] = iroi[panel, 0, y_pixels, x_pixels]
            iroi[panel, 0, y_pixels, x_pixels] = roi_index

            lam[panel, 2, y_pixels, x_pixels] = lam[panel, 1, y_pixels, x_pixels]
            lam[panel, 1, y_pixels, x_pixels] = lam[panel, 0, y_pixels, x_pixels]
            lam[panel, 0, y_pixels, x_pixels] = weights
            sroi[panel, y_pixels, x_pixels] = True
            lambda_all[y_pixels, x_pixels] = weights

            centroid = roi_statistics[roi_index].centroid
            label_text = str(roi_index)
        else:
            label_text = ""
            centroid = (0, 0)

        text_item = pg.TextItem(label_text, color=_ROI_TEXT_COLOR, anchor=(0.5, 0.5))
        text_item.setPos(centroid[1], centroid[0])
        text_item.setFont(QtGui.QFont("Times", _ROI_TEXT_SIZE, weight=QtGui.QFont.Weight.Bold))
        text_labels.append(text_item)

    text_labels.reverse()

    lam_mean = float(lambda_all[lambda_all > _LAMBDA_THRESHOLD].mean())
    lam_norm = np.maximum(
        0,
        np.minimum(1, _LAMBDA_NORM_SCALE * lam[:, 0] / lam_mean),
    )

    roi_maps = ROIIndexMaps(
        sroi=sroi,
        lam=lam,
        iroi=iroi,
        lam_mean=lam_mean,
        lam_norm=lam_norm,
        text_labels=text_labels,
    )

    # Populates RGB overlays for all color channels.
    for color_index in range(color_arrays.cols.shape[0]):
        rgb_masks(
            color_arrays=color_arrays,
            roi_maps=roi_maps,
            color=color_arrays.cols[color_index],
            color_index=color_index,
        )

    return roi_maps


def draw_masks(
    context: ROIViewerData,
    color_arrays: ColorArrays,
    roi_maps: ROIIndexMaps,
    roi_color_mode: int,
    background_view: int,
    selected_roi_index: int,
    merge_roi_indices: list[int],
    roi_opacity: list[int],
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Draws the current mask overlay for both cell and non-cell panels.

    Computes transparency based on ROI weights, then highlights the currently
    selected ROIs with full-white (ROI view) or colored circles (image views).

    Args:
        context: The loaded data context.
        color_arrays: The computed color arrays.
        roi_maps: The ROI index maps.
        roi_color_mode: The active ROI color mode index.
        background_view: The active background view index.
        selected_roi_index: Index of the currently selected ROI.
        merge_roi_indices: Indices of all ROIs staged for multi-selection.
        roi_opacity: Alpha values for ROI overlays in [circle-view, filled-ROI-view] rendering modes.

    Returns:
        Tuple of (cells_overlay, noncells_overlay) RGBA arrays.
    """
    color_index = roi_color_mode
    view_index = background_view
    opacity = roi_opacity

    active_panel = int(1 - context.cell_classification_labels[selected_roi_index])

    # Resets transparency based on ROI weights.
    for panel in range(2):
        color_arrays.rgb[panel, color_index, :, :, 3] = (
            opacity[view_index == 0] * roi_maps.sroi[panel] * roi_maps.lam_norm[panel]
        ).astype(np.uint8)

    overlays = [
        np.array(color_arrays.rgb[0, color_index]),
        np.array(color_arrays.rgb[1, color_index]),
    ]

    roi_statistics = context.roi_statistics
    if view_index == 0:
        # ROI view: highlights selected ROIs with brightness based on overlap depth.
        for roi_index in merge_roi_indices:
            y_pixels = roi_statistics[roi_index].y_pixels.flatten()
            x_pixels = roi_statistics[roi_index].x_pixels.flatten()
            overlap_count = (roi_maps.iroi[active_panel][:, y_pixels, x_pixels] > -1).sum(axis=0) - 1
            brightness = 1 - overlap_count / _OVERLAP_LAYERS
            overlays[active_panel] = _make_chosen_roi(overlays[active_panel], y_pixels, x_pixels, brightness)
    else:
        # Image view: highlights selected ROIs with colored circles.
        for roi_index in merge_roi_indices:
            y_circle = roi_statistics[roi_index].circle_y_pixels
            x_circle = roi_statistics[roi_index].circle_x_pixels
            y_pixels = roi_statistics[roi_index].y_pixels.flatten()
            x_pixels = roi_statistics[roi_index].x_pixels.flatten()
            overlays[active_panel][y_pixels, x_pixels, 3] = 0
            roi_color = color_arrays.cols[color_index, roi_index]
            if y_circle is not None and x_circle is not None:
                overlays[active_panel] = _make_chosen_circle(
                    overlays[active_panel],
                    y_circle,
                    x_circle,
                    roi_color,
                )

    return overlays[0], overlays[1]


def display_masks(
    color1: pg.ImageItem,
    color2: pg.ImageItem,
    masks: tuple[NDArray[np.uint8], NDArray[np.uint8]],
) -> None:
    """Displays the mask overlays on both image panels.

    Args:
        color1: The cells panel overlay image item.
        color2: The non-cells panel overlay image item.
        masks: Tuple of (cells_overlay, noncells_overlay) from ``draw_masks``.
    """
    color1.setImage(masks[0], levels=(0.0, 255.0))
    color2.setImage(masks[1], levels=(0.0, 255.0))
    color1.show()
    color2.show()


def render_colorbar(
    roi_color_mode: int,
    color_arrays: ColorArrays,
    colorbar_widgets: ColorbarWidgets,
    colorbar_image: NDArray[np.uint8],
) -> None:
    """Updates the colorbar image and tick labels for the active color mode.

    Args:
        roi_color_mode: The active ROI color mode index.
        color_arrays: The computed color arrays.
        colorbar_widgets: The colorbar display widgets.
        colorbar_image: The colorbar gradient image from ``draw_colorbar``.
    """
    color_index = roi_color_mode
    if color_index == 0:
        colorbar_widgets.image.setImage(np.zeros((_COLORBAR_ROW_COUNT, _COLORBAR_SAMPLE_COUNT - 1, 3)))
    else:
        colorbar_widgets.image.setImage(colorbar_image)

    for label_index in range(3):
        colorbar_widgets.labels[label_index].setText(f"{color_arrays.colorbar[color_index][label_index]:1.2f}")


def draw_colorbar(colormap: str = "hsv") -> NDArray[np.uint8]:
    """Creates a colorbar image for the given colormap.

    Args:
        colormap: Name of the matplotlib colormap.

    Returns:
        Colorbar image array with shape (20, 101, 3) and dtype uint8.
    """
    gradient = np.linspace(0, 1, _COLORBAR_SAMPLE_COUNT).astype(np.float32)
    rgb = istat_transform(gradient, colormap)
    colormat = np.expand_dims(rgb, axis=0)
    return np.tile(colormat, (_COLORBAR_ROW_COUNT, 1, 1))


def rgb_masks(
    color_arrays: ColorArrays,
    roi_maps: ROIIndexMaps,
    color: NDArray[np.uint8],
    color_index: int,
) -> None:
    """Updates the RGB overlay array for a specific color channel.

    Args:
        color_arrays: The computed color arrays.
        roi_maps: The ROI index maps.
        color: Per-ROI RGB colors with shape (cell_count, 3).
        color_index: Index of the color channel to update.
    """
    for panel in range(2):
        mapped_colors = color[roi_maps.iroi[panel, 0], :]
        color_arrays.rgb[panel, color_index, :, :, :3] = mapped_colors


def update_colormap(
    color_arrays: ColorArrays,
    roi_maps: ROIIndexMaps,
    colormap: str,
) -> NDArray[np.uint8]:
    """Recomputes all color statistics using a new colormap.

    Args:
        color_arrays: The computed color arrays (modified in place).
        roi_maps: The ROI index maps.
        colormap: Name of the new colormap.

    Returns:
        New colorbar gradient image.
    """
    console.echo(message=f"Colormap changed to {colormap}, loading...")
    for color_index in range(1, color_arrays.istat.shape[0]):
        color_arrays.cols[color_index] = istat_transform(color_arrays.istat[color_index], colormap)
        rgb_masks(
            color_arrays=color_arrays,
            roi_maps=roi_maps,
            color=color_arrays.cols[color_index],
            color_index=color_index,
        )
    return draw_colorbar(colormap)


def update_chan2_colors(
    context: ROIViewerData,
    colocalization_threshold: float,
    color_arrays: ColorArrays,
    roi_maps: ROIIndexMaps,
) -> None:
    """Recomputes the channel 2 random coloring after threshold change.

    Args:
        context: The loaded data context.
        colocalization_threshold: The current channel 2 display threshold.
        color_arrays: The computed color arrays (modified in place).
        roi_maps: The ROI index maps.
    """
    is_channel_2 = context.cell_colocalization_probabilities > colocalization_threshold
    color = color_arrays.random_hues.copy()
    color[is_channel_2] = 0
    color = color.flatten()
    color_arrays.cols[0] = hsv2rgb(color)
    rgb_masks(color_arrays=color_arrays, roi_maps=roi_maps, color=color_arrays.cols[0], color_index=0)


def update_correlation_masks(
    color_arrays: ColorArrays,
    roi_maps: ROIIndexMaps,
    binned_fluorescence: NDArray[np.float32],
    fluorescence_std: NDArray[np.float32],
    merge_indices: list[int],
    colormap: str,
) -> None:
    """Computes inter-ROI correlation coloring.

    Correlates each ROI's binned fluorescence with the average of the selected ROIs.

    Args:
        color_arrays: The computed color arrays (modified in place).
        roi_maps: The ROI index maps.
        binned_fluorescence: Binned fluorescence with shape (roi_count, bin_count).
        fluorescence_std: Per-ROI standard deviation with shape (roi_count,).
        merge_indices: Currently selected ROI indices.
        colormap: Name of the active colormap.
    """
    color_index = _COLOR_CORRELATION
    selected = np.array(merge_indices)
    selected_mean = binned_fluorescence[selected].mean(axis=-2).squeeze()
    selected_std = float((selected_mean**2).mean() ** 0.5)
    denominator = binned_fluorescence.shape[-1] * fluorescence_std * selected_std
    correlation = np.dot(binned_fluorescence, selected_mean.T) / denominator
    correlation[selected] = correlation.mean()

    istat = correlation
    istat_min = float(istat.min())
    istat_max = float(istat.max())
    color_arrays.colorbar[color_index] = [istat_min, (istat_max - istat_min) / 2 + istat_min, istat_max]
    istat = istat - istat.min()
    istat = istat / istat.max()
    color = istat_transform(istat, colormap)
    color_arrays.cols[color_index] = color
    color_arrays.istat[color_index] = istat.flatten()
    rgb_masks(color_arrays=color_arrays, roi_maps=roi_maps, color=color, color_index=color_index)


def hsv2rgb(colors: NDArray[np.float64]) -> NDArray[np.uint8]:
    """Converts HSV hue values to RGB uint8 colors.

    Args:
        colors: Array of hue values in [0, 1].

    Returns:
        RGB color array with shape (..., 3) and dtype uint8.
    """
    colors = colors[:, np.newaxis]
    colors = np.concatenate((colors, np.ones_like(colors), np.ones_like(colors)), axis=-1)
    return (255 * hsv_to_rgb(colors)).astype(np.uint8)


def istat_transform(istat: NDArray[np.float32], colormap: str = "hsv") -> NDArray[np.uint8]:
    """Transforms a normalized statistic array into RGB colors using the given colormap.

    Args:
        istat: Statistic values normalized to [0, 1].
        colormap: Name of the matplotlib colormap to use.

    Returns:
        RGB color array with shape (..., 3) and dtype uint8.
    """
    if colormap == "hsv":
        return _istat_hsv(istat)

    try:
        cmap = matplotlib.cm.get_cmap(colormap)
        mapped = cmap(istat)[:, :3]
        mapped *= 255
        return mapped.astype(np.uint8)
    except Exception:
        console.echo(message="Bad colormap, using hsv", level=LogLevel.WARNING)
        return _istat_hsv(istat)


def _istat_hsv(istat: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Applies the HSV color transform to a statistic array.

    Args:
        istat: Normalized statistic values in [0, 1].

    Returns:
        RGB color array with shape (..., 3) and dtype uint8.
    """
    istat = istat / _HSV_DIVISOR
    istat = istat + (_HSV_OFFSET / _HSV_DIVISOR)
    inverted = 1 - istat
    return hsv2rgb(inverted.flatten().astype(np.float64))


def _make_chosen_roi(
    overlay: NDArray[np.uint8],
    y_pixels: NDArray,
    x_pixels: NDArray,
    brightness: NDArray,
) -> NDArray[np.uint8]:
    """Highlights a selected ROI in the overlay with white based on overlap depth.

    Args:
        overlay: RGBA overlay array to modify.
        y_pixels: Row coordinates of the ROI pixels.
        x_pixels: Column coordinates of the ROI pixels.
        brightness: Per-pixel brightness values in [0, 1].

    Returns:
        Modified overlay array.
    """
    overlay[y_pixels, x_pixels, :] = np.tile((255 * brightness[:, np.newaxis]).astype(np.uint8), (1, 4))
    return overlay


def _make_chosen_circle(
    overlay: NDArray[np.uint8],
    y_circle: NDArray,
    x_circle: NDArray,
    color: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Draws a colored circle on the overlay for a selected ROI.

    Args:
        overlay: RGBA overlay array to modify.
        y_circle: Row coordinates of the circle pixels.
        x_circle: Column coordinates of the circle pixels.
        color: RGB color for the circle.

    Returns:
        Modified overlay array.
    """
    overlay[y_circle, x_circle, :3] = color
    overlay[y_circle, x_circle, 3] = 255
    return overlay


# ============================================================ #
#  Trace plot helpers                                           #
# ============================================================ #

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

    # Computes average trace scale.
    average_scale = len(selected) / _AVERAGE_SCALE_DIVISOR + 1
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
            -1 * average_scale + average * average_scale,
            pen=average_pen,
        )
        y_minimum = -1 * average_scale

    y_maximum = (len(selected) - 1) * k_space + 1
    axis.setTicks([tick_labels])
    return y_minimum, y_maximum


# ============================================================ #
#  Custom Qt widget classes                                     #
# ============================================================ #

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

        # Only handle left-click in the read-only viewer.
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


class ROIViewer(QMainWindow):
    """Self-contained read-only viewer for single-day ROI pipeline outputs.

    Displays ROI overlays, background images, and fluorescence traces. Supports left-click ROI
    selection, shift/ctrl multi-select, keyboard shortcuts for view/color switching, and quadrant
    zoom navigation. No data mutation — classification, merge, and manual ROI drawing are excluded.

    Args:
        data: Pre-loaded viewer data. If None the viewer starts empty and the user can load
            a session via the File menu or drag-and-drop.
    """

    def __init__(self, data: ROIViewerData | None = None) -> None:
        super().__init__()
        pg.setConfigOptions(imageAxisOrder="row-major")

        # Display state (replaces ViewState — lives directly on the window).
        self.rois_visible: bool = True
        self.roi_color_mode: int = ROIColorMode.RANDOM
        self.background_view: int = BackgroundView.ROIS_ONLY
        self.roi_opacity: list[int] = [127, 255]
        self.background_saturation: list[int] = [0, 255]
        self.roi_colormap: str = "hsv"
        self.selected_roi_index: int = 0
        self.merge_roi_indices: list[int] = [0]
        self.roi_tool_active: bool = False
        self.roi_tool_panel: int = ROIToolPanel.CELLS
        self.trace_mode: int = TraceMode.NEUROPIL_CORRECTED
        self.temporal_bin_size: int = 1
        self.auto_zoom_to_roi: bool = False
        self.roi_labels_visible: bool = False
        self.session_loaded: bool = False
        self.colocalization_threshold: float = 0.6

        # Core data objects.
        self.context_data: ROIViewerData | None = None
        self.color_arrays: ColorArrays | None = None
        self.roi_maps: ROIIndexMaps | None = None
        self.colorbar_widgets: ColorbarWidgets | None = None
        self.colorbar_image: NDArray[np.uint8] | None = None
        self.views: NDArray[np.uint8] | None = None

        # Binned activity state (used by correlation coloring).
        self.Fbin: NDArray[np.float32] | None = None
        self.Fstd: NDArray[np.float32] | None = None
        self.frame_indices: NDArray | None = None

        # Window geometry and title.
        self.setGeometry(50, 50, 1500, 800)
        self.setWindowTitle("cindra ROI Viewer (load session directory)")

        cindra_dir = Path(cindra.__file__).parent
        icon_path = str(cindra_dir / "logo" / "logo.png")
        app_icon = QtGui.QIcon()
        for size in (16, 24, 32, 48, 64, 256):
            app_icon.addFile(icon_path, QtCore.QSize(size, size))
        self.setWindowIcon(app_icon)
        self.setStyleSheet(STYLE.main_window)

        # File-only menu bar.
        self._build_menus()

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

        # Apply NoFocus policy to all buttons in the control panel.
        for widget in control_panel.findChildren(QWidget):
            widget.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)

        # Accept drag-and-drop of directories.
        self.setAcceptDrops(True)

        # Load data if provided.
        if data is not None:
            self.context_data = data
            self._initialize_gui()

        self.show()
        self._graphics_widget.show()

    # ------------------------------------------------------------------ #
    #  Menu bar                                                           #
    # ------------------------------------------------------------------ #

    def _build_menus(self) -> None:
        """Builds the File-only menu bar for the read-only viewer."""
        file_menu = self.menuBar().addMenu("&File")

        load_action = file_menu.addAction("&Load processed data")
        load_action.setShortcut("Ctrl+L")
        load_action.triggered.connect(self._on_load_dialog)

        load_folder_action = file_menu.addAction("Load &folder with planeX folders")
        load_folder_action.setShortcut("Ctrl+F")
        load_folder_action.triggered.connect(self._on_load_dialog)

        export_action = file_menu.addAction("&Export as image (svg)")
        export_action.triggered.connect(self._on_export_fig)

    def _on_load_dialog(self) -> None:
        """Opens a directory dialog and loads the selected session."""
        self._load_session()

    def _on_export_fig(self) -> None:
        """Opens the pyqtgraph export dialog for the current plot."""
        self._graphics_widget.scene().contextMenuItem = self._cells_view_box
        self._graphics_widget.scene().showExportDialog()

    # ------------------------------------------------------------------ #
    #  Control panel                                                      #
    # ------------------------------------------------------------------ #

    def _build_control_panel(self) -> QWidget:
        """Builds the right-side control panel with QGroupBox sections.

        Returns:
            The control panel widget containing all grouped controls.
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 1. ROI Visibility.
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
        selection_box, self._selection_controls = self._create_selection_buttons()
        layout.addWidget(selection_box)

        # 3. View Toggle.
        toggle_box, self._cell_toggle_controls = self._create_cell_toggle_buttons()
        layout.addWidget(toggle_box)

        # 4. Background Views.
        background_box, self._view_controls = self._create_view_controls()
        layout.addWidget(background_box)

        # 5. ROI Colors + colorbar.
        colors_box, self._color_controls = self._create_color_controls()
        self.colorbar_widgets = self._create_colorbar()
        colors_layout = colors_box.layout()
        assert colors_layout is not None
        colors_layout.addWidget(self.colorbar_widgets.widget)
        layout.addWidget(colors_box)

        # 6. Selected ROI — ROI index edit + stat labels.
        roi_box = QGroupBox("Selected ROI")
        roi_box.setStyleSheet("QGroupBox { color: white; }")
        roi_layout = QVBoxLayout(roi_box)
        self._stats_to_show = [
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
        self._roi_index_edit.returnPressed.connect(self._on_number_chosen)
        roi_layout.addWidget(self._roi_index_edit)
        self._roi_stat_labels: list[QLabel] = []
        for k in range(len(self._stats_to_show)):
            stat_label = QLabel(self._stats_to_show[k])
            stat_label.setFont(lilfont)
            stat_label.setStyleSheet(STYLE.white_label)
            stat_label.resize(stat_label.minimumSizeHint())
            roi_layout.addWidget(stat_label)
            self._roi_stat_labels.append(stat_label)
        layout.addWidget(roi_box)

        # 7. Trace Display.
        trace_box, self._trace_controls = self._create_trace_controls()
        self._zoom_to_cell_checkbox = QCheckBox("zoom to cell")
        self._zoom_to_cell_checkbox.setStyleSheet(STYLE.white_label)
        self._zoom_to_cell_checkbox.stateChanged.connect(self._on_zoom_cell_toggled)
        trace_layout = trace_box.layout()
        assert trace_layout is not None
        trace_layout.addWidget(self._zoom_to_cell_checkbox)
        layout.addWidget(trace_box)

        # 8. Navigation.
        nav_box, self._quadrant_controls = self._create_quadrant_buttons()
        layout.addWidget(nav_box)

        layout.addStretch()
        return panel

    # ------------------------------------------------------------------ #
    #  Widget creation helpers (no signal bus — viewer connects signals)  #
    # ------------------------------------------------------------------ #

    def _create_selection_buttons(self) -> tuple[QGroupBox, SelectionControls]:
        """Creates the cell selection buttons and top-n input field."""
        group_box = QGroupBox("Cell Selection")
        group_box.setStyleSheet("QGroupBox { color: white; }")
        layout = QGridLayout(group_box)

        selection_buttons = QButtonGroup()
        labels = [" draw selection", " select top n", " select bottom n"]
        for button_index in range(3):
            button = QPushButton(labels[button_index], self)
            button.setCheckable(True)
            button.setStyleSheet(STYLE.button_inactive)
            button.setFont(label_font_bold())
            button.resize(button.minimumSizeHint())
            selection_buttons.addButton(button, button_index)
            layout.addWidget(button, button_index, 0, 1, 1)
            button.setEnabled(False)
            button.clicked.connect(lambda _checked, idx=button_index: self._on_selection_button(idx))
        selection_buttons.setExclusive(True)

        count_label = QLabel("n=")
        count_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        count_label.setStyleSheet(STYLE.white_label)
        count_label.setFont(label_font_bold())
        layout.addWidget(count_label, 1, 1, 1, 1)

        top_count_edit = QLineEdit(self)
        top_count_edit.setValidator(QtGui.QIntValidator(0, _MAX_TOP_N))
        top_count_edit.setText(str(_DEFAULT_TOP_N))
        top_count_edit.setFixedWidth(STYLE.small_edit_width)
        top_count_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        top_count_edit.returnPressed.connect(self._on_roi_selection)
        layout.addWidget(top_count_edit, 2, 1, 1, 1)

        controls = SelectionControls(selection_buttons=selection_buttons, top_count_edit=top_count_edit)
        return group_box, controls

    def _create_cell_toggle_buttons(self) -> tuple[QGroupBox, CellToggleControls]:
        """Creates the cell / not-cell / both size-toggle buttons and ROI count labels."""
        group_box = QGroupBox("View Toggle")
        group_box.setStyleSheet("QGroupBox { color: white; }")
        layout = QGridLayout(group_box)

        cell_count_label = QLabel("")
        cell_count_label.setStyleSheet(STYLE.white_label)
        cell_count_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        layout.addWidget(cell_count_label, 0, 0, 1, 1)

        noncell_count_label = QLabel("")
        noncell_count_label.setStyleSheet(STYLE.white_label)
        layout.addWidget(noncell_count_label, 0, 2, 1, 1)

        size_buttons = QButtonGroup(self)
        labels = [" cells", " both", " not cells"]
        for button_index, label_text in enumerate(labels):
            button = QPushButton(label_text, self)
            button.setCheckable(True)
            button.setStyleSheet(STYLE.button_inactive)
            button.setFont(label_font_bold())
            button.resize(button.minimumSizeHint())
            size_buttons.addButton(button, button_index)
            layout.addWidget(button, 1, button_index, 1, 1)
            button.setEnabled(button_index == _VIEW_BOTH)
            button.clicked.connect(lambda _checked: self.update_plot())
        size_buttons.setExclusive(True)

        return group_box, CellToggleControls(
            size_buttons=size_buttons,
            cell_count_label=cell_count_label,
            noncell_count_label=noncell_count_label,
        )

    def _create_view_controls(self) -> tuple[QGroupBox, ViewControls]:
        """Creates background view selection controls inside a group box."""
        group_box = QGroupBox("Background")
        group_box.setStyleSheet("QGroupBox { color: white; }")
        layout = QGridLayout(group_box)

        view_buttons = QButtonGroup(self)
        for button_index, name in enumerate(_VIEW_NAMES):
            button = QPushButton("&" + name, self)
            button.setCheckable(True)
            button.setStyleSheet(STYLE.button_inactive)
            button.setFont(label_font_bold())
            button.resize(button.minimumSizeHint())
            view_buttons.addButton(button, button_index)
            layout.addWidget(button, button_index, 0, 1, 1)
            if button_index == 0:
                saturation_label = QLabel("sat: ")
                saturation_label.setStyleSheet(STYLE.white_label)
                layout.addWidget(saturation_label, button_index, 1, 1, 1)
            button.setEnabled(False)
            button.clicked.connect(lambda _checked, idx=button_index: self._on_view_changed(idx))
        view_buttons.setExclusive(True)

        range_slider = RangeSlider(owner=self, on_release=self._on_saturation_changed)
        range_slider.setMinimum(0)
        range_slider.setMaximum(255)
        range_slider.setLow(0)
        range_slider.setHigh(255)
        range_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        layout.addWidget(range_slider, 1, 1, len(_VIEW_NAMES) - 2, 1)

        controls = ViewControls(view_buttons=view_buttons, range_slider=range_slider)
        return group_box, controls

    def _create_color_controls(self) -> tuple[QGroupBox, ColorControls]:
        """Creates color statistic selection buttons and their associated controls."""
        group_box = QGroupBox("ROI Colors")
        group_box.setStyleSheet("QGroupBox { color: white; }")
        layout = QGridLayout(group_box)

        color_buttons = QButtonGroup(self)

        colormap_chooser = QComboBox()
        colormap_chooser.addItems(_COLORMAPS)
        colormap_chooser.setCurrentIndex(0)
        colormap_chooser.setFont(label_font())
        colormap_chooser.setFixedWidth(_COLOR_EDIT_WIDTH)
        layout.addWidget(colormap_chooser, 0, 1, 1, 1)

        for button_index, name in enumerate(_COLOR_NAMES):
            button = QPushButton("&" + name, self)
            button.setCheckable(True)
            button.setStyleSheet(STYLE.button_inactive)
            button.setFont(label_font_bold())
            button.resize(button.minimumSizeHint())
            color_buttons.addButton(button, button_index)
            if _COLOR_NARROW_RANGE_START <= button_index < _COLOR_NARROW_RANGE_END:
                layout.addWidget(button, button_index, 0, 1, 1)
            else:
                layout.addWidget(button, button_index, 0, 1, 2)
            button.setEnabled(False)
            button.clicked.connect(lambda _checked, idx=button_index: self._on_color_changed(idx))

        channel_2_edit = QLineEdit(self)
        channel_2_edit.setText("0.6")
        channel_2_edit.setFixedWidth(_COLOR_EDIT_WIDTH)
        channel_2_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(channel_2_edit, len(_COLOR_NAMES) - 4, 1, 1, 1)
        channel_2_edit.returnPressed.connect(self.update_plot)

        classifier_edit = QLineEdit(self)
        classifier_edit.setText("0.5")
        classifier_edit.setFixedWidth(_COLOR_EDIT_WIDTH)
        classifier_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(classifier_edit, len(_COLOR_NAMES) - 3, 1, 1, 1)
        classifier_edit.returnPressed.connect(self.update_plot)

        bin_edit = QLineEdit(self)
        bin_edit.setValidator(QtGui.QIntValidator(0, 500))
        bin_edit.setText("1")
        bin_edit.setFixedWidth(_COLOR_EDIT_WIDTH)
        bin_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(bin_edit, len(_COLOR_NAMES) - 2, 1, 1, 1)
        bin_edit.returnPressed.connect(
            lambda: self._on_activity_changed(self._trace_controls.activity_combo.currentIndex())
        )

        colormap_chooser.activated.connect(lambda: self._on_color_changed(self.roi_color_mode))

        controls = ColorControls(
            color_buttons=color_buttons,
            colormap_chooser=colormap_chooser,
            channel_2_edit=channel_2_edit,
            classifier_edit=classifier_edit,
            bin_edit=bin_edit,
        )
        return group_box, controls

    def _create_colorbar(self) -> ColorbarWidgets:
        """Creates the colorbar widget displaying the current color mapping."""
        colorbar_widget = pg.GraphicsLayoutWidget(self)
        colorbar_widget.setMaximumHeight(STYLE.colorbar_max_height)
        colorbar_widget.setMaximumWidth(STYLE.colorbar_max_width)
        colorbar_widget.ci.layout.setRowStretchFactor(0, 2)
        colorbar_widget.ci.layout.setContentsMargins(0, 0, 0, 0)

        image = pg.ImageItem()
        colorbar_view = colorbar_widget.addViewBox(row=0, col=0, colspan=3)
        colorbar_view.setMenuEnabled(False)
        colorbar_view.addItem(image)

        labels = [
            colorbar_widget.addLabel("0.0", color=[255, 255, 255], row=1, col=0),
            colorbar_widget.addLabel("0.5", color=[255, 255, 255], row=1, col=1),
            colorbar_widget.addLabel("1.0", color=[255, 255, 255], row=1, col=2),
        ]
        return ColorbarWidgets(image=image, labels=labels, widget=colorbar_widget)

    def _create_trace_controls(self) -> tuple[QGroupBox, TraceControls]:
        """Creates trace panel controls inside a group box."""
        group_box = QGroupBox("Trace Display")
        group_box.setStyleSheet("QGroupBox { color: white; }")
        layout = QGridLayout(group_box)

        activity_label = QLabel("Activity mode:")
        activity_label.setStyleSheet(STYLE.white_label)
        layout.addWidget(activity_label, 0, 0, 1, 1)

        activity_combo = QComboBox(self)
        activity_combo.setFixedWidth(STYLE.combo_box_width)
        layout.addWidget(activity_combo, 1, 0, 1, 1)
        activity_combo.addItem("F")
        activity_combo.addItem("Fneu")
        activity_combo.addItem("F - 0.7*Fneu")
        activity_combo.addItem("deconvolved")
        activity_combo.setCurrentIndex(_DEFAULT_ACTIVITY_MODE)
        activity_combo.currentIndexChanged.connect(self._on_activity_changed)

        arrow_up = QPushButton(" \u25b2")
        arrow_down = QPushButton(" \u25bc")
        arrow_buttons = [arrow_up, arrow_down]
        for button_index, button in enumerate(arrow_buttons):
            button.setMaximumWidth(STYLE.square_button_max_width)
            button.setFont(arrow_button_font())
            button.setStyleSheet(STYLE.button_unpressed)
            layout.addWidget(button, button_index, 1, 1, 1, QtCore.Qt.AlignmentFlag.AlignRight)

        scale_up = QPushButton(" +")
        scale_down = QPushButton(" -")
        scale_buttons = [scale_up, scale_down]
        for button_index, button in enumerate(scale_buttons):
            button.setMaximumWidth(STYLE.square_button_max_width)
            button.setFont(arrow_button_font())
            button.setStyleSheet(STYLE.button_unpressed)
            layout.addWidget(button, button_index, 2, 1, 1)

        max_plotted_label = QLabel("max # plotted:")
        max_plotted_label.setStyleSheet(STYLE.white_label)
        layout.addWidget(max_plotted_label, 2, 0, 1, 1)

        max_plotted_edit = QLineEdit(self)
        max_plotted_edit.setValidator(QtGui.QIntValidator(0, _MAX_PLOTTED_COUNT))
        max_plotted_edit.setText(str(_DEFAULT_PLOTTED_COUNT))
        max_plotted_edit.setFixedWidth(STYLE.small_edit_width)
        max_plotted_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
        layout.addWidget(max_plotted_edit, 3, 0, 1, 1)

        deconvolved_checkbox = QCheckBox("deconv [N]")
        deconvolved_checkbox.setStyleSheet(STYLE.white_label)
        deconvolved_checkbox.toggle()
        layout.addWidget(deconvolved_checkbox, 3, 1, 1, 1)

        neuropil_checkbox = QCheckBox("neuropil [B]")
        neuropil_checkbox.setStyleSheet(STYLE.red_label)
        neuropil_checkbox.toggle()
        layout.addWidget(neuropil_checkbox, 3, 2, 1, 1)

        traces_checkbox = QCheckBox("raw fluor [V]")
        traces_checkbox.setStyleSheet(STYLE.cyan_label)
        traces_checkbox.toggle()
        layout.addWidget(traces_checkbox, 3, 3, 1, 1)

        controls = TraceControls(
            activity_combo=activity_combo,
            deconvolved_checkbox=deconvolved_checkbox,
            neuropil_checkbox=neuropil_checkbox,
            traces_checkbox=traces_checkbox,
            max_plotted_edit=max_plotted_edit,
            arrow_buttons=arrow_buttons,
            scale_buttons=scale_buttons,
        )

        arrow_up.clicked.connect(lambda: self._adjust_trace_level(1))
        arrow_down.clicked.connect(lambda: self._adjust_trace_level(-1))
        scale_up.clicked.connect(lambda: self._adjust_scale(_SCALE_STEP))
        scale_down.clicked.connect(lambda: self._adjust_scale(-_SCALE_STEP))
        max_plotted_edit.returnPressed.connect(self._refresh_traces)
        deconvolved_checkbox.toggled.connect(lambda: self._on_trace_toggle("deconvolved"))
        neuropil_checkbox.toggled.connect(lambda: self._on_trace_toggle("neuropil"))
        traces_checkbox.toggled.connect(lambda: self._on_trace_toggle("traces"))

        return group_box, controls

    def _create_quadrant_buttons(self) -> tuple[QGroupBox, QuadrantControls]:
        """Creates the 3x3 quadrant zoom navigation buttons."""
        group_box = QGroupBox("Navigation")
        group_box.setStyleSheet("QGroupBox { color: white; }")
        layout = QGridLayout(group_box)

        quadrant_buttons = QButtonGroup(self)
        for button_index in range(9):
            button = QPushButton(" " + str(button_index + 1), self)
            button.setCheckable(True)
            button.setStyleSheet(STYLE.button_inactive)
            button.setFont(label_font_bold())
            button.resize(button.minimumSizeHint())
            button.setMaximumWidth(STYLE.small_edit_width)
            quadrant_buttons.addButton(button, button_index)
            row = button_index // _QUADRANT_COLUMNS
            col = button_index % _QUADRANT_COLUMNS
            layout.addWidget(button, row, col, 1, 1)
            button.setEnabled(False)
            button.clicked.connect(lambda _checked, idx=button_index: self._on_quadrant(idx))
        quadrant_buttons.setExclusive(True)

        return group_box, QuadrantControls(quadrant_buttons=quadrant_buttons)

    # ------------------------------------------------------------------ #
    #  Graphics panels                                                    #
    # ------------------------------------------------------------------ #

    def _build_graphics(self) -> None:
        """Creates the main plotting area with cells, non-cells, and trace panels."""
        self._cells_view_box = ViewBox(panel=ROIToolPanel.CELLS, name="plot1", border=[100, 100, 100], invert_y=True)
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

        self._noncells_view_box = ViewBox(
            panel=ROIToolPanel.NON_CELLS, name="plot2", border=[100, 100, 100], invert_y=True
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

        self._noncells_view_box.setXLink("plot1")
        self._noncells_view_box.setYLink("plot1")

        self._cells_view_box.set_click_handler(self._handle_click)
        self._noncells_view_box.set_click_handler(self._handle_click)
        self._cells_view_box.set_zoom_handler(lambda: self._zoom_plot(_CELLS_PLOT))
        self._noncells_view_box.set_zoom_handler(lambda: self._zoom_plot(_NONCELLS_PLOT))

        self._trace_box = TraceBox()
        self._trace_box.setMouseEnabled(x=True, y=False)
        self._trace_box.enableAutoRange(x=True, y=True)
        self._graphics_widget.addItem(self._trace_box, row=1, col=0, colspan=2)
        self._graphics_widget.ci.layout.setRowStretchFactor(0, 2)
        gl = self._graphics_widget.ci.layout
        gl.setColumnMinimumWidth(0, 1)
        gl.setColumnMinimumWidth(1, 1)
        gl.setHorizontalSpacing(20)

    # ------------------------------------------------------------------ #
    #  Session loading                                                    #
    # ------------------------------------------------------------------ #

    def _load_session(self, session_path: Path | None = None) -> None:
        """Loads a pipeline output directory into the viewer.

        Args:
            session_path: Path to the cindra output directory. If None, opens a dialog.
        """
        if session_path is None:
            name = QFileDialog.getExistingDirectory(parent=self, caption="Open cindra output directory")
            if not name:
                return
            session_path = Path(name)

        console.echo(message=f"Loading session: {session_path}")

        try:
            context_data = ROIViewerData.from_single_day(root_path=session_path)
        except Exception:
            console.echo(message="Failed to load session data.", level=LogLevel.ERROR)
            result = QMessageBox.question(
                self,
                "ERROR",
                "Failed to load session. Try another directory?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if result == QMessageBox.StandardButton.Yes:
                self._load_session()
            return

        self.context_data = context_data
        self._reset_state()
        self._initialize_gui()

    def _reset_state(self) -> None:
        """Resets all display state to defaults before loading new data."""
        self.rois_visible = True
        self.roi_color_mode = ROIColorMode.RANDOM
        self.background_view = BackgroundView.ROIS_ONLY
        self.roi_opacity = [127, 255]
        self.background_saturation = [0, 255]
        self.roi_colormap = "hsv"
        self.selected_roi_index = 0
        self.merge_roi_indices = [0]
        self.roi_tool_active = False
        self.roi_tool_panel = ROIToolPanel.CELLS
        self.trace_mode = TraceMode.NEUROPIL_CORRECTED
        self.temporal_bin_size = 1
        self.auto_zoom_to_roi = False
        self.roi_labels_visible = False
        self.session_loaded = False
        self.colocalization_threshold = 0.6

    def _initialize_gui(self) -> None:
        """Initializes all GUI components after loading context data."""
        context = self.context_data
        if context is None:
            return

        # Resets display controls.
        self._roi_visibility_checkbox.setChecked(True)
        if self._roi_labels_checkbox.isChecked():
            self._roi_text(False)
        self._roi_labels_checkbox.setChecked(False)
        self._roi_labels_checkbox.setEnabled(True)
        self._roi_remove()

        session_title = str(context.output_path) if context.output_path is not None else "unknown session"
        self.setWindowTitle(f"cindra ROI Viewer — {session_title}")

        # Computes default bin size from tau and sampling rate.
        self.temporal_bin_size = max(1, int(context.tau * context.sampling_rate / _BIN_SIZE_DIVISOR))
        self._color_controls.bin_edit.setText(str(self.temporal_bin_size))
        self.colocalization_threshold = _DEFAULT_CHANNEL_2_THRESHOLD
        self._color_controls.channel_2_edit.setText(str(self.colocalization_threshold))

        # Enables buttons.
        self._enable_controls()

        # Builds background views from detection images.
        self.views = build_views(
            frame_height=context.frame_height,
            frame_width=context.frame_width,
            mean_image=context.mean_image,
            enhanced_mean_image=context.enhanced_mean_image,
            correlation_map=context.correlation_map,
            maximum_projection=context.maximum_projection,
            corrected_channel_2_image=context.corrected_structural_mean_image,
            channel_2_mean_image=context.mean_image_channel_2,
            valid_y_range=context.valid_y_range,
            valid_x_range=context.valid_x_range,
        )

        # Computes color statistics and builds ROI index maps.
        self.color_arrays = compute_colors(
            context=context,
            roi_colormap=self.roi_colormap,
            colocalization_threshold=self.colocalization_threshold,
            has_channel_2=context.has_channel_2,
        )
        self.roi_maps = init_roi_maps(context=context, color_arrays=self.color_arrays)

        # Selects the first classified cell as the initial selection.
        first_cell = int(np.nonzero(context.cell_classification_labels)[0][0]) if context.cell_count > 0 else 0
        self.selected_roi_index = first_cell
        self.merge_roi_indices = [first_cell]
        self._ichosen_stats()
        self._trace_controls.activity_combo.setCurrentIndex(_DEFAULT_ACTIVITY_MODE)

        # Draws the colorbar and initial mask overlays.
        self.colorbar_image = draw_colorbar(colormap=self.roi_colormap)
        if self.colorbar_widgets is None or self.colorbar_image is None:
            return
        render_colorbar(
            roi_color_mode=self.roi_color_mode,
            color_arrays=self.color_arrays,
            colorbar_widgets=self.colorbar_widgets,
            colorbar_image=self.colorbar_image,
        )

        masks = draw_masks(
            context=context,
            color_arrays=self.color_arrays,
            roi_maps=self.roi_maps,
            roi_color_mode=self.roi_color_mode,
            background_view=self.background_view,
            selected_roi_index=self.selected_roi_index,
            merge_roi_indices=self.merge_roi_indices,
            roi_opacity=self.roi_opacity,
        )
        display_masks(color1=self._cells_overlay, color2=self._noncells_overlay, masks=masks)

        # Updates cell count labels.
        self._cell_toggle_controls.cell_count_label.setText(f"{int(context.cell_count)}")
        self._cell_toggle_controls.noncell_count_label.setText(f"{int(context.roi_count - context.cell_count)}")

        # Initializes plot ranges.
        self._cells_view_box.setXRange(0, context.frame_width)
        self._cells_view_box.setYRange(0, context.frame_height)
        self._noncells_view_box.setXRange(0, context.frame_width)
        self._noncells_view_box.setYRange(0, context.frame_height)
        self._trace_box.getViewBox().setLimits(xMin=0, xMax=context.frame_count)
        self.frame_indices = np.arange(0, context.frame_count, dtype=np.int32)

        display_views(
            view1=self._cells_background,
            view2=self._noncells_background,
            views=self.views,
            view_index=self.background_view,
            saturation=self.background_saturation,
        )
        plot_trace(
            trace_box=self._trace_box,
            cell_fluorescence=context.cell_fluorescence,
            neuropil_fluorescence=context.neuropil_fluorescence,
            spikes=context.spikes,
            frame_indices=self.frame_indices,
            merge_indices=self.merge_roi_indices,
            activity_mode=self.trace_mode,
        )

        # Sets aspect ratio on both panels.
        self._cells_view_box.setAspectLocked(lock=True, ratio=context.aspect_ratio)
        self._noncells_view_box.setAspectLocked(lock=True, ratio=context.aspect_ratio)

        self.session_loaded = True

        # Computes binned activity and triggers initial full redraw.
        self._on_activity_changed(_DEFAULT_ACTIVITY_MODE)
        self.show()

    def _enable_controls(self) -> None:
        """Enables all view, color, and selection buttons after data loading."""
        if self.context_data is None:
            return
        context = self.context_data

        # Enables quadrant buttons.
        for b in range(9):
            self._quadrant_controls.quadrant_buttons.button(b).setEnabled(True)
            self._quadrant_controls.quadrant_buttons.button(b).setStyleSheet(STYLE.button_unpressed)

        # Enables view buttons.
        for b in range(len(self._view_controls.view_names)):
            self._view_controls.view_buttons.button(b).setEnabled(True)
            self._view_controls.view_buttons.button(b).setStyleSheet(STYLE.button_unpressed)
            if b == 0:
                self._view_controls.view_buttons.button(b).setChecked(True)
                self._view_controls.view_buttons.button(b).setStyleSheet(STYLE.button_pressed)

        # Disables channel 2 views if no channel 2 data is available.
        if context.corrected_structural_mean_image is None:
            self._view_controls.view_buttons.button(5).setEnabled(False)
            self._view_controls.view_buttons.button(5).setStyleSheet(STYLE.button_inactive)
            if context.mean_image_channel_2 is None:
                self._view_controls.view_buttons.button(6).setEnabled(False)
                self._view_controls.view_buttons.button(6).setStyleSheet(STYLE.button_inactive)

        # Enables color mode buttons.
        color_button_count = len(self._color_controls.color_buttons.buttons())
        for b in range(color_button_count):
            if b == _COLOR_CHAN2:
                if context.has_channel_2:
                    self._color_controls.color_buttons.button(b).setEnabled(True)
                    self._color_controls.color_buttons.button(b).setStyleSheet(STYLE.button_unpressed)
            elif b == 0:
                self._color_controls.color_buttons.button(b).setEnabled(True)
                self._color_controls.color_buttons.button(b).setChecked(True)
                self._color_controls.color_buttons.button(b).setStyleSheet(STYLE.button_pressed)
            elif b < _COLOR_STAT_COUNT:
                self._color_controls.color_buttons.button(b).setEnabled(True)
                self._color_controls.color_buttons.button(b).setStyleSheet(STYLE.button_unpressed)

        # Enables size toggle buttons.
        for button_index, btn in enumerate(self._cell_toggle_controls.size_buttons.buttons()):
            btn.setStyleSheet(STYLE.button_unpressed)
            btn.setEnabled(True)
            if button_index == 0:
                btn.setChecked(True)
                btn.setStyleSheet(STYLE.button_pressed)

        # Enables selection buttons (draw enabled, top/bottom disabled until data analyzed).
        for b in range(3):
            if b == 0:
                self._selection_controls.selection_buttons.button(b).setEnabled(True)
                self._selection_controls.selection_buttons.button(b).setStyleSheet(STYLE.button_unpressed)
            else:
                self._selection_controls.selection_buttons.button(b).setEnabled(False)
                self._selection_controls.selection_buttons.button(b).setStyleSheet(STYLE.button_inactive)

    # ------------------------------------------------------------------ #
    #  Event handlers — view, color, activity, trace                     #
    # ------------------------------------------------------------------ #

    def _on_view_changed(self, index: int) -> None:
        """Handles background view mode changes.

        Args:
            index: The background view index selected.
        """
        # Updates button styles.
        for i in range(len(_VIEW_NAMES)):
            btn = self._view_controls.view_buttons.button(i)
            if btn is not None and btn.isEnabled():
                btn.setStyleSheet(STYLE.button_unpressed)
        btn = self._view_controls.view_buttons.button(index)
        if btn is not None:
            btn.setChecked(True)
            btn.setStyleSheet(STYLE.button_pressed)

        self.background_view = BackgroundView(index)
        self.update_plot()

    def _on_color_changed(self, index: int) -> None:
        """Handles ROI color mode changes.

        Args:
            index: The color mode index selected.
        """
        # Updates button styles.
        for i in range(len(_COLOR_NAMES)):
            btn = self._color_controls.color_buttons.button(i)
            if btn is not None and btn.isEnabled():
                btn.setStyleSheet(STYLE.button_unpressed)
        btn = self._color_controls.color_buttons.button(index)
        if btn is not None:
            btn.setChecked(True)
            btn.setStyleSheet(STYLE.button_pressed)

        self.roi_color_mode = ROIColorMode(index)
        if self.context_data is not None and self.color_arrays is not None and self.roi_maps is not None:
            colormap = self._color_controls.colormap_chooser.currentText()
            if colormap != self.roi_colormap:
                self.roi_colormap = colormap
                self.colorbar_image = update_colormap(
                    color_arrays=self.color_arrays,
                    roi_maps=self.roi_maps,
                    colormap=colormap,
                )
            if (
                self.context_data.has_channel_2
                and abs(float(self._color_controls.channel_2_edit.text()) - self.colocalization_threshold)
                > _CHAN2_THRESHOLD_EPSILON
            ):
                self.colocalization_threshold = float(self._color_controls.channel_2_edit.text())
                update_chan2_colors(
                    context=self.context_data,
                    colocalization_threshold=self.colocalization_threshold,
                    color_arrays=self.color_arrays,
                    roi_maps=self.roi_maps,
                )
        self.update_plot()

    def _on_activity_changed(self, i: int) -> None:
        """Changes the activity mode used for multi-neuron display and correlation.

        Args:
            i: The activity mode index to switch to.
        """
        self.trace_mode = TraceMode(i)
        if self.session_loaded and self.context_data is not None:
            self.temporal_bin_size = max(1, int(self._color_controls.bin_edit.text()))
            nb = int(np.floor(float(self.context_data.frame_count) / float(self.temporal_bin_size)))
            if i == 0:
                f = self.context_data.cell_fluorescence
            elif i == 1:
                f = self.context_data.neuropil_fluorescence
            elif i == _ACTIVITY_MODE_SUBTRACTED:
                f = self.context_data.cell_fluorescence - 0.7 * self.context_data.neuropil_fluorescence
            else:
                f = self.context_data.spikes
            ncells = self.context_data.roi_count
            bin_size = self.temporal_bin_size
            self.Fbin = f[:, : nb * bin_size].reshape((ncells, nb, bin_size)).mean(axis=2)
            self.Fbin -= self.Fbin.mean(axis=1)[:, np.newaxis]
            self.Fstd = (self.Fbin**2).mean(axis=1) ** 0.5
            self.frame_indices = np.arange(0, self.context_data.frame_count, dtype=np.int32)
            self.update_plot()

    def _on_saturation_changed(self) -> None:
        """Handles saturation range slider changes."""
        self.background_saturation = self._view_controls.range_slider.saturation_values()
        self.update_plot()

    def _on_selection_button(self, index: int) -> None:
        """Handles selection button presses."""
        self._on_roi_selection()

    def _on_quadrant(self, index: int) -> None:
        """Handles quadrant zoom button presses.

        Args:
            index: The quadrant button index (0-8).
        """
        if self.context_data is None:
            return
        qb = self._quadrant_controls.quadrant_buttons
        for i in range(9):
            if qb.button(i).isEnabled():
                qb.button(i).setStyleSheet(STYLE.button_unpressed)
        qb.button(index).setStyleSheet(STYLE.button_pressed)

        x_column = index % _QUADRANT_COLUMNS
        y_row = index // _QUADRANT_COLUMNS
        fw = self.context_data.frame_width
        fh = self.context_data.frame_height
        x_lo = (x_column - _QUADRANT_ZOOM_MARGIN) * fw / _QUADRANT_COLUMNS
        x_hi = (x_column + 1 + _QUADRANT_ZOOM_MARGIN) * fw / _QUADRANT_COLUMNS
        y_lo = (y_row - _QUADRANT_ZOOM_MARGIN) * fh / _QUADRANT_COLUMNS
        y_hi = (y_row + 1 + _QUADRANT_ZOOM_MARGIN) * fh / _QUADRANT_COLUMNS
        self._cells_view_box.setXRange(x_lo, x_hi)
        self._cells_view_box.setYRange(y_lo, y_hi)
        self._noncells_view_box.setXRange(x_lo, x_hi)
        self._noncells_view_box.setYRange(y_lo, y_hi)

    def _on_number_chosen(self) -> None:
        """Jumps to the ROI number entered in the ROI edit field."""
        if self.session_loaded and self.context_data is not None:
            self.selected_roi_index = int(self._roi_index_edit.text())
            if self.selected_roi_index >= self.context_data.roi_count:
                self.selected_roi_index = self.context_data.roi_count - 1
            self.merge_roi_indices = [self.selected_roi_index]
            self.update_plot()
            self.show()

    def _on_zoom_cell_toggled(self, state: int) -> None:
        """Toggles zoom-to-cell behavior based on checkbox state."""
        if not self.session_loaded:
            return
        self.auto_zoom_to_roi = QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked
        self.update_plot()

    def _on_trace_toggle(self, which: str) -> None:
        """Handles trace visibility checkbox toggles."""
        tc = self._trace_controls
        if which == "deconvolved":
            tc.deconvolved_visible = tc.deconvolved_checkbox.isChecked()
        elif which == "neuropil":
            tc.neuropil_visible = tc.neuropil_checkbox.isChecked()
        elif which == "traces":
            tc.traces_visible = tc.traces_checkbox.isChecked()
        self._refresh_traces()

    def _adjust_trace_level(self, delta: int) -> None:
        """Adjusts the trace panel row stretch factor."""
        tc = self._trace_controls
        tc.trace_level = max(_MIN_TRACE_LEVEL, min(_MAX_TRACE_LEVEL, tc.trace_level + delta))
        self._refresh_traces()

    def _adjust_scale(self, delta: float) -> None:
        """Adjusts the vertical scale factor for multi-trace stacking."""
        tc = self._trace_controls
        tc.scale_factor = max(_MIN_SCALE, min(_MAX_SCALE, tc.scale_factor + delta))
        self._refresh_traces()

    def _refresh_traces(self) -> None:
        """Refreshes the trace panel without redrawing image panels."""
        if self.context_data is None or self.color_arrays is None or self.frame_indices is None:
            return
        plot_trace(
            trace_box=self._trace_box,
            cell_fluorescence=self.context_data.cell_fluorescence,
            neuropil_fluorescence=self.context_data.neuropil_fluorescence,
            spikes=self.context_data.spikes,
            frame_indices=self.frame_indices,
            merge_indices=self.merge_roi_indices,
            activity_mode=self.trace_mode,
            roi_colors=self.color_arrays.cols[self.roi_color_mode],
            traces_visible=self._trace_controls.traces_visible,
            neuropil_visible=self._trace_controls.neuropil_visible,
            deconvolved_visible=self._trace_controls.deconvolved_visible,
            scale_factor=self._trace_controls.scale_factor,
            max_plotted=int(self._trace_controls.max_plotted_edit.text() or "40"),
        )

    # ------------------------------------------------------------------ #
    #  Main plot update                                                   #
    # ------------------------------------------------------------------ #

    def update_plot(self) -> None:
        """Redraws all plot panels including masks, traces, and colorbar."""
        if self.context_data is None or self.color_arrays is None or self.roi_maps is None:
            return
        if self.views is None or self.colorbar_widgets is None or self.colorbar_image is None:
            return
        if self.roi_color_mode == _COLOR_CORRELATION and self.Fbin is not None:
            assert self.Fstd is not None
            update_correlation_masks(
                color_arrays=self.color_arrays,
                roi_maps=self.roi_maps,
                binned_fluorescence=self.Fbin,
                fluorescence_std=self.Fstd,
                merge_indices=self.merge_roi_indices,
                colormap=self.roi_colormap,
            )
        render_colorbar(
            roi_color_mode=self.roi_color_mode,
            color_arrays=self.color_arrays,
            colorbar_widgets=self.colorbar_widgets,
            colorbar_image=self.colorbar_image,
        )
        self._ichosen_stats()
        display_views(
            view1=self._cells_background,
            view2=self._noncells_background,
            views=self.views,
            view_index=self.background_view,
            saturation=self.background_saturation,
        )
        masks = draw_masks(
            context=self.context_data,
            color_arrays=self.color_arrays,
            roi_maps=self.roi_maps,
            roi_color_mode=self.roi_color_mode,
            background_view=self.background_view,
            selected_roi_index=self.selected_roi_index,
            merge_roi_indices=self.merge_roi_indices,
            roi_opacity=self.roi_opacity,
        )
        display_masks(color1=self._cells_overlay, color2=self._noncells_overlay, masks=masks)
        assert self.frame_indices is not None
        plot_trace(
            trace_box=self._trace_box,
            cell_fluorescence=self.context_data.cell_fluorescence,
            neuropil_fluorescence=self.context_data.neuropil_fluorescence,
            spikes=self.context_data.spikes,
            frame_indices=self.frame_indices,
            merge_indices=self.merge_roi_indices,
            activity_mode=self.trace_mode,
            roi_colors=self.color_arrays.cols[self.roi_color_mode],
            traces_visible=self._trace_controls.traces_visible,
            neuropil_visible=self._trace_controls.neuropil_visible,
            deconvolved_visible=self._trace_controls.deconvolved_visible,
            scale_factor=self._trace_controls.scale_factor,
            max_plotted=int(self._trace_controls.max_plotted_edit.text() or "40"),
        )
        if self.auto_zoom_to_roi:
            self._zoom_to_cell()
        self._cells_view_box.show()
        self._noncells_view_box.show()
        self._graphics_widget.show()
        self.show()

        # Update status bar.
        roi_index = self.selected_roi_index
        cell_count = int(self.context_data.cell_count)
        height = self.context_data.frame_height
        width = self.context_data.frame_width
        session_name = str(self.context_data.output_path) if self.context_data.output_path is not None else "unknown"
        self._status_bar.showMessage(
            f"Session: {session_name}  |  ROI: {roi_index}  |  Cells: {cell_count}  |  Size: {height} x {width}"
        )

    # ------------------------------------------------------------------ #
    #  ROI display helpers                                                #
    # ------------------------------------------------------------------ #

    def _ichosen_stats(self) -> None:
        """Updates the ROI statistics labels for the currently selected cell."""
        if self.context_data is None:
            return
        n = self.selected_roi_index
        self._roi_index_edit.setText(str(n))
        roi = self.context_data.roi_statistics[n]
        for k in range(len(self._stats_to_show)):
            key = self._stats_to_show[k]
            ival = getattr(roi, key, None)
            if ival is None:
                continue
            if k + 1 == _CENTROID_STAT_INDEX:
                self._roi_stat_labels[k].setText(f"{key}: [{ival[0]:d}, {ival[1]:d}]")
            elif k + 1 == _PIXEL_COUNT_STAT_INDEX:
                self._roi_stat_labels[k].setText(f"{key}: {ival:d}")
            else:
                self._roi_stat_labels[k].setText(f"{key}: {ival:2.2f}")

    def _toggle_rois(self, state: int) -> None:
        """Toggles ROI overlay visibility on both image panels."""
        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            self.rois_visible = True
            self._cells_view_box.addItem(self._cells_overlay)
            self._noncells_view_box.addItem(self._noncells_overlay)
        else:
            self.rois_visible = False
            self._cells_view_box.removeItem(self._cells_overlay)
            self._noncells_view_box.removeItem(self._noncells_overlay)
        self._graphics_widget.show()
        self.show()

    def _roi_text(self, state: int) -> None:
        """Toggles ROI number text labels on the image panels."""
        if self.roi_maps is None or self.context_data is None:
            return

        if QtCore.Qt.CheckState(state) == QtCore.Qt.CheckState.Checked:
            for n in range(len(self.roi_maps.text_labels)):
                if self.context_data.cell_classification_labels[n] == 1:
                    self._cells_view_box.addItem(self.roi_maps.text_labels[n])
                else:
                    self._noncells_view_box.addItem(self.roi_maps.text_labels[n])
            self.roi_labels_visible = True
        else:
            for n in range(len(self.roi_maps.text_labels)):
                if self.context_data.cell_classification_labels[n] == 1:
                    with suppress(Exception):
                        self._cells_view_box.removeItem(self.roi_maps.text_labels[n])
                else:
                    with suppress(Exception):
                        self._noncells_view_box.removeItem(self.roi_maps.text_labels[n])
            self.roi_labels_visible = False

    def _zoom_to_cell(self) -> None:
        """Zooms both image panels to center on the currently selected cell."""
        if self.context_data is None:
            return
        irange = 0.1 * np.array([self.context_data.frame_height, self.context_data.frame_width]).max()
        roi_statistics = self.context_data.roi_statistics
        if len(self.merge_roi_indices) > 1:
            apix = np.zeros((0, 2))
            for k in self.merge_roi_indices:
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
            icent = np.array(roi_statistics[self.selected_roi_index].centroid)
            imin = icent - irange
            imax = icent + irange
        self._cells_view_box.setYRange(imin[0], imax[0])
        self._cells_view_box.setXRange(imin[1], imax[1])
        self._noncells_view_box.setYRange(imin[0], imax[0])
        self._noncells_view_box.setXRange(imin[1], imax[1])
        self._graphics_widget.show()
        self.show()

    def _zoom_plot(self, panel: int) -> None:
        """Resets the view range for the specified panel."""
        if panel == _CELLS_PLOT:
            self._cells_view_box.autoRange()
            self._noncells_view_box.autoRange()
        elif panel == _NONCELLS_PLOT:
            self._noncells_view_box.autoRange()

    # ------------------------------------------------------------------ #
    #  Click handling (left-click only — no right-click reclassification) #
    # ------------------------------------------------------------------ #

    def _handle_click(self, click_x: int, click_y: int, panel_index: int, is_multi: bool) -> None:
        """Handles mouse clicks on image panels.

        Left-click chooses a cell. Shift/ctrl-click adds or removes from the merge selection.

        Args:
            click_x: Column coordinate of the click.
            click_y: Row coordinate of the click.
            panel_index: Panel index (0=cells, 1=non-cells).
            is_multi: Determines whether shift or ctrl was held during the click.
        """
        if not self.session_loaded or self.roi_maps is None or self.context_data is None:
            return

        if (
            click_y < 0
            or click_y >= self.context_data.frame_height
            or click_x < 0
            or click_x >= self.context_data.frame_width
        ):
            return

        ichosen = int(self.roi_maps.iroi[panel_index, 0, click_y, click_x])
        if ichosen < 0:
            return

        merged = False
        if is_multi and (
            self.context_data.cell_classification_labels[self.merge_roi_indices[0]]
            == self.context_data.cell_classification_labels[ichosen]
        ):
            if ichosen not in self.merge_roi_indices:
                self.merge_roi_indices.append(ichosen)
                self.selected_roi_index = ichosen
                merged = True
            elif len(self.merge_roi_indices) > 1:
                self.merge_roi_indices.remove(ichosen)
                self.selected_roi_index = self.merge_roi_indices[0]
                merged = True
        if not merged:
            self.merge_roi_indices = [ichosen]
            self.selected_roi_index = ichosen

        if self.roi_tool_active:
            self._roi_remove()
        if not self._cell_toggle_controls.size_buttons.button(1).isChecked():
            for btn in self._selection_controls.selection_buttons.buttons():
                if btn.isChecked():
                    btn.setStyleSheet(STYLE.button_unpressed)
        self.update_plot()

    # ------------------------------------------------------------------ #
    #  ROI selection tool                                                 #
    # ------------------------------------------------------------------ #

    def _on_roi_selection(self) -> None:
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
            self.roi_tool_panel = ROIToolPanel(wplot)
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
            self.roi_tool_active = True

    def _roi_remove(self) -> None:
        """Removes the current rectangular ROI selection and resets button styles."""
        if self.roi_tool_active:
            if self.roi_tool_panel == 0:
                self._cells_view_box.removeItem(self._active_roi_selection)
            else:
                self._noncells_view_box.removeItem(self._active_roi_selection)
            self.roi_tool_active = False
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
            if self.roi_tool_panel == 0
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
        """Selects cells whose pixels overlap the given coordinate arrays."""
        if self.roi_maps is None or self.context_data is None:
            return
        i = self.roi_tool_panel
        roi_indices = self.roi_maps.iroi[i, 0, ypix, xpix]
        icells = np.unique(roi_indices[roi_indices >= 0])
        self.merge_roi_indices = []
        for n in icells:
            pixel_count = self.context_data.roi_statistics[n].pixel_count
            if (self.roi_maps.iroi[i, :, ypix, xpix] == n).sum() > 0.6 * pixel_count:
                self.merge_roi_indices.append(n)
        if len(self.merge_roi_indices) > 0:
            self.selected_roi_index = self.merge_roi_indices[0]
            self.update_plot()
            self.show()

    # ------------------------------------------------------------------ #
    #  Keyboard shortcuts (viewing only)                                  #
    # ------------------------------------------------------------------ #

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:  # noqa: N802
        """Handles keyboard shortcuts for view switching, ROI navigation, and toggles.

        Args:
            event: The key press event from Qt.
        """
        if not self.session_loaded:
            return
        if event.modifiers() in {QtCore.Qt.KeyboardModifier.ControlModifier, QtCore.Qt.KeyboardModifier.ShiftModifier}:
            return
        key = event.key()

        # Background views: Q, W, E, R, T, Y, U.
        if key == QtCore.Qt.Key.Key_Q:
            self._on_view_changed(0)
        elif key == QtCore.Qt.Key.Key_W:
            self._on_view_changed(1)
        elif key == QtCore.Qt.Key.Key_E:
            self._on_view_changed(2)
        elif key == QtCore.Qt.Key.Key_R:
            self._on_view_changed(3)
        elif key == QtCore.Qt.Key.Key_T:
            self._on_view_changed(4)
        elif key == QtCore.Qt.Key.Key_U:
            if self.context_data is not None and self.context_data.mean_image_channel_2 is not None:
                self._on_view_changed(6)
        elif key == QtCore.Qt.Key.Key_Y:
            if self.context_data is not None and self.context_data.corrected_structural_mean_image is not None:
                self._on_view_changed(5)

        # Color modes: A, S, D, F, G, H, J, K.
        elif key == QtCore.Qt.Key.Key_A:
            self._on_color_changed(0)
        elif key == QtCore.Qt.Key.Key_S:
            self._on_color_changed(1)
        elif key == QtCore.Qt.Key.Key_D:
            self._on_color_changed(2)
        elif key == QtCore.Qt.Key.Key_F:
            self._on_color_changed(3)
        elif key == QtCore.Qt.Key.Key_G:
            self._on_color_changed(4)
        elif key == QtCore.Qt.Key.Key_H:
            if self.context_data is not None and self.context_data.has_channel_2:
                self._on_color_changed(5)
        elif key == QtCore.Qt.Key.Key_J:
            self._on_color_changed(6)
        elif key == QtCore.Qt.Key.Key_K:
            self._on_color_changed(7)

        # Toggle ROI visibility.
        elif key == QtCore.Qt.Key.Key_Space:
            self._roi_visibility_checkbox.toggle()

        # Trace visibility toggles.
        elif key == QtCore.Qt.Key.Key_N:
            self._trace_controls.deconvolved_checkbox.toggle()
        elif key == QtCore.Qt.Key.Key_B:
            self._trace_controls.neuropil_checkbox.toggle()
        elif key == QtCore.Qt.Key.Key_V:
            self._trace_controls.traces_checkbox.toggle()

        # ROI navigation: Left/Right.
        elif key == QtCore.Qt.Key.Key_Left:
            if self.context_data is None:
                return
            ctype = self.context_data.cell_classification_labels[self.selected_roi_index]
            roi_count = self.context_data.roi_count
            while True:
                self.selected_roi_index = (self.selected_roi_index - 1) % roi_count
                if self.context_data.cell_classification_labels[self.selected_roi_index] is ctype:
                    break
            self.merge_roi_indices = [self.selected_roi_index]
            self._roi_remove()
            self.update_plot()
        elif key == QtCore.Qt.Key.Key_Right:
            if self.context_data is None:
                return
            self._roi_remove()
            ctype = self.context_data.cell_classification_labels[self.selected_roi_index]
            roi_count = self.context_data.roi_count
            while True:
                self.selected_roi_index = (self.selected_roi_index + 1) % roi_count
                if self.context_data.cell_classification_labels[self.selected_roi_index] is ctype:
                    break
            self.merge_roi_indices = [self.selected_roi_index]
            self.update_plot()
            self.show()

        # Quadrant zoom: 1-9.
        elif QtCore.Qt.Key.Key_1 <= key <= QtCore.Qt.Key.Key_9:
            self._on_quadrant(key - QtCore.Qt.Key.Key_1)

        # Reset zoom.
        elif key == QtCore.Qt.Key.Key_Escape:
            self._zoom_plot(_CELLS_PLOT)
            self._trace_box.autoRange()
            self.show()

        # Delete: deselect multi-selection.
        elif key == QtCore.Qt.Key.Key_Delete:
            self._roi_remove()

    # ------------------------------------------------------------------ #
    #  Drag and drop                                                      #
    # ------------------------------------------------------------------ #

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:  # noqa: N802
        """Accepts drag events that contain file URLs."""
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:  # noqa: N802
        """Handles dropped directories by loading session data."""
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        console.echo(message=f"Files dropped: {files}")
        dropped_path = Path(files[0])
        if dropped_path.is_dir():
            self._load_session(session_path=dropped_path)
        else:
            console.echo(
                message=f"Invalid drop target '{dropped_path}'. Drop a cindra output directory.",
                level=LogLevel.ERROR,
            )
