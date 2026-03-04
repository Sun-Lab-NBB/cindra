"""Provides widget-reference and rendering-state dataclasses for all GUI applications."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

from .constants import ROI_CONFIG

if TYPE_CHECKING:
    import numpy as np
    import pyqtgraph as pg  # type: ignore[import-untyped]
    from numpy.typing import NDArray
    from PySide6.QtWidgets import QLabel, QSlider, QCheckBox, QComboBox, QLineEdit, QPushButton


@dataclass(frozen=True)
class ColorControls:
    """Holds references to color statistic control widgets."""

    color_combo: QComboBox
    """The dropdown for selecting the active color statistic."""
    colormap_chooser: QComboBox
    """The dropdown for selecting the active colormap."""
    classifier_edit: QLineEdit
    """The text input for the classifier probability threshold."""
    binning_edit: QLineEdit
    """The text input for the binning size."""
    classification_label_button: QPushButton
    """The checkable push button for toggling between probability gradient and binary cell/non-cell label views in the
    cell classification color mode."""


@dataclass(frozen=True)
class ColorbarWidgets:
    """Holds references to the colorbar display widgets."""

    image: pg.ImageItem
    """The pyqtgraph image item displaying the colorbar gradient."""
    labels: list[pg.LabelItem]
    """The three label items showing the low, mid, and high colorbar values."""
    widget: pg.GraphicsLayoutWidget
    """The pyqtgraph GraphicsLayoutWidget containing the colorbar."""


@dataclass
class ColorArrays:
    """Holds all computed color data for ROI overlay rendering."""

    colors: NDArray[np.uint8]
    """The per-statistic RGB colors with shape (color_count, roi_count, 3)."""
    normalized_statistics: NDArray[np.float32]
    """The per-statistic normalized values with shape (color_count, roi_count)."""
    colorbar: list[list[float]]
    """The per-statistic colorbar range values as [low, mid, high] lists."""
    rgb: NDArray[np.uint8]
    """The RGBA overlay arrays with shape (color_count, height, width, 4)."""
    random_hues: NDArray[np.float64]
    """The per-ROI random hue values with shape (roi_count,)."""
    classification_label_colors: NDArray[np.uint8]
    """The per-ROI binary cell/non-cell RGB colors with shape (roi_count, 3), used as the secondary color set for the
    CELL_CLASSIFICATION slot when the label toggle is active."""


@dataclass
class ROIIndexMaps:
    """Holds the multi-layer ROI index maps for overlay rendering."""

    roi_presence: NDArray[np.bool_]
    """The boolean presence map with shape (height, width)."""
    roi_indices: NDArray[np.int32]
    """The ROI index layers with shape (overlap_layers, height, width)."""
    text_labels: list[pg.TextItem] = field(default_factory=list)
    """The per-ROI text label items for centroid display."""


@dataclass(frozen=True)
class ViewControls:
    """Holds references to background view panel widgets."""

    view_combo: QComboBox
    """The dropdown for selecting the active background view."""
    channel_2_button: QPushButton
    """The checkable push button for toggling channel 2 images."""
    opacity_slider: QSlider
    """The slider controlling ROI mask overlay opacity (0-255)."""


@dataclass
class TraceControls:
    """Holds references to trace panel widgets and their mutable state."""

    activity_combo: QComboBox
    """The combo box for selecting the activity mode."""
    deconvolved_checkbox: QCheckBox
    """The checkbox toggling deconvolved spike trace visibility."""
    neuropil_checkbox: QCheckBox
    """The checkbox toggling neuropil fluorescence trace visibility."""
    traces_checkbox: QCheckBox
    """The checkbox toggling raw fluorescence trace visibility."""
    max_plotted_edit: QLineEdit
    """The text input for the maximum number of plotted traces."""
    deconvolved_visible: bool = True
    """Determines whether the deconvolved trace is drawn."""
    neuropil_visible: bool = True
    """Determines whether the neuropil trace is drawn."""
    traces_visible: bool = True
    """Determines whether the raw fluorescence trace is drawn."""


@dataclass
class SelectionControls:
    """Holds references to cell selection widgets and their mutable state."""

    selection_combo: QComboBox
    """The dropdown for selecting the cell selection mode."""
    top_count_edit: QLineEdit
    """The text input for the number of top/bottom cells to select."""
    top_count: int = ROI_CONFIG.top_selection_count
    """The current top-n/bottom-n count value."""


@dataclass(frozen=True)
class ClassifierControls:
    """Holds references to the classifier builder panel widgets."""

    new_button: QPushButton
    """The button to create a new classifier training dataset file."""
    add_button: QPushButton
    """The button to append samples to an existing classifier training dataset file."""
    status_label: QLabel
    """The label showing the result of the last classifier operation."""
