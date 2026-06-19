"""Provides widget-reference and rendering-state dataclasses for the ROI viewer GUI."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

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
    threshold_edit: QLineEdit
    """The text input for the classifier probability threshold used by the cell classification mode."""
    binning_edit: QLineEdit
    """The text input for the temporal bin size used by the activity correlation mode."""
    opacity_slider: QSlider
    """The slider controlling ROI mask overlay opacity (0-255)."""


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
    random_hues: NDArray[np.float32]
    """The per-ROI random hue values with shape (roi_count,)."""


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
class TraceControls:
    """Holds references to trace panel widgets."""

    fluorescence_checkbox: QCheckBox
    """The checkbox toggling raw fluorescence trace visibility."""
    neuropil_checkbox: QCheckBox
    """The checkbox toggling neuropil fluorescence trace visibility."""
    corrected_checkbox: QCheckBox
    """The checkbox toggling neuropil-corrected fluorescence trace visibility."""
    spikes_checkbox: QCheckBox
    """The checkbox toggling deconvolved spike trace visibility."""
    maximum_trace_count_edit: QLineEdit
    """The text input for the maximum number of plotted traces."""


@dataclass(frozen=True)
class ClassifierControls:
    """Holds references to the classifier builder panel widgets."""

    classify_button: QPushButton
    """The checkable push button for toggling classifier mode, where clicks flip ROIs between cell and non-cell labels
    instead of selecting them for trace plotting."""
    new_button: QPushButton
    """The button to create a new classifier training dataset file."""
    add_button: QPushButton
    """The button to append samples to an existing classifier training dataset file."""
    status_label: QLabel
    """The label showing the result of the last classifier operation."""
