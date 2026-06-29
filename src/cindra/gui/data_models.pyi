from dataclasses import field, dataclass

import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray as NDArray
from PySide6.QtWidgets import QLabel, QSlider, QCheckBox, QComboBox, QLineEdit, QPushButton

@dataclass(frozen=True, slots=True)
class ColorControls:
    color_combo: QComboBox
    colormap_chooser: QComboBox
    threshold_edit: QLineEdit
    binning_edit: QLineEdit
    opacity_slider: QSlider

@dataclass(frozen=True, slots=True)
class ColorbarWidgets:
    image: pg.ImageItem
    labels: list[pg.LabelItem]
    widget: pg.GraphicsLayoutWidget

@dataclass(slots=True)
class ColorArrays:
    colors: NDArray[np.uint8]
    normalized_statistics: NDArray[np.float32]
    colorbar: list[list[float]]
    rgb: NDArray[np.uint8]
    random_hues: NDArray[np.float32]

@dataclass(slots=True)
class ROIIndexMaps:
    roi_presence: NDArray[np.bool_]
    roi_indices: NDArray[np.int32]
    text_labels: list[pg.TextItem] = field(default_factory=list)

@dataclass(frozen=True, slots=True)
class TraceControls:
    fluorescence_checkbox: QCheckBox
    neuropil_checkbox: QCheckBox
    corrected_checkbox: QCheckBox
    spikes_checkbox: QCheckBox
    maximum_trace_count_edit: QLineEdit

@dataclass(frozen=True, slots=True)
class ClassifierControls:
    classify_button: QPushButton
    new_button: QPushButton
    add_button: QPushButton
    status_label: QLabel
