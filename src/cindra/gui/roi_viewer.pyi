from typing import Any

import numpy as np
from PySide6 import QtGui, QtCore
from _typeshed import Incomplete
from numpy.typing import NDArray as NDArray
from PySide6.QtWidgets import QWidget, QCheckBox, QComboBox, QGroupBox, QMainWindow, QVBoxLayout

from .styles import (
    FONTS as FONTS,
    STYLE as STYLE,
    COLORS as COLORS,
    ROI_STYLE as ROI_STYLE,
)
from .widgets import (
    ViewBox as ViewBox,
    TraceBox as TraceBox,
    plot_trace as plot_trace,
    add_plot_legend as add_plot_legend,
    escape_returns_focus as escape_returns_focus,
)
from .overlays import (
    flip_rois as flip_rois,
    draw_masks as draw_masks,
    build_views as build_views,
    display_masks as display_masks,
    display_views as display_views,
    draw_colorbar as draw_colorbar,
    compute_colors as compute_colors,
    render_colorbar as render_colorbar,
    update_colormap as update_colormap,
    initialize_roi_maps as initialize_roi_maps,
    update_correlation_masks as update_correlation_masks,
    recompute_binary_classification as recompute_binary_classification,
)
from .constants import (
    ROI_CONFIG as ROI_CONFIG,
    Colormap as Colormap,
    TraceMode as TraceMode,
    ROIColorMode as ROIColorMode,
    BackgroundView as BackgroundView,
    TraceModeLabel as TraceModeLabel,
    ROIColorModeLabel as ROIColorModeLabel,
    BackgroundViewLabel as BackgroundViewLabel,
)
from .data_models import (
    ColorArrays as ColorArrays,
    ROIIndexMaps as ROIIndexMaps,
    ColorControls as ColorControls,
    TraceControls as TraceControls,
    ColorbarWidgets as ColorbarWidgets,
    ClassifierControls as ClassifierControls,
)
from ..dataclasses import ROIStatistics as ROIStatistics
from .viewer_context import (
    EMPTY as EMPTY,
    ViewerData as ViewerData,
)
from ..classification import Classifier as Classifier

_STATISTICS_TO_SHOW: tuple[str, ...]

class ROIViewer(QMainWindow):
    _roi_color_mode: int
    _background_view: int
    _roi_colormap: str
    _selected_roi_index: int
    _selected_roi_indices: list[int]
    _temporal_bin_size: int
    _recording_loaded: bool
    _colocalization_threshold: float
    _last_reclassified_index: int
    _classify_mode: bool
    _pre_classify_color_mode: int
    _saved_opacity: int
    _all_recordings_visible: bool
    _context_data: ViewerData | None
    _color_arrays: ColorArrays | None
    _roi_maps: ROIIndexMaps | None
    _colorbar_widgets: ColorbarWidgets | None
    _colorbar_image: NDArray[np.uint8] | None
    _views: NDArray[np.uint8] | None
    _roi_statistics: list[ROIStatistics]
    _cell_classification: NDArray[np.float32]
    _cell_colocalization: NDArray[np.float32]
    _two_channels: bool
    _cell_fluorescence: NDArray[np.float32]
    _neuropil_fluorescence: NDArray[np.float32]
    _subtracted_fluorescence: NDArray[np.float32]
    _spikes: NDArray[np.float32]
    _frame_count: int
    _roi_count: int
    _binned_fluorescence: NDArray[np.float32] | None
    _fluorescence_standard_deviation: NDArray[np.float32] | None
    _frame_indices: NDArray[np.int32] | None
    _graphics_splitter: Incomplete
    _image_widget: Incomplete
    _trace_widget: Incomplete
    _status_bar: Incomplete
    def __init__(self, data: ViewerData) -> None: ...
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None: ...
    def eventFilter(self, source: QtCore.QObject, event: QtCore.QEvent) -> bool: ...
    def load_data(self, data: ViewerData) -> None: ...
    def get_state(self) -> dict[str, Any]: ...
    @property
    def _is_multi_recording(self) -> bool: ...
    def _build_toolbar(self, parent_layout: QVBoxLayout) -> None: ...
    _roi_source_group: Incomplete
    _roi_source_combo: QComboBox
    _roi_selection_group: Incomplete
    _roi_index_edit: Incomplete
    _autoselection_label: Incomplete
    _ranked_count_edit: Incomplete
    _top_button: Incomplete
    _bottom_button: Incomplete
    _channel_group: Incomplete
    _channel_2_button: Incomplete
    def _build_control_panel(self) -> QWidget: ...
    _info_label: Incomplete
    def _build_roi_info_bar(self, parent_layout: QVBoxLayout) -> None: ...
    def _create_view_controls(self) -> tuple[QGroupBox, QComboBox]: ...
    _params_container: Incomplete
    _threshold_label: Incomplete
    _binning_label: Incomplete
    def _create_color_controls(self) -> tuple[QGroupBox, ColorControls]: ...
    def _create_colorbar(self) -> ColorbarWidgets: ...
    def _create_classifier_controls(self) -> tuple[QGroupBox, ClassifierControls]: ...
    _all_recordings_button: Incomplete
    def _create_trace_controls(self) -> tuple[QGroupBox, TraceControls]: ...
    _view_box: Incomplete
    _background: Incomplete
    _overlay: Incomplete
    _trace_box: Incomplete
    def _build_graphics(self) -> None: ...
    def _load_recording(self) -> None: ...
    def _reset_state(self) -> None: ...
    def _initialize_gui(self) -> None: ...
    def _enable_controls(self) -> None: ...
    def _on_view_changed(self, index: int) -> None: ...
    def _on_color_changed(self, index: int) -> None: ...
    @property
    def _active_trace_mode(self) -> int: ...
    def _recompute_binned_fluorescence(self) -> None: ...
    def _on_channel_2_toggled(self, checked: bool) -> None: ...
    def _on_classify_toggled(self, checked: bool) -> None: ...
    def _on_threshold_changed(self) -> None: ...
    def _on_number_chosen(self) -> None: ...
    def _enforce_exclusive_trace(self) -> None: ...
    def _on_trace_toggle(self, toggled: QCheckBox) -> None: ...
    def _refresh_traces(self) -> None: ...
    def _update_plot(self) -> None: ...
    def _update_selected_roi_statistics(self) -> None: ...
    def _select_all_rois(self) -> None: ...
    def _deselect_all_rois(self) -> None: ...
    def _zoom_plot(self) -> None: ...
    def _handle_click(self, click_x: int, click_y: int, is_right_button: bool, is_multi_select: bool) -> bool: ...
    def _on_ranked_selection(self, *, top: bool) -> None: ...
    def _on_dataset_source_changed(self, index: int) -> None: ...
    def _on_all_recordings_toggled(self, checked: bool) -> None: ...
    def _refresh_all_recording_traces(self) -> None: ...
    def _extract_classifier_data(
        self,
    ) -> tuple[NDArray[np.bool_], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]] | None: ...
    def _on_classifier_new(self) -> None: ...
    def _on_classifier_add_to_existing(self) -> None: ...
