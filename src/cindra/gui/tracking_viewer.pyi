from typing import Any
from collections.abc import Sequence

import numpy as np
from PySide6 import QtGui, QtCore
from _typeshed import Incomplete
import pyqtgraph as pg
from numpy.typing import NDArray as NDArray
from PySide6.QtWidgets import QWidget, QMainWindow, QPushButton, QToolButton

from .styles import (
    FONTS as FONTS,
    STYLE as STYLE,
    TRACKING_STYLE as TRACKING_STYLE,
)
from .widgets import (
    escape_returns_focus as escape_returns_focus,
    create_play_pause_group as create_play_pause_group,
)
from .overlays import normalize_percentile as normalize_percentile
from .constants import (
    ROI_CONFIG as ROI_CONFIG,
    TRACKING_CONFIG as TRACKING_CONFIG,
    MaskLayer as MaskLayer,
    BackgroundView as BackgroundView,
    CoordinateSpace as CoordinateSpace,
    BackgroundViewLabel as BackgroundViewLabel,
)
from ..dataclasses import (
    ROIMask as ROIMask,
    ROIStatistics as ROIStatistics,
)
from .viewer_context import (
    EMPTY as EMPTY,
    ViewerData as ViewerData,
    MultiRecordingData as MultiRecordingData,
)

class TrackingViewer(QMainWindow):
    data: ViewerData
    _auto_cycle_timer: QtCore.QTimer
    _cached_background: NDArray[np.uint8] | None
    _cached_mask_y: NDArray[np.int32] | None
    _cached_mask_x: NDArray[np.int32] | None
    _cached_mask_colors: NDArray[np.uint8] | None
    _cached_mask_roi_indices: NDArray[np.int32] | None
    _cached_roi_map: NDArray[np.int32] | None
    _cached_mask_count: int
    _selected_rois: set[int] | None
    _selection_was_template: bool
    _selection_recording_index: int
    _file_button: QPushButton
    _graphics_widget: Incomplete
    _view_box: pg.ViewBox
    _image_item: pg.ImageItem
    _status_bar: Incomplete
    def __init__(self, data: ViewerData) -> None: ...
    def load_data(self, data: ViewerData) -> None: ...
    def get_state(self) -> dict[str, Any]: ...
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None: ...
    def eventFilter(self, source: QtCore.QObject, event: QtCore.QEvent) -> bool: ...
    def _load_dataset(self) -> None: ...
    _dataset_combo: Incomplete
    _recording_combo: Incomplete
    _skip_backward_button: QToolButton
    _play_button: Incomplete
    _pause_button: Incomplete
    _skip_forward_button: QToolButton
    _background_combo: Incomplete
    _space_combo: Incomplete
    _mask_combo: Incomplete
    _opacity_slider: Incomplete
    _channel_group: Incomplete
    _channel_2_checkbox: Incomplete
    _roi_edit: Incomplete
    def _build_control_panel(self) -> QWidget: ...
    def _refresh_display(self) -> None: ...
    def _composite_and_display(self) -> None: ...
    def _on_recording_selected(self, index: int) -> None: ...
    def _on_dataset_selected(self, index: int) -> None: ...
    def _on_channel_2_toggled(self, checked: bool) -> None: ...
    def _on_opacity_changed(self) -> None: ...
    def _on_image_clicked(self, event: object) -> None: ...
    def _on_roi_entered(self) -> None: ...
    def _select_all_rois(self) -> None: ...
    def _deselect_all_rois(self) -> None: ...
    def _previous_recording(self) -> None: ...
    def _next_recording(self) -> None: ...
    def _advance_recording(self) -> None: ...
    def _start_cycling(self) -> None: ...
    def _stop_cycling(self) -> None: ...
    def _normalize_image(self, image: NDArray[np.float32]) -> NDArray[np.uint8]: ...
    @staticmethod
    def _resolve_background(
        recording: MultiRecordingData, image_type: BackgroundView, coordinate_space: CoordinateSpace, channel_2: bool
    ) -> NDArray[np.float32]: ...
    @staticmethod
    def _resolve_masks(
        recording: MultiRecordingData, layer: MaskLayer, channel_2: bool
    ) -> Sequence[ROIMask | ROIStatistics]: ...
