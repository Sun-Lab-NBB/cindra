from typing import Any

import numpy as np
from PySide6 import QtGui, QtCore
from _typeshed import Incomplete
import pyqtgraph as pg
from numpy.typing import NDArray as NDArray
from PySide6.QtWidgets import QLabel, QSlider, QWidget, QLineEdit, QGridLayout, QMainWindow, QPushButton, QToolButton

from .styles import (
    FONTS as FONTS,
    STYLE as STYLE,
    COLORS as COLORS,
    PLOT_STYLE as PLOT_STYLE,
    BINARY_STYLE as BINARY_STYLE,
)
from .widgets import (
    configure_plot as configure_plot,
    add_plot_legend as add_plot_legend,
    escape_returns_focus as escape_returns_focus,
    create_play_pause_group as create_play_pause_group,
)
from .constants import BINARY_CONFIG as BINARY_CONFIG
from .viewer_context import SingleRecordingData as SingleRecordingData

class BinaryPlayer(QMainWindow):
    recording_changed: Incomplete
    _central_widget: QWidget
    _layout: QGridLayout
    _channel_2_visible: bool
    data: SingleRecordingData
    _current_frame: int
    _frame_delta: int
    _display_range: NDArray[np.float32]
    _time_step: float
    _image: NDArray[np.int16] | None
    _average_rigid_y_offsets: NDArray[np.float32]
    _average_rigid_x_offsets: NDArray[np.float32]
    _file_button: QPushButton
    _channel_2_button: QPushButton
    _graphics_widget: pg.GraphicsLayoutWidget
    _main_view_box: pg.ViewBox
    _main_image: pg.ImageItem
    _offset_plot: Incomplete
    _step_edit: QLineEdit
    _frame_number_label: QLabel
    _frame_slider: QSlider
    _update_timer: QtCore.QTimer
    def __init__(self, data: SingleRecordingData) -> None: ...
    def load_data(self, data: SingleRecordingData) -> None: ...
    def get_state(self) -> dict[str, Any]: ...
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None: ...
    def eventFilter(self, source: QtCore.QObject, event: QtCore.QEvent) -> bool: ...
    _skip_backward_button: QToolButton
    _play_button: Incomplete
    _pause_button: Incomplete
    _skip_forward_button: QToolButton
    def _create_buttons(self) -> None: ...
    def _update_frame_slider(self) -> None: ...
    def _update_buttons(self) -> None: ...
    def _apply_step(self) -> None: ...
    def _load_recording(self) -> None: ...
    _offset_scatter: Incomplete
    def _setup_views(self) -> None: ...
    def _update_offset_scatter(self, frame_index: int) -> None: ...
    def _next_frame(self) -> None: ...
    def _render_frame(self) -> None: ...
    def _go_to_frame(self) -> None: ...
    def _step_backward(self) -> None: ...
    def _step_forward(self) -> None: ...
    def _start_playback(self) -> None: ...
    def _pause_playback(self) -> None: ...
    def _toggle_channel_2(self) -> None: ...
    def _zoom_image(self) -> None: ...
    def _plot_clicked(self, event: object) -> None: ...
