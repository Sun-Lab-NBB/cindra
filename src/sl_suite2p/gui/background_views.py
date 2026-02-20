"""Provides background image views and their controls for the main GUI viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

import numpy as np
from PySide6 import QtCore
from PySide6.QtGui import QPainter, QMouseEvent, QPaintEvent
from PySide6.QtWidgets import (
    QLabel,
    QStyle,
    QSlider,
    QWidget,
    QPushButton,
    QApplication,
    QButtonGroup,
    QStyleOptionSlider,
)
from ataraxis_base_utilities import LogLevel, console

from .styles import (
    WHITE_LABEL_STYLESHEET,
    RANGE_SLIDER_STYLESHEET,
    BUTTON_PRESSED_STYLESHEET,
    BUTTON_INACTIVE_STYLESHEET,
    BUTTON_UNPRESSED_STYLESHEET,
    header_font,
    label_font_bold,
)

if TYPE_CHECKING:
    import pyqtgraph as pg
    from numpy.typing import NDArray
    from PySide6.QtWidgets import QGridLayout

    from .signals import GUISignals


# Number of background view types available.
_VIEW_COUNT: int = 7

# Index for the ROI overlay view (no background image).
_VIEW_ROIS: int = 0

# Index for the mean image view.
_VIEW_MEAN: int = 1

# Index for the enhanced mean image view.
_VIEW_ENHANCED: int = 2

# Index for the correlation map view.
_VIEW_CORRELATION: int = 3

# Index for the maximum projection view.
_VIEW_MAX_PROJECTION: int = 4

# Index for the corrected channel 2 mean image view.
_VIEW_CHAN2_CORRECTED: int = 5

# Index for the raw channel 2 mean image view.
_VIEW_CHAN2_RAW: int = 6

# Names displayed on view selection buttons, with keyboard shortcut prefixes.
_VIEW_NAMES: list[str] = [
    "Q: ROIs",
    "W: mean img",
    "E: mean img (enhanced)",
    "R: correlation map",
    "T: max projection",
    "Y: mean img chan2, corr",
    "U: mean img chan2",
]


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
    image_space: int = 0,
    transformed_mean_image: NDArray[np.float32] | None = None,
    transformed_enhanced_mean_image: NDArray[np.float32] | None = None,
    transformed_maximum_projection: NDArray[np.float32] | None = None,
) -> NDArray[np.uint8]:
    """Builds the background view stack from detection images.

    Creates a stack of 7 RGB background images, each normalized to [0, 255] uint8 range.
    Views are indexed as: 0=ROIs (black), 1=mean, 2=enhanced mean, 3=correlation map,
    4=max projection, 5=corrected channel 2, 6=raw channel 2.

    When ``image_space`` is 1 (transformed), the mean, enhanced mean, and max projection
    views use the deformed (registered) images from multi-day processing instead of the
    native single-day images.

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
        image_space: Image space selector (0=native, 1=transformed/deformed).
        transformed_mean_image: Multi-day deformed mean image for transformed space.
        transformed_enhanced_mean_image: Multi-day deformed enhanced mean image.
        transformed_maximum_projection: Multi-day deformed maximum projection.

    Returns:
        Array of shape (7, frame_height, frame_width, 3) containing uint8 RGB views.
    """
    # Selects transformed images when image_space is 1 and they are available.
    if image_space == 1:
        if transformed_mean_image is not None:
            mean_image = transformed_mean_image
        if transformed_enhanced_mean_image is not None:
            enhanced_mean_image = transformed_enhanced_mean_image
        if transformed_maximum_projection is not None:
            maximum_projection = transformed_maximum_projection

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


def create_view_controls(
    owner: QWidget,
    layout: QGridLayout,
    row: int,
    signals: GUISignals,
) -> tuple[ViewControls, int]:
    """Creates background view selection controls and adds them to the layout.

    Builds the view button group, saturation range slider, and a label header.
    Each button corresponds to a different background image type.

    Args:
        owner: The parent QWidget that owns the created controls.
        layout: The grid layout to add widgets to.
        row: Starting row index in the layout.
        signals: The central signal bus for emitting view change events.

    Returns:
        Tuple of (view controls container, next available row index).
    """
    view_buttons = QButtonGroup(owner)

    header = QLabel("Background")
    header.setStyleSheet(WHITE_LABEL_STYLESHEET)
    header.setFont(header_font())
    header.resize(header.minimumSizeHint())
    layout.addWidget(header, row, 0, 1, 1)

    button_index = 0
    for name in _VIEW_NAMES:
        button = _ViewButton(
            button_id=button_index,
            text="&" + name,
            owner=owner,
            button_group=view_buttons,
            signals=signals,
        )
        view_buttons.addButton(button, button_index)
        layout.addWidget(button, button_index + row + 1, 0, 1, 1)

        # Adds the saturation label next to the first button.
        if button_index == 0:
            saturation_label = QLabel("sat: ")
            saturation_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
            layout.addWidget(saturation_label, button_index + row + 1, 1, 1, 1)

        button.setEnabled(False)
        button_index += 1

    view_buttons.setExclusive(True)

    range_slider = RangeSlider(owner=owner, signals=signals)
    range_slider.setMinimum(0)
    range_slider.setMaximum(255)
    range_slider.setLow(0)
    range_slider.setHigh(255)
    range_slider.setTickPosition(QSlider.TicksBelow)
    layout.addWidget(range_slider, row + 2, 1, len(_VIEW_NAMES) - 2, 1)

    next_row = row + button_index + 2

    controls = ViewControls(
        view_buttons=view_buttons,
        range_slider=range_slider,
    )
    return controls, next_row


class _ViewButton(QPushButton):
    """Background view selection button.

    Each button corresponds to a different background image type (mean image,
    enhanced mean, correlation map, etc.). On press, emits ``view_mode_changed``
    on the signal bus.

    Args:
        button_id: Zero-based index identifying this button's view type.
        text: Display label for the button (with keyboard shortcut prefix).
        owner: The parent QWidget.
        button_group: The button group this button belongs to.
        signals: The central signal bus for emitting view mode changes.
    """

    def __init__(
        self,
        button_id: int,
        text: str,
        owner: QWidget,
        button_group: QButtonGroup,
        signals: GUISignals,
    ) -> None:
        super().__init__(owner)
        self.setText(text)
        self.setCheckable(True)
        self.setStyleSheet(BUTTON_INACTIVE_STYLESHEET)
        self.setFont(label_font_bold())
        self.resize(self.minimumSizeHint())
        self._button_id: int = button_id
        self._button_group: QButtonGroup = button_group
        self._signals: GUISignals = signals
        self.clicked.connect(self._press)
        self.show()

    def _press(self) -> None:
        """Switches the background view and emits the view_mode_changed signal."""
        for index in range(_VIEW_COUNT):
            button = self._button_group.button(index)
            if button is not None and button.isEnabled():
                button.setStyleSheet(BUTTON_UNPRESSED_STYLESHEET)
        self.setStyleSheet(BUTTON_PRESSED_STYLESHEET)
        self._signals.view_mode_changed.emit(self._button_id)


class RangeSlider(QSlider):
    """Dual-handle range slider for controlling image saturation levels.

    Provides two independently movable slider handles that define a low-high range.
    Dragging between the handles moves both together. Releasing the mouse emits
    ``plot_needs_update`` on the signal bus.

    Args:
        owner: The parent QWidget.
        signals: The central signal bus for emitting plot update requests.
    """

    def __init__(self, owner: QWidget | None = None, signals: GUISignals | None = None) -> None:
        super().__init__(owner)

        self._low: int = self.minimum()
        self._high: int = self.maximum()
        self._pressed_control = QStyle.SC_None
        self._hover_control = QStyle.SC_None
        self._click_offset: int = 0

        self.setOrientation(QtCore.Qt.Vertical)
        self.setTickPosition(QSlider.TicksRight)
        self.setStyleSheet(RANGE_SLIDER_STYLESHEET)

        # 0 for the low handle, 1 for the high handle, -1 for both.
        self._active_slider: int = 0
        self._signals: GUISignals | None = signals

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
            option.subControls = QStyle.SC_SliderHandle

            if self.tickPosition() != self.NoTicks:
                option.subControls |= QStyle.SC_SliderTickmarks

            if self._pressed_control:
                option.activeSubControls = self._pressed_control
                option.state |= QStyle.State_Sunken
            else:
                option.activeSubControls = self._hover_control

            option.sliderPosition = value
            option.sliderValue = value
            style.drawComplexControl(QStyle.CC_Slider, option, painter, self)

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
                hit = style.hitTestComplexControl(
                    style.CC_Slider, option, event.pos(), self
                )
                if hit == style.SC_SliderHandle:
                    self._active_slider = handle_index
                    self._pressed_control = hit
                    self.triggerAction(self.SliderMove)
                    self.setRepeatAction(self.SliderNoAction)
                    self.setSliderDown(True)
                    break

            if self._active_slider < 0:
                self._pressed_control = QStyle.SC_SliderHandle
                self._click_offset = self._pixel_position_to_value(
                    self._pick_coordinate(event.pos())
                )
                self.triggerAction(self.SliderMove)
                self.setRepeatAction(self.SliderNoAction)
        else:
            event.ignore()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        """Handles mouse drag to move the active slider handle."""
        if self._pressed_control != QStyle.SC_SliderHandle:
            event.ignore()
            return

        event.accept()
        new_position = self._pixel_position_to_value(
            self._pick_coordinate(event.pos())
        )
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
        """Emits a plot update request after the slider handles are released."""
        if self._signals is not None:
            self._signals.plot_needs_update.emit()

    def _pick_coordinate(self, point: QtCore.QPoint) -> int:
        """Extracts the relevant coordinate from a point based on slider orientation.

        Args:
            point: The mouse position.

        Returns:
            The x or y coordinate depending on the slider orientation.
        """
        if self.orientation() == QtCore.Qt.Horizontal:
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
            style.CC_Slider, option, style.SC_SliderGroove, self
        )
        handle_rect = style.subControlRect(
            style.CC_Slider, option, style.SC_SliderHandle, self
        )

        if self.orientation() == QtCore.Qt.Horizontal:
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
