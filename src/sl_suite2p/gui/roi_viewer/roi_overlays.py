"""Provides ROI color overlays, mask rendering, and color statistic controls."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

import numpy as np
from PySide6 import QtGui, QtCore
import pyqtgraph as pg
import matplotlib.cm
from PySide6.QtWidgets import QLabel, QComboBox, QLineEdit, QPushButton, QButtonGroup
from matplotlib.colors import hsv_to_rgb
from ataraxis_base_utilities import LogLevel, console

from .styles import (
    COLORBAR_MAX_WIDTH,
    COLORBAR_MAX_HEIGHT,
    WHITE_LABEL_STYLESHEET,
    BUTTON_INACTIVE_STYLESHEET,
    label_font,
    header_font,
    label_font_bold,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from PySide6.QtWidgets import QWidget, QGridLayout

    from .signals import GUISignals
    from .view_state import ViewState
    from .context_data import ContextData
    from ...dataclasses.single_day_data import ROIStatistics

# Width for color panel edit fields and the colormap combo box.
_COLOR_EDIT_WIDTH: int = 65

# Number of overlap layers stored in the ROI index map.
_OVERLAP_LAYERS: int = 3

# ROI weight normalization scale factor.
_LAMBDA_NORM_SCALE: float = 0.75

# Minimum lambda value threshold for computing the mean weight.
_LAMBDA_THRESHOLD: float = 1e-10

# Number of color statistics (random, skew, compact, footprint, aspect, chan2, class, corr).
_COLOR_STAT_COUNT: int = 8

# Fixed colorbar range for statistics without computed percentiles.
_FIXED_COLORBAR_RANGE: list[float] = [0.0, 0.5, 1.0]

# Percentile values for computing istat normalization bounds.
_LOWER_PERCENTILE: float = 2.0
_UPPER_PERCENTILE: float = 98.0

# Random color adjustment factors for channel 2 data.
_CHAN2_COLOR_DIVISOR: float = 1.4
_CHAN2_COLOR_OFFSET: float = 0.1

# HSV transform normalization constants.
_HSV_DIVISOR: float = 1.4
_HSV_OFFSET: float = 0.4

# Number of samples for the colorbar gradient.
_COLORBAR_SAMPLE_COUNT: int = 101

# Number of rows in the colorbar image.
_COLORBAR_ROW_COUNT: int = 20

# Available colormaps for the colormap chooser.
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

# Color statistic names displayed on the color buttons.
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

# Short names extracted from the color buttons (after the prefix).
_COLOR_SHORT_NAMES: list[str] = [name[3:] for name in _COLOR_NAMES]

# Color button indices that require the adjacent edit field column.
_COLOR_NARROW_RANGE_START: int = 5
_COLOR_NARROW_RANGE_END: int = 8

# Channel 2 color index.
_COLOR_CHAN2: int = 5

# Classifier probability color index.
_COLOR_CLASSIFIER: int = 6

# Correlation color index.
_COLOR_CORRELATION: int = 7

# Minimum number of changed cells before incremental flip is used over full reinit.
_FLIP_THRESHOLD: int = 100

# Minimum change in channel 2 threshold to trigger a recoloring update.
_CHAN2_THRESHOLD_EPSILON: float = 1e-3

# Font size for ROI text labels.
_ROI_TEXT_SIZE: int = 8

# Color for ROI text labels.
_ROI_TEXT_COLOR: tuple[int, int, int] = (180, 180, 180)

# Maps color statistic display names to ROIStatistics attribute names.
_STAT_FIELD_MAP: dict[str, str] = {
    "skew": "skewness",
    "compact": "compactness",
    "footprint": "footprint",
    "aspect_ratio": "aspect_ratio",
    "chan2_prob": "colocalization_probability",
}


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
    """

    image: pg.ImageItem
    labels: list[pg.LabelItem]


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


def build_color_controls(
    owner: QWidget,
    layout: QGridLayout,
    row: int,
    signals: GUISignals,
) -> tuple[ColorControls, int]:
    """Creates color statistic selection buttons and their associated controls.

    Builds a button group for selecting which color statistic to display on the ROI
    overlays, plus a colormap chooser dropdown, and edit fields for channel 2 threshold,
    classifier probability, and binning size.

    Args:
        owner: The parent widget for ownership of created widgets.
        layout: The grid layout to add widgets to.
        row: Starting row index in the layout.
        signals: The central signal bus for GUI events.

    Returns:
        Tuple of (color controls container, next available row index).
    """
    color_buttons = QButtonGroup(owner)

    color_label = QLabel("Colors")
    color_label.setStyleSheet(WHITE_LABEL_STYLESHEET)
    color_label.setFont(header_font())
    layout.addWidget(color_label, row, 0, 1, 1)

    # Colormap chooser dropdown.
    colormap_chooser = QComboBox()
    colormap_chooser.addItems(_COLORMAPS)
    colormap_chooser.setCurrentIndex(0)
    colormap_chooser.setFont(label_font())
    colormap_chooser.setFixedWidth(_COLOR_EDIT_WIDTH)
    layout.addWidget(colormap_chooser, row, 1, 1, 1)

    # Color statistic buttons.
    button_row = row
    for button_index, name in enumerate(_COLOR_NAMES):
        button = _ColorButton(
            button_id=button_index,
            text="&" + name,
            owner=owner,
            button_group=color_buttons,
            signals=signals,
        )
        color_buttons.addButton(button, button_index)

        if _COLOR_NARROW_RANGE_START <= button_index < _COLOR_NARROW_RANGE_END:
            layout.addWidget(button, button_row + button_index + 1, 0, 1, 1)
        else:
            layout.addWidget(button, button_row + button_index + 1, 0, 1, 2)

        button.setEnabled(False)

    # Channel 2 probability threshold edit.
    channel_2_edit = QLineEdit(owner)
    channel_2_edit.setText("0.6")
    channel_2_edit.setFixedWidth(_COLOR_EDIT_WIDTH)
    channel_2_edit.setAlignment(QtCore.Qt.AlignRight)
    layout.addWidget(channel_2_edit, button_row + len(_COLOR_NAMES) - 4, 1, 1, 1)

    # Classifier probability edit.
    classifier_edit = QLineEdit(owner)
    classifier_edit.setText("0.5")
    classifier_edit.setFixedWidth(_COLOR_EDIT_WIDTH)
    classifier_edit.setAlignment(QtCore.Qt.AlignRight)
    layout.addWidget(classifier_edit, button_row + len(_COLOR_NAMES) - 3, 1, 1, 1)

    # Binning size edit.
    bin_edit = QLineEdit(owner)
    bin_edit.setValidator(QtGui.QIntValidator(0, 500))
    bin_edit.setText("1")
    bin_edit.setFixedWidth(_COLOR_EDIT_WIDTH)
    bin_edit.setAlignment(QtCore.Qt.AlignRight)
    layout.addWidget(bin_edit, button_row + len(_COLOR_NAMES) - 2, 1, 1, 1)

    controls = ColorControls(
        color_buttons=color_buttons,
        colormap_chooser=colormap_chooser,
        channel_2_edit=channel_2_edit,
        classifier_edit=classifier_edit,
        bin_edit=bin_edit,
    )

    # Connects signals. Colormap and edit press callbacks are handled by MainWindow.
    colormap_chooser.activated.connect(signals.color_mode_changed.emit)
    channel_2_edit.returnPressed.connect(signals.plot_needs_update.emit)
    classifier_edit.returnPressed.connect(signals.plot_needs_update.emit)
    bin_edit.returnPressed.connect(signals.activity_mode_changed.emit)

    return controls, button_row + len(_COLOR_NAMES) + 2


def build_colorbar(
    owner: QWidget,
    layout: QGridLayout,
    row: int,
) -> ColorbarWidgets:
    """Creates the colorbar widget displaying the current color mapping.

    Args:
        owner: The parent widget for ownership of created widgets.
        layout: The grid layout to add widgets to.
        row: Row index in the layout for the colorbar.

    Returns:
        Colorbar widgets container.
    """
    colorbar_widget = pg.GraphicsLayoutWidget(owner)
    colorbar_widget.setMaximumHeight(COLORBAR_MAX_HEIGHT)
    colorbar_widget.setMaximumWidth(COLORBAR_MAX_WIDTH)
    colorbar_widget.ci.layout.setRowStretchFactor(0, 2)
    colorbar_widget.ci.layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(colorbar_widget, row, 0, 1, 2)

    image = pg.ImageItem()
    colorbar_view = colorbar_widget.addViewBox(row=0, col=0, colspan=3)
    colorbar_view.setMenuEnabled(False)
    colorbar_view.addItem(image)

    labels = [
        colorbar_widget.addLabel("0.0", color=[255, 255, 255], row=1, col=0),
        colorbar_widget.addLabel("0.5", color=[255, 255, 255], row=1, col=1),
        colorbar_widget.addLabel("1.0", color=[255, 255, 255], row=1, col=2),
    ]

    return ColorbarWidgets(image=image, labels=labels)


def compute_colors(
    context: ContextData,
    state: ViewState,
) -> ColorArrays:
    """Computes color statistics and RGB color arrays for all ROIs.

    Initializes per-statistic color arrays and normalization values. The first color
    channel uses random HSV coloring; subsequent channels use computed statistics (skew,
    compact, footprint, aspect ratio, etc.).

    Args:
        context: The loaded data context.
        state: The current GUI display state.

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
    if context.has_channel_2:
        random_colors = random_colors / _CHAN2_COLOR_DIVISOR + _CHAN2_COLOR_OFFSET
        is_channel_2 = context.cell_colocalization_probabilities > state.colocalization_threshold
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

            color = istat_transform(stat_values, state.roi_colormap)
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
    context: ContextData,
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

    for roi_index in np.arange(cell_count - 1, -1, -1, int):
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
        text_item.setFont(QtGui.QFont("Times", _ROI_TEXT_SIZE, weight=QtGui.QFont.Bold))
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
    context: ContextData,
    state: ViewState,
    color_arrays: ColorArrays,
    roi_maps: ROIIndexMaps,
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Draws the current mask overlay for both cell and non-cell panels.

    Computes transparency based on ROI weights, then highlights the currently
    selected ROIs with full-white (ROI view) or colored circles (image views).

    Args:
        context: The loaded data context.
        state: The current GUI display state.
        color_arrays: The computed color arrays.
        roi_maps: The ROI index maps.

    Returns:
        Tuple of (cells_overlay, noncells_overlay) RGBA arrays.
    """
    color_index = state.roi_color_mode
    view_index = state.background_view
    opacity = state.roi_opacity

    active_panel = int(1 - context.cell_classification_labels[state.selected_roi_index])

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
        for roi_index in state.merge_roi_indices:
            y_pixels = roi_statistics[roi_index].y_pixels.flatten()
            x_pixels = roi_statistics[roi_index].x_pixels.flatten()
            overlap_count = (roi_maps.iroi[active_panel][:, y_pixels, x_pixels] > -1).sum(axis=0) - 1
            brightness = 1 - overlap_count / _OVERLAP_LAYERS
            overlays[active_panel] = _make_chosen_roi(overlays[active_panel], y_pixels, x_pixels, brightness)
    else:
        # Image view: highlights selected ROIs with colored circles.
        for roi_index in state.merge_roi_indices:
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
    state: ViewState,
    color_arrays: ColorArrays,
    colorbar_widgets: ColorbarWidgets,
    colorbar_image: NDArray[np.uint8],
) -> None:
    """Updates the colorbar image and tick labels for the active color mode.

    Args:
        state: The current GUI display state.
        color_arrays: The computed color arrays.
        colorbar_widgets: The colorbar display widgets.
        colorbar_image: The colorbar gradient image from ``draw_colorbar``.
    """
    color_index = state.roi_color_mode
    if color_index == 0:
        colorbar_widgets.image.setImage(np.zeros((_COLORBAR_ROW_COUNT, _COLORBAR_SAMPLE_COUNT - 1, 3)))
    else:
        colorbar_widgets.image.setImage(colorbar_image)

    for label_index in range(3):
        colorbar_widgets.labels[label_index].setText(f"{color_arrays.colorbar[color_index][label_index]:1.2f}")


def flip_rois(
    context: ContextData,
    state: ViewState,
    color_arrays: ColorArrays,
    roi_maps: ROIIndexMaps,
) -> None:
    """Flips selected ROIs between the cell and non-cell panels.

    Toggles the ``cell_classification_labels`` for all ROIs in ``state.merge_roi_indices``,
    moves them between panels. The caller is responsible for saving and updating the plot.

    Args:
        context: The loaded data context (cell_classification_labels is modified in place).
        state: The current GUI display state.
        color_arrays: The computed color arrays.
        roi_maps: The ROI index maps.
    """
    state.last_reclassified_index = state.selected_roi_index
    for roi_index in state.merge_roi_indices:
        context.cell_classification_labels[roi_index] = ~context.cell_classification_labels[roi_index]
        _flip_roi(
            roi_maps=roi_maps,
            color_arrays=color_arrays,
            roi_statistics=context.roi_statistics,
            cell_classification_labels=context.cell_classification_labels,
            roi_index=roi_index,
        )
        merged = context.roi_statistics[roi_index].merged_roi_indices
        if merged is not None:
            for merged_index in merged:
                context.cell_classification_labels[merged_index] = ~context.cell_classification_labels[merged_index]


def flip_for_class(
    context: ContextData,
    color_arrays: ColorArrays,
    roi_maps: ROIIndexMaps,
    new_classification_labels: NDArray[np.bool_],
) -> bool:
    """Applies new cell classification labels from the classifier.

    For small numbers of changes, flips individual ROIs incrementally. For large
    changes, returns False to signal the caller should reinitialize all masks.

    Args:
        context: The loaded data context.
        color_arrays: The computed color arrays.
        roi_maps: The ROI index maps.
        new_classification_labels: New cell classification label array.

    Returns:
        True if changes were applied incrementally, False if full reinit is needed.
    """
    cell_count = new_classification_labels.size
    if int((new_classification_labels == context.cell_classification_labels).sum()) < _FLIP_THRESHOLD:
        for roi_index in range(cell_count):
            if new_classification_labels[roi_index] != context.cell_classification_labels[roi_index]:
                context.cell_classification_labels[roi_index] = new_classification_labels[roi_index]
                _flip_roi(
                    roi_maps=roi_maps,
                    color_arrays=color_arrays,
                    roi_statistics=context.roi_statistics,
                    cell_classification_labels=context.cell_classification_labels,
                    roi_index=roi_index,
                )
        return True

    context.cell_classification_labels[:] = new_classification_labels
    return False


def add_roi(
    roi_maps: ROIIndexMaps,
    roi_statistics: list[ROIStatistics],
    roi_index: int,
    panel: int,
) -> None:
    """Adds an ROI to a specific panel on top of existing layers.

    Args:
        roi_maps: The ROI index maps.
        roi_statistics: The ROI statistics list.
        roi_index: Index of the ROI to add.
        panel: Panel index (0=cells, 1=non-cells).
    """
    y_pixels = roi_statistics[roi_index].y_pixels
    x_pixels = roi_statistics[roi_index].x_pixels
    weights = roi_statistics[roi_index].pixel_weights

    # Pushes existing layers down.
    roi_maps.iroi[panel, 2, y_pixels, x_pixels] = roi_maps.iroi[panel, 1, y_pixels, x_pixels]
    roi_maps.iroi[panel, 1, y_pixels, x_pixels] = roi_maps.iroi[panel, 0, y_pixels, x_pixels]
    roi_maps.iroi[panel, 0, y_pixels, x_pixels] = roi_index
    roi_maps.lam[panel, 2, y_pixels, x_pixels] = roi_maps.lam[panel, 1, y_pixels, x_pixels]
    roi_maps.lam[panel, 1, y_pixels, x_pixels] = roi_maps.lam[panel, 0, y_pixels, x_pixels]
    roi_maps.lam[panel, 0, y_pixels, x_pixels] = weights

    roi_maps.sroi[panel, y_pixels, x_pixels] = True
    roi_maps.lam_norm[:, y_pixels, x_pixels] = np.maximum(
        0,
        np.minimum(1, _LAMBDA_NORM_SCALE * roi_maps.lam[:, 0, y_pixels, x_pixels] / roi_maps.lam_mean),
    )


def remove_roi(
    roi_maps: ROIIndexMaps,
    roi_statistics: list[ROIStatistics],
    roi_index: int,
    panel: int,
) -> None:
    """Removes an ROI from a specific panel and shifts overlap layers up.

    Args:
        roi_maps: The ROI index maps.
        roi_statistics: The ROI statistics list.
        roi_index: Index of the ROI to remove.
        panel: Panel index (0=cells, 1=non-cells).
    """
    y_pixels = roi_statistics[roi_index].y_pixels
    x_pixels = roi_statistics[roi_index].x_pixels

    # Finds pixels at each overlap layer where this ROI appears.
    layer0_pixels = np.array((roi_maps.iroi[panel, 0, :, :] == roi_index).nonzero()).astype(np.int32)
    layer1_pixels = np.array((roi_maps.iroi[panel, 1, :, :] == roi_index).nonzero()).astype(np.int32)
    layer2_pixels = np.array((roi_maps.iroi[panel, 2, :, :] == roi_index).nonzero()).astype(np.int32)

    # Shifts layers up to fill the gap.
    roi_maps.lam[panel, 0, layer0_pixels[0], layer0_pixels[1]] = roi_maps.lam[
        panel, 1, layer0_pixels[0], layer0_pixels[1]
    ]
    roi_maps.lam[panel, 1, layer0_pixels[0], layer0_pixels[1]] = 0
    roi_maps.lam[panel, 1, layer1_pixels[0], layer1_pixels[1]] = roi_maps.lam[
        panel, 2, layer1_pixels[0], layer1_pixels[1]
    ]
    roi_maps.lam[panel, 2, layer1_pixels[0], layer1_pixels[1]] = 0
    roi_maps.lam[panel, 2, layer2_pixels[0], layer2_pixels[1]] = 0

    roi_maps.iroi[panel, 0, layer0_pixels[0], layer0_pixels[1]] = roi_maps.iroi[
        panel, 1, layer0_pixels[0], layer0_pixels[1]
    ]
    roi_maps.iroi[panel, 1, layer0_pixels[0], layer0_pixels[1]] = -1
    roi_maps.iroi[panel, 1, layer1_pixels[0], layer1_pixels[1]] = roi_maps.iroi[
        panel, 2, layer1_pixels[0], layer1_pixels[1]
    ]
    roi_maps.iroi[panel, 2, layer1_pixels[0], layer1_pixels[1]] = -1
    roi_maps.iroi[panel, 2, layer2_pixels[0], layer2_pixels[1]] = -1

    roi_maps.sroi[panel, y_pixels, x_pixels] = roi_maps.iroi[panel, 0, y_pixels, x_pixels] > 0
    roi_maps.lam_norm[panel, y_pixels, x_pixels] = np.maximum(
        0,
        np.minimum(1, _LAMBDA_NORM_SCALE * roi_maps.lam[panel, 0, y_pixels, x_pixels] / roi_maps.lam_mean),
    )


def redraw_masks(
    color_arrays: ColorArrays,
    roi_maps: ROIIndexMaps,
    y_pixels: NDArray,
    x_pixels: NDArray,
) -> None:
    """Redraws RGB mask colors at specific pixel locations after ROI changes.

    Args:
        color_arrays: The computed color arrays.
        roi_maps: The ROI index maps.
        y_pixels: Row coordinates of pixels to redraw.
        x_pixels: Column coordinates of pixels to redraw.
    """
    for color_index in range(color_arrays.cols.shape[0]):
        for panel in range(2):
            color = color_arrays.cols[color_index]
            rgb = color[roi_maps.iroi[panel, 0, y_pixels, x_pixels], :]
            color_arrays.rgb[panel, color_index, y_pixels, x_pixels, :3] = rgb


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
    context: ContextData,
    state: ViewState,
    color_arrays: ColorArrays,
    roi_maps: ROIIndexMaps,
) -> None:
    """Recomputes the channel 2 random coloring after threshold change.

    Args:
        context: The loaded data context.
        state: The current GUI display state.
        color_arrays: The computed color arrays (modified in place).
        roi_maps: The ROI index maps.
    """
    is_channel_2 = context.cell_colocalization_probabilities > state.colocalization_threshold
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


class _ColorButton(QPushButton):
    """Color statistic selection button for the ROI overlay panel.

    Each button maps to a different color statistic mode (random, skew, compact,
    footprint, etc.). Only one button can be active at a time within the group.
    Pressing emits the color_mode_changed signal.

    Args:
        button_id: Zero-based index identifying this button's color mode.
        text: Display label (with keyboard shortcut prefix).
        owner: The parent widget for ownership.
        button_group: The button group this button belongs to.
        signals: The central signal bus for GUI events.
    """

    def __init__(
        self,
        button_id: int,
        text: str,
        owner: QWidget,
        button_group: QButtonGroup,  # noqa: ARG002
        signals: GUISignals,
    ) -> None:
        super().__init__(owner)
        self.setText(text)
        self.setCheckable(True)
        self.setStyleSheet(BUTTON_INACTIVE_STYLESHEET)
        self.setFont(label_font_bold())
        self.resize(self.minimumSizeHint())
        self._button_id: int = button_id
        self._signals = signals
        self.clicked.connect(self._press)
        self.show()

    def _press(self) -> None:
        """Emits the color_mode_changed signal with this button's index."""
        self._signals.color_mode_changed.emit(self._button_id)


def _flip_roi(
    roi_maps: ROIIndexMaps,
    color_arrays: ColorArrays,
    roi_statistics: list[ROIStatistics],
    cell_classification_labels: NDArray[np.bool_],
    roi_index: int,
) -> None:
    """Flips a single ROI between cell and non-cell panels.

    Moves the ROI from its old panel to its new panel, including updating overlap
    layers and text labels. The caller handles any UI-level text label moves.

    Args:
        roi_maps: The ROI index maps.
        color_arrays: The computed color arrays.
        roi_statistics: The ROI statistics list.
        cell_classification_labels: The current cell classification label array.
        roi_index: Index of the ROI to flip.
    """
    new_panel = int(1 - cell_classification_labels[roi_index])
    old_panel = 1 - new_panel

    remove_roi(roi_maps=roi_maps, roi_statistics=roi_statistics, roi_index=roi_index, panel=old_panel)
    add_roi(roi_maps=roi_maps, roi_statistics=roi_statistics, roi_index=roi_index, panel=new_panel)

    y_pixels = roi_statistics[roi_index].y_pixels
    x_pixels = roi_statistics[roi_index].x_pixels
    redraw_masks(color_arrays=color_arrays, roi_maps=roi_maps, y_pixels=y_pixels, x_pixels=x_pixels)


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
    return hsv2rgb(inverted.flatten())


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
