"""Provides background view construction, ROI overlay rendering, and mask mutation shared by the ROI viewer and
editor.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg  # type: ignore[import-untyped]
import matplotlib.cm
from matplotlib.colors import hsv_to_rgb
from ataraxis_base_utilities import LogLevel, console

from .styles import FONTS, COLORS, ROI_STYLE
from .constants import COMMON_CONFIG, ROI_CONFIG, ROIColorMode, BackgroundView
from .data_models import ColorArrays, ROIIndexMaps

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .data_models import ColorbarWidgets
    from ..dataclasses import ROIStatistics

_STATISTIC_FIELD_MAP: dict[int, str] = {
    ROIColorMode.SKEWNESS: "skewness",
    ROIColorMode.COMPACTNESS: "compactness",
    ROIColorMode.FOOTPRINT: "footprint",
    ROIColorMode.ASPECT_RATIO: "aspect_ratio",
    ROIColorMode.COLOCALIZATION_PROBABILITY: "colocalization_probability",
}
"""Maps ROIColorMode values to the corresponding ROIStatistics attribute names for percentile-based color modes."""


def build_views(
    frame_height: int,
    frame_width: int,
    *,
    mean_image: NDArray[np.float32],
    enhanced_mean_image: NDArray[np.float32],
    correlation_map: NDArray[np.float32],
    maximum_projection: NDArray[np.float32],
    corrected_structural_mean_image: NDArray[np.float32],
    channel_2: bool = False,
    channel_2_mean_image: NDArray[np.float32],
    channel_2_enhanced_mean_image: NDArray[np.float32],
    channel_2_correlation_map: NDArray[np.float32],
    channel_2_maximum_projection: NDArray[np.float32],
    valid_y_range: tuple[int, int] | None = None,
    valid_x_range: tuple[int, int] | None = None,
) -> NDArray[np.uint8]:
    """Builds the background view stack from detection images.

    Creates a stack of 6 RGB background images, each normalized to [0, 255] uint8 range.
    Views are indexed as: 0=ROIs (black), 1=mean, 2=enhanced mean, 3=correlation map,
    4=max projection, 5=corrected structural. When ``channel_2`` is True, slots 1-4
    use channel 2 images (falling back to black where unavailable).

    Args:
        frame_height: Height of the field of view in pixels.
        frame_width: Width of the field of view in pixels.
        mean_image: Channel 1 mean fluorescence image.
        enhanced_mean_image: Channel 1 contrast-enhanced mean image.
        correlation_map: Channel 1 pixel correlation map.
        maximum_projection: Channel 1 maximum intensity projection.
        corrected_structural_mean_image: Corrected structural channel mean image. Empty if unavailable.
        channel_2: Determines whether to use channel 2 images for slots 1-4.
        channel_2_mean_image: Channel 2 mean fluorescence image. Empty if single-channel.
        channel_2_enhanced_mean_image: Channel 2 contrast-enhanced mean image. Empty if single-channel.
        channel_2_correlation_map: Channel 2 pixel correlation map. Empty if single-channel.
        channel_2_maximum_projection: Channel 2 maximum intensity projection. Empty if single-channel.
        valid_y_range: Tuple of (start, end) row indices for the valid image region.
        valid_x_range: Tuple of (start, end) column indices for the valid image region.

    Returns:
        Array of shape (6, frame_height, frame_width, 3) containing uint8 RGB views.
    """
    views = np.zeros((len(BackgroundView), frame_height, frame_width, 3), dtype=np.float32)

    for view_index in range(len(BackgroundView)):
        image = _build_single_view(
            view_index=view_index,
            frame_height=frame_height,
            frame_width=frame_width,
            mean_image=mean_image,
            enhanced_mean_image=enhanced_mean_image,
            correlation_map=correlation_map,
            maximum_projection=maximum_projection,
            corrected_structural_mean_image=corrected_structural_mean_image,
            channel_2=channel_2,
            channel_2_mean_image=channel_2_mean_image,
            channel_2_enhanced_mean_image=channel_2_enhanced_mean_image,
            channel_2_correlation_map=channel_2_correlation_map,
            channel_2_maximum_projection=channel_2_maximum_projection,
            valid_y_range=valid_y_range,
            valid_x_range=valid_x_range,
        )
        image_uint8 = (image * 255).astype(np.uint8)
        views[view_index] = np.tile(image_uint8[:, :, np.newaxis], (1, 1, 3))

    return views.astype(np.uint8)


def display_views(
    view: pg.ImageItem,
    views: NDArray[np.uint8],
    view_index: int,
) -> None:
    """Displays the selected background view on the image panel.

    Args:
        view: The background image item.
        views: The full view stack of shape (6, height, width, 3).
        view_index: Index of the view to display (0-5).
    """
    view.setImage(views[view_index], levels=[0, 255])
    view.show()


def compute_colors(
    roi_statistics: list[ROIStatistics],
    frame_height: int,
    frame_width: int,
    cell_classification: NDArray[np.float32],
    cell_colocalization: NDArray[np.float32],
    roi_colormap: str,
    colocalization_threshold: float,
    *,
    two_channels: bool = False,
) -> ColorArrays:
    """Computes color statistics and RGB color arrays for all ROIs.

    Initializes per-statistic color arrays and normalization values. The first color channel
    uses random HSV coloring; subsequent channels use computed statistics (skew, compact,
    footprint, aspect ratio, etc.).

    Args:
        roi_statistics: The ROI statistics for the current view.
        frame_height: Height of the field of view in pixels.
        frame_width: Width of the field of view in pixels.
        cell_classification: Cell classification array with shape (cell_count, 2).
        cell_colocalization: Cell colocalization array with shape (cell_count, 2).
        roi_colormap: Name of the matplotlib colormap applied when mapping ROI statistics to overlay colors.
        colocalization_threshold: Display threshold applied to cell colocalization probabilities.
        two_channels: Determines whether channel 2 data is available.

    Returns:
        Computed color arrays for all ROIs.
    """
    cell_count = len(roi_statistics)
    color_count = len(ROIColorMode)
    colorbar: list[list[float]] = []

    cols = np.zeros((color_count, cell_count, 3), dtype=np.uint8)
    istat = np.zeros((color_count, cell_count), dtype=np.float32)

    # Generates random colors, adjusting for channel 2 data if present.
    np.random.seed(seed=ROI_CONFIG.random_color_seed)  # noqa: NPY002
    random_colors = np.random.random((cell_count,))  # noqa: NPY002
    if two_channels:
        random_colors = random_colors / ROI_CONFIG.channel_2_color_divisor + ROI_CONFIG.channel_2_color_offset
        is_channel_2 = cell_colocalization[:, 0] > colocalization_threshold
        console.echo(message=f"Number of channel 2 cells: {int(is_channel_2.sum())}")
        random_hues = random_colors.copy()
        random_colors[is_channel_2] = 0
    else:
        random_hues = random_colors.copy()

    istat[0] = random_hues
    cols[0] = hsv2rgb(random_colors)

    # Computes color arrays for percentile-based statistics (skewness, compactness, footprint, aspect_ratio, chan2_prob).
    for color_mode in ROIColorMode:
        if color_mode >= ROIColorMode.CELL_CLASSIFICATION:
            break
        if color_mode == ROIColorMode.RANDOM:
            colorbar.append(list(ROI_CONFIG.fixed_colorbar_range))
            continue

        stat_values = np.zeros((cell_count, 1))
        field_name = _STATISTIC_FIELD_MAP.get(color_mode)
        if field_name is not None:
            for roi_index in range(cell_count):
                value = getattr(roi_statistics[roi_index], field_name, None)
                if value is not None:
                    stat_values[roi_index] = value

        stat_low = np.percentile(stat_values, COMMON_CONFIG.lower_percentile)
        stat_high = np.percentile(stat_values, COMMON_CONFIG.upper_percentile)
        colorbar.append([float(stat_low), float((stat_high - stat_low) / 2 + stat_low), float(stat_high)])
        stat_range = stat_high - stat_low
        if stat_range > 0:
            stat_values = (stat_values - stat_low) / stat_range
            stat_values = np.maximum(0, np.minimum(1, stat_values))
        else:
            stat_values = np.zeros_like(stat_values)

        color = istat_transform(stat_values.astype(np.float32), roi_colormap)
        cols[color_mode] = color
        istat[color_mode] = stat_values.flatten()

    # Computes classifier probability colors into the CELL_CLASSIFICATION slot.
    classifier_values = np.expand_dims(cell_classification[:, 0], axis=1)
    classifier_color = istat_transform(classifier_values.astype(np.float32), roi_colormap)
    cols[ROIColorMode.CELL_CLASSIFICATION] = classifier_color
    istat[ROIColorMode.CELL_CLASSIFICATION] = classifier_values.flatten()
    colorbar.append(list(ROI_CONFIG.fixed_colorbar_range))

    # Appends a fixed range for the correlation color channel.
    colorbar.append(list(ROI_CONFIG.fixed_colorbar_range))

    # Computes binary cell / non-cell label colors into the secondary classification_label_cols array.
    classification_label_cols = np.zeros((cell_count, 3), dtype=np.uint8)
    for roi_index in range(cell_count):
        if cell_classification[:, 1][roi_index]:
            classification_label_cols[roi_index] = COLORS.green
        else:
            classification_label_cols[roi_index] = COLORS.magenta

    # Creates a placeholder RGB array (populated by init_roi_maps via rgb_masks).
    rgb = np.zeros((color_count, frame_height, frame_width, 4), dtype=np.uint8)

    return ColorArrays(
        cols=cols,
        istat=istat,
        colorbar=colorbar,
        rgb=rgb,
        random_hues=random_hues,
        classification_label_cols=classification_label_cols,
    )


def init_roi_maps(
    roi_statistics: list[ROIStatistics],
    frame_height: int,
    frame_width: int,
    color_arrays: ColorArrays,
) -> ROIIndexMaps:
    """Initializes ROI index maps and RGB overlay arrays.

    Creates the multi-layer ROI index map that tracks pixel overlap ordering and generates per-ROI text labels at
    centroids. Also populates the RGB overlay in color_arrays.

    Args:
        roi_statistics: The ROI statistics for the current view.
        frame_height: Height of the field of view in pixels.
        frame_width: Width of the field of view in pixels.
        color_arrays: The computed color arrays (rgb field is populated in place).

    Returns:
        Initialized ROI index maps.
    """
    cell_count = len(roi_statistics)

    sroi = np.zeros((frame_height, frame_width), dtype=bool)
    iroi = -1 * np.ones((ROI_CONFIG.overlap_layers, frame_height, frame_width), dtype=np.int32)

    text_labels: list[pg.TextItem] = []

    for roi_index in np.arange(cell_count - 1, -1, -1, dtype=np.int32):
        y_pixels = roi_statistics[roi_index].mask.y_pixels
        if y_pixels is not None:
            x_pixels = roi_statistics[roi_index].mask.x_pixels

            # Pushes down existing layers and adds cell on top.
            iroi[2, y_pixels, x_pixels] = iroi[1, y_pixels, x_pixels]
            iroi[1, y_pixels, x_pixels] = iroi[0, y_pixels, x_pixels]
            iroi[0, y_pixels, x_pixels] = roi_index
            sroi[y_pixels, x_pixels] = True

            centroid = roi_statistics[roi_index].mask.centroid
            label_text = str(roi_index)
        else:
            label_text = ""
            centroid = (0, 0)

        text_item = pg.TextItem(label_text, color=COLORS.silver, anchor=(0.5, 0.5))
        text_item.setPos(centroid[1], centroid[0])
        text_item.setFont(FONTS.small_bold)
        text_labels.append(text_item)

    text_labels.reverse()

    roi_maps = ROIIndexMaps(
        sroi=sroi,
        iroi=iroi,
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
    roi_statistics: list[ROIStatistics],
    frame_height: int,
    frame_width: int,
    color_arrays: ColorArrays,
    roi_maps: ROIIndexMaps,
    *,
    roi_color_mode: int,
    background_view: int,
    roi_opacity: int,
    merge_roi_indices: list[int],
    classification_label_mode: bool = False,
) -> NDArray[np.uint8]:
    """Draws the current mask overlay for the image panel.

    Computes transparency based on ROI weights, then highlights the currently selected ROIs
    with full-white (ROI view) or colored circles (image views).

    Args:
        roi_statistics: The ROI statistics for the current view.
        frame_height: Height of the field of view in pixels.
        frame_width: Width of the field of view in pixels.
        color_arrays: The computed color arrays.
        roi_maps: The ROI index maps.
        roi_color_mode: Active color statistic index.
        background_view: Active background view index.
        roi_opacity: Alpha value (0-255) for mask overlay opacity.
        merge_roi_indices: Indices of all ROIs staged for merge or multi-selection.
        classification_label_mode: Determines whether to use binary cell/non-cell label colors instead of probability
            gradient colors when the cell classification color mode is active.

    Returns:
        RGBA overlay array.
    """
    color_index = roi_color_mode
    view_index = background_view

    # When the cell classification mode is active, swaps between label colors and probability colors.
    if roi_color_mode == ROIColorMode.CELL_CLASSIFICATION:
        color = color_arrays.classification_label_cols if classification_label_mode else color_arrays.cols[color_index]
        rgb_masks(
            color_arrays=color_arrays,
            roi_maps=roi_maps,
            color=color,
            color_index=color_index,
        )

    # The ROI-only view (view_index == 0, black background) always uses full opacity since partial transparency on
    # black just dims the ROIs. All other views use the slider value.
    effective_opacity = 255 if view_index == 0 else roi_opacity

    # Applies flat opacity to all ROI pixels.
    color_arrays.rgb[color_index, :, :, 3] = (effective_opacity * roi_maps.sroi).astype(np.uint8)

    overlay = np.array(color_arrays.rgb[color_index])

    if view_index == 0:
        # ROI view: highlights selected ROIs with brightness based on overlap depth.
        for roi_index in merge_roi_indices:
            y_pixels = roi_statistics[roi_index].mask.y_pixels.flatten()
            x_pixels = roi_statistics[roi_index].mask.x_pixels.flatten()
            overlap_count = (roi_maps.iroi[:, y_pixels, x_pixels] > -1).sum(axis=0) - 1
            brightness = 1 - overlap_count / ROI_CONFIG.overlap_layers
            overlay = _make_chosen_roi(overlay, y_pixels, x_pixels, brightness)
    else:
        # Image view: highlights selected ROIs with colored circles.
        for roi_index in merge_roi_indices:
            y_circle, x_circle = roi_statistics[roi_index].mask.circle_pixels
            valid = (y_circle >= 0) & (x_circle >= 0) & (y_circle < frame_height) & (x_circle < frame_width)
            y_circle, x_circle = y_circle[valid], x_circle[valid]
            y_pixels = roi_statistics[roi_index].mask.y_pixels.flatten()
            x_pixels = roi_statistics[roi_index].mask.x_pixels.flatten()
            overlay[y_pixels, x_pixels, 3] = 0
            roi_color = color_arrays.cols[color_index, roi_index]
            overlay = _make_chosen_circle(
                overlay,
                y_circle,
                x_circle,
                roi_color,
            )

    return overlay


def display_masks(
    overlay_item: pg.ImageItem,
    mask: NDArray[np.uint8],
) -> None:
    """Displays the mask overlay on the image panel.

    Args:
        overlay_item: The overlay image item.
        mask: RGBA overlay array from ``draw_masks``.
    """
    overlay_item.setImage(mask, levels=(0.0, 255.0))
    overlay_item.show()


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
        colorbar_widgets.image.setImage(
            np.zeros((ROI_STYLE.colorbar_row_count, ROI_STYLE.colorbar_sample_count - 1, 3))
        )
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
    gradient = np.linspace(0, 1, ROI_STYLE.colorbar_sample_count).astype(np.float32)
    rgb = istat_transform(gradient, colormap)
    colormat = np.expand_dims(rgb, axis=0)
    return np.tile(colormat, (ROI_STYLE.colorbar_row_count, 1, 1))


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
    color_arrays.rgb[color_index, :, :, :3] = color[roi_maps.iroi[0], :]


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
    color_index = ROIColorMode.CORRELATIONS
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
    colors = np.nan_to_num(colors, nan=0.0)
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


def flip_rois(
    roi_statistics: list[ROIStatistics],
    cell_classification: NDArray[np.float32],
    color_arrays: ColorArrays,
    roi_maps: ROIIndexMaps,
    *,
    selected_roi_index: int,
    merge_roi_indices: list[int],
) -> int:
    """Reclassifies selected ROIs between cell and non-cell.

    Toggles the classification labels (column 1) for all ROIs in ``merge_roi_indices`` and
    updates their overlay colors. The caller is responsible for saving and updating the plot.

    Args:
        roi_statistics: The ROI statistics for the current view.
        cell_classification: Cell classification array (column 1 is modified in place).
        color_arrays: The computed color arrays.
        roi_maps: The ROI index maps.
        selected_roi_index: Index of the currently selected ROI.
        merge_roi_indices: Indices of all ROIs to flip.

    Returns:
        The selected ROI index, stored by the caller as the last reclassified index.
    """
    labels = cell_classification[:, 1]
    for roi_index in merge_roi_indices:
        labels[roi_index] = 1.0 - labels[roi_index]
        _flip_roi(
            roi_maps=roi_maps,
            color_arrays=color_arrays,
            roi_statistics=roi_statistics,
            cell_classification_labels=labels,
            roi_index=roi_index,
        )

    return selected_roi_index


def _build_single_view(
    view_index: int,
    frame_height: int,
    frame_width: int,
    mean_image: NDArray[np.float32],
    enhanced_mean_image: NDArray[np.float32],
    correlation_map: NDArray[np.float32],
    maximum_projection: NDArray[np.float32],
    corrected_structural_mean_image: NDArray[np.float32],
    channel_2: bool,
    channel_2_mean_image: NDArray[np.float32],
    channel_2_enhanced_mean_image: NDArray[np.float32],
    channel_2_correlation_map: NDArray[np.float32],
    channel_2_maximum_projection: NDArray[np.float32],
    valid_y_range: tuple[int, int] | None,
    valid_x_range: tuple[int, int] | None,
) -> NDArray[np.float32]:
    """Builds a single background view image normalized to [0, 1].

    When ``channel_2`` is True, slots 1-4 use channel 2 images (falling back to black).

    Args:
        view_index: Index of the view to build (0-5).
        frame_height: Height of the field of view in pixels.
        frame_width: Width of the field of view in pixels.
        mean_image: Channel 1 mean fluorescence image.
        enhanced_mean_image: Channel 1 contrast-enhanced mean image.
        correlation_map: Channel 1 pixel correlation map.
        maximum_projection: Channel 1 maximum intensity projection.
        corrected_structural_mean_image: Corrected structural channel mean image. Empty if unavailable.
        channel_2: Determines whether to use channel 2 images for slots 1-4.
        channel_2_mean_image: Channel 2 mean fluorescence image. Empty if single-channel.
        channel_2_enhanced_mean_image: Channel 2 contrast-enhanced mean image. Empty if single-channel.
        channel_2_correlation_map: Channel 2 pixel correlation map. Empty if single-channel.
        channel_2_maximum_projection: Channel 2 maximum intensity projection. Empty if single-channel.
        valid_y_range: Tuple of (start, end) row indices for the valid image region.
        valid_x_range: Tuple of (start, end) column indices for the valid image region.

    Returns:
        Normalized image of shape (frame_height, frame_width) with values in [0, 1].
    """
    if view_index == BackgroundView.ROIS_ONLY:
        return np.zeros((frame_height, frame_width), dtype=np.float32)

    if view_index == BackgroundView.MEAN_IMAGE:
        image = channel_2_mean_image if channel_2 else mean_image
        return _normalize_percentile(image=image, frame_height=frame_height, frame_width=frame_width)

    if view_index == BackgroundView.ENHANCED_MEAN_IMAGE:
        image = channel_2_enhanced_mean_image if channel_2 else enhanced_mean_image
        return _normalize_percentile(image=image, frame_height=frame_height, frame_width=frame_width)

    if view_index == BackgroundView.CORRELATION_MAP:
        image = channel_2_correlation_map if channel_2 else correlation_map
        return _place_in_valid_region(
            image=image,
            frame_height=frame_height,
            frame_width=frame_width,
            valid_y_range=valid_y_range,
            valid_x_range=valid_x_range,
        )

    if view_index == BackgroundView.MAXIMUM_PROJECTION:
        image = channel_2_maximum_projection if channel_2 else maximum_projection
        return _place_in_valid_region(
            image=image,
            frame_height=frame_height,
            frame_width=frame_width,
            valid_y_range=valid_y_range,
            valid_x_range=valid_x_range,
            warn_on_error=True,
        )

    if view_index == BackgroundView.CORRECTED_STRUCTURAL:
        return _normalize_percentile(
            image=corrected_structural_mean_image,
            frame_height=frame_height,
            frame_width=frame_width,
        )

    return np.zeros((frame_height, frame_width), dtype=np.float32)


def _normalize_percentile(
    image: NDArray[np.float32],
    frame_height: int,
    frame_width: int,
) -> NDArray[np.float32]:
    """Normalizes an image to [0, 1] using 1st and 99th percentile clipping.

    Args:
        image: Input image to normalize. A size-0 array produces a zero fallback.
        frame_height: Height for the fallback zero image.
        frame_width: Width for the fallback zero image.

    Returns:
        Normalized image with values clipped to [0, 1].
    """
    if image.size == 0:
        return np.zeros((frame_height, frame_width), dtype=np.float32)

    percentile_1 = np.percentile(image, COMMON_CONFIG.lower_percentile)
    percentile_99 = np.percentile(image, COMMON_CONFIG.upper_percentile)

    if percentile_99 <= percentile_1:
        return np.zeros((frame_height, frame_width), dtype=np.float32)

    normalized = (image - percentile_1) / (percentile_99 - percentile_1)
    return np.clip(normalized, 0, 1).astype(np.float32)


def _place_in_valid_region(
    image: NDArray[np.float32],
    frame_height: int,
    frame_width: int,
    valid_y_range: tuple[int, int] | None,
    valid_x_range: tuple[int, int] | None,
    warn_on_error: bool = False,
) -> NDArray[np.float32]:
    """Normalizes and places an image into the valid subregion of the full frame.

    Args:
        image: Input image to normalize and place. A size-0 array produces a gray fallback.
        frame_height: Height of the full frame.
        frame_width: Width of the full frame.
        valid_y_range: Row range (start, end) for the valid subregion.
        valid_x_range: Column range (start, end) for the valid subregion.
        warn_on_error: Determines whether to log a warning on placement failure.

    Returns:
        Full-frame image with the normalized data placed in the valid region.
    """
    if image.size == 0:
        return 0.5 * np.ones((frame_height, frame_width), dtype=np.float32)

    # Normalizes the image using percentile clipping.
    percentile_1 = np.percentile(image, COMMON_CONFIG.lower_percentile)
    percentile_99 = np.percentile(image, COMMON_CONFIG.upper_percentile)

    if percentile_99 <= percentile_1:
        return np.zeros((frame_height, frame_width), dtype=np.float32)

    normalized = (image - percentile_1) / (percentile_99 - percentile_1)

    # Places in the valid subregion.
    output = percentile_1 * np.ones((frame_height, frame_width), dtype=np.float32)
    if valid_y_range is not None and valid_x_range is not None:
        try:
            output[valid_y_range[0] : valid_y_range[1], valid_x_range[0] : valid_x_range[1]] = normalized
        except ValueError, IndexError:
            if warn_on_error:
                console.echo(
                    message="Max projection not in combined view",
                    level=LogLevel.WARNING,
                )
    else:
        output = normalized

    return np.clip(output, 0, 1).astype(np.float32)


def _flip_roi(
    roi_maps: ROIIndexMaps,
    color_arrays: ColorArrays,
    roi_statistics: list[ROIStatistics],
    cell_classification_labels: NDArray[np.float32],
    roi_index: int,
) -> None:
    """Updates the cell/non-cell overlay color for a reclassified ROI.

    Args:
        roi_maps: The ROI index maps.
        color_arrays: The computed color arrays.
        roi_statistics: The ROI statistics list.
        cell_classification_labels: The current cell classification label array.
        roi_index: Index of the ROI to update.
    """
    if cell_classification_labels[roi_index]:
        color_arrays.classification_label_cols[roi_index] = COLORS.green
    else:
        color_arrays.classification_label_cols[roi_index] = COLORS.magenta

    y_pixels = roi_statistics[roi_index].mask.y_pixels
    x_pixels = roi_statistics[roi_index].mask.x_pixels
    redraw_masks(color_arrays=color_arrays, roi_maps=roi_maps, y_pixels=y_pixels, x_pixels=x_pixels)


def _istat_hsv(istat: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Applies the HSV color transform to a statistic array.

    Args:
        istat: Normalized statistic values in [0, 1].

    Returns:
        RGB color array with shape (..., 3) and dtype uint8.
    """
    istat /= ROI_CONFIG.hsv_divisor
    istat += ROI_CONFIG.hsv_offset / ROI_CONFIG.hsv_divisor
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
