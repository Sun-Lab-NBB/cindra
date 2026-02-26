"""Provides background view construction, ROI overlay rendering, and mask mutation shared by the ROI viewer and editor."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pyqtgraph as pg  # type: ignore[import-untyped]
import matplotlib.cm
from PySide6 import QtGui
from matplotlib.colors import hsv_to_rgb
from ataraxis_base_utilities import LogLevel, console

from .constants import STYLE, CONFIG, BackgroundView
from .data_models import ColorArrays, ROIIndexMaps

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from .data_models import ColorbarWidgets
    from .single_day_context import ROIViewerData
    from ..dataclasses import ROIStatistics


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
    views = np.zeros((CONFIG.view_count, frame_height, frame_width, 3), dtype=np.float32)

    for view_index in range(CONFIG.view_count):
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


def compute_colors(
    context: ROIViewerData,
    roi_colormap: str,
    colocalization_threshold: float,
) -> ColorArrays:
    """Computes color statistics and RGB color arrays for all ROIs.

    Initializes per-statistic color arrays and normalization values. The first color channel
    uses random HSV coloring; subsequent channels use computed statistics (skew, compact,
    footprint, aspect ratio, etc.).

    Args:
        context: The loaded data context.
        roi_colormap: Name of the matplotlib colormap applied when mapping ROI statistics to overlay colors.
        colocalization_threshold: Display threshold applied to cell_colocalization_probabilities.

    Returns:
        Computed color arrays for all ROIs.
    """
    roi_statistics = context.roi_statistics
    cell_count = len(roi_statistics)
    color_count = CONFIG.color_stat_count
    colorbar: list[list[float]] = []

    cols = np.zeros((color_count, cell_count, 3), dtype=np.uint8)
    istat = np.zeros((color_count, cell_count), dtype=np.float32)

    # Generates random colors, adjusting for channel 2 data if present.
    np.random.seed(seed=0)  # noqa: NPY002
    random_colors = np.random.random((cell_count,))  # noqa: NPY002
    if context.has_channel_2:
        random_colors = random_colors / CONFIG.channel_2_color_divisor + CONFIG.channel_2_color_offset
        is_channel_2 = context.cell_colocalization_probabilities > colocalization_threshold
        console.echo(message=f"Number of channel 2 cells: {int(is_channel_2.sum())}")
        random_hues = random_colors.copy()
        random_colors[is_channel_2] = 0
    else:
        random_hues = random_colors.copy()

    istat[0] = random_hues
    cols[0] = hsv2rgb(random_colors)

    # Computes color arrays for each statistic.
    for stat_index, name in enumerate(CONFIG.color_short_names[:-3]):
        if stat_index > 0:
            stat_values = np.zeros((cell_count, 1))
            if stat_index < color_count - 2:
                field_name = CONFIG.stat_field_map.get(name)
                if field_name is not None:
                    for roi_index in range(cell_count):
                        value = getattr(roi_statistics[roi_index], field_name, None)
                        if value is not None:
                            stat_values[roi_index] = value

                stat_low = np.percentile(stat_values, CONFIG.lower_percentile)
                stat_high = np.percentile(stat_values, CONFIG.upper_percentile)
                colorbar.append([float(stat_low), float((stat_high - stat_low) / 2 + stat_low), float(stat_high)])
                stat_values = stat_values - stat_low
                stat_values = stat_values / (stat_high - stat_low)
                stat_values = np.maximum(0, np.minimum(1, stat_values))
            else:
                stat_values = np.expand_dims(context.cell_classification_probabilities, axis=1)
                colorbar.append(list(CONFIG.fixed_colorbar_range))

            color = istat_transform(stat_values.astype(np.float32), roi_colormap)
            cols[stat_index] = color
            istat[stat_index] = stat_values.flatten()
        else:
            colorbar.append(list(CONFIG.fixed_colorbar_range))

    # Appends a fixed range for the correlation color channel.
    colorbar.append(list(CONFIG.fixed_colorbar_range))

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
    lam = np.zeros((2, CONFIG.overlap_layers, frame_height, frame_width), dtype=np.float32)
    iroi = -1 * np.ones((2, CONFIG.overlap_layers, frame_height, frame_width), dtype=np.int32)

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

        text_item = pg.TextItem(label_text, color=STYLE.roi_text_color, anchor=(0.5, 0.5))
        text_item.setPos(centroid[1], centroid[0])
        text_item.setFont(QtGui.QFont("Times", STYLE.roi_text_size, weight=QtGui.QFont.Weight.Bold))
        text_labels.append(text_item)

    text_labels.reverse()

    lam_mean = float(lambda_all[lambda_all > CONFIG.lambda_threshold].mean())
    lam_norm = np.maximum(
        0,
        np.minimum(1, CONFIG.lambda_norm_scale * lam[:, 0] / lam_mean),
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
    *,
    roi_color_mode: int,
    background_view: int,
    roi_opacity: list[int],
    selected_roi_index: int,
    merge_roi_indices: list[int],
) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Draws the current mask overlay for both cell and non-cell panels.

    Computes transparency based on ROI weights, then highlights the currently selected ROIs
    with full-white (ROI view) or colored circles (image views).

    Args:
        context: The loaded data context.
        color_arrays: The computed color arrays.
        roi_maps: The ROI index maps.
        roi_color_mode: Active color statistic index.
        background_view: Active background view index.
        roi_opacity: Alpha values for ROI overlays in [circle-view, filled-ROI-view] modes.
        selected_roi_index: Index of the currently highlighted ROI.
        merge_roi_indices: Indices of all ROIs staged for merge or multi-selection.

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
            brightness = 1 - overlap_count / CONFIG.overlap_layers
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
        colorbar_widgets.image.setImage(np.zeros((STYLE.colorbar_row_count, STYLE.colorbar_sample_count - 1, 3)))
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
    gradient = np.linspace(0, 1, STYLE.colorbar_sample_count).astype(np.float32)
    rgb = istat_transform(gradient, colormap)
    colormat = np.expand_dims(rgb, axis=0)
    return np.tile(colormat, (STYLE.colorbar_row_count, 1, 1))


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
    color_index = CONFIG.color_correlation
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


def flip_rois(
    context: ROIViewerData,
    color_arrays: ColorArrays,
    roi_maps: ROIIndexMaps,
    *,
    selected_roi_index: int,
    merge_roi_indices: list[int],
    last_reclassified_index_out: list[int],
) -> None:
    """Flips selected ROIs between the cell and non-cell panels.

    Toggles the ``cell_classification_labels`` for all ROIs in ``merge_roi_indices``, moves
    them between panels. The caller is responsible for saving and updating the plot.

    Args:
        context: The loaded data context (cell_classification_labels is modified in place).
        color_arrays: The computed color arrays.
        roi_maps: The ROI index maps.
        selected_roi_index: Index of the currently selected ROI.
        merge_roi_indices: Indices of all ROIs to flip.
        last_reclassified_index_out: Single-element list updated with the selected ROI index.
    """
    last_reclassified_index_out[0] = selected_roi_index
    for roi_index in merge_roi_indices:
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
    context: ROIViewerData,
    color_arrays: ColorArrays,
    roi_maps: ROIIndexMaps,
    new_classification_labels: NDArray[np.bool_],
) -> bool:
    """Applies new cell classification labels from the classifier.

    For small numbers of changes, flips individual ROIs incrementally. For large changes,
    returns False to signal the caller should reinitialize all masks.

    Args:
        context: The loaded data context.
        color_arrays: The computed color arrays.
        roi_maps: The ROI index maps.
        new_classification_labels: New cell classification label array.

    Returns:
        True if changes were applied incrementally, False if full reinit is needed.
    """
    cell_count = new_classification_labels.size
    if int((new_classification_labels == context.cell_classification_labels).sum()) < CONFIG.flip_threshold:
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
        np.minimum(1, CONFIG.lambda_norm_scale * roi_maps.lam[:, 0, y_pixels, x_pixels] / roi_maps.lam_mean),
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
        np.minimum(1, CONFIG.lambda_norm_scale * roi_maps.lam[panel, 0, y_pixels, x_pixels] / roi_maps.lam_mean),
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
    if view_index == BackgroundView.ROIS_ONLY:
        return np.zeros((frame_height, frame_width), dtype=np.float32)

    if view_index == BackgroundView.MEAN_IMAGE:
        return _normalize_percentile(
            image=mean_image,
            frame_height=frame_height,
            frame_width=frame_width,
        )

    if view_index == BackgroundView.ENHANCED_MEAN_IMAGE:
        return _normalize_percentile(
            image=enhanced_mean_image,
            frame_height=frame_height,
            frame_width=frame_width,
        )

    if view_index == BackgroundView.CORRELATION_MAP:
        return _place_in_valid_region(
            image=correlation_map,
            frame_height=frame_height,
            frame_width=frame_width,
            valid_y_range=valid_y_range,
            valid_x_range=valid_x_range,
        )

    if view_index == BackgroundView.MAXIMUM_PROJECTION:
        return _place_in_valid_region(
            image=maximum_projection,
            frame_height=frame_height,
            frame_width=frame_width,
            valid_y_range=valid_y_range,
            valid_x_range=valid_x_range,
            warn_on_error=True,
        )

    if view_index == BackgroundView.MEAN_IMAGE_CHANNEL_2:
        return _normalize_percentile(
            image=corrected_channel_2_image,
            frame_height=frame_height,
            frame_width=frame_width,
        )

    if view_index == BackgroundView.ENHANCED_MEAN_IMAGE_CHANNEL_2:
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


def _flip_roi(
    roi_maps: ROIIndexMaps,
    color_arrays: ColorArrays,
    roi_statistics: list[ROIStatistics],
    cell_classification_labels: NDArray[np.bool_],
    roi_index: int,
) -> None:
    """Flips a single ROI between cell and non-cell panels.

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
    istat = istat / CONFIG.hsv_divisor
    istat = istat + (CONFIG.hsv_offset / CONFIG.hsv_divisor)
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
