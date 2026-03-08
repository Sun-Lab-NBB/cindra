"""Provides background view construction, ROI overlay rendering, and mask mutation for all GUI applications."""

from __future__ import annotations

from typing import TYPE_CHECKING
import contextlib

import numpy as np
import pyqtgraph as pg  # type: ignore[import-untyped]
import matplotlib.cm
from matplotlib.colors import hsv_to_rgb
from ataraxis_base_utilities import LogLevel, console

from .styles import FONTS, COLORS, ROI_STYLE
from .constants import ROI_CONFIG, COMMON_CONFIG, ROIColorMode, BackgroundView
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
    ROIColorMode.SOLIDITY: "solidity",
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
    4=maximum projection, 5=corrected structural. When ``channel_2`` is True, slots 1-4
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
    classifier_threshold: float = 0.5,
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
        cell_classification: Cell classification array with shape (roi_count, 2).
        cell_colocalization: Cell colocalization array with shape (roi_count, 2).
        roi_colormap: Name of the matplotlib colormap applied when mapping ROI statistics to overlay colors.
        colocalization_threshold: Display threshold applied to cell colocalization probabilities.
        classifier_threshold: Probability cutoff for initial binary cell/non-cell label assignment.
        two_channels: Determines whether channel 2 data is available.

    Returns:
        Computed color arrays for all ROIs.
    """
    roi_count = len(roi_statistics)
    color_count = len(ROIColorMode)
    colorbar: list[list[float]] = []

    # Allocates output arrays: one RGB triplet per (color_mode, cell) and one scalar per (color_mode, cell).
    colors = np.zeros((color_count, roi_count, 3), dtype=np.uint8)
    normalized_statistics = np.zeros((color_count, roi_count), dtype=np.float32)

    # Generates deterministic random hues (seeded for reproducibility across recordings).
    np.random.seed(seed=ROI_CONFIG.random_color_seed)  # noqa: NPY002
    random_colors = np.random.random((roi_count,)).astype(np.float32)  # noqa: NPY002
    if two_channels:
        # Shifts hues into the channel 2 color range so the two channels are visually distinct.
        random_colors = random_colors / ROI_CONFIG.channel_2_color_divisor + ROI_CONFIG.channel_2_color_offset
        is_channel_2 = cell_colocalization[:, 0] > colocalization_threshold
        # Preserves the original hues for normalization before zeroing channel 2 ROIs.
        random_hues = random_colors.copy()
        # Zeros channel 2 ROIs so they render as black in the random color view.
        random_colors[is_channel_2] = 0
    else:
        random_hues = random_colors.copy()

    # Stores the random hues as the normalization values for the RANDOM color slot.
    normalized_statistics[0] = random_hues
    colors[0] = _convert_hues_to_rgb(random_colors)

    # Pre-extracts per-field statistic arrays into column vectors to avoid repeated getattr loops inside the main
    # color mode loop. Missing attributes default to 0.0.
    precomputed_statistics: dict[str, NDArray[np.float32]] = {}
    for field_name in _STATISTIC_FIELD_MAP.values():
        values = np.array([getattr(roi, field_name, None) or 0.0 for roi in roi_statistics], dtype=np.float32)
        precomputed_statistics[field_name] = values.reshape(-1, 1)

    # Iterates over percentile-based color modes (SKEWNESS through COLOCALIZATION_PROBABILITY), skipping RANDOM
    # and stopping before CELL_PROBABILITY which uses a different coloring strategy.
    for color_mode in ROIColorMode:
        if color_mode >= ROIColorMode.CELL_PROBABILITY:
            break
        if color_mode == ROIColorMode.RANDOM:
            colorbar.append(list(ROI_CONFIG.fixed_colorbar_range))
            continue

        # Looks up the pre-extracted array for this color mode, falling back to zeros for unmapped modes.
        mapped_field = _STATISTIC_FIELD_MAP.get(color_mode)
        statistic_values = (
            precomputed_statistics[mapped_field]
            if mapped_field is not None
            else np.zeros((roi_count, 1), dtype=np.float32)
        )

        # Computes percentile bounds for min-max normalization and stores [low, mid, high] for colorbar labels.
        statistic_low = np.percentile(statistic_values, COMMON_CONFIG.lower_percentile)
        statistic_high = np.percentile(statistic_values, COMMON_CONFIG.upper_percentile)
        colorbar.append(
            [
                float(statistic_low),
                float((statistic_high - statistic_low) / 2 + statistic_low),
                float(statistic_high),
            ]
        )

        # Normalizes values to [0, 1] using the percentile range; collapses to zeros if the range is degenerate.
        statistic_range = statistic_high - statistic_low
        if statistic_range > 0:
            statistic_values = np.clip((statistic_values - statistic_low) / statistic_range, 0, 1)
        else:
            statistic_values = np.zeros_like(statistic_values)

        # Maps the normalized [0, 1] values to RGB through the active colormap.
        colors[color_mode] = _apply_colormap(statistic_values, roi_colormap)
        normalized_statistics[color_mode] = statistic_values.ravel()

    # Uses the classifier probability (column 1) directly as a pre-normalized [0, 1] value for colormap mapping.
    classifier_values = cell_classification[:, 1:2]
    colors[ROIColorMode.CELL_PROBABILITY] = _apply_colormap(classifier_values, roi_colormap)
    normalized_statistics[ROIColorMode.CELL_PROBABILITY] = classifier_values.ravel()
    colorbar.append(list(ROI_CONFIG.fixed_colorbar_range))

    # The correlation slot colorbar is a placeholder; actual values are computed on-demand by
    # update_correlation_masks when the user selects ROIs.
    colorbar.append(list(ROI_CONFIG.fixed_colorbar_range))

    # Assigns binary cell/non-cell colors by thresholding classifier probabilities (column 1). The Classify toggle
    # starts OFF, so initial binary colors reflect the probability threshold rather than the original labels.
    # Uses the active colormap endpoints for non-cell (low) and cell (high) colors.
    non_cell_color, cell_color = _classification_endpoint_colors(roi_colormap)
    is_cell = cell_classification[:, 1] >= classifier_threshold
    binary_colors = np.full((roi_count, 3), non_cell_color, dtype=np.uint8)
    binary_colors[is_cell] = cell_color
    colors[ROIColorMode.CELL_CLASSIFICATION] = binary_colors
    normalized_statistics[ROIColorMode.CELL_CLASSIFICATION] = is_cell.astype(np.float32)
    colorbar.append(list(ROI_CONFIG.fixed_colorbar_range))

    # Creates a placeholder RGBA array; actual pixel colors are written by initialize_roi_maps via _update_rgb_masks
    # once the ROI index maps are available.
    rgb = np.zeros((color_count, frame_height, frame_width, 4), dtype=np.uint8)

    return ColorArrays(
        colors=colors,
        normalized_statistics=normalized_statistics,
        colorbar=colorbar,
        rgb=rgb,
        random_hues=random_hues,
    )


def initialize_roi_maps(
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
    roi_count = len(roi_statistics)

    # Tracks which pixels belong to any ROI (used for flat opacity application in draw_masks).
    roi_presence = np.zeros((frame_height, frame_width), dtype=bool)
    # Multi-layer index map: layer 0 holds the topmost ROI at each pixel, layer 1 the next below, etc. Pixels
    # with no ROI are -1. Used for overlap-aware rendering and selection highlighting.
    roi_indices = np.full((ROI_CONFIG.overlap_layers, frame_height, frame_width), -1, dtype=np.int32)

    # Pre-allocates the label list so each ROI's text item lands at its natural index without reversing.
    text_labels: list[pg.TextItem | None] = [None] * roi_count

    # Iterates from the last ROI to the first so that lower-indexed ROIs end up on top (layer 0). Each ROI
    # shifts existing layers down before inserting itself at the top.
    for roi_index in range(roi_count - 1, -1, -1):
        roi = roi_statistics[roi_index]
        y_pixels = roi.mask.y_pixels
        if y_pixels is not None:
            x_pixels = roi.mask.x_pixels

            # Clips out-of-bounds coordinates that can arise from backward-deformed multi-recording masks. Without
            # clipping, negative indices wrap around in numpy advanced indexing, producing mangled overlays.
            valid = (y_pixels >= 0) & (y_pixels < frame_height) & (x_pixels >= 0) & (x_pixels < frame_width)
            y_pixels = y_pixels[valid]
            x_pixels = x_pixels[valid]

            # Shifts all existing layers down by one (layer N-1 → layer N) to make room at layer 0. NumPy
            # evaluates the RHS into a temporary before assigning, so the copy order is safe.
            roi_indices[1:, y_pixels, x_pixels] = roi_indices[:-1, y_pixels, x_pixels]
            roi_indices[0, y_pixels, x_pixels] = roi_index
            roi_presence[y_pixels, x_pixels] = True

            centroid = roi.mask.centroid
            label_text = str(roi_index)
        else:
            # ROIs without pixel data (e.g. failed detection) get an invisible label at the origin.
            label_text = ""
            centroid = (0, 0)

        # Creates a centered text label at the ROI centroid for the number overlay toggle.
        text_item = pg.TextItem(label_text, color=COLORS.silver, anchor=(0.5, 0.5))
        text_item.setPos(centroid[1], centroid[0])
        text_item.setFont(FONTS.small_bold)
        text_labels[roi_index] = text_item

    roi_maps = ROIIndexMaps(
        roi_presence=roi_presence,
        roi_indices=roi_indices,
        text_labels=text_labels,
    )

    # Populates the RGBA overlay for every color mode using the topmost ROI index at each pixel.
    for color_index in range(color_arrays.colors.shape[0]):
        _update_rgb_masks(
            color_arrays=color_arrays,
            roi_maps=roi_maps,
            color=color_arrays.colors[color_index],
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
    selected_roi_indices: list[int],
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
        selected_roi_indices: Indices of all ROIs staged for merge or multi-selection.

    Returns:
        RGBA overlay array.
    """
    color_index = roi_color_mode
    view_index = background_view

    effective_opacity = roi_opacity

    # Sets alpha to zero everywhere, then writes the effective opacity only at ROI pixels. Avoids a full-frame
    # multiply + cast by using boolean indexing on the sparse ROI presence mask.
    alpha_channel = color_arrays.rgb[color_index, :, :, 3]
    alpha_channel[:] = 0
    alpha_channel[roi_maps.roi_presence] = effective_opacity

    overlay = color_arrays.rgb[color_index].copy()

    if view_index == 0:
        # ROI view: highlights selected ROIs with brightness based on overlap depth.
        for roi_index in selected_roi_indices:
            roi = roi_statistics[roi_index]
            y_pixels = roi.mask.y_pixels.ravel()
            x_pixels = roi.mask.x_pixels.ravel()
            valid = (y_pixels >= 0) & (y_pixels < frame_height) & (x_pixels >= 0) & (x_pixels < frame_width)
            y_pixels = y_pixels[valid]
            x_pixels = x_pixels[valid]
            overlap_count = (roi_maps.roi_indices[:, y_pixels, x_pixels] > -1).sum(axis=0) - 1
            brightness = (1 - overlap_count / ROI_CONFIG.overlap_layers).astype(np.float32)
            overlay = _highlight_selected_roi(overlay, y_pixels, x_pixels, brightness)
    else:
        # Image view: highlights selected ROIs with colored circles.
        for roi_index in selected_roi_indices:
            roi = roi_statistics[roi_index]
            y_circle, x_circle = roi.mask.circle_pixels
            valid = (y_circle >= 0) & (x_circle >= 0) & (y_circle < frame_height) & (x_circle < frame_width)
            y_circle, x_circle = y_circle[valid], x_circle[valid]
            y_pixels = roi.mask.y_pixels.ravel()
            x_pixels = roi.mask.x_pixels.ravel()
            valid = (y_pixels >= 0) & (y_pixels < frame_height) & (x_pixels >= 0) & (x_pixels < frame_width)
            y_pixels = y_pixels[valid]
            x_pixels = x_pixels[valid]
            overlay[y_pixels, x_pixels, 3] = 0
            roi_color = color_arrays.colors[color_index, roi_index]
            overlay = _highlight_selected_circle(
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
    if roi_color_mode == ROIColorMode.CELL_CLASSIFICATION:
        # Renders a two-color bar using the active colormap endpoints: low (non-cell) on the left, high (cell) on
        # the right.
        sample_count = ROI_STYLE.colorbar_sample_count - 1
        row_count = ROI_STYLE.colorbar_row_count
        midpoint = sample_count // 2
        binary_bar = np.zeros((row_count, sample_count, 3), dtype=np.uint8)
        binary_bar[:, :midpoint] = colorbar_image[0, 0]
        binary_bar[:, midpoint:] = colorbar_image[0, -1]
        colorbar_widgets.image.setImage(binary_bar)
        colorbar_widgets.labels[0].setText("Non-Cell")
        colorbar_widgets.labels[1].setText("")
        colorbar_widgets.labels[2].setText("Cell")
        return

    color_index = roi_color_mode
    if color_index == 0:
        colorbar_widgets.image.setImage(
            np.zeros((ROI_STYLE.colorbar_row_count, ROI_STYLE.colorbar_sample_count - 1, 3), dtype=np.uint8)
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
    rgb = _apply_colormap(gradient, colormap)
    color_matrix = np.expand_dims(rgb, axis=0)
    return np.tile(color_matrix, (ROI_STYLE.colorbar_row_count, 1, 1))


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
    for color_index in range(1, color_arrays.normalized_statistics.shape[0]):
        if color_index == ROIColorMode.CELL_CLASSIFICATION:
            # Recolors the binary classification slot using the new colormap endpoints.
            non_cell_color, cell_color = _classification_endpoint_colors(colormap)
            is_cell = color_arrays.normalized_statistics[color_index] > 0
            binary_colors = color_arrays.colors[color_index]
            binary_colors[:] = non_cell_color
            binary_colors[is_cell] = cell_color
            _update_rgb_masks(
                color_arrays=color_arrays,
                roi_maps=roi_maps,
                color=binary_colors,
                color_index=color_index,
            )
            continue
        color_arrays.colors[color_index] = _apply_colormap(color_arrays.normalized_statistics[color_index], colormap)
        _update_rgb_masks(
            color_arrays=color_arrays,
            roi_maps=roi_maps,
            color=color_arrays.colors[color_index],
            color_index=color_index,
        )
    return draw_colorbar(colormap)


def update_correlation_masks(
    color_arrays: ColorArrays,
    roi_maps: ROIIndexMaps,
    binned_fluorescence: NDArray[np.float32],
    fluorescence_standard_deviation: NDArray[np.float32],
    selected_indices: list[int],
    colormap: str,
) -> None:
    """Computes inter-ROI correlation coloring.

    Correlates each ROI's binned fluorescence with the average of the selected ROIs.

    Args:
        color_arrays: The computed color arrays (modified in place).
        roi_maps: The ROI index maps.
        binned_fluorescence: Binned fluorescence with shape (roi_count, bin_count).
        fluorescence_standard_deviation: Per-ROI standard deviation with shape (roi_count,).
        selected_indices: Currently selected ROI indices.
        colormap: Name of the active colormap.
    """
    color_index = ROIColorMode.CORRELATIONS

    # Skips computation when no ROIs are selected; there is no reference trace to correlate against.
    if not selected_indices:
        return

    # Averages the binned fluorescence traces of all selected ROIs into a single reference template.
    selected_array = np.array(selected_indices, dtype=np.int32)
    selected_mean = binned_fluorescence[selected_array].mean(axis=-2).squeeze()

    # Computes the RMS of the reference template for Pearson-style normalization.
    selected_standard_deviation = float((selected_mean**2).mean() ** 0.5)

    # Builds the per-ROI normalization denominator: bin_count * per_roi_std * reference_std. This converts the
    # dot product into an approximate Pearson correlation coefficient for each ROI.
    denominator = binned_fluorescence.shape[-1] * fluorescence_standard_deviation * selected_standard_deviation

    # Dot product of each ROI's trace against the reference template, normalized by the denominator.
    correlation = np.dot(binned_fluorescence, selected_mean) / denominator

    # Replaces the selected ROIs' self-correlation with the population mean to prevent them from dominating the
    # color scale (they would otherwise always be the highest values).
    correlation[selected_indices] = correlation.mean()

    # Computes min/max once and reuses for both the colorbar [low, mid, high] labels and the [0, 1] normalization.
    correlation_min = float(correlation.min())
    correlation_max = float(correlation.max())
    color_arrays.colorbar[color_index] = [
        correlation_min,
        (correlation_max - correlation_min) / 2 + correlation_min,
        correlation_max,
    ]

    # Normalizes to [0, 1] for colormap mapping; falls back to zeros if all correlations are identical.
    correlation_range = correlation_max - correlation_min
    normalized = (
        (correlation - correlation_min) / correlation_range if correlation_range > 0 else np.zeros_like(correlation)
    )

    # Maps normalized correlations to RGB and writes into the CORRELATIONS color slot and overlay.
    color = _apply_colormap(normalized, colormap)
    color_arrays.colors[color_index] = color
    color_arrays.normalized_statistics[color_index] = normalized.ravel()
    _update_rgb_masks(color_arrays=color_arrays, roi_maps=roi_maps, color=color, color_index=color_index)


def flip_rois(
    roi_statistics: list[ROIStatistics],
    cell_classification: NDArray[np.float32],
    color_arrays: ColorArrays,
    roi_maps: ROIIndexMaps,
    selected_roi_indices: list[int],
    colormap: str,
) -> None:
    """Reclassifies selected ROIs between cell and non-cell.

    Toggles the classification labels (column 0) for all ROIs in ``selected_roi_indices`` and
    updates their overlay colors using the active colormap endpoints. The caller is responsible
    for saving and updating the plot.

    Args:
        roi_statistics: The ROI statistics for the current view.
        cell_classification: Cell classification array (column 0 labels are modified in place).
        color_arrays: The computed color arrays.
        roi_maps: The ROI index maps.
        selected_roi_indices: Indices of all ROIs to flip.
        colormap: Name of the active colormap for endpoint color derivation.
    """
    non_cell_color, cell_color = _classification_endpoint_colors(colormap)
    labels = cell_classification[:, 0]
    for roi_index in selected_roi_indices:
        labels[roi_index] = 1.0 - labels[roi_index]
        _flip_roi(
            roi_maps=roi_maps,
            color_arrays=color_arrays,
            roi_statistics=roi_statistics,
            cell_classification_labels=labels,
            roi_index=roi_index,
            cell_color=cell_color,
            non_cell_color=non_cell_color,
        )


def recompute_binary_classification(
    cell_classification: NDArray[np.float32],
    color_arrays: ColorArrays,
    roi_maps: ROIIndexMaps,
    colormap: str,
    threshold: float | None = None,
) -> None:
    """Recomputes the binary cell/non-cell label colors and updates the CELL_CLASSIFICATION overlay.

    When a threshold is provided, classifies ROIs by thresholding the probability column (column 1). When threshold
    is None, uses the original binary labels from column 0 directly. Colors are derived from the active colormap
    endpoints.

    Args:
        cell_classification: Cell classification array with shape (roi_count, 2).
        color_arrays: The computed color arrays (modified in place).
        roi_maps: The ROI index maps.
        colormap: Name of the active colormap for endpoint color derivation.
        threshold: Probability cutoff for cell/non-cell assignment. Uses original labels when None.
    """
    non_cell_color, cell_color = _classification_endpoint_colors(colormap)
    is_cell = cell_classification[:, 1] >= threshold if threshold is not None else cell_classification[:, 0] > 0
    binary_colors = color_arrays.colors[ROIColorMode.CELL_CLASSIFICATION]
    binary_colors[:] = non_cell_color
    binary_colors[is_cell] = cell_color
    color_arrays.normalized_statistics[ROIColorMode.CELL_CLASSIFICATION] = is_cell.astype(np.float32)
    _update_rgb_masks(
        color_arrays=color_arrays,
        roi_maps=roi_maps,
        color=binary_colors,
        color_index=ROIColorMode.CELL_CLASSIFICATION,
    )


def normalize_percentile(
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

    lower_bound = np.percentile(image, COMMON_CONFIG.lower_percentile)
    upper_bound = np.percentile(image, COMMON_CONFIG.upper_percentile)

    if upper_bound <= lower_bound:
        return np.zeros((frame_height, frame_width), dtype=np.float32)

    normalized = (image - lower_bound) / (upper_bound - lower_bound)
    return np.clip(normalized, 0, 1).astype(np.float32)


def _update_rgb_masks(
    color_arrays: ColorArrays,
    roi_maps: ROIIndexMaps,
    color: NDArray[np.uint8],
    color_index: int,
) -> None:
    """Updates the RGB overlay array for a specific color channel.

    Args:
        color_arrays: The computed color arrays.
        roi_maps: The ROI index maps.
        color: Per-ROI RGB colors with shape (roi_count, 3).
        color_index: Index of the color channel to update.
    """
    color_arrays.rgb[color_index, :, :, :3] = color[roi_maps.roi_indices[0], :]


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
        return normalize_percentile(image=image, frame_height=frame_height, frame_width=frame_width)

    if view_index == BackgroundView.ENHANCED_MEAN_IMAGE:
        image = channel_2_enhanced_mean_image if channel_2 else enhanced_mean_image
        return normalize_percentile(image=image, frame_height=frame_height, frame_width=frame_width)

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
        )

    if view_index == BackgroundView.CORRECTED_STRUCTURAL:
        return normalize_percentile(
            image=corrected_structural_mean_image,
            frame_height=frame_height,
            frame_width=frame_width,
        )

    return np.zeros((frame_height, frame_width), dtype=np.float32)


def _place_in_valid_region(
    image: NDArray[np.float32],
    frame_height: int,
    frame_width: int,
    valid_y_range: tuple[int, int] | None,
    valid_x_range: tuple[int, int] | None,
) -> NDArray[np.float32]:
    """Normalizes and places an image into the valid subregion of the full frame.

    Args:
        image: Input image to normalize and place. A size-0 array produces a gray fallback when a valid region is
            specified, or a zero fallback otherwise.
        frame_height: Height of the full frame.
        frame_width: Width of the full frame.
        valid_y_range: Row range (start, end) for the valid subregion.
        valid_x_range: Column range (start, end) for the valid subregion.

    Returns:
        Full-frame image with the normalized data placed in the valid region.
    """
    # Delegates to normalize_percentile when no valid subregion is specified.
    if valid_y_range is None or valid_x_range is None:
        return normalize_percentile(image=image, frame_height=frame_height, frame_width=frame_width)

    if image.size == 0:
        return np.full((frame_height, frame_width), 0.5, dtype=np.float32)

    # Normalizes the image using percentile clipping.
    lower_bound = np.percentile(image, COMMON_CONFIG.lower_percentile)
    upper_bound = np.percentile(image, COMMON_CONFIG.upper_percentile)

    if upper_bound <= lower_bound:
        return np.zeros((frame_height, frame_width), dtype=np.float32)

    normalized = (image - lower_bound) / (upper_bound - lower_bound)

    # Places in the valid subregion, filling the border with the lower percentile value.
    output = np.full((frame_height, frame_width), lower_bound, dtype=np.float32)
    with contextlib.suppress(ValueError, IndexError):
        output[valid_y_range[0] : valid_y_range[1], valid_x_range[0] : valid_x_range[1]] = normalized
    np.clip(output, 0, 1, out=output)
    return output


def _convert_hues_to_rgb(hues: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Converts HSV hue values to RGB uint8 colors with full saturation and value.

    Args:
        hues: Array of hue values in [0, 1].

    Returns:
        RGB color array with shape (..., 3) and dtype uint8.
    """
    hsv = np.empty((hues.size, 3), dtype=hues.dtype)
    hsv[:, 0] = np.nan_to_num(hues.ravel(), nan=0.0)
    hsv[:, 1] = 1.0
    hsv[:, 2] = 1.0
    return (255 * hsv_to_rgb(hsv)).astype(np.uint8)


def _apply_colormap(values: NDArray[np.float32], colormap: str = "hsv") -> NDArray[np.uint8]:
    """Transforms a normalized statistic array into RGB colors using the given colormap.

    Args:
        values: Statistic values normalized to [0, 1].
        colormap: Name of the matplotlib colormap to use.

    Returns:
        RGB color array with shape (..., 3) and dtype uint8.
    """
    if colormap == "hsv":
        return _apply_hsv_colormap(values)

    try:
        color_map = matplotlib.cm.get_cmap(colormap)
        mapped = color_map(values)[:, :3]
        mapped *= 255
        return mapped.astype(np.uint8)
    except ValueError:
        console.echo(message="Unable to apply the requested colormap. Falling back to hsv.", level=LogLevel.WARNING)
        return _apply_hsv_colormap(values)


def _apply_hsv_colormap(values: NDArray[np.float32]) -> NDArray[np.uint8]:
    """Applies the HSV color transform to a statistic array.

    Args:
        values: Normalized statistic values in [0, 1].

    Returns:
        RGB color array with shape (..., 3) and dtype uint8.
    """
    inverted = 1.0 - (values + ROI_CONFIG.hsv_offset) / ROI_CONFIG.hsv_divisor
    return _convert_hues_to_rgb(inverted.ravel().astype(np.float32))


def _classification_endpoint_colors(colormap: str) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]:
    """Computes the non-cell and cell endpoint colors from the given colormap.

    Args:
        colormap: Name of the matplotlib colormap.

    Returns:
        A tuple of (non_cell_color, cell_color), each a uint8 array of shape (3,).
    """
    endpoints = _apply_colormap(np.array([0.0, 1.0], dtype=np.float32), colormap)
    return endpoints[0], endpoints[1]


def _flip_roi(
    roi_maps: ROIIndexMaps,
    color_arrays: ColorArrays,
    roi_statistics: list[ROIStatistics],
    cell_classification_labels: NDArray[np.float32],
    roi_index: int,
    cell_color: NDArray[np.uint8],
    non_cell_color: NDArray[np.uint8],
) -> None:
    """Updates the cell/non-cell overlay color for a reclassified ROI.

    Args:
        roi_maps: The ROI index maps.
        color_arrays: The computed color arrays.
        roi_statistics: The ROI statistics list.
        cell_classification_labels: The current cell classification label array.
        roi_index: Index of the ROI to update.
        cell_color: RGB color for cell ROIs derived from the active colormap high endpoint.
        non_cell_color: RGB color for non-cell ROIs derived from the active colormap low endpoint.
    """
    # Updates the binary classification color and normalized statistic for the flipped ROI.
    is_cell = bool(cell_classification_labels[roi_index])
    color_arrays.colors[ROIColorMode.CELL_CLASSIFICATION, roi_index] = cell_color if is_cell else non_cell_color
    color_arrays.normalized_statistics[ROIColorMode.CELL_CLASSIFICATION, roi_index] = float(is_cell)

    # Refreshes the precomputed RGB overlay pixels for every color slot including the binary classification slot.
    # Clips coordinates to valid frame bounds (backward-deformed multi-recording masks may have out-of-bounds pixels).
    y_pixels = roi_statistics[roi_index].mask.y_pixels
    x_pixels = roi_statistics[roi_index].mask.x_pixels
    frame_height, frame_width = color_arrays.rgb.shape[1], color_arrays.rgb.shape[2]
    valid = (y_pixels >= 0) & (y_pixels < frame_height) & (x_pixels >= 0) & (x_pixels < frame_width)
    y_pixels = y_pixels[valid]
    x_pixels = x_pixels[valid]
    for color_index in range(color_arrays.colors.shape[0]):
        color = color_arrays.colors[color_index]
        color_arrays.rgb[color_index, y_pixels, x_pixels, :3] = color[roi_maps.roi_indices[0, y_pixels, x_pixels], :]


def _highlight_selected_roi(
    overlay: NDArray[np.uint8],
    y_pixels: NDArray[np.int32],
    x_pixels: NDArray[np.int32],
    brightness: NDArray[np.float32],
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
    overlay[y_pixels, x_pixels, :] = (255 * brightness).astype(np.uint8)[:, np.newaxis]
    return overlay


def _highlight_selected_circle(
    overlay: NDArray[np.uint8],
    y_circle: NDArray[np.int32],
    x_circle: NDArray[np.int32],
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
