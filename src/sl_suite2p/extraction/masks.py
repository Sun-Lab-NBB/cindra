"""Provides assets for creating ROI (cell) and neuropil pixel masks associated with each detected ROI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import percentile_filter

from ..detection import extend_roi

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..dataclasses import ROIStatistics

# The rate at which to expand the neuropil mask, in pixels per expansion step.
_NEUROPIL_EXPANSION_STEP: int = 5

# The maximum number of neuropil expansion iterations before stopping.
_MAXIMUM_NEUROPIL_EXPANSION_ITERATIONS: int = 100

# The scaling factor applied to the median cell radius to determine the percentile filter neighborhood size.
_RADIUS_TO_NEIGHBORHOOD_SCALE: int = 5


def _create_roi_pixels(
    roi_statistics: list[ROIStatistics],
    height: int,
    width: int,
    cell_probability_percentile: int,
) -> NDArray[np.bool_]:
    """Creates a binary mask identifying all pixels that belong to any detected ROI.

    Args:
        roi_statistics: The ROI statistics for each ROI to be processed.
        height: The height of the imaged area in pixels.
        width: The width of the imaged area in pixels.
        cell_probability_percentile: The percentile threshold for considering a pixel as belonging to a cell ROI
            object, based on the lambda weight associated with the pixel.

    Returns:
        The created binary mask image.
    """
    cell_likelihood_map = np.zeros((height, width), dtype=np.float32)

    for roi in roi_statistics:
        # Ensures that the cell likelihood is measured using positive numbers only for the percentile filter below.
        cell_likelihood_map[roi.y_pixels, roi.x_pixels] = np.maximum(
            cell_likelihood_map[roi.y_pixels, roi.x_pixels], roi.pixel_weights
        )

    # Computes the median cell radius to determine the local neighborhood size for percentile filtering.
    median_radius = np.median(np.array([roi.radius for roi in roi_statistics]))

    # Selects ROI pixels based on the specified percentile threshold if additional likelihood filtering is enabled.
    if cell_probability_percentile > 0:
        # Selects pixels as 'ROI' if their likelihood is greater than or equal to the specified percentile filter.
        neighborhood_size = int(median_radius * _RADIUS_TO_NEIGHBORHOOD_SCALE)
        cell_threshold_filter = percentile_filter(
            input=cell_likelihood_map,
            percentile=cell_probability_percentile,
            size=neighborhood_size,
        ).astype(np.float32)
        pixel_mask = (cell_likelihood_map > 0.0) & (cell_likelihood_map >= cell_threshold_filter)
    else:
        # Selects all pixels with a weight greater than zero.
        pixel_mask = cell_likelihood_map > 0.0

    return pixel_mask


def _create_roi_masks(
    roi_statistics: list[ROIStatistics],
    width: int,
    include_overlap: bool,
) -> tuple[tuple[NDArray[np.int32], NDArray[np.float32]], ...]:
    """Creates the ROI pixel masks and the normalized lambda weight masks for the input ROIs.

    Args:
        roi_statistics: The ROI statistics for each ROI to be processed.
        width: The width of the imaged area in pixels. Used to compute flat pixel indices.
        include_overlap: Determines whether to include overlapping ROI pixels in the created ROI masks.

    Returns:
        A tuple of two masks for each ROI. The first is the flattened ROI pixel mask. The second is the flattened
        lambda weight mask for the pixels that make up the ROI mask.
    """
    cell_masks: list[tuple[NDArray[np.int32], NDArray[np.float32]]] = []

    for roi in roi_statistics:
        # Selects all pixels or excludes overlapping pixels depending on the include_overlap flag. Treats a missing
        # overlap mask as having no overlapping pixels.
        if include_overlap or roi.overlap_mask is None:
            pixel_mask: slice | NDArray[np.bool_] = slice(None)
        else:
            pixel_mask = ~roi.overlap_mask

        # Computes flat pixel indices directly via arithmetic instead of np.ravel_multi_index to avoid function-call
        # overhead and bounds checking. Applies the pixel selection mask in the same step.
        flat_indices = (roi.y_pixels * width + roi.x_pixels).astype(np.int32)[pixel_mask]
        weights = roi.pixel_weights[pixel_mask]

        # Normalizes the weights to sum to 1.0, creating a probability distribution of each pixel belonging to a
        # cell ROI.
        if weights.size > 0:
            normalized_weights = (weights / weights.sum()).astype(np.float32)
        else:
            normalized_weights = np.empty(0, dtype=np.float32)

        cell_masks.append((flat_indices, normalized_weights))

    return tuple(cell_masks)


def _create_neuropil_masks(
    roi_statistics: list[ROIStatistics],
    height: int,
    width: int,
    inner_neuropil_border_radius: int,
    minimum_neuropil_size: int,
    cell_probability_percentile: int,
    recompute: bool = False,
) -> tuple[NDArray[np.int32], ...]:
    """Creates the neuropil masks for the input ROIs, caching results on each ROIStatistics instance.

    Notes:
        Computed neuropil masks are stored as boolean arrays on each ROI's ``neuropil_mask`` field. When all ROIs
        already have cached masks and ``recompute`` is False, the cached masks are converted to flat indices and
        returned directly, skipping the expensive cell pixel map and iterative expansion computation.

    Args:
        roi_statistics: The ROI statistics for each ROI to be processed. Each ROI's ``neuropil_mask`` field is
            updated in-place with the computed boolean mask.
        height: The height of the imaged area in pixels.
        width: The width of the imaged area in pixels.
        inner_neuropil_border_radius: The radius of the border separating the neuropil region from the surrounded cell
            region, in pixels. Pixels that make up this border are excluded from the neuropil mask.
        minimum_neuropil_size: The minimum number of pixels to use for each created neuropil mask.
        cell_probability_percentile: The percentile threshold for labeling a pixel as belonging to the cell ROI region.
            This is used to determine the size of each ROI's cell region around which to form the neuropil mask.
        recompute: Determines whether to force recomputation of neuropil masks even when cached masks are available.

    Returns:
        The flattened neuropil masks for each ROI.
    """
    # Returns cached masks if all ROIs already have neuropil masks and recomputation is not requested.
    if not recompute and all(roi.neuropil_mask is not None for roi in roi_statistics):
        cached_masks: list[NDArray[np.int32]] = []
        for roi in roi_statistics:
            # Skips ROIs without cached masks. This branch is unreachable due to the all() guard above, but
            # satisfies mypy's type narrowing.
            if roi.neuropil_mask is None:
                continue
            cached_masks.append(np.flatnonzero(roi.neuropil_mask).astype(np.int32))
        return tuple(cached_masks)

    # Creates a binary mask of all cell pixels across all ROIs.
    cell_pixels = _create_roi_pixels(
        roi_statistics=roi_statistics,
        height=height,
        width=width,
        cell_probability_percentile=cell_probability_percentile,
    )

    neuropil_masks: list[NDArray[np.int32]] = []

    for roi in roi_statistics:
        # Extends the ROI to get a ring of pixels around the ROI center. This is the inner border that separates the
        # neuropil region from the cell region.
        inner_y_pixels, inner_x_pixels = extend_roi(
            y_pixels=roi.y_pixels,
            x_pixels=roi.x_pixels,
            height=height,
            width=width,
            iterations=inner_neuropil_border_radius,
        )

        # Determines the number of non-cell pixels within the inner neuropil border.
        exclude_count = int(np.sum(cell_pixels[inner_y_pixels, inner_x_pixels] == 0))

        # Iteratively expands the neuropil mask until it accumulates the requested number of pixels.
        current_y_pixels, current_x_pixels = inner_y_pixels.copy(), inner_x_pixels.copy()
        for _ in range(_MAXIMUM_NEUROPIL_EXPANSION_ITERATIONS):
            # Determines the number of neuropil region pixels at the start of the current iteration. Discounts the
            # inner neuropil border pixels to maintain a clear separation between the neuropil and the cell ROI regions.
            valid_pixels = int(np.sum(cell_pixels[current_y_pixels, current_x_pixels] == 0))
            neuropil_count = valid_pixels - exclude_count

            # Aborts expansion if the accumulated number of neuropil pixels exceeds the minimum required count.
            if neuropil_count > minimum_neuropil_size:
                break

            # Expands the neuropil mask by uniformly extending the neuropil's bounding box to include additional pixels
            # on each side. Clamps to frame boundaries to prevent out-of-bounds indices.
            y_min = max(0, int(current_y_pixels.min()) - _NEUROPIL_EXPANSION_STEP)
            y_max = min(height, int(current_y_pixels.max()) + _NEUROPIL_EXPANSION_STEP + 1)
            x_min = max(0, int(current_x_pixels.min()) - _NEUROPIL_EXPANSION_STEP)
            x_max = min(width, int(current_x_pixels.max()) + _NEUROPIL_EXPANSION_STEP + 1)
            current_y_pixels, current_x_pixels = np.meshgrid(
                np.arange(y_min, y_max, dtype=np.int32),
                np.arange(x_min, x_max, dtype=np.int32),
                indexing="ij",
            )

        # Creates the final neuropil mask for this ROI by excluding all inner border pixels and including all non-cell
        # pixels in the expanded neuropil region.
        is_non_cell = cell_pixels[current_y_pixels, current_x_pixels] == 0
        neuropil_mask = np.zeros((height, width), dtype=np.bool_)
        neuropil_mask[current_y_pixels[is_non_cell], current_x_pixels[is_non_cell]] = True
        neuropil_mask[inner_y_pixels, inner_x_pixels] = False

        # Caches the boolean mask on the ROI and extracts flat indices for the return value.
        roi.neuropil_mask = neuropil_mask
        neuropil_masks.append(np.flatnonzero(neuropil_mask).astype(np.int32))

    return tuple(neuropil_masks)


def create_masks(
    roi_statistics: list[ROIStatistics],
    height: int,
    width: int,
    neuropil: bool,
    include_overlap: bool,
    cell_probability_percentile: int = 50,
    inner_neuropil_border_radius: int = 2,
    minimum_neuropil_pixels: int = 350,
) -> tuple[tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.int32] | None], ...]:
    """Creates pixel masks for the ROI and the surrounding neuropil region of each detected ROI.

    Notes:
        The 'ROI masks' include both the flattened ROI mask pixel indices and the lambda weights associated with
        each mask pixel (the lambda weight masks). The neuropil region pixels are selected based on having
        sub-threshold lambda weights which are assumed to be 0. Therefore, the neuropil masks only include the
        flattened mask pixel indices.

    Args:
        roi_statistics: The ROI statistics for each ROI to be processed.
        height: The height of the imaged area in pixels.
        width: The width of the imaged area in pixels.
        neuropil: Determines whether to create the masks for the surrounding neuropil region for each ROI.
        include_overlap: Determines whether to include overlapping ROI pixels in the created ROI masks.
        cell_probability_percentile: The percentile threshold for classifying pixels as belonging to a cell versus
            neuropil.
        inner_neuropil_border_radius: The width, in pixels, of the exclusion zone between the cell ROI and its
            neuropil mask.
        minimum_neuropil_pixels: The minimum number of pixels required for each neuropil mask.

    Returns:
        A tuple of per-ROI mask data. Each element contains the flattened ROI pixel indices, the corresponding
        normalized lambda weights, and the flattened neuropil pixel indices (or None if neuropil processing is
        disabled).
    """
    roi_masks = _create_roi_masks(
        roi_statistics=roi_statistics,
        width=width,
        include_overlap=include_overlap,
    )

    # Combines ROI masks with None neuropil placeholders if neuropil processing is disabled.
    if not neuropil:
        return tuple((indices, weights, None) for indices, weights in roi_masks)

    neuropil_masks = _create_neuropil_masks(
        roi_statistics=roi_statistics,
        height=height,
        width=width,
        cell_probability_percentile=cell_probability_percentile,
        inner_neuropil_border_radius=inner_neuropil_border_radius,
        minimum_neuropil_size=minimum_neuropil_pixels,
    )

    return tuple(
        (indices, weights, neuropil_indices)
        for (indices, weights), neuropil_indices in zip(roi_masks, neuropil_masks, strict=True)
    )
