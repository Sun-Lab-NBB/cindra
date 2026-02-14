"""Provides algorithms for determining ROI colocalization in multichannel imaging data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import csr_matrix
from scipy.ndimage import gaussian_filter

from .masks import create_masks

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..dataclasses import ROIStatistics

# The number of spatial blocks along each axis used for block-wise bleedthrough regression.
_BLOCK_COUNT: int = 3

# The fraction of the mean block dimension used as the Gaussian smoothing sigma for quadrant masks.
_SMOOTHING_FRACTION: float = 0.25

# The minimum intensity floor used to prevent division by zero in colocalization probability computation.
_INTENSITY_EPSILON: float = 1e-3


def _correct_bleedthrough(
    functional_mean_image: NDArray[np.float32],
    structural_mean_image: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Corrects bleedthrough from the functional channel into the structural channel using local regression.

    Performs non-rigid regression to subtract the contribution of functional channel signal that bleeds
    into the structural channel. The image is divided into a 3x3 grid of blocks, and for each block a
    linear coefficient is computed to predict structural intensity from functional intensity. The predicted
    bleedthrough is then subtracted from the structural channel.

    Notes:
        The block-wise approach accounts for spatial variations in bleedthrough across the field of view.
        Each 2D block mask is a Gaussian-smoothed rectangle, which factors as the outer product of two 1D
        Gaussian-smoothed step functions. This separability reduces 9 2D convolutions to 6 1D convolutions
        and allows the mask-normalized weighted correction to be expressed as a single matrix product.

    Args:
        functional_mean_image: The temporal mean image for the functional channel with shape (height, width).
        structural_mean_image: The temporal mean image for the structural channel with shape (height, width).

    Returns:
        The bleedthrough-corrected structural mean image with non-negative values.
    """
    frame_height, frame_width = functional_mean_image.shape

    # Computes the smoothing sigma based on image dimensions and block count.
    smoothing_sigma = round((frame_height + frame_width) / (_BLOCK_COUNT * 2) * _SMOOTHING_FRACTION)

    # Computes block boundaries for dividing the image.
    y_boundaries = np.linspace(start=0, stop=frame_height, num=_BLOCK_COUNT + 1).astype(np.intp)
    x_boundaries = np.linspace(start=0, stop=frame_width, num=_BLOCK_COUNT + 1).astype(np.intp)

    # Computes 1D Gaussian-smoothed masks along each axis independently. Each 2D block mask is the
    # outer product of a y-axis indicator and an x-axis indicator, and since the Gaussian filter is
    # separable, the smoothed 2D mask equals the outer product of the two 1D smoothed indicators.
    y_masks = np.zeros((_BLOCK_COUNT, frame_height), dtype=np.float32)
    x_masks = np.zeros((_BLOCK_COUNT, frame_width), dtype=np.float32)

    for block_index in range(_BLOCK_COUNT):
        y_indicator = np.zeros(frame_height, dtype=np.float32)
        y_indicator[y_boundaries[block_index] : y_boundaries[block_index + 1]] = 1.0
        y_masks[block_index] = gaussian_filter(input=y_indicator, sigma=smoothing_sigma).astype(np.float32)

        x_indicator = np.zeros(frame_width, dtype=np.float32)
        x_indicator[x_boundaries[block_index] : x_boundaries[block_index + 1]] = 1.0
        x_masks[block_index] = gaussian_filter(input=x_indicator, sigma=smoothing_sigma).astype(np.float32)

    # Computes the linear regression weight for each block.
    weights = np.zeros((_BLOCK_COUNT, _BLOCK_COUNT), dtype=np.float32)
    for y_block in range(_BLOCK_COUNT):
        y_slice = slice(y_boundaries[y_block], y_boundaries[y_block + 1])
        for x_block in range(_BLOCK_COUNT):
            x_slice = slice(x_boundaries[x_block], x_boundaries[x_block + 1])

            functional_block = functional_mean_image[y_slice, x_slice]
            structural_block = structural_mean_image[y_slice, x_slice]

            numerator = (functional_block * structural_block).sum()
            denominator = (functional_block * functional_block).sum()
            weights[y_block, x_block] = numerator / denominator if denominator > 0 else 0.0

    # Normalizes each 1D mask by its per-axis sum. The 2D mask sum factors as the outer product of
    # the 1D mask sums because each 2D mask is itself rank-1.
    y_mask_sum = y_masks.sum(axis=0)
    x_mask_sum = x_masks.sum(axis=0)
    y_mask_sum[y_mask_sum == 0] = 1.0
    x_mask_sum[x_mask_sum == 0] = 1.0

    y_normalized = y_masks / y_mask_sum[np.newaxis, :]
    x_normalized = x_masks / x_mask_sum[np.newaxis, :]

    # Expresses the correction as F * (Y_norm.T @ W @ X_norm), which is equivalent to summing
    # (normalized_mask * weight * F) over all blocks but avoids explicit 2D mask construction.
    weighted_blend: NDArray[np.float32] = (y_normalized.T @ weights @ x_normalized).astype(np.float32)
    correction = functional_mean_image * weighted_blend

    corrected: NDArray[np.float32] = np.maximum(0, structural_mean_image - correction).astype(np.float32)
    return corrected


def _build_sparse_roi_masks(
    rois: list[ROIStatistics],
    frame_height: int,
    frame_width: int,
) -> csr_matrix:
    """Builds a sparse binary mask matrix from ROI pixel coordinates.

    Args:
        rois: The ROI statistics containing pixel coordinates.
        frame_height: The height of the imaging field in pixels.
        frame_width: The width of the imaging field in pixels.

    Returns:
        A Compressed Sparse Row (CSR) matrix of shape (n_rois, frame_height * frame_width) where each row contains
        ones at the flattened pixel indices belonging to that ROI.
    """
    total_pixels = frame_height * frame_width
    roi_count = len(rois)

    # Accumulates COO-format triplet arrays (row, column, value) for all ROIs. Each ROI contributes
    # one entry per pixel, where the row is the ROI index and the column is the flattened pixel index.
    row_indices: list[NDArray[np.intp]] = []
    column_indices: list[NDArray[np.intp]] = []

    for roi_index, roi in enumerate(rois):
        # Converts 2D pixel coordinates to flattened 1D indices via row-major arithmetic.
        flat_pixels = (roi.y_pixels * frame_width + roi.x_pixels).astype(np.intp)

        # Assigns every pixel in this ROI to the same row (roi_index) in the sparse matrix.
        row_indices.append(np.full(len(flat_pixels), fill_value=roi_index, dtype=np.intp))
        column_indices.append(flat_pixels)

    # Merges per-ROI arrays into single COO-format arrays for CSR construction.
    all_rows = np.concatenate(row_indices)
    all_columns = np.concatenate(column_indices)
    data = np.ones(len(all_rows), dtype=np.float32)

    # Constructs the CSR matrix from COO triplets. Duplicate (row, column) entries are summed by
    # default, so any repeated pixel coordinates within an ROI will produce values greater than 1.
    masks = csr_matrix((data, (all_rows, all_columns)), shape=(roi_count, total_pixels))

    # Clips summed duplicates back to binary values to ensure each pixel is counted at most once.
    masks.data = np.minimum(masks.data, np.float32(1.0))

    return masks


def _compute_overlap_matrix(
    rois_channel_1: list[ROIStatistics],
    rois_channel_2: list[ROIStatistics],
    frame_height: int,
    frame_width: int,
) -> NDArray[np.float32]:
    """Computes pairwise overlap fractions between two sets of ROIs.

    The overlap fraction is defined as the intersection size divided by the size of the smaller ROI.
    This normalization ensures that a small ROI fully contained within a large ROI receives an
    overlap score of 1.0.

    Notes:
        Builds sparse binary mask matrices for each channel and computes all pairwise intersection
        counts via a single sparse matrix multiplication, avoiding explicit Python-level set operations.

    Args:
        rois_channel_1: The ROI statistics for channel 1.
        rois_channel_2: The ROI statistics for channel 2.
        frame_height: The height of the imaging field in pixels.
        frame_width: The width of the imaging field in pixels.

    Returns:
        An array of shape (n_rois_1, n_rois_2) where element [i, j] is the overlap fraction between
        ROI i from channel 1 and ROI j from channel 2.
    """
    count_1 = len(rois_channel_1)
    count_2 = len(rois_channel_2)

    # Handles edge cases with empty ROI lists. This case is handled explicitly by external callers, so this fallback is
    # mostly to appease mypy.
    if count_1 == 0 or count_2 == 0:
        return np.zeros((count_1, count_2), dtype=np.float32)

    # Builds sparse binary masks for both channels.
    sparse_masks_1 = _build_sparse_roi_masks(
        rois=rois_channel_1,
        frame_height=frame_height,
        frame_width=frame_width,
    )
    sparse_masks_2 = _build_sparse_roi_masks(
        rois=rois_channel_2,
        frame_height=frame_height,
        frame_width=frame_width,
    )

    # Computes all pairwise intersection counts via sparse matrix multiplication.
    intersection_counts: NDArray[np.float32] = (sparse_masks_1 @ sparse_masks_2.T).toarray().astype(np.float32)

    # Computes per-ROI pixel counts from sparse row sums.
    sizes_1: NDArray[np.float32] = np.asarray(sparse_masks_1.sum(axis=1), dtype=np.float32).ravel()
    sizes_2: NDArray[np.float32] = np.asarray(sparse_masks_2.sum(axis=1), dtype=np.float32).ravel()

    # Normalizes in-place by the smaller ROI size per pair, avoiding a second (count_1, count_2)
    # allocation for the result.
    minimum_sizes = np.minimum(sizes_1[:, np.newaxis], sizes_2[np.newaxis, :])
    minimum_sizes[minimum_sizes == 0] = 1.0
    intersection_counts /= minimum_sizes

    return intersection_counts


def compute_intensity_colocalization(
    rois: list[ROIStatistics],
    functional_mean_image: NDArray[np.float32],
    structural_mean_image: NDArray[np.float32],
    frame_height: int,
    frame_width: int,
    colocalization_threshold: float,
    allow_overlap: bool,
    cell_probability_percentile: int,
    inner_neuropil_border_radius: int,
    minimum_neuropil_pixels: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Computes the intensity colocalization between the functional channel's ROIs and the structural channel.

    Computes a colocalization probability for each functional ROI by comparing the mean intensity inside
    the ROI to the intensity in the surrounding neuropil region in the structural channel. ROIs with a
    high inside-to-surround ratio are likely present in both channels. Bleed-through correction is applied
    automatically to remove functional channel signal that leaks into the structural channel.

    Notes:
        This method is appropriate when one channel contains functional data (e.g., GCaMP calcium
        indicator) and the other contains structural data (e.g., tdTomato cell marker). The intensity ratio
        approach assumes that colocalized ROIs will have higher signal inside the cell boundary than in the
        surrounding neuropil.

    Args:
        rois: The ROI statistics from functional channel detection.
        functional_mean_image: The temporal mean image from the functional channel, used for
            bleedthrough correction.
        structural_mean_image: The temporal mean image from the structural channel, used for
            intensity measurement.
        frame_height: The height of the imaging field in pixels.
        frame_width: The width of the imaging field in pixels.
        colocalization_threshold: The minimum probability for classifying an ROI as colocalized. For intensity-based
            colocalization, this represents the inside-to-total intensity ratio threshold.
        allow_overlap: Determines whether to include overlapping ROI pixels in the created masks.
        cell_probability_percentile: The percentile threshold for classifying pixels as belonging to a cell versus
            neuropil.
        inner_neuropil_border_radius: The width, in pixels, of the exclusion zone between the cell ROI and its
            neuropil mask.
        minimum_neuropil_pixels: The minimum number of pixels required for each neuropil mask.

    Returns:
        A tuple of two arrays. The first array has shape (n_rois, 2) where column 0 contains boolean
        colocalization flags and column 1 contains probability values. The second array is the
        bleedthrough-corrected structural mean image.
    """
    # Handles edge cases with empty ROI lists. This case is handled explicitly by external callers, so this fallback is
    # mostly to appease mypy.
    if len(rois) == 0:
        empty_result = np.zeros((0, 2), dtype=np.float32)
        return empty_result, structural_mean_image.astype(np.float32)

    # Corrects for bleedthrough from functional channel into structural channel.
    corrected_mean_image = _correct_bleedthrough(
        functional_mean_image=functional_mean_image,
        structural_mean_image=structural_mean_image.copy(),
    )

    # Creates cell and neuropil masks from the ROI statistics. Neuropil masks are always required for intensity
    # colocalization since the algorithm compares inside-ROI vs. neuropil-region intensity.
    per_roi_masks = create_masks(
        roi_statistics=rois,
        height=frame_height,
        width=frame_width,
        neuropil=True,
        include_overlap=allow_overlap,
        cell_probability_percentile=cell_probability_percentile,
        inner_neuropil_border_radius=inner_neuropil_border_radius,
        minimum_neuropil_pixels=minimum_neuropil_pixels,
    )

    # Computes per-ROI weighted intensity inside each cell and mean intensity in the neuropil
    # region directly from sparse mask data, avoiding dense (roi_count, total_pixels) allocation.
    roi_count = len(rois)
    flattened_image = corrected_mean_image.ravel()
    intensity_inside = np.zeros(roi_count, dtype=np.float32)
    intensity_outside = np.zeros(roi_count, dtype=np.float32)

    for roi_index, (cell_indices, cell_weights, neuropil_indices) in enumerate(per_roi_masks):
        intensity_inside[roi_index] = np.dot(cell_weights, flattened_image[cell_indices])

        if neuropil_indices is not None and len(neuropil_indices) > 0:
            intensity_outside[roi_index] = flattened_image[neuropil_indices].mean()

    # Computes the colocalization probability as the ratio of inside to total intensity. Adds a small
    # epsilon to prevent division by zero and ensure numerical stability.
    intensity_inside = np.maximum(np.float32(_INTENSITY_EPSILON), intensity_inside)
    colocalization_probability = intensity_inside / (intensity_inside + intensity_outside)

    # Applies the threshold to determine which ROIs are colocalized.
    is_colocalized = colocalization_probability > colocalization_threshold

    # Stacks results into (n_rois, 2) array matching the reference implementation format.
    colocalization_result = np.stack((is_colocalized, colocalization_probability), axis=-1).astype(np.float32)

    return colocalization_result, corrected_mean_image


def compute_spatial_colocalization(
    rois_channel_1: list[ROIStatistics],
    rois_channel_2: list[ROIStatistics],
    frame_height: int,
    frame_width: int,
    colocalization_threshold: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Computes spatial colocalization by matching ROIs between two channels based on pixel overlap.

    Computes pairwise overlap fractions between all ROIs in channel 1 and channel 2, then finds
    mutually consistent best-match pairs. A pair (i, j) is only accepted when channel 1 ROI i's
    best match is channel 2 ROI j AND channel 2 ROI j's best match is channel 1 ROI i. This
    enforces convergent bidirectional matching where every accepted pairing is reciprocal.

    Notes:
        This method is appropriate when both channels contain functional data with independently
        detected ROIs. The overlap fraction is normalized by the smaller ROI size, ensuring that
        a small ROI fully contained within a larger ROI receives an overlap score of 1.0. The
        mutual best-match constraint guarantees that accepted pairings are consistent across both
        output arrays: if channel_1_to_2[i] points to j, then channel_2_to_1[j] points to i.

    Args:
        rois_channel_1: The ROI statistics for channel 1 ROIs.
        rois_channel_2: The ROI statistics for channel 2 ROIs.
        frame_height: The height of the imaging field in pixels.
        frame_width: The width of the imaging field in pixels.
        colocalization_threshold: The minimum overlap fraction for considering ROIs as matched. For spatial
            colocalization, this represents the pixel overlap ratio threshold.

    Returns:
        A tuple of two arrays for bidirectional ROI mappings. The first array has shape
        (n_channel_1_rois, 2) where column 0 contains the matched channel 2 ROI index (-1 if no
        match) and column 1 contains the overlap score. The second array has shape
        (n_channel_2_rois, 2) with the same format for channel 2 to channel 1 mapping.
    """
    count_1 = len(rois_channel_1)
    count_2 = len(rois_channel_2)

    # Handles edge cases with empty ROI lists. This case is handled explicitly by external callers, so this fallback is
    # mostly to appease mypy.
    if count_1 == 0:
        channel_1_to_2 = np.zeros((0, 2), dtype=np.float32)
        channel_2_to_1 = np.column_stack(
            (
                np.full(count_2, fill_value=-1, dtype=np.float32),
                np.zeros(count_2, dtype=np.float32),
            )
        )
        return channel_1_to_2, channel_2_to_1

    if count_2 == 0:
        channel_1_to_2 = np.column_stack(
            (
                np.full(count_1, fill_value=-1, dtype=np.float32),
                np.zeros(count_1, dtype=np.float32),
            )
        )
        channel_2_to_1 = np.zeros((0, 2), dtype=np.float32)
        return channel_1_to_2, channel_2_to_1

    # Computes the pairwise overlap matrix via sparse matrix multiplication.
    overlap_matrix = _compute_overlap_matrix(
        rois_channel_1=rois_channel_1,
        rois_channel_2=rois_channel_2,
        frame_height=frame_height,
        frame_width=frame_width,
    )

    # Finds the best match index and score for each ROI in both directions.
    best_indices_1 = np.argmax(overlap_matrix, axis=1)
    best_scores_1 = np.max(overlap_matrix, axis=1)
    best_indices_2 = np.argmax(overlap_matrix, axis=0)
    best_scores_2 = np.max(overlap_matrix, axis=0)

    # Enforces mutual best matching: a pair (i, j) is accepted only when channel 1 ROI i's best
    # match is j AND channel 2 ROI j's best match is i. For each channel 1 ROI i, looks up its
    # proposed partner j = best_indices_1[i], then checks whether j's best partner points back to i.
    is_mutual_1 = best_indices_2[best_indices_1] == np.arange(count_1)
    is_mutual_2 = best_indices_1[best_indices_2] == np.arange(count_2)

    # Rejects non-mutual matches and pairs below the overlap threshold.
    unmatched_1 = ~is_mutual_1 | (best_scores_1 < colocalization_threshold)
    unmatched_2 = ~is_mutual_2 | (best_scores_2 < colocalization_threshold)

    channel_1_to_2 = np.empty((count_1, 2), dtype=np.float32)
    channel_1_to_2[:, 0] = best_indices_1
    channel_1_to_2[:, 1] = best_scores_1
    channel_1_to_2[unmatched_1, 0] = -1
    channel_1_to_2[unmatched_1, 1] = 0.0

    channel_2_to_1 = np.empty((count_2, 2), dtype=np.float32)
    channel_2_to_1[:, 0] = best_indices_2
    channel_2_to_1[:, 1] = best_scores_2
    channel_2_to_1[unmatched_2, 0] = -1
    channel_2_to_1[unmatched_2, 1] = 0.0

    return channel_1_to_2, channel_2_to_1
