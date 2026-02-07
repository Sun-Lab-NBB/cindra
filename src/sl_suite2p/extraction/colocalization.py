"""Provides assets for determining ROI colocalization in multichannel imaging data."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter

from .masks import create_masks

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..dataclasses import ROIStatistics


def _create_quadrant_mask(
    frame_height: int,
    frame_width: int,
    y_indices: NDArray[np.uintp],
    x_indices: NDArray[np.uintp],
    smoothing_sigma: float,
) -> NDArray[np.float32]:
    """Creates a smoothed quadrant mask for local bleed-through regression.

    The mask is initialized with ones inside the specified region and zeros elsewhere, then smoothed
    with a Gaussian filter to create soft boundaries between regions.

    Args:
        frame_height: The height of the imaging field in pixels.
        frame_width: The width of the imaging field in pixels.
        y_indices: The y-coordinates of pixels to include in the mask.
        x_indices: The x-coordinates of pixels to include in the mask.
        smoothing_sigma: The standard deviation of the Gaussian smoothing kernel.

    Returns:
        The smoothed quadrant mask with shape (frame_height, frame_width).
    """
    mask = np.zeros((frame_height, frame_width), dtype=np.float32)
    mask[np.ix_(y_indices, x_indices)] = 1.0
    return gaussian_filter(input=mask, sigma=smoothing_sigma).astype(np.float32)


def _correct_bleed_through(
    functional_mean_image: NDArray[np.float32],
    structural_mean_image: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Corrects bleed-through from the functional channel into the structural channel using local regression.

    This function performs non-rigid regression to subtract the contribution of functional channel signal
    that bleeds into the structural channel. The image is divided into a 3x3 grid of blocks, and for each
    block a linear coefficient is computed to predict structural intensity from functional intensity. The
    predicted bleed-through is then subtracted from the structural channel.

    Notes:
        The block-wise approach accounts for spatial variations in bleed-through across the field of view.

    Args:
        functional_mean_image: The temporal mean image for the functional channel with shape (height, width).
        structural_mean_image: The temporal mean image for the structural channel with shape (height, width).

    Returns:
        The bleed-through-corrected structural mean image with non-negative values.
    """
    frame_height, frame_width = functional_mean_image.shape
    block_count = 3

    # Computes the smoothing sigma based on image dimensions and block count.
    smoothing_sigma = round((frame_height + frame_width) / (block_count * 2) * 0.25)

    # Computes block boundaries for dividing the image.
    y_boundaries = np.linspace(start=0, stop=frame_height, num=block_count + 1).astype(np.uintp)
    x_boundaries = np.linspace(start=0, stop=frame_width, num=block_count + 1).astype(np.uintp)

    # First pass: computes masks and regression weights, accumulates mask sum for normalization.
    mask_sum = np.zeros((frame_height, frame_width), dtype=np.float32)
    block_data: list[tuple[NDArray[np.float32], float]] = []

    for y_block in range(block_count):
        for x_block in range(block_count):
            # Extracts the pixel indices for the current block.
            y_indices = np.arange(y_boundaries[y_block], y_boundaries[y_block + 1], dtype=np.uintp)
            x_indices = np.arange(x_boundaries[x_block], x_boundaries[x_block + 1], dtype=np.uintp)

            # Creates the smoothed mask for this block.
            mask = _create_quadrant_mask(
                frame_height=frame_height,
                frame_width=frame_width,
                y_indices=y_indices,
                x_indices=x_indices,
                smoothing_sigma=smoothing_sigma,
            )

            # Extracts the pixel values from both channels for this block.
            functional_block = functional_mean_image[np.ix_(y_indices, x_indices)].flatten()
            structural_block = structural_mean_image[np.ix_(y_indices, x_indices)].flatten()

            # Computes the linear regression coefficient predicting structural from functional.
            numerator = (functional_block * structural_block).sum()
            denominator = (functional_block * functional_block).sum()
            weight = numerator / denominator if denominator > 0 else 0.0

            mask_sum += mask
            block_data.append((mask, weight))

    # Avoids division by zero in normalization.
    mask_sum[mask_sum == 0] = 1.0

    # Second pass: computes normalized weighted correction.
    correction = np.zeros((frame_height, frame_width), dtype=np.float32)
    for mask, weight in block_data:
        correction += (mask / mask_sum) * weight * functional_mean_image

    # Generates and returns the corrected image.
    corrected: NDArray[np.float32] = np.maximum(0, structural_mean_image - correction).astype(np.float32)
    return corrected


def compute_intensity_colocalization(
    rois: list[ROIStatistics],
    functional_mean_image: NDArray[np.float32],
    structural_mean_image: NDArray[np.float32],
    frame_height: int,
    frame_width: int,
    colocalization_threshold: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Computes the intensity colocalization between the functional channel's ROIs and the structural channel.

    This function computes a colocalization probability for each functional ROI by comparing the mean intensity
    inside the ROI to the intensity in the surrounding neuropil region in the structural channel.
    ROIs with a high inside-to-surround ratio are likely present in both channels. Bleed-through
    correction is applied automatically to remove functional channel signal that leaks into the
    structural channel.

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
        colocalization_threshold: The minimum probability for classifying an ROI as colocalized,
            sourced from the pipeline configuration. For intensity-based colocalization, this represents
            the inside-to-total intensity ratio threshold.

    Returns:
        A tuple of two arrays. The first array has shape (n_rois, 2) where column 0 contains boolean
        colocalization flags and column 1 contains probability values. The second array is the
        bleedthrough-corrected structural mean image.
    """
    # Handles the edge case of empty ROI list.
    if len(rois) == 0:
        empty_result = np.zeros((0, 2), dtype=np.float32)
        return empty_result, structural_mean_image.astype(np.float32)

    # Corrects for bleedthrough from functional channel into structural channel.
    corrected_mean_image = _correct_bleed_through(
        functional_mean_image=functional_mean_image,
        structural_mean_image=structural_mean_image.copy(),
    )

    # Creates extraction configuration dictionary with required parameters.
    extraction_ops = {
        "allow_overlap": True,
        "lambda_percentile": 50,
        "inner_neuropil_border_radius": 2,
        "minimum_neuropil_pixels": 350,
    }

    # Converts ROIStatistics to dict format for create_masks compatibility.
    roi_dicts = [
        {"y_pixels": roi.y_pixels, "x_pixels": roi.x_pixels, "pixel_weights": roi.pixel_weights, "radius": roi.radius}
        for roi in rois
    ]

    # Creates cell and neuropil masks from the ROI statistics.
    cell_masks_sparse, neuropil_masks_sparse = create_masks(
        roi_statistics=roi_dicts,
        height=frame_height,
        width=frame_width,
        neuropil=True,
        ops=extraction_ops,
    )

    # Verifies that neuropil masks were created since neuropil=True was specified. This check is required for type
    # narrowing, as create_masks can return None when neuropil=False.
    if neuropil_masks_sparse is None:
        message = "Internal error: neuropil masks were not created despite neuropil=True."
        raise RuntimeError(message)

    # Converts sparse masks to dense format for matrix multiplication.
    total_pixels = frame_height * frame_width
    roi_count = len(rois)

    cell_masks_dense = np.zeros((roi_count, total_pixels), dtype=np.float32)
    neuropil_masks_dense = np.zeros((roi_count, total_pixels), dtype=np.float32)

    for roi_index, (cell_mask, neuropil_mask) in enumerate(zip(cell_masks_sparse, neuropil_masks_sparse, strict=True)):
        # Cell mask contains (indices, weights) tuple.
        cell_indices, cell_weights = cell_mask
        cell_masks_dense[roi_index, cell_indices.astype(np.int64)] = cell_weights

        # Neuropil mask contains only indices, with uniform weights.
        neuropil_count = len(neuropil_mask)
        if neuropil_count > 0:
            neuropil_masks_dense[roi_index, neuropil_mask.astype(np.int64)] = 1.0 / neuropil_count

    # Computes the weighted intensity inside each ROI and in the neuropil region.
    flattened_image = corrected_mean_image.flatten()
    intensity_inside = cell_masks_dense @ flattened_image
    intensity_outside = neuropil_masks_dense @ flattened_image

    # Computes the colocalization probability as the ratio of inside to total intensity. Adds a small
    # epsilon to prevent division by zero and ensure numerical stability.
    epsilon = 1e-3
    intensity_inside = np.maximum(epsilon, intensity_inside)
    colocalization_probability = intensity_inside / (intensity_inside + intensity_outside)

    # Applies the threshold to determine which ROIs are colocalized.
    is_colocalized = colocalization_probability > colocalization_threshold

    # Stacks results into (n_rois, 2) array matching the reference implementation format.
    colocalization_result = np.stack((is_colocalized, colocalization_probability), axis=-1).astype(np.float32)

    return colocalization_result, corrected_mean_image


def _compute_roi_pixel_sets(
    rois: list[ROIStatistics],
    frame_height: int,
    frame_width: int,
) -> list[set[int]]:
    """Converts ROI pixel coordinates to sets of flattened indices.

    Args:
        rois: The ROI statistics containing pixel coordinates.
        frame_height: The height of the imaging field in pixels.
        frame_width: The width of the imaging field in pixels.

    Returns:
        A list of sets, where each set contains the flattened pixel indices for one ROI.
    """
    pixel_sets = []

    for roi in rois:
        # Converts 2D coordinates to flattened 1D indices.
        flattened_indices = np.ravel_multi_index(
            multi_index=(roi.y_pixels, roi.x_pixels),
            dims=(frame_height, frame_width),
        )

        pixel_sets.append(set(flattened_indices.tolist()))

    return pixel_sets


def _compute_overlap_matrix(
    pixel_sets_1: list[set[int]],
    pixel_sets_2: list[set[int]],
) -> NDArray[np.float32]:
    """Computes pairwise overlap fractions between two sets of ROIs.

    The overlap fraction is defined as the intersection size divided by the size of the smaller ROI.
    This normalization ensures that a small ROI fully contained within a large ROI receives an
    overlap score of 1.0.

    Args:
        pixel_sets_1: The pixel sets for channel 1 ROIs.
        pixel_sets_2: The pixel sets for channel 2 ROIs.

    Returns:
        An array of shape (n_rois_1, n_rois_2) where element [i, j] is the overlap fraction between
        ROI i from channel 1 and ROI j from channel 2.
    """
    count_1 = len(pixel_sets_1)
    count_2 = len(pixel_sets_2)

    # Handles edge cases with empty ROI lists.
    if count_1 == 0 or count_2 == 0:
        return np.zeros((count_1, count_2), dtype=np.float32)

    overlap_matrix = np.zeros((count_1, count_2), dtype=np.float32)

    for index_1, pixels_1 in enumerate(pixel_sets_1):
        size_1 = len(pixels_1)
        if size_1 == 0:
            continue

        for index_2, pixels_2 in enumerate(pixel_sets_2):
            size_2 = len(pixels_2)
            if size_2 == 0:
                continue

            # Computes the intersection size.
            intersection_size = len(pixels_1 & pixels_2)

            # Normalizes by the smaller ROI size.
            minimum_size = min(size_1, size_2)
            overlap_matrix[index_1, index_2] = intersection_size / minimum_size

    return overlap_matrix


def compute_spatial_colocalization(
    rois_channel_1: list[ROIStatistics],
    rois_channel_2: list[ROIStatistics],
    frame_height: int,
    frame_width: int,
    colocalization_threshold: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Computes spatial colocalization by matching ROIs between two channels based on pixel overlap.

    This function computes pairwise overlap fractions between all ROIs in channel 1 and channel 2,
    then finds the best matching pairs. The matching is performed bidirectionally, meaning each ROI
    in one channel is matched to its best counterpart in the other channel independently.

    Notes:
        This method is appropriate when both channels contain functional data with independently
        detected ROIs. The overlap fraction is normalized by the smaller ROI size, ensuring that
        a small ROI fully contained within a larger ROI receives an overlap score of 1.0. The
        bidirectional matching allows for asymmetric relationships where ROI A's best match is ROI B,
        but ROI B's best match might be a different ROI.

    Args:
        rois_channel_1: The ROI statistics for channel 1 ROIs.
        rois_channel_2: The ROI statistics for channel 2 ROIs.
        frame_height: The height of the imaging field in pixels.
        frame_width: The width of the imaging field in pixels.
        colocalization_threshold: The minimum overlap fraction for considering ROIs as matched,
            sourced from the pipeline configuration. For spatial colocalization, this represents the
            pixel overlap ratio threshold.

    Returns:
        A tuple of two arrays for bidirectional ROI mappings. The first array has shape
        (n_channel_1_rois, 2) where column 0 contains the matched channel 2 ROI index (-1 if no
        match) and column 1 contains the overlap score. The second array has shape
        (n_channel_2_rois, 2) with the same format for channel 2 to channel 1 mappings.
    """
    count_1 = len(rois_channel_1)
    count_2 = len(rois_channel_2)

    # Handles edge cases with empty ROI lists.
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

    # Converts ROI coordinates to pixel sets.
    pixel_sets_1 = _compute_roi_pixel_sets(
        rois=rois_channel_1,
        frame_height=frame_height,
        frame_width=frame_width,
    )
    pixel_sets_2 = _compute_roi_pixel_sets(
        rois=rois_channel_2,
        frame_height=frame_height,
        frame_width=frame_width,
    )

    # Computes the pairwise overlap matrix.
    overlap_matrix = _compute_overlap_matrix(pixel_sets_1=pixel_sets_1, pixel_sets_2=pixel_sets_2)

    # Finds the best match for each channel 1 ROI in channel 2.
    best_match_1_to_2 = np.argmax(overlap_matrix, axis=1).astype(np.float32)
    best_scores_1_to_2 = overlap_matrix[np.arange(count_1), best_match_1_to_2.astype(np.int32)]

    # Applies threshold to channel 1 to channel 2 matches.
    below_threshold_1 = best_scores_1_to_2 < colocalization_threshold
    best_match_1_to_2[below_threshold_1] = -1
    best_scores_1_to_2[below_threshold_1] = 0.0

    # Finds the best match for each channel 2 ROI in channel 1.
    best_match_2_to_1 = np.argmax(overlap_matrix, axis=0).astype(np.float32)
    best_scores_2_to_1 = overlap_matrix[best_match_2_to_1.astype(np.int32), np.arange(count_2)]

    # Applies threshold to channel 2 to channel 1 matches.
    below_threshold_2 = best_scores_2_to_1 < colocalization_threshold
    best_match_2_to_1[below_threshold_2] = -1
    best_scores_2_to_1[below_threshold_2] = 0.0

    # Stacks results into (n_rois, 2) arrays.
    channel_1_to_2 = np.column_stack((best_match_1_to_2, best_scores_1_to_2)).astype(np.float32)
    channel_2_to_1 = np.column_stack((best_match_2_to_1, best_scores_2_to_1)).astype(np.float32)

    return channel_1_to_2, channel_2_to_1
