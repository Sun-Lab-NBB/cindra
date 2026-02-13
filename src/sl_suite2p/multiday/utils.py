"""This module provides helper functions for the multi-day suite2p processing pipeline.

The functions in this module support cell mask overlap detection for multi-session registration workflows.
"""

from typing import Any

from numba import njit, prange
import numpy as np
from numpy.typing import NDArray


@njit(parallel=True, cache=True)
def _find_overlapping_pixels(all_pixels: NDArray[np.int32]) -> NDArray[np.int32]:
    """This service function is used by the add_overlap_info function to find all pixels that appear in more than one
    cell mask.

    Args:
        all_pixels: A numpy array that linearly concatenates the pixel indices from all cell masks.

    Returns:
        A NumPy array that stores the indices of pixels that appear in more than one cell mask.
    """
    # Sorts pixel indices
    sorted_pixels = np.sort(all_pixels)

    # Creates a mask of duplicate pixels
    is_duplicate = np.zeros(len(sorted_pixels), dtype=np.bool_)
    for i in prange(1, len(sorted_pixels)):
        is_duplicate[i] = sorted_pixels[i] == sorted_pixels[i - 1]

    # Get unique duplicate values
    return np.unique(sorted_pixels[is_duplicate])


@njit(parallel=True, cache=True)
def _create_overlap_arrays(
    flat_pixel_indices: NDArray[np.int32],
    overlapping_pixels: NDArray[np.int32],
) -> NDArray[np.bool_]:
    """Creates a boolean array marking overlapping pixels for all masks.

    Args:
        flat_pixel_indices: Flattened array containing all mask pixel indices concatenated together.
        overlapping_pixels: A NumPy array storing the indices of overlapping pixels.

    Returns:
        A flattened boolean array marking overlapping pixels for all masks.
    """
    total_pixels = len(flat_pixel_indices)
    overlap_result = np.zeros(total_pixels, dtype=np.bool_)

    for pixel_index in prange(total_pixels):
        pixel_value = flat_pixel_indices[pixel_index]
        for overlap_index in range(len(overlapping_pixels)):
            if pixel_value == overlapping_pixels[overlap_index]:
                overlap_result[pixel_index] = True
                break

    return overlap_result


def add_overlap_info(masks: list[dict[str, Any]]) -> list[dict[str, NDArray[np.bool]]]:
    """Identifies overlapping pixels across the input cell masks and augments each cell mask dictionary with
    overlapping pixel information.

    Args:
        masks: The list of cell mask dictionaries with 'raveled_pixels' keys.

    Returns:
        The list of modified mask dictionaries, which now contain the added 'overlap_mask' boolean arrays.
    """
    # Extracts pixel indices from all masks
    mask_pixel_indices = [mask["raveled_pixels"] for mask in masks]
    mask_count = len(mask_pixel_indices)

    # Flattens mask pixel indices into a contiguous array with offset pointers.
    # This format avoids Numba's tuple/list size limitations and enables efficient parallel processing.
    mask_sizes = np.array([len(indices) for indices in mask_pixel_indices], dtype=np.uint64)
    mask_offsets = np.zeros(mask_count + 1, dtype=np.uint64)
    mask_offsets[1:] = np.cumsum(mask_sizes)

    # Concatenates all pixel indices into flat array
    all_pixel_indices = np.concatenate(mask_pixel_indices)

    # Finds overlapping pixels using numba
    overlapping_pixels = _find_overlapping_pixels(all_pixel_indices)

    # Creates flattened overlap array using numba
    flat_overlap = _create_overlap_arrays(all_pixel_indices, overlapping_pixels)

    # Assigns overlapping pixel arrays to mask dictionaries by extracting slices from flat result
    for mask_index, mask in enumerate(masks):
        start = mask_offsets[mask_index]
        end = mask_offsets[mask_index + 1]
        mask["overlap_mask"] = flat_overlap[start:end]

    return masks
