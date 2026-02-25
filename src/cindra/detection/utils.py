"""Provides utility functions for filtering and downsampling data arrays during ROI detection."""

from __future__ import annotations

from typing import TYPE_CHECKING
from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter

if TYPE_CHECKING:
    from numpy.typing import NDArray

_MINIMUM_STANDARD_DEVIATION: float = 1e-10
"""The minimum standard deviation threshold to prevent division by zero in downstream processing."""

_GAUSSIAN_KERNEL_THRESHOLD: int = 10
"""The kernel size threshold for selecting Gaussian vs rolling mean high-pass filter."""


def apply_temporal_high_pass_filter(frames: NDArray[np.float32], kernel_size: int) -> None:
    """Applies a temporal high-pass filter to the frames in-place, automatically selecting the optimal algorithm.

    Notes:
        For kernel sizes less than 10 frames, a Gaussian filter is used. For larger sizes, a rolling mean filter
        is used instead because the Gaussian implementation becomes computationally expensive.

    Args:
        frames: The input frame array with shape (num_frames, height, width). Modified in-place.
        kernel_size: The filter kernel size in frames.
    """
    if kernel_size < _GAUSSIAN_KERNEL_THRESHOLD:
        _apply_gaussian_high_pass(frames=frames, kernel_size=kernel_size)
    else:
        _apply_rolling_mean_high_pass(frames=frames, kernel_size=kernel_size)


def compute_temporal_standard_deviation(frames: NDArray[np.float32]) -> NDArray[np.float32]:
    """Computes the standard deviation of frame-to-frame pixel differences across time.

    Notes:
        The result represents the temporal variability of each pixel, which is useful for identifying active regions
        in calcium imaging data.

    Args:
        frames: The input frame array with shape (num_frames, height, width).

    Returns:
        An array with shape (height, width) containing the standard deviation of frame differences for each pixel.
        Values are clipped to a minimum threshold to avoid division by zero in downstream processing.
    """
    frame_differences = np.diff(frames, axis=0)
    result: NDArray[np.float32] = np.maximum(
        _MINIMUM_STANDARD_DEVIATION, np.sqrt((frame_differences**2).sum(axis=0) / frames.shape[0])
    )
    return result


def downsample(data: NDArray[np.float32], taper_edge: bool = True) -> NDArray[np.float32]:
    """Downsamples a 3D array by a factor of 2 in both spatial dimensions.

    Notes:
        Adjacent elements are averaged to produce each output element. When the input dimensions are odd, the final
        row or column can either be tapered (multiplied by 0.5) or preserved at full intensity.

    Args:
        data: The input array with shape (depth, height, width). This can be movie frames with shape
            (num_frames, height, width) or spatial coordinate grids with shape (num_axes, height, width).
        taper_edge: Determines whether to taper edge elements when dimensions are odd. If True, edge elements are
            multiplied by 0.5 to maintain consistent intensity scaling. If False, edge elements retain their
            original values.

    Returns:
        A downsampled array with shape (depth, ceil(height/2), ceil(width/2)).
    """
    # Precomputes the downsampling parameters and the output array.
    depth, height, width = data.shape
    out_height = (height + 1) // 2
    out_width = (width + 1) // 2
    even_height = (height // 2) * 2
    even_width = (width // 2) * 2
    taper_factor = 0.5 if taper_edge else 1.0

    downsampled = np.zeros((depth, out_height, out_width), dtype=np.float32)

    # Processes the main 2x2 blocks using reshape (creates a view, not a copy).
    if even_height > 0 and even_width > 0:
        block = data[:, :even_height, :even_width].reshape(depth, even_height // 2, 2, even_width // 2, 2)
        downsampled[:, : even_height // 2, : even_width // 2] = block.mean(axis=(2, 4))

    # Handles the right edge column when width is odd.
    if width % 2 == 1 and even_height > 0:
        right_column = data[:, :even_height, -1].reshape(depth, even_height // 2, 2).mean(axis=2)
        downsampled[:, : even_height // 2, -1] = right_column * taper_factor

    # Handles the bottom edge row when height is odd.
    if height % 2 == 1 and even_width > 0:
        bottom_row = data[:, -1, :even_width].reshape(depth, even_width // 2, 2).mean(axis=2)
        downsampled[:, -1, : even_width // 2] = bottom_row * taper_factor

    # Handles the bottom-right corner when both dimensions are odd.
    if height % 2 == 1 and width % 2 == 1:
        downsampled[:, -1, -1] = data[:, -1, -1] * taper_factor * taper_factor

    return downsampled


def compute_thresholded_variance(frames: NDArray[np.float32], intensity_threshold: float) -> NDArray[np.float32]:
    """Computes the thresholded standard deviation of pixel intensities across frames.

    Notes:
        This function computes a root-sum-of-squares measure for pixels exceeding the intensity threshold. Uses
        np.where with in-place squaring to avoid allocating separate boolean mask and squared-frames temporaries.

    Args:
        frames: The input frame array with shape (num_frames, height, width).
        intensity_threshold: The minimum pixel intensity required for inclusion in the standard deviation
            calculation. Pixels below this threshold contribute zero to the sum.

    Returns:
        An array with shape (height, width) containing the thresholded standard deviation for each pixel.
    """
    # Zeros out below-threshold values in a single allocation, then squares in-place to avoid a second temporary.
    thresholded = np.where(frames > intensity_threshold, frames, np.float32(0.0))
    thresholded *= thresholded
    result: NDArray[np.float32] = np.sqrt(thresholded.sum(axis=0))
    return result


@lru_cache(maxsize=5)
def compute_spatial_taper_mask(sigma: float, height: int, width: int) -> NDArray[np.float32]:
    """Creates a spatial taper mask with sigmoid falloff at the edges.

    The mask smoothly transitions from 1.0 in the center to ~0 at the edges, suppressing border artifacts
    during phase correlation. The transition follows a sigmoid curve controlled by sigma. Results are cached
    since the same mask is reused across all frames in a recording.

    Args:
        sigma: Controls the steepness of the edge falloff. Larger values produce a more gradual taper.
        height: The height of the frames to be processed with the generated taper mask, in pixels.
        width: The width of the frames to be processed with the generated taper mask, in pixels.

    Returns:
        The multiplicative taper mask with shape (height, width), values in range [0, 1].
    """
    # Creates grids of absolute distances from center for each axis.
    column_distances, row_distances = _mean_centered_meshgrid(height=height, width=width)

    # Computes where taper begins: 2*sigma pixels inward from the edge. This ensures the sigmoid reaches
    # ~0.12 at the edge (when distance equals half-width).
    taper_start_row = np.float32(((height - 1) / 2) - 2 * sigma)
    taper_start_column = np.float32(((width - 1) / 2) - 2 * sigma)

    # Applies sigmoid function: 1.0 at center, 0.5 at taper_start, approaches 0 at edges.
    sigma_f32 = np.float32(sigma)
    row_taper = np.float32(1.0) / (np.float32(1.0) + np.exp((row_distances - taper_start_row) / sigma_f32))
    col_taper = np.float32(1.0) / (np.float32(1.0) + np.exp((column_distances - taper_start_column) / sigma_f32))

    # Combines row and column tapers multiplicatively for 2D falloff.
    taper_mask: NDArray[np.float32] = row_taper * col_taper
    return taper_mask


@lru_cache(maxsize=5)
def compute_block_smoothing_kernel(x_block_count: int, y_block_count: int) -> NDArray[np.float32]:
    """Computes a normalized Gaussian kernel matrix for smoothing nonrigid block shifts.

    Creates a kernel that weights neighboring blocks based on their spatial distance, used to enforce smoothness
    constraints in nonrigid registration. Results are cached since block counts don't change during a recording.

    Args:
        x_block_count: Number of blocks along the x-axis.
        y_block_count: Number of blocks along the y-axis.

    Returns:
        The row-normalized Gaussian kernel matrix with shape (num_blocks, num_blocks).
    """
    # Creates 2D coordinate grids from block indices.
    grid_y, grid_x = np.meshgrid(
        np.arange(x_block_count, dtype=np.float32),
        np.arange(y_block_count, dtype=np.float32),
    )

    # Reshapes to row vectors for pairwise distance computation via broadcasting.
    grid_y = grid_y.reshape(1, -1)
    grid_x = grid_x.reshape(1, -1)

    # Computes pairwise Gaussian weights based on squared Euclidean distance.
    kernel_matrix = np.exp(-((grid_y - grid_y.T) ** 2 + (grid_x - grid_x.T) ** 2), dtype=np.float32)

    # Normalizes each column to sum to 1 for weighted averaging.
    kernel_matrix /= kernel_matrix.sum(axis=0)
    return kernel_matrix


def compute_registration_blocks(
    height: int,
    width: int,
    block_size: tuple[int, int] = (128, 128),
) -> tuple[list[NDArray[np.int32]], list[NDArray[np.int32]], tuple[int, int], tuple[int, int], NDArray[np.float32]]:
    """Computes overlapping blocks for nonrigid registration.

    Divides the field of view into overlapping blocks that are registered independently. The blocks
    are arranged in a regular grid with positions computed to provide approximately 50% overlap
    between adjacent blocks.

    Args:
        height: The imaging field height in pixels.
        width: The imaging field width in pixels.
        block_size: The target block size as (height, width) in pixels. Actual block sizes may differ
            if the image dimensions are smaller than the requested block size.

    Returns:
        A tuple of (y_blocks, x_blocks, block_counts, actual_block_size, smoothing_kernel). The
        y_blocks and x_blocks are lists of 2-element arrays specifying the start and end indices for
        each block. The block_counts tuple gives (y_count, x_count). The actual_block_size tuple gives
        the final block dimensions. The smoothing_kernel is used for interpolating block shifts.
    """
    # Computes block dimensions and counts for each axis. If the requested block size exceeds the image
    # dimension, uses the full dimension as a single block. Otherwise, the 1.5x multiplier produces
    # approximately 50% overlap between adjacent blocks.
    if block_size[0] >= height:
        block_size_y, y_block_count = height, 1
    else:
        block_size_y, y_block_count = block_size[0], int(np.ceil(1.5 * height / block_size[0]))

    if block_size[1] >= width:
        block_size_x, x_block_count = width, 1
    else:
        block_size_x, x_block_count = block_size[1], int(np.ceil(1.5 * width / block_size[1]))

    actual_block_size = (block_size_y, block_size_x)

    # Computes evenly-spaced block start positions spanning from 0 to the last valid position.
    y_starts = np.linspace(0, height - block_size_y, y_block_count).astype(np.int32)
    x_starts = np.linspace(0, width - block_size_x, x_block_count).astype(np.int32)

    # Creates block boundary arrays in row-major order (all x positions for each y position).
    y_blocks = [
        np.array([y_starts[y_index], y_starts[y_index] + block_size_y], dtype=np.int32)
        for y_index in range(y_block_count)
        for _ in range(x_block_count)
    ]
    x_blocks = [
        np.array([x_starts[x_index], x_starts[x_index] + block_size_x], dtype=np.int32)
        for _ in range(y_block_count)
        for x_index in range(x_block_count)
    ]

    # Computes the smoothing kernel used for SNR-based adaptive smoothing during shift estimation.
    smoothing_kernel = compute_block_smoothing_kernel(
        x_block_count=x_block_count,
        y_block_count=y_block_count,
    ).T

    return y_blocks, x_blocks, (y_block_count, x_block_count), actual_block_size, smoothing_kernel


def _apply_gaussian_high_pass(frames: NDArray[np.float32], kernel_size: int) -> None:
    """Applies a high-pass filter to the input frames in-place using a Gaussian kernel.

    Args:
        frames: The input frame array with shape (num_frames, height, width). Modified in-place.
        kernel_size: The Gaussian kernel size in frames.
    """
    frames -= gaussian_filter(input=frames, sigma=[kernel_size, 0, 0])


def _apply_rolling_mean_high_pass(frames: NDArray[np.float32], kernel_size: int) -> None:
    """Applies a high-pass filter to the input frames in-place using a non-overlapping rolling mean kernel.

    Notes:
        This method is more efficient than Gaussian filtering for large kernel sizes. The filter subtracts the mean
        of each non-overlapping temporal window from all frames within that window.

    Args:
        frames: The input frame array with shape (num_frames, height, width). Modified in-place.
        kernel_size: The rolling window size in frames.
    """
    # Determines the number of complete windows based on the frame count.
    num_frames, height, width = frames.shape
    num_complete_windows = num_frames // kernel_size

    # Reshapes to (num_windows, kernel_size, height, width). This creates a view, not a copy.
    if num_complete_windows > 0:
        # Applies the filter to all windows at once.
        complete = frames[: num_complete_windows * kernel_size].reshape(
            num_complete_windows, kernel_size, height, width
        )
        complete -= complete.mean(axis=1, keepdims=True)

    # Handles remaining frames that don't fill a complete window.
    remainder = num_frames % kernel_size
    if remainder > 0:
        frames[-remainder:] -= frames[-remainder:].mean(axis=0)


def _mean_centered_meshgrid(height: int, width: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Creates a mean-centered distance meshgrid of the specified dimensions.

    Each coordinate value represents the absolute distance from the center of that axis. Used internally
    for creating spatial taper masks.

    Args:
        height: The height of the frames or images to generate the meshgrid for, in pixels.
        width: The width of the frames or images to generate the meshgrid for, in pixels.

    Returns:
        A tuple of (column_distances, row_distances) arrays with shape (height, width), where each value
        represents the absolute distance from the center along that axis.
    """
    # Computes absolute distances from center for each axis. For arange(0, n), mean is (n-1)/2. Casts centers to
    # float32 to prevent promotion of the entire distance arrays to float64.
    row_center = np.float32((height - 1) / 2)
    column_center = np.float32((width - 1) / 2)
    row_distances_1d = np.abs(np.arange(height, dtype=np.float32) - row_center)
    column_distances_1d = np.abs(np.arange(width, dtype=np.float32) - column_center)

    # Expands 1D distances into 2D grids. Meshgrid returns (column-varying, row-varying) arrays.
    column_distances, row_distances = np.meshgrid(column_distances_1d, row_distances_1d)

    return column_distances, row_distances
