"""Provides utility functions for filtering and downsampling data arrays during ROI detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import gaussian_filter

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Minimum standard deviation threshold to prevent division by zero in downstream processing.
_MINIMUM_STANDARD_DEVIATION: float = 1e-10

# Kernel size threshold for selecting Gaussian vs rolling mean high-pass filter.
_GAUSSIAN_KERNEL_THRESHOLD: int = 10


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
