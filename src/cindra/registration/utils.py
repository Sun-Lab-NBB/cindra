"""Provides utility functions for image registration and motion correction."""

from __future__ import annotations

from typing import TYPE_CHECKING
from functools import lru_cache

from numba import vectorize  # type: ignore[import-untyped]
import numpy as np
from numpy.fft import ifftshift
from scipy.fft import (
    rfft2 as scipy_rfft2,
    irfft2 as scipy_irfft2,
)
from scipy.ndimage import gaussian_filter1d
from ataraxis_base_utilities import console

if TYPE_CHECKING:
    from numpy.typing import NDArray

NORMALIZATION_EPSILON: float = 1e-5
"""The small epsilon value for numerical stability when normalizing by magnitude."""


def apply_phase_correlation(
    frames: NDArray[np.float32],
    kernel: NDArray[np.complex64],
    workers: int,
) -> NDArray[np.float32]:
    """Applies phase correlation between frames and reference kernel.

    Computes normalized cross-correlation in the frequency domain for motion estimation. Uses real FFT
    for efficiency.

    Args:
        frames: The frames to correlate with shape (num_frames, height, width).
        kernel: The reference kernel from compute_reference_fft.
        workers: The number of parallel workers for FFT computation. Use -1 for all available cores.

    Returns:
        The correlation maps with the same shape as input frames.
    """
    # Stores original width for inverse FFT reconstruction.
    width = frames.shape[-1]

    # Transforms frames to frequency domain.
    frames_fft = scipy_rfft2(frames, axes=(-2, -1), workers=workers)

    # Normalizes by magnitude to extract phase-only information. This makes the correlation robust to
    # intensity variations between frames. Epsilon prevents division by zero at DC component.
    frames_fft /= NORMALIZATION_EPSILON + np.abs(frames_fft)

    # Multiplies by conjugate of reference spectrum. In frequency domain, this computes cross-correlation.
    frames_fft *= kernel

    # Transforms back to spatial domain to get correlation surface. The peak location indicates the offset.
    return scipy_irfft2(frames_fft, s=(frames.shape[-2], width), axes=(-2, -1), workers=workers).astype(
        np.float32, copy=False
    )


@vectorize(
    ["float32(float32, float32, float32)"],
    nopython=True,
    target="parallel",
    cache=True,
)
def apply_mask(
    frames: NDArray[np.float32],
    mask: NDArray[np.float32],
    offset: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Applies spatial mask to frame data.

    Computes (frames * mask + offset) to apply edge tapering and mean offset correction. Uses parallel
    execution for performance on large arrays.

    Args:
        frames: The input frame data with shape (num_frames, height, width).
        mask: The multiplicative taper mask with shape (height, width), typically from compute_spatial_taper_mask.
        offset: The additive offset with shape (height, width), typically reference_image.mean() * (1 - mask).

    Returns:
        The masked frames with the same shape as input.
    """
    return frames * mask + offset


def combine_rigid_offsets(
    offset_list: list[tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]],
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]:
    """Combines rigid registration offsets from multiple processing batches.

    Rigid offsets are 1D arrays with one integer pixel offset per frame, so horizontal stacking
    concatenates all frames into a single array.

    Args:
        offset_list: A list of tuples containing (y_offsets, x_offsets, correlation_values) for each batch.

    Returns:
        A tuple of (y_offsets, x_offsets, correlation_values) arrays combined from all batches.
    """
    # Transposes list of tuples into separate tuples for each offset type.
    y_offsets, x_offsets, correlations = zip(*offset_list, strict=True)
    return np.hstack(y_offsets), np.hstack(x_offsets), np.hstack(correlations)


def combine_nonrigid_offsets(
    offset_list: list[tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Combines nonrigid registration offsets from multiple processing batches.

    Nonrigid offsets are 2D arrays with subpixel offsets per block per frame, so vertical stacking
    preserves the block structure across batches.

    Args:
        offset_list: A list of tuples containing (y_offsets, x_offsets, correlation_values) for each batch.

    Returns:
        A tuple of (y_offsets, x_offsets, correlation_values) arrays combined from all batches.
    """
    # Transposes list of tuples into separate tuples for each offset type.
    y_offsets, x_offsets, correlations = zip(*offset_list, strict=True)
    return np.vstack(y_offsets), np.vstack(x_offsets), np.vstack(correlations)


@lru_cache(maxsize=5)
def compute_gaussian_frequency_filter(sigma: float, height: int, width: int) -> NDArray[np.complex64]:
    """Creates a Gaussian smoothing filter in the Fourier domain using real FFT.

    Constructs a 2D Gaussian kernel in spatial domain, then transforms it to frequency domain for use with phase
    correlation. Results are cached since the same filter is reused across all frames in a recording.

    Args:
        sigma: The standard deviation of the Gaussian kernel in pixels.
        height: The height of the frames or images to be filtered, in pixels.
        width: The width of the frames or images to be filtered, in pixels.

    Returns:
        The smoothing filter in the Fourier domain with shape (height, width // 2 + 1) for real FFT compatibility.
    """
    # Creates grids of distances from center for a spatial-domain kernel.
    column_distances, row_distances = _mean_centered_meshgrid(height=height, width=width)

    # Computes separable 1D Gaussians along each axis, then combines into 2D kernel.
    gaussian_column = np.exp(-np.square(column_distances / sigma) / 2)
    gaussian_row = np.exp(-np.square(row_distances / sigma) / 2)
    gaussian_kernel = gaussian_row * gaussian_column

    # Normalizes kernel to unit sum and transforms to frequency domain.
    gaussian_kernel /= gaussian_kernel.sum()
    return scipy_rfft2(ifftshift(gaussian_kernel), axes=(-2, -1)).astype(np.complex64)


def apply_temporal_smoothing(frames: NDArray[np.float32], sigma: float) -> NDArray[np.float32]:
    """Applies Gaussian filtering along the temporal (first) axis.

    Args:
        frames: The frames with shape (num_frames, height, width) to be smoothed.
        sigma: The standard deviation of the Gaussian kernel.

    Returns:
        The temporally smoothed frames with the same shape as input.
    """
    return gaussian_filter1d(input=frames, sigma=sigma, axis=0).astype(np.float32)


def apply_spatial_smoothing(data: NDArray[np.float32], window: int) -> NDArray[np.float32]:
    """Applies spatial smoothing using cumulative sum with a sliding window.

    Args:
        data: Recording frames with shape (num_frames, height, width) or a single image with shape (height, width).
        window: The window size for smoothing. Must be an even integer.

    Returns:
        The spatially smoothed data with the same shape as input.

    Raises:
        ValueError: If the window size is not an even integer.
    """
    if window and window % 2:
        message = f"Unable to apply spatial smoothing. Filter window must be an even integer, but got {window}."
        console.error(message=message, error=ValueError)

    # Promotes 2D input to 3D for uniform processing.
    if data.ndim == 2:  # noqa: PLR2004
        data = data[np.newaxis, :, :]

    # Pads spatial dimensions to handle window edges. Zero padding ensures border pixels average over partial windows.
    half_pad = window // 2
    data_padded = np.pad(
        array=data,
        pad_width=((0, 0), (half_pad, half_pad), (half_pad, half_pad)),
        mode="constant",
        constant_values=0,
    )

    # Computes integral image (summed area table) via cumulative sums along height then width.
    # Specifies float32 dtype on both cumsum calls to avoid intermediate float64 arrays.
    data_summed = data_padded.cumsum(axis=1, dtype=np.float32).cumsum(axis=2, dtype=np.float32)

    # Extracts box sums using integral image differences. For each pixel, computes sum of (window x window) region
    # centered on that pixel, then normalizes to get the mean.
    data_summed = data_summed[:, window:, :] - data_summed[:, :-window, :]
    data_summed = data_summed[:, :, window:] - data_summed[:, :, :-window]
    data_summed /= window**2

    # Squeezes back to 2D if input was 2D.
    result: NDArray[np.float32] = data_summed.squeeze()
    return result


def apply_spatial_high_pass(data: NDArray[np.float32], window: int) -> NDArray[np.float32]:
    """Applies a spatial high-pass filter using the sliding window method.

    Args:
        data: Recording frames with shape (num_frames, height, width) or a single image with shape (height, width).
        window: The window size for the low-pass component to subtract.

    Returns:
        The high-pass filtered data with the same shape as input.
    """
    # Promotes 2D input to 3D for uniform processing.
    if data.ndim == 2:  # noqa: PLR2004
        data = data[np.newaxis, :, :]

    # Retrieves cached normalization weights that correct for zero-padding at borders.
    normalization = _get_normalization_weights(height=data.shape[1], width=data.shape[2], window=window)

    # Subtracts normalized low-pass (local mean) from original to extract high-frequency components.
    # Uses in-place division to avoid creating an intermediate array.
    low_pass = apply_spatial_smoothing(data=data, window=window)
    low_pass /= normalization
    data_filtered = data - low_pass

    # Squeezes back to 2D if input was 2D.
    return data_filtered.squeeze()


def compute_reference_fft(reference_image: NDArray[np.float32]) -> NDArray[np.complex64]:
    """Computes the complex conjugate of the real FFT for a reference image.

    The complex conjugate is taken because phase correlation requires multiplication by the conjugate of the reference
    spectrum. No padding is applied to ensure dimension compatibility with frame FFTs computed without padding.

    Args:
        reference_image: The 2D reference image with shape (height, width).

    Returns:
        The complex conjugate of the FFT with shape (height, width // 2 + 1).
    """
    return np.conj(scipy_rfft2(reference_image, axes=(-2, -1))).astype(np.complex64)


@lru_cache(maxsize=5)
def compute_upsampling_kernel(padding: int, subpixel: int = 10) -> tuple[NDArray[np.float32], int]:
    """Computes the upsampling matrix for subpixel offset estimation using Gaussian RBF interpolation.

    Builds a kernel that maps low-resolution correlation peaks to a high-resolution grid for precise subpixel
    offset detection. Uses the RBF interpolation formula: inv(K(low, low)) @ K(low, high). Results are cached
    since the same kernel is reused across all frames.

    Args:
        padding: The half-width of the correlation peak region to upsample, in pixels.
        subpixel: The subpixel resolution factor (e.g., 10 means 0.1 pixel precision).

    Returns:
        A tuple of (kernel_matrix, num_upsampled_points) where kernel_matrix is the upsampling transformation
        matrix and num_upsampled_points is the number of points in the upsampled grid.
    """
    # Creates low-resolution grid centered at zero with integer spacing.
    low_resolution_coordinates = np.arange(-padding, padding + 1, dtype=np.float64)

    # Creates high-resolution grid with subpixel spacing. The +0.001 ensures the endpoint is included
    # since arange excludes the stop value.
    high_resolution_coordinates = np.arange(-padding, padding + 0.001, 1.0 / subpixel, dtype=np.float64)
    num_upsampled = high_resolution_coordinates.shape[0]

    # Computes RBF interpolation kernel: inv(K(source, source)) @ K(source, target).
    # Uses float64 internally for numerical stability during matrix inversion.
    source_weights = _compute_gaussian_rbf_weights(
        source_coordinates=low_resolution_coordinates, target_coordinates=low_resolution_coordinates
    )
    interpolation_weights = _compute_gaussian_rbf_weights(
        source_coordinates=low_resolution_coordinates, target_coordinates=high_resolution_coordinates
    )
    kernel_matrix = np.linalg.inv(source_weights) @ interpolation_weights

    # Casts to float32 since precision is no longer critical after inversion.
    return kernel_matrix.astype(np.float32), num_upsampled


def _mean_centered_meshgrid(height: int, width: int) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Creates a mean-centered distance meshgrid of the specified dimensions.

    Each coordinate value represents the absolute distance from the center of that axis. Used internally
    for creating spatial taper masks and Gaussian frequency filters.

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


def _compute_gaussian_rbf_weights(
    source_coordinates: NDArray[np.float64],
    target_coordinates: NDArray[np.float64],
    sigma: float = 0.85,
) -> NDArray[np.float64]:
    """Computes Gaussian radial basis function weights between 2D point grids.

    Creates 2D point grids from the Cartesian product of each 1D coordinate array with itself, then computes
    pairwise Gaussian weights between all source and target grid points, which is used for RBF interpolation.

    Notes:
        Radial Basis Function (RBF) interpolation uses basis functions that depend only on the distance from a
        center point. The Gaussian RBF, exp(-r^2 / 2*sigma^2), produces smooth interpolations where each source point
        contributes to the output based on its distance from the target. The interpolation weights are computed
        as inv(K(source, source)) @ K(source, target), where K is the Gaussian kernel matrix.

    Args:
        source_coordinates: The 1D array of source coordinates. The 2D source grid has n² points where n is
            the array length.
        target_coordinates: The 1D array of target coordinates. The 2D target grid has m² points where m is
            the array length.
        sigma: The Gaussian kernel bandwidth controlling interpolation smoothness. Smaller values produce
            sharper interpolation, larger values produce smoother results.

    Returns:
        The Gaussian RBF weight matrix with shape (n², m²). Float64 precision is used because this matrix
        is inverted during RBF interpolation, and matrix inversion is numerically sensitive.
    """
    # Creates 2D grids from Cartesian product of coordinates with themselves.
    source_grid_x, source_grid_y = np.meshgrid(source_coordinates, source_coordinates)
    target_grid_x, target_grid_y = np.meshgrid(target_coordinates, target_coordinates)

    # Flattens grids and computes pairwise coordinate differences between all source and target points.
    delta_x = source_grid_x.reshape(-1, 1) - target_grid_x.reshape(1, -1)
    delta_y = source_grid_y.reshape(-1, 1) - target_grid_y.reshape(1, -1)

    # Computes Gaussian weights based on squared Euclidean distance.
    return np.exp(-(delta_x**2 + delta_y**2) / (2 * sigma**2))


@lru_cache(maxsize=5)
def _get_normalization_weights(height: int, width: int, window: int) -> NDArray[np.float32]:
    """Computes cached normalization weights for spatial high-pass filtering.

    The weights correct for zero-padding at borders by computing how many valid pixels contribute to each window.
    Since this only depends on dimensions and window size, results are cached to avoid redundant computation.

    Args:
        height: The height of the frames or images to be filtered, in pixels.
        width: The width of the frames or images to be filtered, in pixels.
        window: The smoothing window size.

    Returns:
        The normalization weights with shape (height, width).
    """
    ones_array = np.ones((1, height, width), dtype=np.float32)
    return apply_spatial_smoothing(data=ones_array, window=window)
