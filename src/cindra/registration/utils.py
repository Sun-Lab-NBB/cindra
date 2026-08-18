"""Provides utility functions for image registration and motion correction."""

from __future__ import annotations

from typing import TYPE_CHECKING
from functools import lru_cache

from numba import njit, prange
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
    frames_fft = scipy_rfft2(x=frames, axes=(-2, -1), workers=workers)

    # Normalizes by magnitude to extract phase-only information. This makes the correlation robust to
    # intensity variations between frames. Epsilon prevents division by zero at DC component. Folding the epsilon
    # into the magnitude buffer keeps the normalization down to a single full spectra temporary.
    magnitude = np.abs(frames_fft)
    magnitude += NORMALIZATION_EPSILON
    frames_fft /= magnitude

    # Multiplies by conjugate of reference spectrum. In frequency domain, this computes cross-correlation.
    frames_fft *= kernel

    # Transforms back to spatial domain to get correlation surface. The peak location indicates the offset.
    return scipy_irfft2(x=frames_fft, s=(frames.shape[-2], width), axes=(-2, -1), workers=workers).astype(
        np.float32, copy=False
    )


@njit(parallel=True, cache=True)
def apply_mask(  # pragma: no cover
    frames: NDArray[np.float32],
    mask: NDArray[np.float32],
    offset: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Applies spatial mask to frame data.

    Computes (frames * mask + offset) to apply edge tapering and mean offset correction, parallelized over the leading
    frame axis.

    Notes:
        Callers pass float32 arrays. The output buffer takes its dtype from 'frames', so an integer or float64 input
        propagates that width into the result and through the phase correlation that consumes it. Every caller reads
        its frames from a float32 allocation or an explicit cast, which is what keeps the pipeline at single precision.

        The 'mask' and 'offset' arrays match the shape of a single frame, so this kernel covers both the rigid case,
        where a two-dimensional mask meets three-dimensional frames, and the nonrigid case, where a three-dimensional
        per-block mask meets four-dimensional extracted blocks.

    Args:
        frames: The input frame data with shape (num_frames, height, width) or (num_frames, num_blocks, height, width).
        mask: The multiplicative taper mask shaped like one frame, typically from compute_spatial_taper_mask.
        offset: The additive offset shaped like one frame, typically reference_image.mean() * (1 - mask).

    Returns:
        The masked frames with the same shape and dtype as the input frames.
    """
    masked_frames = np.empty_like(frames)
    for frame_index in prange(frames.shape[0]):
        masked_frames[frame_index] = frames[frame_index] * mask + offset
    return masked_frames


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
    # Measures each axis from the sample the ifftshift below moves to the origin, which is what makes the filter
    # zero-phase. An even-length axis carries its mean between two samples, so a kernel built around that mean lands
    # half a pixel off the origin and shifts every correlation peak the filter smooths by that half pixel.
    row_distances = np.abs(np.arange(height, dtype=np.float32) - np.float32(height // 2))
    column_distances = np.abs(np.arange(width, dtype=np.float32) - np.float32(width // 2))

    # Computes separable 1D Gaussians along each axis, then combines into 2D kernel.
    gaussian_column = np.exp(-np.square(column_distances / sigma) / 2)
    gaussian_row = np.exp(-np.square(row_distances / sigma) / 2)
    gaussian_kernel = np.outer(a=gaussian_row, b=gaussian_column)

    # Normalizes kernel to unit sum and transforms to frequency domain.
    gaussian_kernel /= gaussian_kernel.sum()
    return scipy_rfft2(x=ifftshift(x=gaussian_kernel), axes=(-2, -1)).astype(np.complex64)


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
        window: The window size for smoothing. Must be a positive even integer.

    Returns:
        The spatially smoothed data. A 2D input returns a 2D array, and a 3D input returns a 3D array of the
        same shape.

    Raises:
        ValueError: If the window size is not a positive even integer.
    """
    # Rejects a zero or negative window here rather than letting it reach the integral-image differencing below, where
    # the ':-window' slice bound degenerates to the empty prefix and the box normalization divides by zero.
    if window <= 0 or window % 2:
        message = f"Unable to apply spatial smoothing. Filter window must be a positive even integer, but got {window}."
        console.error(message=message, error=ValueError)

    # Promotes 2D input to 3D for uniform processing. The flag records the promotion so that the output drops only an
    # axis this function added, leaving a genuine single-frame stack three-dimensional.
    promoted = data.ndim == 2  # noqa: PLR2004
    if promoted:
        data = data[np.newaxis, :, :]

    # Pads spatial dimensions with zeros to handle window edges. Border pixels are summed over partial (zero-filled)
    # windows but still divided by the full window**2, so their means are under-estimated and corrected later via the
    # normalization weights in apply_spatial_high_pass.
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

    result: NDArray[np.float32] = data_summed[0] if promoted else data_summed
    return result


def apply_spatial_high_pass(data: NDArray[np.float32], window: int) -> NDArray[np.float32]:
    """Applies a spatial high-pass filter using the sliding window method.

    Args:
        data: Recording frames with shape (num_frames, height, width) or a single image with shape (height, width).
        window: The window size for the low-pass component to subtract.

    Returns:
        The high-pass filtered data. A 2D input returns a 2D array, and a 3D input returns a 3D array of the
        same shape.
    """
    # Promotes 2D input to 3D for uniform processing. The flag records the promotion so that the output drops only an
    # axis this function added, leaving a genuine single-frame stack three-dimensional.
    promoted = data.ndim == 2  # noqa: PLR2004
    if promoted:
        data = data[np.newaxis, :, :]

    # Retrieves cached normalization weights that correct for zero-padding at borders.
    normalization = _get_normalization_weights(height=data.shape[1], width=data.shape[2], window=window)

    # Subtracts normalized low-pass (local mean) from original to extract high-frequency components.
    # Uses in-place division to avoid creating an intermediate array.
    low_pass = apply_spatial_smoothing(data=data, window=window)
    low_pass /= normalization
    data_filtered = data - low_pass

    result: NDArray[np.float32] = data_filtered[0] if promoted else data_filtered
    return result


def compute_reference_fft(reference_image: NDArray[np.float32]) -> NDArray[np.complex64]:
    """Computes the complex conjugate of the real FFT for a reference image.

    The complex conjugate is taken because phase correlation requires multiplication by the conjugate of the reference
    spectrum. No padding is applied to ensure dimension compatibility with frame FFTs computed without padding.

    Args:
        reference_image: The 2D reference image with shape (height, width).

    Returns:
        The complex conjugate of the FFT with shape (height, width // 2 + 1).
    """
    return np.conj(scipy_rfft2(x=reference_image, axes=(-2, -1))).astype(np.complex64)


@lru_cache(maxsize=5)
def compute_upsampling_kernel(padding: int, subpixel: int = 10) -> tuple[NDArray[np.float32], int]:
    """Computes the upsampling matrix for subpixel offset estimation using Gaussian RBF interpolation.

    Builds a kernel that maps low-resolution correlation peaks to a high-resolution grid for precise subpixel
    offset detection. Uses the RBF interpolation formula: inv(K(low, low)) @ K(low, high). Results are cached
    since the same kernel is reused across all frames.

    Args:
        padding: The half-width of the correlation peak region to upsample, in pixels.
        subpixel: The subpixel resolution factor (e.g., 10 means 0.1 pixel precision). Defaults to 10.

    Returns:
        A tuple of (kernel_matrix, upsampled_point_count) where kernel_matrix is the upsampling transformation
        matrix and upsampled_point_count is the number of points in the upsampled grid.
    """
    # Creates low-resolution grid centered at zero with integer spacing.
    low_resolution_coordinates = np.arange(-padding, padding + 1, dtype=np.float64)

    # Creates high-resolution grid with subpixel spacing. The +0.001 ensures the endpoint is included
    # since arange excludes the stop value.
    high_resolution_coordinates = np.arange(-padding, padding + 0.001, 1.0 / subpixel, dtype=np.float64)
    upsampled_point_count = high_resolution_coordinates.shape[0]

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
    return kernel_matrix.astype(np.float32), upsampled_point_count


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
        sigma: The Gaussian kernel bandwidth controlling interpolation smoothness. Defaults to 0.85. Smaller values
            produce sharper interpolation, larger values produce smoother results.

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
    # Smooths a single-image batch and drops the batch axis, since the weights broadcast against both 2D images and
    # 3D frame stacks.
    ones_array = np.ones((1, height, width), dtype=np.float32)
    weights: NDArray[np.float32] = apply_spatial_smoothing(data=ones_array, window=window)[0]
    return weights
