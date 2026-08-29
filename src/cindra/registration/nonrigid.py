"""Provides the nonrigid (piecewise) registration algorithm for motion correction."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit, prange
import numpy as np
from scipy.fft import rfft2

from .utils import (
    NORMALIZATION_EPSILON,
    apply_mask,
    apply_phase_correlation,
    compute_upsampling_kernel,
    compute_gaussian_frequency_filter,
)
from ..detection import compute_spatial_taper_mask

if TYPE_CHECKING:
    from numpy.typing import NDArray

_SNR_EPSILON: float = 1e-10
"""The small epsilon value used to prevent division by zero in SNR calculations."""

_SUBPIXEL_FACTOR: int = 10
"""The upsampling factor for Gaussian RBF subpixel peak localization. A value of 10 provides 0.1 pixel precision."""

_UPSAMPLING_PADDING: int = 3
"""The half-width of the region around integer peaks used for RBF upsampling. A value of 3 uses a 7x7 region."""

_CORRELATION_BATCH_SIZE: int = 64
"""The maximum number of blocks to process in a single FFT batch during phase correlation. Limits memory usage."""


def compute_nonrigid_reference_data(
    reference_image: NDArray[np.float32],
    taper_slope: float,
    smoothing_sigma: float,
    y_blocks: list[NDArray[np.int32]],
    x_blocks: list[NDArray[np.int32]],
    workers: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.complex64]]:
    """Computes edge taper masks and FFT reference kernel for nonrigid phase correlation.

    Prepares the reference data needed for nonrigid registration by extracting each block from the reference image,
    computing per-block taper masks, and generating phase-normalized FFT kernels.

    Args:
        reference_image: The reference image with shape (height, width).
        taper_slope: Controls the steepness of the edge falloff for the spatial taper mask.
        smoothing_sigma: The standard deviation for Gaussian smoothing in the frequency domain. Twice this value also
            sets the falloff of the per-block spatial taper mask.
        y_blocks: The list of y-coordinate ranges for each block.
        x_blocks: The list of x-coordinate ranges for each block.
        workers: The number of threads each transform may use.

    Returns:
        A tuple of (taper_mask, mean_offset, reference_kernel). The taper_mask and mean_offset arrays have shape
        (block_count, block_height, block_width). The reference_kernel contains the phase-normalized FFT of each block
        with shape (block_count, block_height, real_fft_width).
    """
    block_count = len(y_blocks)
    block_height = y_blocks[0][1] - y_blocks[0][0]
    block_width = x_blocks[0][1] - x_blocks[0][0]

    # Real FFT output has shape (height, width // 2 + 1) for the frequency dimension.
    real_fft_width = block_width // 2 + 1
    gaussian_filter = compute_gaussian_frequency_filter(
        sigma=smoothing_sigma, height=block_height, width=block_width, workers=workers
    )
    reference_kernel = np.empty((block_count, block_height, real_fft_width), dtype=np.complex64)

    global_taper = compute_spatial_taper_mask(
        sigma=taper_slope,
        height=reference_image.shape[0],
        width=reference_image.shape[1],
    )

    # Computes the block-level taper mask used for each extracted block. compute_spatial_taper_mask returns float32,
    # and np.tile preserves dtype, so no cast is needed.
    block_taper = compute_spatial_taper_mask(sigma=2 * smoothing_sigma, height=block_height, width=block_width)
    taper_mask = np.tile(A=block_taper, reps=(block_count, 1, 1))
    mean_offset = np.empty((block_count, block_height, block_width), dtype=np.float32)

    for block_index, (y_range, x_range) in enumerate(zip(y_blocks, x_blocks, strict=True)):
        reference_block = reference_image[y_range[0] : y_range[1], x_range[0] : x_range[1]]

        taper_mask[block_index] *= global_taper[y_range[0] : y_range[1], x_range[0] : x_range[1]]
        mean_offset[block_index] = reference_block.mean() * (np.float32(1.0) - taper_mask[block_index])

        # Computes the phase-normalized FFT kernel with Gaussian smoothing.
        block_fft = np.conj(rfft2(reference_block, workers=workers))
        block_fft /= NORMALIZATION_EPSILON + np.absolute(block_fft)
        block_fft *= gaussian_filter
        reference_kernel[block_index] = block_fft

    return taper_mask, mean_offset, reference_kernel


def compute_nonrigid_offsets(
    frames: NDArray[np.float32],
    taper_mask: NDArray[np.float32],
    mean_offset: NDArray[np.float32],
    reference_kernel: NDArray[np.complex64],
    snr_threshold: float,
    smoothing_kernel: NDArray[np.float32],
    x_blocks: list[NDArray[np.int32]],
    y_blocks: list[NDArray[np.int32]],
    maximum_offset: float,
    workers: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Computes nonrigid offsets using block-wise phase correlation.

    Estimates per-block (y, x) subpixel offsets by computing phase correlation between each frame block and the
    corresponding reference kernel. Applies adaptive smoothing based on correlation SNR to improve reliability in
    low-quality regions. Subpixel precision is determined by the module constants _SUBPIXEL_FACTOR (0.1 pixel) and
    _UPSAMPLING_PADDING (7x7 fitting region).

    Args:
        frames: The frame data with shape (frame_count, height, width) to be registered.
        taper_mask: The edge taper mask with shape (block_count, block_height, block_width) from
            compute_nonrigid_reference_data. Suppresses edge artifacts during phase correlation.
        mean_offset: The mean intensity offset with shape (block_count, block_height, block_width) from
            compute_nonrigid_reference_data. Fills tapered regions with uniform intensity.
        reference_kernel: The phase-normalized FFT kernel with shape (block_count, block_height, real_fft_width)
            from compute_nonrigid_reference_data. Used for cross-correlation with frame blocks.
        snr_threshold: The SNR threshold below which additional smoothing is applied to correlation peaks.
            Higher values apply more smoothing. Typical values range from 1.0 to 1.5.
        smoothing_kernel: The block smoothing kernel from compute_registration_blocks. Used for SNR-based
            adaptive smoothing of correlation peaks across neighboring blocks.
        x_blocks: The list of x-coordinate ranges for each block from compute_registration_blocks.
        y_blocks: The list of y-coordinate ranges for each block from compute_registration_blocks.
        maximum_offset: The maximum allowed offset in pixels. Constrains the correlation search window.
        workers: The number of parallel workers for FFT computation. Use -1 for all available cores.

    Returns:
        A tuple of (y_offsets, x_offsets, correlation_maxima) arrays with shape (frame_count, block_count).
        The offsets have subpixel precision determined by _SUBPIXEL_FACTOR.
    """
    upsampling_kernel, upsampled_size = compute_upsampling_kernel(padding=_UPSAMPLING_PADDING)

    frame_count = frames.shape[0]
    block_height, block_width = taper_mask.shape[-2], taper_mask.shape[-1]

    # Computes maximum registration offset, constrained by block dimensions.
    maximum_block_radius = np.floor(np.minimum(block_height, block_width) / 2.0) - _UPSAMPLING_PADDING
    correlation_radius = int(np.minimum(np.round(maximum_offset), maximum_block_radius))
    block_count = len(y_blocks)

    extracted_blocks = np.empty((frame_count, block_count, block_height, block_width), dtype=np.float32)
    for block_index in range(block_count):
        y_range, x_range = y_blocks[block_index], x_blocks[block_index]
        extracted_blocks[:, block_index] = frames[:, y_range[0] : y_range[1], x_range[0] : x_range[1]]

    extracted_blocks = apply_mask(frames=extracted_blocks, mask=taper_mask, offset=mean_offset)
    batch_size = min(_CORRELATION_BATCH_SIZE, extracted_blocks.shape[1])
    for batch_start in np.arange(0, block_count, batch_size):
        batch_end = min(extracted_blocks.shape[1], batch_start + batch_size)
        extracted_blocks[:, batch_start:batch_end] = apply_phase_correlation(
            frames=extracted_blocks[:, batch_start:batch_end],
            kernel=reference_kernel[batch_start:batch_end],
            workers=workers,
        )

    # Extracts the central correlation window containing valid peaks.
    half_window = correlation_radius + _UPSAMPLING_PADDING
    correlation_window = np.real(
        np.block(
            [
                [
                    extracted_blocks[:, :, -half_window:, -half_window:],
                    extracted_blocks[:, :, -half_window:, : half_window + 1],
                ],
                [
                    extracted_blocks[:, :, : half_window + 1, -half_window:],
                    extracted_blocks[:, :, : half_window + 1, : half_window + 1],
                ],
            ]
        )
    )
    correlation_window = correlation_window.transpose(1, 0, 2, 3)
    correlation_window = correlation_window.reshape(correlation_window.shape[0], -1)

    # Applies progressive smoothing based on SNR.
    smoothing_levels = [
        correlation_window,
        smoothing_kernel @ correlation_window,
        smoothing_kernel @ smoothing_kernel @ correlation_window,
    ]
    window_size = 2 * correlation_radius + 2 * _UPSAMPLING_PADDING + 1
    smoothing_levels = [level.reshape(block_count, frame_count, window_size, window_size) for level in smoothing_levels]
    smoothed_correlation = smoothing_levels[0]

    for block_index in range(block_count):
        signal_to_noise_ratio = np.ones(frame_count, dtype=np.float32)
        for smoothing_index, smoothed_data in enumerate(smoothing_levels):
            low_snr_mask = signal_to_noise_ratio < snr_threshold
            if np.sum(low_snr_mask) == 0:
                break
            block_correlation = smoothed_data[block_index, low_snr_mask, :, :]
            if smoothing_index > 0:
                smoothed_correlation[block_index, low_snr_mask, :, :] = block_correlation
            signal_to_noise_ratio[low_snr_mask] = _compute_correlation_snr(
                correlation_data=block_correlation,
                padding=_UPSAMPLING_PADDING,
            )

    # Computes subpixel offsets using Gaussian RBF upsampling (vectorized over all blocks and frames).
    midpoint = upsampled_size // 2
    region_size = 2 * _UPSAMPLING_PADDING + 1
    central_size = 2 * correlation_radius + 1

    central_regions = smoothed_correlation[
        :, :, _UPSAMPLING_PADDING:-_UPSAMPLING_PADDING, _UPSAMPLING_PADDING:-_UPSAMPLING_PADDING
    ]
    central_flat = central_regions.reshape(block_count * frame_count, -1)
    flat_indices = np.argmax(a=central_flat, axis=1)
    y_peaks_flat, x_peaks_flat = np.unravel_index(indices=flat_indices, shape=(central_size, central_size))
    y_peaks = y_peaks_flat.reshape(block_count, frame_count).astype(np.int32)
    x_peaks = x_peaks_flat.reshape(block_count, frame_count).astype(np.int32)

    upsampling_regions = np.empty((block_count, frame_count, region_size, region_size), dtype=np.float32)
    _extract_upsampling_regions(
        correlation=smoothed_correlation,
        y_peaks=y_peaks,
        x_peaks=x_peaks,
        region_size=region_size,
        output=upsampling_regions,
    )

    upsampled_flat = upsampling_regions.reshape(block_count * frame_count, -1) @ upsampling_kernel

    # Finds subpixel peak locations and correlation maxima. Gathering the maxima from the peak indices lets a single
    # scan of the upsampled matrix serve both outputs.
    subpixel_indices = np.argmax(a=upsampled_flat, axis=1)
    correlation_maxima = (
        upsampled_flat[np.arange(upsampled_flat.shape[0]), subpixel_indices]
        .reshape(block_count, frame_count)
        .T.astype(np.float32)
    )
    y_subpixel, x_subpixel = np.unravel_index(indices=subpixel_indices, shape=(upsampled_size, upsampled_size))
    y_subpixel = y_subpixel.reshape(block_count, frame_count)
    x_subpixel = x_subpixel.reshape(block_count, frame_count)

    y_integer_offsets = y_peaks - correlation_radius
    x_integer_offsets = x_peaks - correlation_radius
    y_offsets = ((y_subpixel - midpoint) / _SUBPIXEL_FACTOR + y_integer_offsets).T.astype(np.float32)
    x_offsets = ((x_subpixel - midpoint) / _SUBPIXEL_FACTOR + x_integer_offsets).T.astype(np.float32)

    return y_offsets, x_offsets, correlation_maxima


def apply_nonrigid_correction(
    frames: NDArray[np.float32],
    block_counts: tuple[int, int],
    x_blocks: list[NDArray[np.int32]],
    y_blocks: list[NDArray[np.int32]],
    y_block_offsets: NDArray[np.float32],
    x_block_offsets: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Applies nonrigid motion correction to the input batch of frames using block offsets.

    Transforms frame data by upsampling block-level offset estimates to per-pixel offset maps and applying bilinear
    interpolation to warp each frame.

    Args:
        frames: The frame data with shape (frame_count, height, width).
        block_counts: The number of blocks as (y_count, x_count) from compute_registration_blocks.
        x_blocks: The list of x-coordinate ranges for each block from compute_registration_blocks.
        y_blocks: The list of y-coordinate ranges for each block from compute_registration_blocks.
        y_block_offsets: The y-offsets per block with shape (frame_count, block_count) from compute_nonrigid_offsets.
            Positive values shift content upward.
        x_block_offsets: The x-offsets per block with shape (frame_count, block_count) from compute_nonrigid_offsets.
            Positive values shift content leftward.

    Returns:
        The corrected frames with shape (frame_count, height, width).
    """
    _, height, width = frames.shape
    y_offset_maps, x_offset_maps = _upsample_block_offsets(
        width=width,
        height=height,
        block_counts=block_counts,
        x_blocks=x_blocks,
        y_blocks=y_blocks,
        y_block_offsets=y_block_offsets,
        x_block_offsets=x_block_offsets,
    )

    x_grid, y_grid = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    output = np.empty_like(frames, dtype=np.float32)
    _apply_coordinate_offsets(
        frames=frames,
        y_offset_maps=y_offset_maps,
        x_offset_maps=x_offset_maps,
        y_grid=y_grid,
        x_grid=x_grid,
        output=output,
    )

    return output


@njit(parallel=True, cache=True)
def _compute_correlation_snr(  # pragma: no cover
    correlation_data: NDArray[np.float32],
    padding: int,
) -> NDArray[np.float32]:
    """Computes signal-to-noise ratio of phase correlation peaks.

    Estimates the SNR by comparing the maximum correlation value to the maximum value outside a padding region around
    the peak. Low SNR indicates unreliable offset estimates that may benefit from additional smoothing.

    Args:
        correlation_data: The correlation data with shape (frame_count, window_height, window_width).
        padding: The padding width, in pixels, to exclude around the peak when computing noise.

    Returns:
        The SNR values with shape (frame_count,) representing the ratio of peak signal to background.
    """
    frame_count = correlation_data.shape[0]
    window_height = correlation_data.shape[1]
    window_width = correlation_data.shape[2]
    signal_to_noise_ratio = np.empty(frame_count, dtype=np.float32)

    for frame_index in prange(frame_count):
        # Finds peak value and location in central region (excluding padding).
        peak_value = np.float32(-np.inf)
        peak_y = 0
        peak_x = 0
        for row in range(padding, window_height - padding):
            for column in range(padding, window_width - padding):
                value = correlation_data[frame_index, row, column]
                if value > peak_value:
                    peak_value = value
                    peak_y = row - padding
                    peak_x = column - padding

        # Finds the maximum value outside the peak region.
        background_value = np.float32(-np.inf)
        mask_y_end = peak_y + 2 * padding
        mask_x_end = peak_x + 2 * padding
        for row in range(window_height):
            for column in range(window_width):
                if peak_y <= row < mask_y_end and peak_x <= column < mask_x_end:
                    continue
                value = correlation_data[frame_index, row, column]
                background_value = max(background_value, value)

        # Ensures positivity for outlier cases with very low background.
        signal_to_noise_ratio[frame_index] = peak_value / max(background_value, _SNR_EPSILON)  # type: ignore[operator]

    return signal_to_noise_ratio


@njit(cache=True)
def _apply_bilinear_interpolation(  # pragma: no cover
    source: NDArray[np.float32],
    y_coordinates: NDArray[np.float32],
    x_coordinates: NDArray[np.float32],
    output: NDArray[np.float32],
) -> None:
    """Applies in-place bilinear interpolation to transform an image.

    Maps pixel values from the source image to new locations specified by the coordinate arrays using bilinear
    interpolation. Neighbor indices are clamped to the source bounds, so a coordinate above the last row or column
    resolves to the edge pixel. A coordinate below zero keeps its negative fractional part and is linearly extrapolated
    from the first two edge pixels.

    Args:
        source: The source image, sampled with bilinear interpolation, with shape (source_height, source_width).
        y_coordinates: The target y-coordinates, with the same shape as output.
        x_coordinates: The target x-coordinates, with the same shape as output.
        output: The output array where interpolated values are stored. Its shape defines the sampled grid and does
            not have to match the source shape.
    """
    height, width = source.shape
    output_height, output_width = output.shape

    for row in range(output_height):
        for column in range(output_width):
            y_coordinate = y_coordinates[row, column]
            x_coordinate = x_coordinates[row, column]

            y_floor = int(y_coordinate)
            x_floor = int(x_coordinate)
            y_fraction = y_coordinate - y_floor
            x_fraction = x_coordinate - x_floor

            y_floor = min(height - 1, max(0, y_floor))
            x_floor = min(width - 1, max(0, x_floor))
            y_ceiling = min(height - 1, y_floor + 1)
            x_ceiling = min(width - 1, x_floor + 1)

            # Computes the weighted average of the four neighboring pixels.
            output[row, column] = (
                source[y_floor, x_floor] * (1 - y_fraction) * (1 - x_fraction)
                + source[y_floor, x_ceiling] * (1 - y_fraction) * x_fraction
                + source[y_ceiling, x_floor] * y_fraction * (1 - x_fraction)
                + source[y_ceiling, x_ceiling] * y_fraction * x_fraction
            )


@njit(parallel=True, cache=True)
def _apply_coordinate_offsets(  # pragma: no cover
    frames: NDArray[np.float32],
    y_offset_maps: NDArray[np.float32],
    x_offset_maps: NDArray[np.float32],
    y_grid: NDArray[np.float32],
    x_grid: NDArray[np.float32],
    output: NDArray[np.float32],
) -> None:
    """Applies per-pixel coordinate offsets to a batch of frames.

    Transforms each frame by adding the offset maps to the base coordinate grids and applying bilinear interpolation.

    Args:
        frames: The input frame data with shape (frame_count, height, width) to be transformed.
        y_offset_maps: The per-pixel vertical offsets with shape (frame_count, height, width). Positive values shift
            content upward (sample from higher y-coordinates).
        x_offset_maps: The per-pixel horizontal offsets with shape (frame_count, height, width). Positive values shift
            content leftward (sample from higher x-coordinates).
        y_grid: The base y-coordinate grid with shape (height, width) containing row indices (0 to height-1). Combined
            with y_offset_maps to determine source sampling locations.
        x_grid: The base x-coordinate grid with shape (height, width) containing column indices (0 to width-1).
            Combined with x_offset_maps to determine source sampling locations.
        output: The pre-allocated output array with shape (frame_count, height, width) where transformed frames are
            stored.
    """
    for frame_index in prange(frames.shape[0]):
        _apply_bilinear_interpolation(
            source=frames[frame_index],
            y_coordinates=y_grid + y_offset_maps[frame_index],
            x_coordinates=x_grid + x_offset_maps[frame_index],
            output=output[frame_index],
        )


@njit(parallel=True, cache=True)
def _interpolate_block_offsets(  # pragma: no cover
    y_block_offsets: NDArray[np.float32],
    x_block_offsets: NDArray[np.float32],
    y_grid: NDArray[np.float32],
    x_grid: NDArray[np.float32],
    y_offset_maps: NDArray[np.float32],
    x_offset_maps: NDArray[np.float32],
) -> None:
    """Interpolates block-level offsets to pixel-level offset maps.

    Converts the sparse block offset values to dense per-pixel offset maps using bilinear interpolation. This enables
    smooth transitions between adjacent blocks.

    Args:
        y_block_offsets: The vertical offsets computed for each block with shape (frame_count, y_blocks, x_blocks). Each
            value represents the estimated y-displacement for that block region.
        x_block_offsets: The horizontal offsets computed for each block with shape (frame_count, y_blocks, x_blocks).
            Each value represents the estimated x-displacement for that block region.
        y_grid: The interpolation grid for y with shape (height, width) containing normalized block coordinates that
            map each pixel to its position in block space.
        x_grid: The interpolation grid for x with shape (height, width) containing normalized block coordinates that
            map each pixel to its position in block space.
        y_offset_maps: The pre-allocated output array with shape (frame_count, height, width) where interpolated
            per-pixel y-offsets are stored.
        x_offset_maps: The pre-allocated output array with shape (frame_count, height, width) where interpolated
            per-pixel x-offsets are stored.
    """
    for frame_index in prange(y_block_offsets.shape[0]):
        _apply_bilinear_interpolation(
            source=y_block_offsets[frame_index],
            y_coordinates=y_grid,
            x_coordinates=x_grid,
            output=y_offset_maps[frame_index],
        )
        _apply_bilinear_interpolation(
            source=x_block_offsets[frame_index],
            y_coordinates=y_grid,
            x_coordinates=x_grid,
            output=x_offset_maps[frame_index],
        )


@njit(parallel=True, cache=True)
def _extract_upsampling_regions(  # pragma: no cover
    correlation: NDArray[np.float32],
    y_peaks: NDArray[np.int32],
    x_peaks: NDArray[np.int32],
    region_size: int,
    output: NDArray[np.float32],
) -> None:
    """Extracts upsampling regions around peak locations for all blocks and frames.

    Copies a region of size (region_size, region_size) centered at each peak location from the correlation data.
    Parallelizes over all (block, frame) pairs for efficiency.

    Args:
        correlation: The correlation data with shape (block_count, frame_count, window_height, window_width).
        y_peaks: The y-coordinates of peaks with shape (block_count, frame_count).
        x_peaks: The x-coordinates of peaks with shape (block_count, frame_count).
        region_size: The size of the square region to extract around each peak.
        output: The pre-allocated output array with shape (block_count, frame_count, region_size, region_size).
    """
    block_count = y_peaks.shape[0]
    frame_count = y_peaks.shape[1]

    for index in prange(block_count * frame_count):
        block_index = index // frame_count
        frame_index = index % frame_count
        peak_y = y_peaks[block_index, frame_index]
        peak_x = x_peaks[block_index, frame_index]

        for row in range(region_size):
            for column in range(region_size):
                output[block_index, frame_index, row, column] = correlation[
                    block_index, frame_index, peak_y + row, peak_x + column
                ]


def _upsample_block_offsets(
    width: int,
    height: int,
    block_counts: tuple[int, int],
    x_blocks: list[NDArray[np.int32]],
    y_blocks: list[NDArray[np.int32]],
    y_block_offsets: NDArray[np.float32],
    x_block_offsets: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Upsamples block-level offsets to dense per-pixel offset maps.

    Converts the sparse block offset estimates to full-resolution offset maps for applying nonrigid corrections. Uses
    bilinear interpolation to create smooth transitions between blocks.

    Args:
        width: The imaging field width, in pixels.
        height: The imaging field height, in pixels.
        block_counts: The number of blocks as (y_count, x_count).
        x_blocks: The list of x-coordinate ranges for each block.
        y_blocks: The list of y-coordinate ranges for each block.
        y_block_offsets: The y-offsets per block with shape (frame_count, block_count).
        x_block_offsets: The x-offsets per block with shape (frame_count, block_count).

    Returns:
        A tuple of (y_offset_maps, x_offset_maps) arrays with shape (frame_count, height, width) containing per-pixel
        offset values.
    """
    # Recovers the block center coordinates from the block boundary arrays.
    y_centers = np.array(y_blocks[:: block_counts[1]], dtype=np.float32).mean(axis=1)
    x_centers = np.array(x_blocks[: block_counts[1]], dtype=np.float32).mean(axis=1)

    # Creates interpolation grids mapping pixel positions to block indices.
    y_indices = np.interp(x=np.arange(height), xp=y_centers, fp=np.arange(y_centers.size)).astype(np.float32)
    x_indices = np.interp(x=np.arange(width), xp=x_centers, fp=np.arange(x_centers.size)).astype(np.float32)
    x_grid, y_grid = np.meshgrid(x_indices, y_indices)

    frame_count = y_block_offsets.shape[0]
    y_block_offsets = y_block_offsets.reshape(frame_count, block_counts[0], block_counts[1])
    x_block_offsets = x_block_offsets.reshape(frame_count, block_counts[0], block_counts[1])

    y_offset_maps = np.empty((frame_count, height, width), dtype=np.float32)
    x_offset_maps = np.empty((frame_count, height, width), dtype=np.float32)
    _interpolate_block_offsets(
        y_block_offsets=y_block_offsets,
        x_block_offsets=x_block_offsets,
        y_grid=y_grid,
        x_grid=x_grid,
        y_offset_maps=y_offset_maps,
        x_offset_maps=x_offset_maps,
    )

    return y_offset_maps, x_offset_maps
