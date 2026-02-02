"""Provides rigid (translation-only) registration functions for motion correction."""

from typing import TYPE_CHECKING

import numpy as np

from .utils import (
    NORMALIZATION_EPSILON,
    apply_mask,
    compute_reference_fft,
    apply_phase_correlation,
    apply_temporal_smoothing,
    compute_spatial_taper_mask,
    compute_gaussian_frequency_filter,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_edge_taper(
    reference_image: NDArray[np.float32],
    taper_slope: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Computes edge taper mask and mean offset for phase correlation preprocessing.

    Creates a spatial taper that suppresses edge artifacts during phase correlation. The taper mask
    transitions from 1.0 in the center to ~0 at edges. The mean offset fills tapered regions with uniform
    intensity (the image mean) rather than fading to black, preventing artificial gradients at frame
    borders that could create spurious correlation peaks.

    Args:
        reference_image: The reference image with shape (height, width) used to compute the mean offset.
        taper_slope: Controls the steepness of the edge falloff. Larger values produce a more gradual taper.

    Returns:
        A tuple of (taper_mask, mean_offset) arrays with shape (height, width). The taper_mask contains
        sigmoid-based edge weights, and mean_offset equals reference_image.mean() * (1 - taper_mask).
    """
    height, width = reference_image.shape
    taper_mask = compute_spatial_taper_mask(sigma=taper_slope, height=height, width=width)
    mean_offset = reference_image.mean() * (1.0 - taper_mask)
    return taper_mask, mean_offset.astype(np.float32)


def apply_edge_taper(
    frames: NDArray[np.float32],
    taper_mask: NDArray[np.float32],
    mean_offset: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Applies edge taper to frames for phase correlation preprocessing.

    Computes (frames * taper_mask + mean_offset) to suppress edge artifacts while preserving mean intensity.
    This preprocessing step reduces wraparound artifacts in phase correlation by attenuating frame borders.

    Args:
        frames: The frame data with shape (num_frames, height, width).
        taper_mask: The edge taper mask with shape (height, width) from compute_edge_taper.
        mean_offset: The mean intensity offset with shape (height, width) from compute_edge_taper.

    Returns:
        The tapered frames with the same shape as input.
    """
    return apply_mask(frames=frames, mask=taper_mask, offset=mean_offset)


def compute_phase_correlation_kernel(
    reference_image: NDArray[np.float32],
    smoothing_sigma: float = 0.0,
) -> NDArray[np.complex64]:
    """Computes the phase correlation kernel from a reference image.

    Transforms the reference image to frequency domain, normalizes by magnitude to extract phase-only
    information, and optionally applies Gaussian smoothing. The resulting kernel is used for
    cross-correlation with data frames during motion estimation.

    Args:
        reference_image: The reference image with shape (height, width).
        smoothing_sigma: The standard deviation of Gaussian smoothing in pixels. Values <= 0 disable
            smoothing.

    Returns:
        The phase correlation kernel with shape (height, width // 2 + 1) from real FFT.
    """
    height, width = reference_image.shape
    reference_fft = compute_reference_fft(reference_image=reference_image)
    reference_fft /= NORMALIZATION_EPSILON + np.absolute(reference_fft)

    if smoothing_sigma > 0:
        reference_fft *= compute_gaussian_frequency_filter(
            sigma=smoothing_sigma,
            height=height,
            width=width,
        )

    return reference_fft.astype(np.complex64)


def compute_rigid_shifts(
    frames: NDArray[np.float32],
    reference_kernel: NDArray[np.complex64],
    maximum_shift_fraction: float,
    temporal_smoothing_sigma: float,
    workers: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]:
    """Computes rigid translation shifts using phase correlation.

    Estimates per-frame (y, x) pixel shifts by finding the peak of the phase correlation between
    each frame and the reference kernel. Optionally applies temporal smoothing to the correlation
    maps before peak detection.

    Args:
        frames: The frame data with shape (num_frames, height, width) after edge tapering.
        reference_kernel: The phase correlation kernel from compute_phase_correlation_kernel.
        maximum_shift_fraction: The maximum allowed shift as a fraction of the minimum spatial dimension.
            The search window is limited to min(height, width) * maximum_shift_fraction pixels.
        temporal_smoothing_sigma: The standard deviation for temporal Gaussian smoothing of correlation
            maps. If 0, no smoothing is applied.
        workers: The number of parallel workers for FFT computation. Use -1 for all available cores.

    Returns:
        A tuple of (y_shifts, x_shifts, correlation_maxima) arrays with shape (num_frames,). The shifts
        are pixel offsets from the reference, and correlation_maxima indicates the peak correlation
        value for each frame.
    """
    # Computes the correlation search window size based on maximum allowed shift.
    minimum_dimension = np.minimum(*frames.shape[1:])
    maximum_radius = minimum_dimension // 2
    correlation_radius = int(np.minimum(np.round(maximum_shift_fraction * minimum_dimension), maximum_radius))

    # Applies phase correlation in frequency domain.
    correlation_data = apply_phase_correlation(frames=frames, kernel=reference_kernel, workers=workers)

    # Extracts the central region containing valid correlation peaks. The correlation surface wraps around,
    # so negative shifts appear at the end of each axis. This block rearranges the four quadrants into a
    # contiguous window centered at zero shift.
    correlation_window = np.real(
        np.block(
            [
                [
                    correlation_data[:, -correlation_radius:, -correlation_radius:],
                    correlation_data[:, -correlation_radius:, : correlation_radius + 1],
                ],
                [
                    correlation_data[:, : correlation_radius + 1, -correlation_radius:],
                    correlation_data[:, : correlation_radius + 1, : correlation_radius + 1],
                ],
            ]
        )
    )

    # Applies temporal smoothing to reduce noise in correlation peaks.
    if temporal_smoothing_sigma > 0:
        correlation_window = apply_temporal_smoothing(frames=correlation_window, sigma=temporal_smoothing_sigma)

    # Finds peak location for each frame using vectorized argmax.
    num_frames = frames.shape[0]
    window_size = 2 * correlation_radius + 1
    flat_indices = np.argmax(correlation_window.reshape(num_frames, -1), axis=1)
    y_shifts = (flat_indices // window_size - correlation_radius).astype(np.int32)
    x_shifts = (flat_indices % window_size - correlation_radius).astype(np.int32)

    # Extracts correlation maxima at peak locations.
    correlation_maxima = correlation_window.reshape(num_frames, -1)[np.arange(num_frames), flat_indices]

    return y_shifts, x_shifts, correlation_maxima.astype(np.float32)


def shift_frame(frame: NDArray[np.float32], y_shift: int, x_shift: int) -> NDArray[np.float32]:
    """Applies a rigid translation to a single frame using circular shift.

    Shifts the frame by the specified pixel amounts using numpy roll. Positive shift values move the
    image content in the negative direction (i.e., a positive y_shift moves content upward).

    Args:
        frame: The frame with shape (height, width) to be shifted.
        y_shift: The vertical shift amount in pixels from compute_rigid_shifts. Positive values shift content upward.
        x_shift: The horizontal shift amount in pixels from compute_rigid_shifts. Positive values shift content
            leftward.

    Returns:
        The shifted frame with the same shape as input.
    """
    return np.roll(frame, shift=(-y_shift, -x_shift), axis=(0, 1))
