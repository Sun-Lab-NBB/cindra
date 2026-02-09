"""Provides bidirectional phase offset correction algorithm for line-scanned imaging data."""

from typing import TYPE_CHECKING

import numpy as np
from scipy import fft

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_bidirectional_phase_offset(frames: NDArray[np.float32]) -> int:
    """Computes the bidirectional phase offset from a sample of imaging frames.

    Bidirectional scanning microscopes acquire alternating lines in opposite directions, which can introduce a
    horizontal offset between odd and even lines. This function estimates that offset by computing the phase
    correlation between odd and even lines across all provided frames.

    Args:
        frames: A 3D array of imaging frames with shape (frame_count, height, width). The frames should be a
            representative sample from the recording, typically selected at regular intervals throughout the session.

    Returns:
        The estimated bidirectional phase offset in pixels. Positive values indicate that odd lines should be shifted
        right, negative values indicate they should be shifted left. A value of zero means no correction is needed.
    """
    _, _height, width = frames.shape

    # Computes the real FFT of odd lines (1, 3, 5, ...) along the x-axis. Uses rfft since input is real-valued,
    # which is ~2x faster than fft and uses half the memory. Casts to complex64 to prevent complex128 promotion.
    odd_lines_fft = fft.rfft(frames[:, 1::2, :], axis=2, workers=-1).astype(np.complex64)
    odd_lines_fft /= np.abs(odd_lines_fft) + np.float32(1e-5)

    # Computes the conjugate FFT of even lines (0, 2, 4, ...) along the x-axis.
    even_lines_fft = fft.rfft(frames[:, ::2, :], axis=2, workers=-1).astype(np.complex64)
    np.conj(even_lines_fft, out=even_lines_fft)
    even_lines_fft /= np.abs(even_lines_fft) + np.float32(1e-5)

    # Truncates even lines to match odd lines count (in case of odd height).
    even_lines_fft = even_lines_fft[:, : odd_lines_fft.shape[1], :]

    # Computes the cross-correlation via inverse FFT of the product and averages across all frames and lines.
    cross_correlation = fft.irfft(odd_lines_fft * even_lines_fft, n=width, axis=2, workers=-1).astype(np.float32)
    cross_correlation = cross_correlation.mean(axis=(0, 1))
    cross_correlation = fft.fftshift(cross_correlation)

    # Finds the peak in a small window around zero to determine the offset.
    search_window_half_width = 10
    window_start = -search_window_half_width + width // 2
    window_end = search_window_half_width + 1 + width // 2
    bidirectional_phase_offset = -(np.argmax(cross_correlation[window_start:window_end]) - search_window_half_width)

    return int(bidirectional_phase_offset)


def apply_bidirectional_phase_correction(
    frames: NDArray[np.float32],
    bidirectional_phase_offset: int,
) -> None:
    """Applies bidirectional phase correction to imaging frames in-place.

    Shifts the odd lines (1, 3, 5, ...) of each frame horizontally by the specified offset to correct for
    bidirectional scanning artifacts. The correction is applied in-place to avoid memory allocation overhead.

    Args:
        frames: A 3D array of imaging frames with shape (frame_count, height, width) to be corrected in-place.
        bidirectional_phase_offset: The horizontal offset in pixels to apply to odd lines. Positive values shift
            odd lines to the right, negative values shift them to the left.
    """
    if bidirectional_phase_offset == 0:
        return

    if bidirectional_phase_offset > 0:
        # Shifts odd lines right and zeros the left border for consistency with spatial filtering zero-padding.
        frames[:, 1::2, bidirectional_phase_offset:] = frames[:, 1::2, :-bidirectional_phase_offset]
        frames[:, 1::2, :bidirectional_phase_offset] = 0
    else:
        # Shifts odd lines left and zeros the right border for consistency with spatial filtering zero-padding.
        frames[:, 1::2, :bidirectional_phase_offset] = frames[:, 1::2, -bidirectional_phase_offset:]
        frames[:, 1::2, bidirectional_phase_offset:] = 0
