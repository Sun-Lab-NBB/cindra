"""This module provides tools for estimating and correcting the bidirectional phase offset between scan lines in a
stack of frames."""

import numpy as np
from numpy import fft
from numpy.typing import NDArray


def compute(frames: NDArray[np.int16] | NDArray[np.float32]) -> int:
    """Computes the bidirectional phase offset between alternating scan lines in a stack of frames.

    Args:
        frames: A 3D NumPy array storing the frames to compute the bidirectional phase offset from.

    Returns:
        The estimated pixel offset along the x-axis between even and odd scan lines.
    """
    # Extracts the width of the input frames.
    _, _, width = frames.shape

    # Computes the Fourier transform of the odd-numbered lines along the x-axis.
    fft_odd_lines = fft.fft(frames[:, 1::2, :], axis=2)

    # Normalizes the FFT results to unit magnitude to isolate the phase information.
    fft_odd_lines /= np.abs(fft_odd_lines) + 1e-5

    # Computes the Fourier transform of the even-numbered lines along the x-axis.
    fft_even_lines = np.conj(fft.fft(frames[:, ::2, :], axis=2))

    # Normalizes the FFT results to unit magnitude to isolate the phase information.
    fft_even_lines /= np.abs(fft_even_lines) + 1e-5

    # Ensures the shape between the odd and even lines matches.
    fft_even_lines = fft_even_lines[:, : fft_odd_lines.shape[1], :]

    # Multiplies the odd and even lines' FFTs and computes the inverse FFT to get the cross-correlation function.
    cross_correlation = np.real(fft.ifft(fft_odd_lines * fft_even_lines, axis=2))

    # Averages the cross-correlation over the y-axis and then across all frames, resulting in a 1D NumPy array
    # representing the mean correlation at each horizontal pixel offset.
    cross_correlation = cross_correlation.mean(axis=1).mean(axis=0)

    # Applies FFT shift to center the zero-frequency component of the cross-correlation array.
    cross_correlation = fft.fftshift(cross_correlation)

    # Finds the index of the maximum value within a 10 pixel window centered at the middle of the frame width.
    bidirectional_phase_offset = -(np.argmax(cross_correlation[-10 + width // 2 : 11 + width // 2]) - 10)

    # Returns the computed bidirectional phase offset.
    return int(bidirectional_phase_offset)


def shift(frames: NDArray[np.int16] | NDArray[np.float32], bidirectional_phase_offset: int) -> None:
    """Applies an in-place horizontal shift to the odd-number scan lines in a stack of frames.

    This function corrects for the bidirectional phase offset that occurs in line scanning, by shifting the pixels
    of odd scan lines along the x-axis by the specified offset.

    Args:
        frames: A 3D NumPy array storing the frames to be shifted.
        bidirectional_phase_offset: The pixel offset to shift the odd-number scan lines by. Positive values shift
            pixels to the right, while negative values shift pixels to the left.
    """
    # If the provided offset is positive, shifts the pixels to the right for odd scan lines. Pixels on the right are
    # discarded, and the left side is filled with shifted content.
    if bidirectional_phase_offset > 0:
        frames[:, 1::2, bidirectional_phase_offset:] = frames[:, 1::2, :-bidirectional_phase_offset]
    # If the provided offset is negative, shifts pixels to the left by the absolute value of the provided offset for
    # odd scan lines. Pixels on the left edge are discarded, and the right side is filled with shifted content.
    else:
        frames[:, 1::2, :bidirectional_phase_offset] = frames[:, 1::2, -bidirectional_phase_offset:]
