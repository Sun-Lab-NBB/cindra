"""This module provides tools for rigid image registration and alignment, including the creation and application of
masks, phase correlation for frame registration, and the shifting of frames based on calculated offsets."""

from typing import Any

import numpy as np
from numpy.typing import NDArray

from .utils import convolve, add_multiply, gaussian_fft, spatial_taper, complex_fft_2d, temporal_smooth


def compute_masks(
    reference_image: NDArray[np.int16] | NDArray[np.float32], mask_slope: float
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Computes the spatial taper mask and corresponding mask offset for registration based on the input reference image
    and mask slope.

    Args:
        reference_image: A 2D NumPy array storing the reference image used for image registration.
        mask_slope: Determines the steepness of the tapering effect applied to the reference image edges. A larger
            slope results in a sharper taper.

    Returns:
        A tuple of two elements. The first element is a 2D NumPy array storing the spatial taper mask to be multiplied
        with the image during registration. The second element is a 2D NumPy array storing the mask offset, which
        is later used in the pipeline to help preserve the original mean intensity of the image.
    """
    # Extracts the height and width of the reference image.
    height, width = reference_image.shape

    # Computes the taper mask based on the reference image and the mask slope.
    taper_mask = spatial_taper(sigma=mask_slope, height=height, width=width)

    # Calculates the mask offset by taking the mean intensity of the reference image and adjusting based on the taper
    # mask.
    mask_offset = reference_image.mean() * (1.0 - taper_mask)

    # Returns the taper mask and offset after converting the data types to float32.
    return taper_mask.astype("float32"), mask_offset.astype("float32")


def apply_masks(
    frames: NDArray[np.int16] | NDArray[np.float32],
    mask_multiplier: NDArray[np.float32],
    mask_offset: NDArray[np.float32],
) -> NDArray[np.complex64]:
    """Applies the mask multiplier and mask offset to the input data using element-wise multiplication and addition.

    Args:
        frames: A 3D NumPy array storing the frames to be registered.
        mask_multiplier: A 2D NumPy array storing the mask multiplier.
        mask_offset: A 2D NumPy array representing the mask offset.

    Returns:
        A 3D NumPy array where the masks have been applied to the input frames.
    """
    masked_frames: NDArray[np.complex64] = add_multiply(frames, mask_multiplier, mask_offset)
    return masked_frames


def fft_reference_image(
    reference_image: NDArray[np.int16] | NDArray[np.float32], smooth_sigma: float = 1.0
) -> NDArray[np.complex64]:
    """Computes the fast Fourier transform (FFT) of the reference image and applies a Gaussian filter in the frequency
    domain for phase correlation.

    Args:
        reference_image: A 2D NumPy array storing the reference image used for image registration.
        smooth_sigma: The standard deviation for the Gaussian filter. The default is 1.0.

    Returns:
        A 2D NumPy array of complex values storing the Fourier-transformed reference image, normalized and filtered.
    """
    # Applies fast Fourier transform (FFT) of the reference image and stores its complex conjugate.
    fft_reference_image = complex_fft_2d(reference_image=reference_image)

    # Normalizes the Fourier-transformed image.
    fft_reference_image /= 1e-5 + np.absolute(fft_reference_image)

    # Applies a Gaussian filter in the frequency domain.
    fft_reference_image *= gaussian_fft(
        sigma=smooth_sigma, height=fft_reference_image.shape[0], width=fft_reference_image.shape[1]
    )

    # Returns the Fourier-transformed reference image after converting the data type to complex64.
    return fft_reference_image.astype("complex64")


def phase_correlate(
    frames: NDArray[np.complex64],
    fft_reference_image: NDArray[np.complex64],
    maximum_shift: float,
    smooth_sigma_time: float,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]:
    """Computes the phase correlation between the input frames and the reference image.

    This function calculates per-frame registration shifts by computing the phase correlation with a Fourier-transformed
    reference image. It returns the y- and x-shifts along with the peak correlation scores for each frame, which are
    used for rigid registration.

    Args:
        frames: A 3D NumPy array storing the frames to be registered.
        fft_reference_image: A 2D NumPy array storing the Fourier-transformed reference image.
        maximum_shift: The maximum allowed registration shift as a fraction of the minimum dimension of the frames.
        smooth_sigma_time: The standard deviation in time frames of the Gaussian used to smooth the data before phase
            correlation is computed

    Returns:
        A tuple of three elements. The first and second elements are the computed y- and x-shifts respectively from
        the reference image to each frame. The third element is the maximum phase correlation for each frame.
    """
    # Determines the minimum dimension between the height and width of the frames.
    minimum_dimension = np.minimum(frames.shape[1], frames.shape[2])

    # Computes the maximum allowed shift in pixels, constrained to half the minimum dimension.
    correlation_window = int(np.minimum(np.round(maximum_shift * minimum_dimension), minimum_dimension // 2))

    # Computes the convolution in the frequency domain between each frame and the reference image.
    convolved_frames = convolve(frames, fft_reference_image)

    # Extracts the region of the correlation matrix corresponding to allowed shifts.
    cross_correlation = np.real(
        np.block(
            [
                [
                    convolved_frames[:, -correlation_window:, -correlation_window:],
                    convolved_frames[:, -correlation_window:, : correlation_window + 1],
                ],
                [
                    convolved_frames[:, : correlation_window + 1, -correlation_window:],
                    convolved_frames[:, : correlation_window + 1, : correlation_window + 1],
                ],
            ]
        )
    )

    # Applies temporal smoothing if 'smooth_sigma_time' is provided.
    cross_correlation = (
        temporal_smooth(cross_correlation, smooth_sigma_time) if smooth_sigma_time > 0 else cross_correlation
    )

    # Initializes arrays to store the computed per-frame y- and x-shifts.
    y_shifts, x_shifts = (
        np.zeros(convolved_frames.shape[0], np.int32),
        np.zeros(convolved_frames.shape[0], np.int32),
    )

    # Loops over the frames. For each frame, computes the y- and x-shift by finding the maximum correlation.
    for frame_index in np.arange(convolved_frames.shape[0]):
        y_shifts[frame_index], x_shifts[frame_index] = np.unravel_index(
            np.argmax(cross_correlation[frame_index], axis=None),
            (2 * correlation_window + 1, 2 * correlation_window + 1),
        )

    # Extracts the maximum correlation score at the computed shifts for each frame.
    correlation_scores = cross_correlation[np.arange(len(cross_correlation)), y_shifts, x_shifts]

    # Adjusts the y- and x-shifts to be relative to the center of the correlation window.
    y_shifts, x_shifts = y_shifts - correlation_window, x_shifts - correlation_window

    # Returns the computed y-shift, x-shift, and peak correlation score of each frame.
    return (
        y_shifts,
        x_shifts,
        correlation_scores.astype(np.float32),
    )


def shift_frame(frame: NDArray[Any], y_shift: int | np.int32, x_shift: int | np.int32) -> NDArray[Any]:
    """Shifts a single frame by the input y- and x-offsets.

    Args:
        frame: A 2D NumPy array storing the frame to be shifted.
        y_shift: The number of pixels to shift along the y-axis.
        x_shift: The number of pixels to shift along the x-axis.

    Returns:
        A 2D NumPy array storing the shifted frame.
    """
    return np.roll(frame, (-y_shift, -x_shift), axis=(0, 1))
