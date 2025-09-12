"""This module provides tools for non-rigid image registration and alignment, including the creation and application of
masks, phase correlation for frame registration, and the shifting of frames based on calculated offsets."""

from typing import Any
import platform

from numba import njit, config, prange, float32
import numpy as np
from numpy import fft
from numpy.typing import NDArray

from .utils import (
    convolve,
    add_multiply,
    gaussian_fft,
    spatial_taper,
    generate_upsampling_matrix,
    normalized_gaussian_kernel,
)

if platform.system() == "Darwin":
    config.THREADING_LAYER = "omp"
else:
    config.THREADING_LAYER = "tbb"


def _calculate_block_number(length: int, block_size: int = 128) -> tuple[int, int]:
    """Calculates the block size and the number of blocks needed to divide a dimension of a given length.

    This function computes how to divide a dimension (e.g., height or width) into blocks of a given size. If the block
    size is greater than or equal to the length, the entire dimension will be considered as one block. Otherwise, it
    calculates how many blocks of the specified size are required to cover the dimension, with a small overestimate
    to ensure coverage.

    Args:
        length: The total length of the dimension (e.g., height or width) to be divided into blocks.
        block_size: The desired size of each block. This controls the size of each block in the division. The default
            is 128.

    Returns:
        A tuple of two elements. The first element is the size of each block, and the second element is the number of
        blocks needed to cover the given dimension.
    """
    return (length, 1) if block_size >= length else (block_size, int(np.ceil(1.5 * length / block_size)))


def make_blocks(
    height: int, width: int, block_size: tuple[int, int] = (128, 128)
) -> tuple[list[NDArray[np.int_]], list[NDArray[np.int_]], list[int], tuple[int, int], NDArray[np.float32]]:
    """Computes overlapping blocks to split the input field of view (FOV) into blocks for separate registration.

    This function generates the block boundaries along the y- and x-axes, and it also creates a weighted spatial kernel
    based on block proximity.

    Args:
        height: The height (in pixels) of the field of view (FOV).
        width: The width (in pixels) of the field of view (FOV).
        block_size: A tuple of two elements. The first element is the size of y-blocks, and the second element is the
            size of the x-blocks. The default is (128, 128).

    Returns:
        A tuple of five elements. The first element is a list of 1D NumPy arrays storing the y-block boundaries, and
        the second element is a list of 1D NumPy arrays of the x-block boundaries. The third element is a list of two
        elements storing the number of blocks along the y- and x-axes. The fourth element is a tuple of two elements
        storing the size of the y-blocks and x-blocks. The fifth element is the weighted spatial kernel that defines
        how much influence each block should have on the other blocks based on spatial proximity.
    """
    # Calculates the block size and the number of blocks along the y- and x-axes.
    y_block_size, y_block_number = _calculate_block_number(length=height, block_size=block_size[0])
    x_block_size, x_block_number = _calculate_block_number(length=width, block_size=block_size[1])
    block_size = (y_block_size, x_block_size)

    # Generates evenly spaced start positions for each block along the y- and x-axes.
    y_start_position = np.linspace(0, height - block_size[0], y_block_number).astype("int")
    x_start_position = np.linspace(0, width - block_size[1], x_block_number).astype("int")

    # Generates the boundaries for each block along the y- and x-axes. Each block boundary is represented by a NumPy
    # array storing the start and end positions.
    y_blocks = [
        np.array([y_start_position[y_block_index], y_start_position[y_block_index] + block_size[0]])
        for y_block_index in range(y_block_number)
        for _ in range(x_block_number)
    ]
    x_blocks = [
        np.array([x_start_position[x_block_index], x_start_position[x_block_index] + block_size[1]])
        for _ in range(y_block_number)
        for x_block_index in range(x_block_number)
    ]

    # Generates a weighted spatial kernel based on the block grid.
    smoothing_kernel = normalized_gaussian_kernel(
        y_coordinates=np.arange(y_block_number), x_coordinates=np.arange(x_block_number)
    ).T

    # Returns the boundaries of the y- and x-blocks, the number of y- and x-blocks, the size of the blocks, and the
    # block kernel.
    return y_blocks, x_blocks, [y_block_number, x_block_number], block_size, smoothing_kernel


def fft_reference_image(
    reference_image: NDArray[np.int16],
    mask_slope: float,
    smooth_sigma: float,
    y_blocks: list[NDArray[np.int_]],
    x_blocks: list[NDArray[np.int_]],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.complex64]]:
    """Computes a tapered and Fourier-transformed reference image for phase correlation.

    This function divides the reference image into smaller blocks, applies a spatial taper (mask) to each block,
    computes the Fourier transform of the reference image, and applies a Gaussian filter to the Fourier-transformed
    blocks.

    Args:
        reference_image: A 2D NumPy array storing the reference image used for image registration.
        mask_slope: Determines the steepness of the tapering effect applied to the reference image edges. A larger
            slope results in a sharper taper.
        smooth_sigma: The standard deviation in pixels of the Gaussian filter used to smooth the phase correlation
            between the reference image and the frame which is being registered.
        y-blocks: A list of NumPy arrays storing the y-coordinates for each block (start and end positions along the
            y-axis).
        x-blocks: A list of NumPy arrays storing the x-coordinates for each block (start and end positions along the
            x-axis).

    Returns:
        A tuple of three elements. The first element is a 4D NumPy array storing the taper mask for each
        block, and the second element is a 4D NumPy array storing the mask offset for each block of the reference
        image. The third element is a 4D NumPy array storing the Fourier-transformed reference image blocks, which have
        been smoothed by the Gaussian filter.
    """
    # Extracts the number of blocks and the block height width from the y- and x-coordinates.
    block_number, block_height, block_width = (
        len(y_blocks),
        y_blocks[0][1] - y_blocks[0][0],
        x_blocks[0][1] - x_blocks[0][0],
    )

    # Stores the dimensions of the reference image in terms of number of blocks, height, and width.
    reference_image_dimensions = (block_number, block_height, block_width)

    # Generates the Gaussian filter to be applied in the frequency domain.
    gaussian_filter = gaussian_fft(smooth_sigma, *reference_image_dimensions[1:])

    # Initializes a NumPy array to store the Fourier-transformed reference image.
    fft_reference_image = np.zeros(reference_image_dimensions, "complex64")

    # Computes the taper mask based on the reference image and the mask slope.
    reference_image_taper_mask = spatial_taper(mask_slope, *reference_image.shape)

    # Initializes NumPy arrays to store the taper mask and the mask offsets for each block.
    taper_mask = np.zeros(reference_image_dimensions, "float32")
    taper_mask[:] = spatial_taper(2 * smooth_sigma, block_height, block_width)
    mask_offset = np.zeros(reference_image_dimensions, "float32")

    # Loops over blocks to generate the reference image blocks.
    for y_block, x_block, taper_mask_block, mask_offset_block, fft_reference_image_block in zip(
        y_blocks, x_blocks, taper_mask, mask_offset, fft_reference_image
    ):
        # Extracts the block-specific indices to extract the block from the reference image.
        block_index = np.ix_(
            np.arange(y_block[0], y_block[-1]).astype("int"),
            np.arange(x_block[0], x_block[-1]).astype("int"),
        )

        # Extracts the current reference image block.
        block_reference_image = reference_image[block_index]

        # Applies the taper mask to the current block to adjust the intensity at the edges.
        taper_mask_block *= reference_image_taper_mask[block_index]

        # Calculates the mask offset by taking the mean intensity of the block and adjusting based on the taper mask.
        mask_offset_block[:] = block_reference_image.mean() * (1.0 - taper_mask_block)

        # Applies the Fourier transform to the block and normalizes the result.
        fft_reference_image_block[:] = np.conj(fft.fft2(block_reference_image))
        fft_reference_image_block /= 1e-5 + np.absolute(fft_reference_image_block)
        fft_reference_image_block[:] *= gaussian_filter

    # Adds an extra axis for compatibility. Returns the tapered reference image, mask offset, and the
    # Fourier-transformed reference image.
    return (
        taper_mask[:, np.newaxis, :, :],
        mask_offset[:, np.newaxis, :, :],
        fft_reference_image[:, np.newaxis, :, :],
    )


def _get_snr(cross_correlation: NDArray[np.float32], correlation_window: int, padding: int) -> float:
    """Computes the signal-to-noise ratio (SNR) for phase correlation.

    This function calculates the SNR by first identifying the peak correlation values for each frame and then excluding
    a region around the peak, as defined by the input 'padding'. The peak value is then normalized by the maximum
    correlation in the surrounding region. A higher SNR value indicates a stronger signal relative to noise.

    Args:
        cross_correlation: A 3D NumPy array containing the cross-correlation matrices for each frame.
        correlation_window: The size of the window used to define the correlation region. This is used to specify the
            area around the peak correlation to exclude in the SNR calculation.
        padding: The number of pixels around the peak correlation to exclude from the correlation region when
            calculating SNR.

    Returns:
        The signal-to-noise ratio (SNR) of the phase correlation for each frame.
    """
    # Extracts the region of the cross-correlation map, excluding the input padding around the borders.
    cross_correlation_without_padding = cross_correlation[:, padding:-padding, padding:-padding].reshape(
        cross_correlation.shape[0], -1
    )

    # Copies the cross correlation map to modify during the calculation.
    cross_correlation_copy = cross_correlation.copy()

    # Loops over the frames to identify the peak correlation and excludes the padding region around it.
    for frame, y_coordinate, x_coordinate in zip(
        cross_correlation_copy,
        *np.unravel_index(
            np.argmax(cross_correlation_without_padding, axis=1),
            (2 * correlation_window + 1, 2 * correlation_window + 1),
        ),
    ):
        # Sets the region around the peak correlation to 0 to exclude it from the SNR calculation.
        frame[
            y_coordinate : y_coordinate + 2 * padding,
            x_coordinate : x_coordinate + 2 * padding,
        ] = 0

    # Calculates the SNR as the ratio of the maximum correlation value to the maximum value in the surrounding region.
    snr: float = np.amax(cross_correlation_without_padding, axis=1) / np.maximum(
        1e-10, np.amax(cross_correlation_copy.reshape(cross_correlation.shape[0], -1), axis=1)
    )  # Ensures positivity for outlier cases by using a small epsilon (1e-10)

    # Returns the SNR.
    return snr


def phase_correlate(
    frames: NDArray[np.int16] | NDArray[np.float32],
    taper_mask: NDArray[np.float32],
    mask_offset: NDArray[np.float32],
    fft_reference_image: NDArray[np.complex64],
    snr_threshold: float,
    smoothing_kernel: NDArray[np.float32],
    y_blocks: list[NDArray[np.int_]],
    x_blocks: list[NDArray[np.int_]],
    maximum_shift: float,
    subpixel: int = 10,
    padding: int = 3,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Computes phase correlations for each block in the input frames relative to a reference image.

    This function applies Fourier transforms, convolution, and signal-to-noise ratio (SNR) thresholding to compute the
    phase correlation for each block. It returns the y- and x- shifts, as well as the correlation values for each block
    in the image, with subpixel refinement applied.

    Args:
        frames: A 3D NumPy array storing the frames to be registered.
        taper_mask: A NumPy array representing the taper mask to be applied to the frames.
        mask_offset: A NumPy array representing the mask offset for each block.
        fft_reference_image: A 2D NumPy array storing the Fourier-transformed reference image.
        snr_threshold: The threshold value for Signal-to-Noise Ratio (SNR) below which the correlation is discarded.
        smoothing_kernel: A 2D NumPy array used for smoothing the cross-correlation results.
        y_blocks: A list of NumPy arrays storing the y-coordinates for each block (start and end positions along the
            y-axis).
        x_blocks: A list of NumPy arrays storing the y-coordinates for each block (start and end positions along the
            x-axis).
        maximum_shift: The maximum shift allowed during registration, relative to the reference image.
        subpixel: The step size that defines the precision of subpixel registration. The default is 10.
        padding: The padding around the block used for correlation. The default is 3.

    Returns:
        A tuple of three elements. The first element is the refined y-shift for each frame, and the second element is
        the refined x-shift for each frame. The third element is the peak correlation value for each frame.

    """
    # Generates the sub-pixel upsampling matrix to refine shifts within each block.
    upsampling_matrix, upsampling_number = generate_upsampling_matrix(padding=3)

    # Extracts the number of frames and the height and width of the reference image.
    frame_number = frames.shape[0]
    height, width = fft_reference_image.shape[-2:]

    # Computes the maximum allowed correlation window, ensuring it fits inside the frame.
    correlation_window = int(np.minimum(np.round(maximum_shift), np.floor(np.minimum(height, width) / 2.0) - padding))

    # Determines the number of blocks for processing.
    block_number = len(y_blocks)

    # Initializes a NumPy array to store the extracted frame blocks for registration.
    frame_blocks = np.zeros((frame_number, block_number, height, width), "float32")

    # Loops over the blocks. Extracts the blocks from the frames based on 'y_blocks' and 'x_blocks' coordinates.
    for block_index in range(block_number):
        y_block, x_block = y_blocks[block_index], x_blocks[block_index]
        frame_blocks[:, block_index] = frames[:, y_block[0] : y_block[-1], x_block[0] : x_block[-1]]

    # Applies the taper mask and offset to each block.
    frame_blocks = add_multiply(frame_blocks, taper_mask, mask_offset)

    # Defines the batch size for processing the blocks.
    batch = min(64, frame_blocks.shape[1])

    # Loops through the batches of blocks.
    for block_index in np.arange(0, block_number, batch):
        # Determines the end index of the current batch.
        batch_end = min(frame_blocks.shape[1], block_index + batch)

        # Performs Fourier-domain convolution with the reference image for each batch of blocks.
        frame_blocks[:, block_index:batch_end] = convolve(
            frames=frame_blocks[:, block_index:batch_end],
            fft_reference_image=fft_reference_image[block_index:batch_end],
        )

    # Defines the size of the correlation window around the peak, including padding for subpixel refinement.
    half_window = correlation_window + padding

    # Rearranges the four corner quadrants of the frame blocks to handle wrap-around from FFT-based convolution.
    cross_correlation = np.real(
        np.block(
            [
                [
                    frame_blocks[:, :, -half_window:, -half_window:],
                    frame_blocks[:, :, -half_window:, : half_window + 1],
                ],
                [
                    frame_blocks[:, :, : half_window + 1, -half_window:],
                    frame_blocks[:, :, : half_window + 1, : half_window + 1],
                ],
            ]
        )
    )

    # Transposes the cross-correlation map so that the block dimension comes first for further processing.
    cross_correlation = cross_correlation.transpose(1, 0, 2, 3)

    # Flattens each block's cross-correlation map to prepare for matrix multiplication with the smoothing kernel.
    cross_correlation = cross_correlation.reshape(cross_correlation.shape[0], -1)

    # Calculates three levels of smoothing on the flattened cross-correlation map.
    smoothed_correlation_results = [
        # No smoothing applied.
        cross_correlation,
        # Applies smoothing kernel once on the cross-correlation map.
        smoothing_kernel @ cross_correlation,
        # Applies the smoothing kernel twice on the cross-correlation map.
        smoothing_kernel @ smoothing_kernel @ cross_correlation,
    ]

    # Reshapes the each of the smoothed correlation results to match the required block and frame dimensions.
    smoothed_correlation_results = [
        correlation_results.reshape(
            block_number,
            frame_number,
            2 * correlation_window + 2 * padding + 1,
            2 * correlation_window + 2 * padding + 1,
        )
        for correlation_results in smoothed_correlation_results
    ]

    # Stores the first smoothed result (no additional smoothing) for further processing.
    smoothed_cross_correlation = smoothed_correlation_results[0]

    # Loops over each block. Computes the signal-to-noise ratio (SNR) and applies them to the smoothed correlation map.
    for block_index in range(block_number):
        # Initializes a NumPy array to store the signal-to-noise ratio (SNR) for each frame.
        snr = np.ones(frame_number, "float32")

        # Loops over the smoothed correlation results. Computes the signal-to-noise ratio (SNR) and applies them
        # to the smoothed correlation map.
        for result_index, smoothed_correlation_result in enumerate(smoothed_correlation_results):
            # Creates a mask identifying which frames have an SNR value below the threshold.
            snr_below_threshold = snr < snr_threshold

            # If there are no frames below the SNR threshold, skips further smoothing for this block.
            if np.sum(snr_below_threshold) == 0:
                break

            # Extracts the cross-correlation maps for the current block and the frames that are below the SNR threshold.
            current_cross_correlation = smoothed_correlation_result[block_index, snr_below_threshold, :, :]

            # If this is not the first smoothed result, updates the final smoothed cross-correlation map.
            if result_index > 0:
                smoothed_cross_correlation[block_index, snr_below_threshold, :, :] = current_cross_correlation

            # Recomputes the SNR for the selected frames using current cross-correlation maps.
            snr[snr_below_threshold] = _get_snr(current_cross_correlation, correlation_window, padding)

    # Initializes NumPy arrays to store the final y- and x-shifts and correlation scores.
    upsampling_midpoint = upsampling_number // 2
    y_shifts = np.empty((frame_number, block_number), np.float32)
    x_shifts = np.empty((frame_number, block_number), np.float32)
    correlation_scores = np.empty((frame_number, block_number), np.float32)
    initial_y_shift = np.empty((block_number,), np.int32)
    initial_x_shift = np.empty((block_number,), np.int32)

    # Loops over the frames.
    for frame in range(frame_number):
        # Initializes a NumPy array to store a small window around each block's peak for subpixel refinement.
        cross_correlation_patch = np.empty((block_number, 2 * padding + 1, 2 * padding + 1), np.float32)

        # Loops over each block.
        for block_index in range(block_number):
            # Extracts the indices of the peak correlation score in the region of interest excluding padding.
            correlation_map_index = np.argmax(
                smoothed_cross_correlation[block_index, frame][padding:-padding, padding:-padding], axis=None
            )

            # Converts the flat index to 2D y- and x-coordinates in the correlation window.
            y_coordinates, x_coordinates = np.unravel_index(
                correlation_map_index, (2 * correlation_window + 1, 2 * correlation_window + 1)
            )

            # Extracts a small cross-correlation patch around the peak for subpixel upsampling.
            cross_correlation_patch[block_index] = smoothed_cross_correlation[block_index, frame][
                y_coordinates : y_coordinates + 2 * padding + 1,
                x_coordinates : x_coordinates + 2 * padding + 1,
            ]

            # Stores the initial integer y- and x-shifts relative to the center of the correlation window.
            initial_y_shift[block_index], initial_x_shift[block_index] = (
                y_coordinates - correlation_window,
                x_coordinates - correlation_window,
            )

        # Applies the upsampling matrix to the small correlation patches for subpixel precision.
        reshaped_cross_correlation = cross_correlation_patch.reshape(block_number, -1) @ upsampling_matrix

        # Stores the peak correlation value for each block in the current frame.
        correlation_scores[frame] = np.amax(reshaped_cross_correlation, axis=1)

        # Locates the peak correlation within the upsampled grid to determine subpixel y- and x-shift indices.
        y_shifts[frame], x_shifts[frame] = np.unravel_index(
            np.argmax(reshaped_cross_correlation, axis=1), (upsampling_number, upsampling_number)
        )

        # Refines the y-shift by considering subpixel precision and adjusts based on the initial shift.
        y_shifts[frame] = (y_shifts[frame] - upsampling_midpoint) / subpixel + initial_y_shift

        # Refines the x-shift by considering subpixel precision and adjusts based on the initial shift.
        x_shifts[frame] = (x_shifts[frame] - upsampling_midpoint) / subpixel + initial_x_shift

    # Returns the computed y-shift, x-shift, and peak correlation score of each frame.
    return y_shifts, x_shifts, correlation_scores


@njit(
    [
        "(int16[:, :],float32[:,:], float32[:,:], float32[:,:])",
        "(float32[:, :],float32[:,:], float32[:,:], float32[:,:])",
    ],
    cache=True,
)
def _map_coordinates(
    frame: NDArray[Any],
    target_y_coordinates: NDArray[np.float32],
    target_x_coordinates: NDArray[np.float32],
    shifted_coordinates: NDArray[np.float32],
) -> None:
    """Performs an in-place bilinear transformation of the input frame based on the target y- and x-coordinates.

    This function calculates the transformed pixel values at the target coordinates using bilinear interpolation. It
    modifies the 'shifted_coordinates' in place by computing the transformed pixel values from the input 'frame'.

    Args:
        frame: A 2D NumPy array storing the frame to be transformed.
        y_coordinates: A 2D NumPy array storing the target y-coordinates where pixels should be moved to.
        x_coordinates: A 2D NumPy array storing the target x-coordinates where pixels should be moved to.
        shifted_coordinates: A 2D NumPy array to store the output of the bilinear transformation.

    """
    # Extracts the height and width of the input frame.
    height, width = frame.shape

    # Converts the floating-point coordinates to integer indices for bilinear interpolation.
    floored_y_coordinates = target_y_coordinates.astype(np.int32)
    floored_x_coordinates = target_x_coordinates.astype(np.int32)

    # Computes the fractional part of the coordinates for interpolation.
    y_fractions = target_y_coordinates - floored_y_coordinates
    x_fractions = target_x_coordinates - floored_x_coordinates

    # Loops over pixels in the frame and performs bilinear interpolation.
    for y_pixel in range(floored_y_coordinates.shape[0]):
        for x_pixel in range(floored_y_coordinates.shape[1]):
            # Ensures the pixel indices are within the frame boundaries.
            y_coordinates = min(height - 1, max(0, floored_y_coordinates[y_pixel, x_pixel]))
            x_coordinates = min(width - 1, max(0, floored_x_coordinates[y_pixel, x_pixel]))

            # Calculates the neighboring pixel indices for bilinear interpolation.
            neighbor_y_coordinates = min(height - 1, y_coordinates + 1)
            neighbor_x_coordinates = min(width - 1, x_coordinates + 1)

            # Extracts the fractional parts of the coordinates for interpolation.
            y_fraction = y_fractions[y_pixel, x_pixel]
            x_fraction = x_fractions[y_pixel, x_pixel]

            # Performs bilinear interpolation. Computes the weighted average of the four neighboring pixels.
            shifted_coordinates[y_pixel, x_pixel] = (
                np.float32(frame[y_coordinates, x_coordinates]) * (1 - y_fraction) * (1 - x_fraction)
                + np.float32(frame[y_coordinates, neighbor_x_coordinates]) * (1 - y_fraction) * x_fraction
                + np.float32(frame[neighbor_y_coordinates, x_coordinates]) * y_fraction * (1 - x_fraction)
                + np.float32(frame[neighbor_y_coordinates, neighbor_x_coordinates]) * y_fraction * x_fraction
            )


@njit(
    [
        "int16[:, :,:], float32[:,:,:], float32[:,:,:], float32[:,:], float32[:,:], float32[:,:,:]",
        "float32[:, :,:], float32[:,:,:], float32[:,:,:], float32[:,:], float32[:,:], float32[:,:,:]",
    ],
    parallel=True,
    cache=True,
)
def _shift_coordinates(
    frames: NDArray[Any],
    y_shifts: NDArray[np.float32],
    x_shifts: NDArray[np.float32],
    y_meshgrid: NDArray[np.float32],
    x_meshgrid: NDArray[np.float32],
    shifted_frames: NDArray[np.float32],
) -> None:
    """Shifts the frames based on the provided y- and x-shift coordinates, using bilinear interpolation.

    This function takes the input frames, applies the y- and x-shifts (calculated for each pixel in each frame), and
    stores the resulting transformed frames in the specified 'shifted_frames' array. The shifts are applied using the
    provided meshgrid values for each frame.

    Args:
        frames: A 3D NumPy array storing the frames to be transformed.
        y_shifts: A 2D NumPy array storing the y-shift values for each pixel in each frame.
        x_shifts: A 2D NumPy array storing the x-shift values for each pixel in each frame.
        y_meshgrid: A 2D NumPy array storing the y-coordinates of the meshgrid used for interpolation.
        x_meshgrid: A 2D NumPy array storing the x-coordinates of the meshgrid used for interpolation.
        shifted_frames: A 3D NumPy array to store the shifted frames after applying the interpolation.

    """
    # Loops over frames and applies the shift transformation.
    for frame_index in prange(frames.shape[0]):  # prange for parallelization support
        # Applies the coordinate transformation for each frame using the meshgrids and frame-specific shifts.
        _map_coordinates(
            frame=frames[frame_index],
            target_y_coordinates=y_meshgrid + y_shifts[frame_index],
            target_x_coordinates=x_meshgrid + x_shifts[frame_index],
            shifted_coordinates=shifted_frames[frame_index],
        )


@njit(
    (float32[:, :, :], float32[:, :, :], float32[:, :], float32[:, :], float32[:, :, :], float32[:, :, :]),
    parallel=True,
    cache=True,
)
def _interpolate_blocks(
    y_shifts: NDArray[np.float32],
    x_shifts: NDArray[np.float32],
    y_meshgrid: NDArray[np.float32],
    x_meshgrid: NDArray[np.float32],
    interpolated_y_shifts: NDArray[np.float32],
    interpolated_x_shifts: NDArray[np.float32],
) -> None:
    """Performs interpolation to calculate the shift coordinates for each frame and each block.

    This function uses bilinear interpolation to calculate the shift coordinates (y- and x-shifts) for each block in
    each frame, based on the shifts provided for each block. The meshgrids for the y- and x-coordinates are used to
    compute the exact positions based on the shifts.

    Args:
        y_shifts: A NumPy array storing the shifts in the y-direction for each block in each frame.
        x_shifts: A NumPy array storing the shifts in the x-direction for each block in each frame.
        y_meshgrid: A 2D NumPy array storing the y-coordinates of the meshgrid.
        x_meshgrid: A 2D NumPy array storing the x-coordinates of the meshgrid.
        interpolated_y_shifts: A 3D NumPy array storing the interpolated y-shift values for each frame and the
            corresponding meshgrid.
        interpolated_x_shifts: A 3D NumPy array storing the interpolated x-shift values for each frame and the
            corresponding meshgrid.
    """
    # Loops over frames to interpolate shifts for all blocks.
    for frame_index in prange(y_shifts.shape[0]):  # prange for parallelization support
        # Interpolates the y-shifts for the current frame's blocks and updates the 'interpolated_y_shifts' array.
        _map_coordinates(
            frame=y_shifts[frame_index],
            target_y_coordinates=y_meshgrid,
            target_x_coordinates=x_meshgrid,
            shifted_coordinates=interpolated_y_shifts[frame_index],
        )
        # Interpolates the x-shifts for the current frame's blocks and updates the 'interpolated_x_shifts' array.
        _map_coordinates(
            frame=x_shifts[frame_index],
            target_y_coordinates=y_meshgrid,
            target_x_coordinates=x_meshgrid,
            shifted_coordinates=interpolated_x_shifts[frame_index],
        )


def _upsample_block_shifts(
    height: int,
    width: int,
    block_number: list[int],
    y_blocks: list[NDArray[np.int_]],
    x_blocks: list[NDArray[np.int_]],
    y_shifts: NDArray[np.float32],
    x_shifts: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Upsamples block shifts into full pixel shift maps for later bilinear interpolation.

    This function takes the block shifts 'y_shifts' and 'x_shifts' (defined per block), and interpolates them to
    generate a full pixel shift map. The result is used for precise shifting of the frames later using bilinear
    interpolation.

    Args:
        height: The height (in pixels) of the frame.
        width: The width (in pixels) of the frame.
        block_number: A tuple storing the number of blocks in the y- and x-directions.
        y_blocks: A list of NumPy arrays storing the y-coordinates for each block (start and end positions along the
            y-axis).
        x_blocks: A list of NumPy arrays storing the y-coordinates for each block (start and end positions along the
            x-axis).
        y_shifts: A NumPy array storing the shifts in the y-direction for each block in each frame.
        x_shifts: A NumPy array storing the shifts in the x-direction for each block in each frame.

    Returns:
        A tuple of two elements. The first element is a NumPy array storing the upsampled y-shifts for each pixel, and
        the second element is a NumPy array storing the upsampled x-shifts for each pixel.
    """
    # Calculates the mean (center) coordinates of each blocks in the y- and x-directions.
    y_block_centers = np.array(y_blocks[:: block_number[1]]).mean(
        axis=1
    )  # this recovers the coordinates of the meshgrid from (yblock, xblock)
    x_block_centers = np.array(x_blocks[: block_number[1]]).mean(axis=1)

    # Interpolates from the block centers to all pixel positions in the y- and x-directions.
    y_coordinates = np.interp(np.arange(height), y_block_centers, np.arange(y_block_centers.size)).astype(np.float32)
    x_coordinates = np.interp(np.arange(width), x_block_centers, np.arange(x_block_centers.size)).astype(np.float32)

    # Creates the meshgrid from the interpolated y- and x-coordinates.
    x_meshgrid, y_meshgrid = np.meshgrid(x_coordinates, y_coordinates)

    # Extracts the number of frames.
    frame_number = y_shifts.shape[0]

    # Reshapes the y- and x-shifts to be suitable for interpolation.
    y_shifts = y_shifts.reshape(frame_number, block_number[0], block_number[1])
    x_shifts = x_shifts.reshape(frame_number, block_number[0], block_number[1])

    # Initializes NumPy arrays to store the upsampled y- and x-shifts for each frame.
    upsampled_y_shifts = np.zeros((frame_number, height, width), np.float32)
    upsampled_x_shifts = np.zeros((frame_number, height, width), np.float32)

    # Interpolates the block shifts to full pixel shifts using the meshgrid and upsampling.
    _interpolate_blocks(
        y_shifts=y_shifts,
        x_shifts=x_shifts,
        y_meshgrid=y_meshgrid,
        x_meshgrid=x_meshgrid,
        interpolated_y_shifts=upsampled_y_shifts,
        interpolated_x_shifts=upsampled_x_shifts,
    )

    # Returns the upsampled y- and x-shifts.
    return upsampled_y_shifts, upsampled_x_shifts


def transform_frames(
    frames: NDArray[np.int16] | NDArray[np.float32],
    block_number: list[int],
    y_blocks: list[NDArray[np.int_]],
    x_blocks: list[NDArray[np.int_]],
    y_shifts: NDArray[np.float32],
    x_shifts: NDArray[np.float32],
    bilinear: bool = True,
) -> NDArray[np.float32]:
    """Applies a piecewise affine transformation to the input frames using the input 'y_shifts' and 'x_shifts'.

    This function applies the upsampled shifts to the frames using bilinear interpolation (or nearest-neighbor if
    specified) to transform the frames based on the block shifts. The transformation is performed pixel-wise for
    subpixel precision.

    Args:
        frames: A 3D NumPy array storing the frames to be registered.
        block_number: A tuple storing the number of blocks in the y- and x-directions.
        y_blocks: A list of NumPy arrays storing the y-coordinates for each block (start and end positions along the
            y-axis).
        x_blocks: A list of NumPy arrays storing the y-coordinates for each block (start and end positions along the
            x-axis).
        y_shifts: A NumPy array storing the shifts in the y-direction for each block in each frame.
        x_shifts: A NumPy array storing the shifts in the x-direction for each block in each frame.
        bilinear: Determines whether to perform bilinear interpolation. If set to False, nearest-neighbor interpolation
            is applied. The default is True.

    Returns:
        A 3D NumPy array storing the transformed frames after applying the shifts.
    """
    # Extracts the height and width of the input frames.
    _, height, width = frames.shape

    # Upsamples the input block shifts into full pixel shifts.
    upsampled_y_shifts, upsampled_x_shifts = _upsample_block_shifts(
        height=height,
        width=width,
        block_number=block_number,
        y_blocks=y_blocks,
        x_blocks=x_blocks,
        y_shifts=y_shifts,
        x_shifts=x_shifts,
    )

    # If nearest-neighbor interpolation is enabled, rounds the shift values to integer values.
    if not bilinear:
        upsampled_y_shifts = np.round(upsampled_y_shifts)
        upsampled_x_shifts = np.round(upsampled_x_shifts)

    # Creates a meshgrid of the y- and x-axes of the frame.
    x_meshgrid, y_meshgrid = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))

    # Initializes a NumPy array to store the shifted frames.
    shifted_frames = np.zeros_like(frames, dtype=np.float32)

    # Applies the shift transformations using bilinear (or nearest-neighbor) interpolation.
    _shift_coordinates(frames, upsampled_y_shifts, upsampled_x_shifts, y_meshgrid, x_meshgrid, shifted_frames)

    # Returns the transformed frames.
    return shifted_frames
