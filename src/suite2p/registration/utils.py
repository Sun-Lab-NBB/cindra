"""Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu."""

from typing import Any
from functools import lru_cache

from numba import complex64, vectorize
import numpy as np
import torch
from numpy.fft import ifftshift  # , fft2, ifft2
from scipy.fft import next_fast_len  # , fft2, ifft2
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from ataraxis_base_utilities import console

try:
    # pytorch > 1.7
    from torch.fft import (
        fft2 as torch_fft2,
        ifft2 as torch_ifft2,
    )
except:
    # pytorch <= 1.7
    raise ImportError("pytorch version > 1.7 required")

epsilon = torch.complex(torch.tensor(1e-5), torch.tensor(0.0))


def _fft_2d(data: NDArray[Any]) -> NDArray[np.complex64]:
    """Computes the 2D Fast Fourier Transform (FFT) of the input data over the last two dimensions using PyTorch.

    Args:
        data: A NumPy array storing the input data to be transformed. The last two dimensions are spatial dimensions
            (height and width) over which the FFT will be computed.

    Returns:
        A NumPy array storing the 2D Fourier-transformed result of the input data. The shape of the output matches the
        input data.
    """
    # Converts the input NumPy array to a PyTorch tensor.
    tensor_data = torch.from_numpy(data)

    # Performs 2D FFT on the input data along the last two dimensions.
    tensor_data = torch_fft2(tensor_data, dim=(-2, -1))

    # Moves the result back to CPU, converts to NumPy array, and casts to complex64 for compatibility.
    fft_data: NDArray[np.complex64] = tensor_data.cpu().numpy().astype("complex64")

    # Returns the Fourier-transformed data.
    return fft_data


def convolve(frames: NDArray[np.complex64], fft_reference_image: NDArray[np.complex64]) -> NDArray[np.float32]:
    """Convolves the input frames with a 2D reference image using Fourier domain operations.

    This function performs the convolution of each frame in the input 3D array with a Fourier-transformed reference
    image. The frames are first transformed into the Fourier domain using FFT, then the convolution is performed by
    multiplying the Fourier-transformed frames with the Fourier-transformed reference image. Finally, the inverse FFT
    is applied to transform the result back to the spatial domain.

    Args:
        frames: A 3D NumPy array storing the frames to be convolved.
        fft_reference_image: A 2D NumPy array storing the Fourier-transformed reference image to convolve with the
        frames.

    Returns:
        A 3D NumPy array storing the convolved frames.
    """
    # Converts the input frames to a PyTorch tensor.
    fft_frames = torch.from_numpy(frames)

    # Performs 2D FFT on each frame along the last two dimensions.
    fft_frames = torch_fft2(fft_frames, dim=(-2, -1))

    # Normalizes to avoid division by zero during multiplication with the reference image.
    fft_frames /= epsilon + torch.abs(fft_frames)

    # Performs element-wise multiplication with the FFT of the reference image.
    fft_frames *= torch.from_numpy(fft_reference_image)

    # Performs inverse 2D FFT to transform the result back to the spatial domain.
    fft_frames = torch.real(torch_ifft2(fft_frames, dim=(-2, -1)))

    # Returns the convolved frames as a NumPy array.
    return fft_frames.numpy()


@vectorize(
    ["complex64(int16, float32, float32)", "complex64(float32, float32, float32)"],
    nopython=True,
    target="parallel",
    cache=True,
)
def add_multiply(
    data: NDArray[np.int16] | NDArray[np.float32], multiplier: NDArray[np.float32], addend: NDArray[np.float32]
) -> NDArray[np.complex64]:
    """Multiplies the input data by 'multiplier' and then adds 'adder'.

    Args:
        data: A NumPy array storing the data to be multiplied.
        multiplier: A scalar or array used to multiply each of the input elements.
        addend: A scalar or array that is added to each input element after the multiplication.

    Returns:
        The resulting NumPy array of type complex64 after multiplication and addition.
    """
    return np.complex64(np.float32(data) * multiplier + addend)


def combine_shifts_across_batches(
    shifts: list[tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]]
    | list[tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]],
    rigid: bool,
) -> (
    tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]
    | tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]
):
    """Combines shifts from different batches into single arrays.

    This function handles rigid and non-rigid cases by either stacking the shift values horizontally or vertically.

    Args:
        shifts: A list of tuples, where each tuple contains three arrays corresponding to the y-shifts, x-shifts,
            and cross-correlation values for every batch.
        rigid: Determines whether to combine the shift values for a rigid transformation (True) or a non-rigid
            transformation (False). If set to 'True', the shift values are stacked horizontally. Otherwise, they are
            stacked vertically.

    Returns:
        A tuple of three elements. The first element is a NumPy array of the combined y-shifts, and the second element
        is a NumPy array of the combined x-shifts. The third element is a NumPy array of the combined
        cross-correlation values.
    """
    # Initializes lists to store the combined shift and correlation values.
    y_shifts, x_shifts, correlations = [], [], []

    # Loops over batches and appends each batch's shift and correlation arrays to the respective lists.
    for batch in shifts:
        y_shifts.append(batch[0])
        x_shifts.append(batch[1])
        correlations.append(batch[2])

    # Combines the shift values either horizontally or vertically depending on the 'rigid' flag.
    if rigid:
        return np.hstack(y_shifts), np.hstack(x_shifts), np.hstack(correlations)
    else:
        return np.vstack(y_shifts), np.vstack(x_shifts), np.vstack(correlations)


def _mean_centered_meshgrid(height: int, width: int) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """Generates a mean-centered meshgrid for spatial operations.

    This function creates two 2D NumPy arrays representing the y- and x-coordinates for a grid, then shifts those values
    so that the mean of both the y- and x-coordinates is centered at 0. This centered grid can be useful for spatial
    filtering, registration, or other operations requiring a mean-centered coordinate system.

    Args:
        height: The height of the meshgrid (the number of rows in the resulting grid).
        width: The width of the meshgrid (the number of columns in the resulting grid).

    Returns:
        A tuple of two elements. The first element is a 2D NumPy array storing the y-coordinates of the mean-centered
        meshgrid. The second element is a 2D NumPy array storing the x-coordinates of the mean-centered meshgrid.

    """
    # Generates 1D NumPy arrays storing the height (rows) and width (columns) range.
    height_range = np.arange(0, height)
    width_range = np.arange(0, width)

    # Centers the height and width ranges by subtracting their mean values. This shifts all the values so that the mean
    # is 0.
    height_range = np.abs(height_range - height_range.mean())
    width_range = np.abs(width_range - width_range.mean())

    # Creates a 2D meshgrid using the centered height and width ranges.
    x_coordinates, y_coordinates = np.meshgrid(width_range, height_range)

    # Returns the y- and x-coordinates of the mean-centered meshgrid.
    return y_coordinates, x_coordinates


def gaussian_fft(sigma: float, height: int, width: int) -> NDArray[np.float32]:
    """Generates a Gaussian filter in the Fourier domain with the input standard deviation and dimensions.

    This function constructs a Gaussian filter, centers it using 'ifftshift', then computes its 2D Fast Fourier
    Transform (FFT).

    Args:
        sigma: The standard deviation of the Gaussian filter. This controls the width of the filter. Smaller values
            result in a narrower filter, while larger values result in a wider filter.
        height: The height of the frame (number of rows in the output) for which the Gaussian filter is generated.
        width: The width of the frame (number of columns in the output) for which the Gaussian filter is generated.

    Returns:
        A 2D NumPy array storing the Gaussian filter in the Fourier domain.
    """
    # Generates the y- and x-coordinates of a mean-centered meshgrid based on the dimensions of the frame.
    y_coordinates, x_coordinates = _mean_centered_meshgrid(height=height, width=width)

    # Computes the Gaussian function along the x- and y-directions
    x_gaussian = np.exp(-np.square(x_coordinates / sigma) / 2)
    y_gaussian = np.exp(-np.square(y_coordinates / sigma) / 2)

    # Multiplies y- and x-axes' Gaussian functions to create a 2D Gaussian filter.
    gaussian_filter = y_gaussian * x_gaussian

    # Normalizes the filter so that the sum of all its elements equals 1.
    gaussian_filter /= gaussian_filter.sum()

    # Centers the Gaussian, computes the 2D FFT of the centered Gaussian, and extracts its real part.
    fft_gaussian_filter = np.real(_fft_2d(ifftshift(gaussian_filter)))

    # Returns the Gaussian filter in the Fourier domain.
    return fft_gaussian_filter


def spatial_taper(sigma: float, height: int, width: int) -> NDArray[np.float32]:
    """Computes a spatial taper using a Gaussian function with the input standard deviation 'sigma' on the edges of the
    frame.

    This function generates a taper mask that reduces the intensity of the pixels at the edges based on the Gaussian
    function.

    Args:
        sigma: The standard deviation of the Gaussian function, which controls the width of the taper effect.
        height: The height (in pixels) of the frame.
        width: The width (in pixels) of the frame.

    Returns:
        A 2D NumPy array storing pixel values that gradually decrease towards the edges, forming a taper effect.
    """
    # Generates the y- and x-coordinates of a mean-centered meshgrid based on the dimensions of the frame.
    y_coordinates, x_coordinates = _mean_centered_meshgrid(height=height, width=width)

    # Calculates the y- and x-midpoint, which is the position where the tapering should start for each direction.
    y_midpoint = ((height - 1) / 2) - 2 * sigma
    x_midpoint = ((width - 1) / 2) - 2 * sigma

    # Calculates the y- and x-masks based on the Gaussian function applied to y- and x-coordinates, respectively.
    y_mask = 1.0 / (1.0 + np.exp((y_coordinates - y_midpoint) / sigma))
    x_mask = 1.0 / (1.0 + np.exp((x_coordinates - x_midpoint) / sigma))

    # Combines the y- and x-masks to create the spatial taper mask.
    taper_mask: NDArray[np.float32] = y_mask * x_mask

    # Returns the spatial taper mask.
    return taper_mask


def temporal_smooth(frames: NDArray[Any], sigma: float) -> NDArray[Any]:
    """Applies Gaussian filtering along the first dimension to smooth the input data.

    Args:
        frames: A 3D NumPy array storing the frames to be registered.
        sigma: The standard deviation of the Gaussian filter, which controls the smoothing level. Larger values result
            in more smoothing.

    Returns:
        A 3D NumPy array with the same shape as the input 'frames' but with smoothing applied along the first dimension.
    """
    return gaussian_filter1d(input=frames, sigma=sigma, axis=0)


def spatial_smooth(frames: NDArray[Any], window_size: int) -> NDArray[Any]:
    """Applies spatial smoothing to the input frames using the cumulative sum over spatial dimensions (height and width)
    with a specified window size.

    This function smooths the input data by applying a cumulative sum along the spatial dimensions of the frames. The
    smoothing is achieved by computing the cumulative sum along both the x- and y- axes, and then subtracting the
    cumulative sums to get the smoothed result. The window size determines the extent of the smoothing.

    Args:
        frames: A 2D or 3D NumPy array storing the frames to be smoothed.
        window_size: The size of the smoothing window. Must be an even integer.

    Returns:
        A NumPy array storing the smoothed frames. The shape will be the same as the input frames.

    Raises:
        ValueError: If the input 'window_size' is not an even integer.
    """
    # Verifies the provided window size is an even integer.
    if window_size and window_size % 2:
        message = f"Filter window must be an even integer."
        console.error(message=message, error=ValueError)

    # If the input frame(s) are 2D, adds a single dimension to make it 3D.
    if frames.ndim == 2:
        frames = frames[np.newaxis, :, :]

    # Pads the frames with zeros to ensure boundaries can be smoothed without errors.
    padding = window_size // 2  # Half of the window size is added to both sides of the frame
    padded_frames = np.pad(frames, ((0, 0), (padding, padding), (padding, padding)), mode="constant", constant_values=0)

    # Calculates the cumulative sum along the y-axis and x-axis to apply smoothing.
    summed_frames: NDArray[np.float32] = padded_frames.cumsum(axis=1).cumsum(axis=2, dtype=np.float32)

    # Subtracts to get the sum over the window size along the y- and x-axes.
    summed_frames = summed_frames[:, :, window_size:] - summed_frames[:, :, :-window_size]  # Smoothing over y-axis
    summed_frames = summed_frames[:, window_size:, :] - summed_frames[:, :-window_size, :]  # Smoothing over x-axis

    # Normalizes the smoothed data by dividing by the square of the window size.
    summed_frames /= window_size**2

    # Squeezes to remove single-dimensional entires and returns the result.
    return summed_frames.squeeze()


def spatial_high_pass(frames: NDArray[Any], window_size: int) -> NDArray[Any]:
    """Applies a high-pass filter to the input frames by subtracting the low-pass filtered frames.

    This function removes the low-frequency signals from the input frames by applying spatial smoothing (using a
    low-pass filter) and then subtracting the smoothed low-pass filtered frames from the original frames, which results
    in only the high-frequency details remaining.

    Args:
        frames: A 2D or 3D NumPy array storing the frames to be filtered.
        window_size: The size of the smoothing window used for the low-pass filter.

    Returns:
        A NumPy array storing the high-pass filtered frames. The shape will be the same as the input frames.
    """
    # If the input frame(s) are 2D, adds a single dimension to make it 3D.
    if frames.ndim == 2:
        frames = frames[np.newaxis, :, :]

    # Subtracts the low-pass filtered frames from the original frames to get the high-pass filtered frames.
    filtered_frames: NDArray[Any] = frames - (
        spatial_smooth(frames, window_size)
        / spatial_smooth(np.ones(shape=(1, frames.shape[1], frames.shape[2]), dtype="float32"), window_size)
    )

    # Squeezes to remove single-dimensional entires and returns the result.
    return filtered_frames.squeeze()


def complex_fft_2d(reference_image: NDArray[Any]) -> NDArray[Any]:
    """Computes the complex conjugate of the 2D Fast Fourier Transform (FFT) of the input reference image.

        Args:
            reference_image: A 2D NumPy array storing the input reference image to be transformed.
    s
        Returns:
            A 2D NumPy array storing the complex conjugate of the 2D FFT of the input image.
    """
    # Computes the 2D FFT and returns its complex conjugate.
    conjugate: NDArray[Any] = np.conj(_fft_2d(reference_image))
    return conjugate


def _smoothed_gaussian_kernel(
    y_coordinates: NDArray[Any], x_coordinates: NDArray[Any], smoothing_width: float = 0.85
) -> NDArray[Any]:
    """Computes a 2D Gaussian kernel based on the pairwise distances between the input y- and x-coordinates with an
    optional smoothing parameter for adjusting the kernel width.

    Args:
        y_coordinates: A NumPy array storing the y-coordinates of the grid.
        x_coordinates: A NumPy array storing the x-coordinates of the grid.
        smooth_width: A smoothing parameter that controls the width of the Gaussian kernel (best between values of 0.5
            and 1.0). Smaller values result in a  narrower kernel while larger values correspond to a wider kernel. The
            default is 0.85.

    Returns:
        A 2D NumPy array storing the Gaussian kernel where each element corresponds to the similarity between the
        pairwise distances of coordinates in the y- and x-coordinate grids.
    """
    # Creates mesh grids for the y- and x-coordinates to calculate pairwise distances.
    y_mesh_grid_coordinates_1, y_mesh_grid_coordinates_2 = np.meshgrid(y_coordinates, y_coordinates)
    x_mesh_grid_coordinates_1, x_mesh_grid_coordinates_2 = np.meshgrid(x_coordinates, x_coordinates)

    # Calculates the pairwise differences between the y- and x-coordinates.
    y_difference = x_mesh_grid_coordinates_2.reshape(-1, 1) - y_mesh_grid_coordinates_2.reshape(1, -1)
    x_difference = x_mesh_grid_coordinates_1.reshape(-1, 1) - y_mesh_grid_coordinates_1.reshape(1, -1)

    # Computes the Gaussian kernel based on the pairwise differences.
    gaussian_kernel: NDArray[Any] = np.exp(-(x_difference**2 + y_difference**2) / (2 * smoothing_width**2))

    # Returns the kernel.
    return gaussian_kernel


def normalized_gaussian_kernel(y_coordinates: NDArray[np.int_], x_coordinates: NDArray[np.int_]) -> NDArray[np.float32]:
    """Computes a 2D normalized Gaussian kernel based on the pairwise distances between the input y- and x-coordinates.

    This function computes a kernel

    Args:
        y_coordinates: A NumPy array storing the y-coordinates of the grid.
        x_coordinates: A NumPy array storing the x-coordinates of the grid.

    Returns:
        A 2D NumPy array storing the normalized Gaussian kernel where each element corresponds to the similarity
        between the two points in the grid. The kernel is normalized such that the sum of its elements is 1.
    """
    # Creates a mesh grid for the y- and x-coordinates to represent all pairwise combinations.
    y_coordinates, x_coordinates = np.meshgrid(x_coordinates, y_coordinates)

    # Flattens the mesh grid and reshapes to 1D arrays for pairwise distance computation.
    y_coordinates = y_coordinates.flatten().reshape(1, -1)
    x_coordinates = x_coordinates.flatten().reshape(1, -1)

    # Computes the Gaussian kernel based on the pairwise differences.
    gaussian_kernel: NDArray[np.float32] = np.exp(
        -((y_coordinates - y_coordinates.T) ** 2 + (x_coordinates - x_coordinates.T) ** 2)
    )

    # Normalizes the kernel by summing along one axis and dividing by the sum.
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel, axis=0)

    # Returns the kernel.
    return gaussian_kernel


@lru_cache(maxsize=5)
def generate_upsampling_matrix(padding: int, subpixel: int = 10) -> tuple[NDArray[np.float32], int]:
    """Generates an upsampling matrix using Gaussian kernels for subpixel registration.

    This function creates an upsampling matrix by using smoothed Gaussian kernels to increase the resolution of the
    correlation matrix using the specified padding and subpixel step size. The output matrix is used for subpixel
    registration in phase correlation.

    Args:
        padding: The padding width used to define the y- and x-dimensions of the Gaussian kernel.
        subpixel: The step size that defines the precision of subpixel registration.

    Returns:
        A tuple of two elements. The first element is the upsampling matrix, and the second element is the number of
        upsampling points.
    """
    # Defines the range for the smoothed Gaussian kernels using the input padding.
    kernel_range = np.arange(-padding, padding + 1)

    # Defines the upsampling kernel range based on the subpixel step size.
    upsampling_range = np.arange(-padding, padding + 0.001, 1.0 / subpixel)

    # Calculates the number of upsampling points.
    upsampling_number = upsampling_range.shape[0]

    # Generates the upsampling matrix using the inverse of the smoothed Gaussian kernel.
    upsampling_matrix = np.linalg.inv(
        _smoothed_gaussian_kernel(y_coordinates=kernel_range, x_coordinates=kernel_range)
    ) @ _smoothed_gaussian_kernel(y_coordinates=upsampling_range, x_coordinates=kernel_range)

    # Returns the upsampling matrix and the number of upsampling points.
    return upsampling_matrix, upsampling_number
