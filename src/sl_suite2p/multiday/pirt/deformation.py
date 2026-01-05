"""Unified deformation module for image registration.

This module provides:
- Diffusion filtering functions (diffusionkernel, diffuse)
- Image transformation functions (warp, resize, zoom, deform_backward, deform_forward)
- The `Deformation` class for representing and applying deformations

Copyright 2010-2017 (C) Almar Klein (original pirt library)
"""

import math

import numba
import numpy as np
import scipy.ndimage
from numpy.typing import NDArray
from ataraxis_base_utilities import console

from .spline_grid import SplineGrid, compute_cardinal_coefficients


# =============================================================================
# Diffusion Filtering
# =============================================================================


def diffusionkernel(sigma, N=4, returnt=False):
    """Create a discrete analog to the continuous Gaussian kernel.

    Based on Lindeberg's discrete diffusion kernel using modified Bessel functions.

    Parameters
    ----------
    sigma : float
        The smoothing parameter (scale).
    N : int
        Tail length factor relative to sigma.
    returnt : bool
        If True, also return the t-values.

    Returns
    -------
    k : ndarray
        The diffusion kernel.
    t : ndarray (optional)
        The t-values if returnt=True.
    """
    sigma = float(sigma)
    sigma2 = sigma * sigma

    # Tail length
    if N > 0:
        nstart = int(np.ceil(N * sigma)) + 1
    else:
        nstart = abs(N) + 1

    # Allocate kernel and times
    t = np.arange(-nstart, nstart + 1, dtype="float64")
    k = np.zeros_like(t)

    # Initialize
    n = nstart
    k[n + nstart] = 0
    n = n - 1
    k[n + nstart] = 0.01

    # Iterate using recurrence relation
    for n in range(nstart - 1, 0, -1):
        k[(n - 1) + nstart] = 2 * n / sigma2 * k[n + nstart] + k[(n + 1) + nstart]

    # Use symmetric right part
    k[:nstart] = np.flipud(k[-nstart:])

    # Remove zero tails
    k = k[1:-1]
    t = t[1:-1]

    # Normalize
    k = k / k.sum()

    if returnt:
        return k, t
    return k


def diffuse(L, sigma, mode="nearest"):
    """Apply discrete diffusion filtering.

    Uses Lindeberg's discrete diffusion kernel for true scale-space properties.
    After diffusion, derivatives can be computed with simple operators:
      * Lx = 0.5 * (L[x+1] - L[x-1])
      * Lxx = L[x+1] - 2*L[x] + L[x-1]

    Parameters
    ----------
    L : ndarray
        The input data to filter.
    sigma : scalar or list
        The smoothing parameter, can be given per dimension.
    mode : str
        Border handling mode for convolution.

    Returns
    -------
    ndarray
        The diffused data.
    """
    try:
        sigma = [sig for sig in sigma]
    except TypeError:
        sigma = [sigma for _ in range(L.ndim)]

    if len(sigma) != L.ndim:
        raise ValueError("Number of sigmas must match data dimensions.")

    for d in range(L.ndim):
        k = diffusionkernel(sigma[d])
        L = scipy.ndimage.convolve1d(L, k, d, mode=mode)

    return L


# =============================================================================
# Transformation Functions
# =============================================================================


def make_samples_absolute(samples: tuple[NDArray, ...]) -> tuple[NDArray, ...]:
    """Converts relative deformation coordinates to absolute sample locations using NumPy broadcasting.

    This function transforms relative deformation arrays expressed in world coordinates (x,y,z order) into absolute
    sample location arrays in pixel coordinates. The function assumes the sampling of the data matches the sample
    arrays.

    Notes:
        This function is intended for samples that represent a deformation. The number of dimensions of each array
        should match the number of arrays.

    Args:
        samples: A tuple of arrays representing relative deformation in world coordinates (x,y,z order).

    Returns:
        A tuple of arrays representing absolute sample locations in pixel coordinates.

    Raises:
        ValueError: If the number of dimensions of each array does not match the number of arrays.
    """
    ndim = len(samples)
    absolute_samples = []

    for i in range(ndim):
        sample_array = samples[i]

        # Validates input array dimensions.
        if sample_array.ndim != ndim:
            message = (
                "Unable to compute absolute sample positions. The number of dimensions of each array should match "
                "the number of arrays."
            )
            console.error(message=message, error=ValueError)

        # Gets dimension corresponding to this sampling array (reversed order: x,y,z -> z,y,x in array indices).
        d = ndim - i - 1

        # Gets sampling factor for this dimension.
        sampling = 1.0
        if hasattr(sample_array, "sampling"):
            sampling = sample_array.sampling[d]

        # Creates index grid for dimension d using broadcasting. Shape is [1, 1, ..., shape[d], ..., 1] with
        # shape[d] at position d.
        shape = [1] * ndim
        shape[d] = sample_array.shape[d]
        indices = np.arange(sample_array.shape[d], dtype=sample_array.dtype).reshape(shape)

        # Computes absolute positions: index + offset/sampling (broadcasts across all dimensions).
        result = indices + sample_array / sampling
        absolute_samples.append(result)

    return tuple(absolute_samples)


@numba.njit(cache=True, inline="always")
def _floor(i):
    """Computes floor that handles negatives correctly.

    This is necessary because integer truncation rounds toward zero, not toward negative infinity.
    """
    if i >= 0:
        return int(i)
    return int(i) - 1


def warp(
    data: NDArray,
    samples: tuple[NDArray, ...] | list[NDArray] | NDArray,
    order: int | str = 1,
    tension: float = 0.0,
) -> NDArray:
    """Interpolates (samples) 2D data at the positions specified by samples in pixel coordinates.

    This function performs backward warping, where pixel values are sampled from source locations specified by the
    sample coordinates. Uses Cardinal spline interpolation for cubic order.

    Args:
        data: The 2D numpy array to interpolate. Float32 or float64 recommended.
        samples: The sample positions as a tuple of two numpy arrays (x, y order), or a stacked array (y-x order,
            skimage-compatible).
        order: The interpolation order. Can be an integer (0, 1, 3) or string ('nearest', 'linear', 'cubic').
            Defaults to 1.
        tension: The tension parameter for Cardinal splines, in range [-1, 1]. A value of 0 produces a Catmull-Rom
            spline. Only used when order is 3. Defaults to 0.0.

    Returns:
        A numpy array with the same dtype as data and the same shape as the sample arrays.

    Raises:
        ValueError: If data is not a 2D numpy array, samples format is invalid, or interpolation parameters are
            out of range.
    """
    # Validates data.
    if not isinstance(data, np.ndarray):
        message = "Unable to warp data. Expected data to be a numpy array."
        console.error(message=message, error=ValueError)
    if data.ndim != 2:
        message = "Unable to warp data. Only 2D arrays are supported."
        console.error(message=message, error=ValueError)

    # Normalizes samples to tuple format.
    if isinstance(samples, tuple):
        pass
    elif isinstance(samples, list):
        samples = tuple(samples)
    elif isinstance(samples, np.ndarray) and samples.shape[0] == 2 and samples[0].ndim > 0:
        # skimage API (y-x order).
        samples = (samples[1], samples[0])
    else:
        message = "Unable to warp data. Samples must be a tuple of two arrays, or a 2xHxW array."
        console.error(message=message, error=ValueError)

    if len(samples) != 2:
        message = "Unable to warp data. Samples must contain exactly two arrays for 2D data."
        console.error(message=message, error=ValueError)
    for s in samples:
        if not isinstance(s, np.ndarray):
            message = "Unable to warp data. All values in samples must be numpy arrays."
            console.error(message=message, error=ValueError)
        if s.shape != samples[0].shape:
            message = "Unable to warp data. All sample arrays must have the same shape."
            console.error(message=message, error=ValueError)

    # Validates and converts order.
    orders = {"nearest": 0, "linear": 1, "cubic": 3}
    if isinstance(order, str):
        if order not in orders:
            message = f"Unable to warp data. Unknown interpolation order: {order}."
            console.error(message=message, error=ValueError)
        order = orders[order]
    if order not in [0, 1, 3]:
        message = f"Unable to warp data. Invalid interpolation order: {order}. Use 0 (nearest), 1 (linear), or 3 (cubic)."
        console.error(message=message, error=ValueError)

    # Validates tension for Cardinal splines.
    if not (-1.0 <= tension <= 1.0):
        message = f"Unable to warp data. Tension must be in range [-1, 1], got: {tension}."
        console.error(message=message, error=ValueError)

    # Prepares empty result array.
    result = np.empty(samples[0].shape, data.dtype)

    # Executes 2D warp using Cardinal spline interpolation.
    _warp2(data, result.ravel(), samples[0].ravel(), samples[1].ravel(), order, tension)

    return result


@numba.njit(parallel=True, cache=True, nogil=True)
def _warp2(data_, result_, samples_x_, samples_y_, order, tension):
    """Performs 2D backward warping with Numba JIT compilation using Cardinal splines."""
    num_samples = samples_x_.size
    size_y = data_.shape[0]
    size_x = data_.shape[1]

    if order == 3:
        for sample_idx in numba.prange(num_samples):
            coeff_x = np.empty((4,), np.float32)
            coeff_y = np.empty((4,), np.float32)

            sample_x = samples_x_[sample_idx]
            index_x = _floor(sample_x)
            frac_x = sample_x - index_x
            sample_y = samples_y_[sample_idx]
            index_y = _floor(sample_y)
            frac_y = sample_y - index_y

            if 1 <= index_x < size_x - 2 and 1 <= index_y < size_y - 2:
                # Cubic interpolation (interior).
                compute_cardinal_coefficients(frac_x, coeff_x, tension)
                compute_cardinal_coefficients(frac_y, coeff_y, tension)

                interpolated_value = 0.0
                for coeff_idx_y in range(4):
                    for coeff_idx_x in range(4):
                        interpolated_value += (
                            data_[index_y + coeff_idx_y - 1, index_x + coeff_idx_x - 1]
                            * coeff_y[coeff_idx_y]
                            * coeff_x[coeff_idx_x]
                        )
                result_[sample_idx] = interpolated_value

            elif -0.5 <= sample_x <= size_x - 0.5 and -0.5 <= sample_y <= size_y - 0.5:
                # Edge effects.
                compute_cardinal_coefficients(frac_x, coeff_x, tension)
                compute_cardinal_coefficients(frac_y, coeff_y, tension)

                coeff_start_x, coeff_end_x = 0, 4
                coeff_start_y, coeff_end_y = 0, 4
                if index_x < 1:
                    coeff_start_x += 1 - index_x
                if index_x > size_x - 3:
                    coeff_end_x += (size_x - 3) - index_x
                if index_y < 1:
                    coeff_start_y += 1 - index_y
                if index_y > size_y - 3:
                    coeff_end_y += (size_y - 3) - index_y

                # Normalizes coefficients so the sum is one.
                coeff_sum = 0.0
                for coeff_idx in range(coeff_start_x, coeff_end_x):
                    coeff_sum += coeff_x[coeff_idx]
                coeff_sum = 1.0 / coeff_sum
                for coeff_idx in range(coeff_start_x, coeff_end_x):
                    coeff_x[coeff_idx] *= coeff_sum
                coeff_sum = 0.0
                for coeff_idx in range(coeff_start_y, coeff_end_y):
                    coeff_sum += coeff_y[coeff_idx]
                coeff_sum = 1.0 / coeff_sum
                for coeff_idx in range(coeff_start_y, coeff_end_y):
                    coeff_y[coeff_idx] *= coeff_sum

                interpolated_value = 0.0
                for coeff_idx_y in range(coeff_start_y, coeff_end_y):
                    for coeff_idx_x in range(coeff_start_x, coeff_end_x):
                        interpolated_value += (
                            data_[index_y + coeff_idx_y - 1, index_x + coeff_idx_x - 1]
                            * coeff_y[coeff_idx_y]
                            * coeff_x[coeff_idx_x]
                        )
                result_[sample_idx] = interpolated_value

            else:
                result_[sample_idx] = 0.0

    elif order == 1:
        for sample_idx in numba.prange(num_samples):
            sample_x = samples_x_[sample_idx]
            index_x = _floor(sample_x)
            frac_x = sample_x - index_x
            sample_y = samples_y_[sample_idx]
            index_y = _floor(sample_y)
            frac_y = sample_y - index_y

            if 0 <= index_x < size_x - 1 and 0 <= index_y < size_y - 1:
                interpolated_value = data_[index_y, index_x] * (1.0 - frac_y) * (1.0 - frac_x)
                interpolated_value += data_[index_y, index_x + 1] * (1.0 - frac_y) * frac_x
                interpolated_value += data_[index_y + 1, index_x] * frac_y * (1.0 - frac_x)
                interpolated_value += data_[index_y + 1, index_x + 1] * frac_y * frac_x
                result_[sample_idx] = interpolated_value
            elif -0.5 <= sample_x <= size_x - 0.5 and -0.5 <= sample_y <= size_y - 0.5:
                if index_x < 0:
                    frac_x += index_x
                    index_x = 0
                if index_x > size_x - 2:
                    frac_x += index_x - (size_x - 2)
                    index_x = size_x - 2
                if index_y < 0:
                    frac_y += index_y
                    index_y = 0
                if index_y > size_y - 2:
                    frac_y += index_y - (size_y - 2)
                    index_y = size_y - 2
                interpolated_value = data_[index_y, index_x] * (1.0 - frac_y) * (1.0 - frac_x)
                interpolated_value += data_[index_y, index_x + 1] * (1.0 - frac_y) * frac_x
                interpolated_value += data_[index_y + 1, index_x] * frac_y * (1.0 - frac_x)
                interpolated_value += data_[index_y + 1, index_x + 1] * frac_y * frac_x
                result_[sample_idx] = interpolated_value
            else:
                result_[sample_idx] = 0.0

    else:
        for sample_idx in numba.prange(num_samples):
            sample_x = samples_x_[sample_idx]
            index_x = _floor(sample_x + 0.5)
            sample_y = samples_y_[sample_idx]
            index_y = _floor(sample_y + 0.5)

            if 0 <= index_x < size_x and 0 <= index_y < size_y:
                result_[sample_idx] = data_[index_y, index_x]
            else:
                result_[sample_idx] = 0.0


def project(
    data: NDArray,
    samples: tuple[NDArray, ...] | list[NDArray] | NDArray,
) -> NDArray:
    """Projects 2D data to the positions specified by samples using forward splatting.

    This function performs forward deformation, moving pixels to specified locations rather than sampling from
    them. Unlike warp(), this function uses splatting instead of interpolation.

    Args:
        data: The 2D numpy array to project. Float32 or float64 recommended.
        samples: The target positions as a tuple of two numpy arrays (x, y order). Each array must have the same
            shape as data. Can also be a stacked array (y-x order, skimage-compatible).

    Returns:
        A numpy array with the same dtype and shape as the input data.

    Raises:
        ValueError: If data is not a 2D numpy array or samples format is invalid.
    """
    # Validates data.
    if not isinstance(data, np.ndarray):
        message = "Unable to project data. Expected data to be a numpy array."
        console.error(message=message, error=ValueError)
    if data.ndim != 2:
        message = "Unable to project data. Only 2D arrays are supported."
        console.error(message=message, error=ValueError)

    # Normalizes samples to tuple format.
    if isinstance(samples, tuple):
        pass
    elif isinstance(samples, list):
        samples = tuple(samples)
    elif isinstance(samples, np.ndarray) and samples.shape[0] == 2 and samples[0].ndim > 0:
        # skimage API (y-x order).
        samples = (samples[1], samples[0])
    else:
        message = "Unable to project data. Samples must be a tuple of two arrays, or a 2xHxW array."
        console.error(message=message, error=ValueError)

    if len(samples) != 2:
        message = "Unable to project data. Samples must contain exactly two arrays for 2D data."
        console.error(message=message, error=ValueError)
    for s in samples:
        if not isinstance(s, np.ndarray):
            message = "Unable to project data. All values in samples must be numpy arrays."
            console.error(message=message, error=ValueError)
        if s.shape != data.shape:
            message = "Unable to project data. Sample arrays must all have the same shape as the data."
            console.error(message=message, error=ValueError)

    # Prepares empty result array (uses zeros, not empty, for accumulation).
    result = np.zeros(samples[0].shape, data.dtype)

    # Executes 2D projection.
    _project2(data, result, samples[0], samples[1])

    return result


@numba.njit(parallel=True, cache=True, nogil=True)
def _project2(data_, result_, deformx_, deformy_):
    """Performs 2D forward projection with Numba JIT compilation."""
    size_y = data_.shape[0]
    size_x = data_.shape[1]

    weight_accumulator = np.zeros(data_.shape, dtype=np.float32)

    # Main splatting loop (sequential due to write conflicts).
    for src_y in range(size_y):
        for src_x in range(size_x):
            target_y = deformy_[src_y, src_x]
            target_x = deformx_[src_y, src_x]

            # Determines destination region from surrounding pixel mapping.
            target_min_x = target_x
            target_max_x = target_x
            target_min_y = target_y
            target_max_y = target_y
            for neighbor_dy in range(-1, 2):
                for neighbor_dx in range(-1, 2):
                    if neighbor_dy * neighbor_dx == 0:
                        continue
                    if src_y + neighbor_dy < 0:
                        target_min_y = target_y - 1000000.0
                        continue
                    if src_x + neighbor_dx < 0:
                        target_min_x = target_x - 1000000.0
                        continue
                    if src_y + neighbor_dy >= size_y:
                        target_max_y = target_y + 1000000.0
                        continue
                    if src_x + neighbor_dx >= size_x:
                        target_max_x = target_x + 1000000.0
                        continue
                    neighbor_target_x = deformx_[src_y + neighbor_dy, src_x + neighbor_dx]
                    target_min_x = min(target_min_x, neighbor_target_x)
                    target_max_x = max(target_max_x, neighbor_target_x)
                    neighbor_target_y = deformy_[src_y + neighbor_dy, src_x + neighbor_dx]
                    target_min_y = min(target_min_y, neighbor_target_y)
                    target_max_y = max(target_max_y, neighbor_target_y)

            # Limits to bounds and converts to integer.
            target_min_x = max(0, min(size_x - 1, target_min_x))
            target_max_x = max(0, min(size_x - 1, target_max_x))
            target_min_y = max(0, min(size_y - 1, target_min_y))
            target_max_y = max(0, min(size_y - 1, target_max_y))
            dst_start_x = max(0, _floor(target_min_x))
            dst_end_x = min(size_x - 1, int(math.ceil(target_max_x)))
            dst_start_y = max(0, _floor(target_min_y))
            dst_end_y = min(size_y - 1, int(math.ceil(target_max_y)))

            # Calculates kernel size from max range.
            inv_kernel_radius = 0.1
            inv_kernel_radius = max(inv_kernel_radius, max(abs(target_min_y - target_y), abs(target_max_y - target_y)))
            inv_kernel_radius = max(inv_kernel_radius, max(abs(target_min_x - target_x), abs(target_max_x - target_x)))
            inv_kernel_radius = 1.0 / inv_kernel_radius

            src_value = data_[src_y, src_x]

            # Splats value in destination.
            for dst_y in range(dst_start_y, dst_end_y + 1):
                for dst_x in range(dst_start_x, dst_end_x + 1):
                    weight_y = 1.0 - inv_kernel_radius * abs(dst_y - target_y)
                    weight_x = 1.0 - inv_kernel_radius * abs(dst_x - target_x)
                    weight = max(0.0, weight_y) * max(0.0, weight_x)
                    result_[dst_y, dst_x] += src_value * weight
                    weight_accumulator[dst_y, dst_x] += weight

    # Divides by coefficients (parallel, no write conflicts).
    for dst_y in numba.prange(size_y):
        for dst_x in range(size_x):
            total_weight = weight_accumulator[dst_y, dst_x]
            if total_weight > 0:
                result_[dst_y, dst_x] /= total_weight


def deform_backward(
    data: NDArray,
    deltas: tuple[NDArray, ...],
    order: int | str = 1,
    tension: float = 0.0,
) -> NDArray:
    """Interpolates data according to the deformations specified in deltas using backward warping.

    This function applies backward deformation by sampling from source locations determined by the delta offsets.

    Args:
        data: The numpy array to deform.
        deltas: A tuple of numpy arrays representing relative sample positions expressed in world coordinates
            (x-y-z order). Must contain as many arrays as data has dimensions.
        order: The interpolation order. Defaults to 1.
        tension: The tension parameter for Cardinal splines, in range [-1, 1]. A value of 0 produces a Catmull-Rom
            spline. Only used when order is 3. Defaults to 0.0.

    Returns:
        A numpy array containing the deformed data.

    Raises:
        ValueError: If the number of delta arrays does not match data dimensions.
    """
    if len(deltas) != data.ndim:
        message = "Unable to apply backward deformation. Deltas must contain as many arrays as data has dimensions."
        console.error(message=message, error=ValueError)

    # Creates absolute sample positions.
    samples = make_samples_absolute(deltas)

    # Interpolates and returns.
    return warp(data, samples, order, tension)


def deform_forward(
    data: NDArray,
    deltas: tuple[NDArray, ...],
) -> NDArray:
    """Applies forward deformation using projection/splatting.

    This function applies forward deformation by moving pixel values to target locations determined by the delta
    offsets.

    Args:
        data: The numpy array to deform.
        deltas: A tuple of numpy arrays representing relative sample positions expressed in world coordinates
            (x-y-z order). Must contain as many arrays as data has dimensions.

    Returns:
        A numpy array containing the deformed data.

    Raises:
        ValueError: If the number of delta arrays does not match data dimensions.
    """
    if len(deltas) != data.ndim:
        message = "Unable to apply forward deformation. Deltas must contain as many arrays as data has dimensions."
        console.error(message=message, error=ValueError)

    # Creates absolute sample positions.
    samples = make_samples_absolute(deltas)

    # Projects and returns.
    return project(data, samples)


def resize(
    data: NDArray,
    new_shape: tuple[int, ...] | list[int],
    order: int | str = 3,
    tension: float = 0.0,
    prefilter: bool = False,
    extra: bool = False,
) -> NDArray:
    """Resizes the data to the specified new shape using interpolation.

    This function resamples the data array to match the specified shape, optionally applying antialiasing for
    downsampling.

    Args:
        data: The numpy array to resize.
        new_shape: A tuple specifying the new shape (z-y-x order).
        order: The interpolation order (0, 1, or 3). Defaults to 3.
        tension: The tension parameter for Cardinal splines, in range [-1, 1]. A value of 0 produces a Catmull-Rom
            spline. Only used when order is 3. Defaults to 0.0.
        prefilter: Whether to apply Gaussian antialiasing when downsampling. Defaults to False.
        extra: Whether to extrapolate beyond original data boundaries. When True, each datapoint spans a space equal
            to the distance between data points (Photoshop-style). When False, the first and last datapoints align
            exactly (scipy-style). Defaults to False.

    Returns:
        The resized data as a numpy array.

    Raises:
        ValueError: If new_shape is not a tuple/list or does not match data dimensions.
    """
    if not isinstance(new_shape, (tuple, list)):
        message = "Unable to resize data. Expected new_shape to be a tuple or list."
        console.error(message=message, error=ValueError)
    if not len(new_shape) == len(data.shape):
        message = "Unable to resize data. new_shape must contain as many values as data has dimensions."
        console.error(message=message, error=ValueError)
    new_shape = [int(round(n)) for n in new_shape]

    # Computes sample coordinate ranges for each dimension.
    shape = data.shape
    ranges = []

    for s, n in zip(shape, new_shape):
        if extra:
            dmin, dmax = -0.5, s - 0.5
            drange = dmax - dmin
            dstep = float(drange) / n
            dstep2 = 0.5 * dstep
            r = np.linspace(dmin + dstep2, dmax - dstep2, n)
            ranges.append(r)
        else:
            dmin, dmax = 0, s - 1
            r = np.linspace(dmin, dmax, n)
            ranges.append(r)

    # Applies antialiasing if requested.
    def compute_sigma(x):
        if x < 1.0:
            return 0.8 / x
        return 0.0

    factors = [float(s1) / s2 for s1, s2 in zip(new_shape, shape)]
    sigmas = [compute_sigma(f) for f in factors]
    if prefilter and sum(sigmas):
        data = diffuse(data, sigmas)

    # Creates coordinate grids for interpolation. Ranges are in z-y-x order, but warp expects x-y-z order.
    # The swizzling handles numpy's meshgrid xy-indexing convention.
    ranges.reverse()
    iterators = [r.astype(np.float32) for r in ranges]
    if len(iterators) > 1:
        iterators[0], iterators[1] = iterators[1], iterators[0]
    grids = np.meshgrid(*iterators, indexing="xy")
    grids = [g.astype(np.float32) for g in grids]
    if len(grids) > 1:
        grids[0], grids[1] = grids[1], grids[0]
    grids = tuple(reversed(grids))

    data2 = warp(data, grids, order, tension)

    return data2


def zoom(
    data: NDArray,
    factor: float | tuple[float, ...] | list[float],
    order: int | str = 3,
    tension: float = 0.0,
    prefilter: bool = False,
    extra: bool = False,
) -> NDArray:
    """Resizes the data by the specified scale factor using interpolation.

    This function scales the data array by a uniform or per-dimension factor, similar to scipy.ndimage.zoom but
    approximately three times faster.

    Args:
        data: The numpy array to resize.
        factor: A scalar or tuple specifying the resize factor (z-y-x order for per-dimension factors).
        order: The interpolation order (0, 1, or 3). Defaults to 3.
        tension: The tension parameter for Cardinal splines, in range [-1, 1]. A value of 0 produces a Catmull-Rom
            spline. Only used when order is 3. Defaults to 0.0.
        prefilter: Whether to apply Gaussian antialiasing when downsampling. Defaults to False.
        extra: Whether to extrapolate beyond original data boundaries. Defaults to False.

    Returns:
        The zoomed data as a numpy array.

    Raises:
        ValueError: If factor is not a float, tuple, or list, or if factor length does not match data dimensions.
    """
    # Normalizes factor to list.
    if isinstance(factor, np.ndarray) and factor.size == 1:
        factor = float(factor)
    if isinstance(factor, (float, int)):
        factor = [factor for _ in data.shape]

    if not isinstance(factor, (list, tuple)):
        message = "Unable to zoom data. Expected factor to be a float, tuple, or list."
        console.error(message=message, error=ValueError)
    if len(factor) != data.ndim:
        message = "Unable to zoom data. Factor length does not match data dimensions."
        console.error(message=message, error=ValueError)

    # Calculates new shape.
    new_shape = [float(f) * s for f, s in zip(factor, data.shape)]
    new_shape = [int(round(s)) for s in new_shape]

    return resize(data, new_shape, order, tension, prefilter, extra)


# =============================================================================
# Deformation Class
# =============================================================================


class Deformation:
    """Unified deformation class for 2D image registration.

    A deformation maps one 2D image to another. It can represent:
    - An identity (null) deformation when no fields are provided
    - A field-based deformation when field arrays are provided

    Parameters
    ----------
    *fields : arrays, int, or shape tuple
        The deformation fields (one per dimension, in z-y-x order).
        Can also be:
        - No arguments: creates an identity deformation
        - Single int: creates null deformation with specified ndim
        - Shape tuple: creates null deformation with specified shape

    Notes:
        All deformations use backward mapping where result pixels sample from source locations.
    """

    def __init__(self, *fields):

        if len(fields) == 1 and isinstance(fields[0], (list, tuple)):
            fields = fields[0]

        if not fields:
            # Identity deformation (no fields)
            self._field_shape = (1, 1)
            self._field_sampling = (1.0, 1.0)
            self._fields = []

        elif len(fields) == 1 and isinstance(fields[0], int):
            # Null deformation with specified ndim
            ndim = fields[0]
            self._field_shape = tuple([1 for _ in range(ndim)])
            self._field_sampling = tuple([1.0 for _ in range(ndim)])
            self._fields = []

        elif len(fields) == 1 and isinstance(fields[0], tuple):
            # Null deformation with specified shape tuple
            self._field_shape = fields[0]
            self._field_sampling = tuple(1.0 for _ in fields[0])
            self._fields = []

        else:
            # Actual field deformation
            if not self._check_fields_same_shape(fields):
                raise ValueError("Fields must all have the same shape.")
            if len(fields) != fields[0].ndim:
                raise ValueError("There must be a field for each dimension.")

            self._field_shape = fields[0].shape
            # Sampling is always 1.0 for all dimensions in multiday usage
            self._field_sampling = tuple(1.0 for _ in range(fields[0].ndim))
            self._fields = list(fields)

    def __repr__(self):
        if self.is_identity:
            return f"<Deformation {self.ndim}D identity>"
        shapestr = "x".join([str(s) for s in self.field_shape])
        return f"<Deformation shape {shapestr}>"

    @staticmethod
    def _check_fields_same_shape(fields):
        """Checks whether the given fields all have the same shape."""
        shape = fields[0].shape
        for field in fields:
            if field.shape != shape:
                return False
        return True

    # --- Properties ---

    @property
    def is_identity(self) -> bool:
        """Whether this represents no deformation (identity)."""
        return len(self._fields) == 0

    @property
    def ndim(self) -> int:
        """The number of dimensions of the deformation."""
        return len(self._field_shape)

    @property
    def field_shape(self) -> tuple:
        """The shape of the deformation field."""
        return tuple(self._field_shape)

    @property
    def field_sampling(self) -> tuple:
        """The sampling (pixel spacing) for each dimension."""
        return tuple(self._field_sampling)

    # --- Sequence access ---

    def __len__(self):
        return len(self._fields)

    def __getitem__(self, item):
        if isinstance(item, int):
            if 0 <= item < len(self._fields):
                return self._fields[item]
            raise IndexError("Field index out of range.")
        raise IndexError("Deformation only supports integer indices.")

    def __iter__(self):
        return iter(self._fields)

    # --- Operators ---

    def __add__(self, other):
        return self.add(other)

    def __mul__(self, other):
        if isinstance(other, Deformation):
            return other.compose(self)
        return self.scale(other)

    # --- Core methods ---

    def copy(self):
        """Create a deep copy of this deformation."""
        if self.is_identity:
            return Deformation(self.field_shape)
        return self.scale(1.0)

    def scale(self, factor: float):
        """Scale the deformation by the given factor.

        Note that the result is diffeomorphic only if the original is
        diffeomorphic and the factor is between -1 and 1.
        """
        fields = []
        for d in range(self.ndim):
            if factor == 1.0:
                fields.append(self._fields[d].copy())
            else:
                fields.append(self._fields[d] * factor)
        return Deformation(*fields)

    def add(self, other):
        """Combine two deformations by addition."""
        if not isinstance(other, Deformation):
            raise ValueError("Can only combine Deformations.")

        if self.is_identity:
            return other.copy()
        if other.is_identity:
            return self.copy()

        if self.field_shape != other.field_shape:
            raise ValueError("Can only combine deforms with same field shape.")

        fields = []
        for d in range(self.ndim):
            fields.append(self.get_field(d) + other.get_field(d))
        return Deformation(*fields)

    def compose(self, other):
        """Combine two deformations by composition.

        The left (self) is the "static" deformation, and the right (other)
        is the "delta" deformation. Returns a new Deformation instance.
        """
        if not isinstance(other, Deformation):
            raise ValueError("Can only combine Deformations.")

        if self.is_identity:
            return other.copy()
        if other.is_identity:
            return self.copy()

        if self.field_shape != other.field_shape:
            raise ValueError("Can only combine deforms with same field shape.")

        fields = self._compose_backward(other)
        return Deformation(*fields)

    def _compose_forward(self, other):
        """Compose for forward mapping: sample in other at locations of self."""
        # Get sample positions in pixel coordinates
        sample_locations = self.get_deformation_locations()

        fields = []
        for d in range(self.ndim):
            field1 = self._fields[d]
            field2 = other._fields[d]
            # Composition with a field introduces interpolation artifacts
            field = warp(field2, sample_locations, "linear")
            fields.append(field1 + field)
        return fields

    def _compose_backward(self, other):
        """Compose for backward mapping: sample in self at locations of other."""
        return other._compose_forward(self)

    def resize_field(self, new_shape):
        """Create a new Deformation with the field resized to match new_shape.

        Args:
            new_shape: The target shape as a tuple, numpy array, or Deformation.

        Returns:
            A new deformation with resized fields, or self if already correct size.
        """
        # Extract shape from various input types
        if hasattr(new_shape, "field_shape"):
            target_shape = tuple(new_shape.field_shape)
        elif hasattr(new_shape, "shape"):
            target_shape = tuple(new_shape.shape)
        else:
            target_shape = tuple(new_shape)

        if self.is_identity:
            return Deformation(target_shape)

        if self.field_shape == target_shape:
            return self

        fields = []
        for field in self._fields:
            resized = resize(field, target_shape, order=3, prefilter=False, extra=False)
            fields.append(resized)
        return Deformation(*fields)

    # --- Getting field values ---

    def get_field(self, d: int):
        """Get the field for dimension d."""
        return self._fields[d]

    def get_deformation_locations(self):
        """Get absolute sample locations in pixel coordinates (x-y-z order).

        These locations can be fed directly to interp functions.
        """
        # Reverse fields from z-y-x to x-y-z order
        deltas = [s for s in reversed(self._fields)]
        return make_samples_absolute(deltas)

    # --- Applying deformation ---

    def apply_deformation(self, data, interpolation: int = 3):
        """Apply the deformation to the given data.

        Parameters
        ----------
        data : array
            The data to deform.
        interpolation : int
            Interpolation order (0, 1, or 3).

        Returns:
        -------
        array
            The deformed data.
        """
        if self.is_identity:
            return data

        # Need upsampling?
        deform = self.resize_field(data)

        # Reverse from z-y-x to x-y-z
        samples = [s for s in reversed(deform._fields)]

        return deform_backward(data, samples, interpolation)

    # --- Conversion methods ---

    def inverse(self):
        """Get the inverse deformation.

        Only valid if the current deformation is diffeomorphic.
        """
        if self.is_identity:
            return self

        # Get samples
        samples = [s for s in reversed(self._fields)]

        # Get inverse fields
        fields = []
        for field in self._fields:
            fields.append(deform_forward(-field, samples))

        return Deformation(*fields)

    def regularize(
        self,
        grid_sampling: float,
        weights: np.ndarray | None = None,
        injective: bool | float = True,
        frozenedge: bool = True,
    ) -> "Deformation":
        """Regularizes the deformation using B-spline grid constraints.

        Args:
            grid_sampling: The B-spline grid spacing (knot spacing) in pixels.
            weights: Optional weights for field elements.
            injective: Whether to prevent grid folding. Can be True, False, or a float factor.
            frozenedge: Whether to freeze edges to zero deformation.

        Returns:
            A new regularized Deformation instance.
        """
        if self.is_identity:
            return self

        grid = SplineGrid(self.field_shape, grid_sampling)
        grid.set_from_fields(self._fields, weights, injective, frozenedge)
        return Deformation(*grid.get_fields())
