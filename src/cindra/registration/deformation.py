"""Provides the assets for computing and representing image deformations used in diffeomorphic registrations."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numba  # type: ignore[import-untyped]
import numpy as np
import scipy.ndimage

from .spline_grid import SplineGrid, compute_cardinal_coefficients

_MINIMUM_DIFFUSION_SIGMA: float = 0.1
"""The minimum sigma value below which the diffusion kernel returns a delta function (no smoothing)."""

_CUBIC_INTERPOLATION_ORDER: int = 3
"""The interpolation order constant for cubic cardinal spline interpolation."""

_BILINEAR_INTERPOLATION_ORDER: int = 1
"""The interpolation order constant for bilinear interpolation."""

_BOUNDARY_TOLERANCE: float = -0.5
"""The boundary tolerance for determining if a sample point falls within valid image bounds. Pixels are centered at
integer coordinates, so valid samples extend from -0.5 to (dimension - 0.5)."""

if TYPE_CHECKING:
    from collections.abc import Iterator

    from numpy.typing import NDArray


def diffuse(data: NDArray[np.float32], sigma: float | list[float]) -> NDArray[np.float32]:
    """Filters the input data using the Lindeberg's discrete diffusion kernel.

    Args:
        data: The data to filter with the kernel.
        sigma: The smoothing parameter. Can be a single value applied to all dimensions or a list with one value per
            each data dimension.

    Returns:
        The diffused data array with the same shape as the input.
    """
    # Converts sigma to a list with one value per dimension.
    sigma_list = [sigma] * data.ndim if isinstance(sigma, float) else list(sigma)

    # Applies 1D diffusion filtering along each dimension.
    result = data
    for dimension in range(data.ndim):
        kernel = _create_diffusion_kernel(sigma_list[dimension])
        # noinspection PyTypeChecker
        result = scipy.ndimage.convolve1d(input=result, weights=kernel, axis=dimension, mode="nearest")

    return result


def zoom(
    data: NDArray[np.float32],
    factor: float | tuple[float, float],
    order: int = 3,
) -> NDArray[np.float32]:
    """Resizes 2D data by the specified scale factor using interpolation.

    Args:
        data: The 2D array to resize.
        factor: The factor by which to resize the data. Can be a scalar for uniform scaling, or a
            (height_factor, width_factor) tuple.
        order: The interpolation order (0=nearest, 1=linear, 3=cubic).

    Returns:
        The zoomed array.
    """
    height, width = data.shape

    # Normalizes factor to per-dimension values.
    if isinstance(factor, (float, int)):
        factor_y, factor_x = float(factor), float(factor)
    else:
        factor_y, factor_x = factor[0], factor[1]

    # Calculates the new shape.
    new_height = round(factor_y * height)
    new_width = round(factor_x * width)

    return _resize(data, new_height, new_width, order)


class Deformation:
    """Represents a 2D image deformation using backward mapping.

    A deformation maps pixel locations from a target image back to source locations, enabling image warping through
    interpolation. The class stores relative displacement fields (deltas) for each dimension.

    Notes:
        Use `Deformation.identity()` to create identity (no-op) deformations. Use the regular constructor only for
        creating deformations from explicit displacement field arrays.

    Args:
        field_y: The Y-dimension displacement field array.
        field_x: The X-dimension displacement field array.

    Attributes:
        _field_shape: The shape of the deformation field as (height, width).
        _fields: List of displacement field arrays, one per dimension [Y, X].
    """

    def __init__(self, field_y: NDArray[np.float32], field_x: NDArray[np.float32]) -> None:
        self._field_shape: tuple[int, int] = (field_y.shape[0], field_y.shape[1])
        self._fields: list[NDArray[np.float32]] = [field_y, field_x]

    @classmethod
    def identity(cls, height: int, width: int) -> Deformation:
        """Creates an identity (no-op) deformation with the specified dimensions.

        An identity deformation represents no transformation, where each pixel maps to itself.

        Args:
            height: The height of the deformation field in pixels.
            width: The width of the deformation field in pixels.

        Returns:
            A new identity Deformation instance.
        """
        instance = cls.__new__(cls)
        instance._field_shape = (height, width)
        instance._fields = []
        return instance

    def __repr__(self) -> str:
        """Returns a string representation of the Deformation instance."""
        if self.is_identity:
            return f"<Deformation {self.ndim}D identity>"
        shape_string = "x".join(str(size) for size in self.field_shape)
        return f"<Deformation shape {shape_string}>"

    @property
    def is_identity(self) -> bool:
        """Returns whether this deformation represents an identity (no-op) transformation."""
        return len(self._fields) == 0

    @property
    def ndim(self) -> int:
        """Returns the number of dimensions of the deformation."""
        return len(self._field_shape)

    @property
    def field_shape(self) -> tuple[int, ...]:
        """Returns the shape of the deformation field as (height, width)."""
        return tuple(self._field_shape)

    def __len__(self) -> int:
        """Returns the number of field arrays (one per dimension)."""
        return len(self._fields)

    def __getitem__(self, index: int) -> NDArray[np.float32]:
        """Returns the field array at the specified dimension index.

        Args:
            index: The dimension index (0 for Y, 1 for X).

        Returns:
            The displacement field array for that dimension.
        """
        return self._fields[index]

    def __iter__(self) -> Iterator[NDArray[np.float32]]:
        """Returns an iterator over the field arrays."""
        return iter(self._fields)

    def __add__(self, other: Deformation) -> Deformation:
        """Combines two deformations by element-wise addition of their fields."""
        return self.add(other)

    def copy(self) -> Deformation:
        """Creates a deep copy of this deformation.

        Returns:
            A new Deformation instance with copied field arrays.
        """
        if self.is_identity:
            return Deformation.identity(height=self._field_shape[0], width=self._field_shape[1])
        return self.scale(factor=1.0)

    def scale(self, factor: float) -> Deformation:
        """Scales the deformation magnitude by the given factor.

        Notes:
            The result is diffeomorphic only if the original is diffeomorphic and the factor is in range [-1, 1].

        Args:
            factor: The scaling factor to apply to all displacement values.

        Returns:
            A new Deformation with scaled displacement fields.
        """
        if factor == 1.0:
            field_y = self._fields[0].copy()
            field_x = self._fields[1].copy()
        else:
            field_y = self._fields[0] * factor
            field_x = self._fields[1] * factor
        return Deformation(field_y=field_y, field_x=field_x)

    def add(self, other: Deformation) -> Deformation:
        """Combines two deformations by element-wise addition of their displacement fields.

        Args:
            other: The deformation to add to this one.

        Returns:
            A new Deformation with summed displacement fields.
        """
        if self.is_identity:
            return other.copy()
        if other.is_identity:
            return self.copy()

        field_y = self._fields[0] + other.get_field(dimension=0)
        field_x = self._fields[1] + other.get_field(dimension=1)
        return Deformation(field_y=field_y, field_x=field_x)

    def compose(self, other: Deformation) -> Deformation:
        """Combines two deformations by composition (function composition).

        Computes the deformation that results from applying self followed by the other. Uses backward composition where
        self is sampled at the locations defined by the other.

        Args:
            other: The deformation to compose with (applied second).

        Returns:
            A new Deformation representing the composed transformation.
        """
        if self.is_identity:
            return other.copy()
        if other.is_identity:
            return self.copy()

        # Backward composition: samples self's field at locations defined by the other.
        samples_x, samples_y = other.get_deformation_locations()

        # Warps Y field.
        warped_y = np.empty(samples_x.shape, dtype=np.float32)
        # noinspection PyTypeChecker
        _warp(
            data=self._fields[0],
            result=warped_y.ravel(),
            samples_x=samples_x.ravel(),
            samples_y=samples_y.ravel(),
            order=_BILINEAR_INTERPOLATION_ORDER,
        )
        field_y = other.get_field(dimension=0) + warped_y

        # Warps X field.
        warped_x = np.empty(samples_x.shape, dtype=np.float32)
        # noinspection PyTypeChecker
        _warp(
            data=self._fields[1],
            result=warped_x.ravel(),
            samples_x=samples_x.ravel(),
            samples_y=samples_y.ravel(),
            order=_BILINEAR_INTERPOLATION_ORDER,
        )
        field_x = other.get_field(dimension=1) + warped_x

        return Deformation(field_y=field_y, field_x=field_x)

    def resize_field(self, new_height: int, new_width: int) -> Deformation:
        """Creates a new Deformation with the field resized to the specified dimensions.

        Args:
            new_height: The target height in pixels.
            new_width: The target width in pixels.

        Returns:
            A new Deformation with resized fields, or self if already the correct size.
        """
        if self.is_identity:
            return Deformation.identity(height=new_height, width=new_width)

        if self._field_shape == (new_height, new_width):
            return self

        resized_y = _resize(data=self._fields[0], new_height=new_height, new_width=new_width)
        resized_x = _resize(data=self._fields[1], new_height=new_height, new_width=new_width)
        return Deformation(field_y=resized_y, field_x=resized_x)

    def get_field(self, dimension: int) -> NDArray[np.float32]:
        """Returns the displacement field array for the specified dimension.

        Args:
            dimension: The dimension index (0 for Y, 1 for X).

        Returns:
            The displacement field array for that dimension.
        """
        return self._fields[dimension]

    def get_deformation_locations(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Computes absolute sample locations from the relative displacement fields.

        Converts the relative displacement fields (deltas) to absolute pixel coordinates that can be used directly
        for interpolation.

        Returns:
            A tuple of (samples_x, samples_y) arrays containing the absolute x and y coordinates.
        """
        return _make_samples_absolute(delta_x=self._fields[1], delta_y=self._fields[0])

    def apply_deformation(self, data: NDArray[np.float32], interpolation: int = 3) -> NDArray[np.float32]:
        """Applies this deformation to warp the input data.

        Uses backward mapping to sample the source data at locations defined by the deformation field.

        Args:
            data: The 2D array to deform.
            interpolation: The interpolation order (0=nearest, 1=bilinear, 3=cubic).

        Returns:
            The deformed data array with the same shape as the input.
        """
        if self.is_identity:
            return data

        # Resizes field to match data shape if needed.
        resized_deformation = self.resize_field(new_height=data.shape[0], new_width=data.shape[1])

        # Converts relative deformation to absolute sample positions.
        samples_x, samples_y = _make_samples_absolute(
            delta_x=resized_deformation._fields[1], delta_y=resized_deformation._fields[0]
        )

        # Applies backward warp to sample data at deformed positions.
        result = np.empty(samples_x.shape, dtype=data.dtype)
        _warp(
            data=data,
            result=result.ravel(),
            samples_x=samples_x.ravel(),
            samples_y=samples_y.ravel(),
            order=interpolation,
        )
        return result

    def inverse(self) -> Deformation:
        """Computes the inverse of this deformation.

        Uses forward projection (splatting) to approximate the inverse transformation. The result is only valid if
        the current deformation is diffeomorphic (invertible).

        Returns:
            A new Deformation representing the approximate inverse transformation.
        """
        if self.is_identity:
            return self

        # Converts relative deformation to absolute sample positions.
        samples_x, samples_y = _make_samples_absolute(delta_x=self._fields[1], delta_y=self._fields[0])

        # Computes inverse fields using forward projection (splatting).
        inverse_y = np.zeros(samples_x.shape, dtype=np.float32)
        _project(data=-self._fields[0], result=inverse_y, samples_x=samples_x, samples_y=samples_y)

        inverse_x = np.zeros(samples_x.shape, dtype=np.float32)
        _project(data=-self._fields[1], result=inverse_x, samples_x=samples_x, samples_y=samples_y)

        return Deformation(field_y=inverse_y, field_x=inverse_x)

    def regularize(
        self,
        grid_sampling: float,
        injective: bool = True,
        injective_factor: float = 0.9,
        freeze_edges: bool = True,
    ) -> Deformation | None:
        """Regularizes the deformation using B-spline grid constraints.

        Fits the deformation to a B-spline grid representation, which enforces smoothness. Optionally applies
        injectivity constraints to ensure the deformation remains diffeomorphic (invertible).

        Args:
            grid_sampling: The B-spline grid spacing (knot spacing) in pixels.
            injective: Whether to apply injectivity constraint to prevent grid folding.
            injective_factor: Scaling factor for the injectivity limit (0 < factor <= 1).
            freeze_edges: Whether to freeze edges to zero deformation.

        Returns:
            A new regularized Deformation instance, or None if the grid is too small for the requested constraints.
        """
        if self.is_identity:
            return self

        grid = SplineGrid(
            field_height=self.field_shape[0],
            field_width=self.field_shape[1],
            sampling=grid_sampling,
        )
        success = grid.set_from_fields(
            field_y=self._fields[0],
            field_x=self._fields[1],
            injective=injective,
            injective_factor=injective_factor,
            freeze_edges=freeze_edges,
        )
        if not success:
            return None
        regularized_y, regularized_x = grid.deformation_fields
        return Deformation(field_y=regularized_y, field_x=regularized_x)

    def crop(
        self,
        origin: tuple[int, int],
        crop_size: tuple[int, int],
    ) -> tuple[Deformation, tuple[int, int]]:
        """Creates a cropped view of the deformation field centered on the specified origin.

        This method extracts a local region of the deformation field to reduce memory overhead when applying
        deformations to small regions such as individual ROI masks. The origin is automatically clamped to ensure
        the crop stays within valid field bounds. The returned Deformation contains views into the original arrays,
        not copies.

        Args:
            origin: The top-left corner of the crop region as (y, x) coordinates.
            crop_size: The size of the crop region as (height, width).

        Returns:
            A tuple containing the cropped Deformation instance and the adjusted origin coordinates. The adjusted
            origin accounts for boundary clamping and can be used to map local coordinates back to global space.
        """
        if self.is_identity:
            return Deformation.identity(height=crop_size[0], width=crop_size[1]), origin

        # Clamps origin to valid bounds in a single pass.
        field_height, field_width = self._field_shape
        adjusted_y = max(0, min(origin[0], field_height - crop_size[0]))
        adjusted_x = max(0, min(origin[1], field_width - crop_size[1]))

        # Extracts views of the cropped fields.
        end_y = adjusted_y + crop_size[0]
        end_x = adjusted_x + crop_size[1]
        cropped_y = self._fields[0][adjusted_y:end_y, adjusted_x:end_x]
        cropped_x = self._fields[1][adjusted_y:end_y, adjusted_x:end_x]

        return Deformation(field_y=cropped_y, field_x=cropped_x), (adjusted_y, adjusted_x)


def _create_diffusion_kernel(sigma: float) -> NDArray[np.float32]:
    """Creates a discrete analog to the continuous Gaussian kernel.

    Uses Lindeberg's discrete diffusion kernel based on modified Bessel functions. This kernel provides true
    scale-space properties for image filtering.

    Args:
        sigma: The smoothing parameter (scale) controlling the kernel's width.

    Returns:
        The normalized diffusion kernel as a 1D array. Returns a delta kernel [1.0] if sigma is too small.
    """
    # For very small sigma, return a delta kernel (no smoothing).
    if sigma < _MINIMUM_DIFFUSION_SIGMA:
        return np.array([1.0], dtype=np.float32)

    sigma_squared = sigma * sigma

    # Computes the tail length. The kernel extends to ceil(4 * sigma) + 1 samples in each direction.
    half_length = int(np.ceil(4 * sigma)) + 1

    # Allocates kernel array. Uses 2 * half_length + 1 elements, trimmed to 2 * half_length - 1 after computation.
    kernel = np.zeros(2 * half_length + 1, dtype=np.float32)

    # Initializes the recurrence relation with seed values.
    kernel[half_length] = 0.0
    kernel[half_length - 1] = 0.01

    # Computes kernel values using the Bessel function recurrence relation.
    for n in range(half_length - 1, 0, -1):
        kernel[(n - 1) + half_length] = (2 * n / sigma_squared) * kernel[n + half_length] + kernel[
            (n + 1) + half_length
        ]

    # Mirrors the computed values to create the symmetric kernel.
    kernel[:half_length] = np.flipud(kernel[-half_length:])

    # Removes the zero-padded boundary elements.
    kernel = kernel[1:-1]

    # Normalizes the kernel to sum to 1. Falls back to delta kernel if sum is zero (numerical edge case).
    kernel_sum = kernel.sum()
    if kernel_sum == 0:
        return np.array([1.0], dtype=np.float32)
    return (kernel / kernel_sum).astype(np.float32)


def _make_samples_absolute(
    delta_x: NDArray[np.float32],
    delta_y: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Converts relative 2D deformation deltas to absolute pixel coordinates.

    Args:
        delta_x: The x-component of the relative deformation field.
        delta_y: The y-component of the relative deformation field.

    Returns:
        A tuple of (absolute_x, absolute_y) arrays representing absolute sample locations.
    """
    height, width = delta_x.shape

    # Creates index grids and adds deltas to get absolute positions.
    indices_x = np.arange(width, dtype=np.float32).reshape(1, width)
    indices_y = np.arange(height, dtype=np.float32).reshape(height, 1)

    # noinspection PyTypeChecker
    return indices_x + delta_x, indices_y + delta_y


@numba.njit(parallel=True, cache=True)
def _warp(
    data: NDArray[np.float32],
    result: NDArray[np.float32],
    samples_x: NDArray[np.float32],
    samples_y: NDArray[np.float32],
    order: int,
) -> None:
    """Performs 2D backward warping using parallel Numba JIT compilation.

    Samples the source data at specified coordinates using the chosen interpolation method. Each output pixel is
    computed independently, enabling parallel execution. Uses Catmull-Rom splines (tension=0) for cubic interpolation.

    Args:
        data: The 2D source array to sample from (height x width).
        result: The 1D flattened output array to write interpolated values into.
        samples_x: The 1D flattened array of x-coordinates to sample at.
        samples_y: The 1D flattened array of y-coordinates to sample at.
        order: Interpolation order (0=nearest, 1=bilinear, 3=cubic Cardinal spline).
    """
    num_samples = samples_x.size
    height = data.shape[0]
    width = data.shape[1]

    # Cubic Cardinal Spline Interpolation (order=3). Uses a 4x4 neighborhood around each sample point. Cardinal
    # splines pass through control points exactly, with tension controlling curve tightness.
    if order == _CUBIC_INTERPOLATION_ORDER:
        for sample_index in numba.prange(num_samples):
            # Allocates coefficient arrays inside the parallel loop to avoid race conditions.
            coefficients_x = np.empty((4,), dtype=np.float32)
            coefficients_y = np.empty((4,), dtype=np.float32)

            # Decomposes sample coordinates into integer indices and fractional offsets.
            sample_x = samples_x[sample_index]
            pixel_x = math.floor(sample_x)
            fraction_x = sample_x - pixel_x
            sample_y = samples_y[sample_index]
            pixel_y = math.floor(sample_y)
            fraction_y = sample_y - pixel_y

            # Interior case: full 4x4 neighborhood available (fast path).
            if 1 <= pixel_x < width - 2 and 1 <= pixel_y < height - 2:
                compute_cardinal_coefficients(interpolation_factor=fraction_x, coefficients=coefficients_x)
                compute_cardinal_coefficients(interpolation_factor=fraction_y, coefficients=coefficients_y)

                # Computes weighted sum over the 4x4 neighborhood.
                interpolated_value = 0.0
                for ky in range(4):
                    for kx in range(4):
                        interpolated_value += (
                            data[pixel_y + ky - 1, pixel_x + kx - 1] * coefficients_y[ky] * coefficients_x[kx]
                        )
                result[sample_index] = interpolated_value

            # Edge case: sample is within bounds but near the border.
            # Uses partial neighborhood and renormalizes coefficients.
            elif (
                _BOUNDARY_TOLERANCE <= sample_x <= width + _BOUNDARY_TOLERANCE
                and _BOUNDARY_TOLERANCE <= sample_y <= height + _BOUNDARY_TOLERANCE
            ):
                compute_cardinal_coefficients(interpolation_factor=fraction_x, coefficients=coefficients_x)
                compute_cardinal_coefficients(interpolation_factor=fraction_y, coefficients=coefficients_y)

                # Determines valid coefficient range based on proximity to edges.
                range_start_x, range_end_x = 0, 4
                range_start_y, range_end_y = 0, 4
                if pixel_x < 1:
                    range_start_x += 1 - pixel_x
                if pixel_x > width - 3:
                    range_end_x += (width - 3) - pixel_x
                if pixel_y < 1:
                    range_start_y += 1 - pixel_y
                if pixel_y > height - 3:
                    range_end_y += (height - 3) - pixel_y

                # Renormalizes x-coefficients to sum to 1 over the valid range.
                coefficient_sum = 0.0
                for k in range(range_start_x, range_end_x):
                    coefficient_sum += coefficients_x[k]
                coefficient_sum = 1.0 / coefficient_sum
                for k in range(range_start_x, range_end_x):
                    coefficients_x[k] *= coefficient_sum

                # Renormalizes y-coefficients to sum to 1 over the valid range.
                coefficient_sum = 0.0
                for k in range(range_start_y, range_end_y):
                    coefficient_sum += coefficients_y[k]
                coefficient_sum = 1.0 / coefficient_sum
                for k in range(range_start_y, range_end_y):
                    coefficients_y[k] *= coefficient_sum

                # Computes weighted sum over the valid partial neighborhood.
                interpolated_value = 0.0
                for ky in range(range_start_y, range_end_y):
                    for kx in range(range_start_x, range_end_x):
                        interpolated_value += (
                            data[pixel_y + ky - 1, pixel_x + kx - 1] * coefficients_y[ky] * coefficients_x[kx]
                        )
                result[sample_index] = interpolated_value

            # Out-of-bounds: sample is outside the valid image region.
            else:
                result[sample_index] = 0.0

    # Bilinear Interpolation (order=1). Uses a 2x2 neighborhood with linear weights based on fractional position.
    elif order == _BILINEAR_INTERPOLATION_ORDER:
        for sample_index in numba.prange(num_samples):
            # Decomposes sample coordinates into integer indices and fractional offsets.
            sample_x = samples_x[sample_index]
            pixel_x = math.floor(sample_x)
            fraction_x = sample_x - pixel_x
            sample_y = samples_y[sample_index]
            pixel_y = math.floor(sample_y)
            fraction_y = sample_y - pixel_y

            # Interior case: full 2x2 neighborhood available (fast path).
            if 0 <= pixel_x < width - 1 and 0 <= pixel_y < height - 1:
                interpolated_value = data[pixel_y, pixel_x] * (1.0 - fraction_y) * (1.0 - fraction_x)
                interpolated_value += data[pixel_y, pixel_x + 1] * (1.0 - fraction_y) * fraction_x
                interpolated_value += data[pixel_y + 1, pixel_x] * fraction_y * (1.0 - fraction_x)
                interpolated_value += data[pixel_y + 1, pixel_x + 1] * fraction_y * fraction_x
                result[sample_index] = interpolated_value

            # Edge case: clamps indices to valid range and adjusts fractions accordingly.
            elif (
                _BOUNDARY_TOLERANCE <= sample_x <= width + _BOUNDARY_TOLERANCE
                and _BOUNDARY_TOLERANCE <= sample_y <= height + _BOUNDARY_TOLERANCE
            ):
                if pixel_x < 0:
                    fraction_x += pixel_x
                    pixel_x = 0
                if pixel_x > width - 2:
                    fraction_x += pixel_x - (width - 2)
                    pixel_x = width - 2
                if pixel_y < 0:
                    fraction_y += pixel_y
                    pixel_y = 0
                if pixel_y > height - 2:
                    fraction_y += pixel_y - (height - 2)
                    pixel_y = height - 2

                interpolated_value = data[pixel_y, pixel_x] * (1.0 - fraction_y) * (1.0 - fraction_x)
                interpolated_value += data[pixel_y, pixel_x + 1] * (1.0 - fraction_y) * fraction_x
                interpolated_value += data[pixel_y + 1, pixel_x] * fraction_y * (1.0 - fraction_x)
                interpolated_value += data[pixel_y + 1, pixel_x + 1] * fraction_y * fraction_x
                result[sample_index] = interpolated_value

            # Out-of-bounds: sample is outside the valid image region.
            else:
                result[sample_index] = 0.0

    # Nearest Neighbor Interpolation (order=0). Rounds to the nearest pixel index without blending.
    else:
        for sample_index in numba.prange(num_samples):
            # Rounds sample coordinates to nearest integer pixel.
            sample_x = samples_x[sample_index]
            pixel_x = math.floor(sample_x + 0.5)
            sample_y = samples_y[sample_index]
            pixel_y = math.floor(sample_y + 0.5)

            # Returns pixel value if in bounds, otherwise zero.
            if 0 <= pixel_x < width and 0 <= pixel_y < height:
                result[sample_index] = data[pixel_y, pixel_x]
            else:
                result[sample_index] = 0.0


@numba.njit(parallel=True, cache=True)
def _project(
    data: NDArray[np.float32],
    result: NDArray[np.float32],
    samples_x: NDArray[np.float32],
    samples_y: NDArray[np.float32],
) -> None:
    """Performs 2D forward projection (splatting) using Numba JIT compilation.

    Each source pixel is "splatted" to its target location, distributing its value across nearby destination pixels
    using a tent (linear) kernel. Weights are accumulated and normalized at the end to handle overlapping
    contributions.

    Args:
        data: The 2D source array to project from.
        result: The 2D output array to accumulate projected values into (must be zero-initialized).
        samples_x: The x-coordinates of target positions for each source pixel.
        samples_y: The y-coordinates of target positions for each source pixel.
    """
    height = data.shape[0]
    width = data.shape[1]

    # Accumulates weights for normalization after splatting.
    weight_accumulator = np.zeros(data.shape, dtype=np.float32)

    # Splatting loop: each source pixel distributes its value to nearby destination pixels. Runs sequentially to avoid
    # write conflicts when multiple source pixels contribute to the same destination.
    for source_y in range(height):
        for source_x in range(width):
            target_x = samples_x[source_y, source_x]
            target_y = samples_y[source_y, source_x]

            # Determines the bounding box of the destination region by examining
            # where neighboring source pixels map to. This adaptive approach handles
            # non-uniform deformations where the splat size varies spatially.
            bounds_min_x = target_x
            bounds_max_x = target_x
            bounds_min_y = target_y
            bounds_max_y = target_y

            for neighbor_offset_y in range(-1, 2):
                for neighbor_offset_x in range(-1, 2):
                    # Skips the center pixel (already have its target).
                    if neighbor_offset_y * neighbor_offset_x == 0:
                        continue

                    # Handles boundary cases by extending bounds to large values.
                    if source_y + neighbor_offset_y < 0:
                        bounds_min_y = target_y - 1000000.0
                        continue
                    if source_x + neighbor_offset_x < 0:
                        bounds_min_x = target_x - 1000000.0
                        continue
                    if source_y + neighbor_offset_y >= height:
                        bounds_max_y = target_y + 1000000.0
                        continue
                    if source_x + neighbor_offset_x >= width:
                        bounds_max_x = target_x + 1000000.0
                        continue

                    # Updates bounds based on neighbor's target position.
                    neighbor_target_x = samples_x[source_y + neighbor_offset_y, source_x + neighbor_offset_x]
                    bounds_min_x = min(bounds_min_x, neighbor_target_x)
                    bounds_max_x = max(bounds_max_x, neighbor_target_x)
                    neighbor_target_y = samples_y[source_y + neighbor_offset_y, source_x + neighbor_offset_x]
                    bounds_min_y = min(bounds_min_y, neighbor_target_y)
                    bounds_max_y = max(bounds_max_y, neighbor_target_y)

            # Clamps bounds to valid image region.
            bounds_min_x = max(0, min(width - 1, bounds_min_x))
            bounds_max_x = max(0, min(width - 1, bounds_max_x))
            bounds_min_y = max(0, min(height - 1, bounds_min_y))
            bounds_max_y = max(0, min(height - 1, bounds_max_y))

            # Converts to integer pixel range for iteration.
            destination_start_x = max(0, math.floor(bounds_min_x))
            destination_end_x = min(width - 1, math.ceil(bounds_max_x))
            destination_start_y = max(0, math.floor(bounds_min_y))
            destination_end_y = min(height - 1, math.ceil(bounds_max_y))

            # Computes inverse kernel radius from the maximum extent.
            # The kernel uses a tent function that falls off linearly from the target center.
            inverse_kernel_radius = max(
                0.1,
                abs(bounds_min_y - target_y),
                abs(bounds_max_y - target_y),
                abs(bounds_min_x - target_x),
                abs(bounds_max_x - target_x),
            )
            inverse_kernel_radius = 1.0 / inverse_kernel_radius

            source_value = data[source_y, source_x]

            # Splats the source value to all destination pixels in the bounding box.
            for destination_y in range(destination_start_y, destination_end_y + 1):
                for destination_x in range(destination_start_x, destination_end_x + 1):
                    # Computes tent kernel weight (separable in x and y).
                    weight_y = 1.0 - inverse_kernel_radius * abs(destination_y - target_y)
                    weight_x = 1.0 - inverse_kernel_radius * abs(destination_x - target_x)
                    weight = max(0.0, weight_y) * max(0.0, weight_x)

                    # Accumulates weighted value and weight for later normalization.
                    result[destination_y, destination_x] += source_value * weight
                    weight_accumulator[destination_y, destination_x] += weight

    # Normalization loop: divides accumulated values by total weights. Runs in parallel since each destination pixel
    # is independent and there are no write conflicts.
    for destination_y in numba.prange(height):
        for destination_x in range(width):
            total_weight = weight_accumulator[destination_y, destination_x]
            if total_weight > 0:
                result[destination_y, destination_x] /= total_weight


def _resize(
    data: NDArray[np.float32],
    new_height: int,
    new_width: int,
    order: int = 3,
) -> NDArray[np.float32]:
    """Resizes 2D data to the specified shape using interpolation.

    Args:
        data: The 2D array to resize.
        new_height: The target height.
        new_width: The target width.
        order: The interpolation order (0=nearest, 1=linear, 3=cubic).

    Returns:
        The resized array.
    """
    height, width = data.shape

    # Creates coordinate grids mapping new pixels to source positions.
    range_y = np.linspace(0, height - 1, new_height, dtype=np.float32)
    range_x = np.linspace(0, width - 1, new_width, dtype=np.float32)
    samples_x, samples_y = np.meshgrid(range_x, range_y)

    # Applies warp to resample at the new coordinates.
    result = np.empty((new_height, new_width), dtype=data.dtype)
    _warp(
        data=data,
        result=result.ravel(),
        samples_x=samples_x.ravel(),
        samples_y=samples_y.ravel(),
        order=order,
    )
    return result
