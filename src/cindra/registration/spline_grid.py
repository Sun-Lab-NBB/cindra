"""Provides the assets for B-spline based deformation field representation used in diffeomorphic Demons registration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba
from numba import prange
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

_MINIMUM_KNOTS_FOR_FROZEN_EDGES: int = 6
"""The minimum number of knots required per dimension to freeze edges (2 on each side + 2 interior)."""


@numba.njit(cache=True, inline="always")
def compute_cardinal_coefficients(  # pragma: no cover
    interpolation_factor: float,
    coefficients: NDArray[np.float32],
) -> None:
    """Computes Catmull-Rom spline coefficients for image interpolation.

    Catmull-Rom splines are Cardinal splines with tension=0. They are interpolating splines that pass exactly
    through their control points, making them suitable for image interpolation where pixel values must be preserved
    at grid locations.

    Args:
        interpolation_factor: The position between the central lattice points, in range [0, 1].
        coefficients: The output array where to store the computed coefficients.
    """
    factor = interpolation_factor
    factor_squared = factor * factor
    factor_cubed = factor_squared * factor

    # Coefficient for p0 (leftmost point). Uses tension_factor=0.5 (Catmull-Rom).
    coefficients[0] = -0.5 * (factor_cubed - 2.0 * factor_squared + factor)

    # Coefficient for p3 (rightmost point).
    coefficients[3] = 0.5 * (factor_cubed - factor_squared)

    # Coefficient for p1 (left-center point).
    coefficients[1] = 2.0 * factor_cubed - 3.0 * factor_squared + 1.0 - coefficients[3]

    # Coefficient for p2 (right-center point).
    coefficients[2] = -2.0 * factor_cubed + 3.0 * factor_squared - coefficients[0]


@numba.njit(cache=True, inline="always")
def compute_basis_coefficients(  # pragma: no cover
    interpolation_factor: float,
    coefficients: NDArray[np.float32],
) -> None:
    """Computes uniform cubic B-spline coefficients.

    Notes:
        B-splines (basis splines) are approximating splines that do not pass through their control points but
        instead produce curves that smoothly approximate them. They provide C2 continuity and minimize bending
        energy, making them ideal for representing smooth deformation fields.

    Args:
        interpolation_factor: The position between the central lattice points, in range [0, 1].
        coefficients: The output array where to store the computed coefficients.
    """
    factor = interpolation_factor
    factor_squared = factor * factor
    factor_cubed = factor_squared * factor
    one_minus_factor = 1.0 - factor

    # Coefficient for p0 (leftmost control point)
    coefficients[0] = (one_minus_factor * one_minus_factor * one_minus_factor) / 6.0

    # Coefficient for p1 (left-center control point)
    coefficients[1] = (3.0 * factor_cubed - 6.0 * factor_squared + 4.0) / 6.0

    # Coefficient for p2 (right-center control point)
    coefficients[2] = (-3.0 * factor_cubed + 3.0 * factor_squared + 3.0 * factor + 1.0) / 6.0

    # Coefficient for p3 (rightmost control point)
    coefficients[3] = factor_cubed / 6.0


class SplineGrid:
    """Represents a 2D deformation field using uniform cubic B-splines for diffeomorphic regularization.

    This class stores one knot array per dimension and provides methods for converting between dense deformation
    fields and sparse B-spline representations while enforcing diffeomorphic constraints.

    Notes:
        B-splines provide C2 continuity and minimize bending energy, making them ideal for smooth deformation fields.
        The grid applies two key constraints for diffeomorphic (invertible) deformations: injectivity constraints
        prevent grid folding, and frozen edges ensure zero deformation at image boundaries.

    Args:
        field_height: The height of the image field this grid applies to, in pixels.
        field_width: The width of the image field this grid applies to, in pixels.
        sampling: The spacing between B-spline control points (knots) in pixels. Larger values produce smoother
            deformations with less local detail, while smaller values allow finer deformation control at the cost of
            reduced smoothness.

    Attributes:
        _field_shape: The shape of the image field this grid applies to, as (height, width).
        _grid_sampling: The spacing between B-spline control points (knots) in pixels.
        _grid_shape: The shape of the knot grid, as (height, width), computed from field shape and sampling.
        _knots: A tuple of two knot arrays, one per dimension [Y, X]. Each array stores the B-spline control point
            values for that dimension.
    """

    def __init__(self, field_height: int, field_width: int, sampling: float) -> None:
        self._field_shape: tuple[int, int] = (field_height, field_width)
        self._grid_sampling: float = sampling

        # Computes grid shape: (field_dim - 1) / sampling gives the number of grid intervals spanning the field,
        # truncated to whole intervals. The +4 adds boundary padding since cubic B-spline evaluation at any point
        # requires 4 surrounding knots (2 beyond each field edge).
        grid_height = int((field_height - 1) / self._grid_sampling) + 4
        grid_width = int((field_width - 1) / self._grid_sampling) + 4
        self._grid_shape: tuple[int, int] = (grid_height, grid_width)

        # Initializes the knot (B-spline control point) arrays, one per dimension (Y, X).
        self._knots: tuple[NDArray[np.float32], NDArray[np.float32]] = (
            np.zeros(self._grid_shape, dtype=np.float32),
            np.zeros(self._grid_shape, dtype=np.float32),
        )

    @property
    def ndim(self) -> int:
        """Returns the number of grid dimensions, which is fixed to 2 in the current SplineGrid
        implementation.
        """
        return len(self._field_shape)

    @property
    def field_shape(self) -> tuple[int, int]:
        """Returns the shape of the underlying image field as (height, width)."""
        return self._field_shape

    @property
    def grid_shape(self) -> tuple[int, int]:
        """Returns the shape of the B-spline knot grid."""
        return self._grid_shape

    @property
    def grid_sampling(self) -> float:
        """Returns the spacing between grid knots in pixels."""
        return self._grid_sampling

    @staticmethod
    def compute_grid_shape(field_height: int, field_width: int, grid_sampling: float) -> tuple[int, int]:
        """Computes the grid shape for the given field and sampling parameters without creating a full instance.

        Notes:
            This method is primarily used to ensure that the proposed grid size meets the minimum size requirements.

        Args:
            field_height: The height of the underlying image field.
            field_width: The width of the underlying image field.
            grid_sampling: The spacing between knots (B-spline control points) in pixels.

        Returns:
            The shape of the knot grid as (height, width).
        """
        # Computes grid shape: (field_dim - 1) / sampling gives the number of grid intervals spanning the field,
        # truncated to whole intervals. The +4 adds boundary padding since cubic B-spline evaluation at any point
        # requires 4 surrounding knots (2 beyond each field edge).
        grid_height = int((field_height - 1) / grid_sampling) + 4
        grid_width = int((field_width - 1) / grid_sampling) + 4
        return grid_height, grid_width

    @property
    def deformation_fields(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Returns two arrays (Y, X), representing the deformation fields for each dimension of the underlying image."""
        field_y: NDArray[np.float32] = np.zeros(self.field_shape, dtype=np.float32)
        field_x: NDArray[np.float32] = np.zeros(self.field_shape, dtype=np.float32)
        _sample_grid(result=field_y, grid_sampling=self._grid_sampling, knots=self._get_knots(dimension=0))
        _sample_grid(result=field_x, grid_sampling=self._grid_sampling, knots=self._get_knots(dimension=1))
        return field_y, field_x

    def set_from_fields(
        self,
        field_y: NDArray[np.float32],
        field_x: NDArray[np.float32],
        injective: bool = True,
        injective_factor: float = 0.9,
        freeze_edges: bool = True,
    ) -> bool:
        """Sets the grid knots from dense deformation fields and applies diffeomorphic constraints.

        Args:
            field_y: The Y-dimension displacement field array.
            field_x: The X-dimension displacement field array.
            injective: Determines whether to apply injectivity constraint to prevent grid folding.
            injective_factor: The scaling factor for the injectivity limit (0 < factor <= 1).
            freeze_edges: Determines whether to freeze the edges, preventing them from being deformed.

        Returns:
            True if all constraints were successfully applied, False if the grid is too small for frozen edges.
        """
        # Fits B-spline knots to the deformation fields using least-squares (Lee et al.).
        _fit_knots_to_field(
            grid_sampling=self._grid_sampling,
            knots=self._get_knots(dimension=0),
            field=field_y,
        )
        _fit_knots_to_field(
            grid_sampling=self._grid_sampling,
            knots=self._get_knots(dimension=1),
            field=field_x,
        )

        # Applies injectivity constraint and freezes edges if requested.
        if injective:
            self._unfold(factor=injective_factor)

        return not (freeze_edges and not self._freeze_edges())

    def _get_knots(self, dimension: int) -> NDArray[np.float32]:
        """Returns the knot array for the requested image field dimension.

        Notes:
            Dimension indexing starts from 0. Dimension 0 corresponds to the Y (vertical) dimension and
            dimension 1 corresponds to the X (horizontal) dimension.

        Args:
            dimension: The image field dimension for which to retrieve the knot array.

        Returns:
            The knot array for the requested dimension.
        """
        return self._knots[dimension]

    def _unfold(self, factor: float = 0.9) -> None:
        """Prevents folds in the grid by limiting the B-spline control values (knots) to ensure injectivity.

        Args:
            factor: The scaling factor for the injectivity limit (0 < factor <= 1). Values closer to 1.0 allow larger
                deformations, while smaller values are more conservative.

        Notes:
            Based on Choi & Lee (2000), "Injectivity conditions of 2D and 3D uniform cubic B-spline functions".
        """
        # Computes the maximum allowed B-spline knot displacement to prevent grid folding. The constant 2.046392675 is
        # the theoretical injectivity bound for 2D cubic B-splines - knot values exceeding 1/K of the grid spacing can
        # cause the deformation to become non-injective (folded). The factor scales this limit conservatively.
        limit = (1.0 / 2.046392675) * self._grid_sampling * factor

        # Applies smooth exponential limiting to each knot array. The formula maps the knot values to the range (-limit,
        # +limit) using a soft saturation curve: small values pass through nearly unchanged, while large values are
        # smoothly compressed toward the limit without hard clipping discontinuities.
        for dimension in range(self.ndim):
            knots = self._get_knots(dimension=dimension).ravel()
            knots[:] = limit * (np.exp(-np.abs(knots) / limit) - 1) * -np.sign(knots)

    def _freeze_edges(self) -> bool:
        """Freezes the outer knots to zero to ensure deformation is zero at image edges.

        Returns:
            True if edges were successfully frozen, False if the grid is too small.
        """
        for dimension in range(self.ndim):
            knots = self._get_knots(dimension=dimension)

            if knots.shape[dimension] < _MINIMUM_KNOTS_FOR_FROZEN_EDGES:
                return False

            # Determines where the field's trailing edge falls between grid knots, since the field does not perfectly
            # map onto the grid's knots.
            field_edge = self._field_shape[dimension] - 1
            grid_edge = (self._grid_shape[dimension] - 4) * self._grid_sampling
            edge_interpolation_factor = 1.0 - (field_edge - grid_edge) / self._grid_sampling

            # Computes the B-spline coefficients at the field edge position.
            coefficients: NDArray[np.float32] = np.zeros((4,), dtype=np.float32)
            compute_basis_coefficients(interpolation_factor=edge_interpolation_factor, coefficients=coefficients)

            # Freezes knots for Y dimension (operates on rows).
            if dimension == 0:
                # Leading edge: zero boundary knots.
                knots[0] = 0
                knots[1] = -0.25 * knots[2]
                # Trailing edge: adjusts knots to produce zero deformation at field edge.
                knots[-3] = (1 - edge_interpolation_factor) * knots[-3]
                knots[-1] = 0
                knots[-2] = -(knots[-3] * coefficients[2] + knots[-4] * coefficients[3]) / coefficients[1]

            # Freezes knots for X dimension (operates on columns).
            elif dimension == 1:
                knots[:, 0] = 0
                knots[:, 1] = -0.25 * knots[:, 2]
                knots[:, -3] = (1 - edge_interpolation_factor) * knots[:, -3]
                knots[:, -1] = 0
                knots[:, -2] = -(knots[:, -3] * coefficients[2] + knots[:, -4] * coefficients[3]) / coefficients[1]

        return True


@numba.njit(cache=True, parallel=True)
def _sample_grid(  # pragma: no cover
    result: NDArray[np.float32],
    grid_sampling: float,
    knots: NDArray[np.float32],
) -> None:
    """Samples the B-spline grid at all pixels of the underlying image field.

    For each pixel in the result array, computes the B-spline interpolated value from the surrounding 4x4 knot
    neighborhood and stores it in the result array.

    Args:
        result: The output array to store sampled deformation values, modified in-place.
        grid_sampling: The spacing between B-spline control points (knots) in pixels.
        knots: The 2D array of B-spline knot values.
    """
    # Parallelizes the computation over rows to improve performance.
    for y in prange(result.shape[0]):
        # Each thread gets its own coefficient arrays.
        coefficients_y = np.empty((4,), dtype=np.float32)
        coefficients_x = np.empty((4,), dtype=np.float32)

        for x in range(result.shape[1]):
            # Computes the reference knot index and interpolation factor for each axis.
            # The +1 corrects for boundary padding in the knot grid.
            grid_position_y = y / grid_sampling + 1
            knot_index_y = int(grid_position_y)
            interpolation_factor_y = grid_position_y - knot_index_y
            grid_position_x = x / grid_sampling + 1
            knot_index_x = int(grid_position_x)
            interpolation_factor_x = grid_position_x - knot_index_x

            # Computes B-spline basis coefficients at this pixel position.
            compute_basis_coefficients(interpolation_factor=interpolation_factor_y, coefficients=coefficients_y)
            compute_basis_coefficients(interpolation_factor=interpolation_factor_x, coefficients=coefficients_x)

            # Accumulates weighted contributions from the 4x4 knot neighborhood.
            sampled_value = 0.0
            knot_y = knot_index_y - 1
            for offset_y in range(4):
                knot_x = knot_index_x - 1
                for offset_x in range(4):
                    sampled_value += coefficients_y[offset_y] * coefficients_x[offset_x] * knots[knot_y, knot_x]
                    knot_x += 1
                knot_y += 1

            result[y, x] = sampled_value


@numba.njit(cache=True)
def _fit_knots_to_field(  # pragma: no cover
    grid_sampling: float,
    knots: NDArray[np.float32],
    field: NDArray[np.float32],
) -> None:
    """Fits B-spline knots to a deformation field using least-squares (Lee et al.).

    For each pixel, distributes its contribution to the surrounding 4x4 knot neighborhood. After accumulating
    all contributions, computes final knot values by dividing the accumulated numerator by denominator.

    Args:
        grid_sampling: The spacing between B-spline control points (knots) in pixels.
        knots: The 2D knot array to update in-place.
        field: The 2D deformation field values.
    """
    coefficients_y = np.empty((4,), dtype=np.float32)
    coefficients_x = np.empty((4,), dtype=np.float32)

    numerator = np.zeros_like(knots)
    denominator = np.zeros_like(knots)

    # Accumulates contributions from each pixel to its surrounding knots.
    for y in range(field.shape[0]):
        for x in range(field.shape[1]):
            field_value = field[y, x]

            # Computes the reference knot index and interpolation factor for each axis.
            # The +1 corrects for boundary padding in the knot grid.
            grid_position_y = y / grid_sampling + 1
            knot_index_y = int(grid_position_y)
            interpolation_factor_y = grid_position_y - knot_index_y
            grid_position_x = x / grid_sampling + 1
            knot_index_x = int(grid_position_x)
            interpolation_factor_x = grid_position_x - knot_index_x

            # Computes B-spline basis coefficients at this pixel position.
            compute_basis_coefficients(interpolation_factor=interpolation_factor_y, coefficients=coefficients_y)
            compute_basis_coefficients(interpolation_factor=interpolation_factor_x, coefficients=coefficients_x)

            # Pre-normalizes the value by the sum of squared basis coefficients.
            coefficient_sum_squared = 0.0
            for offset_y in range(4):
                for offset_x in range(4):
                    coefficient = coefficients_y[offset_y] * coefficients_x[offset_x]
                    coefficient_sum_squared += coefficient * coefficient
            normalized_value = field_value / coefficient_sum_squared

            # Accumulates contributions to each knot in the 4x4 neighborhood.
            for offset_y in range(4):
                knot_y = offset_y + knot_index_y - 1
                for offset_x in range(4):
                    knot_x = offset_x + knot_index_x - 1
                    basis_coefficient = coefficients_y[offset_y] * coefficients_x[offset_x]
                    coefficient_squared = basis_coefficient * basis_coefficient
                    numerator[knot_y, knot_x] += coefficient_squared * (normalized_value * basis_coefficient)
                    denominator[knot_y, knot_x] += coefficient_squared

    # Finalizes the knot values by dividing numerator by denominator.
    for i in range(knots.size):
        if denominator.flat[i] > 0.0:
            knots.flat[i] = numerator.flat[i] / denominator.flat[i]
