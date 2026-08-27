"""Provides the assets for B-spline based deformation field representation used in diffeomorphic Demons registration."""

from __future__ import annotations

from typing import TYPE_CHECKING
from functools import lru_cache

import numba
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

MINIMUM_KNOTS_FOR_FROZEN_EDGES: int = 6
"""The minimum number of knots required per dimension to freeze edges (2 on each side + 2 interior)."""


@numba.njit(cache=True, inline="always")
def compute_cardinal_coefficients(  # pragma: no cover
    interpolation_factor: float,
    coefficients: NDArray[np.float32],
) -> None:
    """Computes Catmull-Rom spline coefficients for image interpolation.

    Notes:
        Catmull-Rom splines are Cardinal splines with tension=0. They are interpolating splines that pass exactly
        through their control points, making them suitable for image interpolation where pixel values must be
        preserved at grid locations.

    Args:
        interpolation_factor: The position between the central lattice points, in range [0, 1].
        coefficients: The output array that stores the computed coefficients.
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
        # truncated to whole intervals. Spanning those intervals takes one more knot than intervals, and cubic
        # B-spline evaluation adds one knot before the field's first knot and two past its last, so the total is +4.
        grid_height = int((field_height - 1) / self._grid_sampling) + 4
        grid_width = int((field_width - 1) / self._grid_sampling) + 4
        self._grid_shape: tuple[int, int] = (grid_height, grid_width)

        # Initializes the knot (B-spline control point) arrays, one per dimension (Y, X).
        self._knots: tuple[NDArray[np.float32], NDArray[np.float32]] = (
            np.zeros(self._grid_shape, dtype=np.float32),
            np.zeros(self._grid_shape, dtype=np.float32),
        )

    def __repr__(self) -> str:
        """Returns a string representation of the SplineGrid instance."""
        return (
            f"SplineGrid(field_shape={self._field_shape}, grid_shape={self._grid_shape}, "
            f"grid_sampling={self._grid_sampling})"
        )

    @property
    def dimension_count(self) -> int:
        """Returns the number of grid dimensions, which is fixed to 2 in the current SplineGrid implementation."""
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

    @property
    def deformation_fields(self) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Returns two arrays (Y, X), representing the deformation fields for each dimension of the underlying image."""
        # _sample_grid writes every element of both fields, so the allocation needs no zero fill.
        field_y: NDArray[np.float32] = np.empty(self.field_shape, dtype=np.float32)
        field_x: NDArray[np.float32] = np.empty(self.field_shape, dtype=np.float32)
        _sample_grid(result=field_y, grid_sampling=self._grid_sampling, knots=self._get_knots(dimension=0))
        _sample_grid(result=field_x, grid_sampling=self._grid_sampling, knots=self._get_knots(dimension=1))
        return field_y, field_x

    @staticmethod
    def compute_grid_shape(field_height: int, field_width: int, grid_sampling: float) -> tuple[int, int]:
        """Computes the grid shape for the given field and sampling parameters without creating a full instance.

        Args:
            field_height: The height of the underlying image field.
            field_width: The width of the underlying image field.
            grid_sampling: The spacing between knots (B-spline control points) in pixels.

        Returns:
            The shape of the knot grid as (height, width).
        """
        # Computes grid shape: (field_dim - 1) / sampling gives the number of grid intervals spanning the field,
        # truncated to whole intervals. Spanning those intervals takes one more knot than intervals, and cubic
        # B-spline evaluation adds one knot before the field's first knot and two past its last, so the total is +4.
        grid_height = int((field_height - 1) / grid_sampling) + 4
        grid_width = int((field_width - 1) / grid_sampling) + 4
        return grid_height, grid_width

    def set_from_fields(
        self,
        field_y: NDArray[np.float32],
        field_x: NDArray[np.float32],
        *,
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

        Notes:
            Based on Choi & Lee (2000), "Injectivity conditions of 2D and 3D uniform cubic B-spline functions".

        Args:
            factor: The scaling factor for the injectivity limit (0 < factor <= 1). Values closer to 1.0 allow larger
                deformations, while smaller values are more conservative.
        """
        # Computes the maximum allowed B-spline knot displacement to prevent grid folding. The constant 2.046392675 is
        # the theoretical injectivity bound for 2D cubic B-splines. Knot values exceeding the grid spacing divided by
        # that bound can make the deformation non-injective (folded). The factor scales this limit conservatively.
        limit = (1.0 / 2.046392675) * self._grid_sampling * factor

        # Applies smooth exponential limiting to each knot array. The formula maps the knot values to the range (-limit,
        # +limit) using a soft saturation curve: small values pass through nearly unchanged, while large values are
        # smoothly compressed toward the limit without hard clipping discontinuities.
        for dimension in range(self.dimension_count):
            knots = self._get_knots(dimension=dimension).ravel()
            knots[:] = limit * (np.exp(-np.abs(knots) / limit) - 1) * -np.sign(knots)

    def _freeze_edges(self) -> bool:
        """Freezes the outer knots to zero to ensure deformation is zero at image edges.

        Returns:
            True if edges were successfully frozen, False if the grid is too small.
        """
        for dimension in range(self.dimension_count):
            knots = self._get_knots(dimension=dimension)

            if knots.shape[dimension] < MINIMUM_KNOTS_FOR_FROZEN_EDGES:
                return False

            # Determines where the field's trailing edge falls between grid knots, since the field does not perfectly
            # map onto the grid's knots.
            field_edge = self._field_shape[dimension] - 1
            grid_edge = (self._grid_shape[dimension] - 4) * self._grid_sampling
            edge_interpolation_factor = 1.0 - (field_edge - grid_edge) / self._grid_sampling

            coefficients: NDArray[np.float32] = np.zeros((4,), dtype=np.float32)
            _compute_basis_coefficients(interpolation_factor=edge_interpolation_factor, coefficients=coefficients)

            # Freezes knots for Y dimension (operates on rows).
            if dimension == 0:
                # Leading edge: zeros the outermost knot and offsets the next one to cancel edge deformation.
                knots[0] = 0
                knots[1] = -0.25 * knots[2]
                # Trailing edge: adjusts knots to produce zero deformation at field edge.
                knots[-3] = (1 - edge_interpolation_factor) * knots[-3]
                knots[-1] = 0
                knots[-2] = -(knots[-3] * coefficients[2] + knots[-4] * coefficients[3]) / coefficients[1]

            # Freezes knots for X dimension (operates on columns).
            elif dimension == 1:  # pragma: no branch - SplineGrid is always 2D, so dimension is only ever 0 or 1.
                knots[:, 0] = 0
                knots[:, 1] = -0.25 * knots[:, 2]
                knots[:, -3] = (1 - edge_interpolation_factor) * knots[:, -3]
                knots[:, -1] = 0
                knots[:, -2] = -(knots[:, -3] * coefficients[2] + knots[:, -4] * coefficients[3]) / coefficients[1]

        return True


@numba.njit(cache=True, inline="always")
def _compute_basis_coefficients(  # pragma: no cover
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
        coefficients: The output array that stores the computed coefficients.
    """
    factor = interpolation_factor
    factor_squared = factor * factor
    factor_cubed = factor_squared * factor
    one_minus_factor = 1.0 - factor

    # Coefficient for p0 (leftmost control point).
    coefficients[0] = (one_minus_factor * one_minus_factor * one_minus_factor) / 6.0

    # Coefficient for p1 (left-center control point).
    coefficients[1] = (3.0 * factor_cubed - 6.0 * factor_squared + 4.0) / 6.0

    # Coefficient for p2 (right-center control point).
    coefficients[2] = (-3.0 * factor_cubed + 3.0 * factor_squared + 3.0 * factor + 1.0) / 6.0

    # Coefficient for p3 (rightmost control point).
    coefficients[3] = factor_cubed / 6.0


@lru_cache(maxsize=16)
def _compute_basis_matrices(
    extent: int,
    grid_sampling: float,
    knot_count: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Builds the B-spline basis matrices that relate one axis of the knot grid to one axis of the image field.

    Notes:
        Row 'i' of each matrix carries the four basis coefficients of the pixel at index 'i' along the axis, written
        into the columns of the four knots that pixel reads. The three matrices hold those coefficients raised to the
        first, second, and third power, which are the three weightings the grid sampling and the least-squares fit
        consume. The cubic B-spline basis is separable, so a two-dimensional sample or fit factors into one matrix
        product per axis.

        The matrices depend on the axis geometry alone, so a registration reuses the same few sets across all of its
        iterations, which is what the cache serves. The returned matrices are read-only, because every caller shares
        one instance of them.

    Args:
        extent: The length of the image field axis, in pixels.
        grid_sampling: The spacing between B-spline control points (knots) in pixels.
        knot_count: The number of knots spanning the axis.

    Returns:
        A tuple of three matrices of shape (extent, knot_count), holding the basis coefficients raised to the first,
        the second, and the third power.
    """
    linear: NDArray[np.float32] = np.zeros((extent, knot_count), dtype=np.float32)
    quadratic: NDArray[np.float32] = np.zeros((extent, knot_count), dtype=np.float32)
    cubic: NDArray[np.float32] = np.zeros((extent, knot_count), dtype=np.float32)

    coefficients: NDArray[np.float32] = np.zeros((4,), dtype=np.float32)
    for index in range(extent):
        # The +1 corrects for boundary padding in the knot grid.
        grid_position = index / grid_sampling + 1.0
        knot_index = int(grid_position)
        _compute_basis_coefficients(interpolation_factor=grid_position - knot_index, coefficients=coefficients)

        for offset in range(4):
            column = offset + knot_index - 1
            linear[index, column] += coefficients[offset]
            quadratic[index, column] += coefficients[offset] ** 2
            cubic[index, column] += coefficients[offset] ** 3

    for matrix in (linear, quadratic, cubic):
        matrix.flags.writeable = False

    return linear, quadratic, cubic


def _sample_grid(
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
    row_basis, _, _ = _compute_basis_matrices(
        extent=result.shape[0], grid_sampling=grid_sampling, knot_count=knots.shape[0]
    )
    column_basis, _, _ = _compute_basis_matrices(
        extent=result.shape[1], grid_sampling=grid_sampling, knot_count=knots.shape[1]
    )

    # Each pixel weights its 4x4 knot neighborhood by the outer product of its two coefficient vectors, and the basis
    # matrices carry those vectors on their rows, so the whole field resolves as one matrix product per axis.
    np.matmul(row_basis @ knots, column_basis.T, out=result)


def _fit_knots_to_field(
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
    _, row_squared, row_cubed = _compute_basis_matrices(
        extent=field.shape[0], grid_sampling=grid_sampling, knot_count=knots.shape[0]
    )
    _, column_squared, column_cubed = _compute_basis_matrices(
        extent=field.shape[1], grid_sampling=grid_sampling, knot_count=knots.shape[1]
    )

    # Each pixel is pre-normalized by the sum of the squared basis products over its 4x4 neighborhood. That sum is the
    # product of the two per-axis sums of squared coefficients, which are the row sums of the squared basis matrices.
    row_scale = row_squared.sum(axis=1)
    column_scale = column_squared.sum(axis=1)
    scaled_field = field / (row_scale[:, np.newaxis] * column_scale[np.newaxis, :])

    # A pixel contributes the cube of its basis product to the numerator and the square of it to the denominator, and
    # both weights factor across the two axes, so the accumulation over every pixel becomes a pair of matrix products.
    numerator = row_cubed.T @ scaled_field @ column_cubed
    denominator = np.outer(row_squared.sum(axis=0), column_squared.sum(axis=0))

    # A knot that no pixel reaches carries a zero denominator and keeps the value it already holds, which is what
    # leaves the outer padding knots of a grid fitted more than once under the values the previous fit gave them.
    np.divide(numerator, denominator, out=knots, where=denominator > 0.0)
