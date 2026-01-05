"""Unified spline grid module for B-spline based deformation field representation.

This module provides:
- Spline coefficient computation functions (for interpolation)
- SplineGrid: B-spline grid for diffeomorphic deformation regularization
- Numba-optimized grid sampling and setting functions

Copyright 2010-2017 (C) Almar Klein (original pirt library)
"""

import numba
import numpy as np
from numpy.typing import NDArray


@numba.njit(cache=True, inline="always")
def compute_cardinal_coefficients(
    interpolation_factor: float,
    coefficients: NDArray[np.float32],
    tension: float,
) -> None:
    """Computes Cardinal spline coefficients for the given tension parameter.

    Notes:
        Cardinal splines are interpolating splines that pass exactly through their control points, making them
        suitable for image interpolation where pixel values must be preserved at grid locations. The tension
        parameter controls the tightness of the curve: tension=0 produces a Catmull-Rom spline (smooth, natural
        appearance), tension=1 produces linear interpolation, and tension=-1 produces exaggerated overshoots.

    Args:
        interpolation_factor: The position between the central lattice points, in range [0, 1].
        coefficients: The output array where to store the computed coefficients.
        tension: The tension parameter in range [-1, 1].
    """
    # Tension scaling factor (0.5 at tension=0 for Catmull-Rom)
    tension_factor = 0.5 * (1.0 - tension)

    t = interpolation_factor
    t_squared = t * t
    t_cubed = t_squared * t

    # Coefficient for p0 (leftmost point)
    coefficients[0] = -tension_factor * (t_cubed - 2.0 * t_squared + t)

    # Coefficient for p3 (rightmost point)
    coefficients[3] = tension_factor * (t_cubed - t_squared)

    # Coefficient for p1 (left-center point)
    coefficients[1] = 2.0 * t_cubed - 3.0 * t_squared + 1.0 - coefficients[3]

    # Coefficient for p2 (right-center point)
    coefficients[2] = -2.0 * t_cubed + 3.0 * t_squared - coefficients[0]


@numba.njit(cache=True, inline="always")
def compute_basis_coefficients(
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
    t = interpolation_factor
    t_squared = t * t
    t_cubed = t_squared * t
    one_minus_t = 1.0 - t

    # Coefficient for p0 (leftmost control point)
    coefficients[0] = (one_minus_t * one_minus_t * one_minus_t) / 6.0

    # Coefficient for p1 (left-center control point)
    coefficients[1] = (3.0 * t_cubed - 6.0 * t_squared + 4.0) / 6.0

    # Coefficient for p2 (right-center control point)
    coefficients[2] = (-3.0 * t_cubed + 3.0 * t_squared + 3.0 * t + 1.0) / 6.0

    # Coefficient for p3 (rightmost control point)
    coefficients[3] = t_cubed / 6.0


class SplineGrid:
    """Represents a multidimensional deformation field using uniform cubic B-splines for diffeomorphic regularization.

    This class stores one knot array per dimension and provides methods for converting between dense deformation
    fields and sparse B-spline representations while enforcing diffeomorphic constraints.

    Notes:
        B-splines provide C2 continuity and minimize bending energy, making them ideal for smooth deformation fields.
        The grid applies two key constraints for diffeomorphic (invertible) deformations: injectivity constraints
        prevent grid folding, and frozen edges ensure zero deformation at image boundaries.

    Args:
        field_shape: The shape of the image field this grid applies to, in (height, width) order.
        sampling: The spacing between B-spline control points (knots) in pixels. Larger values produce smoother
            deformations with less local detail, while smaller values allow finer deformation control at the cost of
            reduced smoothness.
    """

    def __init__(self, field_shape: tuple[int, int], sampling: float) -> None:
        self._field_shape = field_shape
        self._grid_sampling = float(sampling)

        # Compute grid shape for each dimension
        # +3 for padding, +1 because first and last row contain knots
        self._grid_shape = tuple(
            int((self._field_shape[d] - 1) / self._grid_sampling) + 4
            for d in range(self.ndim)
        )

        # Initialize knot arrays (one per dimension), created lazily
        self._knots: list[NDArray[np.float64] | None] = [None for _ in range(self.ndim)]

    @property
    def ndim(self) -> int:
        """The number of dimensions."""
        return len(self._field_shape)

    @property
    def field_shape(self) -> tuple[int, int]:
        """The shape of the underlying field."""
        return self._field_shape

    @property
    def grid_shape(self) -> tuple[int, int]:
        """The shape of the knot grid."""
        return self._grid_shape

    @property
    def grid_sampling(self) -> float:
        """The spacing between knots in pixels."""
        return self._grid_sampling

    def _get_knots(self, d: int) -> NDArray[np.float64]:
        """Returns the knot array for dimension d, creating it if necessary."""
        if self._knots[d] is None:
            self._knots[d] = np.zeros(self._grid_shape, dtype=np.float64)
        return self._knots[d]

    @staticmethod
    def compute_grid_shape(field_shape: tuple[int, int], grid_sampling: float) -> tuple[int, int]:
        """Computes the grid shape for given field and sampling parameters without creating a full instance.

        Args:
            field_shape: The shape of the underlying field.
            grid_sampling: The spacing between knots in world units.

        Returns:
            The shape of the knot grid as a tuple.
        """
        return tuple(
            int((field_shape[d] - 1) / grid_sampling) + 4
            for d in range(len(field_shape))
        )

    def get_fields(self) -> list[NDArray[np.float32]]:
        """Samples the grid to produce dense deformation fields.

        Returns:
            A list of 2D arrays, one per dimension, representing the deformation field.
        """
        if self.ndim != 2:
            raise ValueError("Only 2D grids are supported.")

        fields = []
        for d in range(self.ndim):
            result = np.zeros(self.field_shape, dtype=np.float32)
            _get_field2(result, self._grid_sampling, self._get_knots(d))
            fields.append(result)

        return fields

    def set_from_fields(
        self,
        fields: list[NDArray[np.float32]],
        weights: NDArray[np.float32] | None = None,
        injective: bool | float = True,
        frozenedge: bool = True,
    ) -> None:
        """Sets the grid knots from dense deformation fields and applies diffeomorphic constraints.

        Args:
            fields: A list of 2D arrays, one per dimension, representing the deformation field.
            weights: Optional weights for field elements.
            injective: Whether to prevent grid folding. Can be True, False, or a float factor.
            frozenedge: Whether to freeze edges to zero deformation.
        """
        if len(fields) != self.ndim:
            raise ValueError("Must provide a field for each dimension.")
        if self.ndim != 2:
            raise ValueError("Only 2D grids are supported.")

        for d in range(self.ndim):
            field = fields[d]
            if self.field_shape != field.shape:
                raise ValueError("Field shape does not match grid field shape.")

            w = weights if weights is not None else np.ones_like(field)
            if field.dtype != w.dtype:
                raise ValueError("Field and weights must be of the same type.")

            knots = self._get_knots(d)
            num, dnum = _set_field2(self._grid_sampling, knots, field, w)
            _set_field_using_num_and_dnum(knots.ravel(), num.ravel(), dnum.ravel())

        if injective:
            self._unfold(injective)
        if frozenedge:
            self._freeze_edges()

    def _unfold(self, factor: bool | float) -> None:
        """Prevents folds in the grid by limiting knot values to ensure injectivity.

        Notes:
            Based on Choi, Yongchoel, and Seungyong Lee. 2000. "Injectivity conditions of 2D and 3D uniform cubic
            B-spline functions".
        """
        mode = 2
        if factor is False:
            return
        if factor is True:
            factor = 0.9
        elif factor < 0:
            mode = 1
            factor = -factor

        # K factor for 2D B-spline injectivity
        K = 2.046392675
        limit = (1.0 / K) * self._grid_sampling * factor

        for d in range(self.ndim):
            knots = self._get_knots(d).ravel()

            if mode == 1:
                # Hard limit
                (I,) = np.where(np.abs(knots) > limit)
                knots[I] = limit * np.sign(knots[I])
            elif mode == 2:
                # Smooth limit
                f = np.exp(-np.abs(knots) / limit)
                knots[:] = limit * (f - 1) * -np.sign(knots)

    def _freeze_edges(self) -> None:
        """Freezes outer knots to zero to ensure deformation is zero at image edges."""

        def get_t_factor(d: int) -> float:
            field_edge = self._field_shape[d] - 1
            grid_edge = (self._grid_shape[d] - 4) * self._grid_sampling
            return 1.0 - (field_edge - grid_edge) / self._grid_sampling

        for d in range(self.ndim):
            knots = self._get_knots(d)

            # Check if grid is large enough
            if knots.shape[d] < 6:
                knots[:] = 0
                continue

            if d == 0:
                knots[0] = 0
                knots[1] = -0.25 * knots[2]

                t = get_t_factor(d)
                coeffs = np.zeros((4,), np.float32)
                compute_basis_coefficients(t, coeffs)

                knots[-3] = (1 - t) * knots[-3]
                knots[-1] = 0
                k3, k4 = knots[-3], knots[-4]
                knots[-2] = -(k3 * coeffs[2] + k4 * coeffs[3]) / coeffs[1]

            elif d == 1:
                knots[:, 0] = 0
                knots[:, 1] = -0.25 * knots[:, 2]

                t = get_t_factor(d)
                coeffs = np.zeros((4,), np.float32)
                compute_basis_coefficients(t, coeffs)

                knots[:, -3] = (1 - t) * knots[:, -3]
                knots[:, -1] = 0
                k3, k4 = knots[:, -3], knots[:, -4]
                knots[:, -2] = -(k3 * coeffs[2] + k4 * coeffs[3]) / coeffs[1]


# =============================================================================
# Numba-optimized Grid Functions
# =============================================================================


@numba.jit(nopython=True, nogil=True)
def _get_field2(result, grid_sampling, knots):
    """Sample the 2D grid at all pixels of the underlying field."""
    if result.ndim != 2:
        raise ValueError("This function can only sample 2D grids.")

    ccy = np.empty((4,), np.float32)
    ccx = np.empty((4,), np.float32)

    # For each pixel ...
    for y in range(result.shape[0]):
        for x in range(result.shape[1]):
            # Calculate what is the leftmost (reference) knot on the grid,
            # and the ratio between closest and second closest knot.
            # Note the +1 to correct for padding.
            tmp = y / grid_sampling + 1
            gy = int(tmp)
            ty = tmp - gy
            tmp = x / grid_sampling + 1
            gx = int(tmp)
            tx = tmp - gx

            # Get coefficients
            compute_basis_coefficients(ty, ccy)
            compute_basis_coefficients(tx, ccx)

            # Init value
            val = 0.0

            # For each knot ...
            jj = gy - 1  # y-location of first knot
            for j in range(4):
                ii = gx - 1  # x-location of first knot
                for i in range(4):
                    # Calculate interpolated value.
                    val += ccy[j] * ccx[i] * knots[jj, ii]
                    ii += 1
                jj += 1

            # Store value in result array
            result[y, x] = val


@numba.jit(nopython=True, nogil=True)
def _set_field_using_num_and_dnum(knots_, num_, dnum_):
    """Divide numerator by denominator to get final knot values."""
    for i in range(knots_.size):
        n = dnum_[i]
        if n > 0.0:
            knots_[i] = num_[i] / n


@numba.jit(nopython=True, nogil=True)
def _set_field2(grid_sampling, knots, field, weights):
    """Set grid knots from field values using least-squares B-spline fitting."""
    ccy = np.empty((4,), np.float32)
    ccx = np.empty((4,), np.float32)

    # Create temporary arrays the same size as the grid
    num = np.zeros_like(knots)
    dnum = np.zeros_like(knots)

    # For each pixel ...
    for y in range(field.shape[0]):
        for x in range(field.shape[1]):
            # Get val and alpha
            val = field[y, x]
            weight = weights[y, x]

            # Evaluate this one?
            if weight <= 0.0:
                continue

            # Calculate what is the leftmost (reference) knot on the grid,
            # and the ratio between closest and second closest knot.
            # Note the +1 to correct for padding.
            tmp = y / grid_sampling + 1
            gy = int(tmp)
            ty = tmp - gy
            tmp = x / grid_sampling + 1
            gx = int(tmp)
            tx = tmp - gx

            # Get coefficients
            compute_basis_coefficients(ty, ccy)
            compute_basis_coefficients(tx, ccx)

            # Pre-normalize value
            omsum = 0.0
            for j in range(4):
                for i in range(4):
                    omsum += ccy[j] * ccx[i] * ccy[j] * ccx[i]
            val_n = val / omsum

            # For each knot that this point influences
            # Following Lee et al. we update a numerator and a denumerator for
            # each knot.
            for j in range(4):
                jj = j + gy - 1
                for i in range(4):
                    ii = i + gx - 1
                    omega = ccy[j] * ccx[i]
                    omega2 = weight * omega * omega
                    num[jj, ii] += omega2 * (val_n * omega)
                    dnum[jj, ii] += omega2

    # Done
    return num, dnum
