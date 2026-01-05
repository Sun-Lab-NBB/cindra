"""This module provides functions for computing spline interpolation coefficients using the Cardinal, B-spline (Basis),
and Quadratic interpolation methods.
"""

from enum import IntEnum

import numba
import numpy as np
from numpy.typing import NDArray
from ataraxis_base_utilities import console


class SplineTypes(IntEnum):
    """Defines the supported spline interpolation types."""

    CARDINAL = 1
    """Cardinal spline. Setting tension to 0 produces a Catmull-Rom spline. Otherwise, a cubic Hermite spline is 
    used."""

    BASIS = 2
    """B-spline (basis spline)."""

    QUADRATIC = 3
    """Quadratic interpolation spline with 4-point support."""


def compute_spline_coefficients(
    interpolation_factor: float,
    spline_type: SplineTypes = SplineTypes.CARDINAL,
    tension: float = 0.0,
) -> NDArray[np.float32]:
    """Computes the four interpolation coefficients for the specified spline type.

    This function calculates coefficients used to interpolate between four lattice points (p0, p1, p2, p3). The
    interpolation_factor specifies the fractional position between the central points p1 and p2.

    Args:
        interpolation_factor: The fractional position between lattice points p1 and p2, in range [0, 1]. A value of 0
            corresponds to p1, a value of 1 corresponds to p2.
        spline_type: The spline interpolation method. Must be a SplineType enum value.
        tension: The tension parameter for Cardinal splines, in range [-1, 1]. This parameter is only used when
            the spline_type argument is set to SplineTypes.CARDINAL. Setting tension to 0 produces a Catmull-Rom spline.

    Returns:
        A flat numpy array containing the four coefficients (c0, c1, c2, c3) to be applied to the input lattice points
            (p0, p1, p2, p3).

    Raises:
        ValueError: If spline_type is not a valid SplineTypes enumeration member or if tension is outside the range
            [-1, 1].
    """
    spline_type = SplineTypes(spline_type)  # Converts to SplineTypes enum for type checking

    if spline_type == SplineTypes.CARDINAL and not (-1.0 <= tension <= 1.0):
        message = (
            f"Unable to compute the spline coefficients. Expected the 'tension' argument value to be in range [-1, 1], "
            f"but encountered: {tension}."
        )
        console.error(message=message, error=ValueError)

    # Computes and returns the coefficient array to the caller.
    coefficients = np.zeros((4,), np.float32)
    set_spline_coefficients(interpolation_factor, int(spline_type), tension, coefficients)
    return coefficients


@numba.njit(cache=True)
def set_spline_coefficients(
    interpolation_factor: float,
    spline_type: int,
    tension: float,
    coefficients: NDArray[np.float32],
) -> None:
    """Computes spline coefficients and stores them in the provided output array.

    Args:
        interpolation_factor: The position between the central lattice points, in range [0, 1].
        spline_type: The integer code for the spline type. Must be one of the valid SplineTypes enumeration members: 1
            for CARDINAL, 2 for BASIS, 3 for QUADRATIC.
        tension: The tension parameter to use for Cardinal splines, in range [-1, 1]. This parameter is ignored for
            non-Cardinal spline types.
        coefficients: The output array where to store the computed coefficients.
    """
    if spline_type == 1:  # SplineType.CARDINAL
        compute_cardinal_coefficients(
            interpolation_factor=interpolation_factor, coefficients=coefficients, tension=tension
        )
    elif spline_type == 2:  # SplineType.BASIS
        compute_basis_coefficients(interpolation_factor=interpolation_factor, coefficients=coefficients)
    elif spline_type == 3:  # SplineType.QUADRATIC
        compute_quadratic_coefficients(interpolation_factor=interpolation_factor, coefficients=coefficients)


@numba.njit(cache=True, inline="always")
def compute_cardinal_coefficients(
    interpolation_factor: float,
    coefficients: NDArray[np.float32],
    tension: float,
) -> None:
    """Computes Cardinal spline coefficients for the given tension parameter.

    Args:
        interpolation_factor: The position between the central lattice points, in range [0, 1].
        coefficients: The output array where to store the computed coefficients.
        tension: The tension parameter in range [-1, 1]. Setting tension to 0 produces a Catmull-Rom spline.
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


@numba.njit(cache=True, inline="always")
def compute_quadratic_coefficients(
    interpolation_factor: float,
    coefficients: NDArray[np.float32],
) -> None:
    """Computes quadratic interpolation coefficients with 4-point support.

    Notes:
        This function averages two quadratic polynomials fitted to overlapping point triplets.

    Args:
        interpolation_factor: The position between the central lattice points, in range [0, 1].
        coefficients: The output array where to store the computed coefficients.
    """
    t = interpolation_factor
    t_cubed = t * t

    # Coefficients for p0 and p3 (symmetric end points)
    coefficients[0] = 0.25 * t_cubed - 0.25 * t
    coefficients[3] = 0.25 * t_cubed - 0.25 * t

    # Coefficients for p1 and p2 (central points)
    coefficients[1] = -0.25 * t_cubed - 0.75 * t + 1.0
    coefficients[2] = -0.25 * t_cubed + 1.25 * t
