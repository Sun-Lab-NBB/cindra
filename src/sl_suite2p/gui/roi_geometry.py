"""Provides boundary and circle geometry utilities for ROI rendering."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Padding added around the mask bounding box for boundary computation.
_BOUNDARY_PADDING: int = 3

# Scale factor applied to the cell radius for circle rendering.
_CIRCLE_RADIUS_SCALE: float = 1.25

# Number of points used to approximate the circle perimeter.
_CIRCLE_POINT_COUNT: int = 100


def boundary(
    y_pixels: NDArray[np.int32],
    x_pixels: NDArray[np.int32],
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Computes the exterior boundary pixels of a binary mask.

    Takes the y and x coordinates of mask pixels and returns the coordinates of
    pixels that lie on the outer border of the mask region.

    Args:
        y_pixels: Row coordinates of the mask pixels.
        x_pixels: Column coordinates of the mask pixels.

    Returns:
        Tuple of (y_boundary, x_boundary) arrays containing the boundary pixel coordinates.
    """
    y_pixels = np.expand_dims(y_pixels.flatten(), axis=1)
    x_pixels = np.expand_dims(x_pixels.flatten(), axis=1)
    pixel_count = y_pixels.shape[0]

    if pixel_count > 0:
        mask = np.zeros(
            (int(np.ptp(y_pixels)) + 2 * _BOUNDARY_PADDING, int(np.ptp(x_pixels)) + 2 * _BOUNDARY_PADDING),
            dtype=bool,
        )
        mask[
            y_pixels - y_pixels.min() + _BOUNDARY_PADDING,
            x_pixels - x_pixels.min() + _BOUNDARY_PADDING,
        ] = True
        mask = binary_dilation(mask)
        mask = binary_fill_holes(mask)

        # Uses an 8-connected structuring element to find the exterior ring.
        kernel = np.zeros((3, 3), dtype=np.int32)
        kernel[1] = 1
        kernel[:, 1] = 1
        exterior = binary_dilation(mask == 0, structure=kernel) & mask

        y_boundary, x_boundary = np.nonzero(exterior)
        y_boundary = y_boundary + y_pixels.min() - _BOUNDARY_PADDING
        x_boundary = x_boundary + x_pixels.min() - _BOUNDARY_PADDING
    else:
        y_boundary = np.zeros((0,), dtype=np.int32)
        x_boundary = np.zeros((0,), dtype=np.int32)

    return y_boundary, x_boundary


def circle(
    centroid: NDArray[np.float64],
    radius: float,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Computes the pixel coordinates of a circle around a cell centroid.

    Generates a circle with radius scaled by 1.25x the cell radius, returning
    integer pixel coordinates suitable for overlay rendering.

    Args:
        centroid: Two-element array containing the (y, x) center of the cell.
        radius: Radius of the cell in pixels.

    Returns:
        Tuple of (x_circle, y_circle) arrays containing the circle pixel coordinates.
    """
    theta = np.linspace(0.0, 2 * np.pi, _CIRCLE_POINT_COUNT)
    x_circle = (radius * _CIRCLE_RADIUS_SCALE * np.cos(theta) + centroid[0]).astype(np.int32)
    y_circle = (radius * _CIRCLE_RADIUS_SCALE * np.sin(theta) + centroid[1]).astype(np.int32)
    return x_circle, y_circle
