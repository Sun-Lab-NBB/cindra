"""Provides boundary and circle geometry computation utilities for ROI rendering."""

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


def compute_boundary_mask(
    y_pixels: NDArray[np.int32],
    x_pixels: NDArray[np.int32],
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Computes the exterior boundary mask of the ROI specified by the input pixel coordinates.

    Args:
        y_pixels: The row coordinates of the mask pixels.
        x_pixels: The column coordinates of the mask pixels.

    Returns:
        The (y_boundary, x_boundary) arrays containing the boundary mask pixel coordinates.
    """
    # Reshapes the coordinate arrays into column vectors for 2D array indexing.
    y_pixels = np.expand_dims(y_pixels.flatten(), axis=1)
    x_pixels = np.expand_dims(x_pixels.flatten(), axis=1)
    pixel_count = y_pixels.shape[0]

    if not pixel_count:
        return np.zeros((0,), dtype=np.int32), np.zeros((0,), dtype=np.int32)

    # Builds a tight bounding box around the ROI with padding to ensure boundary pixels at the edges of the mask are
    # not clipped during morphological operations.
    y_min = y_pixels.min()
    x_min = x_pixels.min()
    mask = np.zeros(
        (int(y_pixels.max() - y_min) + 2 * _BOUNDARY_PADDING, int(x_pixels.max() - x_min) + 2 * _BOUNDARY_PADDING),
        dtype=np.bool_,
    )

    # Stamps the ROI pixels into the local coordinate system offset by the bounding box origin.
    mask[
        y_pixels - y_min + _BOUNDARY_PADDING,
        x_pixels - x_min + _BOUNDARY_PADDING,
    ] = True

    # Dilates the mask to close single-pixel gaps, then fills interior holes to produce a solid region.
    mask = binary_dilation(mask)
    mask = binary_fill_holes(mask)

    # Uses a 4-connected structuring element (cross pattern) to find the exterior ring. Dilating the background into
    # the foreground and intersecting with the original mask isolates the outermost pixel layer.
    kernel = np.zeros((3, 3), dtype=np.int32)
    kernel[1] = 1
    kernel[:, 1] = 1
    exterior = binary_dilation(mask == 0, structure=kernel) & mask

    # Converts local bounding-box coordinates back to the original frame coordinate system.
    y_boundary, x_boundary = np.nonzero(exterior)
    y_boundary = y_boundary + y_min - _BOUNDARY_PADDING
    x_boundary = x_boundary + x_min - _BOUNDARY_PADDING

    return y_boundary, x_boundary


def compute_circle_mask(
    centroid_y: int,
    centroid_x: int,
    radius: float,
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Computes the pixel coordinates of a circle around the ROI centroid specified by the input coordinates.

    Args:
        centroid_y: The row coordinate of the ROI's center.
        centroid_x: The column coordinate of the ROI's center.
        radius: The radius of the ROI in pixels.

    Returns:
        The (y_circle, x_circle) arrays containing the circle pixel coordinates.
    """
    # Scales the radius up to provide visual clearance around the ROI boundary.
    scaled_radius = radius * _CIRCLE_RADIUS_SCALE
    theta = np.linspace(0.0, 2 * np.pi, num=_CIRCLE_POINT_COUNT)
    y_circle = (scaled_radius * np.sin(theta) + centroid_y).astype(np.int32)
    x_circle = (scaled_radius * np.cos(theta) + centroid_x).astype(np.int32)
    return y_circle, x_circle
