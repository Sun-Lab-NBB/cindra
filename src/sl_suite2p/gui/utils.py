"""This module provides the utility functions used by the sl-suite2p GUI to visualize and manipulate cell
(ROI) masks.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import binary_dilation, binary_fill_holes


def compute_roi_boundary(
    y_pixels: NDArray[np.signedinteger[Any]], x_pixels: NDArray[np.signedinteger[Any]]
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Computes the exterior boundary pixels for the target cell (ROI) mask.

    This service function is used to compute the visualization boundary (border) for the target cell (ROI) mask.

    Args:
        y_pixels: The array of mask pixel y-coordinates.
        x_pixels: The array of mask pixel x-coordinates.

    Returns:
        A tuple of two arrays containing the cell (ROI) boundary pixel coordinates in the order of (y, x).
    """
    # Converts x and y coordinates into column vectors and determines the total number of pixels to process. Assumes
    # that the arrays have the same length.
    y_pixels = np.expand_dims(y_pixels.flatten(), axis=1)
    x_pixels = np.expand_dims(x_pixels.flatten(), axis=1)
    pixel_count = y_pixels.shape[0]

    # If the input ROI does not contain any pixels to process (is an empty ROI), returns empty arrays to indicate that
    # there is no boundary to render.
    if pixel_count < 1:
        boundary_y_pixels = np.zeros((0,), dtype=np.int32)
        boundary_x_pixels = np.zeros((0,), dtype=np.int32)
        return boundary_y_pixels, boundary_x_pixels

    # Creates the binary ROI mask using the input pixel coordinates. Pads the mask with 6 pixels to create a visual
    # boundary region, centers the mask around the original pixel values, and sets all original ROI pixels to 1 (white).
    mask = np.zeros((np.ptp(y_pixels) + 6, np.ptp(x_pixels) + 6), dtype=bool)
    mask[y_pixels - y_pixels.min() + 3, x_pixels - x_pixels.min() + 3] = True

    # Cleans up the mask by dilating and filling the holes (classic binary mask augmentation technique). This creates a
    # uniform 'white' center with a 'black' boundary region.
    mask = binary_dilation(mask)
    mask = binary_fill_holes(mask)

    # Defines the kernel for 8-connected neighborhood and uses it to find the boundary pixels that directly contact the
    # cleaned up cell ROI mask (center region). This generates the 1-pixel-thin mask for the cell (ROI) boundary region.
    kernel = np.zeros((3, 3), dtype=int)
    kernel[1] = 1
    kernel[:, 1] = 1
    boundary_mask = binary_dilation(mask == 0, kernel) & mask

    # Converts the boundary mask back to the original ROI coordinates and returns the x and y coordinates of the
    # boundary pixels.
    boundary_y_pixels, boundary_x_pixels = np.nonzero(boundary_mask)
    boundary_y_pixels, boundary_x_pixels = (
        boundary_y_pixels + y_pixels.min() - 3,
        boundary_x_pixels + x_pixels.min() - 3,
    )
    return boundary_y_pixels.astype(dtype=np.int32), boundary_x_pixels.astype(dtype=np.int32)


def compute_circular_boundary(
    centroid_coordinates: tuple[np.signedinteger[Any], np.signedinteger[Any]], roi_radius: np.signedinteger[Any]
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Generates the pixel coordinates for a circular overlay drawing that covers the target ROI.

    This service function is used to compute circular overlays for cell (ROI) visualization purposes.

    Notes:
        This function uses the following equation for computing the circular overlay radius: 1.25 x roi_radius.

    Args:
        centroid_coordinates: The coordinates of the processed ROI's centroid, given in the order of (x, y).
        roi_radius: The radius of the processed ROI.

    Returns:
        A tuple of two arrays containing the circular overlay pixel coordinates in the order of (y, x).
    """
    theta = np.linspace(0.0, 2 * np.pi, 100)
    x = roi_radius * np.float64(1.25) * np.cos(theta) + centroid_coordinates[0]
    y = roi_radius * np.float64(1.25) * np.sin(theta) + centroid_coordinates[1]
    return y.astype(np.int32), x.astype(np.int32)
