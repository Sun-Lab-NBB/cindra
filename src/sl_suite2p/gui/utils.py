"""Utility functions for mask boundary extraction and circle generation.

This module has helper functions used by the Suite2p GUI to visualize and 
manipulate mask regions. 

Copyright © 2023 Howard Hughes Medical Institute, 
Authored by Carsen Stringer and Marius Pachitariu.
"""


from __future__ import annotations
import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes


def boundary(ypix: np.ndarray, xpix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute the exterior boundary pixels of a mask.

    Args:
        ypix: Array of y-coordinates for the mask pixels.
        xpix: Array of x-coordinates for the mask pixels.

    Returns:
        A tuple (yext, xext) of arrays containing the coordinates of
        boundary pixels.
    """
    ypix = np.expand_dims(ypix.flatten(), axis=1)
    xpix = np.expand_dims(xpix.flatten(), axis=1)
    npix = ypix.shape[0]

    if npix > 0:
        msk = np.zeros((np.ptp(ypix) + 6, np.ptp(xpix) + 6), dtype=bool)
        msk[ypix - ypix.min() + 3, xpix - xpix.min() + 3] = True
        msk = binary_dilation(msk)
        msk = binary_fill_holes(msk)

        # Define kernel for 8-connected neighborhood
        kernel = np.zeros((3, 3), dtype=int)
        kernel[1] = 1
        kernel[:, 1] = 1

        out = binary_dilation(msk == 0, kernel) & msk

        yext, xext = np.nonzero(out)
        yext, xext = yext + ypix.min() - 3, xext + xpix.min() - 3
    else:
        yext = np.zeros((0,))
        xext = np.zeros((0,))

    return yext, xext


def circle(med: tuple[float, float], r: float) -> tuple[np.ndarray, np.ndarray]:
    """Generate the pixel coordinates of a circle around a cell.

    The circle has radius 1.25 × r.

    Args:
        med: The (x, y) coordinates of the circle center (cell centroid).
        r: The base cell radius.

    Returns:
        A tuple (x, y) of arrays containing the circle pixel coordinates.
    """
    theta = np.linspace(0.0, 2 * np.pi, 100)
    x = r * 1.25 * np.cos(theta) + med[0]
    y = r * 1.25 * np.sin(theta) + med[1]

    return x.astype(np.int32), y.astype(np.int32)


__all__ = ["boundary", "circle"]
