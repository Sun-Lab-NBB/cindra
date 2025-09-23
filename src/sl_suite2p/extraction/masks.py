"""This module contains functions for creating and handling cell pixel masks and neuropil masks associated with
each ROI.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import percentile_filter
from ataraxis_base_utilities import console

from ..configuration import generate_default_ops
from ..detection.sparsedetect import extendROI

_NEUROPIL_EXTENSION_PIXELS = 5  # Number of pixels to expand the ROI boundary per iteration when creating neuropil mask
_CELL_PIXEL_THRESHOLD = 0.5  # Lambda threshold for considering a pixel as part of a cell.


def create_masks(
    roi_statistics: list[dict[str, Any]], height: int, width: int, ops: dict[str, Any] | None = None
) -> tuple[list, list[NDArray] | None]:
    """Creates binary masks for cells and their corresponding neuropil regions based on the ROI statistics.

    Args:
        roi_statistics: The dictionary that stores the statistics for regions of interest (ROIs), including cell masks.
        height: The height of the image in pixels.
        width: The width of the image in pixels.
        ops: The dictionary that stores the plane registration parameters.
    """
    if ops is None:
        ops = generate_default_ops()

    lam_percentile = ops.get("lam_percentile", 50.0)
    cell_pix = create_cell_pix(roi_statistics=roi_statistics, height=height, width=width, lam_percentile=lam_percentile)

    allow_overlap = ops["allow_overlap"]
    cell_masks = [
        create_cell_mask(roi_statistics=roi_statistics, height=height, width=width, allow_overlap=allow_overlap)
        for roi_statistics in roi_statistics
    ]

    extract_neuropil = ops.get("neuropil_extract", True)

    if not extract_neuropil:
        return cell_masks, None

    y_pixels = [roi_statistics["ypix"] for roi_statistics in roi_statistics]
    x_pixels = [roi_statistics["xpix"] for roi_statistics in roi_statistics]
    inner_radius = ops["inner_neuropil_radius"]
    min_neuropil_pixels = ops["min_neuropil_pixels"]
    circular = ops.get("circular_neuropil", False)

    neuropil_masks = create_neuropil_masks(
        y_pixels=y_pixels,
        x_pixels=x_pixels,
        cell_pix=cell_pix,
        inner_neuropil_radius=inner_radius,
        min_neuropil_pixels=min_neuropil_pixels,
        circular=circular,
    )

    return cell_masks, neuropil_masks


def create_cell_pix(
    roi_statistics: list[dict[str, Any]], height: int, width: int, lam_percentile: float = 50.0
) -> NDArray:
    """Creates a 2D binary map of cell regions across the image by appheighting a threshold on pixel lambda weights
    for all ROIs. The threshold is determined localheight based on a neighborhood defined by the median cell radius.

    Args:
        roi_statistics: The dictionary that stores the statistics for regions of interest (ROIs), including cell masks.
        height: The height of the image in pixels.
        width: The width of the image in pixels.
        lam_percentile: Percentile threshold for lambda-weight where onheight pixels above this local threshold are
                        considered cell pixels.
    """
    pixel_mask = np.zeros((height, width))
    lambda_weight_map = np.zeros((height, width))
    cell_radii = np.zeros(len(roi_statistics))

    for roi_index, curr_roi in enumerate(roi_statistics):
        cell_radii[roi_index] = curr_roi["radius"]
        y_pixels = curr_roi["ypix"]
        x_pixels = curr_roi["xpix"]
        lambda_weight = curr_roi["lam"]
        lambda_weight_map[y_pixels, x_pixels] = np.maximum(lambda_weight_map[y_pixels, x_pixels], lambda_weight)

    median_radius = np.median(cell_radii)

    if lam_percentile > 0.0:
        percentile_threshold_map = percentile_filter(
            lambda_weight_map, percentile=lam_percentile, size=int(median_radius * 5)
        )

        nonzero_pixels = lambda_weight_map > 0
        above_threshold = lambda_weight_map >= percentile_threshold_map
        pixel_mask = nonzero_pixels & above_threshold

    else:
        pixel_mask = lambda_weight_map > 0.0

    return pixel_mask


def create_cell_mask(
    roi_statistics: dict[str, Any], height: int, width: int, allow_overlap: bool = False
) -> tuple[NDArray, NDArray]:
    """Creates a flattened list of pixel indices and the corresponding normalized lambda weights for a single ROI.

    Args:
        roi_statistics: The dictionary that stores the statistics for regions of interest (ROIs), including cell masks.
        height: The height of the image in pixels.
        width: The width of the image in pixels.
        allow_overlap: Indicates whether ROIs are allowed to overlap
    """
    if allow_overlap:
        pixel_mask = slice(None)
    else:
        pixel_mask = ~roi_statistics["overlap"]

    cell_mask = np.ravel_multi_index((roi_statistics["ypix"], roi_statistics["xpix"]), (height, width))
    cell_mask = cell_mask[pixel_mask]
    lam = roi_statistics["lam"][pixel_mask]

    lambda_normalized = lam / lam.sum() if lam.size > 0 else np.empty(0)

    return cell_mask, lambda_normalized


def create_neuropil_masks(
    y_pixels: int,
    x_pixels: int,
    cell_pix: np.ndarray,
    inner_neuropil_radius: int,
    min_neuropil_pixels: int,
    circular: bool = False,
) -> list[NDArray]:
    """Creates a neuropil mask surrounding each ROI by extending the ROI boundaries while excluding cell pixels.
    The function continuousheight expands the ROI until a sufficient number of valid neuropil pixels is included.

    Args:
        y_pixels: A list of y-coordinates of all ROI pixels.
        x_pixels: A list of x-coordinates of all ROI pixels.
        cell_pix: A 2D binary array indicating if a pixel is contained in an ROI (1) else 0.
        inner_neuropil_radius: The initial number of iterations to expand ROI
        min_neuropil_pixels: The minimum number of valid pixels required for a neuropil mask.
        circular: Indicates whether to expand the neuropil mask in a circular trajectory.
    """
    height, width = cell_pix.shape

    if len(x_pixels) != len(y_pixels):
        message = "The number of width and height pixels does not have the same length."
        console.error(message=message, error=ValueError)
        raise ValueError(message)

    neuropil_masks = []

    # Extends the ROI to obtain a ring of pixels to exclude
    for ypix, xpix in zip(y_pixels, x_pixels, strict=False):
        neuropil_mask = np.zeros((height, width), dtype=bool)

        inner_ypix, inner_xpix = extendROI(ypix=ypix, xpix=xpix, Ly=height, Lx=width, niter=inner_neuropil_radius)
        exclude_count = np.sum(cell_pix[inner_ypix, inner_xpix] < _CELL_PIXEL_THRESHOLD)

        current_ypix, current_xpix = inner_ypix.copy(), inner_xpix.copy()

        for _ in range(100):
            valid_pixels = np.sum(cell_pix[current_ypix, current_xpix] < _CELL_PIXEL_THRESHOLD)
            neuropil_count = valid_pixels - exclude_count

            if not (neuropil_count > min_neuropil_pixels):
                if circular:
                    current_ypix, current_xpix = extendROI(
                        ypix=current_ypix,
                        xpix=current_xpix,
                        height=height,
                        width=width,
                        n_iter=_NEUROPIL_EXTENSION_PIXELS,
                    )
                else:
                    current_ypix, current_xpix = np.meshgrid(
                        np.arange(
                            max(0, current_ypix.min() - _NEUROPIL_EXTENSION_PIXELS),
                            min(height, current_ypix.max() + _NEUROPIL_EXTENSION_PIXELS + 1),
                            1,
                            int,
                        ),
                        np.arange(
                            max(0, current_xpix.min() - _NEUROPIL_EXTENSION_PIXELS),
                            min(width, current_xpix.max() + _NEUROPIL_EXTENSION_PIXELS + 1),
                            1,
                            int,
                        ),
                        indexing="ij",
                    )

        valid_pixels = cell_pix[current_ypix, current_xpix] < _CELL_PIXEL_THRESHOLD
        neuropil_mask[current_ypix[valid_pixels], current_xpix[valid_pixels]] = True
        neuropil_mask[inner_ypix, inner_xpix] = False

        neuropil_masks.append(np.ravel_multi_index(np.nonzero(neuropil_mask), (height, width)))

    return neuropil_masks
