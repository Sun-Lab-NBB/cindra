"""This module provides the functions for creating cell and neuropil pixel masks associated with each extracted ROI."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import percentile_filter

from ..configuration import generate_default_ops
from ..detection.sparsedetect import extend_roi


def create_masks(
    roi_statistics: list[dict[str, Any]],
    height: int,
    width: int,
    neuropil: bool,
    ops: dict[str, Any],
) -> tuple[list[tuple[NDArray[np.uint32], NDArray[np.float32]]], list[NDArray[np.uint32]] | None]:
    """Creates binary pixel masks for cell ROIs and the surrounding neuropil regions.

    Args:
        roi_statistics: The ROI statistics for each ROI to be processed.
        height: The height of the imaged area in pixels.
        width: The width of the imaged area in pixels.
        neuropil: Determines whether to create the masks for the surrounding neuropil region for each ROI.
        ops: The signal extraction parameters.

    Returns:
        A tuple of two elements. The first element contains the flattened cell masks and the corresponding lambda weight
        masks for each ROI. The second element contains the flattened neuropil masks for each ROI or None if neuropil
        processing is disabled.
    """
    if ops is None:
        ops = generate_default_ops()

    # Create cell masks and lambda weight masks for all ROIs
    cell_masks = _create_cell_masks(
        roi_statistics=roi_statistics, height=height, width=width, allow_overlap=ops.get("allow_overlap", False)
    )

    # If neuropil processing is disabled, returns the extracted cell masks and the None placeholder for the neuropil
    # masks.
    if not neuropil:
        return cell_masks, None

    # Creates the neuropil masks for all ROIs
    neuropil_masks = _create_neuropil_masks(
        roi_statistics=roi_statistics,
        height=height,
        width=width,
        cell_lambda_percentile=ops.get("lambda_percentile", 50.0),
        inner_neuropil_border_radius=ops.get("inner_neuropil_border_radius", 2),
        minimum_neuropil_size=ops.get("minimum_neuropil_pixels", 350),
    )

    return cell_masks, neuropil_masks


def _create_cell_masks(
    roi_statistics: list[dict[str, Any]], height: int, width: int, allow_overlap: bool
) -> list[tuple[NDArray[np.uint32], NDArray[np.float32]]]:
    """Creates the cell pixel masks and the normalized lambda weight masks for the target ROIs.

    Args:
        roi_statistics: The ROI statistics for each ROI to be processed.
        height: The height of the imaged area in pixels.
        width: The width of the imaged area in pixels.
        allow_overlap: Determines whether to include overlapping ROI pixels in the created ROI cell masks.

    Returns:
        A tuple of two masks for each ROI. The first is the flattened ROI cell region mask. The second is the flattened
        lambda weight mask for the pixels that make up the cell region mask.
    """
    # Pre-creates the output list
    cell_masks = []

    # Loops over and processes each RI.
    for roi_data in roi_statistics:
        # Extracts the ROI pixels from the data dictionary. Depending on the 'allow_overlap' flag, selects all pixels or
        # excludes any overlapping pixels from the selection.
        pixel_mask = slice(None) if allow_overlap else ~roi_data["overlap"]

        # Convert 2D pixel coordinates (y, x) to 1D flattened indices, applies the pixel selection mask, and extracts
        # the lambda weights for the processed pixels. The lambda weights track the likelihood that the pixel belongs
        # to a cell object.
        # noinspection PyTypeChecker
        cell_mask_indices: NDArray[np.int64] = np.ravel_multi_index(
            multi_index=(roi_data["ypix"], roi_data["xpix"]), dims=(height, width)
        )
        cell_mask_indices = cell_mask_indices[pixel_mask]
        lambda_weights = roi_data["lam"][pixel_mask]

        # Normalizes the lambda weights sum to 1.0, creating a probability distribution of each pixel belonging to
        # a cell ROI.
        normalized_lambda_weights = lambda_weights / lambda_weights.sum() if lambda_weights.size > 0 else np.empty(0)
        cell_masks.append((cell_mask_indices.astype(np.uint32), normalized_lambda_weights.astype(np.float32)))

    return cell_masks


def _create_neuropil_masks(
    roi_statistics: list[dict[str, Any]],
    height: int,
    width: int,
    inner_neuropil_border_radius: int,
    minimum_neuropil_size: int,
    cell_lambda_percentile: float,
) -> list[NDArray[np.uint32]]:
    """Creates the neuropil masks for the target ROIs.

    Args:
        roi_statistics: The ROI statistics for each ROI to be processed.
        height: The height of the imaged area in pixels.
        width: The width of the imaged area in pixels.
        inner_neuropil_border_radius: The radius of the border separating the neuropil region from the surrounded cell
            region, in pixels. Pixels that make up this border are excluded from the neuropil mask.
        minimum_neuropil_size: The minimum number of pixels to use for each created neuropil mask.
        cell_lambda_percentile: The percentile threshold for labeling a pixel as belonging to the cell ROI region. This
            is used to determine the size of each ROI's the cell region around which to form the neuropil mask.

    Returns:
        The flattened neuropil masks for each ROI.
    """
    # Creates a binary mask of all cell pixels across all ROIs
    cell_pixels = _create_cell_pixels_mask(
        roi_statistics=roi_statistics, height=height, width=width, lambda_percentile=cell_lambda_percentile
    )

    # Pre-creates the list that stores the discovered neuropil masks.
    neuropil_masks = []
    neuropil_expansion_step = 5  # Defines the rate at which to expand the neuropil mask, in pixels per expansion step.

    # Process each ROI to create its neuropil mask
    for roi_data in roi_statistics:
        # Extracts the y-coordinates and x-coordinates of the current ROI's pixels.
        y_pixels = np.array(roi_data["ypix"], np.uint32)
        x_pixels = np.array(roi_data["xpix"], np.uint32)

        # Pre-creates the boolean array to store the neuropil mask. Uses the dimensions of the imaging area to define
        # the array
        neuropil_mask = np.zeros((height, width), dtype=bool)

        # Extends the ROI to get a ring of pixels around the ROI center. This is the inner border that separates the
        # neuropil region from the cell region.
        inner_ypix, inner_xpix = extend_roi(
            y_pixels=y_pixels, x_pixels=x_pixels, height=height, width=width, iterations=inner_neuropil_border_radius
        )

        # Determines the number of non-cell pixels within the inner neuropil border.
        exclude_count = np.sum(cell_pixels[inner_ypix, inner_xpix] == 0)

        # Iteratively expands the neuropil masks until it accumulates the requested number of pixels. Ensures that
        # the neuropil expansion runs for at most 100 iterations.
        current_ypix, current_xpix = inner_ypix.copy(), inner_xpix.copy()
        for _ in range(100):
            # Determines the number of neuropil region pixels at the start of the current iteration. To do so, discounts
            # the inner neuropil border pixels to maintain a clear separation between the neuropil and the cell ROI
            # regions.
            valid_pixels = np.sum(cell_pixels[current_ypix, current_xpix] == 0)
            neuropil_count = valid_pixels - exclude_count

            # If the accumulated number of neuropil pixels exceeds the minimum required count, aborts the runtime.
            if neuropil_count > minimum_neuropil_size:
                break

            # Expands the neuropil mask by uniformly extending the neuropil's bounding box to include the expansion_step
            # number of new pixels on each side.
            current_ypix, current_xpix = np.meshgrid(
                np.arange(
                    max(0, current_ypix.min() - neuropil_expansion_step),
                    min(height, current_ypix.max() + neuropil_expansion_step + 1),
                    1,
                    int,
                ),
                np.arange(
                    max(0, current_xpix.min() - neuropil_expansion_step),
                    min(width, current_xpix.max() + neuropil_expansion_step + 1),
                    1,
                    int,
                ),
                indexing="ij",
            )

        # Creates the final neuropil mask for this ROI by excluding all inner border pixels and including all non-cell
        # pixels in the expanded neuropil region.
        is_non_cell = cell_pixels[current_ypix, current_xpix] == 0
        neuropil_mask[current_ypix[is_non_cell], current_xpix[is_non_cell]] = True
        neuropil_mask[inner_ypix, inner_xpix] = False

        # Flattens the neuropil mask array to an array if pixel indices and appends them to the neuropil_masks list.
        neuropil_masks.append(
            np.ravel_multi_index(multi_index=np.nonzero(neuropil_mask), dims=(height, width)).astype(np.uint32)
        )

    return neuropil_masks


def _create_cell_pixels_mask(
    roi_statistics: list[dict[str, Any]], height: int, width: int, lambda_percentile: float
) -> NDArray[np.bool]:
    """Creates a binary mask image that sets all pixels of the input image corresponding to a cell ROI to 1, and all
    other pixels to 0.

    Args:
        roi_statistics: The ROI statistics for each ROI to be processed.
        height: The height of the imaged area in pixels.
        width: The width of the imaged area in pixels.
        lambda_percentile: The percentile threshold for considering a pixel as belonging to a cell ROI object, based on
            the confidence score assigned during the ROI detection procedure.

    Returns:
        The created binary mask image.
    """
    # Pre-creates the temporary arrays used to process the data.
    cell_likelihood_map = np.zeros((height, width))
    cell_radii = np.zeros(len(roi_statistics))

    # Loops over each ROI and extracts the data required for processing.
    for roi_index, data in enumerate(roi_statistics):
        cell_radii[roi_index] = data["radius"]
        y_pixels = data["ypix"]
        x_pixels = data["xpix"]
        # Lambda weight. Measures the likelihood of the pixel belonging to a cell ROI.
        pixel_cell_likelihood = data["lam"]

        # Ensures that the cell likelihood is measured using positive numbers only to make the percentile filter below.
        cell_likelihood_map[y_pixels, x_pixels] = np.maximum(
            cell_likelihood_map[y_pixels, x_pixels], pixel_cell_likelihood
        )

    # Computes the median cell radius to determine the local neighborhood size for percentile filtering.
    median_radius = np.median(cell_radii)

    # If additional percentile (ROI likelihood) filtering is enabled, selects ROI pixels based on the specified
    # percentile threshold.
    if lambda_percentile > 0.0:
        # Pixels are selected as 'ROI' if their likelihood for being an ROI is greater than or equal to the specified
        # percentile filter.
        neighborhood_size = int(median_radius * 5)
        cell_threshold_filter = percentile_filter(
            cell_likelihood_map, percentile=lambda_percentile, size=neighborhood_size
        )
        pixel_mask = (cell_likelihood_map > 0.0) & (cell_likelihood_map >= cell_threshold_filter)

    else:
        # Otherwise, selects all pixels with a weight greater than zero.
        pixel_mask = cell_likelihood_map > 0.0

    return pixel_mask
