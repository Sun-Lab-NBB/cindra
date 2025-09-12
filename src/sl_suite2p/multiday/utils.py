"""This module stores various helper functions used by all other modules of the 'multiday' sl-suite2p package."""

from typing import Any
from pathlib import Path

from pirt import Aarray, DeformationFieldBackward
from numba import njit, prange
import numpy as np
from numpy.typing import NDArray
import scipy.ndimage
from skimage.measure import regionprops, find_contours
from ataraxis_base_utilities import console


# noinspection PyTypeHints
def create_mask_image(
    cell_masks: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    image_size: list[int] | tuple[int, int],
    field: str | None = None,
    mark_overlap: bool = False,
    contours: bool = False,
    contour_upsampling: int = 1,
) -> NDArray[np.uint32 | np.float32]:
    """Creates labeled images from cell masks with optional overlap marking and contour generation.

    Args:
        cell_masks: A list or tuple of cell mask dictionaries containing 'xpix' and 'ypix' keys.
        image_size: The dimensions of the output image in the order of (height, width).
        field: Optional. The cell_masks dictionary field (key) to use for setting the labeled image pixel values. If
            this argument is set to None, the function uses cell mask indices as pixel values.
        mark_overlap: Determines whether to highlight the overlapping regions by setting their pixels to 100.
        contours: Determines whether to generate mask contours instead of filled regions.
        contour_upsampling: The scaling factor for contour resolution. Setting this to a value greater than 1 multiplies
            the drawn ROI contours with this value, generating visually thicker contours. This argument is only used
            when a non-default 'field' argument value is provided.

    Returns:
        The generated label image, stored as a two-dimensional NumPy array.

    Raises:
        ValueError: If both 'mark_overlap' and 'contours' arguments are enabled at the same time.
    """
    if mark_overlap and contours:
        message = (
            "Unable to generate the labeled image. Both 'mark_overlap' and 'contours' are enabled (True) at the same "
            "time, which is not allowed."
        )
        console.error(message=message, error=ValueError)

    # Initializes the output image array with the appropriate dtype
    image: NDArray[Any]
    if (not field) or (field == "id"):
        image = np.zeros(
            (image_size[0] * contour_upsampling, image_size[1] * contour_upsampling),
            dtype=np.uint32,
        )
    else:
        image = np.zeros(image_size, dtype=np.float32)

    for index, mask in enumerate(cell_masks):
        value = mask[field] if field else index  # Determines the pixel value to use for labeling the mask
        ypix, xpix = mask["ypix"], mask["xpix"]  # Determines mask locations

        # If fully rendering cells, sets each pixel defined by the cell mask to the pixel value computed above
        if not contours:
            image[ypix, xpix] = value

            # If overlap rendering is enabled, overwrites all overlapping pixels with the value 100.
            if mark_overlap:
                image[ypix[mask["overlap"]], xpix[mask["overlap"]]] = 100
        else:
            # Otherwise, when only rendering contours, generates a contoured image:

            # Generates a local (within-cell) coordinate system
            origin = [min(ypix - 1), min(xpix - 1)]
            y_local = ypix - origin[0]
            x_local = xpix - origin[1]

            # Creates a binary cell mask image (just large enough to contain the cell)
            temporary_image = np.zeros((max(y_local) + 2, max(x_local) + 2), dtype=bool)
            temporary_image[y_local, x_local] = True

            # If necessary, zooms to upsample the contours of the binary mask
            if contour_upsampling > 1:
                temporary_image = scipy.ndimage.zoom(temporary_image, contour_upsampling, order=0)

            # Finds contour pixel indices (contours of the cell mask)
            contours_ind = np.vstack(find_contours(temporary_image)).astype(int)

            # Maps contour pixel coordinates back to the original image space.
            y_coordinates = contours_ind[:, 0] + (origin[0] * contour_upsampling)
            x_coordinates = contours_ind[:, 1] + (origin[1] * contour_upsampling)
            image[y_coordinates, x_coordinates] = value

    # Returns the generated label image
    return image


def create_cropped_deform_field(
    deform: DeformationFieldBackward, origin: NDArray[Any], crop_size: tuple[int, int]
) -> tuple[DeformationFieldBackward, NDArray[Any]]:
    """Creates a cropped deformation field from the input deformation field.

    Args:
        deform: The original (larger) backward deformation field.
        origin: The coordinates of crop origin in the order of (y, x).
        crop_size: The size of the crop region in the order of (height, width).

    Returns:
        A tuple of two elements. The first element is the cropped deformation field, and the second element is the
        adjusted origin coordinates.
    """
    origin = origin.copy()  # Creates a copy to avoid modifying the input

    # Ensures origin is non-negative
    np.maximum(origin, 0, out=origin)

    # Gets the overall deformation field dimensions and casts crop_size to NumPy array
    image_size = np.asarray(deform.field_shape)  # Convert tuple to NumPy array
    crop_size_array = np.asarray(crop_size)

    # Adjusts origin if the crop exceeds image bounds
    origin = np.minimum(origin, np.maximum(image_size - crop_size, 0))

    # Creates the x and y coordinates for computing the cropped deformation field.
    y_slice = slice(origin[0], origin[0] + crop_size_array[0])
    x_slice = slice(origin[1], origin[1] + crop_size_array[1])

    return DeformationFieldBackward([deform[0][y_slice, x_slice], deform[1][y_slice, x_slice]]), origin


def deform_masks(
    cell_masks: tuple[dict[str, Any], ...] | list[dict[str, Any]], deform: DeformationFieldBackward, crop_bin: int = 500
) -> tuple[dict[str, Any], ...]:
    """Applies deformation field offsets to all input cell masks.

    Args:
        cell_masks: The tuple that stores cell mask dictionaries for each cell ROI to be deformed.
        deform: The DeformationFieldBackward instance generated during multi-day registration to apply to the input
            cell masks.
        crop_bin: The number of masks to transform at the same time. This optional field allows reducing the RAM
            overhead of the function by only processing a subset of masks at a time.

    Returns:
        A tuple of cell masks modified with the deformation field offsets.
    """
    deformed = []
    for mask in cell_masks:
        # Generates cropped deformation field windows to split the procedure into smaller steps
        crop_origin = np.array(mask["med"], dtype=np.int32) - crop_bin // 2
        crop_def, adj_origin = create_cropped_deform_field(deform, crop_origin, (crop_bin, crop_bin))

        # Processes lambda weights
        y_local = mask["ypix"] - adj_origin[0]
        x_local = mask["xpix"] - adj_origin[1]
        lam_img = np.zeros((crop_bin, crop_bin), dtype=np.float32)
        lam_img[y_local, x_local] = mask["lam"]

        # Applies the deformation using the cropped deformation field
        warped_lam = np.array(crop_def.apply_deformation(Aarray(lam_img, origin=tuple(adj_origin)), interpolation=0))

        # Extracts deformed coordinates
        y_new, x_new = np.nonzero(warped_lam)
        lam_values = warped_lam[y_new, x_new]
        y_global = y_new + adj_origin[0]
        x_global = x_new + adj_origin[1]

        # Uses the extracted coordinates to create the cell mask dictionary for the cell in the deformed visual
        # field.
        deformed.append(
            {
                "xpix": x_global,
                "ypix": y_global,
                "ipix": np.ravel_multi_index((y_global, x_global), deform[0].shape),
                "med": [np.median(y_global), np.median(x_global)],
                "lam": lam_values,
                "radius": regionprops(warped_lam.astype(np.uint8))[0].minor_axis_length,
            }
        )

    return tuple(add_overlap_info(deformed))


@njit(parallel=True, cache=True)
def _find_overlapping_pixels(all_pixels: NDArray[np.int32]) -> NDArray[np.int32]:
    """This service function is used by the add_overlap_info function to find all pixels that appear in more than one
    cell mask.

    Args:
        all_pixels: A numpy array that linearly concatenates the pixel indices from all cell masks.

    Returns:
        A NumPy array that stores the indices of pixels that appear in more than one cell mask.
    """
    # Sorts pixel indices
    sorted_pixels = np.sort(all_pixels)

    # Creates a mask of duplicate pixels
    is_duplicate = np.zeros(len(sorted_pixels), dtype=np.bool_)
    for i in prange(1, len(sorted_pixels)):
        is_duplicate[i] = sorted_pixels[i] == sorted_pixels[i - 1]

    # Get unique duplicate values
    return np.unique(sorted_pixels[is_duplicate])


@njit(parallel=True, cache=True)
def _create_overlap_arrays(
    mask_pixel_indices: list[NDArray[np.int32]], overlapping_pixels: NDArray[np.int32]
) -> list[NDArray[np.bool]]:
    """This service function is used by the add_overlap_info function to create boolean arrays that mark overlapping
        pixels for each mask.

    Args:
        mask_pixel_indices: A list of NumPy arrays, where each array stores the indices of cell mask pixels.
        overlapping_pixels: A NumPy array storing the indices of overlapping pixels.

    Returns:
        A list of boolean arrays, where each array marks overlapping pixels for a specific cell mask.
    """
    overlap_arrays: list[NDArray[np.bool]] = []

    for pixel_indices in mask_pixel_indices:
        overlap = np.zeros(len(pixel_indices), dtype=np.bool_)
        for i in prange(len(pixel_indices)):
            for j in range(len(overlapping_pixels)):
                if pixel_indices[i] == overlapping_pixels[j]:
                    overlap[i] = True
                    break
        overlap_arrays.append(overlap)

    return overlap_arrays


def add_overlap_info(masks: list[dict[str, Any]]) -> list[dict[str, NDArray[np.bool]]]:
    """Identifies overlapping pixels across the input cell masks and augments each cell mask dictionary with
    overlapping pixel information.

    Args:
        masks: The list of cell mask dictionaries with 'ipix' keys.

    Returns:
        The list of modified mask dictionaries, which now contain the added 'overlap' boolean arrays.
    """
    # Extracts pixel indices from all masks
    mask_pixel_indices = [mask["ipix"] for mask in masks]

    # Concatenates all pixel indices
    all_pixel_indices = np.concatenate(mask_pixel_indices)

    # Finds overlapping pixels using numba
    overlapping_pixels = _find_overlapping_pixels(all_pixel_indices)

    # Creates arrays to store overlapping pixels
    overlap_arrays = _create_overlap_arrays(mask_pixel_indices, overlapping_pixels)

    # Assigns overlapping pixel arrays to mask dictionaries for cells that make up the overlap
    for i, mask in enumerate(masks):
        mask["overlap"] = overlap_arrays[i]

    return masks


def extract_unique_components(paths: list[Path] | tuple[Path, ...]) -> tuple[str, ...]:
    """For each path in the input iterable, extracts the first component from the end that uniquely identifies the
    path globally, regardless of position.

    This service function is used to automatically build the list of input session IDs from the input list of session
    paths when resolving the multi-day ops.npy file.

    Notes:
        This function presupposes that the input paths contain a unique identifier that can be extracted. With respect
        to processing paths to recording sessions, it assumes that each session is stored in a uniquely named folder
        somewhere along the path hierarchy. If this assumption is violated, the function raises a RuntimeError.

    Args:
        paths: A list or tuple of Path objects.

    Returns:
        A tuple of unique components, one for each path, stored in the same order as the input paths.

    Raises:
        RuntimeError: If one or more paths do not contain unique components.
    """
    paths_list = list(paths)
    result = []

    for path in paths_list:
        # Gets components from right to left
        components = list(path.parts)[::-1]
        found_unique = False

        for component in components:
            # Checks if this component appears in any other path
            is_unique = True

            for other_path in paths_list:
                if path == other_path:
                    continue

                # If the component appears anywhere in the other path, it's not unique
                if component in other_path.parts:
                    is_unique = False
                    break

            if is_unique:
                result.append(component)
                found_unique = True
                break

        if not found_unique:
            message = f"No unique component found for path: {path}, which is not allowed."
            console.error(message=message, error=RuntimeError)

    return tuple(result)
