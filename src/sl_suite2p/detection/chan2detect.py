"""This module provides assets for identify red cells in channel 2 brightness.

main function is detect
takes from ops: "meanImg", "meanImg_chan2", "Ly", "Lx"
takes from stat: "ypix", "xpix", "lam"
"""


import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter

from . import utils
from ..extraction import masks
from ataraxis_base_utilities import LogLevel, console


def _create_gaussian_mask(image_height: int, image_width:int, pixel_y_indices: NDArray[np.int64], pixel_x_indices: NDArray[np.int64], smoothing_radius: float) -> NDArray[np.float32]:
    """
    Creates a mask that smoothens a specified sub area of the imput image by applying gaussian filter.

    Args:
        image_height: The height of the full image in pixels
        image_width: The width of the full image in pixels
        pixel_y_indices: Array of pixel row indices (y-coordinates) defining the sub area.
        pixel_x_indices: Array of pixel column indices (x-coordinates) defining the sub area.
        smoothing_radius: The standard deviation of the Gaussian filter which controls degree of edge softness.
            - Larger values: produce wider, softer transitions, so the block edges fade out over a larger area.
            - Smaller values: produce narrower, sharper transitions, so the block edges stay more confined.

    Returns:
        A mask with same dimensions (image_height x image_width) where the specified block is 
        1 in the center and smoothly tapers to 0 near the edges.
    """
    # Initialize a blank mask with same size as full image
    mask = np.zeros((image_height, image_width), np.float32)

    # Sets the sub area of the mask corresponding to given pixel coordinates to 1 to indicate selection
    mask[np.ix_(pixel_y_indices, pixel_x_indices)] = 1

    # Apply a gaussian filter to smooth the the edges of the sub area
    mask = gaussian_filter(mask, smoothing_radius)

    return mask


def _correct_green_bleedthrough(image_height: int, image_width:int, number_of_blocks: int, green_channel_image: NDArray[np.float32], red_channel_image: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Corrects green-to-red bleedthrough by estimating and subtracting local linear predictions of the green signal within image blocks.

    Args:
        image_height: The height of the full image in pixels
        image_width: The width of the full image in pixels
        number_of_blocks: The number of blocks along each dimension to split the image in (image is split into number_of_blocks x number_of_blocks).
        green_channel_image: The mean image from the green channel of the input image
        red_channel_image: The mean image from the red channel of the input image

    Returns: 
        corrected_red: A 2D NumPy array of shape (image_height, image_width) containing the red_channel_image with predicted green bleedthrough removed.
    """
    # Compute the gaussian smoothing radius based on the image size and the number of blocks you want to split it into.
    # (larger images and fewer blocks → larger radius).
    gaussian_filter_smoothing_radius = np.round((image_height + image_width) / (number_of_blocks * 2) * 0.25)

    # Initialize block-specific masks and weights
    block_masks = np.zeros((image_height, image_width, number_of_blocks, number_of_blocks), np.float32)
    block_weights = np.zeros((number_of_blocks, number_of_blocks), np.float32)

    # Define the indices of the images that corresponds to an edge/start of a block
    block_edge_y_indices = np.linspace(0, image_height, number_of_blocks + 1, dtype=int)
    block_edge_x_indices = np.linspace(0, image_width, number_of_blocks + 1, dtype=int)

    # For each block, estimate how much the corresponding green channel image block explains that of the red channel image
    for block_y_index in range(number_of_blocks):
        for block_x_index in range(number_of_blocks):
            # Get the pixel row and column indices for the current block
            block_y_pixel_indices = np.arange(block_edge_y_indices[block_y_index], block_edge_y_indices[block_y_index + 1])
            block_x_pixel_indices = np.arange(block_edge_x_indices[block_x_index], block_edge_x_indices[block_x_index + 1])
            
            # Create an array containing a smoothing mask for each corresponding block
            block_masks[:, :, block_y_index, block_x_index] = _create_gaussian_mask(image_height, image_width, block_y_pixel_indices, block_x_pixel_indices, gaussian_filter_smoothing_radius)
            
            # Extract the corresponding blocks in the green and red chanel images
            green_values = green_channel_image[np.ix_(block_y_pixel_indices, block_x_pixel_indices)].flatten()
            red_values = red_channel_image[np.ix_(block_y_pixel_indices, block_x_pixel_indices)].flatten()
            
            # Calculate the linear regression slope to find best fit scaling from green to red
            numerator = np.dot(green_values, red_values)
            denominator = np.dot(green_values, green_values)
            block_weights[block_y_index, block_x_index] = numerator / denominator if denominator > 0 else 0.0


    # Normalize the Gaussian-blurred masks so their sum is 1 at every pixel.
    # These normalized masks are then used to smoothly weight block corrections across the image.
    normalization = block_masks.sum(axis=(-1, -2), keepdims=True)
    block_masks /= normalization

    # Scale masks by regression weights and green channel signal
    block_masks *= block_weights
    block_masks *= green_channel_image[:, :, np.newaxis, np.newaxis]

    # Remove the predicted green signal from red channel
    corrected_red = red_channel_image.copy()
    corrected_red -= block_masks.sum(axis=(-1, -2))

    # Clip negative values to zero (cannot have negative fluorescence)
    corrected_red = np.maximum(0, corrected_red)

    return corrected_red


def _compute_red_intensity_ratio(ops: dict, cell_statistics:list[dict]) -> NDArray[np.float32]:
    """
    Computes the per-cell red labeling probabilities based on the intesity of red inside a cell relative to that of its nearby neuropil (surrounding region). 
    For each cell, the function measures red intensity inside the cell and in the
    immediate background, then computes a ratio that reflects how red-labeled the
    cell is.
    
    Args:
        ops: The dictionary containing descriptive parameters of the processed imaging data. 
        Expected keys include:
            - "Ly", "Lx": image height and width in pixels.
            - "meanImg_chan2": mean image from the red fluorescence channel.
            - "allow_overlap": whether cell masks may overlap.
            - "inner_neuropil_radius": radius (pixels) for background ring.
            - "min_neuropil_pixels": minimum pixel count for background mask.
            - "chan2_thres": ratio threshold to classify cells as red-labeled.
        cell_statistics: List containing the statistics for all detected cells

    Returns:
        An NDArray of shape (number_of_cells, 2)
            [:, 0] = Boolean red-cell classification (True/False).
            [:, 1] = Continuous red intensity ratio (0–1).

    """
    # Extract image dimensions
    image_height, image_width = ops["Ly"], ops["Lx"]

    # Create a global "pixel occupancy map" of all cells
    all_cells_pixel_map = masks.create_cell_pix(cell_statistics, Ly=image_height, Lx=image_width)

    # Create weights masks for each cell
    per_cell_weight_masks = [
        masks.create_cell_mask(cell_data, Ly=image_height, Lx=image_width, allow_overlap=ops["allow_overlap"]) for cell_data in cell_statistics
    ]

    # Create neuropil masks (regions around each cell)
    neuropil_pixel_index_lists= masks.create_neuropil_masks(
        ypixs=[cell["ypix"] for cell in cell_statistics],
        xpixs=[cell["xpix"] for cell in cell_statistics],
        cell_pix=all_cells_pixel_map,
        inner_neuropil_radius=ops["inner_neuropil_radius"],
        min_neuropil_pixels=ops["min_neuropil_pixels"],
    )
    # Initialize arrays to hold the per-cell mask data
    number_of_cells = len(cell_statistics)
    number_of_pixels = image_height * image_width
    cell_masks_matrix = np.zeros((number_of_cells, number_of_pixels), np.float32)
    neuropil_masks_matrix = np.zeros((number_of_cells, number_of_pixels), np.float32)

    # Populate the arrays with weights
    for (cell_mask_row, (cell_pixel_indices, cell_pixel_weights), neuropil_mask_row, neuropil_pixel_indices
         )in zip(
        cell_masks_matrix, per_cell_weight_masks, neuropil_masks_matrix, neuropil_pixel_index_lists, strict=False
    ):
        cell_mask_row[cell_pixel_indices] = cell_pixel_weights
        neuropil_mask_row[neuropil_pixel_indices.astype(np.int64)] = 1.0 / len(neuropil_pixel_indices)

    # Project the red channel image through those masks
    flat_red_channel = ops["meanImg_chan2"]
    inside_cell_intensity = cell_masks_matrix @ flat_red_channel.flatten()
    outside_cell_intensity = neuropil_masks_matrix @ flat_red_channel.flatten()

    # Compute per-cell probability of being red
    inside_cell_intensity = np.maximum(1e-3, inside_cell_intensity) # Avoid division by 0
    redprob = inside_cell_intensity / (inside_cell_intensity + outside_cell_intensity)

    # Classify as red cell based on threshold value
    is_red_cell = redprob > ops["chan2_thres"]

    return np.stack((is_red_cell, redprob), axis=-1)


def _cellpose_overlap(cell_statistics:  list[dict], mean_red_image:NDArray[np.float32]) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """
    Calculates overlap of extracted cells with Cellpose anatomical mask. 
    Cellpose segments the red channel anatomical structure, then computes the Intersection-over-Union (IoU).


    Args:
        cell_statistics: List containing the statistics for all detected cells
        mean_red_image: The red-channel mean image


    Returns:
        redstats: An NDArray of shape (number_of_cells, 2)
            [:, 0] = Boolean red-cell classification (True/False).
            [:, 1] = Continuous red intensity ratio (0–1).
        masks:

    """
    from . import anatomical
    # Run Cellpose to detect anatomical regions in the red channel image and extract mask labels
    anatomical_masks = anatomical.roi_detect(mean_red_image)[0]

    # Extract image dimensions
    image_height, image_width = anatomical_masks.shape

    # Initialize array to store overlap information for each cell
    # Column 0: Boolean red label; Column 1: IoU value
    red_cell_statistics = np.zeros((len(cell_statistics), 2), np.float32)

    # Compute overlap with anatomical masks for each cell
    for cell_index, cell_data in enumerate(cell_statistics):
        # Initialize an rmpty binary mask for this cell
        single_cell_mask = np.zeros((image_height, image_width), np.uint16)
        
        # Fill the corresponding pixel to this cell with 1
        cell_y_pixel, cell_x_pixel = cell_data["ypix"], cell_data["xpix"]
        single_cell_mask[cell_y_pixel, cell_x_pixel] = 1

        # Computes the IoU between this cell and all anatomical Cellpose masks, and then the highest among them
        iou = utils.mask_ious(anatomical_masks, single_cell_mask)[0]
        highest_iou = iou.max()

        # Condition to be likely red-labeled
        likely_red_label_flag = highest_iou > 0.25
        
        # Store results for this cell: [boolean classification, IoU value]
        red_cell_statistics[cell_index] = np.array([likely_red_label_flag, highest_iou])  # this had the wrong dimension

    return red_cell_statistics, anatomical_masks


def detect_red_cells(ops: np.ndarray, cell_statistics: list[dict]) -> tuple(dict, NDArray[np.float32]):
    """
    Detects the red cells in channel 2
    
    Args:
        ops: The dictionary containing descriptive parameters of the processed imaging data. 
        cell_statistics: List containing the statistics for all detected cells

    Returns:
        - 
        - red_cell_results: float32 array of shape (num_cells, 2):
            - [:, 0] → Boolean classification (True if red-labeled, else False)
            - [:, 1] → Confidence metric

    """
    green_channel_image = ops["meanImg"].copy()
    red_channel_image = ops["meanImg_chan2"].copy()

    # non-rigid regression with nblks x nblks pieces
    bleedthrough_blocks_per_dimension = 3
    image_height, image_width = ops["Ly"], ops["Lx"]

    # Correct the green bleedthrough and store it in 'ops'
    ops["meanImg_chan2_corrected"] = _correct_green_bleedthrough(
        image_height=image_height, 
        image_width=image_width, 
        number_of_blocks=bleedthrough_blocks_per_dimension, 
        green_channel_image=green_channel_image, 
        red_channel_image=red_channel_image
        )

    red_cell_results = None

    # Attempt to detect red cells using Cellpose
    if ops.get("anatomical_red", True):
        try:
            console.echo(
            message=f">>>> CELLPOSE estimating masks in anatomical channel",
            level=LogLevel.SUCCESS,
            )

            red_cell_results, anatomical_masks = _cellpose_overlap(cell_statistics=cell_statistics, mean_red_image=red_channel_image)
        except:
            console.echo(
                message=f"ERROR importing or running cellpose, continuing without anatomical estimates",
                level=LogLevel.SUCCESS,
            )
            red_cell_results = None

    # If we cannot run cell pose, we use intensity ratio to detect red cells
    if red_cell_results is None:
        red_cell_results = _compute_red_intensity_ratio(ops, cell_statistics)
    else:
        ops["chan2_masks"] = anatomical_masks

    return ops, red_cell_results
