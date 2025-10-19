"""This module contains functions for extracting cell and neuropil fluorescence from the ROI masks."""

from typing import Any

from numba import njit, config, prange
import numpy as np
from scipy import stats, signal
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import LogLevel, console

from .masks import create_masks
from ..io.binary import BinaryFile
from ..configuration import generate_default_ops

# Configures the numba threading layer.
config.THREADING_LAYER = "tbb"

_SCALE_BACKGROUND = 4
_MINIMUM_INTENSITY = -6
_MAXIMUM_INTENSITY = 6


@njit(parallel=True)
def _extract_cell_fluorescence(
    output_prototype: NDArray[np.float32],
    data: NDArray[np.float32],
    cell_masks: list[NDArray[np.uint32]],
    lambda_weight_masks: list[NDArray[np.float32]],
) -> NDArray[np.float32]:
    """Extracts cell fluorescence traces across all frames for each ROI.

    Args:
        output_prototype: The pre-initialized output array to be updated with the extracted fluorescence traces.
        data: The raw activity data from which to extract the fluorescence traces.
        cell_masks: The list of cell mask pixel indices for each ROI, stored as a flattened NumPy array.
        lambda_weight_masks: The list of lambda weight masks for each ROI, stored as a flattened NumPy array.

    Returns:
        The output_prototype array updated with the extracted cell fluorescence traces for each processed ROI.
    """
    cell_count = np.int64(output_prototype.shape[0])

    for cell_index in prange(cell_count):
        cell_pixels_data = data[:, cell_masks[cell_index]]
        lambda_weight = lambda_weight_masks[cell_index]

        # Uses pixel lambda weight to weigh the pixel fluorescence. This biases the trace to base more of the extracted
        # signals on the pixels that are more likely to belong to the cell.
        output_prototype[cell_index] = np.dot(cell_pixels_data, lambda_weight)

    return output_prototype


@njit(parallel=True)
def _extract_neuropil_fluorescence(
    output_prototype: NDArray[np.float32],
    data: NDArray[np.float32],
    neuropil_pixel_indices: list[NDArray[np.uint32]],
    neuropil_pixel_count: NDArray[np.uint32],
) -> NDArray[np.float32]:
    """Extracts the fluorescence signals from pixels in the surrounding neuropil ring and computes
    the mean fluorescence across those pixels.

    Args:
        output_prototype: The pre-initialized output array to be updated with the extracted fluorescence traces.
        data: The raw activity data from which to extract the fluorescence traces.
        neuropil_pixel_indices: A list of arrays containing the pixel indices forming the neuropil ring
                                around each ROI.
        neuropil_pixel_count: An array indicating the number of neuropil pixels for each ROI.
    """
    n_cells = np.int64(output_prototype.shape[0])

    for cell_idx in prange(n_cells):
        output_prototype[cell_idx] = (
            data[:, neuropil_pixel_indices[cell_idx]].sum(axis=1) / neuropil_pixel_count[cell_idx]
        )

    return output_prototype


def extract_traces_from_masks(
    ops: dict[str, Any], cell_masks: list[tuple], neuropil_masks: list
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Computes fluorescence traces for each ROI and its corresponding neuropil region
    from both channels if available.

    Args:
        ops: The dictionary that stores the plane registration parameters.
        cell_masks: A tuple containing an array of flattened ROI pixel indices and a corresponding
                    array of normalized weights to compute the ROI`s fluorescence trace.
        neuropil_masks: An array containing pixel indices of the neuropil surrounding the ROI.
    """
    batch_size = ops["batch_size"]
    height = ops["height"]
    width = ops["width"]

    with BinaryFile(height=height, width=width, file_path=ops["reg_file"]) as f:
        cell_fluorescence, neuropil_fluorescence = extract_traces(
            f_in=f, cell_masks=cell_masks, neuropil_masks=neuropil_masks, batch_size=batch_size
        )

    cell_fluorescence_channel_2 = []
    neuropil_fluorescence_channel_2 = []

    if ops.get("reg_file_chan2"):
        with BinaryFile(height=height, width=width, file_path=ops["reg_file_chan2"]) as f:
            cell_fluorescence_channel_2, neuropil_fluorescence_channel_2 = extract_traces(
                f_in=f, cell_masks=cell_masks, neuropil_masks=neuropil_masks, batch_size=batch_size
            )

    return cell_fluorescence, neuropil_fluorescence, cell_fluorescence_channel_2, neuropil_fluorescence_channel_2


def extract_traces(
    f_in: np.ndarray,
    plane_number: int,
    cell_masks: list[NDArray],
    neuropil_masks: list[NDArray] | None = None,
    batch_size: int = 500,
    session_id: str = "",
) -> tuple[NDArray, NDArray]:
    """Extracts fluorescence traces from imaging data using cell and neuropil masks.

    Args:
        f_in: An np.ndarray or io.BinaryFile of the imaging data with shape [n_frames, height, width].
        plane_number: The index of the image plane being processed for logging purposes.
        cell_masks: A list where each element is a tuple of pixel indices and their
                    corresponding lambda weights. The pixel indices are flattened pixel
                    locations and the lambda weights are normalized to sum to 1.
        neuropil_masks: The neuropil pixel indices for each cell.
        batch_size: The number of frames processed at once, with a maximum of 1000
                    and a default of 500.
        session_id: The session identifier used for logging, which overrides the plane_number
                    in messages.
    """
    if session_id == "":
        console.echo(f"Extracting ROI fluorescence data for plane {plane_number}...", level=LogLevel.INFO)
    else:
        console.echo(f"Extracting ROI fluorescence data for session {session_id}...", level=LogLevel.INFO)

    timer = PrecisionTimer("s")
    timer.reset()

    n_frames, height, width = f_in.shape
    n_cells = len(cell_masks)
    n_pixels = height * width
    actual_batch_size = min(batch_size, 1000)

    # Computes ROI fluorescence as the weighted pixel sums
    fluorescence = np.zeros((n_cells, n_frames), dtype=np.float32)
    neuropil_fluorescence = np.zeros((n_cells, n_frames), dtype=np.float32)

    cell_pixel_indices = []
    cell_lambda_weights = []
    for pixel_indices, lambda_weights in cell_masks:
        cell_pixel_indices.append(pixel_indices.astype(np.int64))
        cell_lambda_weights.append(lambda_weights.astype(np.float32))

    # Computes neuropil fluorescence as the mean intensity over neuropil pixels
    has_neuropil = neuropil_masks is not None
    if has_neuropil:
        neuropil_pixel_indices = []

        if isinstance(neuropil_masks, np.ndarray) and neuropil_masks.shape[1] == n_pixels:
            neuropil_pixel_indices.extend(np.nonzero(mask_row)[0].astype(np.int64) for mask_row in neuropil_masks)
        else:
            neuropil_pixel_indices.extend(mask_indices.astype(np.int64) for mask_indices in neuropil_masks)

    neuropil_pixel_count = np.array([len(indices) for indices in neuropil_pixel_indices], dtype=np.float32)

    current_frame = 0

    for batch_start in range(0, n_frames, actual_batch_size):
        batch_end = min(batch_start + actual_batch_size, n_frames)

        # Reshapes each batch from [frames, height, width] to [frames, pixels]
        batch_data = f_in[batch_start:batch_end].astype(np.float32)
        n_batch_frames = batch_data.shape[0]
        batch_n_pixels = batch_data.reshape(n_batch_frames, n_pixels)
        current_batch_slice = slice(current_frame, current_frame + n_batch_frames)

        # Pre-allocates an array of size [n_cells, n_batch_frames] to store extracted cell
        # and neuropil fluorescence values in the current batch
        output_prototype = np.zeros((n_cells, n_batch_frames), dtype=np.float32)

        fluorescence[:, current_batch_slice] = _extract_cell_fluorescence(
            output_prototype=output_prototype,
            data=batch_n_pixels,
            cell_masks=cell_pixel_indices,
            lambda_weight_masks=cell_lambda_weights,
        )

        if has_neuropil:
            neuropil_fluorescence[:, current_batch_slice] = _extract_neuropil_fluorescence(
                output_prototype=output_prototype,
                data=batch_n_pixels,
                neuropil_pixel_indices=neuropil_pixel_indices,
                neuropil_pixel_count=neuropil_pixel_count,
            )

        current_frame += n_batch_frames

    elapsed_time = timer.elapsed
    if session_id == "":
        message = (
            f"Plane {plane_number} ROI fluorescence: extracted from {n_cells} ROIs in {n_frames} frames. "
            f"Time taken: {elapsed_time:.2f} seconds."
        )
    else:
        message = (
            f"Session {session_id} ROI fluorescence: extracted from {n_cells} ROIs in {n_frames} frames. "
            f"Time taken: {elapsed_time:.2f} seconds."
        )
    console.echo(message=message, level=LogLevel.SUCCESS)

    return fluorescence, neuropil_fluorescence


def extraction_wrapper(
    roi_statistics: list[dict[str, Any]],
    plane_number: int,
    frames: NDArray,
    frames_channel_2: NDArray | None = None,
    cell_masks: list[NDArray] | None = None,
    neuropil_masks: list[NDArray] | None = None,
    ops: dict[str, Any] | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Main extraction function that creates the masks and computes the fluorescence traces.

    Args:
        roi_statistics: The dictionary that stores the statistics for regions of interest (ROIs), including cell masks.
        plane_number: The number (index) of the processed plane.
        frames: The path to the binary file that stores the registered plane frames for which to process the ROIs.
        frames_channel_2: Same as 'frames_path', but for the second functional channel, if the plane data contains
                data from two channels.
        cell_masks: A list where each element is a tuple of pixel indices and their corresponding lambda weights.
                    The pixel indices are flattened pixel locations and the lambda weights are normalized to sum
                    to 1.
        neuropil_masks: The neuropil pixel indices for each cell.
        ops: The dictionary that stores the plane registration parameters.
    """
    if ops is None:
        ops = generate_default_ops()

    timer = PrecisionTimer("s")
    _, height, width = frames.shape
    batch_size = ops["batch_size"]
    neucoeff = ops["neucoeff"]

    # Creates cell and neuropil masks if not provided
    if cell_masks is None:
        console.echo(f"Creating ROI masks for plane {plane_number}...", level=LogLevel.INFO)
        timer.reset()

        cell_masks, new_neuropil_masks = create_masks(
            roi_statistics=roi_statistics, height=height, width=width, ops=ops
        )

        if neuropil_masks is None:
            neuropil_masks = new_neuropil_masks

        console.echo(
            f"Plane {plane_number} ROI masks: created. Time taken: {timer.elapsed} seconds.", level=LogLevel.SUCCESS
        )

    # Extracts fluorescence traces for primary channel
    cell_fluorescence, neuropil_fluorescence = extract_traces(
        f_in=frames,
        plane_number=plane_number,
        cell_masks=cell_masks,
        neuropil_masks=neuropil_masks,
        batch_size=batch_size,
    )

    cell_fluorescence_channel_2 = []
    neuropil_fluorescence_channel_2 = []

    # Processes second channel if available
    if frames_channel_2:
        cell_fluorescence_channel_2, neuropil_fluorescence_channel_2 = extract_traces(
            f_in=frames_channel_2,
            plane_number=plane_number,
            cell_masks=cell_masks,
            neuropil_masks=neuropil_masks,
            batch_size=batch_size,
        )

    # Applies neuropil correction to cell fluorescence
    corrected = cell_fluorescence - neucoeff * neuropil_fluorescence

    # Computes skewness and standard deviation for each ROI and updates the corresponding ROI statistics dictiona
    skew_values = stats.skew(corrected, axis=1)
    std_values = np.std(corrected, axis=1)

    for i, (roi_stat, skew, std) in enumerate(zip(roi_statistics, skew_values, std_values, strict=False)):
        roi_stat.update({"skew": skew, "std": std})
        if neuropil_masks is not None:
            roi_stat["neuropil_mask"] = neuropil_masks[i]

    return (
        roi_statistics,
        cell_fluorescence,
        neuropil_fluorescence,
        cell_fluorescence_channel_2,
        neuropil_fluorescence_channel_2,
    )


def enhanced_mean_image(ops: dict[str, Any]) -> dict[str, Any]:
    """Computes an enhanced mean image by removing background noise and normalizing
    the local variance.

    Args:
        ops: The dictionary that stores the plane registration parameters.

    Returns:
        The input 'ops' dictionary, expanded to include the 'meanImgE' field.
    """
    mean_image = ops["meanImg"].astype(np.float32)

    if "spatscale_pix" not in ops:
        if isinstance(ops["diameter"], int):
            cell_diameter = np.array([ops["diameter"], ops["diameter"]])
        else:
            cell_diameter = np.array(ops["diameter"])

        if cell_diameter[0] == 0:
            cell_diameter[:] = 12

        ops["spatscale_pix"] = cell_diameter[1]
        ops["aspect"] = cell_diameter[0] / cell_diameter[1]

    filter_height = _SCALE_BACKGROUND * np.ceil(ops["spatscale_pix"] * ops["aspect"]) + 1
    filter_width = _SCALE_BACKGROUND * np.ceil(ops["spatscale_pix"]) + 1
    filter_kernel_size = (int(filter_height), int(filter_width))

    background = signal.medfilt2d(mean_image, filter_kernel_size)
    background_removed = mean_image - background

    local_variance = signal.medfilt2d(np.absolute(background_removed), filter_kernel_size)
    normalized_image = background_removed / (1e-10 + local_variance)

    y_start, y_end = ops["yrange"]
    x_start, x_end = ops["xrange"]
    roi_image = normalized_image[y_start:y_end, x_start:x_end]

    scaled_roi = (roi_image - _MINIMUM_INTENSITY) / (_MAXIMUM_INTENSITY - _MINIMUM_INTENSITY)
    scaled_roi = np.clip(scaled_roi, 0, 1)

    height, width = ops["height"], ops["width"]
    enhanced_image = np.full((height, width), scaled_roi.min(), dtype=np.float32)
    enhanced_image[y_start:y_end, x_start:x_end] = scaled_roi

    ops["meanImgE"] = enhanced_image

    return ops
