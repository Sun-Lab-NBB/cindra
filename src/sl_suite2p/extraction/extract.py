"""This module contains functions for extracting cell and neuropil fluorescence from the ROI masks."""

from typing import Any

from numba import njit, config, prange
import numpy as np
from scipy import stats
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import LogLevel, console

from .masks import create_masks
from ..io.binary import BinaryFile, BinaryFileCombined

# Configures the numba threading layer.
config.THREADING_LAYER = "tbb"


@njit(parallel=True)
def _extract_cell_fluorescence(
    output_prototype: NDArray[np.float32],
    data: NDArray[np.float32],
    cell_masks: tuple[NDArray[np.uint32]],
    lambda_weight_masks: tuple[NDArray[np.float32]],
) -> NDArray[np.float32]:
    """Extracts cell fluorescence traces for the requested ROIs.

    Args:
        output_prototype: The pre-initialized output array to be updated with the extracted fluorescence traces.
        data: The raw activity data from which to extract the fluorescence traces.
        cell_masks: The cell masks for each ROI to process.
        lambda_weight_masks: The lambda weight for each pixel in every cell mask.

    Returns:
        The output_prototype array updated with the extracted cell fluorescence traces for each processed ROI.
    """
    cell_count = np.uint32(output_prototype.shape[0])

    for cell_index in prange(cell_count):
        cell_pixels_data = data[:, cell_masks[cell_index]]
        lambda_weight = lambda_weight_masks[cell_index]

        # Uses pixel lambda weight to weigh the pixel fluorescence. This biases the trace to base more of
        # the extracted signals on the pixels that are more likely to belong to the cell.
        output_prototype[cell_index] = np.dot(cell_pixels_data, lambda_weight)

    return output_prototype


@njit(parallel=True)
def _extract_neuropil_fluorescence(
    output_prototype: NDArray[np.float32],
    data: NDArray[np.float32],
    neuropil_masks: tuple[NDArray[np.uint32]],
    neuropil_pixel_count: NDArray[np.uint32],
) -> NDArray[np.float32]:
    """Extracts neuropil fluorescence traces for the requested ROIs.

    Args:
        output_prototype: The pre-initialized output array to be updated with the extracted fluorescence traces.
        data: The raw activity data from which to extract the fluorescence traces.
        neuropil_masks: The neuropil masks for each ROI to process.
        neuropil_pixel_count: The number of pixels in each neuropil mask.
    """
    n_cells = np.uint32(output_prototype.shape[0])

    for cell_idx in prange(n_cells):
        # Computes the average fluorescence over the entire neuropil region.
        output_prototype[cell_idx] = data[:, neuropil_masks[cell_idx]].sum(axis=1) / neuropil_pixel_count[cell_idx]

    return output_prototype


def extract_traces(
    data: BinaryFile | BinaryFileCombined,
    cell_masks: tuple[tuple[NDArray[np.uint32], NDArray[np.float32]]],
    neuropil_masks: tuple[NDArray[np.uint32]] | None = None,
    batch_size: int = 500,
    plane: int = 0,
    session_id: str = "",
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Extracts the fluorescence traces from the raw cell activity data (movie) using cell and neuropil masks.

    Notes:
        If neuropil masks are not provided, the neuropil fluorescence traces are returned as an array of zeroes.

    Args:
        data: The raw cell activity data (movie) to process.
        cell_masks: The cell masks for each ROI. Note; each cell mask must be a tuple that contains the flattened cell
            mask as the first element and the lambda weights for each mask pixel as the second element.
        neuropil_masks: The neuropil masks for each roi.
        batch_size: The number of frames processed at once. Note; the maximum batch size is capped at 1000 frames.
        plane: The index of the image plane being processed, if the function is called as part of the single-day
            processing pipeline.
        session_id: The ID (name) of the session being processed, if the function is called as part of the multi-day
            processing pipeline.

    Returns:
        The extracted cell and neuropil fluorescence traces stored as arrays with dimensions (roi_count, frame_count).
    """
    # Notifies the user about the start of the processing.
    if session_id == "":
        console.echo(message=f"Extracting ROI fluorescence data for plane {plane}...", level=LogLevel.INFO)
    else:
        console.echo(message=f"Extracting ROI fluorescence data for session {session_id}...", level=LogLevel.INFO)

    # Instantiates and starts the execution timer.
    timer = PrecisionTimer("s")
    timer.reset()

    # Extracts processed movie statistics.
    frame_count, height, width = data.shape
    cell_count = len(cell_masks)
    pixel_count = height * width

    # Caps the batch size at 1000 frames
    batch_size = min(batch_size, 1000)

    # Pre-allocates the arrays to store the extracted cell and neuropil fluorescence traces
    fluorescence = np.zeros((cell_count, frame_count), dtype=np.float32)
    neuropil_fluorescence = np.zeros((cell_count, frame_count), dtype=np.float32)

    # Decomposes cell mask inputs into standalone tuples of cell mask indices and lambda weights. This format is
    # preferred for the parallel processing steps below.
    cell_mask_indices = []
    cell_lambda_weights = []
    for pixel_indices, lambda_weights in cell_masks:
        cell_mask_indices.append(pixel_indices)
        cell_lambda_weights.append(lambda_weights)

    # Computes the pixel counts for each neuropil mask
    if neuropil_masks is not None:
        neuropil_pixel_count = np.array([len(indices) for indices in neuropil_masks], dtype=np.uint32)

    # Extracts the cell fluorescence from all frames of the processed cell activity movie.
    current_frame = 0
    for batch_start in range(0, frame_count, batch_size):
        batch_end = min(batch_start + batch_size, frame_count)

        # Reshapes each batch from [frames, height, width] to [frames, pixels]
        batch_data = data[batch_start:batch_end].astype(np.float32)
        batch_frames = batch_data.shape[0]
        batch_pixels = batch_data.reshape(batch_frames, pixel_count)
        current_batch_slice = slice(current_frame, current_frame + batch_frames)

        # Pre-allocates an array of size [cell_count, batch_frames] to store the cell
        # and neuropil fluorescence values extracted from the currently processed batch of frames.
        output_prototype = np.zeros((cell_count, batch_frames), dtype=np.float32)

        # Extracts the cell fluorescence from all frames of the currently processed batch of frames.
        fluorescence[:, current_batch_slice] = _extract_cell_fluorescence(
            output_prototype=output_prototype,
            data=batch_pixels,
            cell_masks=tuple(cell_mask_indices),
            lambda_weight_masks=tuple(cell_lambda_weights),
        )

        # If neuropil masks are provided, extracts the neuropil fluorescence from all frames of the currently
        # processed batch of frames
        if neuropil_masks is not None:
            # noinspection PyUnboundLocalVariable
            neuropil_fluorescence[:, current_batch_slice] = _extract_neuropil_fluorescence(
                output_prototype=output_prototype,
                data=batch_pixels,
                neuropil_masks=tuple(neuropil_masks),
                neuropil_pixel_count=neuropil_pixel_count,
            )

        current_frame += batch_frames

    # Determines the processing time and notifies the user about the completion of the processing.
    elapsed_time = timer.elapsed
    if session_id == "":
        message = (
            f"Plane {plane} ROI fluorescence: extracted from {cell_count} ROIs in {frame_count} frames. "
            f"Time taken: {elapsed_time:.2f} seconds."
        )
    else:
        message = (
            f"Session {session_id} ROI fluorescence: extracted from {cell_count} ROIs in {frame_count} frames. "
            f"Time taken: {elapsed_time:.2f} seconds."
        )
    console.echo(message=message, level=LogLevel.SUCCESS)

    return fluorescence, neuropil_fluorescence


def extract_traces_from_masks(
    ops: dict[str, Any],
    cell_masks: tuple[tuple[NDArray[np.uint32], NDArray[np.float32]]],
    neuropil_masks: tuple[NDArray[np.uint32]] | None,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
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
            data=f, cell_masks=cell_masks, neuropil_masks=neuropil_masks, batch_size=batch_size
        )

    cell_fluorescence_channel_2 = []
    neuropil_fluorescence_channel_2 = []

    if ops.get("reg_file_chan2"):
        with BinaryFile(height=height, width=width, file_path=ops["reg_file_chan2"]) as f:
            cell_fluorescence_channel_2, neuropil_fluorescence_channel_2 = extract_traces(
                data=f, cell_masks=cell_masks, neuropil_masks=neuropil_masks, batch_size=batch_size
            )

    return cell_fluorescence, neuropil_fluorescence, cell_fluorescence_channel_2, neuropil_fluorescence_channel_2


def extraction_wrapper(
    roi_statistics: list[dict[str, Any]],
    plane_number: int,
    frames: BinaryFile | BinaryFileCombined,
    ops: dict[str, Any],
    channel_2_frames: BinaryFile | BinaryFileCombined | None = None,
) -> tuple[list[dict[str, Any]], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Main extraction function that creates the masks and computes the fluorescence traces.

    Args:
        roi_statistics: The dictionary that stores the statistics for regions of interest (ROIs), including cell masks.
        plane_number: The number (index) of the processed plane.
        frames: The path to the binary file that stores the registered plane frames for which to process the ROIs.
        ops: The dictionary that stores the plane registration parameters.
        channel_2_frames: Same as 'frames_path', but for the second functional channel, if the plane data contains
                data from two channels.
    """
    timer = PrecisionTimer("s")
    _, height, width = frames.shape
    batch_size = ops["batch_size"]
    neuropil_coefficient = ops["neuropil_coefficient"]

    # Creates cell and neuropil masks if not provided
    console.echo(f"Creating ROI masks for plane {plane_number}...", level=LogLevel.INFO)
    timer.reset()

    cell_masks, neuropil_masks = create_masks(
        roi_statistics=roi_statistics, height=height, width=width, neuropil=ops.get("extract_neuropil", True), ops=ops
    )

    console.echo(
        f"Plane {plane_number} ROI masks: created. Time taken: {timer.elapsed} seconds.", level=LogLevel.SUCCESS
    )

    # Extracts fluorescence traces for primary channel
    cell_fluorescence, neuropil_fluorescence = extract_traces(
        data=frames,
        plane=plane_number,
        cell_masks=cell_masks,
        neuropil_masks=neuropil_masks,
        batch_size=batch_size,
    )

    cell_fluorescence_channel_2 = []
    neuropil_fluorescence_channel_2 = []

    # Processes second channel if available
    if channel_2_frames:
        cell_fluorescence_channel_2, neuropil_fluorescence_channel_2 = extract_traces(
            data=channel_2_frames,
            plane=plane_number,
            cell_masks=cell_masks,
            neuropil_masks=neuropil_masks,
            batch_size=batch_size,
        )

    # Applies neuropil correction to cell fluorescence
    corrected = cell_fluorescence - neuropil_coefficient * neuropil_fluorescence

    # Computes skewness and standard deviation for each ROI and updates the corresponding ROI statistics dictionary
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
