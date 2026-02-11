"""Provides tools for importing, converting, and saving TIFF imaging data."""

import gc
import math
from typing import TYPE_CHECKING

from tqdm import tqdm
import numpy as np
from natsort import natsorted
from tifffile import TiffFile
from ataraxis_base_utilities import LogLevel, console

from .binary import BinaryFile

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from ..dataclasses import RuntimeContext, AcquisitionParameters

# Supported TIFF file extensions.
_TIFF_EXTENSIONS: tuple[str, ...] = ("tif", "tiff", "TIF", "TIFF")

# Determines the minimum number of image dimensions considered 'multidimensional'.
_MULTIDIMENSIONAL_PROCESSING_THRESHOLD: int = 3


def _discover_tiff_files(
    data_directory: Path,
    ignored_file_names: tuple[str, ...] = (),
) -> list[Path]:
    """Discovers TIFF files in the specified directory.

    Notes:
        This function performs a non-recursive scan of the data_directory for files with valid TIFF extension aliases.

    Args:
        data_directory: The directory to scan for TIFF files. This should be the same directory that contains the
            acquisition parameters JSON file.
        ignored_file_names: A tuple of file names (without extension) to ignore. Files whose stem matches any of
            these names are excluded from the results.

    Returns:
        A list of absolute paths to TIFF files, sorted naturally by filename.

    Raises:
        FileNotFoundError: If no TIFF files are found in the directory.
    """
    if not data_directory.is_dir():
        message = f"Unable to discover TIFF files. The path is not a directory: {data_directory}."
        console.error(message=message, error=ValueError)

    # Performs non-recursive scan for TIFF files.
    file_paths: list[Path] = []
    for extension in _TIFF_EXTENSIONS:
        file_paths.extend(
            file_path.resolve()
            for file_path in data_directory.glob(f"*.{extension}")
            if file_path.stem not in ignored_file_names
        )

    if not file_paths:
        message = f"Unable to find any TIFF files in the data directory: {data_directory}."
        console.error(message=message, error=FileNotFoundError)

    # Sorts files naturally by filename.
    file_paths = natsorted(file_paths)

    message = f"Found {len(file_paths)} valid TIFF files."
    console.echo(message=message, level=LogLevel.INFO)

    return file_paths


def _read_tiff(tiff: TiffFile, start_index: int, batch_size: int) -> NDArray[np.int16] | None:
    """Reads a batch (subset) of frames stored inside the TIFF file wrapped by the input TiffFile instance.

    This function loads the requested subset of data into memory.

    Args:
        tiff: The TiffFile instance that wraps the .tiff file from which to read the data.
        start_index: Index of the first frame to read.
        batch_size: Maximum number of frames to read in this batch.

    Returns:
        A 3D NumPy array with shape (frames, height, width) containing the requested frame data, or None if the
        start_index is beyond the end of the file.
    """
    tiff_length = len(tiff.pages)

    if start_index >= tiff_length:
        return None

    frames_to_read = min(tiff_length - start_index, batch_size)
    frames = tiff.asarray() if tiff_length == 1 else tiff.asarray(key=range(start_index, start_index + frames_to_read))

    # Adds extra dimension for single-frame TIFFs to ensure 3D array.
    if len(frames.shape) < _MULTIDIMENSIONAL_PROCESSING_THRESHOLD:
        frames = np.expand_dims(frames, axis=0)

    # Converts to int16, rescaling where possible. Divides by 2 to shift uint16 (0 to 65535) or int32 values into the
    # int16 range (-32768 to 32767) without overflow.
    if frames.dtype.type in {np.uint16, np.int32}:
        frames = (frames // 2).astype(dtype=np.int16)
    elif frames.dtype.type != np.int16:
        frames = frames.astype(dtype=np.int16)

    # While this should not be possible, ensures that the returned frame number matches the requested number by
    # truncating any extra frames from the array before returning it to the caller.
    if frames.shape[0] > frames_to_read:
        frames = frames[:frames_to_read, :, :]

    return frames


def _get_frame_dimensions(
    tiff_files: list[Path],
    contexts: list[RuntimeContext],
    acquisition: AcquisitionParameters,
) -> tuple[list[int], list[int]]:
    """Pre-scans the first TIFF file to determine frame dimensions for each plane.

    This function reads the first frame from the first TIFF file to get base frame dimensions, then calculates
    per-plane dimensions accounting for MROI slicing if applicable.

    Args:
        tiff_files: The list of TIFF file paths to process.
        contexts: The list of RuntimeContext instances, one per plane.
        acquisition: The acquisition parameters describing the recording setup.

    Returns:
        A tuple of two lists: (heights, widths) where each list has one entry per plane/context.

    Raises:
        ValueError: If the TIFF files are empty or have invalid dimensions.
    """
    # Opens the first TIFF and reads the first frame to get base dimensions.
    tiff = TiffFile(tiff_files[0])
    tiff_length = len(tiff.pages)
    if tiff_length == 0:
        message = f"Unable to determine frame dimensions. The first TIFF file is empty: {tiff_files[0]}"
        console.error(message=message, error=ValueError)

    # Reads a single frame to get dimensions.
    first_frame = tiff.asarray(key=0) if tiff_length > 1 else tiff.asarray()
    base_height, base_width = first_frame.shape[-2], first_frame.shape[-1]

    # Calculates dimensions for each plane/context.
    heights: list[int] = []
    widths: list[int] = []

    for context in contexts:
        io_data = context.runtime.io

        # For MROI data, the height is determined by the ROI line range.
        if acquisition.is_mroi and io_data.mroi_lines:
            plane_height = io_data.mroi_lines[-1] - io_data.mroi_lines[0] + 1
            plane_width = base_width
        else:
            plane_height = base_height
            plane_width = base_width

        heights.append(plane_height)
        widths.append(plane_width)

    return heights, widths


def _create_binary_files(
    contexts: list[RuntimeContext],
    frame_heights: list[int],
    frame_widths: list[int],
    frames_per_plane: int,
) -> tuple[list[BinaryFile], list[BinaryFile]]:
    """Creates BinaryFile instances for writing converted TIFF data for each plane.

    Args:
        contexts: The list of RuntimeContext instances, one per plane. Each context must have IOData with binary file
            paths configured.
        frame_heights: The height of each frame for each plane.
        frame_widths: The width of each frame for each plane.
        frames_per_plane: The total number of frames to be written per plane.

    Returns:
        A tuple of two lists. The first list contains BinaryFile instances for channel 1 (one per plane). The second
        list contains BinaryFile instances for channel 2 (empty if single channel).

    Raises:
        ValueError: If no contexts are provided or if required binary paths are not configured.
    """
    if not contexts:
        message = "Unable to create binary files. At least one RuntimeContext must be provided."
        console.error(message=message, error=ValueError)

    # Uses the first context to get shared acquisition parameters.
    acquisition = contexts[0].acquisition

    # Determines whether the recording uses two channels.
    has_two_channels = acquisition.channel_number > 1

    # Initializes lists to store the BinaryFile instances.
    channel_1_binary_files: list[BinaryFile] = []
    channel_2_binary_files: list[BinaryFile] = []

    # Creates BinaryFile instances for each plane based on the paths in IOData.
    for context_index, context in enumerate(contexts):
        io_data = context.runtime.io
        height = frame_heights[context_index]
        width = frame_widths[context_index]

        # Creates channel 1 binary file.
        registered_path = io_data.registered_binary_path
        if registered_path is None:
            message = (
                f"Unable to create binary file for plane {io_data.plane_index}. The registered_binary_path is not "
                f"configured in IOData."
            )
            console.error(message=message, error=ValueError)
        channel_1_binary_files.append(
            BinaryFile(height=height, width=width, file_path=registered_path, frame_number=frames_per_plane)
        )

        # Creates channel 2 binary file if applicable.
        if has_two_channels:
            registered_path_ch2 = io_data.registered_binary_path_channel_2
            if registered_path_ch2 is None:
                message = (
                    f"Unable to create binary file for plane {io_data.plane_index} channel 2. The "
                    f"registered_binary_path_channel_2 is not configured in IOData."
                )
                console.error(message=message, error=ValueError)
            channel_2_binary_files.append(
                BinaryFile(height=height, width=width, file_path=registered_path_ch2, frame_number=frames_per_plane)
            )

    return channel_1_binary_files, channel_2_binary_files


def convert_tiffs_to_binary(contexts: list[RuntimeContext]) -> None:
    """Converts TIFF files to suite2p binary format for all planes.

    This function performs TIFF to binary conversion using pre-initialized RuntimeContext instances. It discovers TIFF
    files in the data directory, reads them in batches, and writes the converted frames to binary files. The function
    handles both standard TIFF data and MROI (Multi-ROI) data automatically based on the acquisition parameters stored
    in the contexts.

    Notes:
        This function modifies the provided contexts in place, populating frame dimensions, frame counts, and mean
        images in each context's runtime data.

    Args:
        contexts: A list of RuntimeContext instances created by resolve_single_day_contexts(). Each context must have
            valid configuration, acquisition parameters, and IOData with binary file paths configured.

    Raises:
        ValueError: If contexts is empty or data_path is not configured.
        FileNotFoundError: If no TIFF files are found in the data directory.
    """
    if not contexts:
        message = "Unable to convert TIFFs to binary. At least one RuntimeContext must be provided."
        console.error(message=message, error=ValueError)

    # Extracts configuration and acquisition from the first context (shared across all contexts).
    config = contexts[0].configuration
    acquisition = contexts[0].acquisition

    # Uses the data directory stored in IOData during context resolution.
    data_directory = contexts[0].runtime.io.data_directory
    if data_directory is None:
        message = (
            "Unable to convert TIFFs to binary. The data_directory must be set in IOData during context resolution, "
            "but it is currently None."
        )
        console.error(message=message, error=ValueError)

    # Discovers TIFF files in the data directory.
    tiff_files = _discover_tiff_files(data_directory, tuple(config.file_io.ignored_file_names))

    # Extracts processing parameters.
    plane_number = acquisition.plane_number
    channel_number = acquisition.channel_number
    is_mroi = acquisition.is_mroi
    display_progress = config.runtime.display_progress_bars

    # Determines which channel is functional (used for ROI detection).
    functional_channel_index = 0 if config.main.first_channel_functional else 1
    if channel_number == 1:
        functional_channel_index = 0

    # Computes batch size adjusted for planes and channels.
    batch_size = config.registration.batch_size
    batch_size = plane_number * channel_number * math.ceil(batch_size / (plane_number * channel_number))

    # Counts total frames for progress bar and calculates frames per plane.
    total_frames = 0
    for tiff_file in tiff_files:
        total_frames += len(TiffFile(tiff_file).pages)

    # Calculates the number of frames per plane (accounting for interleaved planes and channels).
    frames_per_plane = total_frames // (plane_number * channel_number)

    # Pre-scans TIFF files to determine frame dimensions for each plane.
    frame_heights, frame_widths = _get_frame_dimensions(tiff_files, contexts, acquisition)

    # Creates BinaryFile instances for writing.
    channel_1_binaries, channel_2_binaries = _create_binary_files(
        contexts, frame_heights, frame_widths, frames_per_plane
    )

    # Creates progress bar.
    description = "Converting MROI frames to binary" if is_mroi else "Converting frames to binary"
    pbar = tqdm(total=total_frames, desc=description, unit="frames", disable=not display_progress)

    # Initializes mean image accumulators, frame counters, and write indices for each context.
    mean_images: list[NDArray[np.float32] | None] = [None] * len(contexts)
    mean_images_channel_2: list[NDArray[np.float32] | None] = [None] * len(contexts)
    frame_counts: list[int] = [0] * len(contexts)
    write_indices: list[int] = [0] * len(contexts)

    # Tracks the position within the plane/channel interleave cycle across file boundaries. When a TIFF file ends
    # mid-cycle, the next file must continue from the correct interleave position rather than resetting to zero.
    interleave_stride: int = plane_number * channel_number
    interleave_offset: int = 0

    # Processes each TIFF file.
    for tiff_file in tiff_files:
        tiff = TiffFile(tiff_file)
        start_index = 0

        while True:
            frames = _read_tiff(tiff=tiff, start_index=start_index, batch_size=batch_size)
            if frames is None:
                break

            frame_count = frames.shape[0]
            pbar.update(frame_count)

            # Processes each context (plane or virtual plane).
            for context_index, context in enumerate(contexts):
                io_data = context.runtime.io

                # Determines the physical plane index for frame extraction.
                if is_mroi:
                    physical_plane_index = io_data.plane_index if io_data.plane_index is not None else 0
                    roi_lines = io_data.mroi_lines
                else:
                    physical_plane_index = context_index % plane_number
                    roi_lines = []

                # Generates frame indices for this plane's functional channel, accounting for the interleave
                # offset from previous files.
                target_position = physical_plane_index * channel_number + functional_channel_index
                first_frame_index = (target_position - interleave_offset) % interleave_stride
                frame_indices = list(range(first_frame_index, frame_count, interleave_stride))

                if not frame_indices:
                    continue

                plane_frames = frames[frame_indices]

                # For MROI data, slices frames to extract only the ROI lines.
                if is_mroi and len(roi_lines) > 0:
                    line_start = roi_lines[0]
                    line_end = roi_lines[-1] + 1
                    plane_frames = plane_frames[:, line_start:line_end, :]

                # Initializes mean image accumulator on first batch.
                if mean_images[context_index] is None:
                    mean_images[context_index] = np.zeros(
                        (plane_frames.shape[1], plane_frames.shape[2]), dtype=np.float32
                    )

                # Writes frames to binary file using indexed assignment.
                batch_frame_count = plane_frames.shape[0]
                write_start = write_indices[context_index]
                channel_1_binaries[context_index][write_start : write_start + batch_frame_count] = plane_frames
                write_indices[context_index] += batch_frame_count

                mean_images[context_index] += plane_frames.sum(axis=0, dtype=np.float32)
                frame_counts[context_index] += batch_frame_count

                # Processes channel 2 if applicable.
                if channel_number > 1:
                    second_channel_index = 1 - functional_channel_index
                    target_position_channel_2 = physical_plane_index * channel_number + second_channel_index
                    first_frame_index_channel_2 = (target_position_channel_2 - interleave_offset) % interleave_stride
                    channel_2_frame_indices = list(range(first_frame_index_channel_2, frame_count, interleave_stride))

                    if channel_2_frame_indices:
                        channel_2_frames = frames[channel_2_frame_indices]

                        if is_mroi and len(roi_lines) > 0:
                            line_start = roi_lines[0]
                            line_end = roi_lines[-1] + 1
                            channel_2_frames = channel_2_frames[:, line_start:line_end, :]

                        if mean_images_channel_2[context_index] is None:
                            mean_images_channel_2[context_index] = np.zeros(
                                (channel_2_frames.shape[1], channel_2_frames.shape[2]), dtype=np.float32
                            )

                        # Writes channel 2 frames to binary file using indexed assignment.
                        # Note: write_indices is shared between channels since they have the same frame count.
                        ch2_batch_count = channel_2_frames.shape[0]
                        ch2_write_start = write_indices[context_index] - batch_frame_count
                        channel_2_binaries[context_index][ch2_write_start : ch2_write_start + ch2_batch_count] = (
                            channel_2_frames
                        )
                        mean_images_channel_2[context_index] += channel_2_frames.sum(axis=0, dtype=np.float32)

            start_index += frame_count

        # Updates the interleave offset for the next file based on the total frames in this file.
        interleave_offset = (interleave_offset + start_index) % interleave_stride

        gc.collect()

    pbar.close()

    # Closes binary files and updates runtime data in each context.
    for context_index, context in enumerate(contexts):
        channel_1_binaries[context_index].close()
        if channel_number > 1:
            channel_2_binaries[context_index].close()

        # Computes final mean image by dividing by frame count.
        mean_img = mean_images[context_index]
        if mean_img is not None and frame_counts[context_index] > 0:
            mean_img /= frame_counts[context_index]

        mean_img_ch2 = mean_images_channel_2[context_index]
        if mean_img_ch2 is not None and frame_counts[context_index] > 0:
            mean_img_ch2 /= frame_counts[context_index]

        # Updates IOData with frame dimensions.
        io_data = context.runtime.io
        if mean_img is not None:
            io_data.frame_height = mean_img.shape[0]
            io_data.frame_width = mean_img.shape[1]
        io_data.frame_count = frame_counts[context_index]

        # Updates DetectionData with mean images.
        context.runtime.detection.mean_image = mean_img
        if channel_number > 1:
            context.runtime.detection.mean_image_channel_2 = mean_img_ch2

        # Sets initial valid pixel ranges to full frame (registration will update these).
        context.runtime.registration.valid_y_range = [0, io_data.frame_height]
        context.runtime.registration.valid_x_range = [0, io_data.frame_width]

    message = f"Converted {total_frames} frames across {len(tiff_files)} TIFF files to binary format."
    console.echo(message=message, level=LogLevel.SUCCESS)
