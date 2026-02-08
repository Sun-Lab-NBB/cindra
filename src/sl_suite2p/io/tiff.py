"""Provides tools for importing, converting, and saving TIFF imaging data."""

import gc
import math
from typing import TYPE_CHECKING

from tqdm import tqdm
import numpy as np
from natsort import natsorted
from tifffile import TiffFile
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists

from .binary import BinaryFile
from ..dataclasses import (
    IOData,
    RuntimeContext,
    SingleDayRuntimeData,
    AcquisitionParameters,
    SingleDayConfiguration,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

# Supported TIFF file extensions.
_TIFF_EXTENSIONS = ("tif", "tiff", "TIF", "TIFF")

# Default name for the acquisition parameters JSON file.
_ACQUISITION_PARAMETERS_FILENAME = "suite2p_parameters.json"

# Determines the minimum number of image dimensions considered 'multidimensional'.
_MULTIDIMENSIONAL_PROCESSING_THRESHOLD = 3


def _find_acquisition_parameters(data_path: Path) -> tuple[AcquisitionParameters, Path]:
    """Recursively searches for the acquisition parameters JSON file and loads it.

    This function searches the data_path directory and all subdirectories for a file named
    'suite2p_parameters.json'. Once found, it loads and validates the acquisition parameters.

    Args:
        data_path: The root directory to search for the acquisition parameters file.

    Returns:
        A tuple containing the loaded AcquisitionParameters and the path to the directory containing the JSON file.
        The directory path is used for subsequent non-recursive TIFF discovery.

    Raises:
        FileNotFoundError: If no acquisition parameters file is found in the data directory or its subdirectories.
        ValueError: If required fields are missing from the JSON file.
    """
    if not data_path.is_dir():
        message = f"Unable to find acquisition parameters. The data_path is not a directory: {data_path}"
        console.error(message=message, error=ValueError)

    # Recursively searches for the acquisition parameters file.
    parameter_files = list(data_path.rglob(_ACQUISITION_PARAMETERS_FILENAME))

    if not parameter_files:
        message = (
            f"Unable to find '{_ACQUISITION_PARAMETERS_FILENAME}' in the data directory or its subdirectories: "
            f"{data_path}. This file is required and must contain acquisition metadata."
        )
        console.error(message=message, error=FileNotFoundError)

    # Uses the first found file (there should typically be only one).
    parameters_path = parameter_files[0]
    data_directory = parameters_path.parent

    message = f"Found acquisition parameters at: {parameters_path}."
    console.echo(message=message, level=LogLevel.SUCCESS)

    # Loads and validates the acquisition parameters.
    acquisition = AcquisitionParameters.from_json(parameters_path)

    return acquisition, data_directory


def _discover_tiff_files(
    data_directory: Path,
    ignored_file_names: tuple[str, ...] = (),
) -> list[Path]:
    """Discovers TIFF files in the specified directory (non-recursive).

    This function performs a non-recursive scan of the data_directory for TIFF files. It is designed to be called
    after _find_acquisition_parameters() to ensure TIFF files and their metadata are co-located.

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

    message = f"Found {len(file_paths)} TIFF files."
    console.echo(message=message, level=LogLevel.INFO)

    return file_paths


def _initialize_plane_contexts(
    config: SingleDayConfiguration,
    acquisition: AcquisitionParameters,
) -> list[RuntimeContext]:
    """Creates plane-specific RuntimeContext instances for each imaging plane in the recording.

    This function initializes the runtime data structures needed for processing each plane. For standard single-ROI
    data, one context is created per physical plane. For MROI (Multi-ROI) data, one context is created per virtual
    plane, where virtual planes are ROI x physical plane combinations.

    Args:
        config: The single-day pipeline configuration containing user-defined processing parameters.
        acquisition: The acquisition parameters loaded from the input data's JSON file. This describes the recording
            setup including frame rate, plane count, channel count, and MROI geometry if applicable.

    Returns:
        A list of RuntimeContext instances, one per plane (or virtual plane for MROI data). Each context contains
        references to the shared configuration, acquisition parameters, and a plane-specific SingleDayRuntimeData
        instance with IOData fields initialized.

    Raises:
        ValueError: If the save_path is not configured in the configuration.
    """
    # Validates that the save path is configured.
    save_path_root = config.file_io.save_path
    if save_path_root is None:
        message = (
            "Unable to initialize plane contexts. The save_path must be configured in the FileIO section of the "
            "configuration, but it is currently None."
        )
        console.error(message=message, error=ValueError)

    # Determines the number of contexts to create. For MROI data, creates one context per virtual plane
    # (ROI x physical plane combination). For single-ROI data, creates one context per physical plane.
    plane_count = acquisition.virtual_plane_count if acquisition.is_mroi else acquisition.plane_number

    # Determines whether the recording uses two channels.
    has_two_channels = acquisition.channel_number > 1

    # Initializes the list to store RuntimeContext instances for each plane.
    contexts: list[RuntimeContext] = []

    # Creates a RuntimeContext for each plane.
    for virtual_plane_index in range(plane_count):
        # Resolves the output directory for this plane. Always uses 'suite2p' as the subdirectory.
        plane_output_path = save_path_root / "suite2p" / f"plane_{virtual_plane_index}"

        # Creates the output directory if it does not exist.
        ensure_directory_exists(plane_output_path)

        # Initializes the IOData for this plane with binary file paths.
        io_data = IOData(
            output_directory=plane_output_path,
            registered_binary_path=plane_output_path / "channel_1_data.bin",
            plane_index=virtual_plane_index,
        )

        # Configures second channel binary paths if using two channels.
        if has_two_channels:
            io_data.registered_binary_path_channel_2 = plane_output_path / "channel_2_data.bin"

        # Populates MROI-specific fields if processing multi-ROI data.
        if acquisition.is_mroi:
            # Computes ROI index and physical plane index from the virtual plane index. Virtual planes are organized
            # as: ROI 0 plane 0, ROI 0 plane 1, ..., ROI 1 plane 0, ROI 1 plane 1, etc.
            roi_index = virtual_plane_index // acquisition.plane_number
            physical_plane_index = virtual_plane_index % acquisition.plane_number

            io_data.mroi_lines = list(acquisition.roi_lines[roi_index])
            io_data.mroi_y_offset = acquisition.roi_y_coordinates[roi_index]
            io_data.mroi_x_offset = acquisition.roi_x_coordinates[roi_index]
            # For MROI, stores the physical plane index (which may differ from virtual plane index).
            io_data.plane_index = physical_plane_index

        # Creates the SingleDayRuntimeData with the initialized IOData.
        runtime_data = SingleDayRuntimeData(
            output_path=plane_output_path,
            io=io_data,
        )

        # Creates the RuntimeContext combining the shared configuration, acquisition parameters, and runtime data.
        context = RuntimeContext(
            configuration=config,
            acquisition=acquisition,
            runtime=runtime_data,
        )

        contexts.append(context)

    return contexts


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
    tiff, tiff_length = _open_tiff(tiff_files[0])
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
        channel_1_binary_files.append(BinaryFile(height, width, registered_path, frames_per_plane))

        # Creates channel 2 binary file if applicable.
        if has_two_channels:
            registered_path_ch2 = io_data.registered_binary_path_channel_2
            if registered_path_ch2 is None:
                message = (
                    f"Unable to create binary file for plane {io_data.plane_index} channel 2. The "
                    f"registered_binary_path_channel_2 is not configured in IOData."
                )
                console.error(message=message, error=ValueError)
            channel_2_binary_files.append(BinaryFile(height, width, registered_path_ch2, frames_per_plane))

    return channel_1_binary_files, channel_2_binary_files


def _open_tiff(file_path: Path) -> tuple[TiffFile, int]:
    """Opens a TIFF file and returns the file handle with page count.

    This function is a prerequisite for reading the data stored inside the specified .tiff file. It does not load the
    data into memory.

    Args:
        file_path: The absolute path to the .tiff file from which to read the frame data.

    Returns:
        A tuple containing the TiffFile instance wrapping the specified file and the number of pages (frames) stored
        inside the file.
    """
    tiff = TiffFile(file_path)
    tiff_length = len(tiff.pages)
    return tiff, tiff_length


def _read_tiff(tiff: TiffFile, start_index: int, batch_size: int) -> NDArray[np.int16] | None:
    """Reads a batch (subset) of frames stored inside the .tiff file wrapped by the input TiffFile instance.

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
        frames = (frames // 2).astype(np.int16)
    elif frames.dtype.type != np.int16:
        frames = frames.astype(np.int16)

    # While this should not be possible, ensures that the returned frame number matches the requested number by
    # truncating any extra frames from the array before returning it to the caller.
    if frames.shape[0] > frames_to_read:
        frames = frames[:frames_to_read, :, :]

    return frames


def convert_tiffs_to_binary(config: SingleDayConfiguration) -> list[RuntimeContext]:
    """Converts TIFF files to suite2p binary format for all planes.

    This is the main entry point for TIFF to binary conversion. It performs the complete workflow: finds acquisition
    parameters, discovers TIFF files, initializes plane contexts, and converts the data. The function handles
    both standard TIFF data and MROI (Multi-ROI) data automatically based on the acquisition parameters loaded from the
    .JSON file that accompanies valid data directories.

    Args:
        config: The single-day pipeline configuration. Must have data_path and save_path configured in file_io.

    Returns:
        A list of RuntimeContext instances, one per plane (or virtual plane for MROI data). Each context contains
        the configuration, acquisition parameters, and runtime data with frame dimensions, counts, and mean images
        populated.

    Raises:
        ValueError: If data_path or save_path is not configured.
        FileNotFoundError: If no acquisition parameters file or TIFF files are found.
    """
    # Validates that data_path is configured and assigns it to a non-optional variable for type narrowing.
    if config.file_io.data_path is None:
        message = (
            "Unable to convert TIFFs to binary. The data_path must be configured in the FileIO section of the "
            "configuration, but it is currently None."
        )
        console.error(message=message, error=ValueError)
    data_path: Path = config.file_io.data_path

    # Finds acquisition parameters and data directory.
    acquisition, data_directory = _find_acquisition_parameters(data_path)

    # Discovers TIFF files in the data directory.
    tiff_files = _discover_tiff_files(data_directory, tuple(config.file_io.ignored_file_names))

    # Initializes plane contexts.
    contexts = _initialize_plane_contexts(config, acquisition)

    # Extracts processing parameters.
    plane_number = acquisition.plane_number
    channel_number = acquisition.channel_number
    is_mroi = acquisition.is_mroi
    display_progress = config.main.display_progress_bars

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
        _, tiff_length = _open_tiff(tiff_file)
        total_frames += tiff_length

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

    # Processes each TIFF file.
    for tiff_file in tiff_files:
        tiff, _ = _open_tiff(tiff_file)
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

                # Generates frame indices for this plane's functional channel.
                frame_indices = list(
                    range(
                        physical_plane_index * channel_number + functional_channel_index,
                        frame_count,
                        plane_number * channel_number,
                    )
                )

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
                    channel_2_frame_indices = list(
                        range(
                            physical_plane_index * channel_number + second_channel_index,
                            frame_count,
                            plane_number * channel_number,
                        )
                    )

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

    return contexts
