"""This module provides tools for importing data stored inside .tif or .tiff files."""

import gc
import json
import math
from pathlib import Path

from tqdm import tqdm
import numpy as np
from tifffile import TiffFile, TiffWriter
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists

from .utils import find_files_open_binaries, initialize_plane_parameters
from ..configuration import RuntimeData

# Determines the minimum number of image dimensions considered 'multidimensional'
_MULTIDIMENSIONAL_PROCESSING_THRESHOLD = 3


def generate_tiff_filename(
    functional_channel: int, alignment_channel: int, save_path: str | Path, batch_number: int, channel: int
) -> str:
    """Generates a suite2p .tiff filename and its path based on the input parameters.

    Args:
        functional_channel: The number (index) of the channel that contains the functional signal data.
        alignment_channel: The number (index) of the channel used for frame alignment.
        save_path: The absolute path to the root directory where to save the generated .tiff file.
        batch_number: The number (positional index) of the movie frame batch (subset) to be stored inside the .tiff file
            (frame stack). This is used to determine the number assigned to the output .tiff file.
        channel: The number (positional index) of the channel for which the .tiff file is generated. Note, channel
            indexing starts from 0 and, currently, only two channels (0 and 1) are supported.

    Returns:
        The absolute path to the generated tiff file.
    """
    # Ensures that save_path is a Path object
    path = Path(save_path)

    # Determines output subdirectory and channel index based on the input channel number and the numbers of the
    # functional and alignment channels.
    if channel == 0:  # Channel 0.
        if functional_channel == alignment_channel:
            tiff_root = path.joinpath("channel_1_tiffs")
            channel_index = 0
        else:
            tiff_root = path.joinpath("channel_2_tiffs")
            channel_index = 1
    elif functional_channel == alignment_channel:
        tiff_root = path.joinpath("channel_2_tiffs")
        channel_index = 1
    else:
        tiff_root = path.joinpath("channel_1_tiffs")
        channel_index = 0

    # Creates the directory if it doesn't exist
    ensure_directory_exists(tiff_root)

    # Formats the output filename to include batch number and the resolved channel index
    file_name = f"file_{str(batch_number).zfill(9)}_channel_{channel_index}.tiff"

    # Combines the directory path with the file name and returns the resultant file path to caller
    return str(tiff_root.joinpath(file_name))


def save_tiff(frames: NDArray[np.int16], file_path: str) -> None:
    """Saves the input frame stack array as the specified tiff file.

    If the file already exists, overwrites the file data with the input frame stack data.

    Args:
        frames: The frames to save.
        file_path: The absolute path to the output .tiff file where to save the frame data.
    """
    with TiffWriter(file_path) as tiff:
        # Rounds all frame values down to the nearest integer and casts to int16 before saving the data.
        for frame in np.floor(frames).astype(np.int16):
            tiff.write(frame, contiguous=True)


def _open_tiff(file_path: Path) -> tuple[TiffFile, int]:
    """Returns the TiffFile instance wrapping the specified .tiff file and the number of pages inside the wrapped
    .tiff file.

    This function is a prerequisite for reading the data stored inside the specified .tiff file. It does not load the
    data into memory.

    Args:
        file_path: The absolute path to the .tiff file from which to read the frame data.
    """
    tiff = TiffFile(file_path)
    tiff_length = len(tiff.pages)  # .pages returns TiffPages iterable, len() makes it an integer size value.
    return tiff, tiff_length


# noinspection PyTypeHints
def _read_tiff(tiff: TiffFile, start_index: int, batch_size: int) -> NDArray[np.int16] | None:
    """Reads a batch (subset) of frames stored inside the .tiff file wrapped by the input TiffFile instance.

    This function loads the requested subset of data into memory.

    Args:
        tiff: The TiffFile instance that wraps the .tiff file from which to read the data.
        start_index: Index of the first frame to read.
        batch_size: Maximum number of frames to read in this batch.

    Returns:
        Number of frames, height, and width stored as a 3D NumPy array, or None if there are no frames to be read.
    """
    # Queries the frame number from the input TiffFile instance
    tiff_length = len(tiff.pages)

    # If the start index is outside the available frame range, returns None.
    if start_index >= tiff_length:
        return None

    # Uses the input batch size and the available number of frames to the right of the start index to determine how
    # many frames to read from the file.
    frames_to_read = min(tiff_length - start_index, batch_size)  # Caps at batch_size as the maximum number to read

    # Reads the requested number of frames
    frames = tiff.asarray() if tiff_length == 1 else tiff.asarray(key=range(start_index, start_index + frames_to_read))

    # Since single-frame tiffs only have two-dimensions, but the rest of the codebase is designed to work with
    # 3-dimensional data (x, y, and frames), adds an extra dimension to 2-dimensional arrays.
    if len(frames.shape) < _MULTIDIMENSIONAL_PROCESSING_THRESHOLD:
        frames = np.expand_dims(frames, axis=0)

    # Converts the frame pixel format to int16 type, rescaling the pixel intensity values where possible.
    if frames.dtype.type in {np.uint16, np.int32}:
        frames = (frames // 2).astype(np.int16)
    elif frames.dtype.type != np.int16:
        frames = frames.astype(np.int16)

    # While this should not be possible, ensures that the returned frame number matches the requested number by
    # truncating any extra frames from the array before returning it to the caller.
    if frames.shape[0] > frames_to_read:
        frames = frames[:frames_to_read, :, :]

    return frames


def tiff_to_binary(runtime_data: RuntimeData) -> RuntimeData:
    """Reads the input data stored as .tif and .tiff files and converts them to the suite2p plane binary (.bin)
    file(s).

    Args:
        runtime_data: A RuntimeData instance that stores the suite2p single-day configuration and runtime parameters.

    Returns:
        The RuntimeData of the first available plane to be processed augmented with additional descriptive parameters
        for the processed data. Specifically, the "height", "width", "nframes", "mean_image" and "mean_image_channel_2"
        fields in the RuntimeData are populated.
    """
    # Instantiates and resets the run timer.
    timer = PrecisionTimer("s")
    timer.reset()

    # Uses the input runtime_data to generate plane-specific RuntimeData instances and initialize files.
    runtime_yaml_paths = initialize_plane_parameters(runtime_data=runtime_data)

    # Load the plane-specific RuntimeData files.
    plane_runtime_data_list = [RuntimeData.from_yaml(file_path=yaml_path) for yaml_path in runtime_yaml_paths]

    # Queries the number of planes and channels from the first plane's configuration.
    plane_number = plane_runtime_data_list[0].configuration.main.nplanes
    channel_number = plane_runtime_data_list[0].configuration.main.nchannels

    # Generates and opens the binary files for each plane for writing. If configured, looks for .tiff and .tif files in
    # multiple data folders.
    plane_runtime_data_list, files, channel_1_binary_files, channel_2_binary_files = find_files_open_binaries(
        plane_runtime_data_list=plane_runtime_data_list
    )

    # Queries the batch_size (how many frames to store in memory at the same time) and adjusts it to account for the
    # total number of planes and channels.
    batch_size = plane_runtime_data_list[0].configuration.registration.batch_size
    batch_size = plane_number * channel_number * math.ceil(batch_size / (plane_number * channel_number))

    # Determines the number of frames across all .tiff files. This is used for the progress bar visualization.
    total_frames = 0
    for file in files:
        _, tiff_length = _open_tiff(file)
        total_frames += tiff_length

    # Creates the progress bar
    progress_bars_enabled = plane_runtime_data_list[0].configuration.main.progress_bars
    pbar = tqdm(
        total=total_frames, desc="Converting frames to binary", unit="frames", disable=not progress_bars_enabled
    )

    # Loops over all discovered .tiff and .tif files.
    current_plane_offset = 0  # Tracks which plane is currently being processed
    for file_index, file in enumerate(files):
        # Opens each target file for reading
        tiff, tiff_length = _open_tiff(file)

        # Loops until all frames from the target file are processed.
        start_index = 0
        while True:
            # Reads up to the batch_size of frames from the processed .tiff file.
            frames = _read_tiff(tiff=tiff, start_index=start_index, batch_size=batch_size)

            # If there are no more frames to read, advances to the next file or ends the runtime.
            if frames is None:
                break

            # Initializes mean_image arrays while processing the first frame batch (as soon as the processed frame
            # dimensions are known).
            if file_index == 0 and start_index == 0:
                for plane_runtime_data in plane_runtime_data_list:
                    plane_io_data = plane_runtime_data.data.file_io
                    plane_io_data.mean_image = np.zeros((frames.shape[1], frames.shape[2]), np.float32)
                    plane_io_data.height = frames.shape[1]
                    plane_io_data.width = frames.shape[2]

                    # For 2-channel data, also initializes the mean image placeholder array for the second channel.
                    if channel_number > 1:
                        plane_io_data.mean_image_channel_2 = np.zeros((frames.shape[1], frames.shape[2]), np.float32)

            # Determines the number of frames read from the processed .tiff file.
            nframes = frames.shape[0]

            # Updates progress bar with the number of frames processed in this batch.
            pbar.update(nframes)

            # Resolves the index of the functional channel (the channel that stores signal data).
            functional_channel_index = (
                plane_runtime_data_list[0].configuration.main.functional_chan - 1 if channel_number > 1 else 0
            )

            # Loops over all available planes and iteratively writes the frames for each plane into the plane-specific
            # binary file(s).
            for plane_index in range(plane_number):
                # Gets the RuntimeData for this specific plane.
                plane_runtime_data = plane_runtime_data_list[plane_index]
                plane_io_data = plane_runtime_data.data.file_io

                # Calculates the starting frame index for this plane (assuming that frames for each plane are stacked
                # in the same .tiff file).
                plane_start_in_batch = (current_plane_offset + plane_index) % plane_number

                # Suite2p assumes the frames are stacked in the order of: channels, planes, time. Generates the set of
                # the functional channel frame indices for the current plane using the total number of planes and
                # channels as iteration offsets, and the known plane-specific starting frame index.
                frame_indices = range(
                    plane_start_in_batch * plane_number + functional_channel_index,
                    nframes,
                    plane_number * channel_number,
                )

                # If there are frames to be added to the current plane's binary file, writes the frames to that binary
                # file.
                if frame_indices:
                    # Extracts the set of frames to write to the current plane's binary file.
                    frames_to_write = frames[frame_indices]

                    # Converts all frames to bytes and writes (appends) them to the (functional) channel 1 memory-mapped
                    # binary file.
                    channel_1_binary_files[plane_index].write(frames_to_write.tobytes())

                    # Appends the data from all processed frames to the data arrays in the plane-specific RuntimeData instance.
                    plane_io_data.mean_image += frames_to_write.sum(axis=0, dtype=np.float32)
                    plane_io_data.nframes += frames_to_write.shape[0]

                    # If processed data uses two functional channels, repeats the steps above for the second
                    # functional channel.
                    if channel_number > 1:
                        # Generates indices for channel 2 frames of the processed plane.
                        second_channel_indices = range(
                            plane_start_in_batch * channel_number + (1 - functional_channel_index),
                            nframes,
                            plane_number * channel_number,
                        )

                        # Writes the frames to the channel 2 binary file and mean image array.
                        if second_channel_indices:
                            channel_2_frames_to_write = frames[second_channel_indices]
                            channel_2_binary_files[plane_index].write(channel_2_frames_to_write.tobytes())
                            plane_io_data.mean_image_channel_2 += channel_2_frames_to_write.sum(
                                axis=0, dtype=np.float32
                            )

            # Updates plane offset for the next batch of frames.
            frames_per_plane_channel = nframes // (plane_number * channel_number)
            current_plane_offset = (current_plane_offset + frames_per_plane_channel) % plane_number
            start_index += nframes

        # Releases all resources before processing the next file.
        gc.collect()

    # Closes the progress bar when binary conversion is over.
    pbar.close()

    # Loops over each plane-specific RuntimeData instance and adds descriptive information about the data to be processed
    # (frames).
    for plane_runtime_data in plane_runtime_data_list:
        plane_io_data = plane_runtime_data.data.file_io

        plane_io_data.height_range = np.array([0, plane_io_data.height], dtype=np.uint32)
        plane_io_data.width_range = np.array([0, plane_io_data.width], dtype=np.uint32)

        # Normalizes the mean images by the number of frames.
        plane_io_data.mean_image /= plane_io_data.nframes

        if channel_number > 1:
            plane_io_data.mean_image_channel_2 /= plane_io_data.nframes

        # Saves each plane's RuntimeData to the .yaml file in its directory.
        plane_runtime_data.save()

    # Closes all memory-mapped binary files.
    for plane_index in range(plane_number):
        channel_1_binary_files[plane_index].close()

        if channel_number > 1:
            channel_2_binary_files[plane_index].close()

    # Returns the first (and, potentially, only) plane's RuntimeData instance to caller.
    return plane_runtime_data_list[0]


def mesoscan_to_binary(runtime_data: RuntimeData) -> RuntimeData:
    """Reads the input mesoscope data stored as .tif and .tiff files and converts them to the suite2p plane binary
    (.bin) file(s).

    Args:
        runtime_data: A RuntimeData instance that stores the suite2p single-day configuration and runtime parameters.

    Returns:
        The RuntimeData of the first available plane to be processed augmented with additional descriptive parameters
        for the processed data. Specifically, the "height", "width", "nframes", "mean_image" and "mean_image_channel_2"
        fields in the RuntimeData are populated.
    """
    # Instantiates and resets the run timer.
    timer = PrecisionTimer("s")
    timer.reset()

    # If "lines" are not already provided in ops, loads parameters from the ops.json file expected to be stored inside
    # the data directory. Note, since sl-suite2p version 2.0.0, ops.json processing now happens as part of resolving the
    # 'ops' dictionary (high-level API), so this is mostly kept as a fall-back safety mechanism.
    if runtime_data.data.file_io.lines is None:
        file_path = Path(runtime_data.configuration.file_io.data_path[0])
        files = list(file_path.glob("*ops.json"))
        with files[0].open() as f:
            ops_json = json.load(f)

        # Stores the 'lines' field inside the main 'ops' dictionary.
        runtime_data.data.file_io.lines = ops_json["lines"]

        # If the number of ROIs is specified inside ops.json, directly uses the parameters from the ops.json file.
        if "nrois" in ops_json:
            runtime_data.data.file_io.nrois = ops_json["nrois"]
            runtime_data.configuration.main.nplanes = ops_json["nplanes"]
            runtime_data.data.file_io.dy = ops_json["dy"]
            runtime_data.data.file_io.dx = ops_json["dx"]
            runtime_data.configuration.main.fs = ops_json["fs"]

        # If the number of ROIs isn't specified but the lines are, defaults to using the number of planes as the number
        # of ROIs.
        elif "nplanes" in ops_json and "lines" in ops_json:
            runtime_data.data.file_io.nrois = ops_json["nplanes"]
            runtime_data.configuration.main.nplanes = 1

        # If ops.json does not specify the number of planes or files, assumes that the data inside the ops.json file is
        # nested by planes, so sets nplanes to the number of top-level keys inside the dictionary loaded from the
        # ops.json file.
        else:
            runtime_data.configuration.main.nplanes = len(ops_json)

    # If "lines" already exists, sets the number of ROIs to the number of sub-lists stored inside the 'lines' list.
    # This assumes that the lines for each ROI are stored as separate lists under the main 'lines' list.
    else:
        runtime_data.data.file_io.nrois = len(runtime_data.data.file_io.lines)

    # Extracts the total number of planes inside the input data to reduce the code complexity below.
    plane_number = runtime_data.configuration.main.nplanes
    nrois = runtime_data.data.file_io.nrois

    message = (
        f"Converting input mesoscope data from nested structure with {plane_number} planes and {nrois} ROIs to "
        f"a flattened structure with {nrois * plane_number} ROI x plane combinations. Each combination is now "
        f"treated as a separate plane."
    )
    # noinspection PyTypeChecker
    console.echo(message=message, level=LogLevel.INFO)

    # Copies original parameters to avoid modifying the original runtime_data by reference.
    lines = runtime_data.data.file_io.lines.copy()
    y_coordinates = runtime_data.data.file_io.dy.copy()
    x_coordinates = runtime_data.data.file_io.dx.copy()

    # Pre-initializes lists to hold the data for all available ROIs and planes.
    runtime_data.data.file_io.lines = [None] * plane_number * nrois
    runtime_data.data.file_io.dy = [None] * plane_number * nrois
    runtime_data.data.file_io.dx = [None] * plane_number * nrois
    runtime_data.data.file_io.plane_index = np.zeros((plane_number * nrois,), np.int32)

    # Re-arranges the data to represent all ROI * plane combinations (de-nests planes from ROIs).
    for roi_index in range(nrois):
        runtime_data.data.file_io.lines[roi_index::nrois] = [lines[roi_index]] * plane_number
        runtime_data.data.file_io.dy[roi_index::nrois] = [y_coordinates[roi_index]] * plane_number
        runtime_data.data.file_io.dx[roi_index::nrois] = [x_coordinates[roi_index]] * plane_number
        runtime_data.data.file_io.plane_index[roi_index::nrois] = np.arange(0, plane_number, 1, int)

    # Updates the 'nplanes' to treat each unique ROI x plane combination as a unique plane. This makes mesoscope data
    # behave like regular 2-photon data.
    runtime_data.configuration.main.nplanes = plane_number * nrois

    # Creates plane-specific RuntimeData instances and saves them.
    plane_runtime_data_list = initialize_plane_parameters(runtime_data=runtime_data)

    # Generates and opens the binary files for each plane for writing.
    plane_runtime_data_list, files, channel_1_binary_files, channel_2_binary_files = find_files_open_binaries(
        plane_runtime_data_list=plane_runtime_data_list
    )

    # Queries the number of channels and the batch_size (how many frames to store in memory at the same time) from the
    # first (and, potentially, only) available plane-specific RuntimeData instance, assuming these configuration values
    # will be consistent across the other planes.
    config = plane_runtime_data_list[0].configuration
    channel_number = config.main.nchannels
    batch_size = config.registration.batch_size
    nplanes = config.main.nplanes

    # Determines the number of frames across all .tiff files. This is used for the progress bar visualization.
    total_frames = 0
    for file in files:
        tiff, tiff_length = _open_tiff(file)
        total_frames += tiff_length

    # Creates the progress bar.
    progress_bars_enabled = config.main.progress_bars
    pbar = tqdm(
        total=total_frames,
        desc="Converting mesoscope frames to binary",
        unit="frames",
        disable=not progress_bars_enabled,
    )

    # Loops over all discovered .tiff and .tif files.
    current_plane_offset = 0  # Tracks which plane is currently being processed.
    for file_index, file in enumerate(files):
        # Opens each target file for reading.
        tiff, tiff_length = _open_tiff(file)

        # Loops until all frames from the target file are processed.
        start_index = 0  # Determines the index from which to start reading the frames.
        while True:
            # Reads up to the batch_size of frames from the processed .tiff file.
            frames = _read_tiff(tiff=tiff, start_index=start_index, batch_size=batch_size)

            # If there are no more frames to read, advances to the next file or ends the runtime.
            if frames is None:
                break

            # Determines the number of frames read from the processed .tiff file.
            nframes = frames.shape[0]

            # Updates progress bar with the number of frames processed in this batch.
            pbar.update(nframes)

            # Resolves the index of the functional channel (the channel that stores signal data).
            functional_channel_index = config.main.functional_chan - 1 if channel_number > 1 else 0

            # Loops over all available ROI x plane combinations and iteratively writes frames for each.
            for roi_plane_index in range(nplanes):
                # Get the RuntimeData for this specific ROI x plane combination.
                plane_runtime_data = plane_runtime_data_list[roi_plane_index]
                plane_io_data = plane_runtime_data.data.file_io

                # Queries the set of lines used for the current ROI-plane.
                roi_plane_lines = np.array(plane_io_data.lines).astype(np.int32)

                # Retrieves the plane index (extract scalar from array if needed).
                if isinstance(plane_io_data.plane_index, np.ndarray):
                    plane_index = int(plane_io_data.plane_index[0])
                else:
                    plane_index = int(plane_io_data.plane_index)

                # Initialize mean_image arrays on first batch (when dimensions are known).
                if file_index == 0 and start_index == 0:
                    plane_io_data.mean_image = np.zeros((len(roi_plane_lines), frames.shape[2]), np.float32)
                    plane_io_data.height = len(roi_plane_lines)
                    plane_io_data.width = frames.shape[2]

                    if channel_number > 1:
                        plane_io_data.mean_image_channel_2 = np.zeros(
                            (len(roi_plane_lines), frames.shape[2]), np.float32
                        )

                    plane_io_data.nframes = 0

                # Calculates the starting frame index for this plane (assuming frames for each plane are stacked
                # in the same .tiff file).
                plane_start_in_batch = (current_plane_offset + plane_index) % plane_number

                # Suite2p assumes frames are stacked in the order: channels, planes, time. Generates the set of
                # functional channel frame indices for the current plane.
                frame_indices = range(
                    plane_start_in_batch * plane_number + functional_channel_index,
                    nframes,
                    plane_number * channel_number,
                )

                # If there are frames to be added to the current plane's binary file, writes the frames.
                if frame_indices:
                    # Extracts the set of frames to write, slicing by the ROI lines.
                    frames_to_write = frames[frame_indices, roi_plane_lines[0] : (roi_plane_lines[-1] + 1), :]

                    # Converts all frames to bytes and writes (appends) them to the channel 1 memory-mapped binary file.
                    channel_1_binary_files[roi_plane_index].write(frames_to_write.tobytes())

                    # Appends the data from all processed frames to the plane-specific RuntimeData.
                    plane_io_data.mean_image += frames_to_write.astype(np.float32).sum(axis=0)
                    plane_io_data.nframes += frames_to_write.shape[0]

                    # If processed data uses two functional channels, repeats the steps above for the second channel.
                    if channel_number > 1:
                        # Generates indices for channel 2 frames of the processed plane.
                        second_channel_indices = range(
                            plane_start_in_batch * channel_number + (1 - functional_channel_index),
                            nframes,
                            plane_number * channel_number,
                        )

                        # Writes the frames to the channel 2 binary file and mean image array.
                        if second_channel_indices:
                            channel_2_frames_to_write = frames[
                                second_channel_indices, roi_plane_lines[0] : (roi_plane_lines[-1] + 1), :
                            ]
                            channel_2_binary_files[roi_plane_index].write(channel_2_frames_to_write.tobytes())
                            plane_io_data.mean_image_channel_2 += channel_2_frames_to_write.astype(np.float32).sum(
                                axis=0
                            )

            # Updates plane offset for the next batch of frames.
            frames_per_plane_channel = nframes // (plane_number * channel_number)
            current_plane_offset = (current_plane_offset + frames_per_plane_channel) % plane_number
            start_index += nframes

        # Releases all resources before processing the next file.
        gc.collect()

    # Closes the progress bar when binary conversion is over.
    pbar.close()

    # Determines whether the current runtime is configured to perform motion registration.
    do_registration = config.registration.do_registration

    # Loops over each plane's RuntimeData and finalizes the mean images and metadata.
    for plane_runtime_data in plane_runtime_data_list:
        plane_io_data = plane_runtime_data.data.file_io

        # If registration is disabled, sets the pixel ranges to span the full height and width of the frame. Pixels on
        # the edges of each frame are excluded during registration as they are typically unstable and should be
        # discarded anyway.
        if not do_registration:
            plane_io_data.height_range = np.array([0, plane_io_data.height], dtype=np.uint32)
            plane_io_data.width_range = np.array([0, plane_io_data.width], dtype=np.uint32)

        # Normalizes mean images by the number of frames.
        plane_io_data.mean_image /= plane_io_data.nframes

        if channel_number > 1:
            plane_io_data.mean_image_channel_2 /= plane_io_data.nframes

        # Save each plane's RuntimeData to its YAML file.
        plane_runtime_data.save()

    # Closes all memory-mapped binary files.
    for roi_plane_index in range(nplanes):
        channel_1_binary_files[roi_plane_index].close()

        if channel_number > 1:
            channel_2_binary_files[roi_plane_index].close()

    # Returns the first (and, potentially, only) plane's RuntimeData instance to caller.
    return plane_runtime_data_list[0]
