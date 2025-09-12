"""This module provides tools for importing data stored inside .tif or .tiff files."""

import gc
import json
import math
from typing import Any
from pathlib import Path

from tqdm import tqdm
import numpy as np
from tifffile import TiffFile, TiffWriter
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists

from .utils import initialize_plane_ops, find_files_open_binaries

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


def save_tiff(frames: NDArray[Any], file_path: str) -> None:
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


def tiff_to_binary(ops: dict[str, Any]) -> dict[str, Any]:
    """Reads the input data stored as .tif and .tiff files and converts them to the suite2p plane binary (.bin)
    file(s).

    Args:
        ops: The dictionary that stores the suite2p processing parameters.

    Returns:
        The 'ops' dictionary of the first available plane to be processed augmented with additional descriptive
        parameters for the processed data. Specifically, the dictionary includes the following additional keys:
        "Ly", "Lx", "first_tiffs", "frames_per_folder", "nframes", "meanImg", "meanImg_chan2".
    """
    # Instantiates and resets the run timer
    timer = PrecisionTimer("s")
    timer.reset()

    # Uses the input 'ops' dictionary to generate a list of plane-specific 'ops' dictionaries. Converts the output
    # list to tuple for efficiency
    plane_ops: tuple[dict[str, Any], ...] = tuple(initialize_plane_ops(ops=ops))

    # Queries the number of planes and channels from the first (and, potentially, only) available plane-specific
    # 'ops' dictionary
    plane_number = plane_ops[0]["nplanes"]
    channel_number = plane_ops[0]["nchannels"]

    # Generates and opens the binary files for each plane for writing. If configured, looks for .tiff and .tif files in
    # multiple data folders.
    plane_ops, files, channel_1_binary_file, channel_2_binary_file = find_files_open_binaries(plane_ops=plane_ops)
    ops = plane_ops[0]  # Queries the first (and, potentially, only) plane's 'ops' dictionary for further processing.

    # Queries the batch_size (how many frames to store in memory at the same time) and adjusts it to account for the
    # total number of planes and channels.
    batch_size = ops["batch_size"]
    batch_size = plane_number * channel_number * math.ceil(batch_size / (plane_number * channel_number))

    # Pre-initializes all plane ops to avoid conditional initialization below
    for plane_index in range(plane_number):
        plane_ops[plane_index]["nframes"] = 0
        plane_ops[plane_index]["frames_per_file"] = np.zeros((len(files),), dtype=int)
        plane_ops[plane_index]["frames_per_folder"] = np.zeros((len(ops["data_path"]),), dtype=int)

    # Determines the number of frames across all .tiff files. This is used for the progress bar visualization
    total_frames = 0
    for file in files:
        _, tiff_length = _open_tiff(file)
        total_frames += tiff_length

    # Creates the progress bar.
    pbar = tqdm(total=total_frames, desc="Converting frames to binary", unit="frames", disable=not ops["progress_bars"])

    # Loops over all discovered .tiff and .tif files.
    folder_index = -1
    current_plane_offset = 0  # Tracks which plane is currently being processed
    for file_index, file in enumerate(files):
        # Opens each target file for reading
        tiff, tiff_length = _open_tiff(file)

        # Resets plane offset at the start of each new folder and increments the processed folder index
        if ops["first_tiffs"][file_index]:
            folder_index += 1  # Increments the folder index
            current_plane_offset = 0  # Resets the plane number to 0

        # Loops until all frames from the target file are processed.
        start_index = 0  # Determines the index from which to start reading the frames
        while True:
            # Reads up to the batch_size of frames from the processed .tiff file.
            frames = _read_tiff(tiff=tiff, start_index=start_index, batch_size=batch_size)

            # If there are no more frames to read, advances to the next file or ends the runtime
            if frames is None:
                break

            # Initializes meanImg arrays while processing the first frame batch (as soon as the processed frame
            # dimensions are known)
            if file_index == 0 and start_index == 0:
                for plane_index in range(plane_number):
                    plane_ops[plane_index]["meanImg"] = np.zeros((frames.shape[1], frames.shape[2]), np.float32)

                    # For 2-channel data, also initializes the mean image placeholder array for the second channel.
                    if channel_number > 1:
                        plane_ops[plane_index]["meanImg_chan2"] = np.zeros(
                            (frames.shape[1], frames.shape[2]), np.float32
                        )

            nframes = frames.shape[0]  # Determines the number of frames read from the processed .tiff file.

            # Updates progress bar with the number of frames processed in this batch.
            pbar.update(nframes)

            # Resolves the index of the functional channel (the channel that stores signal data).
            functional_channel_index = ops["functional_chan"] - 1 if channel_number > 1 else 0

            # Loops over all available planes and iteratively writes the frames for each plane into the plane-specific
            # binary file(s).
            for plane_index in range(plane_number):
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
                    channel_1_binary_file[plane_index].write(frames_to_write.tobytes())

                    # Appends the data from all processed frames to the data arrays in the plane-specific 'ops'
                    # dictionary, as this data is used during further processing.
                    plane_ops[plane_index]["meanImg"] += frames_to_write.sum(axis=0, dtype=np.float32)
                    plane_ops[plane_index]["nframes"] += frames_to_write.shape[0]
                    plane_ops[plane_index]["frames_per_file"][file_index] += frames_to_write.shape[0]
                    plane_ops[plane_index]["frames_per_folder"][folder_index] += frames_to_write.shape[0]

                    # If processed data uses two functional channels, repeats the steps above for the second
                    # functional channel
                    if channel_number > 1:
                        # Generates indices for channel 2 frames of the processed plane.
                        second_channel_indices = range(
                            plane_start_in_batch * channel_number + (1 - functional_channel_index),
                            nframes,
                            plane_number * channel_number,
                        )

                        # Writes the frames to the channel 2 binary file and mean image 'ops' array.
                        if second_channel_indices:
                            channel_2_frames_to_write = frames[second_channel_indices]
                            channel_2_binary_file[plane_index].write(channel_2_frames_to_write.tobytes())
                            plane_ops[plane_index]["meanImg_chan2"] += channel_2_frames_to_write.mean(axis=0)

            # Updates plane offset for the next batch of frames
            frames_per_plane_channel = nframes // (plane_number * channel_number)
            current_plane_offset = (current_plane_offset + frames_per_plane_channel) % plane_number
            start_index += nframes

        # Releases all resources before processing the next file.
        gc.collect()

    # Closes the progress bar when binary conversion is over
    pbar.close()

    # Loops over each plane-specific 'ops' dictionary and adds descriptive information about the data to be processed
    # (frames).
    for ops in plane_ops:
        ops["Ly"], ops["Lx"] = ops["meanImg"].shape
        ops["yrange"] = np.array([0, ops["Ly"]])
        ops["xrange"] = np.array([0, ops["Lx"]])
        ops["meanImg"] /= ops["nframes"]
        if channel_number > 1:
            ops["meanImg_chan2"] /= ops["nframes"]

        # Caches each 'ops' dictionary to disk as an ops.npy file. The file is cached into the plane-specific processing
        # subdirectory.
        np.save(ops["ops_path"], ops)

    # Closes all memory-mapped binary files
    for plane_index in range(plane_number):
        channel_1_binary_file[plane_index].close()

        if channel_number > 1:
            channel_2_binary_file[plane_index].close()

    # Returns the first (and, potentially, only) plane's 'ops' dictionary to caller.
    return plane_ops[0]


def mesoscan_to_binary(ops: dict[str, Any]) -> dict[str, Any]:
    """Reads the input mesoscope data stored as .tif and .tiff files and converts them to the suite2p plane binary
    (.bin) file(s).

    Args:
        ops: The dictionary that stores the suite2p processing parameters.

    Returns:
        The 'ops' dictionary of the first available plane to be processed augmented with additional descriptive
        parameters for the processed data. Specifically, the dictionary includes the following additional keys:
        "Ly", "Lx", "first_tiffs", "frames_per_folder", "nframes", "meanImg", "meanImg_chan2".
    """
    # Instantiates and resets the run timer
    timer = PrecisionTimer("s")
    timer.reset()

    # If "lines" are not already provided in ops, loads parameters from the ops.json file expected to be stored inside
    # the data directory. Note, since sl-suite2p version 2.0.0, ops.json processing now happens as part of resolving the
    # 'ops' dictionary (high-level API), so this is mostly kept as a fall-back safety mechanism.
    if "lines" not in ops:
        file_path = Path(ops["data_path"][0])
        files = list(file_path.glob("*.json"))
        with files[0].open() as f:
            ops_json = json.load(f)

        # Stores the 'lines' field inside the main 'ops' dictionary.
        ops["lines"] = ops_json["lines"]

        # If the number of ROIs is specified inside ops.json, directly uses the parameters from the ops.json file.
        if "nrois" in ops_json:
            ops["nrois"] = ops_json["nrois"]
            ops["nplanes"] = ops_json["nplanes"]
            ops["dy"] = ops_json["dy"]
            ops["dx"] = ops_json["dx"]
            ops["fs"] = ops_json["fs"]

        # If the number of ROIs isn't specified but the lines are, defaults to using the number of planes as the number
        # of ROIs.
        elif "nplanes" in ops_json and "lines" in ops_json:
            ops["nrois"] = ops_json["nplanes"]
            ops["nplanes"] = 1

        # If ops.json does not specify the number of planes or files, assumes that the data inside the ops.json file is
        # nested by planes, so sets nplanes to the number of top-level keys inside the dictionary loaded from the
        # ops.json file.
        else:
            ops["nplanes"] = len(ops_json)

    # If "lines" already exists, sets the number of ROIs to the number of sub-lists stored inside the 'lines' list.
    # This assumes that the lines for each ROI are stored as separate lists under the main 'lines' list.
    else:
        ops["nrois"] = len(ops["lines"])

    # Extracts the total number of planes inside the input data to reduce the code complexity below.
    plane_number = ops["nplanes"]

    message = (
        f"Converting input mesoscope data from nested structure with {plane_number} planes and {ops['nrois']} ROIs to "
        f"a flattened structure with {ops['nrois'] * plane_number} ROI x plane combinations. Each combination is now "
        f"treated as a separate plane."
    )
    # noinspection PyTypeChecker
    console.echo(message=message, level=LogLevel.INFO)

    # Copies original parameters to avoid modifying the original 'ops' dictionary by reference.
    lines = ops["lines"].copy()
    y_coordinates = ops["dy"].copy()
    x_coordinates = ops["dx"].copy()

    # Pre-initializes lists to hold the data for all available ROIs and planes.
    ops["lines"] = [None] * plane_number * ops["nrois"]
    ops["dy"] = [None] * plane_number * ops["nrois"]
    ops["dx"] = [None] * plane_number * ops["nrois"]
    ops["iplane"] = np.zeros((plane_number * ops["nrois"],), np.int32)

    # Re-arranges the data to represent all ROI * plane combinations (de-nests planes from ROIs).
    for roi_index in range(ops["nrois"]):
        ops["lines"][roi_index :: ops["nrois"]] = [lines[roi_index]] * plane_number
        ops["dy"][roi_index :: ops["nrois"]] = [y_coordinates[roi_index]] * plane_number
        ops["dx"][roi_index :: ops["nrois"]] = [x_coordinates[roi_index]] * plane_number
        ops["iplane"][roi_index :: ops["nrois"]] = np.arange(0, plane_number, 1, int)

    # Updates the 'nplanes' to treat each unique ROI x plane combination as a unique plane. This makes mesoscope data
    # behave like regular 2-photon data.
    ops["nplanes"] = plane_number * ops["nrois"]

    # Uses the input 'ops' dictionary to generate the list of plane-specific 'ops' dictionaries. Converts the output
    # list to tuple for efficiency
    plane_ops: tuple[dict[str, Any], ...] = tuple(initialize_plane_ops(ops=ops))

    # Generates and opens the binary files for each plane for writing. If configured, looks for .tiff and .tif files in
    # multiple data folders.
    plane_ops, files, channel_1_binary_file, channel_2_binary_file = find_files_open_binaries(plane_ops=plane_ops)
    ops = plane_ops[0]  # Queries the first (and, potentially, only) plane's 'ops' dictionary for further processing.

    # Queries the number of channels and the batch_size (how many frames to store in memory at the same time) from the
    # first (and, potentially, only) available plane-specific 'ops' dictionary
    channel_number = ops["nchannels"]
    batch_size = ops["batch_size"]

    # Determines the number of frames across all .tiff files. This is used for the progress bar visualization
    total_frames = 0
    for file in files:
        tiff, tiff_length = _open_tiff(file)
        total_frames += tiff_length

    # Creates the progress bar.
    pbar = tqdm(
        total=total_frames,
        desc="Converting mesoscope frames to binary",
        unit="frames",
        disable=not ops["progress_bars"],
    )

    # Loops over all discovered .tiff and .tif files.
    folder_index = -1
    current_plane_offset = 0  # Tracks which plane is currently being processed
    for file_index, file in enumerate(files):
        # Opens each target file for reading
        tiff, tiff_length = _open_tiff(file)

        # Resets plane offsets at the start of each new folder and increments the processed folder index.
        if ops["first_tiffs"][file_index]:
            folder_index += 1  # Increments the folder index
            current_plane_offset = 0  # Resets the plane number to 0

        # Loops until all frames from the target file are processed.
        start_index = 0  # Determines the index from which to start reading the frames
        while True:
            # Reads up to the batch_size of frames from the processed .tiff file.
            frames = _read_tiff(tiff=tiff, start_index=start_index, batch_size=batch_size)

            # If there are no more frames to read, advances to the next file or ends the runtime
            if frames is None:
                break

            # Determines the number of frames read from the processed .tiff file.
            nframes = frames.shape[0]

            # Updates progress bar with the number of frames processed in this batch.
            pbar.update(nframes)

            # Resolves the index of the functional channel (the channel that stores signal data).
            functional_channel_index = ops["functional_chan"] - 1 if channel_number > 1 else 0

            # Loops over all available planes and iteratively writes the frames for each plane into the plane-specific
            # binary file(s).
            for roi_plane_index in range(ops["nplanes"]):
                # Queries the set of lines used for the current ROI-plane.
                roi_plane_lines = np.array(plane_ops[roi_plane_index]["lines"]).astype(np.int32)

                # Retrieves the plane index.
                plane_index = plane_ops[roi_plane_index]["iplane"]

                if file_index == 0 and start_index == 0:
                    plane_ops[roi_plane_index]["meanImg"] = np.zeros(
                        (len(roi_plane_lines), frames.shape[2]), np.float32
                    )
                    if channel_number > 1:
                        plane_ops[roi_plane_index]["meanImg_chan2"] = np.zeros(
                            (len(roi_plane_lines), frames.shape[2]), np.float32
                        )
                    plane_ops[roi_plane_index]["nframes"] = 0

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
                    frames_to_write = frames[frame_indices, roi_plane_lines[0] : (roi_plane_lines[-1] + 1), :]

                    # Converts all frames to bytes and writes (appends) them to the (functional) channel 1 memory-mapped
                    # binary file.
                    channel_1_binary_file[roi_plane_index].write(frames_to_write.tobytes())

                    # Appends the data from all processed frames to the data arrays in the plane-specific 'ops'
                    # dictionary, as this data is used during further processing.
                    plane_ops[roi_plane_index]["meanImg"] += frames_to_write.astype(np.float32).sum(axis=0)
                    plane_ops[roi_plane_index]["nframes"] += frames_to_write.shape[0]
                    plane_ops[roi_plane_index]["frames_per_folder"][folder_index] += frames_to_write.shape[0]

                    # If processed data uses two functional channels, repeats the steps above for the second
                    # functional channel
                    if channel_number > 1:
                        # Generates indices for channel 2 frames of the processed plane.
                        second_channel_indices = range(
                            plane_start_in_batch * channel_number + (1 - functional_channel_index),
                            nframes,
                            plane_number * channel_number,
                        )

                        # Writes the frames to the channel 2 binary file and mean image 'ops' array.
                        if second_channel_indices:
                            channel_2_frames_to_write = frames[
                                second_channel_indices, roi_plane_lines[0] : roi_plane_lines[-1] + 1, :
                            ]
                            channel_2_binary_file[roi_plane_index].write(channel_2_frames_to_write.tobytes())
                            plane_ops[roi_plane_index]["meanImg_chan2"] += channel_2_frames_to_write.astype(
                                np.float32
                            ).sum(axis=0)

            # Updates plane offset for the next batch of frames
            frames_per_plane_channel = nframes // (plane_number * channel_number)
            current_plane_offset = (current_plane_offset + frames_per_plane_channel) % plane_number
            start_index += nframes

        # Releases all resources before processing the next file.
        gc.collect()

    # Closes the progress bar when binary conversion is over
    pbar.close()

    # Determines whether the current runtime is configured to perform motion registration
    do_registration = ops["do_registration"]

    # Loops over each plane-specific 'ops' dictionary and adds descriptive information about the data to be processed
    # (frames).
    for ops in plane_ops:
        ops["Ly"], ops["Lx"] = ops["meanImg"].shape

        # If registration is disabled, sets the pixel ranges to span the full height and width of the frame. Pixels on
        # the edges of each frame are excluded during registration as they are typically unstable and should be
        # discarded anyway.
        if not do_registration:
            ops["yrange"] = np.array([0, ops["Ly"]])
            ops["xrange"] = np.array([0, ops["Lx"]])

        ops["meanImg"] /= ops["nframes"]
        if channel_number > 1:
            ops["meanImg_chan2"] /= ops["nframes"]

        # Caches each 'ops' dictionary to disk as an ops.npy file. The file is cached into the plane-specific processing
        # subdirectory.
        np.save(ops["ops_path"], ops)

    # Closes all memory-mapped binary files
    for roi_plane_index in range(ops["nplanes"]):
        channel_1_binary_file[roi_plane_index].close()

        if channel_number > 1:
            channel_2_binary_file[roi_plane_index].close()

    # Returns the first (and, potentially, only) plane's 'ops' dictionary to caller.
    return plane_ops[0]
