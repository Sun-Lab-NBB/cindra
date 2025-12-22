"""This module provides utility functions for handling file searching, path management, and binary file operations."""

from typing import Any
from pathlib import Path

import numpy as np
from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists


def _search_files_by_extension(
    root_directory: Path,
    extensions: tuple[str, ...] = ("tif", "tiff"),
    ignore_names: tuple[str, ...] = (),
) -> tuple[list[Path], list[bool]]:
    """Recursively searches the target directory and all subdirectories for files matching the given extensions.

    Args:
        root_directory: The absolute path to the directory where to search the files.
        extensions: The list of file extensions to search for. Note, the file extensions should NOT include the leading
            dot (e.g., 'tif', 'tiff').
        ignore_names: A tuple of file names to ignore while searching. A file name must match the ignored name
            completely for the file to be excluded from the search results.

    Returns:
        A tuple of two elements. The first element is a list of absolute paths to files found in the specified root
        directory and all subdirectories. The second element is a boolean list that indicates which file is found
        first in a directory or subdirectory after sorting all discovered files naturally.

    Raises:
        FileNotFoundError: If no files with the specified extension(s) are found in the root directory or its
        subdirectories.
    """
    # Initializes lists to store the discovered absolute file paths and a matching list to store binary flags for
    # whether each path is the first file in its parent directory.
    file_paths: list[Path] = []
    first_files: list[bool] = []

    # If the specified root directory exists and is a directory, recursively searches it for files matching the
    # provided extensions.
    if root_directory.is_dir():
        # For each extension, recursively searches the entire directory tree for matching files.
        files: list[Path] = []
        for extension in extensions:
            # Uses recursive glob pattern (**/) to search all subdirectories.
            found_files = [file.resolve() for file in root_directory.rglob(f"*.{extension}")]

            # Filters ignored files.
            filtered_files = [file for file in found_files if file.stem not in ignore_names]

            files.extend(filtered_files)

        # If files were found, groups them by parent directory and processes each group.
        if files:
            # Groups files by their parent directory.
            files_by_directory: dict[Path, list[Path]] = {}
            for file in files:
                parent = file.parent
                if parent not in files_by_directory:
                    files_by_directory[parent] = []
                files_by_directory[parent].append(file)

            # Processes each directory's files in sorted order.
            for directory in natsorted(files_by_directory.keys()):
                directory_files = natsorted(files_by_directory[directory])
                file_paths.extend(directory_files)

                # First file in each directory is marked as True.
                first_files.append(True)
                first_files.extend([False] * (len(directory_files) - 1))

    # If no files were found, raises a FileNotFoundError.
    if not file_paths:
        message = (
            f"Could not find any files with specified extensions '{extensions}' inside the target directory: "
            f"{root_directory}."
        )
        console.error(message=message, error=FileNotFoundError)

    # Returns a list storing the absolute paths of the discovered files and a boolean list marking the first files in
    # each directory and subdirectory.
    return file_paths, first_files


def _get_tiff_list(ops: dict[str, Any]) -> tuple[list[Path], dict[str, Any]]:
    """Creates a list of .tif and .tiff files found in the directory specified by the "data_path" field of the input
    'ops' dictionary.

    The function recursively searches the data_path directory and all subdirectories for TIFF files.

    Args:
        ops: The dictionary that stores the suite2p single-day processing parameters.

    Returns:
        A tuple of two elements. The first element is a list of the absolute paths to the found .tif and .tiff files,
        and the second element is the updated 'ops' dictionary.

    Raises:
        FileNotFoundError: If no .tif or .tiff files are found in the directory or its subdirectories.
    """
    # Queries the absolute path to the root data directory (now a single path, not a list).
    data_path = ops["data_path"]

    # Recursively searches for .tif and .tiff files in the data directory and all subdirectories.
    file_paths, first_tiffs = _search_files_by_extension(
        root_directory=data_path,
        extensions=("tif", "tiff", "TIF", "TIFF"),
        ignore_names=tuple(ops.get("ignored_file_names", [])),
    )

    # If no files were found, raises a FileNotFoundError.
    if len(file_paths) == 0:
        message = f"Could not find any TIF/TIFF files to process in {data_path}."
        console.error(message=message, error=FileNotFoundError)

    # Converts 'first_tiffs' into a boolean NumPy array and updates 'ops'.
    ops["first_tiffs"] = np.array(first_tiffs).astype("bool")
    message = f"Found {len(file_paths)} TIF/TIFF files. Converting to binaries."
    console.echo(message=message, level=LogLevel.INFO)

    # Returns the list of absolute paths to the tiff files and the updated 'ops' dictionary.
    return file_paths, ops


def find_files_open_binaries(
    plane_ops: tuple[dict[str, Any], ...],
) -> tuple[tuple[dict[str, Any], ...], list[Path], list[Any], list[Any]]:
    """Finds the source data files for each plane inside the input list of plane-specific 'ops' dictionaries and
    prepares plane-specific binary files for writing the data.

    This service function resolves the paths to the raw data files and generates memory-mapped binary files for each
    plane. The output from this service function is later used to convert raw data files to the suite2p binary file
    format. This function currently only supports .tif and .tiff files.

    Args:
        plane_ops: The list of plane-specific 'ops' dictionaries that store single-day plane processing parameters.

    Returns:
        A tuple of four elements. The first element is the input 'plane_ops' list, where each plane-specific dictionary
        is updated with paths to source data files. The second element is the list of paths to source data files
        for each plane. The third element is the list of opened binaries for channel 1. The fourth element is the list
        of opened binaries for channel 2 if the data uses two functional channels.
    """
    # Initializes lists to store the binary files of each channel, which are eventually returned.
    channel_1_binary_files = []
    channel_2_binary_files = []

    # Pre-types to appease mypy.
    input_format: str | None

    # Loops through each plane's 'ops' dictionary, processes, and opens the binary files.
    for ops in plane_ops:
        # Queries the number of channels from the plane-specific 'ops' dictionary.
        channel_number = ops["nchannels"]

        # Resolves paths to either raw or registered binary files for both channels, depending on the 'ops'
        # configuration.
        if ops.get("keep_movie_raw"):
            # Opens the raw binary file and appends it to 'channel_1_binary_files'.
            channel_1_binary_files.append(Path(ops["raw_file"]).open(mode="wb"))
            # If there is a second channel, opens the raw binary and appends it to 'channel_2_binary_files'.
            if channel_number > 1:
                channel_2_binary_files.append(Path(ops["raw_file_chan2"]).open(mode="wb"))
        else:
            # Opens the registered binary file and appends it to 'channel_1_binary_files'.
            channel_1_binary_files.append(Path(ops["reg_file"]).open(mode="wb"))
            # If there is a second channel, opens the registered binary and appends it to 'channel_2_binary_files'.
            if channel_number > 1:
                channel_2_binary_files.append(Path(ops["reg_file_chan2"]).open(mode="wb"))

    # Determines the input format based on the first plane's 'ops' dictionary.
    input_format = plane_ops[0].get("input_format", "tiff")

    message = f"Input data format: {input_format}."
    console.echo(message=message, level=LogLevel.SUCCESS)

    # NOTE: The following blocks of code involving the .tif and .tiff files were originally a part of an if-else block
    # that supported deprecated input file types. If support for the deprecated file types is re-implemented in the
    # future, refer to the original Suite2p code for the original if-else 'input_format' logic. :)

    # Retrieves the absolute file paths to the .tif and .tiff files.
    file_paths, ops_updated = _get_tiff_list(plane_ops[0])

    # Stores the updated values for the "first_tiffs" and "frames_per_folder" keys in each plane-specific 'ops'
    # dictionary.
    for ops in plane_ops:
        ops["first_tiffs"] = ops_updated["first_tiffs"]
        ops["frames_per_folder"] = np.zeros((ops_updated["first_tiffs"].sum(),), np.int32)

    # Stores the absolute paths to the files under the "filelist" key for each plane-specific 'ops' dictionary.
    for ops in plane_ops:
        ops["filelist"] = file_paths

    # Returns the list of plane-specific 'ops' dictionaries, the absolute paths to the discovered files, and the opened
    # binary files for both channels.
    return plane_ops, file_paths, channel_1_binary_files, channel_2_binary_files


# noinspection PyUnboundLocalVariable
def initialize_plane_ops(ops: dict[str, Any]) -> list[dict[str, Any]]:
    """Constructs plane-specific 'ops' dictionaries for each plane specified inside the input 'ops' dictionary.

    Args:
        ops: The dictionary that stores the suite2p single-day processing parameters.

    Returns:
        The list of plane-specific 'ops' dictionaries with the same length as the number of planes ('nplanes')
        specified inside the input 'ops' dictionary.
    """
    # Initializes the list that will store each plane's 'ops' dictionary, which is eventually returned.
    plane_ops = []

    # Queries the number of planes and channels from the input 'ops' dictionary.
    plane_number = ops["nplanes"]
    channel_number = ops["nchannels"]

    # If the "lines" and "iplane" keys are populated in the input 'ops' dictionary, makes copies of the values to
    # populate the keys in each plane-specific 'ops' dictionary.
    if "lines" in ops:
        lines = ops["lines"]
    if "iplane" in ops:
        iplane = ops["iplane"]

    # For mesoscope ROIs, makes copies of the values stored under the "dy" and "dx" keys.
    if "dy" in ops and ops["dy"] != "":
        dy = ops["dy"]
        dx = ops["dx"]

    # Converts save_path and data_path from string to Path.
    save_path_root = Path(ops["save_path"])
    ops["data_path"] = Path(ops["data_path"])

    # Loops over each of the planes and constructs each plane-specific 'ops' dictionary. If the keys are populated in
    # the input 'ops' dictionary, stores the plane-specific value under the appropriate key in the plane-specific 'ops'
    # dictionary.
    for plane_index in range(plane_number):
        # Resolves the output directory for the plane data. Always uses 'suite2p' as the subdirectory.
        plane_output_path = save_path_root.joinpath("suite2p", f"plane{plane_index}")
        ops["output_path"] = plane_output_path

        # Defines the paths for the ops.npy file and the first channel's registered data binary file.
        ops["ops_path"] = plane_output_path.joinpath("ops.npy")
        ops["reg_file"] = plane_output_path.joinpath("data.bin")

        # If necessary, generates an additional binary file to store raw (unregistered) data after runtime.
        if ops.get("keep_movie_raw"):
            ops["raw_file"] = plane_output_path.joinpath("data_raw.bin")

        # Sets the "lines" and "iplane" values for the current plane.
        if "lines" in ops:
            ops["lines"] = lines[plane_index]
        if "iplane" in ops:
            ops["iplane"] = iplane[plane_index]

        # If the data contains multiple functional channels, configures the binaries for the second channel.
        if channel_number > 1:
            ops["reg_file_chan2"] = plane_output_path.joinpath("data_chan2.bin")
            if ops.get("keep_movie_raw"):
                ops["raw_file_chan2"] = plane_output_path.joinpath("data_chan2_raw.bin")

        # Stores the mesoscope ROI coordinates (top left corner) "dy" and "dx" for the current plane.
        if "dy" in ops and ops["dy"] != "":
            ops["dy"] = dy[plane_index]
            ops["dx"] = dx[plane_index]

        # Creates the output directory if it does not exist.
        ensure_directory_exists(plane_output_path)

        # Copies the modified 'ops' dictionary and appends it to 'plane_ops'.
        plane_ops.append(ops.copy())

    # Returns the list of 'ops' dictionaries for each plane.
    return plane_ops
