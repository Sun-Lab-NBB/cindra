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
    look_one_level_down: bool = False,
) -> tuple[list[Path], list[bool]]:
    """Searches the target directory and subdirectories (one level down) for files matching the given extensions.

    Notes:
        Originally, this worker function was used by multiple higher-level functions to discover specific data files.
        In the current version of sl-suite2p, support for most container types other than .tif / .tiff has been
        deprecated. Therefore, the function is currently only used to discover .tiff files, despite being
        container-agnostic.

    Args:
        root_directory: The absolute path to the directory where to search the files.
        extensions: The list of file extensions to search for. Note, the file extensions should NOT include the leading
            dot (e.g., 'tif', 'tiff').
        ignore_names: A tuple of file names to ignore while searching. A file name must match the ignored name
            completely for the file to be excluded from the search results.
        look_one_level_down: Determines whether to search the subdirectories of the target directory (one level down).

    Returns:
        A tuple of two elements. The first element is a list of absolute paths to files found in the specified root
        directory and (if applicable) its subdirectories. The second element is a boolean list that indicates which
        file is found first in a directory or subdirectory after sorting al discovered files naturally.

    Raises:
        FileNotFoundError: If no files with the specified extension(s) are found in the root directory or its
        subdirectories.
    """
    # Initializes lists to store the discovered absolute file paths and a matching list to store binary flags for
    # whether each path is the first file in its parent directory.
    file_paths = []
    first_files = []

    # If the specified root directory exists and is a directory, searches it for files matching the provided
    # extensions.
    if root_directory.is_dir():
        # For each extension, searches the provided root directory for matching files and retrieves their paths.
        files = []  # Stores discovered files
        for extension in extensions:
            # Gets all files with the matching extension
            found_files = [file.resolve() for file in root_directory.glob(f"*.{extension}")]

            # Filters ignored files
            filtered_files = [file for file in found_files if file.stem not in ignore_names]

            files.extend(filtered_files)

        # If files were found, updates the storage lists with discovered data.
        if files:
            # Adds the absolute paths of the found files to 'file_paths', after sorting them naturally.
            file_paths.extend(natsorted(files))

            # Updates 'first_files' such that there is a corresponding boolean value for each file found in the
            # directory. The first item in 'first_files' is set to True, since it corresponds to the first file in the
            # directory (following natural sorting). All other values are set to False.
            first_files.append(True)
            first_files.extend([False] * (len(files) - 1))

        # If the argument 'look_one_level_down' is set to True, performs the same search one level down in the
        # subdirectories of the provided root directory.
        if look_one_level_down:
            # Retrieves the subdirectories of the provided root directory, which are sorted in natural order.
            subdirectories = natsorted([path for path in root_directory.iterdir() if path.is_dir()])

            # Loops over all discovered subdirectories.
            for directory in subdirectories:
                # For each extension, searches the subdirectory for matching files and retrieves their absolute
                # paths.
                subdirectory_files = []  # Stores the found files
                for extension in extensions:
                    # Gets all files with the matching extension
                    found_files = [file.resolve() for file in directory.glob(f"*.{extension}")]

                    # Filters ignored files
                    filtered_files = [file for file in found_files if file.stem not in ignore_names]

                    subdirectory_files.extend(filtered_files)

                # If files were found, updates the storage lists with subdirectory data, following the same procedure
                # as for the root directory
                if subdirectory_files:
                    file_paths.extend(natsorted(subdirectory_files))
                    first_files.append(True)
                    first_files.extend([False] * (len(subdirectory_files) - 1))

    # If no files were found, raises a FileNotFoundError.
    if not file_paths:
        message = (
            f"Could not find any files with specified extensions '{extensions}' inside the target directory: "
            f"{root_directory}."
        )
        console.error(message=message, error=FileNotFoundError)

    # Returns a list storing the absolute paths of the discovered files and a boolean list marking the first files in
    # each directory and (if applicable) subdirectory.
    return file_paths, first_files


def _get_tiff_list(ops: dict[str, Any]) -> tuple[list[Path], dict[str, Any]]:
    """Creates a list of .tif and .tiff files found in the directory specified by the "data_path" field of the input
    'ops' dictionary.

    If the "subfolders" field is specified in 'ops', this function additionally retrieves files from the specified
    subdirectories. If "look_one_level_down" is set to True in 'ops', this function additionally retrieves files from
    the subdirectories of the root directories specified by the "data_path" field.

    Args:
        ops: The dictionary that stores the suite2p single-day processing parameters.

    Returns:
        A tuple of two elements. The first element is a list of the absolute paths to the found .tif and .tiff files,
        and the second element is the updated 'ops' dictionary.

    Raises:
        FileNotFoundError: If no .tif or .tiff files are found in the directory or (if applicable) its subdirectories."
    """
    # Queries the absolute path(s) to root data directory.
    directories = ops["data_path"]

    # Initializes a list to store the absolute paths to the discovered .tif and .tiff files.
    file_paths: list[Path] = []

    # If a user-specified list of tiff files is provided in the 'ops' dictionary, adds them to the list.
    if "tiff_list" in ops:
        # Loops over all .tif and .tiff files and stores the absolute path to each target file.
        for tiff in ops["tiff_list"]:
            file_paths.append(Path(file_paths[0]).joinpath(tiff))

        # Initializes an array to store which .tif or .tiff file is the first to process.
        ops["first_tiffs"] = np.zeros((len(file_paths),), dtype="bool")

        # Sets the element matching the first .tif or .tiff file as True, indicating it is the first file to process.
        ops["first_tiffs"][0] = True

        message = (
            f"Parsed the paths to {len(file_paths)} TIF/TIFF files using the user-defined 'tiff_list' processing "
            f"parameter. Converting to binaries."
        )
        console.echo(message=message, level=LogLevel.INFO)

    # Otherwise, searches for .tif and .tiff files in directories or subdirectories.
    else:
        # If only one root directory path is provided and searching the subdirectories is allowed, also searches them
        # for the .tif or .tiff file. To do so, extends the list of search directories with the path to subfolders.
        if len(directories) == 1 and "subfolders" in ops and len(ops["subfolders"]) > 0:
            for subdirectory in ops["subfolders"]:
                directory_path = file_paths[0].joinpath(subdirectory)
                directories.append(directory_path)

        # Initializes a list to store the first .tif or .tiff file of each directory.
        first_tiffs: list[bool] = []

        # Loops over all directories and searches for .TIFF files.
        for directory in directories:
            # Retrieves the absolute paths of the .tif and .tiff files and the list of the first .tif or .tiff file(s)
            # in the target directory, optionally searching subdirectories as well, depending on the
            # "look_one_level_down" configuration in 'ops'.
            file_paths_found, first_tiffs_found = _search_files_by_extension(
                root_directory=directory,
                extensions=("tif", "tiff", "TIF", "TIFF"),
                ignore_names=tuple(ops["ignored_file_names"]),
                look_one_level_down=ops["look_one_level_down"],
            )

            # Extends the returned data into storage lists
            file_paths.extend(file_paths_found)
            first_tiffs.extend(first_tiffs_found)

        # If no files were found in the directories, raises a FileNotFoundError.
        if len(file_paths) == 0:
            message = "Could not find any TIF/TIFF files to process."
            console.error(message=message, error=FileNotFoundError)

        # Otherwise, converts 'first_tiffs' into a boolean NumPy array and updates 'ops'.
        else:
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

    # Resolves the "fast_disk" directory or sets it to the same directory as the base save_path, if not provided
    if ("fast_disk" not in ops) or len(ops["fast_disk"]) == 0:
        ops["fast_disk"] = ops["save_path0"]

    # For mesoscope ROIs, makes copies of the values stored under the "dy" and "dx" keys.
    if "dy" in ops and ops["dy"] != "":
        dy = ops["dy"]
        dx = ops["dx"]

    # Converts known Path instances from string to Path
    ops["fast_disk"] = Path(ops["fast_disk"])
    ops["save_path0"] = Path(ops["save_path0"])
    ops["data_path"] = [Path(path) for path in ops["data_path"]]

    # Loops over each of the planes and constructs each plane-specific 'ops' dictionary. If the keys are populated in
    # the input 'ops' dictionary, stores the plane-specific value under the appropriate key in the plane-specific 'ops'
    # dictionary.
    for plane_index in range(plane_number):
        # Resolves the output directory for the plane data.
        if len(ops["save_folder"]) > 0:
            ops["save_path"] = ops["save_path0"].joinpath(ops["save_folder"], f"plane{plane_index}")
        else:
            ops["save_path"] = ops["save_path0"].joinpath("suite2p", f"plane{plane_index}")

        # Resolves the plane-specific working directory path.
        fast_disk = ops["fast_disk"].joinpath("suite2p", f"plane{plane_index}")

        # Defines the paths for the ops.npy file and the first channel's registered data binary file.
        ops["ops_path"] = ops["save_path"].joinpath("ops.npy")
        ops["reg_file"] = fast_disk.joinpath("data.bin")

        # If necessary, generates an additional binary file to store raw (unregistered) data after runtime.
        if ops.get("keep_movie_raw"):
            ops["raw_file"] = fast_disk.joinpath("data_raw.bin")

        # Sets the "lines" and "iplane" values for the current plane.
        if "lines" in ops:
            ops["lines"] = lines[plane_index]
        if "iplane" in ops:
            ops["iplane"] = iplane[plane_index]

        # If the data contains multiple functional channels, configures the binaries for the second channel.
        if channel_number > 1:
            ops["reg_file_chan2"] = fast_disk.joinpath("data_chan2.bin")
            if ops.get("keep_movie_raw"):
                ops["raw_file_chan2"] = fast_disk.joinpath("data_chan2_raw.bin")

        # Stores the mesoscope ROI coordinates (top left corner) "dy" and "dx" for the current plane.
        if "dy" in ops and ops["dy"] != "":
            ops["dy"] = dy[plane_index]
            ops["dx"] = dx[plane_index]

        # Creates directories for 'fast_disk' location and the save path if they do not exist.
        ensure_directory_exists(fast_disk)
        ensure_directory_exists(ops["save_path"])

        # Copies the modified 'ops' dictionary and appends it to 'plane_ops'.
        plane_ops.append(ops.copy())

    # Returns the list of 'ops' dictionaries for each plane.
    return plane_ops
