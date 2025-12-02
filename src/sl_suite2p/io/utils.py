"""This module provides utility functions for handling file searching, path management, and binary file operations."""

from copy import deepcopy
from typing import Any
from pathlib import Path

from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists

from ..configuration import RuntimeData


def _search_files_by_extension(
    root_directory: Path,
    extensions: tuple[str, ...] = ("tif", "tiff"),
    ignore_names: tuple[str, ...] = (),
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

        # Performs the same search one level down in the subdirectories of the provided root directory.
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


def _get_tiff_list(runtime_data: RuntimeData) -> tuple[list[Path], RuntimeData]:
    """Creates a list of .tif and .tiff files found in the directory specified by the "data_path" field of the
    runtime_data configuration. By default, it searches recursively through all subdirectories within each root
    directory listed in the "data_path" field.

    Args:
        runtime_data: A RuntimeData instance that stores the suite2p single-day configuration and runtime parameters.

    Returns:
        A tuple of two elements. The first element is a sorted list of the absolute paths to the found
        .tif and .tiff files, and the second element is the updated RuntimeData instance.

    Raises:
        FileNotFoundError: If no .tif or .tiff files are found in the directory or (if applicable) its subdirectories.
    """
    # Queries the absolute path(s) to root data directory.
    directories = runtime_data.configuration.file_io.data_path

    # Initializes a list to store the absolute paths to the discovered .tif and .tiff files.
    file_paths: list[Path] = []

    # Loops over all directories and searches for .TIFF files.
    for directory in directories:
        # Retrieves the absolute paths of the .tif and .tiff files in the target directory, searching
        # subdirectories as well.
        file_paths_found, _ = _search_files_by_extension(
            root_directory=directory,
            extensions=("tif", "tiff", "TIF", "TIFF"),
            ignore_names=tuple(runtime_data.configuration.file_io.ignored_file_names),
        )

        # Extends the returned data into storage list
        file_paths.extend(file_paths_found)

    # If no files were found in the directories, raises a FileNotFoundError.
    if len(file_paths) == 0:
        message = "Could not find any TIF/TIFF files to process."
        console.error(message=message, error=FileNotFoundError)

    file_paths = natsorted(file_paths)

    message = f"Found {len(file_paths)} TIF/TIFF files. Converting to binaries."
    console.echo(message=message, level=LogLevel.INFO)

    return file_paths, runtime_data


def find_files_open_binaries(
    plane_runtime_data_list: list[RuntimeData],
) -> tuple[list[RuntimeData], list[Path], list[Any], list[Any]]:
    """Finds the source data files for each plane inside the input list of plane-specific RuntimeData instances and
    prepares plane-specific binary files for writing the data.

    This service function resolves the paths to the raw data files and generates memory-mapped binary files for each
    plane. The output from this service function is later used to convert raw data files to the suite2p binary file
    format. This function currently only supports .tif and .tiff files.

    Args:
        plane_runtime_data_list: A list of all plane-specific RuntimeData instances that store single-day plane
                                 processing parameters.

    Returns:
        A tuple of four elements. The first element is the input 'plane_runtime_data_list', where each instance
        is updated with paths to source data files. The second element is the list of paths to source data files
        for each plane. The third element is the list of opened binaries for channel 1. The fourth element is the list
        of opened binaries for channel 2 if the data uses two functional channels.
    """
    # Initializes lists to store the binary files of each channel, which are eventually returned.
    channel_1_binary_files = []
    channel_2_binary_files = []

    # Iterates over each RuntimeData instance, processes plane data, and opens its binary files.
    for plane_runtime_data in plane_runtime_data_list:
        # Retrieves the number of channels from the plane-specific configuration.
        channel_number = plane_runtime_data.configuration.main.nchannels

        # Resolves paths to either raw or registered binary files for both channels depending on the configuration.
        if plane_runtime_data.configuration.registration.keep_movie_raw:
            # Opens the raw binary file and appends it to 'channel_1_binary_files'.
            channel_1_binary_files.append(plane_runtime_data.data.file_io.raw_file.open(mode="wb"))
            # If there is a second channel, opens the raw binary and appends it to 'channel_2_binary_files'.
            if channel_number > 1:
                channel_2_binary_files.append(plane_runtime_data.data.file_io.raw_file_channel_2.open(mode="wb"))
        else:
            # Opens the registered binary file and appends it to 'channel_1_binary_files'.
            channel_1_binary_files.append(plane_runtime_data.data.file_io.reg_file.open(mode="wb"))
            # If there is a second channel, opens the registered binary for THIS plane
            if channel_number > 1:
                channel_2_binary_files.append(plane_runtime_data.data.file_io.reg_file_channel_2.open(mode="wb"))

    # Determines the input format from the first RuntimeData instance's configuration.
    input_format = plane_runtime_data_list[0].configuration.file_io.input_format

    message = f"Input data format: {input_format}."
    console.echo(message=message, level=LogLevel.SUCCESS)

    # NOTE: The following blocks of code involving the .tif and .tiff files were originally a part of an if-else block
    # that supported deprecated input file types. If support for the deprecated file types is re-implemented in the
    # future, refer to the original Suite2p code for the original if-else 'input_format' logic. :)

    # Retrieves the absolute file paths to the .tif and .tiff files.
    file_paths, _ = _get_tiff_list(plane_runtime_data_list[0])

    return plane_runtime_data_list, file_paths, channel_1_binary_files, channel_2_binary_files


# noinspection PyUnboundLocalVariable
def initialize_plane_parameters(runtime_data: RuntimeData) -> list[RuntimeData]:
    """Constructs plane-specific RuntimeData instances for each plane and saves them to their respective directories.

    This function creates a separate RuntimeData YAML file for each plane, with plane-specific file paths configured.
    Each plane gets its own directory under save_path/suite2p/planeN/ with its own runtime_data.yaml file.

    Args:
        runtime_data: A RuntimeData instance that stores the suite2p single-day configuration and runtime parameters.

    Returns:
        A list of Path objects pointing to each plane's runtime_data.yaml file with the same length as the number of
        planes ('nplanes').
    """
    # Initialize list to store the RuntimeData instances for all planes.
    plane_runtime_data_list = []

    # Queries the number of planes and channels from the configuration.
    plane_number = runtime_data.configuration.main.nplanes
    channel_number = runtime_data.configuration.main.nchannels

    # Store references to mesoscope ROI data from the original runtime_data.
    lines = runtime_data.data.file_io.lines if runtime_data.data.file_io.lines is not None else None
    dy = (
        runtime_data.data.file_io.dy
        if runtime_data.data.file_io.dy is not None and len(runtime_data.data.file_io.dy) > 0
        else None
    )
    dx = (
        runtime_data.data.file_io.dx
        if runtime_data.data.file_io.dx is not None and len(runtime_data.data.file_io.dx) > 0
        else None
    )
    source_plane_indices = (
        runtime_data.data.file_io.plane_index
        if runtime_data.data.file_io.plane_index is not None and len(runtime_data.data.file_io.plane_index) > 0
        else None
    )

    # Loops over each plane and constructs each plane-specific RuntimeData instance.
    for plane_index in range(plane_number):
        # Create a separate RuntimeData instance for this plane with deep copied configuration.
        plane_runtime_data = RuntimeData(configuration=deepcopy(runtime_data.configuration))

        # Sets up the yaml_path for this plane and creates the directory hierarchy.
        plane_directory = set_yaml_path(plane_runtime_data, plane_index)
        plane_directory = plane_runtime_data.yaml_path.parent

        # Gets reference to the current plane's IOData.
        plane_io_data = plane_runtime_data.data.file_io

        # Defines the paths for the first channel's registered data binary file.
        plane_io_data.reg_file = plane_directory.joinpath("data.bin")
        plane_io_data.reg_file.touch()

        # If necessary, generates an additional binary file to store raw (unregistered) data after runtime.
        if plane_runtime_data.configuration.registration.keep_movie_raw:
            plane_io_data.raw_file = plane_directory.joinpath("data_raw.bin")
            plane_io_data.raw_file.touch()

        # If the data contains multiple functional channels, configures the binaries for the second channel.
        if channel_number > 1:
            plane_io_data.reg_file_channel_2 = plane_directory.joinpath("data_chan2.bin")
            plane_io_data.reg_file_channel_2.touch()

            if plane_runtime_data.configuration.registration.keep_movie_raw:
                plane_io_data.raw_file_channel_2 = plane_directory.joinpath("data_chan2_raw.bin")
                plane_io_data.raw_file_channel_2.touch()

        # Initializes the frame counter.
        plane_io_data.nframes = 0

        if lines is not None and plane_index < len(lines):
            plane_io_data.lines = lines[plane_index]

        if source_plane_indices is not None and plane_index < len(source_plane_indices):
            plane_io_data.plane_index = source_plane_indices[plane_index]

        # Stores the mesoscope ROI coordinates (top left corner) "dy" and "dx" for the current plane.
        if dy is not None and plane_index < len(dy):
            plane_io_data.dy = [dy[plane_index]]
            plane_io_data.dx = [dx[plane_index]]

        # Saves the plane-specific RuntimeData to its YAML file.
        plane_runtime_data.save()
        plane_runtime_data_list.append(plane_runtime_data)

    # Returns the list of RuntimeData instances for each plane.
    return plane_runtime_data_list


def set_yaml_path(runtime_data: RuntimeData, plane_index: int) -> Path:
    """Creates the plane-specific directory for the given plane index and sets yaml_path.

    Args:
        runtime_data: The RuntimeData instance to update
        plane_index: The index of the plane being processed.

    Returns:
        The path to the plane directory
    """
    save_path = Path(runtime_data.configuration.output.save_path)
    plane_directory = save_path.joinpath("suite2p", f"plane{plane_index}")
    ensure_directory_exists(plane_directory)
    runtime_data.yaml_path = plane_directory.joinpath("runtime_data.yaml")
    return plane_directory
