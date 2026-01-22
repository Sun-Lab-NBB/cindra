"""Provides utility functions for handling file searching, path management, and binary file operations."""

from copy import deepcopy
from typing import Any
from pathlib import Path

from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists

from ..configuration import RuntimeData


def _get_tiff_list(runtime_data: RuntimeData) -> list[Path]:
    """Recursively searches for .tif and .tiff files under all directories in "data_path".

    Args:
        runtime_data: A RuntimeData instance storing the suite2p configuration and runtime parameters.

    Returns:
        A naturally sorted list of absolute paths to all discovered .tif and .tiff files.

    Raises:
        FileNotFoundError: If no .tif or .tiff files are found in any of the data directories.
    """
    directories = runtime_data.configuration.file_io.data_path
    ignored_names = set(runtime_data.configuration.file_io.ignored_file_names)
    extensions = ("tif", "tiff", "TIF", "TIFF")

    file_paths: list[Path] = []
    for directory in directories:
        for ext in extensions:
            file_paths.extend(
                file.resolve() for file in directory.rglob(f"*.{ext}") if file.stem not in ignored_names
            )

    if not file_paths:
        message = "Could not find any TIF/TIFF files to process."
        console.error(message=message, error=FileNotFoundError)

    file_paths = natsorted(file_paths)

    message = f"Found {len(file_paths)} TIF/TIFF files. Converting to binaries."
    console.echo(message=message, level=LogLevel.INFO)

    return file_paths


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

    # Retrieves the absolute file paths to the .tif and .tiff files.
    file_paths = _get_tiff_list(plane_runtime_data_list[0])

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
