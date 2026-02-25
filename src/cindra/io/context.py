"""Provides assets for resolving pipeline runtime contexts from user configuration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists

from ..dataclasses import (
    IOData,
    CombinedData,
    MultiDayIOData,
    RuntimeContext,
    MultiDayRuntimeData,
    SingleDayRuntimeData,
    AcquisitionParameters,
    MultiDayConfiguration,
    MultiDayRuntimeContext,
    SingleDayConfiguration,
)

if TYPE_CHECKING:
    from pathlib import Path

# Preferred name for the acquisition parameters JSON file.
_PREFERRED_PARAMETERS_FILENAME: str = "cindra_parameters.json"

# Legacy name for the acquisition parameters JSON file (fallback).
_LEGACY_PARAMETERS_FILENAME: str = "suite2p_parameters.json"

# Maximum number of imaging channels supported by the pipeline.
_MAXIMUM_CHANNEL_COUNT: int = 2


def find_data_directory(data_path: Path) -> Path:
    """Recursively searches for the directory containing the acquisition parameters JSON file.

    This function searches the data_path directory and all subdirectories for a file named 'cindra_parameters.json'
    first, then falls back to 'suite2p_parameters.json' (created by sl-experiment). Returns the parent directory
    containing the matched file. This directory is expected to also contain the TIFF files.

    Args:
        data_path: The root directory to search for the acquisition parameters file.

    Returns:
        The path to the directory containing the acquisition parameters JSON file.

    Raises:
        FileNotFoundError: If no acquisition parameters file is found in the data directory or its subdirectories.
        ValueError: If the data_path is not a directory.
    """
    if not data_path.is_dir():
        message = f"Unable to find data directory. The data_path is not a directory: {data_path}"
        console.error(message=message, error=ValueError)

    # Searches for the preferred cindra parameters file first, then falls back to the legacy suite2p file.
    parameter_files = list(data_path.rglob(_PREFERRED_PARAMETERS_FILENAME))
    if not parameter_files:
        parameter_files = list(data_path.rglob(_LEGACY_PARAMETERS_FILENAME))

    if not parameter_files:
        message = (
            f"Unable to find '{_PREFERRED_PARAMETERS_FILENAME}' or '{_LEGACY_PARAMETERS_FILENAME}' in the data "
            f"directory or its subdirectories: {data_path}. This file is required and must contain acquisition "
            f"metadata."
        )
        console.error(message=message, error=FileNotFoundError)

    return parameter_files[0].parent


def resolve_single_day_contexts(configuration: SingleDayConfiguration) -> list[RuntimeContext]:
    """Creates RuntimeContext instances for all imaging planes processed by the target single-day pipeline.

    This function performs the initial setup for single-day processing: it finds acquisition parameters from the data
    directory, creates output directories, and initializes RuntimeContext instances for each of the recording's plains.

    Notes:
        For standard single-ROI data, one context is created per physical plane. For MROI (Multi-ROI) data, one context
        is created per virtual plane, where virtual planes are ROI x physical plane combinations.

        The configuration and acquisition parameters are always saved to disk, ensuring they reflect the current
        settings passed to this function.

        When loading previously processed data (e.g., data moved to a different machine), acquisition parameters are
        loaded from the saved output directory if available, allowing the pipeline to work without raw TIFF data.

    Args:
        configuration: The single-day pipeline configuration. Must have save_path configured in file_io. The data_path
            is only required when raw data needs to be processed (rebinarization) or when no processed data exists.

    Returns:
        A list of RuntimeContext instances, one per plane (or virtual plane for MROI data). Each context contains
        references to the shared configuration, acquisition parameters, and a plane-specific SingleDayRuntimeData
        instance with IOData fields initialized.

    Raises:
        ValueError: If save_path is not configured, or if the acquisition parameters specify more than 2 channels.
        FileNotFoundError: If neither processed data nor raw data with acquisition parameters is available.
    """
    # Validates that the save path is configured.
    save_path_root = configuration.file_io.save_path
    if save_path_root is None:
        message = (
            "Unable to resolve single-day contexts. The save_path must be configured in the FileIO section of the "
            "configuration, but it is currently None."
        )
        console.error(message=message, error=ValueError)

    # Checks if processed data already exists with saved acquisition parameters.
    saved_acquisition_path = save_path_root / "cindra" / "acquisition_parameters.yaml"
    if saved_acquisition_path.exists():
        # Loads acquisition parameters from processed output (supports loading moved data without raw TIFFs).
        acquisition = AcquisitionParameters.from_yaml(file_path=saved_acquisition_path)
        console.echo(message=f"Loaded acquisition parameters from: {saved_acquisition_path}.", level=LogLevel.INFO)
    else:
        # Falls back to finding acquisition parameters from raw data path.
        if configuration.file_io.data_path is None:
            message = (
                "Unable to resolve single-day contexts. No processed data exists at the save_path and data_path is "
                "not configured. Either provide processed data or configure data_path to point to raw TIFF data."
            )
            console.error(message=message, error=ValueError)
        acquisition = _find_acquisition_parameters(configuration.file_io.data_path)

    # Validates that the channel count does not exceed the maximum supported channel count.
    if acquisition.channel_number > _MAXIMUM_CHANNEL_COUNT:
        message = (
            f"Unable to resolve single-day contexts. The pipeline supports at most {_MAXIMUM_CHANNEL_COUNT} channels, "
            f"but the acquisition parameters specify {acquisition.channel_number} channels."
        )
        console.error(message=message, error=ValueError)

    # Determines the number of contexts to create. For MROI data, creates one context per virtual plane
    # (ROI x physical plane combination). For single-ROI data, creates one context per physical plane.
    plane_count = acquisition.virtual_plane_count if acquisition.is_mroi else acquisition.plane_number

    # Determines whether the recording uses two channels.
    has_two_channels = acquisition.channel_number > 1

    # Derives the per-plane sampling rate from the acquisition frame rate and the number of physical planes.
    sampling_rate = acquisition.frame_rate / acquisition.plane_number

    # Initializes the list to store RuntimeContext instances for each plane.
    contexts: list[RuntimeContext] = []

    # Creates a RuntimeContext for each plane.
    for virtual_plane_index in range(plane_count):
        # Resolves the output directory for this plane. Always uses 'cindra' as the subdirectory.
        plane_output_path = save_path_root / "cindra" / f"plane_{virtual_plane_index}"

        # Checks if existing runtime data exists for this plane.
        runtime_yaml_path = plane_output_path / "runtime_data.yaml"
        if runtime_yaml_path.exists():
            # Loads existing runtime data.
            runtime_data = SingleDayRuntimeData.load(output_path=plane_output_path)
            console.echo(message=f"Loaded existing runtime data for plane {virtual_plane_index}.", level=LogLevel.INFO)

            context = RuntimeContext(
                configuration=configuration,
                acquisition=acquisition,
                runtime=runtime_data,
            )
            contexts.append(context)
            continue

        # Creates the output directory if it does not exist.
        ensure_directory_exists(plane_output_path)

        # Initializes the IOData for this plane with binary file paths.
        io_data = IOData(
            output_directory=plane_output_path,
            registered_binary_path=plane_output_path / "channel_1_data.bin",
            plane_index=virtual_plane_index,
            sampling_rate=sampling_rate,
        )

        # Configures second channel binary paths if using two channels.
        if has_two_channels:
            io_data.registered_binary_path_channel_2 = plane_output_path / "channel_2_data.bin"

        # Populates MROI-specific fields if processing multi-ROI data.
        if acquisition.is_mroi:
            # Computes ROI index and physical plane index from the virtual plane index. Virtual planes are organized
            # as: ROI 0 plane 0, ROI 0 plane 1, ..., ROI 1 plane 0, ROI 1 plane 1, etc.
            roi_index = virtual_plane_index // acquisition.plane_number
            io_data.mroi_lines = acquisition.roi_lines[roi_index]
            io_data.mroi_y_offset = acquisition.roi_y_coordinates[roi_index]
            io_data.mroi_x_offset = acquisition.roi_x_coordinates[roi_index]

        # Creates the SingleDayRuntimeData with the initialized IOData.
        runtime_data = SingleDayRuntimeData(
            output_path=plane_output_path,
            io=io_data,
        )

        # Creates the RuntimeContext combining the shared configuration, acquisition parameters, and runtime data.
        context = RuntimeContext(
            configuration=configuration,
            acquisition=acquisition,
            runtime=runtime_data,
        )

        contexts.append(context)

    # Saves shared configuration and acquisition parameters to ensure they are always up to date.
    contexts[0].save_shared()

    # Saves each runtime to persist the initialized IO data.
    for context in contexts:
        context.save_runtime()

    return contexts


def resolve_multiday_contexts(
    configuration: MultiDayConfiguration,
    target_session_id: str | None = None,
) -> list[MultiDayRuntimeContext]:
    """Creates MultiDayRuntimeContext instances for recording sessions processed by the target multi-day pipeline.

    This function performs the initial setup for multi-day processing: it discovers cindra output directories for each
    session, derives multiday output paths, and initializes MultiDayRuntimeContext instances.

    Notes:
        Each session directory must contain exactly one cindra output directory with a combined_metadata.npz file from
        a completed single-day pipeline run. The function extracts unique session identifiers from the directory paths
        to distinguish sessions within the dataset.

        The configuration is always saved to disk, ensuring it reflects the current settings passed to this function.
        Cell selection is performed as a separate step using select_session_cells(), not during context resolution.

        When target_session_id is provided, only the matching session's CombinedData and runtime data are loaded.
        Non-matching sessions are skipped entirely. This avoids the overhead of loading large arrays for sessions
        that will not be used (e.g., during per-session extraction).

    Args:
        configuration: The multi-day pipeline configuration. Must have session_directories and dataset_name configured
            in session_io.
        target_session_id: When provided, only resolves the context for the session matching this identifier. The
            returned list contains a single element. When None (default), all sessions are resolved.

    Returns:
        A list of MultiDayRuntimeContext instances, one per session (or one element when target_session_id is set).
        Each context contains references to the shared configuration and a session-specific MultiDayRuntimeData
        instance with MultiDayIOData fields initialized.

    Raises:
        FileNotFoundError: If no combined_metadata.npz file is found in a session directory.
        RuntimeError: If multiple combined_metadata.npz files are found in a session directory, or if session paths
            do not contain unique identifying components.
        ValueError: If target_session_id does not match any resolved session identifier.
    """
    session_directories = configuration.session_io.session_directories
    session_ids = _extract_unique_components(paths=session_directories)
    dataset_name = configuration.session_io.dataset_name

    # Resolves all cindra directories and output paths upfront. These are cheap path operations needed for every
    # session regardless of target filtering, because dataset_output_paths stores the full set.
    data_paths: list[Path] = []
    output_paths: list[Path] = []
    for session_directory in session_directories:
        data_path = _find_cindra_directory(session_directory=session_directory)
        data_paths.append(data_path)
        output_paths.append(data_path / "multiday" / dataset_name)

    # Validates the target session ID before performing expensive I/O.
    if target_session_id is not None and target_session_id not in session_ids:
        available_ids = list(session_ids)
        message = (
            f"Unable to resolve multi-day context for session '{target_session_id}'. The provided session_id does "
            f"not match any resolved session identifier. Available session IDs: {available_ids}."
        )
        console.error(message=message, error=ValueError)

    contexts: list[MultiDayRuntimeContext] = []

    for index, session_id in enumerate(session_ids):
        # Skips non-target sessions when a specific session is requested.
        if target_session_id is not None and session_id != target_session_id:
            continue

        data_path = data_paths[index]
        output_path = output_paths[index]
        combined_data = CombinedData.load(root_path=data_path)

        runtime_path = output_path / "multiday_runtime_data.yaml"
        if runtime_path.exists():
            # Loads existing runtime data (pure deserialization from known output_path).
            runtime = MultiDayRuntimeData.load(output_path=output_path)

            # Updates IO paths to reflect the current configuration's session directories. This handles cases where
            # session directories have changed or data was moved since the runtime was last saved.
            runtime.io.data_path = data_path
            runtime.io.dataset_output_paths = tuple(output_paths)
            runtime.io.mroi_region_borders = _compute_mroi_region_borders(data_path=data_path)

            # Injects the preloaded CombinedData to ensure it's available regardless of __post_init__ behavior.
            runtime.combined_data = combined_data

            contexts.append(MultiDayRuntimeContext(configuration=configuration, runtime=runtime))
            continue

        # Constructs new IO data with all discovered paths.
        io_data = MultiDayIOData(
            session_id=session_id,
            data_path=data_path,
            dataset_name=dataset_name,
            dataset_output_paths=tuple(output_paths),
        )

        # Computes MROI region borders from acquisition parameters if applicable.
        io_data.mroi_region_borders = _compute_mroi_region_borders(data_path=data_path)

        # Constructs the runtime data with the IO data, output path, and preloaded CombinedData.
        runtime = MultiDayRuntimeData(output_path=output_path, io=io_data, combined_data=combined_data)

        # Creates the output directory for this session.
        ensure_directory_exists(output_path)

        contexts.append(MultiDayRuntimeContext(configuration=configuration, runtime=runtime))

    console.echo(message=f"Loaded existing multi-day runtime data for {len(contexts)} session(s).", level=LogLevel.INFO)

    # Saves shared configuration once via the first context to ensure it is always up to date.
    contexts[0].save_shared()

    # Saves each runtime to persist the fully-resolved IO data (including dataset_output_paths).
    for context in contexts:
        context.save_runtime()

    return contexts


def _load_acquisition_parameters(json_path: Path) -> AcquisitionParameters:
    """Loads acquisition parameters from a JSON file and validates all required fields.

    For single-ROI data, frame_rate, plane_number, and channel_number are required. For MROI data (roi_number > 1),
    roi_lines, roi_x_coordinates, and roi_y_coordinates are additionally required.

    Args:
        json_path: The path to the JSON file containing acquisition parameters.

    Returns:
        An AcquisitionParameters instance populated from the JSON file.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        ValueError: If required fields are missing from the JSON data.
    """
    if not json_path.exists():
        message = f"Acquisition parameters file not found: {json_path}"
        console.error(message=message, error=FileNotFoundError)

    with json_path.open("r") as file:
        data = json.load(file)

    # Extracts frame_rate (required).
    frame_rate = data.get("frame_rate")
    if frame_rate is None:
        message = (
            f"Unable to extract the required field 'frame_rate' from the acquisition parameters file "
            f"located at {json_path}."
        )
        console.error(message=message, error=ValueError)

    # Extracts plane_number (required).
    plane_number = data.get("plane_number")
    if plane_number is None:
        message = (
            f"Unable to extract the required field 'plane_number' from the acquisition parameters file "
            f"located at {json_path}."
        )
        console.error(message=message, error=ValueError)

    # Extracts channel_number (required).
    channel_number = data.get("channel_number")
    if channel_number is None:
        message = (
            f"Unable to extract the required field 'channel_number' from the acquisition parameters file "
            f"located at {json_path}."
        )
        console.error(message=message, error=ValueError)

    # Extracts roi_number (defaults to 1 for single-ROI).
    roi_number = data.get("roi_number", 1)

    # For MROI data (roi_number > 1), validates that all MROI fields are present.
    if roi_number > 1:
        roi_lines = data.get("roi_lines")
        if roi_lines is None:
            message = (
                f"Unable to extract the required field 'roi_lines' from the acquisition parameters file "
                f"located at {json_path}."
            )
            console.error(message=message, error=ValueError)

        roi_x_coordinates = data.get("roi_x_coordinates")
        if roi_x_coordinates is None:
            message = (
                f"Unable to extract the required field 'roi_x_coordinates' from the acquisition parameters "
                f"file located at {json_path}."
            )
            console.error(message=message, error=ValueError)

        roi_y_coordinates = data.get("roi_y_coordinates")
        if roi_y_coordinates is None:
            message = (
                f"Unable to extract the required field 'roi_y_coordinates' from the acquisition parameters "
                f"file located at {json_path}."
            )
            console.error(message=message, error=ValueError)
    else:
        roi_lines = []
        roi_x_coordinates = []
        roi_y_coordinates = []

    return AcquisitionParameters(
        frame_rate=frame_rate,
        plane_number=plane_number,
        channel_number=channel_number,
        roi_number=roi_number,
        roi_lines=roi_lines,
        roi_x_coordinates=roi_x_coordinates,
        roi_y_coordinates=roi_y_coordinates,
    )


def _find_acquisition_parameters(data_path: Path) -> AcquisitionParameters:
    """Finds and loads acquisition parameters from the data directory.

    Searches for the preferred 'cindra_parameters.json' first, then falls back to the legacy
    'suite2p_parameters.json'.

    Args:
        data_path: The root directory to search for the acquisition parameters file.

    Returns:
        The loaded AcquisitionParameters instance.

    Raises:
        FileNotFoundError: If no acquisition parameters file is found.
        ValueError: If the data_path is not a directory, or if required fields are missing from the JSON file.
    """
    data_directory = find_data_directory(data_path)

    # Tries the preferred filename first, then falls back to the legacy filename.
    parameters_path = data_directory / _PREFERRED_PARAMETERS_FILENAME
    if not parameters_path.exists():
        parameters_path = data_directory / _LEGACY_PARAMETERS_FILENAME

    message = f"Found acquisition parameters at: {parameters_path}."
    console.echo(message=message, level=LogLevel.SUCCESS)

    return _load_acquisition_parameters(json_path=parameters_path)


def _extract_unique_components(paths: list[Path] | tuple[Path, ...]) -> tuple[str, ...]:
    """Extracts the first component from the end of each input path that uniquely identifies each path globally.

    Notes:
        This function adapts the multi-day pipeline to directory structures where the unique session identifier appears
        at different levels of the path hierarchy. For example, given paths like ``/data/day1/session`` and
        ``/data/day2/session``, the function identifies ``day1`` and ``day2`` as the unique components (not ``session``,
        which is shared). This allows users to organize sessions using any naming convention, as long as each path
        contains at least one unique component somewhere in its hierarchy.

    Args:
        paths: A list or tuple of Path objects.

    Returns:
        A tuple of unique components, one for each path, stored in the same order as the input paths.

    Raises:
        RuntimeError: If one or more paths do not contain unique components.
    """
    paths_list = list(paths)
    unique_components: list[str] = []

    for path in paths_list:
        # Gets components from right to left.
        components = list(path.parts)[::-1]
        found_unique = False

        for component in components:
            # Checks if this component appears in any other path.
            is_unique = True

            for other_path in paths_list:
                if path == other_path:
                    continue

                # If the component appears anywhere in the other path, it is not unique.
                if component in other_path.parts:
                    is_unique = False
                    break

            if is_unique:
                unique_components.append(component)
                found_unique = True
                break

        if not found_unique:
            message = f"No unique component found for path: {path}, which is not allowed."
            console.error(message=message, error=RuntimeError)

    return tuple(unique_components)


def _find_cindra_directory(session_directory: Path) -> Path:
    """Discovers the cindra output directory within a session directory tree.

    Searches recursively for the combined_metadata.npz file created by the single-day pipeline's combination step.
    Since multi-day session paths are not pre-sanitized, the cindra directory may be
    nested at an arbitrary depth below the session root (e.g., under a processed_data/mesoscope_data/ subdirectory).

    Args:
        session_directory: The path to the session's root directory.

    Returns:
        The path to the cindra output directory that contains the combined_metadata.npz file.

    Raises:
        FileNotFoundError: If no combined_metadata.npz file is found under the session directory.
        RuntimeError: If multiple combined_metadata.npz files are found under the session directory.
    """
    matches = list(session_directory.rglob("combined_metadata.npz"))

    if not matches:
        message = (
            f"Unable to locate cindra output for session {session_directory}. No combined_metadata.npz file was "
            f"found anywhere in the directory tree. Ensure the single-day pipeline has completed successfully for "
            f"this recording session before running multi-day processing."
        )
        console.error(message=message, error=FileNotFoundError)

    if len(matches) > 1:
        message = (
            f"Unable to locate cindra output for session {session_directory}. Found {len(matches)} "
            f"combined_metadata.npz files, but expected exactly one unique match."
        )
        console.error(message=message, error=RuntimeError)

    # The combined_metadata.npz file is saved at the cindra root level by CombinedData.save().
    return matches[0].parent


def _compute_mroi_region_borders(data_path: Path) -> tuple[int, ...]:
    """Computes MROI region border x-coordinates from the acquisition parameters.

    For MROI recordings, the borders are the x-coordinates where one imaging region ends and another begins, computed
    from the sorted ROI x-coordinates (excluding the leftmost region's starting position). For non-MROI recordings,
    returns an empty tuple.

    Args:
        data_path: The path to the cindra output directory containing acquisition_parameters.yaml.

    Returns:
        A tuple of border x-coordinates for MROI recordings, or an empty tuple for non-MROI recordings.
    """
    acquisition_path = data_path / "acquisition_parameters.yaml"
    acquisition = AcquisitionParameters.from_yaml(file_path=acquisition_path)
    if not acquisition.is_mroi:
        return ()

    # Computes region borders from ROI x-coordinates. The borders are the x-coordinates where one region ends and
    # another begins, which are all x-coordinates except the minimum (leftmost region).
    sorted_x = sorted(acquisition.roi_x_coordinates)
    return tuple(sorted_x[1:])
