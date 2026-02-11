"""Provides functions for resolving pipeline runtime contexts from user configuration."""

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

    from ..dataclasses import ROIStatistics

# Default name for the acquisition parameters JSON file.
_ACQUISITION_PARAMETERS_FILENAME: str = "suite2p_parameters.json"


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


def _find_acquisition_parameters(data_path: Path) -> tuple[AcquisitionParameters, Path]:
    """Recursively searches for the acquisition parameters JSON file and imports its content as AcquisitionParameters.

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
    acquisition = _load_acquisition_parameters(json_path=parameters_path)

    return acquisition, data_directory


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
    result = []

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
                result.append(component)
                found_unique = True
                break

        if not found_unique:
            message = f"No unique component found for path: {path}, which is not allowed."
            console.error(message=message, error=RuntimeError)

    return tuple(result)


def _find_suite2p_directory(session_directory: Path) -> Path:
    """Discovers the suite2p output directory within a session directory tree.

    Searches recursively for the combined_metadata.npz file created by the single-day pipeline's combination step.
    Since multi-day session paths are not pre-sanitized, the suite2p directory may be
    nested at an arbitrary depth below the session root (e.g., under a processed_data/mesoscope_data/ subdirectory).

    Args:
        session_directory: The path to the session's root directory.

    Returns:
        The path to the suite2p output directory that contains the combined_metadata.npz file.

    Raises:
        FileNotFoundError: If no combined_metadata.npz file is found under the session directory.
        RuntimeError: If multiple combined_metadata.npz files are found under the session directory.
    """
    matches = list(session_directory.rglob("combined_metadata.npz"))

    if len(matches) == 0:
        message = (
            f"Unable to locate suite2p output for session {session_directory}. No combined_metadata.npz file was "
            f"found anywhere in the directory tree. Ensure the single-day pipeline has completed successfully for "
            f"this recording session before running multi-day processing."
        )
        console.error(message=message, error=FileNotFoundError)

    if len(matches) > 1:
        message = (
            f"Unable to locate suite2p output for session {session_directory}. Found {len(matches)} "
            f"combined_metadata.npz files, but expected exactly one unique match."
        )
        console.error(message=message, error=RuntimeError)

    # The combined_metadata.npz file is saved at the suite2p root level by CombinedData.save().
    return matches[0].parent


def _compute_mroi_region_borders(data_path: Path) -> list[int]:
    """Computes MROI region border x-coordinates from the acquisition parameters.

    For MROI recordings, the borders are the x-coordinates where one imaging region ends and another begins, computed
    from the sorted ROI x-coordinates (excluding the leftmost region's starting position). For non-MROI recordings,
    returns an empty list.

    Args:
        data_path: The path to the suite2p output directory containing acquisition_parameters.yaml.

    Returns:
        A list of border x-coordinates for MROI recordings, or an empty list for non-MROI recordings.
    """
    acquisition_path = data_path / "acquisition_parameters.yaml"
    acquisition = AcquisitionParameters.from_yaml(file_path=acquisition_path)
    if not acquisition.is_mroi:
        return []

    # Computes region borders from ROI x-coordinates. The borders are the x-coordinates where one region ends and
    # another begins, which are all x-coordinates except the minimum (leftmost region).
    sorted_x = sorted(acquisition.roi_x_coordinates)
    return sorted_x[1:]


def _select_session_cells(
    runtime: MultiDayRuntimeData,
    configuration: MultiDayConfiguration,
) -> None:
    """Selects cells from the single-day pipeline output that meet multi-day tracking criteria.

    Filters ROIs from the combined single-day data using the probability threshold, maximum size, and (for MROI
    recordings) region border margin specified in the configuration. The filtered cells are stored directly in
    runtime.extraction.roi_statistics.

    Notes:
        This step is expected to discard some single-day ROIs because the multi-day pipeline typically uses more
        stringent cell identification criteria.

    Args:
        runtime: The per-session runtime data containing the CombinedData instance created by the single-day pipeline
            for that session. The extraction.roi_statistics field is populated with the filtered cells.
        configuration: The multi-day pipeline configuration containing ROI selection parameters for the processed
            pipeline.

    Raises:
        ValueError: If the combined data or its extraction data is not available.
    """
    if runtime.combined_data is None or runtime.combined_data.extraction.roi_statistics is None:
        message = (
            f"Unable to select the cells tot rack across days for session {runtime.io.session_id}. The combined "
            f"single-day data must be loaded before cell selection can be performed."
        )
        console.error(message=message, error=ValueError)

    roi_statistics = runtime.combined_data.extraction.roi_statistics
    cell_classification = runtime.combined_data.extraction.cell_classification

    probability_threshold = configuration.roi_selection.probability_threshold
    maximum_size = configuration.roi_selection.maximum_size

    # Filters ROIs by classifier probability and pixel count. When cell_classification is available, only ROIs whose
    # classifier probability exceeds the threshold and whose pixel count is below the maximum size are retained.
    selected_cells: list[ROIStatistics] = []
    for index, roi in enumerate(roi_statistics):
        # Applies the probability threshold filter if classification data is available.
        if cell_classification is not None and cell_classification[index, 1] <= probability_threshold:
            continue

        # Applies the maximum size filter.
        if roi.pixel_count >= maximum_size:
            continue

        selected_cells.append(roi)

    # Filters ROIs near MROI region borders if applicable.
    mroi_region_borders = runtime.io.mroi_region_borders
    if mroi_region_borders:
        region_margin = configuration.roi_selection.mroi_region_margin
        selected_cells = [
            cell
            for cell in selected_cells
            if all(abs(cell.centroid[1] - border) > region_margin for border in mroi_region_borders)
        ]

    runtime.extraction.roi_statistics = selected_cells


def resolve_single_day_contexts(config: SingleDayConfiguration) -> list[RuntimeContext]:
    """Creates RuntimeContext instances for all planes without converting TIFF data.

    This function performs the initial setup for single-day processing: it finds acquisition parameters from the data
    directory, creates output directories, and initializes RuntimeContext instances with IOData fields populated. It
    does not perform TIFF to binary conversion, which should be done separately via convert_tiffs_to_binary().

    For standard single-ROI data, one context is created per physical plane. For MROI (Multi-ROI) data, one context
    is created per virtual plane, where virtual planes are ROI x physical plane combinations.

    Args:
        config: The single-day pipeline configuration. Must have data_path and save_path configured in file_io.

    Returns:
        A list of RuntimeContext instances, one per plane (or virtual plane for MROI data). Each context contains
        references to the shared configuration, acquisition parameters, and a plane-specific SingleDayRuntimeData
        instance with IOData fields initialized.

    Raises:
        ValueError: If data_path or save_path is not configured in the configuration.
    """
    # Validates that data_path is configured.
    if config.file_io.data_path is None:
        message = (
            "Unable to resolve single-day contexts. The data_path must be configured in the FileIO section of the "
            "configuration, but it is currently None."
        )
        console.error(message=message, error=ValueError)
    data_path: Path = config.file_io.data_path

    # Finds acquisition parameters and the data directory containing TIFFs.
    acquisition, data_directory = _find_acquisition_parameters(data_path)

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

    # Derives the per-plane sampling rate from the acquisition frame rate and the number of physical planes.
    sampling_rate = acquisition.frame_rate / acquisition.plane_number

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
            sampling_rate=sampling_rate,
            data_directory=data_directory,
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


def resolve_multiday_contexts(configuration: MultiDayConfiguration) -> list[MultiDayRuntimeContext]:
    """Resolves MultiDayRuntimeContext instances for all sessions in the target multi-day pipeline.

    For each session directory in the configuration, discovers the suite2p output directory, derives the multiday
    output path, and either loads an existing MultiDayRuntimeData from disk or constructs a new one. New sessions
    undergo cell selection filtering and output directory creation.

    Args:
        configuration: The multi-day pipeline configuration containing session directories and processing parameters.

    Returns:
        A list of MultiDayRuntimeContext instances, one for each session to be processed.
    """
    session_directories = configuration.session_io.session_directories
    session_ids = _extract_unique_components(paths=session_directories)
    dataset_name = configuration.session_io.dataset_name

    # Resolves all suite2p directories (data_paths) and output paths upfront so that the complete
    # dataset_output_paths list can be stored in every session's IO data.
    data_paths: list[Path] = []
    output_paths: list[Path] = []
    for session_directory in session_directories:
        data_path = _find_suite2p_directory(session_directory=session_directory)
        data_paths.append(data_path)
        output_paths.append(data_path.parent / "multiday" / dataset_name)

    contexts: list[MultiDayRuntimeContext] = []

    for index, session_id in enumerate(session_ids):
        data_path = data_paths[index]
        output_path = output_paths[index]

        # Loads CombinedData from the single-day pipeline outputs.
        combined_data = CombinedData.load(root_path=data_path)

        runtime_path = output_path / "multiday_runtime_data.yaml"
        if runtime_path.exists():
            # Loads existing runtime data (pure deserialization from known output_path).
            runtime = MultiDayRuntimeData.load(output_path=output_path)
            runtime.combined_data = combined_data

            # Updates dataset_output_paths in case sessions were added or paths changed.
            runtime.io.dataset_output_paths = list(output_paths)

            contexts.append(MultiDayRuntimeContext(configuration=configuration, runtime=runtime))
            console.echo(
                message=f"Loaded existing multi-day runtime data for session {session_id}.", level=LogLevel.INFO
            )
            continue

        # Constructs new IO data with all discovered paths.
        io_data = MultiDayIOData(
            session_id=session_id,
            data_path=data_path,
            dataset_name=dataset_name,
            dataset_output_paths=list(output_paths),
        )

        # Computes MROI region borders from acquisition parameters if applicable.
        io_data.mroi_region_borders = _compute_mroi_region_borders(data_path=data_path)

        # Constructs the runtime data with the IO data and output path.
        runtime = MultiDayRuntimeData(output_path=output_path, io=io_data)
        runtime.combined_data = combined_data

        # Runs cell selection filtering on the combined single-day data.
        _select_session_cells(runtime=runtime, configuration=configuration)

        # Creates the output directory for this session.
        ensure_directory_exists(output_path)

        contexts.append(MultiDayRuntimeContext(configuration=configuration, runtime=runtime))
        cell_count = len(runtime.extraction.roi_statistics) if runtime.extraction.roi_statistics else 0
        console.echo(
            message=(
                f"Initialized multi-day runtime for session {session_id} with {cell_count} preselected cell candidates."
            ),
            level=LogLevel.SUCCESS,
        )

    # Saves shared configuration once via the first context.
    contexts[0].save_shared()

    # Saves each runtime to persist the fully-resolved IO data (including dataset_output_paths).
    for context in contexts:
        context.save_runtime()

    return contexts
