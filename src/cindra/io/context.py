"""Provides assets for resolving pipeline runtime contexts from user configuration."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists
from ataraxis_data_structures import resolve_unique_roots, discover_marker_files

from ..layout import (
    PARAMETERS_FILENAME,
    OUTPUT_DIRECTORY_NAME,
    CHANNEL_1_BINARY_FILENAME,
    CHANNEL_2_BINARY_FILENAME,
    COMBINED_METADATA_FILENAME,
    MULTI_RECORDING_DIRECTORY_NAME,
    ACQUISITION_PARAMETERS_FILENAME,
    MULTI_RECORDING_RUNTIME_DATA_FILENAME,
    SINGLE_RECORDING_RUNTIME_DATA_FILENAME,
    resolve_plane_specifier,
)
from ..dataclasses import (
    IOData,
    CombinedData,
    RuntimeContext,
    MultiRecordingIOData,
    AcquisitionParameters,
    MultiRecordingRuntimeData,
    SingleRecordingRuntimeData,
    MultiRecordingConfiguration,
    MultiRecordingRuntimeContext,
    SingleRecordingConfiguration,
)

if TYPE_CHECKING:
    from pathlib import Path

MAXIMUM_CHANNEL_COUNT: int = 2
"""The maximum number of imaging channels supported by the pipeline."""


def find_data_directory(data_path: Path) -> Path:
    """Recursively searches for the directory containing the acquisition parameters JSON file.

    Searches the data_path directory and all subdirectories for a file named 'cindra_parameters.json'.
    Returns the parent directory containing the matched file. This directory is expected to also contain the TIFF
    files.

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

    parameter_files = discover_marker_files(directory=data_path, marker_name=PARAMETERS_FILENAME)

    if not parameter_files:
        message = (
            f"Unable to find '{PARAMETERS_FILENAME}' in the data directory or its subdirectories: {data_path}. "
            f"This file is required and must contain acquisition metadata."
        )
        console.error(message=message, error=FileNotFoundError)

    return parameter_files[0].parent


def resolve_single_recording_contexts(
    configuration: SingleRecordingConfiguration,
    *,
    persist: bool = True,
) -> list[RuntimeContext]:
    """Creates RuntimeContext instances for all imaging planes processed by the target single-recording pipeline.

    Performs the initial setup for single-recording processing: finds acquisition parameters from
    the data directory, creates output directories, and initializes RuntimeContext instances for each of the
    recording's planes.

    Notes:
        For standard single-ROI data, one context is created per physical plane. For MROI (Multi-ROI) data, one context
        is created per virtual plane, where virtual planes are ROI x physical plane combinations.

        With ``persist=True`` (the default), the shared configuration and acquisition parameters plus every plane's
        runtime data file are saved to disk at the end of resolution, ensuring they reflect the current settings. This
        is the correct mode for single-threaded bootstrap (e.g., the prepare_single_recording_batch_tool invocation).

        With ``persist=False``, no files are written. This mode is required for worker entry (REMOTE mode), because
        this resolver builds a context for every plane rather than for the one plane the worker was dispatched to
        process. A persisting worker would therefore write each peer plane's runtime_data.yaml from its own
        snapshot, overwriting whatever a peer had already recorded there with content that is whole and parsable
        but stale. Workers must therefore only *load* the bootstrap written by the earlier prepare step. When
        ``persist=False``, any missing runtime_data.yaml is treated as a hard error because it indicates
        prepare_single_recording_batch_tool was not run first.

        When loading previously processed data (e.g., data moved to a different machine), acquisition parameters are
        loaded from the saved output directory if available, allowing the pipeline to work without raw TIFF data.

    Args:
        configuration: The single-recording pipeline configuration. Must have output_path configured in
            file_io. The data_path is only required when raw data needs to be processed (rebinarization) or
            when no processed data exists.
        persist: When True (default), writes the shared configuration plus every plane's runtime_data.yaml at the
            end of resolution. When False, treats the call as load-only and raises FileNotFoundError if any
            expected runtime_data.yaml is missing. Worker entry points (REMOTE mode) must call with
            ``persist=False``. The prepare_single_recording_batch_tool owns bootstrap persistence in a
            single-threaded context.

    Returns:
        A list of RuntimeContext instances, one per plane (or virtual plane for MROI data). Each context contains
        references to the shared configuration, acquisition parameters, and a plane-specific SingleRecordingRuntimeData
        instance with IOData fields initialized.

    Raises:
        ValueError: If output_path is not configured, if no processed data exists at output_path while data_path is
            also not configured, or if the acquisition parameters specify more than 2 channels.
        FileNotFoundError: If the acquisition parameters file is not found under data_path, or if ``persist=False``
            and any plane's runtime_data.yaml does not already exist on disk.
    """
    # Validates that the save path is configured.
    output_path_root = configuration.file_io.output_path
    if output_path_root is None:
        message = (
            "Unable to resolve single-recording contexts. The output_path must be configured in the "
            "FileIO section of the configuration, but it is currently None."
        )
        console.error(message=message, error=ValueError)

    # Checks if processed data already exists with saved acquisition parameters.
    saved_acquisition_path = output_path_root / OUTPUT_DIRECTORY_NAME / ACQUISITION_PARAMETERS_FILENAME
    if saved_acquisition_path.exists():
        # Loads acquisition parameters from processed output (supports loading moved data without raw TIFFs).
        acquisition = AcquisitionParameters.from_yaml(file_path=saved_acquisition_path)
        console.echo(message=f"Loaded acquisition parameters from: {saved_acquisition_path}.", level=LogLevel.INFO)
    else:
        # Falls back to finding acquisition parameters from raw data path.
        if configuration.file_io.data_path is None:
            message = (
                "Unable to resolve single-recording contexts. No processed data exists at the output_path "
                "and data_path is not configured. Either provide processed data or configure data_path to "
                "point to raw TIFF data."
            )
            console.error(message=message, error=ValueError)
        acquisition = _find_acquisition_parameters(data_path=configuration.file_io.data_path)

    if acquisition.channel_number > MAXIMUM_CHANNEL_COUNT:
        message = (
            f"Unable to resolve single-recording contexts. The pipeline supports at most "
            f"{MAXIMUM_CHANNEL_COUNT} channels, but the acquisition parameters specify "
            f"{acquisition.channel_number} channels."
        )
        console.error(message=message, error=ValueError)

    # Determines the number of contexts to create. For MROI data, creates one context per virtual plane
    # (ROI x physical plane combination). For single-ROI data, creates one context per physical plane.
    plane_count = acquisition.virtual_plane_count if acquisition.is_mroi else acquisition.plane_number

    # Determines whether the recording uses two channels.
    has_two_channels = acquisition.channel_number > 1

    # Derives the per-plane sampling rate from the acquisition frame rate and the number of physical planes.
    sampling_rate = acquisition.frame_rate / acquisition.plane_number

    contexts: list[RuntimeContext] = []

    for virtual_plane_index in range(plane_count):
        # Resolves the output directory for this plane, which the layout contract names.
        plane_output_path = (
            output_path_root / OUTPUT_DIRECTORY_NAME / resolve_plane_specifier(plane_index=virtual_plane_index)
        )

        # Checks if existing runtime data exists for this plane.
        runtime_yaml_path = plane_output_path / SINGLE_RECORDING_RUNTIME_DATA_FILENAME
        if runtime_yaml_path.exists():
            # Loads existing runtime data (scalars only). Arrays are loaded on demand by each pipeline function.
            runtime_data = SingleRecordingRuntimeData.load(output_path=plane_output_path)
            console.echo(message=f"Loaded existing runtime data for plane {virtual_plane_index}.", level=LogLevel.INFO)

            context = RuntimeContext(
                configuration=configuration,
                acquisition=acquisition,
                runtime=runtime_data,
            )
            contexts.append(context)
            continue

        ensure_directory_exists(path=plane_output_path, is_file=False)

        io_data = IOData(
            output_path=plane_output_path,
            registered_binary_path=plane_output_path / CHANNEL_1_BINARY_FILENAME,
            plane_index=virtual_plane_index,
            sampling_rate=sampling_rate,
        )

        # Configures second channel binary paths if using two channels.
        if has_two_channels:
            io_data.registered_binary_path_channel_2 = plane_output_path / CHANNEL_2_BINARY_FILENAME

        # Populates MROI-specific fields if processing multi-ROI data.
        if acquisition.is_mroi:
            # Computes ROI index and physical plane index from the virtual plane index. Virtual planes are organized
            # as: ROI 0 plane 0, ROI 0 plane 1, ..., ROI 1 plane 0, ROI 1 plane 1, etc.
            roi_index = virtual_plane_index // acquisition.plane_number
            io_data.mroi_lines = acquisition.roi_lines[roi_index]
            io_data.mroi_y_offset = acquisition.roi_y_coordinates[roi_index]
            io_data.mroi_x_offset = acquisition.roi_x_coordinates[roi_index]

        runtime_data = SingleRecordingRuntimeData(
            output_path=plane_output_path,
            io=io_data,
        )

        context = RuntimeContext(
            configuration=configuration,
            acquisition=acquisition,
            runtime=runtime_data,
        )

        contexts.append(context)

    if persist:
        # Saves shared configuration and acquisition parameters to ensure they are always up to date.
        contexts[0].save_shared()

        # Saves each runtime to persist the initialized IO data.
        for context in contexts:
            context.save_runtime()
    else:
        # Worker entry (REMOTE mode): bootstrap must already exist on disk from a prior prepare step. Treats missing
        # runtime_data.yaml as a hard error rather than silently persisting concurrently alongside peer workers.
        for context in contexts:
            # io.output_path is always populated during per-plane construction above, so the guard is defensive only.
            if context.runtime.io.output_path is None:  # pragma: no cover
                continue
            runtime_yaml = context.runtime.io.output_path / SINGLE_RECORDING_RUNTIME_DATA_FILENAME
            if not runtime_yaml.exists():
                message = (
                    f"Unable to resolve single-recording contexts without bootstrap persistence. The runtime data "
                    f"file was not found at: {runtime_yaml}. Run prepare_single_recording_batch_tool before "
                    f"dispatching workers so the filesystem bootstrap is written exactly once in a single-threaded "
                    f"context."
                )
                console.error(message=message, error=FileNotFoundError)

    return contexts


def resolve_multi_recording_contexts(
    configuration: MultiRecordingConfiguration,
    target_recording_id: str | None = None,
    *,
    persist: bool = True,
) -> list[MultiRecordingRuntimeContext]:
    """Creates MultiRecordingRuntimeContext instances for recordings processed by the target multi-recording pipeline.

    Performs the initial setup for multi-recording processing: discovers cindra output
    directories for each recording, derives multi_recording output paths, and initializes
    MultiRecordingRuntimeContext instances.

    Notes:
        Each recording directory must contain exactly one cindra output directory with a
        combined_metadata.npz file from a completed single-recording pipeline run. The function extracts
        unique recording identifiers from the directory paths to distinguish recordings within the dataset.

        With ``persist=True`` (the default), the shared dataset configuration and every recording's runtime data
        file are saved to disk at the end of resolution, ensuring they reflect the current settings. This is the
        correct mode for single-threaded bootstrap (e.g., the prepare_multi_recording_batch_tool invocation).

        With ``persist=False``, no files are written. This mode is required for worker entry (REMOTE mode), because
        this resolver builds a context for every recording rather than for the one recording the worker was
        dispatched to process. A persisting worker would therefore write each peer recording's
        multi_recording_runtime_data.yaml from its own snapshot, overwriting whatever a peer had already recorded
        there with content that is whole and parsable but stale. Workers must therefore only *load* the bootstrap
        written by the earlier prepare step. When ``persist=False``, any missing
        multi_recording_runtime_data.yaml is treated as a hard error because it indicates
        prepare_multi_recording_batch_tool was not run first.

        ROI selection is performed as a separate step using select_recording_rois(), not during context resolution.

        When target_recording_id is provided, only the matching recording's CombinedData and runtime data are loaded.
        Non-matching recordings are skipped entirely. This avoids the overhead of loading large arrays for recordings
        that will not be used (e.g., during per-recording extraction).

    Args:
        configuration: The multi-recording pipeline configuration. Must have recording_directories and
            dataset_name configured in recording_io.
        target_recording_id: When provided, only resolves the context for the recording matching this identifier. The
            returned list contains a single element. When None (default), all recordings are resolved.
        persist: When True (default), writes the shared configuration and every resolved context's
            multi_recording_runtime_data.yaml at the end of resolution. When False, treats the call as load-only
            and raises FileNotFoundError if any expected runtime data file is missing. Worker entry points (REMOTE
            mode) must call with ``persist=False``. The prepare_multi_recording_batch_tool owns bootstrap
            persistence in a single-threaded context.

    Returns:
        A list of MultiRecordingRuntimeContext instances, one per recording (or one element when
        target_recording_id is set). Each context contains references to the shared configuration and a
        recording-specific MultiRecordingRuntimeData
        instance with MultiRecordingIOData fields initialized.

    Raises:
        FileNotFoundError: If no combined_metadata.npz file is found in a recording directory, or if
            ``persist=False`` and any resolved recording's multi_recording_runtime_data.yaml does not already
            exist on disk.
        RuntimeError: If multiple combined_metadata.npz files are found in a recording directory, or if recording paths
            do not contain unique identifying components.
        ValueError: If target_recording_id does not match any resolved recording identifier.
    """
    recording_directories = configuration.recording_io.recording_directories
    recording_ids = extract_unique_components(paths=recording_directories)
    dataset_name = configuration.recording_io.dataset_name.lower()

    # Resolves all cindra directories and output paths upfront. These are cheap path operations needed for every
    # recording regardless of target filtering, because dataset_output_paths stores the full set.
    data_paths: list[Path] = []
    output_paths: list[Path] = []
    for recording_directory in recording_directories:
        data_path = _find_cindra_directory(recording_directory=recording_directory)
        data_paths.append(data_path)
        output_paths.append(data_path / MULTI_RECORDING_DIRECTORY_NAME / dataset_name)

    # Validates the target recording ID before performing expensive I/O.
    if target_recording_id is not None and target_recording_id not in recording_ids:
        available_ids = list(recording_ids)
        message = (
            f"Unable to resolve multi-recording context for recording '{target_recording_id}'. The "
            f"provided recording_id does not match any resolved recording identifier. Available "
            f"recording IDs: {available_ids}."
        )
        console.error(message=message, error=ValueError)

    contexts: list[MultiRecordingRuntimeContext] = []

    for index, recording_id in enumerate(recording_ids):
        # Skips non-target recordings when a specific recording is requested.
        if target_recording_id is not None and recording_id != target_recording_id:
            continue

        data_path = data_paths[index]
        output_path = output_paths[index]

        # Loads single-recording combined data (scalars only). Arrays are loaded on demand by each pipeline function.
        combined_data = CombinedData.load(root_path=data_path)

        runtime_path = output_path / MULTI_RECORDING_RUNTIME_DATA_FILENAME
        if runtime_path.exists():
            # Loads existing runtime data (scalars only). Arrays are loaded on demand by each pipeline function.
            runtime = MultiRecordingRuntimeData.load(output_path=output_path)

            # Updates IO paths to reflect the current configuration's recording directories. This handles cases where
            # recording directories have changed or data was moved since the runtime was last saved.
            runtime.io.data_path = data_path
            runtime.io.dataset_output_paths = tuple(output_paths)
            runtime.io.mroi_region_borders = _compute_mroi_region_borders(data_path=data_path)

            # Injects the preloaded CombinedData.
            runtime.combined_data = combined_data

            contexts.append(MultiRecordingRuntimeContext(configuration=configuration, runtime=runtime))
            continue

        io_data = MultiRecordingIOData(
            recording_id=recording_id,
            data_path=data_path,
            dataset_name=dataset_name,
            dataset_output_paths=tuple(output_paths),
        )

        # Computes MROI region borders from acquisition parameters if applicable.
        io_data.mroi_region_borders = _compute_mroi_region_borders(data_path=data_path)

        runtime = MultiRecordingRuntimeData(output_path=output_path, io=io_data, combined_data=combined_data)

        ensure_directory_exists(path=output_path, is_file=False)

        contexts.append(MultiRecordingRuntimeContext(configuration=configuration, runtime=runtime))

    console.echo(
        message=f"Loaded existing multi-recording runtime data for {len(contexts)} recording(s).", level=LogLevel.INFO
    )

    if persist:
        # Saves shared configuration once via the first context to ensure it is always up to date.
        contexts[0].save_shared()

        # Saves each runtime to persist the fully-resolved IO data (including dataset_output_paths).
        for context in contexts:
            context.save_runtime()
    else:
        # Worker entry (REMOTE mode): bootstrap must already exist on disk from a prior prepare step. Treats missing
        # multi_recording_runtime_data.yaml as a hard error rather than silently persisting alongside peer workers,
        # which would overwrite each peer's file with this worker's own stale snapshot of it.
        for context in contexts:
            runtime_output_path = context.runtime.output_path
            # output_path is always populated during per-recording construction above, so the guard is defensive only.
            if runtime_output_path is None:  # pragma: no cover
                continue
            runtime_yaml = runtime_output_path / MULTI_RECORDING_RUNTIME_DATA_FILENAME
            if not runtime_yaml.exists():
                message = (
                    f"Unable to resolve multi-recording contexts without bootstrap persistence. The runtime data "
                    f"file was not found at: {runtime_yaml}. Run prepare_multi_recording_batch_tool before "
                    f"dispatching workers so the filesystem bootstrap is written exactly once in a single-threaded "
                    f"context."
                )
                console.error(message=message, error=FileNotFoundError)

    return contexts


def extract_unique_components(paths: list[Path] | tuple[Path, ...]) -> tuple[str, ...]:
    """Extracts the first component from the end of each input path that uniquely identifies each path globally.

    Notes:
        Adapts the multi-recording pipeline to directory structures where the unique
        recording identifier appears at different levels of the path hierarchy. For example, given paths
        like ``/data/day1/recording`` and ``/data/day2/recording``, the function identifies ``day1`` and
        ``day2`` as the unique components (not ``recording``,
        which is shared). This allows users to organize recordings using any naming convention, as long as each path
        contains at least one unique component somewhere in its hierarchy.

        The returned components become the tracker specifiers of the multi-recording extraction jobs, and a
        specifier is hashed into its job identifier alongside the job name with a colon joining the two. A component
        carrying a colon would therefore collide with a differently split pair, so it is rejected here rather than at
        identifier generation, where the diagnostic no longer names the directory that produced it.

        A directory listed twice shares every component with its duplicate, so it carries no unique component and is
        rejected, which keeps two jobs of the same dataset from collapsing onto one tracker record.

    Args:
        paths: A list or tuple of Path objects.

    Returns:
        A tuple of unique components, one for each path, stored in the same order as the input paths.

    Raises:
        RuntimeError: If one or more paths do not contain unique components, or if a resolved component contains a
            colon.
    """
    paths_list = list(paths)
    unique_components: list[str] = []

    for path_index, path in enumerate(paths_list):
        # Gets components from right to left.
        components = list(path.parts)[::-1]
        found_unique = False

        for component in components:
            # Checks if this component appears in any other path.
            is_unique = True

            for other_index, other_path in enumerate(paths_list):
                # Skips the path itself by position rather than by value, so a directory listed twice disqualifies
                # every component it shares with its duplicate instead of yielding the same specifier for both entries.
                if other_index == path_index:
                    continue

                # If the component appears anywhere in the other path, it is not unique.
                if component in other_path.parts:
                    is_unique = False
                    break

            if is_unique:
                if ":" in component:
                    message = (
                        f"Unable to extract a unique component from the given path: {path}. The resolved component "
                        f"'{component}' contains the ':' character, which the tracker reserves for joining a job "
                        f"name to its specifier. Rename the directory to remove the character."
                    )
                    console.error(message=message, error=RuntimeError)
                unique_components.append(component)
                found_unique = True
                break

        if not found_unique:
            message = f"Unable to extract a unique component from the given path: {path}."
            console.error(message=message, error=RuntimeError)

    return tuple(unique_components)


def resolve_recording_roots(paths: list[Path] | tuple[Path, ...]) -> tuple[Path, ...]:
    """Resolves a set of discovered marker-file directories to their recording root directories.

    Recording roots are the meaningful top-level directories that uniquely identify each recording session. A marker
    directory that is itself a pipeline output directory resolves to its parent, which is the path every downstream
    status, cleaning, and preparation tool expects. Any other marker directory, whose depth below the recording root
    is user-defined, is truncated at its deepest distinguishing component, which strips a common structural
    subdirectory without assuming any particular name.

    Notes:
        The output-directory roots are reported ahead of the roots inferred from the remaining directories, so the
        result follows those two groups rather than the order the paths arrived in. Each group is resolved by its own
        rule, and every consumer sorts the result before displaying it.

    Args:
        paths: Directories containing discovered marker files (e.g., parents of ``cindra_parameters.json``
            or ``combined_metadata.npz``). The pipeline resolves sub-paths to raw data and outputs internally, so
            callers should always work with recording roots rather than implementation-specific subdirectories.

    Returns:
        A deduplicated tuple of recording root paths, one per unique recording.
    """
    # The combination stage always writes combined_metadata.npz directly into <output_root>/cindra, so an output
    # marker's parent is the recording root by construction and needs no inference.
    roots: list[Path] = []
    raw_paths: list[Path] = []
    for path in paths:
        if path.name == OUTPUT_DIRECTORY_NAME:
            roots.append(path.parent)
        else:
            raw_paths.append(path)

    # Resolves the remaining directories against each other only, so an authoritative output root is never shortened
    # by a component a raw-data sibling happens to share. A set in which some path shares every one of its components
    # with its peers offers no ancestor to truncate at, so each of those paths stands as its own root.
    distinct_raw_paths = list(dict.fromkeys(raw_paths))
    if distinct_raw_paths:
        try:
            roots.extend(resolve_unique_roots(paths=distinct_raw_paths))
        except ValueError:
            roots.extend(distinct_raw_paths)

    unique_roots: list[Path] = []
    for root in roots:
        if root not in unique_roots:
            unique_roots.append(root)

    return tuple(unique_roots)


def load_acquisition_parameters(json_path: Path) -> AcquisitionParameters:
    """Loads acquisition parameters from a JSON file and validates all required fields.

    For single-ROI data, frame_rate, plane_number, and channel_number are required. For MROI data (roi_number > 1),
    roi_lines, roi_x_coordinates, and roi_y_coordinates are additionally required.

    Notes:
        The file is authored by the user or by a conversion script, so the counts it carries are checked for a positive
        numeric domain alongside their presence. The frame rate divides by the plane count and the plane and channel
        counts size the interleave cycle, so a zero or negative value otherwise surfaces deep inside context
        resolution as a division by zero or an empty context list.

    Args:
        json_path: The path to the JSON file containing acquisition parameters.

    Returns:
        An AcquisitionParameters instance populated from the JSON file.

    Raises:
        FileNotFoundError: If the JSON file does not exist.
        ValueError: If required fields are missing from the JSON data, if frame_rate is not a positive number, or if
            plane_number, channel_number, or roi_number is not a positive integer.
    """
    if not json_path.exists():
        message = f"Unable to load acquisition parameters. The file was not found: {json_path}."
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
    if not isinstance(frame_rate, (int, float)) or frame_rate <= 0:
        message = (
            f"Unable to load the acquisition parameters stored inside the file located at {json_path}. The "
            f"'frame_rate' field must be a positive number, but it is {frame_rate}."
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
    _validate_positive_count(value=plane_number, field_name="plane_number", json_path=json_path)

    # Extracts channel_number (required).
    channel_number = data.get("channel_number")
    if channel_number is None:
        message = (
            f"Unable to extract the required field 'channel_number' from the acquisition parameters file "
            f"located at {json_path}."
        )
        console.error(message=message, error=ValueError)
    _validate_positive_count(value=channel_number, field_name="channel_number", json_path=json_path)

    # Extracts roi_number (defaults to 1 for single-ROI).
    roi_number = data.get("roi_number", 1)
    _validate_positive_count(value=roi_number, field_name="roi_number", json_path=json_path)

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


def _validate_positive_count(value: object, field_name: str, json_path: Path) -> None:
    """Verifies that the given acquisition parameters field holds a positive whole number.

    Args:
        value: The value read from the acquisition parameters file.
        field_name: The name of the acquisition parameters field the value was read from.
        json_path: The path to the acquisition parameters file the value was read from.

    Raises:
        ValueError: If the value is not a positive integer.
    """
    if not isinstance(value, int) or value < 1:
        message = (
            f"Unable to load the acquisition parameters stored inside the file located at {json_path}. The "
            f"'{field_name}' field must be a positive integer, but it is {value}."
        )
        console.error(message=message, error=ValueError)


def _find_acquisition_parameters(data_path: Path) -> AcquisitionParameters:
    """Finds and loads acquisition parameters from the data directory.

    Args:
        data_path: The root directory to search for the acquisition parameters file.

    Returns:
        The loaded AcquisitionParameters instance.

    Raises:
        FileNotFoundError: If no acquisition parameters file is found.
        ValueError: If the data_path is not a directory, or if required fields are missing from the JSON file.
    """
    data_directory = find_data_directory(data_path=data_path)
    parameters_path = data_directory / PARAMETERS_FILENAME

    message = f"Found acquisition parameters at: {parameters_path}."
    console.echo(message=message, level=LogLevel.SUCCESS)

    return load_acquisition_parameters(json_path=parameters_path)


def _find_cindra_directory(recording_directory: Path) -> Path:
    """Discovers the cindra output directory within a recording directory tree.

    Searches recursively for the combined_metadata.npz file created by the single-recording pipeline's combination step.
    The cindra directory may be nested at an arbitrary depth below the recording root.

    Args:
        recording_directory: The path to the recording's root directory.

    Returns:
        The path to the cindra output directory that contains the combined_metadata.npz file.

    Raises:
        FileNotFoundError: If no combined_metadata.npz file is found under the recording directory.
        RuntimeError: If multiple combined_metadata.npz files are found under the recording directory.
    """
    matches = discover_marker_files(directory=recording_directory, marker_name=COMBINED_METADATA_FILENAME)

    if not matches:
        message = (
            f"Unable to locate cindra output for recording {recording_directory}. No "
            f"combined_metadata.npz file was found anywhere in the directory tree. Ensure the "
            f"single-recording pipeline has completed successfully for this recording before running "
            f"multi-recording processing."
        )
        console.error(message=message, error=FileNotFoundError)

    if len(matches) > 1:
        message = (
            f"Unable to locate cindra output for recording {recording_directory}. Found {len(matches)} "
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
    acquisition_path = data_path / ACQUISITION_PARAMETERS_FILENAME
    acquisition = AcquisitionParameters.from_yaml(file_path=acquisition_path)
    if not acquisition.is_mroi:
        return ()

    # Computes region borders from ROI x-coordinates. The borders are the x-coordinates where one region ends and
    # another begins, which are all x-coordinates except the minimum (leftmost region).
    sorted_x = sorted(acquisition.roi_x_coordinates)
    return tuple(sorted_x[1:])
