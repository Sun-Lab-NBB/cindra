"""Provides functions for resolving multi-day pipeline runtime contexts from user configuration."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists

from ..dataclasses import (
    CombinedData,
    MultiDayIOData,
    MultiDayRuntimeData,
    MultiDayConfiguration,
    MultiDayRuntimeContext,
)
from ..dataclasses.single_day_configuration import AcquisitionParameters

if TYPE_CHECKING:
    from pathlib import Path

    from ..dataclasses import ROIStatistics


def _find_suite2p_directory(session_directory: Path) -> Path:
    """Discovers the suite2p output directory within a session directory tree.

    Searches recursively for the combined_metadata.npz file created by the single-day pipeline's combination step.
    Unlike the single-day pipeline, multi-day session paths are not pre-sanitized, so the suite2p directory may be
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


def resolve_multiday_contexts(configuration: MultiDayConfiguration) -> list[MultiDayRuntimeContext]:
    """Resolves MultiDayRuntimeContext instances for all sessions in the multi-day pipeline.

    Follows the single-day resolve_processing_contexts() load-or-create pattern. For each session directory in the
    configuration, discovers the suite2p output directory, derives the multiday output path, and either loads an
    existing MultiDayRuntimeData from disk or constructs a new one. New sessions undergo cell selection filtering
    and output directory creation.

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
        selected_cells = _select_session_cells(runtime=runtime, configuration=configuration)
        runtime.extraction.roi_statistics = selected_cells

        # Creates the output directory for this session.
        ensure_directory_exists(output_path)

        contexts.append(MultiDayRuntimeContext(configuration=configuration, runtime=runtime))
        cell_count = len(selected_cells)
        console.echo(
            message=f"Initialized multi-day runtime for session {session_id} with {cell_count} selected cells.",
            level=LogLevel.SUCCESS,
        )

    # Saves shared configuration once via the first context.
    contexts[0].save_shared()

    # Saves each runtime to persist the fully-resolved IO data (including dataset_output_paths).
    for context in contexts:
        context.save_runtime()

    return contexts


def _select_session_cells(
    runtime: MultiDayRuntimeData,
    configuration: MultiDayConfiguration,
) -> list[ROIStatistics]:
    """Selects cells from the single-day pipeline output that meet multi-day tracking criteria.

    Filters ROIs from the combined single-day data using the probability threshold, maximum size, and (for MROI
    recordings) region border margin specified in the configuration. This step is expected to discard some single-day
    ROIs because the multi-day pipeline typically uses more stringent cell identification criteria.

    Args:
        runtime: The per-session runtime data containing the loaded CombinedData.
        configuration: The multi-day pipeline configuration containing ROI selection parameters.

    Returns:
        A list of ROIStatistics instances that passed all filtering criteria.

    Raises:
        ValueError: If the combined data or its extraction data is not available.
    """
    if runtime.combined_data is None or runtime.combined_data.extraction.roi_statistics is None:
        message = (
            f"Unable to select session cells for session {runtime.io.session_id}. The combined single-day data "
            f"must be loaded before cell selection can be performed."
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

    return selected_cells


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
    if not acquisition_path.exists():
        return []

    acquisition = AcquisitionParameters.from_yaml(file_path=acquisition_path)
    if not acquisition.is_mroi:
        return []

    # Computes region borders from ROI x-coordinates. The borders are the x-coordinates where one region ends and
    # another begins, which are all x-coordinates except the minimum (leftmost region).
    sorted_x = sorted(acquisition.roi_x_coordinates)
    return sorted_x[1:]


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
