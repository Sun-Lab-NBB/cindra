"""Provides assets for selecting cells from single-day outputs for multi-day tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console

if TYPE_CHECKING:
    from ..dataclasses import (
        CombinedData,
        ROIStatistics,
        MultiDayRuntimeData,
        MultiDayConfiguration,
        MultiDayRuntimeContext,
    )


def _filter_cells(
    runtime: MultiDayRuntimeData,
    configuration: MultiDayConfiguration,
    combined_data: CombinedData,
) -> None:
    """Filters ROIs from combined single-day data using the multi-day ROI selection criteria.

    Filters ROIs using the probability threshold, maximum size, and (for MROI recordings) region border margin specified
    in the configuration. The filtered cells are stored directly in runtime.extraction.roi_statistics.

    Notes:
        This step is expected to discard some single-day ROIs because the multi-day pipeline typically uses more
        stringent cell identification criteria.

    Args:
        runtime: The per-session runtime data. The extraction.roi_statistics field is populated with the filtered cells.
        configuration: The multi-day pipeline configuration containing ROI selection parameters.
        combined_data: The combined single-day data containing the ROIs to filter.

    Raises:
        ValueError: If the combined data does not contain ROI statistics.
    """
    if combined_data.extraction.roi_statistics is None:
        message = (
            f"Unable to select cells for session {runtime.io.session_id}. The combined single-day data does not "
            f"contain ROI statistics. Ensure the single-day pipeline completed successfully."
        )
        console.error(message=message, error=ValueError)

    roi_statistics = combined_data.extraction.roi_statistics
    cell_classification = combined_data.extraction.cell_classification

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


def select_session_cells(contexts: list[MultiDayRuntimeContext]) -> None:
    """Selects cells from single-day pipeline outputs that meet multi-day tracking criteria.

    This function performs cell selection filtering on each session using the ROI selection parameters from the
    configuration. The CombinedData for each session is accessed from runtime.combined_data (loaded during context
    resolution), and the filtered results are stored in the session's runtime.extraction.roi_statistics.

    Notes:
        Selection is an on-demand operation. When repeat_selection is False (default), sessions with existing cell
        selections are skipped. When repeat_selection is True, selection is re-run for all sessions even if selections
        already exist.

    Args:
        contexts: The list of MultiDayRuntimeContext instances to process. Each context must have combined_data
            available in its runtime (set during context resolution).

    Raises:
        ValueError: If combined_data is not available in the runtime or does not contain ROI statistics.
    """
    if not contexts:
        return

    configuration = contexts[0].configuration
    repeat_selection = configuration.session_io.repeat_selection

    for context in contexts:
        runtime = context.runtime
        session_id = runtime.io.session_id

        # Checks if cell selection already exists and repeat_selection is not enabled.
        if runtime.extraction.roi_statistics is not None and not repeat_selection:
            cell_count = len(runtime.extraction.roi_statistics)
            console.echo(
                message=f"Session {session_id} already has {cell_count} selected cells. Skipping cell selection.",
                level=LogLevel.INFO,
            )
            continue

        # Validates that CombinedData is available in the runtime (set during context resolution).
        combined_data = runtime.combined_data
        if combined_data is None:
            message = (
                f"Unable to select cells for session {session_id}. The combined_data is not available in the runtime. "
                f"Ensure context resolution completed successfully before calling this function."
            )
            console.error(message=message, error=ValueError)

        # Performs cell selection filtering.
        _filter_cells(runtime=runtime, configuration=configuration, combined_data=combined_data)

        cell_count = len(runtime.extraction.roi_statistics) if runtime.extraction.roi_statistics else 0
        if repeat_selection:
            console.echo(
                message=f"Re-selected {cell_count} cell candidates for session {session_id}.",
                level=LogLevel.INFO,
            )
        else:
            console.echo(
                message=f"Selected {cell_count} cell candidates for session {session_id}.",
                level=LogLevel.SUCCESS,
            )

        # Saves the updated runtime data with the selected cells.
        context.save_runtime()
