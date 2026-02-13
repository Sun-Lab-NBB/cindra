"""Provides assets for selecting cells from single-day outputs for multi-day tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ..dataclasses import (
        ROIStatistics,
        MultiDayRuntimeData,
        MultiDayConfiguration,
        MultiDayRuntimeContext,
    )


def _filter_channel_cells(
    roi_statistics: list[ROIStatistics],
    cell_classification: NDArray[np.float32],
    mroi_region_borders: list[int],
    probability_threshold: float,
    maximum_size: int,
    region_margin: int,
) -> list[ROIStatistics]:
    """Filters ROIs from a single channel using the multi-day ROI selection criteria.

    Applies probability threshold, maximum size, and MROI region border margin filters to select ROIs suitable for
    cross-session tracking. This helper function handles filtering for one channel and is called separately for
    channel 1 and channel 2 data.

    Args:
        roi_statistics: The list of ROIStatistics instances to filter.
        cell_classification: The classification array for this channel. Each row contains [probability, is_cell] for
            one ROI. Only ROIs whose classifier probability exceeds the threshold are retained.
        mroi_region_borders: The x-coordinates of MROI region borders. ROIs near these borders are filtered out
            to avoid tracking ambiguities. Pass an empty list for non-MROI recordings.
        probability_threshold: The minimum classifier probability required for an ROI to be selected.
        maximum_size: The maximum allowed ROI size in pixels. ROIs with more pixels are excluded.
        region_margin: The minimum distance in pixels between an ROI's centroid and MROI region borders.

    Returns:
        A list of ROIStatistics instances that passed all selection filters.
    """
    # Filters ROIs by classifier probability and pixel count.
    selected_cells: list[ROIStatistics] = []
    for index, roi in enumerate(roi_statistics):
        # Applies the probability threshold filter.
        if cell_classification[index, 0] < probability_threshold:
            continue

        # Applies the maximum size filter.
        if roi.pixel_count >= maximum_size:
            continue

        selected_cells.append(roi)

    # Filters ROIs near MROI region borders if applicable.
    if mroi_region_borders:
        selected_cells = [
            cell
            for cell in selected_cells
            if all(abs(cell.centroid[1] - border) > region_margin for border in mroi_region_borders)
        ]

    return selected_cells


def _filter_cells(
    runtime: MultiDayRuntimeData,
    configuration: MultiDayConfiguration,
) -> tuple[int, int]:
    """Filters ROIs from combined single-day data using the multi-day ROI selection criteria.

    Filters ROIs from both channel 1 and channel 2 (if available) using the probability threshold, maximum size, and
    (for MROI recordings) region border margin specified in the configuration. The filtered cells are stored directly
    in runtime.extraction.roi_statistics and runtime.extraction.roi_statistics_channel_2.

    Notes:
        This step is expected to discard some single-day ROIs because the multi-day pipeline typically uses more
        stringent cell identification criteria. Channel 2 filtering only occurs when roi_statistics_channel_2 is
        present in the combined data, indicating the recording used two functional channels.

    Args:
        runtime: The per-session runtime data. The extraction.roi_statistics and extraction.roi_statistics_channel_2
            fields of the input MultiDayRuntimeData instance are populated with the filtered cells in-place.
        configuration: The multi-day pipeline configuration containing ROI selection parameters.

    Returns:
        A tuple of (channel_1_count, channel_2_count) indicating how many cells were selected from each channel.
        Channel 2 count is 0 if channel 2 data is not available.

    Raises:
        ValueError: If combined_data is not available, does not contain ROI statistics, or does not contain
            classification results. Multi-day processing requires both ROI statistics and classification data.
    """
    combined_data = runtime.combined_data
    if combined_data is None:
        message = (
            f"Unable to select cells for session {runtime.io.session_id}. The combined_data is not available in the "
            f"runtime. Ensure context resolution completed successfully before calling this function."
        )
        console.error(message=message, error=ValueError)

    if combined_data.extraction.roi_statistics is None:
        message = (
            f"Unable to select cells for session {runtime.io.session_id}. The combined single-day data does not "
            f"contain ROI statistics. Ensure the single-day pipeline completed successfully."
        )
        console.error(message=message, error=ValueError)

    if combined_data.extraction.cell_classification is None:
        message = (
            f"Unable to select cells for session {runtime.io.session_id}. The combined single-day data does not "
            f"contain cell classification results. Multi-day processing requires classification to filter cells."
        )
        console.error(message=message, error=ValueError)

    # Extracts filtering parameters from configuration.
    probability_threshold = configuration.roi_selection.probability_threshold
    maximum_size = configuration.roi_selection.maximum_size
    region_margin = configuration.roi_selection.mroi_region_margin
    mroi_region_borders = runtime.io.mroi_region_borders

    # Filters channel 1 cells.
    runtime.extraction.roi_statistics = _filter_channel_cells(
        roi_statistics=combined_data.extraction.roi_statistics,
        cell_classification=combined_data.extraction.cell_classification,
        mroi_region_borders=mroi_region_borders,
        probability_threshold=probability_threshold,
        maximum_size=maximum_size,
        region_margin=region_margin,
    )
    channel_1_count = len(runtime.extraction.roi_statistics)

    # Filters channel 2 cells if two-functional-channel data is available.
    channel_2_count = 0
    if combined_data.extraction.roi_statistics_channel_2 is not None:
        if combined_data.extraction.cell_classification_channel_2 is None:
            message = (
                f"Unable to select channel 2 cells for session {runtime.io.session_id}. The combined single-day data "
                f"contains channel 2 ROI statistics but no classification results. Multi-day processing requires "
                f"classification to filter cells."
            )
            console.error(message=message, error=ValueError)

        # Uses channel 2 specific parameters if configured, otherwise falls back to channel 1 parameters.
        roi_selection = configuration.roi_selection
        channel_2_probability_threshold = (
            roi_selection.probability_threshold_channel_2
            if roi_selection.probability_threshold_channel_2 is not None
            else probability_threshold
        )
        channel_2_maximum_size = (
            roi_selection.maximum_size_channel_2
            if roi_selection.maximum_size_channel_2 is not None
            else maximum_size
        )
        channel_2_region_margin = (
            roi_selection.mroi_region_margin_channel_2
            if roi_selection.mroi_region_margin_channel_2 is not None
            else region_margin
        )

        runtime.extraction.roi_statistics_channel_2 = _filter_channel_cells(
            roi_statistics=combined_data.extraction.roi_statistics_channel_2,
            cell_classification=combined_data.extraction.cell_classification_channel_2,
            mroi_region_borders=mroi_region_borders,
            probability_threshold=channel_2_probability_threshold,
            maximum_size=channel_2_maximum_size,
            region_margin=channel_2_region_margin,
        )
        channel_2_count = len(runtime.extraction.roi_statistics_channel_2)

    return channel_1_count, channel_2_count


def select_session_cells(contexts: list[MultiDayRuntimeContext]) -> None:
    """Selects cells from single-day pipeline outputs that meet multi-day tracking criteria.

    This function performs cell selection filtering on each session using the ROI selection parameters from the
    configuration. The CombinedData for each session is accessed from runtime.combined_data (loaded during context
    resolution), and the filtered results are stored in runtime.extraction.roi_statistics (channel 1) and
    runtime.extraction.roi_statistics_channel_2 (channel 2, if available).

    Notes:
        Selection is an on-demand operation. When repeat_selection is False (default), sessions with existing cell
        selections are skipped. When repeat_selection is True, selection is re-run for all sessions even if selections
        already exist.

        For recordings with two functional channels, both channels are filtered independently using the same selection
        criteria. The output messages report cell counts for both channels when channel 2 data is present.

    Args:
        contexts: The list of MultiDayRuntimeContext instances to process. Each context must have combined_data
            available in its runtime (set during context resolution).

    Raises:
        ValueError: If combined_data is not available in the runtime, does not contain ROI statistics, or does not
            contain classification results.
    """
    if not contexts:
        return

    configuration = contexts[0].configuration
    repeat_selection = configuration.session_io.repeat_selection

    for context in contexts:
        runtime = context.runtime
        session_id = runtime.io.session_id

        # Checks if cell selection already exists and repeat_selection is not enabled. Both channel 1 and channel 2
        # (if applicable) must have existing selections to skip.
        has_channel_1_selection = runtime.extraction.roi_statistics is not None
        has_channel_2_data = (
            runtime.combined_data is not None and runtime.combined_data.extraction.roi_statistics_channel_2 is not None
        )
        has_channel_2_selection = runtime.extraction.roi_statistics_channel_2 is not None

        # Skips if channel 1 has selections AND (no channel 2 data OR channel 2 has selections).
        if has_channel_1_selection and (not has_channel_2_data or has_channel_2_selection) and not repeat_selection:
            channel_1_count = len(runtime.extraction.roi_statistics) if runtime.extraction.roi_statistics else 0
            channel_2_count = (
                len(runtime.extraction.roi_statistics_channel_2) if runtime.extraction.roi_statistics_channel_2 else 0
            )
            if channel_2_count > 0:
                message = (
                    f"Session {session_id} already has {channel_1_count} channel 1 and {channel_2_count} channel 2 "
                    f"selected cells. Skipping cell selection."
                )
            else:
                message = f"Session {session_id} already has {channel_1_count} selected cells. Skipping cell selection."
            console.echo(message=message, level=LogLevel.INFO)
            continue

        # Performs cell selection filtering for both channels.
        channel_1_count, channel_2_count = _filter_cells(runtime=runtime, configuration=configuration)

        # Formats output message based on whether channel 2 data is present.
        if channel_2_count > 0:
            count_message = f"{channel_1_count} channel 1 and {channel_2_count} channel 2 cell candidates"
        else:
            count_message = f"{channel_1_count} cell candidates"

        if repeat_selection:
            console.echo(message=f"Re-selected {count_message} for session {session_id}.", level=LogLevel.INFO)
        else:
            console.echo(message=f"Selected {count_message} for session {session_id}.", level=LogLevel.SUCCESS)

        # Saves the updated runtime data with the selected cells.
        context.save_runtime()
