"""Provides assets for selecting ROIs from single-recording outputs for multi-recording tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from ..dataclasses import (
        ROIStatistics,
        MultiRecordingRuntimeData,
        MultiRecordingConfiguration,
        MultiRecordingRuntimeContext,
    )


def select_recording_rois(contexts: list[MultiRecordingRuntimeContext]) -> None:  # pragma: no cover
    """Selects ROIs from single-recording pipeline outputs that meet multi-recording tracking criteria.

    This function performs ROI selection filtering on each recording using the ROI selection parameters from the
    configuration. The CombinedData for each recording is accessed from runtime.combined_data (loaded during context
    resolution), and the selected ROI indices are stored in runtime.io.selected_roi_indices (channel 1) and
    runtime.io.selected_roi_indices_channel_2 (channel 2, if available).

    Notes:
        Selection is an on-demand operation. When repeat_selection is False (default), recordings with existing ROI
        selections are skipped. When repeat_selection is True, selection is re-run for all recordings even if selections
        already exist.

        For recordings with two functional channels, both channels are filtered independently using the same selection
        criteria. The output messages report ROI counts for both channels when channel 2 data is present.

    Args:
        contexts: The list of MultiRecordingRuntimeContext instances to process. Each context must have combined_data
            available in its runtime (set during context resolution).

    Raises:
        ValueError: If combined_data is not available in the runtime, does not contain ROI statistics, or does not
            contain classification results.
    """
    if not contexts:
        return

    configuration = contexts[0].configuration
    repeat_selection = configuration.recording_io.repeat_selection

    for context in contexts:
        runtime = context.runtime
        recording_id = runtime.io.recording_id

        # Checks if ROI selection already exists and repeat_selection is not enabled. Both channel 1 and channel 2
        # (if applicable) must have existing selections to skip.
        has_channel_1_selection = len(runtime.io.selected_roi_indices) > 0
        has_channel_2_data = (
            runtime.combined_data is not None and runtime.combined_data.extraction.roi_statistics_channel_2 is not None
        )
        has_channel_2_selection = len(runtime.io.selected_roi_indices_channel_2) > 0

        # Skips if channel 1 has selections AND (no channel 2 data OR channel 2 has selections).
        if has_channel_1_selection and (not has_channel_2_data or has_channel_2_selection) and not repeat_selection:
            channel_1_count = len(runtime.io.selected_roi_indices)
            channel_2_count = len(runtime.io.selected_roi_indices_channel_2)
            if channel_2_count > 0:
                message = (
                    f"Recording {recording_id} already has {channel_1_count} channel 1 and {channel_2_count} channel 2 "
                    f"selected ROIs. Skipping ROI selection."
                )
            else:
                message = (
                    f"Recording {recording_id} already has {channel_1_count} selected ROIs. Skipping ROI selection."
                )
            console.echo(message=message, level=LogLevel.INFO)
            continue

        # Memory-maps combined extraction arrays needed for ROI selection (roi_statistics, classification).
        combined_data = runtime.combined_data
        if combined_data is not None and runtime.io.data_path is not None:
            combined_data.extraction.memory_map_arrays(runtime.io.data_path)

        # Performs ROI selection filtering for both channels.
        channel_1_count, channel_2_count = _filter_rois(runtime=runtime, configuration=configuration)

        # Formats output message based on whether channel 2 data is present.
        if channel_2_count > 0:
            count_message = f"{channel_1_count} channel 1 and {channel_2_count} channel 2 ROI candidates"
        else:
            count_message = f"{channel_1_count} ROI candidates"

        if repeat_selection:
            console.echo(message=f"Re-selected {count_message} for recording {recording_id}.", level=LogLevel.INFO)
        else:
            console.echo(message=f"Selected {count_message} for recording {recording_id}.", level=LogLevel.SUCCESS)

        # Saves the updated runtime data with the selected ROI indices.
        context.save_runtime()

        # Releases combined extraction arrays to free memory.
        if context.runtime.combined_data is not None:
            context.runtime.combined_data.extraction.release_arrays()


def _filter_channel_rois(
    roi_statistics: list[ROIStatistics],
    cell_classification: NDArray[np.float32],
    mroi_region_borders: tuple[int, ...],
    probability_threshold: float,
    maximum_size: int,
    region_margin: int,
) -> tuple[int, ...]:
    """Filters ROIs from a single channel using the multi-recording ROI selection criteria.

    Applies probability threshold, maximum size, and MROI region border margin filters to select ROIs suitable for
    cross-recording tracking. This helper function handles filtering for one channel and is called separately for
    channel 1 and channel 2 data.

    Args:
        roi_statistics: The list of ROIStatistics instances to filter.
        cell_classification: The classification array for this channel. Each row contains [probability, is_cell] for
            one ROI. Only ROIs whose classifier probability exceeds the threshold are retained.
        mroi_region_borders: The x-coordinates of MROI region borders. ROIs near these borders are filtered out
            to avoid tracking ambiguities. Pass an empty tuple for non-MROI recordings.
        probability_threshold: The minimum classifier probability required for an ROI to be selected.
        maximum_size: The maximum allowed ROI size in pixels. ROIs with more pixels are excluded.
        region_margin: The minimum distance in pixels between an ROI's centroid and MROI region borders.

    Returns:
        A tuple of indices into roi_statistics for ROIs that passed all selection filters.
    """
    # Filters ROIs by classifier probability and pixel count.
    selected_indices: list[int] = []
    for index, roi in enumerate(roi_statistics):
        # Applies the probability threshold filter.
        if cell_classification[index, 0] < probability_threshold:
            continue

        # Applies the maximum size filter.
        if roi.pixel_count >= maximum_size:
            continue

        # Applies MROI region border filter if applicable.
        if mroi_region_borders and not all(
            abs(roi.mask.centroid[1] - border) > region_margin for border in mroi_region_borders
        ):
            continue

        selected_indices.append(index)

    return tuple(selected_indices)


def _filter_rois(
    runtime: MultiRecordingRuntimeData,
    configuration: MultiRecordingConfiguration,
) -> tuple[int, int]:
    """Filters ROIs from combined single-recording data using the multi-recording ROI selection criteria.

    Filters ROIs from both channel 1 and channel 2 (if available) using the probability threshold, maximum size, and
    (for MROI recordings) region border margin specified in the configuration. The selected ROI indices are stored
    in runtime.io.selected_roi_indices and runtime.io.selected_roi_indices_channel_2.

    Notes:
        This step is expected to discard some single-recording ROIs because the multi-recording pipeline
        typically uses more stringent ROI identification criteria. Channel 2 filtering only occurs when
        roi_statistics_channel_2 is
        present in the combined data, indicating the recording used two functional channels.

    Args:
        runtime: The per-recording runtime data. The io.selected_roi_indices and io.selected_roi_indices_channel_2
            fields of the input MultiRecordingRuntimeData instance are populated with the selected indices in-place.
        configuration: The multi-recording pipeline configuration containing ROI selection parameters.

    Returns:
        A tuple of (channel_1_count, channel_2_count) indicating how many ROIs were selected from each channel.
        Channel 2 count is 0 if channel 2 data is not available.

    Raises:
        ValueError: If combined_data is not available, does not contain ROI statistics, or does not contain
            classification results. Multi-recording processing requires both ROI statistics and classification data.
    """
    combined_data = runtime.combined_data
    if combined_data is None:
        message = (
            f"Unable to select ROIs for recording {runtime.io.recording_id}. The combined_data is not available in the "
            f"runtime. Ensure context resolution completed successfully before calling this function."
        )
        console.error(message=message, error=ValueError)

    if combined_data.extraction.roi_statistics is None:
        message = (
            f"Unable to select ROIs for recording {runtime.io.recording_id}. The combined "
            f"single-recording data does not contain ROI statistics. Ensure the single-recording "
            f"pipeline completed successfully."
        )
        console.error(message=message, error=ValueError)

    if combined_data.extraction.cell_classification is None:
        message = (
            f"Unable to select ROIs for recording {runtime.io.recording_id}. The combined "
            f"single-recording data does not contain cell classification results. Multi-recording "
            f"processing requires classification to filter ROIs."
        )
        console.error(message=message, error=ValueError)

    # Extracts filtering parameters from configuration.
    probability_threshold = configuration.roi_selection.probability_threshold
    maximum_size = configuration.roi_selection.maximum_size
    region_margin = configuration.roi_selection.mroi_region_margin
    mroi_region_borders = runtime.io.mroi_region_borders

    # Filters channel 1 ROIs and stores indices.
    runtime.io.selected_roi_indices = _filter_channel_rois(
        roi_statistics=combined_data.extraction.roi_statistics,
        cell_classification=combined_data.extraction.cell_classification,
        mroi_region_borders=mroi_region_borders,
        probability_threshold=probability_threshold,
        maximum_size=maximum_size,
        region_margin=region_margin,
    )
    channel_1_count = len(runtime.io.selected_roi_indices)

    # Filters channel 2 ROIs if two-functional-channel data is available.
    channel_2_count = 0
    if combined_data.extraction.roi_statistics_channel_2 is not None:
        if combined_data.extraction.cell_classification_channel_2 is None:
            message = (
                f"Unable to select channel 2 ROIs for recording {runtime.io.recording_id}. The combined "
                f"single-recording data contains channel 2 ROI statistics but no classification results. "
                f"Multi-recording processing requires classification to filter ROIs."
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
            roi_selection.maximum_size_channel_2 if roi_selection.maximum_size_channel_2 is not None else maximum_size
        )
        channel_2_region_margin = (
            roi_selection.mroi_region_margin_channel_2
            if roi_selection.mroi_region_margin_channel_2 is not None
            else region_margin
        )

        runtime.io.selected_roi_indices_channel_2 = _filter_channel_rois(
            roi_statistics=combined_data.extraction.roi_statistics_channel_2,
            cell_classification=combined_data.extraction.cell_classification_channel_2,
            mroi_region_borders=mroi_region_borders,
            probability_threshold=channel_2_probability_threshold,
            maximum_size=channel_2_maximum_size,
            region_margin=channel_2_region_margin,
        )
        channel_2_count = len(runtime.io.selected_roi_indices_channel_2)

    return channel_1_count, channel_2_count
