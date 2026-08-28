"""Provides assets for selecting ROIs from single-recording outputs for multi-recording tracking."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console

from ..layout import MULTI_RECORDING_DIRECTORY_NAME, MULTI_RECORDING_RUNTIME_DATA_FILENAME
from ..dataclasses import MultiRecordingRuntimeData

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    from numpy.typing import NDArray

    from ..dataclasses import (
        ROIStatistics,
        MultiRecordingConfiguration,
        MultiRecordingRuntimeContext,
    )


def select_recording_rois(contexts: list[MultiRecordingRuntimeContext]) -> None:
    """Selects ROIs from single-recording pipeline outputs that meet multi-recording tracking criteria.

    Performs ROI selection filtering on each recording using the ROI selection parameters from the configuration.
    The CombinedData for each recording is accessed from runtime.combined_data (loaded during context resolution),
    and the selected ROI indices are stored in runtime.io.selected_roi_indices (channel 1) and
    runtime.io.selected_roi_indices_channel_2 (channel 2, if available). Each processed recording's runtime data file
    is written to disk after its selection completes.

    Notes:
        Selection is an on-demand operation. When repeat_selection is False (default), recordings with existing ROI
        selections are skipped. When repeat_selection is True, selection is re-run for all recordings even if selections
        already exist.

        For recordings with two functional channels, both channels are filtered independently. Channel 2 uses its own
        probability_threshold_channel_2, maximum_size_channel_2, and mroi_region_margin_channel_2 parameters when they
        are configured, and falls back to the channel 1 parameters otherwise. The output messages report ROI counts
        for both channels when channel 2 data is present.

    Args:
        contexts: The recordings to process. Each context must have combined_data available in its runtime (set
            during context resolution).

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
        has_channel_1_selection = bool(runtime.io.selected_roi_indices)
        has_channel_2_data = (
            runtime.combined_data is not None and runtime.combined_data.extraction.roi_statistics_channel_2 is not None
        )
        has_channel_2_selection = bool(runtime.io.selected_roi_indices_channel_2)

        if has_channel_1_selection and (not has_channel_2_data or has_channel_2_selection) and not repeat_selection:
            channel_1_count = len(runtime.io.selected_roi_indices)
            channel_2_count = len(runtime.io.selected_roi_indices_channel_2)
            if channel_2_count:
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
            combined_data.extraction.memory_map_arrays(output_path=runtime.io.data_path)

        channel_1_count, channel_2_count = _filter_rois(runtime=runtime, configuration=configuration)

        if channel_2_count:
            count_message = f"{channel_1_count} channel 1 and {channel_2_count} channel 2 ROI candidates"
        else:
            count_message = f"{channel_1_count} ROI candidates"

        if repeat_selection:
            console.echo(message=f"Re-selected {count_message} for recording {recording_id}.", level=LogLevel.INFO)
        else:
            console.echo(message=f"Selected {count_message} for recording {recording_id}.", level=LogLevel.SUCCESS)

        context.save_runtime()

        # Releases combined extraction arrays to free memory.
        if context.runtime.combined_data is not None:  # pragma: no branch - _filter_rois rejects None combined_data.
            context.runtime.combined_data.extraction.release_arrays()


def clear_dataset_selection(dataset_path: Path) -> bool:
    """Clears the region selection one multi-recording dataset holds for one recording.

    Notes:
        A selection names regions by their position in the recording's own region list, so a detection run that
        rebuilds that list invalidates it. Clearing the selection makes the discovery stage select again for this
        recording, while the recording identity and dataset membership the same file carries stay in place.

    Args:
        dataset_path: The recording's directory inside one dataset tree, which holds that dataset's runtime data file.

    Returns:
        True when a selection was cleared, and False when the directory holds no runtime data file or no selection.
    """
    runtime_path = dataset_path / MULTI_RECORDING_RUNTIME_DATA_FILENAME
    if not runtime_path.is_file():
        return False

    runtime_data = MultiRecordingRuntimeData.from_yaml(file_path=runtime_path)
    if not runtime_data.io.selected_roi_indices and not runtime_data.io.selected_roi_indices_channel_2:
        return False

    runtime_data.io.selected_roi_indices = ()
    runtime_data.io.selected_roi_indices_channel_2 = ()
    runtime_data.to_yaml(file_path=runtime_path)
    return True


def clear_recording_selections(cindra_root: Path) -> int:
    """Clears the region selections every multi-recording dataset holds for one recording.

    Args:
        cindra_root: The recording's cindra output directory, which parents its multi_recording directory.

    Returns:
        The number of dataset selections cleared, which is zero when the recording belongs to no dataset.
    """
    datasets_path = cindra_root / MULTI_RECORDING_DIRECTORY_NAME
    if not datasets_path.is_dir():
        return 0

    return sum(clear_dataset_selection(dataset_path=dataset_path) for dataset_path in sorted(datasets_path.iterdir()))


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
    cross-recording tracking.

    Args:
        roi_statistics: The ROI statistics of the channel being filtered.
        cell_classification: The classification array for this channel. Each row contains [is_cell, probability] for
            one ROI. Only ROIs whose classifier probability meets or exceeds the threshold are retained.
        mroi_region_borders: The x-coordinates of MROI region borders. ROIs near these borders are filtered out
            to avoid tracking ambiguities. Pass an empty tuple for non-MROI recordings.
        probability_threshold: The minimum classifier probability required for an ROI to be selected.
        maximum_size: The maximum allowed ROI size in pixels. ROIs with more pixels are excluded.
        region_margin: The minimum distance in pixels between an ROI's centroid and MROI region borders.

    Returns:
        The indices into roi_statistics of the ROIs that passed every selection filter.
    """
    selected_indices: list[int] = []
    for index, roi in enumerate(roi_statistics):
        if cell_classification[index, 1] < probability_threshold:
            continue

        if roi.pixel_count > maximum_size:
            continue

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
        roi_statistics_channel_2 is present in the combined data, indicating the recording used two functional
        channels.

    Args:
        runtime: The per-recording runtime data. The io.selected_roi_indices and io.selected_roi_indices_channel_2
            fields of the input MultiRecordingRuntimeData instance are populated with the selected indices in-place.
        configuration: The multi-recording pipeline configuration containing ROI selection parameters.

    Returns:
        The number of ROIs selected from channel 1 and the number selected from channel 2, the latter being 0 when
        channel 2 data is not available.

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

    probability_threshold = configuration.roi_selection.probability_threshold
    maximum_size = configuration.roi_selection.maximum_size
    region_margin = configuration.roi_selection.mroi_region_margin
    mroi_region_borders = runtime.io.mroi_region_borders

    runtime.io.selected_roi_indices = _filter_channel_rois(
        roi_statistics=combined_data.extraction.roi_statistics,
        cell_classification=combined_data.extraction.cell_classification,
        mroi_region_borders=mroi_region_borders,
        probability_threshold=probability_threshold,
        maximum_size=maximum_size,
        region_margin=region_margin,
    )
    channel_1_count = len(runtime.io.selected_roi_indices)

    channel_2_count = 0
    if combined_data.extraction.roi_statistics_channel_2 is not None:
        if combined_data.extraction.cell_classification_channel_2 is None:
            message = (
                f"Unable to select channel 2 ROIs for recording {runtime.io.recording_id}. The combined "
                f"single-recording data contains channel 2 ROI statistics but no classification results. "
                f"Multi-recording processing requires classification to filter ROIs."
            )
            console.error(message=message, error=ValueError)

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
