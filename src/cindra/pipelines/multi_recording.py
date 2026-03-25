"""Provides the high-level API for the multi-recording processing pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_time import PrecisionTimer, TimerPrecisions, get_timestamp
from ataraxis_base_utilities import LogLevel, console

from ..io import select_recording_rois, resolve_multi_recording_contexts
from ..detection import track_rois_across_recordings
from ..extraction import extract_traces
from ..registration import register_recordings, project_templates_to_recordings

if TYPE_CHECKING:
    from ..dataclasses import MultiRecordingConfiguration


def discover_multi_recording_cells(configuration: MultiRecordingConfiguration) -> None:  # pragma: no cover
    """Discovers reliably identifiable ROIs and tracks them across the processed set of recordings.

    Notes:
        This function executes the first phase of the multi-recording pipeline: it discovers and tracks stable ROIs
        across the processed set of recordings. This process generates the ROIs used during the second processing
        phase (extraction) to iteratively extract the fluorescence of each tracked ROI from each processed recording.

    Args:
        configuration: The multi-recording pipeline configuration.
    """
    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    console.echo(message="Initializing multi-recording discovery phase...", level=LogLevel.INFO)

    # Resolves or reloads MultiRecordingRuntimeContext instances for all recordings. Saves configuration and runtime
    # data to disk during resolution.
    contexts = resolve_multi_recording_contexts(configuration=configuration)

    # Filters ROIs from each recording's single-recording outputs based on the configured selection criteria. Respects
    # the repeat_selection flag to skip recordings with existing selections.
    select_recording_rois(contexts=contexts)

    # Registers all recordings to a shared visual space using diffeomorphic demons registration and applies the
    # deformation fields to transform reference images and ROI masks.
    register_recordings(contexts=contexts)

    # Clusters ROIs across recordings in the shared deformed visual space and generates template masks for ROIs that
    # can be reliably identified across recordings.
    track_rois_across_recordings(contexts=contexts)

    # Projects template masks from the shared visual space back to each recording's original coordinate system for
    # fluorescence extraction.
    project_templates_to_recordings(contexts=contexts)

    # Records total discovery time and processing timestamp for each context.
    total_discovery_time = int(timer.elapsed)
    for context in contexts:
        context.runtime.timing.total_discovery_time = total_discovery_time
        context.runtime.timing.date_processed = str(get_timestamp())
        context.save_runtime()

    console.echo(
        message=f"Multi-recording discovery: complete. Total time: {total_discovery_time} seconds.",
        level=LogLevel.SUCCESS,
    )


def extract_multi_recording_fluorescence(  # pragma: no cover
    configuration: MultiRecordingConfiguration, recording_id: str
) -> None:
    """Extracts fluorescence data from ROIs tracked across imaging recordings for the specified recording.

    Notes:
        This function executes the second phase of the multi-recording pipeline: it locates the runtime context
        matching the input recording_id and extracts the fluorescence of the ROIs tracked across recordings from
        the processed recording. The discovery phase must have completed before attempting extraction. Multiple
        recordings can be processed in parallel, but each recording may use significant memory and CPU resources.

    Args:
        configuration: The multi-recording pipeline configuration.
        recording_id: The unique identifier of the recording for which to extract fluorescence data. Must match
            one of the recording IDs assigned during context resolution.

    Raises:
        ValueError: If the target recording_id does not match any resolved recording context.
        RuntimeError: If backward-transformed ROI statistics are not available, indicating the discovery phase has
            not completed.
    """
    # Reloads only the target recording's context from disk. The target_recording_id parameter avoids loading
    # CombinedData and runtime arrays for every other recording in the dataset.
    contexts = resolve_multi_recording_contexts(configuration=configuration, target_recording_id=recording_id)
    target_context = contexts[0]

    # Memory-maps extraction arrays from disk. resolve_multi_recording_contexts() only loads YAML scalars, so
    # roi_statistics will be None until arrays are explicitly loaded. Uses memory mapping because the data is only
    # needed for validation here; extract_traces() reloads what it needs independently.
    if target_context.runtime.output_path is not None:
        target_context.runtime.extraction.memory_map_arrays(target_context.runtime.output_path)

    # Validates that backward-transformed ROI statistics exist from the discovery phase.
    if target_context.runtime.extraction.roi_statistics is None:
        message = (
            f"Unable to extract multi-recording fluorescence for recording "
            f"'{recording_id}'. Backward-transformed ROI statistics are not available. "
            f"Ensure the multi-recording discovery phase has been completed before "
            f"running extraction."
        )
        console.error(message=message, error=RuntimeError)

    # Delegates to the unified extraction entry point, which dispatches to _extract_multi_recording internally. The
    # extraction function handles fluorescence extraction, deconvolution, timing, and runtime saving.
    extract_traces(context=target_context)
