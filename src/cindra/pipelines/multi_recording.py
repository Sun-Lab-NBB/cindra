"""Provides the high-level API for the multi-recording processing pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_time import PrecisionTimer, TimerPrecisions, get_timestamp
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]
from ataraxis_base_utilities import LogLevel, console

from ..io import select_recording_rois, resolve_multi_recording_contexts
from ..detection import track_rois_across_recordings
from ..extraction import extract_traces
from ..registration import register_recordings, project_templates_to_recordings

if TYPE_CHECKING:
    from ..dataclasses import MultiRecordingConfiguration


def discover_multi_recording_cells(configuration: MultiRecordingConfiguration, *, workers: int) -> None:
    """Discovers reliably identifiable ROIs and tracks them across the processed set of recordings.

    Notes:
        This function executes the first phase of the multi-recording pipeline: it discovers and tracks stable ROIs
        across the processed set of recordings. This process generates the ROIs used during the second processing
        phase (extraction) to iteratively extract the fluorescence of each tracked ROI from each processed recording.

    Args:
        configuration: The multi-recording pipeline configuration.
        workers: The number of parallel workers allocated to this discovery job. Must be a positive integer, which the
            caller resolves before invoking this function.
    """
    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    console.echo(message="Initializing multi-recording discovery phase...", level=LogLevel.INFO)

    # Resolves or reloads MultiRecordingRuntimeContext instances for all recordings. The outer pipeline entry
    # (run_multi_recording_pipeline) or the prepare_multi_recording_batch_tool already wrote the shared configuration
    # and every recording's multi_recording_runtime_data.yaml, so this call is load-only to avoid racing against
    # peer worker threads on the same YAML files.
    contexts = resolve_multi_recording_contexts(configuration=configuration, persist=False)

    # Confines the linear-algebra backends to the allocated worker budget for the whole stage. The batch engine pins
    # every worker process to one backend thread, so the demons registration and the cross-recording clustering would
    # otherwise run their matrix work single-threaded, while the same code invoked outside that engine would size
    # those backends to the whole host instead of to the job.
    with threadpool_limits(limits=workers):
        # Filters ROIs from each recording's single-recording outputs based on the configured selection criteria.
        # Respects the repeat_selection flag to skip recordings with existing selections.
        select_recording_rois(contexts=contexts)

        # Registers all recordings to a shared visual space using diffeomorphic demons registration and applies the
        # deformation fields to transform reference images and ROI masks.
        register_recordings(contexts=contexts, workers=workers)

        # Clusters ROIs across recordings in the shared deformed visual space and generates template masks for ROIs
        # that can be reliably identified across recordings.
        track_rois_across_recordings(contexts=contexts)

        # Projects template masks from the shared visual space back to each recording's original coordinate system for
        # fluorescence extraction.
        project_templates_to_recordings(contexts=contexts, workers=workers)

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


def extract_multi_recording_fluorescence(
    configuration: MultiRecordingConfiguration,
    recording_id: str,
    *,
    workers: int,
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
        workers: The number of parallel workers allocated to this extraction job. Must be a positive integer, which the
            caller resolves before invoking this function.

    Raises:
        ValueError: If the target recording_id does not match any resolved recording context.
        RuntimeError: If backward-transformed ROI statistics are not available, indicating the discovery phase has
            not completed.
    """
    # Reloads only the target recording's context from disk. The target_recording_id parameter avoids loading
    # CombinedData and runtime arrays for every other recording in the dataset. The outer pipeline entry
    # (run_multi_recording_pipeline) or the prepare_multi_recording_batch_tool already wrote the shared configuration
    # and the target recording's multi_recording_runtime_data.yaml. This call is therefore load-only, so that peer
    # worker threads do not race on the same YAML files, because every EXTRACT worker would otherwise re-save the
    # shared configuration.
    contexts = resolve_multi_recording_contexts(
        configuration=configuration,
        target_recording_id=recording_id,
        persist=False,
    )
    target_context = contexts[0]

    # Loads the extraction arrays from disk. resolve_multi_recording_contexts() only loads YAML scalars, so
    # roi_statistics is None until the arrays are read: memory_map_arrays() eagerly loads the ROI statistics archives
    # (.npz cannot be memory-mapped) and memory-maps the classification arrays. The validation below and the
    # extraction itself both read these arrays, and extract_traces() skips its own load while roi_statistics is set.
    # pragma justification: the resolved runtime context always carries a configured output path.
    if target_context.runtime.output_path is not None:  # pragma: no branch
        target_context.runtime.extraction.memory_map_arrays(output_path=target_context.runtime.output_path)

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
    # extraction function handles fluorescence extraction, deconvolution, timing, and runtime saving. The
    # linear-algebra backends are confined to the allocated worker budget for the same reason the discovery stage
    # confines them, since the batch engine pins every worker process to one backend thread.
    with threadpool_limits(limits=workers):
        extract_traces(context=target_context, workers=workers)
