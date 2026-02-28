"""Provides the high-level API for the multi-day processing pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ataraxis_time import PrecisionTimer, TimerPrecisions, get_timestamp
from ataraxis_base_utilities import LogLevel, console

from ..io import select_session_cells, resolve_multiday_contexts
from ..detection import track_rois_across_sessions
from ..extraction import extract_traces
from ..registration import register_sessions, project_templates_to_sessions

if TYPE_CHECKING:
    from ..dataclasses import MultiDayConfiguration


def discover_multiday_cells(configuration: MultiDayConfiguration) -> None:
    """Discovers reliably identifiable cells and tracks them across the processed set of sessions.

    Notes:
        This function executes the first phase of the multi-day pipeline: it discovers and tracks stable ROIs across the
        processed set of sessions. This process generates the ROIs used during the second processing phase (extraction)
        to iteratively extract the fluorescence of each tracked cell from each processed session.

    Args:
        configuration: The multi-day pipeline configuration.
    """
    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    console.echo(message="Initializing multi-day discovery phase...", level=LogLevel.INFO)

    # Resolves or reloads MultiDayRuntimeContext instances for all sessions. Saves configuration and runtime data to
    # disk during resolution.
    contexts = resolve_multiday_contexts(configuration=configuration)

    # Filters ROIs from each session's single-day outputs based on the configured selection criteria. Respects the
    # repeat_selection flag to skip sessions with existing selections.
    select_session_cells(contexts=contexts)

    # Registers all sessions to a shared visual space using diffeomorphic demons registration and applies the
    # deformation fields to transform reference images and cell masks.
    register_sessions(contexts=contexts)

    # Clusters ROIs across sessions in the shared deformed visual space and generates template masks for cells that
    # can be reliably identified across sessions.
    track_rois_across_sessions(contexts=contexts)

    # Projects template masks from the shared visual space back to each session's original coordinate system for
    # fluorescence extraction.
    project_templates_to_sessions(contexts=contexts)

    # Records total discovery time and processing timestamp for each context.
    total_discovery_time = int(timer.elapsed)
    for context in contexts:
        context.runtime.timing.total_discovery_time = total_discovery_time
        context.runtime.timing.date_processed = str(get_timestamp())
        context.save_runtime()

    console.echo(
        message=f"Multi-day discovery: complete. Total time: {total_discovery_time} seconds.", level=LogLevel.SUCCESS
    )


def extract_multiday_fluorescence(configuration: MultiDayConfiguration, session_id: str) -> None:
    """Extracts fluorescence data from cells tracked across imaging sessions for the specified session.

    Notes:
        This function executes the second phase of the multi-day pipeline: it locates the runtime context matching the
        input session_id and extracts the fluorescence of the ROIs tracked across sessions from the processed session's
        recording. The discovery phase must have completed before attempting extraction. Multiple sessions can be
        processed in parallel, but each session may use significant memory and CPU resources.

    Args:
        configuration: The multi-day pipeline configuration.
        session_id: The unique identifier of the session for which to extract fluorescence data. Must match one of the
            session IDs assigned during context resolution.

    Raises:
        ValueError: If the target session_id does not match any resolved session context.
        RuntimeError: If backward-transformed ROI statistics are not available, indicating the discovery phase has
            not completed.
    """
    # Reloads only the target session's context from disk. The target_session_id parameter avoids loading CombinedData
    # and runtime arrays for every other session in the dataset.
    contexts = resolve_multiday_contexts(configuration=configuration, target_session_id=session_id)
    target_context = contexts[0]

    # Memory-maps extraction arrays from disk. resolve_multiday_contexts() only loads YAML scalars, so roi_statistics
    # will be None until arrays are explicitly loaded. Uses memory mapping because the data is only needed for validation
    # here; extract_traces() reloads what it needs independently.
    if target_context.runtime.output_path is not None:
        target_context.runtime.extraction.memory_map_arrays(target_context.runtime.output_path)

    # Validates that backward-transformed ROI statistics exist from the discovery phase.
    if target_context.runtime.extraction.roi_statistics is None:
        message = (
            f"Unable to extract multi-day fluorescence for session '{session_id}'. Backward-transformed ROI "
            f"statistics are not available. Ensure the multi-day discovery phase has been completed before running "
            f"extraction."
        )
        console.error(message=message, error=RuntimeError)

    # Delegates to the unified extraction entry point, which dispatches to _extract_multi_day internally. The
    # extraction function handles fluorescence extraction, deconvolution, timing, and runtime saving.
    extract_traces(context=target_context)
