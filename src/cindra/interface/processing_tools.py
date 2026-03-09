"""Provides MCP tools for batch processing, status monitoring, and cancellation of neural imaging pipelines.

These tools enable AI agents to start, monitor, and cancel both single-recording and multi-recording batch processing
operations. Processing state is tracked via ProcessingTracker YAML files that persist across restarts, rather than
in-memory state. Single-recording processing follows a three-phase workflow (binarize, process, combine), while
multi-recording processing follows a two-phase workflow (discover, extract).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock, Thread

from natsort import natsorted
from ataraxis_base_utilities import resolve_parallel_job_capacity, resolve_worker_count
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker
from ataraxis_time import PrecisionTimer, TimerPrecisions

from ..dataclasses import MultiRecordingConfiguration, SingleRecordingConfiguration
from ..io import resolve_multi_recording_contexts, resolve_single_recording_contexts
from ..pipelines import (
    MULTI_RECORDING_TRACKER_NAME,
    SINGLE_RECORDING_TRACKER_NAME,
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    run_multi_recording_pipeline,
    run_single_recording_pipeline,
)
from .mcp_instance import mcp

_RESERVED_CORES: int = 2
"""The number of CPU cores reserved for system operations."""

_MAXIMUM_JOB_CORES: int = 30
"""The maximum number of CPU cores any single job can use."""

_MAXIMUM_PARALLEL_BINARIZE: int = 3
"""The maximum number of concurrent binarization jobs (I/O bound with TIFF decompression)."""

_MINIMUM_RECORDING_COUNT: int = 2
"""The minimum number of recordings required for multi-recording processing."""


@dataclass
class _SingleRecordingBatchState:
    """Tracks the runtime orchestration state for single-recording batch processing.

    Job completion and failure states are persisted in per-recording ProcessingTracker YAML files. This dataclass only
    holds the minimal state needed for thread orchestration: phase queues, active threads, resource limits, and
    tracker/configuration path mappings.
    """

    tracker_paths: dict[str, Path] = field(default_factory=dict)
    """Recording key to ProcessingTracker file path mapping."""
    configuration_paths: dict[str, Path] = field(default_factory=dict)
    """Recording key to per-recording configuration file path mapping."""

    binarize_jobs: dict[str, str] = field(default_factory=dict)
    """Recording key to binarize job ID mapping."""
    process_jobs: dict[str, list[str]] = field(default_factory=dict)
    """Recording key to ordered list of process job IDs mapping."""
    combine_jobs: dict[str, str] = field(default_factory=dict)
    """Recording key to combine job ID mapping."""

    current_phase: str = "binarize"
    """Current processing phase: 'binarize', 'process', or 'combine'."""
    phase_queue: list[tuple[str, str]] = field(default_factory=list)
    """Ordered (recording_key, job_id) pairs queued for dispatch in the current phase."""
    active_threads: dict[str, Thread] = field(default_factory=dict)
    """Currently running job_id to Thread mapping."""

    workers_per_plane: int = 30
    """CPU cores allocated per plane processing job."""
    max_parallel_planes: int = 1
    """Maximum number of concurrent plane processing jobs."""

    lock: Lock = field(default_factory=Lock)
    """Thread synchronization lock for batch state access."""
    manager_thread: Thread | None = None
    """Background batch manager thread reference."""


@dataclass
class _MultiRecordingBatchState:
    """Tracks the runtime orchestration state for multi-recording batch processing.

    Job completion and failure states are persisted in per-animal ProcessingTracker YAML files. This dataclass only
    holds the minimal state needed for thread orchestration: phase queues, active threads, resource limits, and
    tracker/configuration path mappings.
    """

    tracker_paths: dict[str, Path] = field(default_factory=dict)
    """Animal key to ProcessingTracker file path mapping."""
    configuration_paths: dict[str, Path] = field(default_factory=dict)
    """Animal key to configuration file path mapping."""
    recording_paths: dict[str, list[Path]] = field(default_factory=dict)
    """Animal key to list of recording directory paths mapping."""

    discover_jobs: dict[str, str] = field(default_factory=dict)
    """Animal key to discover job ID mapping."""
    extract_jobs: dict[str, list[str]] = field(default_factory=dict)
    """Animal key to ordered list of extract job IDs mapping."""

    current_phase: str = "discover"
    """Current processing phase: 'discover' or 'extract'."""
    phase_queue: list[tuple[str, str]] = field(default_factory=list)
    """Ordered (animal_key, job_id) pairs queued for dispatch in the current phase."""
    active_threads: dict[str, Thread] = field(default_factory=dict)
    """Currently running job_id to Thread mapping."""

    workers_per_discover: int = 20
    """Workers allocated for the discover phase."""
    max_parallel_discovers: int = 1
    """Maximum number of concurrent discover jobs."""
    workers_per_extract: int = 30
    """Workers allocated for the extract phase."""
    max_parallel_extracts: int = 1
    """Maximum number of concurrent extract jobs."""

    lock: Lock = field(default_factory=Lock)
    """Thread synchronization lock for batch state access."""
    manager_thread: Thread | None = None
    """Background batch manager thread reference."""


_single_recording_batch_state: _SingleRecordingBatchState | None = None
"""The module-level batch processing state for single-recording operations."""

_multi_recording_batch_state: _MultiRecordingBatchState | None = None
"""The module-level batch processing state for multi-recording operations."""


@mcp.tool()
def get_single_recording_status(recording_path: str) -> dict[str, object]:
    """Gets the processing status of a single recording by reading its ProcessingTracker file.

    Reads the tracker YAML file at <recording_path>/single_recording_tracker.yaml to determine how far processing has
    progressed. Returns per-job status grouped by pipeline phase (binarize, process, combine) and an overall status
    string synthesized from the tracker state.

    Args:
        recording_path: The absolute path to the recording data directory.

    Returns:
        On success, contains the 'recording_path', 'tracker_path', per-phase job status in 'jobs', a 'summary' with
        counts by status, and a synthesized 'status' string ('not_started', 'binarizing', 'processing', 'combining',
        'completed', or 'failed'). When no tracker exists, returns 'status' of 'not_started'. On failure, contains an
        'error' describing the issue. Both cases include a 'success' flag.
    """
    recording = Path(recording_path)

    if not recording.exists():
        return {
            "success": False,
            "error": f"Unable to get single-recording status. Recording directory not found: {recording_path}.",
        }

    tracker_path = recording / SINGLE_RECORDING_TRACKER_NAME
    if not tracker_path.exists():
        return {
            "success": True,
            "recording_path": str(recording),
            "status": "not_started",
            "message": "No processing tracker found for this recording.",
        }

    return _read_single_recording_tracker(tracker_path=tracker_path, recording_path=recording)


@mcp.tool()
def get_multi_recording_status(recording_path: str) -> dict[str, object]:
    """Gets the multi-recording processing status for a recording by reading ProcessingTracker files.

    Searches for multi-recording tracker YAML files under <recording_path>/cindra/multi_recording/<dataset>/ and reads
    each tracker to determine per-dataset processing progress. Returns per-job status grouped by pipeline phase
    (discover, extract) for each dataset found.

    Args:
        recording_path: The absolute path to a recording directory.

    Returns:
        On success, contains the 'recording_path' and a 'datasets' mapping where each dataset key maps to its tracker
        status, including per-phase job states, summary counts, and overall status. When no trackers exist, returns
        'status' of 'not_started'. On failure, contains an 'error' describing the issue. Both cases include a 'success'
        flag.
    """
    recording = Path(recording_path)

    if not recording.exists():
        return {
            "success": False,
            "error": f"Unable to get multi-recording status. Recording directory not found: {recording_path}.",
        }

    # Searches for multi-recording tracker files under the cindra output hierarchy.
    multi_recording_base = recording / "cindra" / "multi_recording"
    if not multi_recording_base.exists():
        return {
            "success": True,
            "recording_path": str(recording),
            "status": "not_started",
            "message": "No multi-recording output directory found.",
        }

    tracker_files = list(multi_recording_base.rglob(MULTI_RECORDING_TRACKER_NAME))
    if not tracker_files:
        return {
            "success": True,
            "recording_path": str(recording),
            "status": "not_started",
            "message": "No multi-recording processing trackers found.",
        }

    dataset_statuses: dict[str, dict[str, object]] = {}
    for tracker_file in tracker_files:
        dataset_name = tracker_file.parent.name
        dataset_statuses[dataset_name] = _read_multi_recording_tracker(tracker_path=tracker_file)

    return {
        "success": True,
        "recording_path": str(recording),
        "datasets": dataset_statuses,
    }


@mcp.tool()
def start_batch_processing_tool(
    recording_paths: list[str],
    configuration_path: str,
    *,
    recording_output_paths: list[str] | None = None,
    workers_per_plane: int = -1,
    max_parallel_planes: int = -1,
    progress_bars: bool = False,
) -> dict[str, object]:
    """Starts batch single-recording processing for multiple recordings.

    Manages a three-phase batch workflow: binarize (sequential), process (parallel by plane), combine (sequential).
    Creates per-recording configuration copies with recording-specific paths and runtime settings, initializes
    ProcessingTracker files for each recording, and dispatches jobs via background threads. Use
    get_batch_processing_status_tool to monitor progress.

    Args:
        recording_paths: List of absolute paths to recording data directories (used as file_io.data_path per
            recording).
        configuration_path: The absolute path to the template configuration YAML file.
        recording_output_paths: Optional list of absolute paths for per-recording output directories (used as
            file_io.output_path). Must match the length of recording_paths when provided. When not provided, each
            recording's output_path defaults to its data_path.
        workers_per_plane: CPU cores per plane job (-1 for automatic, max 30).
        max_parallel_planes: Max concurrent plane jobs (-1 for automatic).
        progress_bars: Determines whether to display progress bars during processing.

    Returns:
        On success, contains a 'started' flag, 'total_recordings' count, 'workers_per_plane' and
        'max_parallel_planes' allocation, and any 'invalid_paths' that were skipped. On failure, contains an 'error'
        describing the issue.
    """
    global _single_recording_batch_state

    if not recording_paths:
        return {"error": "Unable to start batch processing. At least one recording path is required."}

    if recording_output_paths is not None and len(recording_output_paths) != len(recording_paths):
        return {
            "error": (
                f"Unable to start batch processing. The recording_output_paths length "
                f"({len(recording_output_paths)}) must match the recording_paths length ({len(recording_paths)})."
            ),
        }

    template_path = Path(configuration_path)
    if not template_path.exists():
        return {"error": f"Unable to start batch processing. Configuration file not found: {configuration_path}."}

    if template_path.suffix != ".yaml":
        return {
            "error": (
                f"Unable to start batch processing. Configuration file must be a .yaml file: {configuration_path}."
            ),
        }

    # Validates recording paths.
    valid_indices: list[int] = []
    valid_paths: list[Path] = []
    invalid_paths: list[str] = []

    for index, path_string in enumerate(recording_paths):
        path = Path(path_string)
        if path.exists() and path.is_dir():
            valid_paths.append(path)
            valid_indices.append(index)
        else:
            invalid_paths.append(path_string)

    if not valid_paths:
        return {
            "error": "Unable to start batch processing. No valid recording paths provided.",
            "invalid_paths": invalid_paths,
        }

    # Resolves per-recording output paths. Defaults to data_path when recording_output_paths is not provided.
    resolved_output_paths: list[Path] = []
    for index, data_path in zip(valid_indices, valid_paths, strict=True):
        if recording_output_paths is not None:
            resolved_output_paths.append(Path(recording_output_paths[index]))
        else:
            resolved_output_paths.append(data_path)

    # Checks if batch processing is already active.
    if _single_recording_batch_state is not None:
        with _single_recording_batch_state.lock:
            if _single_recording_batch_state.active_threads or _single_recording_batch_state.phase_queue:
                return {
                    "error": "Unable to start batch processing. Batch processing is already in progress.",
                    "active_count": len(_single_recording_batch_state.active_threads),
                    "queued_count": len(_single_recording_batch_state.phase_queue),
                }

    # Calculates resource allocation.
    actual_workers = min(
        resolve_worker_count(requested_workers=workers_per_plane, reserved_cores=_RESERVED_CORES), _MAXIMUM_JOB_CORES
    )
    actual_max_parallel = (
        max_parallel_planes
        if max_parallel_planes > 0
        else resolve_parallel_job_capacity(workers_per_job=actual_workers)
    )

    # Creates per-recording configurations, resolves plane counts, and initializes ProcessingTracker files.
    batch_state = _SingleRecordingBatchState(
        workers_per_plane=actual_workers,
        max_parallel_planes=actual_max_parallel,
        lock=Lock(),
    )

    for data_path, output_path in zip(valid_paths, resolved_output_paths, strict=True):
        recording_key = str(data_path)

        # Creates a per-recording configuration copy with recording-specific paths and runtime settings.
        recording_configuration = SingleRecordingConfiguration.from_yaml(file_path=template_path)
        recording_configuration.file_io.data_path = data_path
        recording_configuration.file_io.output_path = output_path
        recording_configuration.runtime.parallel_workers = actual_workers
        recording_configuration.runtime.display_progress_bars = progress_bars
        output_path.mkdir(parents=True, exist_ok=True)
        recording_configuration_path = output_path / "_batch_config.yaml"
        recording_configuration.save(file_path=recording_configuration_path)
        batch_state.configuration_paths[recording_key] = recording_configuration_path

        # Resolves plane count from configuration to build the complete job list.
        contexts = resolve_single_recording_contexts(configuration=recording_configuration)
        plane_count = len(contexts)

        # Builds the job list: binarize, all process planes, combine.
        jobs: list[tuple[str, str]] = [(SingleRecordingJobNames.BINARIZE, "")]
        for plane_index in range(plane_count):
            jobs.append((SingleRecordingJobNames.PROCESS, f"plane_{plane_index}"))
        jobs.append((SingleRecordingJobNames.COMBINE, ""))

        # Initializes the ProcessingTracker with all jobs for this recording.
        tracker_path = output_path / SINGLE_RECORDING_TRACKER_NAME
        tracker = ProcessingTracker(file_path=tracker_path)
        job_ids = tracker.initialize_jobs(jobs=jobs)
        batch_state.tracker_paths[recording_key] = tracker_path

        # Maps job IDs to phases for orchestration.
        batch_state.binarize_jobs[recording_key] = job_ids[0]
        batch_state.process_jobs[recording_key] = job_ids[1 : 1 + plane_count]
        batch_state.combine_jobs[recording_key] = job_ids[-1]

    # Populates the binarize phase queue (naturally sorted for deterministic order).
    for recording_key in natsorted(batch_state.tracker_paths.keys()):
        batch_state.phase_queue.append((recording_key, batch_state.binarize_jobs[recording_key]))

    # Activates the batch state and starts the manager thread.
    _single_recording_batch_state = batch_state
    manager = Thread(target=_single_recording_batch_manager, daemon=True)
    manager.start()
    _single_recording_batch_state.manager_thread = manager

    result: dict[str, object] = {
        "started": True,
        "total_recordings": len(valid_paths),
        "workers_per_plane": actual_workers,
        "max_parallel_planes": actual_max_parallel,
        "message": "Batch processing started. Use get_batch_processing_status_tool to monitor progress.",
    }

    if invalid_paths:
        result["invalid_paths"] = invalid_paths

    return result


@mcp.tool()
def get_batch_processing_status_tool() -> dict[str, object]:
    """Returns the current status of single-recording batch processing.

    Reads ProcessingTracker files for each recording in the batch to report per-recording progress across all three
    phases (binarize, process, combine), including active, completed, and failed counts.

    Returns:
        Contains the 'current_phase', per-recording 'recordings' status list, and a 'summary' with aggregate counts
        for total, succeeded, failed, and running recordings. Returns empty state when no batch processing has been
        started.
    """
    if _single_recording_batch_state is None:
        return {
            "current_phase": "none",
            "recordings": [],
            "summary": {"total": 0, "succeeded": 0, "failed": 0, "running": 0},
        }

    with _single_recording_batch_state.lock:
        recordings_status: list[dict[str, object]] = []
        total_succeeded = 0
        total_failed = 0
        total_running = 0

        for recording_key in natsorted(_single_recording_batch_state.tracker_paths.keys()):
            tracker_path = _single_recording_batch_state.tracker_paths[recording_key]
            tracker = ProcessingTracker(file_path=tracker_path)

            # Reads binarize job status.
            binarize_job_id = _single_recording_batch_state.binarize_jobs[recording_key]
            binarize_status = tracker.get_job_status(job_id=binarize_job_id).name.lower()

            # Reads process job statuses.
            process_job_ids = _single_recording_batch_state.process_jobs[recording_key]
            plane_count = len(process_job_ids)
            process_succeeded = sum(
                1
                for job_id in process_job_ids
                if tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED
            )
            process_failed = sum(
                1
                for job_id in process_job_ids
                if tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED
            )

            if process_failed:
                process_status = f"{process_succeeded}/{plane_count} (failed: {process_failed})"
            else:
                process_status = f"{process_succeeded}/{plane_count}"

            # Reads combine job status.
            combine_job_id = _single_recording_batch_state.combine_jobs[recording_key]
            combine_status = tracker.get_job_status(job_id=combine_job_id).name.lower()

            # Synthesizes overall recording status from tracker state.
            if tracker.complete:
                overall_status = "SUCCEEDED"
                total_succeeded += 1
            elif tracker.encountered_error:
                overall_status = "FAILED"
                total_failed += 1
            elif any(
                job_id in _single_recording_batch_state.active_threads
                for job_id in [binarize_job_id, *process_job_ids, combine_job_id]
            ):
                overall_status = "PROCESSING"
                total_running += 1
            else:
                overall_status = "QUEUED"

            recording_status: dict[str, object] = {
                "recording_name": Path(recording_key).name,
                "status": overall_status,
                "binarize": binarize_status,
                "process": process_status,
                "combine": combine_status,
            }

            # Includes error messages from any failed jobs.
            errors: list[str] = []
            for job_id in [binarize_job_id, *process_job_ids, combine_job_id]:
                job_info = tracker.get_job_info(job_id=job_id)
                if job_info.error_message:
                    errors.append(f"{job_info.job_name}({job_info.specifier}): {job_info.error_message}")
            if errors:
                recording_status["errors"] = errors

            recordings_status.append(recording_status)

        summary = {
            "total": len(_single_recording_batch_state.tracker_paths),
            "succeeded": total_succeeded,
            "failed": total_failed,
            "running": total_running,
        }

        return {
            "current_phase": _single_recording_batch_state.current_phase,
            "recordings": recordings_status,
            "summary": summary,
        }


@mcp.tool()
def cancel_batch_processing_tool() -> dict[str, object]:
    """Cancels any running single-recording batch processing.

    Clears all phase queues to prevent new jobs from starting and resets the batch state. Active jobs will complete
    naturally but no new jobs will be dispatched.

    Returns:
        Contains a 'canceled' flag, a 'message' describing the outcome, and a 'final_state' with counts for
        succeeded_jobs, failed_jobs, and active_jobs_at_cancel.
    """
    global _single_recording_batch_state

    if _single_recording_batch_state is None:
        return {"canceled": False, "message": "No single-recording batch processing is active."}

    with _single_recording_batch_state.lock:
        active_count = len(_single_recording_batch_state.active_threads)

        # Clears the phase queue to prevent new jobs from starting.
        _single_recording_batch_state.phase_queue.clear()

        # Reads final state from trackers.
        total_succeeded = 0
        total_failed = 0
        for tracker_path in _single_recording_batch_state.tracker_paths.values():
            tracker = ProcessingTracker(file_path=tracker_path)
            summary = tracker.get_summary()
            for status, count in summary.items():
                if status == ProcessingStatus.SUCCEEDED:
                    total_succeeded += count
                elif status == ProcessingStatus.FAILED:
                    total_failed += count

        final_state = {
            "succeeded_jobs": total_succeeded,
            "failed_jobs": total_failed,
            "active_jobs_at_cancel": active_count,
        }

    # Resets batch state after releasing lock.
    _single_recording_batch_state = None

    return {
        "canceled": True,
        "message": "Single-recording batch processing canceled. Active jobs will complete but no new jobs will start.",
        "final_state": final_state,
    }


@mcp.tool()
def start_multi_recording_batch_processing_tool(
    animal_configurations: list[dict[str, object]],
    *,
    workers_per_discover: int = 20,
    workers_per_extract: int = -1,
    progress_bars: bool = False,
) -> dict[str, object]:
    """Starts batch multi-recording processing for multiple animals.

    Manages a two-phase batch workflow: discover (parallel by animal), extract (parallel by recording). Each animal
    configuration specifies a configuration file and its associated recording paths. Initializes ProcessingTracker
    files per animal and dispatches jobs via background threads. Use get_multi_recording_batch_processing_status_tool
    to monitor progress.

    Args:
        animal_configurations: List of animal configurations, each a dictionary with 'configuration_path' (absolute
            path to the multi-recording YAML configuration) and 'recording_paths' (list of absolute paths to
            recording directories). At least 2 recording paths per animal are required.
        workers_per_discover: Workers for discover phase (default 20).
        workers_per_extract: Workers for extract phase (-1 for automatic, max 30).
        progress_bars: Determines whether to display progress bars during processing.

    Returns:
        On success, contains a 'started' flag, 'total_animals' and 'total_recordings' counts, worker allocation
        settings, and any 'invalid_configurations' that were skipped. On failure, contains an 'error' describing
        the issue.
    """
    global _multi_recording_batch_state

    if not animal_configurations:
        return {
            "error": "Unable to start multi-recording batch processing. At least one animal configuration is required.",
        }

    # Validates animal configurations.
    valid_animals: list[tuple[str, Path, list[Path]]] = []
    invalid_configurations: list[str] = []

    for animal_configuration in animal_configurations:
        if "configuration_path" not in animal_configuration or "recording_paths" not in animal_configuration:
            invalid_configurations.append(f"Missing required keys: {animal_configuration}")
            continue

        # noinspection PyTypeChecker
        animal_configuration_path = Path(animal_configuration["configuration_path"])
        if not animal_configuration_path.exists():
            invalid_configurations.append(f"Configuration not found: {animal_configuration_path}")
            continue

        # noinspection PyTypeChecker
        animal_recording_paths = [Path(path) for path in animal_configuration["recording_paths"]]
        if len(animal_recording_paths) < _MINIMUM_RECORDING_COUNT:
            invalid_configurations.append(f"Need at least 2 recordings: {animal_configuration_path}")
            continue

        invalid_recordings = [
            str(path) for path in animal_recording_paths if not path.exists() or not path.is_dir()
        ]
        if invalid_recordings:
            invalid_configurations.append(f"Invalid recordings for {animal_configuration_path}: {invalid_recordings}")
            continue

        # Loads configuration to extract the animal key and validate the file format.
        try:
            configuration = MultiRecordingConfiguration.from_yaml(file_path=animal_configuration_path)
            animal_key = configuration.recording_io.dataset_name
        except Exception as error:
            invalid_configurations.append(f"Unable to load configuration {animal_configuration_path}: {error}")
            continue

        valid_animals.append((animal_key, animal_configuration_path, animal_recording_paths))

    if not valid_animals:
        return {
            "error": "Unable to start multi-recording batch processing. No valid animal configurations provided.",
            "invalid_configurations": invalid_configurations,
        }

    # Checks if batch processing is already active.
    if _multi_recording_batch_state is not None:
        with _multi_recording_batch_state.lock:
            if _multi_recording_batch_state.active_threads or _multi_recording_batch_state.phase_queue:
                return {
                    "error": (
                        "Unable to start multi-recording batch processing. Batch processing is already in progress."
                    ),
                    "active_count": len(_multi_recording_batch_state.active_threads),
                    "queued_count": len(_multi_recording_batch_state.phase_queue),
                }

    # Calculates resource allocation.
    actual_workers_discover = min(
        resolve_worker_count(requested_workers=workers_per_discover, reserved_cores=_RESERVED_CORES),
        _MAXIMUM_JOB_CORES,
    )
    actual_workers_extract = min(
        resolve_worker_count(requested_workers=workers_per_extract, reserved_cores=_RESERVED_CORES), _MAXIMUM_JOB_CORES
    )
    max_parallel_discovers = resolve_parallel_job_capacity(workers_per_job=actual_workers_discover)
    max_parallel_extracts = resolve_parallel_job_capacity(workers_per_job=actual_workers_extract)

    # Initializes batch state.
    batch_state = _MultiRecordingBatchState(
        workers_per_discover=actual_workers_discover,
        max_parallel_discovers=max_parallel_discovers,
        workers_per_extract=actual_workers_extract,
        max_parallel_extracts=max_parallel_extracts,
        lock=Lock(),
    )

    total_recordings = 0
    for animal_key, animal_configuration_path, animal_recording_paths in valid_animals:
        # Writes recording directories and runtime settings into the configuration file so the pipeline can read them.
        configuration = MultiRecordingConfiguration.from_yaml(file_path=animal_configuration_path)
        configuration.recording_io.recording_directories = tuple(natsorted(animal_recording_paths))
        configuration.runtime.parallel_workers = actual_workers_discover
        configuration.runtime.display_progress_bars = progress_bars
        configuration.save(file_path=animal_configuration_path)

        # Resolves contexts to determine recording IDs and the tracker output path.
        contexts = resolve_multi_recording_contexts(configuration=configuration)
        recording_ids = [context.runtime.io.recording_id for context in contexts]
        main_recording_path = contexts[0].runtime.output_path

        if main_recording_path is None:
            invalid_configurations.append(f"Unable to resolve output path for animal '{animal_key}'.")
            continue

        # Builds the job list: discover, then extract per recording.
        jobs: list[tuple[str, str]] = [(MultiRecordingJobNames.DISCOVER, "")]
        for recording_id in recording_ids:
            jobs.append((MultiRecordingJobNames.EXTRACT, recording_id))

        # Initializes the ProcessingTracker with all jobs for this animal.
        tracker_path = main_recording_path / MULTI_RECORDING_TRACKER_NAME
        tracker = ProcessingTracker(file_path=tracker_path)
        job_ids = tracker.initialize_jobs(jobs=jobs)

        # Stores per-animal state for orchestration.
        batch_state.tracker_paths[animal_key] = tracker_path
        batch_state.configuration_paths[animal_key] = animal_configuration_path
        batch_state.recording_paths[animal_key] = animal_recording_paths
        batch_state.discover_jobs[animal_key] = job_ids[0]
        batch_state.extract_jobs[animal_key] = job_ids[1:]
        total_recordings += len(recording_ids)

    # Populates the discover phase queue (naturally sorted for deterministic order).
    for animal_key in natsorted(batch_state.tracker_paths.keys()):
        batch_state.phase_queue.append((animal_key, batch_state.discover_jobs[animal_key]))

    # Activates the batch state and starts the manager thread.
    _multi_recording_batch_state = batch_state
    manager = Thread(target=_multi_recording_batch_manager, daemon=True)
    manager.start()
    _multi_recording_batch_state.manager_thread = manager

    result: dict[str, object] = {
        "started": True,
        "total_animals": len(batch_state.tracker_paths),
        "total_recordings": total_recordings,
        "workers_per_discover": actual_workers_discover,
        "workers_per_extract": actual_workers_extract,
        "message": (
            "Multi-recording batch processing started. Use get_multi_recording_batch_processing_status_tool to monitor."
        ),
    }

    if invalid_configurations:
        result["invalid_configurations"] = invalid_configurations

    return result


@mcp.tool()
def get_multi_recording_batch_processing_status_tool() -> dict[str, object]:
    """Returns the current status of multi-recording batch processing.

    Reads ProcessingTracker files for each animal in the batch to report per-animal progress across both phases
    (discover, extract), including active, completed, and failed counts.

    Returns:
        Contains the 'current_phase', per-animal 'animals' status list, and a 'summary' with aggregate counts for
        total_animals, succeeded, failed, and running. Returns empty state when no batch processing has been started.
    """
    if _multi_recording_batch_state is None:
        return {
            "current_phase": "none",
            "animals": [],
            "summary": {"total_animals": 0, "succeeded": 0, "failed": 0, "running": 0},
        }

    with _multi_recording_batch_state.lock:
        animals_status: list[dict[str, object]] = []
        total_succeeded = 0
        total_failed = 0
        total_running = 0

        for animal_key in natsorted(_multi_recording_batch_state.tracker_paths.keys()):
            tracker_path = _multi_recording_batch_state.tracker_paths[animal_key]
            tracker = ProcessingTracker(file_path=tracker_path)

            # Reads discover job status.
            discover_job_id = _multi_recording_batch_state.discover_jobs[animal_key]
            discover_status = tracker.get_job_status(job_id=discover_job_id).name.lower()

            # Reads extract job statuses.
            extract_job_ids = _multi_recording_batch_state.extract_jobs[animal_key]
            extract_total = len(extract_job_ids)
            extract_succeeded = sum(
                1
                for job_id in extract_job_ids
                if tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED
            )
            extract_failed = sum(
                1
                for job_id in extract_job_ids
                if tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED
            )

            # Synthesizes overall animal status from tracker state.
            if tracker.complete:
                overall_status = "SUCCEEDED"
                total_succeeded += 1
            elif tracker.encountered_error:
                overall_status = "FAILED"
                total_failed += 1
            elif any(
                job_id in _multi_recording_batch_state.active_threads
                for job_id in [discover_job_id, *extract_job_ids]
            ):
                overall_status = "PROCESSING"
                total_running += 1
            else:
                overall_status = "QUEUED"

            animal_status: dict[str, object] = {
                "animal_key": animal_key,
                "status": overall_status,
                "discover": discover_status,
                "extract_completed": extract_succeeded,
                "extract_failed": extract_failed,
                "extract_total": extract_total,
            }

            # Includes error messages from any failed jobs.
            errors: list[str] = []
            for job_id in [discover_job_id, *extract_job_ids]:
                job_info = tracker.get_job_info(job_id=job_id)
                if job_info.error_message:
                    errors.append(f"{job_info.job_name}({job_info.specifier}): {job_info.error_message}")
            if errors:
                animal_status["errors"] = errors

            animals_status.append(animal_status)

        summary = {
            "total_animals": len(_multi_recording_batch_state.tracker_paths),
            "succeeded": total_succeeded,
            "failed": total_failed,
            "running": total_running,
        }

        return {
            "current_phase": _multi_recording_batch_state.current_phase,
            "animals": animals_status,
            "summary": summary,
        }


@mcp.tool()
def cancel_multi_recording_batch_processing_tool() -> dict[str, object]:
    """Cancels any running multi-recording batch processing.

    Clears all phase queues to prevent new jobs from starting and resets the batch state. Active jobs will complete
    naturally but no new jobs will be dispatched.

    Returns:
        Contains a 'canceled' flag, a 'message' describing the outcome, and a 'final_state' with counts for
        succeeded_jobs, failed_jobs, and active_jobs_at_cancel.
    """
    global _multi_recording_batch_state

    if _multi_recording_batch_state is None:
        return {"canceled": False, "message": "No multi-recording batch processing is active."}

    with _multi_recording_batch_state.lock:
        active_count = len(_multi_recording_batch_state.active_threads)

        # Clears the phase queue to prevent new jobs from starting.
        _multi_recording_batch_state.phase_queue.clear()

        # Reads final state from trackers.
        total_succeeded = 0
        total_failed = 0
        for tracker_path in _multi_recording_batch_state.tracker_paths.values():
            tracker = ProcessingTracker(file_path=tracker_path)
            summary = tracker.get_summary()
            for status, count in summary.items():
                if status == ProcessingStatus.SUCCEEDED:
                    total_succeeded += count
                elif status == ProcessingStatus.FAILED:
                    total_failed += count

        final_state = {
            "succeeded_jobs": total_succeeded,
            "failed_jobs": total_failed,
            "active_jobs_at_cancel": active_count,
        }

    # Resets batch state after releasing lock.
    _multi_recording_batch_state = None

    return {
        "canceled": True,
        "message": "Multi-recording batch processing canceled. Active jobs will complete but no new jobs will start.",
        "final_state": final_state,
    }


def _pipeline_worker(configuration_path: Path, job_id: str, *, single_recording: bool = True) -> None:
    """Executes a single pipeline job identified by its job ID.

    Calls the appropriate pipeline function in REMOTE mode, passing the job_id so the pipeline reads the job definition
    from the ProcessingTracker and updates tracker state on completion or failure. The pipeline handles all tracker
    state transitions (start_job, complete_job, fail_job) internally.

    Args:
        configuration_path: The path to the recording or animal configuration file.
        job_id: The unique hexadecimal job identifier registered in the ProcessingTracker.
        single_recording: Determines whether to call the single-recording or multi-recording pipeline.
    """
    try:
        if single_recording:
            run_single_recording_pipeline(configuration_path=configuration_path, job_id=job_id)
        else:
            run_multi_recording_pipeline(configuration_path=configuration_path, job_id=job_id)
    except Exception:
        # The pipeline already called tracker.fail_job() before re-raising. The exception is caught here to prevent
        # the daemon thread from propagating it, since the tracker records the failure state persistently.
        pass


def _single_recording_batch_manager() -> None:
    """Orchestrates three-phase single-recording batch processing: binarize, process, combine.

    Runs as a daemon thread, polling at 1-second intervals to dispatch new jobs and advance between phases.
    Binarize and combine phases run up to 3 concurrent jobs (I/O bound), while the process phase runs jobs in parallel
    up to the configured max_parallel_planes limit. Job state is tracked via ProcessingTracker files; this
    manager only handles thread orchestration and phase transitions.
    """
    timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

    if _single_recording_batch_state is None:
        return

    while True:
        with _single_recording_batch_state.lock:
            # Cleans up completed threads.
            completed_job_ids = [
                completed_job_id
                for completed_job_id, thread in _single_recording_batch_state.active_threads.items()
                if not thread.is_alive()
            ]
            for completed_job_id in completed_job_ids:
                _single_recording_batch_state.active_threads.pop(completed_job_id, None)

            # Phase 1: BINARIZE (parallel — up to _MAXIMUM_PARALLEL_BINARIZE concurrent jobs).
            if _single_recording_batch_state.current_phase == "binarize":
                while (
                    len(_single_recording_batch_state.active_threads) < _MAXIMUM_PARALLEL_BINARIZE
                    and _single_recording_batch_state.phase_queue
                ):
                    recording_key, job_id = _single_recording_batch_state.phase_queue.pop(0)
                    thread = Thread(
                        target=_pipeline_worker,
                        kwargs={
                            "configuration_path": _single_recording_batch_state.configuration_paths[recording_key],
                            "job_id": job_id,
                            "single_recording": True,
                        },
                        daemon=True,
                    )
                    thread.start()
                    _single_recording_batch_state.active_threads[job_id] = thread

                # Transitions to process phase when all binarize jobs have been dispatched and completed.
                if not _single_recording_batch_state.active_threads and not _single_recording_batch_state.phase_queue:
                    # Builds process queue from recordings whose binarize job succeeded.
                    for recording_key in natsorted(_single_recording_batch_state.tracker_paths.keys()):
                        tracker = ProcessingTracker(
                            file_path=_single_recording_batch_state.tracker_paths[recording_key]
                        )
                        binarize_status = tracker.get_job_status(
                            job_id=_single_recording_batch_state.binarize_jobs[recording_key]
                        )
                        if binarize_status == ProcessingStatus.SUCCEEDED:
                            for process_job_id in _single_recording_batch_state.process_jobs[recording_key]:
                                _single_recording_batch_state.phase_queue.append((recording_key, process_job_id))

                    _single_recording_batch_state.current_phase = "process"

            # Phase 2: PROCESS (parallel — up to max_parallel_planes concurrent jobs).
            elif _single_recording_batch_state.current_phase == "process":
                while (
                    len(_single_recording_batch_state.active_threads)
                    < _single_recording_batch_state.max_parallel_planes
                    and _single_recording_batch_state.phase_queue
                ):
                    recording_key, job_id = _single_recording_batch_state.phase_queue.pop(0)
                    thread = Thread(
                        target=_pipeline_worker,
                        kwargs={
                            "configuration_path": _single_recording_batch_state.configuration_paths[recording_key],
                            "job_id": job_id,
                            "single_recording": True,
                        },
                        daemon=True,
                    )
                    thread.start()
                    _single_recording_batch_state.active_threads[job_id] = thread

                # Transitions to combine phase when all process jobs have been dispatched and completed.
                if not _single_recording_batch_state.active_threads and not _single_recording_batch_state.phase_queue:
                    # Builds combine queue from recordings whose process jobs succeeded.
                    for recording_key in natsorted(_single_recording_batch_state.tracker_paths.keys()):
                        tracker = ProcessingTracker(
                            file_path=_single_recording_batch_state.tracker_paths[recording_key]
                        )
                        process_job_ids = _single_recording_batch_state.process_jobs[recording_key]
                        all_succeeded = all(
                            tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED
                            for job_id in process_job_ids
                        )
                        if all_succeeded:
                            _single_recording_batch_state.phase_queue.append(
                                (recording_key, _single_recording_batch_state.combine_jobs[recording_key])
                            )

                    _single_recording_batch_state.current_phase = "combine"

            # Phase 3: COMBINE (parallel — up to _MAXIMUM_PARALLEL_BINARIZE concurrent jobs, I/O bound).
            elif _single_recording_batch_state.current_phase == "combine":
                while (
                    len(_single_recording_batch_state.active_threads) < _MAXIMUM_PARALLEL_BINARIZE
                    and _single_recording_batch_state.phase_queue
                ):
                    recording_key, job_id = _single_recording_batch_state.phase_queue.pop(0)
                    thread = Thread(
                        target=_pipeline_worker,
                        kwargs={
                            "configuration_path": _single_recording_batch_state.configuration_paths[recording_key],
                            "job_id": job_id,
                            "single_recording": True,
                        },
                        daemon=True,
                    )
                    thread.start()
                    _single_recording_batch_state.active_threads[job_id] = thread

                # Exits when all combine jobs have been dispatched and completed.
                if not _single_recording_batch_state.active_threads and not _single_recording_batch_state.phase_queue:
                    break

        # Polls at 1-second intervals before checking again.
        timer.delay(delay=1000, allow_sleep=True)


def _multi_recording_batch_manager() -> None:
    """Orchestrates two-phase multi-recording batch processing: discover, extract.

    Runs as a daemon thread, polling at 1-second intervals to dispatch new jobs and advance between phases.
    Both discover and extract phases support parallel execution up to their respective configured limits. Job state
    is tracked via ProcessingTracker files; this manager only handles thread orchestration and phase transitions.
    """
    timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

    if _multi_recording_batch_state is None:
        return

    while True:
        with _multi_recording_batch_state.lock:
            # Cleans up completed threads.
            completed_job_ids = [
                completed_job_id
                for completed_job_id, thread in _multi_recording_batch_state.active_threads.items()
                if not thread.is_alive()
            ]
            for completed_job_id in completed_job_ids:
                _multi_recording_batch_state.active_threads.pop(completed_job_id, None)

            # Phase 1: DISCOVER (parallel — up to max_parallel_discovers concurrent jobs).
            if _multi_recording_batch_state.current_phase == "discover":
                while (
                    len(_multi_recording_batch_state.active_threads)
                    < _multi_recording_batch_state.max_parallel_discovers
                    and _multi_recording_batch_state.phase_queue
                ):
                    animal_key, job_id = _multi_recording_batch_state.phase_queue.pop(0)
                    thread = Thread(
                        target=_pipeline_worker,
                        kwargs={
                            "configuration_path": _multi_recording_batch_state.configuration_paths[animal_key],
                            "job_id": job_id,
                            "single_recording": False,
                        },
                        daemon=True,
                    )
                    thread.start()
                    _multi_recording_batch_state.active_threads[job_id] = thread

                # Transitions to extract phase when all discover jobs have been dispatched and completed.
                if not _multi_recording_batch_state.active_threads and not _multi_recording_batch_state.phase_queue:
                    # Builds extract queue from animals whose discover job succeeded.
                    for animal_key in natsorted(_multi_recording_batch_state.tracker_paths.keys()):
                        tracker = ProcessingTracker(
                            file_path=_multi_recording_batch_state.tracker_paths[animal_key]
                        )
                        discover_status = tracker.get_job_status(
                            job_id=_multi_recording_batch_state.discover_jobs[animal_key]
                        )
                        if discover_status == ProcessingStatus.SUCCEEDED:
                            for extract_job_id in _multi_recording_batch_state.extract_jobs[animal_key]:
                                _multi_recording_batch_state.phase_queue.append((animal_key, extract_job_id))

                    _multi_recording_batch_state.current_phase = "extract"

            # Phase 2: EXTRACT (parallel — up to max_parallel_extracts concurrent jobs).
            elif _multi_recording_batch_state.current_phase == "extract":
                while (
                    len(_multi_recording_batch_state.active_threads)
                    < _multi_recording_batch_state.max_parallel_extracts
                    and _multi_recording_batch_state.phase_queue
                ):
                    animal_key, job_id = _multi_recording_batch_state.phase_queue.pop(0)
                    thread = Thread(
                        target=_pipeline_worker,
                        kwargs={
                            "configuration_path": _multi_recording_batch_state.configuration_paths[animal_key],
                            "job_id": job_id,
                            "single_recording": False,
                        },
                        daemon=True,
                    )
                    thread.start()
                    _multi_recording_batch_state.active_threads[job_id] = thread

                # Exits when all extract jobs have been dispatched and completed.
                if not _multi_recording_batch_state.active_threads and not _multi_recording_batch_state.phase_queue:
                    break

        # Polls at 1-second intervals before checking again.
        timer.delay(delay=1000, allow_sleep=True)


def _read_single_recording_tracker(tracker_path: Path, recording_path: Path) -> dict[str, object]:
    """Reads a single-recording ProcessingTracker and returns structured status information.

    Args:
        tracker_path: The path to the ProcessingTracker YAML file.
        recording_path: The path to the recording directory (for display purposes).

    Returns:
        A dictionary containing the recording path, tracker path, per-phase job status, summary counts, and an
        overall synthesized status string.
    """
    tracker = ProcessingTracker(file_path=tracker_path)
    summary = tracker.get_summary()

    # Groups jobs by pipeline phase using find_jobs.
    binarize_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.BINARIZE)
    process_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.PROCESS)
    combine_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.COMBINE)

    # Reads binarize status.
    binarize_status: dict[str, object] = {}
    for job_id in binarize_jobs:
        job_info = tracker.get_job_info(job_id=job_id)
        binarize_status["status"] = job_info.status.name.lower()
        if job_info.error_message:
            binarize_status["error"] = job_info.error_message

    # Reads per-plane process status.
    process_status: dict[str, object] = {}
    for job_id, (_, specifier) in process_jobs.items():
        job_info = tracker.get_job_info(job_id=job_id)
        process_status[specifier] = job_info.status.name.lower()

    # Reads combine status.
    combine_status: dict[str, object] = {}
    for job_id in combine_jobs:
        job_info = tracker.get_job_info(job_id=job_id)
        combine_status["status"] = job_info.status.name.lower()
        if job_info.error_message:
            combine_status["error"] = job_info.error_message

    # Synthesizes overall status from tracker state.
    if tracker.complete:
        overall_status = "completed"
    elif tracker.encountered_error:
        overall_status = "failed"
    elif combine_jobs and any(
        tracker.get_job_info(job_id=job_id).status == ProcessingStatus.RUNNING for job_id in combine_jobs
    ):
        overall_status = "combining"
    elif process_jobs and any(
        tracker.get_job_info(job_id=job_id).status in (ProcessingStatus.RUNNING, ProcessingStatus.SUCCEEDED)
        for job_id in process_jobs
    ):
        overall_status = "processing"
    elif binarize_jobs and any(
        tracker.get_job_info(job_id=job_id).status in (ProcessingStatus.RUNNING, ProcessingStatus.SUCCEEDED)
        for job_id in binarize_jobs
    ):
        overall_status = "binarizing"
    else:
        overall_status = "scheduled"

    # Formats summary counts using ProcessingStatus enum names.
    summary_counts: dict[str, int] = {status.name.lower(): count for status, count in summary.items()}

    return {
        "success": True,
        "recording_path": str(recording_path),
        "tracker_path": str(tracker_path),
        "status": overall_status,
        "jobs": {
            "binarize": binarize_status,
            "process": process_status,
            "combine": combine_status,
        },
        "summary": summary_counts,
    }


def _read_multi_recording_tracker(tracker_path: Path) -> dict[str, object]:
    """Reads a multi-recording ProcessingTracker and returns structured status information.

    Args:
        tracker_path: The path to the ProcessingTracker YAML file.

    Returns:
        A dictionary containing the tracker path, per-phase job status, summary counts, and an overall synthesized
        status string.
    """
    tracker = ProcessingTracker(file_path=tracker_path)
    summary = tracker.get_summary()

    # Groups jobs by pipeline phase using find_jobs.
    discover_jobs = tracker.find_jobs(job_name=MultiRecordingJobNames.DISCOVER)
    extract_jobs = tracker.find_jobs(job_name=MultiRecordingJobNames.EXTRACT)

    # Reads discover status.
    discover_status: dict[str, object] = {}
    for job_id in discover_jobs:
        job_info = tracker.get_job_info(job_id=job_id)
        discover_status["status"] = job_info.status.name.lower()
        if job_info.error_message:
            discover_status["error"] = job_info.error_message

    # Reads per-recording extract status.
    extract_status: dict[str, object] = {}
    for job_id, (_, specifier) in extract_jobs.items():
        job_info = tracker.get_job_info(job_id=job_id)
        extract_status[specifier] = job_info.status.name.lower()

    # Synthesizes overall status from tracker state.
    if tracker.complete:
        overall_status = "completed"
    elif tracker.encountered_error:
        overall_status = "failed"
    elif extract_jobs and any(
        tracker.get_job_info(job_id=job_id).status in (ProcessingStatus.RUNNING, ProcessingStatus.SUCCEEDED)
        for job_id in extract_jobs
    ):
        overall_status = "extracting"
    elif discover_jobs and any(
        tracker.get_job_info(job_id=job_id).status in (ProcessingStatus.RUNNING, ProcessingStatus.SUCCEEDED)
        for job_id in discover_jobs
    ):
        overall_status = "discovering"
    else:
        overall_status = "scheduled"

    # Formats summary counts using ProcessingStatus enum names.
    summary_counts: dict[str, int] = {status.name.lower(): count for status, count in summary.items()}

    return {
        "tracker_path": str(tracker_path),
        "status": overall_status,
        "jobs": {
            "discover": discover_status,
            "extract": extract_status,
        },
        "summary": summary_counts,
    }
