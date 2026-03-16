"""Provides MCP tools for batch processing, status monitoring, and cancellation of neural imaging pipelines.

These tools enable AI agents to start, monitor, and cancel both single-recording and multi-recording batch processing
operations. Processing state is tracked via ProcessingTracker YAML files that persist across restarts, rather than
in-memory state. Single-recording processing follows a three-phase workflow (binarize, process, combine), while
multi-recording processing follows a two-phase workflow (discover, extract).
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock, Thread
from dataclasses import field, dataclass

from natsort import natsorted
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import resolve_worker_count, resolve_parallel_job_capacity
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from ..io import resolve_multi_recording_contexts, resolve_single_recording_contexts
from ..pipelines import (
    MULTI_RECORDING_TRACKER_NAME,
    SINGLE_RECORDING_TRACKER_NAME,
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    run_multi_recording_pipeline,
    run_single_recording_pipeline,
)
from ..dataclasses import MultiRecordingConfiguration, SingleRecordingConfiguration
from .mcp_instance import mcp

_RESERVED_CORES: int = 2
"""The number of CPU cores reserved for system operations."""

_MAXIMUM_PARALLEL_IO_JOBS: int = 3
"""The maximum number of concurrent I/O-bound jobs (binarize and combine phases)."""

_MINIMUM_RECORDING_COUNT: int = 2
"""The minimum number of recordings required for multi-recording processing."""

_PREFERRED_WORKERS_PER_JOB: int = 30
"""The preferred number of CPU cores per parallel processing job for optimal throughput."""

_MINIMUM_WORKERS_PER_JOB: int = 10
"""The minimum number of CPU cores required per job when running multiple jobs in parallel."""

_WORKER_MULTIPLE: int = 5
"""Worker counts are rounded down to the nearest multiple of this value for clean allocation."""


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
    active_threads: dict[tuple[str, str], Thread] = field(default_factory=dict)
    """Currently running (recording_key, job_id) to Thread mapping."""

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

    Job completion and failure states are persisted in per-dataset ProcessingTracker YAML files. This dataclass only
    holds the minimal state needed for thread orchestration: phase queues, active threads, resource limits, and
    tracker/configuration path mappings.
    """

    tracker_paths: dict[str, Path] = field(default_factory=dict)
    """Dataset key to ProcessingTracker file path mapping."""
    configuration_paths: dict[str, Path] = field(default_factory=dict)
    """Dataset key to configuration file path mapping."""
    recording_paths: dict[str, list[Path]] = field(default_factory=dict)
    """Dataset key to list of recording directory paths mapping."""

    discover_jobs: dict[str, str] = field(default_factory=dict)
    """Dataset key to discover job ID mapping."""
    extract_jobs: dict[str, list[str]] = field(default_factory=dict)
    """Dataset key to ordered list of extract job IDs mapping."""

    current_phase: str = "discover"
    """Current processing phase: 'discover' or 'extract'."""
    phase_queue: list[tuple[str, str]] = field(default_factory=list)
    """Ordered (dataset_key, job_id) pairs queued for dispatch in the current phase."""
    active_threads: dict[tuple[str, str], Thread] = field(default_factory=dict)
    """Currently running (dataset_key, job_id) to Thread mapping."""

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

    Reads the tracker YAML file at <recording_path>/cindra/single_recording_tracker.yaml to determine how far processing
    has progressed. Returns per-job status grouped by pipeline phase (binarize, process, combine) and an overall status
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

    tracker_path = recording / "cindra" / SINGLE_RECORDING_TRACKER_NAME
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
        dataset_name = tracker_file.parent.name.lower()
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
        recording_paths: List of absolute paths to recording root directories (used as file_io.data_path per
            recording). These should be session-level roots, not sub-paths to raw data; the pipeline resolves
            raw data locations internally via recursive search.
        configuration_path: The absolute path to the template configuration YAML file.
        recording_output_paths: Optional list of absolute paths for per-recording output directories (used as
            file_io.output_path). Must match the length of recording_paths when provided. When not provided, each
            recording's output_path defaults to its data_path.
        workers_per_plane: CPU cores per plane job (-1 for automatic, limited by cpu_count - 2).
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

    # Creates per-recording configurations, resolves plane counts, and initializes ProcessingTracker files.
    # Defers configuration saving and resource allocation until total process jobs are known.
    batch_state = _SingleRecordingBatchState(lock=Lock())
    deferred_configs: list[tuple[SingleRecordingConfiguration, Path]] = []
    total_process_jobs = 0

    for data_path, output_path in zip(valid_paths, resolved_output_paths, strict=True):
        recording_key = str(data_path)

        # Creates a per-recording configuration copy with recording-specific paths and runtime settings.
        recording_configuration = SingleRecordingConfiguration.from_yaml(file_path=template_path)
        recording_configuration.file_io.data_path = data_path
        recording_configuration.file_io.output_path = output_path
        recording_configuration.runtime.display_progress_bars = progress_bars

        # Prepares the output directory and defers configuration saving until worker count is resolved.
        cindra_root = output_path / "cindra"
        cindra_root.mkdir(parents=True, exist_ok=True)
        recording_configuration_path = cindra_root / "configuration.yaml"
        batch_state.configuration_paths[recording_key] = recording_configuration_path
        deferred_configs.append((recording_configuration, recording_configuration_path))

        # Resolves plane count from configuration to build the complete job list.
        contexts = resolve_single_recording_contexts(configuration=recording_configuration)
        plane_count = len(contexts)
        total_process_jobs += plane_count

        # Builds the job list: binarize, all process planes, combine.
        jobs: list[tuple[str, str]] = [(SingleRecordingJobNames.BINARIZE, "")]
        jobs.extend((SingleRecordingJobNames.PROCESS, f"plane_{plane_index}") for plane_index in range(plane_count))
        jobs.append((SingleRecordingJobNames.COMBINE, ""))

        # Initializes the ProcessingTracker with all jobs for this recording.
        tracker_path = cindra_root / SINGLE_RECORDING_TRACKER_NAME
        tracker = ProcessingTracker(file_path=tracker_path)
        job_ids = tracker.initialize_jobs(jobs=jobs)
        batch_state.tracker_paths[recording_key] = tracker_path

        # Maps job IDs to phases for orchestration.
        batch_state.binarize_jobs[recording_key] = job_ids[0]
        batch_state.process_jobs[recording_key] = job_ids[1 : 1 + plane_count]
        batch_state.combine_jobs[recording_key] = job_ids[-1]

    # Calculates resource allocation now that total process jobs are known.
    budget = resolve_worker_count(requested_workers=-1, reserved_cores=_RESERVED_CORES)
    if workers_per_plane <= 0 and max_parallel_planes <= 0:
        actual_workers, actual_max_parallel = _resolve_saturating_allocation(budget, total_process_jobs)
    elif workers_per_plane > 0 >= max_parallel_planes:
        actual_workers = resolve_worker_count(requested_workers=workers_per_plane, reserved_cores=_RESERVED_CORES)
        actual_max_parallel = resolve_parallel_job_capacity(workers_per_job=actual_workers)
    elif workers_per_plane <= 0 < max_parallel_planes:
        raw_workers = budget // max_parallel_planes
        actual_workers = max(1, (raw_workers // _WORKER_MULTIPLE) * _WORKER_MULTIPLE)
        actual_max_parallel = max_parallel_planes
    else:
        actual_workers = resolve_worker_count(requested_workers=workers_per_plane, reserved_cores=_RESERVED_CORES)
        actual_max_parallel = max_parallel_planes

    batch_state.workers_per_plane = actual_workers
    batch_state.max_parallel_planes = actual_max_parallel

    # Saves all configurations with the resolved parallel_workers value.
    for recording_configuration, recording_configuration_path in deferred_configs:
        recording_configuration.runtime.parallel_workers = actual_workers
        recording_configuration.save(file_path=recording_configuration_path)

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

    Reads ProcessingTracker files from disk for each recording in the batch to report per-recording progress across
    all three phases (binarize, process, combine), including active, completed, and failed counts. All status
    information is derived from the on-disk tracker files rather than in-memory state.

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

            # Skips recordings whose tracker file no longer exists on disk.
            if not tracker_path.exists():
                continue

            tracker = ProcessingTracker(file_path=tracker_path)

            # Discovers job IDs from the tracker file by job name.
            binarize_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.BINARIZE)
            process_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.PROCESS)
            combine_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.COMBINE)

            # Reads binarize job status from disk.
            binarize_status = "pending"
            for job_id in binarize_jobs:
                binarize_status = tracker.get_job_status(job_id=job_id).name.lower()

            # Reads process job statuses from disk.
            plane_count = len(process_jobs)
            process_succeeded = sum(
                1 for job_id in process_jobs if tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED
            )
            process_failed = sum(
                1 for job_id in process_jobs if tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED
            )

            if process_failed:
                process_status = f"{process_succeeded}/{plane_count} (failed: {process_failed})"
            else:
                process_status = f"{process_succeeded}/{plane_count}"

            # Reads combine job status from disk.
            combine_status = "pending"
            for job_id in combine_jobs:
                combine_status = tracker.get_job_status(job_id=job_id).name.lower()

            # Synthesizes overall recording status from the on-disk tracker state.
            all_job_ids = [*binarize_jobs, *process_jobs, *combine_jobs]
            if tracker.complete:
                overall_status = "SUCCEEDED"
                total_succeeded += 1
            elif tracker.encountered_error:
                overall_status = "FAILED"
                total_failed += 1
            elif any(tracker.get_job_status(job_id=job_id) == ProcessingStatus.RUNNING for job_id in all_job_ids):
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
            for job_id in all_job_ids:
                job_info = tracker.get_job_info(job_id=job_id)
                if job_info.error_message:
                    errors.append(f"{job_info.job_name}({job_info.specifier}): {job_info.error_message}")
            if errors:
                recording_status["errors"] = errors

            recordings_status.append(recording_status)

        # Derives current phase from tracker state rather than in-memory orchestration state.
        current_phase = "none"
        if recordings_status:
            has_pending_or_running_binarize = any(ds["binarize"] in ("pending", "running") for ds in recordings_status)
            has_pending_or_running_combine = any(ds["combine"] in ("pending", "running") for ds in recordings_status)
            if has_pending_or_running_binarize:
                current_phase = "binarize"
            elif has_pending_or_running_combine:
                current_phase = "process"
            else:
                current_phase = "combine"

        summary = {
            "total": len(recordings_status),
            "succeeded": total_succeeded,
            "failed": total_failed,
            "running": total_running,
        }

        return {
            "current_phase": current_phase,
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
    dataset_configurations: list[dict[str, object]],
    *,
    workers_per_discover: int = -1,
    workers_per_extract: int = -1,
    progress_bars: bool = False,
) -> dict[str, object]:
    """Starts batch multi-recording processing for multiple datasets.

    Manages a two-phase batch workflow: discover (parallel by dataset), extract (parallel by recording). Each dataset
    configuration specifies a configuration file and its associated recording paths. Initializes ProcessingTracker
    files per dataset and dispatches jobs via background threads. Use get_multi_recording_batch_processing_status_tool
    to monitor progress.

    Args:
        dataset_configurations: List of dataset configurations, each a dictionary with 'configuration_path' (absolute
            path to the multi-recording YAML configuration), 'recording_paths' (list of absolute paths to recording
            directories), and 'dataset_name' (unique name for this dataset). At least 2 recording paths per dataset
            are required.
        workers_per_discover: Workers for discover phase (-1 for automatic via saturating allocation).
        workers_per_extract: Workers for extract phase (-1 for automatic via saturating allocation).
        progress_bars: Determines whether to display progress bars during processing.

    Returns:
        On success, contains a 'started' flag, 'total_datasets' and 'total_recordings' counts, worker allocation
        settings, and any 'invalid_configurations' that were skipped. On failure, contains an 'error' describing
        the issue.
    """
    global _multi_recording_batch_state

    if not dataset_configurations:
        return {
            "error": "Unable to start multi-recording batch processing. At least one dataset configuration is "
            "required.",
        }

    # Validates dataset configurations.
    valid_datasets: list[tuple[str, Path, list[Path]]] = []
    invalid_configurations: list[str] = []

    for dataset_configuration in dataset_configurations:
        required_keys = {"configuration_path", "recording_paths", "dataset_name"}
        if not required_keys.issubset(dataset_configuration):
            invalid_configurations.append(f"Missing required keys: {dataset_configuration}")
            continue

        dataset_name = str(dataset_configuration["dataset_name"]).strip()
        if not dataset_name:
            invalid_configurations.append(f"Empty dataset_name: {dataset_configuration}")
            continue

        dataset_configuration_path = Path(str(dataset_configuration["configuration_path"]))
        if not dataset_configuration_path.exists():
            invalid_configurations.append(f"Configuration not found: {dataset_configuration_path}")
            continue

        raw_recording_paths = dataset_configuration["recording_paths"]
        if not isinstance(raw_recording_paths, list):
            invalid_configurations.append(f"recording_paths must be a list: {dataset_configuration_path}")
            continue
        dataset_recording_paths = [Path(str(path)) for path in raw_recording_paths]
        if len(dataset_recording_paths) < _MINIMUM_RECORDING_COUNT:
            invalid_configurations.append(f"Need at least 2 recordings: {dataset_configuration_path}")
            continue

        invalid_recordings = [str(path) for path in dataset_recording_paths if not path.exists() or not path.is_dir()]
        if invalid_recordings:
            invalid_configurations.append(f"Invalid recordings for {dataset_configuration_path}: {invalid_recordings}")
            continue

        # Validates the configuration file format.
        try:
            MultiRecordingConfiguration.from_yaml(file_path=dataset_configuration_path)
        except Exception as error:
            invalid_configurations.append(f"Unable to load configuration {dataset_configuration_path}: {error}")
            continue

        dataset_key = dataset_name.lower()
        valid_datasets.append((dataset_key, dataset_configuration_path, dataset_recording_paths))

    if not valid_datasets:
        return {
            "error": "Unable to start multi-recording batch processing. No valid dataset configurations provided.",
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

    # Calculates resource allocation using saturating allocation for auto (-1) parameters.
    budget = resolve_worker_count(requested_workers=-1, reserved_cores=_RESERVED_CORES)
    total_discover_jobs = len(valid_datasets)
    total_extract_jobs = sum(len(rp) for _, _, rp in valid_datasets)

    if workers_per_discover <= 0:
        actual_workers_discover, max_parallel_discovers = _resolve_saturating_allocation(budget, total_discover_jobs)
    else:
        actual_workers_discover = resolve_worker_count(
            requested_workers=workers_per_discover, reserved_cores=_RESERVED_CORES
        )
        max_parallel_discovers = resolve_parallel_job_capacity(workers_per_job=actual_workers_discover)

    if workers_per_extract <= 0:
        actual_workers_extract, max_parallel_extracts = _resolve_saturating_allocation(budget, total_extract_jobs)
    else:
        actual_workers_extract = resolve_worker_count(
            requested_workers=workers_per_extract, reserved_cores=_RESERVED_CORES
        )
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
    for dataset_key, dataset_configuration_path, dataset_recording_paths in valid_datasets:
        # Loads the template configuration and applies runtime-specific overrides. The template file is never modified.
        configuration = MultiRecordingConfiguration.from_yaml(file_path=dataset_configuration_path)
        configuration.recording_io.dataset_name = dataset_key
        configuration.recording_io.recording_directories = tuple(natsorted(dataset_recording_paths))
        configuration.runtime.parallel_workers = actual_workers_discover
        configuration.runtime.display_progress_bars = progress_bars

        # Resolves contexts to determine recording IDs and the output path for the fine-tuned configuration.
        contexts = resolve_multi_recording_contexts(configuration=configuration)
        recording_ids = [context.runtime.io.recording_id for context in contexts]
        main_recording_path = contexts[0].runtime.output_path

        if main_recording_path is None:
            invalid_configurations.append(f"Unable to resolve output path for dataset '{dataset_key}'.")
            continue

        # Saves the resolved configuration to the canonical multi-recording output location.
        configuration_path = main_recording_path / "multi_recording_configuration.yaml"
        configuration.save(file_path=configuration_path)

        # Builds the job list: discover, then extract per recording.
        jobs: list[tuple[str, str]] = [(MultiRecordingJobNames.DISCOVER, "")]
        jobs.extend((MultiRecordingJobNames.EXTRACT, recording_id) for recording_id in recording_ids)

        # Initializes the ProcessingTracker with all jobs for this dataset.
        tracker_path = main_recording_path / MULTI_RECORDING_TRACKER_NAME
        tracker = ProcessingTracker(file_path=tracker_path)
        job_ids = tracker.initialize_jobs(jobs=jobs)

        # Stores per-dataset state for orchestration. Points to the resolved copy, not the template.
        batch_state.tracker_paths[dataset_key] = tracker_path
        batch_state.configuration_paths[dataset_key] = configuration_path
        batch_state.recording_paths[dataset_key] = dataset_recording_paths
        batch_state.discover_jobs[dataset_key] = job_ids[0]
        batch_state.extract_jobs[dataset_key] = job_ids[1:]
        total_recordings += len(recording_ids)

    # Populates the discover phase queue (naturally sorted for deterministic order).
    for dataset_key in natsorted(batch_state.tracker_paths.keys()):
        batch_state.phase_queue.append((dataset_key, batch_state.discover_jobs[dataset_key]))

    # Activates the batch state and starts the manager thread.
    _multi_recording_batch_state = batch_state
    manager = Thread(target=_multi_recording_batch_manager, daemon=True)
    manager.start()
    _multi_recording_batch_state.manager_thread = manager

    result: dict[str, object] = {
        "started": True,
        "total_datasets": len(batch_state.tracker_paths),
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

    Reads ProcessingTracker files from disk for each dataset in the batch to report per-dataset progress across both
    phases (discover, extract), including active, completed, and failed counts. All status information is derived from
    the on-disk tracker files rather than in-memory state.

    Returns:
        Contains the 'current_phase', per-dataset 'datasets' status list, and a 'summary' with aggregate counts for
        total_datasets, succeeded, failed, and running. Returns empty state when no batch processing has been started.
    """
    if _multi_recording_batch_state is None:
        return {
            "current_phase": "none",
            "datasets": [],
            "summary": {"total_datasets": 0, "succeeded": 0, "failed": 0, "running": 0},
        }

    with _multi_recording_batch_state.lock:
        datasets_status: list[dict[str, object]] = []
        total_succeeded = 0
        total_failed = 0
        total_running = 0

        for dataset_key in natsorted(_multi_recording_batch_state.tracker_paths.keys()):
            tracker_path = _multi_recording_batch_state.tracker_paths[dataset_key]

            # Skips datasets whose tracker file no longer exists on disk.
            if not tracker_path.exists():
                continue

            tracker = ProcessingTracker(file_path=tracker_path)

            # Discovers job IDs from the tracker file by job name.
            discover_jobs = tracker.find_jobs(job_name=MultiRecordingJobNames.DISCOVER)
            extract_jobs = tracker.find_jobs(job_name=MultiRecordingJobNames.EXTRACT)

            # Reads discover job status from disk.
            discover_status = "pending"
            for job_id in discover_jobs:
                discover_status = tracker.get_job_status(job_id=job_id).name.lower()

            # Reads extract job statuses from disk.
            extract_total = len(extract_jobs)
            extract_succeeded = sum(
                1 for job_id in extract_jobs if tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED
            )
            extract_failed = sum(
                1 for job_id in extract_jobs if tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED
            )

            # Synthesizes overall dataset status from the on-disk tracker state.
            if tracker.complete:
                overall_status = "SUCCEEDED"
                total_succeeded += 1
            elif tracker.encountered_error:
                overall_status = "FAILED"
                total_failed += 1
            elif any(
                tracker.get_job_status(job_id=job_id) == ProcessingStatus.RUNNING
                for job_id in [*discover_jobs, *extract_jobs]
            ):
                overall_status = "PROCESSING"
                total_running += 1
            else:
                overall_status = "QUEUED"

            dataset_status: dict[str, object] = {
                "dataset_key": dataset_key,
                "status": overall_status,
                "discover": discover_status,
                "extract_completed": extract_succeeded,
                "extract_failed": extract_failed,
                "extract_total": extract_total,
            }

            # Includes error messages from any failed jobs.
            errors: list[str] = []
            for job_id in [*discover_jobs, *extract_jobs]:
                job_info = tracker.get_job_info(job_id=job_id)
                if job_info.error_message:
                    errors.append(f"{job_info.job_name}({job_info.specifier}): {job_info.error_message}")
            if errors:
                dataset_status["errors"] = errors

            datasets_status.append(dataset_status)

        # Derives current phase from tracker state rather than in-memory orchestration state.
        current_phase = "none"
        if datasets_status:
            has_running_discover = any(ds["discover"] == "running" for ds in datasets_status)
            has_pending_discover = any(ds["discover"] == "pending" for ds in datasets_status)
            current_phase = "discover" if has_running_discover or has_pending_discover else "extract"

        summary = {
            "total_datasets": len(datasets_status),
            "succeeded": total_succeeded,
            "failed": total_failed,
            "running": total_running,
        }

        return {
            "current_phase": current_phase,
            "datasets": datasets_status,
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


def _resolve_saturating_allocation(budget: int, total_jobs: int) -> tuple[int, int]:
    """Resolves worker and parallelism counts to saturate available cores across multiple jobs.

    Prefers ~30 workers per job and distributes the CPU budget across as many concurrent jobs as possible, subject to
    a minimum of 10 workers per job when running in parallel. Worker counts are rounded down to the nearest multiple
    of 5 for clean allocation.

    Args:
        budget: The total number of available CPU cores (after reserving system cores).
        total_jobs: The total number of jobs to execute.

    Returns:
        A (workers_per_job, max_parallel_jobs) tuple.
    """
    max_at_preferred = max(1, budget // _PREFERRED_WORKERS_PER_JOB)
    max_parallel = min(total_jobs, max_at_preferred)
    raw_workers = budget // max_parallel
    workers = max(1, (raw_workers // _WORKER_MULTIPLE) * _WORKER_MULTIPLE)

    # Reduces parallelism until each job has at least the minimum worker count.
    while workers < _MINIMUM_WORKERS_PER_JOB and max_parallel > 1:
        max_parallel -= 1
        raw_workers = budget // max_parallel
        workers = max(1, (raw_workers // _WORKER_MULTIPLE) * _WORKER_MULTIPLE)

    return workers, max_parallel


def _pipeline_worker(
    configuration_path: Path, job_id: str, tracker_path: Path, *, single_recording: bool = True
) -> None:
    """Executes a single pipeline job identified by its job ID.

    Calls the appropriate pipeline function in REMOTE mode, passing the job_id so the pipeline reads the job definition
    from the ProcessingTracker and updates tracker state on completion or failure. After the pipeline returns or raises,
    verifies that the tracker reached a terminal state and marks the job as failed if the pipeline terminated without
    updating the tracker.

    Args:
        configuration_path: The path to the recording or dataset configuration file.
        job_id: The unique hexadecimal job identifier registered in the ProcessingTracker.
        tracker_path: The path to the ProcessingTracker file for this job.
        single_recording: Determines whether to call the single-recording or multi-recording pipeline.
    """
    try:
        if single_recording:
            run_single_recording_pipeline(configuration_path=configuration_path, job_id=job_id)
        else:
            run_multi_recording_pipeline(configuration_path=configuration_path, job_id=job_id)
    except Exception:  # noqa: S110 - Pipeline may have persisted failure via tracker.fail_job() before re-raising.
        pass
    finally:
        tracker = ProcessingTracker(file_path=tracker_path)
        if tracker.get_job_status(job_id=job_id) not in (ProcessingStatus.SUCCEEDED, ProcessingStatus.FAILED):
            tracker.fail_job(
                job_id=job_id,
                error_message="Unable to complete job. Worker terminated without reaching a terminal state.",
            )


def _single_recording_batch_manager() -> None:
    """Orchestrates three-phase single-recording batch processing: binarize, process, combine.

    Runs as a daemon thread, polling at 1-second intervals to dispatch new jobs and advance between phases.
    Binarize and combine phases run up to 3 concurrent jobs (I/O bound), while the process phase runs jobs in parallel
    up to the configured max_parallel_planes limit. Job state is tracked via ProcessingTracker files; this
    manager only handles thread orchestration and phase transitions.
    """
    timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

    while True:
        state = _single_recording_batch_state
        if state is None:
            return

        with state.lock:
            # Cleans up completed threads.
            completed_keys = [
                thread_key for thread_key, thread in state.active_threads.items() if not thread.is_alive()
            ]
            for thread_key in completed_keys:
                state.active_threads.pop(thread_key, None)

            # Phase 1: BINARIZE (parallel — up to _MAXIMUM_PARALLEL_IO_JOBS concurrent jobs).
            if state.current_phase == "binarize":
                while len(state.active_threads) < _MAXIMUM_PARALLEL_IO_JOBS and state.phase_queue:
                    recording_key, job_id = state.phase_queue.pop(0)
                    thread = Thread(
                        target=_pipeline_worker,
                        kwargs={
                            "configuration_path": state.configuration_paths[recording_key],
                            "job_id": job_id,
                            "tracker_path": state.tracker_paths[recording_key],
                            "single_recording": True,
                        },
                        daemon=True,
                    )
                    thread.start()
                    state.active_threads[(recording_key, job_id)] = thread

                # Transitions to process phase when all binarize jobs have been dispatched and completed.
                if not state.active_threads and not state.phase_queue:
                    for recording_key in natsorted(state.tracker_paths.keys()):
                        tracker = ProcessingTracker(file_path=state.tracker_paths[recording_key])
                        binarize_status = tracker.get_job_status(job_id=state.binarize_jobs[recording_key])
                        if binarize_status == ProcessingStatus.SUCCEEDED:
                            for process_job_id in state.process_jobs[recording_key]:
                                state.phase_queue.append((recording_key, process_job_id))
                    state.current_phase = "process"

            # Phase 2: PROCESS (parallel — up to max_parallel_planes concurrent jobs).
            elif state.current_phase == "process":
                while len(state.active_threads) < state.max_parallel_planes and state.phase_queue:
                    recording_key, job_id = state.phase_queue.pop(0)
                    thread = Thread(
                        target=_pipeline_worker,
                        kwargs={
                            "configuration_path": state.configuration_paths[recording_key],
                            "job_id": job_id,
                            "tracker_path": state.tracker_paths[recording_key],
                            "single_recording": True,
                        },
                        daemon=True,
                    )
                    thread.start()
                    state.active_threads[(recording_key, job_id)] = thread

                # Transitions to combine phase when all process jobs have been dispatched and completed.
                if not state.active_threads and not state.phase_queue:
                    for recording_key in natsorted(state.tracker_paths.keys()):
                        tracker = ProcessingTracker(file_path=state.tracker_paths[recording_key])
                        process_job_ids = state.process_jobs[recording_key]
                        all_succeeded = all(
                            tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED
                            for job_id in process_job_ids
                        )
                        if all_succeeded:
                            state.phase_queue.append((recording_key, state.combine_jobs[recording_key]))
                    state.current_phase = "combine"

            # Phase 3: COMBINE (parallel — up to _MAXIMUM_PARALLEL_IO_JOBS concurrent jobs, I/O bound).
            elif state.current_phase == "combine":
                while len(state.active_threads) < _MAXIMUM_PARALLEL_IO_JOBS and state.phase_queue:
                    recording_key, job_id = state.phase_queue.pop(0)
                    thread = Thread(
                        target=_pipeline_worker,
                        kwargs={
                            "configuration_path": state.configuration_paths[recording_key],
                            "job_id": job_id,
                            "tracker_path": state.tracker_paths[recording_key],
                            "single_recording": True,
                        },
                        daemon=True,
                    )
                    thread.start()
                    state.active_threads[(recording_key, job_id)] = thread

                # Exits when all combine jobs have been dispatched and completed.
                if not state.active_threads and not state.phase_queue:
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

    while True:
        state = _multi_recording_batch_state
        if state is None:
            return

        with state.lock:
            # Cleans up completed threads.
            completed_keys = [
                thread_key for thread_key, thread in state.active_threads.items() if not thread.is_alive()
            ]
            for thread_key in completed_keys:
                state.active_threads.pop(thread_key, None)

            # Phase 1: DISCOVER (parallel — up to max_parallel_discovers concurrent jobs).
            if state.current_phase == "discover":
                while len(state.active_threads) < state.max_parallel_discovers and state.phase_queue:
                    dataset_key, job_id = state.phase_queue.pop(0)
                    thread = Thread(
                        target=_pipeline_worker,
                        kwargs={
                            "configuration_path": state.configuration_paths[dataset_key],
                            "job_id": job_id,
                            "tracker_path": state.tracker_paths[dataset_key],
                            "single_recording": False,
                        },
                        daemon=True,
                    )
                    thread.start()
                    state.active_threads[(dataset_key, job_id)] = thread

                # Transitions to extract phase when all discover jobs have been dispatched and completed.
                if not state.active_threads and not state.phase_queue:
                    # Builds extract queue from datasets whose discover job succeeded.
                    for dataset_key in natsorted(state.tracker_paths.keys()):
                        tracker = ProcessingTracker(file_path=state.tracker_paths[dataset_key])
                        discover_status = tracker.get_job_status(job_id=state.discover_jobs[dataset_key])
                        if discover_status == ProcessingStatus.SUCCEEDED:
                            for extract_job_id in state.extract_jobs[dataset_key]:
                                state.phase_queue.append((dataset_key, extract_job_id))

                            # Updates configuration to use extract-phase worker count.
                            config = MultiRecordingConfiguration.from_yaml(
                                file_path=state.configuration_paths[dataset_key]
                            )
                            config.runtime.parallel_workers = state.workers_per_extract
                            config.save(file_path=state.configuration_paths[dataset_key])

                    state.current_phase = "extract"

            # Phase 2: EXTRACT (parallel — up to max_parallel_extracts concurrent jobs).
            elif state.current_phase == "extract":
                while len(state.active_threads) < state.max_parallel_extracts and state.phase_queue:
                    dataset_key, job_id = state.phase_queue.pop(0)
                    thread = Thread(
                        target=_pipeline_worker,
                        kwargs={
                            "configuration_path": state.configuration_paths[dataset_key],
                            "job_id": job_id,
                            "tracker_path": state.tracker_paths[dataset_key],
                            "single_recording": False,
                        },
                        daemon=True,
                    )
                    thread.start()
                    state.active_threads[(dataset_key, job_id)] = thread

                # Exits when all extract jobs have been dispatched and completed.
                if not state.active_threads and not state.phase_queue:
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
