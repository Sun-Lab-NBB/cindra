"""Provides MCP tools for preparing, executing, monitoring, and cancelling neural imaging pipeline jobs.

These tools give agents fine-grained control over pipeline execution: prepare builds an execution manifest without
running anything, execute dispatches selected jobs with prerequisite validation, reset selectively reverts completed
phases for re-runs, and status/cancel manage the active execution. Both single-recording (three-phase: binarize,
process, combine) and multi-recording (two-phase: discover, extract) pipelines are supported through a unified
execution model.
"""

from __future__ import annotations

import shutil
from typing import Any
from pathlib import Path
from threading import Lock, Thread
from dataclasses import field, dataclass

import yaml  # type: ignore[import-untyped]
from natsort import natsorted
from ataraxis_time import PrecisionTimer, TimerPrecisions, TimestampFormats, TimestampPrecisions, get_timestamp
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

_MAXIMUM_PARALLEL_IO_JOBS: int = 4
"""The maximum number of concurrent I/O-bound jobs (binarize and combine phases)."""

_MINIMUM_RECORDING_COUNT: int = 2
"""The minimum number of recordings required for multi-recording processing."""

_PREFERRED_WORKERS_PER_JOB: int = 30
"""The preferred number of CPU cores per parallel processing job for optimal throughput."""

_MINIMUM_WORKERS_PER_JOB: int = 10
"""The minimum number of CPU cores required per job when running multiple jobs in parallel."""

_WORKER_MULTIPLE: int = 5
"""Worker counts are rounded down to the nearest multiple of this value for clean allocation."""

_IO_BOUND_JOB_NAMES: frozenset[str] = frozenset({SingleRecordingJobNames.BINARIZE, SingleRecordingJobNames.COMBINE})
"""Job names whose execution is I/O-bound and use a fixed concurrency limit instead of compute-bound allocation."""


@dataclass(slots=True)
class _PendingJob:
    """Describes a single job queued for execution."""

    configuration_path: Path
    """The path to the pipeline configuration file for this job."""
    tracker_path: Path
    """The path to the ProcessingTracker file that tracks this job."""
    job_id: str
    """The unique hexadecimal identifier for this job in the tracker."""
    single_recording: bool
    """Determines whether this job belongs to a single-recording or multi-recording pipeline."""
    io_bound: bool
    """Determines whether this job is I/O-bound (binarize, combine) and should use fixed concurrency limits."""

    @property
    def dispatch_key(self) -> tuple[str, str]:
        """Returns the composite key that uniquely identifies this job across the entire batch, combining the tracker
        path with the job ID.
        """
        return str(self.tracker_path), self.job_id


@dataclass(slots=True)
class _JobExecutionState:
    """Tracks the runtime state for generic job execution across both pipeline types.

    I/O-bound jobs (binarize, combine) and compute-bound jobs (process, discover, extract) are dispatched from
    separate queues with independent concurrency limits. I/O-bound concurrency is fixed at ``_MAXIMUM_PARALLEL_IO_JOBS``
    regardless of the ``max_parallel_jobs`` parameter.
    """

    all_jobs: dict[tuple[str, str], _PendingJob] = field(default_factory=dict)
    """All submitted jobs keyed by (tracker_path, job_id) dispatch key, used for status reporting."""
    io_pending_queue: list[_PendingJob] = field(default_factory=list)
    """I/O-bound jobs awaiting dispatch, capped at _MAXIMUM_PARALLEL_IO_JOBS concurrent."""
    compute_pending_queue: list[_PendingJob] = field(default_factory=list)
    """Compute-bound jobs awaiting dispatch, capped at max_parallel_jobs concurrent."""
    io_active_threads: dict[tuple[str, str], Thread] = field(default_factory=dict)
    """Currently running I/O-bound (tracker_path, job_id) dispatch key to Thread mapping."""
    compute_active_threads: dict[tuple[str, str], Thread] = field(default_factory=dict)
    """Currently running compute-bound (tracker_path, job_id) dispatch key to Thread mapping."""
    max_parallel_jobs: int = 1
    """The maximum number of compute-bound jobs to execute concurrently."""
    lock: Lock = field(default_factory=Lock)
    """Thread synchronization lock for execution state access."""
    manager_thread: Thread | None = None
    """Background execution manager thread reference."""
    phase_groups: list[list[_PendingJob]] = field(default_factory=list)
    """Ordered groups of jobs for phased execution, processed one group at a time by the manager."""


_job_execution_state: _JobExecutionState | None = None
"""The module-level execution state for active processing jobs."""


@mcp.tool()
def get_recording_status_tool(recording_path: str) -> dict[str, object]:
    """Gets the processing status for a recording by reading all available ProcessingTracker files.

    Checks for both single-recording and multi-recording trackers under the recording's cindra output directory and
    returns status for all pipelines found. For single-recording, reads the tracker at
    <recording_path>/cindra/single_recording_tracker.yaml and returns per-phase job status (binarize, process,
    combine). For multi-recording, searches under <recording_path>/cindra/multi_recording/<dataset>/ for tracker files
    and returns per-dataset status (discover, extract).

    Args:
        recording_path: The absolute path to the recording data directory.

    Returns:
        On success, contains the 'recording_path', 'single_recording' status (per-phase jobs, summary, and synthesized
        status string), and 'multi_recording' status (per-dataset tracker status). Each section reports 'not_started'
        when no tracker exists. On failure, contains an 'error' describing the issue. Both cases include a 'success'
        flag.
    """
    recording = Path(recording_path)

    if not recording.exists():
        return {
            "success": False,
            "error": f"Unable to get recording status. Recording directory not found: {recording_path}.",
        }

    # Reads single-recording tracker status.
    single_tracker_path = recording / "cindra" / SINGLE_RECORDING_TRACKER_NAME
    if single_tracker_path.exists():
        single_recording_status = _read_single_recording_tracker(
            tracker_path=single_tracker_path, recording_path=recording
        )
    else:
        single_recording_status = {"status": "not_started"}

    # Reads multi-recording tracker status from all datasets.
    multi_recording_status: dict[str, object]
    multi_recording_base = recording / "cindra" / "multi_recording"
    if multi_recording_base.exists():
        tracker_files = list(multi_recording_base.rglob(MULTI_RECORDING_TRACKER_NAME))
        if tracker_files:
            datasets: dict[str, object] = {}
            for tracker_file in natsorted(tracker_files):
                dataset_key = tracker_file.parent.name
                datasets[dataset_key] = _read_multi_recording_tracker(tracker_path=tracker_file)
            multi_recording_status = {"datasets": datasets}
        else:
            multi_recording_status = {"status": "not_started"}
    else:
        multi_recording_status = {"status": "not_started"}

    return {
        "success": True,
        "recording_path": str(recording),
        "single_recording": single_recording_status,
        "multi_recording": multi_recording_status,
    }


@mcp.tool()
def get_batch_status_overview_tool(root_directory: str) -> dict[str, object]:
    """Discovers and summarizes processing status for all recordings and datasets under a root directory.

    Searches recursively for single-recording and multi-recording ProcessingTracker files, reads each tracker to
    determine per-recording or per-dataset processing progress, and aggregates summary counts across all discovered
    pipelines. Use this for a bird's-eye view of batch processing progress across an entire data directory tree.

    Args:
        root_directory: The absolute path to the root directory to search.

    Returns:
        On success, contains 'single_recordings' and 'multi_recordings' lists with per-tracker status, and a
        'summary' with aggregate counts for completed, failed, in_progress, and not_started pipelines. On failure,
        contains an 'error' describing the issue. Both cases include a 'success' flag.
    """
    root = Path(root_directory)

    if not root.exists():
        return {
            "success": False,
            "error": f"Unable to get batch status overview. Directory not found: {root_directory}.",
        }

    if not root.is_dir():
        return {
            "success": False,
            "error": f"Unable to get batch status overview. Path is not a directory: {root_directory}.",
        }

    permission_errors: list[str] = []

    # Discovers single-recording tracker files.
    single_tracker_paths: list[Path] = []
    try:
        single_tracker_paths.extend(root.rglob(SINGLE_RECORDING_TRACKER_NAME))
    except PermissionError as error:
        permission_errors.append(f"Access denied during single-recording search: {error}")

    # Discovers multi-recording tracker files.
    multi_tracker_paths: list[Path] = []
    try:
        multi_tracker_paths.extend(root.rglob(MULTI_RECORDING_TRACKER_NAME))
    except PermissionError as error:
        permission_errors.append(f"Access denied during multi-recording search: {error}")

    # Reads single-recording trackers. Derives recording_path from tracker location.
    single_recordings: list[dict[str, object]] = []
    for tracker_path in natsorted(single_tracker_paths, key=str):
        recording_path = tracker_path.parent.parent
        single_recordings.append(
            _read_single_recording_tracker(tracker_path=tracker_path, recording_path=recording_path)
        )

    # Reads multi-recording trackers. Extracts dataset name from parent directory.
    multi_recordings: list[dict[str, object]] = []
    for tracker_path in natsorted(multi_tracker_paths, key=str):
        dataset_name = tracker_path.parent.name
        entry = _read_multi_recording_tracker(tracker_path=tracker_path)
        entry["dataset_name"] = dataset_name
        multi_recordings.append(entry)

    # Aggregates summary counts from synthesized status strings.
    completed = 0
    failed = 0
    in_progress = 0
    not_started = 0

    for recording in single_recordings:
        status = recording.get("status", "")
        if status == "completed":
            completed += 1
        elif status == "failed":
            failed += 1
        elif status == "scheduled":
            not_started += 1
        else:
            in_progress += 1

    for dataset_entry in multi_recordings:
        status = dataset_entry.get("status", "")
        if status == "completed":
            completed += 1
        elif status == "failed":
            failed += 1
        elif status == "scheduled":
            not_started += 1
        else:
            in_progress += 1

    result: dict[str, object] = {
        "success": True,
        "root_directory": root_directory,
        "single_recordings": single_recordings,
        "multi_recordings": multi_recordings,
        "summary": {
            "total_single_recordings": len(single_recordings),
            "total_multi_recording_datasets": len(multi_recordings),
            "completed": completed,
            "failed": failed,
            "in_progress": in_progress,
            "not_started": not_started,
        },
    }

    if permission_errors:
        result["permission_errors"] = permission_errors

    return result


@mcp.tool()
def prepare_single_recording_batch_tool(
    recording_paths: list[str],
    configuration_path: str,
    recording_output_paths: list[str],
) -> dict[str, object]:
    """Prepares an execution manifest for single-recording batch processing without starting execution.

    For each recording, creates a per-recording configuration copy with recording-specific paths and runtime settings,
    resolves the plane count, and initializes a ProcessingTracker with all jobs (binarize, per-plane process, combine).
    Idempotent: if a tracker already exists for a recording, returns the existing manifest with current job statuses
    instead of reinitializing. Use execute_processing_jobs_tool to dispatch jobs from the manifest and
    reset_processing_phases_tool to selectively reset completed phases for re-runs.

    Important:
        Worker allocation and parallelism are controlled by execute_processing_jobs_tool, not this tool. The execute
        tool resolves resource allocation at dispatch time and rewrites configuration files for compute-bound jobs
        accordingly.

    Args:
        recording_paths: List of absolute paths to recording root directories (used as file_io.data_path per
            recording). These should be session-level roots, not sub-paths to raw data; the pipeline resolves
            raw data locations internally via recursive search.
        configuration_path: The absolute path to the template configuration YAML file.
        recording_output_paths: List of absolute paths for per-recording output directories (used as
            file_io.output_path). Must match the length of recording_paths.

    Returns:
        On success, contains per-recording manifests in 'recordings' keyed by recording path, with each entry listing
        its configuration_path, tracker_path, pipeline_type, and per-phase job entries (binarize_job, process_jobs,
        combine_job) including job_id, name, specifier, and current status. Also includes 'total_recordings' and
        'total_jobs' counts. On failure, contains an 'error' describing the issue.
    """
    if not recording_paths:
        return {"success": False, "error": "Unable to prepare batch. At least one recording path is required."}

    if len(recording_output_paths) != len(recording_paths):
        return {
            "success": False,
            "error": (
                f"Unable to prepare batch. The recording_output_paths length "
                f"({len(recording_output_paths)}) must match the recording_paths length ({len(recording_paths)})."
            ),
        }

    template_path = Path(configuration_path)
    if not template_path.exists():
        return {
            "success": False,
            "error": f"Unable to prepare batch. Configuration file not found: {configuration_path}.",
        }

    if template_path.suffix != ".yaml":
        return {
            "success": False,
            "error": f"Unable to prepare batch. Configuration file must be a .yaml file: {configuration_path}.",
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
            "success": False,
            "error": "Unable to prepare batch. No valid recording paths provided.",
            "invalid_paths": invalid_paths,
        }

    # Resolves per-recording output paths from the provided list.
    resolved_output_paths: list[Path] = [Path(recording_output_paths[index]) for index in valid_indices]

    # Builds the manifest for each recording.
    recordings_manifest: dict[str, dict[str, object]] = {}
    total_jobs = 0

    for data_path, output_path in zip(valid_paths, resolved_output_paths, strict=True):
        recording_key = str(data_path)
        cindra_root = output_path / "cindra"
        tracker_path = cindra_root / SINGLE_RECORDING_TRACKER_NAME

        if tracker_path.exists():
            # Idempotent path: tracker already exists, returns current state without reinitializing.
            tracker = ProcessingTracker(file_path=tracker_path)
            configuration_file_path = cindra_root / "configuration.yaml"

            binarize_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.BINARIZE)
            process_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.PROCESS)
            combine_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.COMBINE)

            binarize_entry: dict[str, object] = {}
            for job_id, (name, specifier) in binarize_jobs.items():
                binarize_entry = {
                    "job_id": job_id,
                    "name": name,
                    "specifier": specifier,
                    "status": tracker.get_job_status(job_id=job_id).name.lower(),
                }

            process_entries: list[dict[str, object]] = []
            for job_id, (name, specifier) in process_jobs.items():
                process_entries.append(
                    {
                        "job_id": job_id,
                        "name": name,
                        "specifier": specifier,
                        "status": tracker.get_job_status(job_id=job_id).name.lower(),
                    }
                )

            combine_entry: dict[str, object] = {}
            for job_id, (name, specifier) in combine_jobs.items():
                combine_entry = {
                    "job_id": job_id,
                    "name": name,
                    "specifier": specifier,
                    "status": tracker.get_job_status(job_id=job_id).name.lower(),
                }

            total_jobs += len(binarize_jobs) + len(process_jobs) + len(combine_jobs)

            recordings_manifest[recording_key] = {
                "configuration_path": str(configuration_file_path),
                "tracker_path": str(tracker_path),
                "pipeline_type": "single-recording",
                "binarize_job": binarize_entry,
                "process_jobs": process_entries,
                "combine_job": combine_entry,
            }
        else:
            # New recording: creates per-recording config, resolves planes, and initializes tracker.
            recording_configuration = SingleRecordingConfiguration.from_yaml(file_path=template_path)
            recording_configuration.file_io.data_path = data_path
            recording_configuration.file_io.output_path = output_path
            recording_configuration.runtime.display_progress_bars = False

            cindra_root.mkdir(parents=True, exist_ok=True)
            recording_configuration_path = cindra_root / "configuration.yaml"

            # Resolves plane count from configuration to build the complete job list.
            contexts = resolve_single_recording_contexts(configuration=recording_configuration)
            plane_count = len(contexts)

            # Saves per-recording configuration. The execute tool overwrites parallel_workers at dispatch time.
            recording_configuration.save(file_path=recording_configuration_path)

            # Builds the job list: binarize, all process planes, combine.
            jobs: list[tuple[str, str]] = [(SingleRecordingJobNames.BINARIZE, "")]
            jobs.extend((SingleRecordingJobNames.PROCESS, f"plane_{plane_index}") for plane_index in range(plane_count))
            jobs.append((SingleRecordingJobNames.COMBINE, ""))

            tracker = ProcessingTracker(file_path=tracker_path)
            job_ids = tracker.initialize_jobs(jobs=jobs)
            total_jobs += len(jobs)

            # Builds manifest entries from the freshly initialized tracker.
            binarize_entry = {
                "job_id": job_ids[0],
                "name": SingleRecordingJobNames.BINARIZE.value,
                "specifier": "",
                "status": "scheduled",
            }

            process_entries = [
                {
                    "job_id": job_ids[1 + plane_index],
                    "name": SingleRecordingJobNames.PROCESS.value,
                    "specifier": f"plane_{plane_index}",
                    "status": "scheduled",
                }
                for plane_index in range(plane_count)
            ]

            combine_entry = {
                "job_id": job_ids[-1],
                "name": SingleRecordingJobNames.COMBINE.value,
                "specifier": "",
                "status": "scheduled",
            }

            recordings_manifest[recording_key] = {
                "configuration_path": str(recording_configuration_path),
                "tracker_path": str(tracker_path),
                "pipeline_type": "single-recording",
                "binarize_job": binarize_entry,
                "process_jobs": process_entries,
                "combine_job": combine_entry,
            }

    result: dict[str, object] = {
        "success": True,
        "recordings": recordings_manifest,
        "total_recordings": len(recordings_manifest),
        "total_jobs": total_jobs,
    }

    if invalid_paths:
        result["invalid_paths"] = invalid_paths

    return result


@mcp.tool()
def prepare_multi_recording_batch_tool(
    dataset_configurations: list[dict[str, object]],
) -> dict[str, object]:
    """Prepares an execution manifest for multi-recording batch processing without starting execution.

    For each dataset, creates a configuration with resolved recording directories, resolves recording IDs, and
    initializes a ProcessingTracker with all jobs (discover, per-recording extract). Idempotent: if a tracker already
    exists for a dataset, returns the existing manifest with current job statuses instead of reinitializing. Use
    execute_processing_jobs_tool to dispatch jobs from the manifest and reset_processing_phases_tool to selectively
    reset completed phases for re-runs.

    Important:
        Worker allocation and parallelism are controlled by execute_processing_jobs_tool, not this tool. The execute
        tool resolves resource allocation at dispatch time and rewrites configuration files for compute-bound jobs
        accordingly.

    Args:
        dataset_configurations: List of dataset configurations, each a dictionary with 'configuration_path' (absolute
            path to the multi-recording YAML configuration), 'recording_paths' (list of absolute paths to recording
            directories), and 'dataset_name' (unique name for this dataset). At least 2 recording paths per dataset
            are required.

    Returns:
        On success, contains per-dataset manifests in 'datasets' keyed by dataset name, with each entry listing its
        configuration_path, tracker_path, pipeline_type, and per-phase job entries (discover_job, extract_jobs)
        including job_id, name, specifier, and current status. Also includes 'total_datasets' and 'total_jobs' counts.
        On failure, contains an 'error' describing the issue.
    """
    if not dataset_configurations:
        return {
            "success": False,
            "error": "Unable to prepare multi-recording batch. At least one dataset configuration is required.",
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
            "success": False,
            "error": "Unable to prepare multi-recording batch. No valid dataset configurations provided.",
            "invalid_configurations": invalid_configurations,
        }

    # Builds the manifest for each dataset.
    datasets_manifest: dict[str, dict[str, object]] = {}
    total_jobs = 0

    for dataset_key, dataset_configuration_path, dataset_recording_paths in valid_datasets:
        # Loads the template configuration and applies runtime-specific overrides.
        configuration = MultiRecordingConfiguration.from_yaml(file_path=dataset_configuration_path)
        configuration.recording_io.dataset_name = dataset_key
        configuration.recording_io.recording_directories = tuple(natsorted(dataset_recording_paths))
        configuration.runtime.display_progress_bars = False

        # Resolves contexts to determine recording IDs and the output path.
        contexts = resolve_multi_recording_contexts(configuration=configuration)
        recording_ids = [context.runtime.io.recording_id for context in contexts]
        main_recording_path = contexts[0].runtime.output_path

        if main_recording_path is None:
            invalid_configurations.append(f"Unable to resolve output path for dataset '{dataset_key}'.")
            continue

        tracker_path = main_recording_path / MULTI_RECORDING_TRACKER_NAME
        configuration_file_path = main_recording_path / "multi_recording_configuration.yaml"

        if tracker_path.exists():
            # Idempotent path: tracker already exists, returns current state without reinitializing.
            tracker = ProcessingTracker(file_path=tracker_path)

            discover_jobs = tracker.find_jobs(job_name=MultiRecordingJobNames.DISCOVER)
            extract_jobs = tracker.find_jobs(job_name=MultiRecordingJobNames.EXTRACT)

            discover_entry: dict[str, object] = {}
            for job_id, (name, specifier) in discover_jobs.items():
                discover_entry = {
                    "job_id": job_id,
                    "name": name,
                    "specifier": specifier,
                    "status": tracker.get_job_status(job_id=job_id).name.lower(),
                }

            extract_entries: list[dict[str, object]] = []
            for job_id, (name, specifier) in extract_jobs.items():
                extract_entries.append(
                    {
                        "job_id": job_id,
                        "name": name,
                        "specifier": specifier,
                        "status": tracker.get_job_status(job_id=job_id).name.lower(),
                    }
                )

            total_jobs += len(discover_jobs) + len(extract_jobs)

            datasets_manifest[dataset_key] = {
                "configuration_path": str(configuration_file_path),
                "tracker_path": str(tracker_path),
                "pipeline_type": "multi-recording",
                "discover_job": discover_entry,
                "extract_jobs": extract_entries,
            }
        else:
            # New dataset: saves configuration and initializes tracker.
            configuration.save(file_path=configuration_file_path)

            # Builds the job list: discover, then extract per recording.
            jobs: list[tuple[str, str]] = [(MultiRecordingJobNames.DISCOVER, "")]
            jobs.extend((MultiRecordingJobNames.EXTRACT, recording_id) for recording_id in recording_ids)

            tracker = ProcessingTracker(file_path=tracker_path)
            job_ids = tracker.initialize_jobs(jobs=jobs)
            total_jobs += len(jobs)

            discover_entry = {
                "job_id": job_ids[0],
                "name": MultiRecordingJobNames.DISCOVER.value,
                "specifier": "",
                "status": "scheduled",
            }

            extract_entries = [
                {
                    "job_id": job_ids[1 + index],
                    "name": MultiRecordingJobNames.EXTRACT.value,
                    "specifier": recording_ids[index],
                    "status": "scheduled",
                }
                for index in range(len(recording_ids))
            ]

            datasets_manifest[dataset_key] = {
                "configuration_path": str(configuration_file_path),
                "tracker_path": str(tracker_path),
                "pipeline_type": "multi-recording",
                "discover_job": discover_entry,
                "extract_jobs": extract_entries,
            }

    result: dict[str, object] = {
        "success": True,
        "datasets": datasets_manifest,
        "total_datasets": len(datasets_manifest),
        "total_jobs": total_jobs,
    }

    if invalid_configurations:
        result["invalid_configurations"] = invalid_configurations

    return result


@mcp.tool()
def reset_processing_phases_tool(
    tracker_path: str,
    phases: list[str],
    pipeline_type: str,
) -> dict[str, object]:
    """Selectively resets specific phases in an existing tracker for re-runs while preserving upstream phases.

    This is the only way to reset completed phases; prepare tools never reinitialize existing trackers. For each phase
    listed in ``phases``, all matching jobs are reset to SCHEDULED status. Downstream dependent phases are
    automatically included in the reset to maintain consistency (e.g., resetting 'binarization' also resets
    'processing' and 'combination'). Jobs belonging to phases not in the expanded reset set retain their original
    status.

    Important:
        After resetting phases, modify the pipeline configuration file if needed (e.g., change ROI detection
        parameters) before calling execute_processing_jobs_tool. The pipeline reads configuration from disk at
        execution time.

    Args:
        tracker_path: The absolute path to the ProcessingTracker YAML file.
        phases: List of phase names to reset. For single-recording: 'binarization', 'processing', 'combination'. For
            multi-recording: 'discovery', 'extraction'. Downstream phases are automatically included.
        pipeline_type: The pipeline type, either 'single-recording' or 'multi-recording'.

    Returns:
        On success, contains a 'reset' flag, the 'requested_phases' as provided, the 'effective_phases' after
        dependency expansion, and per-job status showing updated states. On failure, contains an 'error' describing
        the issue.
    """
    path = Path(tracker_path)
    if not path.exists():
        return {"success": False, "error": f"Unable to reset phases. Tracker file not found: {tracker_path}."}

    if pipeline_type not in ("single-recording", "multi-recording"):
        return {
            "success": False,
            "error": (
                f"Unable to reset phases. Invalid pipeline_type '{pipeline_type}'. "
                f"Must be 'single-recording' or 'multi-recording'."
            ),
        }

    # Validates phase names against the pipeline type.
    if pipeline_type == "single-recording":
        valid_phases = {member.value for member in SingleRecordingJobNames}
    else:
        valid_phases = {member.value for member in MultiRecordingJobNames}

    invalid_phases = [phase for phase in phases if phase not in valid_phases]
    if invalid_phases:
        return {
            "success": False,
            "error": (
                f"Unable to reset phases. Invalid phase names {invalid_phases} for {pipeline_type}. "
                f"Valid phases: {sorted(valid_phases)}."
            ),
        }

    # Expands the requested phases to include all downstream dependents. Resetting an upstream phase invalidates
    # all phases that depend on its output, so they must be reset too.
    requested_phases = list(phases)
    if pipeline_type == "single-recording":
        # Dependency chain: binarization → processing → combination.
        expanded = set(phases)
        if SingleRecordingJobNames.BINARIZE in expanded:
            expanded.add(SingleRecordingJobNames.PROCESS)
            expanded.add(SingleRecordingJobNames.COMBINE)
        if SingleRecordingJobNames.PROCESS in expanded:
            expanded.add(SingleRecordingJobNames.COMBINE)
        phases = sorted(expanded)
    else:
        # Dependency chain: discovery → extraction.
        expanded = set(phases)
        if MultiRecordingJobNames.DISCOVER in expanded:
            expanded.add(MultiRecordingJobNames.EXTRACT)
        phases = sorted(expanded)

    tracker = ProcessingTracker(file_path=path)

    # Snapshots current job states before reset.
    if pipeline_type == "single-recording":
        all_found_jobs = {
            **tracker.find_jobs(job_name=SingleRecordingJobNames.BINARIZE),
            **tracker.find_jobs(job_name=SingleRecordingJobNames.PROCESS),
            **tracker.find_jobs(job_name=SingleRecordingJobNames.COMBINE),
        }
    else:
        all_found_jobs = {
            **tracker.find_jobs(job_name=MultiRecordingJobNames.DISCOVER),
            **tracker.find_jobs(job_name=MultiRecordingJobNames.EXTRACT),
        }

    # Captures original states for all jobs.
    original_states: dict[str, tuple[str, str, ProcessingStatus, str | None]] = {}
    for job_id, (job_name, specifier) in all_found_jobs.items():
        job_info = tracker.get_job_info(job_id=job_id)
        original_states[job_id] = (job_name, specifier, job_info.status, job_info.error_message)

    # Resets tracker and reinitializes all jobs as SCHEDULED.
    all_jobs_list: list[tuple[str, str]] = list(all_found_jobs.values())
    tracker.reset()
    tracker.initialize_jobs(jobs=all_jobs_list)

    # Replays original status for phases NOT being reset (preserved phases).
    phases_set = set(phases)
    for job_id, (job_name, _specifier, original_status, error_message) in original_states.items():
        if job_name in phases_set:
            continue

        if original_status == ProcessingStatus.SUCCEEDED:
            tracker.start_job(job_id=job_id)
            tracker.complete_job(job_id=job_id)
        elif original_status == ProcessingStatus.FAILED:
            tracker.start_job(job_id=job_id)
            tracker.fail_job(job_id=job_id, error_message=error_message)

    # Builds the response with updated per-job statuses.
    updated_jobs: list[dict[str, object]] = []
    for job_id, (job_name, specifier) in all_found_jobs.items():
        updated_jobs.append(
            {
                "job_id": job_id,
                "name": job_name,
                "specifier": specifier,
                "status": tracker.get_job_status(job_id=job_id).name.lower(),
            }
        )

    return {
        "success": True,
        "reset": True,
        "tracker_path": tracker_path,
        "requested_phases": requested_phases,
        "effective_phases": phases,
        "jobs": updated_jobs,
    }


@mcp.tool()
def clean_processing_output_tool(
    recording_path: str,
    phases: list[str],
    pipeline_type: str,
    dataset: str = "",
) -> dict[str, object]:
    """Deletes output files and directories for specific pipeline phases while preserving configuration and tracker
    state.

    Removes all files generated by the specified phases. Downstream phases are automatically included in the cleanup
    to maintain consistency (e.g., cleaning 'binarization' also cleans 'processing' and 'combination'). Tracker
    files, configuration files, runtime_data.yaml, and acquisition_parameters.yaml are never deleted. Use this to
    reclaim disk space or force a full rerun from specific phases.

    Args:
        recording_path: The absolute path to the recording data directory.
        phases: List of phase names to clean. For single-recording: 'binarization', 'processing', 'combination'. For
            multi-recording: 'discovery', 'extraction'. Downstream phases are automatically included.
        pipeline_type: The pipeline type, either 'single-recording' or 'multi-recording'.
        dataset: The multi-recording dataset name. Required when pipeline_type is 'multi-recording'.

    Returns:
        On success, contains 'deleted_files', 'deleted_dirs', 'total_deleted', the 'requested_phases' and
        'effective_phases' after dependency expansion. On failure, contains an 'error' describing the issue.
        Both cases include a 'success' flag.
    """
    recording = Path(recording_path)

    if not recording.exists():
        return {
            "success": False,
            "error": f"Unable to clean processing output. Recording directory not found: {recording_path}.",
        }

    if pipeline_type not in ("single-recording", "multi-recording"):
        return {
            "success": False,
            "error": (
                f"Unable to clean processing output. Invalid pipeline_type '{pipeline_type}'. "
                f"Must be 'single-recording' or 'multi-recording'."
            ),
        }

    # Validates phase names against the pipeline type.
    if pipeline_type == "single-recording":
        valid_phases = {member.value for member in SingleRecordingJobNames}
    else:
        valid_phases = {member.value for member in MultiRecordingJobNames}

    invalid_phases = [phase for phase in phases if phase not in valid_phases]
    if invalid_phases:
        return {
            "success": False,
            "error": (
                f"Unable to clean processing output. Invalid phase names {invalid_phases} for {pipeline_type}. "
                f"Valid phases: {sorted(valid_phases)}."
            ),
        }

    # Expands the requested phases to include all downstream dependents.
    requested_phases = list(phases)
    if pipeline_type == "single-recording":
        expanded = set(phases)
        if SingleRecordingJobNames.BINARIZE in expanded:
            expanded.add(SingleRecordingJobNames.PROCESS)
            expanded.add(SingleRecordingJobNames.COMBINE)
        if SingleRecordingJobNames.PROCESS in expanded:
            expanded.add(SingleRecordingJobNames.COMBINE)
        effective_phases = sorted(expanded)
    else:
        expanded = set(phases)
        if MultiRecordingJobNames.DISCOVER in expanded:
            expanded.add(MultiRecordingJobNames.EXTRACT)
        effective_phases = sorted(expanded)

    deleted_files: list[str] = []
    deleted_dirs: list[str] = []
    errors: list[str] = []

    if pipeline_type == "single-recording":
        cindra_root = recording / "cindra"
        if not cindra_root.exists():
            return {
                "success": False,
                "error": f"Unable to clean processing output. No cindra directory found at: {recording_path}.",
            }

        effective_set = set(effective_phases)

        # Cleans per-plane files.
        plane_dirs = sorted(d for d in cindra_root.iterdir() if d.is_dir() and d.name.startswith("plane_"))
        for plane_dir in plane_dirs:
            if SingleRecordingJobNames.BINARIZE in effective_set:
                for name in ("channel_1_data.bin", "channel_2_data.bin"):
                    _delete_file(path=plane_dir / name, deleted=deleted_files, errors=errors)

            if SingleRecordingJobNames.PROCESS in effective_set:
                _delete_directory(path=plane_dir / "registration_data", deleted=deleted_dirs, errors=errors)
                _delete_directory(path=plane_dir / "detection_data", deleted=deleted_dirs, errors=errors)
                for name in (
                    "roi_masks.npz",
                    "roi_statistics.npz",
                    "cell_fluorescence.npy",
                    "neuropil_fluorescence.npy",
                    "subtracted_fluorescence.npy",
                    "spikes.npy",
                    "cell_classification.npy",
                    "cell_fluorescence_channel_2.npy",
                    "neuropil_fluorescence_channel_2.npy",
                    "subtracted_fluorescence_channel_2.npy",
                    "spikes_channel_2.npy",
                    "cell_classification_channel_2.npy",
                    "cell_colocalization.npy",
                ):
                    _delete_file(path=plane_dir / name, deleted=deleted_files, errors=errors)

        # Cleans combined files.
        if SingleRecordingJobNames.COMBINE in effective_set:
            _delete_directory(path=cindra_root / "detection_data", deleted=deleted_dirs, errors=errors)
            for name in (
                "combined_metadata.npz",
                "roi_masks.npz",
                "roi_statistics.npz",
                "cell_fluorescence.npy",
                "neuropil_fluorescence.npy",
                "subtracted_fluorescence.npy",
                "spikes.npy",
                "cell_classification.npy",
                "cell_fluorescence_channel_2.npy",
                "neuropil_fluorescence_channel_2.npy",
                "subtracted_fluorescence_channel_2.npy",
                "spikes_channel_2.npy",
                "cell_classification_channel_2.npy",
                "cell_colocalization.npy",
            ):
                _delete_file(path=cindra_root / name, deleted=deleted_files, errors=errors)

    else:
        # Multi-recording cleanup requires the dataset parameter.
        if not dataset:
            return {
                "success": False,
                "error": "Unable to clean processing output. The 'dataset' parameter is required for multi-recording.",
            }

        cindra_root = recording / "cindra"
        dataset_path = cindra_root / "multi_recording" / dataset
        if not dataset_path.exists():
            return {
                "success": False,
                "error": f"Unable to clean processing output. Dataset directory not found: {dataset_path}.",
            }

        # Loads runtime data to discover all recording output paths.
        runtime = _load_runtime_yaml(path=dataset_path / "multi_recording_runtime_data.yaml")
        if runtime is None:
            return {
                "success": False,
                "error": f"Unable to load runtime data from: {dataset_path}.",
            }

        dataset_output_paths = runtime.get("io", {}).get("dataset_output_paths", [str(dataset_path)])
        effective_set = set(effective_phases)

        for output_path_str in dataset_output_paths:
            output_path = Path(output_path_str)
            if not output_path.exists():
                continue

            if MultiRecordingJobNames.DISCOVER in effective_set:
                _delete_directory(path=output_path / "registration_arrays", deleted=deleted_dirs, errors=errors)
                for name in (
                    "registration_deformed_masks.npz",
                    "registration_deformed_masks_channel_2.npz",
                    "tracking_template_masks.npz",
                    "tracking_template_masks_channel_2.npz",
                ):
                    _delete_file(path=output_path / name, deleted=deleted_files, errors=errors)

            if MultiRecordingJobNames.EXTRACT in effective_set:
                for name in (
                    "roi_masks.npz",
                    "roi_statistics.npz",
                    "cell_fluorescence.npy",
                    "neuropil_fluorescence.npy",
                    "subtracted_fluorescence.npy",
                    "spikes.npy",
                    "cell_fluorescence_channel_2.npy",
                    "neuropil_fluorescence_channel_2.npy",
                    "subtracted_fluorescence_channel_2.npy",
                    "spikes_channel_2.npy",
                    "cell_colocalization.npy",
                ):
                    _delete_file(path=output_path / name, deleted=deleted_files, errors=errors)

    result: dict[str, object] = {
        "success": True,
        "cleaned": True,
        "recording_path": recording_path,
        "requested_phases": requested_phases,
        "effective_phases": effective_phases,
        "deleted_files": deleted_files,
        "deleted_dirs": deleted_dirs,
        "total_deleted": len(deleted_files) + len(deleted_dirs),
    }

    if errors:
        result["errors"] = errors

    return result


@mcp.tool()
def execute_processing_jobs_tool(
    jobs: list[dict[str, str]],
    *,
    workers_per_job: int = -1,
    max_parallel_jobs: int = -1,
) -> dict[str, object]:
    """Dispatches pipeline jobs for background execution with prerequisite validation and resource allocation.

    Validates that each job's prerequisites are satisfied (e.g., BINARIZE must be SUCCEEDED before PROCESS can run),
    resolves worker and parallelism counts via saturating allocation, rewrites configuration files for compute-bound
    jobs with the resolved worker count, then starts a background execution manager. I/O-bound jobs (binarize, combine)
    ignore both parameters and always run with a fixed concurrency of 4. Use get_processing_jobs_status_tool to monitor
    progress and cancel_processing_jobs_tool to stop execution.

    Important:
        Only one execution session can be active at a time. Wait for the current session to complete or cancel it before
        starting a new one. The agent is responsible for submitting jobs in the correct phase order; this tool validates
        prerequisites but does not reorder jobs.

    Args:
        jobs: List of job descriptors, each a dictionary with 'configuration_path' (absolute path to the pipeline
            configuration file), 'tracker_path' (absolute path to the ProcessingTracker file), 'job_id' (the
            hexadecimal job identifier from the prepare manifest), and 'pipeline_type' ('single-recording' or
            'multi-recording').
        workers_per_job: CPU cores per compute-bound job. Set to -1 for automatic resolution via saturating allocation
            (prefers ~30 cores per job, minimum 10, rounded to multiples of 5). Ignored for I/O-bound jobs.
        max_parallel_jobs: Maximum concurrent compute-bound jobs. Set to -1 for automatic resolution via saturating
            allocation. Ignored for I/O-bound jobs, which always use a fixed concurrency of 4.

    Returns:
        Contains a 'started' flag, 'total_jobs' dispatched (split into 'compute_jobs' and 'io_jobs'), resolved
        'workers_per_job' and 'max_parallel_jobs' for compute-bound work, and 'invalid_jobs' listing any jobs that
        failed validation with reasons.
    """
    if not jobs:
        return {"success": False, "error": "Unable to execute jobs. At least one job descriptor is required."}

    # Checks if an execution session is already active.
    if _job_execution_state is not None:
        with _job_execution_state.lock:
            has_pending = _job_execution_state.io_pending_queue or _job_execution_state.compute_pending_queue
            has_active = _job_execution_state.io_active_threads or _job_execution_state.compute_active_threads
            if has_pending or has_active:
                return {
                    "success": False,
                    "error": "Unable to execute jobs. An execution session is already active.",
                    "pending_count": (
                        len(_job_execution_state.io_pending_queue) + len(_job_execution_state.compute_pending_queue)
                    ),
                    "active_count": (
                        len(_job_execution_state.io_active_threads) + len(_job_execution_state.compute_active_threads)
                    ),
                }

    # Validates each job entry and categorizes into IO-bound and compute-bound.
    required_keys = {"configuration_path", "tracker_path", "job_id", "pipeline_type"}
    io_jobs: list[_PendingJob] = []
    compute_jobs: list[_PendingJob] = []
    all_jobs_map: dict[tuple[str, str], _PendingJob] = {}
    invalid_jobs: list[dict[str, str]] = []

    for job_entry in jobs:
        # Validates required keys.
        missing_keys = required_keys - set(job_entry)
        if missing_keys:
            invalid_jobs.append({"job": str(job_entry), "reason": f"Missing required keys: {missing_keys}"})
            continue

        configuration_file = Path(job_entry["configuration_path"])
        tracker_file = Path(job_entry["tracker_path"])
        job_id = job_entry["job_id"]
        pipeline_type = job_entry["pipeline_type"]

        if not configuration_file.exists():
            invalid_jobs.append({"job_id": job_id, "reason": f"Configuration file not found: {configuration_file}"})
            continue

        if not tracker_file.exists():
            invalid_jobs.append({"job_id": job_id, "reason": f"Tracker file not found: {tracker_file}"})
            continue

        if pipeline_type not in ("single-recording", "multi-recording"):
            invalid_jobs.append({"job_id": job_id, "reason": f"Invalid pipeline_type: {pipeline_type}"})
            continue

        single_recording = pipeline_type == "single-recording"

        # Validates that the job_id exists in the tracker and reads its job name for categorization.
        tracker = ProcessingTracker(file_path=tracker_file)
        try:
            job_info = tracker.get_job_info(job_id=job_id)
        except Exception:
            invalid_jobs.append({"job_id": job_id, "reason": f"Job ID not found in tracker: {tracker_file}"})
            continue

        # Validates prerequisites using the tracker as the authoritative source.
        prerequisite_error = _validate_job_prerequisites(
            tracker=tracker, job_id=job_id, single_recording=single_recording
        )
        if prerequisite_error is not None:
            invalid_jobs.append({"job_id": job_id, "reason": prerequisite_error})
            continue

        io_bound = job_info.job_name in _IO_BOUND_JOB_NAMES
        pending_job = _PendingJob(
            configuration_path=configuration_file,
            tracker_path=tracker_file,
            job_id=job_id,
            single_recording=single_recording,
            io_bound=io_bound,
        )

        if io_bound:
            io_jobs.append(pending_job)
        else:
            compute_jobs.append(pending_job)

        all_jobs_map[pending_job.dispatch_key] = pending_job

    if not all_jobs_map:
        return {
            "success": False,
            "error": "Unable to execute jobs. No valid jobs after validation.",
            "invalid_jobs": invalid_jobs,
        }

    return _start_execution_session(
        all_jobs=all_jobs_map,
        io_jobs=io_jobs,
        compute_jobs=compute_jobs,
        phase_groups=[],
        workers_per_job=workers_per_job,
        max_parallel_jobs=max_parallel_jobs,
        extra_result_fields={"invalid_jobs": invalid_jobs} if invalid_jobs else {},
    )


@mcp.tool()
def get_processing_jobs_status_tool() -> dict[str, object]:
    """Returns the current status of the active job execution session.

    Reads ProcessingTracker files from disk for each job in the execution session to report per-job progress. All
    status information is derived from the on-disk tracker files rather than in-memory state.

    Returns:
        Contains an 'active' flag, per-job status entries in 'jobs', and a 'summary' with counts for pending, running,
        succeeded, and failed jobs. Returns inactive state when no execution session exists.
    """
    if _job_execution_state is None:
        return {
            "active": False,
            "jobs": [],
            "summary": {"pending": 0, "running": 0, "succeeded": 0, "failed": 0},
        }

    with _job_execution_state.lock:
        io_pending = len(_job_execution_state.io_pending_queue)
        compute_pending = len(_job_execution_state.compute_pending_queue)
        io_active_ids = list(_job_execution_state.io_active_threads.keys())
        compute_active_ids = list(_job_execution_state.compute_active_threads.keys())

    # Reads per-job status from tracker files (outside lock to avoid holding it during I/O).
    jobs_status: list[dict[str, object]] = []
    summary_counts: dict[str, int] = {"pending": 0, "running": 0, "succeeded": 0, "failed": 0}

    for pending_job in _job_execution_state.all_jobs.values():
        tracker = ProcessingTracker(file_path=pending_job.tracker_path)
        status = tracker.get_job_status(job_id=pending_job.job_id)
        job_info = tracker.get_job_info(job_id=pending_job.job_id)
        status_name = status.name.lower()

        # Counts by status category.
        if status == ProcessingStatus.SCHEDULED:
            summary_counts["pending"] += 1
        elif status == ProcessingStatus.RUNNING:
            summary_counts["running"] += 1
        elif status == ProcessingStatus.SUCCEEDED:
            summary_counts["succeeded"] += 1
        elif status == ProcessingStatus.FAILED:
            summary_counts["failed"] += 1

        job_entry: dict[str, object] = {
            "job_id": pending_job.job_id,
            "name": job_info.job_name,
            "specifier": job_info.specifier,
            "status": status_name,
            "io_bound": pending_job.io_bound,
            "pipeline_type": "single-recording" if pending_job.single_recording else "multi-recording",
            "tracker_path": str(pending_job.tracker_path),
        }

        if job_info.error_message:
            job_entry["error"] = job_info.error_message

        jobs_status.append(job_entry)

    # Determines if the manager thread is still alive.
    manager_alive = _job_execution_state.manager_thread is not None and _job_execution_state.manager_thread.is_alive()

    return {
        "active": manager_alive,
        "pending_io": io_pending,
        "pending_compute": compute_pending,
        "active_io": io_active_ids,
        "active_compute": compute_active_ids,
        "jobs": jobs_status,
        "summary": summary_counts,
    }


@mcp.tool()
def get_active_execution_timing_tool() -> dict[str, object]:
    """Returns timing information for all jobs in the active execution session.

    Reports elapsed time for running jobs and duration for completed jobs using microsecond-precision UTC timestamps
    from ProcessingTracker. Also computes session-level statistics including total elapsed time and throughput. Use
    this alongside get_processing_jobs_status_tool for time-aware progress monitoring.

    Returns:
        Contains an 'active' flag, per-job timing in 'jobs', and a 'session' summary with total_elapsed_seconds,
        completed, failed, running, and pending counts, and throughput_jobs_per_hour when applicable.
    """
    if _job_execution_state is None:
        return {
            "active": False,
            "jobs": [],
            "session": {
                "total_elapsed_seconds": 0.0,
                "completed_count": 0,
                "failed_count": 0,
                "running_count": 0,
                "pending_count": 0,
            },
        }

    current_us = int(get_timestamp(output_format=TimestampFormats.INTEGER, precision=TimestampPrecisions.MICROSECOND))

    jobs_timing: list[dict[str, object]] = []
    earliest_start: int | None = None
    completed_count = 0
    failed_count = 0
    running_count = 0
    pending_count = 0

    for pending_job in _job_execution_state.all_jobs.values():
        tracker = ProcessingTracker(file_path=pending_job.tracker_path)
        job_info = tracker.get_job_info(job_id=pending_job.job_id)

        entry: dict[str, object] = {
            "job_id": pending_job.job_id,
            "name": job_info.job_name,
            "specifier": job_info.specifier,
            "status": job_info.status.name.lower(),
        }

        if job_info.started_at is not None:
            started_at_us = int(job_info.started_at)
            entry["started_at"] = started_at_us
            if earliest_start is None or started_at_us < earliest_start:
                earliest_start = started_at_us

        if job_info.completed_at is not None:
            entry["completed_at"] = job_info.completed_at

        if job_info.status == ProcessingStatus.RUNNING and job_info.started_at is not None:
            entry["elapsed_seconds"] = round((current_us - int(job_info.started_at)) / 1_000_000, 2)
            running_count += 1
        elif job_info.status == ProcessingStatus.SUCCEEDED:
            if job_info.started_at is not None and job_info.completed_at is not None:
                entry["duration_seconds"] = round(
                    (int(job_info.completed_at) - int(job_info.started_at)) / 1_000_000, 2
                )
            completed_count += 1
        elif job_info.status == ProcessingStatus.FAILED:
            if job_info.started_at is not None and job_info.completed_at is not None:
                entry["duration_seconds"] = round(
                    (int(job_info.completed_at) - int(job_info.started_at)) / 1_000_000, 2
                )
            failed_count += 1
        else:
            pending_count += 1

        jobs_timing.append(entry)

    # Computes session-level timing.
    total_elapsed = round((current_us - earliest_start) / 1_000_000, 2) if earliest_start is not None else 0.0

    session: dict[str, object] = {
        "total_elapsed_seconds": total_elapsed,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "running_count": running_count,
        "pending_count": pending_count,
    }

    if total_elapsed > 0 and completed_count > 0:
        session["throughput_jobs_per_hour"] = round(completed_count / (total_elapsed / 3600), 2)

    manager_alive = _job_execution_state.manager_thread is not None and _job_execution_state.manager_thread.is_alive()

    return {
        "active": manager_alive,
        "jobs": jobs_timing,
        "session": session,
    }


@mcp.tool()
def cancel_processing_jobs_tool() -> dict[str, object]:
    """Cancels the active job execution session.

    Clears both pending job queues to prevent new jobs from starting and resets the execution state. Active jobs will
    complete naturally but no new jobs will be dispatched.

    Returns:
        Contains a 'canceled' flag, a 'message' describing the outcome, and a 'final_state' with counts for
        succeeded_jobs, failed_jobs, and active_jobs_at_cancel.
    """
    global _job_execution_state

    if _job_execution_state is None:
        return {"canceled": False, "message": "No execution session is active."}

    with _job_execution_state.lock:
        active_count = len(_job_execution_state.io_active_threads) + len(_job_execution_state.compute_active_threads)

        # Clears both pending queues to prevent new jobs from starting.
        _job_execution_state.io_pending_queue.clear()
        _job_execution_state.compute_pending_queue.clear()

        # Reads final state from trackers.
        total_succeeded = 0
        total_failed = 0
        seen_trackers: set[Path] = set()
        for pending_job in _job_execution_state.all_jobs.values():
            if pending_job.tracker_path in seen_trackers:
                continue
            seen_trackers.add(pending_job.tracker_path)
            tracker = ProcessingTracker(file_path=pending_job.tracker_path)
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

    # Resets execution state after releasing lock.
    _job_execution_state = None

    return {
        "canceled": True,
        "message": "Execution session canceled. Active jobs will complete but no new jobs will start.",
        "final_state": final_state,
    }


@mcp.tool()
def execute_full_pipeline_tool(
    pipeline_type: str,
    *,
    recording_paths: list[str] | None = None,
    configuration_path: str | None = None,
    recording_output_paths: list[str] | None = None,
    dataset_configurations: list[dict[str, object]] | None = None,
    workers_per_job: int = -1,
    max_parallel_jobs: int = -1,
) -> dict[str, object]:
    """Executes a complete pipeline from preparation through all phases with automatic phase sequencing.

    Prepares, validates, and dispatches all jobs for a full pipeline run with automatic phase advancement. Jobs are
    grouped by phase and executed sequentially: each phase must complete successfully before the next phase begins.
    If any job in a phase fails, subsequent phases are marked as failed and execution stops.

    For single-recording pipelines, the three phases are: binarize, process, combine. For multi-recording pipelines,
    the two phases are: discover, extract.

    Args:
        pipeline_type: The pipeline type, either 'single-recording' or 'multi-recording'.
        recording_paths: List of absolute paths to recording directories. Required for single-recording pipelines.
        configuration_path: Absolute path to the template configuration file. Required for single-recording pipelines.
        recording_output_paths: List of per-recording output paths for single-recording pipelines. Required for
            single-recording pipelines and must match the length of recording_paths.
        dataset_configurations: List of dataset configuration dictionaries. Required for multi-recording pipelines.
            Each must contain 'configuration_path', 'recording_paths', and 'dataset_name'.
        workers_per_job: CPU cores per compute-bound job. Set to -1 for automatic resolution via saturating
            allocation.
        max_parallel_jobs: Maximum concurrent compute-bound jobs. Set to -1 for automatic resolution.

    Returns:
        On success, contains 'total_jobs', 'phase_count', per-phase 'phases' with job counts and IDs, and resolved
        resource allocation. On failure, contains an 'error' describing the issue. Both cases include a 'success'
        flag.
    """
    if pipeline_type not in ("single-recording", "multi-recording"):
        return {
            "success": False,
            "error": (
                f"Unable to execute full pipeline. Invalid pipeline_type '{pipeline_type}'. "
                f"Must be 'single-recording' or 'multi-recording'."
            ),
        }

    # Checks for an active execution session.
    if _job_execution_state is not None:
        with _job_execution_state.lock:
            has_pending = _job_execution_state.io_pending_queue or _job_execution_state.compute_pending_queue
            has_active = _job_execution_state.io_active_threads or _job_execution_state.compute_active_threads
            if has_pending or has_active:
                return {
                    "success": False,
                    "error": "Unable to execute full pipeline. An execution session is already active.",
                }

    # Calls the appropriate prepare tool to build the execution manifest.
    manifest: dict[str, object]
    if pipeline_type == "single-recording":
        if not recording_paths:
            return {
                "success": False,
                "error": "Unable to execute full pipeline. 'recording_paths' is required for single-recording.",
            }
        if not configuration_path:
            return {
                "success": False,
                "error": "Unable to execute full pipeline. 'configuration_path' is required for single-recording.",
            }
        if not recording_output_paths:
            return {
                "success": False,
                "error": (
                    "Unable to execute full pipeline. 'recording_output_paths' is required for single-recording."
                ),
            }

        manifest = prepare_single_recording_batch_tool(
            recording_paths=recording_paths,
            configuration_path=configuration_path,
            recording_output_paths=recording_output_paths,
        )
    else:
        if not dataset_configurations:
            return {
                "success": False,
                "error": "Unable to execute full pipeline. 'dataset_configurations' is required for multi-recording.",
            }

        manifest = prepare_multi_recording_batch_tool(dataset_configurations=dataset_configurations)

    if not manifest.get("success"):
        return manifest

    # Parses manifest into phased job groups.
    phase_groups: list[tuple[str, list[_PendingJob]]] = []

    if pipeline_type == "single-recording":
        binarize_phase_jobs: list[_PendingJob] = []
        process_phase_jobs: list[_PendingJob] = []
        combine_phase_jobs: list[_PendingJob] = []

        raw_recordings = manifest.get("recordings", {})
        if isinstance(raw_recordings, dict):
            for recording_manifest in raw_recordings.values():
                manifest_dict: dict[str, Any] = recording_manifest
                config_path = Path(str(manifest_dict["configuration_path"]))
                tracker_path = Path(str(manifest_dict["tracker_path"]))

                binarize = manifest_dict.get("binarize_job", {})
                if binarize and binarize.get("status") != "succeeded":
                    binarize_phase_jobs.append(
                        _PendingJob(
                            configuration_path=config_path,
                            tracker_path=tracker_path,
                            job_id=binarize["job_id"],
                            single_recording=True,
                            io_bound=True,
                        )
                    )

                process_phase_jobs.extend(
                    _PendingJob(
                        configuration_path=config_path,
                        tracker_path=tracker_path,
                        job_id=process["job_id"],
                        single_recording=True,
                        io_bound=False,
                    )
                    for process in manifest_dict.get("process_jobs", [])
                    if process.get("status") != "succeeded"
                )

                combine = manifest_dict.get("combine_job", {})
                if combine and combine.get("status") != "succeeded":
                    combine_phase_jobs.append(
                        _PendingJob(
                            configuration_path=config_path,
                            tracker_path=tracker_path,
                            job_id=combine["job_id"],
                            single_recording=True,
                            io_bound=True,
                        )
                    )

        if binarize_phase_jobs:
            phase_groups.append(("binarization", binarize_phase_jobs))
        if process_phase_jobs:
            phase_groups.append(("processing", process_phase_jobs))
        if combine_phase_jobs:
            phase_groups.append(("combination", combine_phase_jobs))

    else:
        discover_phase_jobs: list[_PendingJob] = []
        extract_phase_jobs: list[_PendingJob] = []

        raw_datasets = manifest.get("datasets", {})
        if isinstance(raw_datasets, dict):
            for dataset_manifest in raw_datasets.values():
                manifest_dict = dataset_manifest
                config_path = Path(str(manifest_dict["configuration_path"]))
                tracker_path = Path(str(manifest_dict["tracker_path"]))

                discover = manifest_dict.get("discover_job", {})
                if discover and discover.get("status") != "succeeded":
                    discover_phase_jobs.append(
                        _PendingJob(
                            configuration_path=config_path,
                            tracker_path=tracker_path,
                            job_id=discover["job_id"],
                            single_recording=False,
                            io_bound=False,
                        )
                    )

                extract_phase_jobs.extend(
                    _PendingJob(
                        configuration_path=config_path,
                        tracker_path=tracker_path,
                        job_id=extract["job_id"],
                        single_recording=False,
                        io_bound=False,
                    )
                    for extract in manifest_dict.get("extract_jobs", [])
                    if extract.get("status") != "succeeded"
                )

        if discover_phase_jobs:
            phase_groups.append(("discovery", discover_phase_jobs))
        if extract_phase_jobs:
            phase_groups.append(("extraction", extract_phase_jobs))

    if not phase_groups:
        return {
            "success": True,
            "started": False,
            "message": "All pipeline phases are already completed.",
            "pipeline_type": pipeline_type,
            "total_jobs": 0,
            "phase_count": 0,
            "phases": [],
        }

    # Collects all jobs across all phases for the execution state.
    all_jobs_map: dict[tuple[str, str], _PendingJob] = {}
    for _phase_name, phase_jobs in phase_groups:
        for job in phase_jobs:
            all_jobs_map[job.dispatch_key] = job

    # Splits first phase into initial queues and remaining phases into groups for sequential advancement.
    _first_phase_name, first_phase_jobs = phase_groups[0]
    remaining_groups = [jobs for _, jobs in phase_groups[1:]]

    first_phase_io: list[_PendingJob] = []
    first_phase_compute: list[_PendingJob] = []
    for job in first_phase_jobs:
        if job.io_bound:
            first_phase_io.append(job)
        else:
            first_phase_compute.append(job)

    # Builds phase summary for response before delegating to shared execution setup.
    phases_summary: list[dict[str, object]] = []
    for phase_name, phase_job_list in phase_groups:
        phases_summary.append(
            {
                "phase_name": phase_name,
                "job_count": len(phase_job_list),
                "job_ids": [job.job_id for job in phase_job_list],
            }
        )

    # Delegates to the shared execution session setup.
    extra_fields: dict[str, object] = {
        "pipeline_type": pipeline_type,
        "phase_count": len(phase_groups),
        "phases": phases_summary,
    }
    if "invalid_paths" in manifest:
        extra_fields["invalid_paths"] = manifest["invalid_paths"]
    if "invalid_configurations" in manifest:
        extra_fields["invalid_configurations"] = manifest["invalid_configurations"]

    return _start_execution_session(
        all_jobs=all_jobs_map,
        io_jobs=first_phase_io,
        compute_jobs=first_phase_compute,
        phase_groups=remaining_groups,
        workers_per_job=workers_per_job,
        max_parallel_jobs=max_parallel_jobs,
        extra_result_fields=extra_fields,
    )


def _start_execution_session(
    all_jobs: dict[tuple[str, str], _PendingJob],
    io_jobs: list[_PendingJob],
    compute_jobs: list[_PendingJob],
    phase_groups: list[list[_PendingJob]],
    workers_per_job: int,
    max_parallel_jobs: int,
    extra_result_fields: dict[str, object],
) -> dict[str, object]:
    """Resolves resource allocation, rewrites configuration files, and starts the background execution manager.

    Centralizes the execution setup logic shared by ``execute_processing_jobs_tool`` (flat dispatch) and
    ``execute_full_pipeline_tool`` (phased dispatch). The caller is responsible for validating jobs and checking
    for active sessions before calling this function.

    Args:
        all_jobs: All submitted jobs keyed by dispatch key, used for status reporting.
        io_jobs: I/O-bound jobs to place in the initial pending queue.
        compute_jobs: Compute-bound jobs to place in the initial pending queue.
        phase_groups: Remaining phase groups for sequential advancement (empty for flat dispatch).
        workers_per_job: Requested CPU cores per compute-bound job (-1 for automatic).
        max_parallel_jobs: Requested maximum concurrent compute-bound jobs (-1 for automatic).
        extra_result_fields: Additional key-value pairs to include in the result dictionary.

    Returns:
        A result dictionary containing 'success', 'started', resource allocation details, and any extra fields.
    """
    global _job_execution_state

    # Resolves resource allocation for compute-bound jobs using saturating allocation.
    budget = resolve_worker_count(requested_workers=-1, reserved_cores=_RESERVED_CORES)
    compute_job_count = max(1, len(compute_jobs))

    # Accounts for compute jobs across all phase groups when computing total count.
    total_compute = compute_job_count
    for group in phase_groups:
        total_compute += sum(1 for job in group if not job.io_bound)
    total_compute = max(1, total_compute)

    if workers_per_job <= 0 and max_parallel_jobs <= 0:
        actual_workers, actual_max_parallel = _resolve_saturating_allocation(budget=budget, total_jobs=total_compute)
    elif workers_per_job > 0 >= max_parallel_jobs:
        actual_workers = resolve_worker_count(requested_workers=workers_per_job, reserved_cores=_RESERVED_CORES)
        actual_max_parallel = resolve_parallel_job_capacity(workers_per_job=actual_workers)
    elif workers_per_job <= 0 < max_parallel_jobs:
        raw_workers = budget // max_parallel_jobs
        actual_workers = max(1, (raw_workers // _WORKER_MULTIPLE) * _WORKER_MULTIPLE)
        actual_max_parallel = max_parallel_jobs
    else:
        actual_workers = resolve_worker_count(requested_workers=workers_per_job, reserved_cores=_RESERVED_CORES)
        actual_max_parallel = max_parallel_jobs

    # Rewrites runtime.parallel_workers in configuration files for compute-bound jobs. Groups by config path to
    # avoid redundant rewrites when multiple jobs share the same configuration file.
    rewritten_configs: set[Path] = set()
    all_pending = list(all_jobs.values())
    for pending_job in all_pending:
        if pending_job.io_bound or pending_job.configuration_path in rewritten_configs:
            continue
        rewritten_configs.add(pending_job.configuration_path)

        if pending_job.single_recording:
            single_config = SingleRecordingConfiguration.from_yaml(file_path=pending_job.configuration_path)
            single_config.runtime.parallel_workers = actual_workers
            single_config.save(file_path=pending_job.configuration_path)
        else:
            multi_config = MultiRecordingConfiguration.from_yaml(file_path=pending_job.configuration_path)
            multi_config.runtime.parallel_workers = actual_workers
            multi_config.save(file_path=pending_job.configuration_path)

    # Creates the execution state and starts the manager thread.
    execution_state = _JobExecutionState(
        all_jobs=all_jobs,
        io_pending_queue=list(io_jobs),
        compute_pending_queue=list(compute_jobs),
        max_parallel_jobs=max(1, actual_max_parallel),
        lock=Lock(),
        phase_groups=phase_groups,
    )

    manager = Thread(target=_job_execution_manager, daemon=True)
    manager.start()
    execution_state.manager_thread = manager
    _job_execution_state = execution_state

    result: dict[str, object] = {
        "success": True,
        "started": True,
        "total_jobs": len(all_jobs),
        "compute_jobs": len(compute_jobs),
        "io_jobs": len(io_jobs),
        "workers_per_job": actual_workers,
        "max_parallel_jobs": actual_max_parallel,
        "max_parallel_io_jobs": _MAXIMUM_PARALLEL_IO_JOBS,
    }
    result.update(extra_result_fields)

    return result


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


def _validate_job_prerequisites(tracker: ProcessingTracker, job_id: str, *, single_recording: bool) -> str | None:
    """Validates that a job's prerequisites are satisfied based on tracker state.

    The tracker is the authoritative source for phase completion. Files on disk may be corrupt or incomplete even if
    they exist; the tracker only marks SUCCEEDED when processing is confirmed complete.

    Args:
        tracker: The ProcessingTracker instance for the job's recording or dataset.
        job_id: The unique hexadecimal job identifier to validate.
        single_recording: Determines whether to apply single-recording or multi-recording prerequisite rules.

    Returns:
        None if all prerequisites are satisfied, or an error message string describing the unmet prerequisite.
    """
    job_info = tracker.get_job_info(job_id=job_id)
    job_name = job_info.job_name

    if single_recording:
        if job_name == SingleRecordingJobNames.PROCESS:
            # PROCESS requires BINARIZE to be SUCCEEDED.
            binarize_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.BINARIZE)
            for binarize_id in binarize_jobs:
                if tracker.get_job_status(job_id=binarize_id) != ProcessingStatus.SUCCEEDED:
                    return (
                        f"Unable to execute PROCESS job {job_id}. "
                        f"Prerequisite BINARIZE job {binarize_id} has not succeeded."
                    )

        elif job_name == SingleRecordingJobNames.COMBINE:
            # COMBINE requires ALL PROCESS jobs to be SUCCEEDED.
            process_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.PROCESS)
            for process_id in process_jobs:
                if tracker.get_job_status(job_id=process_id) != ProcessingStatus.SUCCEEDED:
                    return (
                        f"Unable to execute COMBINE job {job_id}. "
                        f"Prerequisite PROCESS job {process_id} has not succeeded."
                    )
    elif job_name == MultiRecordingJobNames.EXTRACT:
        # EXTRACT requires DISCOVER to be SUCCEEDED.
        discover_jobs = tracker.find_jobs(job_name=MultiRecordingJobNames.DISCOVER)
        for discover_id in discover_jobs:
            if tracker.get_job_status(job_id=discover_id) != ProcessingStatus.SUCCEEDED:
                return (
                    f"Unable to execute EXTRACT job {job_id}. "
                    f"Prerequisite DISCOVER job {discover_id} has not succeeded."
                )

    return None


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


def _job_execution_manager() -> None:
    """Dispatches queued jobs from independent I/O-bound and compute-bound queues with separate concurrency limits.

    I/O-bound jobs (binarize, combine) run up to ``_MAXIMUM_PARALLEL_IO_JOBS`` concurrent. Compute-bound jobs (process,
    discover, extract) run up to ``max_parallel_jobs`` concurrent. When phase_groups are present, automatically advances
    to the next phase after the current phase drains. Runs as a daemon thread, polling at 1-second intervals. Exits
    when both queues are empty, no active threads remain, and no phase groups are pending.
    """
    global _job_execution_state

    timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

    while True:
        state = _job_execution_state
        if state is None:
            return

        with state.lock:
            # Cleans up completed I/O threads.
            completed_io = [key for key, thread in state.io_active_threads.items() if not thread.is_alive()]
            for key in completed_io:
                state.io_active_threads.pop(key, None)

            # Cleans up completed compute threads.
            completed_compute = [key for key, thread in state.compute_active_threads.items() if not thread.is_alive()]
            for key in completed_compute:
                state.compute_active_threads.pop(key, None)

            # Dispatches I/O-bound jobs up to the fixed concurrency limit.
            while len(state.io_active_threads) < _MAXIMUM_PARALLEL_IO_JOBS and state.io_pending_queue:
                pending_job = state.io_pending_queue.pop(0)
                thread = Thread(
                    target=_pipeline_worker,
                    kwargs={
                        "configuration_path": pending_job.configuration_path,
                        "job_id": pending_job.job_id,
                        "tracker_path": pending_job.tracker_path,
                        "single_recording": pending_job.single_recording,
                    },
                    daemon=True,
                )
                thread.start()
                state.io_active_threads[pending_job.dispatch_key] = thread

            # Dispatches compute-bound jobs up to the resolved parallelism limit.
            while len(state.compute_active_threads) < state.max_parallel_jobs and state.compute_pending_queue:
                pending_job = state.compute_pending_queue.pop(0)
                thread = Thread(
                    target=_pipeline_worker,
                    kwargs={
                        "configuration_path": pending_job.configuration_path,
                        "job_id": pending_job.job_id,
                        "tracker_path": pending_job.tracker_path,
                        "single_recording": pending_job.single_recording,
                    },
                    daemon=True,
                )
                thread.start()
                state.compute_active_threads[pending_job.dispatch_key] = thread

            # Phase advancement: when current phase drains, advances to the next phase group.
            all_empty = (
                not state.io_pending_queue
                and not state.compute_pending_queue
                and not state.io_active_threads
                and not state.compute_active_threads
            )
            if all_empty:
                if not state.phase_groups:
                    _job_execution_state = None
                    return

                # Checks if the preceding phase had any failures before advancing.
                if _check_current_phase_failures(state):
                    _fail_remaining_phase_groups(state)
                    _job_execution_state = None
                    return

                # Pops the next phase group and distributes jobs into pending queues.
                next_group = state.phase_groups.pop(0)
                for job in next_group:
                    if job.io_bound:
                        state.io_pending_queue.append(job)
                    else:
                        state.compute_pending_queue.append(job)

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


def _check_current_phase_failures(state: _JobExecutionState) -> bool:
    """Checks whether any jobs in the current (completed) phase have failed.

    Identifies which jobs belong to remaining phase groups and checks all other jobs for failures. This determines
    whether the manager should advance to the next phase or abort remaining phases.

    Args:
        state: The current job execution state.

    Returns:
        True if any job outside the remaining phase groups has a FAILED status, False otherwise.
    """
    # Collects dispatch keys that belong to remaining phase groups.
    remaining_keys: set[tuple[str, str]] = set()
    for group in state.phase_groups:
        for job in group:
            remaining_keys.add(job.dispatch_key)

    # Checks all jobs not in remaining groups for failures.
    for dispatch_key, pending_job in state.all_jobs.items():
        if dispatch_key in remaining_keys:
            continue
        tracker = ProcessingTracker(file_path=pending_job.tracker_path)
        if tracker.get_job_status(job_id=pending_job.job_id) == ProcessingStatus.FAILED:
            return True

    return False


def _fail_remaining_phase_groups(state: _JobExecutionState) -> None:
    """Marks all jobs in remaining phase groups as failed due to a preceding phase failure.

    Iterates through all remaining phase groups, starts and immediately fails each job with a dependency-failure
    message so that the tracker records the failure reason.

    Args:
        state: The current job execution state with phase_groups to fail.
    """
    for group in state.phase_groups:
        for job in group:
            tracker = ProcessingTracker(file_path=job.tracker_path)
            tracker.start_job(job_id=job.job_id)
            tracker.fail_job(
                job_id=job.job_id,
                error_message="Unable to execute job. A preceding pipeline phase failed.",
            )

    state.phase_groups.clear()


def _delete_file(path: Path, deleted: list[str], errors: list[str]) -> None:
    """Deletes a single file and records the result.

    Args:
        path: The filesystem path to the file to delete.
        deleted: The list to append the deleted file path to on success.
        errors: The list to append error messages to on failure.
    """
    if not path.exists():
        return
    try:
        path.unlink()
        deleted.append(str(path))
    except Exception as error:
        errors.append(f"Unable to delete file {path}: {error}")


def _delete_directory(path: Path, deleted: list[str], errors: list[str]) -> None:
    """Recursively deletes a directory and records the result.

    Args:
        path: The filesystem path to the directory to delete.
        deleted: The list to append the deleted directory path to on success.
        errors: The list to append error messages to on failure.
    """
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
        deleted.append(str(path))
    except Exception as error:
        errors.append(f"Unable to delete directory {path}: {error}")


def _load_runtime_yaml(path: Path) -> dict[str, Any] | None:
    """Loads a runtime YAML file and returns the parsed dictionary, or None if loading fails.

    Args:
        path: The filesystem path to the YAML file to load.

    Returns:
        The parsed YAML dictionary, or None if the file does not exist or loading fails.
    """
    if not path.exists():
        return None
    try:
        with path.open() as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception:
        return None
