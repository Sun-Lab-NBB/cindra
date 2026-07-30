"""Provides MCP tools for preparing, executing, monitoring, and cancelling neural imaging pipeline jobs.

These tools give agents fine-grained control over pipeline execution: prepare builds an execution manifest without
running anything, execute dispatches selected jobs with prerequisite validation, reset selectively reverts completed
phases for re-runs, and status/cancel manage the active execution. Both single-recording (four-phase: binarize,
register, process, combine) and multi-recording (two-phase: discover, extract) pipelines are supported through a
unified execution model that admits every job as soon as that job's own prerequisites succeed.
"""

from __future__ import annotations

import os
from enum import StrEnum
import shutil
from typing import Any
from pathlib import Path
from threading import Lock, Thread
from dataclasses import field, dataclass

import yaml  # type: ignore[import-untyped]
from natsort import natsorted
from ataraxis_time import (
    TimeUnits,
    PrecisionTimer,
    TimerPrecisions,
    TimestampFormats,
    TimestampPrecisions,
    convert_time,
    get_timestamp,
)
from ataraxis_base_utilities import resolve_worker_count
from ataraxis_data_structures import JobState, ProcessingStatus, ProcessingTracker

from ..io import resolve_multi_recording_contexts, resolve_single_recording_contexts
from ..pipelines import (
    MULTI_RECORDING_TRACKER_NAME,
    SINGLE_RECORDING_TRACKER_NAME,
    run_multi_recording_pipeline,
    run_single_recording_pipeline,
)
from ..allocation import (
    DISCOVERY_WORKERS,
    EXTRACTION_WORKERS,
    PROCESSING_WORKERS,
    BINARIZATION_WORKERS,
    REGISTRATION_WORKERS,
    MULTI_RECORDING_PHASES,
    SINGLE_RECORDING_PHASES,
    PrerequisiteScope,
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    resolve_pipeline_jobs,
    resolve_downstream_phases,
    resolve_multi_recording_jobs,
    resolve_single_recording_jobs,
)
from ..dataclasses import MultiRecordingConfiguration, SingleRecordingConfiguration
from .mcp_instance import mcp

_RESERVED_CORES: int = 2
"""The number of CPU cores reserved for system operations."""

_MAXIMUM_PARALLEL_IO_JOBS: int = 4
"""The maximum number of concurrent I/O-bound jobs (the binarization and combination resource classes)."""

_MINIMUM_RECORDING_COUNT: int = 2
"""The minimum number of recordings required for multi-recording processing."""

_COMBINATION_WORKERS: int = 1
"""The number of CPU cores one combination job holds. The combination stage merges the per-plane result files with
serial input and output and takes no worker argument, so each of its jobs occupies exactly one core."""


_PROCESSING_MEMORY_GIGABYTES_PER_JOB: float = 15.0
"""The peak resident memory, in gigabytes, that one processing job holds. Measured on a real nine-plane run where
detection peaked at 10.5 gigabytes on the smallest (512-line) plane, rounded up to 15 to cover the taller planes of the
same recording. The processing resource class bounds its concurrency by this figure because the job workers are threads
inside the MCP server process, so one out-of-memory kill terminates the whole server rather than a single job."""

_BYTES_PER_GIGABYTE: int = 1024**3
"""The number of bytes in one gigabyte, used to convert the operating system's page counts into a memory budget."""

_KIBIBYTES_PER_GIGABYTE: int = 1024**2
"""The number of kibibytes in one gigabyte, used to convert the Linux memory counters into a memory budget."""

_MEMORY_INFO_PATH: Path = Path("/proc/meminfo")
"""The path to the Linux memory counter file that reports how much memory new allocations can claim."""

_AVAILABLE_MEMORY_FIELDS: int = 3
"""The number of whitespace-separated fields a well-formed Linux memory counter line carries, which are the counter
name, the value, and the unit."""

_AVAILABLE_MEMORY_KEY: str = "MemAvailable:"
"""The Linux memory counter that estimates how much memory new allocations can claim without swapping, counting the
reclaimable page cache that registration leaves behind when it memory-maps a plane binary."""

_PREREQUISITE_FAILURE_MESSAGE: str = "Unable to execute job. A preceding pipeline phase failed."
"""The tracker error message recorded for a job whose prerequisite job failed."""

_UNREACHABLE_PREREQUISITE_MESSAGE: str = (
    "Unable to execute job. Its prerequisite jobs never succeeded and no queued job can still satisfy them."
)
"""The tracker error message recorded for a job that the execution session can no longer admit."""


class _AdmissionDecisions(StrEnum):
    """Defines the outcomes of evaluating one queued job's prerequisites against its own tracker."""

    ADMIT = "admit"
    """Every prerequisite job succeeded, so the job moves into its resource class queue."""
    WAIT = "wait"
    """At least one prerequisite job has not finished, so the job stays in the admission pool."""
    ABORT = "abort"
    """At least one prerequisite job failed or is absent from the tracker, so the job can never run."""


@dataclass(frozen=True, slots=True)
class _ResourceClass:
    """Describes the CPU and memory budget that one class of pipeline jobs holds for its entire duration."""

    name: str
    """The name of the resource class, used as the key of the per-class queues and of the reported allocation."""
    workers_per_job: int
    """The number of CPU cores each job of this class holds, taken from the measured stage defaults in
    cindra.allocation, except for the combination class, whose single core is defined by _COMBINATION_WORKERS."""
    fixed_parallel_jobs: int | None
    """The machine-independent concurrency cap of this class, or None when the cap is derived from the CPU budget and,
    for memory-bound classes, from the available system memory. A per-class cap bounds one class in isolation, so the
    dispatcher additionally holds the sum of the cores committed by every class inside the session CPU budget."""
    memory_gigabytes_per_job: float
    """The peak resident memory one job of this class holds, or 0.0 when the class does not bound its concurrency by
    memory."""


_BINARIZATION_RESOURCES: _ResourceClass = _ResourceClass(
    name="binarization",
    workers_per_job=BINARIZATION_WORKERS,
    fixed_parallel_jobs=_MAXIMUM_PARALLEL_IO_JOBS,
    memory_gigabytes_per_job=0.0,
)
"""The resource class of the binarization jobs. The allocated cores become the TIFF image decode threads, and the
stage streams frames to disk instead of holding them, so the concurrency cap is the fixed I/O limit."""

_REGISTRATION_RESOURCES: _ResourceClass = _ResourceClass(
    name="registration",
    workers_per_job=REGISTRATION_WORKERS,
    fixed_parallel_jobs=None,
    memory_gigabytes_per_job=0.0,
)
"""The resource class of the plane-registration jobs. Registration reads the plane binary through a memory map, so its
resident growth is evictable page cache and its concurrency is bounded by the shared CPU budget alone."""

_PROCESSING_RESOURCES: _ResourceClass = _ResourceClass(
    name="processing",
    workers_per_job=PROCESSING_WORKERS,
    fixed_parallel_jobs=None,
    memory_gigabytes_per_job=_PROCESSING_MEMORY_GIGABYTES_PER_JOB,
)
"""The resource class of the plane-processing jobs. Detection materializes the binned movie in anonymous memory, so
this class bounds its concurrency by both the shared CPU budget and the available system memory."""

_COMBINATION_RESOURCES: _ResourceClass = _ResourceClass(
    name="combination",
    workers_per_job=_COMBINATION_WORKERS,
    fixed_parallel_jobs=_MAXIMUM_PARALLEL_IO_JOBS,
    memory_gigabytes_per_job=0.0,
)
"""The resource class of the combination jobs. Combination merges per-plane result files with serial input and output,
so each job holds one core and the concurrency cap is the fixed I/O limit."""

_DISCOVERY_RESOURCES: _ResourceClass = _ResourceClass(
    name="discovery",
    workers_per_job=DISCOVERY_WORKERS,
    fixed_parallel_jobs=None,
    memory_gigabytes_per_job=0.0,
)
"""The resource class of the multi-recording discovery jobs. Discovery registers every recording of one animal against
the others, so each job holds the stage's saturating allocation and its concurrency is bounded by the shared CPU budget
alone."""

_EXTRACTION_RESOURCES: _ResourceClass = _ResourceClass(
    name="extraction",
    workers_per_job=EXTRACTION_WORKERS,
    fixed_parallel_jobs=None,
    memory_gigabytes_per_job=0.0,
)
"""The resource class of the multi-recording extraction jobs. Extraction reads each frame batch serially before the
kernel consumes it, so the stage plateaus at its measured worker count and the remaining budget is better spent on
running more recordings concurrently."""

_RESOURCE_CLASS_BY_JOB_NAME: dict[str, _ResourceClass] = {
    SingleRecordingJobNames.BINARIZE: _BINARIZATION_RESOURCES,
    SingleRecordingJobNames.REGISTER: _REGISTRATION_RESOURCES,
    SingleRecordingJobNames.PROCESS: _PROCESSING_RESOURCES,
    SingleRecordingJobNames.COMBINE: _COMBINATION_RESOURCES,
    MultiRecordingJobNames.DISCOVER: _DISCOVERY_RESOURCES,
    MultiRecordingJobNames.EXTRACT: _EXTRACTION_RESOURCES,
}
"""Maps every pipeline job name to the resource class that governs its worker count and its concurrency cap."""


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
    resource_class: _ResourceClass
    """The resource class that governs this job's worker count and the concurrency of its queue."""
    resolved_workers: int | None = None
    """The number of parallel workers to allocate to this job, assigned at dispatch time. A value of None makes the
    pipeline fall back to the measured default for the job's stage."""

    @property
    def dispatch_key(self) -> tuple[str, str]:
        """Returns the composite key that uniquely identifies this job across the entire batch, combining the tracker
        path with the job ID.
        """
        return str(self.tracker_path), self.job_id


@dataclass(slots=True)
class _JobExecutionState:
    """Tracks the runtime state for generic job execution across both pipeline types.

    Notes:
        Every submitted job first enters the admission pool. The execution manager admits a job into its resource
        class queue as soon as that job's own prerequisites have succeeded on its own tracker, so each recording
        advances independently. Each resource class then dispatches from its own queue under its own concurrency cap
        and under the session-wide CPU budget shared by every class.
    """

    all_jobs: dict[tuple[str, str], _PendingJob] = field(default_factory=dict)
    """All submitted jobs keyed by (tracker_path, job_id) dispatch key, used for status reporting."""
    admission_pool: list[_PendingJob] = field(default_factory=list)
    """Jobs awaiting prerequisite satisfaction, scanned by the manager on every polling cycle."""
    pending_queues: dict[str, list[_PendingJob]] = field(default_factory=dict)
    """Admitted jobs awaiting dispatch, keyed by resource class name."""
    active_threads: dict[str, dict[tuple[str, str], Thread]] = field(default_factory=dict)
    """Currently running dispatch key to Thread mappings, keyed by resource class name."""
    class_capacities: dict[str, int] = field(default_factory=dict)
    """The resolved maximum number of concurrent jobs for each resource class name."""
    class_workers: dict[str, int] = field(default_factory=dict)
    """The resolved number of CPU cores allocated to each job of each resource class name."""
    cpu_budget: int = 1
    """The total number of CPU cores this session may commit across every resource class at once."""
    lock: Lock = field(default_factory=Lock)
    """Thread synchronization lock for execution state access."""
    manager_thread: Thread | None = None
    """Background execution manager thread reference."""


_job_execution_state: _JobExecutionState | None = None
"""The module-level execution state for active processing jobs."""


@mcp.tool()
def get_recording_status_tool(recording_path: str) -> dict[str, object]:
    """Gets the processing status for a recording by reading all available ProcessingTracker files.

    Checks for both single-recording and multi-recording trackers under the recording's cindra output directory and
    returns status for all pipelines found. For single-recording, reads the tracker at
    <recording_path>/cindra/single_recording_tracker.yaml and returns per-phase job status (binarize, register,
    process, combine). For multi-recording, searches under <recording_path>/cindra/multi_recording/<dataset>/ for
    tracker files and returns per-dataset status (discover, extract).

    Args:
        recording_path: The absolute path to the recording OUTPUT directory, which is the parent of the cindra/ folder
            and equals the recording_output_paths entries returned by the prepare tool when the output root differs from
            the raw-data root. The cindra/ subdirectory is resolved directly under it with no fallback.

    Returns:
        On success, contains the 'recording_path', 'single_recording' status (per-phase jobs, summary, and synthesized
        status string), and 'multi_recording' status (per-dataset tracker status). Each section reports 'not_started'
        when no tracker exists. On failure, contains an 'error' describing the issue. Both cases include a 'success'
        flag. The synthesized single-recording status string is one of completed, failed, scheduled, binarizing,
        registering, processing, or combining. The synthesized multi-recording status string is one of completed,
        failed, scheduled, discovering, or extracting. These differ from the get_processing_jobs_status_tool summary
        vocabulary of pending, running, succeeded, and failed, which describes the same jobs (pending equals scheduled,
        and running spans the *-ing phases).
    """
    recording = Path(recording_path)

    if not recording.exists():
        return {
            "success": False,
            "error": f"Unable to get recording status. Recording directory not found: {recording_path}.",
        }

    single_tracker_path = recording / "cindra" / SINGLE_RECORDING_TRACKER_NAME
    if single_tracker_path.exists():
        single_recording_status = _read_single_recording_tracker(
            tracker_path=single_tracker_path, recording_path=recording
        )
    else:
        single_recording_status = {"status": "not_started"}

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
        On success, contains the 'root_directory', 'single_recordings' and 'multi_recordings' lists with per-tracker
        status, and a 'summary' with 'total_single_recordings', 'total_multi_recording_datasets', and aggregate counts
        for completed, failed, in_progress, and not_started pipelines, plus 'permission_errors' when a search was
        denied access. On failure, contains an 'error' describing the issue. Both cases include a 'success' flag. Each
        per-tracker entry carries a synthesized status string: single-recording entries use completed, failed,
        scheduled, binarizing, registering, processing, or combining, and multi-recording entries use completed,
        failed, scheduled, discovering, or extracting. The summary roll-up counts completed as completed, failed as
        failed, scheduled as not_started, and buckets every other status (the *-ing phases) as in_progress. These
        differ from the get_processing_jobs_status_tool summary vocabulary of pending, running, succeeded, and failed,
        which describes the same jobs (pending equals scheduled, and running spans the *-ing phases).
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

    single_tracker_paths: list[Path] = []
    try:
        single_tracker_paths.extend(root.rglob(SINGLE_RECORDING_TRACKER_NAME))
    except PermissionError as error:
        permission_errors.append(f"Access denied during single-recording search: {error}")

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
    resolves the plane count, and initializes a ProcessingTracker with all jobs (binarize, per-plane register, per-plane
    process, combine). Idempotent: if a tracker already exists for a recording, returns the existing manifest with
    current job statuses instead of reinitializing. The one exception is a tracker that carries process jobs but no
    register jobs, which gains the missing per-plane register jobs while every job it already holds keeps its recorded
    status. Use execute_processing_jobs_tool to dispatch jobs from the manifest and reset_processing_phases_tool to
    selectively reset completed phases for re-runs.

    Important:
        Worker allocation and parallelism are controlled by execute_processing_jobs_tool, not this tool. The execute
        tool resolves resource allocation at dispatch time and passes it to each job as a dispatch argument, so the
        configuration file this tool writes stays immutable and is safe to share between concurrently dispatched jobs.

    Args:
        recording_paths: List of absolute paths to recording root directories (used as file_io.data_path per
            recording). These should be session-level roots, not sub-paths to raw data. The pipeline resolves
            raw data locations internally via recursive search.
        configuration_path: The absolute path to the template configuration YAML file.
        recording_output_paths: List of absolute paths for per-recording output directories (used as
            file_io.output_path). Must match the length of recording_paths.

    Returns:
        On success, contains per-recording manifests in 'recordings' keyed by recording path, with each entry listing
        its configuration_path, tracker_path, output_path, pipeline_type, and per-phase job entries (binarize_job,
        register_jobs, process_jobs, combine_job) including job_id, name, specifier, and current status. The output_path
        is the absolute output directory and the parent of the cindra/ directory, where configuration_path equals
        <output_path>/cindra/configuration.yaml, so downstream verify, status, and clean tools need no re-derivation.
        Also includes 'total_recordings' and 'total_jobs' counts, plus 'migrated_recordings' listing any recording
        whose tracker gained the missing register jobs and 'invalid_paths' listing any provided path that is not an
        existing directory. On failure, contains an 'error' describing the issue.
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
    migrated_recordings: list[str] = []
    total_jobs = 0

    for data_path, output_path in zip(valid_paths, resolved_output_paths, strict=True):
        recording_key = str(data_path)
        cindra_root = output_path / "cindra"
        tracker_path = cindra_root / SINGLE_RECORDING_TRACKER_NAME

        if tracker_path.exists():
            # Idempotent path: tracker already exists, returns current state without reinitializing.
            tracker = ProcessingTracker(file_path=tracker_path)
            registry = tracker.snapshot()
            configuration_file_path = cindra_root / "configuration.yaml"

            binarize_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.BINARIZE)
            register_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.REGISTER)
            process_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.PROCESS)
            combine_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.COMBINE)

            # A tracker that carries process jobs but no register jobs would leave every processing job permanently
            # unable to satisfy its prerequisite. Registering the missing phase is additive, so every existing job
            # keeps its recorded status, and a registration job dispatched against an already-registered plane returns
            # immediately.
            if process_jobs and not register_jobs:
                plane_specifiers = natsorted({specifier for _, specifier in process_jobs.values()})
                # Rebuilds the universe over the tracker's existing plane specifiers, so that a migrated tracker keeps
                # the specifiers its processing jobs already carry.
                migrated_jobs: list[tuple[str, str]] = resolve_pipeline_jobs(
                    phases=SINGLE_RECORDING_PHASES, specifiers=plane_specifiers
                )
                tracker.align_jobs(jobs=migrated_jobs, universe=migrated_jobs)
                registry = tracker.snapshot()
                register_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.REGISTER)
                migrated_recordings.append(recording_key)

            binarize_entry: dict[str, object] = {}
            for job_id, (name, specifier) in binarize_jobs.items():
                job_info = registry[job_id]
                binarize_entry = {
                    "job_id": job_id,
                    "name": name,
                    "specifier": specifier,
                    "status": job_info.status.name.lower(),
                    "executor_id": job_info.executor_id,
                }

            register_entries: list[dict[str, object]] = [
                {
                    "job_id": job_id,
                    "name": name,
                    "specifier": specifier,
                    "status": registry[job_id].status.name.lower(),
                    "executor_id": registry[job_id].executor_id,
                }
                for job_id, (name, specifier) in register_jobs.items()
            ]

            process_entries: list[dict[str, object]] = [
                {
                    "job_id": job_id,
                    "name": name,
                    "specifier": specifier,
                    "status": registry[job_id].status.name.lower(),
                    "executor_id": registry[job_id].executor_id,
                }
                for job_id, (name, specifier) in process_jobs.items()
            ]

            combine_entry: dict[str, object] = {}
            for job_id, (name, specifier) in combine_jobs.items():
                job_info = registry[job_id]
                combine_entry = {
                    "job_id": job_id,
                    "name": name,
                    "specifier": specifier,
                    "status": job_info.status.name.lower(),
                    "executor_id": job_info.executor_id,
                }

            total_jobs += len(binarize_jobs) + len(register_jobs) + len(process_jobs) + len(combine_jobs)

            recordings_manifest[recording_key] = {
                "configuration_path": str(configuration_file_path),
                "tracker_path": str(tracker_path),
                "output_path": str(output_path),
                "pipeline_type": "single-recording",
                "binarize_job": binarize_entry,
                "register_jobs": register_entries,
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

            # Saves the per-recording configuration. The execute tool passes the resolved worker allocation to each
            # job as a dispatch argument, so this one file serves every job dispatched against it.
            recording_configuration.save(file_path=recording_configuration_path)

            # Builds the recording's job universe from the exported phase model, which orders the phases and expands
            # the per-plane ones.
            jobs: list[tuple[str, str]] = resolve_single_recording_jobs(plane_count=plane_count)

            tracker = ProcessingTracker(file_path=tracker_path)
            job_ids = tracker.initialize_jobs(jobs=jobs)
            total_jobs += len(jobs)

            # Builds manifest entries from the freshly initialized tracker. The returned identifiers follow the job
            # list order, so the register block starts at index 1 and the process block starts after it.
            binarize_entry = {
                "job_id": job_ids[0],
                "name": SingleRecordingJobNames.BINARIZE.value,
                "specifier": "",
                "status": "scheduled",
            }

            register_entries = [
                {
                    "job_id": job_ids[1 + plane_index],
                    "name": SingleRecordingJobNames.REGISTER.value,
                    "specifier": f"plane_{plane_index}",
                    "status": "scheduled",
                }
                for plane_index in range(plane_count)
            ]

            process_entries = [
                {
                    "job_id": job_ids[1 + plane_count + plane_index],
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
                "output_path": str(output_path),
                "pipeline_type": "single-recording",
                "binarize_job": binarize_entry,
                "register_jobs": register_entries,
                "process_jobs": process_entries,
                "combine_job": combine_entry,
            }

    result: dict[str, object] = {
        "success": True,
        "recordings": recordings_manifest,
        "total_recordings": len(recordings_manifest),
        "total_jobs": total_jobs,
    }

    if migrated_recordings:
        result["migrated_recordings"] = migrated_recordings

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
        tool resolves resource allocation at dispatch time and passes it to each job as a dispatch argument, so the
        configuration file this tool writes stays immutable and is safe to share between concurrently dispatched jobs.

    Args:
        dataset_configurations: List of dataset configurations, each a dictionary with 'configuration_path' (absolute
            path to the multi-recording YAML configuration), 'recording_paths' (list of absolute paths to recording
            directories), and 'dataset_name' (unique name for this dataset). At least 2 recording paths per dataset
            are required.

    Returns:
        On success, contains per-dataset manifests in 'datasets' keyed by the lowercased dataset name, with each entry
        listing its configuration_path, tracker_path, dataset_name, pipeline_type, and per-phase job entries
        (discover_job, extract_jobs) including job_id, name, specifier, and current status. The dataset_name field is
        the resolved lowercased dataset name. To verify a dataset, call verify_multi_recording_output_tool with the
        dataset_name plus any recording_path belonging to the dataset (one of the input recording_paths, whose cindra/
        subdirectory is resolved automatically). Also includes 'total_datasets' and 'total_jobs' counts, plus
        'invalid_configurations' listing every rejected dataset entry with its reason. On failure, contains an 'error'
        describing the issue.
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
            registry = tracker.snapshot()

            discover_jobs = tracker.find_jobs(job_name=MultiRecordingJobNames.DISCOVER)
            extract_jobs = tracker.find_jobs(job_name=MultiRecordingJobNames.EXTRACT)

            discover_entry: dict[str, object] = {}
            for job_id, (name, specifier) in discover_jobs.items():
                job_info = registry[job_id]
                discover_entry = {
                    "job_id": job_id,
                    "name": name,
                    "specifier": specifier,
                    "status": job_info.status.name.lower(),
                    "executor_id": job_info.executor_id,
                }

            extract_entries: list[dict[str, object]] = [
                {
                    "job_id": job_id,
                    "name": name,
                    "specifier": specifier,
                    "status": registry[job_id].status.name.lower(),
                    "executor_id": registry[job_id].executor_id,
                }
                for job_id, (name, specifier) in extract_jobs.items()
            ]

            total_jobs += len(discover_jobs) + len(extract_jobs)

            datasets_manifest[dataset_key] = {
                "configuration_path": str(configuration_file_path),
                "tracker_path": str(tracker_path),
                "dataset_name": dataset_key,
                "pipeline_type": "multi-recording",
                "discover_job": discover_entry,
                "extract_jobs": extract_entries,
            }
        else:
            # New dataset: saves configuration and initializes tracker.
            configuration.save(file_path=configuration_file_path)

            # Builds the dataset's job universe from the exported phase model.
            jobs: list[tuple[str, str]] = resolve_multi_recording_jobs(recording_ids=recording_ids)

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
                "dataset_name": dataset_key,
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

    This is the only way to reset completed phases. Prepare tools never reinitialize existing trackers. For each phase
    listed in ``phases``, all matching jobs are reset to SCHEDULED status. Downstream dependent phases are
    automatically included in the reset to maintain consistency (e.g., resetting 'binarization' also resets
    'registration', 'processing', and 'combination'). Jobs belonging to phases not in the expanded reset set retain
    their original status.

    Important:
        After resetting phases, modify the pipeline configuration file if needed (e.g., change ROI detection
        parameters) before calling execute_processing_jobs_tool. The pipeline reads configuration from disk at
        execution time.

    Args:
        tracker_path: The absolute path to the ProcessingTracker YAML file.
        phases: List of phase names to reset. For single-recording: 'binarization', 'registration', 'processing',
            'combination'. For multi-recording: 'discovery', 'extraction'. Downstream phases are automatically
            included.
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
    phases = sorted(resolve_downstream_phases(phase_names=phases, single_recording=pipeline_type == "single-recording"))

    tracker = ProcessingTracker(file_path=path)

    # Resets only the jobs whose phase is in the expanded reset set, reading the registry once. Jobs of preserved
    # phases are left untouched, so their recorded status, executor, and timing survive the reset unchanged.
    phases_set = set(phases)
    reset_ids = [job_id for job_id, state in tracker.snapshot().items() if state.job_name in phases_set]
    tracker.reset_jobs(job_ids=reset_ids)

    # Builds the response from a post-reset snapshot, reporting every job of a valid phase for this pipeline.
    updated_jobs: list[dict[str, object]] = [
        {
            "job_id": job_id,
            "name": state.job_name,
            "specifier": state.specifier,
            "status": state.status.name.lower(),
            "executor_id": state.executor_id,
        }
        for job_id, state in tracker.snapshot().items()
        if state.job_name in valid_phases
    ]

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
    to maintain consistency (e.g., cleaning 'binarization' also cleans 'registration', 'processing', and
    'combination'). Tracker files, configuration files, runtime_data.yaml, and acquisition_parameters.yaml are never
    deleted. Use this to reclaim disk space or force a full rerun from specific phases.

    Important:
        Each phase deletes only the files it owns. The per-plane detection_data directory is shared: binarize creates
        mean_image.npy and mean_image_channel_2.npy, and both register and process rewrite them while process also
        writes the remaining detection arrays. Cleaning 'binarization' therefore removes the two mean images, cleaning
        'registration' removes the whole registration_data directory including bad_frames.npy, and cleaning
        'processing' removes only the detection-owned arrays. The shared detection_data directory itself is left in
        place, empty if every phase that writes into it was cleaned.

        Cleaning 'registration' cannot undo registration. The registration stage rewrites channel_1_data.bin in place,
        while the pipeline decides whether a plane is registered from the presence of
        registration_data/reference_image.npy. Removing that sentinel leaves an already motion-corrected binary behind,
        so a re-run registers the corrected data a second time. Clean 'binarization' instead when the plane binary must
        be rebuilt from the raw TIFF files.

    Args:
        recording_path: The absolute path to the recording OUTPUT directory, which is the parent of the cindra/ folder
            and equals the recording_output_paths entries returned by the prepare tool when the output root differs from
            the raw-data root. The cindra/ subdirectory is resolved directly under it with no fallback.
        phases: List of phase names to clean. For single-recording: 'binarization', 'registration', 'processing',
            'combination'. For multi-recording: 'discovery', 'extraction'. Downstream phases are automatically
            included.
        pipeline_type: The pipeline type, either 'single-recording' or 'multi-recording'.
        dataset: The multi-recording dataset name. Required when pipeline_type is 'multi-recording'. It must be the
            resolved lowercased dataset directory name created by the prepare tool, located at
            cindra/multi_recording/<dataset>, and the match is case-sensitive.

    Returns:
        On success, contains a 'cleaned' flag, the 'recording_path', 'deleted_files', 'deleted_dirs', 'total_deleted',
        and the 'requested_phases' and 'effective_phases' after dependency expansion, plus 'errors' when a deletion
        failed. On failure, contains an 'error' describing the issue. Both cases include a 'success' flag.
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
    effective_phases = sorted(
        resolve_downstream_phases(phase_names=phases, single_recording=pipeline_type == "single-recording")
    )

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

        # Cleans per-plane files, partitioning the shared detection_data directory by the phase that owns each array.
        plane_directories = sorted(
            entry for entry in cindra_root.iterdir() if entry.is_dir() and entry.name.startswith("plane_")
        )
        for plane_directory in plane_directories:
            plane_detection_directory = plane_directory / "detection_data"

            if SingleRecordingJobNames.BINARIZE in effective_set:
                for name in ("channel_1_data.bin", "channel_2_data.bin"):
                    _delete_file(path=plane_directory / name, deleted=deleted_files, errors=errors)
                # The mean images are created by binarization and later rewritten by both registration and processing,
                # so binarization, the phase that creates them, owns their removal. Deleting them under a downstream
                # phase would discard the output of the phases between it and binarization.
                for name in ("mean_image.npy", "mean_image_channel_2.npy"):
                    _delete_file(path=plane_detection_directory / name, deleted=deleted_files, errors=errors)

            if SingleRecordingJobNames.REGISTER in effective_set:
                # The registration directory holds bad_frames.npy, which detection reads, so it is removed only when
                # the registration phase itself is cleaned.
                _delete_directory(path=plane_directory / "registration_data", deleted=deleted_dirs, errors=errors)

            if SingleRecordingJobNames.PROCESS in effective_set:
                for name in (
                    "enhanced_mean_image.npy",
                    "maximum_projection.npy",
                    "correlation_map.npy",
                    "enhanced_mean_image_channel_2.npy",
                    "maximum_projection_channel_2.npy",
                    "correlation_map_channel_2.npy",
                ):
                    _delete_file(path=plane_detection_directory / name, deleted=deleted_files, errors=errors)
                for name in (
                    "roi_masks.npz",
                    "roi_masks_channel_2.npz",
                    "roi_statistics.npz",
                    "roi_statistics_channel_2.npz",
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
                    "corrected_structural_mean_image.npy",
                ):
                    _delete_file(path=plane_directory / name, deleted=deleted_files, errors=errors)

        if SingleRecordingJobNames.COMBINE in effective_set:
            _delete_directory(path=cindra_root / "detection_data", deleted=deleted_dirs, errors=errors)
            for name in (
                "combined_metadata.npz",
                "roi_masks.npz",
                "roi_masks_channel_2.npz",
                "roi_statistics.npz",
                "roi_statistics_channel_2.npz",
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
                "corrected_structural_mean_image.npy",
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
                    # Backward-projected per-recording mask and statistics files are produced by the final
                    # discovery step (project_templates_to_recordings), not by extraction, and deleting them under
                    # EXTRACT strands the pipeline because extraction consumes them as inputs.
                    "roi_masks.npz",
                    "roi_masks_channel_2.npz",
                    "roi_statistics.npz",
                    "roi_statistics_channel_2.npz",
                ):
                    _delete_file(path=output_path / name, deleted=deleted_files, errors=errors)

            if MultiRecordingJobNames.EXTRACT in effective_set:
                for name in (
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

    Validates that each job's prerequisites are either already SUCCEEDED or submitted alongside it, resolves the CPU
    and memory allocation of every resource class involved, and starts a background execution manager. The manager
    admits each job into its resource class queue as soon as that job's own prerequisites succeed on its own tracker,
    so the whole dependency graph can be submitted in one call and each recording advances independently. Use
    get_processing_jobs_status_tool to monitor progress and cancel_processing_jobs_tool to stop execution.

    Important:
        Only one execution session can be active at a time. Wait for the current session to complete or cancel it before
        starting a new one. Jobs may be submitted in any order because admission follows the tracked dependency chain
        binarization to registration to processing to combination for single-recording work and discovery to extraction
        for multi-recording work.

        Each job runs under the resource class of its phase. Binarization holds 4 cores per job at a fixed concurrency
        of 4. Combination holds 1 core per job at the same fixed concurrency, because it merges result files serially.
        Registration holds 8 cores per job with a concurrency bounded by the CPU budget. Processing holds 10 cores per
        job with a concurrency bounded by both the CPU budget and the available system memory. The binarization and
        combination classes ignore both parameters below.

        Every class dispatches during the same cycle, so the dispatcher additionally holds the sum of the cores
        committed by the running jobs of every class inside the session CPU budget reported as 'cpu_budget'. That
        budget is the machine's core count minus the cores reserved for the system.

    Args:
        jobs: List of job descriptors, each a dictionary with 'configuration_path' (absolute path to the pipeline
            configuration file), 'tracker_path' (absolute path to the ProcessingTracker file), 'job_id' (the
            hexadecimal job identifier from the prepare manifest), and 'pipeline_type' ('single-recording' or
            'multi-recording').
        workers_per_job: CPU cores per job, overriding the measured default of every non-fixed resource class. Set to
            -1 to accept the measured defaults, which are 4 cores for binarization, 8 for registration, 10 for
            processing, 1 for combination, 30 for multi-recording discovery, and 16 for multi-recording extraction.
            The override is a single scalar applied to every non-fixed class alike.
        max_parallel_jobs: Maximum concurrent jobs per resource class, overriding the derived concurrency cap of every
            non-fixed resource class. Set to -1 to accept the derived caps.

    Returns:
        Always contains a 'success' flag indicating the tool ran. On a started session, also contains a 'started' flag,
        'total_jobs' dispatched, and the session 'cpu_budget' that bounds the classes in aggregate. A started session
        further reports a 'resource_classes' mapping with the resolved workers_per_job, max_parallel_jobs, and
        job_count of every class present in the session, and 'invalid_jobs' listing any jobs that failed validation
        with reasons. On failure, contains success:False and an 'error' describing the issue.
    """
    if not jobs:
        return {"success": False, "error": "Unable to execute jobs. At least one job descriptor is required."}

    active_session_error = _check_active_session(action="execute jobs")
    if active_session_error is not None:
        return active_session_error

    # Validates each job entry and resolves the resource class that governs its allocation.
    required_keys = {"configuration_path", "tracker_path", "job_id", "pipeline_type"}
    candidate_jobs: list[_PendingJob] = []
    submitted_by_tracker: dict[str, set[str]] = {}
    invalid_jobs: list[dict[str, str]] = []

    for job_entry in jobs:
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

        # Validates that the job_id exists in the tracker and reads its job name to resolve the resource class.
        tracker = ProcessingTracker(file_path=tracker_file)
        try:
            job_info = tracker.get_job_info(job_id=job_id)
        except Exception:
            invalid_jobs.append({"job_id": job_id, "reason": f"Job ID not found in tracker: {tracker_file}"})
            continue

        resource_class = _RESOURCE_CLASS_BY_JOB_NAME.get(job_info.job_name)
        if resource_class is None:
            invalid_jobs.append({"job_id": job_id, "reason": f"Unrecognized pipeline phase: {job_info.job_name}"})
            continue

        candidate_jobs.append(
            _PendingJob(
                configuration_path=configuration_file,
                tracker_path=tracker_file,
                job_id=job_id,
                single_recording=pipeline_type == "single-recording",
                resource_class=resource_class,
            )
        )
        submitted_by_tracker.setdefault(str(tracker_file), set()).add(job_id)

    # Validates prerequisites once the full submission is known, so a job whose prerequisite is submitted alongside it
    # is accepted and admitted later, when that prerequisite actually succeeds.
    all_jobs_map: dict[tuple[str, str], _PendingJob] = {}
    for candidate_job in candidate_jobs:
        prerequisite_error = _validate_job_prerequisites(
            tracker=ProcessingTracker(file_path=candidate_job.tracker_path),
            job_id=candidate_job.job_id,
            single_recording=candidate_job.single_recording,
            submitted_job_ids=frozenset(submitted_by_tracker[str(candidate_job.tracker_path)]),
        )
        if prerequisite_error is not None:
            invalid_jobs.append({"job_id": candidate_job.job_id, "reason": prerequisite_error})
            continue
        all_jobs_map[candidate_job.dispatch_key] = candidate_job

    if not all_jobs_map:
        return {
            "success": False,
            "error": "Unable to execute jobs. No valid jobs after validation.",
            "invalid_jobs": invalid_jobs,
        }

    return _start_execution_session(
        all_jobs=all_jobs_map,
        workers_per_job=workers_per_job,
        max_parallel_jobs=max_parallel_jobs,
        extra_result_fields={"invalid_jobs": invalid_jobs} if invalid_jobs else {},
    )


@mcp.tool()
def get_processing_jobs_status_tool() -> dict[str, object]:
    """Returns the current status of the active job execution session.

    Reads ProcessingTracker files from disk for each job in the execution session to report per-job progress. Per-job
    status comes from the on-disk tracker files rather than in-memory state, while the session-level 'active',
    'awaiting_prerequisites', and 'resource_classes' fields come from the in-memory execution state.

    Returns:
        Always contains a 'success' flag indicating the tool ran. On an active session, also contains an 'active' flag,
        per-job status entries in 'jobs', a 'summary' with counts for pending, running, succeeded, and failed jobs, and
        an 'awaiting_prerequisites' count of jobs still in the admission pool. An active session further reports a
        'resource_classes' mapping with the resolved 'workers_per_job' and 'max_parallel_jobs' of every class in the
        session, together with its 'pending' job count and the list of 'active' dispatch keys. The 'active' flag
        reflects manager-thread liveness, not whether jobs ever ran. The execution manager clears session state once
        the session drains, so afterwards this tool reports active:False with empty 'jobs' and a zero 'summary' plus a
        'note'. Final per-job outcomes must then be re-read via get_recording_status_tool,
        get_batch_status_overview_tool, or verify_*_output_tool.
    """
    if _job_execution_state is None:
        return {
            "success": True,
            "active": False,
            "jobs": [],
            "summary": {"pending": 0, "running": 0, "succeeded": 0, "failed": 0},
            "note": (
                "No execution session is active. Final per-job outcomes can be read via get_recording_status_tool "
                "or get_batch_status_overview_tool."
            ),
        }

    # Binds the session to a local name, because the execution manager clears the module-level reference the moment
    # the session drains and this tool must keep reporting on the session it started with.
    state = _job_execution_state

    with state.lock:
        awaiting_prerequisites = len(state.admission_pool)
        class_status: dict[str, object] = {
            class_name: {
                "workers_per_job": state.class_workers.get(class_name, 0),
                "max_parallel_jobs": state.class_capacities.get(class_name, 0),
                "pending": len(state.pending_queues.get(class_name, [])),
                "active": list(state.active_threads.get(class_name, {}).keys()),
            }
            for class_name in state.class_capacities
        }

    # Reads per-job status from tracker files (outside lock to avoid holding it during I/O).
    jobs_status: list[dict[str, object]] = []
    summary_counts: dict[str, int] = {"pending": 0, "running": 0, "succeeded": 0, "failed": 0}

    for pending_job in state.all_jobs.values():
        tracker = ProcessingTracker(file_path=pending_job.tracker_path)
        status = tracker.get_job_status(job_id=pending_job.job_id)
        job_info = tracker.get_job_info(job_id=pending_job.job_id)
        status_name = status.name.lower()

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
            "resource_class": pending_job.resource_class.name,
            "pipeline_type": "single-recording" if pending_job.single_recording else "multi-recording",
            "tracker_path": str(pending_job.tracker_path),
        }

        if job_info.error_message:
            job_entry["error"] = job_info.error_message

        jobs_status.append(job_entry)

    manager_alive = state.manager_thread is not None and state.manager_thread.is_alive()

    return {
        "success": True,
        "active": manager_alive,
        "awaiting_prerequisites": awaiting_prerequisites,
        "resource_classes": class_status,
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
        Always contains a 'success' flag indicating the tool ran. Also contains an 'active' flag, per-job timing in
        'jobs', and a 'session' summary with total_elapsed_seconds, completed_count, failed_count, running_count, and
        pending_count, plus throughput_jobs_per_hour when applicable.
    """
    if _job_execution_state is None:
        return {
            "success": True,
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

    # Binds the session to a local name, because the execution manager clears the module-level reference the moment
    # the session drains and this tool must keep reporting on the session it started with.
    state = _job_execution_state

    current_microseconds = int(
        get_timestamp(output_format=TimestampFormats.INTEGER, precision=TimestampPrecisions.MICROSECOND)
    )

    jobs_timing: list[dict[str, object]] = []
    earliest_start: int | None = None
    completed_count = 0
    failed_count = 0
    running_count = 0
    pending_count = 0

    for pending_job in state.all_jobs.values():
        tracker = ProcessingTracker(file_path=pending_job.tracker_path)
        job_info = tracker.get_job_info(job_id=pending_job.job_id)

        entry: dict[str, object] = {
            "job_id": pending_job.job_id,
            "name": job_info.job_name,
            "specifier": job_info.specifier,
            "status": job_info.status.name.lower(),
        }

        if job_info.started_at is not None:
            started_at_microseconds = int(job_info.started_at)
            entry["started_at"] = started_at_microseconds
            if earliest_start is None or started_at_microseconds < earliest_start:
                earliest_start = started_at_microseconds

        if job_info.completed_at is not None:
            entry["completed_at"] = job_info.completed_at

        if job_info.status == ProcessingStatus.RUNNING and job_info.started_at is not None:
            entry["elapsed_seconds"] = round(
                convert_time(
                    time=current_microseconds - int(job_info.started_at),
                    from_units=TimeUnits.MICROSECOND,
                    to_units=TimeUnits.SECOND,
                    as_float=True,
                ),
                ndigits=2,
            )
            running_count += 1
        elif job_info.status == ProcessingStatus.SUCCEEDED:
            if job_info.started_at is not None and job_info.completed_at is not None:
                entry["duration_seconds"] = round(
                    convert_time(
                        time=int(job_info.completed_at) - int(job_info.started_at),
                        from_units=TimeUnits.MICROSECOND,
                        to_units=TimeUnits.SECOND,
                        as_float=True,
                    ),
                    ndigits=2,
                )
            completed_count += 1
        elif job_info.status == ProcessingStatus.FAILED:
            if job_info.started_at is not None and job_info.completed_at is not None:
                entry["duration_seconds"] = round(
                    convert_time(
                        time=int(job_info.completed_at) - int(job_info.started_at),
                        from_units=TimeUnits.MICROSECOND,
                        to_units=TimeUnits.SECOND,
                        as_float=True,
                    ),
                    ndigits=2,
                )
            failed_count += 1
        else:
            pending_count += 1

        jobs_timing.append(entry)

    total_elapsed = (
        round(
            convert_time(
                time=current_microseconds - earliest_start,
                from_units=TimeUnits.MICROSECOND,
                to_units=TimeUnits.SECOND,
                as_float=True,
            ),
            ndigits=2,
        )
        if earliest_start is not None
        else 0.0
    )

    session: dict[str, object] = {
        "total_elapsed_seconds": total_elapsed,
        "completed_count": completed_count,
        "failed_count": failed_count,
        "running_count": running_count,
        "pending_count": pending_count,
    }

    if total_elapsed > 0 and completed_count > 0:
        session["throughput_jobs_per_hour"] = round(
            completed_count
            / convert_time(time=total_elapsed, from_units=TimeUnits.SECOND, to_units=TimeUnits.HOUR, as_float=True),
            ndigits=2,
        )

    manager_alive = state.manager_thread is not None and state.manager_thread.is_alive()

    return {
        "success": True,
        "active": manager_alive,
        "jobs": jobs_timing,
        "session": session,
    }


@mcp.tool()
def cancel_processing_jobs_tool() -> dict[str, object]:
    """Cancels the active job execution session.

    Clears the admission pool and every resource class queue to prevent new jobs from starting and resets the execution
    state. Already-dispatched worker threads keep running because session state is cleared immediately. The agent
    should therefore poll get_recording_status_tool on the affected recordings or datasets until previously-RUNNING jobs
    leave RUNNING before starting a new session, to avoid colliding with still-running cancelled jobs. Calling cancel
    when no session is active is a safe no-op.

    Returns:
        Always contains a 'success' flag indicating the tool ran. On an active session, also contains a 'canceled'
        flag, a 'message' describing the outcome, and a 'final_state' with counts for succeeded_jobs, failed_jobs, and
        active_jobs_at_cancel. With no active session, contains canceled:False plus a 'note' stating that no session is
        active and that final per-job outcomes can be read via get_recording_status_tool or
        get_batch_status_overview_tool.
    """
    global _job_execution_state

    if _job_execution_state is None:
        return {
            "success": True,
            "canceled": False,
            "message": "No execution session is active.",
            "note": (
                "No execution session is active. Final per-job outcomes can be read via get_recording_status_tool "
                "or get_batch_status_overview_tool."
            ),
        }

    # Binds the session to a local name, because the execution manager can clear the module-level reference while this
    # tool reads the final tracker state.
    state = _job_execution_state

    with state.lock:
        active_count = sum(len(threads) for threads in state.active_threads.values())

        # Clears the admission pool and every class queue to prevent new jobs from starting.
        state.admission_pool.clear()
        for pending_queue in state.pending_queues.values():
            pending_queue.clear()

        total_succeeded = 0
        total_failed = 0
        seen_trackers: set[Path] = set()
        for pending_job in state.all_jobs.values():
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
        "success": True,
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
    """Executes a complete pipeline from preparation through all phases with automatic dependency sequencing.

    Prepares, validates, and dispatches every outstanding job of a full pipeline run in one session. Jobs are grouped
    by phase for reporting, but each job is admitted independently as soon as its own prerequisites succeed on its own
    tracker. A plane whose registration finishes early therefore starts processing while its peers are still
    registering, and a failure in one recording only aborts the jobs that transitively depend on it.

    For single-recording pipelines, the four phases are: binarize, register, process, combine. For multi-recording
    pipelines, the two phases are: discover, extract.

    Args:
        pipeline_type: The pipeline type, either 'single-recording' or 'multi-recording'.
        recording_paths: List of absolute paths to recording directories. Required for single-recording pipelines.
        configuration_path: Absolute path to the template configuration file. Required for single-recording pipelines.
        recording_output_paths: List of per-recording output paths for single-recording pipelines. Required for
            single-recording pipelines and must match the length of recording_paths.
        dataset_configurations: List of dataset configuration dictionaries. Required for multi-recording pipelines.
            Each must contain 'configuration_path', 'recording_paths', and 'dataset_name'.
        workers_per_job: CPU cores per job, overriding the measured default of every non-fixed resource class. Set to
            -1 to accept the measured defaults of 4 cores for binarization, 1 for combination, 8 for registration, 10
            for processing, 30 for multi-recording discovery, and 16 for multi-recording extraction.
        max_parallel_jobs: Maximum concurrent jobs per resource class, overriding the derived concurrency cap of every
            non-fixed resource class. Set to -1 to accept the derived caps.

    Returns:
        Always contains a 'success' flag indicating the tool ran, and callers MUST also check the 'started' flag rather
        than 'success' alone. When jobs are dispatched, contains started:True, 'total_jobs', 'phase_count', per-phase
        'phases' with job counts and IDs, the session 'cpu_budget' that bounds the classes in aggregate, and a
        'resource_classes' mapping with the resolved allocation of every class in the session. When all phases are
        already complete, returns {success:True, started:False, message:"All pipeline phases are already completed.",
        total_jobs:0, phase_count:0, phases:[]} plus a 'next_step' string. On failure, contains success:False and an
        'error' describing the issue. Cascade-aborted downstream jobs are recorded in their trackers as FAILED with
        the exact message "Unable to execute job. A preceding pipeline phase failed.", distinguishing them from
        genuine per-job failures. A job aborted because its prerequisite phase is absent from the tracker instead
        records a message naming that phase and asking for the prepare tool to be re-run, because no phase failed in
        that case.
    """
    if pipeline_type not in ("single-recording", "multi-recording"):
        return {
            "success": False,
            "error": (
                f"Unable to execute full pipeline. Invalid pipeline_type '{pipeline_type}'. "
                f"Must be 'single-recording' or 'multi-recording'."
            ),
        }

    active_session_error = _check_active_session(action="execute full pipeline")
    if active_session_error is not None:
        return active_session_error

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

    # Parses the manifest into phase groups. The groups order the admission pool and shape the response summary, while
    # the actual execution order follows each job's own prerequisites.
    phase_groups: list[tuple[str, list[_PendingJob]]] = []

    if pipeline_type == "single-recording":
        binarize_phase_jobs: list[_PendingJob] = []
        register_phase_jobs: list[_PendingJob] = []
        process_phase_jobs: list[_PendingJob] = []
        combine_phase_jobs: list[_PendingJob] = []

        raw_recordings = manifest.get("recordings", {})
        if isinstance(raw_recordings, dict):
            for recording_manifest in raw_recordings.values():
                manifest_dict: dict[str, Any] = recording_manifest
                job_configuration_path = Path(str(manifest_dict["configuration_path"]))
                tracker_path = Path(str(manifest_dict["tracker_path"]))

                binarize = manifest_dict.get("binarize_job", {})
                if binarize and binarize.get("status") != "succeeded":
                    binarize_phase_jobs.append(
                        _PendingJob(
                            configuration_path=job_configuration_path,
                            tracker_path=tracker_path,
                            job_id=binarize["job_id"],
                            single_recording=True,
                            resource_class=_BINARIZATION_RESOURCES,
                        )
                    )

                register_phase_jobs.extend(
                    _PendingJob(
                        configuration_path=job_configuration_path,
                        tracker_path=tracker_path,
                        job_id=register["job_id"],
                        single_recording=True,
                        resource_class=_REGISTRATION_RESOURCES,
                    )
                    for register in manifest_dict.get("register_jobs", [])
                    if register.get("status") != "succeeded"
                )

                process_phase_jobs.extend(
                    _PendingJob(
                        configuration_path=job_configuration_path,
                        tracker_path=tracker_path,
                        job_id=process["job_id"],
                        single_recording=True,
                        resource_class=_PROCESSING_RESOURCES,
                    )
                    for process in manifest_dict.get("process_jobs", [])
                    if process.get("status") != "succeeded"
                )

                combine = manifest_dict.get("combine_job", {})
                if combine and combine.get("status") != "succeeded":
                    combine_phase_jobs.append(
                        _PendingJob(
                            configuration_path=job_configuration_path,
                            tracker_path=tracker_path,
                            job_id=combine["job_id"],
                            single_recording=True,
                            resource_class=_COMBINATION_RESOURCES,
                        )
                    )

        if binarize_phase_jobs:
            phase_groups.append((SingleRecordingJobNames.BINARIZE.value, binarize_phase_jobs))
        if register_phase_jobs:
            phase_groups.append((SingleRecordingJobNames.REGISTER.value, register_phase_jobs))
        if process_phase_jobs:
            phase_groups.append((SingleRecordingJobNames.PROCESS.value, process_phase_jobs))
        if combine_phase_jobs:
            phase_groups.append((SingleRecordingJobNames.COMBINE.value, combine_phase_jobs))

    else:
        discover_phase_jobs: list[_PendingJob] = []
        extract_phase_jobs: list[_PendingJob] = []

        raw_datasets = manifest.get("datasets", {})
        if isinstance(raw_datasets, dict):
            for dataset_manifest in raw_datasets.values():
                manifest_dict = dataset_manifest
                job_configuration_path = Path(str(manifest_dict["configuration_path"]))
                tracker_path = Path(str(manifest_dict["tracker_path"]))

                discover = manifest_dict.get("discover_job", {})
                if discover and discover.get("status") != "succeeded":
                    discover_phase_jobs.append(
                        _PendingJob(
                            configuration_path=job_configuration_path,
                            tracker_path=tracker_path,
                            job_id=discover["job_id"],
                            single_recording=False,
                            resource_class=_DISCOVERY_RESOURCES,
                        )
                    )

                extract_phase_jobs.extend(
                    _PendingJob(
                        configuration_path=job_configuration_path,
                        tracker_path=tracker_path,
                        job_id=extract["job_id"],
                        single_recording=False,
                        resource_class=_EXTRACTION_RESOURCES,
                    )
                    for extract in manifest_dict.get("extract_jobs", [])
                    if extract.get("status") != "succeeded"
                )

        if discover_phase_jobs:
            phase_groups.append((MultiRecordingJobNames.DISCOVER.value, discover_phase_jobs))
        if extract_phase_jobs:
            phase_groups.append((MultiRecordingJobNames.EXTRACT.value, extract_phase_jobs))

    if not phase_groups:
        return {
            "success": True,
            "started": False,
            "message": "All pipeline phases are already completed.",
            "pipeline_type": pipeline_type,
            "total_jobs": 0,
            "phase_count": 0,
            "phases": [],
            "next_step": (
                "All phases complete; call reset_processing_phases_tool to force a re-run of a specific phase."
            ),
        }

    # Collects all jobs across all phases for the execution state, preserving the phase order so that the admission
    # scan always considers upstream jobs before the jobs that depend on them.
    all_jobs_map: dict[tuple[str, str], _PendingJob] = {}
    for _phase_name, phase_jobs in phase_groups:
        for job in phase_jobs:
            all_jobs_map[job.dispatch_key] = job

    # Builds phase summary for response before delegating to shared execution setup.
    phases_summary: list[dict[str, object]] = [
        {
            "phase_name": phase_name,
            "job_count": len(phase_job_list),
            "job_ids": [job.job_id for job in phase_job_list],
        }
        for phase_name, phase_job_list in phase_groups
    ]

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
        workers_per_job=workers_per_job,
        max_parallel_jobs=max_parallel_jobs,
        extra_result_fields=extra_fields,
    )


def _check_active_session(action: str) -> dict[str, object] | None:
    """Reports whether an execution session is already running and therefore blocks a new one.

    Args:
        action: The lowercase description of the blocked action, interpolated into the error message.

    Returns:
        None when no session is active, or an error result dictionary describing the running session.
    """
    # Binds the session to a local name, because the execution manager can clear the module-level reference between
    # the emptiness check and the lock acquisition.
    state = _job_execution_state
    if state is None:
        return None

    with state.lock:
        pending_count = len(state.admission_pool) + sum(len(queue) for queue in state.pending_queues.values())
        active_count = sum(len(threads) for threads in state.active_threads.values())
        if not pending_count and not active_count:
            return None

        return {
            "success": False,
            "error": f"Unable to {action}. An execution session is already active.",
            "pending_count": pending_count,
            "active_count": active_count,
        }


def _start_execution_session(
    all_jobs: dict[tuple[str, str], _PendingJob],
    workers_per_job: int,
    max_parallel_jobs: int,
    extra_result_fields: dict[str, object],
) -> dict[str, object]:
    """Resolves per-class resource allocation, stamps it onto the queued jobs, and starts the execution manager.

    Centralizes the execution setup logic shared by ``execute_processing_jobs_tool`` and
    ``execute_full_pipeline_tool``. The caller is responsible for validating jobs and checking for active sessions
    before calling this function.

    Notes:
        Every job enters the admission pool, and the manager decides admission from the tracked prerequisites.

        The resolved allocation is stamped onto each job and travels to the pipeline as a dispatch argument, so one
        configuration file serves every job dispatched concurrently against it.

        Each class resolves its own concurrency cap, and the session CPU budget is recorded alongside those caps
        because every class dispatches during the same cycle. The dispatcher holds the sum of the cores committed by
        the running jobs of every class inside that budget, so the per-class caps cannot oversubscribe the machine
        between them.

    Args:
        all_jobs: All submitted jobs keyed by dispatch key, in the order the manager should consider them.
        workers_per_job: Requested CPU cores per job (-1 to accept each resource class default).
        max_parallel_jobs: Requested maximum concurrent jobs per resource class (-1 to accept the derived caps).
        extra_result_fields: Additional key-value pairs to include in the result dictionary.

    Returns:
        A result dictionary containing 'success', 'started', the session 'cpu_budget', per-class resource allocation
        details, and any extra fields.
    """
    global _job_execution_state

    budget = resolve_worker_count(requested_workers=-1, reserved_cores=_RESERVED_CORES)
    available_memory = _resolve_available_memory_gigabytes()

    # Counts the jobs of every resource class present in this session, which bounds each class capacity.
    class_job_counts: dict[str, int] = {}
    classes_by_name: dict[str, _ResourceClass] = {}
    for pending_job in all_jobs.values():
        class_name = pending_job.resource_class.name
        classes_by_name[class_name] = pending_job.resource_class
        class_job_counts[class_name] = class_job_counts.get(class_name, 0) + 1

    # Resolves the allocation of every class and stamps the worker count onto its jobs.
    class_workers: dict[str, int] = {}
    class_capacities: dict[str, int] = {}
    for class_name, resource_class in classes_by_name.items():
        workers, capacity = _resolve_class_allocation(
            resource_class=resource_class,
            budget=budget,
            available_memory=available_memory,
            job_count=class_job_counts[class_name],
            workers_per_job=workers_per_job,
            max_parallel_jobs=max_parallel_jobs,
        )
        class_workers[class_name] = workers
        class_capacities[class_name] = capacity

    for pending_job in all_jobs.values():
        pending_job.resolved_workers = class_workers[pending_job.resource_class.name]

    execution_state = _JobExecutionState(
        all_jobs=all_jobs,
        admission_pool=list(all_jobs.values()),
        pending_queues={class_name: [] for class_name in classes_by_name},
        active_threads={class_name: {} for class_name in classes_by_name},
        class_capacities=class_capacities,
        class_workers=class_workers,
        cpu_budget=budget,
        lock=Lock(),
    )

    # Assigns the global state before starting the manager thread to prevent a race condition where the manager
    # reads _job_execution_state as None and exits immediately.
    _job_execution_state = execution_state
    manager = Thread(target=_job_execution_manager, daemon=True)
    manager.start()
    execution_state.manager_thread = manager

    result: dict[str, object] = {
        "success": True,
        "started": True,
        "total_jobs": len(all_jobs),
        "cpu_budget": budget,
        "resource_classes": {
            class_name: {
                "workers_per_job": class_workers[class_name],
                "max_parallel_jobs": class_capacities[class_name],
                "job_count": class_job_counts[class_name],
            }
            for class_name in classes_by_name
        },
    }
    result.update(extra_result_fields)

    return result


def _resolve_class_allocation(
    resource_class: _ResourceClass,
    *,
    budget: int,
    available_memory: float | None,
    job_count: int,
    workers_per_job: int,
    max_parallel_jobs: int,
) -> tuple[int, int]:
    """Resolves the per-job worker count and the concurrency cap of one resource class.

    Notes:
        A class with a fixed concurrency cap describes I/O-bound work whose throughput does not follow the core count,
        so it keeps its measured allocation and ignores both overrides. Every other class takes its measured worker
        count, bounds its concurrency by the CPU budget, and bounds it further by the available system memory when the
        class declares a per-job memory footprint.

        Every cap resolved here bounds one class in isolation, because a class cannot know which other classes will be
        dispatching alongside it. The dispatcher therefore holds the sum of the cores committed by every class inside
        the same CPU budget at run time.

    Args:
        resource_class: The resource class to resolve the allocation for.
        budget: The number of CPU cores available to the session after reserving system cores.
        available_memory: The available system memory in gigabytes, or None when the platform does not report it.
        job_count: The number of jobs of this class in the session, which caps the useful concurrency.
        workers_per_job: The requested CPU cores per job (-1 to accept the class default).
        max_parallel_jobs: The requested concurrency cap (-1 to accept the derived cap).

    Returns:
        A (workers_per_job, max_parallel_jobs) tuple for this resource class.
    """
    if resource_class.fixed_parallel_jobs is not None:
        workers = resource_class.workers_per_job
        return workers, min(resource_class.fixed_parallel_jobs, max(1, job_count))

    workers = workers_per_job if workers_per_job > 0 else resource_class.workers_per_job

    if max_parallel_jobs > 0:
        return workers, max_parallel_jobs

    capacity = max(1, budget // workers)
    if resource_class.memory_gigabytes_per_job > 0 and available_memory is not None:
        capacity = min(capacity, max(1, int(available_memory // resource_class.memory_gigabytes_per_job)))

    return workers, min(capacity, max(1, job_count))


def _resolve_available_memory_gigabytes() -> float | None:
    """Resolves the amount of system memory that new allocations can claim, in gigabytes.

    Notes:
        On Linux the value comes from the MemAvailable counter, which counts the reclaimable page cache. The POSIX
        SC_AVPHYS_PAGES counter is not used there because it reports MemFree instead, which collapses once registration
        fills the page cache with the plane binaries it memory-maps. That would throttle the memory-bound classes to a
        near-serial concurrency on a machine that is not actually short of memory.

        Hosts without the Linux counter, such as macOS, fall back to the POSIX page counters. Platforms without either
        report no value, and the memory-bound resource classes then bound their concurrency by the CPU budget alone.

        The value is sampled once, when the execution session starts. MemAvailable already discounts the page cache
        that the session itself will fill, so the sample stays representative for the lifetime of the session.

    Returns:
        The available system memory in gigabytes, or None when the platform does not report it.
    """
    linux_available = _read_linux_available_memory_gigabytes()
    if linux_available is not None:
        return linux_available

    try:
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except AttributeError, OSError, ValueError:
        return None

    if available_pages <= 0 or page_size <= 0:
        return None

    return (available_pages * page_size) / _BYTES_PER_GIGABYTE


def _read_linux_available_memory_gigabytes() -> float | None:
    """Reads the Linux MemAvailable counter and converts it to gigabytes.

    Returns:
        The memory that new allocations can claim, in gigabytes, or None when the host does not expose the counter or
        the counter cannot be parsed.
    """
    try:
        memory_info = _MEMORY_INFO_PATH.read_text()
    except OSError:
        return None

    for line in memory_info.splitlines():
        if not line.startswith(_AVAILABLE_MEMORY_KEY):
            continue
        fields = line.split()
        if len(fields) < _AVAILABLE_MEMORY_FIELDS:
            return None
        try:
            available_kibibytes = int(fields[1])
        except ValueError:
            return None
        if available_kibibytes <= 0:
            return None
        return available_kibibytes / _KIBIBYTES_PER_GIGABYTE

    return None


def _resolve_prerequisite_job_ids(
    registry: dict[str, JobState], job_id: str, *, single_recording: bool
) -> tuple[list[str], str | None]:
    """Resolves the tracker job IDs that must succeed before the target job can run.

    Notes:
        The single-recording chain runs binarization to registration to processing to combination and the
        multi-recording chain runs discovery to extraction. Each job depends on its immediate predecessor only, because
        a succeeded predecessor already implies the phases above it. Registration and processing pair up per plane, so
        a processing job depends only on the registration job carrying the same specifier.

        A prerequisite phase that the tracker does not contain is reported as an error rather than treated as
        satisfied, which prevents an incompletely initialized tracker from admitting a job whose input never exists.

    Args:
        registry: The point-in-time job registry of the tracker that owns the target job.
        job_id: The unique hexadecimal identifier of the job to resolve the prerequisites for.
        single_recording: Determines whether to apply single-recording or multi-recording prerequisite rules.

    Returns:
        A tuple of the prerequisite job IDs and an error message. The message is None unless the target job itself is
        not registered in the tracker or its prerequisite phase is absent from the tracker.
    """
    job_state = registry.get(job_id)
    if job_state is None:
        message = (
            f"Unable to resolve the prerequisites for job {job_id}. The job is not registered in the tracker that "
            f"was provided for it."
        )
        return [], message

    # Reads the dependency from the exported phase model, so the rule the interface layer enforces and the rule the
    # phase model publishes to external schedulers cannot drift apart.
    phases = SINGLE_RECORDING_PHASES if single_recording else MULTI_RECORDING_PHASES
    phase = {str(entry.job_name): entry for entry in phases}.get(job_state.job_name)
    if phase is None or phase.prerequisite is None:
        return [], None

    specifier = job_state.specifier if phase.prerequisite_scope == PrerequisiteScope.MATCHING_SPECIFIER else None
    return _collect_phase_job_ids(
        registry=registry,
        job_name=phase.prerequisite,
        specifier=specifier,
        dependent_job_id=job_id,
    )


def _collect_phase_job_ids(
    registry: dict[str, JobState], job_name: str, specifier: str | None, dependent_job_id: str
) -> tuple[list[str], str | None]:
    """Collects the tracker job IDs belonging to a prerequisite phase and reports an absent phase.

    Args:
        registry: The point-in-time job registry of the tracker that owns the dependent job.
        job_name: The name of the prerequisite phase to collect the jobs of.
        specifier: The specifier the prerequisite job must carry, or None to collect every job of the phase.
        dependent_job_id: The identifier of the job that depends on this phase, used in the error message.

    Returns:
        A tuple of the matching job IDs and an error message, where the message is None unless the phase has no
        matching jobs.
    """
    matches = [
        candidate_id
        for candidate_id, state in registry.items()
        if state.job_name == job_name and (specifier is None or state.specifier == specifier)
    ]

    if not matches:
        scope = "" if specifier is None else f" with specifier '{specifier}'"
        message = (
            f"Unable to execute job {dependent_job_id}. Its prerequisite '{job_name}' phase{scope} is not registered "
            f"in the tracker, so the prerequisite can never be satisfied. Re-run the prepare tool for this recording "
            f"or dataset to register the missing phase."
        )
        return [], message

    return matches, None


def _validate_job_prerequisites(
    tracker: ProcessingTracker, job_id: str, *, single_recording: bool, submitted_job_ids: frozenset[str]
) -> str | None:
    """Validates that a job's prerequisites either already succeeded or arrive with the same submission.

    The tracker is the authoritative source for phase completion. Files on disk may be corrupt or incomplete even if
    they exist, and the tracker only marks SUCCEEDED when processing is confirmed complete. A prerequisite that is
    submitted alongside the dependent job passes validation because the execution manager admits the dependent job only
    after that prerequisite actually succeeds.

    Args:
        tracker: The ProcessingTracker instance for the job's recording or dataset.
        job_id: The unique hexadecimal job identifier to validate.
        single_recording: Determines whether to apply single-recording or multi-recording prerequisite rules.
        submitted_job_ids: The identifiers of every job submitted against this tracker in the same call.

    Returns:
        None if all prerequisites are satisfied or pending in this submission, or an error message string describing
        the unmet prerequisite.
    """
    registry = tracker.snapshot()
    prerequisite_ids, missing_message = _resolve_prerequisite_job_ids(
        registry=registry, job_id=job_id, single_recording=single_recording
    )
    if missing_message is not None:
        return missing_message

    for prerequisite_id in prerequisite_ids:
        prerequisite_state = registry[prerequisite_id]
        if prerequisite_state.status == ProcessingStatus.SUCCEEDED or prerequisite_id in submitted_job_ids:
            continue
        return (
            f"Unable to execute job {job_id}. Its prerequisite '{prerequisite_state.job_name}' job "
            f"{prerequisite_id} has not succeeded and is not part of this submission."
        )

    return None


def _pipeline_worker(
    configuration_path: Path,
    job_id: str,
    tracker_path: Path,
    *,
    single_recording: bool = True,
    workers: int | None = None,
) -> None:
    """Executes a single pipeline job identified by its job ID.

    Calls the appropriate pipeline function in REMOTE mode, passing the job_id so the pipeline reads the job definition
    from the ProcessingTracker and updates tracker state on completion or failure. After the pipeline returns or raises,
    verifies that the tracker reached a terminal state and marks the job as failed if the pipeline terminated without
    updating the tracker.

    Notes:
        A remote invocation runs exactly one job, so the allocation the execution manager resolved for that job's
        resource class is given to every stage parameter of the pipeline. Only the parameter of the executed stage is
        read, and a combination job reads none of them because that stage takes no worker allocation.

    Args:
        configuration_path: The path to the recording or dataset configuration file.
        job_id: The unique hexadecimal job identifier registered in the ProcessingTracker.
        tracker_path: The path to the ProcessingTracker file for this job.
        single_recording: Determines whether to call the single-recording or multi-recording pipeline.
        workers: The number of parallel workers to allocate to this job. A value of None makes the pipeline apply the
            measured default for the job's stage.
    """
    try:
        if single_recording:
            run_single_recording_pipeline(
                configuration_path=configuration_path,
                job_id=job_id,
                binarization_workers=workers,
                registration_workers=workers,
                processing_workers=workers,
            )
        else:
            run_multi_recording_pipeline(
                configuration_path=configuration_path,
                job_id=job_id,
                discovery_workers=workers,
                extraction_workers=workers,
            )
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
    """Admits jobs whose prerequisites succeeded and dispatches them under their resource class concurrency caps.

    Runs as a daemon thread, polling at 1-second intervals.

    Notes:
        Every polling cycle reaps finished worker threads, scans the admission pool against a fresh snapshot of each
        tracker, and then dispatches from every resource class queue up to that class's cap. A job is admitted the
        moment its own prerequisites succeed on its own tracker, so each job follows the progress of its own recording.

        A job whose prerequisite failed is marked FAILED on the cycle that observes the failure, and a session that can
        make no further progress fails everything it still holds. Both outcomes clear the session state, so the manager
        always terminates.
    """
    global _job_execution_state

    timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

    while True:
        state = _job_execution_state
        if state is None:
            return

        with state.lock:
            _reap_completed_threads(state=state)
            admitted = _admit_ready_jobs(state=state)
            dispatched = _dispatch_admitted_jobs(state=state)

            queued = any(pending_queue for pending_queue in state.pending_queues.values())
            active = any(threads for threads in state.active_threads.values())

            if not state.admission_pool and not queued and not active:
                _job_execution_state = None
                return

            # Nothing is running, nothing is queued, and this cycle changed nothing, so the jobs still held in the
            # admission pool depend on work that this session cannot perform.
            if not queued and not active and not admitted and not dispatched:
                _fail_pending_jobs(jobs=state.admission_pool, message=_UNREACHABLE_PREREQUISITE_MESSAGE)
                state.admission_pool.clear()
                _job_execution_state = None
                return

        timer.delay(delay=1000, allow_sleep=True)


def _reap_completed_threads(state: _JobExecutionState) -> None:
    """Removes the finished worker threads from every resource class, freeing that class's concurrency.

    Args:
        state: The current job execution state, accessed under its lock.
    """
    for active_threads in state.active_threads.values():
        completed_keys = [key for key, thread in active_threads.items() if not thread.is_alive()]
        for key in completed_keys:
            active_threads.pop(key, None)


def _admit_ready_jobs(state: _JobExecutionState) -> bool:
    """Moves every admission-pool job whose prerequisites succeeded into its resource class queue.

    Notes:
        Each tracker is snapshotted once per scan and the snapshot is reused for every job that tracker owns, which
        keeps a large batch to one tracker read per recording per polling cycle. Jobs whose prerequisites failed or are
        absent from the tracker are marked FAILED here, so they leave the pool on the cycle that detects the failure.
        Each of them records the reason resolved for it, so a missing prerequisite phase is distinguishable from a
        failed one.

    Args:
        state: The current job execution state, accessed under its lock.

    Returns:
        True if at least one job left the admission pool during this scan, False otherwise.
    """
    if not state.admission_pool:
        return False

    registries: dict[Path, dict[str, JobState]] = {}
    remaining: list[_PendingJob] = []
    aborted: list[tuple[_PendingJob, str]] = []
    admitted = False

    for pending_job in state.admission_pool:
        registry = registries.get(pending_job.tracker_path)
        if registry is None:
            registry = ProcessingTracker(file_path=pending_job.tracker_path).snapshot()
            registries[pending_job.tracker_path] = registry

        decision, abort_message = _resolve_job_admission(registry=registry, pending_job=pending_job)
        if decision == _AdmissionDecisions.ADMIT:
            state.pending_queues[pending_job.resource_class.name].append(pending_job)
            admitted = True
        elif decision == _AdmissionDecisions.ABORT:
            aborted.append((pending_job, abort_message))
        else:
            remaining.append(pending_job)

    state.admission_pool = remaining

    # Records the reason resolved for each aborted job, because a job blocked by a missing prerequisite phase needs a
    # different remedy from one blocked by a failed phase.
    for aborted_job, aborted_message in aborted:
        _fail_pending_jobs(jobs=[aborted_job], message=aborted_message)

    return admitted or bool(aborted)


def _resolve_job_admission(registry: dict[str, JobState], pending_job: _PendingJob) -> tuple[_AdmissionDecisions, str]:
    """Decides whether one queued job may start, must keep waiting, or can never run.

    Args:
        registry: The point-in-time job registry of the tracker that owns the job.
        pending_job: The queued job to evaluate.

    Returns:
        A tuple of the admission decision for the job and the reason to record when that decision is ABORT. The reason
        is an empty string for every other decision.
    """
    prerequisite_ids, missing_message = _resolve_prerequisite_job_ids(
        registry=registry, job_id=pending_job.job_id, single_recording=pending_job.single_recording
    )
    if missing_message is not None:
        return _AdmissionDecisions.ABORT, missing_message

    statuses = [registry[prerequisite_id].status for prerequisite_id in prerequisite_ids]
    if any(status == ProcessingStatus.FAILED for status in statuses):
        return _AdmissionDecisions.ABORT, _PREREQUISITE_FAILURE_MESSAGE
    if all(status == ProcessingStatus.SUCCEEDED for status in statuses):
        return _AdmissionDecisions.ADMIT, ""

    return _AdmissionDecisions.WAIT, ""


def _committed_cores(state: _JobExecutionState) -> int:
    """Sums the CPU cores that the currently running jobs of every resource class hold.

    Args:
        state: The current job execution state, accessed under its lock.

    Returns:
        The number of cores the session has committed to running jobs.
    """
    return sum(len(threads) * state.class_workers[class_name] for class_name, threads in state.active_threads.items())


def _dispatch_admitted_jobs(state: _JobExecutionState) -> bool:
    """Starts worker threads for admitted jobs up to each resource class concurrency cap and the session CPU budget.

    Notes:
        A per-class concurrency cap bounds one class in isolation, and every class dispatches during the same cycle, so
        the caps alone would let the classes oversubscribe the machine between them. This function therefore also holds
        the sum of the cores committed by every running job inside the session CPU budget.

        A session whose classes all hold nothing dispatches one job regardless of the budget, so a job whose worker
        count exceeds the whole budget still runs instead of stalling the session forever.

    Args:
        state: The current job execution state, accessed under its lock.

    Returns:
        True if at least one worker thread started during this cycle, False otherwise.
    """
    dispatched = False

    for class_name, pending_queue in state.pending_queues.items():
        active_threads = state.active_threads[class_name]
        capacity = state.class_capacities[class_name]
        workers = state.class_workers[class_name]
        while len(active_threads) < capacity and pending_queue:
            committed = _committed_cores(state=state)
            if committed > 0 and committed + workers > state.cpu_budget:
                break

            pending_job = pending_queue.pop(0)
            thread = Thread(
                target=_pipeline_worker,
                kwargs={
                    "configuration_path": pending_job.configuration_path,
                    "job_id": pending_job.job_id,
                    "tracker_path": pending_job.tracker_path,
                    "single_recording": pending_job.single_recording,
                    "workers": pending_job.resolved_workers,
                },
                daemon=True,
            )
            thread.start()
            active_threads[pending_job.dispatch_key] = thread
            dispatched = True

    return dispatched


def _fail_pending_jobs(jobs: list[_PendingJob], message: str) -> None:
    """Marks every provided job as failed with the given reason recorded on its tracker.

    Args:
        jobs: The jobs that can no longer run.
        message: The error message to record for each job.
    """
    for job in jobs:
        tracker = ProcessingTracker(file_path=job.tracker_path)
        tracker.start_job(job_id=job.job_id)
        tracker.fail_job(job_id=job.job_id, error_message=message)


def _read_single_recording_tracker(tracker_path: Path, recording_path: Path) -> dict[str, object]:
    """Reads a single-recording ProcessingTracker and returns structured status information.

    Args:
        tracker_path: The path to the ProcessingTracker YAML file.
        recording_path: The path to the recording directory (for display purposes).

    Returns:
        A dictionary containing a success flag, the recording path, tracker path, per-phase job status, summary
        counts, and an overall synthesized status string.
    """
    tracker = ProcessingTracker(file_path=tracker_path)
    summary = tracker.get_summary()
    registry = tracker.snapshot()

    binarize_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.BINARIZE)
    register_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.REGISTER)
    process_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.PROCESS)
    combine_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.COMBINE)

    binarize_status: dict[str, object] = {}
    for job_id in binarize_jobs:
        job_info = registry[job_id]
        binarize_status["status"] = job_info.status.name.lower()
        if job_info.error_message:
            binarize_status["error"] = job_info.error_message

    register_status: dict[str, object] = {
        specifier: registry[job_id].status.name.lower() for job_id, (_, specifier) in register_jobs.items()
    }

    process_status: dict[str, object] = {
        specifier: registry[job_id].status.name.lower() for job_id, (_, specifier) in process_jobs.items()
    }

    combine_status: dict[str, object] = {}
    for job_id in combine_jobs:
        job_info = registry[job_id]
        combine_status["status"] = job_info.status.name.lower()
        if job_info.error_message:
            combine_status["error"] = job_info.error_message

    # Synthesizes overall status from tracker state, reporting the furthest phase the recording has reached.
    if tracker.complete:
        overall_status = "completed"
    elif tracker.encountered_error:
        overall_status = "failed"
    elif combine_jobs and any(registry[job_id].status == ProcessingStatus.RUNNING for job_id in combine_jobs):
        overall_status = "combining"
    elif process_jobs and any(
        registry[job_id].status in (ProcessingStatus.RUNNING, ProcessingStatus.SUCCEEDED) for job_id in process_jobs
    ):
        overall_status = "processing"
    elif register_jobs and any(
        registry[job_id].status in (ProcessingStatus.RUNNING, ProcessingStatus.SUCCEEDED) for job_id in register_jobs
    ):
        overall_status = "registering"
    elif binarize_jobs and any(
        registry[job_id].status in (ProcessingStatus.RUNNING, ProcessingStatus.SUCCEEDED) for job_id in binarize_jobs
    ):
        overall_status = "binarizing"
    else:
        overall_status = "scheduled"

    summary_counts: dict[str, int] = {status.name.lower(): count for status, count in summary.items()}

    return {
        "success": True,
        "recording_path": str(recording_path),
        "tracker_path": str(tracker_path),
        "status": overall_status,
        "jobs": {
            "binarize": binarize_status,
            "register": register_status,
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
    registry = tracker.snapshot()

    discover_jobs = tracker.find_jobs(job_name=MultiRecordingJobNames.DISCOVER)
    extract_jobs = tracker.find_jobs(job_name=MultiRecordingJobNames.EXTRACT)

    discover_status: dict[str, object] = {}
    for job_id in discover_jobs:
        job_info = registry[job_id]
        discover_status["status"] = job_info.status.name.lower()
        if job_info.error_message:
            discover_status["error"] = job_info.error_message

    extract_status: dict[str, object] = {
        specifier: registry[job_id].status.name.lower() for job_id, (_, specifier) in extract_jobs.items()
    }

    # Synthesizes overall status from tracker state.
    if tracker.complete:
        overall_status = "completed"
    elif tracker.encountered_error:
        overall_status = "failed"
    elif extract_jobs and any(
        registry[job_id].status in (ProcessingStatus.RUNNING, ProcessingStatus.SUCCEEDED) for job_id in extract_jobs
    ):
        overall_status = "extracting"
    elif discover_jobs and any(
        registry[job_id].status in (ProcessingStatus.RUNNING, ProcessingStatus.SUCCEEDED) for job_id in discover_jobs
    ):
        overall_status = "discovering"
    else:
        overall_status = "scheduled"

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
    """Loads and parses the runtime YAML file at the given path.

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
