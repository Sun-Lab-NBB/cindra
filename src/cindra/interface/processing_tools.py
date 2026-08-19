"""Provides MCP tools for preparing, executing, monitoring, and cancelling neural imaging pipeline jobs.

These tools give agents fine-grained control over pipeline execution: prepare builds an execution manifest without
running anything, execute dispatches selected jobs with prerequisite validation, reset selectively reverts completed
phases for re-runs, and status/cancel manage the active execution. Both single-recording (four-phase: binarize,
register, process, combine) and multi-recording (two-phase: discover, extract) pipelines are supported through a
unified execution model that admits every job as soon as that job's own prerequisites succeed.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any
from pathlib import Path
from importlib.util import find_spec

import yaml
from natsort import natsorted
from ataraxis_time import (
    TimeUnits,
    TimestampFormats,
    TimestampPrecisions,
    convert_time,
    get_timestamp,
)
from ataraxis_base_utilities import console
from ataraxis_data_structures import (
    ProcessingStatus,
    ProcessingTracker,
    delete_directory,
    index_marker_files,
    discover_marker_files,
)

from ..io import (
    is_plane_converted,
    find_data_directory,
    is_dataset_discovered,
    resolve_recording_planes,
    resolve_dataset_recordings,
    resolve_source_frame_geometry,
    resolve_multi_recording_contexts,
)
from ..layout import (
    OUTPUT_DIRECTORY_NAME,
    PLANE_SPECIFIER_PREFIX,
    DEFORMED_MASKS_FILENAME,
    CHANNEL_1_BINARY_FILENAME,
    CHANNEL_2_BINARY_FILENAME,
    COMBINED_METADATA_FILENAME,
    DETECTION_DATA_DIRECTORY_NAME,
    MULTI_RECORDING_DIRECTORY_NAME,
    MULTI_RECORDING_TRACKER_FILENAME,
    REGISTRATION_DATA_DIRECTORY_NAME,
    TRACKING_TEMPLATE_MASKS_FILENAME,
    SINGLE_RECORDING_TRACKER_FILENAME,
    MULTI_RECORDING_ARRAYS_DIRECTORY_NAME,
    MULTI_RECORDING_RUNTIME_DATA_FILENAME,
    MULTI_RECORDING_CONFIGURATION_FILENAME,
    SINGLE_RECORDING_CONFIGURATION_FILENAME,
    DetectionImages,
    RecordingArrays,
    resolve_array_name,
    resolve_channel_2_name,
    resolve_plane_specifier,
)
from ..dataclasses import MultiRecordingConfiguration, SingleRecordingConfiguration
from .mcp_instance import mcp
from ..orchestration import (
    SINGLE_RECORDING_PHASES,
    RESOURCE_CLASS_BY_JOB_NAME,
    PendingJob,
    OpenMPStatus,
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    prime_recording,
    get_execution_state,
    set_execution_state,
    resolve_session_load,
    resolve_pipeline_jobs,
    resolve_openmp_runtime,
    start_execution_session,
    cancel_execution_session,
    size_multi_recording_job,
    order_phases_by_execution,
    resolve_downstream_phases,
    size_single_recording_job,
    resolve_recording_geometry,
    validate_job_prerequisites,
    resolve_multi_recording_jobs,
    resolve_single_recording_jobs,
    load_multi_recording_configuration,
    load_single_recording_configuration,
    resolve_multi_recording_job_universe,
    resolve_single_recording_job_universe,
    estimate_multi_recording_job_memory_mb,
    estimate_single_recording_job_memory_mb,
)

if TYPE_CHECKING:
    from ataraxis_data_structures import JobState

_MINIMUM_RECORDING_COUNT: int = 2
"""The minimum number of recordings required for multi-recording processing."""

_SECONDS_PER_HOUR: float = 3600.0
"""The number of seconds in one hour, which scales an elapsed second count into the reported job throughput."""

_TIFF_HINT_SEARCH_DEPTH: int = 2
"""The number of directory levels below a rejected raw data path that are examined for the TIFF files it is missing."""


@mcp.tool()
def get_recording_status_tool(output_root: str) -> dict[str, object]:
    """Gets the processing status for a recording by reading all available ProcessingTracker files.

    Checks for both single-recording and multi-recording trackers under the recording's cindra output directory and
    returns status for all pipelines found. For single-recording, reads the tracker at
    <output_root>/cindra/single_recording_tracker.yaml and returns per-phase job status (binarize, register,
    process, combine). For multi-recording, searches under <output_root>/cindra/multi_recording/<dataset>/ for
    tracker files and returns per-dataset status (discover, extract).

    Args:
        output_root: The absolute path to the pipeline output root, which is the parent of the cindra/ folder and
            equals the output_roots entries passed to the prepare tool. The cindra/ subdirectory is resolved directly
            under it with no fallback.

    Returns:
        On success, contains the 'output_root', 'single_recording' status (per-phase jobs, summary, and synthesized
        status string), and 'multi_recording' status (per-dataset tracker status). Each section reports 'not_started'
        when no tracker exists. On failure, contains an 'error' describing the issue. Both cases include a 'success'
        flag. The synthesized single-recording status string is one of completed, failed, scheduled, binarizing,
        registering, processing, or combining. The synthesized multi-recording status string is one of completed,
        failed, scheduled, discovering, or extracting. These differ from the get_processing_jobs_status_tool summary
        vocabulary of pending, running, succeeded, and failed, which describes the same jobs (pending equals scheduled,
        and running spans the *-ing phases).
    """
    recording = Path(output_root)

    if not recording.exists():
        return {
            "success": False,
            "error": f"Unable to get recording status. Output root not found: {output_root}.",
        }

    single_tracker_path = recording / OUTPUT_DIRECTORY_NAME / SINGLE_RECORDING_TRACKER_FILENAME
    if single_tracker_path.exists():
        single_recording_status = _read_single_recording_tracker(
            tracker_path=single_tracker_path, output_root=recording
        )
    else:
        single_recording_status = {"status": "not_started"}

    multi_recording_status: dict[str, object]
    multi_recording_base = recording / OUTPUT_DIRECTORY_NAME / MULTI_RECORDING_DIRECTORY_NAME
    if multi_recording_base.exists():
        # Falls back to the tolerant scan when a subdirectory denies the strict one, so a single unreadable dataset
        # directory reports the datasets that are readable instead of failing the whole status query.
        try:
            tracker_files = discover_marker_files(
                directory=multi_recording_base, marker_name=MULTI_RECORDING_TRACKER_FILENAME
            )
        except OSError:
            tracker_files = list(multi_recording_base.rglob(MULTI_RECORDING_TRACKER_FILENAME))
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
        "output_root": str(recording),
        "single_recording": single_recording_status,
        MULTI_RECORDING_DIRECTORY_NAME: multi_recording_status,
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

    # Indexes both tracker names in a single traversal of the tree, which halves the walk a two-name search would
    # otherwise cost over a whole data root. The indexer refuses to narrow its result to the readable subset, so a
    # single unreadable directory anywhere under the root fails the whole scan. This tool surveys a root the caller
    # chose rather than a path the pipeline owns, so reporting no recording at all because one sibling directory is
    # unreadable would be the wrong answer. The denial is recorded and the tolerant scan supplies the rest.
    try:
        tracker_index = index_marker_files(
            directory=root, marker_names=(SINGLE_RECORDING_TRACKER_FILENAME, MULTI_RECORDING_TRACKER_FILENAME)
        )
        single_tracker_paths: list[Path] = list(tracker_index[SINGLE_RECORDING_TRACKER_FILENAME])
        multi_tracker_paths: list[Path] = list(tracker_index[MULTI_RECORDING_TRACKER_FILENAME])
    except OSError as error:
        permission_errors.append(
            _collapse_whitespace(text=f"Access denied during the processing tracker search: {error}")
        )
        single_tracker_paths = list(root.rglob(SINGLE_RECORDING_TRACKER_FILENAME))
        multi_tracker_paths = list(root.rglob(MULTI_RECORDING_TRACKER_FILENAME))

    # The tracker sits in the recording's cindra output directory, so a tracker file's grandparent is the output root.
    single_recordings: list[dict[str, object]] = [
        _read_single_recording_tracker(tracker_path=tracker_path, output_root=tracker_path.parent.parent)
        for tracker_path in natsorted(single_tracker_paths, key=str)
    ]

    # Reads multi-recording trackers. Extracts dataset name from parent directory.
    multi_recordings: list[dict[str, object]] = []
    for tracker_path in natsorted(multi_tracker_paths, key=str):
        dataset_name = tracker_path.parent.name
        entry = _read_multi_recording_tracker(tracker_path=tracker_path)
        entry["dataset_name"] = dataset_name
        multi_recordings.append(entry)

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
    raw_data_paths: list[str],
    configuration_path: str,
    output_roots: list[str],
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
        raw_data_paths: List of absolute paths to each recording's raw imaging data (used as file_io.data_path per
            recording). An entry may name the imaging directory itself or any parent of it, because the conversion
            locates the cindra_parameters.json file beneath the path and reads the directory holding it. The TIFF
            files must sit beside that file, since only that one directory is scanned. A path whose subtree carries
            no source file the conversion accepts is rejected here rather than at dispatch.
        configuration_path: The absolute path to the template configuration YAML file.
        output_roots: List of absolute paths to the pipeline output roots, each the parent of the cindra/ folder the
            recording's results are written under (used as file_io.output_path). Must match the length of
            raw_data_paths.

    Returns:
        On success, contains per-recording manifests in 'recordings' keyed by the raw data path. Each entry lists its
        configuration_path, tracker_path, output_root, pipeline_type, and per-phase job entries (binarize_job,
        register_jobs, process_jobs, combine_job) including job_id, name, specifier, and current status, plus
        executor_id for a tracker that already existed. The output_root is the absolute output directory and the parent
        of the cindra/ directory, where configuration_path equals <output_root>/cindra/configuration.yaml, so downstream
        verify, status, and clean tools need no re-derivation. A job_id is derived from the job name and specifier
        alone, so the same phase carries the same identifier in every recording and (tracker_path, job_id) is the only
        key that identifies a job across the batch. Keying a dictionary by job_id alone merges recordings. Also
        includes 'total_recordings' and 'total_jobs' counts, plus 'migrated_recordings' listing any recording whose
        tracker gained the missing register jobs and 'invalid_paths' listing any provided path that is not an existing
        directory. A recording whose preparation fails, such as one holding no readable TIFF file or no acquisition
        parameters file, is reported with its reason under 'invalid_recordings' and receives no manifest, while every
        other recording keeps its manifest. A recording whose existing configuration records paths other than the ones
        passed here is reported under 'path_conflicts', naming the recording, the stored value, and the passed value.
        A caller MUST read both keys rather than treat the absence of an 'error' as full preparation. On failure,
        contains an 'error' describing the issue.
    """
    if not raw_data_paths:
        return {"success": False, "error": "Unable to prepare batch. At least one raw data path is required."}

    if len(output_roots) != len(raw_data_paths):
        return {
            "success": False,
            "error": (
                f"Unable to prepare batch. The output_roots length ({len(output_roots)}) must match the "
                f"raw_data_paths length ({len(raw_data_paths)})."
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

    valid_indices: list[int] = []
    valid_paths: list[Path] = []
    invalid_paths: list[str] = []

    for index, path_string in enumerate(raw_data_paths):
        path = Path(path_string)
        if path.exists() and path.is_dir():
            valid_paths.append(path)
            valid_indices.append(index)
        else:
            invalid_paths.append(path_string)

    if not valid_paths:
        return {
            "success": False,
            "error": "Unable to prepare batch. No valid raw data paths provided.",
            "invalid_paths": invalid_paths,
        }

    resolved_output_roots: list[Path] = [Path(output_roots[index]) for index in valid_indices]

    # Reads the ignored file stems the template declares, so the raw data scan below accepts exactly the files the
    # conversion would accept. A template the loader rejects leaves the tuple empty, and every recording then reports
    # that rejection through its own configuration load.
    try:
        ignored_file_names = SingleRecordingConfiguration.from_yaml(file_path=template_path).file_io.ignored_file_names
    except Exception:
        ignored_file_names = ()

    recordings_manifest: dict[str, dict[str, object]] = {}
    migrated_recordings: list[str] = []
    invalid_recordings: list[str] = []
    path_conflicts: list[dict[str, str]] = []
    total_jobs = 0

    for data_path, output_root in zip(valid_paths, resolved_output_roots, strict=True):
        recording_key = str(data_path)
        cindra_root = output_root / OUTPUT_DIRECTORY_NAME
        tracker_path = cindra_root / SINGLE_RECORDING_TRACKER_FILENAME

        if tracker_path.exists():
            # Idempotent path: tracker already exists, returns current state without reinitializing.
            tracker = ProcessingTracker(file_path=tracker_path)
            registry = tracker.snapshot()
            configuration_file_path = cindra_root / SINGLE_RECORDING_CONFIGURATION_FILENAME

            binarize_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.BINARIZE)
            register_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.REGISTER)
            process_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.PROCESS)
            combine_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.COMBINE)

            # A recording whose binarization has already succeeded never reads its raw data again, so the scan of
            # the raw imaging directory only excludes a recording whose conversion is still outstanding.
            if any(registry[job_id].status != ProcessingStatus.SUCCEEDED for job_id in binarize_jobs):
                raw_data_failure = _resolve_raw_data_failure(
                    raw_data_path=data_path, ignored_file_names=ignored_file_names
                )
                if raw_data_failure is not None:
                    invalid_recordings.append(f"{recording_key}: {raw_data_failure}")
                    continue

            # Reports the paths the existing configuration records when they disagree with the ones the caller passed.
            # The tracker is never reinitialized, so the recording keeps running against the stored paths until its
            # output directory is removed.
            path_conflicts.extend(
                _resolve_single_recording_path_conflicts(
                    recording_key=recording_key,
                    configuration_path=configuration_file_path,
                    output_root=output_root,
                    data_path=data_path,
                )
            )

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
                "output_root": str(output_root),
                "pipeline_type": "single-recording",
                "binarize_job": binarize_entry,
                "register_jobs": register_entries,
                "process_jobs": process_entries,
                "combine_job": combine_entry,
            }
        else:
            # A recording whose subtree holds nothing the conversion reads cannot run any stage, so it is rejected
            # here rather than at dispatch, and no tracker is written for it. The gate resolves the imaging directory
            # the way the conversion does, so it accepts every path the conversion would.
            raw_data_failure = _resolve_raw_data_failure(raw_data_path=data_path, ignored_file_names=ignored_file_names)
            if raw_data_failure is not None:
                invalid_recordings.append(f"{recording_key}: {raw_data_failure}")
                continue

            # New recording: creates per-recording config, resolves planes, and initializes tracker. The bootstrap
            # reads the recording's acquisition parameters, so a recording that carries none, or whose configuration
            # the pipeline rejects, is reported through 'invalid_recordings' instead of aborting the batch and
            # discarding the manifest of every recording prepared before it.
            try:
                recording_configuration = SingleRecordingConfiguration.from_yaml(file_path=template_path)
                recording_configuration.file_io.data_path = data_path
                recording_configuration.file_io.output_path = output_root
                recording_configuration.runtime.display_progress_bars = False

                cindra_root.mkdir(parents=True, exist_ok=True)
                recording_configuration_path = cindra_root / SINGLE_RECORDING_CONFIGURATION_FILENAME

                # Saves the per-recording configuration. The execute tool passes the resolved worker allocation to
                # each job as a dispatch argument, so this one file serves every job dispatched against it.
                recording_configuration.save(file_path=recording_configuration_path)

                # Writes the shared bootstrap every later job reads and reports the planes the recording holds.
                plane_count = prime_recording(configuration_path=recording_configuration_path).plane_count

                # Builds the recording's job universe from the exported phase model, which orders the phases and
                # expands the per-plane ones.
                jobs: list[tuple[str, str]] = resolve_single_recording_jobs(plane_count=plane_count)

                tracker = ProcessingTracker(file_path=tracker_path)
                identifiers = _resolve_job_identifiers(tracker=tracker, jobs=jobs)

                binarize_entry = _manifest_entry(
                    identifiers=identifiers, job_name=SingleRecordingJobNames.BINARIZE, specifier=""
                )

                register_entries = [
                    _manifest_entry(
                        identifiers=identifiers,
                        job_name=SingleRecordingJobNames.REGISTER,
                        specifier=resolve_plane_specifier(plane_index=plane_index),
                    )
                    for plane_index in range(plane_count)
                ]

                process_entries = [
                    _manifest_entry(
                        identifiers=identifiers,
                        job_name=SingleRecordingJobNames.PROCESS,
                        specifier=resolve_plane_specifier(plane_index=plane_index),
                    )
                    for plane_index in range(plane_count)
                ]

                combine_entry = _manifest_entry(
                    identifiers=identifiers, job_name=SingleRecordingJobNames.COMBINE, specifier=""
                )
            except Exception as error:
                invalid_recordings.append(_collapse_whitespace(text=f"{recording_key}: {error}"))
                continue

            total_jobs += len(jobs)

            recordings_manifest[recording_key] = {
                "configuration_path": str(recording_configuration_path),
                "tracker_path": str(tracker_path),
                "output_root": str(output_root),
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

    if invalid_recordings:
        result["invalid_recordings"] = invalid_recordings

    if path_conflicts:
        result["path_conflicts"] = path_conflicts

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
        dataset_configurations: List of dataset configurations, each a dictionary with three keys. 'configuration_path'
            is the absolute path to the multi-recording YAML configuration. 'output_roots' is the list of absolute paths
            to the pipeline output roots of the COMPLETED single-recording runs the dataset spans, each the parent of a
            cindra/ folder. 'dataset_name' is a unique name for this dataset. At least 2 output roots per dataset are
            required. These are output roots rather than raw imaging directories, because the dataset tracks ROIs across
            finished single-recording results.

    Returns:
        On success, contains per-dataset manifests in 'datasets' keyed by the lowercased dataset name. Each entry lists
        its configuration_path, tracker_path, dataset_name, pipeline_type, and per-phase job entries (discover_job,
        extract_jobs) including job_id, name, specifier, and current status, plus executor_id for a tracker that already
        existed. The dataset_name field is the resolved lowercased dataset name. To verify a dataset, call
        verify_multi_recording_output_tool with the dataset_name plus any output_root belonging to the dataset (one of
        the input output_roots, whose cindra/ subdirectory is resolved automatically). A job_id is derived from the job
        name and specifier alone, so the same phase carries the same identifier in every dataset and (tracker_path,
        job_id) is the only key that identifies a job across the batch. Keying a dictionary by job_id alone merges
        datasets. Also includes 'total_datasets' and 'total_jobs' counts, plus 'invalid_configurations' listing every
        rejected dataset entry with its reason. A dataset whose existing configuration records output roots other than
        the ones passed here is reported under 'path_conflicts', naming the dataset, the stored value, and the passed
        value. A caller MUST read that key rather than treat the returned manifest as proof the passed paths were
        adopted. On failure, contains an 'error' describing the issue.
    """
    if not dataset_configurations:
        return {
            "success": False,
            "error": "Unable to prepare multi-recording batch. At least one dataset configuration is required.",
        }

    valid_datasets: list[tuple[str, Path, list[Path]]] = []
    invalid_configurations: list[str] = []

    for dataset_configuration in dataset_configurations:
        required_keys = {"configuration_path", "output_roots", "dataset_name"}
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

        raw_output_roots = dataset_configuration["output_roots"]
        if not isinstance(raw_output_roots, list):
            invalid_configurations.append(f"output_roots must be a list: {dataset_configuration_path}")
            continue
        dataset_output_roots = [Path(str(path)) for path in raw_output_roots]
        if len(dataset_output_roots) < _MINIMUM_RECORDING_COUNT:
            invalid_configurations.append(f"Need at least 2 recordings: {dataset_configuration_path}")
            continue

        invalid_recordings = [str(path) for path in dataset_output_roots if not path.exists() or not path.is_dir()]
        if invalid_recordings:
            invalid_configurations.append(f"Invalid recordings for {dataset_configuration_path}: {invalid_recordings}")
            continue

        try:
            MultiRecordingConfiguration.from_yaml(file_path=dataset_configuration_path)
        except Exception as error:
            invalid_configurations.append(
                _collapse_whitespace(text=f"Unable to load configuration {dataset_configuration_path}: {error}")
            )
            continue

        dataset_key = dataset_name.lower()
        valid_datasets.append((dataset_key, dataset_configuration_path, dataset_output_roots))

    if not valid_datasets:
        return {
            "success": False,
            "error": "Unable to prepare multi-recording batch. No valid dataset configurations provided.",
            "invalid_configurations": invalid_configurations,
        }

    datasets_manifest: dict[str, dict[str, object]] = {}
    path_conflicts: list[dict[str, str]] = []
    total_jobs = 0

    for dataset_key, dataset_configuration_path, dataset_output_roots in valid_datasets:
        configuration = MultiRecordingConfiguration.from_yaml(file_path=dataset_configuration_path)
        configuration.recording_io.dataset_name = dataset_key
        resolved_output_roots = tuple(natsorted(dataset_output_roots))
        configuration.recording_io.recording_directories = resolved_output_roots
        configuration.runtime.display_progress_bars = False

        contexts = resolve_multi_recording_contexts(configuration=configuration)
        recording_ids = [context.runtime.io.recording_id for context in contexts]
        dataset_directory = contexts[0].runtime.output_path

        if dataset_directory is None:
            invalid_configurations.append(f"Unable to resolve output path for dataset '{dataset_key}'.")
            continue

        tracker_path = dataset_directory / MULTI_RECORDING_TRACKER_FILENAME
        configuration_file_path = dataset_directory / MULTI_RECORDING_CONFIGURATION_FILENAME

        if tracker_path.exists():
            # Idempotent path: tracker already exists, returns current state without reinitializing.
            tracker = ProcessingTracker(file_path=tracker_path)
            registry = tracker.snapshot()

            # Reports the output roots the stored dataset configuration records when they disagree with the ones the
            # caller passed. The tracker is never reinitialized, so the dataset keeps running against the stored roots
            # until its output directory is removed.
            path_conflicts.extend(
                _resolve_multi_recording_path_conflicts(
                    dataset_key=dataset_key,
                    configuration_path=configuration_file_path,
                    output_roots=resolved_output_roots,
                )
            )

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
            identifiers = _resolve_job_identifiers(tracker=tracker, jobs=jobs)
            total_jobs += len(jobs)

            discover_entry = _manifest_entry(
                identifiers=identifiers, job_name=MultiRecordingJobNames.DISCOVER, specifier=""
            )

            extract_entries = [
                _manifest_entry(
                    identifiers=identifiers, job_name=MultiRecordingJobNames.EXTRACT, specifier=recording_id
                )
                for recording_id in recording_ids
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

    if path_conflicts:
        result["path_conflicts"] = path_conflicts

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
        dependency expansion, and per-job status showing updated states. A 'warnings' list is present when a reset
        phase is governed by a repeat flag that is false while that phase's output already exists on disk, because the
        stage then returns immediately and records success without redoing its work. Each warning names the dotted
        configuration flag to set and set_config_values_tool as the way to set it, and a caller MUST act on it before
        dispatching the reset phase. On failure, contains an 'error' describing the issue.
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
    single_recording = pipeline_type == "single-recording"
    phases = order_phases_by_execution(
        phase_names=resolve_downstream_phases(phase_names=phases, single_recording=single_recording),
        single_recording=single_recording,
    )

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

    # A stage whose output already exists returns immediately unless its repeat flag is set, so resetting its tracker
    # entry alone produces a job that reports success without redoing any work.
    warnings = _resolve_repeat_flag_warnings(tracker_path=path, phase_names=phases, single_recording=single_recording)

    result: dict[str, object] = {
        "success": True,
        "reset": True,
        "tracker_path": tracker_path,
        "requested_phases": requested_phases,
        "effective_phases": phases,
        "jobs": updated_jobs,
    }

    if warnings:
        result["warnings"] = warnings

    return result


@mcp.tool()
def clean_processing_output_tool(
    output_root: str,
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
        output_root: The absolute path to the pipeline output root, which is the parent of the cindra/ folder and
            equals the output_roots entries passed to the prepare tool. The cindra/ subdirectory is resolved directly
            under it with no fallback.
        phases: List of phase names to clean. For single-recording: 'binarization', 'registration', 'processing',
            'combination'. For multi-recording: 'discovery', 'extraction'. Downstream phases are automatically
            included.
        pipeline_type: The pipeline type, either 'single-recording' or 'multi-recording'.
        dataset: The multi-recording dataset name. Required when pipeline_type is 'multi-recording'. It must be the
            resolved lowercased dataset directory name created by the prepare tool, located at
            cindra/multi_recording/<dataset>, and the match is case-sensitive.

    Returns:
        On success, contains a 'cleaned' flag, the 'output_root', 'deleted_files', 'deleted_dirs', 'total_deleted',
        and the 'requested_phases' and 'effective_phases' after dependency expansion, plus 'errors' when a deletion
        failed. On failure, contains an 'error' describing the issue. Both cases include a 'success' flag.
    """
    recording = Path(output_root)

    if not recording.exists():
        return {
            "success": False,
            "error": f"Unable to clean processing output. Output root not found: {output_root}.",
        }

    if pipeline_type not in ("single-recording", "multi-recording"):
        return {
            "success": False,
            "error": (
                f"Unable to clean processing output. Invalid pipeline_type '{pipeline_type}'. "
                f"Must be 'single-recording' or 'multi-recording'."
            ),
        }

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

    requested_phases = list(phases)
    single_recording = pipeline_type == "single-recording"
    effective_phases = order_phases_by_execution(
        phase_names=resolve_downstream_phases(phase_names=phases, single_recording=single_recording),
        single_recording=single_recording,
    )

    deleted_files: list[str] = []
    deleted_dirs: list[str] = []
    errors: list[str] = []

    if pipeline_type == "single-recording":
        cindra_root = recording / OUTPUT_DIRECTORY_NAME
        if not cindra_root.exists():
            return {
                "success": False,
                "error": f"Unable to clean processing output. No cindra directory found at: {output_root}.",
            }

        effective_set = set(effective_phases)

        # Unlinks the completion marker ahead of every array it vouches for, inverting the order the combination
        # stage writes them in. An interrupted clean then leaves an unmarked partial output, which the discovery and
        # inventory paths already treat as unfinished, instead of a marker standing over deleted data.
        if SingleRecordingJobNames.COMBINE in effective_set:
            _delete_file(path=cindra_root / COMBINED_METADATA_FILENAME, deleted=deleted_files, errors=errors)

        # Cleans per-plane files, partitioning the shared detection_data directory by the phase that owns each array.
        plane_directories = natsorted(
            entry for entry in cindra_root.iterdir() if entry.is_dir() and entry.name.startswith(PLANE_SPECIFIER_PREFIX)
        )
        for plane_directory in plane_directories:
            plane_detection_directory = plane_directory / DETECTION_DATA_DIRECTORY_NAME

            if SingleRecordingJobNames.BINARIZE in effective_set:
                for name in (CHANNEL_1_BINARY_FILENAME, CHANNEL_2_BINARY_FILENAME):
                    _delete_file(path=plane_directory / name, deleted=deleted_files, errors=errors)
                # The mean images are created by binarization and later rewritten by both registration and processing,
                # so binarization, the phase that creates them, owns their removal. Deleting them under a downstream
                # phase would discard the output of the phases between it and binarization.
                for name in (
                    DetectionImages.MEAN_IMAGE,
                    resolve_array_name(array=DetectionImages.MEAN_IMAGE, second_channel=True),
                ):
                    _delete_file(path=plane_detection_directory / name, deleted=deleted_files, errors=errors)

            if SingleRecordingJobNames.REGISTER in effective_set:
                # The registration directory holds bad_frames.npy, which detection reads, so it is removed only when
                # the registration phase itself is cleaned.
                _delete_directory(
                    path=plane_directory / REGISTRATION_DATA_DIRECTORY_NAME, deleted=deleted_dirs, errors=errors
                )

            if SingleRecordingJobNames.PROCESS in effective_set:
                for name in (
                    DetectionImages.ENHANCED_MEAN_IMAGE,
                    DetectionImages.MAXIMUM_PROJECTION,
                    DetectionImages.CORRELATION_MAP,
                    resolve_array_name(array=DetectionImages.ENHANCED_MEAN_IMAGE, second_channel=True),
                    resolve_array_name(array=DetectionImages.MAXIMUM_PROJECTION, second_channel=True),
                    resolve_array_name(array=DetectionImages.CORRELATION_MAP, second_channel=True),
                ):
                    _delete_file(path=plane_detection_directory / name, deleted=deleted_files, errors=errors)
                for name in (
                    RecordingArrays.ROI_MASKS,
                    resolve_array_name(array=RecordingArrays.ROI_MASKS, second_channel=True),
                    RecordingArrays.ROI_STATISTICS,
                    resolve_array_name(array=RecordingArrays.ROI_STATISTICS, second_channel=True),
                    RecordingArrays.CELL_FLUORESCENCE,
                    RecordingArrays.NEUROPIL_FLUORESCENCE,
                    RecordingArrays.SUBTRACTED_FLUORESCENCE,
                    RecordingArrays.SPIKES,
                    RecordingArrays.CELL_CLASSIFICATION,
                    resolve_array_name(array=RecordingArrays.CELL_FLUORESCENCE, second_channel=True),
                    resolve_array_name(array=RecordingArrays.NEUROPIL_FLUORESCENCE, second_channel=True),
                    resolve_array_name(array=RecordingArrays.SUBTRACTED_FLUORESCENCE, second_channel=True),
                    resolve_array_name(array=RecordingArrays.SPIKES, second_channel=True),
                    resolve_array_name(array=RecordingArrays.CELL_CLASSIFICATION, second_channel=True),
                    RecordingArrays.CELL_COLOCALIZATION,
                    RecordingArrays.CORRECTED_STRUCTURAL_MEAN_IMAGE,
                ):
                    _delete_file(path=plane_directory / name, deleted=deleted_files, errors=errors)

        if SingleRecordingJobNames.COMBINE in effective_set:
            _delete_directory(path=cindra_root / DETECTION_DATA_DIRECTORY_NAME, deleted=deleted_dirs, errors=errors)
            for name in (
                RecordingArrays.ROI_MASKS,
                resolve_array_name(array=RecordingArrays.ROI_MASKS, second_channel=True),
                RecordingArrays.ROI_STATISTICS,
                resolve_array_name(array=RecordingArrays.ROI_STATISTICS, second_channel=True),
                RecordingArrays.CELL_FLUORESCENCE,
                RecordingArrays.NEUROPIL_FLUORESCENCE,
                RecordingArrays.SUBTRACTED_FLUORESCENCE,
                RecordingArrays.SPIKES,
                RecordingArrays.CELL_CLASSIFICATION,
                resolve_array_name(array=RecordingArrays.CELL_FLUORESCENCE, second_channel=True),
                resolve_array_name(array=RecordingArrays.NEUROPIL_FLUORESCENCE, second_channel=True),
                resolve_array_name(array=RecordingArrays.SUBTRACTED_FLUORESCENCE, second_channel=True),
                resolve_array_name(array=RecordingArrays.SPIKES, second_channel=True),
                resolve_array_name(array=RecordingArrays.CELL_CLASSIFICATION, second_channel=True),
                RecordingArrays.CELL_COLOCALIZATION,
                RecordingArrays.CORRECTED_STRUCTURAL_MEAN_IMAGE,
            ):
                _delete_file(path=cindra_root / name, deleted=deleted_files, errors=errors)

    else:
        # Multi-recording cleanup requires the dataset parameter.
        if not dataset:
            return {
                "success": False,
                "error": "Unable to clean processing output. The 'dataset' parameter is required for multi-recording.",
            }

        cindra_root = recording / OUTPUT_DIRECTORY_NAME
        dataset_path = cindra_root / MULTI_RECORDING_DIRECTORY_NAME / dataset
        if not dataset_path.exists():
            return {
                "success": False,
                "error": f"Unable to clean processing output. Dataset directory not found: {dataset_path}.",
            }

        runtime = _load_runtime_yaml(path=dataset_path / MULTI_RECORDING_RUNTIME_DATA_FILENAME)
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
                # Unlinks the discovery completion marker ahead of every array it vouches for, so an interruption
                # leaves an unmarked partial output rather than a marker standing over deleted data.
                for name in (
                    TRACKING_TEMPLATE_MASKS_FILENAME,
                    resolve_channel_2_name(name=TRACKING_TEMPLATE_MASKS_FILENAME),
                ):
                    _delete_file(path=output_path / name, deleted=deleted_files, errors=errors)

                _delete_directory(
                    path=output_path / MULTI_RECORDING_ARRAYS_DIRECTORY_NAME, deleted=deleted_dirs, errors=errors
                )
                for name in (
                    DEFORMED_MASKS_FILENAME,
                    resolve_channel_2_name(name=DEFORMED_MASKS_FILENAME),
                    # Backward-projected per-recording mask and statistics files are produced by the final
                    # discovery step (project_templates_to_recordings), not by extraction, and deleting them under
                    # EXTRACT strands the pipeline because extraction consumes them as inputs.
                    RecordingArrays.ROI_MASKS,
                    resolve_array_name(array=RecordingArrays.ROI_MASKS, second_channel=True),
                    RecordingArrays.ROI_STATISTICS,
                    resolve_array_name(array=RecordingArrays.ROI_STATISTICS, second_channel=True),
                ):
                    _delete_file(path=output_path / name, deleted=deleted_files, errors=errors)

            if MultiRecordingJobNames.EXTRACT in effective_set:
                for name in (
                    RecordingArrays.CELL_FLUORESCENCE,
                    RecordingArrays.NEUROPIL_FLUORESCENCE,
                    RecordingArrays.SUBTRACTED_FLUORESCENCE,
                    RecordingArrays.SPIKES,
                    resolve_array_name(array=RecordingArrays.CELL_FLUORESCENCE, second_channel=True),
                    resolve_array_name(array=RecordingArrays.NEUROPIL_FLUORESCENCE, second_channel=True),
                    resolve_array_name(array=RecordingArrays.SUBTRACTED_FLUORESCENCE, second_channel=True),
                    resolve_array_name(array=RecordingArrays.SPIKES, second_channel=True),
                    RecordingArrays.CELL_COLOCALIZATION,
                ):
                    _delete_file(path=output_path / name, deleted=deleted_files, errors=errors)

    result: dict[str, object] = {
        "success": True,
        "cleaned": True,
        "output_root": output_root,
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
    workers_per_job: int | None = None,
    max_parallel_jobs: int | None = None,
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

        Each job runs under the resource class of its phase. Binarization holds 3 cores per job at a fixed concurrency
        of 4. Combination holds 1 core per job, because it merges result files serially, and its concurrency is bounded
        by the CPU budget alone. Registration holds 4 cores per job and processing holds 10, each with a concurrency
        bounded by the CPU budget. The session memory budget bounds dispatch for every class alike rather than the
        concurrency cap of any one of them. The binarization class alone ignores both parameters below.

        Every class dispatches during the same cycle, so the dispatcher additionally holds the sum of the cores
        committed by the running jobs of every class inside the session CPU budget reported as 'cpu_budget'. That
        budget is the machine's core count minus the cores reserved for the system.

    Args:
        jobs: List of job descriptors, each a dictionary with 'configuration_path' (absolute path to the pipeline
            configuration file), 'tracker_path' (absolute path to the ProcessingTracker file), 'job_id' (the
            hexadecimal job identifier from the prepare manifest), and 'pipeline_type' ('single-recording' or
            'multi-recording').
        workers_per_job: CPU cores per job, overriding the measured default of every class that carries no hard
            concurrency ceiling. Leave as None to accept the measured defaults, which are 3 cores for binarization, 4
            for registration, 10 for processing, 1 for combination, 2 for multi-recording discovery, and 16 for
            multi-recording extraction. Set to -1 to give every job the whole session core budget. The override is a
            single scalar applied to every non-fixed class alike.
        max_parallel_jobs: Maximum concurrent jobs per resource class, overriding the derived concurrency cap of every
            non-fixed resource class. Leave as None to accept the derived caps, or set to -1 to lift them so that only
            the job count bounds concurrency.

    Returns:
        Always contains a 'success' flag indicating the tool ran. On a started session, also contains a 'started' flag,
        'total_jobs' dispatched, and the session 'cpu_budget' and 'memory_budget_mb' that bound the classes in
        aggregate. A started session further reports a 'resource_classes' mapping with the resolved workers_per_job,
        max_parallel_jobs, and job_count of every class present in the session, and 'invalid_jobs' listing any jobs that
        failed validation with reasons. On failure, contains success:False and an 'error' describing the issue.
    """
    if not jobs:
        return {"success": False, "error": "Unable to execute jobs. At least one job descriptor is required."}

    active_session_error = _check_active_session(action="execute jobs")
    if active_session_error is not None:
        return active_session_error

    required_keys = {"configuration_path", "tracker_path", "job_id", "pipeline_type"}
    candidate_jobs: list[PendingJob] = []
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

        tracker = ProcessingTracker(file_path=tracker_file)
        try:
            job_info = tracker.get_job_info(job_id=job_id)
        except Exception:
            invalid_jobs.append({"job_id": job_id, "reason": f"Job ID not found in tracker: {tracker_file}"})
            continue

        resource_class = RESOURCE_CLASS_BY_JOB_NAME.get(job_info.job_name)
        if resource_class is None:
            invalid_jobs.append({"job_id": job_id, "reason": f"Unrecognized pipeline phase: {job_info.job_name}"})
            continue

        single_recording = pipeline_type == "single-recording"

        # Sizing the job reads its configuration, which the descriptor names and which the pipeline rejects when it
        # carries no output path, so a descriptor naming the wrong file joins the invalid list rather than aborting
        # the whole submission.
        try:
            memory_megabytes = _estimate_pending_job_memory(
                configuration_path=configuration_file,
                job_name=job_info.job_name,
                specifier=job_info.specifier,
                single=single_recording,
            )
        except Exception as error:
            invalid_jobs.append(
                {
                    "job_id": job_id,
                    "reason": _collapse_whitespace(text=f"Unable to size the job from its configuration: {error}"),
                }
            )
            continue

        candidate_jobs.append(
            PendingJob(
                configuration_path=configuration_file,
                tracker_path=tracker_file,
                job_id=job_id,
                single_recording=single_recording,
                resource_class=resource_class,
                memory_megabytes=memory_megabytes,
            )
        )
        submitted_by_tracker.setdefault(str(tracker_file), set()).add(job_id)

    # Validates prerequisites once the full submission is known, so a job whose prerequisite is submitted alongside it
    # is accepted and admitted later, when that prerequisite actually succeeds.
    all_jobs_map: dict[tuple[str, str], PendingJob] = {}
    for candidate_job in candidate_jobs:
        prerequisite_error = validate_job_prerequisites(
            registry=ProcessingTracker(file_path=candidate_job.tracker_path).snapshot(),
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

    return _start_session(
        all_jobs=all_jobs_map,
        workers_per_job=workers_per_job,
        max_parallel_jobs=max_parallel_jobs,
        extra_result_fields={"invalid_jobs": invalid_jobs} if invalid_jobs else {},
    )


@mcp.tool()
def get_processing_jobs_status_tool(*, summary_only: bool = False) -> dict[str, object]:
    """Returns the current status of the active job execution session.

    Reads ProcessingTracker files from disk for each job in the execution session to report per-job progress. Per-job
    status comes from the on-disk tracker files rather than in-memory state, while the session-level 'active',
    'awaiting_prerequisites', and 'resource_classes' fields come from the in-memory execution state.

    Args:
        summary_only: Determines whether the response omits the per-job 'jobs' list and carries the session fields and
            the summary counts alone. Poll a large batch with this enabled, because the full list grows with the job
            count while the counts it summarizes do not.

    Returns:
        Always contains a 'success' flag indicating the tool ran. On an active session, also contains an 'active' flag,
        per-job status entries in 'jobs', a 'summary' with counts for pending, running, succeeded, and failed jobs, and
        an 'awaiting_prerequisites' count of jobs still in the admission pool. The 'jobs' list is absent when
        summary_only is enabled. An active session further reports a 'resource_classes' mapping with the resolved
        'workers_per_job' and 'max_parallel_jobs' of every class in the session, together with its 'pending' job count
        and the list of 'active' dispatch keys. A job_id identifies a job only within its own tracker, because it is
        derived from the job name and specifier alone. Each 'jobs' entry therefore carries the 'tracker_path' its
        'job_id' belongs to, and only that pair identifies a job across recordings. A dictionary keyed by job_id alone
        merges recordings. The 'active' flag reflects manager-thread liveness, not whether jobs ever ran. The execution
        manager clears session state once the session drains, so afterwards this tool reports active:False with empty
        'jobs' and a zero 'summary' plus a 'note'. Final per-job outcomes must then be re-read via
        get_recording_status_tool, get_batch_status_overview_tool, or verify_*_output_tool.
    """
    # Binds the session to a local name, because the execution manager clears the module-level reference the moment
    # the session drains and this tool must keep reporting on the session it started with.
    state = get_execution_state()
    if state is None:
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

    with state.lock:
        awaiting_prerequisites = len(state.admission_pool)
        class_status: dict[str, object] = {
            class_name: {
                "workers_per_job": state.class_workers.get(class_name, 0),
                "max_parallel_jobs": state.class_capacities.get(class_name, 0),
                "pending": len(state.pending_queues.get(class_name, [])),
                "active": list(state.active_futures.get(class_name, {}).keys()),
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

        if summary_only:
            continue

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
            job_entry["error"] = _collapse_whitespace(text=job_info.error_message)

        jobs_status.append(job_entry)

    manager_alive = state.manager_thread is not None and state.manager_thread.is_alive()

    result: dict[str, object] = {
        "success": True,
        "active": manager_alive,
        "awaiting_prerequisites": awaiting_prerequisites,
        "resource_classes": class_status,
        "summary": summary_counts,
    }

    if not summary_only:
        result["jobs"] = jobs_status

    return result


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
    # Binds the session to a local name, because the execution manager clears the module-level reference the moment
    # the session drains and this tool must keep reporting on the session it started with.
    state = get_execution_state()
    if state is None:
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
        # Scales the rate rather than the elapsed time, because the unit converter rounds every result to three
        # decimals, which quantizes any sub-hour denominator and pins one below 1.8 seconds to zero.
        session["throughput_jobs_per_hour"] = round(completed_count * _SECONDS_PER_HOUR / total_elapsed, ndigits=2)

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
    state. Already-dispatched jobs keep running in their worker processes, because cancellation only empties the queues
    and never cancels a running job. The agent should therefore poll get_recording_status_tool on the affected
    recordings or datasets until previously-RUNNING jobs leave RUNNING before starting a new session, to avoid colliding
    with still-running cancelled jobs. Calling cancel when no session is active is a safe no-op.

    Returns:
        Always contains a 'success' flag indicating the tool ran. On an active session, also contains a 'canceled'
        flag, a 'message' describing the outcome, and a 'final_state' with counts for succeeded_jobs, failed_jobs, and
        active_jobs_at_cancel. With no active session, contains canceled:False plus a 'note' stating that no session is
        active and that final per-job outcomes can be read via get_recording_status_tool or
        get_batch_status_overview_tool.
    """
    # Binds the session to a local name, because the execution manager can clear the module-level reference while this
    # tool reads the final tracker state.
    state = get_execution_state()
    if state is None:
        return {
            "success": True,
            "canceled": False,
            "message": "No execution session is active.",
            "note": (
                "No execution session is active. Final per-job outcomes can be read via get_recording_status_tool "
                "or get_batch_status_overview_tool."
            ),
        }

    _canceled_count, active_count = cancel_execution_session()

    total_succeeded = 0
    total_failed = 0
    seen_trackers: set[Path] = set()
    for pending_job in state.all_jobs.values():
        if pending_job.tracker_path in seen_trackers:
            continue
        seen_trackers.add(pending_job.tracker_path)
        counts = ProcessingTracker(file_path=pending_job.tracker_path).summarize()["summary"]
        total_succeeded += counts["succeeded"]
        total_failed += counts["failed"]

    final_state = {
        "succeeded_jobs": total_succeeded,
        "failed_jobs": total_failed,
        "active_jobs_at_cancel": active_count,
    }

    set_execution_state(state=None)

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
    raw_data_paths: list[str] | None = None,
    configuration_path: str | None = None,
    output_roots: list[str] | None = None,
    dataset_configurations: list[dict[str, object]] | None = None,
    workers_per_job: int | None = None,
    max_parallel_jobs: int | None = None,
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
        raw_data_paths: List of absolute paths to each recording's raw imaging data. An entry may name the imaging
            directory itself or any parent of it, because the conversion locates the cindra_parameters.json file
            beneath the path and reads the directory holding it. The TIFF files must sit beside that file, since only
            that one directory is scanned. A path whose subtree carries no source file the conversion accepts is
            rejected during preparation. Required for single-recording pipelines.
        configuration_path: Absolute path to the template configuration file. Required for single-recording pipelines.
        output_roots: List of absolute paths to the pipeline output roots, each the parent of the cindra/ folder a
            recording's results are written under. Required for single-recording pipelines and must match the length
            of raw_data_paths.
        dataset_configurations: List of dataset configuration dictionaries. Required for multi-recording pipelines.
            Each must contain 'configuration_path', 'output_roots' (the output roots of the completed
            single-recording runs the dataset spans), and 'dataset_name'.
        workers_per_job: CPU cores per job, overriding the measured default of every class that carries no hard
            concurrency ceiling. Leave as None to accept the measured defaults of 3 cores for binarization, 1 for
            combination, 4 for registration, 10 for processing, 2 for multi-recording discovery, and 16 for
            multi-recording extraction. Set to -1 to give every job the whole session core budget.
        max_parallel_jobs: Maximum concurrent jobs per resource class, overriding the derived concurrency cap of every
            non-fixed resource class. Leave as None to accept the derived caps, or set to -1 to lift them so that only
            the job count bounds concurrency.

    Returns:
        Always contains a 'success' flag indicating the tool ran, and callers MUST also check the 'started' flag rather
        than 'success' alone. When jobs are dispatched, contains started:True, 'total_jobs', 'phase_count', per-phase
        'phases' with job counts and IDs, the session 'cpu_budget' and 'memory_budget_mb' that bound the classes in
        aggregate, and a 'resource_classes' mapping with the resolved allocation of every class in the session. When all
        phases are already complete, returns {success:True, started:False, message:"All pipeline phases are already
        completed.", pipeline_type:<the requested type>, total_jobs:0, phase_count:0, phases:[]} plus a 'next_step'
        string. Every outcome carries 'pipeline_type'. The preparation step produces the rejection lists
        'invalid_paths', 'invalid_recordings', 'invalid_configurations', and 'path_conflicts', and this step's own
        sizing pass produces 'unsizable_recordings' and 'unsizable_datasets'. Each of those lists is forwarded when
        non-empty, and together they name every recording or dataset the session omits or runs against paths other than
        the ones passed here. 'migrated_recordings' is forwarded on the same terms, naming every recording whose tracker
        gained the missing per-plane register jobs. A batch whose preparation accepted no input at all returns
        success:False alongside those lists, because it holds no phase to report as complete. On failure, contains
        success:False and an 'error' describing the issue. Cascade-aborted downstream jobs are recorded in their
        trackers as FAILED with the exact message "Unable to execute job. A preceding pipeline phase failed.",
        distinguishing them from genuine per-job failures. A job aborted because its prerequisite phase is absent from
        the tracker instead records a message naming that phase and asking for the prepare tool to be re-run, because no
        phase failed in that case.
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

    manifest: dict[str, object]
    if pipeline_type == "single-recording":
        if not raw_data_paths:
            return {
                "success": False,
                "error": "Unable to execute full pipeline. 'raw_data_paths' is required for single-recording.",
            }
        if not configuration_path:
            return {
                "success": False,
                "error": "Unable to execute full pipeline. 'configuration_path' is required for single-recording.",
            }
        if not output_roots:
            return {
                "success": False,
                "error": "Unable to execute full pipeline. 'output_roots' is required for single-recording.",
            }

        manifest = prepare_single_recording_batch_tool(
            raw_data_paths=raw_data_paths,
            configuration_path=configuration_path,
            output_roots=output_roots,
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

    # Carries the lists the preparation step produces into every outcome below, so that no response accounts for the
    # batch without naming the recordings or datasets it leaves out, or the trackers it migrated.
    rejection_fields: dict[str, object] = {
        key: manifest[key]
        for key in (
            "invalid_paths",
            "invalid_recordings",
            "invalid_configurations",
            "migrated_recordings",
            "path_conflicts",
        )
        if key in manifest
    }

    # Records the recordings and datasets this step excludes, which the response carries beside the preparation
    # step's own rejections. An input nothing can size leaves the batch on its own, so every other input still runs.
    unsizable_recordings: list[dict[str, str]] = []
    unsizable_datasets: list[dict[str, str]] = []

    # Parses the manifest into phase groups. The groups order the admission pool and shape the response summary, while
    # the actual execution order follows each job's own prerequisites.
    phase_groups: list[tuple[str, list[PendingJob]]] = []

    if pipeline_type == "single-recording":
        binarize_phase_jobs: list[PendingJob] = []
        register_phase_jobs: list[PendingJob] = []
        process_phase_jobs: list[PendingJob] = []
        combine_phase_jobs: list[PendingJob] = []

        raw_recordings = manifest.get("recordings", {})
        if isinstance(raw_recordings, dict):
            for recording_key, recording_manifest in raw_recordings.items():
                manifest_dict: dict[str, Any] = recording_manifest
                job_configuration_path = Path(str(manifest_dict["configuration_path"]))
                tracker_path = Path(str(manifest_dict["tracker_path"]))

                # Sizes the recording's whole job set before any of it joins a phase group, so a recording that
                # cannot be sized leaves the batch whole rather than contributing the jobs resolved before the
                # failure. Every other recording of the batch still runs.
                try:
                    recording_jobs = _resolve_recording_phase_jobs(
                        manifest_dict=manifest_dict,
                        configuration_path=job_configuration_path,
                        tracker_path=tracker_path,
                    )
                except Exception as error:
                    unsizable_recordings.append(
                        {"recording": str(recording_key), "error": _collapse_whitespace(text=str(error))}
                    )
                    continue

                binarize_phase_jobs.extend(recording_jobs[0])
                register_phase_jobs.extend(recording_jobs[1])
                process_phase_jobs.extend(recording_jobs[2])
                combine_phase_jobs.extend(recording_jobs[3])

        if binarize_phase_jobs:
            phase_groups.append((SingleRecordingJobNames.BINARIZE.value, binarize_phase_jobs))
        if register_phase_jobs:
            phase_groups.append((SingleRecordingJobNames.REGISTER.value, register_phase_jobs))
        if process_phase_jobs:
            phase_groups.append((SingleRecordingJobNames.PROCESS.value, process_phase_jobs))
        if combine_phase_jobs:
            phase_groups.append((SingleRecordingJobNames.COMBINE.value, combine_phase_jobs))

    else:
        discover_phase_jobs: list[PendingJob] = []
        extract_phase_jobs: list[PendingJob] = []

        raw_datasets = manifest.get("datasets", {})
        if isinstance(raw_datasets, dict):
            for dataset_key, dataset_manifest in raw_datasets.items():
                manifest_dict = dataset_manifest
                job_configuration_path = Path(str(manifest_dict["configuration_path"]))
                tracker_path = Path(str(manifest_dict["tracker_path"]))

                # The multi-recording pipeline refuses a dataset whose recordings do not all carry combined output,
                # and the estimate refuses the same ones, so a dataset that cannot be sized leaves the batch whole.
                try:
                    dataset_jobs = _resolve_dataset_phase_jobs(
                        manifest_dict=manifest_dict,
                        configuration_path=job_configuration_path,
                        tracker_path=tracker_path,
                    )
                except Exception as error:
                    unsizable_datasets.append(
                        {"dataset": str(dataset_key), "error": _collapse_whitespace(text=str(error))}
                    )
                    continue

                discover_phase_jobs.extend(dataset_jobs[0])
                extract_phase_jobs.extend(dataset_jobs[1])

        if discover_phase_jobs:
            phase_groups.append((MultiRecordingJobNames.DISCOVER.value, discover_phase_jobs))
        if extract_phase_jobs:
            phase_groups.append((MultiRecordingJobNames.EXTRACT.value, extract_phase_jobs))

    # Folds this step's own exclusions into the shared rejection fields, so every outcome below names the inputs it
    # left out whether the preparation step or the sizing pass excluded them.
    if unsizable_recordings:
        rejection_fields["unsizable_recordings"] = unsizable_recordings
    if unsizable_datasets:
        rejection_fields["unsizable_datasets"] = unsizable_datasets

    if not phase_groups:
        prepared_entries = manifest.get("recordings" if pipeline_type == "single-recording" else "datasets", {})

        # Reports the rejections rather than a completion message when the preparation step accepted no input at all,
        # because a batch holding no prepared entry has no phase whose absence means completion.
        if not prepared_entries:
            return {
                "success": False,
                "started": False,
                "error": (
                    "Unable to execute full pipeline. The preparation step accepted none of the provided inputs, so "
                    "the session holds no jobs."
                ),
                "pipeline_type": pipeline_type,
                "total_jobs": 0,
                **rejection_fields,
            }

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
            **rejection_fields,
        }

    # Collects all jobs across all phases for the execution state, preserving the phase order so that the admission
    # scan always considers upstream jobs before the jobs that depend on them.
    all_jobs_map: dict[tuple[str, str], PendingJob] = {}
    for _phase_name, phase_jobs in phase_groups:
        for job in phase_jobs:
            all_jobs_map[job.dispatch_key] = job

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
        **rejection_fields,
    }

    return _start_session(
        all_jobs=all_jobs_map,
        workers_per_job=workers_per_job,
        max_parallel_jobs=max_parallel_jobs,
        extra_result_fields=extra_fields,
    )


@mcp.tool()
def size_pipeline_jobs_tool(
    configuration_path: str,
    pipeline_type: str,
    planned_roi_count: int | None = None,
) -> dict[str, object]:
    """Reports the cores and memory every job of a pipeline holds, without preparing or dispatching anything.

    Sizes each job of the configuration's whole job universe from the data that exists right now, so a caller plans a
    batch before any tracker or output directory is created. A single-recording job is sized from the recording's
    acquisition metadata and one source file header, and a multi-recording job is sized from the completed
    single-recording output the dataset runs on.

    Important:
        The figures are the same ones the execute tools charge against the session memory budget at dispatch, so a
        batch whose peak_memory_mb exceeds a host's free memory will admit its jobs serially rather than concurrently.
        Compare peak_memory_mb against the memory the host has free to predict that, and compare total_memory_mb
        against it to learn whether every job could ever run at once.

    Args:
        configuration_path: The absolute path to the pipeline configuration YAML file.
        pipeline_type: The pipeline type, either 'single-recording' or 'multi-recording'.
        planned_roi_count: The regions to plan for across every plane of a single-recording job. Use None to accept
            the ceiling the detection iteration bound provides. Must be a positive integer when supplied.

    Returns:
        On success, contains a 'jobs' list holding the 'name', 'specifier', 'cores', and 'memory_mb' of every job the
        pipeline can execute, in execution order, plus 'total_jobs', the 'peak_memory_mb' of the single largest job,
        the 'total_memory_mb' every job would hold at once, and the 'pipeline_type'. On failure, contains an 'error'
        describing the issue, which names the recording or dataset that could not be sized. Both cases include a
        'success' flag.
    """
    if pipeline_type not in ("single-recording", "multi-recording"):
        return {
            "success": False,
            "error": (
                f"Unable to size pipeline jobs. Invalid pipeline_type '{pipeline_type}'. "
                f"Must be 'single-recording' or 'multi-recording'."
            ),
        }

    configuration_file = Path(configuration_path)
    if not configuration_file.exists():
        return {
            "success": False,
            "error": f"Unable to size pipeline jobs. Configuration file not found: {configuration_path}.",
        }

    # Sizing reads the data the recording or dataset already holds, so an input the models cannot measure raises
    # instead of returning a figure. Reporting the reason keeps the caller from treating an unsizable input as free.
    try:
        if pipeline_type == "single-recording":
            sized_jobs = _size_single_recording_universe(
                configuration_path=configuration_file, planned_roi_count=planned_roi_count
            )
        else:
            sized_jobs = _size_multi_recording_universe(configuration_path=configuration_file)
    except Exception as error:
        return {"success": False, "error": _collapse_whitespace(text=f"Unable to size pipeline jobs. {error}")}

    memory_figures = [memory_mb for _, _, _, memory_mb in sized_jobs]

    return {
        "success": True,
        "pipeline_type": pipeline_type,
        "jobs": [
            {"name": name, "specifier": specifier, "cores": cores, "memory_mb": memory_mb}
            for name, specifier, cores, memory_mb in sized_jobs
        ],
        "total_jobs": len(sized_jobs),
        "peak_memory_mb": max(memory_figures) if memory_figures else 0,
        "total_memory_mb": sum(memory_figures),
    }


@mcp.tool()
def check_threading_runtime_tool() -> dict[str, object]:
    """Reports whether the numeric threading layer every parallelized stage needs is loadable on this host.

    The library selects the OpenMP threading layer on macOS and the TBB layer on every other platform. A macOS host
    carrying no loadable OpenMP runtime aborts every job at the pipeline entry point, before any stage runs, because
    both pipeline entry points verify the runtime first. Every other platform carries no such check, so a host missing
    the TBB runtime instead fails at the job's first parallelized call. Either outcome surfaces as a per-job tracker
    failure rather than as a tool error, so call this before dispatching a batch to gate on 'ready'.

    Returns:
        Always contains a 'success' flag indicating the tool ran, a 'ready' flag reporting whether the layer loads,
        the 'platform' this host runs, the 'required_layer' the library selects for it ('omp' or 'tbb'), and a
        'detail' sentence describing the outcome. A host that is not ready also carries a 'remedy' naming the command
        that resolves it. On macOS, the report additionally carries 'discovered_runtimes' holding the single runtime the
        discovery would link, which is the first candidate that resolves to a file, and 'searched_paths' listing the
        candidates examined. When no runtime was found, 'discovered_runtimes' is empty while 'searched_paths' still
        lists every candidate the discovery examined. Both are empty only when the runtime already loads, because a
        host that already loads one runs no discovery.
    """
    platform_name = sys.platform

    if platform_name != "darwin":
        # Every non-macOS platform runs TBB, whose runtime ships in the tbb4py distribution rather than with Numba.
        tbb_available = find_spec(name="tbb") is not None
        result: dict[str, object] = {
            "success": True,
            "ready": tbb_available,
            "platform": platform_name,
            "required_layer": "tbb",
        }
        if tbb_available:
            result["detail"] = "The TBB threading layer runtime is installed, so every parallelized stage can run."
        else:
            result["detail"] = (
                "The TBB threading layer runtime is missing, so every parallelized stage fails at its first "
                "parallelized call."
            )
            result["remedy"] = "pip install tbb4py"
        return result

    summary = resolve_openmp_runtime(execute=False)

    # describe() phrases its outcome for the 'cindra omp' command, which names flags an MCP caller cannot pass, so the
    # ready case carries its own sentence and every other case reuses the discovery detail describe() already builds.
    detail = (
        "The OpenMP runtime loads, so every parallelized stage can run."
        if summary.status == OpenMPStatus.AVAILABLE
        else summary.describe()
    )

    result = {
        "success": True,
        "ready": summary.loadable,
        "platform": platform_name,
        "required_layer": "omp",
        "discovered_runtimes": [str(summary.runtime_path)] if summary.runtime_path is not None else [],
        "searched_paths": [str(path) for path in summary.searched_paths],
        "detail": detail,
    }

    if not summary.loadable:
        result["remedy"] = (
            "brew install libomp" if summary.status == OpenMPStatus.UNRESOLVED else "sudo cindra omp --yes"
        )

    return result


@mcp.tool()
def get_pipeline_job_universe_tool(configuration_path: str, pipeline_type: str) -> dict[str, object]:
    """Reports every job a recording or dataset declares and which of them can run right now.

    Reads the inventory the output directories already hold, so it answers what is runnable before a tracker exists and
    without preparing anything. A recording carrying nothing resolves to an empty universe rather than failing, which
    makes this safe to call on an unprocessed recording.

    Important:
        The 'ready' flag reports that a job's own input exists on disk, which is a different question from whether its
        prerequisite job has succeeded on a tracker. Use this tool to plan a selective re-run, and
        get_recording_status_tool to read recorded job outcomes once a batch has been prepared.

    Args:
        configuration_path: The absolute path to the pipeline configuration YAML file.
        pipeline_type: The pipeline type, either 'single-recording' or 'multi-recording'.

    Returns:
        On success, contains a 'jobs' list holding the 'name', 'specifier', and 'ready' flag of every job the pipeline
        declares, in execution order, plus 'total_jobs', 'ready_jobs' counting the runnable subset, a 'resolved' flag
        reporting whether the universe follows from the recording's own parameters rather than from their absence, and
        the 'pipeline_type'. A single-recording report also carries 'plane_count', and a multi-recording report carries
        'dataset_name' and 'recording_ids'. On failure, contains an 'error' describing the issue. Both cases include a
        'success' flag.
    """
    if pipeline_type not in ("single-recording", "multi-recording"):
        return {
            "success": False,
            "error": (
                f"Unable to resolve the pipeline job universe. Invalid pipeline_type '{pipeline_type}'. "
                f"Must be 'single-recording' or 'multi-recording'."
            ),
        }

    configuration_file = Path(configuration_path)
    if not configuration_file.exists():
        return {
            "success": False,
            "error": (
                f"Unable to resolve the pipeline job universe. Configuration file not found: {configuration_path}."
            ),
        }

    # Both universe resolvers report an empty universe for a recording or dataset carrying nothing rather than
    # raising, but the configuration load and the identifier derivation behind them both can, so the guard spans them.
    try:
        if pipeline_type == "single-recording":
            configuration, output_path = load_single_recording_configuration(configuration_path=configuration_file)
            single_universe = resolve_single_recording_job_universe(
                output_root=output_path, data_path=configuration.file_io.data_path
            )
            possible = set(single_universe.possible)
            return {
                "success": True,
                "pipeline_type": pipeline_type,
                "resolved": single_universe.resolved,
                "plane_count": single_universe.plane_count,
                "jobs": [
                    {"name": name, "specifier": specifier, "ready": (name, specifier) in possible}
                    for name, specifier in single_universe.universe
                ],
                "total_jobs": len(single_universe.universe),
                "ready_jobs": len(single_universe.possible),
            }

        dataset_configuration = load_multi_recording_configuration(configuration_path=configuration_file)
        multi_universe = resolve_multi_recording_job_universe(
            recording_roots=dataset_configuration.recording_io.recording_directories,
            dataset_name=dataset_configuration.recording_io.dataset_name,
        )
    except Exception as error:
        return {
            "success": False,
            "error": _collapse_whitespace(text=f"Unable to resolve the pipeline job universe. {error}"),
        }

    multi_possible = set(multi_universe.possible)
    return {
        "success": True,
        "pipeline_type": pipeline_type,
        "resolved": multi_universe.resolved,
        "dataset_name": multi_universe.dataset_name,
        "recording_ids": list(multi_universe.recording_ids),
        "jobs": [
            {"name": name, "specifier": specifier, "ready": (name, specifier) in multi_possible}
            for name, specifier in multi_universe.universe
        ],
        "total_jobs": len(multi_universe.universe),
        "ready_jobs": len(multi_universe.possible),
    }


def _collapse_whitespace(text: str) -> str:
    """Folds every run of whitespace in a message into a single space.

    Notes:
        The console wraps its messages at a fixed column, so an exception raised through it carries hard newlines that
        would otherwise reach the caller inside a JSON string.

    Args:
        text: The message to fold.

    Returns:
        The message on a single line.
    """
    return " ".join(text.split())


def _resolve_raw_data_failure(raw_data_path: Path, ignored_file_names: tuple[str, ...]) -> str | None:
    """Reports why the conversion would find no usable source file in one raw imaging directory.

    Notes:
        Resolves the imaging directory the way the conversion does, by locating the acquisition parameters file
        beneath the named path and reading the directory that holds it. A path that parents the imaging directory
        therefore passes this gate exactly as it passes the conversion, and only a path whose subtree carries no
        usable source file is rejected.

    Args:
        raw_data_path: The path the caller named as the recording's raw imaging path.
        ignored_file_names: The file stems the configuration excludes from discovery.

    Returns:
        None when the conversion would find at least one TIFF file beneath the path, or the reason it would find none.
    """
    try:
        data_directory = find_data_directory(data_path=raw_data_path)
    except FileNotFoundError, OSError, ValueError:
        data_directory = raw_data_path

    try:
        resolve_source_frame_geometry(data_directory=data_directory, ignored_file_names=ignored_file_names)
    except FileNotFoundError:
        message = (
            f"The conversion would find no TIFF file it accepts beneath {raw_data_path}. The conversion reads the "
            f"directory holding the recording's cindra_parameters.json file, and scans that one directory without "
            f"descending further, so the TIFF files must sit beside that file."
        )
        subdirectory = _resolve_tiff_subdirectory(raw_data_path=raw_data_path, ignored_file_names=ignored_file_names)
        if subdirectory is not None:
            message = f"{message} The subdirectory {subdirectory} holds such files and is the likely intended path."
        return message
    except Exception as error:
        return _collapse_whitespace(text=f"Unable to read the source files of {raw_data_path}: {error}")

    return None


def _resolve_tiff_subdirectory(raw_data_path: Path, ignored_file_names: tuple[str, ...]) -> Path | None:
    """Searches the directories below a rejected raw data path for the one holding the recording's TIFF files.

    Notes:
        The search spans the levels _TIFF_HINT_SEARCH_DEPTH covers, which reaches the layout a session root nests its
        imaging directory in while leaving a deep tree unwalked.

    Args:
        raw_data_path: The directory the caller named, which holds no TIFF file the conversion accepts.
        ignored_file_names: The file stems the configuration excludes from discovery.

    Returns:
        The first subdirectory holding a TIFF file the conversion accepts, or None when the search finds none.
    """
    frontier: list[Path] = [raw_data_path]
    for _level in range(_TIFF_HINT_SEARCH_DEPTH):
        children: list[Path] = []
        for directory in frontier:
            try:
                children.extend(entry for entry in natsorted(directory.iterdir(), key=str) if entry.is_dir())
            except OSError:
                continue

        for child in children:
            # A child that cannot be read is simply not the imaging directory. The reason is not reported, because
            # this search only ever refines a hint inside an error the caller already receives.
            try:
                resolve_source_frame_geometry(data_directory=child, ignored_file_names=ignored_file_names)
            except Exception:  # noqa: S112 - The failure only means this child is not the imaging directory.
                continue
            return child

        frontier = children

    return None


def _resolve_single_recording_path_conflicts(
    recording_key: str, configuration_path: Path, output_root: Path, data_path: Path
) -> list[dict[str, str]]:
    """Compares the paths a prepared recording already records against the paths the caller passed.

    Args:
        recording_key: The raw data path the manifest keys the recording by.
        configuration_path: The path to the per-recording configuration file the previous preparation wrote.
        output_root: The output root the caller passed.
        data_path: The raw imaging directory the caller passed.

    Returns:
        One entry per disagreeing path, naming the recording, the stored value, the passed value, and the removal that
        allows the recording to be prepared again. The list is empty when both paths agree or the stored configuration
        cannot be read.
    """
    try:
        configuration = SingleRecordingConfiguration.from_yaml(file_path=configuration_path)
    except Exception:
        return []

    resolution = (
        f"Preparation does not reinitialize an existing tracker, so the recording keeps running against the stored "
        f"paths. Remove {output_root / OUTPUT_DIRECTORY_NAME} to prepare this recording again with different paths."
    )

    return [
        {
            "recording": recording_key,
            "field": field_name,
            "stored": str(stored_path),
            "passed": str(passed_path),
            "resolution": resolution,
        }
        for field_name, stored_path, passed_path in (
            ("file_io.data_path", configuration.file_io.data_path, data_path),
            ("file_io.output_path", configuration.file_io.output_path, output_root),
        )
        if stored_path is None or Path(stored_path) != passed_path
    ]


def _resolve_multi_recording_path_conflicts(
    dataset_key: str, configuration_path: Path, output_roots: tuple[Path, ...]
) -> list[dict[str, str]]:
    """Compares the output roots a prepared dataset already records against the ones the caller passed.

    Args:
        dataset_key: The lowercased dataset name the manifest keys the dataset by.
        configuration_path: The path to the dataset configuration file the previous preparation wrote.
        output_roots: The output roots the caller passed, in the order the preparation would store them.

    Returns:
        A single entry naming the dataset, the stored roots, the passed roots, and the removal that allows the dataset
        to be prepared again. The list is empty when the roots agree or the stored configuration cannot be read.
    """
    try:
        configuration = MultiRecordingConfiguration.from_yaml(file_path=configuration_path)
    except Exception:
        return []

    stored_roots = tuple(Path(path) for path in configuration.recording_io.recording_directories)
    if stored_roots == tuple(output_roots):
        return []

    return [
        {
            "dataset": dataset_key,
            "field": "recording_io.recording_directories",
            "stored": ", ".join(str(path) for path in stored_roots),
            "passed": ", ".join(str(path) for path in output_roots),
            "resolution": (
                f"Preparation does not reinitialize an existing tracker, so the dataset keeps running against the "
                f"stored output roots. Remove {configuration_path.parent} to prepare this dataset again with "
                f"different output roots."
            ),
        }
    ]


def _resolve_repeat_flag_warnings(tracker_path: Path, phase_names: list[str], *, single_recording: bool) -> list[str]:
    """Reports every reset phase whose stage would skip its work because its repeat flag is disabled.

    Notes:
        Binarization, registration, and multi-recording discovery each read their own output before running and return
        immediately when it exists and their repeat flag is false. Resetting the tracker entry of such a phase
        therefore produces a job that records success in seconds without redoing the work.

    Args:
        tracker_path: The path to the tracker the reset was applied to, whose directory holds the configuration file.
        phase_names: The phases the reset covers after downstream expansion.
        single_recording: Determines whether the tracker belongs to the single-recording or multi-recording pipeline.

    Returns:
        One sentence per phase that would skip its work, naming the configuration flag that lifts the skip. The list
        is empty when the configuration cannot be read.
    """
    warnings: list[str] = []

    if single_recording:
        try:
            configuration = SingleRecordingConfiguration.from_yaml(
                file_path=tracker_path.parent / SINGLE_RECORDING_CONFIGURATION_FILENAME
            )
        except Exception:
            return warnings

        output_root = configuration.file_io.output_path
        if output_root is None:
            return warnings

        planes = resolve_recording_planes(output_root=output_root, data_path=configuration.file_io.data_path)
        converted = any(
            is_plane_converted(output_root=output_root, plane_index=plane_index)
            for plane_index in range(planes.plane_count)
        )

        if (
            SingleRecordingJobNames.BINARIZE in phase_names
            and not configuration.file_io.repeat_binarization
            and converted
        ):
            warnings.append(
                "The binarization phase was reset while file_io.repeat_binarization is false and at least one plane "
                "binary already exists, so the job reuses that binary and records success without rebuilding it. Set "
                "file_io.repeat_binarization to true with set_config_values_tool before dispatching the phase."
            )

        if (
            SingleRecordingJobNames.REGISTER in phase_names
            and not configuration.registration.repeat_registration
            and planes.registered_planes
        ):
            warnings.append(
                "The registration phase was reset while registration.repeat_registration is false and at least one "
                "plane already carries registration output, so the job returns immediately and records success "
                "without re-registering. Set registration.repeat_registration to true with set_config_values_tool "
                "before dispatching the phase."
            )

        return warnings

    try:
        dataset_configuration = MultiRecordingConfiguration.from_yaml(
            file_path=tracker_path.parent / MULTI_RECORDING_CONFIGURATION_FILENAME
        )
    except Exception:
        return warnings

    dataset_name = dataset_configuration.recording_io.dataset_name
    discovered = any(
        is_dataset_discovered(output_root=recording_root, dataset_name=dataset_name)
        for recording_root in dataset_configuration.recording_io.recording_directories
    )

    if (
        MultiRecordingJobNames.DISCOVER in phase_names
        and not dataset_configuration.recording_io.repeat_selection
        and discovered
    ):
        warnings.append(
            "The discovery phase was reset while recording_io.repeat_selection is false and the dataset already "
            "carries selected ROIs, so the job reuses that selection and records success without repeating it. Set "
            "recording_io.repeat_selection to true with set_config_values_tool before dispatching the phase."
        )

    return warnings


def _check_active_session(action: str) -> dict[str, object] | None:
    """Reports whether an execution session is already running and therefore blocks a new one.

    Args:
        action: The lowercase description of the blocked action, interpolated into the error message.

    Returns:
        None when no session is active, or an error result dictionary describing the running session.
    """
    pending_count, active_count = resolve_session_load()
    if not pending_count and not active_count:
        return None

    return {
        "success": False,
        "error": f"Unable to {action}. An execution session is already active.",
        "pending_count": pending_count,
        "active_count": active_count,
    }


def _start_session(
    all_jobs: dict[tuple[str, str], PendingJob],
    workers_per_job: int | None,
    max_parallel_jobs: int | None,
    extra_result_fields: dict[str, object],
) -> dict[str, object]:
    """Starts a batch execution session and shapes its allocation report into an MCP tool response.

    Args:
        all_jobs: All submitted jobs keyed by dispatch key, in the order the manager should consider them.
        workers_per_job: Requested CPU cores per job, -1 for every available core, or None to accept each resource
            class default.
        max_parallel_jobs: Requested maximum concurrent jobs per resource class, -1 to lift the caps, or None to
            accept the derived caps.
        extra_result_fields: Additional key-value pairs to include in the result dictionary.

    Returns:
        A result dictionary containing 'success', 'started', the session 'cpu_budget' and 'memory_budget_mb', per-class
        resource allocation details, and any extra fields. A rejected override yields success:False and an 'error'
        instead.
    """
    try:
        session = start_execution_session(
            all_jobs=all_jobs, workers_per_job=workers_per_job, max_parallel_jobs=max_parallel_jobs
        )
    except ValueError as error:
        return {"success": False, "started": False, "error": _collapse_whitespace(text=str(error))}

    result: dict[str, object] = {"success": True, "started": True, **session}
    result.update(extra_result_fields)
    return result


def _group_jobs_by_name(registry: dict[str, JobState], job_name: str) -> dict[str, JobState]:
    """Selects the jobs of one pipeline phase from a tracker registry snapshot.

    Args:
        registry: The point-in-time job registry read from the tracker.
        job_name: The name of the phase whose jobs are selected.

    Returns:
        The state of every job carrying the requested phase name, keyed by that job's identifier.
    """
    return {job_id: job_state for job_id, job_state in registry.items() if job_state.job_name == job_name}


def _read_single_recording_tracker(tracker_path: Path, output_root: Path) -> dict[str, object]:
    """Reads a single-recording ProcessingTracker and returns structured status information.

    Args:
        tracker_path: The path to the ProcessingTracker YAML file.
        output_root: The pipeline output root the tracker belongs to, reported back to the caller.

    Returns:
        A dictionary containing a success flag, the output root, tracker path, per-phase job status, summary counts,
        and an overall synthesized status string.
    """
    # Reads the whole registry once and derives every phase grouping, the counts, and the overall status from that one
    # snapshot. Asking the tracker for each phase separately costs one lock acquisition and one YAML parse per
    # question, which the batch overview then pays once per discovered tracker across a whole data root.
    registry = ProcessingTracker(file_path=tracker_path).snapshot()

    binarize_jobs = _group_jobs_by_name(registry=registry, job_name=SingleRecordingJobNames.BINARIZE)
    register_jobs = _group_jobs_by_name(registry=registry, job_name=SingleRecordingJobNames.REGISTER)
    process_jobs = _group_jobs_by_name(registry=registry, job_name=SingleRecordingJobNames.PROCESS)
    combine_jobs = _group_jobs_by_name(registry=registry, job_name=SingleRecordingJobNames.COMBINE)

    binarize_status: dict[str, object] = {}
    for job_state in binarize_jobs.values():
        binarize_status["status"] = job_state.status.name.lower()
        if job_state.error_message:
            binarize_status["error"] = _collapse_whitespace(text=job_state.error_message)

    register_status: dict[str, object] = {
        job_state.specifier: job_state.status.name.lower() for job_state in register_jobs.values()
    }

    process_status: dict[str, object] = {
        job_state.specifier: job_state.status.name.lower() for job_state in process_jobs.values()
    }

    combine_status: dict[str, object] = {}
    for job_state in combine_jobs.values():
        combine_status["status"] = job_state.status.name.lower()
        if job_state.error_message:
            combine_status["error"] = _collapse_whitespace(text=job_state.error_message)

    # Synthesizes overall status from tracker state, reporting the furthest phase the recording has reached.
    statuses = [job_state.status for job_state in registry.values()]
    if statuses and all(status == ProcessingStatus.SUCCEEDED for status in statuses):
        overall_status = "completed"
    elif any(status == ProcessingStatus.FAILED for status in statuses):
        overall_status = "failed"
    elif combine_jobs and any(job_state.status == ProcessingStatus.RUNNING for job_state in combine_jobs.values()):
        overall_status = "combining"
    elif process_jobs and any(
        job_state.status in (ProcessingStatus.RUNNING, ProcessingStatus.SUCCEEDED)
        for job_state in process_jobs.values()
    ):
        overall_status = "processing"
    elif register_jobs and any(
        job_state.status in (ProcessingStatus.RUNNING, ProcessingStatus.SUCCEEDED)
        for job_state in register_jobs.values()
    ):
        overall_status = "registering"
    elif binarize_jobs and any(
        job_state.status in (ProcessingStatus.RUNNING, ProcessingStatus.SUCCEEDED)
        for job_state in binarize_jobs.values()
    ):
        overall_status = "binarizing"
    else:
        overall_status = "scheduled"

    summary_counts: dict[str, int] = {
        status.name.lower(): sum(1 for job_status in statuses if job_status == status) for status in ProcessingStatus
    }

    return {
        "success": True,
        "output_root": str(output_root),
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
    registry = ProcessingTracker(file_path=tracker_path).snapshot()

    discover_jobs = _group_jobs_by_name(registry=registry, job_name=MultiRecordingJobNames.DISCOVER)
    extract_jobs = _group_jobs_by_name(registry=registry, job_name=MultiRecordingJobNames.EXTRACT)

    discover_status: dict[str, object] = {}
    for job_state in discover_jobs.values():
        discover_status["status"] = job_state.status.name.lower()
        if job_state.error_message:
            discover_status["error"] = _collapse_whitespace(text=job_state.error_message)

    extract_status: dict[str, object] = {
        job_state.specifier: job_state.status.name.lower() for job_state in extract_jobs.values()
    }

    statuses = [job_state.status for job_state in registry.values()]
    if statuses and all(status == ProcessingStatus.SUCCEEDED for status in statuses):
        overall_status = "completed"
    elif any(status == ProcessingStatus.FAILED for status in statuses):
        overall_status = "failed"
    elif extract_jobs and any(
        job_state.status in (ProcessingStatus.RUNNING, ProcessingStatus.SUCCEEDED)
        for job_state in extract_jobs.values()
    ):
        overall_status = "extracting"
    elif discover_jobs and any(
        job_state.status in (ProcessingStatus.RUNNING, ProcessingStatus.SUCCEEDED)
        for job_state in discover_jobs.values()
    ):
        overall_status = "discovering"
    else:
        overall_status = "scheduled"

    summary_counts: dict[str, int] = {
        status.name.lower(): sum(1 for job_status in statuses if job_status == status) for status in ProcessingStatus
    }

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
        errors.append(_collapse_whitespace(text=f"Unable to delete file {path}: {error}"))


def _delete_directory(path: Path, deleted: list[str], errors: list[str]) -> None:
    """Recursively deletes a directory and records the result.

    Notes:
        The ataraxis directory remover unlinks the files of each directory in parallel and retries every emptied
        directory whose removal the host refuses, which suits a tree of memory-mapped arrays another process may still
        hold open. It reports a directory it could not remove through a warning and returns rather than raising, so
        the path is re-examined afterwards to decide between the deleted and the failed list.

    Args:
        path: The filesystem path to the directory to delete.
        deleted: The list to append the deleted directory path to on success.
        errors: The list to append error messages to on failure.
    """
    if not path.exists():
        return
    try:
        delete_directory(directory_path=path)
    except Exception as error:
        errors.append(_collapse_whitespace(text=f"Unable to delete directory {path}: {error}"))
        return

    if path.exists():
        errors.append(f"Unable to delete directory {path}: the directory could not be removed after every attempt.")
        return

    deleted.append(str(path))


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


def _resolve_dataset_phase_jobs(
    manifest_dict: dict[str, Any], configuration_path: Path, tracker_path: Path
) -> tuple[list[PendingJob], list[PendingJob]]:
    """Builds the pending jobs of one tracked dataset, grouped by the phase each belongs to.

    Notes:
        Sizing every job of the dataset here rather than at each phase group lets the caller admit or exclude a
        dataset whole. That matches the pipeline, which refuses a dataset whose recordings do not all carry the
        combined output a completed single-recording run leaves behind.

    Args:
        manifest_dict: The dataset's entry in the prepared batch manifest.
        configuration_path: The path to the dataset's pipeline configuration file.
        tracker_path: The path to the dataset's processing tracker.

    Returns:
        The discovery and extraction jobs the dataset contributes, in that order.

    Raises:
        FileNotFoundError: If any recording the dataset spans carries nothing its stages can be sized against.
    """
    discover_jobs: list[PendingJob] = []
    extract_jobs: list[PendingJob] = []

    discover = manifest_dict.get("discover_job", {})
    if discover and discover.get("status") != "succeeded":
        discover_jobs.append(
            PendingJob(
                configuration_path=configuration_path,
                tracker_path=tracker_path,
                job_id=discover["job_id"],
                single_recording=False,
                resource_class=RESOURCE_CLASS_BY_JOB_NAME[MultiRecordingJobNames.DISCOVER],
                memory_megabytes=_estimate_pending_job_memory(
                    configuration_path=configuration_path,
                    job_name=str(discover["name"]),
                    specifier=str(discover["specifier"]),
                    single=False,
                ),
            )
        )

    extract_jobs.extend(
        PendingJob(
            configuration_path=configuration_path,
            tracker_path=tracker_path,
            job_id=extract["job_id"],
            single_recording=False,
            resource_class=RESOURCE_CLASS_BY_JOB_NAME[MultiRecordingJobNames.EXTRACT],
            memory_megabytes=_estimate_pending_job_memory(
                configuration_path=configuration_path,
                job_name=str(extract["name"]),
                specifier=str(extract["specifier"]),
                single=False,
            ),
        )
        for extract in manifest_dict.get("extract_jobs", [])
        if extract.get("status") != "succeeded"
    )

    return discover_jobs, extract_jobs


def _resolve_recording_phase_jobs(
    manifest_dict: dict[str, Any], configuration_path: Path, tracker_path: Path
) -> tuple[list[PendingJob], list[PendingJob], list[PendingJob], list[PendingJob]]:
    """Builds the pending jobs of one recording, grouped by the phase each belongs to.

    Notes:
        Sizing every job of the recording here rather than at each phase group lets the caller admit or exclude a
        recording whole. A recording that cannot be sized contributes no job at all, so the batch never runs a
        recording's early phases against an estimate its later phases could not produce.

    Args:
        manifest_dict: The recording's entry in the prepared batch manifest.
        configuration_path: The path to the recording's pipeline configuration file.
        tracker_path: The path to the recording's processing tracker.

    Returns:
        The binarization, registration, processing, and combination jobs the recording contributes, in that order.

    Raises:
        FileNotFoundError: If the recording carries nothing its stages can be sized against.
    """
    binarize_jobs: list[PendingJob] = []
    register_jobs: list[PendingJob] = []
    process_jobs: list[PendingJob] = []
    combine_jobs: list[PendingJob] = []

    binarize = manifest_dict.get("binarize_job", {})
    if binarize and binarize.get("status") != "succeeded":
        binarize_jobs.append(
            PendingJob(
                configuration_path=configuration_path,
                tracker_path=tracker_path,
                job_id=binarize["job_id"],
                single_recording=True,
                resource_class=RESOURCE_CLASS_BY_JOB_NAME[SingleRecordingJobNames.BINARIZE],
                memory_megabytes=_estimate_pending_job_memory(
                    configuration_path=configuration_path,
                    job_name=str(binarize["name"]),
                    specifier=str(binarize["specifier"]),
                    single=True,
                ),
            )
        )

    register_jobs.extend(
        PendingJob(
            configuration_path=configuration_path,
            tracker_path=tracker_path,
            job_id=register["job_id"],
            single_recording=True,
            resource_class=RESOURCE_CLASS_BY_JOB_NAME[SingleRecordingJobNames.REGISTER],
            memory_megabytes=_estimate_pending_job_memory(
                configuration_path=configuration_path,
                job_name=str(register["name"]),
                specifier=str(register["specifier"]),
                single=True,
            ),
        )
        for register in manifest_dict.get("register_jobs", [])
        if register.get("status") != "succeeded"
    )

    process_jobs.extend(
        PendingJob(
            configuration_path=configuration_path,
            tracker_path=tracker_path,
            job_id=process["job_id"],
            single_recording=True,
            resource_class=RESOURCE_CLASS_BY_JOB_NAME[SingleRecordingJobNames.PROCESS],
            memory_megabytes=_estimate_pending_job_memory(
                configuration_path=configuration_path,
                job_name=str(process["name"]),
                specifier=str(process["specifier"]),
                single=True,
            ),
        )
        for process in manifest_dict.get("process_jobs", [])
        if process.get("status") != "succeeded"
    )

    combine = manifest_dict.get("combine_job", {})
    if combine and combine.get("status") != "succeeded":
        combine_jobs.append(
            PendingJob(
                configuration_path=configuration_path,
                tracker_path=tracker_path,
                job_id=combine["job_id"],
                single_recording=True,
                resource_class=RESOURCE_CLASS_BY_JOB_NAME[SingleRecordingJobNames.COMBINE],
                memory_megabytes=_estimate_pending_job_memory(
                    configuration_path=configuration_path,
                    job_name=str(combine["name"]),
                    specifier=str(combine["specifier"]),
                    single=True,
                ),
            )
        )

    return binarize_jobs, register_jobs, process_jobs, combine_jobs


def _estimate_pending_job_memory(configuration_path: Path, job_name: str, specifier: str, *, single: bool) -> int:
    """Estimates the memory one queued job holds, from the recording or dataset it will process.

    Args:
        configuration_path: The path to the job's pipeline configuration file.
        job_name: The name of the pipeline stage the job runs.
        specifier: The job's tracker specifier.
        single: Determines whether the job belongs to the single-recording or the multi-recording pipeline.

    Returns:
        The memory the job holds in megabytes.

    Raises:
        FileNotFoundError: If the recording or dataset the job runs on carries nothing its stage can be sized
            against.
    """
    if single:
        configuration, output_path = load_single_recording_configuration(configuration_path=configuration_path)
        return estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames(job_name),
            specifier=specifier,
            output_root=output_path,
            configuration=configuration,
            data_path=configuration.file_io.data_path,
        )

    dataset_configuration = load_multi_recording_configuration(configuration_path=configuration_path)
    return estimate_multi_recording_job_memory_mb(
        job_name=MultiRecordingJobNames(job_name),
        specifier=specifier,
        recording_directories=dataset_configuration.recording_io.recording_directories,
        configuration=dataset_configuration,
    )


def _manifest_entry(identifiers: dict[tuple[str, str], str], job_name: str, specifier: str) -> dict[str, object]:
    """Builds the manifest entry describing one scheduled job.

    Args:
        identifiers: The identifier of every job, keyed by its name and specifier.
        job_name: The name of the job the entry describes.
        specifier: The specifier of the job the entry describes.

    Returns:
        The manifest entry holding the job's identifier, name, specifier, and scheduled status.
    """
    return {
        "job_id": identifiers[str(job_name), specifier],
        "name": str(job_name),
        "specifier": specifier,
        "status": "scheduled",
    }


def _resolve_job_identifiers(tracker: ProcessingTracker, jobs: list[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """Initializes a tracker's jobs and keys every returned identifier by the job it belongs to.

    Notes:
        The tracker returns its identifiers in the order the job list carries them. Keying them by name and specifier
        keeps a manifest entry bound to its own job, so inserting or reordering a pipeline phase cannot shift an
        identifier onto a different job.

    Args:
        tracker: The tracker to initialize the jobs on.
        jobs: The job universe, as job name and specifier pairs.

    Returns:
        The identifier of every job, keyed by its name and specifier.
    """
    job_ids = tracker.initialize_jobs(jobs=jobs)
    return dict(zip(jobs, job_ids, strict=True))


def _size_single_recording_universe(
    configuration_path: Path, planned_roi_count: int | None
) -> list[tuple[str, str, int, int]]:
    """Sizes every job the single-recording pipeline can execute for one recording.

    Notes:
        The plane count comes from the recording's acquisition geometry rather than from a tracker, so the universe
        resolves before the recording has been prepared.

    Args:
        configuration_path: The path to the recording's configuration file.
        planned_roi_count: The regions to plan for across every plane, or None to accept the detection ceiling.

    Returns:
        The name, specifier, cores, and memory in megabytes of every job, in execution order.

    Raises:
        FileNotFoundError: If the recording's raw imaging directory holds no readable source file.
        ValueError: If planned_roi_count is not a positive integer, or if the recording declares no imaging plane.
    """
    configuration, output_path = load_single_recording_configuration(configuration_path=configuration_path)
    geometry = resolve_recording_geometry(
        output_root=output_path,
        data_path=configuration.file_io.data_path,
        ignored_file_names=configuration.file_io.ignored_file_names,
    )

    if not geometry.planes:
        message = (
            f"Unable to size the jobs of the recording configured at {configuration_path}. The recording must declare "
            f"at least one imaging plane, but it declares none. Verify that its acquisition parameters file is "
            f"readable."
        )
        console.error(message=message, error=ValueError)

    sized: list[tuple[str, str, int, int]] = []
    for job_name, specifier in resolve_single_recording_jobs(plane_count=len(geometry.planes)):
        sizing = size_single_recording_job(
            job_name=SingleRecordingJobNames(job_name),
            specifier=specifier,
            output_root=output_path,
            configuration=configuration,
            data_path=configuration.file_io.data_path,
            planned_roi_count=planned_roi_count,
        )
        sized.append((job_name, specifier, sizing.cores, sizing.memory_mb))

    return sized


def _size_multi_recording_universe(configuration_path: Path) -> list[tuple[str, str, int, int]]:
    """Sizes every job the multi-recording pipeline can execute for one tracked dataset.

    Notes:
        The recording identifiers come from a read-only inventory of the completed single-recording output, so the
        universe resolves before the dataset has been prepared.

    Args:
        configuration_path: The path to the dataset's configuration file.

    Returns:
        The name, specifier, cores, and memory in megabytes of every job, in execution order.

    Raises:
        FileNotFoundError: If no recording the dataset names carries a combined metadata archive.
    """
    configuration = load_multi_recording_configuration(configuration_path=configuration_path)
    recording_directories = configuration.recording_io.recording_directories
    inventory = resolve_dataset_recordings(
        recording_roots=recording_directories, dataset_name=configuration.recording_io.dataset_name
    )

    sized: list[tuple[str, str, int, int]] = []
    for job_name, specifier in resolve_multi_recording_jobs(recording_ids=inventory.recording_ids):
        sizing = size_multi_recording_job(
            job_name=MultiRecordingJobNames(job_name),
            specifier=specifier,
            recording_directories=recording_directories,
            configuration=configuration,
        )
        sized.append((job_name, specifier, sizing.cores, sizing.memory_mb))

    return sized
