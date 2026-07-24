"""Provides the centralized pipeline for processing the acquired brain activity data."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console, resolve_worker_count
from ataraxis_data_structures import ProcessingTracker

from ..io import resolve_multi_recording_contexts, resolve_single_recording_contexts
from ..dataclasses import RuntimeContext, MultiRecordingConfiguration, SingleRecordingConfiguration
from .multi_recording import discover_multi_recording_cells, extract_multi_recording_fluorescence
from .single_recording import process_plane, binarize_recording, save_combined_data

if TYPE_CHECKING:
    from pathlib import Path

SINGLE_RECORDING_TRACKER_NAME: str = "single_recording_tracker.yaml"
"""The tracker file name for the single-recording processing pipeline."""

MULTI_RECORDING_TRACKER_NAME: str = "multi_recording_tracker.yaml"
"""The tracker file name for the multi-recording processing pipeline."""


class SingleRecordingJobNames(StrEnum):
    """Defines the job names for the single-recording processing pipeline components."""

    BINARIZE = "binarization"
    """The name for the binarization (step 1) processing job."""
    PROCESS = "processing"
    """The generic name for the plane-processing (step 2) job. During runtime, the processed plane is identified
    by the tracker's specifier field using the format 'plane_{plane_index}'."""
    COMBINE = "combination"
    """The name for the combination (step 3) processing job."""


class MultiRecordingJobNames(StrEnum):
    """Defines the job names for the multi-recording processing pipeline components."""

    DISCOVER = "discovery"
    """The name for the ROI discovery (step 1) processing job."""
    EXTRACT = "extraction"
    """The generic name for the fluorescence extraction (step 2) processing job. During runtime, the processed recording
    is identified by the tracker's specifier field, which stores the recording ID string."""


def run_single_recording_pipeline(
    configuration_path: Path,
    job_id: str | None = None,
    *,
    binarize: bool = False,
    process: bool = False,
    combine: bool = False,
    target_plane: int = -1,
) -> None:
    """Executes the requested single-recording processing pipeline steps for the target data.

    The caller is responsible for writing all runtime overrides (``file_io.data_path``, ``file_io.output_path``,
    ``runtime.parallel_workers``, ``runtime.display_progress_bars``) into the configuration file before invoking this
    function. The pipeline reads these values from the file at ``configuration_path`` and does not accept them as
    direct parameters.

    Args:
        configuration_path: The path to the single-recording configuration YAML file.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed. If not provided, all requested jobs are run sequentially.
        binarize: Determines whether to resolve the binary files for plane-specific processing (step 1).
        process: Determines whether to process the target plane(s) to remove motion, discover ROIs, and extract their
            fluorescence (step 2).
        combine: Determines whether to combine processed plane data into a uniform dataset (step 3).
        target_plane: The index of the plane to process. Setting this to '-1' processes all available planes
            sequentially.

    Raises:
        FileNotFoundError: If the single-recording configuration data cannot be loaded from the specified file.
        ValueError: If the recording's data validation fails or the specified job_id does not match any available jobs.
    """
    configuration, output_path = _load_single_recording_configuration(configuration_path=configuration_path)

    # Resolves RuntimeContext instances for all planes upfront. This determines the plane count without requiring
    # binarization to run first, mirroring how run_multi_recording_pipeline resolves contexts before building jobs.
    # In REMOTE mode (job_id provided, i.e., when dispatched by the MCP job executor), disables bootstrap persistence
    # because the prepare tool already wrote the shared configuration and per-plane runtime_data.yaml files
    # single-threaded. Skipping the per-worker re-save prevents concurrent worker threads from racing on the same
    # YAML files and producing corrupted output.
    contexts = resolve_single_recording_contexts(configuration=configuration, persist=job_id is None)
    plane_count = len(contexts)

    # Derives the tracker path from the configuration. The tracker lives under the cindra/ subdirectory, consistent
    # with where batch tools create it and where get_recording_status_tool looks for it.
    tracker_path: Path = output_path / "cindra" / SINGLE_RECORDING_TRACKER_NAME

    requested_jobs: dict[str, bool] = {
        SingleRecordingJobNames.BINARIZE: binarize,
        SingleRecordingJobNames.PROCESS: process,
        SingleRecordingJobNames.COMBINE: combine,
    }

    # If all requested job flags are False, treats them as all True (run all jobs).
    if not any(requested_jobs.values()):
        requested_jobs = dict.fromkeys(requested_jobs, True)

    jobs_to_run = [job_name for job_name, requested in requested_jobs.items() if requested]

    # Builds the universe of every valid job the pipeline could execute for this configuration. PROCESS always
    # expands to every available plane regardless of ``target_plane`` so that foreign-entry detection treats
    # the universe as a configuration fingerprint, not an invocation fingerprint.
    universe: list[tuple[str, str]] = [
        (SingleRecordingJobNames.BINARIZE, ""),
        *((SingleRecordingJobNames.PROCESS, f"plane_{plane_index}") for plane_index in range(plane_count)),
        (SingleRecordingJobNames.COMBINE, ""),
    ]

    tracker = ProcessingTracker(file_path=tracker_path)

    # Determines the execution mode and resolves job IDs accordingly.
    if job_id is not None:
        # REMOTE mode: Validates the requested job_id against the pipeline's universe, aligns the tracker with
        # every valid job so that ``start_job`` finds the requested ID, and executes the matching job.
        id_to_job: dict[str, tuple[str, str]] = {
            ProcessingTracker.generate_job_id(job_name=name, specifier=spec): (name, spec) for name, spec in universe
        }

        if job_id not in id_to_job:
            message = (
                f"Unable to execute the requested job with ID '{job_id}'. The input identifier does not match "
                f"any jobs available for the current pipeline configuration. Valid job IDs: "
                f"{sorted(id_to_job.keys())}."
            )
            console.error(message=message, error=ValueError)

        tracker.align_jobs(jobs=universe, universe=universe)

        resolved_name, resolved_specifier = id_to_job[job_id]
        _execute_single_recording_job(
            configuration=configuration,
            job_name=SingleRecordingJobNames(resolved_name),
            specifier=resolved_specifier,
            job_id=job_id,
            tracker=tracker,
        )
    else:
        # LOCAL mode: Builds all requested jobs upfront using the pre-resolved plane count, aligns the tracker
        # with the requested subset, then runs them sequentially. This mirrors the approach used by
        # run_multi_recording_pipeline.
        jobs: list[tuple[str, str]] = []
        for base_job_name in jobs_to_run:
            if base_job_name == SingleRecordingJobNames.PROCESS:
                if target_plane == -1:
                    jobs.extend((SingleRecordingJobNames.PROCESS, f"plane_{p}") for p in range(plane_count))
                else:
                    jobs.append((SingleRecordingJobNames.PROCESS, f"plane_{target_plane}"))
            else:
                jobs.append((base_job_name, ""))

        tracker.align_jobs(jobs=jobs, universe=universe)

        for name, spec in jobs:
            _execute_single_recording_job(
                configuration=configuration,
                job_name=SingleRecordingJobNames(name),
                specifier=spec,
                job_id=ProcessingTracker.generate_job_id(job_name=name, specifier=spec),
                tracker=tracker,
            )

    console.echo(message="Single-recording processing: Complete.", level=LogLevel.SUCCESS)


def execute_single_recording_job(
    configuration_path: Path,
    job_name: SingleRecordingJobNames,
    specifier: str,
    job_id: str,
    tracker: ProcessingTracker,
    *,
    persist_bootstrap: bool = False,
) -> None:
    """Executes one single-recording job and records its state on a caller-provided tracker.

    Notes:
        This is the tracker-injection entry point. Unlike run_single_recording_pipeline, it neither constructs its own
        tracker nor aligns the tracker's job universe. The caller owns the tracker, aligns it with a universe that
        contains job_id, and passes both in, so cindra stages this job into a foreign tracker whose job names and
        granularity the caller controls. The job's start, completion, and failure are recorded onto the provided
        tracker under job_id.

        The binarization and plane-processing stages re-load the shared bootstrap with persistence disabled, so it must
        already exist on disk. Enable persist_bootstrap for the first job dispatched against a configuration, the
        binarization job, which runs single-threaded and writes the bootstrap for every plane. Leave it disabled for
        the later per-plane processing jobs, which may run concurrently and read the bootstrap the binarization job
        already wrote.

    Args:
        configuration_path: The path to the single-recording configuration YAML file.
        job_name: The single-recording job to run, a member of the SingleRecordingJobNames enumeration.
        specifier: The job specifier. For a PROCESS job this encodes the plane index as 'plane_{index}', and for a
            BINARIZE or COMBINE job it is an empty string.
        job_id: The unique hexadecimal identifier under which the job's state is recorded on the provided tracker. It
            must already be present in the tracker's aligned job set.
        tracker: The caller-owned ProcessingTracker onto which this job's start, completion, or failure is recorded.
        persist_bootstrap: Determines whether to write the shared single-recording bootstrap to disk before running the
            job. Enable it only for the single-threaded binarization job that precedes plane processing.

    Raises:
        FileNotFoundError: If the configuration file is missing, is not a .yaml file, or is not a valid single-recording
            configuration.
        ValueError: If the configuration does not configure an output path, or if job_name is not a recognized
            single-recording job.
    """
    configuration, _ = _load_single_recording_configuration(configuration_path=configuration_path)

    # The binarization and plane-processing stages re-load the shared bootstrap with persistence disabled, so it must
    # exist before they run. The single-threaded binarization job opts in to write it, and later jobs rely on it.
    if persist_bootstrap:
        resolve_single_recording_contexts(configuration=configuration, persist=True)

    _execute_single_recording_job(
        configuration=configuration,
        job_name=SingleRecordingJobNames(job_name),
        specifier=specifier,
        job_id=job_id,
        tracker=tracker,
    )


def run_multi_recording_pipeline(
    configuration_path: Path,
    job_id: str | None = None,
    *,
    discover: bool = False,
    extract: bool = False,
    target_recording: str | None = None,
) -> None:
    """Executes the requested multi-recording processing pipeline steps for the target data.

    The caller is responsible for writing all runtime overrides (``recording_io.recording_directories``,
    ``runtime.parallel_workers``, ``runtime.display_progress_bars``) into the configuration file before invoking this
    function. The pipeline reads these values from the file at ``configuration_path`` and does not accept them as
    direct parameters.

    Args:
        configuration_path: The path to the multi-recording configuration YAML file. The configuration must include the
            ``recording_io.recording_directories`` list of recording paths and ``recording_io.dataset_name``.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed. If not provided, all requested jobs are run sequentially.
        discover: Determines whether to discover ROIs whose activity can be tracked across recordings (step 1).
        extract: Determines whether to extract fluorescence from the ROIs tracked across multiple recordings (step 2).
        target_recording: The unique identifier of the recording to process when running the 'extract' job. If None,
            processes all recordings.

    Raises:
        FileNotFoundError: If the multi-recording configuration data cannot be loaded from the specified file.
        ValueError: If recording validation fails, recording_directories is empty, or the specified job_id does not
            match any available jobs.
    """
    config = _load_multi_recording_configuration(configuration_path=configuration_path)

    console.echo(
        message=f"Processing {len(config.recording_io.recording_directories)} recordings for dataset "
        f"'{config.recording_io.dataset_name}'..."
    )

    # Resolves MultiRecordingRuntimeContext instances to extract recording IDs and the main recording output
    # path. This also validates that all recording directories contain valid single-recording outputs and
    # handles relocated data. In REMOTE mode (job_id provided, i.e., when dispatched by the MCP job executor),
    # disables bootstrap persistence because the prepare tool already wrote the shared configuration and every
    # recording's multi_recording_runtime_data.yaml single-threaded. Skipping the per-worker re-save prevents
    # concurrent worker threads from racing on the same YAML files and producing corrupted output.
    contexts = resolve_multi_recording_contexts(configuration=config, persist=job_id is None)
    recording_ids: list[str] = [context.runtime.io.recording_id for context in contexts]
    main_recording_path = contexts[0].runtime.output_path
    if main_recording_path is None:
        message = (
            "Unable to run the multi-recording pipeline. The main recording's "
            "output path is not configured in the resolved runtime context."
        )
        console.error(message=message, error=ValueError)

    requested_jobs: dict[str, bool] = {
        MultiRecordingJobNames.DISCOVER: discover,
        MultiRecordingJobNames.EXTRACT: extract,
    }

    # If all requested job flags are False, treats them as all True (run all jobs).
    if not any(requested_jobs.values()):
        requested_jobs = dict.fromkeys(requested_jobs, True)

    jobs_to_run = [job_name for job_name, requested in requested_jobs.items() if requested]

    # Builds the universe of every valid job the pipeline could execute for this configuration. EXTRACT always
    # expands to every resolved recording ID regardless of ``target_recording`` so that foreign-entry detection
    # treats the universe as a configuration fingerprint, not an invocation fingerprint.
    universe: list[tuple[str, str]] = [
        (MultiRecordingJobNames.DISCOVER, ""),
        *((MultiRecordingJobNames.EXTRACT, recording_id) for recording_id in recording_ids),
    ]

    tracker = ProcessingTracker(file_path=main_recording_path.joinpath(MULTI_RECORDING_TRACKER_NAME))

    # Determines the execution mode and resolves job IDs accordingly.
    if job_id is not None:
        # REMOTE mode: Validates the requested job_id against the pipeline's universe, aligns the tracker with
        # every valid job so that ``start_job`` finds the requested ID, and executes the matching job.
        id_to_job: dict[str, tuple[str, str]] = {
            ProcessingTracker.generate_job_id(job_name=name, specifier=spec): (name, spec) for name, spec in universe
        }

        if job_id not in id_to_job:
            message = (
                f"Unable to execute the requested job with ID '{job_id}'. The input identifier does not match "
                f"any jobs available for the current pipeline configuration. Valid job IDs: "
                f"{sorted(id_to_job.keys())}."
            )
            console.error(message=message, error=ValueError)

        tracker.align_jobs(jobs=universe, universe=universe)

        resolved_name, resolved_specifier = id_to_job[job_id]
        _execute_multi_recording_job(
            configuration=config,
            job_name=MultiRecordingJobNames(resolved_name),
            specifier=resolved_specifier,
            job_id=job_id,
            tracker=tracker,
        )
    else:
        # LOCAL mode: Builds all requested jobs, aligns the tracker with the requested subset, then runs
        # them sequentially. For EXTRACT jobs, expands to recording-specific jobs if target_recording is None.
        jobs: list[tuple[str, str]] = []
        for base_job_name in jobs_to_run:
            if base_job_name == MultiRecordingJobNames.EXTRACT:
                if target_recording is None:
                    jobs.extend((MultiRecordingJobNames.EXTRACT, recording_id) for recording_id in recording_ids)
                else:
                    jobs.append((MultiRecordingJobNames.EXTRACT, target_recording))
            else:
                jobs.append((base_job_name, ""))

        tracker.align_jobs(jobs=jobs, universe=universe)

        for name, spec in jobs:
            _execute_multi_recording_job(
                configuration=config,
                job_name=MultiRecordingJobNames(name),
                specifier=spec,
                job_id=ProcessingTracker.generate_job_id(job_name=name, specifier=spec),
                tracker=tracker,
            )

    console.echo(message="Multi-recording processing: Complete.", level=LogLevel.SUCCESS)


def execute_multi_recording_job(
    configuration_path: Path,
    job_name: MultiRecordingJobNames,
    specifier: str,
    job_id: str,
    tracker: ProcessingTracker,
    *,
    persist_bootstrap: bool = False,
) -> None:
    """Executes one multi-recording job and records its state on a caller-provided tracker.

    Notes:
        This is the tracker-injection entry point. Unlike run_multi_recording_pipeline, it neither constructs its own
        tracker nor aligns the tracker's job universe. The caller owns the tracker, aligns it with a universe that
        contains job_id, and passes both in, so cindra stages this job into a foreign tracker whose job names and
        granularity the caller controls. The job's start, completion, and failure are recorded onto the provided
        tracker under job_id.

        The discovery and extraction stages re-load the shared bootstrap with persistence disabled, so it must already
        exist on disk. Enable persist_bootstrap for the first job dispatched against a configuration, the discovery
        job, which runs single-threaded and writes the bootstrap for every recording. Leave it disabled for the later
        per-recording extraction jobs, which may run concurrently and read the bootstrap the discovery job already
        wrote.

    Args:
        configuration_path: The path to the multi-recording configuration YAML file.
        job_name: The multi-recording job to run, a member of the MultiRecordingJobNames enumeration.
        specifier: The job specifier. For an EXTRACT job this is the recording identifier, and for a DISCOVER job it is
            an empty string.
        job_id: The unique hexadecimal identifier under which the job's state is recorded on the provided tracker. It
            must already be present in the tracker's aligned job set.
        tracker: The caller-owned ProcessingTracker onto which this job's start, completion, or failure is recorded.
        persist_bootstrap: Determines whether to write the shared multi-recording bootstrap to disk before running the
            job. Enable it only for the single-threaded discovery job that precedes extraction.

    Raises:
        FileNotFoundError: If the configuration file is missing, is not a .yaml file, or is not a valid multi-recording
            configuration.
        ValueError: If the configuration specifies no recording directories or no dataset name, or if job_name is not a
            recognized multi-recording job.
    """
    config = _load_multi_recording_configuration(configuration_path=configuration_path)

    # The discovery and extraction stages re-load the shared bootstrap with persistence disabled, so it must exist
    # before they run. The single-threaded discovery job opts in to write it, and later extraction jobs rely on it.
    if persist_bootstrap:
        resolve_multi_recording_contexts(configuration=config, persist=True)

    _execute_multi_recording_job(
        configuration=config,
        job_name=MultiRecordingJobNames(job_name),
        specifier=specifier,
        job_id=job_id,
        tracker=tracker,
    )


def _load_single_recording_configuration(configuration_path: Path) -> tuple[SingleRecordingConfiguration, Path]:
    """Loads, validates, and runtime-configures a single-recording configuration from a YAML file.

    Notes:
        Shared by the whole-pipeline entry point and the single-job executor so both apply identical path validation,
        dataclass loading, worker-count resolution, progress configuration, and output-path checks.

    Args:
        configuration_path: The path to the single-recording configuration YAML file.

    Returns:
        A tuple of the loaded SingleRecordingConfiguration, with its worker count resolved and progress display state
        applied, and its validated output path.

    Raises:
        FileNotFoundError: If the configuration file is missing, is not a .yaml file, or is not a valid single-recording
            configuration.
        ValueError: If the configuration does not configure an output path.
    """
    # Ensures the input configuration file is valid.
    if not configuration_path.exists() or configuration_path.suffix != ".yaml":
        message = (
            f"Unable to run the single-recording cindra processing pipeline. Expected the configuration file to "
            f"end with a '.yaml' extension and exist at the specified path, but encountered: {configuration_path}."
        )
        console.error(message=message, error=FileNotFoundError)

    # Loads configuration data from the provided file.
    try:
        configuration: SingleRecordingConfiguration = SingleRecordingConfiguration.from_yaml(
            file_path=configuration_path
        )
    except Exception:
        message = (
            "Unable to run the single-recording cindra processing pipeline, as the input configuration file is not a "
            "valid single-recording pipeline configuration file. Specifically, failed to load the file's data as a "
            "SingleRecordingConfiguration dataclass instance. Ensure that the 'configuration_path' argument "
            "points to a valid single-recording configuration .yaml file."
        )
        console.error(message=message, error=FileNotFoundError)

    # Resolves the requested worker count to a valid positive integer based on available CPU cores.
    configuration.runtime.parallel_workers = resolve_worker_count(
        requested_workers=configuration.runtime.parallel_workers
    )

    # Configures the console's progress bar display state based on the configuration flag.
    if configuration.runtime.display_progress_bars:
        console.enable_progress()
    else:
        console.disable_progress()

    # Validates that the output_path is configured.
    if configuration.file_io.output_path is None:
        message = (
            "Unable to run the single-recording cindra processing pipeline. The output_path must be configured in the "
            "FileIO section of the configuration, but it is currently None."
        )
        console.error(message=message, error=ValueError)

    return configuration, configuration.file_io.output_path


def _load_multi_recording_configuration(configuration_path: Path) -> MultiRecordingConfiguration:
    """Loads, validates, and runtime-configures a multi-recording configuration from a YAML file.

    Notes:
        Shared by the whole-pipeline entry point and the single-job executor so both apply identical path validation,
        dataclass loading, worker-count resolution, progress configuration, and required-field checks.

    Args:
        configuration_path: The path to the multi-recording configuration YAML file.

    Returns:
        The loaded MultiRecordingConfiguration with its worker count resolved and progress display state applied.

    Raises:
        FileNotFoundError: If the configuration file is missing, is not a .yaml file, or is not a valid multi-recording
            configuration.
        ValueError: If the configuration specifies no recording directories or no dataset name.
    """
    # Ensures the input configuration file is valid.
    if not configuration_path.exists() or configuration_path.suffix != ".yaml":
        message = (
            f"Unable to run the multi-recording cindra processing pipeline. "
            f"Expected the configuration file to end with a '.yaml' extension and "
            f"exist at the specified path, but encountered: {configuration_path}."
        )
        console.error(message=message, error=FileNotFoundError)

    # Loads configuration data from the provided file.
    try:
        config: MultiRecordingConfiguration = MultiRecordingConfiguration.from_yaml(file_path=configuration_path)
    except Exception:
        message = (
            "Unable to run the multi-recording cindra processing pipeline, as the input configuration file is not a "
            "valid multi-recording pipeline configuration file. Specifically, failed to load the file's data as a "
            "MultiRecordingConfiguration dataclass instance. Ensure that the 'configuration_path' argument points to a "
            "valid multi-recording configuration .yaml file."
        )
        console.error(message=message, error=FileNotFoundError)

    # Validates that the configuration contains the required recording directories.
    if not config.recording_io.recording_directories:
        message = (
            "Unable to run the multi-recording cindra processing pipeline. The "
            "configuration file must specify at least two recording directories "
            "under 'recording_io.recording_directories'. The provided configuration "
            "has no recording directories specified."
        )
        console.error(message=message, error=ValueError)

    # Validates that the configuration contains a dataset name.
    if not config.recording_io.dataset_name:
        message = (
            "Unable to run the multi-recording cindra processing pipeline. The "
            "configuration file must specify a dataset name under "
            "'recording_io.dataset_name'. The provided configuration has no "
            "dataset name specified."
        )
        console.error(message=message, error=ValueError)

    # Resolves the requested worker count to a valid positive integer based on available CPU cores.
    config.runtime.parallel_workers = resolve_worker_count(requested_workers=config.runtime.parallel_workers)

    # Configures the console's progress bar display state based on the configuration flag.
    if config.runtime.display_progress_bars:
        console.enable_progress()
    else:
        console.disable_progress()

    return config


def _execute_single_recording_job(
    configuration: SingleRecordingConfiguration,
    job_name: SingleRecordingJobNames,
    specifier: str,
    job_id: str,
    tracker: ProcessingTracker,
) -> None:
    """Executes a single processing job of the single-recording pipeline.

    Args:
        configuration: The SingleRecordingConfiguration instance for the pipeline.
        job_name: The job name identifying the job to run. Must be a valid member of the
            SingleRecordingJobNames enumeration.
        specifier: The job specifier string. For PROCESS jobs, this encodes the plane index as 'plane_{index}'.
            For BINARIZE and COMBINE jobs, this is an empty string.
        job_id: The unique hexadecimal identifier for this processing job.
        tracker: The ProcessingTracker instance used to track the pipeline's runtime status.

    Raises:
        ValueError: If the job_name is not recognized.
    """
    console.echo(message=f"Running '{job_name}' job (specifier='{specifier}') with ID {job_id}...")
    tracker.start_job(job_id=job_id)

    try:
        if job_name == SingleRecordingJobNames.BINARIZE:
            binarize_recording(configuration=configuration)

        elif job_name == SingleRecordingJobNames.PROCESS:
            plane_index = int(specifier.removeprefix("plane_"))
            process_plane(configuration=configuration, plane_index=plane_index)

        elif job_name == SingleRecordingJobNames.COMBINE:
            # Validates that output_path is configured before loading contexts.
            if configuration.file_io.output_path is None:
                message = (
                    "Unable to execute the combination job. The output_path must be configured in the FileIO section "
                    "of the configuration, but it is currently None."
                )
                console.error(message=message, error=ValueError)

            # Loads contexts from disk and combines all processed planes into a dataset. Arrays are not
            # loaded automatically due to their memory footprint, so they must be loaded explicitly before
            # combining. Detection arrays provide background images; extraction arrays provide ROI statistics
            # and fluorescence traces.
            root_path = configuration.file_io.output_path / "cindra"
            contexts = RuntimeContext.load(root_path=root_path, plane_index=-1)
            if not isinstance(contexts, list):  # pragma: no cover — load with plane_index=-1 always returns a list
                contexts = [contexts]
            for context in contexts:
                # pragma justification: resolved plane contexts always carry a configured output path.
                if context.runtime.output_path is not None:  # pragma: no branch
                    context.runtime.detection.memory_map_arrays(context.runtime.output_path)
                    context.runtime.extraction.memory_map_arrays(context.runtime.output_path)
                    context.runtime.extraction.memory_map_results(context.runtime.output_path)
            save_combined_data(contexts=contexts)

        else:
            message = (
                f"Unable to execute the requested job '{job_name}' with ID '{job_id}'. The input job name is not "
                f"recognized. Use one of the valid Job names: {list(SingleRecordingJobNames)}."
            )
            console.error(message=message, error=ValueError)

        tracker.complete_job(job_id=job_id)

    except Exception as error:
        tracker.fail_job(job_id=job_id, error_message=str(error))
        raise


def _execute_multi_recording_job(
    configuration: MultiRecordingConfiguration,
    job_name: MultiRecordingJobNames,
    specifier: str,
    job_id: str,
    tracker: ProcessingTracker,
) -> None:
    """Executes a single processing job of the multi-recording pipeline.

    Args:
        configuration: The MultiRecordingConfiguration instance for the pipeline.
        job_name: The job name identifying the job to run. Must be a valid member of the
            MultiRecordingJobNames enumeration.
        specifier: The job specifier string. For EXTRACT jobs, this is the recording ID. For DISCOVER jobs, this is an
            empty string.
        job_id: The unique hexadecimal identifier for this processing job.
        tracker: The ProcessingTracker instance used to track the pipeline's runtime status.

    Raises:
        ValueError: If the job_name is not recognized.
    """
    console.echo(message=f"Running '{job_name}' job (specifier='{specifier}') with ID {job_id}...")
    tracker.start_job(job_id=job_id)

    try:
        if job_name == MultiRecordingJobNames.DISCOVER:
            discover_multi_recording_cells(configuration=configuration)

        elif job_name == MultiRecordingJobNames.EXTRACT:
            extract_multi_recording_fluorescence(configuration=configuration, recording_id=specifier)

        else:
            message = (
                f"Unable to execute the requested job '{job_name}' with ID '{job_id}'. The input job name is not "
                f"recognized. Use one of the valid Job names: {list(MultiRecordingJobNames)}."
            )
            console.error(message=message, error=ValueError)

        tracker.complete_job(job_id=job_id)

    except Exception as error:
        tracker.fail_job(job_id=job_id, error_message=str(error))
        raise
