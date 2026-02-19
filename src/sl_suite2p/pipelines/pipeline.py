"""Provides the centralized pipeline for processing the brain activity data acquired in the Sun lab."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from ataraxis_base_utilities import LogLevel, console
from ataraxis_data_structures import ProcessingTracker

from ..io import resolve_multiday_contexts
from .multi_day import discover_multiday_cells, extract_multiday_fluorescence
from .single_day import process_plane, binarize_recording, save_combined_data
from ..dataclasses import RuntimeContext, MultiDayConfiguration, SingleDayConfiguration

if TYPE_CHECKING:
    from pathlib import Path

# The tracker file name for the single-day processing pipeline.
_SINGLE_DAY_TRACKER_NAME: str = "single_day_tracker"

# The tracker file name for the multi-day processing pipeline.
_MULTI_DAY_TRACKER_NAME: str = "multi_day_tracker"


class SingleDayJobNames(StrEnum):
    """Defines the job names for the single-day processing pipeline components."""

    BINARIZE = "binarization"
    """The name for the binarization (step 1) processing job."""
    PROCESS = "processing"
    """The generic name for the processing (step 2) processing job. During runtime, the processed plane is identified
    by the tracker's specifier field using the format 'plane_{plane_index}'."""
    COMBINE = "combination"
    """The name for the combination (step 3) processing job."""


class MultiDayJobNames(StrEnum):
    """Defines the job names for the multi-day processing pipeline components."""

    DISCOVER = "discovery"
    """The name for the cell discovery (step 1) processing job."""
    EXTRACT = "extraction"
    """The generic name for the fluorescence extraction (step 2) processing job. During runtime, the processed session
    is identified by the tracker's specifier field, which stores the session ID string."""


def _execute_single_day_job(
    configuration: SingleDayConfiguration,
    job_name: SingleDayJobNames | str,
    specifier: str,
    job_id: str,
    tracker: ProcessingTracker,
) -> None:
    """Executes a single processing job of the single-day pipeline.

    Args:
        configuration: The SingleDayConfiguration instance for the pipeline.
        job_name: The job name identifying the job to run. Accepts a SingleDayJobNames enum member or an equivalent
            string value.
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
        if job_name == SingleDayJobNames.BINARIZE:
            binarize_recording(configuration=configuration)

        elif job_name == SingleDayJobNames.PROCESS:
            plane_index = int(specifier.removeprefix("plane_"))
            process_plane(configuration=configuration, plane_index=plane_index)

        elif job_name == SingleDayJobNames.COMBINE:
            # Validates that save_path is configured before loading contexts.
            if configuration.file_io.save_path is None:
                message = (
                    "Unable to execute the combination job. The save_path must be configured in the FileIO section "
                    "of the configuration, but it is currently None."
                )
                console.error(message=message, error=ValueError)

            # Loads contexts from disk and combines all processed planes into a dataset.
            root_path = configuration.file_io.save_path / "suite2p"
            contexts = RuntimeContext.load(root_path=root_path, plane_index=-1)
            if not isinstance(contexts, list):
                contexts = [contexts]
            save_combined_data(contexts=contexts)

        else:
            message = (
                f"Unable to execute the requested job '{job_name}' with ID '{job_id}'. The input job name is not "
                f"recognized. Use one of the valid Job names: {list(SingleDayJobNames)}."
            )
            console.error(message=message, error=ValueError)

        tracker.complete_job(job_id=job_id)

    except Exception:
        tracker.fail_job(job_id=job_id)
        raise


def _execute_multi_day_job(
    configuration: MultiDayConfiguration,
    job_name: MultiDayJobNames | str,
    specifier: str,
    job_id: str,
    tracker: ProcessingTracker,
) -> None:
    """Executes a single processing job of the multi-day pipeline.

    Args:
        configuration: The MultiDayConfiguration instance for the pipeline.
        job_name: The job name identifying the job to run. Accepts a MultiDayJobNames enum member or an equivalent
            string value.
        specifier: The job specifier string. For EXTRACT jobs, this is the session ID. For DISCOVER jobs, this is an
            empty string.
        job_id: The unique hexadecimal identifier for this processing job.
        tracker: The ProcessingTracker instance used to track the pipeline's runtime status.

    Raises:
        ValueError: If the job_name is not recognized.
    """
    console.echo(message=f"Running '{job_name}' job (specifier='{specifier}') with ID {job_id}...")
    tracker.start_job(job_id=job_id)

    try:
        if job_name == MultiDayJobNames.DISCOVER:
            discover_multiday_cells(configuration=configuration)

        elif job_name == MultiDayJobNames.EXTRACT:
            extract_multiday_fluorescence(configuration=configuration, session_id=specifier)

        else:
            message = (
                f"Unable to execute the requested job '{job_name}' with ID '{job_id}'. The input job name is not "
                f"recognized. Use one of the valid Job names: {list(MultiDayJobNames)}."
            )
            console.error(message=message, error=ValueError)

        tracker.complete_job(job_id=job_id)

    except Exception:
        tracker.fail_job(job_id=job_id)
        raise


def process_single_day(
    configuration_path: Path,
    job_id: str | None = None,
    *,
    binarize: bool = False,
    process: bool = False,
    combine: bool = False,
    target_plane: int = -1,
    workers: int = -1,
    progress_bars: bool = False,
) -> None:
    """Processes the brain activity data recorded during the target data acquisition session using the single-day
    processing pipeline.

    Args:
        configuration_path: The path to the single-day configuration YAML file.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed. If not provided, all requested jobs are run sequentially.
        binarize: Determines whether to resolve the binary files for plane-specific processing (step 1).
        process: Determines whether to process the target plane(s) to remove motion, discover ROIs, and extract their
            fluorescence (step 2).
        combine: Determines whether to combine processed plane data into a uniform dataset (step 3).
        target_plane: The index of the plane to process. Setting this to '-1' (default value) processes all available
            planes sequentially.
        workers: The number of parallel workers to use when processing the data. Setting this to '-1' (default value)
            uses all available CPU cores.
        progress_bars: Determines whether to show progress bars during processing.

    Raises:
        FileNotFoundError: If the single-day configuration data cannot be loaded from the specified file.
        ValueError: If session's data validation fails or the specified job_id does not match any available jobs.
    """
    # Ensures the input configuration file is valid.
    if not configuration_path.exists() or configuration_path.suffix != ".yaml":
        message = (
            f"Unable to run the single-day sl-suite2p processing pipeline. Expected the configuration file to end with "
            f"a '.yaml' extension and exist at the specified path, but encountered: {configuration_path}."
        )
        console.error(message=message, error=FileNotFoundError)

    # Loads configuration data from the provided file.
    try:
        configuration: SingleDayConfiguration = SingleDayConfiguration.from_yaml(file_path=configuration_path)
    except Exception:
        message = (
            "Unable to run the single-day sl-suite2p processing pipeline, as the input configuration file is not a "
            "valid single-day pipeline configuration file. Specifically, failed to load the file's data as a "
            "SingleDayConfiguration dataclass instance. Ensure that the 'configuration_path' argument points to a "
            "valid single-day configuration .yaml file."
        )
        console.error(message=message, error=FileNotFoundError)

    # Overrides the 'workers' and 'progress_bars' parameters with the provided values.
    configuration.runtime.display_progress_bars = progress_bars
    configuration.runtime.parallel_workers = workers

    # Validates that the save_path is configured.
    if configuration.file_io.save_path is None:
        message = (
            "Unable to run the single-day sl-suite2p processing pipeline. The save_path must be configured in the "
            "FileIO section of the configuration, but it is currently None."
        )
        console.error(message=message, error=ValueError)

    # Derives the tracker path from the configuration.
    tracker_path: Path = configuration.file_io.save_path / _SINGLE_DAY_TRACKER_NAME

    # Determines which jobs to run based on the flags.
    requested_jobs: dict[str, bool] = {
        SingleDayJobNames.BINARIZE: binarize,
        SingleDayJobNames.PROCESS: process,
        SingleDayJobNames.COMBINE: combine,
    }

    # If all requested job flags are False, treats them as all True (run all jobs).
    if not any(requested_jobs.values()):
        requested_jobs = dict.fromkeys(requested_jobs, True)

    # Builds the list of jobs to run.
    jobs_to_run = [job_name for job_name, requested in requested_jobs.items() if requested]

    # Initializes the tracker instance.
    tracker = ProcessingTracker(file_path=tracker_path)

    # Determines the execution mode and resolves job IDs accordingly.
    if job_id is not None:
        # REMOTE mode: Retrieves the job name and specifier directly from the tracker using the provided job_id.
        job_info = tracker.get_job_info(job_id=job_id)
        _execute_single_day_job(
            configuration=configuration,
            job_name=SingleDayJobNames(job_info.job_name),
            specifier=job_info.specifier,
            job_id=job_id,
            tracker=tracker,
        )
    else:
        # LOCAL mode: Runs BINARIZE first (if requested) to determine the plane count, then expands and runs the
        # remaining jobs.

        # Runs BINARIZE first if requested, as it determines the number of planes.
        if SingleDayJobNames.BINARIZE in jobs_to_run:
            console.echo(message="Initializing the processing tracker with BINARIZE job...")
            binarize_ids = tracker.initialize_jobs(jobs=[(SingleDayJobNames.BINARIZE, "")])
            _execute_single_day_job(
                configuration=configuration,
                job_name=SingleDayJobNames.BINARIZE,
                specifier="",
                job_id=binarize_ids[0],
                tracker=tracker,
            )

        # Builds the list of remaining jobs (excluding BINARIZE which already ran).
        remaining_jobs = [job for job in jobs_to_run if job != SingleDayJobNames.BINARIZE]

        if remaining_jobs:
            # Loads contexts to determine the number of planes after binarization.
            root_path = configuration.file_io.save_path / "suite2p"
            contexts = RuntimeContext.load(root_path=root_path, plane_index=-1)
            if not isinstance(contexts, list):
                contexts = [contexts]
            plane_count = len(contexts)

            # Builds (job_name, specifier) tuples for the remaining jobs. Expands PROCESS jobs to plane-specific
            # jobs if target_plane == -1.
            jobs: list[tuple[str, str]] = []
            for base_job_name in remaining_jobs:
                if base_job_name == SingleDayJobNames.PROCESS:
                    if target_plane == -1:
                        jobs.extend((SingleDayJobNames.PROCESS, f"plane_{p}") for p in range(plane_count))
                    else:
                        jobs.append((SingleDayJobNames.PROCESS, f"plane_{target_plane}"))
                else:
                    jobs.append((base_job_name, ""))

            # Adds remaining jobs to the tracker. The initialize_jobs method only adds jobs that don't already
            # exist, so this safely extends the tracker initialized with BINARIZE above.
            console.echo(message=f"Adding {len(jobs)} remaining job(s) to the processing tracker...")
            job_ids = tracker.initialize_jobs(jobs=jobs)

            for (name, spec), jid in zip(jobs, job_ids, strict=True):
                _execute_single_day_job(
                    configuration=configuration,
                    job_name=SingleDayJobNames(name),
                    specifier=spec,
                    job_id=jid,
                    tracker=tracker,
                )

    console.echo(message="Single-day processing: Complete.", level=LogLevel.SUCCESS)


def process_multi_day(
    configuration_path: Path,
    job_id: str | None = None,
    *,
    discover: bool = False,
    extract: bool = False,
    target_session: str | None = None,
    workers: int = -1,
    progress_bars: bool = False,
) -> None:
    """Processes the brain activity data from cells tracked across multiple sessions using the multi-day
    processing pipeline.

    Notes:
        Sessions are specified directly in the configuration file's `session_io.session_directories` field. The
        sessions are natural-sorted, and the first session becomes the 'main session' which stores the processing
        tracker file.

    Args:
        configuration_path: The path to the multi-day configuration YAML file. The configuration must include the
            `session_io.session_directories` list of session paths and `session_io.dataset_name`.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed. If not provided, all requested jobs are run sequentially.
        discover: Determines whether to discover cells whose activity can be tracked across days (step 1).
        extract: Determines whether to extract fluorescence from the cells tracked across multiple days (step 2).
        target_session: The unique identifier of the session to process when running the 'extract' job. If None,
            processes all sessions.
        workers: The number of parallel workers to use when processing the data. Setting this to '-1' (default value)
            uses all available CPU cores.
        progress_bars: Determines whether to show progress bars during processing.

    Raises:
        FileNotFoundError: If the multi-day configuration data cannot be loaded from the specified file.
        ValueError: If session validation fails, session_directories is empty, or the specified job_id does not match
            any available jobs.
    """
    # Ensures the input configuration file is valid.
    if not configuration_path.exists() or configuration_path.suffix != ".yaml":
        message = (
            f"Unable to run the multi-day sl-suite2p processing pipeline. Expected the configuration file to end with "
            f"a '.yaml' extension and exist at the specified path, but encountered: {configuration_path}."
        )
        console.error(message=message, error=FileNotFoundError)

    # Loads configuration data from the provided file.
    try:
        config: MultiDayConfiguration = MultiDayConfiguration.from_yaml(file_path=configuration_path)
    except Exception:
        message = (
            "Unable to run the multi-day sl-suite2p processing pipeline, as the input configuration file is not a "
            "valid multi-day pipeline configuration file. Specifically, failed to load the file's data as a "
            "MultiDayConfiguration dataclass instance. Ensure that the 'configuration_path' argument points to a "
            "valid multi-day configuration .yaml file."
        )
        console.error(message=message, error=FileNotFoundError)

    # Validates that the configuration contains the required session directories.
    if not config.session_io.session_directories:
        message = (
            "Unable to run the multi-day sl-suite2p processing pipeline. The configuration file must specify at least "
            "two session directories under 'session_io.session_directories'. The provided configuration has no session "
            "directories specified."
        )
        console.error(message=message, error=ValueError)

    # Validates that the configuration contains a dataset name.
    if not config.session_io.dataset_name:
        message = (
            "Unable to run the multi-day sl-suite2p processing pipeline. The configuration file must specify a dataset "
            "name under 'session_io.dataset_name'. The provided configuration has no dataset name specified."
        )
        console.error(message=message, error=ValueError)

    # Overrides the 'workers' and 'progress_bars' parameters with the provided values.
    config.runtime.display_progress_bars = progress_bars
    config.runtime.parallel_workers = workers

    console.echo(
        message=f"Processing {len(config.session_io.session_directories)} sessions for dataset "
        f"'{config.session_io.dataset_name}'..."
    )

    # Resolves MultiDayRuntimeContext instances to extract session IDs and the main session output path. This also
    # validates that all session directories contain valid single-day outputs.
    contexts = resolve_multiday_contexts(configuration=config)
    session_ids: list[str] = [ctx.runtime.io.session_id for ctx in contexts]
    main_session_path = contexts[0].runtime.output_path
    if main_session_path is None:
        message = (
            "Unable to run the multi-day pipeline. The main session's output path is not configured in the resolved "
            "runtime context."
        )
        console.error(message=message, error=ValueError)

    # Determines which jobs to run based on the flags.
    requested_jobs: dict[str, bool] = {
        MultiDayJobNames.DISCOVER: discover,
        MultiDayJobNames.EXTRACT: extract,
    }

    # If all requested job flags are False, treats them as all True (run all jobs).
    if not any(requested_jobs.values()):
        requested_jobs = dict.fromkeys(requested_jobs, True)

    # Builds the list of jobs to run.
    jobs_to_run = [job_name for job_name, requested in requested_jobs.items() if requested]

    # Determines the execution mode and resolves job IDs accordingly.
    if job_id is not None:
        # REMOTE mode: Retrieves the job name and specifier directly from the tracker using the provided job_id.
        tracker = ProcessingTracker(file_path=main_session_path.joinpath(_MULTI_DAY_TRACKER_NAME))
        job_info = tracker.get_job_info(job_id=job_id)
        _execute_multi_day_job(
            configuration=config,
            job_name=MultiDayJobNames(job_info.job_name),
            specifier=job_info.specifier,
            job_id=job_id,
            tracker=tracker,
        )
    else:
        # LOCAL mode: Initializes the tracker and runs all requested jobs. For EXTRACT jobs, expands to
        # session-specific jobs if target_session is None.
        jobs: list[tuple[str, str]] = []
        for base_job_name in jobs_to_run:
            if base_job_name == MultiDayJobNames.EXTRACT:
                if target_session is None:
                    jobs.extend((MultiDayJobNames.EXTRACT, session_id) for session_id in session_ids)
                else:
                    jobs.append((MultiDayJobNames.EXTRACT, target_session))
            else:
                jobs.append((base_job_name, ""))

        console.echo(message=f"Initializing the processing tracker for {len(jobs)} job(s)...")
        tracker = ProcessingTracker(file_path=main_session_path.joinpath(_MULTI_DAY_TRACKER_NAME))
        job_ids = tracker.initialize_jobs(jobs=jobs)

        for (name, spec), jid in zip(jobs, job_ids, strict=True):
            _execute_multi_day_job(
                configuration=config,
                job_name=MultiDayJobNames(name),
                specifier=spec,
                job_id=jid,
                tracker=tracker,
            )

    console.echo(message="Multi-day processing: Complete.", level=LogLevel.SUCCESS)
