"""This module provides the centralized pipeline for processing the brain activity data acquired in the Sun lab. The
pipeline supports both local and remote processing modes.
"""

from enum import StrEnum
from typing import Any
from pathlib import Path

import numpy as np
from sl_shared_assets import (
    SessionData,
    SessionTypes,
    ProcessingTracker,
    AcquisitionSystems,
    ProcessingTrackers,
)
from ataraxis_base_utilities import LogLevel, console

from .multi_day import resolve_multiday_ops, discover_multiday_cells, extract_multiday_fluorescence
from .single_day import resolve_ops, process_plane, combine_planes, resolve_binaries
from .configuration import MultiDayConfiguration, SingleDayConfiguration

# Defines the session types and acquisition systems currently supported by the processing pipeline.
_supported_systems = tuple(AcquisitionSystems)
_supported_sessions = (SessionTypes.MESOSCOPE_EXPERIMENT,)


def get_session_root(session: SessionData) -> Path:
    """Returns the canonical session root path for consistent job ID generation.

    The session root is the parent directory of the raw_data directory. This path format matches what sl-forgery uses
    when submitting jobs to the remote compute server, ensuring consistent job IDs across local and remote processing.

    Args:
        session: The loaded SessionData instance.

    Returns:
        The canonical session root path (parent of raw_data).
    """
    return session.raw_data.raw_data_path.parent


class SingleDayJobNames(StrEnum):
    """Defines the job names for the single-day suite2p processing pipeline components."""

    BINARIZE = "ss2p_binarization"
    """The name for the binarization (step 1) processing job."""
    PROCESS = "ss2p_processing"
    """The generic name for the processing (step 2) processing job. During runtime, the name is further modified to
    reference the processed plane using the format 'ss2p_processing_plane_{plane_index}'."""
    COMBINE = "ss2p_combination"
    """The name for the combination (step 3) processing job."""


class MultiDayJobNames(StrEnum):
    """Defines the job names for the multi-day suite2p processing pipeline components."""

    DISCOVER = "ss2p_discovery"
    """The name for the cell discovery (step 1) processing job."""
    EXTRACT = "ss2p_extraction"
    """The generic name for the fluorescence extraction (step 2) processing job. During runtime, the name is further
    modified to reference the processed session using the format 'ss2p_extraction_session_{session_name}'."""


def _generate_job_ids(root_path: Path, data_name: str, base_job_names: list[str]) -> dict[str, str]:
    """Generates unique processing job identifiers for the specified pipeline jobs.

    Args:
        root_path: The path to the root directory that stores the data to be processed by the chosen pipeline
            (session path for single-day, dataset path for multi-day).
        data_name: The unique identifier of the data being processed (session name for single-day, dataset name
            for multi-day).
        base_job_names: The list of base job names for which to generate the IDs.

    Returns:
        A dictionary mapping full job names (with data name prefix) to their generated job IDs.
    """
    job_ids: dict[str, str] = {}
    for base_job_name in base_job_names:
        full_job_name = f"{data_name}_{base_job_name}"
        job_ids[full_job_name] = ProcessingTracker.generate_job_id(session_path=root_path, job_name=full_job_name)
    return job_ids


def _initialize_single_day_processing_tracker(
    session_path: Path,
    session_name: str,
    base_job_names: list[str],
) -> dict[str, str]:
    """Initializes the processing tracker file for the single-day suite2p pipeline using the requested job IDs.

    Notes:
        This function is used to process the data in the 'local' processing mode. During remote data processing, the
        tracker file is pre-generated before submitting the processing jobs to the remote compute server.

    Args:
        session_path: The path to the session's data directory.
        session_name: The unique identifier of the session being processed.
        base_job_names: The base job names for the processing jobs to track.

    Returns:
        A dictionary mapping full job names (with session prefix) to their generated job IDs.
    """
    session = SessionData.load(session_path=session_path)

    # Initializes the processing tracker for this pipeline.
    tracker = ProcessingTracker(file_path=session.tracking_data.tracking_data_path.joinpath(ProcessingTrackers.SUITE2P))

    # Generates job IDs for each requested job.
    job_ids = _generate_job_ids(root_path=session_path, data_name=session_name, base_job_names=base_job_names)

    # Initializes all jobs in the tracker file.
    tracker.initialize_jobs(job_ids=list(job_ids.values()))

    return job_ids


def _execute_single_day_job(
    ops_path: Path,
    job_name: str,
    job_id: str,
    tracker: ProcessingTracker,
) -> None:
    """Executes a single processing job of the single-day suite2p pipeline.

    Args:
        ops_path: The path to the ops.npy file that stores the single-day suite2p processing parameters.
        job_name: The name of the job to run.
        job_id: The unique hexadecimal identifier for this processing job.
        tracker: The ProcessingTracker instance used to track the pipeline's runtime status.

    Raises:
        ValueError: If the job_name is not recognized.
    """
    console.echo(message=f"Running {job_name} job with ID {job_id}...")
    tracker.start_job(job_id=job_id)

    try:
        if job_name.endswith(SingleDayJobNames.BINARIZE):
            resolve_binaries(ops_path=ops_path)

        elif f"_{SingleDayJobNames.PROCESS}_plane_" in job_name:
            process_plane(ops_path=ops_path, plane_index=int(job_name.split("_plane_")[1]))

        elif job_name.endswith(SingleDayJobNames.COMBINE):
            combine_planes(ops_path=ops_path)

        else:
            message = (
                f"Unable to execute the requested job {job_name} with ID '{job_id}'. The input job name is not "
                f"recognized. Use one of the valid Job names: {list(SingleDayJobNames)}."
            )
            console.error(message=message, error=ValueError)

        tracker.complete_job(job_id=job_id)

    except Exception:
        tracker.fail_job(job_id=job_id)
        raise


def process_single_day(
    configuration_path: Path,
    session_path: Path,
    job_id: str | None = None,
    *,
    binarize: bool = False,
    process: bool = False,
    combine: bool = False,
    target_plane: int = -1,
    workers: int = -1,
    progress_bars: bool = False,
    overrides: dict[str, Any] | None = None,
) -> None:
    """Processes the brain activity data recorded during the target data acquisition session using the single-day
    suite2p processing pipeline.

    Args:
        configuration_path: The path to the single-day configuration YAML file.
        session_path: The path to the root data directory of the session to process.
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
        overrides: An optional dictionary of configuration parameter overrides. This allows dynamically overriding the
            configuration parameters loaded from the .YAML file before executing the processing.

    Raises:
        FileNotFoundError: If the single-day configuration data cannot be loaded from the specified file.
        ValueError: If session's data validation fails or the specified job_id does not match any available jobs.
    """
    # Ensures the input configuration file is valid.
    if not configuration_path.exists() or configuration_path.suffix != ".yaml":
        message = (
            f"Unable to run the single-day suite2p processing pipeline. Expected the configuration file to end with a "
            f"'.yaml' extension and exist at the specified path, but encountered: {configuration_path}."
        )
        console.error(message=message, error=FileNotFoundError)

    # Loads configuration data from the provided file.
    try:
        config: SingleDayConfiguration = SingleDayConfiguration.from_yaml(file_path=configuration_path)
    except Exception:
        message = (
            "Unable to run the single-day sl-suite2p processing pipeline, as the input configuration file is not a "
            "valid single-day pipeline configuration file. Specifically, failed to load the file's data as a "
            "SingleDayConfiguration dataclass instance. Ensure that the 'configuration_path' argument points to a "
            "valid single-day configuration .yaml file."
        )
        console.error(message=message, error=FileNotFoundError)
        return  # Fallback to appease mypy

    # Overrides the 'workers' and 'progress_bars' parameters with the provided values.
    config.main.progress_bars = progress_bars
    config.main.parallel_workers = workers

    # Instantiates the SessionData instance for the processed session.
    session_data = SessionData.load(session_path=session_path)

    # Normalizes session_path to the canonical session root for consistent job ID generation.
    # This ensures job IDs match regardless of whether the user passed raw_data or session root paths.
    session_path = get_session_root(session=session_data)

    # Ensures that the session supports this type of processing.
    if session_data.acquisition_system not in _supported_systems:
        message = (
            f"Unable to specialize the single-day sl-suite2p configuration file for the session "
            f"{session_data.session_name} performed by animal {session_data.animal_id} for the "
            f"{session_data.project_name} project. The session was acquired using an unsupported acquisition "
            f"system {session_data.acquisition_system}. Currently, only the following acquisition systems are "
            f"supported: {', '.join(_supported_systems)}."
        )
        console.error(message=message, error=ValueError)
    if session_data.session_type not in _supported_sessions:
        message = (
            f"Unable to run the single-day suite2p pipeline for the session {session_data.session_name} "
            f"performed by animal {session_data.animal_id} for the {session_data.project_name} project. The "
            f"session is of an unsupported type {session_data.session_type}. Currently, only the following "
            f"session types are supported: {', '.join(_supported_sessions)}."
        )
        console.error(message=message, error=ValueError)

    # Adjusts the runtime configuration to work with the Sun lab data hierarchy.
    config.file_io.save_path = str(session_data.processed_data.mesoscope_data_path)
    config.file_io.data_path = str(session_data.raw_data.mesoscope_data_path)

    # Converts the dataclass to an 'ops' dictionary instance.
    ops = config.to_ops()

    # Parses the override parameters as a 'db' dictionary.
    db = overrides if overrides is not None else {}

    # Generates the ops.npy file for the runtime.
    ops_path = resolve_ops(ops=ops, db=db)

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
    tracker = ProcessingTracker(
        file_path=session_data.tracking_data.tracking_data_path.joinpath(ProcessingTrackers.SUITE2P)
    )

    # Extracts the session name for job naming.
    session_name = session_data.session_name

    # Determines the execution mode and resolves job IDs accordingly.
    if job_id is not None:
        # REMOTE mode: Finds the job name matching the provided job_id.
        # Generates all possible base job names including plane-specific PROCESS jobs.
        final_ops = np.load(ops_path, allow_pickle=True).item()
        all_base_job_names: list[str] = [SingleDayJobNames.BINARIZE, SingleDayJobNames.COMBINE]
        for plane in range(final_ops["nplanes"]):
            all_base_job_names.append(f"{SingleDayJobNames.PROCESS}_plane_{plane}")

        all_job_ids = _generate_job_ids(
            root_path=session_path, data_name=session_name, base_job_names=all_base_job_names
        )
        id_to_name: dict[str, str] = {v: k for k, v in all_job_ids.items()}

        if job_id not in id_to_name:
            tracker.fail_job(job_id=job_id)
            message = (
                f"Unable to execute the requested job with ID '{job_id}'. The input identifier does not match any "
                f"jobs available for this session. Use one of the valid job IDs: {list(all_job_ids.values())}."
            )
            console.error(message=message, error=ValueError)

        # Runs the job whose id matches the target job_id.
        job_name = id_to_name[job_id]
        _execute_single_day_job(ops_path=ops_path, job_name=job_name, job_id=job_id, tracker=tracker)
    else:
        # LOCAL mode: Runs BINARIZE first (if requested) to determine nplanes, then expands and runs remaining jobs.

        # Initializes the tracker and runs BINARIZE first if requested, as it determines the number of planes.
        if SingleDayJobNames.BINARIZE in jobs_to_run:
            console.echo(message="Initializing the processing tracker with BINARIZE job...")
            binarize_job_ids = _initialize_single_day_processing_tracker(
                session_path=session_path, session_name=session_name, base_job_names=[SingleDayJobNames.BINARIZE]
            )
            full_binarize_name = f"{session_name}_{SingleDayJobNames.BINARIZE}"
            _execute_single_day_job(
                ops_path=ops_path,
                job_name=full_binarize_name,
                job_id=binarize_job_ids[full_binarize_name],
                tracker=tracker,
            )

        # Builds the list of remaining jobs (excluding BINARIZE which already ran).
        remaining_jobs = [job for job in jobs_to_run if job != SingleDayJobNames.BINARIZE]

        if remaining_jobs:
            # Reloads ops to get the correct nplanes after binarization.
            final_ops = np.load(ops_path, allow_pickle=True).item()

            # Expands PROCESS jobs to plane-specific jobs if target_plane == -1.
            expanded_jobs: list[tuple[str, int]] = []
            for base_job_name in remaining_jobs:
                if base_job_name == SingleDayJobNames.PROCESS and target_plane == -1:
                    for plane in range(final_ops["nplanes"]):
                        expanded_jobs.append((base_job_name, plane))
                else:
                    expanded_jobs.append((base_job_name, target_plane))

            # Generates base job names for tracking (plane-specific for PROCESS jobs).
            base_job_names_for_tracking = [
                f"{base_job_name}_plane_{plane}" if base_job_name == SingleDayJobNames.PROCESS else base_job_name
                for base_job_name, plane in expanded_jobs
            ]

            # Adds remaining jobs to the tracker. The initialize_jobs method only adds jobs that don't already exist,
            # so this safely extends the tracker initialized with BINARIZE above.
            console.echo(message=f"Adding {len(expanded_jobs)} remaining job(s) to the processing tracker...")
            job_ids = _initialize_single_day_processing_tracker(
                session_path=session_path, session_name=session_name, base_job_names=base_job_names_for_tracking
            )

            for base_job_name in base_job_names_for_tracking:
                full_job_name = f"{session_name}_{base_job_name}"
                _execute_single_day_job(
                    ops_path=ops_path,
                    job_name=full_job_name,
                    job_id=job_ids[full_job_name],
                    tracker=tracker,
                )

    console.echo(message="Single-day processing: Complete.", level=LogLevel.SUCCESS)


def _initialize_multi_day_processing_tracker(
    main_session_path: Path,
    dataset_name: str,
    base_job_names: list[str],
) -> dict[str, str]:
    """Initializes the processing tracker file for the multi-day suite2p pipeline using the requested job IDs.

    Notes:
        This function is used to process the data in the 'local' processing mode. During remote data processing, the
        tracker file is pre-generated before submitting the processing jobs to the remote compute server.

        The tracker is stored in the main session's multiday output folder (the first session after natural sorting).

    Args:
        main_session_path: The path to the main session's multiday output folder.
        dataset_name: The unique identifier of the dataset being processed.
        base_job_names: The base job names for the processing jobs to track.

    Returns:
        A dictionary mapping full job names (with dataset prefix) to their generated job IDs.
    """
    # Initializes the processing tracker for this pipeline. The tracker is stored in the main session's
    # multiday output folder.
    tracker = ProcessingTracker(file_path=main_session_path.joinpath("multiday_tracker.json"))

    # Generates job IDs for each requested job using the main session path.
    job_ids = _generate_job_ids(root_path=main_session_path, data_name=dataset_name, base_job_names=base_job_names)

    # Initializes all jobs in the tracker file.
    tracker.initialize_jobs(job_ids=list(job_ids.values()))

    return job_ids


def _execute_multi_day_job(
    ops_path: Path,
    job_name: str,
    job_id: str,
    tracker: ProcessingTracker,
) -> None:
    """Executes a single processing job of the multi-day suite2p pipeline.

    Args:
        ops_path: The path to the ops.npy file that stores the multi-day suite2p processing parameters.
        job_name: The name of the job to run.
        job_id: The unique hexadecimal identifier for this processing job.
        tracker: The ProcessingTracker instance used to track the pipeline's runtime status.

    Raises:
        ValueError: If the job_name is not recognized.
    """
    console.echo(message=f"Running {job_name} job with ID {job_id}...")
    tracker.start_job(job_id=job_id)

    try:
        if job_name.endswith(MultiDayJobNames.DISCOVER):
            discover_multiday_cells(ops_path=ops_path)

        elif f"_{MultiDayJobNames.EXTRACT}_session_" in job_name:
            extract_multiday_fluorescence(
                ops_path=ops_path, session_id=job_name.split(f"{MultiDayJobNames.EXTRACT}_session_")[1]
            )

        else:
            message = (
                f"Unable to execute the requested job {job_name} with ID '{job_id}'. The input job name is not "
                f"recognized. Use one of the valid Job names: {list(MultiDayJobNames)}."
            )
            console.error(message=message, error=ValueError)

        tracker.complete_job(job_id=job_id)

    except Exception:
        tracker.fail_job(job_id=job_id)
        raise


def process_multi_day(
    configuration_path: Path,
    job_id: str | None = None,
    *,
    discover: bool = False,
    extract: bool = False,
    target_session: str | None = None,
    workers: int = -1,
    progress_bars: bool = False,
    overrides: dict[str, Any] | None = None,
) -> None:
    """Processes the brain activity data from cells tracked across multiple sessions using the multi-day suite2p
    processing pipeline.

    Notes:
        Sessions are specified directly in the configuration file's `io.session_directories` field. The sessions are
        natural-sorted, and the first session becomes the 'main session' which stores the processing tracker file.

    Args:
        configuration_path: The path to the multi-day configuration YAML file. The configuration must include the
            `io.session_directories` list of session paths and `io.dataset_name`.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed. If not provided, all requested jobs are run sequentially.
        discover: Determines whether to discover cells whose activity can be tracked across days (step 1).
        extract: Determines whether to extract fluorescence from the cells tracked across multiple days (step 2).
        target_session: The unique identifier of the session to process when running the 'extract' job. If None,
            processes all sessions.
        workers: The number of parallel workers to use when processing the data. Setting this to '-1' (default value)
            uses all available CPU cores.
        progress_bars: Determines whether to show progress bars during processing.
        overrides: An optional dictionary of configuration parameter overrides. This allows dynamically overriding the
            configuration parameters loaded from the .YAML file before executing the processing.

    Raises:
        FileNotFoundError: If the multi-day configuration data cannot be loaded from the specified file.
        ValueError: If session validation fails, session_directories is empty, or the specified job_id does not match
            any available jobs.
    """
    # Ensures the input configuration file is valid.
    if not configuration_path.exists() or configuration_path.suffix != ".yaml":
        message = (
            f"Unable to run the multi-day suite2p processing pipeline. Expected the configuration file to end with a "
            f"'.yaml' extension and exist at the specified path, but encountered: {configuration_path}."
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
        return  # Fallback to appease mypy

    # Validates that the configuration contains the required session directories.
    if not config.io.session_directories:
        message = (
            "Unable to run the multi-day suite2p processing pipeline. The configuration file must specify at least "
            "two session directories under 'io.session_directories'. The provided configuration has no session "
            "directories specified."
        )
        console.error(message=message, error=ValueError)
        return  # Fallback to appease mypy

    # Validates that the configuration contains a dataset name.
    if not config.io.dataset_name:
        message = (
            "Unable to run the multi-day suite2p processing pipeline. The configuration file must specify a dataset "
            "name under 'io.dataset_name'. The provided configuration has no dataset name specified."
        )
        console.error(message=message, error=ValueError)
        return  # Fallback to appease mypy

    # Overrides the 'workers' and 'progress_bars' parameters with the provided values.
    config.main.progress_bars = progress_bars
    config.main.parallel_workers = workers

    console.echo(f"Processing {len(config.io.session_directories)} sessions for dataset '{config.io.dataset_name}'...")

    # Converts the dataclass to an 'ops' dictionary instance.
    ops = config.to_ops()

    # Parses the override parameters as a 'db' dictionary.
    db = overrides if overrides is not None else {}

    # Generates the ops.npy file for the runtime. This returns the path to ops.npy in the main session's
    # multiday output folder.
    ops_path = resolve_multiday_ops(ops=ops, db=db)

    # Loads the resolved ops file to access the session_ids. The main session path is the parent of ops_path
    # since resolve_multiday_ops returns the path to ops.npy in the main session's multiday folder.
    final_ops = np.load(ops_path, allow_pickle=True).item()
    session_ids: list[str] = final_ops["session_ids"]
    main_session_path = ops_path.parent
    dataset_name: str = final_ops["dataset_name"]

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
        # REMOTE mode: Finds the job name matching the provided job_id.
        # Initializes the tracker from the main session path.
        tracker = ProcessingTracker(file_path=main_session_path.joinpath("multiday_tracker.json"))

        # Generates all possible base job names including session-specific EXTRACT jobs.
        all_base_job_names: list[str] = [MultiDayJobNames.DISCOVER]
        for session_id in session_ids:
            all_base_job_names.append(f"{MultiDayJobNames.EXTRACT}_session_{session_id}")

        all_job_ids = _generate_job_ids(
            root_path=main_session_path, data_name=dataset_name, base_job_names=all_base_job_names
        )
        id_to_name: dict[str, str] = {v: k for k, v in all_job_ids.items()}

        if job_id not in id_to_name:
            tracker.fail_job(job_id=job_id)
            message = (
                f"Unable to execute the requested job with ID '{job_id}'. The input identifier does not match any "
                f"jobs available for this dataset. Use one of the valid job IDs: {list(all_job_ids.values())}."
            )
            console.error(message=message, error=ValueError)

        # Runs the job whose id matches the target job_id.
        job_name = id_to_name[job_id]
        _execute_multi_day_job(
            ops_path=ops_path,
            job_name=job_name,
            job_id=job_id,
            tracker=tracker,
        )
    else:
        # LOCAL mode: Initializes the tracker and runs all requested jobs.
        # For EXTRACT jobs, expands to session-specific jobs if target_session is None.
        expanded_jobs: list[tuple[str, str | None]] = []  # (base_job_name, session_id) pairs
        for base_job_name in jobs_to_run:
            if base_job_name == MultiDayJobNames.EXTRACT and target_session is None:
                for session_id in session_ids:
                    expanded_jobs.append((base_job_name, session_id))
            else:
                expanded_jobs.append((base_job_name, target_session))

        # Generates base job names for tracking (session-specific for EXTRACT jobs).
        base_job_names_for_tracking = [
            f"{base_job_name}_session_{session}" if base_job_name == MultiDayJobNames.EXTRACT else base_job_name
            for base_job_name, session in expanded_jobs
        ]

        console.echo(message=f"Initializing the processing tracker for {len(expanded_jobs)} job(s)...")
        tracker = ProcessingTracker(file_path=main_session_path.joinpath("multiday_tracker.json"))
        job_ids = _initialize_multi_day_processing_tracker(
            main_session_path=main_session_path, dataset_name=dataset_name, base_job_names=base_job_names_for_tracking
        )

        for base_job_name in base_job_names_for_tracking:
            full_job_name = f"{dataset_name}_{base_job_name}"
            _execute_multi_day_job(
                ops_path=ops_path,
                job_name=full_job_name,
                job_id=job_ids[full_job_name],
                tracker=tracker,
            )

    console.echo(message="Multi-day processing: Complete.", level=LogLevel.SUCCESS)
