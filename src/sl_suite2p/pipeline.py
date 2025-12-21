"""This module provides the centralized pipeline for processing the brain activity data acquired in the Sun lab. The
pipeline supports both local and remote processing modes.
"""

from enum import StrEnum
from typing import Any
from pathlib import Path

import numpy as np
from ataraxis_base_utilities import LogLevel, console
from sl_shared_assets import SessionData, ProcessingTracker, ProcessingTrackers, SessionTypes, AcquisitionSystems

from .single_day import resolve_ops, resolve_binaries, process_plane, combine_planes
from .configuration import SingleDayS2PConfiguration


# Defines the session types and acquisition systems currently supported by the processing pipeline.
_supported_systems = tuple(AcquisitionSystems)
_supported_sessions = (SessionTypes.MESOSCOPE_EXPERIMENT,)


class SingleDayJobNames(StrEnum):
    """Defines the job names for the single-day suite2p processing pipeline components."""

    BINARIZE = "single_day_binarize"
    """The name for the binarization step (step 1) processing job."""
    PROCESS = "single_day_process"
    """The name for the processing step (step 2) processing job."""
    COMBINE = "single_day_combine"
    """The name for the combination step (step 3) processing job."""


def _generate_single_day_job_ids(session_path: Path, job_names: list[str]) -> dict[str, str]:
    """Generates unique processing job identifiers for the specified single-day pipeline jobs.

    Args:
        session_path: The path to the processed session's data directory.
        job_names: The list of job names for which to generate the IDs.

    Returns:
        A dictionary mapping job names to their generated job IDs.
    """
    job_ids: dict[str, str] = {}
    for job_name in job_names:
        job_ids[job_name] = ProcessingTracker.generate_job_id(session_path=session_path, job_name=job_name)
    return job_ids


def _initialize_single_day_processing_tracker(
    session_path: Path,
    job_names: list[str],
) -> dict[str, str]:
    """Initializes the processing tracker file for the single-day suite2p pipeline using the requested job IDs.

    Notes:
        This function is used to process the data in the 'local' processing mode. During remote data processing, the
        tracker file is pre-generated before submitting the processing jobs to the remote compute server.

    Args:
        session_path: The path to the session's data directory.
        job_names: The names for the processing jobs to track.

    Returns:
        A dictionary mapping job names to their generated job IDs.
    """
    session = SessionData.load(session_path=session_path)

    # Initializes the processing tracker for this pipeline.
    tracker = ProcessingTracker(
        file_path=session.tracking_data.tracking_data_path.joinpath(ProcessingTrackers.SUITE2P)
    )

    # Generates job IDs for each requested job.
    job_ids = _generate_single_day_job_ids(session_path=session_path, job_names=job_names)

    # Initializes all jobs in the tracker file.
    tracker.initialize_jobs(job_ids=list(job_ids.values()))

    return job_ids


def _execute_single_day_job(
    ops_path: Path,
    job_name: str,
    job_id: str,
    target_plane: int,
    tracker: ProcessingTracker,
) -> None:
    """Executes a single processing job of the single-day suite2p pipeline with tracker management.

    Args:
        ops_path: The path to the ops.npy file that stores the single-day suite2p processing parameters.
        job_name: The name of the job to run.
        job_id: The unique hexadecimal identifier for this processing job.
        target_plane: The plane index to process (-1 for all planes).
        tracker: The ProcessingTracker instance used to track the pipeline's runtime status.

    Raises:
        ValueError: If the job_name is not recognized.
    """
    console.echo(message=f"Running job: {job_name} (ID: {job_id})...")
    tracker.start_job(job_id=job_id)

    try:
        if job_name == SingleDayJobNames.BINARIZE:
            resolve_binaries(ops_path=ops_path)

        elif job_name == SingleDayJobNames.PROCESS:
            # Loads the resolved ops file to access the runtime configuration parameters.
            final_ops = np.load(ops_path, allow_pickle=True).item()

            # Either processes all available planes sequentially or only the requested plane.
            if target_plane != -1:
                process_plane(ops_path=ops_path, plane_index=target_plane)
            else:
                for plane in range(final_ops["nplanes"]):
                    process_plane(ops_path=ops_path, plane_index=plane)

        elif job_name == SingleDayJobNames.COMBINE:
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
        session_path: The path to the processed session's root data directory.
        job_id: The unique hexadecimal identifier for the processing job to execute. If provided, only the job
            matching this ID is executed. If not provided, all requested jobs are run sequentially with automatic
            tracker management.
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
        config: SingleDayS2PConfiguration = SingleDayS2PConfiguration.from_yaml(file_path=configuration_path)
    except Exception:
        message = (
            "Unable to run the single-day sl-suite2p processing pipeline, as the input configuration file is not a "
            "valid single-day pipeline configuration file. Specifically, failed to load the file's data as a "
            "SingleDayS2PConfiguration dataclass instance. Ensure that the 'configuration_path' argument points to a "
            "valid single-day configuration .yaml file."
        )
        console.error(message=message, error=FileNotFoundError)
        return  # Fallback to appease mypy

    # Overrides the 'workers' and 'progress_bars' parameters with the provided values.
    config.main.progress_bars = progress_bars
    config.main.parallel_workers = workers

    # Instantiates the SessionData instance for the processed session.
    session_data = SessionData.load(session_path=session_path)

    # Ensures that the session supports this type of processing.
    if session_data.acquisition_system not in _supported_systems:
        message = (
            f"Unable to specialize the single-day sl-suite2p configuration file for the session "
            f"'{session_data.session_name}' performed by animal '{session_data.animal_id}' for the "
            f"'{session_data.project_name}' project. The session was acquired using an unsupported acquisition "
            f"system '{session_data.acquisition_system}'. Currently, only the following acquisition systems are "
            f"supported: {', '.join(_supported_systems)}."
        )
        console.error(message=message, error=ValueError)
    if session_data.session_type not in _supported_sessions:
        message = (
            f"Unable to run the single-day suite2p pipeline for the session '{session_data.session_name}' "
            f"performed by animal '{session_data.animal_id}' for the '{session_data.project_name}' project. The "
            f"session is of an unsupported type '{session_data.session_type}'. Currently, only the following "
            f"session types are supported: {', '.join(_supported_sessions)}."
        )
        console.error(message=message, error=ValueError)

    # Adjusts the runtime configuration to work with the Sun lab data hierarchy.
    config.file_io.save_path0 = str(session_data.processed_data.mesoscope_data_path)
    config.file_io.data_path = [str(session_data.raw_data.mesoscope_data_path)]

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

    # Determines the execution mode and resolves job IDs accordingly.
    if job_id is not None:

        # REMOTE mode: Finds the job name matching the provided job_id.
        all_job_ids = _generate_single_day_job_ids(session_path=session_path, job_names=list(SingleDayJobNames))
        id_to_name: dict[str, str] = {v: k for k, v in all_job_ids.items()}

        if job_id not in id_to_name:
            tracker.fail_job(job_id=job_id)
            message = (
                f"Unable to execute the requested job with ID '{job_id}'. The input identifier does not match any "
                f"jobs available for this session. Use one of the valid job IDs: {list(all_job_ids.values())}."
            )
            console.error(message=message, error=ValueError)

        # Runs the jobs whose id matches the target job_id.
        job_name = id_to_name[job_id]
        _execute_single_day_job(
            ops_path=ops_path, job_name=job_name, job_id=job_id, target_plane=target_plane, tracker=tracker
        )
    else:

        # LOCAL mode: Initializes the tracker and runs all requested jobs.
        console.echo(message=f"Initializing the processing tracker for {len(jobs_to_run)} job(s)...")
        job_ids = _initialize_single_day_processing_tracker(session_path=session_path, job_names=jobs_to_run)

        for job_name in jobs_to_run:
            _execute_single_day_job(
                ops_path=ops_path, job_name=job_name, job_id=job_ids[job_name], target_plane=target_plane,
                tracker=tracker
            )

    console.echo(message="All processing jobs completed successfully.", level=LogLevel.SUCCESS)
