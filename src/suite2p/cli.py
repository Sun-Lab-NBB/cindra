"""This module provides general command-line interfaces (CLIs) that are installed into the host-environment together
with the sl-suite2p library. The CLIs from this module provide a complete terminal-based interface to run all pipelines
supported by the sl-suite2p library.
"""

import ast
from typing import Any
from pathlib import Path

import click
import numpy as np
from sl_shared_assets import (
    SessionData,
    SessionLock,
    SessionTypes,
    TrackerFileNames,
    ProcessingTracker,
    AcquisitionSystems,
    generate_project_manifest,
)
from ataraxis_base_utilities import LogLevel, console

from .gui import run
from .multi_day import run_s2p_multiday, resolve_multiday_ops, discover_multiday_cells, extract_multiday_fluorescence
from .single_day import run_s2p, resolve_ops, process_plane, combine_planes, resolve_binaries
from .configuration import (
    MultiDayS2PConfiguration,
    SingleDayS2PConfiguration,
    generate_default_ops,
    generate_default_multiday_ops,
)

# Ensures that displayed CLICK help messages are formatted according to the lab standard.
CONTEXT_SETTINGS = dict(max_content_width=120)


# Defines supported Sun lab sessions and acquisition systems.
_supported_systems = tuple(AcquisitionSystems)
_supported_sessions = (SessionTypes.MESOSCOPE_EXPERIMENT,)


@click.group("ss2p", context_settings=CONTEXT_SETTINGS)
def ss2p() -> None:
    """This Command-Line Interface (CLI) functions as an entry-point for all interactions with the Sun lab's suite2p
    implementation (sl-suite2p library).
    """


@ss2p.command("gui")
def ss2p_gui() -> None:
    """Starts the sl-suite2p Graphical User Interface (GUI) application. Use this command to work with the
    single-day sl-suite2p processing pipeline via a graphical interface. At this time, the GUI does not support the
    multi-day processing pipeline.
    """
    run()


@ss2p.group("configure")
@click.option(
    "-od",
    "--output-directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The absolute path to the (existing) directory where to generate the requested configuration file.",
)
@click.option(
    "-n",
    "--name",
    type=str,
    default="single_day_sls2p_configuration",
    required=True,
    help="The name to use for the generated configuration file.",
)
@click.pass_context
def ss2p_config(ctx: Any, output_directory: Path, name: str) -> None:
    """This group provides commands for generating the sl-suite2p single-day and the multi-day processing pipeline
    configuration files.

    Commands from this group generate the configuration files which are used to run sl-suite2p processing pipelines.
    Modifying the parameters stored in the file(s) generated via this command group allows configuring all aspects of
    the target processing pipeline. Provide the path to the modified file to the 'run' CLI command group to execute the
    desired pipeline with the parameters specified inside the file.
    """
    ctx.ensure_object(dict)
    ctx.obj["file_path"] = output_directory.joinpath(name).with_suffix(".yaml")


# noinspection PyUnresolvedReferences
@ss2p_config.command("single-day")
@click.pass_context
def ss2p_sd_config(ctx: Any) -> None:
    """Generates a single-day sl-suite2p processing pipeline configuration file."""
    # Unpacks the shared parameters
    file_path = Path(ctx.obj["file_path"])

    # Generates the precursor configuration file in the specified output directory.
    precursor: SingleDayS2PConfiguration = generate_default_ops(as_dict=False)
    precursor.to_config(file_path=file_path)

    message = (
        f"Default single-day pipeline configuration file: generated in the {file_path.parent} directory. Modify "
        f"the configuration parameters in the file to finish the configuration process."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)

    message = (
        "See the original suite2p documentation (https://suite2p.readthedocs.io/en/latest/) and the Sun lab "
        "repository (https://github.com/Sun-Lab-NBB/suite2p) for more information about sl-suite2p and its "
        "configuration parameters. Note! The sun-lab suite2p library overlaps, but does not have the same "
        "configuration parameters as the original suite2p library."
    )
    console.echo(message=message, level=LogLevel.INFO)


# noinspection PyUnresolvedReferences
@ss2p_config.command("multi-day")
@click.pass_context
def ss2p_md_config(ctx: Any) -> None:
    """Generates a multi-day sl-suite2p processing pipeline configuration file."""
    # Unpacks the shared parameters
    file_path = Path(ctx.obj["file_path"])

    # Generates the precursor configuration file in the specified output directory.
    precursor: MultiDayS2PConfiguration = generate_default_multiday_ops(as_dict=False)
    precursor.to_config(file_path=file_path)

    message = (
        f"Default multi-day pipeline configuration file: generated in the {file_path.parent} directory. Modify "
        f"the configuration parameters in the file to finish the configuration process."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)

    message = (
        "See the original suite2p documentation (https://suite2p.readthedocs.io/en/latest/) and the Sun lab "
        "repository (https://github.com/Sun-Lab-NBB/suite2p) for more information about sl-suite2p and its "
        "configuration parameters. Note! The sun-lab suite2p library overlaps, but does not have the same "
        "configuration parameters as the original suite2p library."
    )
    console.echo(message=message, level=LogLevel.INFO)


@ss2p.group("run")
@click.option(
    "-i",
    "--input_path",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help=(
        "The absolute path to the configuration .yaml file that stores the runtime parameters for the target pipeline."
    ),
)
@click.option(
    "-o",
    "--overrides",
    type=str,
    default="{}",
    help=(
        "Additional processing parameters used to augment or override the parameters loaded from the configuration "
        "file. The input parameters have to be provided as a dictionary-formatted string, e.g.: "
        "{parallel_workers: 5, progress_bars: False}"
    ),
)
@click.option(
    "-w",
    "--workers",
    type=int,
    default=-1,
    help=(
        "The number of parallel workers to use when executing multiprocessing tasks. Most runtimes should set this to "
        "a value between 10 and 20. Setting this to a value of -1 or 0 makes the system use all available cores to "
        "parallelize multiprocessing tasks."
    ),
)
@click.option(
    "-pb",
    "--progress-bars",
    type=bool,
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to use progress bars during long-running tasks to visualize progress.",
)
@click.pass_context
def ss2p_run(
    ctx: Any,
    input_path: Path,
    overrides: str,
    workers: int,
    progress_bars: bool,
) -> None:
    """This group provides commands for running the single-day and multi-day sl-suite2p processing pipelines.

    Use commands from this group to execute the desired processing pipeline. Each command in the group has two versions:
    the 'general' version intended for all users and the Sun-lab-specific (sl-) version used within the Sun lab. Do not
    use the sl-specific version unless you are a Sun lab member, as it makes specific assumptions about the type and
    organization of the processed data.
    """
    # Ensures the input configuration file is valid
    if input_path.suffix != ".yaml":
        message = (
            f"Unable to run the requested suite2p processing pipeline. Expected the configuration file to end with a "
            f"'.yaml' extension, but encountered the file with extension {input_path.suffix}."
        )
        console.error(message=message, error=FileNotFoundError)

    ctx.ensure_object(dict)
    ctx.obj["config_path"] = input_path
    ctx.obj["overrides"] = overrides
    ctx.obj["workers"] = workers
    ctx.obj["progress_bars"] = progress_bars


# noinspection PyUnresolvedReferences
@ss2p_run.command("single-day")
@click.option(
    "-b",
    "--binarize",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to resolve the binary files for plane-specific processing (step 1). This step prepares "
        "the data for further processing during step 2."
    ),
)
@click.option(
    "-p",
    "--process",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to process the target plane(s) to remove motion, discover ROIs, and extract their "
        "fluorescence (step 2). This step aggregates most data processing logic of the pipeline."
    ),
)
@click.option(
    "-c",
    "--combine",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to combine processed plane data into a uniform dataset (step 2). Note, this step is "
        "required to later process the data as part of a multi-day pipeline."
    ),
)
@click.option(
    "-t",
    "--target",
    type=int,
    default=-1,
    help=(
        "The index of the plane to process. Setting this to '-1' (default value) processes all available planes "
        "sequentially."
    ),
)
@click.pass_context
def run_sd_pipeline(
    ctx: Any,
    binarize: bool,
    process: bool,
    combine: bool,
    target: int,
) -> None:
    """Runs the single-day sl-suite2p pipeline step(s) using the configuration parameters from the target file.

    This command functions as the central entry point for running all single-day sl-suite2p pipeline steps via the
    terminal. It can be flexibly configured using the parameters stored in .yaml configuration files and provided as
    manual 'overrides'.
    """
    # Extracts shared configuration parameters passed as the context dictionary.
    input_path = ctx.obj["config_path"]
    progress_bars = ctx.obj["progress_bars"]
    workers = ctx.obj["workers"]
    overrides = ctx.obj["overrides"]

    ops_path = Path()  # This variable is precreated here to appease mypy
    try:
        # Loads configuration data from the provided file.
        config: SingleDayS2PConfiguration = SingleDayS2PConfiguration.from_yaml(file_path=input_path)

        # Overrides the 'workers' and 'progress_bars' parameters with the provided values.
        config.main.progress_bars = progress_bars
        config.main.parallel_workers = workers

        # Converts the dataclass to an 'ops' dictionary instance.
        ops = config.to_ops()

    except Exception:
        # If the file cannot be loaded as the expected configuration class instance, raises an exception.
        message = (
            "Unable to run the single-day sl-suite2p processing pipeline, as the input configuration file is not a "
            "valid single-day pipeline configuration file. Specifically, failed to load the file's data as a "
            "SingleDayS2PConfiguration dataclass instance. Ensure that the 'input_path' argument points to a valid "
            "single-day configuration .yaml file."
        )
        console.error(message=message, error=FileNotFoundError)

    else:
        # Parses the override parameters as a 'db' dictionary.
        db = _parse_db(overrides)

        # Generates the ops.npy file for the runtime, using the 'ops' loaded above and additional overrides, 'db'
        # (if any)
        ops_path = resolve_ops(ops=ops, db=db)

    # Loads the resolved ops file to access the runtime configuration parameters below.
    final_ops = np.load(ops_path, allow_pickle=True).item()

    # If all three single-day steps are set to the same values, runs the entire single-day pipeline. Note, since
    # it does not make sense to call the single-day pipeline with all steps disabled, this function treats the case
    # where all are disabled the same as when all are enabled.
    if binarize == process == combine:
        run_s2p(ops_path=ops_path)
        return  # Explicit return to prevent repeating processing steps below

    # Otherwise, executes the requested single-day pipeline steps.
    if binarize:  # Step 1
        resolve_binaries(ops_path=ops_path)

    if process:  # Step 2
        # Either processes all available planes sequentially or only the requested plane
        if target != -1:
            process_plane(ops_path=ops_path, plane_index=target)
        else:
            for plane in final_ops["nplanes"]:
                process_plane(ops_path=ops_path, plane_index=plane)

    if combine:  # Step 3
        combine_planes(ops_path=ops_path)


# noinspection PyUnresolvedReferences
@ss2p_run.command("sl-single-day")
@click.option(
    "-b",
    "--binarize",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to resolve the binary files for plane-specific processing (step 1). This step prepares "
        "the data for further processing during step 2."
    ),
)
@click.option(
    "-p",
    "--process",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to process the target plane(s) to remove motion, discover ROIs, and extract their "
        "fluorescence (step 2). This step aggregates most data processing logic of the pipeline."
    ),
)
@click.option(
    "-c",
    "--combine",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to combine processed plane data into a uniform dataset (step 2). Note, this step is "
        "required to later process the data as part of a multi-day pipeline."
    ),
)
@click.option(
    "-t",
    "--target",
    type=int,
    default=-1,
    help=(
        "The index of the plane to process. Setting this to '-1' (default value) processes all available planes "
        "sequentially."
    ),
)
@click.option(
    "-sp",
    "--session-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help=(
        "The absolute path to the root session directory to process. This directory must contain the 'raw_data' "
        "subdirectory."
    ),
)
@click.option(
    "-pdr",
    "--processed-data-root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=False,
    help=(
        "The absolute path to the directory that stores the processed data from all Sun lab projects, if it is "
        "different from the root directory included in the 'session-path' argument value."
    ),
)
@click.option(
    "-id",
    "--manager-id",
    type=int,
    required=True,
    default=0,
    show_default=True,
    help="The unique identifier of the process that manages this runtime.",
)
@click.option(
    "-j",
    "--jobs",
    type=int,
    required=True,
    show_default=True,
    help=(
        "The total number of individual processing jobs to be executed as part of the target processing pipeline. "
        "This value is used to track when the processing pipeline as a whole finishes its runtime."
    ),
)
@click.option(
    "-r",
    "--reset-tracker",
    is_flag=True,
    required=False,
    help=(
        "Determines whether to forcibly reset the processing tracker file for the target session before "
        "processing runtime. This flag should only be used in exceptional cases to recover from improper runtime "
        "terminations."
    ),
)
@click.pass_context
def run_sd_pipeline_sl(
    ctx: Any,
    binarize: bool,
    process: bool,
    combine: bool,
    target: int,
    session_path: Path,
    processed_data_root: Path | None,
    jobs: int,
    manager_id: int,
    reset_tracker: bool | None,
) -> None:
    """Runs the requested single-day sl-suite2p pipeline step(s) using the configuration parameters from the target
    file.

    This command is a version of the general 'single-day' command specialized for the Sun lab data processing workflow.
    """
    # Extracts shared configuration parameters passed as the context dictionary.
    input_path = ctx.obj["config_path"]
    progress_bars = ctx.obj["progress_bars"]
    workers = ctx.obj["workers"]
    overrides = ctx.obj["overrides"]

    # Instantiates the SessionData instance for the processed session
    session_data = SessionData.load(
        session_path=session_path,
        processed_data_root=processed_data_root,
    )

    # Ensures that the session supports this type of processing
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

    ops_path = Path()  # This variable is precreated here to appease mypy
    try:
        # Loads configuration data from the provided file.
        config: SingleDayS2PConfiguration = SingleDayS2PConfiguration.from_yaml(file_path=input_path)

        # Overrides the 'workers' and 'progress_bars' parameters with the provided values.
        config.main.progress_bars = progress_bars
        config.main.parallel_workers = workers

        # Adjusts the runtime configuration to work with the Sun lab data hierarchy.
        config.file_io.save_path0 = str(session_data.processed_data.mesoscope_data_path)
        config.file_io.data_path = [str(session_data.source_data.mesoscope_data_path)]

        # Converts the dataclass to an 'ops' dictionary instance.
        ops = config.to_ops()

    except Exception:
        # If the file cannot be loaded as the expected configuration class instance, raises an exception.
        message = (
            "Unable to run the single-day sl-suite2p processing pipeline, as the input configuration file is not a "
            "valid single-day pipeline configuration file. Specifically, failed to load the file's data as a "
            "SingleDayS2PConfiguration dataclass instance. Ensure that the 'input_path' argument points to a valid "
            "single-day configuration .yaml file."
        )
        console.error(message=message, error=FileNotFoundError)

    else:
        # Parses the override parameters as a 'db' dictionary.
        db = _parse_db(overrides)

        # Generates the ops.npy file for the runtime, using the 'ops' loaded above and additional overrides, 'db'
        # (if any)
        ops_path = resolve_ops(ops=ops, db=db)

    # Loads the resolved ops file to access the runtime configuration parameters below.
    final_ops = np.load(ops_path, allow_pickle=True).item()

    # Ensures that the manager has exclusive access to the session's data.
    lock = SessionLock(file_path=session_data.tracking_data.session_lock_path)
    lock.check_owner(manager_id=manager_id)

    # Instantiates the ProcessingTracker instance for single-day suite2p processing and configures the underlying
    # tracker file to indicate that the processing is ongoing.
    tracker = ProcessingTracker(
        file_path=session_data.tracking_data.tracking_data_path.joinpath(TrackerFileNames.SUITE2P)
    )

    # If requested, resets the processing tracker before starting the runtime.
    if reset_tracker:
        tracker.abort()

    # Configures the tracker to indicate that the processing is ongoing.
    tracker.start(manager_id=manager_id, job_count=jobs)

    try:
        # If all three single-day steps are set to the same values, runs the entire single-day pipeline. Note, since
        # it does not make sense to call the single-day pipeline with all steps disabled, this function treats the case
        # where all are disabled the same as when all are enabled.
        if binarize == process == combine:
            run_s2p(ops_path=ops_path)
            return  # Explicit return to prevent repeating processing steps below

        # Otherwise, executes the requested single-day pipeline steps.
        if binarize:  # Step 1
            resolve_binaries(ops_path=ops_path)

        if process:  # Step 2
            # Either processes all available planes sequentially or only the requested plane
            if target != -1:
                process_plane(ops_path=ops_path, plane_index=target)
            else:
                for plane in final_ops["nplanes"]:
                    process_plane(ops_path=ops_path, plane_index=plane)

        if combine:  # Step 3
            combine_planes(ops_path=ops_path)

    # If the runtime encounters an error, configures the tracker to indicate that the processing was interrupted.
    except Exception:
        tracker.error(manager_id=manager_id)
        raise

    finally:
        # Configures the tracker to indicate that the processing is finished.
        if tracker.is_running:
            tracker.stop(manager_id=manager_id)

        # Updates the project manifest file to reflect the processing outcome.
        if session_data is not None:
            generate_project_manifest(
                raw_project_directory=session_data.raw_data.root_path.joinpath(session_data.project_name),
                processed_data_root=processed_data_root,
                manager_id=manager_id,
            )


# noinspection PyUnresolvedReferences
@ss2p_run.command("multi-day")
@click.option(
    "-o",
    "--output_path",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    required=False,
    help=(
        "The absolute path to the directory where to set up the output multi-day suite2p data hierarchy and save the "
        "pipeline output data."
    ),
)
@click.option(
    "-d",
    "--discover",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to run multi-day suite2p pipeline step 1 (discover cells trackable across days).",
)
@click.option(
    "-e",
    "--extract",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to run multi-day suite2p pipeline step 2 (extract fluorescence from cells tracked "
        "across days)."
    ),
)
@click.option(
    "-t",
    "--target",
    type=str,
    required=False,
    help=(
        "The unique identifier of the session to process when running the 'extract' step. If this argument is not "
        "provided, the pipeline processes all available sessions."
    ),
)
@click.pass_context
def run_md_pipeline(
    ctx: Any,
    output_path: Path | None,
    discover: bool,
    extract: bool,
    target: str | None,
) -> None:
    """Runs the multi-day sl-suite2p pipeline step(s) using the configuration parameters from the target file.

    This command functions as the central entry point for running all multi-day sl-suite2p pipeline steps via the
    terminal. It can be flexibly configured using the parameters stored in .yaml configuration files and provided as
    manual 'overrides'.
    """
    # Extracts shared configuration parameters passed as the context dictionary.
    input_path = ctx.obj["config_path"]
    progress_bars = ctx.obj["progress_bars"]
    workers = ctx.obj["workers"]
    overrides = ctx.obj["overrides"]

    ops_path = Path()  # This variable is precreated here to appease mypy
    try:
        # Loads configuration data from the provided file.
        config: MultiDayS2PConfiguration = MultiDayS2PConfiguration.from_yaml(file_path=input_path)

        # Specializes the config to work with the target data
        config.main.progress_bars = progress_bars
        config.main.parallel_workers = workers
        if output_path is not None:
            config.io.multiday_save_path = str(output_path)

        # Converts the dataclass to an 'ops' dictionary instance.
        ops = config.to_ops()

    except Exception:
        # If the file cannot be loaded as the expected configuration class instance, raises an exception.
        message = (
            "Unable to run the multi-day sl-suite2p processing pipeline, as the input configuration file is not a "
            "valid multi-day pipeline configuration file. Specifically, failed to load the file's data as a "
            "MultiDayS2PConfiguration dataclass instance. Ensure that the 'input_path' argument points to a valid "
            "multi-day configuration .yaml file."
        )
        console.error(message=message, error=FileNotFoundError)

    else:
        # Parses the override parameters as a 'db' dictionary.
        db = _parse_db(overrides)

        # Generates the ops.npy file for the runtime, using the 'ops' loaded above and additional overrides, 'db'
        # (if any)
        ops_path = resolve_multiday_ops(ops=ops, db=db)

    # Loads the resolved ops file to access the runtime configuration parameters below.
    final_ops = np.load(ops_path, allow_pickle=True).item()

    # Same idea as in the single-day pipeline: If both flags are set to the same value, this is interpreted as
    # a request to run the entire multi-day pipeline.
    if discover == extract:
        run_s2p_multiday(ops_path=ops_path)
        return

    if discover:  # Step 1
        discover_multiday_cells(ops_path=ops_path)

    if extract:  # Step 2
        # Same idea as with single-day planes, either processes all sessions sequentially or only the target session
        if target is not None:
            session_id = target
            extract_multiday_fluorescence(ops_path=ops_path, session_id=session_id)
        else:
            for session in final_ops["session_ids"]:
                extract_multiday_fluorescence(ops_path=ops_path, session_id=session)


# noinspection PyUnresolvedReferences
@ss2p_run.command("sl-multi-day")
@click.option(
    "-o",
    "--output_path",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help=(
        "The absolute path to the directory where to set up the output multi-day suite2p data hierarchy and save the "
        "pipeline output data."
    ),
)
@click.option(
    "-d",
    "--discover",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to run multi-day suite2p pipeline step 1 (discover cells trackable across days).",
)
@click.option(
    "-e",
    "--extract",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to run multi-day suite2p pipeline step 2 (extract fluorescence from cells tracked "
        "across days)."
    ),
)
@click.option(
    "-t",
    "--target",
    type=str,
    required=False,
    help=(
        "The unique identifier of the session to process when running the 'extract' step. If this argument is not "
        "provided, the pipeline processes all available sessions."
    ),
)
@click.option(
    "-sp",
    "--session-paths",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    multiple=True,
    help=(
        "The absolute path to the root directory of each sessions to process. Use multiple '-sp' arguments to provide "
        "at least two paths for each CLI call. The directory specified by each argument must contain the 'raw_data' "
        "subdirectory."
    ),
)
@click.option(
    "-pdr",
    "--processed-data-root",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=False,
    help=(
        "The absolute path to the directory that stores the processed data from all Sun lab projects, if it is "
        "different from the root directory included in the 'session-path' argument value."
    ),
)
@click.option(
    "-id",
    "--manager-id",
    type=int,
    required=True,
    default=0,
    show_default=True,
    help="The unique identifier of the process that manages this runtime.",
)
@click.option(
    "-j",
    "--jobs",
    type=int,
    required=True,
    show_default=True,
    help=(
        "The total number of individual processing jobs to be executed as part of the target processing pipeline. "
        "This value is used to track when the processing pipeline as a whole finishes its runtime."
    ),
)
@click.option(
    "-r",
    "--reset-tracker",
    is_flag=True,
    required=False,
    help=(
        "Determines whether to forcibly reset the processing tracker file for the target session before "
        "processing runtime. This flag should only be used in exceptional cases to recover from improper runtime "
        "terminations."
    ),
)
@click.pass_context
def run_md_pipeline(
    ctx: Any,
    output_path: Path,
    discover: bool,
    extract: bool,
    target: str | None,
    session_paths: list[Path],
    processed_data_root: Path,
    jobs: int,
    manager_id: int,
    reset_tracker: bool,
) -> None:
    """Runs the requested multi-day sl-suite2p pipeline step(s) using the configuration parameters from the target
    file.

    This command is a version of the general 'multi-day' command specialized for the Sun lab data processing workflow.
    """
    # Extracts shared configuration parameters passed as the context dictionary.
    input_path = ctx.obj["config_path"]
    progress_bars = ctx.obj["progress_bars"]
    workers = ctx.obj["workers"]
    overrides = ctx.obj["overrides"]

    # If the processed data is a Sun lab session, resolves additional processing configuration parameters.
    session_data: SessionData | None = None
    animal_id = ""
    session_inputs = []

    # Ensures that the user provided at least two session paths
    if len(session_paths) < 2:
        message = (
            f"Unable to run the multi-day sl-suite2p processing pipeline due to receiving an invalid number of "
            f"'session-path' inputs. The multi-day pipeline expects at least two session paths provided via the "
            f"'--session-path (-sp)' arguments, but instead encountered {len(session_paths)} inputs."
        )
        console.error(message=message, error=ValueError)

    # Loops over the target sessions and verifies that all support multi-day processing.
    for session_paths in session_paths:
        # Instantiates the SessionData instance for the processed session
        session_data = SessionData.load(
            session_path=session_paths,
            processed_data_root=processed_data_root,
        )

        # Ensures that all processed sessions belong to the same animal
        if animal_id == "":
            animal_id = session_data.animal_id
        elif animal_id != session_data.animal_id:
            message = (
                f"Unable to run the multi-day sl-suite2p processing pipeline, as the input set of sessions comes from "
                f"at least two different animals: {animal_id} and {session_data.animal_id}. Multi-day tracking "
                f"requires all sessions to be acquired from the same animal."
            )
            console.error(message=message, error=ValueError)

        # Ensures that the session supports this type of processing
        if session_data.acquisition_system not in _supported_systems:
            message = (
                f"Unable to run the multi-day suite2p pipeline for the session '{session_data.session_name}' "
                f"performed by animal '{session_data.animal_id}' for the '{session_data.project_name}' project. "
                f"The session was acquired using an unsupported acquisition system "
                f"'{session_data.acquisition_system}'. Currently, only the following acquisition systems are "
                f"supported: {', '.join(_supported_systems)}."
            )
            console.error(message=message, error=ValueError)
        if session_data.session_type not in _supported_sessions:
            message = (
                f"Unable to run the multi-day suite2p pipeline for the session '{session_data.session_name}' "
                f"performed by animal '{session_data.animal_id}' for the '{session_data.project_name}' project. "
                f"The session is of an unsupported type '{session_data.session_type}'. Currently, only the "
                f"following session types are supported: {', '.join(_supported_sessions)}."
            )
            console.error(message=message, error=ValueError)

        # Resolves and adds the path to the session's single-day suite2p-processed data folder as the input to the
        # multi-day pipeline
        session_inputs.append(str(session_data.processed_data.mesoscope_data_path))

    ops_path = Path()  # This variable is precreated here to appease mypy
    try:
        # Loads configuration data from the provided file.
        config: MultiDayS2PConfiguration = MultiDayS2PConfiguration.from_yaml(file_path=input_path)

        # Specializes the config to work with the target data
        config.main.progress_bars = progress_bars
        config.main.parallel_workers = workers
        config.io.multiday_save_path = str(output_path)
        config.io.multiday_save_folder = animal_id
        config.io.session_directories = session_inputs

    except Exception:
        # If the file cannot be loaded as the expected configuration class instance, raises an exception.
        message = (
            "Unable to run the multi-day sl-suite2p processing pipeline, as the input configuration file is not a "
            "valid multi-day pipeline configuration file. Specifically, failed to load the file's data as a "
            "MultiDayS2PConfiguration dataclass instance. Ensure that the 'input_path' argument points to a valid "
            "multi-day configuration .yaml file."
        )
        console.error(message=message, error=FileNotFoundError)

    else:
        # Converts the dataclass to an 'ops' dictionary instance.
        ops = config.to_ops()

        # Parses the override parameters as a 'db' dictionary.
        db = _parse_db(overrides)

        # Generates the ops.npy file for the runtime, using the 'ops' loaded above and additional overrides, 'db'
        # (if any)
        ops_path = resolve_multiday_ops(ops=ops, db=db)

    # Loads the resolved ops file to access the runtime configuration parameters below.
    final_ops = np.load(ops_path, allow_pickle=True).item()

    # Ensures that the manager has exclusive access to the session's data.
    lock = SessionLock(file_path=session_data.tracking_data.session_lock_path)
    lock.check_owner(manager_id=manager_id)

    # Instantiates the ProcessingTracker instance for single-day suite2p processing and configures the underlying
    # tracker file to indicate that the processing is ongoing.
    tracker = ProcessingTracker(
        file_path=session_data.tracking_data.tracking_data_path.joinpath(TrackerFileNames.MULTIDAY)
    )

    # If requested, resets the processing tracker before starting the runtime.
    if reset_tracker:
        tracker.abort()

    # Configures the tracker to indicate that the processing is ongoing.
    tracker.start(manager_id=manager_id, job_count=jobs)

    try:
        # Same idea as in the single-day pipeline: If both flags are set to the same value, this is interpreted as
        # a request to run the entire multi-day pipeline.
        if discover == extract:
            run_s2p_multiday(ops_path=ops_path)
            return

        if discover:  # Step 1
            discover_multiday_cells(ops_path=ops_path)

        if extract:  # Step 2
            # Same idea as with single-day planes, either processes all sessions sequentially or only the target session
            if target is not None:
                session_id = target
                extract_multiday_fluorescence(ops_path=ops_path, session_id=session_id)
            else:
                for session in final_ops["session_ids"]:
                    extract_multiday_fluorescence(ops_path=ops_path, session_id=session)

    # If the runtime encounters an error, configures the tracker to indicate that the processing was interrupted.
    except Exception:
        tracker.error(manager_id=manager_id)
        raise

    finally:
        # Configures the tracker to indicate that the processing is finished.
        if tracker.is_running:
            tracker.stop(manager_id=manager_id)


def _parse_db(data_string: str) -> dict[str, Any]:
    """This service function parses the value passed to the --overrides (-o) argument of the 'run' 'ss2p' CLI group
    function as a Python dictionary.

    Args:
        data_string: A string that contains the override data to be parsed.

    Returns:
        The parsed data as a dictionary compatible with the 'db' and 'ops' input arguments of the resolve_ops()
        or resolve_multiday_ops() functions.

    Raises:
        ValueError: If the input data_string cannot be parsed as a Python dictionary.
    """

    def _ensure_dict(value: Any) -> None:
        """This worker function ensures that the input value is a dictionary."""
        if not isinstance(value, dict):
            raise TypeError

    # If the user provided no overrides, returns an empty 'db' dictionary.
    if data_string == "{}":
        return {}

    try:
        # Parses the string as a Python literal
        parsed = ast.literal_eval(data_string)

        # Ensures the parsed result is a dictionary. If not, propagates the error to be evaluated by the 'try' block
        _ensure_dict(ast.literal_eval(data_string))

    except (SyntaxError, TypeError):
        message = (
            "Unable to parse the input 'overrides' argument as a python dictionary. Ensure the value of the "
            "--overrides (-o) argument is formatted like a python dictionary, "
            "e.g.: '{'key1': value1, 'key2': 'value2'}'"
        )
        console.error(message=message, error=TypeError)

        # Fallback to appease mypy, should not be reachable.
        raise TypeError(message)
    else:
        # Otherwise, returns the parsed dictionary
        return parsed
