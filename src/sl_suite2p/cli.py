"""This module provides the command-line interfaces (CLIs) that are installed into the host-environment together
with the sl-suite2p library. The CLIs from this module provide a complete terminal-based interface for running all
pipelines supported by the sl-suite2p library.
"""

import ast
from typing import Any
from pathlib import Path

import click
from ataraxis_base_utilities import LogLevel, console

from .gui import run
from .pipeline import process_multi_day, process_single_day
from .configuration import (
    MultiDayS2PConfiguration,
    SingleDayS2PConfiguration,
    generate_default_ops,
    generate_default_multiday_ops,
)

# Ensures that displayed CLICK help messages are formatted according to the lab standard.
CONTEXT_SETTINGS = dict(max_content_width=120)


@click.group("ss2p", context_settings=CONTEXT_SETTINGS)
def ss2p() -> None:
    """This Command-Line Interface (CLI) functions as an entry-point for all interactions with the Sun lab's suite2p
    implementation (sl-suite2p library).
    """


@ss2p.command("gui")
def ss2p_gui() -> None:
    """Starts the sl-suite2p Graphical User Interface (GUI) application.

    Use this command to work with the single-day sl-suite2p processing pipeline via a graphical interface. At this
    time, the GUI does not support the multi-day processing pipeline.
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
    """Generates the single-day or the multi-day processing pipeline configuration file.

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
    """Generates the single-day sl-suite2p processing pipeline configuration file."""
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
    """Generates the multi-day sl-suite2p processing pipeline configuration file."""
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
    """Runs the single-day or multi-day sl-suite2p processing pipeline."""
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
    "-sp",
    "--session-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The absolute path to the root data directory of the session to process.",
)
@click.option(
    "-id",
    "--job-id",
    type=str,
    required=False,
    default=None,
    help=(
        "The unique hexadecimal identifier for this processing job. If provided, runs only the matching job "
        "(remote mode)."
    ),
)
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
        "Determines whether to combine processed plane data into a uniform dataset (step 3). Note, this step is "
        "required to later process the data as part of a multi-day pipeline."
    ),
)
@click.option(
    "-t",
    "--target",
    type=int,
    default=-1,
    help=(
        "The index of the plane to process when running the PROCESS step (2). Setting this to '-1' (default value) "
        "processes all available planes sequentially."
    ),
)
@click.pass_context
def run_sd_pipeline(
    ctx: Any,
    session_path: Path,
    job_id: str | None,
    binarize: bool,
    process: bool,
    combine: bool,
    target: int,
) -> None:
    """Runs the requested single-day pipeline step(s)."""
    # Extracts shared configuration parameters passed as the context dictionary.
    config_path = ctx.obj["config_path"]
    progress_bars = ctx.obj["progress_bars"]
    workers = ctx.obj["workers"]
    overrides = ctx.obj["overrides"]

    # Parses the override parameters as a dictionary.
    db = _parse_db(overrides)

    # Calls the unified pipeline API.
    process_single_day(
        configuration_path=config_path,
        session_path=session_path,
        job_id=job_id,
        binarize=binarize,
        process=process,
        combine=combine,
        target_plane=target,
        workers=workers,
        progress_bars=progress_bars,
        overrides=db,
    )


# noinspection PyUnresolvedReferences
@ss2p_run.command("multi-day")
@click.option(
    "-dp",
    "--dataset-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The absolute path to the root data directory of the dataset to process.",
)
@click.option(
    "-id",
    "--job-id",
    type=str,
    required=False,
    default=None,
    help=(
        "The unique hexadecimal identifier for this processing job. If provided, runs only the matching job "
        "(remote mode)."
    ),
)
@click.option(
    "-d",
    "--discover",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to discover cells trackable across days (step 1). This step discovers the candidates for "
        "the fluorescence extraction performed during the second processing step."
    ),
)
@click.option(
    "-e",
    "--extract",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to extract the fluorescence from cells tracked across days, identified during the first "
        "processing step."
    ),
)
@click.option(
    "-t",
    "--target",
    type=str,
    required=False,
    help=(
        "The unique identifier of the sessions to process when running the 'extract' step. If this argument is not "
        "provided, the pipeline processes all available sessions."
    ),
)
@click.option(
    "-a",
    "--animal",
    type=str,
    required=False,
    default=None,
    help=(
        "The unique identifier of the animal whose sessions to process. If not provided and the dataset contains "
        "multiple animals, each animal's sessions are processed sequentially as separate multi-day runs."
    ),
)
@click.pass_context
def run_md_pipeline(
    ctx: Any,
    dataset_path: Path,
    job_id: str | None,
    discover: bool,
    extract: bool,
    target: str | None,
    animal: str | None,
) -> None:
    """Runs the requested multi-day pipeline step(s)."""
    # Extracts shared configuration parameters passed as the context dictionary.
    config_path = ctx.obj["config_path"]
    progress_bars = ctx.obj["progress_bars"]
    workers = ctx.obj["workers"]
    overrides = ctx.obj["overrides"]

    # Parses the override parameters as a dictionary.
    db = _parse_db(overrides)

    # Calls the unified pipeline API.
    process_multi_day(
        configuration_path=config_path,
        dataset_path=dataset_path,
        job_id=job_id,
        discover=discover,
        extract=extract,
        target_session=target,
        target_animal=animal,
        workers=workers,
        progress_bars=progress_bars,
        overrides=db,
    )


def _parse_db(data_string: str) -> dict[str, Any]:
    """Parses the value passed to the --overrides (-o) argument of the 'run' 'ss2p' CLI group function as a Python
    dictionary.

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
