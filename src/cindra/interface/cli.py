"""This module provides the command-line interfaces (CLIs) that are installed into the host-environment together
with the cindra library. The CLIs from this module provide a complete terminal-based interface for running all
pipelines supported by the cindra library.
"""

from pathlib import Path

import click
from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console

from ..pipelines import run_multi_day_pipeline, run_single_day_pipeline
from .mcp_server import run_server
from ..dataclasses import MultiDayConfiguration, SingleDayConfiguration

# Ensures that displayed CLICK help messages are formatted according to the lab standard.
CONTEXT_SETTINGS = {"max_content_width": 120}


@click.group("cindra", context_settings=CONTEXT_SETTINGS)
def cindra_cli() -> None:
    """This Command-Line Interface (CLI) functions as an entry-point for all interactions with the Sun lab's suite2p
    implementation (cindra library).
    """


@cindra_cli.command("mcp")
@click.option(
    "-t",
    "--transport",
    type=click.Choice(["stdio", "sse", "streamable-http"]),
    default="stdio",
    show_default=True,
    help="The transport protocol to use for MCP communication.",
)
def cindra_mcp(transport: str) -> None:
    """Starts the Model Context Protocol (MCP) server for agentic neural imaging data processing.

    The MCP server exposes tools that enable AI agents to discover sessions, execute pipelines,
    monitor processing status, and manage batch operations for both single-day and multi-day workflows.
    """
    run_server(transport=transport)  # type: ignore[arg-type]


@cindra_cli.group("configure")
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
    default="single_day_cindra_configuration",
    required=True,
    help="The name to use for the generated configuration file.",
)
@click.pass_context
def cindra_config(ctx: click.Context, output_directory: Path, name: str) -> None:
    """Generates the single-day or the multi-day processing pipeline configuration file.

    Commands from this group generate the configuration files for running processing pipelines.
    Modifying the parameters stored in the file(s) generated via this command group allows configuring all aspects of
    the target processing pipeline. Provide the path to the modified file to the 'run' CLI command group to execute the
    desired pipeline with the parameters specified inside the file.
    """
    ctx.ensure_object(dict)
    ctx.obj["file_path"] = output_directory.joinpath(name).with_suffix(".yaml")


# noinspection PyUnresolvedReferences
@cindra_config.command("single-day")
@click.pass_context
def cindra_sd_conf(ctx: click.Context) -> None:
    """Generates the single-day cindra processing pipeline configuration file."""
    # Unpacks the shared parameters
    file_path = Path(ctx.obj["file_path"])

    # Generates the precursor configuration file in the specified output directory.
    config = SingleDayConfiguration()
    config.save(file_path=file_path)

    message = (
        f"Default single-day pipeline configuration file: generated in the {file_path.parent} directory. Modify "
        f"the configuration parameters in the file to finish the configuration process."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)

    message = (
        "See the original suite2p documentation (https://suite2p.readthedocs.io/en/latest/) and the Sun lab "
        "repository (https://github.com/Sun-Lab-NBB/suite2p) for more information about cindra and its "
        "configuration parameters. Note! The sun-lab suite2p library overlaps, but does not have the same "
        "configuration parameters as the original suite2p library."
    )
    console.echo(message=message, level=LogLevel.INFO)


# noinspection PyUnresolvedReferences
@cindra_config.command("multi-day")
@click.pass_context
def cindra_md_conf(ctx: click.Context) -> None:
    """Generates the multi-day cindra processing pipeline configuration file."""
    # Unpacks the shared parameters
    file_path = Path(ctx.obj["file_path"])

    # Generates the precursor configuration file in the specified output directory.
    config = MultiDayConfiguration()
    config.save(file_path=file_path)

    message = (
        f"Default multi-day pipeline configuration file: generated in the {file_path.parent} directory. Modify "
        f"the configuration parameters in the file to finish the configuration process."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)

    message = (
        "See the original suite2p documentation (https://suite2p.readthedocs.io/en/latest/) and the Sun lab "
        "repository (https://github.com/Sun-Lab-NBB/suite2p) for more information about cindra and its "
        "configuration parameters. Note! The sun-lab suite2p library overlaps, but does not have the same "
        "configuration parameters as the original suite2p library."
    )
    console.echo(message=message, level=LogLevel.INFO)


@cindra_cli.group("run")
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
def cindra_run(
    ctx: click.Context,
    input_path: Path,
    workers: int,
    progress_bars: bool,
) -> None:
    """Runs the single-day or multi-day cindra processing pipeline."""
    # Ensures the input configuration file is valid.
    if input_path.suffix != ".yaml":
        message = (
            f"Unable to run the requested suite2p processing pipeline. Expected the configuration file to end with a "
            f"'.yaml' extension, but encountered the file with extension {input_path.suffix}."
        )
        console.error(message=message, error=FileNotFoundError)

    ctx.ensure_object(dict)
    ctx.obj["config_path"] = input_path
    ctx.obj["workers"] = workers
    ctx.obj["progress_bars"] = progress_bars


# noinspection PyUnresolvedReferences
@cindra_run.command("single-day")
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
@click.option(
    "-d",
    "--data-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=False,
    default=None,
    help="Overrides the configuration's file_io.data_path with the specified directory.",
)
@click.option(
    "-s",
    "--save-path",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    required=False,
    default=None,
    help="Overrides the configuration's file_io.save_path with the specified directory.",
)
@click.pass_context
def run_sd_pipeline(
    ctx: click.Context,
    job_id: str | None,
    binarize: bool,
    process: bool,
    combine: bool,
    target: int,
    data_path: Path | None,
    save_path: Path | None,
) -> None:
    """Runs the requested single-day pipeline step(s)."""
    # Extracts shared configuration parameters passed as the context dictionary.
    config_path: Path = ctx.obj["config_path"]
    progress_bars: bool = ctx.obj["progress_bars"]
    workers: int = ctx.obj["workers"]

    # Writes CLI overrides into the configuration file before running the pipeline.
    configuration = SingleDayConfiguration.from_yaml(file_path=config_path)
    configuration.runtime.parallel_workers = workers
    configuration.runtime.display_progress_bars = progress_bars
    if data_path is not None:
        configuration.file_io.data_path = data_path
    if save_path is not None:
        configuration.file_io.save_path = save_path
    configuration.save(file_path=config_path)

    run_single_day_pipeline(
        configuration_path=config_path,
        job_id=job_id,
        binarize=binarize,
        process=process,
        combine=combine,
        target_plane=target,
    )


# noinspection PyUnresolvedReferences
@cindra_run.command("multi-day")
@click.option(
    "-sp",
    "--session-path",
    "session_paths",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    multiple=True,
    required=True,
    help=(
        "The absolute path to a session directory to include in multi-day processing. Specify this option multiple "
        "times to include multiple sessions (at least two required)."
    ),
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
@click.pass_context
def run_md_pipeline(
    ctx: click.Context,
    session_paths: tuple[Path, ...],
    job_id: str | None,
    discover: bool,
    extract: bool,
    target: str | None,
) -> None:
    """Runs the requested multi-day pipeline step(s)."""
    # Extracts shared configuration parameters passed as the context dictionary.
    config_path: Path = ctx.obj["config_path"]
    progress_bars: bool = ctx.obj["progress_bars"]
    workers: int = ctx.obj["workers"]

    # Writes CLI overrides into the configuration file before running the pipeline.
    configuration = MultiDayConfiguration.from_yaml(file_path=config_path)
    configuration.session_io.session_directories = tuple(natsorted(session_paths))
    configuration.runtime.parallel_workers = workers
    configuration.runtime.display_progress_bars = progress_bars
    configuration.save(file_path=config_path)

    run_multi_day_pipeline(
        configuration_path=config_path,
        job_id=job_id,
        discover=discover,
        extract=extract,
        target_session=target,
    )
