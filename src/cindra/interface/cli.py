"""Provides the terminal-based interface for running all processing pipelines supported by the library."""

from pathlib import Path

import click
from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console

from ..io import discover_recordings
from ..pipelines import run_multi_day_pipeline, run_single_day_batch, run_single_day_pipeline
from .mcp_server import run_server
from ..dataclasses import PipelineType, MultiDayConfiguration, SingleDayConfiguration, detect_pipeline_type

CONTEXT_SETTINGS = {"max_content_width": 120}
"""The Click context settings that ensure displayed help messages are formatted according to the Sun lab standard."""


@click.group("cindra", context_settings=CONTEXT_SETTINGS)
def cindra_cli() -> None:
    """This Command-Line Interface (CLI) is the entry-point for all headless interactions with the cindra library."""


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

    The MCP server exposes tools that enable AI agents to discover recording data, execute pipelines,
    monitor processing status, and manage batch operations for both single-day and multi-day workflows.
    """
    run_server(transport=transport)  # type: ignore[arg-type]


@cindra_cli.command("configure")
@click.option(
    "-p",
    "--pipeline",
    type=click.Choice(["single-day", "sd", "multi-day", "md"], case_sensitive=False),
    required=True,
    help="The type of processing pipeline to generate the configuration file for.",
)
@click.option(
    "-od",
    "--output-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="The absolute path to the (existing) directory where to generate the requested configuration file.",
)
@click.option(
    "-n",
    "--name",
    type=str,
    required=False,
    default=None,
    help="The name to use for the generated configuration file. Defaults to 'cindra_sd_conf' or 'cindra_md_conf'.",
)
def cindra_config(pipeline: str, output_path: Path, name: str | None) -> None:
    """Generates the configuration file for the specified processing pipeline.

    Modifying the parameters stored in the generated file allows configuring all aspects of the target processing
    pipeline. Provide the path to the modified file to the 'run' CLI command group to execute the desired pipeline
    with the parameters specified inside the file.
    """
    # Normalizes shorthand aliases and resolves pipeline-specific parameters.
    single_day = pipeline in ("single-day", "sd")
    resolved_name = name if name is not None else ("cindra_sd_conf" if single_day else "cindra_md_conf")
    file_path = output_path.joinpath(resolved_name).with_suffix(".yaml")

    # Generates the precursor configuration file in the specified output directory.
    config = SingleDayConfiguration() if single_day else MultiDayConfiguration()
    config.save(file_path=file_path)

    message = (
        f"Default {'single-day' if single_day else 'multi-day'} pipeline configuration file: generated in the "
        f"{file_path.parent} directory. Modify the configuration parameters in the file to finish the configuration "
        f"process."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)

    message = (
        "See the cindra repository (https://github.com/Sun-Lab-NBB/cindra) for more information about cindra and "
        "its configuration parameters."
    )
    console.echo(message=message, level=LogLevel.INFO)


@cindra_cli.command("run")
@click.option(
    "-i",
    "--input-path",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="The absolute path to the configuration .yaml file for the executed pipeline.",
)
@click.option(
    "-w",
    "--workers",
    type=int,
    default=-1,
    help=(
        "The number of parallel workers to use when executing multiprocessing tasks. For machines with a large number "
        "of cores a value between 10 and 20 is optimal. Setting this to a value of -1 or 0 makes the system use "
        "all available cores to parallelize multiprocessing tasks."
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
@click.option(
    "-id",
    "--job-id",
    type=str,
    required=False,
    default=None,
    help=(
        "The unique hexadecimal identifier for this processing job. If provided, the pipeline type is inferred from "
        "the configuration file and only the matching job is executed (remote mode). All step flags are ignored."
    ),
)
@click.option(
    "-b",
    "--binarize",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "[Single-day] Determines whether to resolve the binary files for plane-specific processing (step 1). "
        "This step prepares the data for further processing during step 2."
    ),
)
@click.option(
    "-p",
    "--process",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "[Single-day] Determines whether to process the target plane(s) to remove motion, discover ROIs, and extract "
        "their fluorescence (step 2). This step aggregates most data processing logic of the pipeline."
    ),
)
@click.option(
    "-c",
    "--combine",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "[Single-day] Determines whether to combine processed plane data into a uniform dataset (step 3). Note, this "
        "step is required to later process the data as part of a multi-day pipeline."
    ),
)
@click.option(
    "-tp",
    "--target-plane",
    type=int,
    default=-1,
    help=(
        "[Single-day] The index of the plane to process when running the PROCESS step (2). Setting this to '-1' "
        "(default value) processes all available planes sequentially."
    ),
)
@click.option(
    "-dp",
    "--data-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=False,
    default=None,
    help=(
        "[Single-day] The path to the root directory that stores the processed recording's data. When provided, "
        "this path overrides the matching field in the pipeline's configuration file."
    ),
)
@click.option(
    "-s",
    "--output-path",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    required=False,
    default=None,
    help=(
        "[Single-day] The path to the root directory where to create the cindra's output hierarchy and store the "
        "processed data. When provided, this path overrides the matching field in the pipeline's configuration file."
    ),
)
@click.option(
    "-d",
    "--discover",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "[Multi-day] Determines whether to discover ROIs trackable across days (recordings) (step 1). This step "
        "discovers the candidates for the fluorescence extraction performed during the second processing step."
    ),
)
@click.option(
    "-e",
    "--extract",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "[Multi-day] Determines whether to extract the fluorescence from ROIs tracked across days, identified "
        "during the first processing step."
    ),
)
@click.option(
    "-tr",
    "--target-recording",
    type=str,
    required=False,
    default=None,
    help=(
        "[Multi-day] The unique identifier of the recording session to process when running the 'extract' step. If "
        "this argument is not provided, the pipeline processes all available recordings in the dataset."
    ),
)
@click.option(
    "-rp",
    "--recording-path",
    "recording_paths",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    multiple=True,
    required=False,
    help=(
        "[Multi-day] The path to the recording processed with the single-day cindra pipeline to include in the "
        "processed multi-day dataset. Specify this option multiple times to include multiple recording sessions "
        "(at least two required). When provided, these paths override the matching fields in the pipeline's "
        "configuration file."
    ),
)
def cindra_run(
    input_path: Path,
    workers: int,
    progress_bars: bool,
    job_id: str | None,
    binarize: bool,
    process: bool,
    combine: bool,
    target_plane: int,
    data_path: Path | None,
    output_path: Path | None,
    discover: bool,
    extract: bool,
    target_recording: str | None,
    recording_paths: tuple[Path, ...],
) -> None:
    """Runs the cindra processing pipeline using the specified configuration file.

    The pipeline type (single-day or multi-day) is automatically detected from the configuration file. When --job-id
    is provided, only the matching job is executed and all step flags are ignored.
    """
    # Detects the pipeline type from the configuration file and dispatches to the appropriate pipeline runner.
    pipeline_type = detect_pipeline_type(file_path=input_path)

    if pipeline_type == PipelineType.SINGLE_DAY:
        # Writes CLI overrides into the configuration file before running the pipeline.
        configuration = SingleDayConfiguration.from_yaml(file_path=input_path)
        configuration.runtime.parallel_workers = workers
        configuration.runtime.display_progress_bars = progress_bars
        if data_path is not None:
            configuration.file_io.data_path = data_path
        if output_path is not None:
            configuration.file_io.output_path = output_path
        configuration.save(file_path=input_path)

        run_single_day_pipeline(
            configuration_path=input_path,
            job_id=job_id,
            binarize=binarize,
            process=process,
            combine=combine,
            target_plane=target_plane,
        )
    else:
        # Writes CLI overrides into the configuration file before running the pipeline.
        multi_day_configuration = MultiDayConfiguration.from_yaml(file_path=input_path)
        if recording_paths:
            multi_day_configuration.session_io.session_directories = tuple(natsorted(recording_paths))
        multi_day_configuration.runtime.parallel_workers = workers
        multi_day_configuration.runtime.display_progress_bars = progress_bars
        multi_day_configuration.save(file_path=input_path)

        run_multi_day_pipeline(
            configuration_path=input_path,
            job_id=job_id,
            discover=discover,
            extract=extract,
            target_session=target_recording,
        )


@cindra_cli.command("batch")
@click.option(
    "-i",
    "--input-path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="The absolute path to the single-day configuration .yaml template file.",
)
@click.option(
    "-d",
    "--data-directory",
    "data_directories",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    multiple=True,
    required=True,
    help=(
        "The root directory to search for recording sessions. Specify this option multiple times to search multiple "
        "directories."
    ),
)
@click.option(
    "-w",
    "--workers",
    type=int,
    default=-1,
    help=(
        "The number of parallel workers to use when executing multiprocessing tasks. For machines with a large number "
        "of cores a value between 10 and 20 is optimal. Setting this to a value of -1 or 0 makes the system use "
        "all available cores to parallelize multiprocessing tasks."
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
@click.option(
    "-b",
    "--binarize",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to run the binarization step (step 1).",
)
@click.option(
    "-p",
    "--process",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to run the processing step (step 2).",
)
@click.option(
    "-c",
    "--combine",
    is_flag=True,
    show_default=True,
    default=False,
    help="Determines whether to run the combination step (step 3).",
)
@click.option(
    "-tp",
    "--target-plane",
    type=int,
    default=-1,
    help="The index of the plane to process. Setting this to '-1' (default) processes all available planes.",
)
def cindra_batch(
    input_path: Path,
    data_directories: tuple[Path, ...],
    workers: int,
    progress_bars: bool,
    binarize: bool,
    process: bool,
    combine: bool,
    target_plane: int,
) -> None:
    """Discovers and processes all recording sessions under the specified root directories.

    This command searches the provided root directories for recording sessions (identified by cindra_parameters.json
    marker files), then runs the single-day pipeline for each discovered session using the provided configuration
    template. Failures for individual sessions do not abort the batch.
    """
    # Validates that the configuration is a single-day pipeline type.
    pipeline_type = detect_pipeline_type(file_path=input_path)
    if pipeline_type != PipelineType.SINGLE_DAY:
        message = (
            "The 'batch' command only supports single-day pipeline configurations. The provided configuration file "
            f"is a '{pipeline_type.value}' pipeline type."
        )
        console.error(message=message, error=ValueError)

    # Applies worker and progress-bar overrides to the in-memory configuration template. The updated template is saved
    # to a temporary file so the user's original config is never modified. Each session receives a local copy of this
    # temporary file, eliminating race conditions when multiple batch processes run in parallel.
    configuration = SingleDayConfiguration.from_yaml(file_path=input_path)
    configuration.runtime.parallel_workers = workers
    configuration.runtime.display_progress_bars = progress_bars
    template_path = input_path.parent / ".batch_template.yaml"
    configuration.save(file_path=template_path)

    # Discovers recording sessions across all specified root directories.
    all_sessions: list[Path] = []
    for directory in data_directories:
        sessions = discover_recordings(root_directory=directory)
        for session in sessions:
            if session not in all_sessions:
                all_sessions.append(session)

    if not all_sessions:
        console.echo(message="No recording sessions found in the specified directories.", level=LogLevel.WARNING)
        return

    # Natural-sorts the aggregated session list for consistent ordering.
    all_sessions = list(natsorted(all_sessions))

    console.echo(
        message=f"Found {len(all_sessions)} unique recording session(s) across {len(data_directories)} "
        f"director{'y' if len(data_directories) == 1 else 'ies'}.",
        level=LogLevel.INFO,
    )

    # Runs the single-day pipeline for each discovered session.
    try:
        run_single_day_batch(
            configuration_path=template_path,
            session_paths=all_sessions,
            binarize=binarize,
            process=process,
            combine=combine,
            target_plane=target_plane,
        )
    finally:
        if template_path.exists():
            template_path.unlink()
