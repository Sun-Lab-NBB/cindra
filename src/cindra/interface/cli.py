"""Provides the terminal-based interface for running all processing pipelines supported by the library."""

from pathlib import Path

import click
from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console

from ..pipelines import run_multi_recording_pipeline, run_single_recording_pipeline
from .mcp_server import run_server
from ..dataclasses import PipelineType, MultiRecordingConfiguration, SingleRecordingConfiguration, detect_pipeline_type

CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""The Click context settings that ensure displayed help messages are formatted according to the cindra standard."""


@click.group("cindra", context_settings=CONTEXT_SETTINGS)
def cindra_cli() -> None:
    """Provides the entry-point for all headless command-line interactions with the cindra library."""


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
    monitor processing status, and manage batch operations for both single-recording and multi-recording workflows.
    """
    run_server(transport=transport)  # type: ignore[arg-type]


@cindra_cli.command("configure")
@click.option(
    "-p",
    "--pipeline",
    type=click.Choice(["single-recording", "sd", "multi-recording", "md"], case_sensitive=False),
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
    single_recording = pipeline in ("single-recording", "sd")
    resolved_name = name if name is not None else ("cindra_sd_conf" if single_recording else "cindra_md_conf")
    file_path = output_path.joinpath(resolved_name).with_suffix(".yaml")

    # Generates the precursor configuration file in the specified output directory.
    config = SingleRecordingConfiguration() if single_recording else MultiRecordingConfiguration()
    config.save(file_path=file_path)

    message = (
        f"Default {'single-recording' if single_recording else 'multi-recording'} pipeline configuration file: "
        f"generated in the {file_path.parent} directory. Modify the configuration parameters in the file to finish "
        f"the configuration process."
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
        "[Single-recording] Determines whether to resolve the binary files for plane-specific processing (step 1). "
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
        "[Single-recording] Determines whether to process the target plane(s) to remove motion, discover ROIs,"
        " and extract their fluorescence (step 2). This step aggregates most data processing logic of the"
        " pipeline."
    ),
)
@click.option(
    "-c",
    "--combine",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "[Single-recording] Determines whether to combine processed plane data into a uniform dataset"
        " (step 3). Note, this step is required to later process the data as part of a multi-recording"
        " pipeline."
    ),
)
@click.option(
    "-tp",
    "--target-plane",
    type=int,
    default=-1,
    help=(
        "[Single-recording] The index of the plane to process when running the PROCESS step (2). Setting this to '-1' "
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
        "[Single-recording] The path to the root directory containing the recording's raw input TIFF files. When "
        "provided, this path overrides the matching field in the pipeline's configuration file."
    ),
)
@click.option(
    "-s",
    "--output-path",
    type=click.Path(exists=False, file_okay=False, dir_okay=True, path_type=Path),
    required=False,
    default=None,
    help=(
        "[Single-recording] The path to the root directory where to create the cindra's output hierarchy and store the "
        "processed data. When provided, this path overrides the matching field in the pipeline's configuration file. "
        "The output_path must be set either in the configuration file or via this flag."
    ),
)
@click.option(
    "-d",
    "--discover",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "[Multi-recording] Determines whether to discover ROIs trackable across recordings (step 1). This step "
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
        "[Multi-recording] Determines whether to extract the fluorescence from ROIs tracked across"
        " recordings, identified during the first processing step."
    ),
)
@click.option(
    "-tr",
    "--target-recording",
    type=str,
    required=False,
    default=None,
    help=(
        "[Multi-recording] The unique identifier of the recording to process when running the 'extract' step. If "
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
        "[Multi-recording] The path to the recording processed with the single-recording cindra pipeline"
        " to include in the processed multi-recording dataset. Specify this option multiple times to include"
        " multiple recordings "
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

    The pipeline type (single-recording or multi-recording) is automatically detected from the
    configuration file. When --job-id is provided, only the matching job is executed and all step flags
    are ignored.
    """
    # Detects the pipeline type from the configuration file and dispatches to the appropriate pipeline runner.
    pipeline_type = detect_pipeline_type(file_path=input_path)

    if pipeline_type == PipelineType.SINGLE_RECORDING:
        # Writes CLI overrides into the configuration file before running the pipeline.
        configuration = SingleRecordingConfiguration.from_yaml(file_path=input_path)
        configuration.runtime.parallel_workers = workers
        configuration.runtime.display_progress_bars = progress_bars
        if data_path is not None:
            configuration.file_io.data_path = data_path
        if output_path is not None:
            configuration.file_io.output_path = output_path
        if configuration.file_io.output_path is None:
            message = (
                "Unable to run the single-recording pipeline. The output_path must be configured either in the "
                "configuration file or via the --output-path flag, but it is currently None."
            )
            console.error(message=message, error=ValueError)
        configuration.save(file_path=input_path)

        run_single_recording_pipeline(
            configuration_path=input_path,
            job_id=job_id,
            binarize=binarize,
            process=process,
            combine=combine,
            target_plane=target_plane,
        )
    else:
        # Writes CLI overrides into the configuration file before running the pipeline.
        multi_recording_configuration = MultiRecordingConfiguration.from_yaml(file_path=input_path)
        if recording_paths:
            multi_recording_configuration.recording_io.recording_directories = tuple(natsorted(recording_paths))
        multi_recording_configuration.runtime.parallel_workers = workers
        multi_recording_configuration.runtime.display_progress_bars = progress_bars
        multi_recording_configuration.save(file_path=input_path)

        run_multi_recording_pipeline(
            configuration_path=input_path,
            job_id=job_id,
            discover=discover,
            extract=extract,
            target_recording=target_recording,
        )
