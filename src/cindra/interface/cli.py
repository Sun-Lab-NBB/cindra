"""Provides the terminal-based interface for running all processing pipelines supported by the library."""

from typing import Literal
from pathlib import Path

import click
from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console

from .mcp_server import run_server
from ..dataclasses import PipelineType, MultiRecordingConfiguration, SingleRecordingConfiguration, detect_pipeline_type
from ..orchestration import (
    OpenMpStatus,
    resolve_openmp_runtime,
    run_multi_recording_pipeline,
    run_single_recording_pipeline,
)

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
def cindra_mcp(transport: Literal["stdio", "sse", "streamable-http"]) -> None:
    """Starts the Model Context Protocol (MCP) server for agentic neural imaging data processing.

    The MCP server exposes tools that enable AI agents to discover recording data, execute pipelines,
    monitor processing status, and manage batch operations for both single-recording and multi-recording workflows.
    """
    # The stdio transport carries the JSON-RPC message stream over stdout, which is also where the console writes
    # every message up to the WARNING level. Silencing the console keeps library output out of that stream, as a
    # single logged line renders the message it interleaves with unparsable for the connected client.
    if transport == "stdio":
        console.disable()

    run_server(transport=transport)


@cindra_cli.command("omp")
@click.option(
    "-s",
    "--source",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    default=None,
    help="The path to the OpenMP runtime to link. Omit to search the Homebrew library directories for it.",
)
@click.option(
    "-t",
    "--target",
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    required=False,
    default=None,
    help="The path to write the link to. Omit to derive it from the directory the dynamic loader searches by default.",
)
@click.option(
    "-f",
    "--force",
    is_flag=True,
    help="Determines whether to link a runtime on a host whose OpenMP runtime already loads.",
)
@click.option(
    "-y",
    "--yes",
    is_flag=True,
    help="Actually creates the resolved link. Without this flag the command reports what it would do and changes "
    "nothing.",
)
def cindra_omp(source: Path | None, target: Path | None, *, force: bool, yes: bool) -> None:
    """Links the OpenMP runtime that the Numba threading layer loads on macOS into a directory the loader searches.

    The Numba macOS wheel names its OpenMP dependency through an rpath that carries no entries, so the runtime
    resolves from the dynamic loader's default search path alone. This command finds an installed runtime and links it
    into that path. Writing the link usually requires running the command through sudo. Running the command on any
    other platform errors, because those platforms run the TBB threading layer instead.
    """
    try:
        summary = resolve_openmp_runtime(runtime_path=source, link_path=target, execute=yes, force=force)
    except RuntimeError as error:
        raise click.ClickException(message=str(error)) from error

    if summary.searched_paths:
        console.echo(message=f"searched: {', '.join(str(path) for path in summary.searched_paths)}")
    if summary.runtime_path is not None:
        console.echo(message=f"runtime:  {summary.runtime_path}")
        console.echo(message=f"link:     {summary.link_path}")
    console.echo(message=summary.describe())
    if summary.status == OpenMpStatus.UNRESOLVED:
        raise SystemExit(1)


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
    help="The absolute path to the (existing) directory in which to generate the requested configuration file.",
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
    pipeline. Provide the path to the modified file to the 'run' CLI command to execute the desired pipeline
    with the parameters specified inside the file.
    """
    # Normalizes shorthand aliases and resolves pipeline-specific parameters.
    single_recording = pipeline in ("single-recording", "sd")
    resolved_name = name if name is not None else ("cindra_sd_conf" if single_recording else "cindra_md_conf")
    file_path = output_path.joinpath(resolved_name).with_suffix(".yaml")

    # Generates the precursor configuration file in the specified output directory.
    configuration = SingleRecordingConfiguration() if single_recording else MultiRecordingConfiguration()
    configuration.save(file_path=file_path)

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
    "-bw",
    "--binarize-workers",
    type=int,
    required=False,
    default=None,
    help=(
        "[Single-recording] The number of parallel workers to allocate to the binarization step. When this option is "
        "omitted, the step receives its measured default allocation of 3 workers, which is the point where the "
        "allocated cores become the TIFF image decode threads. Setting this to -1 uses every available core, minus "
        "the cores reserved for system use."
    ),
)
@click.option(
    "-rw",
    "--register-workers",
    type=int,
    required=False,
    default=None,
    help=(
        "[Single-recording] The number of parallel workers to allocate to each plane-registration step. When this "
        "option is omitted, the step receives its measured default allocation of 12 workers, which is the knee of the "
        "measured registration scaling curve. Setting this to -1 uses every available core, minus the cores reserved "
        "for system use."
    ),
)
@click.option(
    "-pw",
    "--process-workers",
    type=int,
    required=False,
    default=None,
    help=(
        "[Single-recording] The number of parallel workers to allocate to each plane-processing step. When this option "
        "is omitted, the step receives its measured default allocation of 10 workers, where detection reaches its "
        "measured throughput plateau. Setting this to -1 uses every available core, minus the cores reserved for "
        "system use."
    ),
)
@click.option(
    "-dw",
    "--discover-workers",
    type=int,
    required=False,
    default=None,
    help=(
        "[Multi-recording] The number of parallel workers to allocate to the discovery step. When this option is "
        "omitted, the step receives its measured default allocation of 30 workers, which is the saturating allocation "
        "the step is admitted at. Setting this to -1 uses every available core, minus the cores reserved for system "
        "use."
    ),
)
@click.option(
    "-ew",
    "--extract-workers",
    type=int,
    required=False,
    default=None,
    help=(
        "[Multi-recording] The number of parallel workers to allocate to each per-recording extraction step. When this "
        "option is omitted, the step receives its measured default allocation of 16 workers, which is the point where "
        "the step stops shortening. Setting this to -1 uses every available core, minus the cores reserved for system "
        "use."
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
    "-r",
    "--register",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "[Single-recording] Determines whether to register the target plane(s) to remove motion and compute the "
        "registration quality metrics (step 2). This step must complete for a plane before that plane can be "
        "processed."
    ),
)
@click.option(
    "-p",
    "--process",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "[Single-recording] Determines whether to process the target plane(s) to discover ROIs and extract their "
        "fluorescence (step 3). This step aggregates most data processing logic of the pipeline."
    ),
)
@click.option(
    "-c",
    "--combine",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "[Single-recording] Determines whether to combine processed plane data into a uniform dataset "
        "(step 4). Note that this step is required to later process the data as part of a multi-recording "
        "pipeline."
    ),
)
@click.option(
    "-tp",
    "--target-plane",
    type=int,
    default=-1,
    help=(
        "[Single-recording] The index of the plane to process when running the REGISTER (2) or PROCESS (3) steps. "
        "Setting this to '-1' (default value) processes all available planes sequentially."
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
        "[Single-recording] The path to the root directory in which to create the cindra output hierarchy and store "
        "the processed data. When provided, this path overrides the matching field in the pipeline's configuration "
        "file. The output_path must be set either in the configuration file or via this flag."
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
        "[Multi-recording] Determines whether to extract the fluorescence from ROIs tracked across "
        "recordings, identified during the first processing step."
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
        "[Multi-recording] The path to the recording processed with the single-recording cindra pipeline "
        "to include in the processed multi-recording dataset. Specify this option multiple times to include "
        "multiple recordings "
        "(at least two required). When provided, these paths override the matching fields in the pipeline's "
        "configuration file."
    ),
)
def cindra_run(
    input_path: Path,
    binarize_workers: int | None,
    register_workers: int | None,
    process_workers: int | None,
    discover_workers: int | None,
    extract_workers: int | None,
    *,
    progress_bars: bool,
    job_id: str | None,
    binarize: bool,
    register: bool,
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
    configuration file. When no step flag is set, every step of the detected pipeline runs in phase order.
    When --job-id is provided, only the matching job is executed and all step flags are ignored. The
    combination step merges the per-plane result files with serial input and output.
    """
    pipeline_type = detect_pipeline_type(file_path=input_path)

    if pipeline_type == PipelineType.SINGLE_RECORDING:
        # Writes CLI overrides into the configuration file before running the pipeline.
        configuration = SingleRecordingConfiguration.from_yaml(file_path=input_path)
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
            register=register,
            process=process,
            combine=combine,
            target_plane=target_plane,
            binarization_workers=binarize_workers,
            registration_workers=register_workers,
            processing_workers=process_workers,
        )
    else:
        # Writes CLI overrides into the configuration file before running the pipeline.
        multi_recording_configuration = MultiRecordingConfiguration.from_yaml(file_path=input_path)
        if recording_paths:
            multi_recording_configuration.recording_io.recording_directories = tuple(natsorted(recording_paths))
        multi_recording_configuration.runtime.display_progress_bars = progress_bars
        multi_recording_configuration.save(file_path=input_path)

        run_multi_recording_pipeline(
            configuration_path=input_path,
            job_id=job_id,
            discover=discover,
            extract=extract,
            target_recording=target_recording,
            discovery_workers=discover_workers,
            extraction_workers=extract_workers,
        )
