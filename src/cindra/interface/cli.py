"""Provides the terminal-based interface for running all processing pipelines supported by the library."""

from typing import TYPE_CHECKING, Literal
from pathlib import Path
from functools import wraps

import click
from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console

from .mcp_server import run_server
from ..dataclasses import PipelineType, MultiRecordingConfiguration, SingleRecordingConfiguration, detect_pipeline_type
from ..orchestration import (
    OpenMPStatus,
    resolve_gpu_devices,
    resolve_openmp_runtime,
    run_multi_recording_pipeline,
    run_single_recording_pipeline,
)

if TYPE_CHECKING:
    from collections.abc import Callable

CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""The Click context settings that ensure displayed help messages are formatted according to the cindra standard."""


# Sits above the commands because a decorator is resolved where it is applied rather than where the module ends.
def report_command_failure[**P](command: Callable[P, None]) -> Callable[P, None]:
    """Reports the failure of a command through the console instead of an interpreter traceback.

    Notes:
        A traceback buries the message under a stack the user of a command line tool cannot act on.

        A Click exception passes through, so the usage errors a command body raises keep the usage banner and the
        exit status Click gives its own parameter validation. A caller therefore reads one exit status for every
        malformed invocation, whether Click or a command body detected it.

        A SystemExit also passes through, so a command that reports its own outcome and then raises SystemExit sets
        the process exit status the caller reads.

    Args:
        command: The command function to wrap.

    Returns:
        The wrapped command, which reports any other Exception its body raises and returns normally.
    """

    @wraps(command)
    def report(*args: P.args, **kwargs: P.kwargs) -> None:
        try:
            command(*args, **kwargs)
        except click.ClickException:
            raise
        except Exception as error:
            console.echo(message=str(error), level=LogLevel.ERROR)

    return report


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
@report_command_failure
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
    help=(
        "The path to the OpenMP runtime to link. Omit to search the macOS package manager directories, the active "
        "conda environment, and the installed Python distributions for one."
    ),
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
    help=(
        "Determines whether to create the resolved link. Without this flag the command reports what it would do and "
        "changes nothing."
    ),
)
@report_command_failure
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
        message = f"Unable to resolve the macOS OpenMP runtime. {error}"
        console.error(message=message, error=RuntimeError)

    if summary.searched_paths:
        console.echo(message=f"searched: {', '.join(str(path) for path in summary.searched_paths)}", raw=True)
    if summary.runtime_path is not None:
        console.echo(message=f"runtime:  {summary.runtime_path}", raw=True)
        console.echo(message=f"link:     {summary.link_path}", raw=True)
    console.echo(message=summary.describe())
    if summary.status == OpenMPStatus.UNRESOLVED:
        raise SystemExit(1)


@cindra_cli.command("gpu")
@report_command_failure
def cindra_gpu() -> None:
    """Reports the CUDA devices the registration stage runs on, and why it reaches none.

    Single-recording planes register on a CUDA device through the CuPy runtime when the run names one. This reports
    every device the host exposes, together with its memory and compute capability, after transforming a small array on
    the device the runtime selects by default. That transform is what separates a reachable device from an importable
    module, because CuPy resolves the CUDA math libraries on first use rather than at import. A host that reaches no
    device reports the reason and the installation that resolves it, and exits with a non-zero status. Running the
    command on macOS reports that CuPy publishes no wheel for the platform, where registration runs on the host CPU.
    """
    summary = resolve_gpu_devices()

    console.echo(message=summary.describe())
    if summary.remedy:
        console.echo(message=f"remedy:   {summary.remedy}", raw=True)
    if not summary.available:
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
@report_command_failure
def cindra_config(pipeline: str, output_path: Path, name: str | None) -> None:
    """Generates the configuration file for the specified processing pipeline.

    Modifying the parameters stored in the generated file allows configuring all aspects of the target processing
    pipeline. Provide the path to the modified file to the 'run' CLI command to execute the desired pipeline
    with the parameters specified inside the file.
    """
    single_recording = pipeline in ("single-recording", "sd")
    resolved_name = name if name is not None else ("cindra_sd_conf" if single_recording else "cindra_md_conf")
    if not resolved_name.strip():
        message = (
            f"Unable to generate the pipeline configuration file. The configuration file name must carry at least one "
            f"non-whitespace character, but got {resolved_name!r}."
        )
        console.error(message=message, error=click.UsageError)

    # Appends the extension the pipeline loader requires, keeping every component of the supplied name. Path
    # 'with_suffix' would instead replace the component that follows the name's last dot.
    if resolved_name.endswith(".yml"):
        resolved_name = f"{resolved_name.removesuffix('.yml')}.yaml"
    elif not resolved_name.endswith(".yaml"):
        resolved_name = f"{resolved_name}.yaml"
    file_path = output_path / resolved_name

    configuration = SingleRecordingConfiguration() if single_recording else MultiRecordingConfiguration()
    configuration.save(file_path=file_path)

    message = (
        f"Default {'single-recording' if single_recording else 'multi-recording'} pipeline configuration file: "
        f"generated as {file_path}. Modify the configuration parameters in the file to finish the configuration "
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
    "-bw",
    "--binarize-workers",
    type=int,
    required=False,
    default=None,
    help=(
        "[Single-recording] The number of parallel workers to allocate to the binarization step. When this option is "
        "omitted, the step receives its measured default allocation of 4 workers, which is the decode ceiling "
        "itself. A larger request is capped at that ceiling, because added decode threads stop shortening the "
        "conversion past that point. Setting this to -1 uses every available core, minus the cores reserved for "
        "system use."
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
        "option is omitted, the step receives its measured default allocation of 8 workers on the host CPU, or 2 "
        "workers when --register-device names a CUDA device. Setting this to -1 uses every available core, minus "
        "the cores reserved for system use."
    ),
)
@click.option(
    "-rd",
    "--register-device",
    type=int,
    required=False,
    default=None,
    help=(
        "[Single-recording] The zero-based index of the CUDA device that registers every plane of this run. When this "
        "option is omitted, every plane of the run registers on the host CPU."
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
        "is omitted, the step receives its measured default allocation of 8 workers. Setting this to -1 uses every "
        "available core, minus the cores reserved for system use."
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
        "omitted, the step receives its measured default allocation of 2 workers. Setting this to -1 uses every "
        "available core, minus the cores reserved for system use."
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
        "option is omitted, the step receives its measured default allocation of 16 workers. Setting this to -1 "
        "uses every available core, minus the cores reserved for system use."
    ),
)
@click.option(
    "-np",
    "--no-progress",
    is_flag=True,
    show_default=True,
    default=False,
    help=(
        "Determines whether to suppress the progress bars displayed during long-running tasks. The progress bars are "
        "displayed by default."
    ),
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
        "[Multi-recording] The path to the recording processed with the single-recording cindra pipeline to include in "
        "the processed multi-recording dataset. Specify this option multiple times to include multiple recordings (at "
        "least two required). When provided, these paths override the matching fields in the pipeline's configuration "
        "file."
    ),
)
@report_command_failure
def cindra_run(
    input_path: Path,
    binarize_workers: int | None,
    register_workers: int | None,
    register_device: int | None,
    process_workers: int | None,
    discover_workers: int | None,
    extract_workers: int | None,
    *,
    no_progress: bool,
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

    The pipeline type (single-recording or multi-recording) is automatically detected from the configuration file. When
    no step flag is set, every step of the detected pipeline runs in phase order. When --job-id is provided, only the
    matching job is executed and all step flags are ignored. The combination step merges the per-plane result files with
    serial input and output.
    """
    pipeline_type = detect_pipeline_type(file_path=input_path)

    if pipeline_type == PipelineType.SINGLE_RECORDING:
        if register_device is not None and register_device < 0:
            message = (
                f"Unable to run the single-recording pipeline. The --register-device option must name a zero-based "
                f"CUDA device index, but encountered {register_device}."
            )
            console.error(message=message, error=click.UsageError)

        configuration = SingleRecordingConfiguration.from_yaml(file_path=input_path)
        configuration.runtime.display_progress_bars = not no_progress
        if data_path is not None:
            configuration.file_io.data_path = data_path
        if output_path is not None:
            configuration.file_io.output_path = output_path
        if configuration.file_io.output_path is None:
            message = (
                "Unable to run the single-recording pipeline. The output_path must be configured either in the "
                "configuration file or via the --output-path flag, but it is currently None."
            )
            console.error(message=message, error=click.UsageError)
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
            registration_device=register_device,
        )
    else:
        if register_device is not None:
            message = (
                "Unable to run the multi-recording pipeline. The --register-device option names the CUDA device the "
                "single-recording pipeline registers its planes on, and no multi-recording stage runs on a device."
            )
            console.error(message=message, error=click.UsageError)

        multi_recording_configuration = MultiRecordingConfiguration.from_yaml(file_path=input_path)
        if recording_paths:
            multi_recording_configuration.recording_io.recording_directories = tuple(natsorted(recording_paths))
        multi_recording_configuration.runtime.display_progress_bars = not no_progress
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
