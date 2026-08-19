from typing import Literal
from pathlib import Path
from collections.abc import Callable as Callable

from .mcp_server import run_server as run_server
from ..dataclasses import (
    PipelineType as PipelineType,
    MultiRecordingConfiguration as MultiRecordingConfiguration,
    SingleRecordingConfiguration as SingleRecordingConfiguration,
    detect_pipeline_type as detect_pipeline_type,
)
from ..orchestration import (
    OpenMPStatus as OpenMPStatus,
    resolve_openmp_runtime as resolve_openmp_runtime,
    run_multi_recording_pipeline as run_multi_recording_pipeline,
    run_single_recording_pipeline as run_single_recording_pipeline,
)

CONTEXT_SETTINGS: dict[str, int]

def report_command_failure[**P](command: Callable[P, None]) -> Callable[P, None]: ...
def cindra_cli() -> None: ...
@report_command_failure
def cindra_mcp(transport: Literal["stdio", "sse", "streamable-http"]) -> None: ...
@report_command_failure
def cindra_omp(source: Path | None, target: Path | None, *, force: bool, yes: bool) -> None: ...
@report_command_failure
def cindra_config(pipeline: str, output_path: Path, name: str | None) -> None: ...
@report_command_failure
def cindra_run(
    input_path: Path,
    binarize_workers: int | None,
    register_workers: int | None,
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
) -> None: ...
