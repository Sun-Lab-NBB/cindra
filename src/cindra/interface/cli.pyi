from pathlib import Path

from ..pipelines import (
    run_multi_recording_pipeline as run_multi_recording_pipeline,
    run_single_recording_pipeline as run_single_recording_pipeline,
)
from .mcp_server import run_server as run_server
from ..dataclasses import (
    PipelineType as PipelineType,
    MultiRecordingConfiguration as MultiRecordingConfiguration,
    SingleRecordingConfiguration as SingleRecordingConfiguration,
    detect_pipeline_type as detect_pipeline_type,
)

CONTEXT_SETTINGS: dict[str, int]

def cindra_cli() -> None: ...
def cindra_mcp(transport: str) -> None: ...
def cindra_config(pipeline: str, output_path: Path, name: str | None) -> None: ...
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
) -> None: ...
