from typing import Literal
from pathlib import Path

from .cli import (
    CONTEXT_SETTINGS as CONTEXT_SETTINGS,
    report_command_failure as report_command_failure,
)
from ..gui import (
    run_roi_viewer as run_roi_viewer,
    run_tracking_viewer as run_tracking_viewer,
    run_registration_viewer as run_registration_viewer,
)
from .gui_mcp_server import run_gui_server as run_gui_server

def cindra_gui() -> None: ...
@report_command_failure
def gui_roi(recording_path: Path, dataset: str | None, state_file: Path | None) -> None: ...
@report_command_failure
def gui_registration(recording_path: Path, state_file: Path | None) -> None: ...
@report_command_failure
def gui_tracking(recording_path: Path, dataset: str | None, state_file: Path | None) -> None: ...
@report_command_failure
def gui_mcp(transport: Literal["stdio", "sse", "streamable-http"]) -> None: ...
