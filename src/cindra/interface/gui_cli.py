"""Provides the command-line interface for launching the cindra Graphical User Interface (GUI) applications and the GUI
MCP server.

This CLI is installed as a separate entry-point from the main 'cindra' CLI to avoid loading GUI dependencies during
headless pipeline execution.
"""

from pathlib import Path

import click

from .cli import CONTEXT_SETTINGS
from ..gui import run_roi_viewer, run_tracking_viewer, run_registration_viewer
from .gui_mcp_server import run_gui_server


@click.group("cindra-gui", context_settings=CONTEXT_SETTINGS)
def cindra_gui() -> None:
    """Launches cindra GUI applications for visualizing pipeline outputs.

    Use this command group to launch the ROI viewer, registration quality viewer, multi-recording tracking
    viewer, or other GUI tools.
    """


@cindra_gui.command("roi")
@click.option(
    "-r",
    "--recording-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Path to a cindra output directory to load on startup.",
)
@click.option(
    "-d",
    "--dataset",
    type=str,
    default=None,
    help="Multi-recording dataset name to load. Stays in single-recording mode if not provided.",
)
@click.option(
    "-sf",
    "--state-file",
    type=click.Path(path_type=Path),
    default=None,
    hidden=True,
    help="Path to the state file for cross-process state exchange with the GUI MCP server.",
)
def gui_roi(recording_path: Path, dataset: str | None, state_file: Path | None) -> None:
    """Launches the ROI viewer for single-recording pipeline output."""
    run_roi_viewer(recording_path=recording_path, dataset=dataset, state_path=state_file)


@cindra_gui.command("registration")
@click.option(
    "-r",
    "--recording-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Path to a cindra output directory containing registration results.",
)
@click.option(
    "-sf",
    "--state-file",
    type=click.Path(path_type=Path),
    default=None,
    hidden=True,
    help="Path to the state file for cross-process state exchange with the GUI MCP server.",
)
def gui_registration(recording_path: Path, state_file: Path | None) -> None:
    """Launches the registration quality viewer for inspecting motion correction results."""
    run_registration_viewer(recording_path=recording_path, state_path=state_file)


@cindra_gui.command("tracking")
@click.option(
    "-r",
    "--recording-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Path to any recording's cindra output directory that is part of a multi-recording dataset.",
)
@click.option(
    "-d",
    "--dataset",
    type=str,
    default=None,
    help="Multi-recording dataset name to load. Defaults to the first available dataset.",
)
@click.option(
    "-sf",
    "--state-file",
    type=click.Path(path_type=Path),
    default=None,
    hidden=True,
    help="Path to the state file for cross-process state exchange with the GUI MCP server.",
)
def gui_tracking(recording_path: Path, dataset: str | None, state_file: Path | None) -> None:
    """Launches the multi-recording tracking quality viewer for inspecting across-recording ROI tracking results."""
    run_tracking_viewer(recording_path=recording_path, dataset=dataset, state_path=state_file)


@cindra_gui.command("mcp")
@click.option(
    "-t",
    "--transport",
    type=click.Choice(["stdio", "sse", "streamable-http"]),
    default="stdio",
    show_default=True,
    help="The transport protocol to use for MCP communication.",
)
def gui_mcp(transport: str) -> None:
    """Starts the GUI MCP server for agentic viewer lifecycle management and display state queries."""
    run_gui_server(transport=transport)  # type: ignore[arg-type]
