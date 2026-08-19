"""Provides the command-line interface for launching the cindra Graphical User Interface (GUI) applications and the GUI
MCP server.

This CLI is installed as a separate entry-point from the main 'cindra' CLI to avoid loading GUI dependencies during
headless pipeline execution.
"""

from typing import Literal
from pathlib import Path

import click
from ataraxis_base_utilities import console

from .cli import CONTEXT_SETTINGS, report_command_failure
from ..gui import run_roi_viewer, run_tracking_viewer, run_registration_viewer
from .gui_mcp_server import run_gui_server


@click.group("cindra-gui", context_settings=CONTEXT_SETTINGS)
def cindra_gui() -> None:
    """Launches cindra GUI applications for visualizing pipeline outputs."""


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
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    hidden=True,
    help="Path to the state file for cross-process state exchange with the GUI MCP server.",
)
@report_command_failure
def gui_roi(recording_path: Path, dataset: str | None, state_file: Path | None) -> None:
    """Launches the ROI viewer for single-recording pipeline output.

    Providing --dataset switches the viewer to the multi-recording tracked-ROI view for the named dataset.
    """
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
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    hidden=True,
    help="Path to the state file for cross-process state exchange with the GUI MCP server.",
)
@report_command_failure
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
    type=click.Path(exists=False, file_okay=True, dir_okay=False, path_type=Path),
    default=None,
    hidden=True,
    help="Path to the state file for cross-process state exchange with the GUI MCP server.",
)
@report_command_failure
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
@report_command_failure
def gui_mcp(transport: Literal["stdio", "sse", "streamable-http"]) -> None:
    """Starts the GUI MCP server for agentic viewer lifecycle management and display state queries."""
    # The stdio transport carries the JSON-RPC message stream over stdout, which is also where the console writes
    # every message up to the WARNING level. Silencing the console keeps library output out of that stream, as a
    # single logged line renders the message it interleaves with unparsable for the connected client.
    if transport == "stdio":
        console.disable()

    run_gui_server(transport=transport)
