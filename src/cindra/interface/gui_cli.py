"""Provides the command-line interface for launching the cindra Graphical User Interface (GUI) applications and the GUI
MCP server.

This CLI is installed as a separate entry-point from the main 'cindra' CLI to avoid loading GUI dependencies during
headless pipeline execution. The GUI MCP server module avoids Qt imports at module level, so the 'mcp' subcommand
starts without loading PySide6.
"""

from pathlib import Path

import click

from ..gui import run_roi_viewer, run_tracking_viewer, run_registration_viewer
from .gui_mcp_server import run_gui_server

_CONTEXT_SETTINGS: dict[str, int] = {"max_content_width": 120}
"""The Click context settings that ensure displayed help messages are formatted according to the lab standard."""


@click.group("cindra-gui", context_settings=_CONTEXT_SETTINGS)
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
def gui_roi(recording_path: Path, dataset: str | None) -> None:
    """Launches the ROI viewer for single-recording pipeline output."""
    run_roi_viewer(recording_path=recording_path, dataset=dataset)


@cindra_gui.command("registration")
@click.option(
    "-r",
    "--recording-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Path to a cindra output directory containing registration results.",
)
def gui_registration(recording_path: Path) -> None:
    """Launches the registration quality viewer for inspecting motion correction results."""
    run_registration_viewer(recording_path=recording_path)


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
def gui_tracking(recording_path: Path, dataset: str | None) -> None:
    """Launches the multi-recording tracking quality viewer for inspecting across-recording ROI tracking results."""
    run_tracking_viewer(recording_path=recording_path, dataset=dataset)


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
    """Starts the GUI MCP server for agentic viewer control and data querying.

    The GUI MCP server exposes tools that enable AI agents to launch GUI viewers, control them in real-time, and query
    processed data such as ROI statistics, fluorescence traces, and tracking results.
    """
    run_gui_server(transport=transport)  # type: ignore[arg-type]
