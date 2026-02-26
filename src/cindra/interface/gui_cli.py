"""This module provides the command-line interface for launching the cindra Graphical User Interface (GUI)
applications. This CLI is installed as a separate entry-point from the main 'cindra' CLI to avoid loading GUI
dependencies during headless pipeline execution.
"""

from pathlib import Path

import click

from ..gui import run_roi_viewer, run_tracking_viewer, run_registration_viewer

CONTEXT_SETTINGS = {"max_content_width": 120}
"""The Click context settings that ensure displayed help messages are formatted according to the lab standard."""


@click.group("cindra-gui", context_settings=CONTEXT_SETTINGS)
def cindra_gui() -> None:
    """Launches cindra GUI applications for visualizing pipeline outputs.

    Use this command group to launch the ROI viewer, registration quality viewer, multi-day tracking viewer, or other
    GUI tools.
    """


@cindra_gui.command("roi")
@click.option(
    "-s",
    "--session-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Path to a cindra output directory to load on startup.",
)
def gui_roi(session_path: Path | None) -> None:
    """Launches the ROI viewer for single-day pipeline output."""
    run_roi_viewer(session_path=session_path)


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
    help="Path to any recording's cindra output directory that is part of a multi-day dataset.",
)
def gui_tracking(recording_path: Path) -> None:
    """Launches the multi-day tracking quality viewer for inspecting across-day ROI tracking results."""
    run_tracking_viewer(recording_path=recording_path)
