"""This module provides the command-line interface for launching the cindra Graphical User Interface (GUI)
applications. This CLI is installed as a separate entry-point from the main 'cindra' CLI to avoid loading GUI
dependencies during headless pipeline execution.
"""

from pathlib import Path

import click

from ..gui import run
from ..gui.registration import run_registration_viewer

# Ensures that displayed CLICK help messages are formatted according to the lab standard.
CONTEXT_SETTINGS = {"max_content_width": 120}


@click.group("cindra-gui", context_settings=CONTEXT_SETTINGS)
def cindra_gui() -> None:
    """Launches cindra GUI applications for visualizing pipeline outputs.

    Use this command group to launch the ROI viewer, registration quality viewer, or other GUI tools. At this time,
    the GUI does not support the multi-day processing pipeline.
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
    """Launches the ROI viewer and editor for single-day pipeline output."""
    run(session_path=session_path)


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
