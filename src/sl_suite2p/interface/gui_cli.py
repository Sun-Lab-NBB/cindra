"""This module provides the command-line interface for launching the sl-suite2p Graphical User Interface (GUI)
applications. This CLI is installed as a separate entry-point from the main 'ss2p' CLI to avoid loading GUI
dependencies during headless pipeline execution.
"""

from pathlib import Path

import click

from ..gui import run
from ..gui.registration import run_registration_viewer

# Ensures that displayed CLICK help messages are formatted according to the lab standard.
CONTEXT_SETTINGS = {"max_content_width": 120}


@click.group("ss2p-gui", context_settings=CONTEXT_SETTINGS)
def ss2p_gui() -> None:
    """Launches sl-suite2p GUI applications for visualizing pipeline outputs.

    Use this command group to launch the ROI viewer, registration quality viewer, or other GUI tools. At this time,
    the GUI does not support the multi-day processing pipeline.
    """


@ss2p_gui.command("roi")
@click.option(
    "-s",
    "--session-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Path to a suite2p output directory to load on startup.",
)
def gui_roi(session_path: Path | None) -> None:
    """Launches the ROI viewer and editor for single-day pipeline output."""
    run(session_path=session_path)


@ss2p_gui.command("registration")
@click.option(
    "-s",
    "--session-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    default=None,
    help="Path to a suite2p output directory to load on startup.",
)
def gui_registration(session_path: Path | None) -> None:
    """Launches the registration quality viewer for inspecting motion correction results."""
    run_registration_viewer(session_path=session_path)
