"""This module provides the command-line interface for launching the sl-suite2p Graphical User Interface (GUI)
application. This CLI is installed as a separate entry-point from the main 'ss2p' CLI to avoid loading GUI dependencies
during headless pipeline execution.
"""

import click

from ..gui import run

# Ensures that displayed CLICK help messages are formatted according to the lab standard.
CONTEXT_SETTINGS = {"max_content_width": 120}


@click.command("ss2p-gui", context_settings=CONTEXT_SETTINGS)
def ss2p_gui() -> None:
    """Starts the sl-suite2p Graphical User Interface (GUI) application.

    Use this command to work with the single-day sl-suite2p processing pipeline via a graphical interface. At this
    time, the GUI does not support the multi-day processing pipeline.
    """
    run()
