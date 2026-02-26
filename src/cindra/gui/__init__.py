"""Provides the interactive GUIs for visualizing the processing outcomes of the single-day and multi-day pipelines."""

from .app import (
    run_roi_editor,
    run_roi_viewer as run_roi_viewer_standalone,
    run_registration_viewer,
    run_tracking_viewer,
)

# Backward-compatible alias: the CLI ``roi`` command calls ``run_roi_viewer`` which opens the editor.
run_roi_viewer = run_roi_editor

__all__ = [
    "run_registration_viewer",
    "run_roi_editor",
    "run_roi_viewer",
    "run_roi_viewer_standalone",
    "run_tracking_viewer",
]
