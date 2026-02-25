"""Provides the interactive desktop GUIs for cindra data visualization and analysis.

Sub-packages:
    roi: ROI viewer and editor for single-day pipeline output.
    registration: Registration quality viewer for motion correction inspection.
    tracking: Multi-day cell tracking quality viewer.
"""

from .roi import run
from .tracking import run_tracking_viewer

__all__ = [
    "run",
    "run_tracking_viewer",
]
