"""Provides the interactive desktop GUIs for suite2p data visualization and analysis.

Sub-packages:
    roi_viewer: ROI viewer and editor for single-day pipeline output.
    registration_viewer: Registration quality viewer for motion correction inspection.
    tracking_viewer: Multi-day cell tracking quality viewer.
"""

from .roi_viewer import run

__all__ = [
    "run",
]
