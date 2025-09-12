"""This package provides the algorithms and tools for carrying out the multi-day sl-suite2p processing pipeline.
This pipeline is based on the original implementation found here:
https://github.com/sprustonlab/multiday-suite2p-public/tree/main.
"""

from .io import import_sessions, export_masks_and_images
from .gui import show_images_with_masks
from .utils import extract_unique_components
from .process import extract_session_traces
from .transform import (
    register_sessions,
    generate_template_masks,
    backward_transform_masks,
)
from .dataclasses import Session, MultiDayData

__all__ = [
    "MultiDayData",
    "Session",
    "backward_transform_masks",
    "export_masks_and_images",
    "extract_session_traces",
    "extract_unique_components",
    "generate_template_masks",
    "import_sessions",
    "register_sessions",
    "show_images_with_masks",
]
