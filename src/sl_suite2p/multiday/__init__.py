"""This package provides the algorithms and tools for carrying out the multi-day sl-suite2p processing pipeline.
This pipeline is based on the original implementation found here:
https://github.com/sprustonlab/multiday-suite2p-public/tree/main.
"""

from .gui import show_images_with_masks
from .process import extract_session_traces
from .transform import generate_template_masks

__all__ = [
    "extract_session_traces",
    "generate_template_masks",
    "show_images_with_masks",
]
