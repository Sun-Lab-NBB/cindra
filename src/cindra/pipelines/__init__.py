"""Provides the stage entry points that the single-recording and multi-recording pipelines dispatch."""

from .multi_recording import discover_multi_recording_cells, extract_multi_recording_fluorescence
from .single_recording import process_plane, binarize_recording, save_combined_data, register_recording_plane

__all__ = [
    "binarize_recording",
    "discover_multi_recording_cells",
    "extract_multi_recording_fluorescence",
    "process_plane",
    "register_recording_plane",
    "save_combined_data",
]
