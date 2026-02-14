"""Provides the processing pipeline orchestration logic for single-day and multi-day workflows."""

from .pipeline import (
    MultiDayJobNames,
    SingleDayJobNames,
    process_multi_day,
    process_single_day,
)
from .multi_day import (
    run_multiday_pipeline,
    discover_multiday_cells,
    extract_multiday_fluorescence,
)
from .single_day import process_plane, binarize_recording, save_combined_data, run_single_day_pipeline

__all__ = [
    "MultiDayJobNames",
    "SingleDayJobNames",
    "binarize_recording",
    "discover_multiday_cells",
    "extract_multiday_fluorescence",
    "process_multi_day",
    "process_plane",
    "process_single_day",
    "run_multiday_pipeline",
    "run_single_day_pipeline",
    "save_combined_data",
]
