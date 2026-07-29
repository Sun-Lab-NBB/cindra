"""Provides the processing pipeline orchestration logic for single-recording and multi-recording workflows."""

from .pipeline import (
    MULTI_RECORDING_TRACKER_NAME,
    SINGLE_RECORDING_TRACKER_NAME,
    execute_multi_recording_job,
    execute_single_recording_job,
    run_multi_recording_pipeline,
    run_single_recording_pipeline,
)

__all__ = [
    "MULTI_RECORDING_TRACKER_NAME",
    "SINGLE_RECORDING_TRACKER_NAME",
    "execute_multi_recording_job",
    "execute_single_recording_job",
    "run_multi_recording_pipeline",
    "run_single_recording_pipeline",
]
