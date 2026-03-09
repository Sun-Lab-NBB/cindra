"""Provides the processing pipeline orchestration logic for single-recording and multi-recording workflows."""

from .pipeline import (
    MULTI_RECORDING_TRACKER_NAME,
    SINGLE_RECORDING_TRACKER_NAME,
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    run_multi_recording_pipeline,
    run_single_recording_pipeline,
)

__all__ = [
    "MULTI_RECORDING_TRACKER_NAME",
    "MultiRecordingJobNames",
    "SINGLE_RECORDING_TRACKER_NAME",
    "SingleRecordingJobNames",
    "run_multi_recording_pipeline",
    "run_single_recording_pipeline",
]
