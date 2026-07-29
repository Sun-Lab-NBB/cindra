"""Provides the pipeline job name enumerations and the measured CPU worker allocation defaults."""

from .workers import (
    ALL_CORES_REQUEST,
    PROCESSING_WORKERS,
    TIFF_DECODE_CEILING,
    BINARIZATION_WORKERS,
    REGISTRATION_WORKERS,
    resolve_stage_workers,
)
from .job_names import MultiRecordingJobNames, SingleRecordingJobNames

__all__ = [
    "ALL_CORES_REQUEST",
    "BINARIZATION_WORKERS",
    "PROCESSING_WORKERS",
    "REGISTRATION_WORKERS",
    "TIFF_DECODE_CEILING",
    "MultiRecordingJobNames",
    "SingleRecordingJobNames",
    "resolve_stage_workers",
]
