"""Provides the pipeline job name enumerations, the phase model, and the measured CPU worker allocation defaults."""

from .phases import (
    MULTI_RECORDING_PHASES,
    PLANE_SPECIFIER_PREFIX,
    SINGLE_RECORDING_PHASES,
    PipelinePhase,
    PrerequisiteScope,
    resolve_pipeline_jobs,
    resolve_plane_specifier,
    resolve_downstream_phases,
    resolve_multi_recording_jobs,
    resolve_single_recording_jobs,
    resolve_multi_recording_prerequisites,
    resolve_single_recording_prerequisites,
)
from .workers import (
    ALL_CORES_REQUEST,
    DISCOVERY_WORKERS,
    EXTRACTION_WORKERS,
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
    "DISCOVERY_WORKERS",
    "EXTRACTION_WORKERS",
    "MULTI_RECORDING_PHASES",
    "PLANE_SPECIFIER_PREFIX",
    "PROCESSING_WORKERS",
    "REGISTRATION_WORKERS",
    "SINGLE_RECORDING_PHASES",
    "TIFF_DECODE_CEILING",
    "MultiRecordingJobNames",
    "PipelinePhase",
    "PrerequisiteScope",
    "SingleRecordingJobNames",
    "resolve_downstream_phases",
    "resolve_multi_recording_jobs",
    "resolve_multi_recording_prerequisites",
    "resolve_pipeline_jobs",
    "resolve_plane_specifier",
    "resolve_single_recording_jobs",
    "resolve_single_recording_prerequisites",
    "resolve_stage_workers",
]
