"""Provides the job name enumerations that identify the stages of the single and multi-recording pipelines.

The enumerations live in this leaf package alongside the worker defaults that key off them, so the io and interface
packages can import the package directly while the pipeline package keeps importing io.
"""

from enum import StrEnum


class SingleRecordingJobNames(StrEnum):
    """Defines the job names for the single-recording processing pipeline components.

    Notes:
        The members are declared in execution order. That order is rendered into error messages and seeds the phase
        validation sets used by the interface layer, so it must match the order in which the pipeline runs the stages.
    """

    BINARIZE = "binarization"
    """The name for the binarization (step 1) processing job."""
    REGISTER = "registration"
    """The generic name for the plane-registration (step 2) job, which removes motion and computes the
    registration-quality principal components. During runtime, the registered plane is identified by the tracker's
    specifier field using the format 'plane_{plane_index}'."""
    PROCESS = "processing"
    """The generic name for the plane-processing (step 3) job, which discovers ROIs and extracts their fluorescence.
    During runtime, the processed plane is identified by the tracker's specifier field using the format
    'plane_{plane_index}'."""
    COMBINE = "combination"
    """The name for the combination (step 4) processing job."""


class MultiRecordingJobNames(StrEnum):
    """Defines the job names for the multi-recording processing pipeline components."""

    DISCOVER = "discovery"
    """The name for the ROI discovery (step 1) processing job."""
    EXTRACT = "extraction"
    """The generic name for the fluorescence extraction (step 2) processing job. During runtime, the processed recording
    is identified by the tracker's specifier field, which stores the recording ID string."""
