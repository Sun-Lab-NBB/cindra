"""Provides configuration and runtime data classes for the single-day and multi-day sl-suite2p pipelines."""

from .multi_day import MultiDayConfiguration
from .single_day import (
    IOData,
    TimingData,
    CombinedData,
    ROIDetection,
    DetectionData,
    ROIStatistics,
    BaselineMethod,
    ExtractionData,
    RuntimeContext,
    RegistrationData,
    SingleDayRuntimeData,
    AcquisitionParameters,
    SingleDayConfiguration,
)

__all__ = [
    "AcquisitionParameters",
    "BaselineMethod",
    "CombinedData",
    "DetectionData",
    "ExtractionData",
    "IOData",
    "MultiDayConfiguration",
    "ROIDetection",
    "ROIStatistics",
    "RegistrationData",
    "RuntimeContext",
    "SingleDayConfiguration",
    "SingleDayRuntimeData",
    "TimingData",
]
