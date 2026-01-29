"""Provides configuration and runtime data classes for the single-day and multi-day sl-suite2p pipelines."""

from .multi_day import MultiDayConfiguration
from .single_day import (
    IOData,
    TimingData,
    InputFormat,
    DetectionData,
    BaselineMethod,
    RuntimeContext,
    RegistrationData,
    SingleDayRuntimeData,
    SingleDayConfiguration,
)

__all__ = [
    "BaselineMethod",
    "DetectionData",
    "IOData",
    "InputFormat",
    "MultiDayConfiguration",
    "RegistrationData",
    "RuntimeContext",
    "SingleDayConfiguration",
    "SingleDayRuntimeData",
    "TimingData",
]
