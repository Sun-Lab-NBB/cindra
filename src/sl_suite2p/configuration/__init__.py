"""This package provides the configuration classes used to configure the single-day and the multi-day suite2p
pipelines and functions to instantiate these classes with default parameters.
"""

from .multi_day import MultiDayS2PConfiguration, generate_default_multiday_ops
from .single_day import (
    IOData,
    RuntimeData,
    SingleDayS2PConfiguration,
    generate_default_configuration,
)

__all__ = [
    "IOData",
    "MultiDayS2PConfiguration",
    "RuntimeData",
    "SingleDayS2PConfiguration",
    "generate_default_configuration",
    "generate_default_multiday_ops",
]
