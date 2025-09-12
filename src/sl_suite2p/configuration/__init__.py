"""This package provides the configuration classes used to configure the single-day and the multi-day suite2p
pipelines and functions to instantiate these classes with default parameters.
"""

from .multi_day import MultiDayS2PConfiguration, generate_default_multiday_ops
from .single_day import SingleDayS2PConfiguration, generate_default_ops

__all__ = [
    "MultiDayS2PConfiguration",
    "SingleDayS2PConfiguration",
    "generate_default_multiday_ops",
    "generate_default_ops",
]
