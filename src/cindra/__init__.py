"""Provides pipelines for processing neural imaging data and tracking Regions of Interest across multiple recordings.

Cindra is a ground-up reimplementation of the suite2p (https://github.com/MouseLand/suite2p) library that features a
novel multi-recording ROI tracking pipeline, optimized algorithms, expanded documentation, and an agentic interface
based on Claude.

See the source code repository (https://github.com/Sun-Lab-NBB/cindra) for documentation and additional information.

Authors: Ivan Kondratyev, Natalie Yeung
"""

# Configures numba threading layer for parallel execution across all modules. This must be set before any numba
# functions are compiled, hence it appears before other imports.
from numba import config  # type: ignore[import-untyped]

config.THREADING_LAYER = "tbb"

from ataraxis_base_utilities import console  # noqa: E402

from .pipelines import (  # noqa: E402
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    run_multi_recording_pipeline,
    run_single_recording_pipeline,
)
from .dataclasses import (  # noqa: E402
    MultiRecordingConfiguration,
    SingleRecordingConfiguration,
)

# Ensures console output is enabled whenever the cindra library is imported. The 'Console' class is
# used over 'print' for all terminal outputs. With minimal configuration, this class can be extended to log terminal
# outputs instead of or in addition to sending them to the terminal.
if not console.enabled:
    console.enable()

__all__ = [
    "MultiRecordingConfiguration",
    "MultiRecordingJobNames",
    "SingleRecordingConfiguration",
    "SingleRecordingJobNames",
    "run_multi_recording_pipeline",
    "run_single_recording_pipeline",
]
