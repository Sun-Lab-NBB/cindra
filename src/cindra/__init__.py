"""A reimplementation of the suite2p library with novel data architecture and across-day tracking pipeline.

cindra is a reimplementation of the popular suite2p (https://github.com/MouseLand/suite2p) library with expanded
documentation, modern Python support, and a new multi-recording ROI tracking pipeline based on the OSM manuscript
(https://www.nature.com/articles/s41586-024-08548-w).

Original suite2p copyright:
Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu.

Sun lab code changes copyright:
Copyright © 2025 Cornell University, Authored by Ivan Kondratyev and Kushaan Gupta.

See the `source code repository <https://github.com/Sun-Lab-NBB/cindra>`_ for documentation and additional
information.

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
