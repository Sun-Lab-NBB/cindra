"""Provides pipelines for processing neural imaging data and tracking Regions of Interest across multiple recordings.

See the `documentation <https://cindra-api-docs.netlify.app/>`_ for the description of available assets. See the
`source code repository <https://github.com/Sun-Lab-NBB/cindra>`_ for more details.

Authors: Ivan Kondratyev, Natalie Yeung
"""

# Configures numba threading layer for parallel execution across all modules. This must be set before any numba
# functions are compiled, hence it appears before other imports. macOS uses OpenMP (libomp via llvm-openmp) because
# tbb4py publishes no Apple Silicon wheel; all other platforms use TBB for lower overhead on flat prange loops.
import sys

from numba import config

config.THREADING_LAYER = "omp" if sys.platform == "darwin" else "tbb"

from ataraxis_base_utilities import console  # noqa: E402

from .pipelines import (  # noqa: E402
    execute_multi_recording_job,
    execute_single_recording_job,
    run_multi_recording_pipeline,
    run_single_recording_pipeline,
)
from .allocation import (  # noqa: E402
    DISCOVERY_WORKERS,
    EXTRACTION_WORKERS,
    PROCESSING_WORKERS,
    TIFF_DECODE_CEILING,
    BINARIZATION_WORKERS,
    REGISTRATION_WORKERS,
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    resolve_stage_workers,
)
from .dataclasses import (  # noqa: E402
    MultiRecordingConfiguration,
    SingleRecordingConfiguration,
)

# Ensures console output is enabled whenever the cindra library is imported. The 'Console' class is
# used over 'print' for all terminal outputs. With minimal configuration, this class can be extended to log terminal
# outputs instead of or in addition to sending them to the terminal.
if not console.enabled:  # pragma: no branch — the console-enabled state is only reachable as False on first import.
    console.enable()

__all__ = [
    "BINARIZATION_WORKERS",
    "DISCOVERY_WORKERS",
    "EXTRACTION_WORKERS",
    "PROCESSING_WORKERS",
    "REGISTRATION_WORKERS",
    "TIFF_DECODE_CEILING",
    "MultiRecordingConfiguration",
    "MultiRecordingJobNames",
    "SingleRecordingConfiguration",
    "SingleRecordingJobNames",
    "execute_multi_recording_job",
    "execute_single_recording_job",
    "resolve_stage_workers",
    "run_multi_recording_pipeline",
    "run_single_recording_pipeline",
]
