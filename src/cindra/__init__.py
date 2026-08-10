"""Provides pipelines for processing neural imaging data and tracking Regions of Interest across multiple recordings.

See the `documentation <https://cindra-api-docs.netlify.app/>`_ for the description of available assets. See the
`source code repository <https://github.com/Sun-Lab-NBB/cindra>`_ for more details.

Authors: Ivan Kondratyev, Natalie Yeung
"""

# Configures numba threading layer for parallel execution across all modules. This must be set before any numba
# functions are compiled, hence it appears before other imports. macOS uses OpenMP because the numba macOS wheel ships
# no tbbpool extension, which leaves the TBB layer unavailable there whatever runtime is installed. All other
# platforms use TBB for lower overhead on flat prange loops.
import sys

from numba import config

config.THREADING_LAYER = "omp" if sys.platform == "darwin" else "tbb"

from ataraxis_base_utilities import console  # noqa: E402

from .io import (  # noqa: E402
    TIFF_DECODE_CEILING,
    RecordingPlanes,
    DatasetRecordings,
    is_plane_registered,
    is_dataset_discovered,
    is_recording_processed,
    resolve_recording_planes,
    resolve_dataset_recordings,
)
from .layout import (  # noqa: E402
    PARAMETERS_FILENAME,
    OUTPUT_DIRECTORY_NAME,
    PLANE_SPECIFIER_PREFIX,
    COMBINED_METADATA_FILENAME,
    MULTI_RECORDING_DIRECTORY_NAME,
    MULTI_RECORDING_TRACKER_FILENAME,
    SINGLE_RECORDING_TRACKER_FILENAME,
    DetectionImages,
    RecordingArrays,
    RegistrationArrays,
    MultiRecordingArrays,
    resolve_array_path,
    resolve_plane_path,
    resolve_output_path,
    resolve_dataset_path,
    parse_plane_specifier,
    resolve_plane_specifier,
)
from .dataclasses import (  # noqa: E402
    MultiRecordingConfiguration,
    SingleRecordingConfiguration,
)
from .orchestration import (  # noqa: E402
    DISCOVERY_WORKERS,
    EXTRACTION_WORKERS,
    PROCESSING_WORKERS,
    COMBINATION_WORKERS,
    BINARIZATION_WORKERS,
    REGISTRATION_WORKERS,
    MultiRecordingJobs,
    SingleRecordingJobs,
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    resolve_stage_workers,
    resolve_recording_geometry,
    execute_multi_recording_job,
    execute_single_recording_job,
    run_multi_recording_pipeline,
    run_single_recording_pipeline,
    resolve_multi_recording_job_universe,
    resolve_single_recording_job_universe,
    estimate_multi_recording_job_memory_mb,
    estimate_single_recording_job_memory_mb,
)

# Ensures console output is enabled whenever the cindra library is imported. The 'Console' class is
# used over 'print' for all terminal outputs. With minimal configuration, this class can be extended to log terminal
# outputs instead of or in addition to sending them to the terminal.
if not console.enabled:  # pragma: no branch - the console-enabled state is only reachable as False on first import.
    console.enable()

__all__ = [
    "BINARIZATION_WORKERS",
    "COMBINATION_WORKERS",
    "COMBINED_METADATA_FILENAME",
    "DISCOVERY_WORKERS",
    "EXTRACTION_WORKERS",
    "MULTI_RECORDING_DIRECTORY_NAME",
    "MULTI_RECORDING_TRACKER_FILENAME",
    "OUTPUT_DIRECTORY_NAME",
    "PARAMETERS_FILENAME",
    "PLANE_SPECIFIER_PREFIX",
    "PROCESSING_WORKERS",
    "REGISTRATION_WORKERS",
    "SINGLE_RECORDING_TRACKER_FILENAME",
    "TIFF_DECODE_CEILING",
    "DatasetRecordings",
    "DetectionImages",
    "MultiRecordingArrays",
    "MultiRecordingConfiguration",
    "MultiRecordingJobNames",
    "MultiRecordingJobs",
    "RecordingArrays",
    "RecordingPlanes",
    "RegistrationArrays",
    "SingleRecordingConfiguration",
    "SingleRecordingJobNames",
    "SingleRecordingJobs",
    "estimate_multi_recording_job_memory_mb",
    "estimate_single_recording_job_memory_mb",
    "execute_multi_recording_job",
    "execute_single_recording_job",
    "is_dataset_discovered",
    "is_plane_registered",
    "is_recording_processed",
    "parse_plane_specifier",
    "resolve_array_path",
    "resolve_dataset_path",
    "resolve_dataset_recordings",
    "resolve_multi_recording_job_universe",
    "resolve_output_path",
    "resolve_plane_path",
    "resolve_plane_specifier",
    "resolve_recording_geometry",
    "resolve_recording_planes",
    "resolve_single_recording_job_universe",
    "resolve_stage_workers",
    "run_multi_recording_pipeline",
    "run_single_recording_pipeline",
]
