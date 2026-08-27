from pathlib import Path

from ..io import (
    resolve_multi_recording_contexts as resolve_multi_recording_contexts,
    resolve_single_recording_contexts as resolve_single_recording_contexts,
)
from .gpu import verify_gpu_runtime as verify_gpu_runtime
from .jobs import (
    PER_PLANE_JOB_NAMES as PER_PLANE_JOB_NAMES,
    MultiRecordingJobNames as MultiRecordingJobNames,
    SingleRecordingJobNames as SingleRecordingJobNames,
    resolve_multi_recording_jobs as resolve_multi_recording_jobs,
    resolve_single_recording_jobs as resolve_single_recording_jobs,
)
from .openmp import verify_openmp_runtime as verify_openmp_runtime
from .worker import (
    dispatch_multi_recording_job as dispatch_multi_recording_job,
    dispatch_single_recording_job as dispatch_single_recording_job,
    load_multi_recording_configuration as load_multi_recording_configuration,
    load_single_recording_configuration as load_single_recording_configuration,
)
from ..layout import (
    OUTPUT_DIRECTORY_NAME as OUTPUT_DIRECTORY_NAME,
    MULTI_RECORDING_TRACKER_FILENAME as MULTI_RECORDING_TRACKER_FILENAME,
    SINGLE_RECORDING_TRACKER_FILENAME as SINGLE_RECORDING_TRACKER_FILENAME,
    resolve_plane_specifier as resolve_plane_specifier,
)
from ..dataclasses import (
    RegistrationBackend as RegistrationBackend,
    SingleRecordingConfiguration as SingleRecordingConfiguration,
)

def run_single_recording_pipeline(
    configuration_path: Path,
    job_id: str | None = None,
    *,
    binarize: bool = False,
    register: bool = False,
    process: bool = False,
    combine: bool = False,
    target_plane: int = -1,
    binarization_workers: int | None = None,
    registration_workers: int | None = None,
    processing_workers: int | None = None,
    registration_device: int | None = None,
) -> None: ...
def run_multi_recording_pipeline(
    configuration_path: Path,
    job_id: str | None = None,
    *,
    discover: bool = False,
    extract: bool = False,
    target_recording: str | None = None,
    discovery_workers: int | None = None,
    extraction_workers: int | None = None,
) -> None: ...
def _verify_registration_device(configuration: SingleRecordingConfiguration, job_names: list[str]) -> None: ...
