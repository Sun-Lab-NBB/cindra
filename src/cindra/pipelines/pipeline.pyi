from pathlib import Path

from ataraxis_data_structures import ProcessingTracker

from ..io import (
    resolve_multi_recording_contexts as resolve_multi_recording_contexts,
    resolve_single_recording_contexts as resolve_single_recording_contexts,
)
from ..allocation import (
    SINGLE_RECORDING_PHASES as SINGLE_RECORDING_PHASES,
    MultiRecordingJobNames as MultiRecordingJobNames,
    SingleRecordingJobNames as SingleRecordingJobNames,
    resolve_stage_workers as resolve_stage_workers,
    resolve_plane_specifier as resolve_plane_specifier,
    resolve_multi_recording_jobs as resolve_multi_recording_jobs,
    resolve_single_recording_jobs as resolve_single_recording_jobs,
)
from ..dataclasses import (
    RuntimeContext as RuntimeContext,
    MultiRecordingConfiguration as MultiRecordingConfiguration,
    SingleRecordingConfiguration as SingleRecordingConfiguration,
)
from .multi_recording import (
    discover_multi_recording_cells as discover_multi_recording_cells,
    extract_multi_recording_fluorescence as extract_multi_recording_fluorescence,
)
from .single_recording import (
    process_plane as process_plane,
    binarize_recording as binarize_recording,
    save_combined_data as save_combined_data,
    register_recording_plane as register_recording_plane,
)

SINGLE_RECORDING_TRACKER_NAME: str
MULTI_RECORDING_TRACKER_NAME: str
_PER_PLANE_JOB_NAMES: frozenset[str]

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
) -> None: ...
def execute_single_recording_job(
    configuration_path: Path,
    job_name: SingleRecordingJobNames,
    specifier: str,
    job_id: str,
    tracker: ProcessingTracker,
    *,
    persist_bootstrap: bool = False,
    workers: int | None = None,
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
def execute_multi_recording_job(
    configuration_path: Path,
    job_name: MultiRecordingJobNames,
    specifier: str,
    job_id: str,
    tracker: ProcessingTracker,
    *,
    persist_bootstrap: bool = False,
    workers: int | None = None,
) -> None: ...
def _load_single_recording_configuration(configuration_path: Path) -> tuple[SingleRecordingConfiguration, Path]: ...
def _load_multi_recording_configuration(configuration_path: Path) -> MultiRecordingConfiguration: ...
def _execute_single_recording_job(
    configuration: SingleRecordingConfiguration,
    job_name: SingleRecordingJobNames,
    specifier: str,
    job_id: str,
    tracker: ProcessingTracker,
    workers: int | None,
) -> None: ...
def _execute_multi_recording_job(
    configuration: MultiRecordingConfiguration,
    job_name: MultiRecordingJobNames,
    specifier: str,
    job_id: str,
    tracker: ProcessingTracker,
    workers: int | None,
) -> None: ...
