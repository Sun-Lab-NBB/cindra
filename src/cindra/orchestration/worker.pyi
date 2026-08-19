from pathlib import Path

from ataraxis_data_structures import ProcessingTracker as ProcessingTracker

from ..io import (
    RecordingPlanes as RecordingPlanes,
    DatasetRecordings as DatasetRecordings,
    resolve_recording_planes as resolve_recording_planes,
    resolve_dataset_recordings as resolve_dataset_recordings,
    resolve_multi_recording_contexts as resolve_multi_recording_contexts,
    resolve_single_recording_contexts as resolve_single_recording_contexts,
)
from .jobs import (
    MultiRecordingJobNames as MultiRecordingJobNames,
    SingleRecordingJobNames as SingleRecordingJobNames,
)
from ..layout import (
    OUTPUT_DIRECTORY_NAME as OUTPUT_DIRECTORY_NAME,
    parse_plane_specifier as parse_plane_specifier,
)
from ..pipelines import (
    process_plane as process_plane,
    binarize_recording as binarize_recording,
    save_combined_data as save_combined_data,
    register_recording_plane as register_recording_plane,
    discover_multi_recording_cells as discover_multi_recording_cells,
    extract_multi_recording_fluorescence as extract_multi_recording_fluorescence,
)
from .allocation import resolve_stage_workers as resolve_stage_workers
from ..dataclasses import (
    RuntimeContext as RuntimeContext,
    MultiRecordingConfiguration as MultiRecordingConfiguration,
    SingleRecordingConfiguration as SingleRecordingConfiguration,
)

_MINIMUM_DATASET_RECORDINGS: int

def execute_single_recording_job(
    configuration_path: Path,
    job_name: SingleRecordingJobNames,
    specifier: str,
    job_id: str,
    tracker: ProcessingTracker,
    *,
    workers: int | None = None,
) -> None: ...
def execute_multi_recording_job(
    configuration_path: Path,
    job_name: MultiRecordingJobNames,
    specifier: str,
    job_id: str,
    tracker: ProcessingTracker,
    *,
    workers: int | None = None,
) -> None: ...
def load_single_recording_configuration(configuration_path: Path) -> tuple[SingleRecordingConfiguration, Path]: ...
def load_multi_recording_configuration(configuration_path: Path) -> MultiRecordingConfiguration: ...
def dispatch_single_recording_job(
    configuration: SingleRecordingConfiguration,
    job_name: SingleRecordingJobNames,
    specifier: str,
    job_id: str,
    tracker: ProcessingTracker,
    workers: int | None,
) -> None: ...
def dispatch_multi_recording_job(
    configuration: MultiRecordingConfiguration,
    job_name: MultiRecordingJobNames,
    specifier: str,
    job_id: str,
    tracker: ProcessingTracker,
    workers: int | None,
) -> None: ...
def prime_recording(configuration_path: Path) -> RecordingPlanes: ...
def prime_dataset(configuration_path: Path) -> DatasetRecordings: ...
def _resolve_job_plane_index(job_name: str, specifier: str) -> int: ...
