from enum import StrEnum
from pathlib import Path

from ataraxis_data_structures import ProcessingTracker

from ..io import (
    resolve_multi_recording_contexts as resolve_multi_recording_contexts,
    resolve_single_recording_contexts as resolve_single_recording_contexts,
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
)

SINGLE_RECORDING_TRACKER_NAME: str
MULTI_RECORDING_TRACKER_NAME: str

class SingleRecordingJobNames(StrEnum):
    BINARIZE = "binarization"
    PROCESS = "processing"
    COMBINE = "combination"

class MultiRecordingJobNames(StrEnum):
    DISCOVER = "discovery"
    EXTRACT = "extraction"

def run_single_recording_pipeline(
    configuration_path: Path,
    job_id: str | None = None,
    *,
    binarize: bool = False,
    process: bool = False,
    combine: bool = False,
    target_plane: int = -1,
) -> None: ...
def execute_single_recording_job(
    configuration_path: Path,
    job_name: SingleRecordingJobNames,
    specifier: str,
    job_id: str,
    tracker: ProcessingTracker,
    *,
    persist_bootstrap: bool = False,
) -> None: ...
def run_multi_recording_pipeline(
    configuration_path: Path,
    job_id: str | None = None,
    *,
    discover: bool = False,
    extract: bool = False,
    target_recording: str | None = None,
) -> None: ...
def execute_multi_recording_job(
    configuration_path: Path,
    job_name: MultiRecordingJobNames,
    specifier: str,
    job_id: str,
    tracker: ProcessingTracker,
    *,
    persist_bootstrap: bool = False,
) -> None: ...
def _load_single_recording_configuration(configuration_path: Path) -> tuple[SingleRecordingConfiguration, Path]: ...
def _load_multi_recording_configuration(configuration_path: Path) -> MultiRecordingConfiguration: ...
def _execute_single_recording_job(
    configuration: SingleRecordingConfiguration,
    job_name: SingleRecordingJobNames,
    specifier: str,
    job_id: str,
    tracker: ProcessingTracker,
) -> None: ...
def _execute_multi_recording_job(
    configuration: MultiRecordingConfiguration,
    job_name: MultiRecordingJobNames,
    specifier: str,
    job_id: str,
    tracker: ProcessingTracker,
) -> None: ...
