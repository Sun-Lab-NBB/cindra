from typing import Any
from pathlib import Path

from ataraxis_data_structures import (
    JobState as JobState,
    ProcessingTracker,
)

from ..io import (
    is_plane_converted as is_plane_converted,
    find_data_directory as find_data_directory,
    is_dataset_discovered as is_dataset_discovered,
    resolve_recording_planes as resolve_recording_planes,
    resolve_dataset_recordings as resolve_dataset_recordings,
    resolve_source_frame_geometry as resolve_source_frame_geometry,
    resolve_multi_recording_contexts as resolve_multi_recording_contexts,
)
from ..layout import (
    OUTPUT_DIRECTORY_NAME as OUTPUT_DIRECTORY_NAME,
    PLANE_SPECIFIER_PREFIX as PLANE_SPECIFIER_PREFIX,
    DEFORMED_MASKS_FILENAME as DEFORMED_MASKS_FILENAME,
    CHANNEL_1_BINARY_FILENAME as CHANNEL_1_BINARY_FILENAME,
    CHANNEL_2_BINARY_FILENAME as CHANNEL_2_BINARY_FILENAME,
    COMBINED_METADATA_FILENAME as COMBINED_METADATA_FILENAME,
    DETECTION_DATA_DIRECTORY_NAME as DETECTION_DATA_DIRECTORY_NAME,
    MULTI_RECORDING_DIRECTORY_NAME as MULTI_RECORDING_DIRECTORY_NAME,
    MULTI_RECORDING_TRACKER_FILENAME as MULTI_RECORDING_TRACKER_FILENAME,
    REGISTRATION_DATA_DIRECTORY_NAME as REGISTRATION_DATA_DIRECTORY_NAME,
    TRACKING_TEMPLATE_MASKS_FILENAME as TRACKING_TEMPLATE_MASKS_FILENAME,
    SINGLE_RECORDING_TRACKER_FILENAME as SINGLE_RECORDING_TRACKER_FILENAME,
    MULTI_RECORDING_ARRAYS_DIRECTORY_NAME as MULTI_RECORDING_ARRAYS_DIRECTORY_NAME,
    MULTI_RECORDING_RUNTIME_DATA_FILENAME as MULTI_RECORDING_RUNTIME_DATA_FILENAME,
    MULTI_RECORDING_CONFIGURATION_FILENAME as MULTI_RECORDING_CONFIGURATION_FILENAME,
    SINGLE_RECORDING_CONFIGURATION_FILENAME as SINGLE_RECORDING_CONFIGURATION_FILENAME,
    DetectionImages as DetectionImages,
    RecordingArrays as RecordingArrays,
    resolve_array_name as resolve_array_name,
    resolve_channel_2_name as resolve_channel_2_name,
    resolve_plane_specifier as resolve_plane_specifier,
)
from ..dataclasses import (
    MultiRecordingConfiguration as MultiRecordingConfiguration,
    SingleRecordingConfiguration as SingleRecordingConfiguration,
)
from .mcp_instance import mcp as mcp
from ..orchestration import (
    SINGLE_RECORDING_PHASES as SINGLE_RECORDING_PHASES,
    RESOURCE_CLASS_BY_JOB_NAME as RESOURCE_CLASS_BY_JOB_NAME,
    PendingJob as PendingJob,
    OpenMPStatus as OpenMPStatus,
    MultiRecordingJobNames as MultiRecordingJobNames,
    SingleRecordingJobNames as SingleRecordingJobNames,
    prime_recording as prime_recording,
    get_execution_state as get_execution_state,
    set_execution_state as set_execution_state,
    resolve_session_load as resolve_session_load,
    resolve_pipeline_jobs as resolve_pipeline_jobs,
    resolve_openmp_runtime as resolve_openmp_runtime,
    start_execution_session as start_execution_session,
    cancel_execution_session as cancel_execution_session,
    size_multi_recording_job as size_multi_recording_job,
    order_phases_by_execution as order_phases_by_execution,
    resolve_downstream_phases as resolve_downstream_phases,
    size_single_recording_job as size_single_recording_job,
    resolve_recording_geometry as resolve_recording_geometry,
    validate_job_prerequisites as validate_job_prerequisites,
    resolve_multi_recording_jobs as resolve_multi_recording_jobs,
    resolve_single_recording_jobs as resolve_single_recording_jobs,
    load_multi_recording_configuration as load_multi_recording_configuration,
    load_single_recording_configuration as load_single_recording_configuration,
    resolve_multi_recording_job_universe as resolve_multi_recording_job_universe,
    resolve_single_recording_job_universe as resolve_single_recording_job_universe,
    estimate_multi_recording_job_memory_mb as estimate_multi_recording_job_memory_mb,
    estimate_single_recording_job_memory_mb as estimate_single_recording_job_memory_mb,
)

_MINIMUM_RECORDING_COUNT: int
_SECONDS_PER_HOUR: float
_TIFF_HINT_SEARCH_DEPTH: int

def get_recording_status_tool(output_root: str) -> dict[str, object]: ...
def get_batch_status_overview_tool(root_directory: str) -> dict[str, object]: ...
def prepare_single_recording_batch_tool(
    raw_data_paths: list[str], configuration_path: str, output_roots: list[str]
) -> dict[str, object]: ...
def prepare_multi_recording_batch_tool(dataset_configurations: list[dict[str, object]]) -> dict[str, object]: ...
def reset_processing_phases_tool(tracker_path: str, phases: list[str], pipeline_type: str) -> dict[str, object]: ...
def clean_processing_output_tool(
    output_root: str, phases: list[str], pipeline_type: str, dataset: str = ""
) -> dict[str, object]: ...
def execute_processing_jobs_tool(
    jobs: list[dict[str, str]], *, workers_per_job: int | None = None, max_parallel_jobs: int | None = None
) -> dict[str, object]: ...
def get_processing_jobs_status_tool(*, summary_only: bool = False) -> dict[str, object]: ...
def get_active_execution_timing_tool() -> dict[str, object]: ...
def cancel_processing_jobs_tool() -> dict[str, object]: ...
def execute_full_pipeline_tool(
    pipeline_type: str,
    *,
    raw_data_paths: list[str] | None = None,
    configuration_path: str | None = None,
    output_roots: list[str] | None = None,
    dataset_configurations: list[dict[str, object]] | None = None,
    workers_per_job: int | None = None,
    max_parallel_jobs: int | None = None,
) -> dict[str, object]: ...
def size_pipeline_jobs_tool(
    configuration_path: str, pipeline_type: str, planned_roi_count: int | None = None
) -> dict[str, object]: ...
def check_threading_runtime_tool() -> dict[str, object]: ...
def get_pipeline_job_universe_tool(configuration_path: str, pipeline_type: str) -> dict[str, object]: ...
def _collapse_whitespace(text: str) -> str: ...
def _resolve_raw_data_failure(raw_data_path: Path, ignored_file_names: tuple[str, ...]) -> str | None: ...
def _resolve_tiff_subdirectory(raw_data_path: Path, ignored_file_names: tuple[str, ...]) -> Path | None: ...
def _resolve_single_recording_path_conflicts(
    recording_key: str, configuration_path: Path, output_root: Path, data_path: Path
) -> list[dict[str, str]]: ...
def _resolve_multi_recording_path_conflicts(
    dataset_key: str, configuration_path: Path, output_roots: tuple[Path, ...]
) -> list[dict[str, str]]: ...
def _resolve_repeat_flag_warnings(
    tracker_path: Path, phase_names: list[str], *, single_recording: bool
) -> list[str]: ...
def _check_active_session(action: str) -> dict[str, object] | None: ...
def _start_session(
    all_jobs: dict[tuple[str, str], PendingJob],
    workers_per_job: int | None,
    max_parallel_jobs: int | None,
    extra_result_fields: dict[str, object],
) -> dict[str, object]: ...
def _group_jobs_by_name(registry: dict[str, JobState], job_name: str) -> dict[str, JobState]: ...
def _read_single_recording_tracker(tracker_path: Path, output_root: Path) -> dict[str, object]: ...
def _read_multi_recording_tracker(tracker_path: Path) -> dict[str, object]: ...
def _delete_file(path: Path, deleted: list[str], errors: list[str]) -> None: ...
def _delete_directory(path: Path, deleted: list[str], errors: list[str]) -> None: ...
def _load_runtime_yaml(path: Path) -> dict[str, Any] | None: ...
def _resolve_dataset_phase_jobs(
    manifest_dict: dict[str, Any], configuration_path: Path, tracker_path: Path
) -> tuple[list[PendingJob], list[PendingJob]]: ...
def _resolve_recording_phase_jobs(
    manifest_dict: dict[str, Any], configuration_path: Path, tracker_path: Path
) -> tuple[list[PendingJob], list[PendingJob], list[PendingJob], list[PendingJob]]: ...
def _estimate_pending_job_memory(configuration_path: Path, job_name: str, specifier: str, *, single: bool) -> int: ...
def _manifest_entry(identifiers: dict[tuple[str, str], str], job_name: str, specifier: str) -> dict[str, object]: ...
def _resolve_job_identifiers(tracker: ProcessingTracker, jobs: list[tuple[str, str]]) -> dict[tuple[str, str], str]: ...
def _size_single_recording_universe(
    configuration_path: Path, planned_roi_count: int | None
) -> list[tuple[str, str, int, int]]: ...
def _size_multi_recording_universe(configuration_path: Path) -> list[tuple[str, str, int, int]]: ...
