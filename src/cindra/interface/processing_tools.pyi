from enum import StrEnum
from typing import Any
from pathlib import Path
from threading import Lock, Thread
from dataclasses import field, dataclass
from collections.abc import Iterable

from ataraxis_data_structures import (
    JobState as JobState,
    ProcessingTracker,
)

from ..io import (
    resolve_multi_recording_contexts as resolve_multi_recording_contexts,
    resolve_single_recording_contexts as resolve_single_recording_contexts,
)
from ..pipelines import (
    MULTI_RECORDING_TRACKER_NAME as MULTI_RECORDING_TRACKER_NAME,
    SINGLE_RECORDING_TRACKER_NAME as SINGLE_RECORDING_TRACKER_NAME,
    run_multi_recording_pipeline as run_multi_recording_pipeline,
    run_single_recording_pipeline as run_single_recording_pipeline,
)
from ..allocation import (
    ALL_CORES_REQUEST as ALL_CORES_REQUEST,
    DISCOVERY_WORKERS as DISCOVERY_WORKERS,
    EXTRACTION_WORKERS as EXTRACTION_WORKERS,
    PROCESSING_WORKERS as PROCESSING_WORKERS,
    BINARIZATION_WORKERS as BINARIZATION_WORKERS,
    REGISTRATION_WORKERS as REGISTRATION_WORKERS,
    MULTI_RECORDING_PHASES as MULTI_RECORDING_PHASES,
    SINGLE_RECORDING_PHASES as SINGLE_RECORDING_PHASES,
    PrerequisiteScope as PrerequisiteScope,
    MultiRecordingJobNames as MultiRecordingJobNames,
    SingleRecordingJobNames as SingleRecordingJobNames,
    resolve_pipeline_jobs as resolve_pipeline_jobs,
    resolve_downstream_phases as resolve_downstream_phases,
    resolve_multi_recording_jobs as resolve_multi_recording_jobs,
    resolve_single_recording_jobs as resolve_single_recording_jobs,
)
from ..dataclasses import (
    MultiRecordingConfiguration as MultiRecordingConfiguration,
    SingleRecordingConfiguration as SingleRecordingConfiguration,
)
from .mcp_instance import mcp as mcp

_RESERVED_CORES: int
_MAXIMUM_PARALLEL_IO_JOBS: int
_MINIMUM_RECORDING_COUNT: int
_COMBINATION_WORKERS: int
_PROCESSING_MEMORY_GIGABYTES_PER_JOB: float
_BYTES_PER_GIGABYTE: int
_KIBIBYTES_PER_GIGABYTE: int
_MEMORY_INFO_PATH: Path
_AVAILABLE_MEMORY_FIELDS: int
_AVAILABLE_MEMORY_KEY: str
_PREREQUISITE_FAILURE_MESSAGE: str
_UNREACHABLE_PREREQUISITE_MESSAGE: str

class _AdmissionDecisions(StrEnum):
    ADMIT = "admit"
    WAIT = "wait"
    ABORT = "abort"

@dataclass(frozen=True, slots=True)
class _ResourceClass:
    name: str
    workers_per_job: int
    fixed_parallel_jobs: int | None
    memory_gigabytes_per_job: float

_BINARIZATION_RESOURCES: _ResourceClass
_REGISTRATION_RESOURCES: _ResourceClass
_PROCESSING_RESOURCES: _ResourceClass
_COMBINATION_RESOURCES: _ResourceClass
_DISCOVERY_RESOURCES: _ResourceClass
_EXTRACTION_RESOURCES: _ResourceClass
_RESOURCE_CLASS_BY_JOB_NAME: dict[str, _ResourceClass]

@dataclass(slots=True)
class _PendingJob:
    configuration_path: Path
    tracker_path: Path
    job_id: str
    single_recording: bool
    resource_class: _ResourceClass
    resolved_workers: int | None = ...
    @property
    def dispatch_key(self) -> tuple[str, str]: ...

@dataclass(slots=True)
class _JobExecutionState:
    all_jobs: dict[tuple[str, str], _PendingJob] = field(default_factory=dict)
    admission_pool: list[_PendingJob] = field(default_factory=list)
    pending_queues: dict[str, list[_PendingJob]] = field(default_factory=dict)
    active_threads: dict[str, dict[tuple[str, str], Thread]] = field(default_factory=dict)
    class_capacities: dict[str, int] = field(default_factory=dict)
    class_workers: dict[str, int] = field(default_factory=dict)
    cpu_budget: int = ...
    lock: Lock = field(default_factory=Lock)
    manager_thread: Thread | None = ...

_job_execution_state: _JobExecutionState | None

def get_recording_status_tool(recording_path: str) -> dict[str, object]: ...
def get_batch_status_overview_tool(root_directory: str) -> dict[str, object]: ...
def prepare_single_recording_batch_tool(
    recording_paths: list[str], configuration_path: str, recording_output_paths: list[str]
) -> dict[str, object]: ...
def prepare_multi_recording_batch_tool(dataset_configurations: list[dict[str, object]]) -> dict[str, object]: ...
def reset_processing_phases_tool(tracker_path: str, phases: list[str], pipeline_type: str) -> dict[str, object]: ...
def clean_processing_output_tool(
    recording_path: str, phases: list[str], pipeline_type: str, dataset: str = ""
) -> dict[str, object]: ...
def execute_processing_jobs_tool(
    jobs: list[dict[str, str]], *, workers_per_job: int | None = None, max_parallel_jobs: int | None = None
) -> dict[str, object]: ...
def get_processing_jobs_status_tool() -> dict[str, object]: ...
def get_active_execution_timing_tool() -> dict[str, object]: ...
def cancel_processing_jobs_tool() -> dict[str, object]: ...
def execute_full_pipeline_tool(
    pipeline_type: str,
    *,
    recording_paths: list[str] | None = None,
    configuration_path: str | None = None,
    recording_output_paths: list[str] | None = None,
    dataset_configurations: list[dict[str, object]] | None = None,
    workers_per_job: int | None = None,
    max_parallel_jobs: int | None = None,
) -> dict[str, object]: ...
def _check_active_session(action: str) -> dict[str, object] | None: ...
def _start_execution_session(
    all_jobs: dict[tuple[str, str], _PendingJob],
    workers_per_job: int | None,
    max_parallel_jobs: int | None,
    extra_result_fields: dict[str, object],
) -> dict[str, object]: ...
def _order_phases_by_execution(phase_names: Iterable[str], *, single_recording: bool) -> list[str]: ...
def _resolve_class_allocation(
    resource_class: _ResourceClass,
    *,
    budget: int,
    available_memory: float | None,
    job_count: int,
    workers_per_job: int | None,
    max_parallel_jobs: int | None,
) -> tuple[int, int]: ...
def _resolve_available_memory_gigabytes() -> float | None: ...
def _read_linux_available_memory_gigabytes() -> float | None: ...
def _resolve_prerequisite_job_ids(
    registry: dict[str, JobState], job_id: str, *, single_recording: bool
) -> tuple[list[str], str | None]: ...
def _collect_phase_job_ids(
    registry: dict[str, JobState], job_name: str, specifier: str | None, dependent_job_id: str
) -> tuple[list[str], str | None]: ...
def _validate_job_prerequisites(
    tracker: ProcessingTracker, job_id: str, *, single_recording: bool, submitted_job_ids: frozenset[str]
) -> str | None: ...
def _pipeline_worker(
    configuration_path: Path,
    job_id: str,
    tracker_path: Path,
    *,
    single_recording: bool = True,
    workers: int | None = None,
) -> None: ...
def _job_execution_manager() -> None: ...
def _reap_completed_threads(state: _JobExecutionState) -> None: ...
def _admit_ready_jobs(state: _JobExecutionState) -> bool: ...
def _resolve_job_admission(
    registry: dict[str, JobState], pending_job: _PendingJob
) -> tuple[_AdmissionDecisions, str]: ...
def _committed_cores(state: _JobExecutionState) -> int: ...
def _dispatch_admitted_jobs(state: _JobExecutionState) -> bool: ...
def _fail_pending_jobs(jobs: list[_PendingJob], message: str) -> None: ...
def _read_single_recording_tracker(tracker_path: Path, recording_path: Path) -> dict[str, object]: ...
def _read_multi_recording_tracker(tracker_path: Path) -> dict[str, object]: ...
def _delete_file(path: Path, deleted: list[str], errors: list[str]) -> None: ...
def _delete_directory(path: Path, deleted: list[str], errors: list[str]) -> None: ...
def _load_runtime_yaml(path: Path) -> dict[str, Any] | None: ...
