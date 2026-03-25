from typing import Any
from pathlib import Path
from threading import Lock, Thread
from dataclasses import field, dataclass

from ataraxis_data_structures import ProcessingTracker

from ..io import (
    resolve_multi_recording_contexts as resolve_multi_recording_contexts,
    resolve_single_recording_contexts as resolve_single_recording_contexts,
)
from ..pipelines import (
    MULTI_RECORDING_TRACKER_NAME as MULTI_RECORDING_TRACKER_NAME,
    SINGLE_RECORDING_TRACKER_NAME as SINGLE_RECORDING_TRACKER_NAME,
    MultiRecordingJobNames as MultiRecordingJobNames,
    SingleRecordingJobNames as SingleRecordingJobNames,
    run_multi_recording_pipeline as run_multi_recording_pipeline,
    run_single_recording_pipeline as run_single_recording_pipeline,
)
from ..dataclasses import (
    MultiRecordingConfiguration as MultiRecordingConfiguration,
    SingleRecordingConfiguration as SingleRecordingConfiguration,
)
from .mcp_instance import mcp as mcp

_RESERVED_CORES: int
_MAXIMUM_PARALLEL_IO_JOBS: int
_MINIMUM_RECORDING_COUNT: int
_PREFERRED_WORKERS_PER_JOB: int
_MINIMUM_WORKERS_PER_JOB: int
_WORKER_MULTIPLE: int
_IO_BOUND_JOB_NAMES: frozenset[str]

@dataclass(slots=True)
class _PendingJob:
    configuration_path: Path
    tracker_path: Path
    job_id: str
    single_recording: bool
    io_bound: bool
    @property
    def dispatch_key(self) -> tuple[str, str]: ...

@dataclass(slots=True)
class _JobExecutionState:
    all_jobs: dict[tuple[str, str], _PendingJob] = field(default_factory=dict)
    io_pending_queue: list[_PendingJob] = field(default_factory=list)
    compute_pending_queue: list[_PendingJob] = field(default_factory=list)
    io_active_threads: dict[tuple[str, str], Thread] = field(default_factory=dict)
    compute_active_threads: dict[tuple[str, str], Thread] = field(default_factory=dict)
    max_parallel_jobs: int = ...
    lock: Lock = field(default_factory=Lock)
    manager_thread: Thread | None = ...
    phase_groups: list[list[_PendingJob]] = field(default_factory=list)

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
    jobs: list[dict[str, str]], *, workers_per_job: int = -1, max_parallel_jobs: int = -1
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
    workers_per_job: int = -1,
    max_parallel_jobs: int = -1,
) -> dict[str, object]: ...
def _start_execution_session(
    all_jobs: dict[tuple[str, str], _PendingJob],
    io_jobs: list[_PendingJob],
    compute_jobs: list[_PendingJob],
    phase_groups: list[list[_PendingJob]],
    workers_per_job: int,
    max_parallel_jobs: int,
    extra_result_fields: dict[str, object],
) -> dict[str, object]: ...
def _resolve_saturating_allocation(budget: int, total_jobs: int) -> tuple[int, int]: ...
def _validate_job_prerequisites(tracker: ProcessingTracker, job_id: str, *, single_recording: bool) -> str | None: ...
def _pipeline_worker(
    configuration_path: Path, job_id: str, tracker_path: Path, *, single_recording: bool = True
) -> None: ...
def _job_execution_manager() -> None: ...
def _read_single_recording_tracker(tracker_path: Path, recording_path: Path) -> dict[str, object]: ...
def _read_multi_recording_tracker(tracker_path: Path) -> dict[str, object]: ...
def _check_current_phase_failures(state: _JobExecutionState) -> bool: ...
def _fail_remaining_phase_groups(state: _JobExecutionState) -> None: ...
def _delete_file(path: Path, deleted: list[str], errors: list[str]) -> None: ...
def _delete_directory(path: Path, deleted: list[str], errors: list[str]) -> None: ...
def _load_runtime_yaml(path: Path) -> dict[str, Any] | None: ...
