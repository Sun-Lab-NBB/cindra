from enum import StrEnum
from pathlib import Path
from threading import Lock, Thread
from dataclasses import field, dataclass
from concurrent.futures import Future, Executor

from ataraxis_data_structures import JobState as JobState

from .gpu import (
    ALL_DEVICES_REQUEST as ALL_DEVICES_REQUEST,
    resolve_gpu_devices as resolve_gpu_devices,
)
from .jobs import (
    PREREQUISITE_FAILURE_MESSAGE as PREREQUISITE_FAILURE_MESSAGE,
    UNREACHABLE_PREREQUISITE_MESSAGE as UNREACHABLE_PREREQUISITE_MESSAGE,
    SingleRecordingJobNames as SingleRecordingJobNames,
    resolve_prerequisite_job_ids as resolve_prerequisite_job_ids,
)
from .pipeline import (
    run_multi_recording_pipeline as run_multi_recording_pipeline,
    run_single_recording_pipeline as run_single_recording_pipeline,
)
from .allocation import (
    ALL_CORES_REQUEST as ALL_CORES_REQUEST,
    RESOURCE_CLASS_BY_JOB_NAME as RESOURCE_CLASS_BY_JOB_NAME,
    ResourceClass as ResourceClass,
    resolve_core_budget as resolve_core_budget,
    class_requires_device as class_requires_device,
    resolve_class_allocation as resolve_class_allocation,
    resolve_dispatch_workers as resolve_dispatch_workers,
    resolve_memory_budget_mb as resolve_memory_budget_mb,
    summarize_class_allocation as summarize_class_allocation,
)

_DISPATCH_POLL_MILLISECONDS: int
_WORKER_THREAD_CEILING: int
_POOL_START_METHOD: str
_BROKEN_POOL_MESSAGE: str
_CANCELED_JOB_MESSAGE: str

class _AdmissionDecisions(StrEnum):
    ADMIT = "admit"
    WAIT = "wait"
    ABORT = "abort"

class _JobOutcomes(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    ABANDONED = "abandoned"

@dataclass(slots=True)
class PendingJob:
    configuration_path: Path
    tracker_path: Path
    job_id: str
    single_recording: bool
    resource_class: ResourceClass
    resolved_workers: int | None = ...
    assigned_device: int | None = ...
    memory_megabytes: int = ...
    @property
    def dispatch_key(self) -> tuple[str, str]: ...

@dataclass(slots=True)
class JobExecutionState:
    all_jobs: dict[tuple[str, str], PendingJob] = field(default_factory=dict)
    admission_pool: list[PendingJob] = field(default_factory=list)
    pending_queues: dict[str, list[PendingJob]] = field(default_factory=dict)
    active_futures: dict[str, dict[tuple[str, str], Future[None]]] = field(default_factory=dict)
    class_capacities: dict[str, int] = field(default_factory=dict)
    class_workers: dict[str, int] = field(default_factory=dict)
    class_reservations: dict[str, int] = field(default_factory=dict)
    elastic_workers: bool = ...
    cpu_budget: int = ...
    memory_budget_mb: int = ...
    device_budget: int = ...
    available_devices: list[int] = field(default_factory=list)
    lock: Lock = field(default_factory=Lock)
    manager_thread: Thread | None = ...

_execution_state: JobExecutionState | None

def get_execution_state() -> JobExecutionState | None: ...
def set_execution_state(state: JobExecutionState | None) -> None: ...
def resolve_session_load() -> tuple[int, int]: ...
def start_execution_session(
    all_jobs: dict[tuple[str, str], PendingJob],
    workers_per_job: int | None,
    max_parallel_jobs: int | None,
    gpu_devices: list[int] | None = None,
) -> dict[str, object]: ...
def cancel_execution_session() -> tuple[int, int]: ...
def _job_execution_manager(state: JobExecutionState) -> None: ...
def _resolve_session_devices(gpu_devices: list[int] | None) -> list[int]: ...
def _validate_session_device_agreement(
    all_jobs: dict[tuple[str, str], PendingJob], session_devices: list[int]
) -> None: ...
def _clear_owned_session(state: JobExecutionState) -> None: ...
def _reap_completed_jobs(state: JobExecutionState) -> None: ...
def _resolve_job_outcome(future: Future[None]) -> tuple[_JobOutcomes, str]: ...
def _admit_ready_jobs(state: JobExecutionState) -> bool: ...
def _resolve_job_admission(
    registry: dict[str, JobState], pending_job: PendingJob
) -> tuple[_AdmissionDecisions, str]: ...
def _dispatch_admitted_jobs(state: JobExecutionState, pool: Executor) -> bool: ...
def _dispatch_pass(state: JobExecutionState, pool: Executor, *, release_reservations: bool) -> bool: ...
def _release_device(state: JobExecutionState, job: PendingJob | None) -> None: ...
def _create_job_pool(max_workers: int) -> Executor: ...
def _resolve_pool_size(state: JobExecutionState) -> int: ...
def _committed_memory(state: JobExecutionState) -> int: ...
def _count_competing_classes(state: JobExecutionState) -> int: ...
def _class_is_elastic(resource_class: ResourceClass) -> bool: ...
def _committed_cores(state: JobExecutionState) -> int: ...
def _resolve_committed_width(state: JobExecutionState, class_name: str, dispatch_key: tuple[str, str]) -> int: ...
def _pipeline_worker(
    configuration_path: Path,
    job_id: str,
    tracker_path: Path,
    *,
    single_recording: bool = True,
    workers: int | None = None,
    device: int | None = None,
) -> None: ...
def _fail_broken_session(state: JobExecutionState) -> None: ...
def _fail_dispatched_job(job: PendingJob, message: str) -> None: ...
def _fail_pending_jobs(jobs: list[PendingJob], message: str) -> None: ...
