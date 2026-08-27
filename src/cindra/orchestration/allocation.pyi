from dataclasses import dataclass
from collections.abc import Mapping

from .jobs import (
    MultiRecordingJobNames as MultiRecordingJobNames,
    SingleRecordingJobNames as SingleRecordingJobNames,
)

BINARIZATION_WORKERS: int
REGISTRATION_WORKERS: int
PROCESSING_WORKERS: int
DISCOVERY_WORKERS: int
EXTRACTION_WORKERS: int
COMBINATION_WORKERS: int
REGISTRATION_MAXIMUM_WORKERS: int
PROCESSING_MAXIMUM_WORKERS: int
DISCOVERY_MAXIMUM_WORKERS: int
EXTRACTION_MAXIMUM_WORKERS: int
ALL_CORES_REQUEST: int
_RESERVED_CORES: int
_BINARIZATION_CONCURRENCY_LIMIT: int
_REGISTRATION_CONCURRENCY_RESERVATION: int
_PROCESSING_CONCURRENCY_RESERVATION: int
_BYTES_PER_MEGABYTE: int
_STAGE_WORKER_DEFAULTS: dict[SingleRecordingJobNames | MultiRecordingJobNames, int]

@dataclass(frozen=True, slots=True)
class ResourceClass:
    name: str
    workers_per_job: int
    maximum_workers_per_job: int | None
    concurrency_limit: int | None
    concurrency_reservation: int | None

_BINARIZATION_RESOURCES: ResourceClass
_REGISTRATION_RESOURCES: ResourceClass
_PROCESSING_RESOURCES: ResourceClass
_COMBINATION_RESOURCES: ResourceClass
_DISCOVERY_RESOURCES: ResourceClass
_EXTRACTION_RESOURCES: ResourceClass
RESOURCE_CLASS_BY_JOB_NAME: dict[str, ResourceClass]

def resolve_stage_workers(
    job_name: SingleRecordingJobNames | MultiRecordingJobNames, requested_workers: int | None = None
) -> int: ...
def resolve_core_budget() -> int: ...
def resolve_class_allocation(
    resource_class: ResourceClass,
    *,
    budget: int,
    job_count: int,
    workers_per_job: int | None,
    max_parallel_jobs: int | None,
) -> tuple[int, int]: ...
def resolve_dispatch_workers(
    resource_class: ResourceClass, *, free_cores: int, pending_jobs: int, running_jobs: int, concurrency_cap: int
) -> int: ...
def resolve_memory_budget_mb() -> int: ...
def summarize_class_allocation(
    class_workers: Mapping[str, int], class_capacities: Mapping[str, int], class_job_counts: Mapping[str, int]
) -> dict[str, dict[str, int]]: ...
