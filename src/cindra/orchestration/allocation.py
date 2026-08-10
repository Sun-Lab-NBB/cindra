"""Provides the measured CPU worker defaults of the single and multi-recording pipeline stages, the resource-class
model that sizes a batch of jobs, and the resolvers that turn a caller's request into a concrete allocation.

The defaults encode the knee of each stage's measured scaling curve, so a caller that expresses no preference gets the
allocation that maximizes batch throughput rather than the allocation that minimizes the wall time of one job. The
resource classes extend that per-stage figure with the concurrency each class sustains, which is what lets a scheduler
plan a batch mixing several stages against one host.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

import psutil
from ataraxis_base_utilities import console, resolve_worker_count

from .jobs import MultiRecordingJobNames, SingleRecordingJobNames

if TYPE_CHECKING:
    from collections.abc import Mapping

BINARIZATION_WORKERS: int = 3
"""The number of workers allocated to the binarization stage by default, measured as the processors the stage occupies
while many recordings convert at once. The conversion waits on each batch reaching it rather than on the threads that
consume the batch, so the stage settles below the decode ceiling it is permitted."""

REGISTRATION_WORKERS: int = 12
"""The number of workers allocated to the registration stage by default, measured as the processors the stage occupies
while many planes register at once. The compiled kernels and the linear-algebra routines the phase correlation
dispatches run concurrently rather than taking turns, so the stage occupies more than the kernels alone suggest."""

PROCESSING_WORKERS: int = 10
"""The number of workers allocated to the processing stage by default, measured on a one-plane worker sweep. Detection
held at 114.0, 113.9, and 113.8 seconds for 10, 20, and 30 workers, bound by movie binning IO and a serial detection
loop. Extraction scaled from 61.8 to 44.1 to 38.6 seconds over the same sweep, but running more planes concurrently
outweighs that gain, which places the default at 10."""

DISCOVERY_WORKERS: int = 30
"""The number of workers allocated to the multi-recording discovery stage by default, which is the saturating
allocation the stage is admitted at. The stage registers every recording of one animal against the others, so its cost
grows with the square of the recording count."""

EXTRACTION_WORKERS: int = 16
"""The number of workers allocated to the multi-recording extraction stage by default, measured as the point where the
stage stops shortening. Every frame batch the extraction kernel consumes is read serially before the kernel runs, so
the stage plateaus below the width it is given and further cores are spent waiting on batch reads."""

COMBINATION_WORKERS: int = 1
"""The number of CPU cores one combination job holds. The combination stage merges the per-plane result files with
serial input and output and takes no worker argument, so each of its jobs occupies exactly one core."""

ALL_CORES_REQUEST: int = -1
"""The requested worker count that asks for every available CPU core."""

_RESERVED_CORES: int = 2
"""The number of CPU cores held back for host-system operations when a core budget auto-resolves."""

_MAXIMUM_PARALLEL_IO_JOBS: int = 4
"""The maximum number of concurrent I/O-bound jobs (the binarization and combination resource classes)."""

_PROCESSING_MEMORY_GIGABYTES_PER_JOB: float = 15.0
"""The peak resident memory, in gigabytes, that one processing job holds. Measured on a real nine-plane run where
detection peaked at 10.5 gigabytes on the smallest (512-line) plane, rounded up to 15 to cover the taller planes of the
same recording. The processing resource class bounds its concurrency by this figure so that a batch plans against the
memory its jobs actually demand. Each job runs in its own worker process, so a host that kills one for exhausting
memory ends that job rather than the process every other job shares."""

_BYTES_PER_GIGABYTE: int = 1024**3
"""The number of bytes in one gigabyte, used to convert the host memory counter into a memory budget."""

_STAGE_WORKER_DEFAULTS: dict[SingleRecordingJobNames | MultiRecordingJobNames, int] = {
    SingleRecordingJobNames.BINARIZE: BINARIZATION_WORKERS,
    SingleRecordingJobNames.REGISTER: REGISTRATION_WORKERS,
    SingleRecordingJobNames.PROCESS: PROCESSING_WORKERS,
    SingleRecordingJobNames.COMBINE: COMBINATION_WORKERS,
    MultiRecordingJobNames.DISCOVER: DISCOVERY_WORKERS,
    MultiRecordingJobNames.EXTRACT: EXTRACTION_WORKERS,
}
"""Maps every single and multi-recording pipeline stage to its measured default worker count.

Notes:
    The combination stage carries the single core its serial merge occupies rather than being absent, so a caller
    resolving an allocation never has to special-case one of the six stages.
"""


@dataclass(frozen=True, slots=True)
class ResourceClass:
    """Describes the CPU and memory budget that one class of pipeline jobs holds for its entire duration."""

    name: str
    """The name of the resource class, used as the key of the per-class queues and of the reported allocation."""
    workers_per_job: int
    """The number of CPU cores each job of this class holds, taken from the measured stage defaults, except for the
    combination class, whose single core is defined by COMBINATION_WORKERS."""
    fixed_parallel_jobs: int | None
    """The machine-independent concurrency cap of this class, or None when the cap is derived from the CPU budget and,
    for memory-bound classes, from the available system memory. A per-class cap bounds one class in isolation, so the
    dispatcher additionally holds the sum of the cores committed by every class inside the session CPU budget."""
    memory_gigabytes_per_job: float
    """The peak resident memory one job of this class holds, or 0.0 when the class does not bound its concurrency by
    memory."""


_BINARIZATION_RESOURCES: ResourceClass = ResourceClass(
    name="binarization",
    workers_per_job=BINARIZATION_WORKERS,
    fixed_parallel_jobs=_MAXIMUM_PARALLEL_IO_JOBS,
    memory_gigabytes_per_job=0.0,
)
"""The resource class of the binarization jobs. The allocated cores become the TIFF image decode threads, and the
stage streams frames to disk instead of holding them, so the concurrency cap is the fixed I/O limit."""

_REGISTRATION_RESOURCES: ResourceClass = ResourceClass(
    name="registration",
    workers_per_job=REGISTRATION_WORKERS,
    fixed_parallel_jobs=None,
    memory_gigabytes_per_job=0.0,
)
"""The resource class of the plane-registration jobs. Registration reads the plane binary through a memory map, so its
resident growth is evictable page cache and its concurrency is bounded by the shared CPU budget alone."""

_PROCESSING_RESOURCES: ResourceClass = ResourceClass(
    name="processing",
    workers_per_job=PROCESSING_WORKERS,
    fixed_parallel_jobs=None,
    memory_gigabytes_per_job=_PROCESSING_MEMORY_GIGABYTES_PER_JOB,
)
"""The resource class of the plane-processing jobs. Detection materializes the binned movie in anonymous memory, so
this class bounds its concurrency by both the shared CPU budget and the available system memory."""

_COMBINATION_RESOURCES: ResourceClass = ResourceClass(
    name="combination",
    workers_per_job=COMBINATION_WORKERS,
    fixed_parallel_jobs=_MAXIMUM_PARALLEL_IO_JOBS,
    memory_gigabytes_per_job=0.0,
)
"""The resource class of the combination jobs. Combination merges per-plane result files with serial input and output,
so each job holds one core and the concurrency cap is the fixed I/O limit."""

_DISCOVERY_RESOURCES: ResourceClass = ResourceClass(
    name="discovery",
    workers_per_job=DISCOVERY_WORKERS,
    fixed_parallel_jobs=None,
    memory_gigabytes_per_job=0.0,
)
"""The resource class of the multi-recording discovery jobs. Discovery registers every recording of one animal against
the others, so each job holds the stage's saturating allocation and its concurrency is bounded by the shared CPU budget
alone."""

_EXTRACTION_RESOURCES: ResourceClass = ResourceClass(
    name="extraction",
    workers_per_job=EXTRACTION_WORKERS,
    fixed_parallel_jobs=None,
    memory_gigabytes_per_job=0.0,
)
"""The resource class of the multi-recording extraction jobs. Extraction reads each frame batch serially before the
kernel consumes it, so the stage plateaus at its measured worker count and the remaining budget is better spent on
running more recordings concurrently."""

RESOURCE_CLASS_BY_JOB_NAME: dict[str, ResourceClass] = {
    SingleRecordingJobNames.BINARIZE: _BINARIZATION_RESOURCES,
    SingleRecordingJobNames.REGISTER: _REGISTRATION_RESOURCES,
    SingleRecordingJobNames.PROCESS: _PROCESSING_RESOURCES,
    SingleRecordingJobNames.COMBINE: _COMBINATION_RESOURCES,
    MultiRecordingJobNames.DISCOVER: _DISCOVERY_RESOURCES,
    MultiRecordingJobNames.EXTRACT: _EXTRACTION_RESOURCES,
}
"""Maps every pipeline job name to the resource class that governs its worker count and its concurrency cap."""


def resolve_stage_workers(
    job_name: SingleRecordingJobNames | MultiRecordingJobNames,
    requested_workers: int | None = None,
) -> int:
    """Resolves the number of workers to allocate to the target pipeline stage.

    Notes:
        A requested count of None resolves to the measured default for the stage, which is the knee of that stage's
        scaling curve. A requested count of -1 resolves to every available CPU core, minus the cores the ataraxis
        worker resolver holds back for system use. A positive requested count is honored exactly. A requested count of
        zero, or any negative count other than -1, is rejected.

        Every pipeline stage resolves through this function. The combination stage takes no worker argument of its
        own, so its default is the single core its serial merge occupies.

    Args:
        job_name: The single or multi-recording pipeline stage to allocate workers for.
        requested_workers: The number of workers the caller asks for. Use None to accept the measured default for the
            stage and -1 to request every available core.

    Returns:
        The number of workers to allocate to the stage, always at least 1.

    Raises:
        ValueError: If job_name does not name a pipeline stage, or if requested_workers is zero or is a negative value
            other than -1.
    """
    default_workers: int | None = _STAGE_WORKER_DEFAULTS.get(job_name)
    if default_workers is None:
        message = (
            f"Unable to resolve the worker count for the '{job_name}' processing stage. The input job name does not "
            f"name a pipeline stage. Use one of the valid stage names: "
            f"{[stage.value for stage in _STAGE_WORKER_DEFAULTS]}."
        )
        console.error(message=message, error=ValueError)

    if requested_workers is None:
        return default_workers

    if requested_workers == ALL_CORES_REQUEST:
        return resolve_worker_count(requested_workers=ALL_CORES_REQUEST)

    if requested_workers <= 0:
        message = (
            f"Unable to resolve the worker count for the '{job_name}' processing stage. The requested worker count "
            f"must be a positive integer, -1 to request every available core, or None to accept the measured stage "
            f"default, but encountered {requested_workers}."
        )
        console.error(message=message, error=ValueError)

    return requested_workers


def resolve_core_budget() -> int:
    """Resolves the cores one execution session may commit across all of its concurrently running jobs.

    Returns:
        The cores the session may commit, which is every available core minus the cores held back for the host.
    """
    return resolve_worker_count(requested_workers=ALL_CORES_REQUEST, reserved_cores=_RESERVED_CORES)


def resolve_class_allocation(
    resource_class: ResourceClass,
    *,
    budget: int,
    available_memory: float,
    job_count: int,
    workers_per_job: int | None,
    max_parallel_jobs: int | None,
) -> tuple[int, int]:
    """Resolves the per-job worker count and the concurrency cap of one resource class.

    Notes:
        A class with a fixed concurrency cap describes I/O-bound work whose throughput does not follow the core count,
        so it keeps its measured allocation and ignores both overrides. Every other class takes its measured worker
        count, bounds its concurrency by the CPU budget, and bounds it further by the available system memory when the
        class declares a per-job memory footprint.

        Every cap resolved here bounds one class in isolation, because a class cannot know which other classes will be
        dispatching alongside it. The dispatcher therefore holds the sum of the cores committed by every class inside
        the same CPU budget at run time.

    Args:
        resource_class: The resource class to resolve the allocation for.
        budget: The number of CPU cores available to the session after reserving system cores.
        available_memory: The available system memory in gigabytes.
        job_count: The number of jobs of this class in the session, which caps the useful concurrency.
        workers_per_job: The requested CPU cores per job, -1 to request every available core, or None to accept
            the class default.
        max_parallel_jobs: The requested concurrency cap, -1 to lift the cap, or None to accept the derived cap.

    Returns:
        A (workers_per_job, max_parallel_jobs) tuple for this resource class.
    """
    if resource_class.fixed_parallel_jobs is not None:
        workers = resource_class.workers_per_job
        return workers, min(resource_class.fixed_parallel_jobs, max(1, job_count))

    if workers_per_job is None:
        workers = resource_class.workers_per_job
    elif workers_per_job == ALL_CORES_REQUEST:
        workers = budget
    else:
        workers = workers_per_job

    # An all-cores concurrency request lifts the derived cap, leaving the job count as the only bound.
    if max_parallel_jobs == ALL_CORES_REQUEST:
        return workers, max(1, job_count)

    if max_parallel_jobs is not None:
        return workers, max_parallel_jobs

    capacity = max(1, budget // workers)
    if resource_class.memory_gigabytes_per_job > 0:
        capacity = min(capacity, max(1, int(available_memory // resource_class.memory_gigabytes_per_job)))

    return workers, min(capacity, max(1, job_count))


def resolve_available_memory_gigabytes() -> float:
    """Resolves the amount of system memory that new allocations can claim, in gigabytes.

    Notes:
        The counter discounts the reclaimable page cache, which matters because registration fills that cache with the
        plane binaries it memory-maps. A counter reporting free memory alone would collapse behind that cache and
        throttle the memory-bound classes to a near-serial concurrency on a host that is not actually short of memory.

        The value is sampled once, when the execution session starts. It already discounts the page cache that the
        session itself will fill, so the sample stays representative for the lifetime of the session.

    Returns:
        The available system memory in gigabytes.
    """
    return float(psutil.virtual_memory().available) / _BYTES_PER_GIGABYTE


def summarize_class_allocation(
    class_workers: Mapping[str, int], class_capacities: Mapping[str, int], class_job_counts: Mapping[str, int]
) -> dict[str, dict[str, int]]:
    """Assembles the per-class allocation report that an execution session publishes to its caller.

    Args:
        class_workers: The resolved per-job worker count of each resource class, keyed by class name.
        class_capacities: The resolved concurrency cap of each resource class, keyed by class name.
        class_job_counts: The number of jobs of each resource class in the session, keyed by class name.

    Returns:
        The worker count, the concurrency cap, and the job count of every resource class, keyed by class name.
    """
    return {
        class_name: {
            "workers_per_job": class_workers[class_name],
            "max_parallel_jobs": class_capacities[class_name],
            "job_count": class_job_counts[class_name],
        }
        for class_name in class_job_counts
    }
