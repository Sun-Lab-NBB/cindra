"""Provides the measured CPU worker defaults of the single and multi-recording pipeline stages, the resource-class model
that sizes a batch of jobs, and the resolvers that turn a caller's request into a concrete allocation. The defaults
encode the knee of each stage's measured scaling curve, so a caller expressing no preference receives the allocation
that maximizes batch throughput rather than the one that minimizes the wall time of a single job.
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

REGISTRATION_WORKERS: int = 4
"""The number of workers allocated to the registration stage by default, which is the knee of its measured scaling
curve. The stage is largely serial, so wall time flattens close to this allocation."""

PROCESSING_WORKERS: int = 10
"""The number of workers allocated to the processing stage by default, measured on a one-plane worker sweep. Detection
held at 114.0, 113.9, and 113.8 seconds for 10, 20, and 30 workers, bound by movie binning IO and a serial detection
loop. Extraction scaled from 61.8 to 44.1 to 38.6 seconds over the same sweep, but running more planes concurrently
outweighs that gain, which places the default at 10."""

DISCOVERY_WORKERS: int = 2
"""The number of workers allocated to the multi-recording discovery stage by default. Sixteen workers shorten a
ten-recording dataset from 28.9 to 19.1 seconds while spending twelve times the processor time, and the wall time
turns back upward past that width. Two is the smallest allocation that engages the deformation pool, which the stage
builds only while the budget exceeds one."""

EXTRACTION_WORKERS: int = 16
"""The number of workers allocated to the multi-recording extraction stage by default. The stage keeps shortening well
past this width, so the figure follows the concurrency a host sustains rather than a plateau. Sixteen leaves room for
the six to eight datasets a compute node extracts at once while still reaching a sevenfold single-job speedup."""

COMBINATION_WORKERS: int = 1
"""The number of CPU cores one combination job holds. The combination stage merges the per-plane result files with
serial input and output and takes no worker argument, so each of its jobs occupies exactly one core."""

REGISTRATION_MAXIMUM_WORKERS: int = 32
"""The widest allocation one registration job converts into wall clock. A 6000-frame recording of 794 by 376 pixel
frames, registered with the quality metrics enabled, took 61.0, 42.2, 33.5, and 30.5 seconds at 4, 8, 16, and 32
workers, and held at 31.2 and 31.0 seconds at 64 and 126. Only 43 cores are ever busy at the widest allocation, so the
stage cannot saturate a wider budget."""

PROCESSING_MAXIMUM_WORKERS: int = 10
"""The widest allocation one processing job converts into wall clock, which is the measured stage default. A
6000-frame recording of 794 by 376 pixel frames holding 2190 regions took 31.9, 30.9, 29.2, 29.0, and 28.4 seconds at
10, 16, 32, 64, and 126 workers. Detection holds at 22 seconds across that whole sweep, because it waits on the
movie binning it reads and on a serial detection loop, and it spends three quarters of the job."""

DISCOVERY_MAXIMUM_WORKERS: int = 8
"""The widest allocation one multi-recording discovery job converts into wall clock. A ten-recording dataset took 28.9,
26.4, 22.9, and 20.4 seconds at 1, 2, 4, and 8 workers, while 16 workers bought 6.4 percent over 8 and 32 workers turned
the wall time back upward to 19.7 seconds."""

EXTRACTION_MAXIMUM_WORKERS: int = 32
"""The widest allocation one multi-recording extraction job converts into wall clock. The extraction the
multi-recording stage runs is the extraction a processing job runs, so the figure follows that sub-stage, which took
6, 4, 3, 3, and 3 seconds at 10, 16, 32, 64, and 126 workers over 2190 regions and 6000 frames. The plateau sits at
32, and a processing job stops short of it because detection spends three quarters of that job at a width no wider
allocation shortens.

Notes:
    The sweep behind this figure measured the sub-stage inside a processing job rather than a multi-recording
    extraction job over a whole dataset, so the plateau is established on the kernels the two stages share rather
    than on the dataset scale the multi-recording stage reaches.
"""

ALL_CORES_REQUEST: int = -1
"""The requested worker count that asks for every available CPU core."""

_RESERVED_CORES: int = 2
"""The number of CPU cores held back for host-system operations when a core budget auto-resolves."""

_BINARIZATION_CONCURRENCY_LIMIT: int = 4
"""The binarization jobs that may run at once regardless of the cores the budget could still supply.

Notes:
    The stage sits at the root of the single-recording chain, so each job that finishes releases that recording's
    plane jobs. Four run at their full decode rate, which is what returns those plane jobs to the queue soonest,
    while spreading the same capacity over more recordings would delay all of them equally. Spare cores never lift
    this ceiling, because a job held by it waits on something spare cores do not supply.
"""

_REGISTRATION_CONCURRENCY_RESERVATION: int = 4
"""The registration jobs that run at once while other work can still use the capacity the stage gives up.

Notes:
    Registration gates the processing job that waits on it, so holding a share back keeps the stages that wait on no
    other job running while a recording's planes are still being registered.
"""

_PROCESSING_CONCURRENCY_RESERVATION: int = 5
"""The processing jobs that run at once while other work can still use the capacity the stage gives up.

Notes:
    Processing cores are the batch's scarcest resource once the single-recording chain opens, so holding a share back
    keeps the conversion jobs at the root of that chain running rather than starving behind a full budget.
"""

_BYTES_PER_MEGABYTE: int = 1024**2
"""The number of bytes in one megabyte, used to convert the host memory counter into a memory budget."""

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
    """Describes the cores one class of pipeline jobs holds and the concurrency terms that bound the class."""

    name: str
    """The name of the resource class, used as the key of the per-class queues and of the reported allocation."""
    workers_per_job: int
    """The number of CPU cores each job of this class holds, taken from the measured stage defaults."""
    maximum_workers_per_job: int | None
    """The most CPU cores one job of this class holds when the host has capacity to spare, or None when the class is
    not elastic and every one of its jobs runs at workers_per_job.

    Notes:
        The ceiling is the width at which the stage stops converting cores into wall clock, so an allocation past it
        holds capacity another job would turn into throughput. A class carries None when its work waits on something
        the host cores do not supply, which covers the storage the conversion decodes from and the serial merge the
        combination stage performs.
    """
    concurrency_limit: int | None
    """The jobs of this class that may run at once regardless of the capacity the budgets could still supply, or None
    when the class is bounded by the budgets alone.

    Notes:
        This is a hard ceiling. It exists for a class whose own throughput stops climbing before its cores run out, so
        spare capacity never lifts it.
    """
    concurrency_reservation: int | None
    """The jobs of this class that run at once while other work can still use the capacity the class gives up, or None
    when the class competes at its full derived width.

    Notes:
        This is a soft counterpart to the ceiling. It exists to leave room for other jobs rather than because the
        class stops gaining from concurrency, so the dispatcher releases it over whatever capacity remains once every
        other runnable job has been offered that room.
    """


_BINARIZATION_RESOURCES: ResourceClass = ResourceClass(
    name="binarization",
    workers_per_job=BINARIZATION_WORKERS,
    maximum_workers_per_job=None,
    concurrency_limit=_BINARIZATION_CONCURRENCY_LIMIT,
    concurrency_reservation=None,
)
"""The resource class of the binarization jobs. The allocated cores become the TIFF image decode threads, and the
stage streams frames to disk instead of holding them, so its concurrency is the hard ceiling this class alone
carries."""

_REGISTRATION_RESOURCES: ResourceClass = ResourceClass(
    name="registration",
    workers_per_job=REGISTRATION_WORKERS,
    maximum_workers_per_job=REGISTRATION_MAXIMUM_WORKERS,
    concurrency_limit=None,
    concurrency_reservation=_REGISTRATION_CONCURRENCY_RESERVATION,
)
"""The resource class of the plane-registration jobs. Its concurrency derives from the shared CPU budget, and it holds
a reservation so the stages waiting on no other job keep a share of the host while a recording's planes register."""

_PROCESSING_RESOURCES: ResourceClass = ResourceClass(
    name="processing",
    workers_per_job=PROCESSING_WORKERS,
    maximum_workers_per_job=PROCESSING_MAXIMUM_WORKERS,
    concurrency_limit=None,
    concurrency_reservation=_PROCESSING_CONCURRENCY_RESERVATION,
)
"""The resource class of the plane-processing jobs. Detection materializes the binned movie in anonymous memory, so its
jobs carry the largest per-job memory estimates the dispatcher admits against, and it holds a reservation so the stages
waiting on no other job keep a share of the host."""

_COMBINATION_RESOURCES: ResourceClass = ResourceClass(
    name="combination",
    workers_per_job=COMBINATION_WORKERS,
    maximum_workers_per_job=None,
    concurrency_limit=None,
    concurrency_reservation=None,
)
"""The resource class of the combination jobs. Combination merges per-plane result files with serial input and output,
so each job holds one core and its concurrency is bounded by the shared CPU budget alone."""

_DISCOVERY_RESOURCES: ResourceClass = ResourceClass(
    name="discovery",
    workers_per_job=DISCOVERY_WORKERS,
    maximum_workers_per_job=DISCOVERY_MAXIMUM_WORKERS,
    concurrency_limit=None,
    concurrency_reservation=None,
)
"""The resource class of the multi-recording discovery jobs. Discovery registers every recording of one animal against
the others, so each job holds the stage's saturating allocation and its concurrency is bounded by the shared CPU budget
alone."""

_EXTRACTION_RESOURCES: ResourceClass = ResourceClass(
    name="extraction",
    workers_per_job=EXTRACTION_WORKERS,
    maximum_workers_per_job=EXTRACTION_MAXIMUM_WORKERS,
    concurrency_limit=None,
    concurrency_reservation=None,
)
"""The resource class of the multi-recording extraction jobs. The stage keeps shortening well past its measured worker
count, so that count follows the concurrency a compute node sustains and the remaining budget is better spent on
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
    job_count: int,
    workers_per_job: int | None,
    max_parallel_jobs: int | None,
) -> tuple[int, int]:
    """Resolves the per-job worker count and the concurrency cap of one resource class.

    Notes:
        A class carrying a hard concurrency ceiling describes work whose throughput stops climbing before its cores
        run out, so it keeps its measured allocation and ignores both overrides. Every other class takes its measured
        worker count and bounds its concurrency by the CPU budget. Memory bounds admission rather than concurrency,
        because the memory one job holds follows the recording it processes rather than the class it belongs to.

        Every cap resolved here bounds one class in isolation, because a class cannot know which other classes will be
        dispatching alongside it. The dispatcher therefore holds the sum of the cores committed by every class inside
        the same CPU budget at run time.

    Args:
        resource_class: The resource class to resolve the allocation for.
        budget: The number of CPU cores available to the session after reserving system cores.
        job_count: The number of jobs of this class in the session, which caps the useful concurrency.
        workers_per_job: The requested CPU cores per job, -1 to request every available core, or None to accept
            the class default.
        max_parallel_jobs: The requested concurrency cap, -1 to lift the cap, or None to accept the derived cap.

    Returns:
        A (workers_per_job, max_parallel_jobs) tuple for this resource class.
    """
    if resource_class.concurrency_limit is not None:
        return resource_class.workers_per_job, min(resource_class.concurrency_limit, max(1, job_count))

    if workers_per_job is None:
        workers = resource_class.workers_per_job
    elif workers_per_job == ALL_CORES_REQUEST:
        workers = budget
    else:
        workers = workers_per_job

    if max_parallel_jobs == ALL_CORES_REQUEST:
        return workers, max(1, job_count)

    if max_parallel_jobs is not None:
        return workers, max_parallel_jobs

    return workers, min(max(1, budget // workers), max(1, job_count))


def resolve_dispatch_workers(
    resource_class: ResourceClass,
    *,
    free_cores: int,
    pending_jobs: int,
    running_jobs: int,
    concurrency_cap: int,
) -> int:
    """Resolves the number of workers one job of the target resource class takes when it is dispatched.

    Notes:
        The share divides the cores no running job holds across the jobs that can still start alongside the dispatched
        one, and the class default and the class ceiling bound the result. A full queue therefore resolves to the class
        default, which is the allocation a session running at its full concurrency gives every job. A queue holding one
        job resolves toward the ceiling, because that job is the only claim on the free capacity. A draining queue sits
        between the two, so the jobs a batch has left to run widen as their peers finish.

        A class carrying no ceiling is not elastic and takes its measured allocation whatever the host holds free.

    Args:
        resource_class: The resource class of the job being dispatched.
        free_cores: The cores of the session budget that no running job holds.
        pending_jobs: The jobs of this class awaiting dispatch, counting the one being dispatched.
        running_jobs: The jobs of this class already running.
        concurrency_cap: The jobs of this class that may run at once.

    Returns:
        The number of workers to allocate to the dispatched job.
    """
    if resource_class.maximum_workers_per_job is None:
        return resource_class.workers_per_job

    # Holds the divisor at one, so a class dispatching the last job of a drained queue gives that job the whole share.
    competitors = max(1, min(pending_jobs, max(1, concurrency_cap - running_jobs)))
    share = free_cores // competitors

    return min(max(share, resource_class.workers_per_job), resource_class.maximum_workers_per_job)


def resolve_memory_budget_mb() -> int:
    """Resolves the amount of system memory that new allocations can claim, in megabytes.

    Notes:
        The counter discounts the reclaimable page cache, which matters because registration fills that cache with the
        plane binaries it memory-maps. A counter reporting free memory alone would collapse behind that cache and
        report a host that is not actually short of memory as full.

        The value is sampled once, when the execution session starts. It already discounts the page cache that the
        session itself will fill, so the sample stays representative for the lifetime of the session.

    Returns:
        The available system memory in megabytes, which is the scale the per-job estimates report.
    """
    return int(psutil.virtual_memory().available / _BYTES_PER_MEGABYTE)


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
