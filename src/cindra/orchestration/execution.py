"""Provides the local batch execution engine that admits queued pipeline jobs as their own prerequisites succeed and
dispatches them under the per-class concurrency terms and the session core and memory budgets.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING
from threading import Lock, Thread
from dataclasses import field, dataclass
from multiprocessing import get_context
from concurrent.futures import Executor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import console
from ataraxis_data_structures import (
    JobState,
    ProcessingStatus,
    ProcessingTracker,
    limit_worker_threads,
    initialize_worker_threads,
)

from .jobs import (
    PREREQUISITE_FAILURE_MESSAGE,
    UNREACHABLE_PREREQUISITE_MESSAGE,
    resolve_prerequisite_job_ids,
)
from .pipeline import run_multi_recording_pipeline, run_single_recording_pipeline
from .allocation import (
    ALL_CORES_REQUEST,
    ResourceClass,
    resolve_core_budget,
    resolve_class_allocation,
    resolve_memory_budget_mb,
    summarize_class_allocation,
)

if TYPE_CHECKING:
    from pathlib import Path
    from concurrent.futures import Future

_DISPATCH_POLL_MILLISECONDS: int = 1000
"""The interval at which the manager re-examines the queues for freed capacity."""

_WORKER_THREAD_CEILING: int = 1
"""The number of threads every pool worker pins its numeric backends to while it starts.

Notes:
    A spawned worker re-imports the numeric backends rather than inheriting the parent's, and each of them sizes its
    thread pool to the whole host at import unless the environment says otherwise. A pool running one worker per core
    would therefore hold the square of the core count in threads while using one of them. The ceiling of one is the
    floor each job raises from, not the width it runs at, because every stage sets its own budget once it starts.
"""

_POOL_START_METHOD: str = "spawn"
"""The multiprocessing start method every worker process of an execution session is created with.

Notes:
    Spawn is the only method available on every platform the library supports. Requesting it explicitly therefore
    gives a Linux session the process semantics a macOS or Windows session gets by default, rather than the platform
    default of the host it happens to run on. The alternatives are also unsound for this pipeline. A forked worker
    inherits the parent's already-sized numeric backends, which leaves the thread pin inert and hands every concurrent
    job a host-wide backend pool, and it inherits the parent's Numba and threading state at whatever point the fork
    interrupted it.
"""

_BROKEN_POOL_MESSAGE: str = (
    "Unable to execute job. The worker process pool was terminated, which happens when a job's process is killed by "
    "the host, most often for exhausting memory."
)
"""The tracker error message recorded for a job the session can no longer dispatch or complete."""

_CANCELED_JOB_MESSAGE: str = (
    "Unable to execute job. The worker process pool canceled the job before any worker started it, which happens "
    "when the pool shuts down while the job is still waiting inside it."
)
"""The tracker error message recorded for a job whose worker pool future was canceled before it ran."""


class _AdmissionDecisions(StrEnum):
    """Defines the outcomes of evaluating one queued job's prerequisites against its own tracker."""

    ADMIT = "admit"
    """Every prerequisite job succeeded, so the job moves into its resource class queue."""
    WAIT = "wait"
    """At least one prerequisite job has not finished, so the job stays in the admission pool."""
    ABORT = "abort"
    """At least one prerequisite job failed or is absent from the tracker, so the job can never run."""


class _JobOutcomes(StrEnum):
    """Defines the outcomes of examining the worker pool future the session holds for one dispatched job."""

    RUNNING = "running"
    """The worker has not finished the job, so the job keeps its place in its resource class running set."""
    COMPLETED = "completed"
    """The worker returned, so the job already recorded its own terminal state on its tracker."""
    ABANDONED = "abandoned"
    """The future carries an exception or a cancellation rather than a result. That covers a worker the host killed, a
    worker that raised on its way out of the job, and a job the pool canceled before any worker started it."""


@dataclass(slots=True)
class PendingJob:
    """Describes a single pipeline job queued for batch execution."""

    configuration_path: Path
    """The path to the pipeline configuration file for this job."""
    tracker_path: Path
    """The path to the ProcessingTracker file that tracks this job."""
    job_id: str
    """The unique hexadecimal identifier for this job in the tracker."""
    single_recording: bool
    """Determines whether this job belongs to a single-recording or multi-recording pipeline."""
    resource_class: ResourceClass
    """The resource class that governs this job's worker count and the concurrency of its queue."""
    resolved_workers: int | None = None
    """The number of parallel workers to allocate to this job, assigned at dispatch time. A value of None makes the
    pipeline fall back to the measured default for the job's stage."""
    memory_megabytes: int = 0
    """The memory this job holds while it runs, as the caller's sizing pass estimated it.

    Notes:
        A value of zero states that the caller supplied no estimate, which leaves the job admitted on the core budget
        alone. Memory is carried per job rather than per resource class, because the memory one job holds follows the
        recording it processes rather than the stage it runs.
    """

    @property
    def dispatch_key(self) -> tuple[str, str]:
        """Returns the composite tracker path and job identifier pair that identifies this job across the batch."""
        return str(self.tracker_path), self.job_id


@dataclass(slots=True)
class JobExecutionState:
    """Tracks the runtime state for one batch execution session across both pipeline types.

    Notes:
        Each admitted job runs in its own worker process at the width its resource class was allocated, so a
        registration job and a processing job run side by side, each holding its own numeric-backend budget. Running
        them as threads of one process instead would let the later of the two overwrite the BLAS width the earlier one
        set. That width is a property of the process rather than of the thread that asked for it.
    """

    all_jobs: dict[tuple[str, str], PendingJob] = field(default_factory=dict)
    """All submitted jobs keyed by their tracker path and job identifier pair, used for status reporting."""
    admission_pool: list[PendingJob] = field(default_factory=list)
    """Jobs awaiting prerequisite satisfaction, scanned by the manager on every polling cycle."""
    pending_queues: dict[str, list[PendingJob]] = field(default_factory=dict)
    """Admitted jobs awaiting dispatch, keyed by resource class name."""
    active_futures: dict[str, dict[tuple[str, str], Future[None]]] = field(default_factory=dict)
    """Currently running dispatch key to Future mappings, keyed by resource class name."""
    class_capacities: dict[str, int] = field(default_factory=dict)
    """The resolved maximum number of concurrent jobs for each resource class name."""
    class_workers: dict[str, int] = field(default_factory=dict)
    """The resolved number of CPU cores allocated to each job of each resource class name."""
    class_reservations: dict[str, int] = field(default_factory=dict)
    """The jobs of each resource class that run before the dispatcher releases that class's reservation, keyed by
    class name. A class absent from this map competes at its full derived width in both passes."""
    cpu_budget: int = 1
    """The total number of CPU cores this session may commit across every resource class at once."""
    memory_budget_mb: int = 0
    """The total memory this session may commit across every running job at once, in megabytes."""
    lock: Lock = field(default_factory=Lock)
    """The lock guarding every mutation of the job queues."""
    manager_thread: Thread | None = None
    """The background thread running the execution manager, or None before the session starts it."""


_execution_state: JobExecutionState | None = None
"""Stores the active execution state for batch processing jobs, or None when no session exists."""


def get_execution_state() -> JobExecutionState | None:
    """Returns the active batch processing execution state, or None when no session exists."""
    return _execution_state


def set_execution_state(state: JobExecutionState | None) -> None:
    """Stores the active batch processing execution state, replacing any existing session reference.

    Args:
        state: The execution state to store, or None to clear the active session.
    """
    global _execution_state
    _execution_state = state


def resolve_session_load() -> tuple[int, int]:
    """Counts the jobs the active execution session still holds.

    Returns:
        The number of jobs awaiting dispatch and the number of jobs currently running, in that order. Both counts are
        zero when no session exists or the session has drained.
    """
    # Binds the session to a local name, because the execution manager can clear the module-level reference between
    # the emptiness check and the lock acquisition.
    state = _execution_state
    if state is None:
        return 0, 0

    with state.lock:
        pending_count = len(state.admission_pool) + sum(len(queue) for queue in state.pending_queues.values())
        active_count = sum(len(futures) for futures in state.active_futures.values())
        return pending_count, active_count


def start_execution_session(
    all_jobs: dict[tuple[str, str], PendingJob],
    workers_per_job: int | None,
    max_parallel_jobs: int | None,
) -> dict[str, object]:
    """Resolves per-class resource allocation, stamps it onto the queued jobs, and starts the execution manager.

    Notes:
        Every job enters the admission pool, and the manager decides admission from the tracked prerequisites.

        The resolved allocation is stamped onto each job and travels to the pipeline as a dispatch argument, so one
        configuration file serves every job dispatched concurrently against it.

        Each class resolves its own concurrency cap, and the session CPU budget is recorded alongside those caps
        because every class dispatches during the same cycle. The dispatcher holds the sum of the cores committed by
        the running jobs of every class inside that budget, so the per-class caps cannot oversubscribe the machine
        between them.

    Args:
        all_jobs: All submitted jobs keyed by dispatch key, in the order the manager should consider them.
        workers_per_job: Requested CPU cores per job, -1 for every available core, or None to accept each resource
            class default.
        max_parallel_jobs: Requested maximum concurrent jobs per resource class, -1 to lift the caps, or None to
            accept the derived caps.

    Returns:
        A dictionary carrying the submitted job total under 'total_jobs', the session core budget under 'cpu_budget',
        the session memory budget under 'memory_budget_mb', and the per-class worker count, concurrency cap, and job
        count under 'resource_classes'.

    Raises:
        ValueError: If either override is zero or is a negative value other than -1.
    """
    # Rejects a non-positive override rather than letting it fall through as a negative core count. None is the only
    # way to ask for a default, so a caller passing 0 or a negative value has confused the two.
    for override_name, override_value in (
        ("workers_per_job", workers_per_job),
        ("max_parallel_jobs", max_parallel_jobs),
    ):
        if override_value is not None and override_value <= 0 and override_value != ALL_CORES_REQUEST:
            message = (
                f"Unable to start the execution session. The '{override_name}' override must be a positive integer, "
                f"-1 to request every available core, or None to accept the measured default, but encountered "
                f"{override_value}."
            )
            console.error(message=message, error=ValueError)

    budget = resolve_core_budget()
    memory_budget = resolve_memory_budget_mb()

    # Counts the jobs of every resource class present in this session, which bounds each class capacity.
    class_job_counts: dict[str, int] = {}
    classes_by_name: dict[str, ResourceClass] = {}
    for pending_job in all_jobs.values():
        class_name = pending_job.resource_class.name
        classes_by_name[class_name] = pending_job.resource_class
        class_job_counts[class_name] = class_job_counts.get(class_name, 0) + 1

    class_workers: dict[str, int] = {}
    class_capacities: dict[str, int] = {}
    for class_name, resource_class in classes_by_name.items():
        workers, capacity = resolve_class_allocation(
            resource_class=resource_class,
            budget=budget,
            job_count=class_job_counts[class_name],
            workers_per_job=workers_per_job,
            max_parallel_jobs=max_parallel_jobs,
        )
        class_workers[class_name] = workers
        class_capacities[class_name] = capacity

    for pending_job in all_jobs.values():
        pending_job.resolved_workers = class_workers[pending_job.resource_class.name]

    execution_state = JobExecutionState(
        all_jobs=all_jobs,
        admission_pool=list(all_jobs.values()),
        pending_queues={class_name: [] for class_name in classes_by_name},
        active_futures={class_name: {} for class_name in classes_by_name},
        class_capacities=class_capacities,
        class_workers=class_workers,
        class_reservations={
            class_name: resource_class.concurrency_reservation
            for class_name, resource_class in classes_by_name.items()
            if resource_class.concurrency_reservation is not None
        },
        cpu_budget=budget,
        memory_budget_mb=memory_budget,
        lock=Lock(),
    )

    # Assigns the session state before starting the manager thread to prevent a race condition where the manager
    # reads the state as None and exits immediately.
    set_execution_state(state=execution_state)
    manager = Thread(target=_job_execution_manager, kwargs={"state": execution_state}, daemon=True)
    manager.start()
    execution_state.manager_thread = manager

    return {
        "total_jobs": len(all_jobs),
        "cpu_budget": budget,
        "memory_budget_mb": memory_budget,
        "resource_classes": summarize_class_allocation(
            class_workers=class_workers,
            class_capacities=class_capacities,
            class_job_counts=class_job_counts,
        ),
    }


def cancel_execution_session() -> tuple[int, int]:
    """Clears every queued job of the active session, leaving the running jobs to finish.

    Notes:
        Cancellation empties the admission pool and every resource class queue, so the manager terminates once the
        running set drains. A job already dispatched keeps its worker process, since interrupting it partway would
        leave its output directory holding a partial result the tracker reports as running.

    Returns:
        The number of jobs cleared from the queues and the number of jobs left running, in that order.
    """
    state = get_execution_state()
    if state is None:
        return 0, 0

    with state.lock:
        canceled_count = len(state.admission_pool) + sum(len(queue) for queue in state.pending_queues.values())
        state.admission_pool.clear()
        for pending_queue in state.pending_queues.values():
            pending_queue.clear()
        active_count = sum(len(futures) for futures in state.active_futures.values())

    return canceled_count, active_count


def _job_execution_manager(state: JobExecutionState) -> None:
    """Admits jobs whose prerequisites succeeded and dispatches them under their resource class concurrency caps.

    Runs as a daemon thread, polling at 1-second intervals.

    Notes:
        Every polling cycle reaps finished jobs, scans the admission pool against a fresh snapshot of each tracker, and
        then dispatches from every resource class queue up to that class's cap. A job is admitted the moment its own
        prerequisites succeed on its own tracker, so each job follows the progress of its own recording.

        A job whose prerequisite failed is marked FAILED on the cycle that observes the failure, and a session that can
        make no further progress fails everything it still holds. Both outcomes clear the session state, so the manager
        always terminates.

        The session is taken as an argument rather than read from the module global on every cycle, so a manager whose
        session was canceled cannot adopt the session that replaced it and dispatch its queues alongside that session's
        own manager. The global is cleared only when it still names the session this manager owns.

        The worker pool lives for the whole session, and the pin on the numeric backends encloses its whole lifetime
        rather than its construction alone. A pool spawns each worker when work first reaches it rather than when the
        pool is created.

    Args:
        state: The execution state this manager owns. Mutated under its own lock as jobs move between the queues.
    """
    timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

    with (
        limit_worker_threads(thread_count=_WORKER_THREAD_CEILING),
        _create_job_pool(max_workers=_resolve_pool_size(state=state)) as pool,
    ):
        while True:
            if get_execution_state() is not state:
                return

            with state.lock:
                _reap_completed_jobs(state=state)
                admitted = _admit_ready_jobs(state=state)
                try:
                    dispatched = _dispatch_admitted_jobs(state=state, pool=pool)
                except BrokenProcessPool:
                    # A worker process died outside the job's own control, most often an out-of-memory kill, which
                    # leaves the pool unable to accept further work. Everything this session still holds is recorded
                    # as failed here, since no later cycle can dispatch or complete it.
                    _fail_broken_session(state=state)
                    _clear_owned_session(state=state)
                    return

                queued = any(pending_queue for pending_queue in state.pending_queues.values())
                active = any(futures for futures in state.active_futures.values())

                if not state.admission_pool and not queued and not active:
                    _clear_owned_session(state=state)
                    return

                # Nothing is running, nothing is queued, and this cycle changed nothing, so the jobs still held in the
                # admission pool depend on work that this session cannot perform.
                if not queued and not active and not admitted and not dispatched:
                    _fail_pending_jobs(jobs=state.admission_pool, message=UNREACHABLE_PREREQUISITE_MESSAGE)
                    state.admission_pool.clear()
                    _clear_owned_session(state=state)
                    return

            timer.delay(delay=_DISPATCH_POLL_MILLISECONDS, allow_sleep=True)


def _clear_owned_session(state: JobExecutionState) -> None:
    """Clears the module-level session reference while it still names the target state.

    Args:
        state: The execution state whose manager is terminating.
    """
    if get_execution_state() is state:
        set_execution_state(state=None)


def _reap_completed_jobs(state: JobExecutionState) -> None:
    """Removes the finished jobs from every resource class, freeing that class's concurrency.

    Notes:
        A job whose worker returned recorded its own terminal state before its future resolved, so the reaper only
        frees the capacity that job held. A future carrying an exception or a cancellation instead covers a worker the
        host killed and a job the pool never started. The reaper records a failure for such a job only when its tracker
        holds no terminal state, so a worker that raised after succeeding keeps its success.

    Args:
        state: The current job execution state, accessed under its lock.
    """
    for active_futures in state.active_futures.values():
        outcomes = [(key, _resolve_job_outcome(future=future)) for key, future in active_futures.items()]
        for key, (outcome, outcome_message) in outcomes:
            if outcome == _JobOutcomes.RUNNING:
                continue

            reaped_job = state.all_jobs.get(key)
            if outcome == _JobOutcomes.ABANDONED and reaped_job is not None:
                _fail_dispatched_job(job=reaped_job, message=outcome_message)
            active_futures.pop(key, None)


def _resolve_job_outcome(future: Future[None]) -> tuple[_JobOutcomes, str]:
    """Classifies one dispatched job by the state of the worker pool future that carries it.

    Notes:
        A finished future alone does not state that its job ran to completion. A worker the host killed, most often
        for exhausting memory, leaves its future finished carrying a BrokenProcessPool, and a pool shutting down
        leaves the futures it never started canceled. Both leave the job in the same place as a worker whose own
        terminal-state guard raised, which is a job holding no outcome of its own.

    Args:
        future: The worker pool future the session holds for the dispatched job.

    Returns:
        A tuple of the job's outcome and the error message to record when that outcome is ABANDONED. The message is
        an empty string for every other outcome.
    """
    if not future.done():
        return _JobOutcomes.RUNNING, ""

    if future.cancelled():
        return _JobOutcomes.ABANDONED, _CANCELED_JOB_MESSAGE

    error = future.exception()
    if error is None:
        return _JobOutcomes.COMPLETED, ""

    if isinstance(error, BrokenProcessPool):
        return _JobOutcomes.ABANDONED, _BROKEN_POOL_MESSAGE

    message = (
        f"Unable to execute job. The worker process raised {type(error).__name__} outside the job's own error "
        f"handling, which leaves the job holding no terminal state of its own. The reported reason is '{error}'."
    )
    return _JobOutcomes.ABANDONED, message


def _admit_ready_jobs(state: JobExecutionState) -> bool:
    """Moves every admission-pool job whose prerequisites succeeded into its resource class queue.

    Notes:
        Each tracker is snapshotted once per scan and the snapshot is reused for every job that tracker owns, which
        keeps a large batch to one tracker read per recording per polling cycle. Jobs whose prerequisites failed or are
        absent from the tracker are marked FAILED here, so they leave the pool on the cycle that detects the failure.
        Each of them records the reason resolved for it, so a missing prerequisite phase is distinguishable from a
        failed one.

    Args:
        state: The current job execution state, accessed under its lock.

    Returns:
        True if at least one job left the admission pool during this scan, False otherwise.
    """
    if not state.admission_pool:
        return False

    registries: dict[Path, dict[str, JobState]] = {}
    remaining: list[PendingJob] = []
    aborted: list[tuple[PendingJob, str]] = []
    admitted = False

    for pending_job in state.admission_pool:
        registry = registries.get(pending_job.tracker_path)
        if registry is None:
            registry = ProcessingTracker(file_path=pending_job.tracker_path).snapshot()
            registries[pending_job.tracker_path] = registry

        decision, abort_message = _resolve_job_admission(registry=registry, pending_job=pending_job)
        if decision == _AdmissionDecisions.ADMIT:
            state.pending_queues[pending_job.resource_class.name].append(pending_job)
            admitted = True
        elif decision == _AdmissionDecisions.ABORT:
            aborted.append((pending_job, abort_message))
        else:
            remaining.append(pending_job)

    state.admission_pool = remaining

    # Records the reason resolved for each aborted job, because a job blocked by a missing prerequisite phase needs a
    # different remedy from one blocked by a failed phase.
    for aborted_job, aborted_message in aborted:
        _fail_pending_jobs(jobs=[aborted_job], message=aborted_message)

    return admitted or bool(aborted)


def _resolve_job_admission(registry: dict[str, JobState], pending_job: PendingJob) -> tuple[_AdmissionDecisions, str]:
    """Decides whether one queued job may start, must keep waiting, or can never run.

    Args:
        registry: The point-in-time job registry of the tracker that owns the job.
        pending_job: The queued job to evaluate.

    Returns:
        A tuple of the admission decision for the job and the reason to record when that decision is ABORT. The reason
        is an empty string for every other decision.
    """
    prerequisite_ids, missing_message = resolve_prerequisite_job_ids(
        registry=registry, job_id=pending_job.job_id, single_recording=pending_job.single_recording
    )
    if missing_message is not None:
        return _AdmissionDecisions.ABORT, missing_message

    statuses = [registry[prerequisite_id].status for prerequisite_id in prerequisite_ids]
    if any(status == ProcessingStatus.FAILED for status in statuses):
        return _AdmissionDecisions.ABORT, PREREQUISITE_FAILURE_MESSAGE
    if all(status == ProcessingStatus.SUCCEEDED for status in statuses):
        return _AdmissionDecisions.ADMIT, ""

    return _AdmissionDecisions.WAIT, ""


def _dispatch_admitted_jobs(state: JobExecutionState, pool: Executor) -> bool:
    """Submits admitted jobs to the worker pool up to each resource class cap and the session CPU budget.

    Notes:
        Dispatch runs in two passes. The first offers every class the capacity its reservation leaves free, so a class
        holding a reservation cannot take the room the stages that wait on no other job need. The second releases
        every reservation over whatever capacity remains, so a reserved class runs at its full derived width rather
        than idling a host whose other queues have drained. Holding a wide compute stage to a reservation while cores
        sit unused and its own queue is deep would waste the very capacity the reservation protects.

        A per-class cap bounds one class in isolation, and every class dispatches during the same cycle, so the caps
        alone would let the classes oversubscribe the machine between them. Both passes therefore hold the sum of the
        cores and the memory committed by every running job inside the session budgets.

        A session whose classes all hold nothing dispatches one job regardless of the budget, so a job whose worker
        count exceeds the whole budget still runs instead of stalling the session forever.

        Admission alone decides how many jobs run at once, and the pool is sized to accept every job admission
        permits, so a submitted job starts rather than queueing behind the pool's own limit and reordering the
        sequence the prerequisites imply.

    Args:
        state: The current job execution state, accessed under its lock.
        pool: The worker pool the session dispatches its jobs into.

    Returns:
        True if at least one job was submitted during this cycle, False otherwise.

    Raises:
        BrokenProcessPool: If a worker process died outside its job's control, leaving the pool unable to accept work.
    """
    dispatched = False

    for release_reservations in (False, True):
        dispatched |= _dispatch_pass(state=state, pool=pool, release_reservations=release_reservations)

    return dispatched


def _dispatch_pass(state: JobExecutionState, pool: Executor, *, release_reservations: bool) -> bool:
    """Submits admitted jobs during one pass of the dispatcher.

    Notes:
        Each submission precedes the queue mutation it belongs to, so a job the pool refuses stays at the head of its
        class queue. That keeps it inside a collection the broken-pool handler scans, which is what lets the one job
        the pool refused reach a terminal state alongside its peers.

    Args:
        state: The current job execution state, accessed under its lock.
        pool: The worker pool the session dispatches its jobs into.
        release_reservations: Determines whether a class holding a reservation dispatches at its full derived width
            rather than at the width its reservation leaves free.

    Returns:
        True if at least one job was submitted during this pass, False otherwise.

    Raises:
        BrokenProcessPool: If a worker process died outside its job's control, leaving the pool unable to accept work.
    """
    dispatched = False

    for class_name, pending_queue in state.pending_queues.items():
        active_futures = state.active_futures[class_name]
        capacity = state.class_capacities[class_name]
        reservation = state.class_reservations.get(class_name)
        if not release_reservations and reservation is not None:
            capacity = min(capacity, reservation)
        workers = state.class_workers[class_name]
        while len(active_futures) < capacity and pending_queue:
            committed = _committed_cores(state=state)
            if committed > 0 and committed + workers > state.cpu_budget:
                break

            pending_job = pending_queue[0]
            committed_memory = _committed_memory(state=state)
            if (
                committed_memory > 0
                and state.memory_budget_mb > 0
                and committed_memory + pending_job.memory_megabytes > state.memory_budget_mb
            ):
                break

            future: Future[None] = pool.submit(
                _pipeline_worker,
                configuration_path=pending_job.configuration_path,
                job_id=pending_job.job_id,
                tracker_path=pending_job.tracker_path,
                single_recording=pending_job.single_recording,
                workers=pending_job.resolved_workers,
            )
            pending_queue.pop(0)
            active_futures[pending_job.dispatch_key] = future
            dispatched = True

    return dispatched


def _create_job_pool(max_workers: int) -> Executor:
    """Creates the worker pool one execution session dispatches its jobs into.

    Notes:
        Each job runs in its own process, which gives it its own numeric-backend thread budget and its own address
        space. Both matter for this pipeline. The BLAS width a job sets is a property of the process rather than of
        the thread that set it, so concurrent jobs sharing one process would overwrite each other's width and leave
        the last one standing. A detection job additionally materializes the binned movie in anonymous memory, so a
        job the host kills for exhausting it takes down its own process rather than every job beside it.

        Each worker is spawned rather than forked, so it re-executes the interpreter and re-imports the numeric
        backends under the pinned environment the manager holds. That is what makes the pin reach a backend sizing
        itself at import, and it gives every supported platform the same worker semantics. Each worker then pins its
        backends again as it starts, which reaches the backends that read their variable the first time they are
        asked to do work rather than while they are imported.

    Args:
        max_workers: The number of worker processes the pool may hold.

    Returns:
        The pool to dispatch the session's jobs into.
    """
    return ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=get_context(method=_POOL_START_METHOD),
        initializer=initialize_worker_threads,
        initargs=(_WORKER_THREAD_CEILING,),
    )


def _resolve_pool_size(state: JobExecutionState) -> int:
    """Resolves the number of worker processes one execution session's pool may hold.

    Notes:
        Sized to the concurrency every resource class may reach at once, so admission remains the only thing bounding
        how many jobs run. A pool narrower than that would hold an admitted job behind its own queue, which would
        reorder the dispatch sequence the prerequisite graph implies.

    Args:
        state: The execution state whose per-class concurrency caps size the pool.

    Returns:
        The number of worker processes the pool may hold, always at least one.
    """
    return max(1, sum(state.class_capacities.values()))


def _committed_memory(state: JobExecutionState) -> int:
    """Sums the memory that the currently running jobs of every resource class hold.

    Notes:
        A job the caller sized at zero contributes nothing, so a session whose jobs carry no estimates admits on the
        core budget alone.

    Args:
        state: The current job execution state, accessed under its lock.

    Returns:
        The memory the session has committed to running jobs, in megabytes.
    """
    return sum(
        state.all_jobs[dispatch_key].memory_megabytes
        for futures in state.active_futures.values()
        for dispatch_key in futures
    )


def _committed_cores(state: JobExecutionState) -> int:
    """Sums the CPU cores that the currently running jobs of every resource class hold.

    Args:
        state: The current job execution state, accessed under its lock.

    Returns:
        The number of cores the session has committed to running jobs.
    """
    return sum(len(futures) * state.class_workers[class_name] for class_name, futures in state.active_futures.items())


def _pipeline_worker(
    configuration_path: Path,
    job_id: str,
    tracker_path: Path,
    *,
    single_recording: bool = True,
    workers: int | None = None,
) -> None:
    """Executes a single pipeline job identified by its job ID.

    Calls the appropriate pipeline function in REMOTE mode, passing the job_id so the pipeline reads the job definition
    from the ProcessingTracker and updates tracker state on completion or failure. After the pipeline returns or raises,
    verifies that the tracker reached a terminal state and marks the job as failed if the pipeline terminated without
    updating the tracker.

    Notes:
        A remote invocation runs exactly one job, so the allocation the execution manager resolved for that job's
        resource class is given to every stage parameter of the pipeline. Only the parameter of the executed stage is
        read, and a combination job reads none of them because that stage takes no worker allocation.

    Args:
        configuration_path: The path to the recording or dataset configuration file.
        job_id: The unique hexadecimal job identifier registered in the ProcessingTracker.
        tracker_path: The path to the ProcessingTracker file for this job.
        single_recording: Determines whether to call the single-recording or multi-recording pipeline.
        workers: The number of parallel workers to allocate to this job. A value of None makes the pipeline apply the
            measured default for the job's stage.
    """
    try:
        if single_recording:
            run_single_recording_pipeline(
                configuration_path=configuration_path,
                job_id=job_id,
                binarization_workers=workers,
                registration_workers=workers,
                processing_workers=workers,
            )
        else:
            run_multi_recording_pipeline(
                configuration_path=configuration_path,
                job_id=job_id,
                discovery_workers=workers,
                extraction_workers=workers,
            )
    except Exception:  # noqa: S110 - Pipeline may have persisted failure via tracker.fail_job() before re-raising.
        pass
    finally:
        tracker = ProcessingTracker(file_path=tracker_path)
        if tracker.get_job_status(job_id=job_id) not in (ProcessingStatus.SUCCEEDED, ProcessingStatus.FAILED):
            tracker.fail_job(
                job_id=job_id,
                error_message="Unable to complete job. Worker terminated without reaching a terminal state.",
            )


def _fail_broken_session(state: JobExecutionState) -> None:
    """Records a terminal outcome for every job a broken worker pool leaves stranded.

    Notes:
        A pool that lost a worker outside that worker's own control accepts no further work, so neither the jobs it
        never started nor the ones it was running can reach a terminal state on their own. Both sets are recorded here
        rather than left reporting as scheduled or running forever, which is what a consumer polling the trackers
        would otherwise see.

        A dispatched job that already recorded a terminal state of its own keeps that state, so a stage that succeeded
        before the pool broke stays a satisfied prerequisite instead of being failed and re-run.

    Args:
        state: The execution state whose pool broke, accessed under its lock.
    """
    stranded: list[PendingJob] = list(state.admission_pool)
    for pending_queue in state.pending_queues.values():
        stranded.extend(pending_queue)
        pending_queue.clear()
    state.admission_pool.clear()

    for active_futures in state.active_futures.values():
        for key, future in active_futures.items():
            outcome, outcome_message = _resolve_job_outcome(future=future)
            if outcome == _JobOutcomes.COMPLETED or key not in state.all_jobs:
                continue

            # A job still running when the pool broke holds no reason of its own, so it takes the pool's reason.
            reason = outcome_message if outcome == _JobOutcomes.ABANDONED else _BROKEN_POOL_MESSAGE
            _fail_dispatched_job(job=state.all_jobs[key], message=reason)
        active_futures.clear()

    _fail_pending_jobs(jobs=stranded, message=_BROKEN_POOL_MESSAGE)


def _fail_dispatched_job(job: PendingJob, message: str) -> None:
    """Records a failure for one dispatched job, unless that job already reached a terminal state of its own.

    Notes:
        A worker records its own terminal state before its future resolves, and it can then raise on the way out of
        the guard that verified that state. Reading the tracker keeps the outcome the job recorded, so a stage that
        succeeded stays a satisfied prerequisite rather than being failed and re-run.

    Args:
        job: The dispatched job whose future carried no result of its own.
        message: The error message to record when the job holds no terminal state.
    """
    tracker = ProcessingTracker(file_path=job.tracker_path)
    if tracker.get_job_status(job_id=job.job_id) in (ProcessingStatus.SUCCEEDED, ProcessingStatus.FAILED):
        return

    tracker.start_job(job_id=job.job_id)
    tracker.fail_job(job_id=job.job_id, error_message=message)


def _fail_pending_jobs(jobs: list[PendingJob], message: str) -> None:
    """Marks every provided job as failed with the given reason recorded on its tracker.

    Args:
        jobs: The jobs that can no longer run.
        message: The error message to record for each job.
    """
    for job in jobs:
        tracker = ProcessingTracker(file_path=job.tracker_path)
        tracker.start_job(job_id=job.job_id)
        tracker.fail_job(job_id=job.job_id, error_message=message)
