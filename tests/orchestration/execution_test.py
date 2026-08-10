"""Contains tests for the local batch execution engine that admits, dispatches, and reaps queued pipeline jobs."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from concurrent.futures import Future, ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import pytest
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from cindra.orchestration import (
    RESOURCE_CLASS_BY_JOB_NAME,
    PendingJob,
    JobExecutionState,
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    execution,
    get_execution_state,
    set_execution_state,
    resolve_session_load,
    start_execution_session,
    cancel_execution_session,
    resolve_multi_recording_jobs,
    resolve_single_recording_jobs,
)
from cindra.orchestration.jobs import (
    PREREQUISITE_FAILURE_MESSAGE,
    UNREACHABLE_PREREQUISITE_MESSAGE,
)
from cindra.orchestration.execution import (
    _committed_cores,
    _pipeline_worker,
    _admit_ready_jobs,
    _fail_pending_jobs,
    _AdmissionDecisions,
    _reap_completed_jobs,
    _resolve_job_admission,
    _dispatch_admitted_jobs,
)
from cindra.orchestration.allocation import BINARIZATION_WORKERS, resolve_core_budget

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable, Iterator, Sequence

_JOIN_TIMEOUT: float = 30.0
"""The number of seconds a test waits for a worker or manager thread to terminate before failing."""

_DRAIN_TIMEOUT: float = 60.0
"""The number of seconds a test waits for a started execution session to drain before failing."""

_DRAIN_POLL_SECONDS: float = 0.05
"""The interval at which a test re-checks whether the started execution session has cleared its state."""

_TERMINAL_STATE_MESSAGE: str = "Unable to complete job. Worker terminated without reaching a terminal state."
"""The tracker error message the pipeline worker records when the pipeline returns without a terminal tracker state."""


def _build_single_recording_tracker(tracker_path: Path, *, plane_count: int = 1) -> ProcessingTracker:
    """Creates a single-recording tracker whose registry holds the full job universe of the given plane count."""
    universe = resolve_single_recording_jobs(plane_count=plane_count)
    tracker = ProcessingTracker(file_path=tracker_path)
    tracker.align_jobs(jobs=universe, universe=universe)
    return tracker


def _build_multi_recording_tracker(tracker_path: Path, *, recording_ids: Sequence[str]) -> ProcessingTracker:
    """Creates a multi-recording tracker whose registry holds the full job universe of the given recordings."""
    universe = resolve_multi_recording_jobs(recording_ids=recording_ids)
    tracker = ProcessingTracker(file_path=tracker_path)
    tracker.align_jobs(jobs=universe, universe=universe)
    return tracker


def _make_job(tracker_path: Path, job_name: str, specifier: str = "", *, single_recording: bool = True) -> PendingJob:
    """Builds a queued job addressing the named tracker entry through the resource class of its pipeline stage."""
    return PendingJob(
        configuration_path=tracker_path.parent / "configuration.yaml",
        tracker_path=tracker_path,
        job_id=ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier),
        single_recording=single_recording,
        resource_class=RESOURCE_CLASS_BY_JOB_NAME[job_name],
    )


def _make_state(
    jobs: Sequence[PendingJob],
    *,
    admitted: bool = False,
    capacity: int = 4,
    workers: int = 1,
    cpu_budget: int = 64,
) -> JobExecutionState:
    """Builds an execution state holding the given jobs in the admission pool or in their resource class queues."""
    class_names = {job.resource_class.name for job in jobs}
    queues: dict[str, list[PendingJob]] = {class_name: [] for class_name in class_names}
    if admitted:
        for job in jobs:
            queues[job.resource_class.name].append(job)

    return JobExecutionState(
        all_jobs={job.dispatch_key: job for job in jobs},
        admission_pool=[] if admitted else list(jobs),
        pending_queues=queues,
        active_futures={class_name: {} for class_name in class_names},
        class_capacities=dict.fromkeys(class_names, capacity),
        class_workers=dict.fromkeys(class_names, workers),
        cpu_budget=cpu_budget,
    )


def _use_same_process_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replaces the session's worker pool with a same-process one, so a patched worker stays reachable."""
    monkeypatch.setattr(execution, "_create_job_pool", lambda max_workers: ThreadPoolExecutor(max_workers=max_workers))


def _make_completing_worker(observed: list[str]) -> Callable[..., None]:
    """Returns a pipeline worker stub that records the dispatched job identifier and marks that job succeeded."""

    def _worker(
        configuration_path: Path,
        job_id: str,
        tracker_path: Path,
        *,
        single_recording: bool = True,
        workers: int | None = None,
    ) -> None:
        observed.append(job_id)
        ProcessingTracker(file_path=tracker_path).complete_job(job_id=job_id)

    return _worker


def _make_recording_worker(observed: list[dict[str, Any]]) -> Callable[..., None]:
    """Returns a pipeline worker stub that records the keyword arguments the dispatcher hands it."""

    def _worker(**kwargs: Any) -> None:
        observed.append(kwargs)

    return _worker


def _make_finished_future() -> Future[None]:
    """Returns a resolved future, which the reaper treats as a completed job."""
    future: Future[None] = Future()
    future.set_result(None)
    return future


def _drain_active_futures(state: JobExecutionState) -> None:
    """Waits for every job the dispatcher submitted to finish."""
    for futures in state.active_futures.values():
        for future in futures.values():
            future.result(timeout=_JOIN_TIMEOUT)


def _wait_for_session_end() -> None:
    """Blocks until the execution manager clears the module-global session state or the drain timeout elapses."""
    deadline = time.monotonic() + _DRAIN_TIMEOUT
    while get_execution_state() is not None and time.monotonic() < deadline:
        time.sleep(_DRAIN_POLL_SECONDS)


@pytest.fixture(autouse=True)
def _isolated_execution_state() -> Iterator[None]:
    """Clears the module-global execution state around every test, so no session leaks between them."""
    set_execution_state(state=None)
    yield
    state = get_execution_state()
    set_execution_state(state=None)
    if state is not None and state.manager_thread is not None:
        state.manager_thread.join(timeout=_JOIN_TIMEOUT)


class TestPendingJob:
    """Tests the queued job descriptor the execution engine passes between its stages."""

    @pytest.mark.xdist_group(name="execution_state")
    def test_dispatch_key_pairs_the_tracker_path_with_the_job_identifier(self, tmp_path: Path) -> None:
        """Verifies that the dispatch key identifies a job by its tracker path string and its job identifier."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        job = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.BINARIZE)

        assert job.dispatch_key == (str(tracker_path), job.job_id)
        assert job.resolved_workers is None


class TestExecutionStateAccessors:
    """Tests the module-global accessors that publish the active execution session."""

    @pytest.mark.xdist_group(name="execution_state")
    def test_state_is_absent_before_a_session_starts(self) -> None:
        """Verifies that the accessor reports no session while the module-global reference is cleared."""
        assert get_execution_state() is None

    @pytest.mark.xdist_group(name="execution_state")
    def test_stored_state_is_returned_until_it_is_cleared(self) -> None:
        """Verifies that a stored session is handed back unchanged and that None clears the stored reference."""
        state = JobExecutionState()

        set_execution_state(state=state)
        assert get_execution_state() is state

        set_execution_state(state=None)
        assert get_execution_state() is None


class TestResolveSessionLoad:
    """Tests the queued and running job counts the session reports."""

    @pytest.mark.xdist_group(name="execution_state")
    def test_absent_session_reports_no_load(self) -> None:
        """Verifies that a cleared session reports neither pending nor running jobs."""
        assert resolve_session_load() == (0, 0)

    @pytest.mark.xdist_group(name="execution_state")
    def test_active_session_counts_pooled_queued_and_running_jobs(self, tmp_path: Path) -> None:
        """Verifies that the pending count spans the admission pool and every class queue, apart from running jobs."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        pooled = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.COMBINE)
        queued = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0")
        running = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.PROCESS, specifier="plane_0")
        state = _make_state(jobs=[pooled, queued, running])
        state.admission_pool = [pooled]
        state.pending_queues[queued.resource_class.name] = [queued]
        state.active_futures[running.resource_class.name] = {running.dispatch_key: _make_finished_future()}
        set_execution_state(state=state)

        assert resolve_session_load() == (2, 1)


class TestCancelExecutionSession:
    """Tests the cancellation that empties the queues while leaving the running jobs alone."""

    @pytest.mark.xdist_group(name="execution_state")
    def test_absent_session_cancels_nothing(self) -> None:
        """Verifies that cancelling without an active session reports no cleared and no running jobs."""
        assert cancel_execution_session() == (0, 0)

    @pytest.mark.xdist_group(name="execution_state")
    def test_active_session_clears_every_queue_and_reports_running_jobs(self, tmp_path: Path) -> None:
        """Verifies that cancellation empties the admission pool and the class queues but keeps the running set."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        pooled = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.COMBINE)
        queued = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0")
        running = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.PROCESS, specifier="plane_0")
        state = _make_state(jobs=[pooled, queued, running])
        state.admission_pool = [pooled]
        state.pending_queues[queued.resource_class.name] = [queued]
        state.active_futures[running.resource_class.name] = {running.dispatch_key: _make_finished_future()}
        set_execution_state(state=state)

        canceled_count, active_count = cancel_execution_session()

        assert canceled_count == 2
        assert active_count == 1
        assert state.admission_pool == []
        assert all(not pending_queue for pending_queue in state.pending_queues.values())
        assert len(state.active_futures[running.resource_class.name]) == 1


class TestStartExecutionSession:
    """Tests the session bootstrap that resolves the allocation and starts the execution manager."""

    @pytest.mark.xdist_group(name="execution_state")
    @pytest.mark.parametrize(
        ("workers_per_job", "max_parallel_jobs", "override_name"),
        [
            (0, None, "workers_per_job"),
            (-2, None, "workers_per_job"),
            (None, 0, "max_parallel_jobs"),
            (None, -3, "max_parallel_jobs"),
        ],
    )
    def test_invalid_override_is_rejected(
        self, tmp_path: Path, workers_per_job: int | None, max_parallel_jobs: int | None, override_name: str
    ) -> None:
        """Verifies that zero and every negative override other than the all-cores request raise before any dispatch."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        job = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.BINARIZE)

        with pytest.raises(ValueError, match=f"The '{override_name}' override must be a positive integer"):
            start_execution_session(
                all_jobs={job.dispatch_key: job},
                workers_per_job=workers_per_job,
                max_parallel_jobs=max_parallel_jobs,
            )

        assert get_execution_state() is None

    @pytest.mark.xdist_group(name="execution_state")
    def test_session_stamps_the_allocation_and_drains_every_job(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verifies that the session reports its allocation, stamps it onto the jobs, and runs them in phase order."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        tracker = _build_single_recording_tracker(tracker_path=tracker_path)
        binarize_job = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.BINARIZE)
        register_job = _make_job(
            tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0"
        )
        observed: list[str] = []
        monkeypatch.setattr(execution, "_pipeline_worker", _make_completing_worker(observed=observed))
        _use_same_process_pool(monkeypatch=monkeypatch)

        summary = start_execution_session(
            all_jobs={job.dispatch_key: job for job in (binarize_job, register_job)},
            workers_per_job=2,
            max_parallel_jobs=1,
        )

        assert summary["total_jobs"] == 2
        assert summary["cpu_budget"] == resolve_core_budget()
        assert summary["resource_classes"] == {
            "binarization": {"workers_per_job": BINARIZATION_WORKERS, "max_parallel_jobs": 1, "job_count": 1},
            "registration": {"workers_per_job": 2, "max_parallel_jobs": 1, "job_count": 1},
        }
        assert binarize_job.resolved_workers == BINARIZATION_WORKERS
        assert register_job.resolved_workers == 2

        state = get_execution_state()
        assert state is not None
        manager = state.manager_thread
        assert manager is not None

        _wait_for_session_end()
        manager.join(timeout=_JOIN_TIMEOUT)

        assert get_execution_state() is None
        assert observed == [binarize_job.job_id, register_job.job_id]
        assert tracker.get_job_status(job_id=binarize_job.job_id) == ProcessingStatus.SUCCEEDED
        assert tracker.get_job_status(job_id=register_job.job_id) == ProcessingStatus.SUCCEEDED


class TestReapCompletedJobs:
    """Tests the reaper that frees the concurrency held by finished jobs."""

    @pytest.mark.xdist_group(name="execution_state")
    def test_finished_jobs_are_removed_and_running_jobs_are_kept(self, tmp_path: Path) -> None:
        """Verifies that only the jobs whose future resolved leave the running set of their resource class."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        finished = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0")
        running = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_1")
        state = _make_state(jobs=[finished, running])
        state.active_futures[finished.resource_class.name] = {
            finished.dispatch_key: _make_finished_future(),
            running.dispatch_key: Future(),
        }

        _reap_completed_jobs(state=state)

        assert list(state.active_futures[finished.resource_class.name]) == [running.dispatch_key]


class TestResolveJobAdmission:
    """Tests the per-job decision the manager takes from a tracker snapshot."""

    @pytest.mark.xdist_group(name="execution_state")
    def test_first_phase_job_is_admitted(self, tmp_path: Path) -> None:
        """Verifies that a job whose phase opens the pipeline admits without waiting on any prerequisite."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        tracker = _build_single_recording_tracker(tracker_path=tracker_path)
        job = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.BINARIZE)

        decision, message = _resolve_job_admission(registry=tracker.snapshot(), pending_job=job)

        assert decision == _AdmissionDecisions.ADMIT
        assert message == ""

    @pytest.mark.xdist_group(name="execution_state")
    def test_succeeded_prerequisite_admits_a_multi_recording_job(self, tmp_path: Path) -> None:
        """Verifies that a multi-recording extraction job admits once its dataset discovery job has succeeded."""
        tracker_path = tmp_path / "multi_recording_tracker.yaml"
        tracker = _build_multi_recording_tracker(tracker_path=tracker_path, recording_ids=["recording_a"])
        discover_id = ProcessingTracker.generate_job_id(job_name=MultiRecordingJobNames.DISCOVER, specifier="")
        tracker.complete_job(job_id=discover_id)
        job = _make_job(
            tracker_path=tracker_path,
            job_name=MultiRecordingJobNames.EXTRACT,
            specifier="recording_a",
            single_recording=False,
        )

        decision, message = _resolve_job_admission(registry=tracker.snapshot(), pending_job=job)

        assert decision == _AdmissionDecisions.ADMIT
        assert message == ""

    @pytest.mark.xdist_group(name="execution_state")
    def test_unfinished_prerequisite_keeps_the_job_waiting(self, tmp_path: Path) -> None:
        """Verifies that a job whose prerequisite is still scheduled stays in the admission pool."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        tracker = _build_single_recording_tracker(tracker_path=tracker_path)
        job = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0")

        decision, message = _resolve_job_admission(registry=tracker.snapshot(), pending_job=job)

        assert decision == _AdmissionDecisions.WAIT
        assert message == ""

    @pytest.mark.xdist_group(name="execution_state")
    def test_failed_prerequisite_aborts_the_job(self, tmp_path: Path) -> None:
        """Verifies that a failed prerequisite aborts the dependent job with the shared prerequisite failure reason."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        tracker = _build_single_recording_tracker(tracker_path=tracker_path)
        binarize_id = ProcessingTracker.generate_job_id(job_name=SingleRecordingJobNames.BINARIZE, specifier="")
        tracker.start_job(job_id=binarize_id)
        tracker.fail_job(job_id=binarize_id, error_message="Unable to binarize the recording.")
        job = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0")

        decision, message = _resolve_job_admission(registry=tracker.snapshot(), pending_job=job)

        assert decision == _AdmissionDecisions.ABORT
        assert message == PREREQUISITE_FAILURE_MESSAGE

    @pytest.mark.xdist_group(name="execution_state")
    def test_missing_prerequisite_phase_aborts_the_job(self, tmp_path: Path) -> None:
        """Verifies that a tracker holding no job of the prerequisite phase aborts the dependent job."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        registered = [(str(SingleRecordingJobNames.REGISTER), "plane_0")]
        tracker = ProcessingTracker(file_path=tracker_path)
        tracker.align_jobs(jobs=registered, universe=registered)
        job = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0")

        decision, message = _resolve_job_admission(registry=tracker.snapshot(), pending_job=job)

        assert decision == _AdmissionDecisions.ABORT
        assert "prerequisite 'binarization' phase" in message

    @pytest.mark.xdist_group(name="execution_state")
    def test_unregistered_job_aborts_the_job(self, tmp_path: Path) -> None:
        """Verifies that a job absent from its own tracker registry aborts rather than waiting forever."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        job = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.BINARIZE)

        decision, message = _resolve_job_admission(registry={}, pending_job=job)

        assert decision == _AdmissionDecisions.ABORT
        assert "is not registered in the tracker" in message


class TestAdmitReadyJobs:
    """Tests the admission scan that moves pooled jobs into their resource class queues."""

    @pytest.mark.xdist_group(name="execution_state")
    def test_empty_pool_reports_no_admission(self) -> None:
        """Verifies that scanning an empty admission pool reads no tracker and reports no admission."""
        assert _admit_ready_jobs(state=JobExecutionState()) is False

    @pytest.mark.xdist_group(name="execution_state")
    def test_ready_job_is_queued_while_a_waiting_job_stays_pooled(self, tmp_path: Path) -> None:
        """Verifies that one tracker snapshot serves every job it owns, queueing the ready one and pooling the rest."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        _build_single_recording_tracker(tracker_path=tracker_path)
        binarize_job = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.BINARIZE)
        combine_job = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.COMBINE)
        state = _make_state(jobs=[binarize_job, combine_job])

        admitted = _admit_ready_jobs(state=state)

        assert admitted is True
        assert state.pending_queues[binarize_job.resource_class.name] == [binarize_job]
        assert state.pending_queues[combine_job.resource_class.name] == []
        assert state.admission_pool == [combine_job]

    @pytest.mark.xdist_group(name="execution_state")
    def test_failed_prerequisite_fails_the_job_and_empties_the_pool(self, tmp_path: Path) -> None:
        """Verifies that a job blocked by a failed phase leaves the pool and records the prerequisite failure."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        tracker = _build_single_recording_tracker(tracker_path=tracker_path)
        binarize_id = ProcessingTracker.generate_job_id(job_name=SingleRecordingJobNames.BINARIZE, specifier="")
        tracker.start_job(job_id=binarize_id)
        tracker.fail_job(job_id=binarize_id, error_message="Unable to binarize the recording.")
        job = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0")
        state = _make_state(jobs=[job])

        admitted = _admit_ready_jobs(state=state)

        assert admitted is True
        assert state.admission_pool == []
        assert state.pending_queues[job.resource_class.name] == []
        assert tracker.get_job_status(job_id=job.job_id) == ProcessingStatus.FAILED
        assert tracker.get_job_info(job_id=job.job_id).error_message == PREREQUISITE_FAILURE_MESSAGE

    @pytest.mark.xdist_group(name="execution_state")
    def test_missing_prerequisite_phase_fails_the_job_with_its_own_reason(self, tmp_path: Path) -> None:
        """Verifies that a job blocked by an unregistered phase records that reason instead of a phase failure."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        registered = [(str(SingleRecordingJobNames.REGISTER), "plane_0")]
        tracker = ProcessingTracker(file_path=tracker_path)
        tracker.align_jobs(jobs=registered, universe=registered)
        job = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0")
        state = _make_state(jobs=[job])

        admitted = _admit_ready_jobs(state=state)

        assert admitted is True
        assert state.admission_pool == []
        assert tracker.get_job_status(job_id=job.job_id) == ProcessingStatus.FAILED
        assert "prerequisite 'binarization' phase" in str(tracker.get_job_info(job_id=job.job_id).error_message)

    @pytest.mark.xdist_group(name="execution_state")
    def test_waiting_job_alone_reports_no_admission(self, tmp_path: Path) -> None:
        """Verifies that a scan admitting and aborting nothing leaves the pool intact and reports no progress."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        _build_single_recording_tracker(tracker_path=tracker_path)
        job = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0")
        state = _make_state(jobs=[job])

        assert _admit_ready_jobs(state=state) is False
        assert state.admission_pool == [job]


class TestDispatchAdmittedJobs:
    """Tests the dispatcher that submits jobs under the class caps and the session CPU budget."""

    @pytest.mark.xdist_group(name="execution_state")
    @pytest.mark.parametrize("cpu_budget", [10, 4])
    def test_first_job_dispatches_before_the_budget_bounds_the_rest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cpu_budget: int
    ) -> None:
        """Verifies that an idle session always dispatches one job, after which the budget stops the next dispatch."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        first = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0")
        second = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_1")
        first.resolved_workers = 8
        second.resolved_workers = 8
        state = _make_state(jobs=[first, second], admitted=True, capacity=4, workers=8, cpu_budget=cpu_budget)
        observed: list[dict[str, Any]] = []
        monkeypatch.setattr(execution, "_pipeline_worker", _make_recording_worker(observed=observed))

        with ThreadPoolExecutor(max_workers=4) as pool:
            dispatched = _dispatch_admitted_jobs(state=state, pool=pool)

        _drain_active_futures(state=state)

        assert dispatched is True
        assert list(state.active_futures[first.resource_class.name]) == [first.dispatch_key]
        assert state.pending_queues[first.resource_class.name] == [second]
        assert observed == [
            {
                "configuration_path": first.configuration_path,
                "job_id": first.job_id,
                "tracker_path": tracker_path,
                "single_recording": True,
                "workers": 8,
            }
        ]

    @pytest.mark.xdist_group(name="execution_state")
    def test_dispatch_continues_while_the_budget_holds(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifies that a session whose committed cores stay inside the budget drains its whole class queue."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        first = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0")
        second = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_1")
        state = _make_state(jobs=[first, second], admitted=True, capacity=4, workers=8, cpu_budget=32)
        observed: list[dict[str, Any]] = []
        monkeypatch.setattr(execution, "_pipeline_worker", _make_recording_worker(observed=observed))

        with ThreadPoolExecutor(max_workers=4) as pool:
            dispatched = _dispatch_admitted_jobs(state=state, pool=pool)

        _drain_active_futures(state=state)

        assert dispatched is True
        assert state.pending_queues[first.resource_class.name] == []
        assert len(state.active_futures[first.resource_class.name]) == 2
        assert [entry["job_id"] for entry in observed] == [first.job_id, second.job_id]

    @pytest.mark.xdist_group(name="execution_state")
    def test_saturated_class_dispatches_nothing(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifies that a class already holding its concurrency cap leaves its queued job untouched."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        running = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0")
        queued = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_1")
        state = _make_state(jobs=[queued], admitted=True, capacity=1, workers=8, cpu_budget=64)
        state.active_futures[running.resource_class.name] = {running.dispatch_key: _make_finished_future()}
        observed: list[dict[str, Any]] = []
        monkeypatch.setattr(execution, "_pipeline_worker", _make_recording_worker(observed=observed))

        with ThreadPoolExecutor(max_workers=4) as pool:
            dispatched = _dispatch_admitted_jobs(state=state, pool=pool)

        assert dispatched is False
        assert state.pending_queues[queued.resource_class.name] == [queued]
        assert observed == []


class TestCommittedCores:
    """Tests the running core total the dispatcher holds inside the session budget."""

    @pytest.mark.xdist_group(name="execution_state")
    def test_idle_session_commits_no_cores(self, tmp_path: Path) -> None:
        """Verifies that a session running no jobs commits no cores at all."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        job = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0")

        assert _committed_cores(state=_make_state(jobs=[job], workers=8)) == 0

    @pytest.mark.xdist_group(name="execution_state")
    def test_committed_cores_sum_every_running_job_of_every_class(self, tmp_path: Path) -> None:
        """Verifies that the total weights each class's running job count by that class's per-job worker count."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        register = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0")
        process = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.PROCESS, specifier="plane_0")
        state = _make_state(jobs=[register, process])
        state.class_workers = {register.resource_class.name: 8, process.resource_class.name: 10}
        state.active_futures[register.resource_class.name] = {
            ("first", "job"): _make_finished_future(),
            ("second", "job"): _make_finished_future(),
        }
        state.active_futures[process.resource_class.name] = {("third", "job"): _make_finished_future()}

        assert _committed_cores(state=state) == 26


class TestPipelineWorker:
    """Tests the worker that runs one dispatched job and guarantees a terminal tracker state."""

    @pytest.mark.xdist_group(name="execution_state")
    def test_single_recording_job_runs_the_single_recording_pipeline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verifies that a single-recording job reaches the single-recording pipeline at its resolved worker width."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        tracker = _build_single_recording_tracker(tracker_path=tracker_path)
        job_id = ProcessingTracker.generate_job_id(job_name=SingleRecordingJobNames.BINARIZE, specifier="")
        observed: list[dict[str, Any]] = []

        def _pipeline(**kwargs: Any) -> None:
            observed.append(kwargs)
            ProcessingTracker(file_path=tracker_path).complete_job(job_id=job_id)

        monkeypatch.setattr(execution, "run_single_recording_pipeline", _pipeline)

        _pipeline_worker(
            configuration_path=tmp_path / "configuration.yaml",
            job_id=job_id,
            tracker_path=tracker_path,
            single_recording=True,
            workers=4,
        )

        assert observed == [
            {
                "configuration_path": tmp_path / "configuration.yaml",
                "job_id": job_id,
                "binarization_workers": 4,
                "registration_workers": 4,
                "processing_workers": 4,
            }
        ]
        assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.SUCCEEDED

    @pytest.mark.xdist_group(name="execution_state")
    def test_multi_recording_job_without_a_terminal_state_is_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verifies that a multi-recording pipeline returning without a terminal tracker state fails its job."""
        tracker_path = tmp_path / "multi_recording_tracker.yaml"
        tracker = _build_multi_recording_tracker(tracker_path=tracker_path, recording_ids=["recording_a"])
        job_id = ProcessingTracker.generate_job_id(job_name=MultiRecordingJobNames.EXTRACT, specifier="recording_a")
        observed: list[dict[str, Any]] = []

        def _pipeline(**kwargs: Any) -> None:
            observed.append(kwargs)

        monkeypatch.setattr(execution, "run_multi_recording_pipeline", _pipeline)

        _pipeline_worker(
            configuration_path=tmp_path / "configuration.yaml",
            job_id=job_id,
            tracker_path=tracker_path,
            single_recording=False,
            workers=6,
        )

        assert observed == [
            {
                "configuration_path": tmp_path / "configuration.yaml",
                "job_id": job_id,
                "discovery_workers": 6,
                "extraction_workers": 6,
            }
        ]
        assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED
        assert tracker.get_job_info(job_id=job_id).error_message == _TERMINAL_STATE_MESSAGE

    @pytest.mark.xdist_group(name="execution_state")
    def test_raising_pipeline_is_swallowed_and_the_job_is_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verifies that a pipeline error does not escape the worker and leaves the job in a terminal failed state."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        tracker = _build_single_recording_tracker(tracker_path=tracker_path)
        job_id = ProcessingTracker.generate_job_id(job_name=SingleRecordingJobNames.BINARIZE, specifier="")

        def _pipeline(**kwargs: Any) -> None:
            raise RuntimeError("Unable to execute the pipeline.")

        monkeypatch.setattr(execution, "run_single_recording_pipeline", _pipeline)

        _pipeline_worker(
            configuration_path=tmp_path / "configuration.yaml",
            job_id=job_id,
            tracker_path=tracker_path,
            single_recording=True,
            workers=None,
        )

        assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED
        assert tracker.get_job_info(job_id=job_id).error_message == _TERMINAL_STATE_MESSAGE


class TestFailPendingJobs:
    """Tests the bulk failure the manager records for jobs it can no longer run."""

    @pytest.mark.xdist_group(name="execution_state")
    def test_every_job_is_marked_failed_with_the_given_reason(self, tmp_path: Path) -> None:
        """Verifies that each named job is started and failed on its own tracker with the provided message."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        tracker = _build_single_recording_tracker(tracker_path=tracker_path)
        register = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0")
        combine = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.COMBINE)

        _fail_pending_jobs(jobs=[register, combine], message=UNREACHABLE_PREREQUISITE_MESSAGE)

        for job in (register, combine):
            assert tracker.get_job_status(job_id=job.job_id) == ProcessingStatus.FAILED
            assert tracker.get_job_info(job_id=job.job_id).error_message == UNREACHABLE_PREREQUISITE_MESSAGE


class TestClearOwnedSession:
    """Tests the ownership check that guards the module-level session reference."""

    @pytest.mark.xdist_group(name="execution_state")
    def test_owned_session_is_cleared(self) -> None:
        """Verifies that a terminating manager clears the reference while it still names its own session."""
        state = JobExecutionState()
        set_execution_state(state=state)

        execution._clear_owned_session(state=state)

        assert get_execution_state() is None

    @pytest.mark.xdist_group(name="execution_state")
    def test_replaced_session_is_left_installed(self) -> None:
        """Verifies that a terminating manager leaves a session that replaced its own untouched."""
        replacement = JobExecutionState()
        set_execution_state(state=replacement)

        execution._clear_owned_session(state=JobExecutionState())

        assert get_execution_state() is replacement


class TestJobPool:
    """Tests the worker pool one execution session dispatches its jobs into."""

    @pytest.mark.xdist_group(name="execution_state")
    def test_pool_runs_each_job_in_its_own_process(self) -> None:
        """Verifies that the session pool is a process pool, so a job holds its own numeric-backend budget."""
        pool = execution._create_job_pool(max_workers=1)
        try:
            assert isinstance(pool, ProcessPoolExecutor)
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    @pytest.mark.xdist_group(name="execution_state")
    def test_pool_spawns_its_workers_on_every_platform(self) -> None:
        """Verifies that the session pool requests the spawn start method rather than the host default."""
        pool = execution._create_job_pool(max_workers=1)
        try:
            assert pool._mp_context.get_start_method() == "spawn"
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    @pytest.mark.xdist_group(name="execution_state")
    @pytest.mark.parametrize(
        ("capacities", "expected_size"),
        [({}, 1), ({"registration": 3}, 3), ({"registration": 3, "processing": 2}, 5)],
    )
    def test_pool_size_covers_every_class_concurrency(self, capacities: dict[str, int], expected_size: int) -> None:
        """Verifies that the pool holds a worker for every job the per-class caps allow to run at once."""
        state = JobExecutionState(class_capacities=capacities)

        assert execution._resolve_pool_size(state=state) == expected_size


class TestBrokenPool:
    """Tests the outcome recorded when a worker process dies outside its job's control."""

    @pytest.mark.xdist_group(name="execution_state")
    def test_broken_pool_fails_every_job_the_session_holds(self, tmp_path: Path) -> None:
        """Verifies that a broken pool records a terminal outcome for the pooled, queued, and running jobs alike."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        tracker = _build_single_recording_tracker(tracker_path=tracker_path, plane_count=3)
        pooled = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0")
        queued = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_1")
        running = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_2")
        state = _make_state(jobs=[pooled, queued, running])
        state.pending_queues[queued.resource_class.name].append(queued)
        state.active_futures[running.resource_class.name] = {running.dispatch_key: Future()}

        execution._fail_broken_session(state=state)

        assert state.admission_pool == []
        assert state.pending_queues[queued.resource_class.name] == []
        assert state.active_futures[running.resource_class.name] == {}
        for job in (pooled, queued, running):
            assert tracker.get_job_status(job_id=job.job_id) == ProcessingStatus.FAILED
            assert tracker.get_job_info(job_id=job.job_id).error_message == execution._BROKEN_POOL_MESSAGE

    @pytest.mark.xdist_group(name="execution_state")
    def test_manager_ends_the_session_when_the_pool_breaks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verifies that a dispatch the pool refuses fails the session's jobs and clears the session state."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        tracker = _build_single_recording_tracker(tracker_path=tracker_path)
        job = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.BINARIZE)
        state = _make_state(jobs=[job], admitted=True)
        _use_same_process_pool(monkeypatch=monkeypatch)

        def _refuse(state: JobExecutionState, pool: Any) -> bool:
            """Stands in for a dispatch against a pool whose worker process died."""
            raise BrokenProcessPool

        monkeypatch.setattr(execution, "_dispatch_admitted_jobs", _refuse)
        set_execution_state(state=state)

        execution._job_execution_manager(state=state)

        assert get_execution_state() is None
        assert tracker.get_job_status(job_id=job.job_id) == ProcessingStatus.FAILED


class TestJobExecutionManager:
    """Tests the polling loop that admits, dispatches, and terminates a batch execution session."""

    @pytest.mark.xdist_group(name="execution_state")
    def test_unowned_session_terminates_immediately(self) -> None:
        """Verifies that a manager whose session no longer holds the module reference returns without dispatching."""
        replaced = JobExecutionState()
        set_execution_state(state=JobExecutionState())

        execution._job_execution_manager(state=replaced)

        # The manager left the session that replaced its own installed, rather than clearing it.
        assert get_execution_state() is not None

    @pytest.mark.xdist_group(name="execution_state")
    def test_drained_session_clears_the_state(self) -> None:
        """Verifies that a session holding no pooled, queued, or running job clears itself and terminates."""
        state = JobExecutionState()
        set_execution_state(state=state)

        execution._job_execution_manager(state=state)

        assert get_execution_state() is None

    @pytest.mark.xdist_group(name="execution_state")
    def test_unreachable_prerequisites_fail_the_whole_pool(self, tmp_path: Path) -> None:
        """Verifies that a session that can make no further progress fails every job it still holds."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        tracker = _build_single_recording_tracker(tracker_path=tracker_path)
        job = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0")
        state = _make_state(jobs=[job])
        set_execution_state(state=state)

        execution._job_execution_manager(state=state)

        assert get_execution_state() is None
        assert state.admission_pool == []
        assert tracker.get_job_status(job_id=job.job_id) == ProcessingStatus.FAILED
        assert tracker.get_job_info(job_id=job.job_id).error_message == UNREACHABLE_PREREQUISITE_MESSAGE

    @pytest.mark.xdist_group(name="execution_state")
    def test_dispatched_job_is_reaped_before_the_session_drains(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verifies that the manager keeps polling while a job runs and terminates on the cycle that reaps it."""
        tracker_path = tmp_path / "single_recording_tracker.yaml"
        tracker = _build_single_recording_tracker(tracker_path=tracker_path)
        job = _make_job(tracker_path=tracker_path, job_name=SingleRecordingJobNames.BINARIZE)
        state = _make_state(jobs=[job])
        observed: list[str] = []
        monkeypatch.setattr(execution, "_pipeline_worker", _make_completing_worker(observed=observed))
        _use_same_process_pool(monkeypatch=monkeypatch)
        set_execution_state(state=state)

        execution._job_execution_manager(state=state)

        assert get_execution_state() is None
        assert observed == [job.job_id]
        assert state.admission_pool == []
        assert state.active_futures[job.resource_class.name] == {}
        assert tracker.get_job_status(job_id=job.job_id) == ProcessingStatus.SUCCEEDED
