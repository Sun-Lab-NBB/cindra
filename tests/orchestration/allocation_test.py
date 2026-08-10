"""Contains tests for the per-stage worker defaults, the resource-class model, and the allocation resolvers."""

import pytest
from ataraxis_base_utilities import resolve_worker_count

from cindra.orchestration import (
    ALL_CORES_REQUEST,
    DISCOVERY_WORKERS,
    EXTRACTION_WORKERS,
    PROCESSING_WORKERS,
    BINARIZATION_WORKERS,
    REGISTRATION_WORKERS,
    RESOURCE_CLASS_BY_JOB_NAME,
    ResourceClass,
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    resolve_stage_workers,
)
from cindra.orchestration.allocation import (
    _RESERVED_CORES,
    COMBINATION_WORKERS,
    _BYTES_PER_GIGABYTE,
    _DISCOVERY_RESOURCES,
    _EXTRACTION_RESOURCES,
    _PROCESSING_RESOURCES,
    _COMBINATION_RESOURCES,
    _BINARIZATION_RESOURCES,
    _REGISTRATION_RESOURCES,
    _MAXIMUM_PARALLEL_IO_JOBS,
    _PROCESSING_MEMORY_GIGABYTES_PER_JOB,
    resolve_core_budget,
    resolve_class_allocation,
    summarize_class_allocation,
    resolve_available_memory_gigabytes,
)


class _VirtualMemory:
    """Stands in for the psutil memory record, carrying only the available-byte counter the resolver reads."""

    def __init__(self, available: int) -> None:
        self.available = available


class TestStageDefaults:
    """Tests the measured worker default each pipeline stage resolves to."""

    @pytest.mark.parametrize(
        ("job_name", "expected_workers"),
        [
            (SingleRecordingJobNames.BINARIZE, BINARIZATION_WORKERS),
            (SingleRecordingJobNames.REGISTER, REGISTRATION_WORKERS),
            (SingleRecordingJobNames.PROCESS, PROCESSING_WORKERS),
            (MultiRecordingJobNames.DISCOVER, DISCOVERY_WORKERS),
            (MultiRecordingJobNames.EXTRACT, EXTRACTION_WORKERS),
        ],
    )
    def test_unspecified_request_resolves_to_the_stage_default(
        self, job_name: SingleRecordingJobNames | MultiRecordingJobNames, expected_workers: int
    ) -> None:
        """Verifies that omitting the requested count resolves to the measured default of the target stage."""
        assert resolve_stage_workers(job_name=job_name) == expected_workers
        assert resolve_stage_workers(job_name=job_name, requested_workers=None) == expected_workers

    def test_stage_defaults_are_positive(self) -> None:
        """Verifies that every measured stage default is a usable positive worker count."""
        assert BINARIZATION_WORKERS > 0
        assert REGISTRATION_WORKERS > 0
        assert PROCESSING_WORKERS > 0
        assert DISCOVERY_WORKERS > 0
        assert EXTRACTION_WORKERS > 0


class TestExplicitRequests:
    """Tests the worker counts a caller asks for explicitly."""

    @pytest.mark.parametrize("requested_workers", [1, 7, 30, 512])
    def test_positive_request_is_honored_exactly(self, requested_workers: int) -> None:
        """Verifies that a positive requested count reaches the stage unchanged."""
        resolved = resolve_stage_workers(job_name=SingleRecordingJobNames.REGISTER, requested_workers=requested_workers)
        assert resolved == requested_workers

    def test_all_cores_request_resolves_through_the_worker_resolver(self) -> None:
        """Verifies that requesting every available core defers to the ataraxis worker resolver."""
        resolved = resolve_stage_workers(job_name=SingleRecordingJobNames.PROCESS, requested_workers=ALL_CORES_REQUEST)
        assert resolved == resolve_worker_count(requested_workers=ALL_CORES_REQUEST)
        assert resolved >= 1

    def test_all_cores_request_is_honored_for_every_allocating_stage(self) -> None:
        """Verifies that every single-recording stage taking an allocation accepts the all-cores request."""
        expected = resolve_worker_count(requested_workers=ALL_CORES_REQUEST)
        for job_name in (
            SingleRecordingJobNames.BINARIZE,
            SingleRecordingJobNames.REGISTER,
            SingleRecordingJobNames.PROCESS,
        ):
            assert resolve_stage_workers(job_name=job_name, requested_workers=ALL_CORES_REQUEST) == expected


class TestRejectedRequests:
    """Tests the requests the resolver rejects."""

    def test_combination_stage_is_rejected(self) -> None:
        """Verifies that the combination stage, which takes no allocation, raises rather than returning a count."""
        with pytest.raises(ValueError, match=r"does not name a\s+pipeline\s+stage"):
            resolve_stage_workers(job_name=SingleRecordingJobNames.COMBINE)

    def test_combination_stage_is_rejected_before_the_requested_count_is_read(self) -> None:
        """Verifies that the stage check precedes the requested-count check, so a valid count cannot mask it."""
        with pytest.raises(ValueError, match=r"does not name a\s+pipeline\s+stage"):
            resolve_stage_workers(job_name=SingleRecordingJobNames.COMBINE, requested_workers=8)

    @pytest.mark.parametrize("requested_workers", [0, -2, -3, -100])
    def test_invalid_worker_count_is_rejected(self, requested_workers: int) -> None:
        """Verifies that zero and every negative count other than the all-cores request raise an error."""
        with pytest.raises(ValueError, match=r"must be a positive\s+integer"):
            resolve_stage_workers(job_name=SingleRecordingJobNames.PROCESS, requested_workers=requested_workers)


class TestResolveCoreBudget:
    """Tests the CPU core budget one execution session may commit."""

    def test_budget_matches_the_worker_resolver_with_the_reserved_host_cores(self) -> None:
        """Verifies that the budget is every available core minus the cores held back for the host system."""
        assert resolve_core_budget() == resolve_worker_count(
            requested_workers=ALL_CORES_REQUEST, reserved_cores=_RESERVED_CORES
        )

    def test_budget_is_a_usable_positive_core_count(self) -> None:
        """Verifies that the resolved budget always admits at least one job."""
        assert resolve_core_budget() >= 1
        assert _RESERVED_CORES > 0


class TestResourceClasses:
    """Tests the resource classes that size a batch of jobs and the job name map that selects them."""

    @pytest.mark.parametrize(
        ("resource_class", "name", "workers_per_job", "fixed_parallel_jobs", "memory_gigabytes_per_job"),
        [
            (_BINARIZATION_RESOURCES, "binarization", BINARIZATION_WORKERS, _MAXIMUM_PARALLEL_IO_JOBS, 0.0),
            (_REGISTRATION_RESOURCES, "registration", REGISTRATION_WORKERS, None, 0.0),
            (_PROCESSING_RESOURCES, "processing", PROCESSING_WORKERS, None, _PROCESSING_MEMORY_GIGABYTES_PER_JOB),
            (_COMBINATION_RESOURCES, "combination", COMBINATION_WORKERS, _MAXIMUM_PARALLEL_IO_JOBS, 0.0),
            (_DISCOVERY_RESOURCES, "discovery", DISCOVERY_WORKERS, None, 0.0),
            (_EXTRACTION_RESOURCES, "extraction", EXTRACTION_WORKERS, None, 0.0),
        ],
    )
    def test_class_fields_carry_the_measured_stage_budget(
        self,
        resource_class: ResourceClass,
        name: str,
        workers_per_job: int,
        fixed_parallel_jobs: int | None,
        memory_gigabytes_per_job: float,
    ) -> None:
        """Verifies that every exported resource class declares its measured worker, concurrency, and memory budget."""
        assert resource_class.name == name
        assert resource_class.workers_per_job == workers_per_job
        assert resource_class.fixed_parallel_jobs == fixed_parallel_jobs
        assert resource_class.memory_gigabytes_per_job == memory_gigabytes_per_job

    @pytest.mark.parametrize(
        ("job_name", "expected_class"),
        [
            (SingleRecordingJobNames.BINARIZE, _BINARIZATION_RESOURCES),
            (SingleRecordingJobNames.REGISTER, _REGISTRATION_RESOURCES),
            (SingleRecordingJobNames.PROCESS, _PROCESSING_RESOURCES),
            (SingleRecordingJobNames.COMBINE, _COMBINATION_RESOURCES),
            (MultiRecordingJobNames.DISCOVER, _DISCOVERY_RESOURCES),
            (MultiRecordingJobNames.EXTRACT, _EXTRACTION_RESOURCES),
        ],
    )
    def test_job_name_selects_its_resource_class(
        self, job_name: SingleRecordingJobNames | MultiRecordingJobNames, expected_class: ResourceClass
    ) -> None:
        """Verifies that every pipeline job name resolves to the resource class that governs its budget."""
        assert RESOURCE_CLASS_BY_JOB_NAME[job_name] is expected_class

    def test_map_covers_every_declared_job_name(self) -> None:
        """Verifies that the map leaves no single or multi-recording job name without a resource class."""
        assert set(RESOURCE_CLASS_BY_JOB_NAME) == set(SingleRecordingJobNames) | set(MultiRecordingJobNames)

    def test_only_the_processing_class_bounds_its_concurrency_by_memory(self) -> None:
        """Verifies that the processing class is the sole class declaring a per-job memory footprint."""
        memory_bound = {
            resource_class.name
            for resource_class in RESOURCE_CLASS_BY_JOB_NAME.values()
            if resource_class.memory_gigabytes_per_job > 0
        }
        assert memory_bound == {"processing"}


class TestFixedCapacityAllocation:
    """Tests the allocation of the resource classes that carry a machine-independent concurrency cap."""

    @pytest.mark.parametrize(
        ("resource_class", "job_count", "expected"),
        [
            (_BINARIZATION_RESOURCES, 10, (BINARIZATION_WORKERS, _MAXIMUM_PARALLEL_IO_JOBS)),
            (_BINARIZATION_RESOURCES, 2, (BINARIZATION_WORKERS, 2)),
            (_BINARIZATION_RESOURCES, 0, (BINARIZATION_WORKERS, 1)),
            (_COMBINATION_RESOURCES, 7, (COMBINATION_WORKERS, _MAXIMUM_PARALLEL_IO_JOBS)),
            (_COMBINATION_RESOURCES, 1, (COMBINATION_WORKERS, 1)),
        ],
    )
    def test_fixed_cap_class_bounds_its_concurrency_by_the_job_count(
        self, resource_class: ResourceClass, job_count: int, expected: tuple[int, int]
    ) -> None:
        """Verifies that a fixed-cap class holds its measured workers and caps concurrency at the smaller bound."""
        allocation = resolve_class_allocation(
            resource_class=resource_class,
            budget=64,
            available_memory=256.0,
            job_count=job_count,
            workers_per_job=None,
            max_parallel_jobs=None,
        )

        assert allocation == expected

    @pytest.mark.parametrize(
        ("workers_per_job", "max_parallel_jobs"),
        [(64, 32), (ALL_CORES_REQUEST, ALL_CORES_REQUEST), (1, 1), (7, None), (None, 3)],
    )
    def test_fixed_cap_class_ignores_both_overrides(
        self, workers_per_job: int | None, max_parallel_jobs: int | None
    ) -> None:
        """Verifies that neither the worker nor the concurrency override reaches a fixed-cap class."""
        allocation = resolve_class_allocation(
            resource_class=_BINARIZATION_RESOURCES,
            budget=64,
            available_memory=64.0,
            job_count=8,
            workers_per_job=workers_per_job,
            max_parallel_jobs=max_parallel_jobs,
        )

        assert allocation == (BINARIZATION_WORKERS, _MAXIMUM_PARALLEL_IO_JOBS)


class TestDerivedCapacityAllocation:
    """Tests the allocation of the resource classes whose concurrency derives from the session CPU budget."""

    @pytest.mark.parametrize(
        ("budget", "job_count", "workers_per_job", "max_parallel_jobs", "expected"),
        [
            (24, 10, None, None, (REGISTRATION_WORKERS, 2)),
            (24, 1, None, None, (REGISTRATION_WORKERS, 1)),
            (24, 0, None, None, (REGISTRATION_WORKERS, 1)),
            (24, 5, ALL_CORES_REQUEST, None, (24, 1)),
            (48, 5, ALL_CORES_REQUEST, None, (48, 1)),
            (24, 10, 6, None, (6, 4)),
            (8, 4, 12, None, (12, 1)),
            (24, 10, 6, ALL_CORES_REQUEST, (6, 10)),
            (24, 0, None, ALL_CORES_REQUEST, (REGISTRATION_WORKERS, 1)),
            (24, 10, None, 3, (REGISTRATION_WORKERS, 3)),
            (24, 2, None, 9, (REGISTRATION_WORKERS, 9)),
        ],
    )
    def test_allocation_follows_the_budget_the_overrides_and_the_job_count(
        self,
        budget: int,
        job_count: int,
        workers_per_job: int | None,
        max_parallel_jobs: int | None,
        expected: tuple[int, int],
    ) -> None:
        """Verifies the worker and concurrency pair a derived-cap class resolves for every override combination."""
        allocation = resolve_class_allocation(
            resource_class=_REGISTRATION_RESOURCES,
            budget=budget,
            available_memory=64.0,
            job_count=job_count,
            workers_per_job=workers_per_job,
            max_parallel_jobs=max_parallel_jobs,
        )

        assert allocation == expected

    @pytest.mark.parametrize("available_memory", [0.0, 0.5, 4096.0])
    def test_class_without_a_memory_footprint_ignores_the_available_memory(self, available_memory: float) -> None:
        """Verifies that a class declaring no per-job memory keeps its CPU-derived cap whatever memory is free."""
        allocation = resolve_class_allocation(
            resource_class=_DISCOVERY_RESOURCES,
            budget=DISCOVERY_WORKERS * 3,
            available_memory=available_memory,
            job_count=8,
            workers_per_job=None,
            max_parallel_jobs=None,
        )

        assert allocation == (DISCOVERY_WORKERS, 3)

    def test_extraction_class_derives_its_capacity_from_the_budget(self) -> None:
        """Verifies that the multi-recording extraction class splits the budget across its measured worker count."""
        allocation = resolve_class_allocation(
            resource_class=_EXTRACTION_RESOURCES,
            budget=EXTRACTION_WORKERS * 2,
            available_memory=1024.0,
            job_count=6,
            workers_per_job=None,
            max_parallel_jobs=None,
        )

        assert allocation == (EXTRACTION_WORKERS, 2)


class TestMemoryBoundAllocation:
    """Tests the allocation of the processing class, whose concurrency additionally follows the available memory."""

    @pytest.mark.parametrize(
        ("budget", "available_memory", "job_count", "expected"),
        [
            (100, 100.0, 20, (PROCESSING_WORKERS, 6)),
            (100, 5.0, 20, (PROCESSING_WORKERS, 1)),
            (100, 0.0, 20, (PROCESSING_WORKERS, 1)),
            (100, 4096.0, 20, (PROCESSING_WORKERS, 10)),
            (100, 4096.0, 3, (PROCESSING_WORKERS, 3)),
        ],
    )
    def test_capacity_takes_the_smallest_of_the_cpu_memory_and_job_bounds(
        self, budget: int, available_memory: float, job_count: int, expected: tuple[int, int]
    ) -> None:
        """Verifies that the memory-bound class caps its concurrency at the tightest of its three bounds."""
        allocation = resolve_class_allocation(
            resource_class=_PROCESSING_RESOURCES,
            budget=budget,
            available_memory=available_memory,
            job_count=job_count,
            workers_per_job=None,
            max_parallel_jobs=None,
        )

        assert allocation == expected

    def test_explicit_concurrency_override_bypasses_the_memory_bound(self) -> None:
        """Verifies that an explicit concurrency cap is honored even when the memory bound would be tighter."""
        allocation = resolve_class_allocation(
            resource_class=_PROCESSING_RESOURCES,
            budget=100,
            available_memory=1.0,
            job_count=20,
            workers_per_job=None,
            max_parallel_jobs=8,
        )

        assert allocation == (PROCESSING_WORKERS, 8)


@pytest.mark.xdist_group(name="allocation_system_probes")
class TestResolveAvailableMemory:
    """Tests the memory probe that sizes the memory-bound resource classes."""

    @pytest.mark.parametrize("available_bytes", [0, 4096, 8 * _BYTES_PER_GIGABYTE, 1536 * _BYTES_PER_GIGABYTE])
    def test_probe_converts_the_available_byte_counter_to_gigabytes(self, monkeypatch, available_bytes: int) -> None:
        """Verifies that the probe divides the host's available-byte counter into a gigabyte figure."""
        monkeypatch.setattr(
            "cindra.orchestration.allocation.psutil.virtual_memory",
            lambda: _VirtualMemory(available=available_bytes),
        )

        assert resolve_available_memory_gigabytes() == available_bytes / _BYTES_PER_GIGABYTE

    def test_host_probe_reports_a_non_negative_figure(self) -> None:
        """Verifies that the unpatched probe reports a usable memory figure on the test host."""
        assert resolve_available_memory_gigabytes() >= 0


class TestSummarizeClassAllocation:
    """Tests the per-class allocation report an execution session publishes."""

    def test_report_carries_the_workers_capacity_and_job_count_of_every_class(self) -> None:
        """Verifies that the report pairs each class name with its worker count, concurrency cap, and job count."""
        summary = summarize_class_allocation(
            class_workers={"registration": 12, "processing": 10},
            class_capacities={"registration": 2, "processing": 6},
            class_job_counts={"registration": 4, "processing": 9},
        )

        assert summary == {
            "registration": {"workers_per_job": 12, "max_parallel_jobs": 2, "job_count": 4},
            "processing": {"workers_per_job": 10, "max_parallel_jobs": 6, "job_count": 9},
        }

    def test_report_covers_only_the_classes_that_carry_jobs(self) -> None:
        """Verifies that the job count mapping selects the classes the report describes."""
        summary = summarize_class_allocation(
            class_workers={"registration": 12, "combination": 1},
            class_capacities={"registration": 2, "combination": 4},
            class_job_counts={"registration": 4},
        )

        assert summary == {"registration": {"workers_per_job": 12, "max_parallel_jobs": 2, "job_count": 4}}

    def test_session_without_jobs_reports_no_classes(self) -> None:
        """Verifies that a session carrying no jobs publishes an empty report."""
        assert summarize_class_allocation(class_workers={}, class_capacities={}, class_job_counts={}) == {}
