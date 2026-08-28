"""Contains tests for the per-stage worker defaults, the resource-class model, and the allocation resolvers."""

import pytest
from ataraxis_base_utilities import error_format, resolve_worker_count

from cindra.orchestration import (
    DISCOVERY_WORKERS,
    EXTRACTION_WORKERS,
    PROCESSING_WORKERS,
    BINARIZATION_WORKERS,
    REGISTRATION_WORKERS,
    REGISTRATION_GPU_WORKERS,
    RESOURCE_CLASS_BY_JOB_NAME,
    ResourceClass,
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    resolve_stage_workers,
    resolve_registration_resource_class,
)
from cindra.orchestration.gpu import resolve_device_budget
from cindra.orchestration.allocation import (
    _RESERVED_CORES,
    ALL_CORES_REQUEST,
    COMBINATION_WORKERS,
    _BYTES_PER_MEGABYTE,
    _DISCOVERY_RESOURCES,
    _EXTRACTION_RESOURCES,
    _PROCESSING_RESOURCES,
    _COMBINATION_RESOURCES,
    _STAGE_WORKER_DEFAULTS,
    _BINARIZATION_RESOURCES,
    _REGISTRATION_RESOURCES,
    DISCOVERY_MAXIMUM_WORKERS,
    EXTRACTION_MAXIMUM_WORKERS,
    PROCESSING_MAXIMUM_WORKERS,
    REGISTRATION_MAXIMUM_WORKERS,
    _BINARIZATION_CONCURRENCY_LIMIT,
    _PROCESSING_CONCURRENCY_RESERVATION,
    _REGISTRATION_CONCURRENCY_RESERVATION,
    resolve_core_budget,
    class_requires_device,
    resolve_class_allocation,
    resolve_dispatch_workers,
    resolve_memory_budget_mb,
    summarize_class_allocation,
    _resolve_registration_gpu_resources,
)


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


class TestDevicePlannedDefaults:
    """Tests the stage default the registration resolves while its job is planned for a CUDA device."""

    def test_registration_takes_its_device_default(self) -> None:
        """Verifies that a registration planned for a device resolves the host-side count that stage occupies."""
        resolved = resolve_stage_workers(job_name=SingleRecordingJobNames.REGISTER, gpu_registration=True)

        assert resolved == REGISTRATION_GPU_WORKERS
        assert resolved != resolve_stage_workers(job_name=SingleRecordingJobNames.REGISTER)

    @pytest.mark.parametrize(
        ("job_name", "expected_workers"),
        [
            (SingleRecordingJobNames.BINARIZE, BINARIZATION_WORKERS),
            (SingleRecordingJobNames.PROCESS, PROCESSING_WORKERS),
            (SingleRecordingJobNames.COMBINE, COMBINATION_WORKERS),
            (MultiRecordingJobNames.DISCOVER, DISCOVERY_WORKERS),
            (MultiRecordingJobNames.EXTRACT, EXTRACTION_WORKERS),
        ],
    )
    def test_every_other_stage_ignores_the_device_plan(
        self, job_name: SingleRecordingJobNames | MultiRecordingJobNames, expected_workers: int
    ) -> None:
        """Verifies that the registration stage alone responds to the flag that plans a job for a device."""
        assert resolve_stage_workers(job_name=job_name, gpu_registration=True) == expected_workers

    def test_explicit_request_still_overrides_the_device_default(self) -> None:
        """Verifies that the sentinel contract holds unchanged while the registration is planned for a device."""
        resolved = resolve_stage_workers(
            job_name=SingleRecordingJobNames.REGISTER, requested_workers=9, gpu_registration=True
        )

        assert resolved == 9

    def test_unknown_stage_is_rejected_before_the_device_default_applies(self) -> None:
        """Verifies that a name the stage map does not hold raises rather than reaching the device default."""
        expected_message = (
            "Unable to resolve the worker count for the 'recording_denoise' processing stage. The input job name "
            "does not name a pipeline stage. Use one of the valid stage names: "
            f"{[stage.value for stage in _STAGE_WORKER_DEFAULTS]}."
        )
        with pytest.raises(ValueError, match=error_format(expected_message)):
            resolve_stage_workers(job_name="recording_denoise", gpu_registration=True)  # type: ignore[arg-type]


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


class TestCombinationStage:
    """Tests the allocation of the stage whose merge is serial."""

    def test_combination_stage_resolves_to_its_single_core(self) -> None:
        """Verifies that the combination stage resolves rather than forcing the caller to special-case it."""
        assert resolve_stage_workers(job_name=SingleRecordingJobNames.COMBINE) == COMBINATION_WORKERS

    def test_combination_stage_honors_an_explicit_count(self) -> None:
        """Verifies that the combination stage follows the shared sentinel contract like every other stage."""
        assert resolve_stage_workers(job_name=SingleRecordingJobNames.COMBINE, requested_workers=4) == 4


class TestRejectedRequests:
    """Tests the requests the resolver rejects."""

    def test_unknown_stage_is_rejected(self) -> None:
        """Verifies that a name that is not a pipeline stage raises rather than returning a count."""
        expected_message = (
            "Unable to resolve the worker count for the 'recording_denoise' processing stage. The input job name "
            "does not name a pipeline stage. Use one of the valid stage names: "
            f"{[stage.value for stage in _STAGE_WORKER_DEFAULTS]}."
        )
        with pytest.raises(ValueError, match=error_format(expected_message)):
            resolve_stage_workers(job_name="recording_denoise")  # type: ignore[arg-type]

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
        ("resource_class", "name", "workers_per_job", "concurrency_limit", "concurrency_reservation"),
        [
            (_BINARIZATION_RESOURCES, "binarization", BINARIZATION_WORKERS, _BINARIZATION_CONCURRENCY_LIMIT, None),
            (
                _REGISTRATION_RESOURCES,
                "registration",
                REGISTRATION_WORKERS,
                None,
                _REGISTRATION_CONCURRENCY_RESERVATION,
            ),
            (_PROCESSING_RESOURCES, "processing", PROCESSING_WORKERS, None, _PROCESSING_CONCURRENCY_RESERVATION),
            (_COMBINATION_RESOURCES, "combination", COMBINATION_WORKERS, None, None),
            (_DISCOVERY_RESOURCES, "discovery", DISCOVERY_WORKERS, None, None),
            (_EXTRACTION_RESOURCES, "extraction", EXTRACTION_WORKERS, None, None),
        ],
    )
    def test_class_fields_carry_the_measured_stage_budget(
        self,
        resource_class: ResourceClass,
        name: str,
        workers_per_job: int,
        concurrency_limit: int | None,
        concurrency_reservation: int | None,
    ) -> None:
        """Verifies that every resource class declares its measured worker count, its ceiling, and its reservation."""
        assert resource_class.name == name
        assert resource_class.workers_per_job == workers_per_job
        assert resource_class.concurrency_limit == concurrency_limit
        assert resource_class.concurrency_reservation == concurrency_reservation

    def test_only_the_conversion_stage_carries_a_hard_ceiling(self) -> None:
        """Verifies that the disk-bound conversion stage is the only class spare capacity cannot widen."""
        limited = {
            resource_class.name
            for resource_class in RESOURCE_CLASS_BY_JOB_NAME.values()
            if resource_class.concurrency_limit is not None
        }

        assert limited == {"binarization"}

    def test_only_the_compute_stages_carry_a_reservation(self) -> None:
        """Verifies that the two stages competing for the scarcest cores are the ones holding capacity back."""
        reserved = {
            resource_class.name
            for resource_class in RESOURCE_CLASS_BY_JOB_NAME.values()
            if resource_class.concurrency_reservation is not None
        }

        assert reserved == {"registration", "processing"}

    @pytest.mark.parametrize(
        ("resource_class", "maximum_workers_per_job"),
        [
            (_BINARIZATION_RESOURCES, None),
            (_REGISTRATION_RESOURCES, REGISTRATION_MAXIMUM_WORKERS),
            (_PROCESSING_RESOURCES, PROCESSING_MAXIMUM_WORKERS),
            (_COMBINATION_RESOURCES, None),
            (_DISCOVERY_RESOURCES, DISCOVERY_MAXIMUM_WORKERS),
            (_EXTRACTION_RESOURCES, EXTRACTION_MAXIMUM_WORKERS),
        ],
    )
    def test_class_carries_the_ceiling_its_stage_stops_gaining_at(
        self, resource_class: ResourceClass, maximum_workers_per_job: int | None
    ) -> None:
        """Verifies that every resource class declares the widest allocation its stage converts into wall clock."""
        assert resource_class.maximum_workers_per_job == maximum_workers_per_job

    def test_only_the_storage_bound_and_serial_stages_are_inelastic(self) -> None:
        """Verifies that the conversion and the merge are the classes a host with spare cores cannot widen."""
        inelastic = {
            resource_class.name
            for resource_class in RESOURCE_CLASS_BY_JOB_NAME.values()
            if resource_class.maximum_workers_per_job is None
        }

        assert inelastic == {"binarization", "combination"}

    def test_every_ceiling_holds_at_or_above_the_class_default(self) -> None:
        """Verifies that widening a job never narrows it below the allocation its class declares."""
        ceilings = [
            (resource_class.maximum_workers_per_job, resource_class.workers_per_job)
            for resource_class in RESOURCE_CLASS_BY_JOB_NAME.values()
            if resource_class.maximum_workers_per_job is not None
        ]

        assert ceilings
        assert all(ceiling >= default for ceiling, default in ceilings)

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

    def test_no_class_declares_a_memory_footprint(self) -> None:
        """Verifies that memory bounds admission per job rather than concurrency per class."""
        assert not any(
            hasattr(resource_class, "memory_gigabytes_per_job")
            for resource_class in RESOURCE_CLASS_BY_JOB_NAME.values()
        )


class TestDeviceResourceClass:
    """Tests the resource class of the registration jobs that run on a CUDA device."""

    def test_class_carries_the_device_budget_as_its_concurrency_limit(self) -> None:
        """Verifies that the devices the host exposes are what bound how many device-backed jobs run at once."""
        assert _resolve_registration_gpu_resources().concurrency_limit == resolve_device_budget()

    def test_class_declares_its_host_side_worker_count(self) -> None:
        """Verifies that a device-backed job still holds the cores its host-side work occupies."""
        assert _resolve_registration_gpu_resources().name == "registration_gpu"
        assert _resolve_registration_gpu_resources().workers_per_job == REGISTRATION_GPU_WORKERS
        assert _resolve_registration_gpu_resources().maximum_workers_per_job is None
        assert _resolve_registration_gpu_resources().concurrency_reservation is None

    @pytest.mark.parametrize(
        ("gpu_registration", "expected_name"), [(True, "registration_gpu"), (False, "registration")]
    )
    def test_device_plan_selects_the_registration_class(self, gpu_registration: bool, expected_name: str) -> None:
        """Verifies that planning the registration for a device selects the class its jobs run under."""
        assert resolve_registration_resource_class(gpu_registration=gpu_registration).name == expected_name

    def test_only_the_device_class_holds_a_device(self) -> None:
        """Verifies that the device predicate separates the device-backed class from every host-only class."""
        assert class_requires_device(resource_class=_resolve_registration_gpu_resources())
        assert not any(
            class_requires_device(resource_class=resource_class)
            for resource_class in RESOURCE_CLASS_BY_JOB_NAME.values()
        )

    def test_job_name_map_holds_the_host_class(self) -> None:
        """Verifies that the name map reports the host class, leaving the device class to the backend resolver."""
        assert RESOURCE_CLASS_BY_JOB_NAME[SingleRecordingJobNames.REGISTER] is _REGISTRATION_RESOURCES

    @pytest.mark.parametrize("job_count", [1, 4])
    def test_device_class_caps_its_concurrency_at_the_device_count(self, job_count: int) -> None:
        """Verifies that a queue deeper than the device count dispatches no more jobs than there are devices."""
        _, capacity = resolve_class_allocation(
            resource_class=_resolve_registration_gpu_resources(),
            budget=64,
            job_count=job_count,
            workers_per_job=None,
            max_parallel_jobs=None,
        )

        assert capacity == max(1, min(resolve_device_budget(), job_count))

    def test_absent_resource_still_resolves_one_dispatch_slot(self) -> None:
        """Verifies that a class whose counted resource is absent dispatches one job rather than holding its queue."""
        absent = ResourceClass(
            name="absent",
            workers_per_job=2,
            maximum_workers_per_job=None,
            concurrency_limit=0,
            concurrency_reservation=None,
        )

        allocation = resolve_class_allocation(
            resource_class=absent, budget=64, job_count=4, workers_per_job=None, max_parallel_jobs=None
        )

        assert allocation == (2, 1)


class TestFixedCapacityAllocation:
    """Tests the allocation of the resource classes that carry a machine-independent concurrency cap."""

    @pytest.mark.parametrize(
        ("resource_class", "job_count", "expected"),
        [
            (_BINARIZATION_RESOURCES, 10, (BINARIZATION_WORKERS, _BINARIZATION_CONCURRENCY_LIMIT)),
            (_BINARIZATION_RESOURCES, 2, (BINARIZATION_WORKERS, 2)),
            (_BINARIZATION_RESOURCES, 0, (BINARIZATION_WORKERS, 1)),
            (_BINARIZATION_RESOURCES, 3, (BINARIZATION_WORKERS, 3)),
        ],
    )
    def test_fixed_cap_class_bounds_its_concurrency_by_the_job_count(
        self, resource_class: ResourceClass, job_count: int, expected: tuple[int, int]
    ) -> None:
        """Verifies that a fixed-cap class holds its measured workers and caps concurrency at the smaller bound."""
        allocation = resolve_class_allocation(
            resource_class=resource_class,
            budget=64,
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
            job_count=8,
            workers_per_job=workers_per_job,
            max_parallel_jobs=max_parallel_jobs,
        )

        assert allocation == (BINARIZATION_WORKERS, _BINARIZATION_CONCURRENCY_LIMIT)


class TestDerivedCapacityAllocation:
    """Tests the allocation of the resource classes whose concurrency derives from the session CPU budget."""

    @pytest.mark.parametrize(
        ("budget", "job_count", "workers_per_job", "max_parallel_jobs", "expected"),
        [
            (24, 10, None, None, (REGISTRATION_WORKERS, 6)),
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
            job_count=job_count,
            workers_per_job=workers_per_job,
            max_parallel_jobs=max_parallel_jobs,
        )

        assert allocation == expected

    def test_discovery_class_derives_its_capacity_from_the_budget(self) -> None:
        """Verifies that the discovery class splits the budget across its measured worker count."""
        allocation = resolve_class_allocation(
            resource_class=_DISCOVERY_RESOURCES,
            budget=DISCOVERY_WORKERS * 3,
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
            job_count=6,
            workers_per_job=None,
            max_parallel_jobs=None,
        )

        assert allocation == (EXTRACTION_WORKERS, 2)


class TestDispatchWorkers:
    """Tests the width one job takes when the dispatcher submits it against the capacity the host holds free."""

    @pytest.mark.parametrize("resource_class", [_BINARIZATION_RESOURCES, _COMBINATION_RESOURCES])
    def test_class_without_a_ceiling_keeps_its_default_width(self, resource_class: ResourceClass) -> None:
        """Verifies that a class carrying no ceiling holds its default allocation however much of the host sits free."""
        resolved = resolve_dispatch_workers(
            resource_class=resource_class, free_cores=126, pending_jobs=1, running_jobs=0, concurrency_cap=8
        )

        assert resolved == resource_class.workers_per_job

    @pytest.mark.parametrize("free_cores", [16, 32, 64])
    def test_saturated_queue_resolves_to_the_class_default(self, free_cores: int) -> None:
        """Verifies that a queue holding a job for every free core reproduces the allocation the class declares."""
        resolved = resolve_dispatch_workers(
            resource_class=_REGISTRATION_RESOURCES,
            free_cores=free_cores,
            pending_jobs=16,
            running_jobs=0,
            concurrency_cap=16,
        )

        assert resolved == REGISTRATION_WORKERS

    def test_nearly_empty_queue_resolves_to_the_class_ceiling(self) -> None:
        """Verifies that the last job of a class takes the free budget up to the width its stage still spends."""
        resolved = resolve_dispatch_workers(
            resource_class=_REGISTRATION_RESOURCES, free_cores=126, pending_jobs=1, running_jobs=0, concurrency_cap=16
        )

        assert resolved == REGISTRATION_MAXIMUM_WORKERS

    def test_discovery_queue_resolves_to_its_own_narrower_ceiling(self) -> None:
        """Verifies that each class stops at the ceiling it declares rather than at one shared across classes."""
        resolved = resolve_dispatch_workers(
            resource_class=_DISCOVERY_RESOURCES, free_cores=126, pending_jobs=1, running_jobs=0, concurrency_cap=8
        )

        assert resolved == DISCOVERY_MAXIMUM_WORKERS

    def test_draining_queue_widens_the_jobs_it_has_left(self) -> None:
        """Verifies that the width climbs from the class default to the class ceiling as the queue empties."""
        widths = [
            resolve_dispatch_workers(
                resource_class=_REGISTRATION_RESOURCES,
                free_cores=64,
                pending_jobs=pending_jobs,
                running_jobs=0,
                concurrency_cap=16,
            )
            for pending_jobs in (16, 8, 4, 2, 1)
        ]

        assert widths == [REGISTRATION_WORKERS, 8, 16, REGISTRATION_MAXIMUM_WORKERS, REGISTRATION_MAXIMUM_WORKERS]

    def test_running_peers_bound_the_jobs_that_still_start_alongside_the_dispatched_one(self) -> None:
        """Verifies that the share divides the free cores by the concurrency the class has left rather than its cap."""
        idle_class = resolve_dispatch_workers(
            resource_class=_REGISTRATION_RESOURCES, free_cores=32, pending_jobs=8, running_jobs=0, concurrency_cap=8
        )
        loaded_class = resolve_dispatch_workers(
            resource_class=_REGISTRATION_RESOURCES, free_cores=32, pending_jobs=8, running_jobs=6, concurrency_cap=8
        )

        assert idle_class == REGISTRATION_WORKERS
        assert loaded_class == 16

    def test_saturated_class_still_resolves_a_usable_width(self) -> None:
        """Verifies that a class already holding its whole cap divides the free cores by one job rather than by zero."""
        resolved = resolve_dispatch_workers(
            resource_class=_REGISTRATION_RESOURCES, free_cores=2, pending_jobs=4, running_jobs=8, concurrency_cap=8
        )

        assert resolved == REGISTRATION_WORKERS

    @pytest.mark.parametrize(("resource_class", "expected_workers"), [(_PROCESSING_RESOURCES, PROCESSING_WORKERS)])
    def test_class_whose_ceiling_meets_its_default_never_widens(
        self, resource_class: ResourceClass, expected_workers: int
    ) -> None:
        """Verifies that a class whose ceiling meets its default holds that width on an idle host."""
        resolved = resolve_dispatch_workers(
            resource_class=resource_class, free_cores=126, pending_jobs=1, running_jobs=0, concurrency_cap=12
        )

        assert resolved == expected_workers


@pytest.mark.xdist_group(name="allocation_system_probes")
class TestResolveMemoryBudget:
    """Tests the memory probe that bounds how much a session may commit to running jobs."""

    @pytest.mark.parametrize("available_bytes", [0, 4096, 8 * _BYTES_PER_MEGABYTE, 1536 * _BYTES_PER_MEGABYTE])
    def test_probe_converts_the_available_byte_counter_to_megabytes(self, monkeypatch, available_bytes: int) -> None:
        """Verifies that the probe reports the host counter on the scale the per-job estimates use."""
        monkeypatch.setattr(
            "cindra.orchestration.allocation.psutil.virtual_memory",
            lambda: _VirtualMemory(available=available_bytes),
        )

        assert resolve_memory_budget_mb() == int(available_bytes / _BYTES_PER_MEGABYTE)

    def test_host_probe_reports_a_non_negative_figure(self) -> None:
        """Verifies that the unpatched probe reports a usable memory figure on the test host."""
        assert resolve_memory_budget_mb() >= 0


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


class _VirtualMemory:
    """Stands in for the psutil memory record, carrying only the available-byte counter the resolver reads."""

    def __init__(self, available: int) -> None:
        self.available = available


class TestExtractionCeiling:
    """Tests the widening the multi-recording extraction class performs."""

    def test_extraction_widens_toward_its_ceiling(self) -> None:
        """Verifies that an extraction job with the host to itself widens past the stage default."""
        resolved = resolve_dispatch_workers(
            resource_class=_EXTRACTION_RESOURCES,
            free_cores=126,
            pending_jobs=1,
            running_jobs=0,
            concurrency_cap=8,
        )

        assert resolved == EXTRACTION_MAXIMUM_WORKERS

    def test_extraction_holds_its_default_while_the_queue_is_full(self) -> None:
        """Verifies that a saturated extraction queue resolves the width the stage default names."""
        resolved = resolve_dispatch_workers(
            resource_class=_EXTRACTION_RESOURCES,
            free_cores=126,
            pending_jobs=32,
            running_jobs=0,
            concurrency_cap=32,
        )

        assert resolved == EXTRACTION_WORKERS


class TestCompetingClasses:
    """Tests the division of the free cores among the elastic classes holding queued work."""

    def test_one_competing_class_takes_the_whole_budget(self) -> None:
        """Verifies that a lone elastic class divides the free cores among its own jobs alone."""
        resolved = resolve_dispatch_workers(
            resource_class=_REGISTRATION_RESOURCES,
            free_cores=126,
            pending_jobs=3,
            running_jobs=0,
            concurrency_cap=8,
            competing_classes=1,
        )

        assert resolved == REGISTRATION_MAXIMUM_WORKERS

    def test_two_competing_classes_split_the_budget(self) -> None:
        """Verifies that a second elastic class holding work halves the share the first one divides."""
        resolved = resolve_dispatch_workers(
            resource_class=_REGISTRATION_RESOURCES,
            free_cores=126,
            pending_jobs=4,
            running_jobs=0,
            concurrency_cap=8,
            competing_classes=2,
        )

        assert resolved == 15

    def test_competition_never_pushes_a_job_below_its_class_default(self) -> None:
        """Verifies that dividing the budget among many classes still leaves each job its class default."""
        resolved = resolve_dispatch_workers(
            resource_class=_REGISTRATION_RESOURCES,
            free_cores=16,
            pending_jobs=8,
            running_jobs=0,
            concurrency_cap=8,
            competing_classes=4,
        )

        assert resolved == REGISTRATION_WORKERS
