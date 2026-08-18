"""Contains tests for the pipeline phase model, the job universe resolvers, and the prerequisite graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from ataraxis_data_structures import JobState, ProcessingStatus, ProcessingTracker

from cindra.layout import PLANE_SPECIFIER_PREFIX
from cindra.orchestration import (
    MULTI_RECORDING_PHASES,
    SINGLE_RECORDING_PHASES,
    MULTI_RECORDING_TRACKER_FILENAME,
    SINGLE_RECORDING_TRACKER_FILENAME,
    PrerequisiteScope,
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    generate_job_ids,
    resolve_pipeline_jobs,
    resolve_plane_specifier,
    order_phases_by_execution,
    resolve_downstream_phases,
    validate_job_prerequisites,
    resolve_multi_recording_jobs,
    resolve_single_recording_jobs,
    resolve_multi_recording_prerequisites,
    resolve_single_recording_prerequisites,
)
from cindra.orchestration.jobs import (
    PER_PLANE_JOB_NAMES,
    PREREQUISITE_FAILURE_MESSAGE,
    UNREACHABLE_PREREQUISITE_MESSAGE,
    _collect_phase_job_ids,
    _resolve_prerequisites,
    resolve_prerequisite_job_ids,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Sequence


class TestJobModelConstants:
    """Tests the module-level constants the trackers and the execution engine consume."""

    def test_per_plane_job_names_hold_the_per_specifier_single_recording_phases(self) -> None:
        """Verifies that the per-plane name set holds the registration and processing phases and nothing else."""
        assert frozenset({SingleRecordingJobNames.REGISTER, SingleRecordingJobNames.PROCESS}) == PER_PLANE_JOB_NAMES
        assert SingleRecordingJobNames.BINARIZE not in PER_PLANE_JOB_NAMES
        assert SingleRecordingJobNames.COMBINE not in PER_PLANE_JOB_NAMES

    def test_tracker_file_names_identify_the_pipeline_they_track(self) -> None:
        """Verifies that each pipeline's tracker file name holds its expected .yaml document name."""
        assert SINGLE_RECORDING_TRACKER_FILENAME == "single_recording_tracker.yaml"
        assert MULTI_RECORDING_TRACKER_FILENAME == "multi_recording_tracker.yaml"

    def test_tracker_error_messages_hold_their_expected_text(self) -> None:
        """Verifies that both tracker error message constants hold the text the execution engine records."""
        assert PREREQUISITE_FAILURE_MESSAGE == "Unable to execute job. A preceding pipeline phase failed."
        assert UNREACHABLE_PREREQUISITE_MESSAGE == (
            "Unable to execute job. Its prerequisite jobs never succeeded and no queued job can still satisfy them."
        )


class TestPhaseModel:
    """Tests the declarative phase tuples the resolvers expand."""

    def test_single_recording_phases_are_declared_in_execution_order(self) -> None:
        """Verifies that the single-recording phase chain runs binarization, registration, processing, combination."""
        assert [phase.job_name for phase in SINGLE_RECORDING_PHASES] == [
            SingleRecordingJobNames.BINARIZE,
            SingleRecordingJobNames.REGISTER,
            SingleRecordingJobNames.PROCESS,
            SingleRecordingJobNames.COMBINE,
        ]

    def test_multi_recording_phases_are_declared_in_execution_order(self) -> None:
        """Verifies that the multi-recording phase chain runs discovery followed by extraction."""
        assert [phase.job_name for phase in MULTI_RECORDING_PHASES] == [
            MultiRecordingJobNames.DISCOVER,
            MultiRecordingJobNames.EXTRACT,
        ]

    @pytest.mark.parametrize("phases", [SINGLE_RECORDING_PHASES, MULTI_RECORDING_PHASES])
    def test_only_the_first_phase_lacks_a_prerequisite(self, phases: tuple[object, ...]) -> None:
        """Verifies that each phase after the first names its immediate predecessor as its prerequisite."""
        job_names = [phase.job_name for phase in phases]  # type: ignore[attr-defined]
        prerequisites = [phase.prerequisite for phase in phases]  # type: ignore[attr-defined]

        assert prerequisites[0] is None
        assert prerequisites[1:] == job_names[:-1]


class TestResolvePlaneSpecifier:
    """Tests the per-plane tracker specifier format."""

    @pytest.mark.parametrize("plane_index", [0, 1, 7, 42])
    def test_specifier_carries_the_prefix_and_the_plane_index(self, plane_index: int) -> None:
        """Verifies that the specifier concatenates the shared prefix with the plane index."""
        assert resolve_plane_specifier(plane_index=plane_index) == f"{PLANE_SPECIFIER_PREFIX}{plane_index}"


class TestResolvePipelineJobs:
    """Tests the expansion of a phase tuple into a job list."""

    def test_per_specifier_phases_expand_and_others_carry_an_empty_specifier(self) -> None:
        """Verifies that per-specifier phases yield one job per specifier while single jobs carry an empty specifier."""
        jobs = resolve_pipeline_jobs(phases=SINGLE_RECORDING_PHASES, specifiers=["plane_0", "plane_1"])

        assert jobs == [
            (SingleRecordingJobNames.BINARIZE, ""),
            (SingleRecordingJobNames.REGISTER, "plane_0"),
            (SingleRecordingJobNames.REGISTER, "plane_1"),
            (SingleRecordingJobNames.PROCESS, "plane_0"),
            (SingleRecordingJobNames.PROCESS, "plane_1"),
            (SingleRecordingJobNames.COMBINE, ""),
        ]

    def test_empty_specifiers_drop_every_per_specifier_job(self) -> None:
        """Verifies that a pipeline given no specifiers retains only the phases that do not expand."""
        jobs = resolve_pipeline_jobs(phases=SINGLE_RECORDING_PHASES, specifiers=[])

        assert jobs == [(SingleRecordingJobNames.BINARIZE, ""), (SingleRecordingJobNames.COMBINE, "")]


class TestResolveJobUniverses:
    """Tests the job universes the two pipelines resolve for a recording and a dataset."""

    def test_single_recording_universe_covers_every_plane(self) -> None:
        """Verifies that the single-recording universe expands the per-plane phases over every plane index."""
        jobs = resolve_single_recording_jobs(plane_count=3)

        assert jobs == [
            (SingleRecordingJobNames.BINARIZE, ""),
            (SingleRecordingJobNames.REGISTER, "plane_0"),
            (SingleRecordingJobNames.REGISTER, "plane_1"),
            (SingleRecordingJobNames.REGISTER, "plane_2"),
            (SingleRecordingJobNames.PROCESS, "plane_0"),
            (SingleRecordingJobNames.PROCESS, "plane_1"),
            (SingleRecordingJobNames.PROCESS, "plane_2"),
            (SingleRecordingJobNames.COMBINE, ""),
        ]

    def test_multi_recording_universe_covers_every_recording(self) -> None:
        """Verifies that the multi-recording universe expands extraction over every recording identifier."""
        jobs = resolve_multi_recording_jobs(recording_ids=["recording_a", "recording_b"])

        assert jobs == [
            (MultiRecordingJobNames.DISCOVER, ""),
            (MultiRecordingJobNames.EXTRACT, "recording_a"),
            (MultiRecordingJobNames.EXTRACT, "recording_b"),
        ]


class TestGenerateJobIds:
    """Tests the job identifiers a caller derives from a resolved job universe."""

    def test_identifiers_match_the_tracker_registration(self, tmp_path: Path) -> None:
        """Verifies that the derived identifiers equal the ones a tracker registers for the same universe."""
        universe = resolve_single_recording_jobs(plane_count=2)
        tracker = ProcessingTracker(file_path=tmp_path / SINGLE_RECORDING_TRACKER_FILENAME)

        registered = tracker.initialize_jobs(jobs=universe)

        assert generate_job_ids(jobs=universe) == dict(zip(universe, registered, strict=True))

    def test_multi_recording_universe_resolves_every_job(self) -> None:
        """Verifies that every multi-recording job receives its own identifier."""
        universe = resolve_multi_recording_jobs(recording_ids=["recording_a", "recording_b"])

        identifiers = generate_job_ids(jobs=universe)

        assert set(identifiers) == set(universe)
        assert len(set(identifiers.values())) == len(universe)

    def test_empty_universe_resolves_no_identifiers(self) -> None:
        """Verifies that an empty job universe resolves to an empty identifier mapping."""
        assert generate_job_ids(jobs=[]) == {}

    def test_colon_in_a_job_name_is_rejected(self) -> None:
        """Verifies that a job name carrying the identifier separator raises an error."""
        with pytest.raises(ValueError, match=r"must\s+not contain the ':' character"):
            generate_job_ids(jobs=[("invalid:name", "")])


class TestSingleRecordingPrerequisites:
    """Tests the prerequisite graph of the single-recording pipeline."""

    def test_graph_matches_the_declared_phase_chain(self) -> None:
        """Verifies the prerequisite of every job, covering both the matching-specifier and all-jobs scopes."""
        jobs = resolve_single_recording_jobs(plane_count=2)

        prerequisites = resolve_single_recording_prerequisites(jobs=jobs)

        assert prerequisites == {
            (SingleRecordingJobNames.BINARIZE, ""): (),
            (SingleRecordingJobNames.REGISTER, "plane_0"): ((SingleRecordingJobNames.BINARIZE, ""),),
            (SingleRecordingJobNames.REGISTER, "plane_1"): ((SingleRecordingJobNames.BINARIZE, ""),),
            (SingleRecordingJobNames.PROCESS, "plane_0"): ((SingleRecordingJobNames.REGISTER, "plane_0"),),
            (SingleRecordingJobNames.PROCESS, "plane_1"): ((SingleRecordingJobNames.REGISTER, "plane_1"),),
            (SingleRecordingJobNames.COMBINE, ""): (
                (SingleRecordingJobNames.PROCESS, "plane_0"),
                (SingleRecordingJobNames.PROCESS, "plane_1"),
            ),
        }

    def test_processing_depends_only_on_the_registration_of_its_own_plane(self) -> None:
        """Verifies that the matching-specifier scope isolates each plane's processing job from its siblings."""
        jobs = resolve_single_recording_jobs(plane_count=4)

        prerequisites = resolve_single_recording_prerequisites(jobs=jobs)

        for plane_index in range(4):
            specifier = resolve_plane_specifier(plane_index=plane_index)
            assert prerequisites[(SingleRecordingJobNames.PROCESS, specifier)] == (
                (SingleRecordingJobNames.REGISTER, specifier),
            )

    def test_missing_prerequisite_job_resolves_to_an_empty_tuple(self) -> None:
        """Verifies that a job whose prerequisite phase is absent from the universe depends on nothing."""
        jobs = [(SingleRecordingJobNames.REGISTER, "plane_0")]

        prerequisites = resolve_single_recording_prerequisites(jobs=jobs)

        assert prerequisites == {(SingleRecordingJobNames.REGISTER, "plane_0"): ()}

    def test_foreign_job_name_resolves_to_an_empty_tuple(self) -> None:
        """Verifies that a job name outside the phase model is retained without acquiring prerequisites."""
        jobs = [("foreign_job", ""), (SingleRecordingJobNames.BINARIZE, "")]

        prerequisites = resolve_single_recording_prerequisites(jobs=jobs)

        assert prerequisites == {("foreign_job", ""): (), (SingleRecordingJobNames.BINARIZE, ""): ()}

    def test_empty_universe_resolves_to_an_empty_graph(self) -> None:
        """Verifies that resolving the prerequisites of no jobs yields an empty mapping."""
        assert resolve_single_recording_prerequisites(jobs=[]) == {}


class TestMultiRecordingPrerequisites:
    """Tests the prerequisite graph of the multi-recording pipeline."""

    def test_every_extraction_depends_on_the_single_discovery_job(self) -> None:
        """Verifies that extraction jobs share the dataset-wide discovery job as their only prerequisite."""
        jobs = resolve_multi_recording_jobs(recording_ids=["recording_a", "recording_b"])

        prerequisites = resolve_multi_recording_prerequisites(jobs=jobs)

        assert prerequisites == {
            (MultiRecordingJobNames.DISCOVER, ""): (),
            (MultiRecordingJobNames.EXTRACT, "recording_a"): ((MultiRecordingJobNames.DISCOVER, ""),),
            (MultiRecordingJobNames.EXTRACT, "recording_b"): ((MultiRecordingJobNames.DISCOVER, ""),),
        }

    def test_empty_universe_resolves_to_an_empty_graph(self) -> None:
        """Verifies that resolving the prerequisites of no jobs yields an empty mapping."""
        assert resolve_multi_recording_prerequisites(jobs=[]) == {}


class TestResolvePrerequisitesScopes:
    """Tests the prerequisite scope selection of the shared graph builder."""

    def test_all_jobs_scope_collects_every_predecessor_specifier(self) -> None:
        """Verifies that the all-jobs scope depends on each predecessor job whatever specifier it carries."""
        jobs = [
            (SingleRecordingJobNames.PROCESS, "plane_0"),
            (SingleRecordingJobNames.PROCESS, "plane_1"),
            (SingleRecordingJobNames.COMBINE, ""),
        ]

        prerequisites = _resolve_prerequisites(jobs=jobs, phases=SINGLE_RECORDING_PHASES)

        assert prerequisites[(SingleRecordingJobNames.COMBINE, "")] == (
            (SingleRecordingJobNames.PROCESS, "plane_0"),
            (SingleRecordingJobNames.PROCESS, "plane_1"),
        )

    def test_matching_specifier_scope_ignores_non_matching_predecessors(self) -> None:
        """Verifies that the matching-specifier scope filters predecessors down to the job's own specifier."""
        jobs = [
            (SingleRecordingJobNames.REGISTER, "plane_0"),
            (SingleRecordingJobNames.REGISTER, "plane_1"),
            (SingleRecordingJobNames.PROCESS, "plane_1"),
        ]

        prerequisites = _resolve_prerequisites(jobs=jobs, phases=SINGLE_RECORDING_PHASES)

        assert prerequisites[(SingleRecordingJobNames.PROCESS, "plane_1")] == (
            (SingleRecordingJobNames.REGISTER, "plane_1"),
        )

    def test_every_declared_scope_is_exercised_by_the_single_recording_chain(self) -> None:
        """Verifies that the single-recording chain declares both prerequisite scopes, so both paths stay reachable."""
        scopes = {phase.prerequisite_scope for phase in SINGLE_RECORDING_PHASES}

        assert scopes == {PrerequisiteScope.ALL_JOBS, PrerequisiteScope.MATCHING_SPECIFIER}


class TestResolveDownstreamPhases:
    """Tests the expansion of a requested phase into itself plus its dependents."""

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [
            (
                [SingleRecordingJobNames.BINARIZE],
                {
                    SingleRecordingJobNames.BINARIZE,
                    SingleRecordingJobNames.REGISTER,
                    SingleRecordingJobNames.PROCESS,
                    SingleRecordingJobNames.COMBINE,
                },
            ),
            (
                [SingleRecordingJobNames.REGISTER],
                {
                    SingleRecordingJobNames.REGISTER,
                    SingleRecordingJobNames.PROCESS,
                    SingleRecordingJobNames.COMBINE,
                },
            ),
            (
                [SingleRecordingJobNames.PROCESS],
                {SingleRecordingJobNames.PROCESS, SingleRecordingJobNames.COMBINE},
            ),
            ([SingleRecordingJobNames.COMBINE], {SingleRecordingJobNames.COMBINE}),
        ],
    )
    def test_single_recording_request_expands_to_its_dependents(self, requested: list[str], expected: set[str]) -> None:
        """Verifies that requesting a single-recording phase also selects every phase below it in the chain."""
        assert resolve_downstream_phases(phase_names=requested, single_recording=True) == expected

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [
            (
                [MultiRecordingJobNames.DISCOVER],
                {MultiRecordingJobNames.DISCOVER, MultiRecordingJobNames.EXTRACT},
            ),
            ([MultiRecordingJobNames.EXTRACT], {MultiRecordingJobNames.EXTRACT}),
        ],
    )
    def test_multi_recording_request_expands_to_its_dependents(self, requested: list[str], expected: set[str]) -> None:
        """Verifies that requesting a multi-recording phase also selects every phase below it in the chain."""
        assert resolve_downstream_phases(phase_names=requested, single_recording=False) == expected

    def test_the_earliest_requested_phase_determines_the_expansion(self) -> None:
        """Verifies that requesting several phases expands from the earliest of them onward."""
        requested = [SingleRecordingJobNames.COMBINE, SingleRecordingJobNames.REGISTER]

        expanded = resolve_downstream_phases(phase_names=requested, single_recording=True)

        assert expanded == {
            SingleRecordingJobNames.REGISTER,
            SingleRecordingJobNames.PROCESS,
            SingleRecordingJobNames.COMBINE,
        }

    def test_unknown_phase_name_is_returned_unchanged(self) -> None:
        """Verifies that a name outside the pipeline chain expands to nothing beyond itself."""
        assert resolve_downstream_phases(phase_names=["foreign_phase"], single_recording=True) == {"foreign_phase"}

    def test_empty_request_expands_to_an_empty_set(self) -> None:
        """Verifies that requesting no phases selects no phases."""
        assert resolve_downstream_phases(phase_names=[], single_recording=True) == set()

    def test_multi_recording_chain_does_not_leak_single_recording_phases(self) -> None:
        """Verifies that the multi-recording chain expands independently of the single-recording phase names."""
        expanded = resolve_downstream_phases(phase_names=[SingleRecordingJobNames.BINARIZE], single_recording=False)

        assert expanded == {SingleRecordingJobNames.BINARIZE}


class TestOrderPhasesByExecution:
    """Tests the ordering of phase job names by the order the pipeline executes them."""

    @pytest.mark.parametrize(
        ("requested", "expected"),
        [
            (
                [
                    SingleRecordingJobNames.COMBINE,
                    SingleRecordingJobNames.PROCESS,
                    SingleRecordingJobNames.REGISTER,
                    SingleRecordingJobNames.BINARIZE,
                ],
                [
                    SingleRecordingJobNames.BINARIZE,
                    SingleRecordingJobNames.REGISTER,
                    SingleRecordingJobNames.PROCESS,
                    SingleRecordingJobNames.COMBINE,
                ],
            ),
            (
                [SingleRecordingJobNames.COMBINE, SingleRecordingJobNames.REGISTER],
                [SingleRecordingJobNames.REGISTER, SingleRecordingJobNames.COMBINE],
            ),
            ([SingleRecordingJobNames.PROCESS], [SingleRecordingJobNames.PROCESS]),
            ([], []),
        ],
    )
    def test_single_recording_names_follow_the_single_recording_chain(
        self, requested: list[str], expected: list[str]
    ) -> None:
        """Verifies that single-recording phase names order by execution rather than alphabetically."""
        assert order_phases_by_execution(phase_names=requested, single_recording=True) == expected

    def test_execution_order_differs_from_alphabetical_order(self) -> None:
        """Verifies that the full single-recording chain is not the alphabetical ordering of the same names."""
        requested = [
            SingleRecordingJobNames.BINARIZE,
            SingleRecordingJobNames.REGISTER,
            SingleRecordingJobNames.PROCESS,
            SingleRecordingJobNames.COMBINE,
        ]

        ordered = order_phases_by_execution(phase_names=requested, single_recording=True)

        assert ordered == requested
        assert ordered != sorted(requested)

    def test_multi_recording_names_follow_the_multi_recording_chain(self) -> None:
        """Verifies that the multi-recording chain orders discovery ahead of extraction."""
        requested = [MultiRecordingJobNames.EXTRACT, MultiRecordingJobNames.DISCOVER]

        ordered = order_phases_by_execution(phase_names=requested, single_recording=False)

        assert ordered == [MultiRecordingJobNames.DISCOVER, MultiRecordingJobNames.EXTRACT]

    def test_unknown_names_are_appended_alphabetically_after_the_known_names(self) -> None:
        """Verifies that names outside the phase model trail the ordered chain in alphabetical order."""
        requested = [
            "zeta_phase",
            SingleRecordingJobNames.COMBINE,
            "alpha_phase",
            SingleRecordingJobNames.BINARIZE,
        ]

        ordered = order_phases_by_execution(phase_names=requested, single_recording=True)

        assert ordered == [
            SingleRecordingJobNames.BINARIZE,
            SingleRecordingJobNames.COMBINE,
            "alpha_phase",
            "zeta_phase",
        ]

    def test_single_recording_names_are_unknown_to_the_multi_recording_chain(self) -> None:
        """Verifies that the multi-recording chain treats single-recording names as names outside the model."""
        requested = [
            SingleRecordingJobNames.REGISTER,
            MultiRecordingJobNames.EXTRACT,
            SingleRecordingJobNames.BINARIZE,
        ]

        ordered = order_phases_by_execution(phase_names=requested, single_recording=False)

        assert ordered == [
            MultiRecordingJobNames.EXTRACT,
            SingleRecordingJobNames.BINARIZE,
            SingleRecordingJobNames.REGISTER,
        ]


class TestResolvePrerequisiteJobIds:
    """Tests the resolution of a job's prerequisite identifiers against a tracker job registry."""

    def test_unregistered_job_reports_an_error(self) -> None:
        """Verifies that resolving a job identifier the registry does not hold reports an error message."""
        registry = _build_single_recording_registry(plane_count=1)

        prerequisite_ids, message = resolve_prerequisite_job_ids(
            registry=registry, job_id="0123456789abcdef", single_recording=True
        )

        assert prerequisite_ids == []
        assert message is not None
        assert "Unable to resolve the prerequisites for job 0123456789abcdef." in message
        assert "not registered in the tracker" in message

    def test_first_phase_job_depends_on_nothing(self) -> None:
        """Verifies that the binarization job, which opens the chain, resolves to no prerequisites and no error."""
        registry = _build_single_recording_registry(plane_count=2)

        prerequisite_ids, message = resolve_prerequisite_job_ids(
            registry=registry, job_id=_job_id(job_name=SingleRecordingJobNames.BINARIZE), single_recording=True
        )

        assert prerequisite_ids == []
        assert message is None

    def test_all_jobs_scope_collects_the_single_predecessor_job(self) -> None:
        """Verifies that a registration job depends on the recording's only binarization job."""
        registry = _build_single_recording_registry(plane_count=2)

        prerequisite_ids, message = resolve_prerequisite_job_ids(
            registry=registry,
            job_id=_job_id(job_name=SingleRecordingJobNames.REGISTER, specifier="plane_1"),
            single_recording=True,
        )

        assert message is None
        assert prerequisite_ids == [_job_id(job_name=SingleRecordingJobNames.BINARIZE)]

    def test_all_jobs_scope_collects_every_predecessor_specifier(self) -> None:
        """Verifies that the combination job depends on the processing job of every plane."""
        registry = _build_single_recording_registry(plane_count=2)

        prerequisite_ids, message = resolve_prerequisite_job_ids(
            registry=registry, job_id=_job_id(job_name=SingleRecordingJobNames.COMBINE), single_recording=True
        )

        assert message is None
        assert set(prerequisite_ids) == {
            _job_id(job_name=SingleRecordingJobNames.PROCESS, specifier="plane_0"),
            _job_id(job_name=SingleRecordingJobNames.PROCESS, specifier="plane_1"),
        }

    def test_matching_specifier_scope_selects_the_predecessor_of_the_same_plane(self) -> None:
        """Verifies that a processing job depends only on the registration job carrying the same plane specifier."""
        registry = _build_single_recording_registry(plane_count=3)

        prerequisite_ids, message = resolve_prerequisite_job_ids(
            registry=registry,
            job_id=_job_id(job_name=SingleRecordingJobNames.PROCESS, specifier="plane_1"),
            single_recording=True,
        )

        assert message is None
        assert prerequisite_ids == [_job_id(job_name=SingleRecordingJobNames.REGISTER, specifier="plane_1")]

    def test_multi_recording_extraction_depends_on_the_discovery_job(self) -> None:
        """Verifies that the multi-recording rules resolve each extraction job onto the dataset's discovery job."""
        jobs = resolve_multi_recording_jobs(recording_ids=["recording_a", "recording_b"])
        registry = _build_registry(
            jobs=[(job_name, specifier, ProcessingStatus.SCHEDULED) for job_name, specifier in jobs]
        )

        prerequisite_ids, message = resolve_prerequisite_job_ids(
            registry=registry,
            job_id=_job_id(job_name=MultiRecordingJobNames.EXTRACT, specifier="recording_a"),
            single_recording=False,
        )

        assert message is None
        assert prerequisite_ids == [_job_id(job_name=MultiRecordingJobNames.DISCOVER)]

    def test_job_name_outside_the_phase_model_depends_on_nothing(self) -> None:
        """Verifies that a registered job whose name the phase model does not declare resolves to no prerequisites."""
        registry = _build_registry(jobs=[("foreign_job", "", ProcessingStatus.SCHEDULED)])

        prerequisite_ids, message = resolve_prerequisite_job_ids(
            registry=registry, job_id=_job_id(job_name="foreign_job"), single_recording=True
        )

        assert prerequisite_ids == []
        assert message is None

    def test_absent_prerequisite_phase_reports_an_error(self) -> None:
        """Verifies that a registry missing the prerequisite phase reports an error rather than an empty list."""
        registry = _build_registry(jobs=[(SingleRecordingJobNames.PROCESS, "plane_0", ProcessingStatus.SCHEDULED)])

        prerequisite_ids, message = resolve_prerequisite_job_ids(
            registry=registry,
            job_id=_job_id(job_name=SingleRecordingJobNames.PROCESS, specifier="plane_0"),
            single_recording=True,
        )

        assert prerequisite_ids == []
        assert message is not None
        assert "Its prerequisite 'registration' phase with specifier 'plane_0' is not registered" in message


class TestCollectPhaseJobIds:
    """Tests the collection of a prerequisite phase's job identifiers from a tracker job registry."""

    def test_unscoped_collection_returns_every_job_of_the_phase(self) -> None:
        """Verifies that omitting the specifier collects each job the prerequisite phase registered."""
        registry = _build_single_recording_registry(plane_count=2)

        matches, message = _collect_phase_job_ids(
            registry=registry,
            job_name=SingleRecordingJobNames.REGISTER,
            specifier=None,
            dependent_job_id="dependent",
        )

        assert message is None
        assert set(matches) == {
            _job_id(job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0"),
            _job_id(job_name=SingleRecordingJobNames.REGISTER, specifier="plane_1"),
        }

    def test_scoped_collection_returns_the_job_carrying_the_specifier(self) -> None:
        """Verifies that supplying a specifier narrows the phase down to the job that carries it."""
        registry = _build_single_recording_registry(plane_count=3)

        matches, message = _collect_phase_job_ids(
            registry=registry,
            job_name=SingleRecordingJobNames.REGISTER,
            specifier="plane_2",
            dependent_job_id="dependent",
        )

        assert message is None
        assert matches == [_job_id(job_name=SingleRecordingJobNames.REGISTER, specifier="plane_2")]

    def test_absent_phase_reports_the_unscoped_message(self) -> None:
        """Verifies that an empty registry reports the phase as unregistered without naming a specifier."""
        matches, message = _collect_phase_job_ids(
            registry={}, job_name=SingleRecordingJobNames.BINARIZE, specifier=None, dependent_job_id="dependent"
        )

        assert matches == []
        assert message == (
            "Unable to execute job dependent. Its prerequisite 'binarization' phase is not registered in the "
            "tracker, so the prerequisite can never be satisfied. Re-run the prepare tool for this recording or "
            "dataset to register the missing phase."
        )

    def test_absent_scoped_phase_reports_the_specifier_in_the_message(self) -> None:
        """Verifies that a phase whose registered jobs carry other specifiers reports the requested specifier."""
        registry = _build_single_recording_registry(plane_count=2)

        matches, message = _collect_phase_job_ids(
            registry=registry,
            job_name=SingleRecordingJobNames.REGISTER,
            specifier="plane_5",
            dependent_job_id="dependent",
        )

        assert matches == []
        assert message == (
            "Unable to execute job dependent. Its prerequisite 'registration' phase with specifier 'plane_5' is not "
            "registered in the tracker, so the prerequisite can never be satisfied. Re-run the prepare tool for this "
            "recording or dataset to register the missing phase."
        )


class TestValidateJobPrerequisites:
    """Tests the admission check that guards a job against unmet prerequisites."""

    def test_succeeded_prerequisite_is_accepted(self) -> None:
        """Verifies that a job whose prerequisite already succeeded passes validation."""
        registry = _build_registry(
            jobs=[
                (SingleRecordingJobNames.BINARIZE, "", ProcessingStatus.SUCCEEDED),
                (SingleRecordingJobNames.REGISTER, "plane_0", ProcessingStatus.SCHEDULED),
            ]
        )

        message = validate_job_prerequisites(
            registry=registry,
            job_id=_job_id(job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0"),
            single_recording=True,
            submitted_job_ids=frozenset(),
        )

        assert message is None

    @pytest.mark.parametrize(
        "prerequisite_status", [ProcessingStatus.SCHEDULED, ProcessingStatus.RUNNING, ProcessingStatus.FAILED]
    )
    def test_prerequisite_submitted_alongside_the_job_is_accepted(self, prerequisite_status: ProcessingStatus) -> None:
        """Verifies that a prerequisite arriving with the same submission passes whatever status it currently holds."""
        registry = _build_registry(
            jobs=[
                (SingleRecordingJobNames.BINARIZE, "", prerequisite_status),
                (SingleRecordingJobNames.REGISTER, "plane_0", ProcessingStatus.SCHEDULED),
            ]
        )

        message = validate_job_prerequisites(
            registry=registry,
            job_id=_job_id(job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0"),
            single_recording=True,
            submitted_job_ids=frozenset({_job_id(job_name=SingleRecordingJobNames.BINARIZE)}),
        )

        assert message is None

    @pytest.mark.parametrize(
        "prerequisite_status", [ProcessingStatus.SCHEDULED, ProcessingStatus.RUNNING, ProcessingStatus.FAILED]
    )
    def test_prerequisite_neither_succeeded_nor_submitted_is_rejected(
        self, prerequisite_status: ProcessingStatus
    ) -> None:
        """Verifies that an unmet prerequisite outside the submission reports the job and the prerequisite."""
        registry = _build_registry(
            jobs=[
                (SingleRecordingJobNames.BINARIZE, "", prerequisite_status),
                (SingleRecordingJobNames.REGISTER, "plane_0", ProcessingStatus.SCHEDULED),
            ]
        )
        job_id = _job_id(job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0")
        prerequisite_id = _job_id(job_name=SingleRecordingJobNames.BINARIZE)

        message = validate_job_prerequisites(
            registry=registry, job_id=job_id, single_recording=True, submitted_job_ids=frozenset()
        )

        assert message == (
            f"Unable to execute job {job_id}. Its prerequisite 'binarization' job {prerequisite_id} has not "
            f"succeeded and is not part of this submission."
        )

    def test_first_phase_job_is_accepted_without_prerequisites(self) -> None:
        """Verifies that the job opening the chain passes validation with no prerequisite to check."""
        registry = _build_single_recording_registry(plane_count=1)

        message = validate_job_prerequisites(
            registry=registry,
            job_id=_job_id(job_name=SingleRecordingJobNames.BINARIZE),
            single_recording=True,
            submitted_job_ids=frozenset(),
        )

        assert message is None

    def test_absent_prerequisite_phase_message_is_passed_through(self) -> None:
        """Verifies that the resolver's missing-phase message reaches the caller instead of a prerequisite check."""
        registry = _build_registry(jobs=[(SingleRecordingJobNames.PROCESS, "plane_0", ProcessingStatus.SCHEDULED)])

        message = validate_job_prerequisites(
            registry=registry,
            job_id=_job_id(job_name=SingleRecordingJobNames.PROCESS, specifier="plane_0"),
            single_recording=True,
            submitted_job_ids=frozenset(),
        )

        assert message is not None
        assert "Its prerequisite 'registration' phase with specifier 'plane_0' is not registered" in message

    def test_unregistered_job_message_is_passed_through(self) -> None:
        """Verifies that validating a job the registry does not hold reports the resolver's unregistered message."""
        registry = _build_single_recording_registry(plane_count=1)

        message = validate_job_prerequisites(
            registry=registry, job_id="0123456789abcdef", single_recording=True, submitted_job_ids=frozenset()
        )

        assert message is not None
        assert "Unable to resolve the prerequisites for job 0123456789abcdef." in message


def _job_id(job_name: str, specifier: str = "") -> str:
    """Returns the tracker identifier that the job with the given name and specifier is registered under."""
    return ProcessingTracker.generate_job_id(job_name=str(job_name), specifier=specifier)


def _build_registry(jobs: Sequence[tuple[str, str, ProcessingStatus]]) -> dict[str, JobState]:
    """Builds a tracker job registry that maps the identifier of each given job to its state."""
    return {
        _job_id(job_name=job_name, specifier=specifier): JobState(
            job_name=str(job_name), specifier=specifier, status=status
        )
        for job_name, specifier, status in jobs
    }


def _build_single_recording_registry(
    plane_count: int, status: ProcessingStatus = ProcessingStatus.SCHEDULED
) -> dict[str, JobState]:
    """Builds the registry of a single-recording tracker whose every job carries the given status."""
    jobs = resolve_single_recording_jobs(plane_count=plane_count)
    return _build_registry(jobs=[(job_name, specifier, status) for job_name, specifier in jobs])
