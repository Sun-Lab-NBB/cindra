"""Contains tests for the pipeline phase model, the job universe resolvers, and the prerequisite graph."""

from __future__ import annotations

import pytest

from cindra.allocation import (
    MULTI_RECORDING_PHASES,
    PLANE_SPECIFIER_PREFIX,
    SINGLE_RECORDING_PHASES,
    PrerequisiteScope,
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    resolve_pipeline_jobs,
    resolve_plane_specifier,
    resolve_downstream_phases,
    resolve_multi_recording_jobs,
    resolve_single_recording_jobs,
    resolve_multi_recording_prerequisites,
    resolve_single_recording_prerequisites,
)
from cindra.allocation.phases import _resolve_prerequisites


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
