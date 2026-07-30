"""Provides the phase model of the single and multi-recording pipelines, which defines the job universe of a recording
and the prerequisite graph over that universe.

The pipelines, the interface layer, and any external scheduler all need the same answers about which jobs exist for a
recording and which jobs must succeed before a given job runs. Exporting the model keeps those answers in one place, so
that inserting or reordering a phase does not require every consumer to be edited in step.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING
from dataclasses import dataclass

from .job_names import MultiRecordingJobNames, SingleRecordingJobNames

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

PLANE_SPECIFIER_PREFIX: str = "plane_"
"""The prefix of the tracker specifier that identifies which imaging plane a per-plane job processes."""


class PrerequisiteScope(StrEnum):
    """Defines how a phase's prerequisite jobs are selected from the phase that precedes it."""

    ALL_JOBS = "all_jobs"
    """Every job of the preceding phase must succeed, whatever specifier it carries."""
    MATCHING_SPECIFIER = "matching_specifier"
    """Only the preceding phase's job that carries the same specifier must succeed."""


@dataclass(frozen=True, slots=True)
class PipelinePhase:
    """Describes one phase of a cindra processing pipeline."""

    job_name: str
    """The tracker job name that identifies the phase."""

    per_specifier: bool
    """Determines whether the phase expands into one job per specifier instead of a single job."""

    prerequisite: str | None
    """The job name of the phase that must succeed before this phase runs, or None for the pipeline's first phase."""

    prerequisite_scope: PrerequisiteScope
    """Determines which jobs of the preceding phase this phase's jobs depend on."""


SINGLE_RECORDING_PHASES: tuple[PipelinePhase, ...] = (
    PipelinePhase(
        job_name=SingleRecordingJobNames.BINARIZE,
        per_specifier=False,
        prerequisite=None,
        prerequisite_scope=PrerequisiteScope.ALL_JOBS,
    ),
    PipelinePhase(
        job_name=SingleRecordingJobNames.REGISTER,
        per_specifier=True,
        prerequisite=SingleRecordingJobNames.BINARIZE,
        prerequisite_scope=PrerequisiteScope.ALL_JOBS,
    ),
    PipelinePhase(
        job_name=SingleRecordingJobNames.PROCESS,
        per_specifier=True,
        prerequisite=SingleRecordingJobNames.REGISTER,
        prerequisite_scope=PrerequisiteScope.MATCHING_SPECIFIER,
    ),
    PipelinePhase(
        job_name=SingleRecordingJobNames.COMBINE,
        per_specifier=False,
        prerequisite=SingleRecordingJobNames.PROCESS,
        prerequisite_scope=PrerequisiteScope.ALL_JOBS,
    ),
)
"""The ordered phases of the single-recording pipeline."""

MULTI_RECORDING_PHASES: tuple[PipelinePhase, ...] = (
    PipelinePhase(
        job_name=MultiRecordingJobNames.DISCOVER,
        per_specifier=False,
        prerequisite=None,
        prerequisite_scope=PrerequisiteScope.ALL_JOBS,
    ),
    PipelinePhase(
        job_name=MultiRecordingJobNames.EXTRACT,
        per_specifier=True,
        prerequisite=MultiRecordingJobNames.DISCOVER,
        prerequisite_scope=PrerequisiteScope.ALL_JOBS,
    ),
)
"""The ordered phases of the multi-recording pipeline."""


def resolve_plane_specifier(plane_index: int) -> str:
    """Returns the tracker specifier that identifies a per-plane single-recording job.

    Args:
        plane_index: The index of the imaging plane the job processes.

    Returns:
        The specifier string the tracker stores for the plane's jobs.
    """
    return f"{PLANE_SPECIFIER_PREFIX}{plane_index}"


def resolve_single_recording_jobs(plane_count: int) -> list[tuple[str, str]]:
    """Returns every job the single-recording pipeline can execute for a recording with the given plane count.

    Notes:
        The returned list is the recording's job universe, which a tracker uses to distinguish the jobs that belong to
        the recording from foreign entries. It therefore covers every plane, independently of which planes a particular
        invocation intends to run.

    Args:
        plane_count: The number of virtual imaging planes the recording holds.

    Returns:
        A list of (job name, specifier) pairs in execution order. Jobs that do not expand per plane carry an empty
        specifier.
    """
    specifiers = [resolve_plane_specifier(plane_index=plane_index) for plane_index in range(plane_count)]
    return resolve_pipeline_jobs(phases=SINGLE_RECORDING_PHASES, specifiers=specifiers)


def resolve_multi_recording_jobs(recording_ids: Sequence[str]) -> list[tuple[str, str]]:
    """Returns every job the multi-recording pipeline can execute for a dataset of the given recordings.

    Args:
        recording_ids: The identifier of every recording the tracked dataset spans.

    Returns:
        A list of (job name, specifier) pairs in execution order. The discovery job carries an empty specifier, and
        each extraction job carries its recording identifier.
    """
    return resolve_pipeline_jobs(phases=MULTI_RECORDING_PHASES, specifiers=list(recording_ids))


def resolve_single_recording_prerequisites(
    jobs: Iterable[tuple[str, str]],
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]:
    """Returns the jobs that must succeed before each single-recording job can run.

    Args:
        jobs: The (job name, specifier) pairs to resolve the prerequisites of, usually a recording's job universe.

    Returns:
        A mapping of every input job to the tuple of jobs it depends on. The tuple is empty for jobs that depend on
        nothing.
    """
    return _resolve_prerequisites(jobs=jobs, phases=SINGLE_RECORDING_PHASES)


def resolve_multi_recording_prerequisites(
    jobs: Iterable[tuple[str, str]],
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]:
    """Returns the jobs that must succeed before each multi-recording job can run.

    Args:
        jobs: The (job name, specifier) pairs to resolve the prerequisites of, usually a dataset's job universe.

    Returns:
        A mapping of every input job to the tuple of jobs it depends on. The tuple is empty for jobs that depend on
        nothing.
    """
    return _resolve_prerequisites(jobs=jobs, phases=MULTI_RECORDING_PHASES)


def resolve_downstream_phases(phase_names: Iterable[str], *, single_recording: bool) -> set[str]:
    """Returns the requested phases together with every phase that depends on them.

    Notes:
        Each pipeline runs its phases in a single chain, so every phase below the earliest requested one consumes
        output the requested phase produces. Resetting or cleaning a phase therefore invalidates all of them.

    Args:
        phase_names: The job names of the phases the caller requested.
        single_recording: Determines whether to apply the single-recording or the multi-recording phase chain.

    Returns:
        The set of phase job names to act on.
    """
    phases = SINGLE_RECORDING_PHASES if single_recording else MULTI_RECORDING_PHASES
    ordered_names = [str(phase.job_name) for phase in phases]
    expanded = set(phase_names)
    for index, name in enumerate(ordered_names):
        if name in expanded:
            expanded.update(ordered_names[index:])
            break
    return expanded


def resolve_pipeline_jobs(phases: tuple[PipelinePhase, ...], specifiers: Sequence[str]) -> list[tuple[str, str]]:
    """Expands a pipeline's phases into a job list over the given specifiers.

    Notes:
        Use this function when the specifiers are already known, for example when rebuilding the universe of a tracker
        whose existing per-specifier jobs must be preserved exactly. Prefer resolve_single_recording_jobs when the
        specifiers follow from a plane count.

    Args:
        phases: The ordered phases of the pipeline.
        specifiers: The specifiers that the per-specifier phases expand over.

    Returns:
        A list of (job name, specifier) pairs in phase order. Phases that do not expand per specifier carry an empty
        specifier.
    """
    jobs: list[tuple[str, str]] = []
    for phase in phases:
        if phase.per_specifier:
            jobs.extend((str(phase.job_name), specifier) for specifier in specifiers)
        else:
            jobs.append((str(phase.job_name), ""))
    return jobs


def _resolve_prerequisites(
    jobs: Iterable[tuple[str, str]],
    phases: tuple[PipelinePhase, ...],
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]:
    """Builds the prerequisite graph over a set of jobs using a pipeline's phase chain.

    Notes:
        Each job depends on its immediate predecessor phase alone, because a succeeded predecessor already implies
        every phase above it.

    Args:
        jobs: The (job name, specifier) pairs to resolve the prerequisites of.
        phases: The ordered phases of the pipeline the jobs belong to.

    Returns:
        A mapping of every input job to the tuple of jobs it depends on.
    """
    job_list = list(jobs)
    phase_by_name = {str(phase.job_name): phase for phase in phases}

    jobs_by_name: dict[str, list[tuple[str, str]]] = {}
    for job in job_list:
        jobs_by_name.setdefault(job[0], []).append(job)

    prerequisites: dict[tuple[str, str], tuple[tuple[str, str], ...]] = {}
    for job_name, specifier in job_list:
        phase = phase_by_name.get(job_name)
        if phase is None or phase.prerequisite is None:
            prerequisites[(job_name, specifier)] = ()
            continue

        candidates = jobs_by_name.get(str(phase.prerequisite), [])
        if phase.prerequisite_scope == PrerequisiteScope.MATCHING_SPECIFIER:
            prerequisites[(job_name, specifier)] = tuple(
                candidate for candidate in candidates if candidate[1] == specifier
            )
        else:
            prerequisites[(job_name, specifier)] = tuple(candidates)

    return prerequisites
