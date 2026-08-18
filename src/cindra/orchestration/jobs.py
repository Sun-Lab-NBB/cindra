"""Provides the job model of the single and multi-recording pipelines, which names every pipeline stage, defines the
job universe of a recording, and resolves the prerequisite graph over that universe.

The pipelines, the execution engine, the interface layer, and any external scheduler all need the same answers about
which jobs exist for a recording and which jobs must succeed before a given job runs. Keeping the model in one leaf
module gives every consumer the same answers, so that inserting or reordering a phase does not require each of them to
be edited in step.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING
from dataclasses import dataclass

from ataraxis_data_structures import JobState, ProcessingStatus, ProcessingTracker

from ..layout import resolve_plane_specifier

if TYPE_CHECKING:
    from collections.abc import Mapping, Iterable, Sequence

PREREQUISITE_FAILURE_MESSAGE: str = "Unable to execute job. A preceding pipeline phase failed."
"""The tracker error message recorded for a job whose prerequisite job failed."""

UNREACHABLE_PREREQUISITE_MESSAGE: str = (
    "Unable to execute job. Its prerequisite jobs never succeeded and no queued job can still satisfy them."
)
"""The tracker error message recorded for a job that the execution session can no longer admit."""


class SingleRecordingJobNames(StrEnum):
    """Defines the job names for the single-recording processing pipeline components.

    Notes:
        The members are declared in execution order, and that order is rendered into the error messages that list
        the valid job names. The authoritative phase order and prerequisite graph live in SINGLE_RECORDING_PHASES.
        The interface layer derives only an unordered validation set from these members.
    """

    BINARIZE = "binarization"
    """The name for the binarization (step 1) processing job."""
    REGISTER = "registration"
    """The generic name for the plane-registration (step 2) job, which removes motion and computes the
    registration-quality principal components. During runtime, the registered plane is identified by the tracker's
    specifier field using the format 'plane_{plane_index}'."""
    PROCESS = "processing"
    """The generic name for the plane-processing (step 3) job, which discovers ROIs and extracts their fluorescence.
    During runtime, the processed plane is identified by the tracker's specifier field using the format
    'plane_{plane_index}'."""
    COMBINE = "combination"
    """The name for the combination (step 4) processing job."""


class MultiRecordingJobNames(StrEnum):
    """Defines the job names for the multi-recording processing pipeline components."""

    DISCOVER = "discovery"
    """The name for the ROI discovery (step 1) processing job."""
    EXTRACT = "extraction"
    """The generic name for the fluorescence extraction (step 2) processing job. During runtime, the processed recording
    is identified by the tracker's specifier field, which stores the recording ID string."""


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

PER_PLANE_JOB_NAMES: frozenset[str] = frozenset(
    str(phase.job_name) for phase in SINGLE_RECORDING_PHASES if phase.per_specifier
)
"""The single-recording job names that expand into one job per imaging plane, each carrying a 'plane_{index}'
specifier. Derived from the phase model, so that adding a per-plane phase updates it automatically."""


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


def generate_job_ids(jobs: Iterable[tuple[str, str]]) -> dict[tuple[str, str], str]:
    """Generates the processing job identifier of every job in a resolved job universe.

    Notes:
        The identifier derives from the job name and specifier alone, and a tracker records each job under the same
        derivation. A caller therefore names a job for the pipeline and per-job entry points from the universe the
        job resolvers return, rather than by reading the tracker the pipeline maintains.

    Args:
        jobs: The jobs to generate identifiers for, as the (job name, specifier) pairs the job resolvers return.

    Returns:
        The hexadecimal identifier of every job, keyed by its name and specifier.

    Raises:
        ValueError: If a job name or a specifier contains a colon.
    """
    return {
        (job_name, specifier): ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier)
        for job_name, specifier in jobs
    }


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


def order_phases_by_execution(phase_names: Iterable[str], *, single_recording: bool) -> list[str]:
    """Orders phase job names by the order the pipeline executes them.

    Notes:
        Callers report the resulting list to the user, who reads a phase list as a sequence. Alphabetical order would
        render the single-recording chain as binarization, combination, processing, registration, which inverts the
        two middle phases relative to the order they run in.

    Args:
        phase_names: The phase job names to order.
        single_recording: Determines whether to apply the single-recording or the multi-recording phase chain.

    Returns:
        The phase job names in pipeline execution order. Names outside the phase model are appended alphabetically.
    """
    phases = SINGLE_RECORDING_PHASES if single_recording else MULTI_RECORDING_PHASES
    execution_order = {str(phase.job_name): index for index, phase in enumerate(phases)}
    known = [name for name in phase_names if name in execution_order]
    unknown = [name for name in phase_names if name not in execution_order]
    return sorted(known, key=lambda name: execution_order[name]) + sorted(unknown)


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


def resolve_prerequisite_job_ids(
    registry: Mapping[str, JobState], job_id: str, *, single_recording: bool
) -> tuple[list[str], str | None]:
    """Resolves the tracker job IDs that must succeed before the target job can run.

    Notes:
        Each job depends on its immediate predecessor only, because a succeeded predecessor already implies the phases
        above it. Registration and processing pair up per plane, so a processing job depends only on the registration
        job carrying the same specifier.

        A prerequisite phase that the tracker does not contain is reported as an error rather than treated as
        satisfied, which prevents an incompletely initialized tracker from admitting a job whose input never exists.

    Args:
        registry: The point-in-time job registry of the tracker that owns the target job.
        job_id: The unique hexadecimal identifier of the job to resolve the prerequisites for.
        single_recording: Determines whether to apply single-recording or multi-recording prerequisite rules.

    Returns:
        A tuple of the prerequisite job IDs and an error message. The message is None unless the target job itself is
        not registered in the tracker or its prerequisite phase is absent from the tracker.
    """
    job_state = registry.get(job_id)
    if job_state is None:
        message = (
            f"Unable to resolve the prerequisites for job {job_id}. The job is not registered in the tracker that "
            f"was provided for it."
        )
        return [], message

    phases = SINGLE_RECORDING_PHASES if single_recording else MULTI_RECORDING_PHASES
    phase = {str(entry.job_name): entry for entry in phases}.get(job_state.job_name)
    if phase is None or phase.prerequisite is None:
        return [], None

    specifier = job_state.specifier if phase.prerequisite_scope == PrerequisiteScope.MATCHING_SPECIFIER else None
    return _collect_phase_job_ids(
        registry=registry,
        job_name=phase.prerequisite,
        specifier=specifier,
        dependent_job_id=job_id,
    )


def validate_job_prerequisites(
    registry: Mapping[str, JobState], job_id: str, *, single_recording: bool, submitted_job_ids: frozenset[str]
) -> str | None:
    """Validates that a job's prerequisites either already succeeded or arrive with the same submission.

    The tracker is the authoritative source for phase completion. Files on disk may be corrupt or incomplete even if
    they exist, and the tracker only marks SUCCEEDED when processing is confirmed complete. A prerequisite that is
    submitted alongside the dependent job passes validation because the execution manager admits the dependent job only
    after that prerequisite actually succeeds.

    Args:
        registry: The point-in-time job registry of the tracker that owns the target job.
        job_id: The unique hexadecimal job identifier to validate.
        single_recording: Determines whether to apply single-recording or multi-recording prerequisite rules.
        submitted_job_ids: The identifiers of every job submitted against this tracker in the same call.

    Returns:
        None if all prerequisites are satisfied or pending in this submission, or an error message string describing
        the unmet prerequisite.
    """
    prerequisite_ids, missing_message = resolve_prerequisite_job_ids(
        registry=registry, job_id=job_id, single_recording=single_recording
    )
    if missing_message is not None:
        return missing_message

    for prerequisite_id in prerequisite_ids:
        prerequisite_state = registry[prerequisite_id]
        if prerequisite_state.status == ProcessingStatus.SUCCEEDED or prerequisite_id in submitted_job_ids:
            continue
        return (
            f"Unable to execute job {job_id}. Its prerequisite '{prerequisite_state.job_name}' job "
            f"{prerequisite_id} has not succeeded and is not part of this submission."
        )

    return None


def _collect_phase_job_ids(
    registry: Mapping[str, JobState], job_name: str, specifier: str | None, dependent_job_id: str
) -> tuple[list[str], str | None]:
    """Collects the tracker job IDs belonging to a prerequisite phase and reports an absent phase.

    Args:
        registry: The point-in-time job registry of the tracker that owns the dependent job.
        job_name: The name of the prerequisite phase to collect the jobs of.
        specifier: The specifier the prerequisite job must carry, or None to collect every job of the phase.
        dependent_job_id: The identifier of the job that depends on this phase, used in the error message.

    Returns:
        A tuple of the matching job IDs and an error message, where the message is None unless the phase has no
        matching jobs.
    """
    matches = [
        candidate_id
        for candidate_id, state in registry.items()
        if state.job_name == job_name and (specifier is None or state.specifier == specifier)
    ]

    if not matches:
        scope = "" if specifier is None else f" with specifier '{specifier}'"
        message = (
            f"Unable to execute job {dependent_job_id}. Its prerequisite '{job_name}' phase{scope} is not registered "
            f"in the tracker, so the prerequisite can never be satisfied. Re-run the prepare tool for this recording "
            f"or dataset to register the missing phase."
        )
        return [], message

    return matches, None


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
