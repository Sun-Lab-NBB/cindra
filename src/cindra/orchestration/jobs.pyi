from enum import StrEnum
from dataclasses import dataclass
from collections.abc import Mapping, Iterable, Sequence

from ataraxis_data_structures import JobState as JobState

from ..layout import resolve_plane_specifier as resolve_plane_specifier

PREREQUISITE_FAILURE_MESSAGE: str
UNREACHABLE_PREREQUISITE_MESSAGE: str

class SingleRecordingJobNames(StrEnum):
    BINARIZE = "binarization"
    REGISTER = "registration"
    PROCESS = "processing"
    COMBINE = "combination"

class MultiRecordingJobNames(StrEnum):
    DISCOVER = "discovery"
    EXTRACT = "extraction"

class PrerequisiteScope(StrEnum):
    ALL_JOBS = "all_jobs"
    MATCHING_SPECIFIER = "matching_specifier"

@dataclass(frozen=True, slots=True)
class PipelinePhase:
    job_name: str
    per_specifier: bool
    prerequisite: str | None
    prerequisite_scope: PrerequisiteScope

SINGLE_RECORDING_PHASES: tuple[PipelinePhase, ...]
MULTI_RECORDING_PHASES: tuple[PipelinePhase, ...]
PER_PLANE_JOB_NAMES: frozenset[str]

def resolve_single_recording_jobs(plane_count: int) -> list[tuple[str, str]]: ...
def resolve_multi_recording_jobs(recording_ids: Sequence[str]) -> list[tuple[str, str]]: ...
def generate_job_ids(jobs: Iterable[tuple[str, str]]) -> dict[tuple[str, str], str]: ...
def resolve_single_recording_prerequisites(
    jobs: Iterable[tuple[str, str]],
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]: ...
def resolve_multi_recording_prerequisites(
    jobs: Iterable[tuple[str, str]],
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]: ...
def resolve_downstream_phases(phase_names: Iterable[str], *, single_recording: bool) -> set[str]: ...
def order_phases_by_execution(phase_names: Iterable[str], *, single_recording: bool) -> list[str]: ...
def resolve_pipeline_jobs(phases: tuple[PipelinePhase, ...], specifiers: Sequence[str]) -> list[tuple[str, str]]: ...
def resolve_prerequisite_job_ids(
    registry: Mapping[str, JobState], job_id: str, *, single_recording: bool
) -> tuple[list[str], str | None]: ...
def validate_job_prerequisites(
    registry: Mapping[str, JobState], job_id: str, *, single_recording: bool, submitted_job_ids: frozenset[str]
) -> str | None: ...
def _collect_phase_job_ids(
    registry: Mapping[str, JobState], job_name: str, specifier: str | None, dependent_job_id: str
) -> tuple[list[str], str | None]: ...
def _resolve_prerequisites(
    jobs: Iterable[tuple[str, str]], phases: tuple[PipelinePhase, ...]
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]: ...
