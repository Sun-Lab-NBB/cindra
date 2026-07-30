from enum import StrEnum
from dataclasses import dataclass
from collections.abc import Iterable, Sequence

from .job_names import (
    MultiRecordingJobNames as MultiRecordingJobNames,
    SingleRecordingJobNames as SingleRecordingJobNames,
)

PLANE_SPECIFIER_PREFIX: str

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

def resolve_plane_specifier(plane_index: int) -> str: ...
def resolve_single_recording_jobs(plane_count: int) -> list[tuple[str, str]]: ...
def resolve_multi_recording_jobs(recording_ids: Sequence[str]) -> list[tuple[str, str]]: ...
def resolve_single_recording_prerequisites(
    jobs: Iterable[tuple[str, str]],
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]: ...
def resolve_multi_recording_prerequisites(
    jobs: Iterable[tuple[str, str]],
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]: ...
def resolve_downstream_phases(phase_names: Iterable[str], *, single_recording: bool) -> set[str]: ...
def resolve_pipeline_jobs(phases: tuple[PipelinePhase, ...], specifiers: Sequence[str]) -> list[tuple[str, str]]: ...
def _resolve_prerequisites(
    jobs: Iterable[tuple[str, str]], phases: tuple[PipelinePhase, ...]
) -> dict[tuple[str, str], tuple[tuple[str, str], ...]]: ...
