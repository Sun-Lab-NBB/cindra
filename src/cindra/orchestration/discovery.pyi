from pathlib import Path
from dataclasses import dataclass
from collections.abc import Sequence

from ..io import (
    is_plane_converted as is_plane_converted,
    is_plane_processed as is_plane_processed,
    is_recording_processed as is_recording_processed,
    is_recording_extractable as is_recording_extractable,
    resolve_recording_planes as resolve_recording_planes,
    resolve_dataset_recordings as resolve_dataset_recordings,
)
from .jobs import (
    MultiRecordingJobNames as MultiRecordingJobNames,
    SingleRecordingJobNames as SingleRecordingJobNames,
    resolve_multi_recording_jobs as resolve_multi_recording_jobs,
    resolve_single_recording_jobs as resolve_single_recording_jobs,
)
from ..layout import parse_plane_specifier as parse_plane_specifier

@dataclass(frozen=True, slots=True)
class SingleRecordingJobs:
    output_root: Path
    plane_count: int
    universe: tuple[tuple[str, str], ...] = ...
    possible: tuple[tuple[str, str], ...] = ...
    resolved: bool = ...

@dataclass(frozen=True, slots=True)
class MultiRecordingJobs:
    dataset_name: str
    recording_ids: tuple[str, ...] = ...
    universe: tuple[tuple[str, str], ...] = ...
    possible: tuple[tuple[str, str], ...] = ...
    resolved: bool = ...

def resolve_single_recording_job_universe(output_root: Path, data_path: Path | None = None) -> SingleRecordingJobs: ...
def resolve_multi_recording_job_universe(recording_roots: Sequence[Path], dataset_name: str) -> MultiRecordingJobs: ...
def _is_single_recording_job_ready(
    job_name: str, specifier: str, converted: set[int], registered: set[int], *, every_plane_processed: bool
) -> bool: ...
def _is_multi_recording_job_ready(
    job_name: str, specifier: str, *, every_recording_processed: bool, extractable: set[str]
) -> bool: ...
