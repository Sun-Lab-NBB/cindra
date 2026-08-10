"""Provides the job universe of a recording or a tracked dataset, resolved from what its directories already hold.

The job model names every job a pipeline can run. This module pairs that model with the inventory of what is on disk,
so a scheduler learns both the jobs a recording declares and the subset whose own inputs exist right now. Every
resolver is read-only, and a recording carrying nothing resolves to an empty universe rather than raising.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from ..io import is_recording_processed, resolve_recording_planes, resolve_dataset_recordings
from .jobs import (
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    resolve_multi_recording_jobs,
    resolve_single_recording_jobs,
)
from ..layout import resolve_plane_specifier

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class SingleRecordingJobs:
    """Describes the single-recording jobs one recording declares and the subset whose inputs already exist."""

    output_root: Path
    """The output root the universe was resolved against."""
    plane_count: int
    """The virtual imaging planes the recording holds, which is zero when its parameters were not found."""
    universe: tuple[tuple[str, str], ...] = ()
    """Every job the recording declares, as job name and specifier pairs.

    Notes:
        This is a recording fingerprint rather than an invocation fingerprint, so every invocation aligns a tracker
        against the same set whatever subset it intends to run.
    """
    possible: tuple[tuple[str, str], ...] = ()
    """The subset of the universe whose own input exists on disk right now."""
    resolved: bool = False
    """Determines whether the universe follows from the recording's own parameters rather than from their absence."""


@dataclass(frozen=True, slots=True)
class MultiRecordingJobs:
    """Describes the multi-recording jobs one tracked dataset declares and the subset whose inputs already exist."""

    dataset_name: str
    """The name of the tracked dataset, already lowered to the fold the output directory carries."""
    recording_ids: tuple[str, ...] = ()
    """The identifier of every recording the dataset spans."""
    universe: tuple[tuple[str, str], ...] = ()
    """Every job the dataset declares, as job name and specifier pairs."""
    possible: tuple[tuple[str, str], ...] = ()
    """The subset of the universe whose own input exists on disk right now."""
    resolved: bool = False
    """Determines whether the universe follows from the dataset spanning at least one recording."""


def resolve_single_recording_job_universe(output_root: Path, data_path: Path | None = None) -> SingleRecordingJobs:
    """Resolves the single-recording jobs one recording declares and the subset ready to run.

    Notes:
        The conversion job is ready whenever the recording's parameters resolve. A registration job is ready once its
        plane carries the runtime data the conversion writes, and a processing job once its plane carries the
        reference image the registration writes. The combination job is ready once every plane is registered, which is
        the earliest point its inputs can exist.

    Args:
        output_root: The output root the recording was configured with.
        data_path: The raw imaging directory, consulted only when the recording carries no output yet.

    Returns:
        The recording's job universe.
    """
    inventory = resolve_recording_planes(output_root=output_root, data_path=data_path)
    if not inventory.resolved:
        return SingleRecordingJobs(output_root=output_root, plane_count=0)

    universe = tuple(resolve_single_recording_jobs(plane_count=inventory.plane_count))
    registered = set(inventory.registered_planes)
    converted = {plane_index for plane_index, plane_path in enumerate(inventory.plane_paths) if plane_path.is_dir()}
    every_plane_registered = len(registered) == inventory.plane_count and inventory.plane_count > 0

    possible = tuple(
        (job_name, specifier)
        for job_name, specifier in universe
        if _is_single_recording_job_ready(
            job_name=job_name,
            specifier=specifier,
            converted=converted,
            registered=registered,
            every_plane_registered=every_plane_registered,
        )
    )

    return SingleRecordingJobs(
        output_root=output_root,
        plane_count=inventory.plane_count,
        universe=universe,
        possible=possible,
        resolved=True,
    )


def resolve_multi_recording_job_universe(recording_roots: Sequence[Path], dataset_name: str) -> MultiRecordingJobs:
    """Resolves the multi-recording jobs one tracked dataset declares and the subset ready to run.

    Notes:
        The discovery job is ready once every recording the dataset spans carries its single-recording output, which
        is what the cross-recording registration reads. An extraction job is ready once the discovery job has written
        the template masks its recording projects.

    Args:
        recording_roots: The output root of every recording the dataset spans.
        dataset_name: The name of the tracked dataset, in any casing.

    Returns:
        The dataset's job universe.
    """
    inventory = resolve_dataset_recordings(recording_roots=recording_roots, dataset_name=dataset_name)
    if not inventory.recording_ids:
        return MultiRecordingJobs(dataset_name=inventory.dataset_name)

    universe = tuple(resolve_multi_recording_jobs(recording_ids=list(inventory.recording_ids)))
    every_recording_processed = all(is_recording_processed(output_root=root) for root in inventory.recording_roots)

    possible = tuple(
        (job_name, specifier)
        for job_name, specifier in universe
        if _is_multi_recording_job_ready(
            job_name=job_name,
            every_recording_processed=every_recording_processed,
            discovered=inventory.discovered,
        )
    )

    return MultiRecordingJobs(
        dataset_name=inventory.dataset_name,
        recording_ids=inventory.recording_ids,
        universe=universe,
        possible=possible,
        resolved=True,
    )


def _is_single_recording_job_ready(
    job_name: str,
    specifier: str,
    converted: set[int],
    registered: set[int],
    *,
    every_plane_registered: bool,
) -> bool:
    """Determines whether one single-recording job's own input exists on disk.

    Args:
        job_name: The pipeline stage the job runs.
        specifier: The job's tracker specifier, which names a plane for the per-plane stages.
        converted: The indices of the planes carrying an output directory.
        registered: The indices of the planes carrying registration output.
        every_plane_registered: Determines whether every plane of the recording is registered.

    Returns:
        True when the job's input exists.
    """
    if job_name == SingleRecordingJobNames.BINARIZE:
        return True
    if job_name == SingleRecordingJobNames.COMBINE:
        return every_plane_registered

    ready_planes = converted if job_name == SingleRecordingJobNames.REGISTER else registered
    return any(resolve_plane_specifier(plane_index=plane_index) == specifier for plane_index in ready_planes)


def _is_multi_recording_job_ready(job_name: str, *, every_recording_processed: bool, discovered: bool) -> bool:
    """Determines whether one multi-recording job's own input exists on disk.

    Args:
        job_name: The pipeline stage the job runs.
        every_recording_processed: Determines whether every recording carries its single-recording output.
        discovered: Determines whether the dataset carries the tracked template masks.

    Returns:
        True when the job's input exists.
    """
    if job_name == MultiRecordingJobNames.DISCOVER:
        return every_recording_processed
    return discovered
