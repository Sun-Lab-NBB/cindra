"""Provides the job universe of a recording or a tracked dataset, resolved from what its directories already hold. Every
resolver is read-only, and a recording carrying nothing resolves to an empty universe rather than raising.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

from ..io import (
    is_plane_converted,
    is_plane_processed,
    is_recording_processed,
    is_recording_extractable,
    resolve_recording_planes,
    resolve_dataset_recordings,
)
from .jobs import (
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    resolve_multi_recording_jobs,
    resolve_single_recording_jobs,
)
from ..layout import parse_plane_specifier

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class SingleRecordingJobs:
    """Describes the single-recording jobs one recording declares and the subset whose inputs already exist."""

    output_root: Path
    """The output root the universe resolution used."""
    plane_count: int
    """The virtual imaging planes the recording holds, which is zero when its parameters were not found."""
    universe: tuple[tuple[str, str], ...] = ()
    """Every job the recording declares, as job name and specifier pairs.

    Notes:
        This is a recording fingerprint rather than an invocation fingerprint, so every invocation aligns a tracker
        against the same set whatever subset it intends to run.
    """
    possible: tuple[tuple[str, str], ...] = ()
    """The subset of the universe whose own input exists on disk right now.

    Notes:
        The conversion job is reported ready whenever the recording's parameters resolve, which is the weakest of the
        four conditions, because its own input is the raw image set this record does not read.
    """
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
        plane carries the channel binary the conversion writes, and a processing job once its plane carries the
        reference image the registration writes. The combination job is ready once every plane carries the traces the
        processing stage writes, which are the arrays the combination stage concatenates.

    Args:
        output_root: The recording's configured output root.
        data_path: The raw imaging directory, consulted only when the recording carries no output yet.

    Returns:
        The recording's job universe.
    """
    inventory = resolve_recording_planes(output_root=output_root, data_path=data_path)
    if not inventory.resolved:
        return SingleRecordingJobs(output_root=output_root, plane_count=0)

    universe = tuple(resolve_single_recording_jobs(plane_count=inventory.plane_count))
    planes = range(inventory.plane_count)
    converted = {index for index in planes if is_plane_converted(output_root=output_root, plane_index=index)}
    registered = set(inventory.registered_planes)
    processed = {index for index in planes if is_plane_processed(output_root=output_root, plane_index=index)}
    every_plane_processed = len(processed) == inventory.plane_count and inventory.plane_count > 0

    possible = tuple(
        (job_name, specifier)
        for job_name, specifier in universe
        if _is_single_recording_job_ready(
            job_name=job_name,
            specifier=specifier,
            converted=converted,
            registered=registered,
            every_plane_processed=every_plane_processed,
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
    extractable = {
        recording_id
        for recording_id, root in zip(inventory.recording_ids, inventory.recording_roots, strict=True)
        if is_recording_extractable(output_root=root, dataset_name=dataset_name)
    }

    possible = tuple(
        (job_name, specifier)
        for job_name, specifier in universe
        if _is_multi_recording_job_ready(
            job_name=job_name,
            specifier=specifier,
            every_recording_processed=every_recording_processed,
            extractable=extractable,
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
    every_plane_processed: bool,
) -> bool:
    """Determines whether one single-recording job's own input exists on disk.

    Args:
        job_name: The pipeline stage the job runs.
        specifier: The job's tracker specifier, which names a plane for the per-plane stages.
        converted: The indices of the planes carrying the channel binary the conversion stage writes.
        registered: The indices of the planes carrying registration output.
        every_plane_processed: Determines whether every plane of the recording carries its extracted traces.

    Returns:
        True when the job's input exists.
    """
    if job_name == SingleRecordingJobNames.BINARIZE:
        return True
    if job_name == SingleRecordingJobNames.COMBINE:
        return every_plane_processed

    ready_planes = converted if job_name == SingleRecordingJobNames.REGISTER else registered
    plane_index = parse_plane_specifier(specifier=specifier)
    return plane_index is not None and plane_index in ready_planes


def _is_multi_recording_job_ready(
    job_name: str, specifier: str, *, every_recording_processed: bool, extractable: set[str]
) -> bool:
    """Determines whether one multi-recording job's own input exists on disk.

    Args:
        job_name: The pipeline stage the job runs.
        specifier: The job's tracker specifier, which names a recording for the extraction stage.
        every_recording_processed: Determines whether every recording carries its single-recording output.
        extractable: The identifiers of the recordings carrying their own projected ROI statistics.

    Returns:
        True when the job's input exists.
    """
    if job_name == MultiRecordingJobNames.DISCOVER:
        return every_recording_processed
    return specifier in extractable
