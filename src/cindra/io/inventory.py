"""Provides the read-only inventory of what a recording or a tracked dataset already holds on disk, which an external
scheduler reads to size and sequence its own work without driving the pipeline.

Every resolver here reads the acquisition parameters and stats the output tree. None of them decodes an image, builds a
runtime context, or creates a directory, so asking what a recording holds costs a few small reads and changes nothing.
A recording that carries neither parameters nor output resolves to an empty record rather than raising, because an
absent recording is an answer rather than a failure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import field, dataclass

from ..layout import (
    CHANNEL_1_BINARY_FILENAME,
    COMBINED_METADATA_FILENAME,
    ACQUISITION_PARAMETERS_FILENAME,
    REGISTRATION_DATA_DIRECTORY_NAME,
    TRACKING_TEMPLATE_MASKS_FILENAME,
    RecordingArrays,
    RegistrationArrays,
    resolve_plane_path,
    resolve_output_path,
    resolve_dataset_path,
    resolve_plane_specifier,
)
from .context import PARAMETERS_FILENAME, extract_unique_components, load_acquisition_parameters
from ..dataclasses import AcquisitionParameters

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Sequence


@dataclass(frozen=True, slots=True)
class RecordingPlanes:
    """Describes the virtual imaging planes one recording holds and how far each of them has been processed."""

    output_root: Path
    """The output root the inventory was resolved against."""
    plane_count: int
    """The number of virtual imaging planes the recording holds, which is zero when its parameters were not found."""
    plane_paths: tuple[Path, ...] = ()
    """The output directory of every virtual plane, ordered by plane index."""
    plane_specifiers: tuple[str, ...] = ()
    """The specifier identifying every virtual plane, ordered by plane index."""
    registered_planes: tuple[int, ...] = ()
    """The indices of the planes carrying registration output, which are the planes the processing stage can run."""
    processed: bool = False
    """Determines whether the recording carries the combined metadata archive that marks its pipeline complete."""
    resolved: bool = False
    """Determines whether the plane count follows from the recording's own acquisition parameters rather than from
    their absence."""


@dataclass(frozen=True, slots=True)
class DatasetRecordings:
    """Describes the recordings one tracked multi-recording dataset spans and how far the dataset has been processed."""

    dataset_name: str
    """The name of the tracked dataset, already lowered to the fold the output directory carries."""
    recording_roots: tuple[Path, ...] = field(default=())
    """The output root of every recording the dataset spans, in the order the caller supplied them."""
    recording_ids: tuple[str, ...] = ()
    """The identifier of every recording, derived from the component of its path that distinguishes it."""
    dataset_paths: tuple[Path, ...] = ()
    """The dataset output directory inside every recording's output directory."""
    extracted_recordings: tuple[str, ...] = ()
    """The identifiers of the recordings whose tracked fluorescence has already been extracted."""
    discovered: bool = False
    """Determines whether the dataset carries the tracked template mask archive.

    Notes:
        The archive is written by the clustering step, which the projection back into each recording still follows, so
        it marks the dataset as tracked rather than the discovery job as finished.
    """


def resolve_recording_planes(output_root: Path, data_path: Path | None = None) -> RecordingPlanes:
    """Resolves the virtual imaging planes one recording holds and how far each of them has been processed.

    Notes:
        The plane count is read from the acquisition parameters the recording's output directory carries, falling back
        to the parameters file in the raw imaging directory when the recording has not been processed yet. A recording
        offering neither resolves to a record whose plane count is zero and whose resolved flag is False, which a
        caller reads as a floor to plan around rather than as an error.

        The plane count follows the same rule the context resolver applies, so the planes named here are the planes
        the pipeline creates.

    Args:
        output_root: The output root the recording was configured with.
        data_path: The raw imaging directory, consulted only when the output directory carries no acquisition
            parameters. Use None when the recording has already been processed.

    Returns:
        The recording's plane inventory.
    """
    acquisition = resolve_acquisition_parameters(output_root=output_root, data_path=data_path)
    if acquisition is None:
        return RecordingPlanes(output_root=output_root, plane_count=0)

    plane_count = acquisition.virtual_plane_count if acquisition.is_mroi else acquisition.plane_number
    plane_paths = tuple(
        resolve_plane_path(output_root=output_root, plane_index=plane_index) for plane_index in range(plane_count)
    )
    plane_specifiers = tuple(resolve_plane_specifier(plane_index=plane_index) for plane_index in range(plane_count))
    registered_planes = tuple(
        plane_index
        for plane_index in range(plane_count)
        if is_plane_registered(output_root=output_root, plane_index=plane_index)
    )

    return RecordingPlanes(
        output_root=output_root,
        plane_count=plane_count,
        plane_paths=plane_paths,
        plane_specifiers=plane_specifiers,
        registered_planes=registered_planes,
        processed=is_recording_processed(output_root=output_root),
        resolved=True,
    )


def resolve_dataset_recordings(recording_roots: Sequence[Path], dataset_name: str) -> DatasetRecordings:
    """Resolves the recordings one tracked dataset spans and how far the dataset has been processed.

    Notes:
        The dataset is considered discovered when the first recording's dataset directory carries the template mask
        archive, matching where the discovery stage writes it. A recording is considered extracted when its own
        dataset directory carries the tracked fluorescence trace, which only the extraction stage writes.

    Args:
        recording_roots: The output root of every recording the dataset spans.
        dataset_name: The name of the tracked dataset, in any casing.

    Returns:
        The dataset's recording inventory.
    """
    roots = tuple(recording_roots)
    if not roots:
        return DatasetRecordings(dataset_name=dataset_name.lower())

    dataset_paths = tuple(resolve_dataset_path(output_root=root, dataset_name=dataset_name) for root in roots)
    recording_ids = extract_unique_components(paths=roots)
    extracted_recordings = tuple(
        recording_id
        for recording_id, dataset_path in zip(recording_ids, dataset_paths, strict=True)
        if (dataset_path / RecordingArrays.CELL_FLUORESCENCE).exists()
    )

    return DatasetRecordings(
        dataset_name=dataset_name.lower(),
        recording_roots=roots,
        recording_ids=recording_ids,
        dataset_paths=dataset_paths,
        extracted_recordings=extracted_recordings,
        discovered=is_dataset_discovered(output_root=roots[0], dataset_name=dataset_name),
    )


def is_recording_processed(output_root: Path) -> bool:
    """Determines whether a recording's single-recording pipeline has completed.

    Notes:
        Reads the combined metadata archive, which the combination stage publishes after its payload arrays through an
        atomic write, so its presence marks every array it describes as already on disk.

    Args:
        output_root: The output root the recording was configured with.

    Returns:
        True when the recording carries the completion marker.
    """
    return (resolve_output_path(output_root=output_root) / COMBINED_METADATA_FILENAME).exists()


def is_plane_registered(output_root: Path, plane_index: int) -> bool:
    """Determines whether one virtual imaging plane carries registration output.

    Args:
        output_root: The output root the recording was configured with.
        plane_index: The index of the virtual imaging plane.

    Returns:
        True when the plane carries the reference image the registration stage writes.
    """
    plane_path = resolve_plane_path(output_root=output_root, plane_index=plane_index)
    return (plane_path / REGISTRATION_DATA_DIRECTORY_NAME / RegistrationArrays.REFERENCE_IMAGE).exists()


def is_plane_converted(output_root: Path, plane_index: int) -> bool:
    """Determines whether one virtual imaging plane carries the binary the conversion stage writes.

    Args:
        output_root: The output root the recording was configured with.
        plane_index: The index of the virtual imaging plane.

    Returns:
        True when the plane carries its functional channel binary.
    """
    plane_path = resolve_plane_path(output_root=output_root, plane_index=plane_index)
    return (plane_path / CHANNEL_1_BINARY_FILENAME).exists()


def is_plane_processed(output_root: Path, plane_index: int) -> bool:
    """Determines whether one virtual imaging plane carries the traces the processing stage writes.

    Notes:
        The traces are what the combination stage concatenates, so this is the condition under which a combination
        job's own inputs exist.

    Args:
        output_root: The output root the recording was configured with.
        plane_index: The index of the virtual imaging plane.

    Returns:
        True when the plane carries its extracted fluorescence trace.
    """
    plane_path = resolve_plane_path(output_root=output_root, plane_index=plane_index)
    return (plane_path / RecordingArrays.CELL_FLUORESCENCE).exists()


def is_recording_extractable(output_root: Path, dataset_name: str) -> bool:
    """Determines whether one recording carries the tracked masks its own extraction job reads.

    Notes:
        The discovery stage projects a template mask set back into every recording it spans, and the extraction stage
        refuses to run without that recording's own projected statistics. The dataset-wide template archive marks the
        clustering step rather than the projection, so it is not the condition an extraction job waits on.

    Args:
        output_root: The output root the recording was configured with.
        dataset_name: The name of the tracked dataset, in any casing.

    Returns:
        True when the recording carries its own projected ROI statistics.
    """
    dataset_path = resolve_dataset_path(output_root=output_root, dataset_name=dataset_name)
    return (dataset_path / RecordingArrays.ROI_STATISTICS).exists()


def is_dataset_discovered(output_root: Path, dataset_name: str) -> bool:
    """Determines whether a tracked dataset's discovery phase has completed.

    Args:
        output_root: The output root of the recording holding the dataset directory.
        dataset_name: The name of the tracked dataset, in any casing.

    Returns:
        True when the dataset carries the tracked template mask archive, which the clustering step writes partway
        through the discovery job.
    """
    dataset_path = resolve_dataset_path(output_root=output_root, dataset_name=dataset_name)
    return (dataset_path / TRACKING_TEMPLATE_MASKS_FILENAME).exists()


def resolve_acquisition_parameters(output_root: Path, data_path: Path | None) -> AcquisitionParameters | None:
    """Resolves a recording's acquisition parameters from its output directory or its raw imaging directory.

    Args:
        output_root: The output root the recording was configured with.
        data_path: The raw imaging directory, consulted only when the output directory carries no parameters.

    Returns:
        The recording's acquisition parameters, or None when neither directory carries them.
    """
    saved_path = resolve_output_path(output_root=output_root) / ACQUISITION_PARAMETERS_FILENAME
    if saved_path.exists():
        return AcquisitionParameters.from_yaml(file_path=saved_path)

    if data_path is None or not data_path.is_dir():
        return None

    candidates = sorted(data_path.rglob(PARAMETERS_FILENAME))
    if not candidates:
        return None
    return load_acquisition_parameters(json_path=candidates[0])
