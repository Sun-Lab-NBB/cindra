"""Provides the per-stage memory models of the single and multi-recording pipeline jobs, which project each stage's
peak anonymous working set from the shape of the data it processes.

Every model here is read from the stage's own allocations rather than measured from the outside, so a change to what a
kernel holds is a change to the model beside it. The estimates cover anonymous memory alone, which is the term that
forces a host to swap and a scheduler to kill a job, so the reclaimable pages a memory-mapping stage leaves resident
are excluded. Each estimate carries a flag stating whether it follows from the recording's own geometry, so a caller
holding an unprocessed recording reads a floor to plan around rather than handling an exception.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
from numpy.lib.format import read_magic, read_array_header_1_0, read_array_header_2_0

from ..io import RecordingPlanes, resolve_recording_planes, resolve_acquisition_parameters
from .jobs import MultiRecordingJobNames, SingleRecordingJobNames
from ..layout import (
    COMBINED_METADATA_FILENAME,
    RecordingArrays,
    resolve_array_path,
    resolve_output_path,
    resolve_dataset_path,
    parse_plane_specifier,
)
from ..dataclasses import SingleRecordingRuntimeData

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Sequence

    from ..dataclasses import (
        AcquisitionParameters,
        MultiRecordingConfiguration,
        SingleRecordingConfiguration,
    )

MEMORY_ESTIMATE_TOLERANCE: float = 1.15
"""The margin applied to every estimate before it is reported.

Notes:
    The margin covers the working sets a model does not enumerate and the variation between recordings of the same
    shape. Understating is the asymmetric failure, since a local batch overcommits its host and a scheduled job is
    killed outright, so every estimate rounds up. The value matches the margin the sibling ataraxis libraries apply,
    which is what lets a scheduler weigh a cindra job against theirs on one scale.
"""

WORKER_MEMORY_MB: int = 384
"""The resident memory a worker process occupies before it runs any job, covering the interpreter and the import graph
of this library. The term is charged once per job."""

SPAWNED_CHILD_MEMORY_MB: int = 200
"""The resident memory each child of a job's own worker pool occupies before it touches data. A worker is spawned
rather than forked, so every child re-imports the module its target function lives in."""

_BYTES_PER_MEGABYTE: int = 1024 * 1024
"""The divisor converting a byte count into megabytes."""

_MEGABYTES_PER_GIGABYTE: int = 1024
"""The megabytes one gigabyte holds, which every reportable estimate is rounded up to a multiple of."""

_SINGLE_PRECISION_BYTES: int = 4
"""The width of one single-precision element, which is the type every modeled working array holds."""

_BINARIZATION_BYTES_PER_PIXEL: int = 8
"""The memory the conversion stage holds per pixel of the batch it reads.

Notes:
    The stage decodes a batch, halves it, clips the halved copy, and casts the clipped copy to the internal width. A
    unsigned 16-bit source therefore holds the decoded batch, the halved copy, the clipped copy, and the cast result
    at once, each two bytes per pixel.
"""

_DETECTION_ARRAY_COPIES: int = 3
"""The copies of the binned movie the detection stage holds at its peak.

Notes:
    The peak is the temporal standard deviation, which holds the binned frames, their successive differences, and the
    squared differences at once. The reduction over the squared differences returns a two-dimensional image, so it
    adds no further copy at movie scale.
"""

_DETECTION_DENOISE_ARRAY_COPIES: int = 7
"""The copies of the binned movie the detection stage holds when PCA denoising runs.

Notes:
    Denoising allocates a reconstruction the size of the movie and holds the centered blocks and their reconstructions
    at once. The blocks overlap by half in each axis, so both block lists together hold roughly four and a half copies
    of the movie beside the frames and the reconstruction.
"""

_REGISTRATION_METRIC_ARRAY_COPIES: int = 3
"""The copies of the sampled movie the registration quality metrics hold at once, which are the single-precision
subsample, the mean-centered copy, and the reordering the principal component fit makes of its transposed input."""

_REFERENCE_STAGE_ARRAY_COPIES: int = 5
"""The copies of the reference sample the reference image stage holds at once, covering the single-precision sample,
its mean-centered and normalized copies, and the transforms the phase correlation allocates."""

_MINIMUM_METRIC_SAMPLE_COUNT: int = 2000
"""The frames the registration metrics sample from a recording that is short or whose planes are large."""

_MAXIMUM_METRIC_SAMPLE_COUNT: int = 5000
"""The frames the registration metrics sample from a long recording whose planes are small."""

_MAXIMUM_EXTENT_FOR_LARGE_SAMPLE: int = 700
"""The plane height or width above which the registration metrics fall back to the smaller sample.

Notes:
    This threshold is what makes the registration footprint move opposite to the plane extent. A plane crossing it
    samples fewer frames, so a taller plane holds a smaller working set than a shorter one.
"""

_MINIMUM_METRIC_FRAME_COUNT: int = 1500
"""The frames a plane must hold before the registration quality metrics run at all."""

_COMBINATION_TRACE_KINDS: int = 4
"""The trace arrays the combination stage concatenates, which are the raw, neuropil, subtracted, and spike traces.

Notes:
    The per-plane sources are memory-mapped, so the anonymous peak is one concatenated copy of each kind rather than
    the sources plus the result.
"""

_EXTRACTION_TRACE_COPIES: int = 5
"""The full trace arrays the extraction stage holds at its peak.

Notes:
    The stage returns the raw, neuropil, subtracted, and spike traces together, so all four are live at once, and the
    baseline filter that produces the subtracted trace holds a fifth array of the same size.
"""

_EXTRACTION_BATCH_BYTES_PER_PIXEL: int = 6
"""The memory one extraction batch holds per combined pixel, covering the batch at its stored width and the
single-precision copy the kernel consumes."""

_OASIS_WORKSPACE_BYTES: int = 16
"""The memory the deconvolution workspace holds per region and sample of the batch it processes, covering the pool
amplitude, weight, start frame, and length arrays."""

_DISCOVERY_PLANES_PER_RECORDING: int = 12
"""The single-precision planes a discovery job holds per recording.

Notes:
    The planes cover each recording's reference image, its scale-space pyramid, its accumulated and cached deformation
    fields, and its transformed reference images. The count is linear in the recording count, because the groupwise
    registration visits each unordered pair once and caches one image and one gradient per image rather than one
    deformation per pair.
"""

_DISCOVERY_TRANSIENT_PLANES: int = 10
"""The single-precision planes one pairwise deformation holds while it is computed. The term is constant, because one
pair is in flight at a time whatever the group size."""

_TRACE_ARRAY_DIMENSIONS: int = 2
"""The axes a trace array carries, which are its regions and its samples."""

_TRACKING_CLUSTERING_MEMORY_MB: int = 5632
"""The memory the cross-recording clustering stage is charged.

Notes:
    The stage builds a pairwise matrix over the regions falling inside one spatial bin, so its size follows local
    region crowding rather than any count the processed data reports. The allowance covers the crowding a dense
    recording produces.
"""


_STAGE_FALLBACK_MEGABYTES: dict[str, int] = {
    SingleRecordingJobNames.BINARIZE: 4096,
    SingleRecordingJobNames.REGISTER: 14848,
    SingleRecordingJobNames.PROCESS: 15360,
    SingleRecordingJobNames.COMBINE: 8192,
    MultiRecordingJobNames.DISCOVER: 16384,
    MultiRecordingJobNames.EXTRACT: 8192,
}
"""The memory each stage is charged when its own geometry cannot be read.

Notes:
    An estimate that cannot be derived falls back to a conservative allowance rather than to a floor, because
    understating is the asymmetric failure. A job admitted against a floor overcommits its host and is killed, while a
    job admitted against an allowance merely waits longer than it had to.

    The registration, processing, combination, and discovery figures are the widest footprints those stages have been
    observed to reach. The processing figure is the measured nine-plane peak of 10.5 gigabytes rounded up to cover the
    taller planes of the same recording. The binarization and extraction figures are allowances rather than observed
    peaks, since neither stage has been measured at its widest, and both sit above every projection their own models
    produce for the recordings this corpus holds.
"""


@dataclass(frozen=True, slots=True)
class PlaneGeometry:
    """Describes the shape of one virtual imaging plane as its binarized output reports it."""

    height: int
    """The height of the plane in pixels."""
    width: int
    """The width of the plane in pixels."""
    frame_count: int
    """The frames the plane holds."""
    sampling_rate: float
    """The rate at which the recording sampled this plane."""


@dataclass(frozen=True, slots=True)
class RecordingGeometry:
    """Describes the shape of one recording as its own output and acquisition parameters report it."""

    planes: tuple[PlaneGeometry, ...] = ()
    """The geometry of every virtual imaging plane, ordered by plane index."""
    raw_frame_pixels: int = 0
    """The pixels one unsliced acquisition frame holds, which the conversion stage reads a batch of at a time."""
    combined_pixels: int = 0
    """The pixels one combined multi-plane frame holds, which every multi-recording stage works at."""
    combined_frame_count: int = 0
    """The frames the combined view holds, trimmed to the shortest contributing plane."""
    region_count: int = 0
    """The regions the pipeline detected across every plane, which is zero before extraction has run."""
    resolved: bool = False
    """Determines whether the geometry follows from the recording's own data rather than from its absence."""


def resolve_recording_geometry(output_root: Path, data_path: Path | None = None) -> RecordingGeometry:
    """Resolves the shape of one recording from the output its earlier stages wrote.

    Notes:
        Reads the per-plane runtime data, the combined metadata archive, and the header of the combined trace array.
        No frame is decoded and no array is mapped, so the cost stays flat as a recording grows.

    Args:
        output_root: The output root the recording was configured with.
        data_path: The raw imaging directory, consulted only for the plane count when the recording carries no
            output yet.

    Returns:
        The recording's geometry, whose resolved flag is False when the recording carries no readable output.
    """
    inventory = resolve_recording_planes(output_root=output_root, data_path=data_path)
    acquisition = resolve_acquisition_parameters(output_root=output_root, data_path=data_path)
    planes = _read_plane_geometries(inventory=inventory)
    combined_pixels, combined_frame_count = _read_combined_geometry(output_root=output_root)
    region_count = _read_region_count(
        array_path=resolve_array_path(
            root_path=resolve_output_path(output_root=output_root), array=RecordingArrays.CELL_FLUORESCENCE
        )
    )

    return RecordingGeometry(
        planes=planes,
        raw_frame_pixels=_resolve_raw_frame_pixels(planes=planes, acquisition=acquisition),
        combined_pixels=combined_pixels,
        combined_frame_count=combined_frame_count,
        region_count=region_count,
        resolved=bool(planes) or combined_pixels > 0,
    )


def estimate_single_recording_job_memory_mb(
    job_name: SingleRecordingJobNames,
    specifier: str,
    output_root: Path,
    configuration: SingleRecordingConfiguration,
    data_path: Path | None = None,
) -> tuple[int, bool]:
    """Estimates the memory one single-recording job occupies at its peak.

    Notes:
        A per-plane job whose specifier names a plane is estimated from that plane alone. A per-plane job whose
        specifier does not resolve is charged the largest per-plane estimate, so an unmatched job never understates.

    Args:
        job_name: The pipeline stage the job runs.
        specifier: The job's tracker specifier, which names a plane for the per-plane stages and is empty otherwise.
        output_root: The output root the recording was configured with.
        configuration: The recording's processing configuration.
        data_path: The raw imaging directory, consulted only when the recording carries no output yet.

    Returns:
        The memory the job occupies in megabytes, and a flag that is True when the figure follows from the recording's
        own geometry rather than from the worker baseline alone.
    """
    geometry = resolve_recording_geometry(output_root=output_root, data_path=data_path)
    if not geometry.resolved:
        return _resolve_stage_fallback(job_name=job_name), False

    if job_name == SingleRecordingJobNames.BINARIZE:
        return _apply_tolerance(
            memory_mb=_estimate_binarization_mb(geometry=geometry, configuration=configuration)
        ), True

    if job_name == SingleRecordingJobNames.COMBINE:
        return _apply_tolerance(memory_mb=_estimate_combination_mb(geometry=geometry)), True

    if not geometry.planes:
        return _resolve_stage_fallback(job_name=job_name), False

    estimator = _estimate_registration_mb if job_name == SingleRecordingJobNames.REGISTER else _estimate_processing_mb
    estimates = [estimator(plane=plane, configuration=configuration) for plane in geometry.planes]
    plane_index = _resolve_specifier_index(specifier=specifier, count=len(estimates))
    memory_mb = estimates[plane_index] if plane_index is not None else max(estimates)
    return _apply_tolerance(memory_mb=memory_mb), True


def estimate_multi_recording_job_memory_mb(
    job_name: MultiRecordingJobNames,
    specifier: str,
    recording_roots: Sequence[Path],
    dataset_name: str,
    configuration: MultiRecordingConfiguration,
) -> tuple[int, bool]:
    """Estimates the memory one multi-recording job occupies at its peak.

    Notes:
        A discovery job spans every recording of the dataset, so it is estimated from all of them. An extraction job
        runs on one recording, which its specifier names, and is estimated from that recording alone.

    Args:
        job_name: The pipeline stage the job runs.
        specifier: The job's tracker specifier, which names a recording for the extraction stage.
        recording_roots: The output root of every recording the dataset spans.
        dataset_name: The name of the tracked dataset.
        configuration: The dataset's processing configuration.

    Returns:
        The memory the job occupies in megabytes, and a flag that is True when the figure follows from the dataset's
        own geometry rather than from the worker baseline alone.
    """
    geometries = [resolve_recording_geometry(output_root=root) for root in recording_roots]
    resolved = [geometry for geometry in geometries if geometry.combined_pixels > 0]
    if not resolved:
        return _resolve_stage_fallback(job_name=job_name), False

    if job_name == MultiRecordingJobNames.DISCOVER:
        return _apply_tolerance(memory_mb=_estimate_discovery_mb(geometries=resolved)), True

    target = _resolve_target_geometry(
        recording_roots=recording_roots, geometries=geometries, resolved=resolved, specifier=specifier
    )
    regions = _read_region_count(
        array_path=resolve_array_path(
            root_path=resolve_dataset_path(
                output_root=recording_roots[geometries.index(target)], dataset_name=dataset_name
            ),
            array=RecordingArrays.CELL_FLUORESCENCE,
        )
    )
    return _apply_tolerance(
        memory_mb=_estimate_extraction_mb(
            geometry=target, tracked_regions=regions or target.region_count, configuration=configuration
        )
    ), True


def _estimate_binarization_mb(geometry: RecordingGeometry, configuration: SingleRecordingConfiguration) -> int:
    """Estimates the memory the conversion stage holds, from the raw frame batch it decodes.

    Notes:
        The stage rounds its configured batch up to a whole interleave stride, which adds fewer frames than one
        stride and therefore stays inside the margin every estimate carries.

    Args:
        geometry: The recording's geometry.
        configuration: The recording's processing configuration.

    Returns:
        The memory the stage holds in megabytes, before the shared tolerance.
    """
    batch_bytes = configuration.registration.batch_size * geometry.raw_frame_pixels * _BINARIZATION_BYTES_PER_PIXEL
    return WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=batch_bytes)


def _resolve_raw_frame_pixels(planes: Sequence[PlaneGeometry], acquisition: AcquisitionParameters | None) -> int:
    """Resolves the pixels one unsliced acquisition frame holds.

    Notes:
        A recording interleaving several regions into one frame holds every region in each acquisition frame, so its
        raw frame spans the regions together. A recording imaging one region holds one plane per acquisition frame
        however many planes it images, so its raw frame spans a single plane.

    Args:
        planes: The geometry of every virtual plane.
        acquisition: The recording's acquisition parameters, or None when they were not readable.

    Returns:
        The pixels one raw acquisition frame holds, or zero when the recording holds no readable plane.
    """
    if not planes:
        return 0
    if acquisition is not None and acquisition.is_mroi:
        return sum(plane.height * plane.width for plane in planes) // max(1, acquisition.plane_number)
    return max(plane.height * plane.width for plane in planes)


def _estimate_registration_mb(plane: PlaneGeometry, configuration: SingleRecordingConfiguration) -> int:
    """Estimates the memory one plane registration job holds, from the samples its stages read.

    Notes:
        The peak is the larger of the reference image stage and the registration quality metrics. The metric sample
        count steps down when the plane is large or the recording is short, which is why a taller plane can hold a
        smaller working set than a shorter one.

    Args:
        plane: The plane's geometry.
        configuration: The recording's processing configuration.

    Returns:
        The memory the job holds in megabytes, before the shared tolerance.
    """
    plane_pixels = plane.height * plane.width
    reference_bytes = (
        _REFERENCE_STAGE_ARRAY_COPIES
        * min(configuration.registration.reference_frame_count, plane.frame_count)
        * plane_pixels
        * _SINGLE_PRECISION_BYTES
    )

    metric_bytes = 0
    metrics_run = (
        configuration.registration.registration_metric_principal_components > 0
        and plane.frame_count >= _MINIMUM_METRIC_FRAME_COUNT
    )
    if metrics_run:
        metric_bytes = (
            _REGISTRATION_METRIC_ARRAY_COPIES
            * _resolve_metric_sample_count(plane=plane)
            * plane_pixels
            * _SINGLE_PRECISION_BYTES
        )

    return WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=max(reference_bytes, metric_bytes))


def _estimate_processing_mb(plane: PlaneGeometry, configuration: SingleRecordingConfiguration) -> int:
    """Estimates the memory one plane processing job holds, from the binned movie detection materializes.

    Args:
        plane: The plane's geometry.
        configuration: The recording's processing configuration.

    Returns:
        The memory the job holds in megabytes, before the shared tolerance.
    """
    binned_frames = _resolve_binned_frame_count(plane=plane, configuration=configuration)
    copies = _DETECTION_DENOISE_ARRAY_COPIES if configuration.roi_detection.denoise else _DETECTION_ARRAY_COPIES
    peak_bytes = copies * binned_frames * plane.height * plane.width * _SINGLE_PRECISION_BYTES
    return WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=peak_bytes)


def _estimate_combination_mb(geometry: RecordingGeometry) -> int:
    """Estimates the memory the combination stage holds, from the traces it concatenates.

    Args:
        geometry: The recording's geometry.

    Returns:
        The memory the stage holds in megabytes, before the shared tolerance.
    """
    trace_bytes = (
        _COMBINATION_TRACE_KINDS * geometry.region_count * geometry.combined_frame_count * _SINGLE_PRECISION_BYTES
    )
    return WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=trace_bytes)


def _estimate_discovery_mb(geometries: Sequence[RecordingGeometry]) -> int:
    """Estimates the memory one cross-recording discovery job holds for a whole dataset.

    Notes:
        The registration term is linear in the recording count, because the groupwise registration caches one image
        and one gradient per image rather than one deformation per unordered pair. The clustering term is a flat
        allowance, because the pairwise matrix it builds follows local region crowding rather than any count the
        processed data reports.

    Args:
        geometries: The geometry of every recording the dataset spans.

    Returns:
        The memory the job holds in megabytes, before the shared tolerance.
    """
    widest_pixels = max(geometry.combined_pixels for geometry in geometries)
    planes = _DISCOVERY_PLANES_PER_RECORDING * len(geometries) + _DISCOVERY_TRANSIENT_PLANES
    registration_bytes = planes * widest_pixels * _SINGLE_PRECISION_BYTES
    return WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=registration_bytes) + _TRACKING_CLUSTERING_MEMORY_MB


def _estimate_extraction_mb(
    geometry: RecordingGeometry, tracked_regions: int, configuration: MultiRecordingConfiguration
) -> int:
    """Estimates the memory one tracked extraction job holds for a single recording.

    Args:
        geometry: The recording's geometry.
        tracked_regions: The regions the job extracts.
        configuration: The dataset's processing configuration.

    Returns:
        The memory the job holds in megabytes, before the shared tolerance.
    """
    trace_bytes = _EXTRACTION_TRACE_COPIES * tracked_regions * geometry.combined_frame_count * _SINGLE_PRECISION_BYTES
    batch_size = configuration.signal_extraction.batch_size
    workspace_bytes = _OASIS_WORKSPACE_BYTES * min(batch_size, max(1, tracked_regions)) * geometry.combined_frame_count
    batch_bytes = _EXTRACTION_BATCH_BYTES_PER_PIXEL * batch_size * geometry.combined_pixels
    return WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=trace_bytes + workspace_bytes + batch_bytes)


def _resolve_metric_sample_count(plane: PlaneGeometry) -> int:
    """Resolves the frames the registration quality metrics sample from one plane.

    Args:
        plane: The plane's geometry.

    Returns:
        The frames the metrics sample.
    """
    use_small_sample = (
        plane.frame_count < _MAXIMUM_METRIC_SAMPLE_COUNT
        or plane.height > _MAXIMUM_EXTENT_FOR_LARGE_SAMPLE
        or plane.width > _MAXIMUM_EXTENT_FOR_LARGE_SAMPLE
    )
    sample_count = _MINIMUM_METRIC_SAMPLE_COUNT if use_small_sample else _MAXIMUM_METRIC_SAMPLE_COUNT
    return min(sample_count, plane.frame_count)


def _resolve_binned_frame_count(plane: PlaneGeometry, configuration: SingleRecordingConfiguration) -> int:
    """Resolves the frames the detection stage bins one plane's movie down to.

    Args:
        plane: The plane's geometry.
        configuration: The recording's processing configuration.

    Returns:
        The frames the binned movie holds.
    """
    bin_size = int(
        max(
            1,
            plane.frame_count // max(1, configuration.roi_detection.maximum_binned_frames),
            round(configuration.main.tau * plane.sampling_rate),
        )
    )
    return max(1, plane.frame_count // bin_size)


def _resolve_specifier_index(specifier: str, count: int) -> int | None:
    """Resolves the index a per-plane specifier names, bounded by the planes the recording holds.

    Args:
        specifier: The job's tracker specifier.
        count: The planes the recording holds.

    Returns:
        The plane index, or None when the specifier names no plane the recording holds.
    """
    plane_index = parse_plane_specifier(specifier=specifier)
    if plane_index is None or not 0 <= plane_index < count:
        return None
    return plane_index


def _resolve_target_geometry(
    recording_roots: Sequence[Path],
    geometries: Sequence[RecordingGeometry],
    resolved: Sequence[RecordingGeometry],
    specifier: str,
) -> RecordingGeometry:
    """Resolves the geometry of the recording one extraction job runs on.

    Notes:
        Charges the widest readable recording when the specifier names none of them, so an unmatched job never
        understates.

    Args:
        recording_roots: The output root of every recording the dataset spans.
        geometries: The geometry of every recording, in the same order.
        resolved: The geometries carrying combined output, which is never empty.
        specifier: The specifier naming the target recording.

    Returns:
        The target recording's geometry.
    """
    for root, geometry in zip(recording_roots, geometries, strict=True):
        if root.name == specifier and geometry.combined_pixels > 0:
            return geometry
    return max(resolved, key=lambda geometry: geometry.combined_pixels)


def _read_plane_geometries(inventory: RecordingPlanes) -> tuple[PlaneGeometry, ...]:
    """Reads the geometry of every plane from the runtime data each plane directory carries.

    Args:
        inventory: The recording's plane inventory.

    Returns:
        The geometry of every plane whose runtime data was readable.
    """
    geometries: list[PlaneGeometry] = []
    for plane_path in inventory.plane_paths:
        try:
            runtime = SingleRecordingRuntimeData.load(output_path=plane_path)
        except FileNotFoundError, ValueError:
            continue
        if runtime.io.frame_count <= 0:
            continue
        geometries.append(
            PlaneGeometry(
                height=runtime.io.frame_height,
                width=runtime.io.frame_width,
                frame_count=runtime.io.frame_count,
                sampling_rate=runtime.io.sampling_rate,
            )
        )
    return tuple(geometries)


def _read_combined_geometry(output_root: Path) -> tuple[int, int]:
    """Reads the combined field extent and frame count from the metadata archive the combination stage wrote.

    Args:
        output_root: The output root the recording was configured with.

    Returns:
        The pixels one combined frame holds and the frames the combined view holds, both zero when the archive is
        absent.
    """
    metadata_path = resolve_output_path(output_root=output_root) / COMBINED_METADATA_FILENAME
    if not metadata_path.is_file():
        return 0, 0
    with np.load(file=metadata_path) as metadata:
        pixels = int(metadata["combined_height"][0]) * int(metadata["combined_width"][0])
        frame_count = int(metadata["frame_count"][0]) if "frame_count" in metadata else 0
    return pixels, frame_count


def _read_region_count(array_path: Path) -> int:
    """Reads the region count a trace array's own header reports.

    Notes:
        Parses the header alone, so a recording of any length costs one small read and no part of the array is mapped.

    Args:
        array_path: The path to the trace array whose header is parsed.

    Returns:
        The regions the array holds, or zero when it is absent or carries another rank.
    """
    if not array_path.is_file():
        return 0
    with array_path.open("rb") as array_file:
        reader = read_array_header_1_0 if read_magic(array_file) == (1, 0) else read_array_header_2_0
        shape, _, _ = reader(array_file)
    return int(shape[0]) if len(shape) == _TRACE_ARRAY_DIMENSIONS else 0


def _bytes_to_megabytes(byte_count: float) -> int:
    """Converts a byte count into whole megabytes, rounding up so an estimate never understates its demand.

    Args:
        byte_count: The number of bytes to convert.

    Returns:
        The equivalent size in megabytes.
    """
    return max(0, int(byte_count / _BYTES_PER_MEGABYTE) + 1) if byte_count > 0 else 0


def _apply_tolerance(memory_mb: int) -> int:
    """Applies the shared estimate tolerance to a modeled figure and rounds it up to a whole gigabyte.

    Notes:
        A scheduler reserves memory in whole gigabytes, so rounding here keeps the figure a plan records identical to
        the figure a submission requests.

    Args:
        memory_mb: The modeled memory in megabytes, before any margin.

    Returns:
        The reportable memory in megabytes.
    """
    reportable = int(memory_mb * MEMORY_ESTIMATE_TOLERANCE) + 1
    return math.ceil(reportable / _MEGABYTES_PER_GIGABYTE) * _MEGABYTES_PER_GIGABYTE


def _resolve_stage_fallback(job_name: str) -> int:
    """Resolves the conservative allowance one stage is charged when its own geometry cannot be read.

    Args:
        job_name: The name of the pipeline stage the job runs.

    Returns:
        The reportable allowance in megabytes.
    """
    return _apply_tolerance(memory_mb=WORKER_MEMORY_MB + _STAGE_FALLBACK_MEGABYTES[str(job_name)])
