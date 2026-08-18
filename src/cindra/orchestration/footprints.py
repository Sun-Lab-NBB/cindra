"""Provides the per-stage memory models of the single and multi-recording pipeline jobs, which project each stage's peak
anonymous working set from the shape of the data it processes. The models exclude the reclaimable pages a memory-mapping
stage leaves resident, because anonymous memory is the term that forces a host to swap and a scheduler to kill a job.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
from numpy.lib.format import read_magic, read_array_header_1_0, read_array_header_2_0
from ataraxis_base_utilities import console

from ..io import (
    SourceFrameGeometry,
    find_cindra_directory,
    extract_unique_components,
    resolve_source_frame_geometry,
    resolve_acquisition_parameters,
)
from .jobs import MultiRecordingJobNames, SingleRecordingJobNames
from ..layout import (
    COMBINED_METADATA_FILENAME,
    RecordingArrays,
    resolve_array_path,
    parse_plane_specifier,
)
from .allocation import resolve_stage_workers

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
"""The resident memory each child of a job's own process pool occupies before it touches data.

Notes:
    Carried at the value the sibling ataraxis libraries use, so a scheduler composing a batch across them charges one
    scale. No cindra stage opens a process pool of its own, so no estimate here applies the term. It is exported for a
    scheduler that wraps a cindra job in a pool it owns.
"""

_BYTES_PER_MEGABYTE: int = 1024 * 1024
"""The divisor converting a byte count into megabytes."""

_MEGABYTES_PER_GIGABYTE: int = 1024
"""The megabytes one gigabyte holds, which every reportable estimate is rounded up to a multiple of."""

_SINGLE_PRECISION_BYTES: int = 4
"""The width of one single-precision element, which is the type every modeled working array holds."""

_BINARIZATION_LIVE_BATCHES: int = 2
"""The decoded batches the conversion stage holds at once, because the caller keeps the previous batch bound across
the whole of the next read."""

_INTERNAL_ELEMENT_BYTES: int = 2
"""The width of one element of the internal binary format, which a wider source is halved and cast down to."""

_DETECTION_ARRAY_COPIES: float = 3.60
"""The copies of the binned movie the detection stage holds at its peak, which is the scale-0 thresholded variance.
Live at that moment are the binned frames, the five convolved scales, the comparison output, and the boolean
predicate."""

_DETECTION_DENOISE_ARRAY_COPIES: float = 4.7
"""The copies of the binned movie the detection stage holds at its peak when PCA denoising runs, which adds the
reconstruction and the block reconstructions the block pool retains."""

_BIN_BATCH_SIZE: int = 500
"""The frames the movie binning reads per batch. Binning happens inside each batch and each batch truncates its own
remainder, so the binned frame count falls below a plain division."""

_DETECTION_ITERATION_MULTIPLIER: int = 250
"""The multiplier the detection stage applies to the configured iteration limit to reach its actual loop bound. The
sparse detection loop appends at most one region per iteration, so the product bounds what one plane can produce."""


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
    the sources plus the result. One combination job merges both channels, so a recording carrying a second channel
    concatenates the same four kinds again and the estimate doubles.
"""

_EXTRACTION_TRACE_COPIES: int = 5
"""The full trace arrays the extraction stage holds at its peak.

Notes:
    The stage returns the raw, neuropil, subtracted, and spike traces together, so all four are live at once, and the
    baseline filter that produces the subtracted trace holds a fifth array of the same size. One extraction job
    extracts both channels and assigns each onto the same record without releasing the first, so a recording carrying
    a second channel doubles the estimate.
"""

_EXTRACTION_BATCH_BYTES_PER_PIXEL: int = 6
"""The memory one extraction batch holds per combined pixel, covering the batch at its stored width and the
single-precision copy the kernel consumes."""

_OASIS_WORKSPACE_BYTES: int = 16
"""The memory the deconvolution workspace holds per region and sample of the batch it processes, covering the pool
amplitude, weight, start frame, and length arrays."""

_DISCOVERY_PLANES_PER_RECORDING: int = 12
"""The single-precision planes a discovery job holds per recording, covering each recording's reference image, its
scale-space pyramid, its accumulated and cached deformation fields, and its transformed reference images. The
groupwise registration caches one image and one gradient per image rather than one deformation per unordered pair, so
the term is linear in the recording count."""

_DISCOVERY_TRANSIENT_PLANES: int = 18
"""The single-precision planes one pairwise deformation holds while it is computed. The term is constant, because one
pair is in flight at a time whatever the group size."""

_TRACE_ARRAY_DIMENSIONS: int = 2
"""The axes a trace array carries, which are its regions and its samples."""

_TRACKING_PAIRWISE_BYTES_PER_SQUARED_REGION: float = 0.55
"""The memory the cross-recording clustering stage holds per squared region it clusters.

Notes:
    The stage clusters the regions of one spatial bin at a time. Each bin holds a condensed distance array, its
    thresholded copy, that copy's square and upper triangle, a condensed Jaccard array, and the copy the linkage
    makes of it. Every one of those scales with the square of the regions the bin holds, and bin occupancy scales
    with the regions the dataset spans, so the term is quadratic in that count and independent of the combined frame.
    The value bounds the coefficient measured across five animals, over which it spans a factor of four, because bin
    occupancy follows local region crowding rather than any count the processed data reports.

    The count it multiplies is the regions each recording's combined trace array holds, which the discovery stage
    narrows by ROI selection before it clusters. The coefficient therefore prices a bin the selection has not yet
    thinned, which overstates rather than understates the stage it sizes.
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
    index: int = 0
    """The position of the plane within the recording, which a per-plane job's specifier names."""


@dataclass(frozen=True, slots=True)
class RecordingGeometry:
    """Describes the shape of one recording as its own output and acquisition parameters report it."""

    planes: tuple[PlaneGeometry, ...] = ()
    """The geometry of every virtual imaging plane, ordered by plane index."""
    raw_frame_pixels: int = 0
    """The pixels one unsliced acquisition frame holds, which the conversion stage reads a batch of at a time."""
    source_element_bytes: int = _INTERNAL_ELEMENT_BYTES
    """The width of one element of the recording's source files."""
    combined_pixels: int = 0
    """The pixels one combined multi-plane frame holds, which every multi-recording stage works at."""
    combined_frame_count: int = 0
    """The frames the combined view holds, trimmed to the shortest contributing plane."""
    two_channels: bool = False
    """Determines whether the recording carries a second channel, which both the combination and the tracked
    extraction stages process alongside the first inside one job."""
    region_count: int = 0
    """The regions the recording's combined trace array holds, which the multi-recording stages read from the
    single-recording output they run on."""
    resolved: bool = False
    """Determines whether the geometry follows from the recording's own data rather than from its absence."""


@dataclass(frozen=True, slots=True)
class JobSizing:
    """Describes the resources one job receives, as one sizing pass resolved them."""

    cores: int
    """The CPU cores the job occupies while it runs, as the stage's measured default declares them."""
    memory_mb: int
    """The memory the job occupies at its peak, in megabytes."""


def resolve_recording_geometry(
    output_root: Path, data_path: Path | None = None, ignored_file_names: tuple[str, ...] = ()
) -> RecordingGeometry:
    """Resolves the shape of one recording from the raw acquisition its conversion stage will read.

    Notes:
        Reads the acquisition metadata and one source file header, which fix every shape the pipeline will write
        before any of it runs. A caller therefore sizes a whole job graph up front and receives the same answer at
        every point in the run. No frame is decoded and no array is mapped, so the cost stays flat as a recording
        grows.

    Args:
        output_root: The output root the recording was configured with.
        data_path: The raw imaging directory holding the recording's source files.
        ignored_file_names: The source file stems the recording excludes from conversion.

    Returns:
        The recording's geometry, whose resolved flag is False when the raw acquisition is unreadable.
    """
    acquisition = resolve_acquisition_parameters(output_root=output_root, data_path=data_path)
    source = _read_source_geometry(data_path=data_path, ignored_file_names=ignored_file_names)
    planes = _derive_plane_geometries(acquisition=acquisition, source=source)

    return RecordingGeometry(
        planes=planes,
        raw_frame_pixels=source.frame_height * source.frame_width if source is not None else 0,
        source_element_bytes=source.element_bytes if source is not None else _INTERNAL_ELEMENT_BYTES,
        two_channels=acquisition is not None and acquisition.channel_number > 1,
        resolved=bool(planes),
    )


def resolve_maximum_roi_count(plane_count: int, configuration: SingleRecordingConfiguration) -> int:
    """Resolves the regions a recording can provably not exceed.

    Notes:
        The sparse detection loop runs a bounded number of iterations and appends at most one region per iteration,
        and every step after the append can only remove regions. The product of the iteration multiplier, the
        configured iteration limit, and the plane count is therefore a ceiling rather than an observed maximum. A second
        functional channel is detected into its own arrays, which the trace models account for through their own
        channel factor.

    Args:
        plane_count: The virtual imaging planes the recording holds.
        configuration: The recording's processing configuration.

    Returns:
        The regions the recording can provably not exceed.
    """
    per_plane = _DETECTION_ITERATION_MULTIPLIER * configuration.roi_detection.maximum_iterations
    return per_plane * max(1, plane_count)


def estimate_single_recording_job_memory_mb(
    job_name: SingleRecordingJobNames,
    specifier: str,
    output_root: Path,
    configuration: SingleRecordingConfiguration,
    data_path: Path | None = None,
    *,
    planned_roi_count: int | None = None,
) -> int:
    """Estimates the memory one single-recording job occupies at its peak.

    Notes:
        Every figure follows from the recording's acquisition geometry, which the raw data fixes before any job runs,
        so a caller sizing a whole job graph up front receives the same answer at every point in the run. A per-plane
        job whose specifier does not resolve is charged the largest per-plane estimate, so an unmatched job never
        understates.

        The regions detection will find are the one input the acquisition leaves open. A caller that knows them
        passes them through planned_roi_count, and the detection ceiling bounds them otherwise.

    Args:
        job_name: The pipeline stage the job runs.
        specifier: The job's tracker specifier, which names a plane for the per-plane stages and is empty otherwise.
        output_root: The output root the recording was configured with.
        configuration: The recording's processing configuration.
        data_path: The raw imaging directory holding the recording's source files, whose header every estimate
            reads.
        planned_roi_count: The regions to plan for, counting every plane of the recording together. Use None to
            accept the ceiling the detection iteration bound provides. Must be a positive integer when supplied.

    Returns:
        The memory the job occupies in megabytes.

    Raises:
        FileNotFoundError: If the recording's raw imaging directory holds no readable source file, in which case no
            stage of it can run.
        ValueError: If planned_roi_count is supplied and is not a positive integer, or if a per-plane job's specifier
            names an imaging plane the recording does not hold.
    """
    if planned_roi_count is not None and planned_roi_count <= 0:
        message = (
            f"Unable to estimate the memory of the '{job_name}' job. The planned region count must be a positive "
            f"integer counting every plane together, or None to accept the detection ceiling, but encountered "
            f"{planned_roi_count}."
        )
        console.error(message=message, error=ValueError)

    geometry = resolve_recording_geometry(
        output_root=output_root,
        data_path=data_path,
        ignored_file_names=tuple(configuration.file_io.ignored_file_names),
    )
    if not geometry.planes:
        message = (
            f"Unable to estimate the memory of the '{job_name}' job. The recording configured with the output root "
            f"{output_root} carries no readable raw imaging data, so no stage of it can run. Verify that the "
            f"configured data path holds the recording's source files."
        )
        console.error(message=message, error=FileNotFoundError)

    regions = _resolve_planned_regions(
        geometry=geometry, configuration=configuration, planned_roi_count=planned_roi_count
    )

    if job_name == SingleRecordingJobNames.BINARIZE:
        return _apply_tolerance(memory_mb=_estimate_binarization_mb(geometry=geometry, configuration=configuration))
    if job_name == SingleRecordingJobNames.COMBINE:
        return _apply_tolerance(memory_mb=_estimate_combination_mb(geometry=geometry, regions=regions))

    plane_index = parse_plane_specifier(specifier=specifier)
    planes = (
        geometry.planes
        if plane_index is None
        else tuple(plane for plane in geometry.planes if plane.index == plane_index)
    )
    if not planes:
        message = (
            f"Unable to estimate the memory of the '{job_name}' job. Its specifier names imaging plane "
            f"'{specifier}', which the recording configured with the output root {output_root} does not hold. The "
            f"recording holds {len(geometry.planes)} plane(s)."
        )
        console.error(message=message, error=ValueError)

    if job_name == SingleRecordingJobNames.REGISTER:
        return _apply_tolerance(
            memory_mb=max(_estimate_registration_mb(plane=plane, configuration=configuration) for plane in planes)
        )

    # A per-plane job is charged as though every planned region fell on its own plane, because the planned figure
    # counts the recording rather than any one plane and the planes of one recording rarely hold similar counts.
    per_plane_ceiling = resolve_maximum_roi_count(plane_count=1, configuration=configuration)
    return _apply_tolerance(
        memory_mb=max(
            _estimate_processing_mb(
                plane=plane,
                configuration=configuration,
                regions=min(regions, per_plane_ceiling),
                channels=2 if geometry.two_channels else 1,
            )
            for plane in planes
        )
    )


def estimate_multi_recording_job_memory_mb(
    job_name: MultiRecordingJobNames,
    specifier: str,
    recording_directories: Sequence[Path],
    configuration: MultiRecordingConfiguration,
) -> int:
    """Estimates the memory one multi-recording job occupies at its peak.

    Notes:
        A discovery job spans every recording of the dataset, so it is estimated from all of them. An extraction job
        runs on one recording, which its specifier names, and is estimated from that recording alone.

    Args:
        job_name: The pipeline stage the job runs.
        specifier: The job's tracker specifier, which names a recording for the extraction stage.
        recording_directories: The root directory of every recording the dataset spans, as the configuration's
            recording_directories field holds them. Each is either the recording's pipeline output directory or a
            directory containing it, matching the latitude the context resolver allows.
        configuration: The dataset's processing configuration.

    Returns:
        The memory the job occupies in megabytes.

    Raises:
        FileNotFoundError: If no recording the dataset names carries a combined metadata archive, in which case
            neither multi-recording stage can run.
    """
    cindra_roots = _resolve_cindra_directories(recording_directories=recording_directories)
    geometries = _resolve_dataset_geometries(job_name=job_name, cindra_roots=cindra_roots)

    if job_name == MultiRecordingJobNames.DISCOVER:
        return _apply_tolerance(memory_mb=_estimate_discovery_mb(geometries=geometries))

    # A template gathers regions that co-locate across recordings, and no recording contributes more than one region
    # to one template, so the regions any single recording holds bound the templates the dataset can track.
    geometry = _resolve_target_geometry(cindra_roots=cindra_roots, geometries=geometries, specifier=specifier)
    tracked_regions = max(entry.region_count for entry in geometries)
    return _apply_tolerance(
        memory_mb=_estimate_extraction_mb(
            geometry=geometry, tracked_regions=tracked_regions, configuration=configuration
        )
    )


def size_single_recording_job(
    job_name: SingleRecordingJobNames,
    specifier: str,
    output_root: Path,
    configuration: SingleRecordingConfiguration,
    data_path: Path | None = None,
    *,
    planned_roi_count: int | None = None,
) -> JobSizing:
    """Sizes one single-recording job from the recording it processes.

    Args:
        job_name: The pipeline stage the job runs.
        specifier: The job's tracker specifier, which names a plane for the per-plane stages and is empty otherwise.
        output_root: The output root the recording was configured with.
        configuration: The recording's processing configuration.
        data_path: The raw imaging directory holding the recording's source files, whose header every estimate
            reads.
        planned_roi_count: The regions to plan for, counting every plane of the recording together. Use None to
            accept the ceiling the detection iteration bound provides. Must be a positive integer when supplied.

    Returns:
        The cores the job occupies and the memory it holds.

    Raises:
        FileNotFoundError: If the recording's raw imaging directory holds no readable source file, in which case no
            stage of it can run.
        ValueError: If planned_roi_count is supplied and is not a positive integer, or if a per-plane job's specifier
            names an imaging plane the recording does not hold.
    """
    memory_mb = estimate_single_recording_job_memory_mb(
        job_name=job_name,
        specifier=specifier,
        output_root=output_root,
        configuration=configuration,
        data_path=data_path,
        planned_roi_count=planned_roi_count,
    )

    return JobSizing(cores=resolve_stage_workers(job_name=job_name), memory_mb=memory_mb)


def size_multi_recording_job(
    job_name: MultiRecordingJobNames,
    specifier: str,
    recording_directories: Sequence[Path],
    configuration: MultiRecordingConfiguration,
) -> JobSizing:
    """Sizes one multi-recording job from the dataset it processes.

    Args:
        job_name: The pipeline stage the job runs.
        specifier: The job's tracker specifier, which names a recording for the extraction stage.
        recording_directories: The root directory of every recording the dataset spans, as the configuration's
            recording_directories field holds them.
        configuration: The dataset's processing configuration.

    Returns:
        The cores the job occupies and the memory it holds.

    Raises:
        FileNotFoundError: If no recording the dataset names carries a combined metadata archive, in which case
            neither multi-recording stage can run.
    """
    memory_mb = estimate_multi_recording_job_memory_mb(
        job_name=job_name,
        specifier=specifier,
        recording_directories=recording_directories,
        configuration=configuration,
    )

    return JobSizing(cores=resolve_stage_workers(job_name=job_name), memory_mb=memory_mb)


def _estimate_binarization_mb(geometry: RecordingGeometry, configuration: SingleRecordingConfiguration) -> int:
    """Estimates the memory the conversion stage holds, from the raw frame batches it decodes.

    Notes:
        Two decoded batches are live at once. A source wider than the internal format adds the halved copy and the
        cast copy on top of them, while a source already at the internal width is written through untouched and adds
        neither. The stage rounds its configured batch up to a whole interleave stride, which adds fewer frames than
        one stride and therefore stays inside the margin every estimate carries.

    Args:
        geometry: The recording's geometry.
        configuration: The recording's processing configuration.

    Returns:
        The memory the stage holds in megabytes, before the shared tolerance.
    """
    source_bytes = geometry.source_element_bytes
    conversion_bytes = 0 if source_bytes == _INTERNAL_ELEMENT_BYTES else source_bytes + _INTERNAL_ELEMENT_BYTES
    batch_bytes = (
        configuration.registration.batch_size
        * geometry.raw_frame_pixels
        * (_BINARIZATION_LIVE_BATCHES * source_bytes + conversion_bytes)
    )
    accumulator_bytes = sum(plane.height * plane.width for plane in geometry.planes) * _SINGLE_PRECISION_BYTES
    return WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=batch_bytes + accumulator_bytes)


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


def _estimate_processing_mb(
    plane: PlaneGeometry, configuration: SingleRecordingConfiguration, regions: int, channels: int
) -> int:
    """Estimates the memory one plane processing job holds.

    Notes:
        The job runs detection and then extraction, and each releases its working set before the other allocates, so
        the peak is the larger of the two rather than their sum. Detection dominates at every region count a normal
        recording reaches, and the extraction term is carried so that a dense recording does not walk off the end of
        the model.

    Args:
        plane: The plane's geometry.
        configuration: The recording's processing configuration.
        regions: The regions this plane's extraction is sized for.
        channels: The channels the recording carries.

    Returns:
        The memory the job holds in megabytes, before the shared tolerance.
    """
    binned_frames = _resolve_binned_frame_count(plane=plane, configuration=configuration)
    copies = _DETECTION_DENOISE_ARRAY_COPIES if configuration.roi_detection.denoise else _DETECTION_ARRAY_COPIES
    frame_pixels = plane.height * plane.width
    detection_bytes = copies * binned_frames * frame_pixels * _SINGLE_PRECISION_BYTES
    extraction_bytes = _EXTRACTION_TRACE_COPIES * channels * regions * plane.frame_count * _SINGLE_PRECISION_BYTES
    return WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=max(detection_bytes, extraction_bytes))


def _estimate_combination_mb(geometry: RecordingGeometry, regions: int) -> int:
    """Estimates the memory the combination stage holds, from the traces it concatenates.

    Args:
        geometry: The recording's geometry.
        regions: The regions the combined trace arrays are sized for.

    Returns:
        The memory the stage holds in megabytes, before the shared tolerance.
    """
    channels = 2 if geometry.two_channels else 1
    frame_count = min((plane.frame_count for plane in geometry.planes), default=0)
    trace_bytes = _COMBINATION_TRACE_KINDS * channels * regions * frame_count * _SINGLE_PRECISION_BYTES
    return WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=trace_bytes)


def _estimate_discovery_mb(geometries: Sequence[RecordingGeometry]) -> int:
    """Estimates the memory one cross-recording discovery job holds for a whole dataset.

    Notes:
        The registration term is linear in the recording count and scales with the combined frame, while the
        clustering term is quadratic in the regions the dataset spans and does not scale with the frame at all. The
        registration working set stays resident while the clustering runs, so the two terms add.

    Args:
        geometries: The geometry of every recording the dataset spans.

    Returns:
        The memory the job holds in megabytes, before the shared tolerance.
    """
    widest_pixels = max(geometry.combined_pixels for geometry in geometries)
    planes = _DISCOVERY_PLANES_PER_RECORDING * len(geometries) + _DISCOVERY_TRANSIENT_PLANES
    registration_mb = _bytes_to_megabytes(byte_count=planes * widest_pixels * _SINGLE_PRECISION_BYTES)

    regions = sum(geometry.region_count for geometry in geometries)
    clustering_mb = _bytes_to_megabytes(byte_count=_TRACKING_PAIRWISE_BYTES_PER_SQUARED_REGION * regions * regions)
    return WORKER_MEMORY_MB + registration_mb + clustering_mb


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
    channels = 2 if geometry.two_channels else 1
    trace_bytes = (
        _EXTRACTION_TRACE_COPIES * channels * tracked_regions * geometry.combined_frame_count * _SINGLE_PRECISION_BYTES
    )
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

    Notes:
        Binning happens inside each read batch and each batch truncates its own remainder, so the movie loses up to
        one bin per batch rather than one bin overall. Every frame is treated as good, which is the planning bound,
        because a rejected frame only ever lowers the count.

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
    batch = max(1, min(plane.frame_count, _BIN_BATCH_SIZE))
    binned = 0
    for start in range(0, plane.frame_count, batch):
        span = min(batch, plane.frame_count - start)
        binned += (span // bin_size) if span > bin_size else 1
    return max(1, binned)


def _resolve_planned_regions(
    geometry: RecordingGeometry, configuration: SingleRecordingConfiguration, planned_roi_count: int | None
) -> int:
    """Resolves the regions the region-scaled estimates are sized for.

    Notes:
        The regions detection finds are the one input the recording's acquisition leaves open, so a caller that
        knows them supplies them and the detection ceiling bounds them otherwise. Both sources are fixed before the
        recording's first job runs, so one recording sizes to the same figure at every point in its pipeline.

    Args:
        geometry: The recording's geometry.
        configuration: The recording's processing configuration.
        planned_roi_count: The regions the caller asked to plan for, or None to accept the detection ceiling.

    Returns:
        The regions to size the region-scaled estimates for.
    """
    if planned_roi_count is not None:
        return planned_roi_count
    return resolve_maximum_roi_count(plane_count=len(geometry.planes), configuration=configuration)


def _resolve_target_geometry(
    cindra_roots: Sequence[Path], geometries: Sequence[RecordingGeometry], specifier: str
) -> RecordingGeometry:
    """Resolves the recording one extraction job runs on, together with the geometry the estimate reads from it.

    Notes:
        The specifier carries the identifying component of the recording's path, which is what the dataset resolver
        derives its recording identifiers from, so the match is made against that component rather than against the
        directory's own name. The widest recording is charged when the specifier matches none of them, so an
        unmatched job never understates.

    Args:
        cindra_roots: The pipeline output directory of every recording the dataset spans.
        geometries: The geometry of every recording, in the order the dataset names them.
        specifier: The specifier naming the target recording.

    Returns:
        The geometry of the recording the extraction job runs on.
    """
    identifiers = extract_unique_components(paths=list(cindra_roots))
    matched = [
        (root, geometry)
        for root, identifier, geometry in zip(cindra_roots, identifiers, geometries, strict=False)
        if identifier == specifier
    ]
    target_root, target = (
        matched[0]
        if matched
        else max(zip(cindra_roots, geometries, strict=True), key=lambda entry: entry[1].combined_pixels)
    )

    acquisition = resolve_acquisition_parameters(output_root=target_root.parent, data_path=None)
    return RecordingGeometry(
        combined_pixels=target.combined_pixels,
        combined_frame_count=target.combined_frame_count,
        two_channels=target.two_channels or (acquisition is not None and acquisition.channel_number > 1),
        region_count=target.region_count,
        resolved=True,
    )


def _derive_plane_geometries(
    acquisition: AcquisitionParameters | None, source: SourceFrameGeometry | None
) -> tuple[PlaneGeometry, ...]:
    """Derives the geometry the conversion will write, from the acquisition metadata and one source file header.

    Notes:
        A multi-region recording takes each plane's height from the span of that region's line list and its width
        from the acquisition page. Every plane receives the frames one whole interleave cycle delivers.

    Args:
        acquisition: The recording's acquisition parameters, or None when they were not readable.
        source: The geometry the recording's source files hold, or None when they were not readable.

    Returns:
        The geometry every plane will hold, empty when either input is unreadable.
    """
    if acquisition is None or source is None:
        return ()
    stride = max(1, acquisition.plane_number * acquisition.channel_number)
    frame_count = source.frame_count // stride
    if frame_count <= 0:
        return ()
    sampling_rate = acquisition.frame_rate / max(1, acquisition.plane_number)
    plane_count = acquisition.virtual_plane_count if acquisition.is_mroi else acquisition.plane_number

    geometries: list[PlaneGeometry] = []
    for virtual_plane_index in range(plane_count):
        if acquisition.is_mroi and acquisition.roi_lines:
            lines = acquisition.roi_lines[virtual_plane_index // max(1, acquisition.plane_number)]
            height, width = lines[-1] - lines[0] + 1, source.frame_width
        else:
            height, width = source.frame_height, source.frame_width
        geometries.append(
            PlaneGeometry(
                height=height,
                width=width,
                frame_count=frame_count,
                sampling_rate=sampling_rate,
                index=virtual_plane_index,
            )
        )
    return tuple(geometries)


def _read_source_geometry(data_path: Path | None, ignored_file_names: tuple[str, ...]) -> SourceFrameGeometry | None:
    """Reads the geometry a recording's source files hold, tolerating their absence.

    Args:
        data_path: The raw imaging directory, or None when the caller named none.
        ignored_file_names: The source file stems the recording excludes from conversion.

    Returns:
        The geometry the source files hold, or None when the directory holds none the discovery accepts.
    """
    if data_path is None or not data_path.is_dir():
        return None
    try:
        return resolve_source_frame_geometry(data_directory=data_path, ignored_file_names=ignored_file_names)
    except FileNotFoundError, OSError, ValueError:
        return None


def _read_tracked_recording_geometry(cindra_root: Path) -> RecordingGeometry:
    """Reads the geometry every multi-recording model reads from one recording of a tracked dataset.

    Notes:
        Covers the combined field extent, the combined frame count, and the second channel the metadata archive
        records. The per-plane geometry and the raw frame extent stay unread, because every multi-recording stage
        works at the combined view.

    Args:
        cindra_root: The recording's pipeline output directory, which carries the combined metadata archive.

    Returns:
        The recording's geometry, whose resolved flag is False when the recording carries no combined output.
    """
    metadata_path = cindra_root / COMBINED_METADATA_FILENAME
    if not metadata_path.is_file():
        return RecordingGeometry()
    combined_pixels, combined_frame_count, two_channels = _read_combined_geometry(metadata_path=metadata_path)
    return RecordingGeometry(
        combined_pixels=combined_pixels,
        combined_frame_count=combined_frame_count,
        two_channels=two_channels,
        region_count=_read_region_count(
            array_path=resolve_array_path(root_path=cindra_root, array=RecordingArrays.CELL_FLUORESCENCE)
        ),
        resolved=combined_pixels > 0,
    )


def _resolve_cindra_directories(recording_directories: Sequence[Path]) -> tuple[Path, ...]:
    """Resolves the pipeline output directory of every recording a dataset names.

    Notes:
        A configured recording directory either is the pipeline output directory or contains it at an arbitrary
        depth, which is the same latitude the context resolver allows, so the resolution is repeated here rather
        than assumed. A recording whose output cannot be located resolves to the directory as configured, which
        carries no combined archive and therefore contributes no geometry.

    Args:
        recording_directories: The recording directories the dataset configuration names.

    Returns:
        The pipeline output directory of every named recording, in the order they were named.
    """
    resolved: list[Path] = []
    for directory in recording_directories:
        if (directory / COMBINED_METADATA_FILENAME).is_file():
            resolved.append(directory)
            continue
        try:
            resolved.append(find_cindra_directory(recording_directory=directory))
        except FileNotFoundError, RuntimeError, OSError:
            resolved.append(directory)
    return tuple(resolved)


def _read_combined_geometry(metadata_path: Path) -> tuple[int, int, bool]:
    """Reads the combined field extent and frame count from the metadata archive the combination stage wrote.

    Args:
        metadata_path: The path to the recording's combined metadata archive, which its caller has located.

    Returns:
        The pixels one combined frame holds, the frames the combined view holds, and whether the recording carries a
        second channel.
    """
    with np.load(file=metadata_path) as metadata:
        pixels = int(metadata["combined_height"][0]) * int(metadata["combined_width"][0])
        frame_count = int(metadata["frame_count"][0]) if "frame_count" in metadata else 0
        two_channels = "registered_binary_paths_channel_2" in metadata
    return pixels, frame_count, two_channels


def _resolve_dataset_geometries(
    job_name: MultiRecordingJobNames, cindra_roots: Sequence[Path]
) -> tuple[RecordingGeometry, ...]:
    """Resolves the geometry of every recording the dataset spans, rejecting a dataset any recording leaves short.

    Notes:
        The multi-recording pipeline resolves its context from every recording at once and refuses a dataset whose
        recordings do not all carry the combined output a completed single-recording run leaves behind. The estimate
        refuses the same datasets, so a partial dataset is reported rather than sized from whichever recordings
        happen to be complete, which would understate the stage by the share the dataset is missing.

    Args:
        job_name: The pipeline stage the job runs.
        cindra_roots: The pipeline output directory of every recording the dataset spans.

    Returns:
        The geometry of every recording, in the order the dataset names them.

    Raises:
        FileNotFoundError: If the dataset names no recording, if any recording carries no combined metadata archive,
            or if any recording reports no regions in its combined trace array.
    """
    if not cindra_roots:
        message = (
            f"Unable to estimate the memory of the '{job_name}' job. The dataset names no recording directory, so "
            f"the stage has nothing to size against."
        )
        console.error(message=message, error=FileNotFoundError)

    geometries = tuple(_read_tracked_recording_geometry(cindra_root=root) for root in cindra_roots)
    incomplete = [
        str(root) for root, geometry in zip(cindra_roots, geometries, strict=True) if geometry.combined_pixels <= 0
    ]
    if incomplete:
        message = (
            f"Unable to estimate the memory of the '{job_name}' job. {len(incomplete)} of the {len(cindra_roots)} "
            f"recording(s) the dataset spans carry no combined metadata archive: {', '.join(incomplete)}. Run the "
            f"single-recording pipeline to completion over every recording of the dataset first."
        )
        console.error(message=message, error=FileNotFoundError)

    unreported = [
        str(root) for root, geometry in zip(cindra_roots, geometries, strict=True) if geometry.region_count <= 0
    ]
    if unreported:
        message = (
            f"Unable to estimate the memory of the '{job_name}' job. {len(unreported)} of the {len(cindra_roots)} "
            f"recording(s) the dataset spans report no regions in their combined trace array: "
            f"{', '.join(unreported)}. Run the single-recording pipeline to completion over every recording of the "
            f"dataset first."
        )
        console.error(message=message, error=FileNotFoundError)

    return geometries


def _read_region_count(array_path: Path) -> int:
    """Reads the region count a trace array's own header reports.

    Notes:
        Parses the header alone, so a recording of any length costs one small read and no part of the array is
        mapped.

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
