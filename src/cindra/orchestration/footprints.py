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
    find_data_directory,
    find_cindra_directory,
    extract_unique_components,
    resolve_source_frame_geometry,
    resolve_acquisition_parameters,
)
from .jobs import MultiRecordingJobNames, SingleRecordingJobNames
from ..layout import (
    PARAMETERS_FILENAME,
    COMBINED_METADATA_FILENAME,
    ACQUISITION_PARAMETERS_FILENAME,
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
    killed outright, so every estimate rounds up. The value matches the margin ataraxis-video-system applies,
    which is what lets a scheduler weigh a cindra job against its jobs on one scale.
"""

WORKER_MEMORY_MB: int = 384
"""The resident memory a worker process occupies before it runs any job, covering the interpreter and the import graph
of this library. The term is charged once per job."""

SPAWNED_CHILD_MEMORY_MB: int = 200
"""The resident memory each child of a job's own process pool occupies before it touches data.

Notes:
    Carried on the same scale as the sibling ataraxis libraries, so a scheduler composing a batch across them prices a
    child the same way. No cindra stage opens a process pool of its own, so no estimate here applies the term. It is
    exported for a scheduler that wraps a cindra job in a pool it owns.
"""

_BYTES_PER_MEGABYTE: int = 1024 * 1024
"""The divisor converting a byte count into megabytes."""

_MEGABYTES_PER_GIGABYTE: int = 1024
"""The megabytes one gigabyte holds, which sets the rounding granularity of every reportable estimate."""

_SINGLE_PRECISION_BYTES: int = 4
"""The width of one single-precision element, which is the type every modeled working array holds."""

_BINARIZATION_LIVE_BATCHES: int = 2
"""The decoded batches the conversion stage holds at once, because the caller keeps the previous batch bound across
the whole of the next read."""

_INTERNAL_ELEMENT_BYTES: int = 2
"""The width of one element of the internal binary format. A wider source is halved and cast down to it."""

_DETECTION_ARRAY_COPIES: float = 2.35
"""The copies of the binned movie the detection stage holds at its peak, which is the scale-0 thresholded variance.
Live at that moment are the binned frames and the five convolved scales. The variance accumulates one frame at a
time, so the transient it adds is a single frame rather than a copy of the movie."""

_DETECTION_DENOISE_ARRAY_COPIES: float = 3.45
"""The copies of the binned movie the detection stage holds at its peak when PCA denoising runs, which adds the
reconstruction and the block reconstructions still in flight. The pool releases each reconstruction as it is
accumulated, so the blocks resident at once follow the worker count rather than the block count."""

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

_BLOCK_OVERLAP_FACTOR: float = 1.5
"""The multiplier the nonrigid block tiling applies to a plane extent, which spaces its blocks at roughly half a block
of overlap."""

_UPSAMPLING_PADDING: int = 3
"""The half-width of the correlation region the subpixel stage upsamples, from which the nonrigid window size
derives."""

_DEVICE_PIPELINE_SLOTS: int = 2
"""The staging slots that carry each frame batch to the device."""

_DEVICE_STAGING_DIRECTIONS: int = 2
"""The transfer directions of the page-locked host buffers each staging slot holds, which are the upload of one batch
and the download of its registered frames."""

_DEVICE_STAGING_BATCH_PIXEL_BYTES: int = 16
"""The device staging buffers the backend holds per batch pixel, covering both pipeline slots at the plane binary's
storage width and every batch geometry a plane's registration passes retain."""

_DEVICE_RIGID_BATCH_PIXEL_BYTES: int = 40
"""The device working set one rigid-only batch registration holds at its peak, in bytes per batch pixel."""

_DEVICE_NONRIGID_BATCH_PIXEL_BYTES: int = 84
"""The device working set one nonrigid batch registration holds at its peak, in bytes per batch pixel."""

_DEVICE_BLOCK_BATCH_PIXEL_BYTES: int = 24
"""The device working set the nonrigid block phase correlation holds at its peak, in bytes per batch block pixel."""

_DEVICE_WINDOW_COPY_BYTES: int = 20
"""The per-block correlation window copies the nonrigid smoothing and subpixel stages hold at once, in bytes per batch
block and window sample."""

_DEVICE_SUBPIXEL_BLOCK_BYTES: int = 15080
"""The gathered correlation region and the upsampled correlation surface one block of one frame holds, in bytes."""

_DEVICE_REFERENCE_FRAME_PIXEL_BYTES: int = 32
"""The frame-shaped reference state the backend holds for its lifetime, in bytes per frame pixel, covering the rigid
taper mask, the rigid mean offset, the four nonrigid interpolation grids, and the normalization weight cache."""

_DEVICE_REFERENCE_BLOCK_PIXEL_BYTES: int = 12
"""The block-shaped reference state the backend holds for its lifetime, in bytes per block pixel, covering the
per-block taper mask, the per-block mean offset, and the block index arrays."""

_DEVICE_COMPLEX_BYTES: int = 8
"""The width of one complex single-precision element, which every phase correlation kernel the backend holds
carries."""

_DEVICE_UPSAMPLING_MATRIX_BYTES: int = 729316
"""The device memory the Gaussian RBF upsampling matrix occupies, which is one 49 by 3721 single-precision matrix."""

_DEVICE_CONTEXT_BYTES: int = 536870912
"""The device memory the CUDA primary context, the cuBLAS handle, and the FFT plan cache occupy alongside the arrays
the registration allocates."""

_DEVICE_LIVE_BACKENDS: int = 2
"""The multiple of one backend's resident state that a plane registration's plan assumes.

Notes:
    One backend is live at a time, because each pass releases its device and page-locked host allocations before the
    next pass builds its own. The figure stays at twice that state as a margin, since the pool sorts its free blocks by
    size and a pass whose plane geometry differs from the one before it allocates fresh blocks rather than reusing the
    cached ones. Admission holds a device-backed job against this estimate, so the margin is the room a job is given
    beyond the arrays it names.
"""

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

_TRACKED_REGION_HEADROOM: float = 1.5
"""The bound on the templates a tracked dataset holds, expressed as a multiple of the region count of its most populated
recording.

Notes:
    This is a domain assumption about how a dataset's recordings overlap rather than a figure derived from the pipeline.
    A tracked dataset holds at most every region of its most populated recording, plus about half that count again
    contributed by regions the other recordings hold and it does not. Because it is an assumption rather than a proof,
    the bound that carries it is taken alongside the combinatorial ceiling the pooled region count sets, and the smaller
    of the two is used.
"""


@dataclass(frozen=True, slots=True)
class PlaneGeometry:
    """Describes the shape one virtual imaging plane will hold, as the acquisition metadata and the source file header
    fix it before the conversion runs.
    """

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
    """The pixels one unsliced acquisition frame holds, which the conversion stage reads in batches."""
    source_element_bytes: int = _INTERNAL_ELEMENT_BYTES
    """The width of one element of the recording's source files."""
    combined_pixels: int = 0
    """The pixels one combined multi-plane frame holds, which every multi-recording stage uses."""
    combined_frame_count: int = 0
    """The frames the combined view holds, trimmed to the shortest contributing plane."""
    two_channels: bool = False
    """Determines whether the recording carries a second channel, which both the combination and the tracked
    extraction stages process alongside the first inside one job."""
    region_count: int = 0
    """The regions the recording's combined trace array holds, which the multi-recording stages read from the
    single-recording output they process."""
    resolved: bool = False
    """Determines whether the geometry follows from the recording's own data rather than from its absence."""
    acquisition_resolved: bool = False
    """Determines whether the recording's acquisition parameters were readable, which the raw acquisition resolution
    alone reports."""
    source_resolved: bool = False
    """Determines whether the recording's source files were readable, which the raw acquisition resolution alone
    reports."""

    def _describe_unresolved_inputs(self, data_path: Path | None) -> str:
        """Describes the input that left the recording without an imaging plane, together with the remedy it requires.

        Notes:
            The acquisition parameters and the source files fail independently and call for different remedies, so the
            resolution keeps the two apart rather than reporting one absence for both. A recording whose two inputs were
            both readable can still describe no imaging plane. That case is reported on its own, because its inputs
            disagree rather than go missing.

        Args:
            data_path: The recording's configured raw imaging directory, or None when it names none.

        Returns:
            The sentences naming the unresolved input and its remedy, empty when the recording holds a plane.
        """
        if self.planes:
            return ""

        source_absence = (
            f"The configured data path {data_path} holds no readable source files"
            if data_path is not None
            else "The recording's configuration names no raw imaging data path"
        )
        source_remedy = (
            "The imaging directory is resolved by locating the acquisition parameters file beneath the data path, "
            "and only the directory holding that file is scanned, so the recording's TIFF files must sit beside it."
            if data_path is not None
            else "Point the configured data path at the recording's imaging directory, or at any parent of it that "
            "carries the acquisition parameters file beneath it."
        )
        parameters_remedy = (
            f"Verify that {ACQUISITION_PARAMETERS_FILENAME} sits in the recording's output directory or that "
            f"{PARAMETERS_FILENAME} sits under its raw imaging directory."
        )

        if not self.acquisition_resolved and not self.source_resolved:
            return (
                f"Neither of its two inputs resolved. {source_absence}, and its acquisition parameters were not "
                f"readable either. {source_remedy} {parameters_remedy}"
            )
        if not self.acquisition_resolved:
            return (
                f"Its acquisition parameters were not readable, so the planes its conversion writes cannot be "
                f"derived. {parameters_remedy}"
            )
        if not self.source_resolved:
            return f"{source_absence}, so the frames its conversion reads cannot be counted. {source_remedy}"
        return (
            "Its acquisition parameters and its source files were both readable, but together they describe fewer "
            "frames than one whole plane and channel interleave cycle, so the conversion writes no imaging plane."
        )


@dataclass(frozen=True, slots=True)
class JobSizing:
    """Describes the resources one job receives, as one sizing pass resolved them."""

    cores: int
    """The CPU cores the job occupies while it runs, as the stage's default declares them."""
    memory_mb: int
    """The memory the job occupies at its peak, in megabytes."""
    device_memory_mb: int
    """The device memory the job occupies at its peak, in megabytes, which is zero for a job that holds no CUDA
    device."""


@dataclass(frozen=True, slots=True)
class _NonrigidBlockGeometry:
    """Describes the overlapping blocks whose offsets one plane's nonrigid registration resolves."""

    count: int
    """The blocks that tile one frame, which is zero while nonrigid registration is disabled."""
    height: int
    """The height of one block in pixels."""
    width: int
    """The width of one block in pixels."""
    window_size: int
    """The side length of the correlation window the subpixel stage reads around each block's peak."""


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
        output_root: The recording's configured output root.
        data_path: The recording's configured raw imaging path, which is either the directory holding its source files
            or any parent of the directory that holds its acquisition parameters file. Every estimate reads the header
            of the first source file that directory holds.
        ignored_file_names: The source file stems the recording excludes from conversion.

    Returns:
        The recording's geometry, whose resolved flag is False when the raw acquisition is unreadable and whose
        acquisition_resolved and source_resolved flags report which of its two inputs failed.
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
        acquisition_resolved=acquisition is not None,
        source_resolved=source is not None,
    )


def read_tracked_recording_geometry(cindra_root: Path) -> RecordingGeometry:
    """Reads the geometry every multi-recording model reads from one recording of a tracked dataset.

    Notes:
        Reads a recording's processed output rather than its raw acquisition, so the caller has already run the
        single-recording pipeline over the recording to completion.

        Covers the combined field extent, the combined frame count, the second channel the metadata archive records,
        and the regions the combined trace array's own header reports. The per-plane geometry and the raw frame
        extent stay unread, because every multi-recording stage works at the combined view. Only headers are parsed,
        so a recording of any length costs a pair of small reads and no array is mapped.

        The region count this reports is the regions one recording holds on its own. It is not the templates a tracked
        dataset holds, which only the cross-recording discovery stage produces. It therefore does not make the tracked
        count readable, and a caller planning a dataset whose discovery has not yet run must not read it as one.

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
        The ceiling, counting every plane of the recording together.
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
    gpu_registration: bool = False,
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
        output_root: The recording's configured output root.
        configuration: The recording's processing configuration.
        data_path: The recording's configured raw imaging path, which is either the directory holding its source files
            or any parent of the directory that holds its acquisition parameters file. Every estimate reads the header
            of the first source file that directory holds.
        planned_roi_count: The regions the plan covers, counting every plane of the recording together. Use None to
            accept the ceiling the detection iteration bound provides. Must be a positive integer when supplied.
        gpu_registration: Determines whether the registration jobs are planned for a CUDA device rather than the host
            CPU. Every other job name resolves the same figure whatever it holds.

    Returns:
        The memory the job occupies in megabytes.

    Raises:
        FileNotFoundError: If the recording's acquisition parameters were not readable or its raw imaging directory
            holds no readable source file, in which case no stage of it can run.
        ValueError: If planned_roi_count is supplied and is not a positive integer, if both inputs were readable and
            still describe no whole imaging plane, or if a per-plane job's specifier names an imaging plane the
            recording does not hold.
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
            f"{output_root} declares no imaging plane, so no stage of it can run. "
            f"{geometry._describe_unresolved_inputs(data_path=data_path)}"
        )
        error = ValueError if geometry.acquisition_resolved and geometry.source_resolved else FileNotFoundError
        console.error(message=message, error=error)

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
            memory_mb=max(
                _estimate_registration_mb(plane=plane, configuration=configuration, gpu_registration=gpu_registration)
                for plane in planes
            )
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
    *,
    planned_roi_count: int | None = None,
) -> int:
    """Estimates the memory one multi-recording job occupies at its peak.

    Notes:
        A discovery job spans every recording of the dataset, so it is estimated from all of them. An extraction job
        runs on one recording, which its specifier names, and is estimated from that recording alone.

        The templates the tracking stage produces are the one figure the completed single-recording output does not
        report, and they do not exist when a plan covering the discovery stage is built. A caller that knows them
        passes them through planned_roi_count, and the bound the per-recording region counts provide covers them
        otherwise. Only the extraction stage reads the figure, because the discovery stage scales with the regions
        each recording reports rather than with the templates it produces.

    Args:
        job_name: The pipeline stage the job runs.
        specifier: The job's tracker specifier, which names a recording for the extraction stage.
        recording_directories: The root directory of every recording the dataset spans, as the configuration's
            recording_directories field holds them. Each is either the recording's pipeline output directory or a
            directory containing it, matching the latitude the context resolver allows.
        configuration: The dataset's processing configuration.
        planned_roi_count: The tracked templates the plan covers, counting the dataset as a whole. Use None to accept
            the bound the per-recording region counts provide. Must be a positive integer when supplied.

    Returns:
        The memory the job occupies in megabytes.

    Raises:
        FileNotFoundError: If the dataset names no recording directory, if any recording carries no combined metadata
            archive, or if any recording reports no regions in its combined trace array, in which case neither
            multi-recording stage can run.
        ValueError: If planned_roi_count is supplied and is not a positive integer.
    """
    if planned_roi_count is not None and planned_roi_count <= 0:
        message = (
            f"Unable to estimate the memory of the '{job_name}' job. The planned region count must be a positive "
            f"integer counting the templates the dataset tracks, or None to accept the bound the per-recording "
            f"region counts provide, but encountered {planned_roi_count}."
        )
        console.error(message=message, error=ValueError)

    cindra_roots = _resolve_cindra_directories(recording_directories=recording_directories)
    geometries = _resolve_dataset_geometries(job_name=job_name, cindra_roots=cindra_roots)

    if job_name == MultiRecordingJobNames.DISCOVER:
        return _apply_tolerance(memory_mb=_estimate_discovery_mb(geometries=geometries))

    geometry = _resolve_target_geometry(cindra_roots=cindra_roots, geometries=geometries, specifier=specifier)
    tracked_regions = _resolve_tracked_regions(
        geometries=geometries, configuration=configuration, planned_roi_count=planned_roi_count
    )
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
    gpu_registration: bool = False,
) -> JobSizing:
    """Sizes one single-recording job from the recording it processes.

    Args:
        job_name: The pipeline stage the job runs.
        specifier: The job's tracker specifier, which names a plane for the per-plane stages and is empty otherwise.
        output_root: The recording's configured output root.
        configuration: The recording's processing configuration.
        data_path: The recording's configured raw imaging path, which is either the directory holding its source files
            or any parent of the directory that holds its acquisition parameters file. Every estimate reads the header
            of the first source file that directory holds.
        planned_roi_count: The regions the plan covers, counting every plane of the recording together. Use None to
            accept the ceiling the detection iteration bound provides. Must be a positive integer when supplied.
        gpu_registration: Determines whether the registration jobs are planned for a CUDA device rather than the host
            CPU. A job of any other stage reports no device memory whatever it holds.

    Returns:
        The cores the job occupies, the memory it holds, and the device memory it holds.

    Raises:
        FileNotFoundError: If the recording's acquisition parameters were not readable or its raw imaging directory
            holds no readable source file, in which case no stage of it can run.
        ValueError: If planned_roi_count is supplied and is not a positive integer, if both inputs were readable and
            still describe no whole imaging plane, or if a per-plane job's specifier names an imaging plane the
            recording does not hold.
    """
    memory_mb = estimate_single_recording_job_memory_mb(
        job_name=job_name,
        specifier=specifier,
        output_root=output_root,
        configuration=configuration,
        data_path=data_path,
        planned_roi_count=planned_roi_count,
        gpu_registration=gpu_registration,
    )

    device_memory_mb = 0
    if gpu_registration and job_name == SingleRecordingJobNames.REGISTER:
        device_memory_mb = _estimate_registration_device_memory_mb(
            specifier=specifier, output_root=output_root, configuration=configuration, data_path=data_path
        )

    return JobSizing(
        cores=resolve_stage_workers(job_name=job_name, gpu_registration=gpu_registration),
        memory_mb=memory_mb,
        device_memory_mb=device_memory_mb,
    )


def size_multi_recording_job(
    job_name: MultiRecordingJobNames,
    specifier: str,
    recording_directories: Sequence[Path],
    configuration: MultiRecordingConfiguration,
    *,
    planned_roi_count: int | None = None,
) -> JobSizing:
    """Sizes one multi-recording job from the dataset it processes.

    Args:
        job_name: The pipeline stage the job runs.
        specifier: The job's tracker specifier, which names a recording for the extraction stage.
        recording_directories: The root directory of every recording the dataset spans, as the configuration's
            recording_directories field holds them.
        configuration: The dataset's processing configuration.
        planned_roi_count: The tracked templates the plan covers, counting the dataset as a whole. Use None to accept
            the bound the per-recording region counts provide. Must be a positive integer when supplied.

    Returns:
        The cores the job occupies and the memory it holds, alongside a device memory of zero, because no
        multi-recording stage runs on a CUDA device.

    Raises:
        FileNotFoundError: If the dataset names no recording directory, if any recording carries no combined metadata
            archive, or if any recording reports no regions in its combined trace array, in which case neither
            multi-recording stage can run.
        ValueError: If planned_roi_count is supplied and is not a positive integer.
    """
    memory_mb = estimate_multi_recording_job_memory_mb(
        job_name=job_name,
        specifier=specifier,
        recording_directories=recording_directories,
        configuration=configuration,
        planned_roi_count=planned_roi_count,
    )

    return JobSizing(cores=resolve_stage_workers(job_name=job_name), memory_mb=memory_mb, device_memory_mb=0)


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


def _estimate_registration_mb(
    plane: PlaneGeometry, configuration: SingleRecordingConfiguration, *, gpu_registration: bool
) -> int:
    """Estimates the host memory one plane registration job holds, from the samples its stages read.

    Notes:
        The peak is the larger of the reference image stage and the registration quality metrics. The metric sample
        count steps down when the plane is large or the recording is short, which is why a taller plane can hold a
        smaller working set than a shorter one.

        A job registering on a CUDA device adds the page-locked staging buffers of both pipeline slots and both
        transfer directions to that peak, because those buffers stay resident for the whole job while the stages above
        run.

    Args:
        plane: The plane's geometry.
        configuration: The recording's processing configuration.
        gpu_registration: Determines whether the job is planned for a CUDA device rather than the host CPU.

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

    peak_bytes = max(reference_bytes, metric_bytes)
    if gpu_registration:
        peak_bytes += (
            _DEVICE_PIPELINE_SLOTS
            * _DEVICE_STAGING_DIRECTIONS
            * _resolve_device_batch_size(plane=plane, configuration=configuration)
            * plane_pixels
            * _SINGLE_PRECISION_BYTES
        )

    return WORKER_MEMORY_MB + _bytes_to_megabytes(byte_count=peak_bytes)


def _estimate_registration_device_memory_mb(
    specifier: str,
    output_root: Path,
    configuration: SingleRecordingConfiguration,
    data_path: Path | None,
) -> int:
    """Estimates the device memory one plane registration job occupies at its peak.

    Notes:
        A job whose specifier names no single plane is charged the largest per-plane figure, matching the host
        estimate. That estimate resolves the same geometry first, so every recording this reads holds at least one
        imaging plane.

    Args:
        specifier: The job's tracker specifier, which names the plane the job registers.
        output_root: The recording's configured output root.
        configuration: The recording's processing configuration.
        data_path: The recording's configured raw imaging path, which is either the directory holding its source files
            or any parent of the directory that holds its acquisition parameters file.

    Returns:
        The device memory the job occupies in megabytes.
    """
    geometry = resolve_recording_geometry(
        output_root=output_root,
        data_path=data_path,
        ignored_file_names=tuple(configuration.file_io.ignored_file_names),
    )
    plane_index = parse_plane_specifier(specifier=specifier)
    planes = (
        geometry.planes
        if plane_index is None
        else tuple(plane for plane in geometry.planes if plane.index == plane_index)
    )

    return _apply_tolerance(
        memory_mb=max(_estimate_registration_device_mb(plane=plane, configuration=configuration) for plane in planes)
    )


def _estimate_registration_device_mb(plane: PlaneGeometry, configuration: SingleRecordingConfiguration) -> int:
    """Estimates the device memory one plane registration job holds while it runs on a CUDA device.

    Notes:
        The batch the device stages is the larger of the two configured sizes, because the alignment pass honors the
        device batch while the secondary channel pass reads the shared one. The staging term and every working term
        scale with that batch, so it is the one setting that fits a job to a card.

        The frame-shaped, block-shaped, and per-block terms are summed rather than maxed. The device memory pool
        retains a freed block rather than returning it to the driver, and the three phases request different shapes.

        A configuration enabling two-step registration is charged for two backends as a margin. One backend is live at a
        time, because each pass releases its allocations before the next pass builds its own. The pool sorts its free
        blocks by size, so a pass whose geometry differs allocates fresh blocks rather than reusing the cached ones. The
        context term is charged once, because one process holds one primary context per device.

    Args:
        plane: The plane's geometry.
        configuration: The recording's processing configuration.

    Returns:
        The device memory the job holds in megabytes, before the shared tolerance.
    """
    plane_pixels = plane.height * plane.width
    half_spectrum = plane.height * (plane.width // 2 + 1)
    batch = _resolve_device_batch_size(plane=plane, configuration=configuration)

    blocks = _resolve_nonrigid_block_geometry(plane=plane, configuration=configuration)
    block_pixels = blocks.count * blocks.height * blocks.width
    block_half_spectrum = blocks.count * blocks.height * (blocks.width // 2 + 1)
    frame_bytes = _DEVICE_NONRIGID_BATCH_PIXEL_BYTES if blocks.count else _DEVICE_RIGID_BATCH_PIXEL_BYTES

    # The staging buffers, the reference uploads, and the upsampling matrix belong to one backend and stay on the
    # device for its whole lifetime, so the two-backend margin charges them twice.
    resident_bytes = (
        _DEVICE_STAGING_BATCH_PIXEL_BYTES * batch * plane_pixels
        + _DEVICE_REFERENCE_FRAME_PIXEL_BYTES * plane_pixels
        + _DEVICE_COMPLEX_BYTES * half_spectrum
        + _DEVICE_REFERENCE_BLOCK_PIXEL_BYTES * block_pixels
        + _DEVICE_COMPLEX_BYTES * block_half_spectrum
        + _SINGLE_PRECISION_BYTES * blocks.count**2
        + _DEVICE_UPSAMPLING_MATRIX_BYTES
    )
    working_bytes = (
        frame_bytes * batch * plane_pixels
        + _DEVICE_BLOCK_BATCH_PIXEL_BYTES * batch * block_pixels
        + batch * blocks.count * (_DEVICE_WINDOW_COPY_BYTES * blocks.window_size**2 + _DEVICE_SUBPIXEL_BLOCK_BYTES)
    )
    backends = _DEVICE_LIVE_BACKENDS if configuration.registration.two_step_registration else 1

    return _bytes_to_megabytes(byte_count=backends * (resident_bytes + working_bytes) + _DEVICE_CONTEXT_BYTES)


def _resolve_device_batch_size(plane: PlaneGeometry, configuration: SingleRecordingConfiguration) -> int:
    """Resolves the frames one plane's device-backed registration stages at once.

    Args:
        plane: The plane's geometry.
        configuration: The recording's processing configuration.

    Returns:
        The frames the widest of the plane's registration passes stages, bounded by the frames the plane holds.
    """
    configured = max(configuration.registration.gpu_batch_size, configuration.registration.batch_size)
    return min(configured, plane.frame_count)


def _resolve_nonrigid_block_geometry(
    plane: PlaneGeometry, configuration: SingleRecordingConfiguration
) -> _NonrigidBlockGeometry:
    """Resolves the block tiling whose offsets one plane's nonrigid registration resolves.

    Args:
        plane: The plane's geometry.
        configuration: The recording's processing configuration.

    Returns:
        The block count, the block extent, and the correlation window size, every one of them zero while nonrigid
        registration is disabled.
    """
    if not configuration.nonrigid_registration.enabled:
        return _NonrigidBlockGeometry(count=0, height=0, width=0, window_size=0)

    requested_height, requested_width = configuration.nonrigid_registration.block_size
    if requested_height >= plane.height:
        block_height, row_blocks = plane.height, 1
    else:
        block_height = requested_height
        row_blocks = math.ceil(_BLOCK_OVERLAP_FACTOR * plane.height / requested_height)

    if requested_width >= plane.width:
        block_width, column_blocks = plane.width, 1
    else:
        block_width = requested_width
        column_blocks = math.ceil(_BLOCK_OVERLAP_FACTOR * plane.width / requested_width)

    correlation_radius = min(
        round(configuration.nonrigid_registration.maximum_block_offset),
        min(block_height, block_width) // 2 - _UPSAMPLING_PADDING,
    )

    return _NonrigidBlockGeometry(
        count=row_blocks * column_blocks,
        height=block_height,
        width=block_width,
        window_size=2 * correlation_radius + 2 * _UPSAMPLING_PADDING + 1,
    )


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
        regions: The regions that size this plane's extraction.
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
        regions: The regions that size the combined trace arrays.

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
    """Resolves the frames one plane's movie holds after the detection stage bins it.

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
    """Resolves the regions that size the region-scaled estimates.

    Notes:
        The regions detection finds are the one input the recording's acquisition leaves open, so a caller that
        knows them supplies them and the detection ceiling bounds them otherwise. Both sources are fixed before the
        recording's first job runs, so one recording sizes to the same figure at every point in its pipeline.

    Args:
        geometry: The recording's geometry.
        configuration: The recording's processing configuration.
        planned_roi_count: The regions the caller's plan covers, or None to accept the detection ceiling.

    Returns:
        The caller's planned count when it supplied one, and the detection ceiling otherwise.
    """
    if planned_roi_count is not None:
        return planned_roi_count
    return resolve_maximum_roi_count(plane_count=len(geometry.planes), configuration=configuration)


def _resolve_tracked_regions(
    geometries: Sequence[RecordingGeometry],
    configuration: MultiRecordingConfiguration,
    planned_roi_count: int | None,
) -> int:
    """Resolves the tracked templates that size the extraction estimate.

    Notes:
        The templates tracking produces do not exist when a dataset is sized, because one planning pass covers the
        discovery stage that produces them and the extraction jobs that read them together. The count is therefore
        estimated from the regions each recording reports on its own, which the completed single-recording pipeline
        already wrote, and a caller that knows the count it expects supplies it instead.

        Two bounds are taken and the smaller is used, because neither follows from the other. The headroom bound is
        the domain assumption _TRACKED_REGION_HEADROOM states: a tracked dataset holds at most every region of its
        most populated recording plus about half that count again, contributed by regions the other recordings hold
        and it does not. The pooled bound is a combinatorial ceiling: every template consumes at least
        minimum_recordings regions and consumes each of them exclusively, so the pooled region count divided by that
        demand is a count no dataset can exceed. The pooled bound is the tighter of the two whenever a dataset spans
        few recordings, and the headroom bound is the tighter one whenever it spans many, so the pooled bound is not
        taken alone.

    Args:
        geometries: The geometry of every recording the dataset spans.
        configuration: The dataset's processing configuration.
        planned_roi_count: The tracked templates the caller's plan covers, or None to accept the bound.

    Returns:
        The caller's planned count when it supplied one, and the smaller of the two bounds otherwise.
    """
    if planned_roi_count is not None:
        return planned_roi_count

    # Mirrors the recording count the tracking stage derives from its mask prevalence, which is the number of recordings
    # that must hold a cluster before it is kept as a template. The floor of one keeps a prevalence of zero, which the
    # configuration admits, from dividing the pooled count by nothing.
    minimum_recordings = max(1, math.ceil(configuration.roi_tracking.mask_prevalence / 100 * len(geometries)))
    pooled_ceiling = sum(entry.region_count for entry in geometries) // minimum_recordings
    headroom_bound = math.ceil(max(entry.region_count for entry in geometries) * _TRACKED_REGION_HEADROOM)
    return min(pooled_ceiling, headroom_bound)


def _resolve_target_geometry(
    cindra_roots: Sequence[Path], geometries: Sequence[RecordingGeometry], specifier: str
) -> RecordingGeometry:
    """Resolves the recording one extraction job uses, together with the geometry the estimate reads from it.

    Notes:
        The specifier carries the identifying component of the recording's path, which is the component the dataset
        resolver uses to derive its recording identifiers, so the match is made against that component rather than
        against the directory's own name. The widest recording is charged when the specifier matches none of them, so an
        unmatched job never understates.

    Args:
        cindra_roots: The pipeline output directory of every recording the dataset spans.
        geometries: The geometry of every recording, in the order the dataset names them.
        specifier: The specifier naming the target recording.

    Returns:
        The geometry of the recording the extraction job uses.
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

    return tuple(
        PlaneGeometry(
            height=_resolve_plane_height(acquisition=acquisition, source=source, plane_index=virtual_plane_index),
            width=source.frame_width,
            frame_count=frame_count,
            sampling_rate=sampling_rate,
            index=virtual_plane_index,
        )
        for virtual_plane_index in range(plane_count)
    )


def _resolve_plane_height(acquisition: AcquisitionParameters, source: SourceFrameGeometry, plane_index: int) -> int:
    """Resolves the frame height one virtual plane holds.

    Args:
        acquisition: The recording's acquisition parameters.
        source: The geometry the recording's source files hold.
        plane_index: The zero-based index of the virtual plane whose height is resolved.

    Returns:
        The plane's frame height, which spans the line list of the plane's region for a multi-region recording.
    """
    if acquisition.is_mroi and acquisition.roi_lines:
        lines = acquisition.roi_lines[plane_index // max(1, acquisition.plane_number)]
        return lines[-1] - lines[0] + 1
    return source.frame_height


def _read_source_geometry(data_path: Path | None, ignored_file_names: tuple[str, ...]) -> SourceFrameGeometry | None:
    """Reads the geometry a recording's source files hold, tolerating their absence.

    Notes:
        Resolves the imaging directory the way the conversion does, by locating the acquisition parameters file beneath
        the configured path and reading the directory that holds it. Sizing a recording therefore measures the same
        files the conversion will read. The resolution accepts a configured path that parents the imaging directory and
        reads the geometry from the files that directory holds.

    Args:
        data_path: The configured raw imaging path, or None when the caller named none.
        ignored_file_names: The source file stems the recording excludes from conversion.

    Returns:
        The geometry the source files hold, or None when the path holds none the discovery accepts.
    """
    if data_path is None or not data_path.is_dir():
        return None
    try:
        data_directory = find_data_directory(data_path=data_path)
    except FileNotFoundError, OSError, ValueError:
        # A path carrying no acquisition parameters file still sizes from its own imaging files, because the
        # geometry the conversion needs is the frame shape rather than the metadata that names the planes.
        data_directory = data_path
    try:
        return resolve_source_frame_geometry(data_directory=data_directory, ignored_file_names=ignored_file_names)
    except FileNotFoundError, OSError, ValueError:
        return None


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

    geometries = tuple(read_tracked_recording_geometry(cindra_root=root) for root in cindra_roots)
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
