"""Provides the fluorescence extraction entry points for the single-recording and multi-recording
processing pipelines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba
from numba import njit, prange
import numpy as np
from scipy import stats
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import LogLevel, console

from ..io import BinaryFile, BinaryFileCombined, resolve_active_binary_marker
from .masks import create_masks
from .deconvolve import apply_oasis_deconvolution, compute_delta_fluorescence
from ..dataclasses import RuntimeContext
from .colocalization import compute_spatial_colocalization, compute_intensity_colocalization
from ..classification import classify

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from ..dataclasses import ROIStatistics, SignalExtraction, SpikeDeconvolution, MultiRecordingRuntimeContext


def extract_traces(context: RuntimeContext | MultiRecordingRuntimeContext, *, workers: int) -> None:
    """Extracts fluorescence traces, classifies ROIs, and deconvolves spikes from registered binary data.

    Notes:
        Dispatches to the appropriate internal handler based on the runtime context type. For single-recording contexts,
        the full extraction pipeline runs including classification and interleaved extraction statistics. For
        multi-recording contexts, backward-transformed tracked ROI masks are used without reclassification.

        Extraction and deconvolution run entirely inside Numba kernels parallelized over ROIs, so the worker count is
        applied as the Numba thread mask before dispatch and covers both branches. The mask is thread-local, so
        concurrently dispatched recordings can hold different worker budgets inside a single process.

    Args:
        context: The runtime context for the recording being processed. Modified in-place to store extraction
            outputs including fluorescence traces, deconvolved spikes, and colocalization data.
        workers: The number of parallel workers allocated to this extraction job. Must be a positive integer.
    """
    # The Numba thread mask is thread-local and cannot exceed the core count Numba detected at import time.
    numba.set_num_threads(min(workers, numba.config.NUMBA_NUM_THREADS))

    if isinstance(context, RuntimeContext):
        _extract_single_recording(context=context)
    else:
        _extract_multi_recording(context=context)


@njit(cache=True, parallel=True)
def _extract_cell_fluorescence(  # pragma: no cover
    output_prototype: NDArray[np.float32],
    data: NDArray[np.float32],
    flat_roi_masks: NDArray[np.int32],
    flat_lambda_weights: NDArray[np.float32],
    mask_offsets: NDArray[np.int32],
) -> NDArray[np.float32]:
    """Extracts cell fluorescence traces for the requested ROIs.

    Notes:
        Fuses the pixel gather and weighted reduction into a single pass to avoid allocating a temporary
        (frame_count, mask_size) array per cell. Since Numba's np.dot on 2D x 1D compiles to a plain scalar loop,
        the fused version performs the same arithmetic with fewer memory operations.

    Args:
        output_prototype: The pre-initialized output array to be updated with the extracted fluorescence traces.
        data: The raw activity data from which to extract the ROI fluorescence traces.
        flat_roi_masks: Flattened array containing all cell mask indices concatenated together.
        flat_lambda_weights: Flattened array containing all lambda weights concatenated together.
        mask_offsets: Array of offsets indicating where each cell's mask starts in the flattened arrays.
            Has length (roi_count + 1), where mask_offsets[i+1] - mask_offsets[i] gives the mask size for cell i.

    Returns:
        The output_prototype array updated with the extracted cell fluorescence traces for each processed ROI.
    """
    roi_count = output_prototype.shape[0]
    frame_count = data.shape[0]

    for cell_index in prange(roi_count):
        start = mask_offsets[cell_index]
        end = mask_offsets[cell_index + 1]

        # Accumulates lambda-weighted pixel fluorescence directly from scattered reads, avoiding a per-cell
        # temporary array allocation. Weights bias the trace toward pixels more likely to belong to the cell.
        for frame_index in range(frame_count):
            accumulator = np.float32(0.0)
            for pixel_offset in range(start, end):
                accumulator += data[frame_index, flat_roi_masks[pixel_offset]] * flat_lambda_weights[pixel_offset]
            output_prototype[cell_index, frame_index] = accumulator

    return output_prototype


@njit(cache=True, parallel=True)
def _extract_neuropil_fluorescence(  # pragma: no cover
    output_prototype: NDArray[np.float32],
    data: NDArray[np.float32],
    flat_neuropil_masks: NDArray[np.int32],
    mask_offsets: NDArray[np.int32],
    neuropil_pixel_count: NDArray[np.int32],
) -> NDArray[np.float32]:
    """Extracts neuropil fluorescence traces for the requested ROIs.

    Notes:
        An ROI whose neuropil mask holds no pixels reports zero neuropil fluorescence, which matches the traces an
        ROI receives when neuropil extraction is disabled.

    Args:
        output_prototype: The pre-initialized output array to be updated with the extracted fluorescence traces.
        data: The raw activity data from which to extract the fluorescence traces.
        flat_neuropil_masks: Flattened array containing all neuropil mask indices concatenated together.
        mask_offsets: Array of offsets indicating where each cell's neuropil mask starts in the flattened array.
        neuropil_pixel_count: The number of pixels in each neuropil mask.

    Returns:
        The output_prototype array updated with the extracted neuropil fluorescence traces for each processed ROI.
    """
    roi_count = output_prototype.shape[0]
    frame_count = data.shape[0]

    for cell_index in prange(roi_count):
        start = mask_offsets[cell_index]
        end = mask_offsets[cell_index + 1]

        # Pre-computes the reciprocal of the neuropil pixel count to replace per-frame division with multiplication.
        # A mask with no pixels takes a zero reciprocal, since float32 division by zero yields infinity, which the
        # empty accumulator below would turn into a NaN trace that propagates into every downstream array.
        pixel_count = neuropil_pixel_count[cell_index]
        reciprocal = np.float32(0.0) if pixel_count == 0 else np.float32(1.0) / np.float32(pixel_count)

        # Computes the average fluorescence over the entire neuropil region for each frame.
        for frame_index in range(frame_count):
            accumulator = np.float32(0.0)
            for pixel_offset in range(start, end):
                accumulator += data[frame_index, flat_neuropil_masks[pixel_offset]]
            output_prototype[cell_index, frame_index] = accumulator * reciprocal

    return output_prototype


def _create_and_unpack_masks(
    roi_statistics: list[ROIStatistics],
    frame_height: int,
    frame_width: int,
    *,
    extract_neuropil: bool,
    allow_overlap: bool,
    cell_probability_percentile: int,
    inner_neuropil_border_radius: int,
    minimum_neuropil_pixels: int,
    channel_label: str,
) -> tuple[tuple[tuple[NDArray[np.int32], NDArray[np.float32]], ...], tuple[NDArray[np.int32], ...] | None]:
    """Creates cell and neuropil masks and unpacks them into the format expected by the extraction functions.

    Args:
        roi_statistics: The ROI statistics for each ROI to process.
        frame_height: The height of the imaging field in pixels.
        frame_width: The width of the imaging field in pixels.
        extract_neuropil: Determines whether to create neuropil masks.
        allow_overlap: Determines whether to include overlapping ROI pixels in the created masks.
        cell_probability_percentile: The percentile threshold for classifying pixels as belonging to a cell versus
            neuropil.
        inner_neuropil_border_radius: The width, in pixels, of the exclusion zone between the cell ROI and its
            neuropil mask.
        minimum_neuropil_pixels: The minimum number of pixels required for each neuropil mask.
        channel_label: A descriptive label for the channel being processed, used in log messages.

    Returns:
        A tuple of two elements. The first is a tuple of (pixel_indices, lambda_weights) pairs for each ROI cell mask.
        The second is a tuple of neuropil pixel index arrays for each ROI, or None if neuropil extraction is disabled.
    """
    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()
    console.echo(message=f"Creating {channel_label} ROI masks...", level=LogLevel.INFO)

    per_roi_masks = create_masks(
        roi_statistics=roi_statistics,
        height=frame_height,
        width=frame_width,
        neuropil=extract_neuropil,
        include_overlap=allow_overlap,
        cell_probability_percentile=cell_probability_percentile,
        inner_neuropil_border_radius=inner_neuropil_border_radius,
        minimum_neuropil_pixels=minimum_neuropil_pixels,
    )

    # Unpacks the per-ROI mask tuples into the separate formats expected by the extraction functions.
    roi_masks = tuple((indices, weights) for indices, weights, _ in per_roi_masks)
    neuropil_masks = (
        tuple(neuropil for _, _, neuropil in per_roi_masks if neuropil is not None)
        if per_roi_masks[0][2] is not None
        else None
    )

    console.echo(
        message=f"{channel_label.capitalize()} ROI masks: created. Time taken: {timer.elapsed} seconds.",
        level=LogLevel.SUCCESS,
    )

    return roi_masks, neuropil_masks


def _extract_fluorescence_traces(
    frames: BinaryFile | BinaryFileCombined,
    roi_masks: tuple[tuple[NDArray[np.int32], NDArray[np.float32]], ...],
    neuropil_masks: tuple[NDArray[np.int32], ...] | None,
    batch_size: int,
    channel_label: str,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Extracts the fluorescence traces from the raw activity data using cell and neuropil masks.

    Notes:
        If neuropil masks are not provided, the neuropil fluorescence traces are returned as an array of zeroes.

    Args:
        frames: The raw activity data (movie) to process.
        roi_masks: The cell masks for each ROI, where each element is a tuple of (flattened pixel indices,
            normalized lambda weights).
        neuropil_masks: The neuropil masks for each ROI, or None to skip neuropil extraction.
        batch_size: The number of frames to process at the same time.
        channel_label: A descriptive label for the channel being processed, used in log messages.

    Returns:
        The extracted cell and neuropil fluorescence traces stored as arrays with dimensions (roi_count, frame_count).
    """
    console.echo(message=f"Extracting {channel_label} ROI fluorescence data...", level=LogLevel.INFO)

    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    # BinaryFileCombined stores the combined height and width as direct attributes, while BinaryFile exposes them
    # through the shape tuple.
    if isinstance(frames, BinaryFileCombined):
        frame_count = frames.frame_number
        height = frames.height
        width = frames.width
    else:
        frame_count, height, width = frames.shape
    roi_count = len(roi_masks)
    pixel_count = height * width

    fluorescence = np.zeros((roi_count, frame_count), dtype=np.float32)
    neuropil_fluorescence = np.zeros((roi_count, frame_count), dtype=np.float32)

    # Flattens cell masks and lambda weights into contiguous arrays with offset pointers. This format avoids Numba's
    # tuple size limitations and enables efficient parallel processing.
    roi_mask_sizes = np.array([len(pixel_indices) for pixel_indices, _ in roi_masks], dtype=np.int32)
    roi_mask_offsets = np.zeros(roi_count + 1, dtype=np.int32)
    roi_mask_offsets[1:] = np.cumsum(roi_mask_sizes)

    total_roi_pixels = int(roi_mask_offsets[-1])
    flat_roi_masks = np.empty(total_roi_pixels, dtype=np.int32)
    flat_lambda_weights = np.empty(total_roi_pixels, dtype=np.float32)

    for mask_index, (pixel_indices, lambda_weights) in enumerate(roi_masks):
        start = roi_mask_offsets[mask_index]
        end = roi_mask_offsets[mask_index + 1]
        flat_roi_masks[start:end] = pixel_indices
        flat_lambda_weights[start:end] = lambda_weights

    # Flattens neuropil masks into contiguous arrays with offset pointers if provided.
    flat_neuropil_masks: NDArray[np.int32] | None = None
    neuropil_mask_offsets: NDArray[np.int32] | None = None
    neuropil_pixel_count: NDArray[np.int32] | None = None

    if neuropil_masks is not None:
        neuropil_mask_sizes = np.array([len(indices) for indices in neuropil_masks], dtype=np.int32)
        neuropil_mask_offsets = np.zeros(roi_count + 1, dtype=np.int32)
        neuropil_mask_offsets[1:] = np.cumsum(neuropil_mask_sizes)

        total_neuropil_pixels = int(neuropil_mask_offsets[-1])
        flat_neuropil_masks = np.empty(total_neuropil_pixels, dtype=np.int32)
        neuropil_pixel_count = np.zeros(roi_count, dtype=np.int32)

        for mask_index, indices in enumerate(neuropil_masks):
            start = neuropil_mask_offsets[mask_index]
            end = neuropil_mask_offsets[mask_index + 1]
            flat_neuropil_masks[start:end] = indices
            neuropil_pixel_count[mask_index] = len(indices)

    # Pre-allocates a reusable buffer for the extraction kernels. Both kernels write every element unconditionally,
    # so zeroing is unnecessary. Re-allocated only for the last batch if it is smaller than the standard batch size.
    output_prototype = np.empty((roi_count, batch_size), dtype=np.float32)

    # Pre-allocates the float32 destination the binary source is converted into. The binary stores int16, so the
    # conversion is required, while a fresh destination per batch is not. A leading-axis slice of this buffer stays
    # C-contiguous, which is the layout both kernels read.
    batch_buffer = np.empty((min(batch_size, frame_count), pixel_count), dtype=np.float32)

    for batch_start in range(0, frame_count, batch_size):
        batch_end = min(batch_start + batch_size, frame_count)

        batch_source = frames[batch_start:batch_end]
        batch_frames = batch_source.shape[0]
        batch_pixels = batch_buffer[:batch_frames]
        np.copyto(dst=batch_pixels, src=batch_source.reshape(batch_frames, pixel_count))
        batch_slice = slice(batch_start, batch_start + batch_frames)

        if batch_frames < output_prototype.shape[1]:
            output_prototype = np.empty((roi_count, batch_frames), dtype=np.float32)

        fluorescence[:, batch_slice] = _extract_cell_fluorescence(
            output_prototype=output_prototype,
            data=batch_pixels,
            flat_roi_masks=flat_roi_masks,
            flat_lambda_weights=flat_lambda_weights,
            mask_offsets=roi_mask_offsets,
        )

        if neuropil_masks is not None:
            neuropil_fluorescence[:, batch_slice] = _extract_neuropil_fluorescence(
                output_prototype=output_prototype,
                data=batch_pixels,
                flat_neuropil_masks=flat_neuropil_masks,
                mask_offsets=neuropil_mask_offsets,
                neuropil_pixel_count=neuropil_pixel_count,
            )

    console.echo(
        message=(
            f"{channel_label.capitalize()} ROI fluorescence: extracted from {roi_count} ROIs in {frame_count} "
            f"frames. Time taken: {timer.elapsed} seconds."
        ),
        level=LogLevel.SUCCESS,
    )

    return fluorescence, neuropil_fluorescence


def _update_roi_extraction_statistics(
    roi_statistics: list[ROIStatistics],
    cell_fluorescence: NDArray[np.float32],
    neuropil_fluorescence: NDArray[np.float32],
    neuropil_coefficient: float,
) -> None:
    """Computes neuropil-corrected skewness and stores it in the ROI statistics.

    Args:
        roi_statistics: The ROI statistics to update in-place with the computed skewness values.
        cell_fluorescence: The extracted cell fluorescence traces with shape (roi_count, frame_count).
        neuropil_fluorescence: The extracted neuropil fluorescence traces with shape (roi_count, frame_count).
        neuropil_coefficient: The scaling factor applied to neuropil fluorescence before subtraction.
    """
    # Scaling by the negated coefficient and accumulating in place holds one full-size buffer instead of two. IEEE 754
    # defines the difference to equal the sum with the negated operand, so the values are unchanged.
    corrected = neuropil_fluorescence * np.float32(-neuropil_coefficient)
    corrected += cell_fluorescence
    skew_values = np.asarray(stats.skew(a=corrected, axis=1))

    for roi, skewness_value in zip(roi_statistics, skew_values, strict=True):
        roi.skewness = float(skewness_value)


def _extract_single_recording(context: RuntimeContext) -> None:
    """Extracts fluorescence traces, classifies ROIs, and deconvolves spikes from registered binary data.

    Notes:
        Orchestrates the full extraction pipeline for one or both channels. For structural channel 2
        data, channel 1 masks are reused and intensity colocalization is computed. For functional channel 2 data,
        independent masks are created and spatial colocalization is computed between the two channels' ROIs. Results
        are written into context.runtime.extraction and context.runtime.timing.

    Args:
        context: The RuntimeContext containing configuration, file paths, and mutable runtime data structures. Modified
            in-place to store extraction outputs including fluorescence traces, classification results, deconvolved
            spikes, and colocalization data.

    Raises:
        RuntimeError: If detection has not been run (no ROI statistics available) or if the registered binary path is
            not set.
    """
    extraction_config = context.configuration.signal_extraction
    deconvolution_config = context.configuration.spike_deconvolution
    main_config = context.configuration.main

    io_data = context.runtime.io
    extraction_data = context.runtime.extraction
    timing = context.runtime.timing

    # Loads extraction arrays from the previous stage (detection) if not in memory.
    output_path = context.runtime.io.output_path
    if output_path is not None and extraction_data.roi_statistics is None:
        extraction_data.load_arrays(output_path=output_path)

    plane_index = io_data.plane_index if io_data.plane_index is not None else 0
    frame_height = io_data.frame_height
    frame_width = io_data.frame_width
    batch_size = extraction_config.batch_size

    if extraction_data.roi_statistics is None:
        message = (
            f"Unable to run extraction for plane {plane_index}. ROI detection must run before extraction, but "
            f"no ROI statistics are available."
        )
        console.error(message=message, error=RuntimeError)

    channel_1_path = io_data.registered_binary_path
    if channel_1_path is None:
        message = (
            f"Unable to run extraction for plane {plane_index}. The registered binary file path is not set "
            f"for channel 1."
        )
        console.error(message=message, error=RuntimeError)

    roi_statistics = extraction_data.roi_statistics
    channel_1_label = f"plane {plane_index} channel 1"

    roi_masks, neuropil_masks = _create_and_unpack_masks(
        roi_statistics=roi_statistics,
        frame_height=frame_height,
        frame_width=frame_width,
        extract_neuropil=extraction_config.extract_neuropil,
        allow_overlap=extraction_config.allow_overlap,
        cell_probability_percentile=extraction_config.cell_probability_percentile,
        inner_neuropil_border_radius=extraction_config.inner_neuropil_border_radius,
        minimum_neuropil_pixels=extraction_config.minimum_neuropil_pixels,
        channel_label=channel_1_label,
    )

    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    with BinaryFile(
        height=frame_height,
        width=frame_width,
        file_path=channel_1_path,
        frame_number=io_data.frame_count,
    ) as binary:
        extraction_data.cell_fluorescence, extraction_data.neuropil_fluorescence = _extract_fluorescence_traces(
            frames=binary,
            roi_masks=roi_masks,
            neuropil_masks=neuropil_masks,
            batch_size=batch_size,
            channel_label=channel_1_label,
        )

    timing.extraction_time = int(timer.elapsed)

    _update_roi_extraction_statistics(
        roi_statistics=roi_statistics,
        cell_fluorescence=extraction_data.cell_fluorescence,
        neuropil_fluorescence=extraction_data.neuropil_fluorescence,
        neuropil_coefficient=deconvolution_config.neuropil_coefficient,
    )

    timer.reset()
    extraction_data.cell_classification = classify(
        roi_statistics=roi_statistics,
        classification_threshold=extraction_config.classification_threshold,
        custom_classifier_path=main_config.custom_classifier_path,
    )
    timing.classification_time = int(timer.elapsed)
    console.echo(
        message=(
            f"Plane {plane_index} channel 1 ROI classification: complete. "
            f"Time taken: {timing.classification_time} seconds."
        ),
        level=LogLevel.SUCCESS,
    )

    timer.reset()
    if deconvolution_config.extract_spikes:
        extraction_data.subtracted_fluorescence = compute_delta_fluorescence(
            cell_fluorescence=extraction_data.cell_fluorescence,
            neuropil_fluorescence=extraction_data.neuropil_fluorescence,
            neuropil_coefficient=deconvolution_config.neuropil_coefficient,
            baseline_method=str(deconvolution_config.baseline_method),
            baseline_window=deconvolution_config.baseline_window,
            baseline_sigma=deconvolution_config.baseline_sigma,
            baseline_percentile=deconvolution_config.baseline_percentile,
            sampling_rate=io_data.sampling_rate,
        )
        extraction_data.spikes = apply_oasis_deconvolution(
            cell_fluorescence=extraction_data.subtracted_fluorescence,
            batch_size=batch_size,
            time_constant=main_config.tau,
            sampling_rate=io_data.sampling_rate,
        )
        timing.deconvolution_time = int(timer.elapsed)
        console.echo(
            message=(
                f"Plane {plane_index} channel 1 spike deconvolution: complete. "
                f"Time taken: {timing.deconvolution_time} seconds."
            ),
            level=LogLevel.SUCCESS,
        )
    else:
        console.echo(
            message=(
                f"Skipping plane {plane_index} channel 1 spike deconvolution, as the 'extract_spikes' configuration "
                f"parameter is set to False."
            ),
            level=LogLevel.WARNING,
        )
        extraction_data.subtracted_fluorescence = np.zeros_like(extraction_data.cell_fluorescence)
        extraction_data.spikes = np.zeros_like(extraction_data.cell_fluorescence)

    # Processes channel 2 if the recording has two channels. When both hardware channels are functional,
    # channel_2_data.bin contains independently detectable data and receives functional extraction. When only the
    # second hardware channel is functional, the import layer swaps it into channel_1_data.bin, so channel_2_data.bin
    # holds non-functional data and receives structural extraction instead.
    if main_config.two_channels and io_data.registered_binary_path_channel_2 is not None:
        if main_config.first_channel_functional and main_config.second_channel_functional:
            _extract_functional_channel_2(context=context, batch_size=batch_size)
        else:
            _extract_structural_channel_2(
                context=context,
                batch_size=batch_size,
                roi_masks=roi_masks,
                neuropil_masks=neuropil_masks,
            )

    context.save_runtime()

    context.runtime.extraction.release_arrays()


def _extract_structural_channel_2(
    context: RuntimeContext,
    batch_size: int,
    roi_masks: tuple[tuple[NDArray[np.int32], NDArray[np.float32]], ...],
    neuropil_masks: tuple[NDArray[np.int32], ...] | None,
) -> None:
    """Extracts structural channel 2 fluorescence using channel 1 masks and computes intensity colocalization.

    Args:
        context: The RuntimeContext containing configuration and mutable runtime data. Modified in-place to store
            channel 2 fluorescence traces, colocalization results, and the corrected structural mean image.
        batch_size: The number of frames to process at the same time.
        roi_masks: The channel 1 cell masks to reuse for channel 2 extraction.
        neuropil_masks: The channel 1 neuropil masks to reuse for channel 2 extraction.

    Raises:
        RuntimeError: If the registered binary file path is not set for channel 2.
    """
    io_data = context.runtime.io
    detection_data = context.runtime.detection
    extraction_data = context.runtime.extraction

    plane_index = io_data.plane_index if io_data.plane_index is not None else 0
    channel_2_path = io_data.registered_binary_path_channel_2
    channel_2_label = f"plane {plane_index} channel 2"

    if channel_2_path is None:
        message = (
            f"Unable to run extraction for {channel_2_label}. The registered binary file path is not set for channel 2."
        )
        console.error(message=message, error=RuntimeError)

    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    with BinaryFile(
        height=io_data.frame_height,
        width=io_data.frame_width,
        file_path=channel_2_path,
        frame_number=io_data.frame_count,
    ) as binary:
        extraction_data.cell_fluorescence_channel_2, extraction_data.neuropil_fluorescence_channel_2 = (
            _extract_fluorescence_traces(
                frames=binary,
                roi_masks=roi_masks,
                neuropil_masks=neuropil_masks,
                batch_size=batch_size,
                channel_label=channel_2_label,
            )
        )

    context.runtime.timing.extraction_time_channel_2 = int(timer.elapsed)

    # Re-acquires the two mean images colocalization consumes. Detection releases them before extraction runs, and
    # both are on disk by this point, the functional one written by detection and the structural one by registration.
    # Memory-mapped arrays are skipped by the save that follows, so re-mapping them here does not rewrite the files.
    output_path = io_data.output_path
    if output_path is not None and (detection_data.mean_image is None or detection_data.mean_image_channel_2 is None):
        detection_data.memory_map_arrays(output_path=output_path)

    extraction_config = context.configuration.signal_extraction
    if (
        extraction_data.roi_statistics is not None
        and detection_data.mean_image is not None
        and detection_data.mean_image_channel_2 is not None
    ):
        extraction_data.cell_colocalization, extraction_data.corrected_structural_mean_image = (
            compute_intensity_colocalization(
                rois=extraction_data.roi_statistics,
                functional_mean_image=detection_data.mean_image,
                structural_mean_image=detection_data.mean_image_channel_2,
                frame_height=io_data.frame_height,
                frame_width=io_data.frame_width,
                colocalization_threshold=extraction_config.colocalization_threshold,
                allow_overlap=extraction_config.allow_overlap,
                cell_probability_percentile=extraction_config.cell_probability_percentile,
                inner_neuropil_border_radius=extraction_config.inner_neuropil_border_radius,
                minimum_neuropil_pixels=extraction_config.minimum_neuropil_pixels,
            )
        )
    else:
        console.echo(
            message=(
                f"Skipping {channel_2_label} intensity colocalization. The ROI statistics or one of the two mean "
                f"images required to measure it are not available for this plane."
            ),
            level=LogLevel.WARNING,
        )


def _extract_functional_channel_2(
    context: RuntimeContext,
    batch_size: int,
) -> None:
    """Extracts functional channel 2 fluorescence with independent masks and computes spatial colocalization.

    Notes:
        When both channels are functional, channel 2 has its own independently detected ROIs. Creates masks from
        those ROIs, extracts fluorescence, computes neuropil-corrected skewness for the channel 2 ROI statistics,
        classifies ROIs, computes delta fluorescence and spike deconvolution, and finally computes spatial
        colocalization between channel 1 and channel 2 ROIs.

    Args:
        context: The RuntimeContext containing configuration and mutable runtime data. Modified in-place to store
            channel 2 extraction results and colocalization data.
        batch_size: The number of frames to process at the same time.

    Raises:
        RuntimeError: If the registered binary file path is not set for channel 2 or if channel 2 ROI detection has not
            been run (no channel 2 ROI statistics available).
    """
    extraction_config = context.configuration.signal_extraction
    deconvolution_config = context.configuration.spike_deconvolution
    main_config = context.configuration.main
    io_data = context.runtime.io
    extraction_data = context.runtime.extraction
    timing = context.runtime.timing

    plane_index = io_data.plane_index if io_data.plane_index is not None else 0
    channel_2_path = io_data.registered_binary_path_channel_2
    channel_2_label = f"plane {plane_index} channel 2"
    frame_height = io_data.frame_height
    frame_width = io_data.frame_width

    if channel_2_path is None:
        message = (
            f"Unable to run extraction for {channel_2_label}. The registered binary file path is not set for channel 2."
        )
        console.error(message=message, error=RuntimeError)

    roi_statistics_channel_2 = extraction_data.roi_statistics_channel_2
    if roi_statistics_channel_2 is None:
        message = (
            f"Unable to run functional channel 2 extraction for plane {plane_index}. Channel 2 ROI detection "
            f"must run before extraction, but no channel 2 ROI statistics are available."
        )
        console.error(message=message, error=RuntimeError)

    channel_2_roi_masks, channel_2_neuropil_masks = _create_and_unpack_masks(
        roi_statistics=roi_statistics_channel_2,
        frame_height=frame_height,
        frame_width=frame_width,
        extract_neuropil=extraction_config.extract_neuropil,
        allow_overlap=extraction_config.allow_overlap,
        cell_probability_percentile=extraction_config.cell_probability_percentile,
        inner_neuropil_border_radius=extraction_config.inner_neuropil_border_radius,
        minimum_neuropil_pixels=extraction_config.minimum_neuropil_pixels,
        channel_label=channel_2_label,
    )

    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    with BinaryFile(
        height=frame_height,
        width=frame_width,
        file_path=channel_2_path,
        frame_number=io_data.frame_count,
    ) as binary:
        extraction_data.cell_fluorescence_channel_2, extraction_data.neuropil_fluorescence_channel_2 = (
            _extract_fluorescence_traces(
                frames=binary,
                roi_masks=channel_2_roi_masks,
                neuropil_masks=channel_2_neuropil_masks,
                batch_size=batch_size,
                channel_label=channel_2_label,
            )
        )

    timing.extraction_time_channel_2 = int(timer.elapsed)

    _update_roi_extraction_statistics(
        roi_statistics=roi_statistics_channel_2,
        cell_fluorescence=extraction_data.cell_fluorescence_channel_2,
        neuropil_fluorescence=extraction_data.neuropil_fluorescence_channel_2,
        neuropil_coefficient=deconvolution_config.neuropil_coefficient,
    )

    timer.reset()
    extraction_data.cell_classification_channel_2 = classify(
        roi_statistics=roi_statistics_channel_2,
        classification_threshold=extraction_config.classification_threshold,
        custom_classifier_path=main_config.custom_classifier_path,
    )
    timing.classification_time_channel_2 = int(timer.elapsed)
    console.echo(
        message=(
            f"Plane {plane_index} channel 2 ROI classification: complete. "
            f"Time taken: {timing.classification_time_channel_2} seconds."
        ),
        level=LogLevel.SUCCESS,
    )

    timer.reset()
    if deconvolution_config.extract_spikes:
        extraction_data.subtracted_fluorescence_channel_2 = compute_delta_fluorescence(
            cell_fluorescence=extraction_data.cell_fluorescence_channel_2,
            neuropil_fluorescence=extraction_data.neuropil_fluorescence_channel_2,
            neuropil_coefficient=deconvolution_config.neuropil_coefficient,
            baseline_method=str(deconvolution_config.baseline_method),
            baseline_window=deconvolution_config.baseline_window,
            baseline_sigma=deconvolution_config.baseline_sigma,
            baseline_percentile=deconvolution_config.baseline_percentile,
            sampling_rate=io_data.sampling_rate,
        )
        extraction_data.spikes_channel_2 = apply_oasis_deconvolution(
            cell_fluorescence=extraction_data.subtracted_fluorescence_channel_2,
            batch_size=batch_size,
            time_constant=main_config.tau,
            sampling_rate=io_data.sampling_rate,
        )
        timing.deconvolution_time_channel_2 = int(timer.elapsed)
        console.echo(
            message=(
                f"Plane {plane_index} channel 2 spike deconvolution: complete. "
                f"Time taken: {timing.deconvolution_time_channel_2} seconds."
            ),
            level=LogLevel.SUCCESS,
        )
    else:
        console.echo(
            message=(
                f"Skipping plane {plane_index} channel 2 spike deconvolution, as the 'extract_spikes' configuration "
                f"parameter is set to False."
            ),
            level=LogLevel.WARNING,
        )
        extraction_data.subtracted_fluorescence_channel_2 = np.zeros_like(extraction_data.cell_fluorescence_channel_2)
        extraction_data.spikes_channel_2 = np.zeros_like(extraction_data.cell_fluorescence_channel_2)

    # Channel 1 ROI statistics always exist at this point, so the guard never takes the negative branch.
    if extraction_data.roi_statistics is not None:  # pragma: no branch
        extraction_data.cell_colocalization = compute_spatial_colocalization(
            rois_channel_1=extraction_data.roi_statistics,
            rois_channel_2=roi_statistics_channel_2,
            frame_height=frame_height,
            frame_width=frame_width,
            colocalization_threshold=extraction_config.colocalization_threshold,
        )


def _extract_multi_recording_channel(
    frames: BinaryFileCombined,
    roi_statistics: list[ROIStatistics],
    extraction_config: SignalExtraction,
    deconvolution_config: SpikeDeconvolution,
    channel_label: str,
    time_constant: float,
    sampling_rate: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], int]:
    """Extracts fluorescence, computes delta-F, and deconvolves spikes for one channel of a multi-recording extraction.

    Notes:
        Always uses ``allow_overlap=True`` since multi-recording template masks are spatially distinct by construction.
        No reclassification is performed because tracked ROIs are already known cells.

    Args:
        frames: The combined multi-plane binary data source for the channel being processed.
        roi_statistics: The backward-transformed ROI statistics for the channel.
        extraction_config: The signal extraction configuration parameters.
        deconvolution_config: The spike deconvolution configuration parameters.
        channel_label: A descriptive label for the channel being processed, used in log messages.
        time_constant: The timescale of the calcium indicator sensor in seconds.
        sampling_rate: The per-plane sampling rate in Hertz.

    Returns:
        A tuple of four arrays and the deconvolution time in seconds. The arrays are the cell fluorescence, the
        neuropil fluorescence, the neuropil-and-baseline-corrected delta fluorescence, and the deconvolved spikes,
        each with shape (roi_count, frame_count). If spike extraction is disabled, the delta fluorescence and spikes
        arrays are filled with zeroes and the reported time is zero.
    """
    roi_masks, neuropil_masks = _create_and_unpack_masks(
        roi_statistics=roi_statistics,
        frame_height=frames.height,
        frame_width=frames.width,
        extract_neuropil=extraction_config.extract_neuropil,
        allow_overlap=True,
        cell_probability_percentile=extraction_config.cell_probability_percentile,
        inner_neuropil_border_radius=extraction_config.inner_neuropil_border_radius,
        minimum_neuropil_pixels=extraction_config.minimum_neuropil_pixels,
        channel_label=channel_label,
    )

    cell_fluorescence, neuropil_fluorescence = _extract_fluorescence_traces(
        frames=frames,
        roi_masks=roi_masks,
        neuropil_masks=neuropil_masks,
        batch_size=extraction_config.batch_size,
        channel_label=channel_label,
    )

    # The deconvolution is timed separately from the trace extraction above, because the caller persists the two
    # durations as disjoint fields of the recording's timing data.
    deconvolution_time = 0
    if deconvolution_config.extract_spikes:
        timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
        timer.reset()
        subtracted_fluorescence = compute_delta_fluorescence(
            cell_fluorescence=cell_fluorescence,
            neuropil_fluorescence=neuropil_fluorescence,
            neuropil_coefficient=deconvolution_config.neuropil_coefficient,
            baseline_method=str(deconvolution_config.baseline_method),
            baseline_window=deconvolution_config.baseline_window,
            baseline_sigma=deconvolution_config.baseline_sigma,
            baseline_percentile=deconvolution_config.baseline_percentile,
            sampling_rate=sampling_rate,
        )
        spikes = apply_oasis_deconvolution(
            cell_fluorescence=subtracted_fluorescence,
            batch_size=extraction_config.batch_size,
            time_constant=time_constant,
            sampling_rate=sampling_rate,
        )
        deconvolution_time = int(timer.elapsed)
        console.echo(
            message=(
                f"{channel_label.capitalize()} spike deconvolution: complete. Time taken: {deconvolution_time} seconds."
            ),
            level=LogLevel.SUCCESS,
        )
    else:
        console.echo(
            message=(
                f"Skipping {channel_label} spike deconvolution, as the 'extract_spikes' configuration parameter is "
                f"set to False."
            ),
            level=LogLevel.WARNING,
        )
        subtracted_fluorescence = np.zeros_like(cell_fluorescence)
        spikes = np.zeros_like(cell_fluorescence)

    return cell_fluorescence, neuropil_fluorescence, subtracted_fluorescence, spikes, deconvolution_time


def _validate_registered_binaries(binary_paths: list[Path], recording_id: str) -> None:
    """Verifies that no interrupted write left any of the read plane binaries in an indeterminate state.

    Notes:
        The binarization stage fills each plane binary and the registration stage rewrites it in place, and both mark
        the binary for the duration of that write, each under its own phase name. A run that dies partway therefore
        leaves finished frames up to an unknown point and unfinished frames after it.

    Args:
        binary_paths: The paths of the plane binaries the extraction reads.
        recording_id: The identifier of the recording being processed, used to identify it in the error message.

    Raises:
        RuntimeError: If a marker shows that a previous write of one of the binaries was interrupted.
    """
    for binary_path in binary_paths:
        marker_path = resolve_active_binary_marker(binary_path=binary_path)
        if marker_path is not None:
            message = (
                f"Unable to extract multi-recording traces for recording {recording_id}. A previous write of the "
                f"binary file '{binary_path}' was interrupted, so the file holds finished frames up to an unknown "
                f"point and unfinished frames after it. Enable 'file_io.repeat_binarization' in that recording's "
                f"single-recording configuration and re-run its single-recording pipeline, which rebuilds the binary "
                f"from its source TIFF files and clears the marker at '{marker_path}'."
            )
            console.error(message=message, error=RuntimeError)


def _extract_multi_recording(context: MultiRecordingRuntimeContext) -> None:
    """Extracts fluorescence traces from ROIs tracked across multiple recordings for a single recording.

    Notes:
        Expects that the multi-recording discovery phase has already been completed, meaning backward-transformed ROI
        statistics are available in the recording's extraction data.

    Args:
        context: The MultiRecordingRuntimeContext for the recording being processed. Modified in-place to
            store extraction outputs including fluorescence traces, delta fluorescence, deconvolved spikes,
            and colocalization data.

    Raises:
        RuntimeError: If the combined single-recording data is not loaded, if backward-transformed ROI statistics are
            not available, or if an interrupted registration left one of the recording's plane binaries marked.
    """
    extraction_config = context.configuration.signal_extraction
    deconvolution_config = context.configuration.spike_deconvolution
    extraction_data = context.runtime.extraction
    combined_data = context.runtime.combined_data
    recording_id = context.runtime.io.recording_id

    # Loads extraction arrays from the previous stage (backward projection) if not in memory.
    output_path = context.runtime.output_path
    if output_path is not None and extraction_data.roi_statistics is None:
        extraction_data.load_arrays(output_path=output_path)

    if combined_data is None:
        message = (
            f"Unable to extract multi-recording traces for recording {recording_id}. The combined "
            f"single-recording data is not loaded. Ensure the single-recording pipeline completed "
            f"successfully and the data has not been moved or deleted."
        )
        console.error(message=message, error=RuntimeError)

    frame_height = combined_data.combined_height
    frame_width = combined_data.combined_width
    tau = combined_data.tau
    sampling_rate = combined_data.sampling_rate

    # Reads per-plane geometry and binary paths from combined data, which caches this information from the
    # single-recording pipeline to avoid reloading full single-recording contexts.
    plane_heights = combined_data.plane_heights
    plane_widths = combined_data.plane_widths
    y_offsets = combined_data.plane_y_offsets
    x_offsets = combined_data.plane_x_offsets

    roi_statistics = extraction_data.roi_statistics
    if roi_statistics is None:
        message = (
            f"Unable to extract multi-recording traces for recording {recording_id}. "
            f"Backward-transformed ROI statistics are not available. Ensure the multi-recording "
            f"discovery phase (registration, tracking, backward transform) has been completed before "
            f"running extraction."
        )
        console.error(message=message, error=RuntimeError)

    channel_1_binary_paths: list[Path] = list(combined_data.registered_binary_paths)
    _validate_registered_binaries(binary_paths=channel_1_binary_paths, recording_id=recording_id)

    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    with BinaryFileCombined(
        height=frame_height,
        width=frame_width,
        plane_heights=plane_heights,
        plane_widths=plane_widths,
        plane_y_coordinates=y_offsets,
        plane_x_coordinates=x_offsets,
        file_paths=channel_1_binary_paths,
    ) as binary:
        (
            extraction_data.cell_fluorescence,
            extraction_data.neuropil_fluorescence,
            extraction_data.subtracted_fluorescence,
            extraction_data.spikes,
            deconvolution_time,
        ) = _extract_multi_recording_channel(
            frames=binary,
            roi_statistics=roi_statistics,
            extraction_config=extraction_config,
            deconvolution_config=deconvolution_config,
            channel_label=f"recording {recording_id} channel 1",
            time_constant=tau,
            sampling_rate=sampling_rate,
        )

    _update_roi_extraction_statistics(
        roi_statistics=roi_statistics,
        cell_fluorescence=extraction_data.cell_fluorescence,
        neuropil_fluorescence=extraction_data.neuropil_fluorescence,
        neuropil_coefficient=deconvolution_config.neuropil_coefficient,
    )

    # Records the two segments disjointly, so that the deconvolution time is not also counted as extraction time and
    # their sum is the whole phase.
    timing = context.runtime.timing
    timing.deconvolution_time = deconvolution_time
    timing.extraction_time = int(timer.elapsed) - deconvolution_time

    # Backward-transformed channel 2 tracked ROI statistics exist only for a dual-channel recording whose channels were
    # both functional during single-recording processing.
    roi_statistics_channel_2 = extraction_data.roi_statistics_channel_2
    if roi_statistics_channel_2 is not None:
        channel_2_binary_paths: list[Path] = list(
            combined_data.registered_binary_paths_channel_2  # type: ignore[arg-type]
        )
        _validate_registered_binaries(binary_paths=channel_2_binary_paths, recording_id=recording_id)

        timer.reset()

        with BinaryFileCombined(
            height=frame_height,
            width=frame_width,
            plane_heights=plane_heights,
            plane_widths=plane_widths,
            plane_y_coordinates=y_offsets,
            plane_x_coordinates=x_offsets,
            file_paths=channel_2_binary_paths,
        ) as binary_channel_2:
            (
                extraction_data.cell_fluorescence_channel_2,
                extraction_data.neuropil_fluorescence_channel_2,
                extraction_data.subtracted_fluorescence_channel_2,
                extraction_data.spikes_channel_2,
                deconvolution_time_channel_2,
            ) = _extract_multi_recording_channel(
                frames=binary_channel_2,
                roi_statistics=roi_statistics_channel_2,
                extraction_config=extraction_config,
                deconvolution_config=deconvolution_config,
                channel_label=f"recording {recording_id} channel 2",
                time_constant=tau,
                sampling_rate=sampling_rate,
            )

        _update_roi_extraction_statistics(
            roi_statistics=roi_statistics_channel_2,
            cell_fluorescence=extraction_data.cell_fluorescence_channel_2,
            neuropil_fluorescence=extraction_data.neuropil_fluorescence_channel_2,
            neuropil_coefficient=deconvolution_config.neuropil_coefficient,
        )

        timing.deconvolution_time += deconvolution_time_channel_2
        timing.extraction_time += int(timer.elapsed) - deconvolution_time_channel_2

        extraction_data.cell_colocalization = compute_spatial_colocalization(
            rois_channel_1=roi_statistics,
            rois_channel_2=roi_statistics_channel_2,
            frame_height=frame_height,
            frame_width=frame_width,
            colocalization_threshold=extraction_config.colocalization_threshold,
        )

    # Totals the phase before the save, since the save is what persists the timing data for this recording.
    total_extraction_time = timing.extraction_time + timing.deconvolution_time
    timing.total_extraction_time = total_extraction_time

    context.save_runtime()

    context.runtime.extraction.release_arrays()

    console.echo(
        message=(
            f"Recording {recording_id} multi-recording extraction: complete. "
            f"Total time: {total_extraction_time} seconds."
        ),
        level=LogLevel.SUCCESS,
    )
