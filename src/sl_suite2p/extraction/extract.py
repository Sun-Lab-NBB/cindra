"""Provides the fluorescence extraction entry point for the single-day and the multi-day processing pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit, prange  # type: ignore[import-untyped]
import numpy as np
from scipy import stats
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import LogLevel, console

from ..io import BinaryFile
from .masks import create_masks
from .deconvolve import apply_oasis_deconvolution, compute_delta_fluorescence
from .colocalization import compute_spatial_colocalization, compute_intensity_colocalization
from ..classification import classify

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..dataclasses import ROIStatistics, RuntimeContext


@njit(cache=True, parallel=True)  # type: ignore[untyped-decorator]
def _extract_cell_fluorescence(
    output_prototype: NDArray[np.float32],
    data: NDArray[np.float32],
    flat_cell_masks: NDArray[np.int32],
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
        flat_cell_masks: Flattened array containing all cell mask indices concatenated together.
        flat_lambda_weights: Flattened array containing all lambda weights concatenated together.
        mask_offsets: Array of offsets indicating where each cell's mask starts in the flattened arrays.
            Has length (cell_count + 1), where mask_offsets[i+1] - mask_offsets[i] gives the mask size for cell i.

    Returns:
        The output_prototype array updated with the extracted cell fluorescence traces for each processed ROI.
    """
    cell_count = output_prototype.shape[0]
    frame_count = data.shape[0]

    for cell_index in prange(cell_count):
        start = mask_offsets[cell_index]
        end = mask_offsets[cell_index + 1]

        # Accumulates lambda-weighted pixel fluorescence directly from scattered reads, avoiding a per-cell
        # temporary array allocation. Weights bias the trace toward pixels more likely to belong to the cell.
        for frame_index in range(frame_count):
            accumulator = np.float32(0.0)
            for pixel_offset in range(start, end):
                accumulator += data[frame_index, flat_cell_masks[pixel_offset]] * flat_lambda_weights[pixel_offset]
            output_prototype[cell_index, frame_index] = accumulator

    return output_prototype


@njit(cache=True, parallel=True)  # type: ignore[untyped-decorator]
def _extract_neuropil_fluorescence(
    output_prototype: NDArray[np.float32],
    data: NDArray[np.float32],
    flat_neuropil_masks: NDArray[np.int32],
    mask_offsets: NDArray[np.int32],
    neuropil_pixel_count: NDArray[np.int32],
) -> NDArray[np.float32]:
    """Extracts neuropil fluorescence traces for the requested ROIs.

    Args:
        output_prototype: The pre-initialized output array to be updated with the extracted fluorescence traces.
        data: The raw activity data from which to extract the fluorescence traces.
        flat_neuropil_masks: Flattened array containing all neuropil mask indices concatenated together.
        mask_offsets: Array of offsets indicating where each cell's neuropil mask starts in the flattened array.
        neuropil_pixel_count: The number of pixels in each neuropil mask.

    Returns:
        The output_prototype array updated with the extracted neuropil fluorescence traces for each processed ROI.
    """
    cell_count = output_prototype.shape[0]
    frame_count = data.shape[0]

    for cell_index in prange(cell_count):
        start = mask_offsets[cell_index]
        end = mask_offsets[cell_index + 1]

        # Pre-computes the reciprocal of the neuropil pixel count to replace per-frame division with multiplication.
        reciprocal = np.float32(1.0) / np.float32(neuropil_pixel_count[cell_index])

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
    cell_masks = tuple((indices, weights) for indices, weights, _ in per_roi_masks)
    neuropil_masks = (
        tuple(neuropil for _, _, neuropil in per_roi_masks if neuropil is not None)
        if per_roi_masks[0][2] is not None
        else None
    )

    console.echo(
        message=f"{channel_label.capitalize()} ROI masks: created. Time taken: {timer.elapsed} seconds.",
        level=LogLevel.SUCCESS,
    )

    return cell_masks, neuropil_masks


def _extract_fluorescence_traces(
    frames: BinaryFile,
    cell_masks: tuple[tuple[NDArray[np.int32], NDArray[np.float32]], ...],
    neuropil_masks: tuple[NDArray[np.int32], ...] | None,
    batch_size: int,
    channel_label: str,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Extracts the fluorescence traces from the raw cell activity data using cell and neuropil masks.

    Notes:
        If neuropil masks are not provided, the neuropil fluorescence traces are returned as an array of zeroes.

    Args:
        frames: The raw cell activity data (movie) to process.
        cell_masks: The cell masks for each ROI, where each element is a tuple of (flattened pixel indices,
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

    # Extracts processed recording statistics.
    frame_count, height, width = frames.shape
    cell_count = len(cell_masks)
    pixel_count = height * width

    # Pre-allocates the arrays to store the extracted cell and neuropil fluorescence traces.
    fluorescence = np.zeros((cell_count, frame_count), dtype=np.float32)
    neuropil_fluorescence = np.zeros((cell_count, frame_count), dtype=np.float32)

    # Flattens cell masks and lambda weights into contiguous arrays with offset pointers. This format avoids Numba's
    # tuple size limitations and enables efficient parallel processing.
    cell_mask_sizes = np.array([len(pixel_indices) for pixel_indices, _ in cell_masks], dtype=np.int32)
    cell_mask_offsets = np.zeros(cell_count + 1, dtype=np.int32)
    cell_mask_offsets[1:] = np.cumsum(cell_mask_sizes)

    total_cell_pixels = int(cell_mask_offsets[-1])
    flat_cell_masks = np.empty(total_cell_pixels, dtype=np.int32)
    flat_lambda_weights = np.empty(total_cell_pixels, dtype=np.float32)

    for mask_index, (pixel_indices, lambda_weights) in enumerate(cell_masks):
        start = cell_mask_offsets[mask_index]
        end = cell_mask_offsets[mask_index + 1]
        flat_cell_masks[start:end] = pixel_indices
        flat_lambda_weights[start:end] = lambda_weights

    # Flattens neuropil masks into contiguous arrays with offset pointers if provided.
    if neuropil_masks is not None:
        neuropil_mask_sizes = np.array([len(indices) for indices in neuropil_masks], dtype=np.int32)
        neuropil_mask_offsets = np.zeros(cell_count + 1, dtype=np.int32)
        neuropil_mask_offsets[1:] = np.cumsum(neuropil_mask_sizes)

        total_neuropil_pixels = int(neuropil_mask_offsets[-1])
        flat_neuropil_masks = np.empty(total_neuropil_pixels, dtype=np.int32)
        neuropil_pixel_count = np.zeros(cell_count, dtype=np.int32)

        for mask_index, indices in enumerate(neuropil_masks):
            start = neuropil_mask_offsets[mask_index]
            end = neuropil_mask_offsets[mask_index + 1]
            flat_neuropil_masks[start:end] = indices
            neuropil_pixel_count[mask_index] = len(indices)

    # Pre-allocates a reusable buffer for the extraction kernels. Both kernels write every element unconditionally,
    # so zeroing is unnecessary. Re-allocated only for the last batch if it is smaller than the standard batch size.
    output_prototype = np.empty((cell_count, batch_size), dtype=np.float32)

    # Extracts the cell fluorescence from all frames of the processed cell activity movie.
    for batch_start in range(0, frame_count, batch_size):
        batch_end = min(batch_start + batch_size, frame_count)

        # Reshapes each batch from [frames, height, width] to [frames, pixels].
        batch_data = frames[batch_start:batch_end].astype(np.float32)
        batch_frames = batch_data.shape[0]
        batch_pixels = batch_data.reshape(batch_frames, pixel_count)
        batch_slice = slice(batch_start, batch_start + batch_frames)

        # Re-allocates the buffer for the last batch if it is smaller than the standard batch size.
        if batch_frames < output_prototype.shape[1]:
            output_prototype = np.empty((cell_count, batch_frames), dtype=np.float32)

        # Extracts the cell fluorescence from all frames of the currently processed batch.
        fluorescence[:, batch_slice] = _extract_cell_fluorescence(
            output_prototype=output_prototype,
            data=batch_pixels,
            flat_cell_masks=flat_cell_masks,
            flat_lambda_weights=flat_lambda_weights,
            mask_offsets=cell_mask_offsets,
        )

        # If neuropil masks are provided, extracts the neuropil fluorescence from the current batch.
        if neuropil_masks is not None:
            # noinspection PyUnboundLocalVariable
            neuropil_fluorescence[:, batch_slice] = _extract_neuropil_fluorescence(
                output_prototype=output_prototype,
                data=batch_pixels,
                flat_neuropil_masks=flat_neuropil_masks,
                mask_offsets=neuropil_mask_offsets,
                neuropil_pixel_count=neuropil_pixel_count,
            )

    console.echo(
        message=(
            f"{channel_label.capitalize()} ROI fluorescence: extracted from {cell_count} ROIs in {frame_count} "
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
    """Computes neuropil-corrected skewness and standard deviation and stores them in the ROI statistics.

    Args:
        roi_statistics: The ROI statistics to update in-place with the computed skewness and standard deviation values.
        cell_fluorescence: The extracted cell fluorescence traces with shape (roi_count, frame_count).
        neuropil_fluorescence: The extracted neuropil fluorescence traces with shape (roi_count, frame_count).
        neuropil_coefficient: The scaling factor applied to neuropil fluorescence before subtraction.
    """
    corrected = cell_fluorescence - np.float32(neuropil_coefficient) * neuropil_fluorescence
    skew_values = np.asarray(stats.skew(a=corrected, axis=1))
    std_values = np.std(corrected, axis=1)

    for roi, skewness_value, standard_deviation_value in zip(roi_statistics, skew_values, std_values, strict=True):
        roi.skewness = float(skewness_value)
        roi.standard_deviation = float(standard_deviation_value)


def extract_traces(context: RuntimeContext) -> None:
    """Extracts fluorescence traces, classifies ROIs, and deconvolves spikes from registered binary data.

    Notes:
        This function orchestrates the full extraction pipeline for one or both channels. For structural channel 2
        data, channel 1 masks are reused and intensity colocalization is computed. For functional channel 2 data,
        independent masks are created and spatial colocalization is computed between the two channel's ROIs. Results
        are written into context.runtime.extraction and context.runtime.timing.

    Args:
        context: The RuntimeContext containing configuration, file paths, and mutable runtime data structures. Modified
            in-place to store extraction outputs including fluorescence traces, classification results, deconvolved
            spikes, and colocalization data.

    Raises:
        RuntimeError: If detection has not been run (no ROI statistics available) or if the registered binary path is
            not set.
    """
    # Extracts configuration.
    extraction_config = context.configuration.signal_extraction
    deconvolution_config = context.configuration.spike_deconvolution
    main_config = context.configuration.main

    # Extracts runtime data.
    io_data = context.runtime.io
    extraction_data = context.runtime.extraction
    timing = context.runtime.timing

    plane_index = io_data.plane_index if io_data.plane_index is not None else 0
    frame_height = io_data.frame_height
    frame_width = io_data.frame_width
    batch_size = extraction_config.batch_size

    # Validates that detection has been run and the registered binary path is available.
    if extraction_data.roi_statistics is None:
        console.error(
            message=(
                f"Unable to run extraction for plane {plane_index}. ROI detection must run before extraction, but "
                f"no ROI statistics are available."
            ),
            error=RuntimeError,
        )

    channel_1_path = io_data.registered_binary_path
    if channel_1_path is None:
        console.error(
            message=(
                f"Unable to run extraction for plane {plane_index}. The registered binary file path is not set "
                f"for channel 1."
            ),
            error=RuntimeError,
        )

    roi_statistics = extraction_data.roi_statistics
    channel_1_label = f"plane {plane_index} channel 1"

    # Creates cell and neuropil masks for channel 1.
    cell_masks, neuropil_masks = _create_and_unpack_masks(
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

    # Extracts channel 1 fluorescence traces.
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
            cell_masks=cell_masks,
            neuropil_masks=neuropil_masks,
            batch_size=batch_size,
            channel_label=channel_1_label,
        )

    timing.extraction_time = int(timer.elapsed)

    # Computes neuropil-corrected skewness and standard deviation for channel 1 ROIs.
    _update_roi_extraction_statistics(
        roi_statistics=roi_statistics,
        cell_fluorescence=extraction_data.cell_fluorescence,
        neuropil_fluorescence=extraction_data.neuropil_fluorescence,
        neuropil_coefficient=deconvolution_config.neuropil_coefficient,
    )

    # Classifies channel 1 ROIs.
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

    # Computes delta fluorescence and spike deconvolution for channel 1.
    timer.reset()
    if deconvolution_config.extract_spikes:
        extraction_data.subtracted_fluorescence = compute_delta_fluorescence(
            roi_fluorescence=extraction_data.cell_fluorescence,
            neuropil_fluorescence=extraction_data.neuropil_fluorescence,
            neuropil_coefficient=deconvolution_config.neuropil_coefficient,
            baseline_method=str(deconvolution_config.baseline_method),
            baseline_window=deconvolution_config.baseline_window,
            baseline_sigma=deconvolution_config.baseline_sigma,
            baseline_percentile=deconvolution_config.baseline_percentile,
            sampling_rate=io_data.sampling_rate,
        )
        extraction_data.spikes = apply_oasis_deconvolution(
            roi_fluorescence=extraction_data.subtracted_fluorescence,
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
            # Functional channel 2: creates independent masks from channel 2 ROI statistics.
            _extract_functional_channel_2(context=context, batch_size=batch_size)
        else:
            # Structural channel 2: reuses channel 1 masks for extraction and computes intensity colocalization.
            _extract_structural_channel_2(
                context=context,
                batch_size=batch_size,
                cell_masks=cell_masks,
                neuropil_masks=neuropil_masks,
            )

    # Saves updated runtime data to disk.
    context.save_runtime()


def _extract_structural_channel_2(
    context: RuntimeContext,
    batch_size: int,
    cell_masks: tuple[tuple[NDArray[np.int32], NDArray[np.float32]], ...],
    neuropil_masks: tuple[NDArray[np.int32], ...] | None,
) -> None:
    """Extracts structural channel 2 fluorescence using channel 1 masks and computes intensity colocalization.

    Args:
        context: The RuntimeContext containing configuration and mutable runtime data. Modified in-place to store
            channel 2 fluorescence traces, colocalization results, and the corrected structural mean image.
        batch_size: The number of frames to process at the same time.
        cell_masks: The channel 1 cell masks to reuse for channel 2 extraction.
        neuropil_masks: The channel 1 neuropil masks to reuse for channel 2 extraction.
    """
    io_data = context.runtime.io
    detection_data = context.runtime.detection
    extraction_data = context.runtime.extraction
    main_config = context.configuration.main

    plane_index = io_data.plane_index if io_data.plane_index is not None else 0
    channel_2_path = io_data.registered_binary_path_channel_2
    channel_2_label = f"plane {plane_index} channel 2"

    if channel_2_path is None:
        console.error(
            message=(
                f"Unable to run extraction for {channel_2_label}. The registered binary file path is not set "
                f"for channel 2."
            ),
            error=RuntimeError,
        )

    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    # Extracts channel 2 fluorescence using channel 1 masks.
    with BinaryFile(
        height=io_data.frame_height,
        width=io_data.frame_width,
        file_path=channel_2_path,
        frame_number=io_data.frame_count,
    ) as binary:
        extraction_data.cell_fluorescence_channel_2, extraction_data.neuropil_fluorescence_channel_2 = (
            _extract_fluorescence_traces(
                frames=binary,
                cell_masks=cell_masks,
                neuropil_masks=neuropil_masks,
                batch_size=batch_size,
                channel_label=channel_2_label,
            )
        )

    context.runtime.timing.extraction_time_channel_2 = int(timer.elapsed)

    # Computes intensity colocalization between functional channel 1 ROIs and the structural channel 2 image.
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
                colocalization_threshold=main_config.colocalization_threshold,
                allow_overlap=extraction_config.allow_overlap,
                cell_probability_percentile=extraction_config.cell_probability_percentile,
                inner_neuropil_border_radius=extraction_config.inner_neuropil_border_radius,
                minimum_neuropil_pixels=extraction_config.minimum_neuropil_pixels,
            )
        )


def _extract_functional_channel_2(
    context: RuntimeContext,
    batch_size: int,
) -> None:
    """Extracts functional channel 2 fluorescence with independent masks and computes spatial colocalization.

    Notes:
        When both channels are functional, channel 2 has its own independently detected ROIs. This function creates
        masks from those ROIs, extracts fluorescence, classifies ROIs, computes delta fluorescence and spike
        deconvolution, and finally computes spatial colocalization between channel 1 and channel 2 ROIs.

    Args:
        context: The RuntimeContext containing configuration and mutable runtime data. Modified in-place to store
            channel 2 extraction results and colocalization data.
        batch_size: The number of frames to process at the same time.
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
        console.error(
            message=(
                f"Unable to run extraction for {channel_2_label}. The registered binary file path is not set "
                f"for channel 2."
            ),
            error=RuntimeError,
        )

    # Validates that channel 2 ROI statistics exist from detection.
    roi_statistics_channel_2 = extraction_data.roi_statistics_channel_2
    if roi_statistics_channel_2 is None:
        console.error(
            message=(
                f"Unable to run functional channel 2 extraction for plane {plane_index}. Channel 2 ROI detection "
                f"must run before extraction, but no channel 2 ROI statistics are available."
            ),
            error=RuntimeError,
        )

    # Creates independent masks from channel 2 ROI statistics.
    channel_2_cell_masks, channel_2_neuropil_masks = _create_and_unpack_masks(
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

    # Extracts channel 2 fluorescence traces using channel 2 masks.
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
                cell_masks=channel_2_cell_masks,
                neuropil_masks=channel_2_neuropil_masks,
                batch_size=batch_size,
                channel_label=channel_2_label,
            )
        )

    timing.extraction_time_channel_2 = int(timer.elapsed)

    # Computes neuropil-corrected skewness and standard deviation for channel 2 ROIs.
    _update_roi_extraction_statistics(
        roi_statistics=roi_statistics_channel_2,
        cell_fluorescence=extraction_data.cell_fluorescence_channel_2,
        neuropil_fluorescence=extraction_data.neuropil_fluorescence_channel_2,
        neuropil_coefficient=deconvolution_config.neuropil_coefficient,
    )

    # Classifies channel 2 ROIs.
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

    # Computes delta fluorescence and spike deconvolution for channel 2.
    timer.reset()
    if deconvolution_config.extract_spikes:
        extraction_data.subtracted_fluorescence_channel_2 = compute_delta_fluorescence(
            roi_fluorescence=extraction_data.cell_fluorescence_channel_2,
            neuropil_fluorescence=extraction_data.neuropil_fluorescence_channel_2,
            neuropil_coefficient=deconvolution_config.neuropil_coefficient,
            baseline_method=str(deconvolution_config.baseline_method),
            baseline_window=deconvolution_config.baseline_window,
            baseline_sigma=deconvolution_config.baseline_sigma,
            baseline_percentile=deconvolution_config.baseline_percentile,
            sampling_rate=io_data.sampling_rate,
        )
        extraction_data.spikes_channel_2 = apply_oasis_deconvolution(
            roi_fluorescence=extraction_data.subtracted_fluorescence_channel_2,
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

    # Computes spatial colocalization between channel 1 and channel 2 ROIs.
    if extraction_data.roi_statistics is not None:
        channel_1_to_2, _channel_2_to_1 = compute_spatial_colocalization(
            rois_channel_1=extraction_data.roi_statistics,
            rois_channel_2=roi_statistics_channel_2,
            frame_height=frame_height,
            frame_width=frame_width,
            colocalization_threshold=main_config.colocalization_threshold,
        )
        extraction_data.cell_colocalization = channel_1_to_2
