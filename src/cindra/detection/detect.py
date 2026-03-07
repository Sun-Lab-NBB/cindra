"""Provides the ROI detection entry point for the single-day and the multi-day processing pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING
from itertools import compress

import numpy as np
from scipy.signal import medfilt2d
from ataraxis_time import PrecisionTimer, TimerPrecisions
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]
from ataraxis_base_utilities import LogLevel, console

from ..io import BinaryFile
from .denoise import pca_denoise
from .detect_rois import detect
from .roi_statistics import compute_roi_statistics
from ..classification import classify

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from ..dataclasses import ROIDetection, ROIStatistics, RuntimeContext

_ITERATION_MULTIPLIER: int = 250
"""The multiplier applied to the user-facing maximum_iterations parameter to determine the actual iteration limit used
by the sparse detection algorithm."""

_BACKGROUND_SCALE: int = 4
"""The spatial multiplier applied to the cell diameter to compute the median filter kernel size for background removal
in the enhanced mean image."""

_ENHANCED_MINIMUM_INTENSITY: float = -6.0
"""The lower intensity clipping bound applied after local contrast normalization when producing the enhanced mean
image."""

_ENHANCED_MAXIMUM_INTENSITY: float = 6.0
"""The upper intensity clipping bound applied after local contrast normalization when producing the enhanced mean
image."""

_VARIANCE_EPSILON: float = 1e-10
"""The small constant added to local variance to avoid division by zero during contrast normalization."""

_DEFAULT_CELL_DIAMETER: int = 12
"""The default cell diameter in pixels, used when the estimated diameter is zero or negative."""

type _ChannelDetectionResult = tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], int, list[ROIStatistics]
]
"""The type alias for the _detect_channel return signature containing mean image, enhanced mean image, maximum
projection, correlation map, cell diameter, and ROI statistics."""


def detect_plane_rois(context: RuntimeContext) -> None:
    """Detects ROIs from registered binary data and updates the runtime context in-place.

    Notes:
        This function orchestrates the full detection pipeline for one or both functional channels. When both channels
        are functional (independent ROI detection), the pipeline runs independently on each channel since different
        cell populations may have different soma sizes and spatial scales. Results are written into
        context.runtime.detection, context.runtime.extraction, and context.runtime.timing.

    Args:
        context: The RuntimeContext containing configuration, file paths, and mutable runtime data structures. Modified
            in-place to store detection outputs including ROI statistics, image projections, and timing data.

    Raises:
        ValueError: If no ROIs are detected on either channel.
    """
    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    # Extracts configuration.
    detection_config = context.configuration.roi_detection
    main_config = context.configuration.main
    nonrigid_block_size = context.configuration.nonrigid_registration.block_size
    custom_classifier_path = main_config.custom_classifier_path

    # Extracts runtime data.
    io_data = context.runtime.io
    registration_data = context.runtime.registration
    detection_data = context.runtime.detection

    # Memory-maps registration arrays from the previous stage (needed for bad_frames).
    output_path = context.runtime.io.output_path
    if output_path is not None and registration_data.bad_frames is None:
        registration_data.memory_map_arrays(output_path)

    plane_index = io_data.plane_index if io_data.plane_index is not None else 0
    frame_height = io_data.frame_height
    frame_width = io_data.frame_width

    # Computes the bin size for temporal averaging. The bin size is the maximum of 1, the ratio of total frames to the
    # maximum number of binned frames, and the number of frames per sensor time constant.
    bin_size = int(
        max(
            1,
            io_data.frame_count // detection_config.maximum_binned_frames,
            np.round(main_config.tau * io_data.sampling_rate),
        )
    )

    valid_y_range = registration_data.valid_y_range
    valid_x_range = registration_data.valid_x_range
    parallel_workers = context.configuration.runtime.parallel_workers

    # Validates that the registered binary path exists. This is always satisfied when called from the processing
    # pipeline, since registration creates the binary file before detection runs.
    channel_1_path = io_data.registered_binary_path
    if channel_1_path is None:
        console.error(
            message="Unable to run ROI detection: registered binary file path is not set for channel 1.",
            error=RuntimeError,
        )

    # Runs channel 1 detection.
    mean_image, enhanced_mean_image, maximum_projection, correlation_map, cell_diameter, roi_statistics = (
        _detect_channel(
            binary_path=channel_1_path,
            frame_height=frame_height,
            frame_width=frame_width,
            frame_count=io_data.frame_count,
            bin_size=bin_size,
            valid_y_range=valid_y_range,
            valid_x_range=valid_x_range,
            bad_frames=registration_data.bad_frames,
            detection_config=detection_config,
            nonrigid_block_size=nonrigid_block_size,
            parallel_workers=parallel_workers,
            custom_classifier_path=custom_classifier_path,
            plane_index=plane_index,
            channel_label="channel 1",
        )
    )

    # Computes the aggregate aspect ratio as the median across all detected ROIs.
    aspect_ratios = np.array([roi.aspect_ratio for roi in roi_statistics], dtype=np.float32)
    detection_data.aspect_ratio = float(np.median(aspect_ratios)) if len(aspect_ratios) > 0 else 0.0

    # Stores channel 1 detection results.
    detection_data.mean_image = mean_image
    detection_data.enhanced_mean_image = enhanced_mean_image
    detection_data.maximum_projection = maximum_projection
    detection_data.correlation_map = correlation_map
    detection_data.cell_diameter = cell_diameter
    for roi in roi_statistics:
        roi.mask.frame_width = frame_width
    context.runtime.extraction.roi_statistics = roi_statistics

    # Records channel 1 detection time.
    elapsed_seconds = int(timer.elapsed)
    context.runtime.timing.detection_time = elapsed_seconds
    console.echo(
        message=f"Plane {plane_index} channel 1 ROI detection: complete. Time taken: {elapsed_seconds} seconds.",
        level=LogLevel.SUCCESS,
    )

    # Runs channel 2 detection only when both hardware channels are functional, meaning channel_2_data.bin contains
    # independently detectable functional data. When only the second hardware channel is functional, the import layer
    # swaps it into channel_1_data.bin, so channel_2_data.bin holds non-functional data and must not be detected.
    channel_2_path = io_data.registered_binary_path_channel_2
    if main_config.first_channel_functional and main_config.second_channel_functional and channel_2_path is not None:
        timer.reset()

        (
            mean_image_channel_2,
            enhanced_mean_image_channel_2,
            maximum_projection_channel_2,
            correlation_map_channel_2,
            cell_diameter_channel_2,
            roi_statistics_channel_2,
        ) = _detect_channel(
            binary_path=channel_2_path,
            frame_height=frame_height,
            frame_width=frame_width,
            frame_count=io_data.frame_count,
            bin_size=bin_size,
            valid_y_range=valid_y_range,
            valid_x_range=valid_x_range,
            bad_frames=registration_data.bad_frames,
            detection_config=detection_config,
            nonrigid_block_size=nonrigid_block_size,
            parallel_workers=parallel_workers,
            custom_classifier_path=custom_classifier_path,
            plane_index=plane_index,
            channel_label="channel 2",
        )

        # Stores channel 2 detection results.
        detection_data.mean_image_channel_2 = mean_image_channel_2
        detection_data.enhanced_mean_image_channel_2 = enhanced_mean_image_channel_2
        detection_data.maximum_projection_channel_2 = maximum_projection_channel_2
        detection_data.correlation_map_channel_2 = correlation_map_channel_2
        detection_data.cell_diameter_channel_2 = cell_diameter_channel_2
        for roi in roi_statistics_channel_2:
            roi.mask.frame_width = frame_width
        context.runtime.extraction.roi_statistics_channel_2 = roi_statistics_channel_2

        # Records channel 2 detection time.
        elapsed_seconds = int(timer.elapsed)
        context.runtime.timing.detection_time_channel_2 = elapsed_seconds
        console.echo(
            message=f"Plane {plane_index} channel 2 ROI detection: complete. Time taken: {elapsed_seconds} seconds.",
            level=LogLevel.SUCCESS,
        )

    # Persists detection results to disk so that ROI statistics and detection images are not lost if extraction fails.
    context.save_runtime()

    # Releases registration and detection arrays to free memory. Extraction arrays (roi_statistics) are preserved
    # because the caller checks them to decide whether to run trace extraction.
    context.runtime.registration.release_arrays()
    context.runtime.detection.release_arrays()


def _create_enhanced_mean_image(
    mean_image: NDArray[np.float32],
    cell_diameter: int,
    valid_y_range: tuple[int, int],
    valid_x_range: tuple[int, int],
    frame_height: int,
    frame_width: int,
) -> NDArray[np.float32]:
    """Creates an enhanced version of the mean image by removing background fluorescence and normalizing local contrast.

    Notes:
        The enhancement pipeline applies a median filter at a scale proportional to the cell diameter to estimate and
        subtract the slowly varying background. The residual is then divided by its local absolute median to normalize
        contrast across the field of view. Finally, the result is clipped and rescaled to the [0, 1] range. Border
        regions outside the valid registration crop are filled with the minimum value of the enhanced interior.

    Args:
        mean_image: The mean image to enhance, already cropped to the valid registration region.
        cell_diameter: The estimated cell diameter in pixels, used to compute the median filter kernel size.
        valid_y_range: The valid Y pixel range (start, end) after registration cropping.
        valid_x_range: The valid X pixel range (start, end) after registration cropping.
        frame_height: The height of the full frame in pixels.
        frame_width: The width of the full frame in pixels.

    Returns:
        The enhanced mean image with shape (frame_height, frame_width), background-subtracted and contrast-normalized
        with values in [0, 1] inside the valid region.
    """
    # Uses cell diameter for spatial scaling, with a default fallback.
    spatial_scale_pixels = cell_diameter if cell_diameter > 0 else _DEFAULT_CELL_DIAMETER

    # Computes median filter kernel size proportional to the cell diameter.
    kernel_dimension = int(_BACKGROUND_SCALE * np.ceil(spatial_scale_pixels) + 1)
    filter_kernel_size = (kernel_dimension, kernel_dimension)

    # Subtracts background fluorescence using a median filter. Casts medfilt2d output to float32 to prevent float64
    # promotion of the entire downstream chain. Reuses the background array for the result.
    background_removed = medfilt2d(mean_image, kernel_size=filter_kernel_size).astype(np.float32)
    np.subtract(mean_image, background_removed, out=background_removed)

    # Normalizes cell contrast by dividing by local absolute median.
    abs_background_removed = np.abs(background_removed)
    local_variance = medfilt2d(abs_background_removed, kernel_size=filter_kernel_size).astype(np.float32)
    np.add(local_variance, _VARIANCE_EPSILON, out=local_variance)
    np.divide(background_removed, local_variance, out=background_removed)

    # Clips intensities and scales to [0, 1] range. The mean_image is already cropped to the valid region, so no
    # additional slicing is needed.
    clipped_roi = np.clip(background_removed, _ENHANCED_MINIMUM_INTENSITY, _ENHANCED_MAXIMUM_INTENSITY)
    scaled_roi = (clipped_roi - _ENHANCED_MINIMUM_INTENSITY) / (
        _ENHANCED_MAXIMUM_INTENSITY - _ENHANCED_MINIMUM_INTENSITY
    )

    # Places the enhanced image into a full-size array with border set to the minimum value.
    y_start, y_end = valid_y_range
    x_start, x_end = valid_x_range
    enhanced_image = np.full((frame_height, frame_width), scaled_roi.min(), dtype=np.float32)
    enhanced_image[y_start:y_end, x_start:x_end] = scaled_roi

    return enhanced_image


def _apply_preclassification(
    roi_statistics: list[ROIStatistics],
    frame_height: int,
    frame_width: int,
    preclassification_threshold: float,
    crop_to_soma: bool,
    custom_classifier_path: Path | None,
    plane_index: int,
    channel_label: str,
    diameter: int = 10,
) -> list[ROIStatistics]:
    """Filters detected ROIs using a lightweight pre-classification model before signal extraction.

    Notes:
        This function computes the minimal shape statistics needed for classification (compactness and normalized pixel
        count), runs a 2-feature logistic regression model, and removes ROIs whose cell probability falls below the
        threshold. Unlike the final classification stage performed by the extraction package, this does not require
        extracted fluorescence traces.

    Args:
        roi_statistics: The list of ROIStatistics instances to filter.
        frame_height: The height of the frame that contains the processed ROIs, in pixels.
        frame_width: The width of the frame that contains the processed ROIs, in pixels.
        preclassification_threshold: The minimum classifier probability for an ROI to be kept.
        crop_to_soma: Determines whether to crop dendritic regions before computing classification features.
        custom_classifier_path: The path to a custom classifier file, or None to use the built-in classifier.
        plane_index: The index of the imaging plane being processed, used for logging.
        channel_label: The channel identifier string used in log messages (e.g., "channel 1" or "channel 2").
        diameter: The estimated cell diameter in pixels, used for distance normalization in compactness computation.

    Returns:
        The filtered list of ROIStatistics instances that passed the preclassification threshold.
    """
    # Computes only the minimal statistics (compactness and normalized_pixel_count) needed by the pre-classifier,
    # skipping the expensive ellipse fitting, convex hull, and overlap computations.
    compute_roi_statistics(
        rois=roi_statistics,
        frame_height=frame_height,
        frame_width=frame_width,
        diameter=diameter,
        crop=crop_to_soma,
        lightweight=True,
    )

    is_cell = classify(
        roi_statistics=roi_statistics,
        custom_classifier_path=custom_classifier_path,
        preclassification=True,
    )

    # Vectorizes the threshold comparison in numpy, then filters with C-level itertools.compress.
    pass_mask = is_cell[:, 1] > preclassification_threshold
    kept = list(compress(roi_statistics, pass_mask))
    removed_count = len(roi_statistics) - len(kept)

    message = (
        f"Plane {plane_index} {channel_label} preclassification pass with confidence threshold "
        f"{preclassification_threshold}: complete. Removed {removed_count} ROIs."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)

    return kept


def _detect_channel(
    binary_path: Path,
    frame_height: int,
    frame_width: int,
    frame_count: int,
    bin_size: int,
    valid_y_range: tuple[int, int],
    valid_x_range: tuple[int, int],
    bad_frames: NDArray[np.bool_] | None,
    detection_config: ROIDetection,
    nonrigid_block_size: tuple[int, int],
    parallel_workers: int,
    custom_classifier_path: Path | None,
    plane_index: int,
    channel_label: str,
) -> _ChannelDetectionResult:
    """Runs the full detection pipeline for a single imaging channel.

    Notes:
        This function handles binning, optional denoising, sparse ROI detection, coordinate offset correction,
        optional preclassification, and final overlap filtering for one channel. It is called by detect_plane_rois
        once per functional channel.

    Args:
        binary_path: The absolute path to the registered binary file for this channel.
        frame_height: The height of each frame in pixels.
        frame_width: The width of each frame in pixels.
        frame_count: The total number of frames in the binary file.
        bin_size: The temporal bin size in frames.
        valid_y_range: The valid Y pixel range (start, end) after registration cropping.
        valid_x_range: The valid X pixel range (start, end) after registration cropping.
        bad_frames: A boolean array with shape (num_frames,) marking frames to exclude from binning, or None if no
            frames are excluded.
        detection_config: The ROIDetection configuration dataclass containing detection parameters.
        nonrigid_block_size: The nonrigid registration block size (height, width), used to derive the PCA denoising
            block dimensions.
        parallel_workers: The number of parallel threads for PCA denoising. Values of -1 or 0 use all available cores.
            A value of 1 disables parallelism.
        custom_classifier_path: The path to a custom classifier file, or None to use the built-in classifier.
        plane_index: The index of the imaging plane being processed, used for logging.
        channel_label: The channel identifier string used in log messages (e.g., "channel 1" or "channel 2").

    Returns:
        A tuple of the mean image, the enhanced mean image, the maximum intensity projection, the pixel-wise
        correlation map, the estimated cell diameter in pixels, and the list of ROIStatistics instances for the
        detected ROIs.

    Raises:
        ValueError: If no ROIs are detected after the sparse detection step.
    """
    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    console.echo(
        message=f"Binning plane {plane_index} {channel_label} frames in chunks of length {bin_size}...",
        level=LogLevel.INFO,
    )
    timer.reset()

    # Opens the registered binary file and bins frames for detection.
    with BinaryFile(
        file_path=binary_path,
        height=frame_height,
        width=frame_width,
        frame_number=frame_count,
    ) as binary_file:
        binned_frames = binary_file.bin_movie(
            bin_size=bin_size,
            y_range=(valid_y_range[0], valid_y_range[1]),
            x_range=(valid_x_range[0], valid_x_range[1]),
            bad_frames=bad_frames,
        )

    message = (
        f"Plane {plane_index} {channel_label} frames: binned. Resultant dimensions: {binned_frames.shape[0]} frames, "
        f"{binned_frames.shape[1]} height, {binned_frames.shape[2]} width. Time taken: {timer.elapsed} seconds."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)

    # Stores the mean image before detect() modifies binned_frames in-place.
    mean_image = binned_frames.mean(axis=0)

    # Applies optional PCA denoising to improve signal-to-noise ratio.
    if detection_config.denoise:
        pca_denoise(
            frames=binned_frames,
            block_size=(nonrigid_block_size[0] // 2, nonrigid_block_size[1] // 2),
            component_fraction=0.5,
            parallel_workers=parallel_workers,
        )

    # Runs the sparse iterative ROI detection algorithm. Limits BLAS and OpenMP thread count to match the requested
    # parallel worker budget, since the detection loop performs many matrix operations (matmul, norm, outer) that
    # otherwise spawn threads for all CPU cores via the underlying BLAS library.
    console.echo(
        message=f"Discovering ROIs for plane {plane_index} {channel_label}...",
        level=LogLevel.INFO,
    )

    with threadpool_limits(limits=parallel_workers if parallel_workers > 0 else None):
        maximum_projection, correlation_map, spatial_scale_pixels, roi_statistics = detect(
            frames=binned_frames,
            temporal_highpass_window=detection_config.temporal_highpass_window,
            spatial_highpass_window=detection_config.spatial_highpass_window,
            threshold_scaling=detection_config.threshold_scaling,
            maximum_iterations=_ITERATION_MULTIPLIER * detection_config.maximum_iterations,
            plane_index=plane_index,
        )

    message = (
        f"Plane {plane_index} {channel_label} ROIs: discovered. Detected ROIs: {len(roi_statistics)}. "
        f"Time taken: {timer.elapsed} seconds."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)

    # The spatial scale in pixels doubles as the cell diameter for ROI statistics and classification.
    cell_diameter = spatial_scale_pixels

    # Computes the enhanced mean image using the cell diameter for spatial filtering scale.
    enhanced_mean_image = _create_enhanced_mean_image(
        mean_image=mean_image,
        cell_diameter=cell_diameter,
        valid_y_range=valid_y_range,
        valid_x_range=valid_x_range,
        frame_height=frame_height,
        frame_width=frame_width,
    )

    # Embeds cropped detection images into full-frame arrays. The mean_image, maximum_projection, and correlation_map
    # are computed from cropped binned frames and must be placed into the full frame coordinate space.
    y_start, y_end = valid_y_range
    x_start, x_end = valid_x_range

    full_mean_image = np.zeros((frame_height, frame_width), dtype=np.float32)
    full_mean_image[y_start:y_end, x_start:x_end] = mean_image
    mean_image = full_mean_image

    full_maximum_projection = np.zeros((frame_height, frame_width), dtype=np.float32)
    full_maximum_projection[y_start:y_end, x_start:x_end] = maximum_projection
    maximum_projection = full_maximum_projection

    full_correlation_map = np.zeros((frame_height, frame_width), dtype=np.float32)
    full_correlation_map[y_start:y_end, x_start:x_end] = correlation_map
    correlation_map = full_correlation_map

    if len(roi_statistics) == 0:
        message = (
            f"Unable to complete ROI detection for plane {plane_index} {channel_label}. No ROIs found. "
            f"Check the binary file and consider adjusting the threshold_scaling parameter."
        )
        console.error(message=message, error=ValueError)

    # Offsets ROI pixel coordinates from the cropped frame space to full-frame space.
    y_pixel_offset = int(valid_y_range[0])
    x_pixel_offset = int(valid_x_range[0])
    for roi in roi_statistics:
        roi.mask.y_pixels += y_pixel_offset
        roi.mask.x_pixels += x_pixel_offset
        roi.mask.centroid = (roi.mask.centroid[0] + y_pixel_offset, roi.mask.centroid[1] + x_pixel_offset)

    # Applies optional preclassification filtering to remove unlikely cell candidates early.
    if detection_config.preclassification_threshold > 0:
        roi_statistics = _apply_preclassification(
            roi_statistics=roi_statistics,
            frame_height=frame_height,
            frame_width=frame_width,
            preclassification_threshold=detection_config.preclassification_threshold,
            crop_to_soma=detection_config.crop_to_soma,
            diameter=cell_diameter,
            custom_classifier_path=custom_classifier_path,
            plane_index=plane_index,
            channel_label=channel_label,
        )

    # Computes final ROI shape statistics with overlap-based filtering.
    console.echo(
        message=f"Computing ROI statistics and removing overlapping ROIs for plane {plane_index} {channel_label}...",
        level=LogLevel.INFO,
    )
    compute_roi_statistics(
        rois=roi_statistics,
        frame_height=frame_height,
        frame_width=frame_width,
        diameter=cell_diameter,
        maximum_overlap_fraction=detection_config.maximum_overlap,
        crop=detection_config.crop_to_soma,
    )

    message = (
        f"Plane {plane_index} {channel_label} overlapping ROI filtering: complete. Kept {len(roi_statistics)} ROIs."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)

    return mean_image, enhanced_mean_image, maximum_projection, correlation_map, cell_diameter, roi_statistics
