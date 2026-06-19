"""Provides diffeomorphic across-recording registration entry point for the multi-recording cindra processing
pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import LogLevel, console

from ..detection import compute_roi_statistics
from .deformation import Deformation
from ..dataclasses import ROIMask, ROIStatistics, ReferenceImageType, MultiRecordingRuntimeContext
from .diffeomorphic import DiffeomorphicDemonsRegistration

if TYPE_CHECKING:
    from numpy.typing import NDArray


def register_recordings(contexts: list[MultiRecordingRuntimeContext]) -> None:  # pragma: no cover
    """Registers multiple recording reference images to a common visual space using diffeomorphic demons registration.

    This function computes deformation fields that align all recordings to a shared coordinate system, then applies
    those deformations to transform reference images and ROI masks. The deformation fields and transformed data
    are stored in each recording's runtime registration data.

    Notes:
        This is the entry point for multi-recording registration. It orchestrates the full registration workflow
        including deformation field computation, image transformation, and mask deformation. The function modifies
        the runtime data in each context in-place.

        When all recordings already have registration data (deformation fields and deformed ROI masks) and
        repeat_registration is False (default), the function returns early without re-running the expensive
        diffeomorphic registration. When repeat_registration is True, existing registration data is cleared before
        re-computing.

    Args:
        contexts: The list of MultiRecordingRuntimeContext instances, one per recording. All contexts must share
            the same configuration. Each context's runtime.combined_data must be loaded with single-recording
            detection results.
    """
    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    configuration = contexts[0].configuration
    registration_config = configuration.diffeomorphic_registration
    runtime_config = configuration.runtime

    # Checks if registration should be skipped (all recordings already registered and not forcing re-registration).
    all_registered = all(
        context.runtime.registration.is_registered(output_path=context.runtime.output_path) for context in contexts
    )
    if all_registered and not registration_config.repeat_registration:
        console.echo(
            message=(
                "Multi-recording registration: skipped. All recordings are already registered and "
                "re-registration is disabled."
            ),
            level=LogLevel.INFO,
        )
        return

    # Clears existing registration data if re-registering.
    if all_registered:
        console.echo(
            message="Multi-recording registration: forced. Clearing existing data and re-running registration.",
            level=LogLevel.INFO,
        )
        for context in contexts:
            context.runtime.registration.clear()

    # Memory-maps combined detection arrays needed for registration (reference images, ROI masks).
    for context in contexts:
        combined = context.runtime.combined_data
        if combined is not None and context.runtime.io.data_path is not None:
            combined.detection.memory_map_arrays(context.runtime.io.data_path)

    # Collects reference images from all recordings based on configured image type.
    image_type = registration_config.image_type
    reference_images: list[NDArray[np.float32]] = []
    for context in contexts:
        combined_data = context.runtime.combined_data
        if combined_data is None:
            message = (
                f"Unable to register recording '{context.runtime.io.recording_id}' to shared visual space. "
                f"The recording's combined_data must be loaded before registration."
            )
            console.error(message=message, error=ValueError)
        detection = combined_data.detection
        if image_type == ReferenceImageType.MEAN:
            image = detection.mean_image
        elif image_type == ReferenceImageType.ENHANCED_MEAN:
            image = detection.enhanced_mean_image
        else:
            image = detection.maximum_projection
        if image is None:
            message = (
                f"Unable to register recording '{context.runtime.io.recording_id}' to shared visual space. "
                f"The required reference image ({image_type!s}) is not available in combined_data."
            )
            console.error(message=message, error=ValueError)
        reference_images.append(image.astype(np.float32))

    # Performs groupwise diffeomorphic registration.
    registration = DiffeomorphicDemonsRegistration(
        images=reference_images,
        grid_sampling_factor=registration_config.grid_sampling_factor,
        scale_sampling=registration_config.scale_sampling,
        speed_factor=registration_config.speed_factor,
    )
    registration.register(progress=runtime_config.display_progress_bars)

    # Applies deformation fields to each recording in parallel.
    if runtime_config.parallel_workers > 1:
        with ThreadPoolExecutor(max_workers=runtime_config.parallel_workers) as executor:
            futures = {
                executor.submit(
                    _apply_forward_deformation,
                    context=context,
                    deformation=registration.get_deformation(image_index=index),
                ): index
                for index, context in enumerate(contexts)
            }

            for future in console.track(
                as_completed(futures),
                description="Transforming recording ROIs to a shared visual space",
                total=len(futures),
                unit="recording",
            ):
                future.result()
    else:
        for index, context in console.track(
            enumerate(contexts),
            description="Transforming recording ROIs to a shared visual space",
            total=len(contexts),
            unit="recording",
        ):
            _apply_forward_deformation(
                context=context,
                deformation=registration.get_deformation(image_index=index),
            )

    # Records registration timing and persists runtime data for each recording.
    registration_time = int(timer.elapsed)
    for context in contexts:
        context.runtime.timing.registration_time = registration_time
        context.save_runtime()

    # Releases registration and combined detection arrays to free memory.
    for context in contexts:
        context.runtime.registration.release_arrays()
        if context.runtime.combined_data is not None:
            context.runtime.combined_data.detection.release_arrays()

    console.echo(
        message=f"Multi-recording registration: complete. Time: {registration_time} seconds.", level=LogLevel.SUCCESS
    )


def project_templates_to_recordings(contexts: list[MultiRecordingRuntimeContext]) -> None:  # pragma: no cover
    """Projects template masks from shared visual space back to each recording's original coordinate system.

    After ROI tracking produces template masks in the shared deformed space, this function applies the inverse
    deformation to map those masks back to each recording's native coordinates. This enables fluorescence extraction
    using the original registered binary data.

    Args:
        contexts: The list of MultiRecordingRuntimeContext instances, one per recording. Each context must have
            deformation fields stored in runtime.registration from a prior call to register_recordings(), and
            template masks set in runtime.tracking from ROI tracking.
    """
    # Skips projection when the backward-transformed ROI statistics already exist and registration was not repeated.
    # Checks for roi_statistics.npz (the file this function produces) rather than tracking_template_masks.npz (which
    # is produced by the tracking phase).
    first_output = contexts[0].runtime.output_path
    repeat_registration = contexts[0].configuration.diffeomorphic_registration.repeat_registration
    if not repeat_registration and first_output is not None and (first_output / "roi_statistics.npz").exists():
        console.echo(
            message="Template projection: skipped. Projection output already exists and re-registration is disabled.",
            level=LogLevel.INFO,
        )
        return

    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()
    runtime_config = contexts[0].configuration.runtime

    # Loads registration and tracking arrays needed for backward deformation.
    for context in contexts:
        output_path = context.runtime.output_path
        if output_path is not None:
            context.runtime.registration.memory_map_arrays(output_path)
            context.runtime.tracking.load_arrays(output_path)

    if runtime_config.parallel_workers > 1:
        with ThreadPoolExecutor(max_workers=runtime_config.parallel_workers) as executor:
            futures = {
                executor.submit(_apply_backward_deformation, context=context): index
                for index, context in enumerate(contexts)
            }

            for future in console.track(
                as_completed(futures),
                description="Projecting tracked ROIs to individual recording's visual space",
                total=len(futures),
                unit="recording",
            ):
                future.result()
    else:
        for context in console.track(
            contexts,
            description="Projecting tracked ROIs to individual recording's visual space",
            total=len(contexts),
            unit="recording",
        ):
            _apply_backward_deformation(context=context)

    # Records backward transform timing and persists runtime data for each recording.
    backward_transform_time = int(timer.elapsed)
    for context in contexts:
        context.runtime.timing.backward_transform_time = backward_transform_time
        context.save_runtime()

    # Releases registration, tracking, and extraction arrays to free memory.
    for context in contexts:
        context.runtime.registration.release_arrays()
        context.runtime.tracking.release_arrays()
        context.runtime.extraction.release_arrays()

    console.echo(
        message=f"Template projection: complete. Time: {backward_transform_time} seconds.", level=LogLevel.SUCCESS
    )


def _warp_mask_pixels(
    mask: ROIMask,
    deformation: Deformation,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32], tuple[int, int]]:
    """Applies a deformation field to transform a single ROI mask's pixel coordinates.

    Extracts a local crop of the deformation field centered on the mask's bounding box, applies the deformation using
    nearest-neighbor interpolation, and returns the transformed pixel coordinates, weights, and centroid.

    Args:
        mask: The ROIMask instance to transform.
        deformation: The Deformation instance to apply.

    Returns:
        A tuple of (y_pixels, x_pixels, pixel_weights, centroid) for the transformed mask.
    """
    margin = 50
    y_min, y_max = int(mask.y_pixels.min()) - margin, int(mask.y_pixels.max()) + margin + 1
    x_min, x_max = int(mask.x_pixels.min()) - margin, int(mask.x_pixels.max()) + margin + 1
    crop_height, crop_width = y_max - y_min, x_max - x_min

    cropped_deform, adjusted_origin = deformation.crop(origin=(y_min, x_min), crop_size=(crop_height, crop_width))

    local_y = mask.y_pixels - adjusted_origin[0]
    local_x = mask.x_pixels - adjusted_origin[1]

    actual_height, actual_width = cropped_deform.field_shape
    weight_image = np.zeros((actual_height, actual_width), dtype=np.float32)

    valid_mask = (local_y >= 0) & (local_y < actual_height) & (local_x >= 0) & (local_x < actual_width)
    weight_image[local_y[valid_mask], local_x[valid_mask]] = mask.pixel_weights[valid_mask]

    warped_weights = cropped_deform.apply_deformation(data=weight_image, interpolation=0)

    new_local_y, new_local_x = np.nonzero(warped_weights)
    new_weights = warped_weights[new_local_y, new_local_x]

    new_global_y = (new_local_y + adjusted_origin[0]).astype(np.int32)
    new_global_x = (new_local_x + adjusted_origin[1]).astype(np.int32)
    new_centroid = (int(np.median(new_global_y)), int(np.median(new_global_x)))

    return new_global_y, new_global_x, new_weights.astype(np.float32), new_centroid


def _forward_deform_masks(
    masks: list[ROIMask],
    deformation: Deformation,
    frame_width: int,
) -> list[ROIMask]:
    """Applies a forward deformation to transform ROI masks to shared visual space.

    Creates lightweight ROIMask instances with transformed coordinates. No shape statistics are computed since the
    multi-recording pipeline only needs spatial data for tracking.

    Args:
        masks: The list of ROIMask instances to transform.
        deformation: The Deformation instance to apply.
        frame_width: The width of the image frame in pixels, stored in each ROIMask for raveled pixel computation.

    Returns:
        A list of new ROIMask instances with transformed coordinates.
    """
    transformed: list[ROIMask] = []
    for mask in masks:
        y_pixels, x_pixels, pixel_weights, centroid = _warp_mask_pixels(mask=mask, deformation=deformation)
        transformed.append(
            ROIMask(
                y_pixels=y_pixels,
                x_pixels=x_pixels,
                pixel_weights=pixel_weights,
                centroid=centroid,
                frame_width=frame_width,
                radius=float(np.sqrt(len(y_pixels) / np.pi)),
            )
        )
    return transformed


def _backward_deform_masks(
    masks: list[ROIMask],
    deformation: Deformation,
    frame_height: int,
    frame_width: int,
    roi_diameter: int,
) -> list[ROIStatistics]:
    """Applies an inverse deformation to project template masks back to a recording's native coordinate system.

    Creates ROIStatistics instances with full shape statistics computed via compute_roi_statistics for downstream
    extraction and GUI use.

    Args:
        masks: The list of ROIMask instances (template masks) to transform.
        deformation: The inverse Deformation instance to apply.
        frame_height: The height of the image frame in pixels, needed for statistics computation.
        frame_width: The width of the image frame in pixels, needed for statistics computation.
        roi_diameter: The estimated ROI diameter in pixels, used for distance normalization in statistics.

    Returns:
        A list of ROIStatistics instances with transformed coordinates and full shape statistics.
    """
    roi_statistics: list[ROIStatistics] = []
    for mask in masks:
        y_pixels, x_pixels, pixel_weights, centroid = _warp_mask_pixels(mask=mask, deformation=deformation)
        roi_statistics.append(
            ROIStatistics(
                mask=ROIMask(
                    y_pixels=y_pixels,
                    x_pixels=x_pixels,
                    pixel_weights=pixel_weights,
                    centroid=centroid,
                    frame_width=frame_width,
                    cluster_id=mask.cluster_id,
                    recording_count=mask.recording_count,
                ),
            )
        )

    compute_roi_statistics(
        rois=roi_statistics,
        frame_height=frame_height,
        frame_width=frame_width,
        diameter=roi_diameter,
        crop=False,
    )

    # Zeros footprint for tracked ROIs since they bypass multi-scale detection and have no meaningful hop size.
    for roi in roi_statistics:
        roi.footprint = 0

    return roi_statistics


def _apply_forward_deformation(  # pragma: no cover
    context: MultiRecordingRuntimeContext, deformation: Deformation
) -> None:
    """Applies a forward deformation to transform the processed recording's images and ROI masks to shared visual space.

    Stores the deformation field components and transforms all reference images and selected ROI masks to the shared
    multi-recording coordinate system. Results are stored in the recording's runtime registration data.

    Args:
        context: The MultiRecordingRuntimeContext for the recording to process.
        deformation: The Deformation instance computed by groupwise registration.
    """
    registration_data = context.runtime.registration
    combined_data = context.runtime.combined_data
    if combined_data is None:
        message = (
            f"Unable to register recording '{context.runtime.io.recording_id}' to shared visual space. The recording's "
            f"combined_data must be loaded before transforming images and ROI masks."
        )
        console.error(message=message, error=ValueError)
    detection = combined_data.detection

    # Stores deformation field components.
    registration_data.deform_field_y = deformation.get_field(dimension=0)
    registration_data.deform_field_x = deformation.get_field(dimension=1)

    # Transforms channel 1 reference images.
    if detection.mean_image is not None:
        registration_data.transformed_mean_image = deformation.apply_deformation(
            data=detection.mean_image.astype(np.float32)
        )
    if detection.enhanced_mean_image is not None:
        registration_data.transformed_enhanced_mean_image = deformation.apply_deformation(
            data=detection.enhanced_mean_image.astype(np.float32)
        )
    if detection.maximum_projection is not None:
        registration_data.transformed_maximum_projection = deformation.apply_deformation(
            data=detection.maximum_projection.astype(np.float32)
        )

    # Transforms channel 2 reference images if available.
    if detection.mean_image_channel_2 is not None:
        registration_data.transformed_mean_image_channel_2 = deformation.apply_deformation(
            data=detection.mean_image_channel_2.astype(np.float32)
        )
    if detection.enhanced_mean_image_channel_2 is not None:
        registration_data.transformed_enhanced_mean_image_channel_2 = deformation.apply_deformation(
            data=detection.enhanced_mean_image_channel_2.astype(np.float32)
        )
    if detection.maximum_projection_channel_2 is not None:
        registration_data.transformed_maximum_projection_channel_2 = deformation.apply_deformation(
            data=detection.maximum_projection_channel_2.astype(np.float32)
        )

    # Gets frame dimensions for deformation.
    frame_width = combined_data.combined_width

    # Loads single-recording ROI masks and slices by selected ROI indices for channel 1.
    selected_indices = tuple(index for index in context.runtime.io.selected_roi_indices if index is not None)
    single_recording_output = context.runtime.io.data_path
    if selected_indices and single_recording_output is not None:
        masks_path = single_recording_output / "roi_masks.npz"
        if masks_path.exists():
            all_masks = ROIMask.load_list(masks_path)
            selected_masks = [all_masks[index] for index in selected_indices]
            registration_data.deformed_roi_masks = _forward_deform_masks(
                masks=selected_masks,
                deformation=deformation,
                frame_width=frame_width,
            )

    # Loads single-recording ROI masks and slices by selected ROI indices for channel 2.
    selected_indices_channel_2 = tuple(
        index for index in context.runtime.io.selected_roi_indices_channel_2 if index is not None
    )
    if selected_indices_channel_2 and single_recording_output is not None:
        masks_path_channel_2 = single_recording_output / "roi_masks_channel_2.npz"
        if masks_path_channel_2.exists():
            all_masks_channel_2 = ROIMask.load_list(masks_path_channel_2)
            selected_masks_channel_2 = [all_masks_channel_2[index] for index in selected_indices_channel_2]
            registration_data.deformed_roi_masks_channel_2 = _forward_deform_masks(
                masks=selected_masks_channel_2,
                deformation=deformation,
                frame_width=frame_width,
            )


def _apply_backward_deformation(context: MultiRecordingRuntimeContext) -> None:  # pragma: no cover
    """Applies the inverse deformation to transform shared template masks back to the target recording's visual space.

    Retrieves template masks from the context's tracking data and transforms them using the inverse of the stored
    deformation field. Results are stored in the recording's runtime extraction data for both channel 1 and channel 2
    if available.

    Args:
        context: The MultiRecordingRuntimeContext for the recording to process. Must have
            tracking.template_masks set and registration.deform_field_y/x populated from a prior forward
            deformation.
    """
    registration_data = context.runtime.registration
    tracking_data = context.runtime.tracking
    combined_data = context.runtime.combined_data
    if combined_data is None:
        message = (
            f"Unable to project templates to recording '{context.runtime.io.recording_id}'. The recording's "
            f"combined_data must be loaded before transforming template masks."
        )
        console.error(message=message, error=ValueError)
    detection = combined_data.detection

    # Gets frame dimensions for statistics computation.
    frame_height = combined_data.combined_height
    frame_width = combined_data.combined_width

    # Validates deformation fields are available from prior forward deformation.
    if registration_data.deform_field_y is None or registration_data.deform_field_x is None:
        message = (
            f"Unable to project templates to recording '{context.runtime.io.recording_id}'. Deformation fields must be "
            f"computed by register_recordings() before applying backward transformation."
        )
        console.error(message=message, error=ValueError)

    # Reconstructs the deformation and computes its inverse.
    deformation = Deformation(
        field_y=registration_data.deform_field_y,
        field_x=registration_data.deform_field_x,
    )
    inverse_deformation = deformation.inverse()

    # Transforms channel 1 template masks if available. Uses the template diameter estimated during tracking
    # rather than the per-recording detection diameter, since templates only contain the most stable consensus pixels
    # and have a different effective size. Falls back to the detection diameter when loading older runtime data that
    # lacks a stored template diameter.
    if tracking_data.template_masks is not None:
        template_diameter = tracking_data.template_diameter or detection.roi_diameter
        context.runtime.extraction.roi_statistics = _backward_deform_masks(
            masks=tracking_data.template_masks,
            deformation=inverse_deformation,
            frame_height=frame_height,
            frame_width=frame_width,
            roi_diameter=template_diameter,
        )

    # Transforms channel 2 template masks if available.
    if tracking_data.template_masks_channel_2 is not None:
        template_diameter_channel_2 = (
            tracking_data.template_diameter_channel_2 or detection.roi_diameter_channel_2 or detection.roi_diameter
        )
        context.runtime.extraction.roi_statistics_channel_2 = _backward_deform_masks(
            masks=tracking_data.template_masks_channel_2,
            deformation=inverse_deformation,
            frame_height=frame_height,
            frame_width=frame_width,
            roi_diameter=template_diameter_channel_2,
        )
