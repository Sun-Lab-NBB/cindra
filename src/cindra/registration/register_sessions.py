"""Provides diffeomorphic across-session registration entry point for the multi-day cindra processing pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import LogLevel, console

from ..detection import compute_roi_statistics
from .deformation import Deformation
from ..dataclasses import ROIMask, ROIStatistics, ReferenceImageType, MultiDayRuntimeContext
from .diffeomorphic import DiffeomorphicDemonsRegistration

if TYPE_CHECKING:
    from numpy.typing import NDArray


def register_sessions(contexts: list[MultiDayRuntimeContext]) -> None:
    """Registers multiple session reference images to a common visual space using diffeomorphic demons registration.

    This function computes deformation fields that align all sessions to a shared coordinate system, then applies
    those deformations to transform reference images and cell masks. The deformation fields and transformed data
    are stored in each session's runtime registration data.

    Notes:
        This is the entry point for multi-day registration. It orchestrates the full registration workflow including
        deformation field computation, image transformation, and mask deformation. The function modifies the runtime
        data in each context in-place.

        When all sessions already have registration data (deformation fields and deformed cell masks) and
        repeat_registration is False (default), the function returns early without re-running the expensive
        diffeomorphic registration. When repeat_registration is True, existing registration data is cleared before
        re-computing.

    Args:
        contexts: The list of MultiDayRuntimeContext instances, one per session. All contexts must share the same
            configuration. Each context's runtime.combined_data must be loaded with single-day detection results.
    """
    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    configuration = contexts[0].configuration
    registration_config = configuration.diffeomorphic_registration
    runtime_config = configuration.runtime

    # Checks if registration should be skipped (all sessions already registered and not forcing re-registration).
    all_registered = all(context.runtime.registration.is_registered() for context in contexts)
    if all_registered and not registration_config.repeat_registration:
        console.echo(
            message=(
                "Multi-day registration: skipped. All sessions are already registered and re-registration is disabled."
            ),
            level=LogLevel.INFO,
        )
        return

    # Clears existing registration data if re-registering.
    if all_registered:
        console.echo(
            message="Multi-day registration: forced. Clearing existing data and re-running registration.",
            level=LogLevel.INFO,
        )
        for context in contexts:
            context.runtime.registration.clear()

    # Collects reference images from all sessions based on configured image type.
    image_type = registration_config.image_type
    reference_images: list[NDArray[np.float32]] = []
    for context in contexts:
        combined_data = context.runtime.combined_data
        if combined_data is None:
            message = (
                f"Unable to register session '{context.runtime.io.session_id}' to shared visual space. The session's "
                f"combined_data must be loaded before registration."
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
                f"Unable to register session '{context.runtime.io.session_id}' to shared visual space. The required "
                f"reference image ({image_type!s}) is not available in combined_data."
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

    # Applies deformation fields to each session in parallel.
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
                description="Transforming session ROIs to a shared visual space",
                total=len(futures),
                unit="session",
            ):
                future.result()
    else:
        for index, context in console.track(
            enumerate(contexts),
            description="Transforming session ROIs to a shared visual space",
            total=len(contexts),
            unit="session",
        ):
            _apply_forward_deformation(
                context=context,
                deformation=registration.get_deformation(image_index=index),
            )

    # Records registration timing and persists runtime data for each session.
    registration_time = int(timer.elapsed)
    for context in contexts:
        context.runtime.timing.registration_time = registration_time
        context.save_runtime()

    console.echo(
        message=f"Multi-day registration: complete. Time: {registration_time} seconds.", level=LogLevel.SUCCESS
    )


def project_templates_to_sessions(contexts: list[MultiDayRuntimeContext]) -> None:
    """Projects template masks from shared visual space back to each session's original coordinate system.

    After cell tracking produces template masks in the shared deformed space, this function applies the inverse
    deformation to map those masks back to each session's native coordinates. This enables fluorescence extraction
    using the original registered binary data.

    Args:
        contexts: The list of MultiDayRuntimeContext instances, one per session. Each context must have deformation
            fields stored in runtime.registration from a prior call to register_sessions(), and template masks set
            in runtime.tracking from cell tracking.
    """
    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()
    runtime_config = contexts[0].configuration.runtime

    if runtime_config.parallel_workers > 1:
        with ThreadPoolExecutor(max_workers=runtime_config.parallel_workers) as executor:
            futures = {
                executor.submit(_apply_backward_deformation, context=context): index
                for index, context in enumerate(contexts)
            }

            for future in console.track(
                as_completed(futures),
                description="Projecting tracked ROIs to individual session's visual space",
                total=len(futures),
                unit="session",
            ):
                future.result()
    else:
        for context in console.track(
            contexts,
            description="Projecting tracked ROIs to individual session's visual space",
            total=len(contexts),
            unit="session",
        ):
            _apply_backward_deformation(context=context)

    # Records backward transform timing and persists runtime data for each session.
    backward_transform_time = int(timer.elapsed)
    for context in contexts:
        context.runtime.timing.backward_transform_time = backward_transform_time
        context.save_runtime()

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
    multi-day pipeline only needs spatial data for tracking.

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
    cell_diameter: int,
) -> tuple[list[ROIMask], list[ROIStatistics]]:
    """Applies an inverse deformation to project template masks back to a session's native coordinate system.

    Creates both ROIMask instances (for spatial data persistence) and ROIStatistics instances (for downstream
    extraction and GUI use) with full shape statistics computed via compute_roi_statistics.

    Args:
        masks: The list of ROIMask instances (template masks) to transform.
        deformation: The inverse Deformation instance to apply.
        frame_height: The height of the image frame in pixels, needed for statistics computation.
        frame_width: The width of the image frame in pixels, needed for statistics computation.
        cell_diameter: The estimated cell diameter in pixels, used for distance normalization in statistics.

    Returns:
        A tuple of (roi_masks, roi_statistics) where roi_masks contains the transformed spatial data and
        roi_statistics contains full shape statistics for extraction and GUI use.
    """
    roi_masks: list[ROIMask] = []
    roi_statistics: list[ROIStatistics] = []
    for mask in masks:
        y_pixels, x_pixels, pixel_weights, centroid = _warp_mask_pixels(mask=mask, deformation=deformation)
        roi_masks.append(
            ROIMask(
                y_pixels=y_pixels,
                x_pixels=x_pixels,
                pixel_weights=pixel_weights,
                centroid=centroid,
                frame_width=frame_width,
                radius=float(np.sqrt(len(y_pixels) / np.pi)),
                cluster_id=mask.cluster_id,
                session_count=mask.session_count,
            )
        )
        roi_statistics.append(
            ROIStatistics(
                y_pixels=y_pixels,
                x_pixels=x_pixels,
                pixel_weights=pixel_weights,
                centroid=centroid,
                cluster_id=mask.cluster_id,
                session_count=mask.session_count,
            )
        )

    compute_roi_statistics(
        rois=roi_statistics,
        frame_height=frame_height,
        frame_width=frame_width,
        diameter=cell_diameter,
        crop=False,
    )

    return roi_masks, roi_statistics


def _apply_forward_deformation(context: MultiDayRuntimeContext, deformation: Deformation) -> None:
    """Applies a forward deformation to transform the processed session's images and ROI masks to shared visual space.

    Stores the deformation field components and transforms all reference images and selected cell masks to the shared
    multi-day coordinate system. Results are stored in the session's runtime registration data.

    Args:
        context: The MultiDayRuntimeContext for the session to process.
        deformation: The Deformation instance computed by groupwise registration.
    """
    registration_data = context.runtime.registration
    combined_data = context.runtime.combined_data
    if combined_data is None:
        message = (
            f"Unable to register session '{context.runtime.io.session_id}' to shared visual space. The session's "
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

    # Loads single-day ROI masks and slices by selected cell indices for channel 1.
    selected_indices = context.runtime.io.selected_cell_indices
    single_day_output = context.runtime.io.data_path
    if selected_indices and single_day_output is not None:
        masks_path = single_day_output / "roi_masks.npz"
        if masks_path.exists():
            all_masks = ROIMask.load_list(masks_path)
            # Updates frame_width on loaded masks to use the combined visual space width.
            selected_masks = [all_masks[i] for i in selected_indices]
            for mask in selected_masks:
                mask.frame_width = frame_width
            registration_data.deformed_cell_masks = _forward_deform_masks(
                masks=selected_masks,
                deformation=deformation,
                frame_width=frame_width,
            )

    # Loads single-day ROI masks and slices by selected cell indices for channel 2.
    selected_indices_channel_2 = context.runtime.io.selected_cell_indices_channel_2
    if selected_indices_channel_2 and single_day_output is not None:
        masks_path_channel_2 = single_day_output / "roi_masks_channel_2.npz"
        if masks_path_channel_2.exists():
            all_masks_channel_2 = ROIMask.load_list(masks_path_channel_2)
            selected_masks_channel_2 = [all_masks_channel_2[i] for i in selected_indices_channel_2]
            for mask in selected_masks_channel_2:
                mask.frame_width = frame_width
            registration_data.deformed_cell_masks_channel_2 = _forward_deform_masks(
                masks=selected_masks_channel_2,
                deformation=deformation,
                frame_width=frame_width,
            )


def _apply_backward_deformation(context: MultiDayRuntimeContext) -> None:
    """Applies the inverse deformation to transform shared template masks back to the target session's visual space.

    Retrieves template masks from the context's tracking data and transforms them using the inverse of the stored
    deformation field. Results are stored in the session's runtime extraction data for both channel 1 and channel 2
    if available.

    Args:
        context: The MultiDayRuntimeContext for the session to process. Must have tracking.template_masks set and
            registration.deform_field_y/x populated from a prior forward deformation.
    """
    registration_data = context.runtime.registration
    tracking_data = context.runtime.tracking
    combined_data = context.runtime.combined_data
    if combined_data is None:
        message = (
            f"Unable to project templates to session '{context.runtime.io.session_id}'. The session's combined_data "
            f"must be loaded before transforming template masks."
        )
        console.error(message=message, error=ValueError)
    detection = combined_data.detection

    # Gets frame dimensions for statistics computation.
    frame_height = combined_data.combined_height
    frame_width = combined_data.combined_width

    # Validates deformation fields are available from prior forward deformation.
    if registration_data.deform_field_y is None or registration_data.deform_field_x is None:
        message = (
            f"Unable to project templates to session '{context.runtime.io.session_id}'. Deformation fields must be "
            f"computed by register_sessions() before applying backward transformation."
        )
        console.error(message=message, error=ValueError)

    # Reconstructs the deformation and computes its inverse.
    deformation = Deformation(
        field_y=registration_data.deform_field_y,
        field_x=registration_data.deform_field_x,
    )
    inverse_deformation = deformation.inverse()

    # Transforms channel 1 template masks if available. Uses the template diameter estimated during tracking
    # rather than the per-session detection diameter, since templates only contain the most stable consensus pixels
    # and have a different effective size. Falls back to the detection diameter when loading older runtime data that
    # lacks a stored template diameter.
    if tracking_data.template_masks is not None:
        template_diameter = tracking_data.template_diameter or detection.cell_diameter
        tracked_masks, tracked_stats = _backward_deform_masks(
            masks=tracking_data.template_masks,
            deformation=inverse_deformation,
            frame_height=frame_height,
            frame_width=frame_width,
            cell_diameter=template_diameter,
        )
        context.runtime.extraction.roi_statistics = tracked_stats

    # Transforms channel 2 template masks if available.
    if tracking_data.template_masks_channel_2 is not None:
        template_diameter_channel_2 = (
            tracking_data.template_diameter_channel_2 or detection.cell_diameter_channel_2 or detection.cell_diameter
        )
        tracked_masks_channel_2, tracked_stats_channel_2 = _backward_deform_masks(
            masks=tracking_data.template_masks_channel_2,
            deformation=inverse_deformation,
            frame_height=frame_height,
            frame_width=frame_width,
            cell_diameter=template_diameter_channel_2,
        )
        context.runtime.extraction.roi_statistics_channel_2 = tracked_stats_channel_2
