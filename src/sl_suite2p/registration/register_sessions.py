"""Provides diffeomorphic across-session registration entry point for the multi-day suite2p processing pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import LogLevel, console

from ..detection import compute_roi_statistics
from .deformation import Deformation
from ..dataclasses import ROIStatistics, ReferenceImageType, MultiDayRuntimeContext
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
                f"reference image ({image_type.value}) is not available in combined_data."
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


def _deform_masks(
    masks: list[ROIStatistics],
    deformation: Deformation,
    frame_height: int,
    frame_width: int,
    cell_diameter: int,
    crop: bool = True,
) -> list[ROIStatistics]:
    """Applies the input deformation field to transform ROI mask coordinates and recomputes shape statistics.

    For each mask, extracts a local crop of the deformation field centered on the mask's centroid, applies the
    deformation using nearest-neighbor interpolation, and recomputes all shape statistics for the transformed ROI.
    This approach reduces memory overhead compared to applying the full deformation field to each mask.

    Notes:
        This function creates new ROIStatistics instances with transformed coordinates and properly recomputed
        statistics. The original masks are not modified. Shape statistics (compactness, solidity, mean_radius, etc.)
        are recomputed using compute_roi_statistics since deformation changes the ROI geometry.

    Args:
        masks: The list of ROIStatistics instances to transform.
        deformation: The Deformation instance to apply.
        frame_height: The height of the image frame in pixels, needed for statistics computation.
        frame_width: The width of the image frame in pixels, needed for statistics computation.
        cell_diameter: The estimated cell diameter in pixels, used for distance normalization in statistics.
        crop: Determines whether to crop processed ROIs to the soma region before computing statistics. Should be
            disabled for template masks that are already consensus masks from cross-session tracking.

    Returns:
        A list of new ROIStatistics instances with transformed coordinates and recomputed statistics.
    """
    transformed_masks: list[ROIStatistics] = []
    for mask in masks:
        # Computes crop region from mask bounding box with margin.
        margin = 50
        y_min, y_max = int(mask.y_pixels.min()) - margin, int(mask.y_pixels.max()) + margin + 1
        x_min, x_max = int(mask.x_pixels.min()) - margin, int(mask.x_pixels.max()) + margin + 1
        crop_height, crop_width = y_max - y_min, x_max - x_min

        # Extracts the local deformation field crop.
        cropped_deform, adjusted_origin = deformation.crop(origin=(y_min, x_min), crop_size=(crop_height, crop_width))

        # Converts mask coordinates to local crop space.
        local_y = mask.y_pixels - adjusted_origin[0]
        local_x = mask.x_pixels - adjusted_origin[1]

        # Creates a local weight image for the mask.
        actual_height, actual_width = cropped_deform.field_shape
        weight_image = np.zeros((actual_height, actual_width), dtype=np.float32)

        # Clamps coordinates to valid crop bounds.
        valid_mask = (local_y >= 0) & (local_y < actual_height) & (local_x >= 0) & (local_x < actual_width)
        valid_local_y = local_y[valid_mask]
        valid_local_x = local_x[valid_mask]
        valid_weights = mask.pixel_weights[valid_mask]

        weight_image[valid_local_y, valid_local_x] = valid_weights

        # Applies the deformation using nearest-neighbor interpolation to preserve discrete pixel values.
        warped_weights = cropped_deform.apply_deformation(data=weight_image, interpolation=0)

        # Extracts transformed coordinates from non-zero pixels.
        new_local_y, new_local_x = np.nonzero(warped_weights)
        new_weights = warped_weights[new_local_y, new_local_x]

        # Converts back to global coordinates.
        new_global_y = new_local_y + adjusted_origin[0]
        new_global_x = new_local_x + adjusted_origin[1]

        # Computes new centroid from transformed coordinates.
        new_centroid = [int(np.median(new_global_y)), int(np.median(new_global_x))]

        # Computes raveled pixel indices for multi-day tracking. The raveled index is y * width + x in the deformed
        # visual space.
        raveled_pixels = (new_global_y * frame_width + new_global_x).astype(np.int32)

        # Creates the transformed ROIStatistics with coordinate data. Statistics are recomputed below.
        transformed = ROIStatistics(
            y_pixels=new_global_y.astype(np.int32),
            x_pixels=new_global_x.astype(np.int32),
            pixel_weights=new_weights.astype(np.float32),
            centroid=new_centroid,
            footprint=mask.footprint,
            raveled_pixels=raveled_pixels,
        )
        transformed_masks.append(transformed)

    # Recomputes all shape statistics for the transformed masks.
    compute_roi_statistics(
        rois=transformed_masks,
        frame_height=frame_height,
        frame_width=frame_width,
        diameter=cell_diameter,
        crop=crop,
    )

    return transformed_masks


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
    extraction = combined_data.extraction

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

    # Gets frame dimensions and cell diameter for statistics computation.
    frame_height = combined_data.combined_height
    frame_width = combined_data.combined_width
    cell_diameter = detection.cell_diameter

    # Transforms channel 1 selected cell masks.
    selected_indices = context.runtime.io.selected_cell_indices
    if selected_indices and extraction.roi_statistics is not None:
        selected_masks = [extraction.roi_statistics[i] for i in selected_indices]
        registration_data.deformed_cell_masks = _deform_masks(
            masks=selected_masks,
            deformation=deformation,
            frame_height=frame_height,
            frame_width=frame_width,
            cell_diameter=cell_diameter,
        )

    # Transforms channel 2 selected cell masks if available.
    selected_indices_channel_2 = context.runtime.io.selected_cell_indices_channel_2
    if selected_indices_channel_2 and extraction.roi_statistics_channel_2 is not None:
        selected_masks_channel_2 = [extraction.roi_statistics_channel_2[i] for i in selected_indices_channel_2]
        cell_diameter_channel_2 = detection.cell_diameter_channel_2 or cell_diameter
        registration_data.deformed_cell_masks_channel_2 = _deform_masks(
            masks=selected_masks_channel_2,
            deformation=deformation,
            frame_height=frame_height,
            frame_width=frame_width,
            cell_diameter=cell_diameter_channel_2,
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
        context.runtime.extraction.roi_statistics = _deform_masks(
            masks=tracking_data.template_masks,
            deformation=inverse_deformation,
            frame_height=frame_height,
            frame_width=frame_width,
            cell_diameter=template_diameter,
            crop=False,
        )

    # Transforms channel 2 template masks if available.
    if tracking_data.template_masks_channel_2 is not None:
        template_diameter_channel_2 = (
            tracking_data.template_diameter_channel_2 or detection.cell_diameter_channel_2 or detection.cell_diameter
        )
        context.runtime.extraction.roi_statistics_channel_2 = _deform_masks(
            masks=tracking_data.template_masks_channel_2,
            deformation=inverse_deformation,
            frame_height=frame_height,
            frame_width=frame_width,
            cell_diameter=template_diameter_channel_2,
            crop=False,
        )
