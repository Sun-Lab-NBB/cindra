"""Provides assets for combining multi-plane data into unified datasets."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
from ataraxis_base_utilities import LogLevel, console

from ..dataclasses import CombinedData, DetectionData, ROIStatistics, ExtractionData

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from ..dataclasses import RuntimeContext


def combine_planes(plane_contexts: list[RuntimeContext]) -> CombinedData:
    """Combines processed data from multiple planes into a unified dataset.

    The combined product carries the detection images, the extraction data of both channels, the per-plane geometry, the
    registered binary paths, tau, and the sampling rate. Planes that did not complete detection or extraction contribute
    no traces, ROI statistics, or summary images, while their frame geometry, frame count, and registered binary path
    still reach the product. The combined traces are trimmed to the frame count of the shortest contributing plane,
    which is stored as CombinedData.frame_count alongside each plane's own count in CombinedData.plane_frame_counts.

    Args:
        plane_contexts: The runtime context of every plane being combined.

    Returns:
        The combined detection and extraction data.

    Raises:
        ValueError: If no valid planes with ROI statistics are found.
        RuntimeError: If a plane's registered binary path (or channel 2 registered binary path, when the second
            channel is functional) is not set, indicating registration did not complete successfully.
    """
    plane_directories = [context.runtime.io.output_path for context in plane_contexts]

    # Computes the y-axis and x-axis displacement for each plane. These displacement values are used to arrange
    # individual planes back into the original recording movie.
    y_offsets, x_offsets = _compute_plane_offsets(plane_contexts=plane_contexts)

    heights = np.array([context.runtime.io.frame_height for context in plane_contexts], dtype=np.uint16)
    widths = np.array([context.runtime.io.frame_width for context in plane_contexts], dtype=np.uint16)

    combined_height = int(np.amax(y_offsets + heights))
    combined_width = int(np.amax(x_offsets + widths))

    # Determines channel configuration. The channel_2_data.bin binary only contains independently detectable functional
    # data when both hardware channels are functional. When only the second hardware channel is functional, the import
    # layer swaps it into channel_1_data.bin, leaving channel_2_data.bin with non-functional data.
    has_two_channels = plane_contexts[0].configuration.main.two_channels
    main_configuration = plane_contexts[0].configuration.main
    second_channel_functional = (
        main_configuration.first_channel_functional and main_configuration.second_channel_functional
    )

    combined_mean_image = np.zeros((combined_height, combined_width), dtype=np.float32)
    combined_enhanced_mean_image = np.zeros((combined_height, combined_width), dtype=np.float32)
    combined_correlation_map = np.zeros((combined_height, combined_width), dtype=np.float32)

    has_maximum_projection = any(context.runtime.detection.maximum_projection is not None for context in plane_contexts)
    combined_maximum_projection: NDArray[np.float32] | None = None
    if has_maximum_projection:
        combined_maximum_projection = np.zeros((combined_height, combined_width), dtype=np.float32)

    # Initializes the combined corrected structural mean image if any plane has one. This image is produced by
    # intensity colocalization when one channel is structural (not functional).
    has_corrected_structural = any(
        context.runtime.extraction.corrected_structural_mean_image is not None for context in plane_contexts
    )
    combined_corrected_structural_mean_image: NDArray[np.float32] | None = None
    if has_corrected_structural:
        combined_corrected_structural_mean_image = np.zeros((combined_height, combined_width), dtype=np.float32)

    combined_mean_image_channel_2: NDArray[np.float32] | None = None
    combined_enhanced_mean_image_channel_2: NDArray[np.float32] | None = None
    combined_correlation_map_channel_2: NDArray[np.float32] | None = None
    combined_maximum_projection_channel_2: NDArray[np.float32] | None = None
    if has_two_channels:
        combined_mean_image_channel_2 = np.zeros((combined_height, combined_width), dtype=np.float32)
        if second_channel_functional:
            combined_enhanced_mean_image_channel_2 = np.zeros((combined_height, combined_width), dtype=np.float32)
            combined_correlation_map_channel_2 = np.zeros((combined_height, combined_width), dtype=np.float32)
            if has_maximum_projection:
                combined_maximum_projection_channel_2 = np.zeros((combined_height, combined_width), dtype=np.float32)

    channel_count = 2 if has_two_channels else 1
    directory_names = [directory.name for directory in plane_directories if directory is not None]
    console.echo(
        message=f"Combining processed data for {channel_count} channel(s) from {directory_names}...",
        level=LogLevel.INFO,
    )

    # Resolves the frame count of the combined product. Planes that did not complete extraction contribute nothing, and
    # the product is trimmed to the shortest plane that did, which keeps every combined column backed by real data on
    # every plane rather than padding the shorter planes with fabricated zeros.
    contributing_frame_counts = [
        frame_count
        for frame_count in (_resolve_plane_frame_count(context=context) for context in plane_contexts)
        if frame_count is not None
    ]
    combined_frame_count = min(contributing_frame_counts) if contributing_frame_counts else 0
    if contributing_frame_counts and max(contributing_frame_counts) > combined_frame_count:
        console.echo(
            message=(
                f"Trimming the combined traces to {combined_frame_count} frames. The recording's planes hold between "
                f"{combined_frame_count} and {max(contributing_frame_counts)} frames, so every combined column past "
                f"that count would carry the traces of some planes alone."
            ),
            level=LogLevel.WARNING,
        )

    combined_roi_statistics: list[ROIStatistics] = []
    combined_roi_statistics_channel_2: list[ROIStatistics] = []
    combined_cell_fluorescence_list: list[NDArray[np.float32]] = []
    combined_neuropil_fluorescence_list: list[NDArray[np.float32]] = []
    combined_subtracted_fluorescence_list: list[NDArray[np.float32]] = []
    combined_spikes_list: list[NDArray[np.float32]] = []
    combined_cell_classification_list: list[NDArray[np.float32]] = []
    combined_cell_colocalization_list: list[NDArray[np.float32]] = []
    combined_cell_fluorescence_channel_2_list: list[NDArray[np.float32]] = []
    combined_neuropil_fluorescence_channel_2_list: list[NDArray[np.float32]] = []
    combined_subtracted_fluorescence_channel_2_list: list[NDArray[np.float32]] = []
    combined_spikes_channel_2_list: list[NDArray[np.float32]] = []
    combined_cell_classification_channel_2_list: list[NDArray[np.float32]] = []

    for plane_index, context in enumerate(plane_contexts):
        if context.runtime.extraction.roi_statistics is None:
            continue

        # Skips planes without fluorescence traces (extraction not completed). This check must precede ROI statistics
        # collection to avoid adding ROI entries that lack corresponding fluorescence data.
        if (
            context.runtime.extraction.cell_fluorescence is None
            or context.runtime.extraction.neuropil_fluorescence is None
            or context.runtime.extraction.subtracted_fluorescence is None
            or context.runtime.extraction.spikes is None
            or context.runtime.extraction.cell_classification is None
        ):
            continue

        y_start = y_offsets[plane_index]
        y_end = y_offsets[plane_index] + heights[plane_index]
        x_start = x_offsets[plane_index]
        x_end = x_offsets[plane_index] + widths[plane_index]
        y_range = np.arange(y_start, y_end, dtype=np.int32)
        x_range = np.arange(x_start, x_end, dtype=np.int32)

        if context.runtime.detection.mean_image is not None:
            combined_mean_image[np.ix_(y_range, x_range)] = context.runtime.detection.mean_image
        if context.runtime.detection.enhanced_mean_image is not None:
            combined_enhanced_mean_image[np.ix_(y_range, x_range)] = context.runtime.detection.enhanced_mean_image
        if (
            has_two_channels
            and combined_mean_image_channel_2 is not None
            and context.runtime.detection.mean_image_channel_2 is not None
        ):
            combined_mean_image_channel_2[np.ix_(y_range, x_range)] = context.runtime.detection.mean_image_channel_2
        if (
            second_channel_functional
            and combined_enhanced_mean_image_channel_2 is not None
            and context.runtime.detection.enhanced_mean_image_channel_2 is not None
        ):
            combined_enhanced_mean_image_channel_2[np.ix_(y_range, x_range)] = (
                context.runtime.detection.enhanced_mean_image_channel_2
            )

        # Updates correlation map. The detection pipeline embeds cropped correlation maps into full-frame arrays, so
        # these use the full plane range (not the valid pixel range).
        if context.runtime.detection.correlation_map is not None:
            combined_correlation_map[np.ix_(y_range, x_range)] = context.runtime.detection.correlation_map
        if (
            second_channel_functional
            and combined_correlation_map_channel_2 is not None
            and context.runtime.detection.correlation_map_channel_2 is not None
        ):
            combined_correlation_map_channel_2[np.ix_(y_range, x_range)] = (
                context.runtime.detection.correlation_map_channel_2
            )

        # Updates maximum projection if available. Like correlation maps, these are full-frame arrays.
        if (
            has_maximum_projection
            and combined_maximum_projection is not None
            and context.runtime.detection.maximum_projection is not None
        ):
            combined_maximum_projection[np.ix_(y_range, x_range)] = context.runtime.detection.maximum_projection
        if (
            second_channel_functional
            and combined_maximum_projection_channel_2 is not None
            and context.runtime.detection.maximum_projection_channel_2 is not None
        ):
            combined_maximum_projection_channel_2[np.ix_(y_range, x_range)] = (
                context.runtime.detection.maximum_projection_channel_2
            )

        if (
            has_corrected_structural
            and combined_corrected_structural_mean_image is not None
            and context.runtime.extraction.corrected_structural_mean_image is not None
        ):
            combined_corrected_structural_mean_image[np.ix_(y_range, x_range)] = (
                context.runtime.extraction.corrected_structural_mean_image
            )

        # Creates deep copies of ROI statistics to avoid modifying the original and updates coordinates.
        for roi in context.runtime.extraction.roi_statistics:
            roi_copy = copy.deepcopy(roi)
            roi_copy.mask.x_pixels = roi_copy.mask.x_pixels + x_offsets[plane_index]
            roi_copy.mask.y_pixels = roi_copy.mask.y_pixels + y_offsets[plane_index]
            roi_copy.mask.centroid = (
                roi_copy.mask.centroid[0] + int(y_offsets[plane_index]),
                roi_copy.mask.centroid[1] + int(x_offsets[plane_index]),
            )
            roi_copy.plane_index = plane_index
            roi_copy.mask.frame_width = combined_width
            combined_roi_statistics.append(roi_copy)

        if second_channel_functional and context.runtime.extraction.roi_statistics_channel_2 is not None:
            for roi in context.runtime.extraction.roi_statistics_channel_2:
                roi_copy = copy.deepcopy(roi)
                roi_copy.mask.x_pixels = roi_copy.mask.x_pixels + x_offsets[plane_index]
                roi_copy.mask.y_pixels = roi_copy.mask.y_pixels + y_offsets[plane_index]
                roi_copy.mask.centroid = (
                    roi_copy.mask.centroid[0] + int(y_offsets[plane_index]),
                    roi_copy.mask.centroid[1] + int(x_offsets[plane_index]),
                )
                roi_copy.plane_index = plane_index
                roi_copy.mask.frame_width = combined_width
                combined_roi_statistics_channel_2.append(roi_copy)

        plane_cell_fluorescence = context.runtime.extraction.cell_fluorescence
        plane_neuropil_fluorescence = context.runtime.extraction.neuropil_fluorescence
        plane_subtracted_fluorescence = context.runtime.extraction.subtracted_fluorescence
        plane_spikes = context.runtime.extraction.spikes
        plane_cell_classification = context.runtime.extraction.cell_classification

        # Trims fluorescence data to the combined frame count, so that every plane contributes the same frames.
        plane_cell_fluorescence = plane_cell_fluorescence[:, :combined_frame_count]
        plane_neuropil_fluorescence = plane_neuropil_fluorescence[:, :combined_frame_count]
        plane_subtracted_fluorescence = plane_subtracted_fluorescence[:, :combined_frame_count]
        plane_spikes = plane_spikes[:, :combined_frame_count]

        combined_cell_fluorescence_list.append(plane_cell_fluorescence)
        combined_neuropil_fluorescence_list.append(plane_neuropil_fluorescence)
        combined_subtracted_fluorescence_list.append(plane_subtracted_fluorescence)
        combined_spikes_list.append(plane_spikes)
        combined_cell_classification_list.append(plane_cell_classification)

        if context.runtime.extraction.cell_colocalization is not None:
            combined_cell_colocalization_list.append(context.runtime.extraction.cell_colocalization)

        if second_channel_functional:
            plane_cell_fluorescence_channel_2 = context.runtime.extraction.cell_fluorescence_channel_2
            plane_neuropil_fluorescence_channel_2 = context.runtime.extraction.neuropil_fluorescence_channel_2
            plane_subtracted_fluorescence_channel_2 = context.runtime.extraction.subtracted_fluorescence_channel_2
            plane_spikes_channel_2 = context.runtime.extraction.spikes_channel_2
            plane_cell_classification_channel_2 = context.runtime.extraction.cell_classification_channel_2

            if (
                plane_cell_fluorescence_channel_2 is not None
                and plane_neuropil_fluorescence_channel_2 is not None
                and plane_subtracted_fluorescence_channel_2 is not None
                and plane_spikes_channel_2 is not None
                and plane_cell_classification_channel_2 is not None
            ):
                plane_cell_fluorescence_channel_2 = plane_cell_fluorescence_channel_2[:, :combined_frame_count]
                plane_neuropil_fluorescence_channel_2 = plane_neuropil_fluorescence_channel_2[:, :combined_frame_count]
                plane_subtracted_fluorescence_channel_2 = plane_subtracted_fluorescence_channel_2[
                    :, :combined_frame_count
                ]
                plane_spikes_channel_2 = plane_spikes_channel_2[:, :combined_frame_count]

                combined_cell_fluorescence_channel_2_list.append(plane_cell_fluorescence_channel_2)
                combined_neuropil_fluorescence_channel_2_list.append(plane_neuropil_fluorescence_channel_2)
                combined_subtracted_fluorescence_channel_2_list.append(plane_subtracted_fluorescence_channel_2)
                combined_spikes_channel_2_list.append(plane_spikes_channel_2)
                combined_cell_classification_channel_2_list.append(plane_cell_classification_channel_2)

        console.echo(message=f"Appended plane {plane_index} data to combined view.", level=LogLevel.SUCCESS)

    if not combined_roi_statistics:
        message = (
            "Unable to combine plane data. No valid planes with ROI statistics were found. Ensure that at least one "
            "plane has been processed successfully before attempting to combine the data."
        )
        console.error(message=message, error=ValueError)

    combined_cell_fluorescence = np.concatenate(combined_cell_fluorescence_list, axis=0)
    combined_neuropil_fluorescence = np.concatenate(combined_neuropil_fluorescence_list, axis=0)
    combined_subtracted_fluorescence = np.concatenate(combined_subtracted_fluorescence_list, axis=0)
    combined_spikes = np.concatenate(combined_spikes_list, axis=0)
    combined_cell_classification = np.concatenate(combined_cell_classification_list, axis=0)

    combined_cell_colocalization: NDArray[np.float32] | None = None
    if combined_cell_colocalization_list:
        combined_cell_colocalization = np.concatenate(combined_cell_colocalization_list, axis=0)

    combined_cell_fluorescence_channel_2: NDArray[np.float32] | None = None
    combined_neuropil_fluorescence_channel_2: NDArray[np.float32] | None = None
    combined_subtracted_fluorescence_channel_2: NDArray[np.float32] | None = None
    combined_spikes_channel_2: NDArray[np.float32] | None = None
    combined_cell_classification_channel_2: NDArray[np.float32] | None = None
    if combined_cell_fluorescence_channel_2_list:
        combined_cell_fluorescence_channel_2 = np.concatenate(combined_cell_fluorescence_channel_2_list, axis=0)
        combined_neuropil_fluorescence_channel_2 = np.concatenate(combined_neuropil_fluorescence_channel_2_list, axis=0)
        combined_subtracted_fluorescence_channel_2 = np.concatenate(
            combined_subtracted_fluorescence_channel_2_list, axis=0
        )
        combined_spikes_channel_2 = np.concatenate(combined_spikes_channel_2_list, axis=0)
        combined_cell_classification_channel_2 = np.concatenate(combined_cell_classification_channel_2_list, axis=0)

    detection = DetectionData(
        mean_image=combined_mean_image,
        enhanced_mean_image=combined_enhanced_mean_image,
        correlation_map=combined_correlation_map,
        maximum_projection=combined_maximum_projection,
        mean_image_channel_2=combined_mean_image_channel_2,
        enhanced_mean_image_channel_2=combined_enhanced_mean_image_channel_2,
        correlation_map_channel_2=combined_correlation_map_channel_2,
        maximum_projection_channel_2=combined_maximum_projection_channel_2,
    )

    extraction = ExtractionData(
        roi_statistics=combined_roi_statistics or None,
        cell_fluorescence=combined_cell_fluorescence,
        neuropil_fluorescence=combined_neuropil_fluorescence,
        subtracted_fluorescence=combined_subtracted_fluorescence,
        spikes=combined_spikes,
        cell_classification=combined_cell_classification,
        roi_statistics_channel_2=combined_roi_statistics_channel_2 or None,
        cell_fluorescence_channel_2=combined_cell_fluorescence_channel_2,
        neuropil_fluorescence_channel_2=combined_neuropil_fluorescence_channel_2,
        subtracted_fluorescence_channel_2=combined_subtracted_fluorescence_channel_2,
        spikes_channel_2=combined_spikes_channel_2,
        cell_classification_channel_2=combined_cell_classification_channel_2,
        cell_colocalization=combined_cell_colocalization,
        corrected_structural_mean_image=combined_corrected_structural_mean_image,
    )

    # Gathers registered binary paths for each plane so the multi-recording extraction pipeline can construct
    # BinaryFileCombined without reloading single-recording contexts. These paths are guaranteed to be set after
    # registration completes, so None values indicate a corrupted or incomplete pipeline run.
    channel_1_paths: list[Path] = []
    for context in plane_contexts:
        path = context.runtime.io.registered_binary_path
        if path is None:
            message = (
                f"Unable to combine plane data. The registered binary path is not set for plane "
                f"{context.runtime.io.plane_index}. Ensure registration completed successfully."
            )
            console.error(message=message, error=RuntimeError)
        channel_1_paths.append(path)
    registered_binary_paths = tuple(channel_1_paths)

    registered_binary_paths_channel_2: tuple[Path, ...] | None = None
    if second_channel_functional:
        channel_2_paths: list[Path] = []
        for context in plane_contexts:
            path_channel_2 = context.runtime.io.registered_binary_path_channel_2
            if path_channel_2 is None:
                message = (
                    f"Unable to combine plane data. The registered binary path for channel 2 is not set for "
                    f"plane {context.runtime.io.plane_index}. Ensure registration completed successfully."
                )
                console.error(message=message, error=RuntimeError)
            channel_2_paths.append(path_channel_2)
        registered_binary_paths_channel_2 = tuple(channel_2_paths)

    # Builds and returns the CombinedData instance. Caches per-plane geometry, binary paths, tau, and sampling_rate
    # from the single-recording contexts so that the multi-recording extraction pipeline can access them directly.
    # Records the per-plane frame counts alongside the combined count, so that a consumer can tell whether the combined
    # traces were trimmed instead of having to infer the frame count from an array shape.
    plane_frame_counts = np.array([context.runtime.io.frame_count for context in plane_contexts], dtype=np.uint32)

    combined_data = CombinedData(
        detection=detection,
        extraction=extraction,
        plane_count=len(plane_contexts),
        frame_count=combined_frame_count,
        plane_frame_counts=plane_frame_counts,
        combined_height=combined_height,
        combined_width=combined_width,
        tau=plane_contexts[0].configuration.main.tau,
        sampling_rate=plane_contexts[0].runtime.io.sampling_rate,
        plane_heights=heights,
        plane_widths=widths,
        plane_y_offsets=y_offsets,
        plane_x_offsets=x_offsets,
        registered_binary_paths=registered_binary_paths,
        registered_binary_paths_channel_2=registered_binary_paths_channel_2,
    )

    console.echo(message="Combined data prepared successfully.", level=LogLevel.SUCCESS)
    return combined_data


def _compute_plane_offsets(
    plane_contexts: list[RuntimeContext],
) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Computes the pixel displacement for each plane to arrange them in a combined view.

    Handles three scenarios based on the recording type. For standard multi-plane recordings without MROI data, computes
    a simple grid layout where each plane is tiled sequentially. For MROI recordings with a single z-plane per ROI, uses
    the MROI offsets directly to preserve spatial relationships between ROIs. For MROI recordings with multiple z-planes
    per ROI, applies two-level tiling: ROI positions are preserved within each tile, and tiles are offset for each
    z-plane to prevent overlap.

    Notes:
        The plane contexts are expected in the order produced by the single-recording context resolver. For MROI
        recordings, that order is ROI-major, so a virtual plane's index equals its ROI index times the z-plane count
        plus its z-plane index.

    Args:
        plane_contexts: The runtime context of every plane being processed.

    Returns:
        The y-displacement values and the x-displacement values, one of each per plane.
    """
    first_context = plane_contexts[0]
    plane_number = len(plane_contexts)

    y_displacement = np.zeros(plane_number, dtype=np.int32)
    x_displacement = np.zeros(plane_number, dtype=np.int32)

    # Handles standard (non-MROI) recordings by computing a simple grid layout for all planes.
    if first_context.runtime.io.mroi_y_offset is None or first_context.runtime.io.mroi_x_offset is None:
        height = first_context.runtime.io.frame_height
        width = first_context.runtime.io.frame_width

        # Calculates the number of columns needed to arrange planes in a roughly square grid.
        column_number = int(np.ceil(np.sqrt(height * width * plane_number) / width))

        for plane_index in range(plane_number):
            x_displacement[plane_index] = (plane_index % column_number) * width
            y_displacement[plane_index] = (plane_index // column_number) * height

    # Handles MROI (Multi-ROI) recordings where each ROI has a known spatial position in the original field of view.
    else:
        # Starts with the MROI offsets, which position each ROI correctly relative to each other.
        x_displacement = np.array([context.runtime.io.mroi_x_offset for context in plane_contexts], dtype=np.int32)
        y_displacement = np.array([context.runtime.io.mroi_y_offset for context in plane_contexts], dtype=np.int32)

        # Checks if multiple virtual planes share the same (x, y) position. This happens when MROI recordings have
        # multiple z-planes per ROI: all z-planes within one ROI share the same spatial position.
        unique_positions = np.unique(np.vstack((y_displacement, x_displacement)), axis=1)
        roi_number = unique_positions.shape[1]

        # Fewer unique positions than virtual planes means the recording holds multiple z-planes per ROI, which
        # requires two-level tiling: ROI positions are preserved within each tile, and entire tiles are offset for
        # each z-plane.
        if roi_number < plane_number:
            z_plane_number = plane_number // roi_number

            heights_array = np.array([context.runtime.io.frame_height for context in plane_contexts], dtype=np.uint16)
            widths_array = np.array([context.runtime.io.frame_width for context in plane_contexts], dtype=np.uint16)

            # Calculates the tile size as the bounding box that contains all ROIs at their MROI positions.
            maximum_height = (y_displacement + heights_array).max()
            maximum_width = (x_displacement + widths_array).max()

            # Calculates the number of columns needed to arrange z-plane tiles in a roughly square grid.
            column_number = int(np.ceil(np.sqrt(maximum_height * maximum_width * z_plane_number) / maximum_width))

            # Adds tile offsets to the base MROI positions. Each z-plane gets its own tile, and within each tile the
            # ROIs maintain their relative MROI positions. The context resolver lays virtual planes out ROI-major, as
            # virtual_plane_index = roi_index * z_plane_number + z_index, so a virtual plane's z-plane index is its own
            # index modulo the z-plane count.
            for virtual_plane_index in range(plane_number):
                z_index = virtual_plane_index % z_plane_number
                x_displacement[virtual_plane_index] += (z_index % column_number) * maximum_width
                y_displacement[virtual_plane_index] += (z_index // column_number) * maximum_height

    return y_displacement, x_displacement


def _resolve_plane_frame_count(context: RuntimeContext) -> int | None:
    """Returns the number of frames the plane's fluorescence traces span, or None when the plane cannot contribute.

    Notes:
        A plane contributes to the combined product only when it holds ROI statistics and a full set of fluorescence
        traces. A plane whose processing stage did not complete holds neither, and contributes nothing.

    Args:
        context: The plane's runtime context.

    Returns:
        The frame count of the plane's cell fluorescence traces, or None when the plane did not complete extraction.
    """
    extraction = context.runtime.extraction
    if (
        extraction.roi_statistics is None
        or extraction.cell_fluorescence is None
        or extraction.neuropil_fluorescence is None
        or extraction.subtracted_fluorescence is None
        or extraction.spikes is None
        or extraction.cell_classification is None
    ):
        return None
    return int(extraction.cell_fluorescence.shape[1])
