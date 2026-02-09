"""Provides tools for exporting multiplane data processed by the single-day pipeline as a unified dataset."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
from ataraxis_base_utilities import LogLevel, console

from ..dataclasses import CombinedData, DetectionData, ROIStatistics, ExtractionData

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..dataclasses import RuntimeContext


def compute_plane_offsets(plane_contexts: list[RuntimeContext]) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Computes the pixel displacement for each plane to arrange them in a combined view.

    Handles three scenarios based on the recording type. For standard multi-plane recordings without MROI data, computes
    a simple grid layout where each plane is tiled sequentially. For MROI recordings with a single z-plane per ROI, uses
    the MROI offsets directly to preserve spatial relationships between ROIs. For MROI recordings with multiple z-planes
    per ROI, applies two-level tiling: ROI positions are preserved within each tile, and tiles are offset for each
    z-plane to prevent overlap.

    Notes:
        The output of this function is used to properly arrange the data from multiple planes in the 'shared' recording
        space, re-assembling the recording from individually processed planes. This is used as part of outputting the
        suite2p-processed data as a 'combined' dataset that integrates the data from all available planes.

    Args:
        plane_contexts: A list of RuntimeContext instances, one for each plane being processed.

    Returns:
        A tuple of two elements. The first element is an array of y-displacement values, and the second element is an
        array of x-displacement values.
    """
    # Extracts data from the first context for reference.
    first_context = plane_contexts[0]

    # Determines the number of planes.
    plane_number = len(plane_contexts)

    # Initializes NumPy arrays to store the calculated displacement values for y-axis and x-axis.
    y_displacement = np.zeros(plane_number, dtype=np.int32)
    x_displacement = np.zeros(plane_number, dtype=np.int32)

    # Handles standard (non-MROI) recordings by computing a simple grid layout for all planes.
    if first_context.runtime.io.mroi_y_offset is None or first_context.runtime.io.mroi_x_offset is None:
        height = first_context.runtime.io.frame_height
        width = first_context.runtime.io.frame_width

        # Calculates the number of columns needed to arrange planes in a roughly square grid.
        column_number = int(np.ceil(np.sqrt(height * width * plane_number) / width))

        # Assigns each plane to a grid position based on its index.
        for plane_index in range(plane_number):
            x_displacement[plane_index] = (plane_index % column_number) * width
            y_displacement[plane_index] = (plane_index // column_number) * height

    # Handles MROI (Multi-ROI) recordings where each ROI has a known spatial position in the original field of view.
    else:
        # Starts with the MROI offsets, which position each ROI correctly relative to each other.
        x_displacement = np.array([ctx.runtime.io.mroi_x_offset for ctx in plane_contexts], dtype=np.int32)
        y_displacement = np.array([ctx.runtime.io.mroi_y_offset for ctx in plane_contexts], dtype=np.int32)

        # Checks if multiple virtual planes share the same (x, y) position. This happens when MROI recordings have
        # multiple z-planes per ROI: all z-planes within one ROI share the same spatial position.
        unique_positions = np.unique(np.vstack((y_displacement, x_displacement)), axis=1)
        roi_number = unique_positions.shape[1]

        # If fewer unique positions than virtual planes exist, we have multiple z-planes per ROI. In this case, we need
        # two-level tiling: preserve ROI positions within each tile, but offset entire tiles for each z-plane.
        if roi_number < plane_number:
            # Computes the number of z-planes (total virtual planes divided by unique ROI positions).
            plane_number //= roi_number

            heights_array = np.array([ctx.runtime.io.frame_height for ctx in plane_contexts])
            widths_array = np.array([ctx.runtime.io.frame_width for ctx in plane_contexts])

            # Calculates the tile size as the bounding box that contains all ROIs at their MROI positions.
            maximum_height = (y_displacement + heights_array).max()
            maximum_width = (x_displacement + widths_array).max()

            # Calculates the number of columns needed to arrange z-plane tiles in a roughly square grid.
            column_number = int(np.ceil(np.sqrt(maximum_height * maximum_width * plane_number) / maximum_width))

            # Adds tile offsets to the base MROI positions. Each z-plane gets its own tile, and within each tile the
            # ROIs maintain their relative MROI positions.
            for plane_index in range(plane_number):
                for roi_index in range(roi_number):
                    roi_plane_index = plane_index * roi_number + roi_index
                    x_displacement[roi_plane_index] += (plane_index % column_number) * maximum_width
                    y_displacement[roi_plane_index] += (plane_index // column_number) * maximum_height

    # Returns the lists of the y-axis and x-axis displacement values.
    return y_displacement, x_displacement


# noinspection PyUnboundLocalVariable
def combine_planes(plane_contexts: list[RuntimeContext]) -> CombinedData:
    """Combines processed data from multiple planes into a unified dataset.

    This function combines multi-plane and multi-ROI recording data into a unified dataset, reassembling the original
    recording from individually processed planes. The combined data is returned as a CombinedData instance containing
    detection images and extraction data for both channels.

    Args:
        plane_contexts: A list of RuntimeContext instances, one for each plane being combined.

    Returns:
        A CombinedData instance containing the combined detection and extraction data.

    Raises:
        ValueError: If no valid planes with ROI statistics are found.
    """
    # Extracts plane directories from the RuntimeContext instances.
    plane_directories = [ctx.runtime.io.output_directory for ctx in plane_contexts]

    # Computes the y-axis and x-axis displacement for each plane. These displacement values are used to arrange
    # individual planes back into the original recording movie.
    y_offsets, x_offsets = compute_plane_offsets(plane_contexts)

    # Queries the height and width for each plane.
    heights = np.array([ctx.runtime.io.frame_height for ctx in plane_contexts], dtype=np.uint16)
    widths = np.array([ctx.runtime.io.frame_width for ctx in plane_contexts], dtype=np.uint16)

    # Calculates the overall height and width of the entire recording plane after accounting for plane displacement.
    combined_height = int(np.amax(y_offsets + heights))
    combined_width = int(np.amax(x_offsets + widths))

    # Determines channel configuration. The channel_2_data.bin binary only contains independently detectable functional
    # data when both hardware channels are functional. When only the second hardware channel is functional, the import
    # layer swaps it into channel_1_data.bin, leaving channel_2_data.bin with non-functional data.
    has_two_channels = plane_contexts[0].configuration.main.two_channels
    main_config = plane_contexts[0].configuration.main
    second_channel_functional = main_config.first_channel_functional and main_config.second_channel_functional

    # Initializes 2D NumPy arrays to store the combined images.
    combined_mean_image = np.zeros((combined_height, combined_width), dtype=np.float32)
    combined_enhanced_mean_image = np.zeros((combined_height, combined_width), dtype=np.float32)
    combined_correlation_map = np.zeros((combined_height, combined_width), dtype=np.float32)

    # Checks if maximum projection images are available in any plane.
    has_max_projection = any(ctx.runtime.detection.maximum_projection is not None for ctx in plane_contexts)
    combined_max_projection: NDArray[np.float32] | None = None
    if has_max_projection:
        combined_max_projection = np.zeros((combined_height, combined_width), dtype=np.float32)

    # Initializes channel 2 image arrays if two channels are present.
    combined_mean_image_channel_2: NDArray[np.float32] | None = None
    combined_enhanced_mean_image_channel_2: NDArray[np.float32] | None = None
    combined_correlation_map_channel_2: NDArray[np.float32] | None = None
    combined_max_projection_channel_2: NDArray[np.float32] | None = None
    if has_two_channels:
        combined_mean_image_channel_2 = np.zeros((combined_height, combined_width), dtype=np.float32)
        if second_channel_functional:
            combined_enhanced_mean_image_channel_2 = np.zeros((combined_height, combined_width), dtype=np.float32)
            combined_correlation_map_channel_2 = np.zeros((combined_height, combined_width), dtype=np.float32)
            if has_max_projection:
                combined_max_projection_channel_2 = np.zeros((combined_height, combined_width), dtype=np.float32)

    # Logs the combining operation.
    channel_count = 2 if has_two_channels else 1
    directory_names = [d.name for d in plane_directories if d is not None]
    console.echo(
        message=f"Combining processed data for {channel_count} channel(s) from {directory_names}...",
        level=LogLevel.INFO,
    )

    # Finds the maximum number of frames across all planes.
    max_frame_count = max(ctx.runtime.io.frame_count for ctx in plane_contexts)

    # Initializes lists to accumulate combined data across planes.
    combined_roi_stats: list[ROIStatistics] = []
    combined_roi_stats_channel_2: list[ROIStatistics] = []
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

    # Loops over all available planes to process each plane's data.
    for plane_index, context in enumerate(plane_contexts):
        # Skips planes without ROI statistics (no detected cells).
        if context.runtime.extraction.roi_statistics is None:
            continue

        # Calculates the pixel ranges for placing this plane's data in the combined view.
        y_start = y_offsets[plane_index]
        y_end = y_offsets[plane_index] + heights[plane_index]
        x_start = x_offsets[plane_index]
        x_end = x_offsets[plane_index] + widths[plane_index]
        y_range = np.arange(y_start, y_end, dtype=np.int32)
        x_range = np.arange(x_start, x_end, dtype=np.int32)

        # Updates combined images with this plane's data.
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

        # Updates correlation map using valid pixel range.
        valid_y_start, valid_y_end = context.runtime.registration.valid_y_range
        valid_x_start, valid_x_end = context.runtime.registration.valid_x_range
        corr_y_range = np.arange(
            y_offsets[plane_index] + valid_y_start, y_offsets[plane_index] + valid_y_end, dtype=np.int32,
        )
        corr_x_range = np.arange(
            x_offsets[plane_index] + valid_x_start, x_offsets[plane_index] + valid_x_end, dtype=np.int32,
        )
        if context.runtime.detection.correlation_map is not None:
            combined_correlation_map[np.ix_(corr_y_range, corr_x_range)] = context.runtime.detection.correlation_map
        if (
            second_channel_functional
            and combined_correlation_map_channel_2 is not None
            and context.runtime.detection.correlation_map_channel_2 is not None
        ):
            combined_correlation_map_channel_2[np.ix_(corr_y_range, corr_x_range)] = (
                context.runtime.detection.correlation_map_channel_2
            )

        # Updates maximum projection if available.
        if (
            has_max_projection
            and combined_max_projection is not None
            and context.runtime.detection.maximum_projection is not None
        ):
            combined_max_projection[np.ix_(corr_y_range, corr_x_range)] = context.runtime.detection.maximum_projection
        if (
            second_channel_functional
            and combined_max_projection_channel_2 is not None
            and context.runtime.detection.maximum_projection_channel_2 is not None
        ):
            combined_max_projection_channel_2[np.ix_(corr_y_range, corr_x_range)] = (
                context.runtime.detection.maximum_projection_channel_2
            )

        # Creates deep copies of ROI statistics to avoid modifying the original and updates coordinates.
        for roi in context.runtime.extraction.roi_statistics:
            roi_copy = copy.deepcopy(roi)
            roi_copy.x_pixels = roi_copy.x_pixels + x_offsets[plane_index]
            roi_copy.y_pixels = roi_copy.y_pixels + y_offsets[plane_index]
            roi_copy.centroid[0] += y_offsets[plane_index]
            roi_copy.centroid[1] += x_offsets[plane_index]
            roi_copy.plane_index = plane_index
            combined_roi_stats.append(roi_copy)

        # Processes channel 2 ROI statistics if second channel is functional.
        if second_channel_functional and context.runtime.extraction.roi_statistics_channel_2 is not None:
            for roi in context.runtime.extraction.roi_statistics_channel_2:
                roi_copy = copy.deepcopy(roi)
                roi_copy.x_pixels = roi_copy.x_pixels + x_offsets[plane_index]
                roi_copy.y_pixels = roi_copy.y_pixels + y_offsets[plane_index]
                roi_copy.centroid[0] += y_offsets[plane_index]
                roi_copy.centroid[1] += x_offsets[plane_index]
                roi_copy.plane_index = plane_index
                combined_roi_stats_channel_2.append(roi_copy)

        # Extracts fluorescence and classification data from the RuntimeContext.
        plane_cell_fluorescence = context.runtime.extraction.cell_fluorescence
        plane_neuropil_fluorescence = context.runtime.extraction.neuropil_fluorescence
        plane_subtracted_fluorescence = context.runtime.extraction.subtracted_fluorescence
        plane_spikes = context.runtime.extraction.spikes
        plane_cell_classification = context.runtime.extraction.cell_classification

        # Skips fluorescence processing if data is not available.
        if (
            plane_cell_fluorescence is None
            or plane_neuropil_fluorescence is None
            or plane_subtracted_fluorescence is None
            or plane_spikes is None
            or plane_cell_classification is None
        ):
            continue

        # Pads fluorescence data if this plane has fewer frames than the maximum.
        cell_count, frame_count = plane_cell_fluorescence.shape
        if frame_count < max_frame_count:
            padding = np.zeros((cell_count, max_frame_count - frame_count), dtype=np.float32)
            plane_cell_fluorescence = np.concatenate((plane_cell_fluorescence, padding), axis=1)
            plane_neuropil_fluorescence = np.concatenate((plane_neuropil_fluorescence, padding), axis=1)
            plane_subtracted_fluorescence = np.concatenate((plane_subtracted_fluorescence, padding), axis=1)
            plane_spikes = np.concatenate((plane_spikes, padding), axis=1)

        # Appends channel 1 data to combined lists.
        combined_cell_fluorescence_list.append(plane_cell_fluorescence)
        combined_neuropil_fluorescence_list.append(plane_neuropil_fluorescence)
        combined_subtracted_fluorescence_list.append(plane_subtracted_fluorescence)
        combined_spikes_list.append(plane_spikes)
        combined_cell_classification_list.append(plane_cell_classification)

        # Extracts and appends colocalization data if available.
        if context.runtime.extraction.cell_colocalization is not None:
            combined_cell_colocalization_list.append(context.runtime.extraction.cell_colocalization)

        # Extracts and appends channel 2 extraction data if second channel is functional.
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
                cell_count_channel_2, frame_count_channel_2 = plane_cell_fluorescence_channel_2.shape
                if frame_count_channel_2 < max_frame_count:
                    padding_channel_2 = np.zeros(
                        (cell_count_channel_2, max_frame_count - frame_count_channel_2), dtype=np.float32
                    )
                    plane_cell_fluorescence_channel_2 = np.concatenate(
                        (plane_cell_fluorescence_channel_2, padding_channel_2), axis=1
                    )
                    plane_neuropil_fluorescence_channel_2 = np.concatenate(
                        (plane_neuropil_fluorescence_channel_2, padding_channel_2), axis=1
                    )
                    plane_subtracted_fluorescence_channel_2 = np.concatenate(
                        (plane_subtracted_fluorescence_channel_2, padding_channel_2), axis=1
                    )
                    plane_spikes_channel_2 = np.concatenate((plane_spikes_channel_2, padding_channel_2), axis=1)

                combined_cell_fluorescence_channel_2_list.append(plane_cell_fluorescence_channel_2)
                combined_neuropil_fluorescence_channel_2_list.append(plane_neuropil_fluorescence_channel_2)
                combined_subtracted_fluorescence_channel_2_list.append(plane_subtracted_fluorescence_channel_2)
                combined_spikes_channel_2_list.append(plane_spikes_channel_2)
                combined_cell_classification_channel_2_list.append(plane_cell_classification_channel_2)

        console.echo(message=f"Appended plane {plane_index} data to combined view.", level=LogLevel.SUCCESS)

    # Raises an error if no valid planes were found.
    if not combined_roi_stats:
        message = (
            "Unable to combine plane data. No valid planes with ROI statistics were found. Ensure that at least one "
            "plane has been processed successfully before attempting to combine the data."
        )
        console.error(message=message, error=ValueError)

    # Concatenates all accumulated arrays.
    combined_cell_fluorescence = np.concatenate(combined_cell_fluorescence_list, axis=0)
    combined_neuropil_fluorescence = np.concatenate(combined_neuropil_fluorescence_list, axis=0)
    combined_subtracted_fluorescence = np.concatenate(combined_subtracted_fluorescence_list, axis=0)
    combined_spikes = np.concatenate(combined_spikes_list, axis=0)
    combined_cell_classification = np.concatenate(combined_cell_classification_list, axis=0)

    # Concatenates colocalization data if available.
    combined_cell_colocalization: NDArray[np.float32] | None = None
    if combined_cell_colocalization_list:
        combined_cell_colocalization = np.concatenate(combined_cell_colocalization_list, axis=0)

    # Concatenates channel 2 extraction data if available.
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

    # Builds the DetectionData instance with combined images.
    detection = DetectionData(
        mean_image=combined_mean_image,
        enhanced_mean_image=combined_enhanced_mean_image,
        correlation_map=combined_correlation_map,
        maximum_projection=combined_max_projection,
        mean_image_channel_2=combined_mean_image_channel_2,
        enhanced_mean_image_channel_2=combined_enhanced_mean_image_channel_2,
        correlation_map_channel_2=combined_correlation_map_channel_2,
        maximum_projection_channel_2=combined_max_projection_channel_2,
    )

    # Builds the ExtractionData instance with combined extraction data.
    extraction = ExtractionData(
        roi_statistics=combined_roi_stats if combined_roi_stats else None,
        cell_fluorescence=combined_cell_fluorescence,
        neuropil_fluorescence=combined_neuropil_fluorescence,
        subtracted_fluorescence=combined_subtracted_fluorescence,
        spikes=combined_spikes,
        cell_classification=combined_cell_classification,
        roi_statistics_channel_2=combined_roi_stats_channel_2 if combined_roi_stats_channel_2 else None,
        cell_fluorescence_channel_2=combined_cell_fluorescence_channel_2,
        neuropil_fluorescence_channel_2=combined_neuropil_fluorescence_channel_2,
        subtracted_fluorescence_channel_2=combined_subtracted_fluorescence_channel_2,
        spikes_channel_2=combined_spikes_channel_2,
        cell_classification_channel_2=combined_cell_classification_channel_2,
        cell_colocalization=combined_cell_colocalization,
    )

    # Builds and returns the CombinedData instance.
    combined_data = CombinedData(
        detection=detection,
        extraction=extraction,
        plane_count=len(plane_contexts),
        combined_height=combined_height,
        combined_width=combined_width,
    )

    console.echo(message="Combined data prepared successfully.", level=LogLevel.SUCCESS)
    return combined_data
