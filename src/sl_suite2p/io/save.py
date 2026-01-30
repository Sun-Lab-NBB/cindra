"""This module provides tools for exporting the multiplane data processed by a Suite2p single-day pipeline as a unified
suite2p dataset.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..configuration.single_day import RuntimeContext


def compute_plane_offsets(plane_contexts: list[RuntimeContext]) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
    """Computes the displacement for each plane in the input list of plane-specific RuntimeContext instances.

    The displacement values are calculated based on the dimensions and configuration parameters provided in each
    RuntimeContext. If mroi_x_offset and mroi_y_offset are already specified in the RuntimeContext, those values are
    used. If not, the function computes the displacement using the dimensions of each plane.

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

    # Calculates the number of planes.
    plane_number = len(plane_contexts)

    # Initializes NumPy arrays to store the calculated displacement values for y-axis and x-axis.
    y_displacement = np.zeros(plane_number, dtype=np.int32)
    x_displacement = np.zeros(plane_number, dtype=np.int32)

    # If mroi_y_offset and mroi_x_offset are not already provided, computes them based on the dimensions.
    if first_context.runtime.io.mroi_y_offset is None or first_context.runtime.io.mroi_x_offset is None:
        # Queries the height and width of the first plane.
        height = first_context.runtime.io.frame_height
        width = first_context.runtime.io.frame_width

        # Calculates the number of pixel columns needed to arrange the planes, based on their dimension.
        column_number = int(np.ceil(np.sqrt(height * width * plane_number) / width))

        # Loops over all available planes and calculates the displacement values of each plane based on the column and
        # row positions.
        for plane_index in range(plane_number):
            x_displacement[plane_index] = (plane_index % column_number) * width
            y_displacement[plane_index] = (plane_index // column_number) * height

    # Otherwise, uses mroi_y_offset and mroi_x_offset values directly.
    else:
        # Queries the values of mroi_x_offset and mroi_y_offset from each plane-specific RuntimeContext.
        x_displacement = np.array([ctx.runtime.io.mroi_x_offset for ctx in plane_contexts], dtype=np.int32)
        y_displacement = np.array([ctx.runtime.io.mroi_y_offset for ctx in plane_contexts], dtype=np.int32)

        # Identifies the unique (dy, dx) pairs and determines the number of unique regions of interests (ROIs).
        unique_positions = np.unique(np.vstack((y_displacement, x_displacement)), axis=1)
        roi_number = unique_positions.shape[1]

        # If the number of regions of interest (ROIs) is lower than the number of planes, recalculates the displacement
        # values based on the maximum dimensions.
        if roi_number < plane_number:
            # Recalculates the number of planes.
            plane_number //= roi_number

            # Queries the widths and heights for each plane.
            height = np.array([ctx.runtime.io.frame_height for ctx in plane_contexts])
            width = np.array([ctx.runtime.io.frame_width for ctx in plane_contexts])

            # Calculates the maximum height and width based on the computed displacement values and plane dimensions.
            maximum_height = (y_displacement + height).max()
            maximum_width = (x_displacement + width).max()

            # Recalculates the number of columns needed to arrange the planes.
            column_number = int(np.ceil(np.sqrt(maximum_height * maximum_width * plane_number) / maximum_width))

            # Loops over all available planes and updates the displacement values for each region of interest (ROI)
            # based on the column and row positions.
            for plane_index in range(plane_number):
                for roi_index in range(roi_number):
                    roi_plane_index = plane_index * roi_number + roi_index
                    x_displacement[roi_plane_index] += (plane_index % column_number) * maximum_width
                    y_displacement[roi_plane_index] += (plane_index // column_number) * maximum_height

    # Returns the lists of the y-axis and x-axis displacement values.
    return y_displacement, x_displacement


# noinspection PyUnboundLocalVariable
def combine_planes(plane_contexts: list[RuntimeContext], save: bool = True) -> None:
    """Combines processed data from multiple planes into a unified 'combined' directory.

    This function combines multi-plane and multi-ROI recording data into a unified dataset, reassembling the original
    recording from individually processed planes. The combined data is saved to a 'combined' subdirectory alongside the
    plane directories.

    Args:
        plane_contexts: A list of RuntimeContext instances, one for each plane being combined. All contexts must have
            their runtime.io.output_directory set to valid plane output directories.
        save: Determines whether to save the combined data to disk.
    """
    # Extracts plane directories from the RuntimeContext instances.
    plane_directories = [ctx.runtime.io.output_directory for ctx in plane_contexts]

    # Derives the save directory from the first plane's output directory (parent of plane0, plane1, etc.).
    save_directory = plane_directories[0].parent

    # Computes the y-axis and x-axis displacement for each plane. These displacement values are used to arrange
    # individual planes back into the original recording movie.
    y_offsets, x_offsets = compute_plane_offsets(plane_contexts)

    # Queries the height and width for each plane.
    heights = np.array([ctx.runtime.io.frame_height for ctx in plane_contexts], dtype=np.int32)
    widths = np.array([ctx.runtime.io.frame_width for ctx in plane_contexts], dtype=np.int32)

    # Calculates the overall height and width of the entire recording plane after accounting for plane displacement.
    combined_height = int(np.amax(y_offsets + heights))
    combined_width = int(np.amax(x_offsets + widths))

    # Determines whether two channels are present.
    has_two_channels = plane_contexts[0].config.main.two_channels

    # Initializes 2D NumPy arrays to store the combined images.
    combined_mean_image = np.zeros((combined_height, combined_width), dtype=np.float32)
    combined_enhanced_mean_image = np.zeros((combined_height, combined_width), dtype=np.float32)
    combined_correlation_map = np.zeros((combined_height, combined_width), dtype=np.float32)

    # Checks if maximum projection images are available in any plane.
    has_max_projection = any(ctx.runtime.detection.maximum_projection is not None for ctx in plane_contexts)
    if has_max_projection:
        combined_max_projection = np.zeros((combined_height, combined_width), dtype=np.float32)

    # Initializes channel 2 arrays if two channels are present.
    if has_two_channels:
        combined_mean_image_channel_2 = np.zeros((combined_height, combined_width), dtype=np.float32)

    # Logs the combining operation.
    channel_count = 2 if has_two_channels else 1
    directory_names = [d.name for d in plane_directories]
    console.echo(
        f"Combining processed data for {channel_count} channels from {directory_names}...", level=LogLevel.INFO
    )

    # Finds the maximum number of frames across all planes.
    max_frame_count = max(ctx.runtime.io.frame_count for ctx in plane_contexts)

    # Tracks whether a valid plane has been processed (used to initialize combined arrays).
    first_valid_plane = True

    # Loops over all available planes to process each plane's data.
    for plane_index, ctx in enumerate(plane_contexts):
        # Skips planes without ROI statistics (no detected cells).
        if ctx.runtime.extraction.roi_statistics is None:
            continue

        # Creates a copy of ROI statistics to avoid modifying the original.
        plane_roi_stats = ctx.runtime.extraction.roi_statistics.copy()

        # Calculates the pixel ranges for placing this plane's data in the combined view.
        y_start, y_end = y_offsets[plane_index], y_offsets[plane_index] + heights[plane_index]
        x_start, x_end = x_offsets[plane_index], x_offsets[plane_index] + widths[plane_index]
        y_range = np.arange(y_start, y_end)
        x_range = np.arange(x_start, x_end)

        # Updates combined images with this plane's data.
        if ctx.runtime.detection.mean_image is not None:
            combined_mean_image[np.ix_(y_range, x_range)] = ctx.runtime.detection.mean_image
        if ctx.runtime.detection.enhanced_mean_image is not None:
            combined_enhanced_mean_image[np.ix_(y_range, x_range)] = ctx.runtime.detection.enhanced_mean_image
        if has_two_channels and ctx.runtime.detection.mean_image_channel_2 is not None:
            combined_mean_image_channel_2[np.ix_(y_range, x_range)] = ctx.runtime.detection.mean_image_channel_2

        # Updates correlation map using valid pixel range.
        valid_y_start, valid_y_end = ctx.runtime.registration.valid_y_range
        valid_x_start, valid_x_end = ctx.runtime.registration.valid_x_range
        corr_y_range = np.arange(y_offsets[plane_index] + valid_y_start, y_offsets[plane_index] + valid_y_end)
        corr_x_range = np.arange(x_offsets[plane_index] + valid_x_start, x_offsets[plane_index] + valid_x_end)
        if ctx.runtime.detection.correlation_map is not None:
            combined_correlation_map[np.ix_(corr_y_range, corr_x_range)] = ctx.runtime.detection.correlation_map

        # Updates maximum projection if available.
        if has_max_projection and ctx.runtime.detection.maximum_projection is not None:
            combined_max_projection[np.ix_(corr_y_range, corr_x_range)] = ctx.runtime.detection.maximum_projection

        # Updates ROI statistics with combined-view coordinates and plane index.
        for roi_index in range(len(plane_roi_stats)):
            plane_roi_stats[roi_index]["x_pixels"] += x_offsets[plane_index]
            plane_roi_stats[roi_index]["y_pixels"] += y_offsets[plane_index]
            plane_roi_stats[roi_index]["centroid"][0] += y_offsets[plane_index]
            plane_roi_stats[roi_index]["centroid"][1] += x_offsets[plane_index]
            plane_roi_stats[roi_index]["plane_index"] = plane_index

        # Extracts fluorescence and classification data from the RuntimeContext.
        plane_cell_fluorescence = ctx.runtime.extraction.cell_fluorescence
        plane_neuropil_fluorescence = ctx.runtime.extraction.neuropil_fluorescence
        plane_subtracted_fluorescence = ctx.runtime.extraction.subtracted_fluorescence
        plane_spikes = ctx.runtime.extraction.spikes
        plane_cell_classification = ctx.runtime.extraction.cell_classification

        # Extracts channel 2 colocalization data if available.
        plane_cell_colocalization = ctx.runtime.extraction.cell_colocalization
        has_colocalization = plane_cell_colocalization is not None

        # Pads fluorescence data if this plane has fewer frames than the maximum.
        cell_count, frame_count = plane_cell_fluorescence.shape
        if frame_count < max_frame_count:
            padding = np.zeros((cell_count, max_frame_count - frame_count), dtype=np.float32)
            plane_cell_fluorescence = np.concatenate((plane_cell_fluorescence, padding), axis=1)
            plane_neuropil_fluorescence = np.concatenate((plane_neuropil_fluorescence, padding), axis=1)
            plane_subtracted_fluorescence = np.concatenate((plane_subtracted_fluorescence, padding), axis=1)
            plane_spikes = np.concatenate((plane_spikes, padding), axis=1)

        # Initializes or concatenates combined arrays.
        if first_valid_plane:
            combined_roi_stats = plane_roi_stats
            combined_cell_fluorescence = plane_cell_fluorescence
            combined_neuropil_fluorescence = plane_neuropil_fluorescence
            combined_subtracted_fluorescence = plane_subtracted_fluorescence
            combined_spikes = plane_spikes
            combined_cell_classification = plane_cell_classification
            combined_cell_colocalization = plane_cell_colocalization
            first_valid_plane = False
        else:
            combined_roi_stats = np.concatenate((combined_roi_stats, plane_roi_stats))
            combined_cell_fluorescence = np.concatenate((combined_cell_fluorescence, plane_cell_fluorescence))
            combined_neuropil_fluorescence = np.concatenate(
                (combined_neuropil_fluorescence, plane_neuropil_fluorescence)
            )
            combined_subtracted_fluorescence = np.concatenate(
                (combined_subtracted_fluorescence, plane_subtracted_fluorescence)
            )
            combined_spikes = np.concatenate((combined_spikes, plane_spikes))
            combined_cell_classification = np.concatenate((combined_cell_classification, plane_cell_classification))
            if has_colocalization:
                combined_cell_colocalization = np.concatenate((combined_cell_colocalization, plane_cell_colocalization))

        console.echo(f"Appended plane {plane_index} data to combined view.", level=LogLevel.SUCCESS)

    # Raises an error if no valid planes were found.
    if first_valid_plane:
        message = (
            "Unable to combine plane data. No valid planes with ROI statistics (stat.npy) were found. "
            "Ensure that at least one plane has been processed successfully before combining."
        )
        console.error(message=message, error=ValueError)

    # Prepares the combined output directory.
    combined_directory = save_directory / "combined"
    ensure_directory_exists(combined_directory)

    # Saves classification data (always saved for GUI compatibility).
    np.save(combined_directory / "iscell.npy", combined_cell_classification)
    if has_colocalization:
        np.save(combined_directory / "redcell.npy", combined_cell_colocalization)

    # Saves all combined data if requested.
    if save:
        np.save(combined_directory / "F.npy", combined_cell_fluorescence)
        np.save(combined_directory / "Fneu.npy", combined_neuropil_fluorescence)
        np.save(combined_directory / "Fsub.npy", combined_subtracted_fluorescence)
        np.save(combined_directory / "spks.npy", combined_spikes)
        np.save(combined_directory / "stat.npy", combined_roi_stats)
        np.save(combined_directory / "mean_image.npy", combined_mean_image)
        np.save(combined_directory / "enhanced_mean_image.npy", combined_enhanced_mean_image)
        np.save(combined_directory / "correlation_map.npy", combined_correlation_map)
        if has_max_projection:
            np.save(combined_directory / "maximum_projection.npy", combined_max_projection)
        if has_two_channels:
            np.save(combined_directory / "mean_image_channel_2.npy", combined_mean_image_channel_2)

    console.echo(f"Combined data saved to {combined_directory}.", level=LogLevel.SUCCESS)
