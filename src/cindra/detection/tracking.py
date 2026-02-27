"""Provides algorithms for tracking ROIs across multiple imaging sessions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ataraxis_time import PrecisionTimer, TimerPrecisions
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform
from ataraxis_base_utilities import LogLevel, console

from ..dataclasses import ROIMask
from .roi_statistics import estimate_diameter_from_rois

if TYPE_CHECKING:
    from ..dataclasses import MultiDayRuntimeContext

_DEFAULT_JACCARD_DISTANCE: float = 10000.0
"""The default Jaccard distance value used to initialize the distance matrix. This large value ensures that ROI pairs
that are not evaluated (due to centroid distance filtering) are never clustered together."""


def track_rois_across_sessions(contexts: list[MultiDayRuntimeContext]) -> None:
    """Tracks ROIs across multiple sessions using Jaccard distance-based hierarchical clustering.

    This function clusters ROI masks from multiple sessions based on spatial overlap in the shared deformed visual
    space. ROIs that consistently appear in the same location across sessions are grouped together, and a template
    mask is created for each cluster representing the consensus ROI. When dual-channel data is present, each channel
    is processed independently.

    Notes:
        This function modifies the input contexts in-place, updating each context's ``runtime.tracking.template_masks``
        (and ``template_masks_channel_2`` for dual-channel recordings) with the generated template ROIs.

    Args:
        contexts: The list of MultiDayRuntimeContext instances, one per session. Each context must have completed
            diffeomorphic registration with deformed ROI masks available in
            ``runtime.registration.deformed_cell_masks`` (and optionally ``deformed_cell_masks_channel_2``).
    """
    if not contexts:
        return

    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    # Determines which channels have data to process by checking the first context with available registration data.
    has_channel_1 = False
    has_channel_2 = False
    for context in contexts:
        if context.runtime.registration.deformed_cell_masks is not None:
            has_channel_1 = True
        if context.runtime.registration.deformed_cell_masks_channel_2 is not None:
            has_channel_2 = True
        if has_channel_1 or has_channel_2:
            break

    # Processes each channel that has available data.
    if has_channel_1:
        _track_channel_rois(contexts=contexts, channel_2=False)

    if has_channel_2:
        _track_channel_rois(contexts=contexts, channel_2=True)

    # Records tracking timing and persists runtime data for each session.
    tracking_time = int(timer.elapsed)
    for context in contexts:
        context.runtime.timing.tracking_time = tracking_time
        context.save_runtime()

    console.echo(message=f"ROI tracking: complete. Time: {tracking_time} seconds.", level=LogLevel.SUCCESS)


def _compute_overlap(rois: list[ROIMask]) -> None:
    """Computes overlapping pixels across ROIs and updates each ROI's overlap_mask field in-place.

    Args:
        rois: The list of ROIMask instances to process. Each ROI's ``overlap_mask`` field is updated in-place.
    """
    # Collects all pixel index arrays from the input ROIs.
    mask_pixel_indices = [roi.raveled_pixels for roi in rois]
    if not mask_pixel_indices:
        return

    # Computes cumulative offsets to track where each ROI's pixels start and end in the concatenated array.
    mask_sizes = np.array([len(indices) for indices in mask_pixel_indices], dtype=np.int32)
    mask_offsets = np.concatenate(([0], np.cumsum(mask_sizes)))
    all_pixel_indices = np.concatenate(mask_pixel_indices)

    # Uses np.unique to count occurrences of each pixel index across all ROIs. The inverse array maps each element
    # back to its position in the unique array, allowing counts[inverse] to give the count for each original element.
    _, inverse, counts = np.unique(all_pixel_indices, return_inverse=True, return_counts=True)
    flat_overlap = counts[inverse] > 1

    # Slices the flat overlap array back into per-ROI segments using the precomputed offsets.
    for roi_index, roi in enumerate(rois):
        roi.overlap_mask = flat_overlap[mask_offsets[roi_index] : mask_offsets[roi_index + 1]]


def _compute_condensed_index(row_index: int, column_index: int, matrix_size: int) -> int:
    """Converts square form matrix indices to condensed form indices.

    Args:
        row_index: The row index in the square distance matrix.
        column_index: The column index in the square distance matrix.
        matrix_size: The dimension of the square matrix (number of rows or columns).

    Returns:
        The index in the condensed distance matrix that corresponds to the input square form indices.

    Raises:
        ValueError: If diagonal elements are detected when converting indices to the condensed matrix form.
    """
    if row_index == column_index:
        message = "Unable to convert matrix indices to condensed form. Diagonal elements are not allowed."
        console.error(message=message, error=ValueError)
    if row_index < column_index:
        row_index, column_index = column_index, row_index
    return int(matrix_size * column_index - column_index * (column_index + 1) / 2 + row_index - 1 - column_index)


def _cluster_rois_in_bin(
    rois: list[ROIMask],
    roi_sessions: list[int],
    threshold: float,
    maximum_distance: int,
) -> list[tuple[list[ROIMask], list[int]]]:
    """Clusters ROIs within a spatial bin using Jaccard distance and hierarchical clustering.

    Args:
        rois: The ROIMask instances describing the ROIs within the spatial bin.
        roi_sessions: The index of the imaging session that contributed each ROI in the 'rois' list.
        threshold: The Jaccard distance threshold for hierarchical clustering. ROI pairs with a Jaccard distance
            below this value (indicating higher spatial overlap) are clustered together. A value of 0 means
            identical ROIs, while 1 means no overlap.
        maximum_distance: The maximum centroid distance in pixels for candidate ROI pairs. Only ROI pairs whose
            centroids are within this distance are evaluated for Jaccard similarity, reducing computation by
            filtering out spatially distant ROIs that cannot represent the same structure.

    Returns:
        A list of (cluster_rois, cluster_sessions) tuples, where each tuple contains the ROIs and their session
        indices that were clustered together as the same ROI across sessions.
    """
    roi_count = len(rois)
    if roi_count == 0:
        return []

    # Extracts centroids and computes pairwise distances to find candidate pairs within maximum_distance. This
    # preemptively excludes the pairs that are too distant to be the same object across days.
    centroids = np.array([roi.centroid for roi in rois], dtype=np.float32)
    pairwise_distances = pdist(centroids)
    within_distance = (pairwise_distances < maximum_distance).astype(np.int8)
    distance_matrix = np.triu(squareform(within_distance))
    candidate_pairs = np.column_stack(np.where(distance_matrix))

    if candidate_pairs.shape[0] == 0:
        return []

    # Filters to keep only pairs from different sessions using vectorized comparison. This excludes within-session
    # clusters.
    sessions_array = np.array(roi_sessions, dtype=np.int32)
    different_session_mask = sessions_array[candidate_pairs[:, 0]] != sessions_array[candidate_pairs[:, 1]]
    valid_pairs = candidate_pairs[different_session_mask]

    if len(valid_pairs) == 0:
        return []

    # Initializes the Jaccard distance matrix with a large default value for unevaluated pairs.
    condensed_size = int(((roi_count * roi_count) / 2) - (roi_count / 2))
    jaccard_matrix = np.full(shape=condensed_size, fill_value=_DEFAULT_JACCARD_DISTANCE, dtype=np.float32)

    # Computes Jaccard distance for each valid pair based on pixel overlap.
    for roi_1_index, roi_2_index in valid_pairs:
        roi_1_pixels = rois[roi_1_index].raveled_pixels
        roi_2_pixels = rois[roi_2_index].raveled_pixels

        intersection_size = np.intersect1d(roi_1_pixels, roi_2_pixels, assume_unique=True).shape[0]
        union_size = roi_1_pixels.shape[0] + roi_2_pixels.shape[0] - intersection_size
        jaccard_distance = 0.0 if union_size == 0 else 1 - intersection_size / union_size

        condensed_index = _compute_condensed_index(
            row_index=roi_1_index,
            column_index=roi_2_index,
            matrix_size=roi_count,
        )
        jaccard_matrix[condensed_index] = jaccard_distance

    # Performs hierarchical clustering and extracts cluster assignments.
    linkage_matrix = hierarchy.complete(jaccard_matrix)
    cluster_labels = hierarchy.fcluster(Z=linkage_matrix, t=threshold, criterion="distance")

    # Groups ROIs by their cluster label.
    clustered_rois: list[tuple[list[ROIMask], list[int]]] = []
    for cluster_id in np.unique(cluster_labels):
        member_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_rois = [rois[i] for i in member_indices]
        cluster_sessions = [roi_sessions[i] for i in member_indices]
        clustered_rois.append((cluster_rois, cluster_sessions))

    return clustered_rois


def _create_template_roi(
    cluster_rois: list[ROIMask],
    cluster_id: int,
    image_shape: tuple[int, int],
    pixel_prevalence: int,
) -> ROIMask | None:
    """Creates a template ROI from a cluster of matched ROIs across sessions.

    Args:
        cluster_rois: The ROIs from the cluster representing the same ROI across sessions.
        cluster_id: The unique identifier for this cluster.
        image_shape: The height and width of the deformed visual space as a tuple.
        pixel_prevalence: The minimum percentage of sessions a pixel must appear in for it to be included in the
            generated template mask.

    Returns:
        A new ROIMask instance representing the template, or None if the template would be empty.
    """
    cluster_pixels = np.hstack([roi.raveled_pixels for roi in cluster_rois])
    cluster_weights = np.hstack([roi.pixel_weights for roi in cluster_rois])

    # Uses np.unique with return_inverse to enable efficient weight aggregation via bincount.
    unique_pixels, inverse, counts = np.unique(cluster_pixels, return_inverse=True, return_counts=True)
    prevalence_mask = (counts / len(cluster_rois)) > (pixel_prevalence / 100)
    filtered_pixels = unique_pixels[prevalence_mask]

    if len(filtered_pixels) == 0:
        return None

    # Computes average weight per pixel using bincount for O(m) aggregation instead of O(n*m) loop.
    weight_sums = np.bincount(inverse, weights=cluster_weights)
    average_weights = (weight_sums[prevalence_mask] / counts[prevalence_mask]).astype(np.float32)

    pixel_coordinates = np.unravel_index(indices=filtered_pixels, shape=image_shape)
    y_pixels = pixel_coordinates[0].astype(np.int32)
    x_pixels = pixel_coordinates[1].astype(np.int32)

    centroid = (int(np.median(y_pixels)), int(np.median(x_pixels)))
    radius = float(np.mean([roi.radius for roi in cluster_rois]))

    return ROIMask(
        y_pixels=y_pixels,
        x_pixels=x_pixels,
        pixel_weights=average_weights,
        centroid=centroid,
        frame_width=image_shape[1],
        radius=radius,
        cluster_id=cluster_id,
        session_count=len(cluster_rois),
    )


def _collect_session_rois(
    contexts: list[MultiDayRuntimeContext],
    channel_2: bool,
) -> tuple[list[ROIMask], list[int]]:
    """Collects all unclustered ROIs from the registered sessions.

    Args:
        contexts: The list of MultiDayRuntimeContext instances, one per session.
        channel_2: Determines whether to collect channel 2 ROIs instead of channel 1.

    Returns:
        A tuple containing the list of ROIMask instances and their corresponding session indices.
    """
    all_rois: list[ROIMask] = []
    all_sessions: list[int] = []

    for session_index, context in enumerate(contexts):
        if channel_2:
            deformed_masks = context.runtime.registration.deformed_cell_masks_channel_2
        else:
            deformed_masks = context.runtime.registration.deformed_cell_masks

        if deformed_masks is None:
            continue

        for roi in deformed_masks:
            if roi.cluster_id == 0:
                all_rois.append(roi)
                all_sessions.append(session_index)

    return all_rois, all_sessions


def _build_roi_grid(
    rois: list[ROIMask],
    sessions: list[int],
    grid_size: int,
) -> dict[tuple[int, int], list[tuple[ROIMask, int]]]:
    """Builds a spatial grid index for efficient ROI lookup by location.

    Args:
        rois: The list of ROIMask instances to index.
        sessions: The session index for each ROI.
        grid_size: The size of each grid cell in pixels.

    Returns:
        A dictionary mapping grid cell coordinates to lists of (ROI, session) tuples.
    """
    roi_grid: dict[tuple[int, int], list[tuple[ROIMask, int]]] = {}
    for roi, session in zip(rois, sessions, strict=True):
        grid_y = roi.centroid[0] // grid_size
        grid_x = roi.centroid[1] // grid_size
        roi_grid.setdefault((grid_y, grid_x), []).append((roi, session))
    return roi_grid


def _collect_bin_rois(
    roi_grid: dict[tuple[int, int], list[tuple[ROIMask, int]]],
    bin_origin_y: int,
    bin_origin_x: int,
    bin_height: int,
    bin_width: int,
    overlap_margin: int,
    grid_cell_size: int,
) -> tuple[list[ROIMask], list[int]]:
    """Collects unclustered ROIs within a spatial bin including its overlap margins.

    Args:
        roi_grid: The spatial grid index mapping grid cell coordinates to lists of (ROI, session_index) tuples.
        bin_origin_y: The y-coordinate of the bin's top-left corner in pixels.
        bin_origin_x: The x-coordinate of the bin's top-left corner in pixels.
        bin_height: The height of the bin's core region in pixels (excluding overlap margins).
        bin_width: The width of the bin's core region in pixels (excluding overlap margins).
        overlap_margin: The margin in pixels added to each bin boundary to capture ROIs near edges. This ensures
            ROIs straddling bin boundaries are considered by both adjacent bins during clustering.
        grid_cell_size: The size of each grid cell in pixels, used to convert pixel coordinates to grid indices.

    Returns:
        A tuple containing the list of ROIs within the bin and their session indices.
    """
    # Computes the pixel boundaries of the extended search region (core bin + overlap margins on all sides).
    search_y_min = bin_origin_y - overlap_margin
    search_y_max = bin_origin_y + bin_height + overlap_margin
    search_x_min = bin_origin_x - overlap_margin
    search_x_max = bin_origin_x + bin_width + overlap_margin

    # Converts pixel boundaries to grid cell indices. The +1 ensures the range includes the cell containing the
    # maximum boundary coordinate.
    grid_row_start = search_y_min // grid_cell_size
    grid_row_end = search_y_max // grid_cell_size + 1
    grid_col_start = search_x_min // grid_cell_size
    grid_col_end = search_x_max // grid_cell_size + 1

    collected_rois: list[ROIMask] = []
    collected_sessions: list[int] = []

    # Iterates over all grid cells that could contain ROIs within the search region.
    for grid_row in range(grid_row_start, grid_row_end):
        for grid_col in range(grid_col_start, grid_col_end):
            grid_cell = roi_grid.get((grid_row, grid_col))
            if grid_cell is None:
                continue

            for roi, session_index in grid_cell:
                # Skips ROIs that have already been assigned to a cluster in a previous bin.
                if roi.cluster_id != 0:
                    continue

                # Performs precise boundary check using the ROI centroid. The grid cell lookup is a coarse filter,
                # but ROIs near cell edges may fall outside the actual search region.
                centroid_y, centroid_x = roi.centroid
                if search_y_min < centroid_y < search_y_max and search_x_min < centroid_x < search_x_max:
                    collected_rois.append(roi)
                    collected_sessions.append(session_index)

    return collected_rois, collected_sessions


def _filter_templates(
    template_masks: list[ROIMask],
    minimum_size: int,
) -> list[ROIMask]:
    """Filters template masks by removing those that are too small after overlap removal.

    Args:
        template_masks: The list of template ROIMask instances to filter.
        minimum_size: The minimum number of non-overlapping pixels required to keep the mask for further processing.

    Returns:
        The filtered list of template masks that meet the size requirement.
    """
    filtered_templates: list[ROIMask] = []
    for mask in template_masks:
        if mask.overlap_mask is None:
            filtered_templates.append(mask)
        else:
            non_overlapping_pixels = len(mask.y_pixels) - int(np.sum(mask.overlap_mask))
            if non_overlapping_pixels >= minimum_size:
                filtered_templates.append(mask)
    return filtered_templates


def _track_channel_rois(contexts: list[MultiDayRuntimeContext], channel_2: bool) -> None:
    """Tracks ROIs for a single channel across multiple sessions.

    This internal worker performs the core tracking algorithm for either channel 1 or channel 2 ROIs.

    Args:
        contexts: The list of MultiDayRuntimeContext instances, one per session.
        channel_2: Determines whether to track channel 2 ROIs instead of channel 1.
    """
    # Extracts tracking configuration parameters. All contexts share the same configuration, so the first is used.
    config = contexts[0].configuration.roi_tracking

    # Spatial binning parameters control how the image is partitioned for parallel-friendly processing.
    step_y = config.step_sizes[0]
    step_x = config.step_sizes[1]
    bin_size = config.bin_size

    # Clustering parameters control which ROIs are grouped together as the same cell across sessions.
    maximum_distance = config.maximum_distance
    threshold = config.threshold

    # Prevalence thresholds determine which clusters and pixels are retained in the final templates.
    mask_prevalence = config.mask_prevalence
    pixel_prevalence = config.pixel_prevalence
    minimum_size = config.minimum_size

    # Converts mask_prevalence percentage to an absolute session count threshold. Uses ceiling to ensure clusters
    # must appear in at least this many sessions (e.g., 50% of 5 sessions = 3 sessions minimum).
    minimum_sessions = int(np.ceil((mask_prevalence / 100) * len(contexts)))

    # Collects all unclustered ROIs (cluster_id == 0) from the deformed masks across all sessions.
    all_rois, all_sessions = _collect_session_rois(contexts=contexts, channel_2=channel_2)
    if not all_rois:
        return

    # Retrieves the combined image dimensions from the first context. These define the coordinate space for all
    # deformed ROI masks after diffeomorphic registration.
    combined_data = contexts[0].runtime.combined_data
    if combined_data is None:
        return
    image_height = combined_data.combined_height
    image_width = combined_data.combined_width
    image_shape = (image_height, image_width)

    # Builds a spatial grid index for O(1) lookup of ROIs by approximate location. The grid cell size is set to
    # the larger step dimension to ensure each ROI maps to exactly one cell.
    grid_size = max(step_x, step_y)
    roi_grid = _build_roi_grid(rois=all_rois, sessions=all_sessions, grid_size=grid_size)

    # Generates the set of unique grid positions that tile the image. Using a set prevents duplicate processing
    # when step sizes don't evenly divide the image dimensions.
    grid_positions = set()
    for y in range(0, image_height, step_y):
        for x in range(0, image_width, step_x):
            grid_y = y // grid_size
            grid_x = x // grid_size
            grid_positions.add((grid_y, grid_x))

    template_masks: list[ROIMask] = []
    cluster_counter = 0

    # Processes each spatial bin independently. Sorting ensures deterministic ordering across runs.
    for grid_pos in console.track(
        sorted(grid_positions),
        description=f"Tracking {'channel 2' if channel_2 else 'channel 1'} ROIs across sessions",
        unit="bins",
    ):
        # Converts grid indices back to pixel coordinates for boundary calculations.
        grid_y, grid_x = grid_pos
        y_position = grid_y * grid_size
        x_position = grid_x * grid_size

        # Collects ROIs within the current bin plus overlap margins. The margins ensure ROIs near bin edges are
        # clustered with their true neighbors, which may fall in adjacent bins.
        bin_rois, bin_sessions = _collect_bin_rois(
            roi_grid=roi_grid,
            bin_origin_y=y_position,
            bin_origin_x=x_position,
            bin_height=step_y,
            bin_width=step_x,
            overlap_margin=bin_size,
            grid_cell_size=grid_size,
        )

        if not bin_rois:
            continue

        # Clusters ROIs based on spatial overlap (Jaccard distance). Each cluster represents candidate matches
        # of the same cell observed across different sessions.
        clustered_rois = _cluster_rois_in_bin(
            rois=bin_rois,
            roi_sessions=bin_sessions,
            threshold=threshold,
            maximum_distance=maximum_distance,
        )

        # Processes each cluster to create template masks for cells that meet the prevalence threshold.
        for cluster_rois, cluster_sessions in clustered_rois:
            # Filters clusters that don't appear in enough sessions. A cell must be detected in at least
            # minimum_sessions to be considered reliably trackable across days.
            unique_sessions = len(set(cluster_sessions))
            if unique_sessions < minimum_sessions:
                continue

            # Computes the cluster centroid to assign ownership to exactly one bin. This prevents duplicate
            # template creation when the same cluster appears in overlapping regions of adjacent bins.
            centroids = np.array([roi.centroid for roi in cluster_rois], dtype=np.float32)
            cluster_center = centroids.mean(axis=0)

            # Only the bin containing the cluster center "owns" the cluster. Other bins that see this cluster
            # in their overlap margins will skip it.
            if not (
                y_position <= cluster_center[0] < y_position + step_y
                and x_position <= cluster_center[1] < x_position + step_x
            ):
                continue

            cluster_counter += 1

            # Creates a consensus template mask from all ROIs in the cluster. The template includes only pixels
            # that appear in at least pixel_prevalence percent of the cluster's ROIs.
            template = _create_template_roi(
                cluster_rois=cluster_rois,
                cluster_id=cluster_counter,
                image_shape=image_shape,
                pixel_prevalence=pixel_prevalence,
            )

            if template is not None:
                template_masks.append(template)

                # Marks all source ROIs with the cluster ID to prevent them from being re-clustered in
                # subsequent bins. This assignment persists in the original deformed_cell_masks.
                for roi in cluster_rois:
                    roi.cluster_id = cluster_counter

    # Identifies pixels shared between multiple template masks. Overlapping regions are ambiguous and may be
    # excluded from signal extraction.
    _compute_overlap(rois=template_masks)

    # Removes templates that become too small after excluding overlapping pixels. Small templates typically
    # represent partial cells or segmentation artifacts.
    filtered_templates = _filter_templates(template_masks=template_masks, minimum_size=minimum_size)

    # Estimates template diameter from pixel counts for use by _backward_deform_masks. Shape statistics are not
    # computed here since templates are lightweight ROIMask instances; full statistics are only computed after
    # backward deformation when ROIStatistics are needed for extraction and GUI.
    template_diameter = 0
    if filtered_templates:
        template_diameter = estimate_diameter_from_rois(rois=filtered_templates)

    # Stores the same template mask list and the estimated template diameter in all session contexts. All sessions
    # share identical templates since they represent consensus ROIs in the common registered coordinate space.
    for context in contexts:
        if channel_2:
            context.runtime.tracking.template_masks_channel_2 = filtered_templates
            context.runtime.tracking.template_diameter_channel_2 = template_diameter
        else:
            context.runtime.tracking.template_masks = filtered_templates
            context.runtime.tracking.template_diameter = template_diameter
