"""This module provides the functions used during the first step of the multi-day suite2p processing pipeline to
discover and track a set of cell ROIs across multiple sessions.
"""

import os
from typing import Any
from concurrent.futures import ProcessPoolExecutor

from pirt import DiffeomorphicDemonsRegistration
from tqdm import tqdm
import numpy as np
from ataraxis_time import PrecisionTimer
from scipy.spatial.distance import pdist, squareform
from ataraxis_base_utilities import LogLevel, console
import scipy.cluster.hierarchy

from .utils import deform_masks, add_overlap_info, create_mask_image
from .dataclasses import Session, MultiDayData


def register_sessions(ops: dict[str, Any], data: MultiDayData) -> MultiDayData:
    """Registers session reference images and ROI masks to the same visual space using DiffeomorphicDemonsRegistration.

    Args:
        ops: The dictionary that stores the multi-day registration parameters.
        data: A MultiDayData instance that stores intermediate pipeline data.

    Returns:
        The MultiDayData instance updated with the outcome of the registration process.
    """
    # Initializes the runtime timer
    timer = PrecisionTimer("s")

    # Extracts the type of reference image to use for the registration process.
    images = [session.reference_images[ops["image_type"]] for session in data.sessions]

    # Instantiates the DiffeomorphicDemonsRegistration object for the reference images to be registered to each-other.
    registration = DiffeomorphicDemonsRegistration(*images)

    # Updates registration parameters using the user-defined settings.
    registration.params.grid_sampling_factor = ops["grid_sampling_factor"]
    registration.params.scale_sampling = ops["scale_sampling"]
    registration.params.speed_factor = ops["speed_factor"]

    # Runs the registration process
    console.echo(message=f"Computing deformation fields for {ops['image_type']} session images...")
    timer.reset()
    registration.register(verbose=0)
    console.echo(message=f"Deformation fields: computed. Time taken: {timer.elapsed} seconds.", level=LogLevel.SUCCESS)
    timer.delay_noblock(delay=1, allow_sleep=False)  # Delays for one second to optimize terminal message order

    # Resolves the number of parallel workers used to apply deformations to session's data.
    parallel_workers = ops["parallel_workers"]
    if parallel_workers < 1:
        parallel_workers = os.cpu_count()

    # Applies the deformation (registration) offsets generated during registration to all reference images of each
    # session to align them in the deformed visual space. Also applies the deformations to all cell masks to register
    # all single-day cells to the same visual space.
    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        # Submits each session to be processed in parallel
        futures = [
            executor.submit(_register_session, registration=registration, deform_index=index, session=session)
            for index, session in enumerate(data.sessions)
        ]

        # Collects results with the progress bar using list comprehension
        results = [
            future.result()
            for future in tqdm(
                futures, desc="Applying deformation fields", unit="session", disable=not ops["progress_bars"]
            )
        ]

    # Updates the sessions stored inside data with the results of the above pipeline
    data.sessions = results

    timer.delay_noblock(delay=1, allow_sleep=False)  # Delays for one second to optimize terminal message order
    # Returns the updated MultiDayData instance
    return data


def _register_session(registration: DiffeomorphicDemonsRegistration, deform_index: int, session: Session) -> Session:
    """This worker function is used by the main register_sessions function to apply deformation offsets to all sessions
    in parallel.
    """
    # Extracts and saves the DeformationField object. Due to how pirt is implemented and contrary to pirt's
    # docstrings, since we do not override the default transform_mapping parameter, the mapping is BACKWARD.
    session.deform = registration.get_deform(deform_index)

    # Uses the deformation field object to transform reference images
    session.transformed_images = {}
    for field in ["mean", "enhanced", "max"]:
        session.transformed_images[field] = np.array(
            object=session.deform.apply_deformation(session.reference_images[field]), dtype=np.float32, copy=True
        )

    # Transforms single-day cell data using the deformation offsets to create deformed cell masks.
    deformed_cells = deform_masks(session.cell_masks, session.deform)

    # Adds the session number (index) to each cell dictionary and sets clustering ID to 0 (unassigned/not clustered)
    session.deformed_cell_masks = tuple([dict(item, session=deform_index, id=0) for item in deformed_cells])

    return session


def square_to_condensed(i: int, j: int, n: int) -> int:
    """Converts square form matrix indices to condensed form indices.

    Args:
        i: Index 1.
        j: Index 2.
        n: The number of entries (the length of one square-form matrix dimension).

    Returns:
        The index in the condensed distance matrix that corresponds to the input square form indices.

    Raises:
        ArithmeticError: If diagonal elements are detected when converting indices to the condensed matrix form.
    """
    if i == j:
        message = "No diagonal elements allowed in condensed matrix."
        console.error(message=message, error=ArithmeticError)
    if i < j:
        i, j = j, i
    return int(n * j - j * (j + 1) / 2 + i - 1 - j)


def generate_template_masks(ops: dict[str, Any], data: MultiDayData) -> MultiDayData:
    """Clusters cell masks across sessions using Jaccard distance matrix and creates the averaged template mask for
    each group of clustered cells.

    Notes:
        Each clustered group represents the same cell tracked across all processed sessions.

    Args:
        ops: The dictionary that stores the multi-day clustering parameters.
        data: A MultiDayData instance that stores the multi-day registration data for all sessions.

    Returns:
        The MultiDayData instance updated with the outcome of the template (multi-day-tracked) cell mask generation.
    """
    # noinspection PyTypeChecker
    image_size: tuple[int, ...] = data.sessions[0].transformed_images["mean"].shape

    # Pre-calculates constants to avoid repeated dictionary lookups during loop
    step_x = ops["step_sizes"][0]
    step_y = ops["step_sizes"][1]
    bin_size = ops["bin_size"]
    maximum_distance = ops["maximum_distance"]
    threshold = ops["threshold"]
    criterion = ops["criterion"]
    mask_prevalence = ops["mask_prevalence"]
    pixel_prevalence = ops["pixel_prevalence"]
    minimum_sessions = int(np.ceil((mask_prevalence / 100) * len(data.sessions)))

    # Pre-extracts all cells with ID=0 from all sessions
    all_cells = [cell for session in data.sessions for cell in session.deformed_cell_masks if cell["id"] == 0]

    # Organizes the cells by the spatial grid before entering the main processing loop. Specifically, bins the cells
    # based on their position in the 2-dimensional space of each session. By pre-binning the sessions, the clustering
    # algorithm works with a small subset of all cells already clustered in approximately the same 2-dimensional space.
    # This improves the efficiency of the clustering process.
    cell_grid: dict[tuple[int, int], list[dict[str, Any]]] = {}
    grid_size = max(step_x, step_y)
    for cell in all_cells:
        grid_y = cell["med"][0] // grid_size
        grid_x = cell["med"][1] // grid_size
        cell_grid.setdefault((grid_y, grid_x), []).append(cell)

    # Creates a set of grid positions to process. This streamlines the main processing loop and prepares the code
    # for further optimizations, such as pre-filtering empty grids and adding multithreaded execution.
    grid_positions = set()
    for y in range(0, image_size[0], step_y):
        for x in range(0, image_size[1], step_x):
            grid_y = y // grid_size
            grid_x = x // grid_size
            grid_positions.add((grid_y, grid_x))

    # Process grid positions with progress tracking
    template_masks: list[dict[str, Any]] = []
    counter = 0
    # Process each grid cell directly
    for grid_pos in tqdm(
        sorted(grid_positions),
        desc="Clustering cells across sessions...",
        unit="bins",
        disable=not ops["progress_bars"],
    ):
        # Converts grid position back to absolute session image coordinates
        grid_y, grid_x = grid_pos
        y_position = grid_y * grid_size
        x_position = grid_x * grid_size

        # Collects all cell masks that fall inside the processed spatial bin. To do so, first determines the range of
        # x and y image-coordinates where the cell median (center-point) should fall for the cell to be assigned to the
        # processed bin. Note, for this process, expands the cell search area to extend into neighboring grids using
        # bin_size.
        grid_y_min = (y_position - bin_size) // grid_size
        grid_y_max = (y_position + step_x + bin_size) // grid_size + 1
        grid_x_min = (x_position - bin_size) // grid_size
        grid_x_max = (x_position + step_y + bin_size) // grid_size + 1

        # Samples all cells whose masks fall within the processed spatial bin.
        cell_info: list[dict[str, Any]] = []
        for gy in range(grid_y_min, grid_y_max):
            for gx in range(grid_x_min, grid_x_max):
                if (gy, gx) in cell_grid:
                    for cell in cell_grid[(gy, gx)]:
                        if cell["id"] == 0:  # Only processes cells that haven't been clustered yet
                            y, x = cell["med"]
                            if (
                                y_position - bin_size < y < y_position + step_x + bin_size
                                and x_position - bin_size < x < x_position + step_y + bin_size
                            ):
                                cell_info.append(cell)

        # If the sampling discovers no cells to process within the bin, advances to the next bin with no further
        # processing.
        cell_number = len(cell_info)
        if cell_number == 0:
            continue

        # Extracts the centerpoints of each processed cell into a separate array
        centers = np.array([cell["med"] for cell in cell_info])

        # Calculates the distances between each pair of cells inside the spatial bin and compares each distance against
        # the distance threshold. Distances smaller than the threshold are considered as possible cell pairs.
        distances = np.triu(squareform(pdist(centers) < maximum_distance))
        is_possible_pair = np.array(np.where(distances)).T

        # If there are no possible cell pairs in the processed bin, advances to the next bin.
        if is_possible_pair.shape[0] == 0:
            continue

        # Pre-calculates possible across-session cell pair indices. Since this clustering process is explicitly designed
        # to track the same cells across sessions, it only keeps across-session pairs, discarding all within-session
        # pairs.
        pair_indices = []
        for i, pair in enumerate(is_possible_pair):
            # Only keeps the pairs formed by cells from different sessions.
            # noinspection PyTypeChecker
            if cell_info[pair[0]]["session"] != cell_info[pair[1]]["session"]:
                pair_indices.append((i, pair[0], pair[1]))

        # If no across-session pairs are found, advances to the next bin.
        if not pair_indices:
            continue

        # Pre-allocates the jaccard matrix
        jaccard_matrix_shape = int(((cell_number * cell_number) / 2) - (cell_number / 2))
        jaccard_matrix = np.ones(jaccard_matrix_shape) * 10000

        # Calculates jaccard distances between all pre-filtered across-session cell pairs
        for _i, cell_1, cell_2 in pair_indices:
            # noinspection PyTypeChecker
            cell_1_pixels = cell_info[cell_1]["ipix"]
            # noinspection PyTypeChecker
            cell_2_pixels = cell_info[cell_2]["ipix"]

            # Determines how many pixels from cell 1 are also in cell 2. This assumes that the same cell would generally
            # appear in roughly the same spatial position across motion-corrected session images. In turn, the cell
            # would then strongly overlap with itself across sessions.
            num_both = np.intersect1d(cell_1_pixels, cell_2_pixels, assume_unique=True).shape[0]

            # Computes the jaccard distance. The jaccard distance captures the similarity (overlap) between the two
            # cells as a ratio from 1 (perfect overlap) to 0 (no overlap).
            union_size = cell_1_pixels.shape[0] + cell_2_pixels.shape[0] - num_both

            # Handles a rare, but possible case of both cells being completely non-overlapping (avoiding division by
            # zero).
            jaccard_distance = 0 if union_size == 0 else 1 - num_both / union_size

            # Converts the 2d jaccard ratios into condensed 1D format used by the hierarchical clustering algorithm and
            # stores them in the pre-allocated matrix.
            jaccard_matrix[square_to_condensed(cell_1, cell_2, cell_number)] = jaccard_distance

        # Carries out hierarchical clustering using jaccard distance matrix. This groups similar cells across sessions
        # based on their jaccard distance (how much they overlap session-to-session). Again, the idea here is that the
        # same cell would be spatially similar to itself and spatially different from other cells across sessions.
        linkage_matrix = scipy.cluster.hierarchy.complete(jaccard_matrix)
        clusters = scipy.cluster.hierarchy.fcluster(Z=linkage_matrix, t=threshold, criterion=criterion)

        # Applies session prevalence filtering. Specifically, this only keeps cells that can be identified across
        # a certain number of sessions. This allows filtering cells that are not active during most sessions or that
        # cannot be reliably identified across sessions. Also, this step ensures that each cell cluster is only found
        # once in the output dataset, filtering duplicates that arise from the initial clustering pass.
        unique_clusters, counts = np.unique(clusters, return_counts=True)
        clusters[np.isin(clusters, unique_clusters[counts < minimum_sessions])] = 0
        unique_clusters = np.unique(clusters)

        # If the filtering process removes all cell clusters, advances to the next bin.
        for cluster_id in unique_clusters:
            if cluster_id == 0:
                continue

            # Computes the average (center) coordinate of the entire cell cluster (across all sessions).
            cluster_cell_pixels = clusters == cluster_id
            cluster_center = centers[cluster_cell_pixels].mean(axis=0)

            # Check if the cluster center is inside the currently processed bin. If not, skips processing the cluster
            # as part of this bin. This is necessary to avoid processing clusters on the edges of bins multiple times.
            if (
                y_position <= cluster_center[0] < y_position + step_x
                and x_position <= cluster_center[1] < x_position + step_y
            ):
                counter += 1  # Increments the discovered multi-day cell counter (putative cell counter)

                # Extracts the cell masks for all cells making up the cluster across sessions.
                cluster_cells = [cell_info[i] for i in np.where(cluster_cell_pixels)[0]]

                # Extracts the pixel coordinates (x, y) and lambda (intensity) values for all cluster cells.
                cluster_cell_pixels = np.hstack([mask["ipix"] for mask in cluster_cells])
                cluster_cell_intensities = np.hstack([mask["lam"] for mask in cluster_cells])

                # Finds pixels that consistently appear in most cell masks across sessions and only keeps these pixels.
                # This resolves the 'consensus' cell mask across session, which only includes the pixels that are
                # present in most sessions.
                unique, counts = np.unique(cluster_cell_pixels, return_counts=True)
                filtered_pixels = unique[(counts / len(cluster_cells)) > (pixel_prevalence / 100)]

                # If the cluster does not contain stable pixels, skips processing the cluster
                if len(filtered_pixels) == 0:
                    continue

                # Converts linear pixel indices to 2-dimensional coordinates.
                pixel_coordinates = np.unravel_index(filtered_pixels, image_size)
                xpix = pixel_coordinates[1]
                ypix = pixel_coordinates[0]

                # Calculates the properties of the template mask (center-point, lambda, and radius).
                center = [np.median(ypix), np.median(xpix)]
                radius = np.asarray([mask["radius"] for mask in cluster_cells]).mean()
                average_lambda = [cluster_cell_intensities[cluster_cell_pixels == i].mean() for i in filtered_pixels]

                # Adds the template mask to the temporary storage list.
                template_masks.append(
                    {
                        "id": counter,
                        "ipix": filtered_pixels,
                        "xpix": xpix,
                        "ypix": ypix,
                        "med": center,
                        "lam": np.array(average_lambda),
                        "radius": radius,
                        "num_sessions": len(cluster_cells),
                    }
                )

                # Updates cluster IDs for all cells in the processed cluster. This excludes these cells from further
                # processing iterations.
                for cell in cluster_cells:
                    cell["id"] = counter

    # Adds overlap pixel information to each template mask and records the number of template masks before the final
    # filtering step.
    template_masks = add_overlap_info(template_masks)
    before_size = len(template_masks)

    # Evaluates the non-overlapping portion of each template against the minimum template mask size threshold. This
    # filters out templates that appear too small to be considered valid cells. Adds the filtered tuple of template
    # masks to the MultiDayData object.
    data.template_cell_masks = tuple(
        [mask for mask in template_masks if (len(mask["ipix"]) - sum(mask["overlap"])) >= ops["minimum_size"]]
    )
    after_size = len(data.template_cell_masks)
    removed_templates = before_size - after_size
    message = f"Template cell masks: filtered. Removed {removed_templates} cells, kept {after_size} cells."
    console.echo(message=message, level=LogLevel.SUCCESS)

    # Duplicates the shared mask data for each session class. This is a bit wasteful but simplifies generating the
    # output .json file that stores all masks and images generated during registration at a later point in the
    # processing hierarchy.
    for session in data.sessions:
        session.shared_cell_masks = data.template_cell_masks

    return data


def backward_transform_masks(ops: dict[str, Any], data: MultiDayData) -> MultiDayData:
    """Backward-transforms the multi-day template cell masks to the original (unregistered) visual space of each
    session.

    Args:
        ops: The dictionary that stores the multi-day registration parameters.
        data: A MultiDayData instance that stores the template (across-session tracked) cell masks.

    Returns:
        The MultiDayData instance updated with the outcome of converting template cell masks back to each session's
        visual space.
    """
    # Resolves the number of parallel workers used to apply deformations to session's data.
    parallel_workers = ops["parallel_workers"]
    if parallel_workers < 1:
        parallel_workers = os.cpu_count()

    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        # Submits the backwards transform task for each session in parallel
        futures = [
            executor.submit(_backward_transform_session, template_masks=data.template_cell_masks, session=session)
            for session in data.sessions
        ]

        # Collects results with the progress bar using list comprehension
        results = [
            future.result()
            for future in tqdm(
                futures,
                desc="Transforming template masks to unregistered visual space",
                unit="session",
                disable=not ops["progress_bars"],
            )
        ]

    # Updates the sessions stored inside data with the results of the above pipeline
    data.sessions = results

    # Returns the updated MultiDayData instance
    return data


def _backward_transform_session(template_masks: tuple[dict[str, Any], ...], session: Session) -> Session:
    """This worker function is used by the main backward_transform_masks function to apply backward transformations
    to template masks for each session in parallel.
    """
    # Transform the template cell masks to the original (unregistered) visual space of this session
    session.template_cell_masks = deform_masks(
        cell_masks=template_masks,
        deform=session.deform.as_backward_inverse(),
    )

    # This step was performed before visualizing backwards-transformed data in the original multi-day notebook. Since
    # we have completely refactored the source code and the API, the step is now statically performed here so that the
    # results can be stored and loaded with other transformed image information stored in the transformed_images.npy
    # output file.
    session.transformed_images["lambda_weights"] = create_mask_image(
        session.template_cell_masks, session.deform.field_shape, field="lam"
    )
    return session
