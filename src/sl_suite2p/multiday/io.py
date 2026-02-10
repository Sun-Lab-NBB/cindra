"""This module provides the functions used to import and export data during the first step (registration) of the
multi-day processing pipeline.
"""

from typing import Any
from pathlib import Path

import numpy as np
from ataraxis_base_utilities import LogLevel, console

from ..dataclasses import Session, MultiDayData


def import_sessions(ops: dict[str, Any]) -> MultiDayData:
    """Imports the single-session suite2p data used for multiday registration from all requested sessions.

    This method extracts the data generated during single-day suite2p processing that is required to carry out the
    multi-day registration.

    Args:
        ops: The dictionary that contains the multiday registration parameters.

    Returns:
        MultiDayData class that stores the data extracted for all requested sessions.

    Raises:
        FileNotFoundError: If the functions cannot find the /combined plane folder using the paths provided via
            'multiday_output_paths' field in the 'ops' dictionary for one or more target sessions.
    """
    # Retrieves session IDs and multiday output paths from ops. These are populated by resolve_multiday_ops().
    session_ids: list[str] = ops["session_ids"]
    multiday_output_paths: list[str] = ops["multiday_output_paths"]

    # Temporarily storage for imported session data, before it is packaged into the MultiDayData class.
    session_classes = []

    # Processes each session sequentially. The suite2p folder is at the same level as the 'multiday' folder.
    for session_id, multiday_output in zip(session_ids, multiday_output_paths, strict=True):
        # Derives the suite2p folder path from the multiday output path.
        # multiday_output is: {suite2p_parent}/multiday/{dataset_name}/
        # suite2p folder is: {suite2p_parent}/suite2p/
        suite2p_parent = Path(multiday_output).parent.parent
        session_path = suite2p_parent.joinpath("suite2p")
        # Ensures that the input session paths point to the root single-session suite2p output folder. Specifically,
        # uses the heuristic that the folder contains the 'combined' plane folder. In turn, that folder has to contain
        # the 'combined' ops.npy file.
        combined_folder = session_path.joinpath("combined")
        if not combined_folder.is_dir() and combined_folder.joinpath("ops.npy").exists():
            message = (
                f"Could not find the 'combined' suite2p folder for session: {session_id}. All sessions have to be "
                f"processed with single-session suite2p pipeline before being submitted to multi-session pipeline. "
                f"Additionally, all sessions, regardless of the number of planes processed for that session, must "
                f"generate the 'combined' folder as part of the single-session processing."
            )
            console.error(message=message, error=FileNotFoundError)

        # Extracts single-day .npy files from the combined folder:
        # Configuration parameters and general processing data.
        single_day_ops = np.load(combined_folder.joinpath("ops.npy"), allow_pickle=True).item()
        # Cell masks
        single_day_stat = np.load(combined_folder.joinpath("stat.npy"), allow_pickle=True)
        # Cell classification data
        single_day_iscell = np.load(combined_folder.joinpath("iscell.npy"), allow_pickle=True)

        # Extracts reference images. These images will be used to register the sessions to each-other across days
        images = {
            "mean": single_day_ops["mean_image"].astype(np.float32),
            "enhanced": single_day_ops["enhanced_mean_image"].astype(np.float32),
            "max": single_day_ops["maximum_projection"].astype(np.float32),
        }

        # Resolves parameters for the list comprehension below to make it visually simpler
        keys_to_keep = ["x_pixels", "y_pixels", "pixel_weights", "centroid", "radius", "overlap_mask"]
        prob_threshold = ops["probability_threshold"]
        max_size = ops["maximum_size"]

        # Subsamples suite2p-extracted cell ROIs. The multiday pipeline typically uses more stringent cell
        # identification criteria than the single-day pipeline, so this step is expected to discard some
        # single-day ROIs.
        selected_cells = [
            {key: mask[key] for key in keys_to_keep}
            for cell_index, mask in enumerate(single_day_stat)
            if single_day_iscell[cell_index, 1] > prob_threshold and mask["pixel_count"] < max_size
        ]  # Loads cell data for all single-day ROIs that satisfy the size and probability thresholds

        # Removes ROIs too close to region borders. This step is skipped if the runtime is not configured to filter
        # cells around borders (if the region borders list is empty).
        if ops["mroi_region_borders"]:
            region_margin = ops["region_margin"]
            filtered_cells = [
                cell
                for cell in selected_cells
                if all(abs(cell["centroid"][1] - border) > region_margin for border in ops["mroi_region_borders"])
            ]
        else:
            # Otherwise, all selected cells automatically pass the region filtering step.
            filtered_cells = selected_cells

        # Uses the 'mean' image to determine the shape (height and width) of the combined session movie.
        image_size = images["mean"].shape

        # Packages imported data into a Session class instance
        session_data = Session(
            session_id=session_id,
            suite2p_folder=session_path,
            reference_images=images,
            image_size=image_size,
            cell_masks=tuple(filtered_cells),
        )

        # Appends each Session class to the temporary list
        session_classes.append(session_data)

        message = f"Extracted single-session suite2p data for {len(filtered_cells)} cells from session {session_id}."
        console.echo(message, level=LogLevel.SUCCESS)

    # Packages extracted data into the MultiDayData instance before returning it to the caller.
    return MultiDayData(sessions=session_classes)


def export_masks_and_images(
    ops: dict[str, Any],
    data: MultiDayData,
) -> None:
    """Exports multi-day registration (processing step 1) data to the multi-day folder of each processed session.

    The multi-day registration data primarily includes the multi-day tracked cell masks (both in that session's
    original and multi-day visual space). It also includes the ops.npy settings file and the reference images used
    during single-day and multi-day registration. This information is then used to extract the activity (fluorescence)
    data for cells tracked across sessions as part of the second multi-day processing step.

    Notes:
        This step also saves various mask images generated by each sub-step of the registration step. This information
        is not used by the pipeline itself but allows users to visually assess the quality of multi-day registration.

    Args:
        ops: The dictionary that stores the multi-day processing parameters.
        data: A MultiDayData instance that stores the data aggregated during the multi-day registration (step 1)
            pipeline.
    """
    # Retrieves the multiday output paths from ops.
    multiday_output_paths: list[str] = ops["multiday_output_paths"]
    session_ids: list[str] = ops["session_ids"]

    # Loops over sessions and exports data to each session's multiday output folder.
    for session in data.sessions:
        # Finds the multiday output folder for this session using the session ID.
        session_index = session_ids.index(session.session_id)
        output_folder = Path(multiday_output_paths[session_index])

        # Template cell masks translated to the original session visual space
        np.save(
            output_folder.joinpath("backwards_deformed_cell_masks.npy"),
            session.template_cell_masks,
        )

        # Template cell masks in the multi-day registered (deformed) visual space
        np.save(output_folder.joinpath("template_cell_masks.npy"), data.template_cell_masks)

        # Reference images modified by deformation (multi-day registration) offsets
        np.save(output_folder.joinpath("transformed_images.npy"), session.transformed_images)

        # Original (single-day) reference images
        np.save(output_folder.joinpath("original_images.npy"), session.reference_images)

        # Multi-day processing parameters.
        np.save(output_folder.joinpath("ops.npy"), ops)

        # Cell mask arrays
        np.save(output_folder.joinpath("unregistered_masks.npy"), session.unregistered_masks)
        np.save(output_folder.joinpath("registered_masks.npy"), session.registered_masks)
        np.save(output_folder.joinpath("shared_multiday_masks.npy"), session.shared_template_masks)
        np.save(output_folder.joinpath("session_multiday_masks.npy"), session.session_template_masks)
