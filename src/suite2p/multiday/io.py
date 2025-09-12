"""This module provides the functions used to import and export data during the first step (registration) of the
multi-day processing pipeline.
"""

from typing import Any
from pathlib import Path

import numpy as np
from natsort import natsorted
from ataraxis_base_utilities import LogLevel, console

from .dataclasses import Session, MultiDayData
from ..configuration import SingleDayS2PConfiguration


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
            'session_folders' field of the 'io' section inside the 'ops' dictionary for one or more target sessions.
    """
    # Reconstructs the path to the output folder from 'ops' parameters
    output_folder = Path(ops["multiday_save_path"]).joinpath(ops["multiday_save_folder"])

    # The output folder contains .npy and .yaml files and directories named after each processed session ID.
    # This re-generates the list of session IDs from the directories stored in the output folder.
    session_ids = [folder.stem for folder in output_folder.glob("*") if folder.is_dir()]

    # Sorts session IDs for consistency
    session_ids = natsorted(session_ids)

    # Loops over the output session directories and reconstructs the path to the single-day output folder of each
    # session using the data inside the single-day s2p config from the multi-day folder.
    sessions = []
    for session in session_ids:
        configuration: SingleDayS2PConfiguration = SingleDayS2PConfiguration.from_yaml(
            output_folder.joinpath(session, "single_day_ss2p_configuration.yaml")
        )
        session_folder = Path(configuration.file_io.save_path0).joinpath(configuration.file_io.save_folder)
        sessions.append(session_folder)

    # Temporarily storage for imported session data, before it is packaged into the MultiDayData class.
    session_classes = []

    # Processes each session sequentially
    for session_id, session_path in zip(session_ids, sessions, strict=False):
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
            raise FileNotFoundError(message)  # Fallback to appease mypy, should not be reachable

        # Extracts single-day .npy files from the combined folder:
        # Configuration parameters and general processing data.
        single_day_ops = np.load(combined_folder.joinpath("ops.npy"), allow_pickle=True).item()
        # Cell masks
        single_day_stat = np.load(combined_folder.joinpath("stat.npy"), allow_pickle=True)
        # Cell classification data
        single_day_iscell = np.load(combined_folder.joinpath("iscell.npy"), allow_pickle=True)

        # Extracts reference images. These images will be used to register the sessions to each-other across days
        images = {
            "mean": single_day_ops["meanImg"].astype(np.float32),
            "enhanced": single_day_ops["meanImgE"].astype(np.float32),
            "max": single_day_ops["max_proj"].astype(np.float32),
        }

        # Resolves parameters for the list comprehension below to make it visually simpler
        keys_to_keep = ["xpix", "ypix", "lam", "med", "radius", "overlap"]
        prob_threshold = ops["probability_threshold"]
        max_size = ops["maximum_size"]

        # Subsamples suite2p-extracted cell ROIs. The multiday pipeline typically uses more stringent cell
        # identification criteria than the single-day pipeline, so this step is expected to discard some
        # single-day ROIs.
        selected_cells = [
            {key: mask[key] for key in keys_to_keep}
            for cell_index, mask in enumerate(single_day_stat)
            if single_day_iscell[cell_index, 1] > prob_threshold and mask["npix"] < max_size
        ]  # Loads cell data for all single-day ROIs that satisfy the size and probability thresholds

        # Removes ROIs too close to stripe margins. This step is skipped if the runtime is not configured to filter
        # cells around borders (if the stripe borders list is empty).
        if ops["mesoscope_stripe_borders"]:
            stripe_margin = ops["stripe_margin"]
            filtered_cells = [
                cell
                for cell in selected_cells
                if all(abs(cell["med"][1] - border) > stripe_margin for border in ops["mesoscope_stripe_borders"])
            ]
        else:
            # Otherwise, all selected cells automatically pass the stripe filtering step.
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
    # Loops over sessions and, for each, generates a new directory under the single-day generated 'suite2p' folder.
    # This directory is found at the same level as the 'combined' and 'plane' folders.
    for session in data.sessions:
        # Reconstructs the path to the multi-day output folder of each session from 'ops' parameters
        output_folder = Path(ops["multiday_save_path"]).joinpath(ops["multiday_save_folder"], session.session_id)

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
