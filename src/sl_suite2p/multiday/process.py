"""This module provides the functions used to extract cell and neuropil fluorescence traces from all cells tracked
across multiple sessions and generate deconvolved cell spike traces.
"""

from typing import Any
from pathlib import Path

import numpy as np
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import LogLevel, console

from .. import extraction
from ..io import compute_dydx
from ..io.binary import BinaryFileCombined
from ..detection.stats import roi_stats


def extract_session_traces(ops: dict[str, Any], session_folder: Path, session_id: str) -> None:
    """Extracts the cell and neuropil fluorescence traces for a single session using multi-day registered cell masks.

    This function extracts the fluorescence of cells tracked across multiple sessions. It is designed to be called
    in parallel for all processed sessions and requires all sessions to be first registered using the first processing
    step of the multi-day suite2p pipeline.

    Args:
        ops: The dictionary that stores the multi-day processing parameters.
        session_folder: The path to the root suite2p output folder of the processed session. Typically, this is the
            default 'suite2p' folder, which stores 'plane' and 'combined' folders.
        session_id: The unique identifier of the processed session.

    Raises:
        FileNotFoundError: If the session's suite2p output folder does not contain the expected multi-day output,
            indicating that the session has not been processed with the multi-day registration pipeline before calling
            the trace extraction pipeline.
    """
    # Initializes the run timer
    timer = PrecisionTimer("s")

    # Resolves the path to the multi-day output folder of the session, which stores cached multi-day registration data.
    multiday_folder = Path(ops["multiday_save_path"]).joinpath(ops["multiday_save_folder"], session_id)

    # Loads single-day suite2p processed data for all planes of the session.
    console.echo(f"Collecting session {session_id} data...")
    timer.reset()
    plane_folders = list(session_folder.glob("plane[0-9]"))
    plane_ops: list[dict[str, Any]] = [
        np.load(plane_folder.joinpath("ops.npy"), allow_pickle=True).item() for plane_folder in plane_folders
    ]
    registered_data_path = [plane_folder.joinpath("data.bin") for plane_folder in plane_folders]
    plane_y_coordinate, plane_x_coordinate = compute_dydx(plane_ops)
    plane_heights = np.array([ops["Ly"] for ops in plane_ops])
    plane_widths = np.array([ops["Lx"] for ops in plane_ops])
    movie_height = int(np.amax(plane_y_coordinate + plane_heights))
    movie_width = int(np.amax(plane_x_coordinate + plane_widths))

    # Loads multi-day tracked cell masks for the processed session
    cell_masks: list[dict[str, Any]] = np.load(
        multiday_folder.joinpath("backwards_deformed_cell_masks.npy"), allow_pickle=True
    )

    # Loads the ops.npy file stored inside the multiday folder. It contains the necessary parameters for extracting the
    # trace data.
    console.echo(f"Session {session_id} data: collected. Time taken {timer.elapsed} seconds.", level=LogLevel.SUCCESS)

    # Creates multi-day cell and neuropil masks in the combined (stitched) view
    console.echo(f"Creating session {session_id} multi-day cell masks...")
    timer.reset()
    # Re-computes the ROI stats for all multi-day tracked cells
    cell_masks = roi_stats(cell_masks, ops["Ly"], ops["Lx"], ops["aspect"], ops["diameter"])
    cell_masks, neuropil_masks = extraction.masks.create_masks(cell_masks, ops["Ly"], ops["Lx"], ops)
    message = f"Session {session_id} multi-day masks: created. Time taken {timer.elapsed} seconds."
    console.echo(message=message, level=LogLevel.SUCCESS)

    # Extracts traces from the single-day registered binary files
    console.echo(f"Extracting session {session_id} fluorescence traces for cells tracked across days...")
    with BinaryFileCombined(
        movie_height,
        movie_width,
        plane_heights,
        plane_widths,
        plane_y_coordinate,
        plane_x_coordinate,
        registered_data_path,
    ) as file:
        cell_fluorescence, neuropil_fluorescence = extraction.extract.extract_traces(
            f_in=file,
            plane_number="combined",
            cell_masks=cell_masks,
            neuropil_masks=neuropil_masks,
            batch_size=ops["batch_size"],
            session_id=session_id,
        )

    # Computes delta f/f (neuropil-subtracted ROI fluorescence)
    dff = cell_fluorescence.copy() - ops["neucoeff"] * neuropil_fluorescence
    dff = extraction.preprocess(
        F=dff,
        baseline=ops["baseline"],
        win_baseline=ops["win_baseline"],
        sig_baseline=ops["sig_baseline"],
        fs=ops["fs"],
        prctile_baseline=ops["prctile_baseline"],
    )

    # Cell activity spike deconvolution
    if ops.get("spikedetect", True):
        message = f"Processing session {session_id} activity spikes..."
        console.echo(message=message, level=LogLevel.INFO)
        timer.reset()

        # Extracts the cell fluorescence spikes using the OASIS algorithm.
        spikes = extraction.oasis(F=dff, batch_size=ops["batch_size"], tau=ops["tau"], fs=ops["fs"])
        ops["timing"]["multiday_deconvolution"] = timer.elapsed

        message = (
            f"Session {session_id} spikes: computed. Time taken: {ops['timing']['multiday_deconvolution']} seconds."
        )
        console.echo(message=message, level=LogLevel.SUCCESS)
    else:
        message = (
            f"Skipping session {session_id} spike deconvolution, as it is disabled via the 'spikedetect' "
            f"configuration parameter."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        spikes = np.zeros_like(cell_fluorescence)

    # Saves extracted data to disk
    console.echo(f"Saving extracted data to the {ops['multiday_save_folder']} session {session_id} folder...")
    np.save(multiday_folder.joinpath("ops.npy"), ops)
    np.save(multiday_folder.joinpath("F.npy"), cell_fluorescence)
    np.save(multiday_folder.joinpath("Fneu.npy"), neuropil_fluorescence)
    np.save(multiday_folder.joinpath("Fsub.npy"), dff)
    np.save(multiday_folder.joinpath("spks.npy"), spikes)
