"""This module provides the functions used to extract cell and neuropil fluorescence traces from all cells tracked
across multiple sessions and generate deconvolved cell spike traces.
"""

from typing import Any
from pathlib import Path

import numpy as np
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import LogLevel, console

from .. import extraction
from ..io import compute_plane_offsets
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
    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)

    # Resolves the path to the multi-day output folder of the session, which stores cached multi-day registration data.
    session_ids: list[str] = ops["session_ids"]
    multiday_output_paths: list[str] = ops["multiday_output_paths"]
    session_index = session_ids.index(session_id)
    multiday_folder = Path(multiday_output_paths[session_index])

    # Loads single-day suite2p processed data for all planes of the session.
    console.echo(f"Collecting session {session_id} data...")
    timer.reset()
    plane_folders = list(session_folder.glob("plane[0-9]"))
    plane_ops: list[dict[str, Any]] = [
        np.load(plane_folder.joinpath("ops.npy"), allow_pickle=True).item() for plane_folder in plane_folders
    ]
    registered_data_path = [plane_folder.joinpath("data.bin") for plane_folder in plane_folders]
    plane_y_coordinate, plane_x_coordinate = compute_plane_offsets(plane_contexts)
    plane_heights = np.array([ops["frame_height"] for ops in plane_ops])
    plane_widths = np.array([ops["frame_width"] for ops in plane_ops])
    movie_height = int(np.amax(plane_y_coordinate + plane_heights))
    movie_width = int(np.amax(plane_x_coordinate + plane_widths))

    # Loads multi-day tracked cell masks for the processed session
    multiday_cell_masks: list[dict[str, Any]] = np.load(
        multiday_folder.joinpath("backwards_deformed_cell_masks.npy"), allow_pickle=True
    )

    # Loads the ops.npy file stored inside the multiday folder. It contains the necessary parameters for extracting the
    # trace data.
    console.echo(f"Session {session_id} data: collected. Time taken {timer.elapsed} seconds.", level=LogLevel.SUCCESS)

    # Creates multi-day cell and neuropil masks in the combined (stitched) view
    console.echo(f"Creating session {session_id} multi-day cell masks...")
    timer.reset()
    # Re-computes the ROI stats for all multi-day tracked cells
    roi_statistics = roi_stats(
        multiday_cell_masks, ops["frame_height"], ops["frame_width"], ops["aspect_ratio"], ops["cell_diameter"]
    )
    cell_masks, neuropil_masks = extraction.masks.create_masks(
        roi_statistics=roi_statistics,
        height=ops["frame_height"],
        width=ops["frame_width"],
        neuropil=ops.get("extract_neuropil", True),
        ops=ops,
    )
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
            data=file,
            cell_masks=cell_masks,
            neuropil_masks=neuropil_masks,
            batch_size=ops["batch_size"],
            session_id=session_id,
        )

    # Computes delta fluorescence (dF) (neuropil-and-baseline-subtracted ROI fluorescence)
    df = extraction.preprocess(
        roi_fluorescence=cell_fluorescence,
        neuropil_fluorescence=neuropil_fluorescence,
        ops=ops,
    )

    # Cell activity spike deconvolution
    if ops.get("extract_spikes", True):
        message = f"Processing session {session_id} activity spikes..."
        console.echo(message=message, level=LogLevel.INFO)
        timer.reset()

        # Extracts the cell fluorescence spikes using the OASIS algorithm.
        spikes = extraction.oasis(
            cell_fluorescence=df,
            batch_size=ops["batch_size"],
            time_constant=ops["tau"],
            sampling_rate=ops["sampling_rate"],
        )
        ops["timing"]["multiday_deconvolution"] = timer.elapsed

        message = (
            f"Session {session_id} spikes: computed. Time taken: {ops['timing']['multiday_deconvolution']} seconds."
        )
        console.echo(message=message, level=LogLevel.SUCCESS)
    else:
        message = (
            f"Skipping session {session_id} spike deconvolution, as the 'extract_spikes' configuration parameter is "
            f"set to False."
        )
        console.echo(message=message, level=LogLevel.WARNING)
        spikes = np.zeros_like(cell_fluorescence)

    # Saves extracted data to disk
    console.echo(f"Saving extracted data to the {ops['dataset_name']} session {session_id} directory...")
    np.save(multiday_folder.joinpath("ops.npy"), ops)
    np.save(multiday_folder.joinpath("F.npy"), cell_fluorescence)
    np.save(multiday_folder.joinpath("Fneu.npy"), neuropil_fluorescence)
    np.save(multiday_folder.joinpath("Fsub.npy"), df)
    np.save(multiday_folder.joinpath("spks.npy"), spikes)
