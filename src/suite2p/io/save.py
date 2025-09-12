"""This module provides tools for exporting the multiplane data processed by a Suite2p single-day pipeline as a unified
suite2p or MATLAB dataset.
"""

from typing import Any
from pathlib import Path
from datetime import datetime

import numpy as np
import scipy
from natsort import natsorted
from numpy.typing import NDArray
from ataraxis_base_utilities import LogLevel, console, ensure_directory_exists


# noinspection PyTypeHints
def save_matlab(
    ops: dict[str, Any],
    roi_statistics: NDArray[Any],
    cell_fluorescence: NDArray[np.float32],
    neuropil_fluorescence: NDArray[np.float32],
    spikes: NDArray[np.float32],
    is_cell: NDArray[Any],
    red_cell: NDArray[Any],
    cell_fluorescence_channel_2: NDArray[np.float32] | None = None,
    neuropil_fluorescence_channel_2: NDArray[np.float32] | None = None,
) -> None:
    """Saves the input data to a MATLAB-compatible (.mat) file.

    Args:
        ops: The dictionary that stores the suite2p processing parameters.
        roi_statistics: The dictionary that stores the statistics for regions of interest (ROIs), including cell masks.
        cell_fluorescence: A NumPy array that stores the cell fluorescence data for the first channel.
        neuropil_fluorescence: A NumPy array that stores the neuropil fluorescence data for the first channel.
        spikes: A NumPy array that stores the deconvolved spike activity data for the first channel.
        is_cell: A NumPy array that specifies which regions of interest (ROIs) are cells for the first channel.
        red_cell: A NumPy array that specifies which regions of interests (ROIs) are cells for the second channel.
        cell_fluorescence_channel_2: A NumPy array that stores the cell fluorescence data for the second channel.
        neuropil_fluorescence_channel_2: A NumPy array that stores the neuropil fluorescence data for the second
            channel.
    """
    # Copies the input 'ops' dictionary to prevent modification by referencing.
    ops_matlab = ops.copy()

    # Formats the data processing date to make it compatible with MATLAB formatting.
    if "date_processed" in ops_matlab:
        ops_matlab["date_processed"] = str(datetime.strftime(ops_matlab["date_proc"], "%Y-%m-%d %H:%M:%S.%f"))

    # Loops over the items in the 'ops' dictionary and converts all Path instances to strings.
    for key, value in ops_matlab.items():
        # If the value is a Path object, converts the path to a string.
        if isinstance(value, Path):
            ops_matlab[key] = value.as_posix()

        # If the value is a list of Path objects, converts each path in the list into a string.
        elif isinstance(value, list) and value and isinstance(value[0], Path):
            ops_matlab[key] = [path.as_posix() for path in value]

    # Converts the regions of interest (ROI) statistics into a NumPy array of objects.
    roi_statistics = np.array(roi_statistics, dtype=object)

    # Creates a dictionary for saving the processed data as a .mat file.
    mat_dictionary: dict[str, Any] = {
        "stat": roi_statistics,
        "ops": ops_matlab,
        "F": cell_fluorescence,
        "Fneu": neuropil_fluorescence,
        "spks": spikes,
        "iscell": is_cell,
        "redcell": red_cell,
    }

    # If the processed data uses two channels, adds the second channel data to 'mat_dictionary' for saving.
    if cell_fluorescence_channel_2 is not None and neuropil_fluorescence_channel_2 is not None:
        mat_dictionary["F_chan2"] = cell_fluorescence_channel_2
        mat_dictionary["Fneu_chan2"] = neuropil_fluorescence_channel_2

    # Saves the data in 'mat_dictionary' to a MATLAB-compatible (.mat) file in the "save_path" directory.
    scipy.io.savemat(file_name=ops["save_path"].joinpath("Fall.mat"), mdict=mat_dictionary)


def compute_dydx(plane_ops: list[dict[str, Any]]) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Computes the displacement for each plane in the input list of plane-specific 'ops' dictionaries.

    The displacement values are calculated based on the dimensions and configuration parameters provided in the 'ops'
    dictionary. If "dx" and "dy" are already specified in the 'ops' dictionaries, those values are used. If not, the
    function computes the displacement using the dimensions of each plane.

    Notes:
        The output of this function is used to properly arrange the data from multiple planes in the 'shared' recording
        space, re-assembling the recording from individually processed planes. This is used as part of outputting the
        suite2p-processed data as a 'combined' dataset that integrates the data from all available planes.

    Args:
        plane_ops: A list of plane-specific dictionaries that stores the suite2p processing parameters.

    Returns:
        A tuple of two elements. The first element is an array of y-displacement values, and the second element is an
        array of x-displacement values.
    """
    # Copies the first available plane-specific 'ops' dictionary to prevent modification by referencing.
    ops = plane_ops[0].copy()

    # Calculates the number of planes.
    plane_number = len(plane_ops)

    # Initializes NumPy arrays to store the calculated displacement values for y-axis and x-axis.
    y_displacement = np.zeros(plane_number, np.int64)
    x_displacement = np.zeros(plane_number, np.int64)

    # If "dy" and "dx" are not already provided in the first 'ops' dictionary, computes them based on the dimensions.
    if ("dy" not in ops) or ("dx" not in ops):
        # Queries the height and width of the first plane's 'ops' dictionary.
        height = ops["Ly"]
        width = ops["Lx"]

        # Calculates the number of pixel columns needed to arrange the planes, based on their dimension.
        column_number = np.ceil(np.sqrt(ops["Ly"] * ops["Lx"] * plane_number) / ops["Lx"]).astype(int)

        # Loops over all available planes and calculates the displacement values of each plane based on the column and
        # row positions.
        for plane_index in range(plane_number):
            x_displacement[plane_index] = (plane_index % column_number) * width
            y_displacement[plane_index] = np.floor_divide(plane_index, column_number) * height

    # Otherwise, uses "dy" and "dx" values directly.
    else:
        # Queries the values of "dy" and "dx" from each plane-specific 'ops' dictionary.
        x_displacement = np.array([ops["dx"] for ops in plane_ops])
        y_displacement = np.array([ops["dy"] for ops in plane_ops])

        # Identifies the unique (dy, dx) pairs and determines the number of unique regions of interests (ROIs).
        unique_positions = np.unique(np.vstack((y_displacement, x_displacement)), axis=1)
        roi_number = unique_positions.shape[1]

        # If the number of regions of interest (ROIs) is lower than the number of planes, recalculates the displacement
        # values based on the maximum dimensions.
        if roi_number < plane_number:
            # Recalculates the number of planes.
            plane_number //= roi_number

            # Queries the widths and heights for each plane.
            height = np.array([ops["Ly"] for ops in plane_ops])
            width = np.array([ops["Lx"] for ops in plane_ops])

            # Calculates the maximum height and width based on the computed displacement values and plane dimensions.
            maximum_height = (y_displacement + height).max()
            maximum_width = (x_displacement + width).max()

            # Recalculates the number of columns needed to arrange the planes.
            column_number = np.ceil(np.sqrt(maximum_height * maximum_width * plane_number) / maximum_width).astype(int)

            # Loops over all available planes and updates the displacement values for each region of interest (ROI)
            # based on the column and row positions.
            for plane_index in range(plane_number):
                for roi_index in range(roi_number):
                    roi_plane_index = plane_index * roi_number + roi_index
                    x_displacement[roi_plane_index] += (plane_index % column_number) * maximum_width
                    y_displacement[roi_plane_index] += np.floor_divide(plane_index, column_number) * maximum_height

    # Returns the lists of the y-axis and x-axis displacement values.
    return y_displacement, x_displacement


# noinspection PyUnboundLocalVariable
def combined(
    save_directory: Path, save: bool = True
) -> tuple[
    NDArray[Any],
    dict[str, Any],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[np.float32],
    NDArray[Any],
    NDArray[Any],
    NDArray[Any],
    NDArray[Any],
    bool,
]:
    """Combines the data from each input plane-specific directory under 'save_directory' into a single resulting
    'combined' directory.

    This function combines multi-plane and multi-roi recording data into a unified recording dataset, effectively
    reassembling the original recording from individually processed planes, adjusting their placement according to the
    specified displacement values for both planes and ROIs.

    Args:
        save_directory: The directory containing all processed plane subdirectories.
        save: Determines whether to save the combined data to disk.

    Returns:
        A tuple of 11 elements. The first element is a NumPy array that stores the ROI statistics. The second element
        is the 'ops' dictionary that combines the data for all processed planes. The third, fourth, fifth, and sixths
        elements are the NumPy arrays storing the cell fluorescence, neuropil fluorescence,
        baseline-and-neuropil-subtracted cell fluorescence, and deconvolved spike data. The seventh and eights elements
        are NumPy arrays that store the boolean cell classification and likelihood data (in this order) for the first
        channel. The ninth and tenth elements are NumPy arrays that store the boolean cell classification and
        likelihood data for the second channel. The elevens element is a boolean flag that indicates whether the
        processed recording contained the second channel.
    """
    # Extracts the list of plane directories in the save folder and sorts them in the natural (ascending) order.
    plane_directories = natsorted(
        [directory for directory in save_directory.iterdir() if directory.is_dir() and directory.name[:5] == "plane"]
    )

    # Loads the ops.npy file for each plane as a dictionary.
    plane_ops = [np.load(directory.joinpath("ops.npy"), allow_pickle=True).item() for directory in plane_directories]

    # Computes the y-axis and x-axis displacement for each plane. These displacement values are used to arrange
    # individual planes back into the original recording movie.
    y_displacement, x_displacement = compute_dydx(plane_ops)

    # Queries the height and width for each plane.
    height = np.array([ops["Ly"] for ops in plane_ops])
    width = np.array([ops["Lx"] for ops in plane_ops])

    # Calculates the overall height and width of the entire recording plane after accounting for plane displacement.
    maximum_height = int(np.amax(y_displacement + height))
    maximum_width = int(np.amax(x_displacement + width))

    # Initializes 2D NumPy arrays to store the mean images and filtered mean images for the combined data.
    channel_1_mean_image = np.zeros((maximum_height, maximum_width))
    filtered_mean_image = np.zeros((maximum_height, maximum_width))

    message = (
        f"Combining the processed data for {plane_ops[0]['nchannels']} channels from folders "
        f"{[Path(folder).name for folder in plane_directories]}..."
    )
    console.echo(message=message, level=LogLevel.INFO)

    # If the processed data uses two channels, initializes a 2D NumPy array to store the second channel's mean image.
    if plane_ops[0]["nchannels"] > 1:
        channel_2_mean_image = np.zeros((maximum_height, maximum_width))

    # Initializes a 2D NumPy array to store the second channel's corrected mean image if specified in any of the
    # plane-specific 'ops' dictionaries.
    if any("meanImg_chan2_corrected" in ops for ops in plane_ops):
        channel_2_corrected_mean_image = np.zeros((maximum_height, maximum_width))

    # Initializes a 2D NumPy array to store the maximum projection image if specified in any of the plane-specific
    # 'ops' dictionaries.
    if any("max_proj" in ops for ops in plane_ops):
        maximum_projection = np.zeros((maximum_height, maximum_width))

    # Initializes a 2D NumPy array to store the correlation map.
    correlation_map = np.zeros((maximum_height, maximum_width))

    # Finds the maximum number of frames across all planes.
    maximum_frame_number = np.amax(np.array([ops["nframes"] for ops in plane_ops]))

    # Loops over all available planes to process each plane's data.
    for plane_index, ops in enumerate(plane_ops):
        # Queries the path to the directory of the processed plane.
        plane_directory_path = plane_directories[plane_index]

        # If there is no stats.npy file in the directory (the plane has no ROIs), skips to the next plane.
        if not plane_directory_path.joinpath("stat.npy").exists():
            continue

        # Loads the regions of interest (ROI) statistics from stat.npy for the processed plane.
        plane_roi_statistics = np.load(plane_directory_path.joinpath("stat.npy"), allow_pickle=True)

        # Calculates the y-pixel and x-pixel ranges based on the displacement and the dimensions of the processed plane.
        y_range = np.arange(y_displacement[plane_index], y_displacement[plane_index] + height[plane_index])
        x_range = np.arange(x_displacement[plane_index], x_displacement[plane_index] + width[plane_index])

        # Updates the mean image and the filtered mean image with the processed plane's mean image and filtered mean
        # image data.
        channel_1_mean_image[np.ix_(y_range, x_range)] = ops["meanImg"]
        filtered_mean_image[np.ix_(y_range, x_range)] = ops["meanImgE"]

        # If the processed data uses two channels, updates the second channels' mean image with the processed plane's
        # data.
        if "meanImg_chan2" in ops:
            channel_2_mean_image[np.ix_(y_range, x_range)] = ops["meanImg_chan2"]

        # Updates the corrected second channel's mean image if specified in the processed plane's 'ops' dictionary.
        if "meanImg_chan2_corrected" in ops:
            channel_2_corrected_mean_image[np.ix_(y_range, x_range)] = ops["meanImg_chan2_corrected"]

        # Updates the correlation map using the processed plane's data.
        y_range = np.arange(
            y_displacement[plane_index] + ops["yrange"][0], y_displacement[plane_index] + ops["yrange"][-1]
        )
        x_range = np.arange(
            x_displacement[plane_index] + ops["xrange"][0], x_displacement[plane_index] + ops["xrange"][-1]
        )
        correlation_map[np.ix_(y_range, x_range)] = ops["Vcorr"]

        # Updates the maximum projection image if specified in the processed plane's 'ops' dictionary.
        if "max_proj" in ops:
            maximum_projection[np.ix_(y_range, x_range)] = ops["max_proj"]

        # Updates the regions of interest (ROI) statistics with the processed plane's displacement values and index.
        for plane_roi_index in range(len(plane_roi_statistics)):
            plane_roi_statistics[plane_roi_index]["xpix"] += x_displacement[plane_index]
            plane_roi_statistics[plane_roi_index]["ypix"] += y_displacement[plane_index]
            plane_roi_statistics[plane_roi_index]["med"][0] += y_displacement[plane_index]
            plane_roi_statistics[plane_roi_index]["med"][1] += x_displacement[plane_index]
            plane_roi_statistics[plane_roi_index]["iplane"] = plane_index

        # Loads the 'cell_fluorescence', 'neuropil_fluorescence', 'spikes', and 'is_cell' data for the processed plane.
        plane_cell_fluorescence = np.load(plane_directory_path.joinpath("F.npy"))
        plane_neuropil_fluorescence = np.load(plane_directory_path.joinpath("Fneu.npy"))
        plane_baseline_subtracted_fluorescence = np.load(plane_directory_path.joinpath("Fsub.npy"))
        plane_spikes = np.load(plane_directory_path.joinpath("spks.npy"))
        plane_is_cell = np.load(plane_directory_path.joinpath("iscell.npy"))

        # Checks if the 'red_cell' (channel 2 cell classification data) data is available and loads it if present.
        if plane_directory_path.joinpath("redcell.npy").is_file():
            plane_red_cell = np.load(plane_directory_path.joinpath("redcell.npy"))
            has_red = True
        else:
            plane_red_cell = []
            has_red = False

        # Extracts the number of cells (ROIs) and frame count from the processed plane's cell fluorescence data.
        cell_count, frame_count = plane_cell_fluorescence.shape

        # Ensures the number of frames in the processed plane's data matches the maximum frame number by padding
        # with zeros to match the recording's maximum frame number.
        if frame_count < maximum_frame_number:
            # noinspection PyUnresolvedReferences
            padding = np.zeros((cell_count, maximum_frame_number - frame_count), "float32")
            plane_cell_fluorescence = np.concatenate((plane_cell_fluorescence, padding), axis=1)
            plane_spikes = np.concatenate((plane_spikes, padding), axis=1)
            plane_neuropil_fluorescence = np.concatenate((plane_neuropil_fluorescence, padding), axis=1)
            plane_baseline_subtracted_fluorescence = np.concatenate(
                (plane_baseline_subtracted_fluorescence, padding), axis=1
            )

        # Appends the processed plane's data to the combined arrays.
        if plane_index == 0:
            (
                cell_fluorescence,
                neuropil_fluorescence,
                subtracted_fluorescence,
                spikes,
                roi_statistics,
                is_cell,
                red_cell,
            ) = (
                plane_cell_fluorescence,
                plane_neuropil_fluorescence,
                plane_baseline_subtracted_fluorescence,
                plane_spikes,
                plane_roi_statistics,
                plane_is_cell,
                plane_red_cell,
            )
        else:
            cell_fluorescence = np.concatenate((cell_fluorescence, plane_cell_fluorescence))
            neuropil_fluorescence = np.concatenate((neuropil_fluorescence, plane_neuropil_fluorescence))
            subtracted_fluorescence = np.concatenate((subtracted_fluorescence, plane_baseline_subtracted_fluorescence))
            spikes = np.concatenate((spikes, plane_spikes))
            roi_statistics = np.concatenate((roi_statistics, plane_roi_statistics))
            is_cell = np.concatenate((is_cell, plane_is_cell))
            if has_red:
                red_cell = np.concatenate((red_cell, plane_red_cell))

        console.echo(message=f"Appended plane {plane_index} data to combined view.", level=LogLevel.SUCCESS)

    # Modifies the combined 'ops' dictionary to include descriptive information about the combined recording data
    ops["meanImg"] = channel_1_mean_image
    ops["meanImgE"] = filtered_mean_image
    if ops["nchannels"] > 1:
        ops["meanImg_chan2"] = channel_2_mean_image
    if "meanImg_chan2_corrected" in ops:
        ops["meanImg_chan2_corrected"] = channel_2_corrected_mean_image
    if "max_proj" in ops:
        ops["max_proj"] = maximum_projection
    ops["Vcorr"] = correlation_map
    ops["Ly"] = maximum_height
    ops["Lx"] = maximum_width
    ops["xrange"] = [0, ops["Lx"]]
    ops["yrange"] = [0, ops["Ly"]]

    # Prepares the path to the directory that will store the combined data files
    combined_directory_path = save_directory.joinpath("combined")

    # Creates the save directory if it does not exist.
    ensure_directory_exists(combined_directory_path)

    # Stores the path to the save directory in the 'ops' dictionary.
    ops["save_path"] = combined_directory_path

    # Caches cell classification data to disk. Since cell classification data is required for the suite2p GUI to work
    # is expected, this is done regardless of the 'save' argument value.
    np.save(combined_directory_path.joinpath("iscell.npy"), is_cell)
    if has_red:
        np.save(combined_directory_path.joinpath("redcell.npy"), red_cell)
    else:
        red_cell = np.zeros_like(is_cell)

    # If 'save' is set to True, saves the combined data in the save directory.
    if save:
        np.save(combined_directory_path.joinpath("F.npy"), cell_fluorescence)
        np.save(combined_directory_path.joinpath("Fneu.npy"), neuropil_fluorescence)
        np.save(combined_directory_path.joinpath("Fsub.npy"), subtracted_fluorescence)
        np.save(combined_directory_path.joinpath("spks.npy"), spikes)
        np.save(combined_directory_path.joinpath("ops.npy"), ops)
        np.save(combined_directory_path.joinpath("stat.npy"), roi_statistics)

        # If "save_mat" is set to True, saves the data to a MATLAB-compatible (.mat) file.
        if ops.get("save_mat"):
            save_matlab(
                ops,
                roi_statistics,
                cell_fluorescence,
                neuropil_fluorescence,
                subtracted_fluorescence,
                spikes,
                is_cell,
                red_cell,
            )

    # Returns the combined data as a tuple.
    return (
        roi_statistics,
        ops,
        cell_fluorescence,
        neuropil_fluorescence,
        subtracted_fluorescence,
        spikes,
        is_cell[:, 0],
        is_cell[:, 1],
        red_cell[:, 0],
        red_cell[:, 1],
        has_red,
    )
