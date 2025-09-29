"""Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu."""

from typing import Any, List
import platform

from numba import njit, config, prange
import numpy as np
from scipy import stats, signal
from numba.typed import List
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import LogLevel, console

from .masks import create_masks
from ..io.binary import BinaryFile
from ..configuration import generate_default_ops

if platform.system() == "Darwin":
    config.THREADING_LAYER = "omp"
else:
    config.THREADING_LAYER = "tbb"


@njit(parallel=True)
def matmul_traces(
    cell_fluorescence: NDArray[np.float64],
    data_matrix: NDArray[np.float64],
    cell_pixel_indices: List[NDArray[np.int64]],
    lambda_weights: List[NDArray[np.float64]],
) -> NDArray[np.float64]:
    """
    Computes cell fluorescence traces by weighted matrix multiplication with their 
    corresponding weights (lambda values). The weighted sum across all pixels gives 
    the final fluorescence trace for that ROI across all frames.

    Args:
        data_matrix: A data array of size n_frames × n_pixels.
        cell_pixel_indices: A list of arrays containing the pixel indices belonging to each ROI.
        lambda_weights: A list of arrays specifying the weight of each pixel within its ROI
    """
    num_cells = cell_fluorescence.shape[0]

    for cell_idx in prange(num_cells):
        cell_idx = np.int64(cell_idx)  # This is here to fix Numba's 'unsafe uint64 -> int64 cast warning.'

        cell_pixels_data = data_matrix[:, cell_pixel_indices[cell_idx]]
        pixel_weights = lambda_weights[cell_idx]
        cell_fluorescence[cell_idx] = np.dot(cell_pixels_data, pixel_weights)

    return cell_fluorescence


@njit(parallel=True)
def matmul_neuropil(
    neuropil_fluorescence: NDArray[np.float64],
    data_matrix: NDArray[np.float64],
    neuropil_pixel_indices: List[NDArray[np.int64]],
    neuropil_pixel_count: NDArray[np.int64],
) -> NDArray[np.float64]:
    """
    Extracts the fluorescence signals from pixels in the surrounding neuropil ring and computes 
    the mean fluorescence across those pixels.

    Args:
        data_matrix: A data array of size n_rois × n_frames.
        neuropil_pixel_indices: A list of arrays containing the pixel indices forming the neuropil ring
                                around each ROI.
        neuropil_pixel_count: An array indicating the number of neuropil pixels for each ROI.                   
    """
    num_cells = neuropil_fluorescence.shape[0]

    for cell_idx in prange(num_cells):
        cell_idx = np.int64(cell_idx)  # This is here to fix Numba's 'unsafe uint64 -> int64 cast warning.'

        neuropil_fluorescence[cell_idx] = (
            data_matrix[:, neuropil_pixel_indices[cell_idx]].sum(axis=1) / neuropil_pixel_count[cell_idx]
        )

    return neuropil_fluorescence


def extract_traces_from_masks(
    ops: dict[str, Any], cell_masks: List[tuple], neuropil_masks: List
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Computes fluorescence traces for each ROI and its corresponding neuropil region 
    from both channels, if available. This function is also used in drawroi.py. 
    
    Args:
        ops: The dictionary that stores the plane registration parameters.
        cell_masks: A tuple containing an array of flattened ROI pixel indices and a corresponding 
                    array of normalized weights to compute the ROI’s fluorescence trace.
        neuropil_masks: An array containing pixel indices of the neuropil surrounding the ROI.
    """
    batch_size = ops["batch_size"]
    height = ops["Ly"]
    width = ops["Lx"]

    with BinaryFile(height=height, width=width, file_path=ops["reg_file"]) as f:
        cell_fluorescence, neuropil_fluorescence = extract_traces(
            f_in=f, cell_masks=cell_masks, neuropil_masks=neuropil_masks, batch_size=batch_size
        )

    cell_fluorescence_channel_2 = []
    neuropil_fluorescence_channel_2 = []

    if ops.get("reg_file_chan2"):
        with BinaryFile(height=height, width=width, file_path=ops["reg_file_chan2"]) as f:
            cell_fluorescence_channel_2, neuropil_fluorescence_channel_2 = extract_traces(
                f_in=f, cell_masks=cell_masks, neuropil_masks=neuropil_masks, batch_size=batch_size
            )

    return cell_fluorescence, neuropil_fluorescence, cell_fluorescence_channel_2, neuropil_fluorescence_channel_2


def extraction_wrapper(
    roi_statistics: List[dict[str, Any]],
    plane_number: int,
    frames: NDArray,
    frames_channel_2: NDArray | None = None,
    cell_masks: List[NDArray] | None = None,
    neuropil_masks: List[NDArray] | None = None,
    ops: dict[str, Any] | None = None,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """
    Main extraction function that creates the masks and computes fluorescence traces.

    This function generates cell and neuropil masks if not provided, extracts raw fluorescence traces 
    from the imaging frames, and computes the skewness and standard deviation on the signals after 
    subtracting the neuropil.

    roi_statistics: The dictionary that stores the statistics for regions of interest (ROIs), including cell masks.
    plane_number: The number (index) of the processed plane.
    frames_path: The path to the binary file that stores the registered plane frames for which to process the ROIs.
    frames_channel_2_path: Same as 'frames_path', but for the second functional channel, if the plane data contains
            data from two channels.
    ops: The dictionary that stores the plane registration parameters.
    """
    if ops is None:
        ops = generate_default_ops()

    timer = PrecisionTimer("s")
    n_frames, height, width = frames.shape
    batch_size = ops["batch_size"]
    neucoeff = ops["neucoeff"]

    if cell_masks is None:
        console.echo(f"Creating ROI masks for plane {plane_number}...", level=LogLevel.INFO)
        timer.reset()

        cell_masks, new_neuropil_masks = create_masks(
            roi_statistics=roi_statistics, height=height, width=width, ops=ops
        )

        if neuropil_masks is None:
            neuropil_masks = new_neuropil_masks

        console.echo(
            f"Plane {plane_number} ROI masks: created. Time taken: {timer.elapsed} seconds.", level=LogLevel.SUCCESS
        )

    cell_fluorescence, neuropil_fluorescence = extract_traces(
        f_in=frames,
        plane_number=plane_number,
        cell_masks=cell_masks,
        neuropil_masks=neuropil_masks,
        batch_size=batch_size,
    )

    cell_fluorescence_channel_2 = []
    neuropil_fluorescence_channel_2 = []

    if frames_channel_2:
        cell_fluorescence_channel_2, neuropil_fluorescence_channel_2 = extract_traces(
            f_in=frames_channel_2,
            plane_number=plane_number,
            cell_masks=cell_masks,
            neuropil_masks=neuropil_masks,
            batch_size=batch_size,
        )

    corrected = cell_fluorescence - neucoeff * neuropil_fluorescence
    skew_values = stats.skew(corrected, axis=1)
    std_values = np.std(corrected, axis=1)

    for i, (roi_stat, skew, std) in enumerate(zip(roi_statistics, skew_values, std_values)):
        roi_stat.update({"skew": skew, "std": std})
        if neuropil_masks is not None:
            roi_stat["neuropil_mask"] = neuropil_masks[i]

    return (
        roi_statistics,
        cell_fluorescence,
        neuropil_fluorescence,
        cell_fluorescence_channel_2,
        neuropil_fluorescence_channel_2,
    )


def extract_traces(f_in, plane_number: int, cell_masks, neuropil_masks, batch_size: int = 500, session_id: str = ""):
    """Extracts activity from f_in using masks in stat and neuropil_masks

    computes fluorescence F as sum of pixels weighted by "lam"
    computes neuropil fluorescence Fneu as sum of pixels in neuropil_masks

    data is from reg_file ops["batch_size"] by pixels:
    .. code-block:: python
        F[n] = data[:, stat[n]["ipix"]] @ stat[n]["lam"]
        Fneu = neuropil_masks @ data.T

    Parameters
    ----------------

    f_in : np.ndarray or io.BinaryFile object
        size n_frames, Ly, Lx


    cell_masks : list
        each is a tuple where first element are cell pixels (flattened), and
        second element are pixel weights normalized to sum 1 (lam)

    neuropil_masks : list
        each element is neuropil pixels in (Ly*Lx) coordinates
        GOING TO BE DEPRECATED: size [ncells x npixels] where weights of each mask are elements

    batch_size : int
        function will run with at most batch size of 1000

    Returns:
    ----------------
    F : float, 2D array
        size [ROIs x time]

    Fneu : float, 2D array
        size [ROIs x time]

    ops : dictionary

    """
    if session_id == "":
        console.echo(f"Extracting ROI fluorescence data for plane {plane_number}...", level=LogLevel.INFO)
    else:
        console.echo(f"Extracting ROI fluorescence data for session {session_id}...", level=LogLevel.INFO)

    timer = PrecisionTimer("s")
    timer.reset()
    n_frames, Ly, Lx = f_in.shape
    batch_size = min(batch_size, 1000)
    ncells = len(cell_masks)

    F = np.zeros((ncells, n_frames), np.float32)
    Fneu = np.zeros((ncells, n_frames), np.float32)

    batch_size = int(batch_size)

    cell_ipix, cell_lam = List(), List()
    [cell_ipix.append(cell_mask[0].astype(np.int64)) for cell_mask in cell_masks]
    [cell_lam.append(cell_mask[1].astype(np.float32)) for cell_mask in cell_masks]

    if neuropil_masks is not None:
        neuropil_ipix = List()
        if isinstance(neuropil_masks, np.ndarray) and neuropil_masks.shape[1] == Ly * Lx:
            [neuropil_ipix.append(np.nonzero(neuropil_mask)[0].astype(np.int64)) for neuropil_mask in neuropil_masks]
        else:
            [neuropil_ipix.append(neuropil_mask.astype(np.int64)) for neuropil_mask in neuropil_masks]
        neuropil_npix = np.array([len(neuropil_ipixi) for neuropil_ipixi in neuropil_ipix]).astype(np.float32)
    else:
        neuropil_ipix = None

    ix = 0
    for k in np.arange(0, n_frames, batch_size):
        data = f_in[k : min(k + batch_size, n_frames)].astype("float32")
        nimg = data.shape[0]
        if nimg == 0:
            break
        inds = ix + np.arange(0, nimg, 1, int)
        data = np.reshape(data, (nimg, -1)).astype(np.float32)
        Fi = np.zeros((ncells, data.shape[0]), np.float32)

        # Extract traces and neuropil
        F[:, inds] = matmul_traces(
            cell_fluorescence=Fi, data_matrix=data, cell_pixel_indices=cell_ipix, lambda_weights=cell_lam
        )
        if neuropil_ipix is not None:
            Fneu[:, inds] = matmul_neuropil(
                neuropil_fluorescence=Fi,
                data_matrix=data,
                neuropil_pixel_indices=neuropil_ipix,
                neuropil_pixel_count=neuropil_npix,
            )

        ix += nimg

    if session_id == "":
        message = (
            f"Plane {plane_number} ROI fluorescence: extracted from {ncells} ROIs in {n_frames} frames. "
            f"Time taken: {timer.elapsed} seconds."
        )
    else:
        message = (
            f"Session {session_id} ROI fluorescence: extracted from {ncells} ROIs in {n_frames} frames. "
            f"Time taken: {timer.elapsed} seconds."
        )
    console.echo(message=message, level=LogLevel.SUCCESS)
    return F, Fneu


def enhanced_mean_image(ops):
    """Computes enhanced mean image and adds it to ops

    Median filters ops["meanImg"] with 4*diameter in 2D and subtracts and
    divides by this median-filtered image to return a high-pass filtered
    image ops["meanImgE"]

    Parameters
    ----------
    ops : dictionary
        uses "meanImg", "aspect", "spatscale_pix", "yrange" and "xrange"

    Returns:
    -------
        ops : dictionary
            "meanImgE" field added

    """
    I = ops["meanImg"].astype(np.float32)
    if "spatscale_pix" not in ops:
        if isinstance(ops["diameter"], int):
            diameter = np.array([ops["diameter"], ops["diameter"]])
        else:
            diameter = np.array(ops["diameter"])
        if diameter[0] == 0:
            diameter[:] = 12
        ops["spatscale_pix"] = diameter[1]
        ops["aspect"] = diameter[0] / diameter[1]

    diameter = 4 * np.ceil(np.array([ops["spatscale_pix"] * ops["aspect"], ops["spatscale_pix"]])) + 1
    diameter = diameter.flatten().astype(np.int64)
    Imed = signal.medfilt2d(I, [diameter[0], diameter[1]])
    I = I - Imed
    Idiv = signal.medfilt2d(np.absolute(I), [diameter[0], diameter[1]])
    I = I / (1e-10 + Idiv)
    mimg1 = -6
    mimg99 = 6
    mimg0 = I

    mimg0 = mimg0[ops["yrange"][0] : ops["yrange"][1], ops["xrange"][0] : ops["xrange"][1]]
    mimg0 = (mimg0 - mimg1) / (mimg99 - mimg1)
    mimg0 = np.maximum(0, np.minimum(1, mimg0))
    mimg = mimg0.min() * np.ones((ops["Ly"], ops["Lx"]), np.float32)
    mimg[ops["yrange"][0] : ops["yrange"][1], ops["xrange"][0] : ops["xrange"][1]] = mimg0
    ops["meanImgE"] = mimg
    return ops
