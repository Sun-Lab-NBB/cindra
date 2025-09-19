"""This module contains the high-level API for the single-day suite2p processing pipeline."""

import os
import shutil
from typing import Any
from pathlib import Path
from datetime import datetime
import contextlib

import numba
import numpy as np
from natsort import natsorted
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import LogLevel, console

from . import io, detection, extraction, registration, classification
from .version import version, sl_version, python_version
from .io.binary import BinaryFile
from .configuration import SingleDayS2PConfiguration, generate_default_ops

# Defines constants used in this module
# Frame binning parameters. Determine how to bin the frames for steps that support splitting the processed movie into
# batches of frames.
_MINIMUM_FRAMES_PER_BIN = 2000  # Minimum number of frames in each batch.
_MAXIMUM_FRAMES_PER_BIN = 5000  # Maximum number of frames in each batch.
_MAXIMUM_HEIGHT_PER_BIN = 700  # The maximum height of each image that can be binned using the maximum bin size.
_MAXIMUM_WIDTH_PER_BIN = 700  # The maximum width of each image that can be binned using the maximum bin size.

# Specifies the maximum number of channels in processed movie images.
_MAXIMUM_SUPPORTED_CHANNELS = 2  # At most two channels: red and green.

# Movie processing thresholds.
_MINIMUM_PROCESSING_FRAMES = 50  # The minimum number of frames in the processed movie to allow processing.
_RECOMMENDED_PROCESSING_FRAMES = 200  # The recommended number of frames in the processed movie.

_MINIMUM_REGISTRATION_METRIC_FRAMES = 1500  # The minimum number of frames required to compute registration metrics.


def _register_plane(
    ops: dict[str, Any],
    plane_number: int,
    frames_path: str,
    raw_frames_path: str | None = None,
    frames_channel_2_path: str | None = None,
    raw_frames_channel_2_path: str | None = None,
) -> dict[str, Any]:
    """Registers (motion-corrects) the frames acquired at the target imaging plane of the processed movie.

    The registration process involves computing rigid and non-rigid offsets (deformation) to register all frames inside
    the processed movie to the reference image. This corrects motion in the X and Y directions and is a prerequisite
    for all other pipeline steps.

    Notes:
        This process does not correct Z-drift. For the best results, z-drift correction should be performed 'online',
        as the movie is being acquired.

    Args:
        ops: The dictionary that stores the plane registration parameters.
        plane_number: The number (index) of the processed plane.
        frames_path: The path to the binary file that stores registered or unregistered frames to process. During
            processing, the contents of the file are overwritten with registered frames (frames with registration
            offsets applied).
        raw_frames_path: Same as 'raw_frames_path', but the data in this file is not overwritten during processing,
            keeping it 'raw'.
        frames_channel_2_path: Same as 'frames_path', but for the second functional channel, if plane data contains
            data from two channels.
        raw_frames_channel_2_path: Same as 'raw_frames_channel_2_path', but for the second functional channel.

    Returns:
        The input 'ops' dictionary, modified to include additional parameters and data generated during registration
        runtime.

    """
    timer = PrecisionTimer("s")

    # Memory-maps the necessary binary files.
    n_frames, height, width = ops["nframes"], ops["Ly"], ops["Lx"]
    null = contextlib.nullcontext()
    with (
        BinaryFile(height=height, width=width, file_path=raw_frames_path, frame_number=n_frames)
        if raw_frames_path
        else null as raw_frames,
        BinaryFile(height=height, width=width, file_path=frames_path, frame_number=n_frames) as frames,
        BinaryFile(height=height, width=width, file_path=raw_frames_channel_2_path, frame_number=n_frames)
        if raw_frames_channel_2_path
        else null as raw_frames_channel_2,
        BinaryFile(height=height, width=width, file_path=frames_channel_2_path, frame_number=n_frames)
        if frames_channel_2_path
        else null as frames_channel_2,
    ):
        # Skips applying bidiphase correction if frames have already been bidiphase-corrected.
        if raw_frames is None and ops["do_bidiphase"] and ops["bidiphase"] != 0:
            ops["bidi_corrected"] = True

        # First registration step:
        message = f"Running plane {plane_number} registration step one..."
        console.echo(message=message, level=LogLevel.INFO)
        timer.reset()

        # Determines whether to compute the reference image or use an existing reference image (if available). During
        # registration, each frame is registered to this reference image.
        reference_image = ops["refImg"] if "refImg" in ops and ops.get("force_ref_img", False) else None

        # Determines whether the registration should be performed using the first or the second functional channel if
        # the processed data contains two functional channels.
        align_by_channel_2 = ops["functional_chan"] != ops["align_by_chan"]

        # Runs the registration pipeline.
        registration_outputs = registration.registration_wrapper(
            frames,
            plane_number=plane_number,
            f_raw=raw_frames,
            f_reg_chan2=frames_channel_2,
            f_raw_chan2=raw_frames_channel_2,
            refImg=reference_image,
            align_by_chan2=align_by_channel_2,
            ops=ops,
        )

        # Adds registration outputs to the plane 'ops' file.
        ops = registration.save_registration_outputs_to_ops(registration_outputs, ops)

        # Computes and adds the enhanced mean image to the plane 'ops' file.
        ops["meanImgE"] = registration.compute_enhanced_mean_image(ops["meanImg"].astype(np.float32), ops)

        # Adds registration time to the plane 'ops' file.
        ops["timing"]["registration"] = timer.elapsed

        message = (
            f"Plane {plane_number} registration step one: complete. Time taken: {ops['timing']['registration']} "
            f"seconds."
        )
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Caches the registration output to disk by overwriting the plane ops file.
        if ops.get("ops_path"):
            np.save(ops["ops_path"], ops)

        # If necessary, carries out the second registration step.
        if ops["two_step_registration"] and ops["keep_movie_raw"]:
            message = f"Running plane {plane_number} second-step registration..."
            console.echo(message=message, level=LogLevel.INFO)

            # Resets the timer for the second step.
            timer.reset()

            message = (
                f"Generating plane {plane_number} mean image that excludes bad frames detected during first "
                f"registration step..."
            )
            console.echo(message=message, level=LogLevel.INFO)

            # Bins all available frames into groups of 1000 frames.
            n_samples = min(frames.shape[0], 1000)
            indices = np.linspace(0, frames.shape[0], 1 + n_samples).astype(np.int64)[:-1]

            # Resolves the reference image, depending on the alignment channel setting.
            if align_by_channel_2 and frames_channel_2 is not None:
                reference_image = frames_channel_2[indices].astype(np.float32).mean(axis=0)
            else:
                reference_image = frames[indices].astype(np.float32).mean(axis=0)

            # Runs the registration pipeline.
            registration.registration_wrapper(
                frames,
                plane_number=plane_number,
                f_raw=None,
                f_reg_chan2=frames_channel_2,
                f_raw_chan2=None,
                refImg=reference_image,
                align_by_chan2=align_by_channel_2,
                ops=ops,
            )

            # Adds the second registration step time to the plane 'ops' file.
            ops["timing"]["two_step_registration"] = timer.elapsed

            # Caches the registration output to disk by overwriting the plane ops file.
            if ops.get("ops_path"):
                np.save(ops["ops_path"], ops)

            message = (
                f"Plane {plane_number} second-step registration: complete. Time taken: "
                f"{ops['timing']['two_step_registration']} seconds."
            )
            console.echo(message=message, level=LogLevel.SUCCESS)

    # Returns the modified ops to caller.
    return ops


def _compute_registration_metrics(ops: dict[str, Any], plane_number: int, frames_path: str) -> dict[str, Any]:
    """Computes frame registration (motion-correction) quality metrics for the target imaging plane of the processed
    movie.

    Notes:
        This step is optional. The metrics are designed for human operators to assess the quality of registration and
        should be used in conjunction with the visual inspection of registered movies. Skipping this step altogether may
        result in a significant processing time reduction for some runtimes.

    Args:
        ops: The dictionary that stores the plane registration quality metrics computation parameters.
        plane_number: The number (index) of the processed plane.
        frames_path: The path to the binary file that stores the registered plane frames to evaluate.

    Returns:
        The input 'ops' dictionary, modified to include the calculated registration metrics. Specifically, the
        dictionary is expanded to include the "regPC", "tPC", and "regDX" fields, in addition to the
        'registration_metrics' subfield stored under the 'timing' field.
    """
    timer = PrecisionTimer("s")

    message = f"Computing plane {plane_number} registration quality metrics..."
    console.echo(message=message, level=LogLevel.INFO)
    timer.reset()

    # Memory-maps the necessary binary files.
    n_frames, height, width = ops["nframes"], ops["Ly"], ops["Lx"]
    with (
        BinaryFile(height=height, width=width, file_path=frames_path, frame_number=n_frames) as frames,
    ):
        # Determines how to bin the processed movie to optimize memory overhead.
        n_frames, height, width = frames.shape
        n_samples = min(
            _MINIMUM_FRAMES_PER_BIN
            if n_frames < _MAXIMUM_FRAMES_PER_BIN or height > _MAXIMUM_HEIGHT_PER_BIN or width > _MAXIMUM_WIDTH_PER_BIN
            else _MAXIMUM_FRAMES_PER_BIN,
            n_frames,
        )

        # Bins the processed movie according to the binning criteria calculated above
        indices = np.linspace(0, n_frames - 1, n_samples).astype("int")
        movie = frames[indices]
        movie = movie[:, ops["yrange"][0] : ops["yrange"][-1], ops["xrange"][0] : ops["xrange"][-1]]

        # Runs the registration evaluation pipeline.
        ops = registration.get_pc_metrics(movie, ops, plane_number=plane_number)

        reg_metrics_time = timer.elapsed
        message = (
            f"Plane {plane_number} registration quality metrics: computed. Time taken: {reg_metrics_time} seconds."
        )
        console.echo(message=message, level=LogLevel.SUCCESS)

        # Adds processing time information to ops
        ops["timing"]["registration_metrics"] = reg_metrics_time

    # Returns the modified ops to caller
    return ops


def _process_rois(
    ops: dict[str, Any], plane_number: int, frames_path: str, frames_channel_2_path: str | None = None
) -> dict[str, Any]:
    """Detects and processes ROI (cell) activity data for the target plane.

    Specifically, carries out ROI discovery, trace extraction, and spike deconvolution for the target plane.

    Args:
        ops: The dictionary that stores the plane roi processing parameters.
        plane_number: The number (index) of the processed plane.
        frames_path: The path to the binary file that stores the registered plane frames for which to process the ROIs.
        frames_channel_2_path: Same as 'frames_path', but for the second functional channel, if the plane data contains
            data from two channels.

    Returns:
        The input 'ops' dictionary, modified to include the processed ROI information. Also, caches extracted ROI data
        to disk as a series of .npy files (stats.npy, spks.npy, etc.).
    """
    timer = PrecisionTimer("s")

    # Selects the classifier file, based on the processing configuration
    ops_classifier_file = ops.get("classifier_path")
    builtin_classifier_file = classification.builtin_classfile
    user_classifier_file = classification.user_classfile
    if ops_classifier_file:
        message = f"Applying target classifier {Path(ops_classifier_file).name} to plane {plane_number}..."
        console.echo(message=message, level=LogLevel.INFO)
        classifier_file = ops_classifier_file
    elif ops["use_builtin_classifier"] or not user_classifier_file.is_file():
        message = f"Applying builtin classifier {builtin_classifier_file.name} to plane {plane_number}..."
        console.echo(message=message, level=LogLevel.INFO)
        classifier_file = builtin_classifier_file
    else:
        message = f"Applying default classifier {user_classifier_file.name} to plane {plane_number}..."
        console.echo(message=message, level=LogLevel.INFO)
        classifier_file = user_classifier_file

    # Memory-maps the necessary binary files.
    n_frames, height, width = ops["nframes"], ops["Ly"], ops["Lx"]
    null = contextlib.nullcontext()
    with (
        BinaryFile(height=height, width=width, file_path=frames_path, frame_number=n_frames) as frames,
        BinaryFile(height=height, width=width, file_path=frames_channel_2_path, frame_number=n_frames)
        if frames_channel_2_path
        else null as frames_channel_2,
    ):
        # Cell ROI detection:
        message = f"Detecting plane {plane_number} ROIs (cells)..."
        console.echo(message=message, level=LogLevel.INFO)
        timer.reset()

        ops, roi_statistics = detection.detection_wrapper(
            frames, plane_number=plane_number, ops=ops, classfile=classifier_file
        )
        ops["timing"]["detection"] = timer.elapsed
        message = f"Plane {plane_number} ROIs: detected. Time taken: {ops['timing']['detection']} seconds."
        console.echo(message=message, level=LogLevel.SUCCESS)

        # If ROIs (cells) were discovered or provided, extracts the fluorescence for each cell and the surrounding
        # neuropil.
        if len(roi_statistics) > 0:
            # ROI (cell fluorescence) extraction
            message = f"Extracting plane {plane_number} ROI fluorescence..."
            console.echo(message=message, level=LogLevel.INFO)
            timer.reset()

            # Extracts cell and neuropil fluorescence traces
            (
                roi_statistics,
                cell_fluorescence,
                neuropil_fluorescence,
                cell_fluorescence_channel_2,
                neuropil_fluorescence_channel_2,
            ) = extraction.extraction_wrapper(
                roi_statistics,
                plane_number=plane_number,
                f_reg=frames,
                f_reg_chan2=frames_channel_2,
                ops=ops,
            )

            # Caches the fluorescence extraction output to disk by overwriting the plane ops file.
            if ops.get("ops_path"):
                np.save(ops["ops_path"], ops)

            ops["timing"]["extraction"] = timer.elapsed

            # ROI classification (filtering)
            message = f"Filtering out non-cell plane {plane_number} ROIs..."
            console.echo(message=message, level=LogLevel.INFO)
            timer.reset()

            # Only applies cell classifier if at least one ROI was detected
            if len(roi_statistics):
                iscell = classification.classify(stat=roi_statistics, classfile=classifier_file)
            else:
                iscell = np.zeros((0, 2))

            ops["timing"]["classification"] = timer.elapsed
            message = f"Plane {plane_number} ROIs: filtered. Time taken: {ops['timing']['classification']} seconds."
            console.echo(message=message, level=LogLevel.SUCCESS)

            # Cell activity spike deconvolution
            if ops.get("spikedetect", True):
                message = f"Processing plane {plane_number} activity spikes..."
                console.echo(message=message, level=LogLevel.INFO)
                timer.reset()

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

                # Extracts the cell fluorescence spikes using the OASIS algorithm.
                spikes = extraction.oasis(F=dff, batch_size=ops["batch_size"], tau=ops["tau"], fs=ops["fs"])
                ops["timing"]["deconvolution"] = timer.elapsed

                message = (
                    f"Plane {plane_number} spikes: computed. Time taken: {ops['timing']['deconvolution']} seconds."
                )
                console.echo(message=message, level=LogLevel.SUCCESS)

            else:
                message = (
                    f"Skipping plane {plane_number} spike deconvolution, as it is disabled via the 'spikedetect' "
                    f"configuration parameter."
                )
                console.echo(message=message, level=LogLevel.WARNING)
                spikes = np.zeros_like(cell_fluorescence)

            # Saves pipeline output to disk as .npy files.
            fpath = Path(ops["save_path"])
            if ops.get("save_path"):
                np.save(fpath.joinpath("stat.npy"), roi_statistics)
                np.save(fpath.joinpath("F.npy"), cell_fluorescence)
                np.save(fpath.joinpath("Fneu.npy"), neuropil_fluorescence)
                np.save(fpath.joinpath("Fsub.npy"), dff)
                np.save(fpath.joinpath("iscell.npy"), iscell)
                np.save(fpath.joinpath("spks.npy"), spikes)

                # If the data contains two functional channels, also saves the data for the second channel.
                if "meanImg_chan2" in ops:
                    np.save(fpath.joinpath("F_chan2.npy"), cell_fluorescence_channel_2)
                    np.save(fpath.joinpath("Fneu_chan2.npy"), neuropil_fluorescence_channel_2)

            # If requested, also saves the data as a .mat (matlab-compatible) file.
            if ops.get("save_mat"):
                roi_statistics = np.load(fpath.joinpath("stat.npy"), allow_pickle=True)
                iscell = np.load(fpath.joinpath("iscell.npy"))
                redcell = (
                    np.load(fpath.joinpath("redcell.npy"))
                    if ops["nchannels"] == _MAXIMUM_SUPPORTED_CHANNELS
                    else np.zeros_like(iscell)
                )
                io.save_matlab(
                    ops,
                    roi_statistics,
                    cell_fluorescence,
                    neuropil_fluorescence,
                    spikes,
                    iscell,
                    redcell,
                    cell_fluorescence_channel_2,
                    neuropil_fluorescence_channel_2,
                )
        else:
            message = f"No ROIs found for plane {plane_number}."
            console.echo(message=message, level=LogLevel.WARNING)

    # Returns the updated ops dictionary to caller
    return ops


def _resolve_plane_paths(ops: dict[str, Any]) -> tuple[list[Path], list[Path], Path]:
    """Resolves and returns the paths to all output 'plane' folders and their respective 'ops.npy' files.

    This service function is used by other processing functions to resolve the paths to all files used during
    suite2p single-day processing.

    Args:
        ops: The dictionary that stores single-day processing parameters, including the plane save directory.

    Returns:
        A tuple of 3 elements. The first element is a list of paths to all 'plane' folders, the second is a list of
        paths to all plane 'ops.npy' files, and the third is the path to the root save folder.
    """
    save_folder = Path(ops["save_path0"]).joinpath(ops["save_folder"])
    save_folder.mkdir(parents=True, exist_ok=True)
    plane_folders = natsorted([file for file in save_folder.glob("*") if file.is_dir() and file.name[:5] == "plane"])
    ops_paths = [folder.joinpath("ops.npy") for folder in plane_folders]
    return plane_folders, ops_paths, save_folder


def resolve_ops(ops: dict[str, Any], db: dict[str, Any]) -> Path:
    """Generates the output directory hierarchy and the main 'ops.npy' file for the single-day suite2p pipeline, using
    the configuration parameters from ops and db.

    This function should be used before the first step (binary file creation) of the single-day suite2p pipeline to
    generate the necessary directories and files used by all further pipeline steps. During step-wise pipeline
    execution, this function can also be used between steps to update the 'ops.npy' file with new runtime parameters.

    Notes:
        All single-day pipeline functions require the path generated by this function as the 'ops_path' argument.

        If both 'ops' and 'db' do not contain some of the expected parameters, they are set using
        the 'default' dictionary generated using the SingleDayS2PConfiguration configuration class.

    Args:
        ops: A dictionary that contains the single-day configuration parameters.
        db: An optional dictionary that contains the same keys as 'ops'. Values from this dictionary are used to
            override the matching keys in the 'ops' dictionary.

    Returns:
        The path to the generated 'ops.npy' file.
    """
    # Builds up the 'ops' dictionary. Specifically, first fills the dictionary with the 'default' keys. Then overwrites
    # all default keys with keys from the input 'ops' dictionary. Finally, overwrites any keys from the input 'db'
    # dictionary with values from 'db'. This way, there is the following order of precedence: db > ops > default.
    ops = {**generate_default_ops(as_dict=True), **ops, **db}

    # Adjusts the aspect ratio for the processed data, if necessary. This is only used in gui, so may or may not be
    # important for certain processing runtimes.
    if isinstance(ops["diameter"], list) and len(ops["diameter"]) > 1 and ops["aspect"] == 1.0:
        ops["aspect"] = ops["diameter"][0] / ops["diameter"][1]

    # If necessary, resolves the output (save folder) path, based on the input data path.
    if "save_path0" not in ops or len(ops["save_path0"]) == 0:
        ops["save_path0"] = ops["data_path"][0]

    # Ensures that the root save folder name is provided via ops. Defaults to 'suite2p' if not provided.
    if "save_folder" not in ops or len(ops["save_folder"]) == 0:
        ops["save_folder"] = "suite2p"

    # Ensures the root save directory exists, creating it, if necessary.
    save_folder = Path(ops["save_path0"]).joinpath(ops["save_folder"])
    save_folder.mkdir(parents=True, exist_ok=True)

    # Saves the updated 'ops' file to the root output folder. During the rest of the runtime, this file is loaded
    # back into memory for various processing steps.
    ops_path = save_folder.joinpath("ops.npy")

    # Actualizes version information to the generated ops.npy file.
    ops["base_suite2p_version"] = version
    ops["sl_suite2p_version"] = sl_version
    ops["python_version"] = python_version

    # If the user does not specify the maximum parallel worker limit, sets it based on the number of available
    # CPU cores
    if ops["parallel_workers"] < 1:
        # noinspection PyUnresolvedReferences
        ops["parallel_workers"] = os.process_cpu_count()

    # Caches the generated ops dictionary to disk as ops.npy
    np.save(ops_path, ops, allow_pickle=True)

    # Also generates a 'single_day_ss2p_configuration.yaml' file that stores the same configuration data as the
    # ops.npy file. This ensures each runtime always has a human-readable configuration file that does not rely on
    # unpickling (which is inherently unsafe).
    config = SingleDayS2PConfiguration.from_ops(ops_dict=ops)
    config.to_config(file_path=save_folder.joinpath("single_day_ss2p_configuration.yaml"))

    # Returns the path to the generated 'ops.npy' file.
    return ops_path


def run_s2p(ops_path: Path) -> None:
    """Executes the single-day suite2p processing pipeline using the parameters stored in the target ops.npy file.

    This function sequentially calls all steps of the suite2p single-day processing pipeline, converting raw data
    frames into extracted cell fluorescence data.

    Notes:
        This function works as a high-level API wrapper over the three pipeline step functions: 'resolve_binaries',
        'process_plane', and 'combine_planes'.

    Args:
        ops_path: The path to the ops.npy file that stores the single-day suite2p processing parameters. Compatible
            ops.npy files are generated by the resolve_ops() function.
    """
    # Guards against invalid inputs.
    if not ops_path.exists() or not ops_path.is_file() or ops_path.suffix != ".npy":
        message = (
            f"Unable to run the single-day suite2p pipeline, as the 'ops.npy' file does not exist at the specified "
            f"path {ops_path}."
        )
        console.error(message=message, error=FileNotFoundError)

    # Instantiates and resets the execution timer
    timer = PrecisionTimer("s")
    timer.reset()

    console.echo(message="Initializing single-day suite2p runtime...", level=LogLevel.INFO)

    # Loads the runtime configuration parameters
    ops = np.load(ops_path, allow_pickle=True).item()

    # Step 1: Ensures the data to analyze is stored as one or more 'plane' folders. Each plane should contain the
    # data.bin file that stores the frame data and the ops.npy file that stores the processing settings.
    resolve_binaries(ops_path=ops_path)

    save_folder = Path(ops["save_path0"]).joinpath(ops["save_folder"])
    plane_folders = natsorted([file for file in save_folder.glob("*") if file.is_dir() and file.name[:5] == "plane"])
    ops_paths = [folder.joinpath("ops.npy") for folder in plane_folders]

    # Step 2: Processes each plane. This applies the suite2p processing pipeline to each of the planes resolved during
    # the first step.
    for index in range(len(ops_paths)):
        process_plane(plane_index=index, ops_path=ops_path)

    # Step 3: Applies post-plane-processing transformations, such as generating the 'combined' folder used during
    # multi-day processing.
    combine_planes(ops_path=ops_path)

    message = f"Single-day suite2p runtime: Complete. Total time: {timer.elapsed} seconds."
    console.echo(message=message, level=LogLevel.SUCCESS)


def resolve_binaries(ops_path: Path) -> None:
    """Ensures that the data.bin files and settings ops.npy dictionaries exist for all imaging planes in the processed
    movie.

    Notes:
        If necessary, the function converts the input data to the suite2p binary (.bin) data format. Following the
        conversion, most other processing steps utilize memory-mapping to reduce the RAM overhead during runtime.

    Args:
        ops_path: The path to the ops.npy file that stores the single-day suite2p processing parameters. Compatible
            ops.npy files are generated by the resolve_ops() function.
    """
    # Guards against invalid inputs.
    if not ops_path.exists() or not ops_path.is_file() or ops_path.suffix != ".npy":
        message = (
            f"Unable to run the single-day suite2p pipeline, as the 'ops.npy' file does not exist at the specified "
            f"path {ops_path}."
        )
        console.error(message=message, error=FileNotFoundError)

    # Initializes the execution timer
    timer = PrecisionTimer("s")
    timer.reset()

    # Loads the 'ops' dictionary from the specified storage file
    ops = np.load(ops_path, allow_pickle=True).item()

    # Determines the path to all existing plane folders and ops.npy files (if any)
    plane_folders, ops_paths, save_folder = _resolve_plane_paths(ops)

    # If at least one plane folder exists and the input format is 'binary', uses existing binary files stored under
    # each available plane folder.
    if len(plane_folders) > 0 and ("input_format" in ops and ops["input_format"] == "binary"):
        # Not sure where this is used in the current version, searching the codebase does not return any use outside
        # a single reference in the gui code. The purpose of this transformation is likely to make the code below
        # (setting Ly and Lx) work regardless of whether Lys/Lxs are lists or integers.
        if isinstance(ops["Lys"], int):
            ops["Lys"] = [ops["Lys"]]
            ops["Lxs"] = [ops["Lxs"]]

        # Loops over plane folders and 'ops.npy' files. Reconfigures and overwrites each existing .npy file to
        # facilitate further data (re)processing.
        for i, (folder, ops_file) in enumerate(zip(plane_folders, ops_paths, strict=False)):
            ops["bin_file"] = folder.joinpath("data.bin")  # Path to the existing binary file.

            # Computes plane dimensions.
            ops["Ly"] = ops["Lys"][i]
            ops["Lx"] = ops["Lxs"][i]

            # Uses plane dimension and a static assumption that each pixel uses 16-bit encoding to determine the
            # number of frames stored inside the plane binary file.
            n_bytes_read = np.int64(2 * ops["Ly"] * ops["Lx"])
            ops["nframes"] = Path(ops["bin_file"]).stat().st_size // n_bytes_read
            np.save(ops_file, ops)  # Overwrites the ops.npy with modified settings

        files_found_flag = True

    # Otherwise, if the input format is not 'binary', tries to find binaries and ops for each plane. Unlike with the
    # previous case, where input data is in binary and, therefore, binaries have to exist, here the binaries might
    # not be available for all planes.
    elif len(plane_folders) > 0:
        # Ensures that both ops.npy setting files and binary files are available for each plane and, if so, sets the
        # file discovery flag to true.
        ops_found_flag = all(ops_paths)
        binaries_found_flag = all(
            folder.joinpath("data_raw.bin").is_file() or folder.joinpath("data.bin").is_file()
            for folder in plane_folders
        )
        files_found_flag = ops_found_flag and binaries_found_flag

    # If there are no plane folders or the code above was not able to find binary files and ops.npy files for all
    # available planes, sets the file discovery flag to false. In turn, this triggers the binary file recreation below.
    else:
        files_found_flag = False

    # If binary and ops.npy files already exist, removes all processed data files from each plane folder
    # before rerunning registration on the already available binary files.
    if files_found_flag:
        message = (
            f"Found existing binaries (.bin files) and ops (ops.npy files) inside {len(plane_folders)} "
            f"available plane folders."
        )
        console.echo(message=message, level=LogLevel.SUCCESS)

        message = "Removing previous cell detection and fluorescence extraction files, if present."
        console.echo(message=message, level=LogLevel.INFO)

        files_to_remove = (
            "stat.npy",
            "F.npy",
            "Fneu.npy",
            "Fsub.npy",
            "F_chan2.npy",
            "Fneu_chan2.npy",
            "iscell.npy",
            "redcell.npy",
        )

        # Loops over each plane folder and, for each, removes any existing files from the remove tuple.
        for path in plane_folders:
            for file in files_to_remove:
                path.joinpath(file).unlink(missing_ok=True)

    # Otherwise, regenerates all binaries from the source data.
    else:
        # For mesoscope scan, manually overrides the input format field. This distinct step is likely due to mesoscope
        # scanning being implemented as an extension of the general .tiff processing method.
        if ops.get("mesoscan"):
            ops["input_format"] = "mesoscan"

        # Defaults to 'tif' input format if the explicit format is not provided.
        elif "input_format" not in ops:
            message = (
                "No explicit input data format specified via the 'input_format' field of the 'ops' dictionary. "
                "Substituting the default 'tif' format instead."
            )
            console.echo(message=message, level=LogLevel.WARNING)
            ops["input_format"] = "tiff"

        # Builds a dictionary of callable 'converter' functions. These functions are used to convert various supported
        # input formats to the binary format used during the rest of the processing pipeline. Note, each converter
        # function also writes the plane-specific ops.npy file to the output plane folder.
        convert_functions = {
            "mesoscan": io.mesoscan_to_binary,
            "raw": io.raw_to_binary,
            "tiff": io.tiff_to_binary,
        }

        # If input data matches one of the supported conversion formats, retrieves and calls the converter function
        # for that format
        if ops["input_format"] in convert_functions:
            ops0 = convert_functions[ops["input_format"]](ops.copy())

        # Otherwise, if the format is not supported, defaults to using the 'tiff' format.
        else:
            message = (
                f"The 'input_format' configuration (ops) dictionary field is set to an unsupported format "
                f"{ops['input_format']}. Only the following formats are supported: {convert_functions.keys()}. "
                f"Substituting the default 'tif' format instead."
            )
            console.echo(message=message, level=LogLevel.WARNING)
            ops0 = io.tiff_to_binary(ops.copy())

        # Rebuilds the plane_folders list to use in the printout below. This is done to account for any new plane
        # folders added as part of the processing
        plane_folders = natsorted(
            [file for file in save_folder.glob("*") if file.is_dir() and file.name[:5] == "plane"]
        )

        message = (
            f"Binaries: Created. Took {timer.elapsed} seconds and wrote {ops0['nframes']} frames per binary for "
            f"{len(plane_folders)} planes."
        )
        console.echo(message=message, level=LogLevel.SUCCESS)


def process_plane(ops_path: Path, plane_index: int) -> None:
    """Runs the single-day suite2p pipeline on the target imaging plane's data.

    Notes:
        This function can be parallelized to process multiple planes at the same time. Many processing steps executed
        by this function are also internally parallelized by numba. Depending on the execution context, this function
        may use up to 2 x plane binary file size of RAM and use multiple CPU cores for each plane. Processing multiple
        planes in parallel may therefore require considerable memory and CPU resources.

    Args:
        ops_path: The path to the ops.npy file used to store the suite2p processing parameters. Compatible ops.npy
            files are generated by the resolve_ops() function.
        plane_index: The index of the imaging plane to process.
    """
    # Guards against invalid inputs.
    if not ops_path.exists() or not ops_path.is_file() or ops_path.suffix != ".npy":
        message = (
            f"Unable to run the single-day suite2p pipeline, as the 'ops.npy' file does not exist at the specified "
            f"path {ops_path}."
        )
        console.error(message=message, error=FileNotFoundError)

    # Loads the 'ops' dictionary from the specified storage file and extracts the paths to all available plane folders
    ops: dict[str, Any]
    ops = np.load(ops_path, allow_pickle=True).item()
    plane_folders, ops_paths, _ = _resolve_plane_paths(ops)
    available_indices = range(len(plane_folders))

    # Ensures that the target plane index is valid
    if plane_index not in range(len(plane_folders)):
        message = (
            f"Unable to process the plane with index {plane_index}, as the index is not valid. Available "
            f"plane indices: {available_indices}."
        )
        console.error(message=message, error=IndexError)

    # Selects the specific plane to process based on the input index
    ops_path = ops_paths[plane_index]

    # Aborts the processing early if the input plane is the flyback plane.
    if plane_index in ops["ignore_flyback"]:
        console.echo(message=f"Skipping processing the flyback plane {plane_index}.", level=LogLevel.SUCCESS)
        return

    # Loads the plane-specific settings 'ops' file.
    plane_ops = np.load(ops_path, allow_pickle=True).item()

    # Replaces most plane processing settings with data from the input ops file. However, avoids overwriting the
    # data directories configuration. This allows flexibly (re)configuring the plane processing via the ops.npy file
    # and the 'resolve_ops' function. This code was modified to also include 'nplanes', 'nrois', and 'nchannels' as
    # non-editable fields, as this directly affects how binary files are read and organized.
    for key in generate_default_ops(as_dict=True):
        if (
            key
            not in [
                "data_path",
                "save_path0",
                "fast_disk",
                "save_folder",
                "subfolders",
                "nplanes",
                "nchannels",
                "nrois",
            ]
            and key in ops
        ):
            plane_ops[key] = ops[key]

    console.echo(f"Processing plane {plane_index}...", level=LogLevel.INFO)

    # Ensures that the 'ops' dictionary contains all necessary runtime parameters, filling any missing parameters with
    # default values. Also overwrites the processing date with the current data.
    ops = {
        **generate_default_ops(as_dict=True),
        **plane_ops,
        "date_processed": datetime.now().astimezone(),
    }
    if "timing" not in ops:
        ops["timing"] = {}

    # Configures the maximum number of cores this function is allowed to use when parallelizing processing steps.
    numba.set_num_threads(ops["parallel_workers"])

    # Ensures that the plane contains enough frames for the processing to work as expected and, if not, either
    # aborts or notifies the user about the unexpected behavior possibility.
    if ops["nframes"] < _MINIMUM_PROCESSING_FRAMES:
        message = (
            f"Unable to process plane {plane_index}. A plane must contain at least 50 frames to be processed, but "
            f"the input plane contains only {ops['nframes']} frames."
        )
        console.error(message=message, error=ValueError)

    if ops["nframes"] < _RECOMMENDED_PROCESSING_FRAMES:
        message = (
            f"The number of frames for plane {plane_index} is below 200, unexpected behavior may occur during "
            f"processing."
        )
        console.echo(message=message, level=LogLevel.WARNING)

    # Determines whether the target plane needs to be registered. If so, adjusts the configuration parameters to
    # support (re)running the registration process.
    if ops["do_registration"] > 0:
        # If the plane has not been registered or registration is forced, carries out registration
        if "refImg" not in ops or "yoff" not in ops or ops["do_registration"] > 1:
            message = (
                f"Plane {plane_index} registration: enabled. The plane either has not been registered or "
                f"re-registration was forced via the 'do_registration' configuration parameter."
            )
            console.echo(message=message, level=LogLevel.INFO)

            # If the frame is being forcibly re-registered, ensures previous registration offsets are removed from the
            # plane ops file.
            ops.pop("yoff", None)
            ops.pop("xoff", None)
            ops.pop("corrXY", None)
            run_registration = True

        # Otherwise, if the plane has already been registered and re-registration is not forced, skips registration
        else:
            console.echo(
                message=f"Plane {plane_index} registration: disabled. The plane is already registered.",
                level=LogLevel.INFO,
            )
            console.echo(message=f"Plane {plane_index} binary path: {ops['reg_file']}.", level=LogLevel.INFO)
            run_registration = False

    # If the user specifically disables plane registration, skips plane registration
    else:
        message = (
            f"Plane {plane_index} registration: disabled. Plane registration was disabled via the 'do_registration' "
            f"configuration parameter."
        )
        console.echo(message=message, level=LogLevel.INFO)
        console.echo(message=f"Plane {plane_index} binary path: {ops['reg_file']}.", level=LogLevel.INFO)
        run_registration = False

    # Determines whether suite2p is configured to preserve the unregistered (raw) binary files.
    raw_file_available = ops.get("keep_movie_raw") and "raw_file" in ops and Path(ops["raw_file"]).is_file()

    # Resolves the paths to registered and, if available, raw binary files for channel 1
    registered_file = ops["reg_file"]
    raw_file = ops.get("raw_file", 0) if raw_file_available else registered_file

    # Gets the number of frames in each binary file to use to initialize channel 2 files, if needed. This is only used
    # when processing data with two functional channels.
    if ops["nchannels"] > 1:
        registered_file_channel_2 = ops["reg_file_chan2"]
        raw_file_channel_2 = ops.get("raw_file_chan2", 0) if raw_file_available else registered_file_channel_2
    else:
        registered_file_channel_2 = registered_file
        raw_file_channel_2 = registered_file

    # Determines the shape and layout of each binary file
    n_frames, _height, _width = ops["nframes"], ops["Ly"], ops["Lx"]

    # Determines which binary files are available for this plane and resolves their paths. Previously, this step
    # constructed the BinaryFile instances. Now, each processing step resolves the necessary binary files as part of its
    # runtime to make it easier to parallelize the processing steps.
    two_channels = ops["nchannels"] > 1
    raw_frames_path = str(raw_file) if raw_file_available else None
    frames_path = str(registered_file)
    raw_frames_channel_2_path = str(raw_file_channel_2) if raw_file_available and two_channels else None
    frames_channel_2_path = str(registered_file_channel_2) if two_channels else None

    # Initializes processing timer for the plane
    timer = PrecisionTimer("s")
    timer.reset()

    # If plane registration is enabled, runs the registration pipeline
    if run_registration:
        ops = _register_plane(
            ops=ops,
            plane_number=plane_index,
            frames_path=frames_path,
            raw_frames_path=raw_frames_path,
            frames_channel_2_path=frames_channel_2_path,
            raw_frames_channel_2_path=raw_frames_channel_2_path,
        )

    # If the processed movie contains at least 1500 frames and registration metric computation is enabled, computes
    # registration quality metrics. Note, depending on the registration metric computation parameter, this step is
    # either tied to registration or executed independently of carrying out registration.
    if n_frames >= _MINIMUM_REGISTRATION_METRIC_FRAMES and (
        ops["compute_registration_metrics"] > 1 or (ops["compute_registration_metrics"] == 1 and run_registration)
    ):
        ops = _compute_registration_metrics(ops=ops, plane_number=plane_index, frames_path=frames_path)
    else:
        message = (
            f"Skipping computing plane {plane_index} registration quality metrics, as it is either disabled via the "
            f"'compute_registration_metrics' configuration parameter, is skipped because plane registration was also "
            f"skipped, or because the plane has less than 1500 frames (it has {n_frames} frames)."
        )
        console.echo(message=message, level=LogLevel.INFO)

    # If ROI (cell) segmentation is enabled, segments (detects) cell ROIs
    if ops.get("roidetect", True):
        ops = _process_rois(
            ops=ops, plane_number=plane_index, frames_path=frames_path, frames_channel_2_path=frames_channel_2_path
        )
    else:
        message = (
            f"Skipping plane {plane_index} cell detection, as it is disabled via the 'roidetect' "
            f"configuration parameter."
        )
        console.echo(message=message, level=LogLevel.WARNING)

    # Appends the overall plane processing time to the 'ops' file.
    ops["timing"]["total_plane_runtime"] = timer.elapsed

    # Caches plane processing results to disk
    if ops.get("ops_path"):
        np.save(ops["ops_path"], ops)

    # If suite2p runs on a machine with a fast processing disk and a slow storage disk and is configured to move
    # binary files after processing, moves the files to the storage disk.
    if ops.get("move_bin") and ops["save_path"] != ops["fast_disk"]:
        console.echo(message=f"Moving plane {plane_index} binary files to {ops['save_path']}...", level=LogLevel.INFO)

        # Registered binary file for channel 1
        shutil.move(ops["reg_file"], Path(ops["save_path"]).joinpath("data.bin"))

        # Registered binary file for channel 2
        if ops["nchannels"] > 1:
            shutil.move(ops["reg_file_chan2"], Path(ops["save_path"]).joinpath("data_chan2.bin"))

        # Raw binary file for channel 1
        if "raw_file" in ops:
            shutil.move(ops["raw_file"], Path(ops["save_path"]).joinpath("data_raw.bin"))

        # Raw binary file for channel 2
        if "raw_file" in ops and ops["nchannels"] > 1:
            shutil.move(ops["raw_file_chan2"], Path(ops["save_path"]).joinpath("data_chan2_raw.bin"))

    # Alternatively, if suite2p is configured to delete binary files after processing, removes the necessary files
    elif ops.get("delete_bin"):
        console.echo(message=f"Deleting plane {plane_index} binary files...", level=LogLevel.INFO)

        # Registered binary file for channel 1
        Path(ops["reg_file"]).unlink()

        # Registered binary file for channel 2
        if ops["nchannels"] > 1:
            Path(ops["reg_file_chan2"]).unlink()

        # Raw binary file for channel 1
        if "raw_file" in ops:
            Path(ops["raw_file"]).unlink()

        # Raw binary file for channel 2
        if "raw_file" in ops and ops["nchannels"] > 1:
            Path(ops["raw_file_chan2"]).unlink()

    message = (
        f"Plane {plane_index} processed in {ops['timing']['total_plane_runtime']} seconds. Processing results "
        f"can now be viewed in the GUI."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)


def combine_planes(ops_path: Path) -> None:
    """Assembles all data processed as part of the single-day suite2p pipeline into a 'combined' suite2p folder or
    MATLAB dataset.

    This function combines the result of processing individual imaging planes of the target movie into a unified
    dataset.

    Notes:
        Assembling all data into the 'combined' dataset is a prerequisite for running the multi-day processing pipeline.

    Args:
        ops_path: The path to the ops.npy file that stores the single-day suite2p processing parameters. Compatible
            ops.npy files are generated by the resolve_ops() function.
    """
    # Guards against invalid inputs.
    if not ops_path.exists() or not ops_path.is_file() or ops_path.suffix != ".npy":
        message = (
            f"Unable to run the single-day suite2p pipeline, as the 'ops.npy' file does not exist at the specified "
            f"path {ops_path}."
        )
        console.error(message=message, error=FileNotFoundError)

    # Loads the 'ops' dictionary from the specified storage file
    ops = np.load(ops_path, allow_pickle=True).item()

    # Resolves the paths to all plane folders and 'ops' files
    plane_folders, ops_paths, save_folder = _resolve_plane_paths(ops)

    # Generates a 'combined' plane folder, which includes the data from all processed planes. This method is safe to
    # call on one or more planes. Note, if you intend to run the multi-day (across-session) suite2p pipeline on the
    # processed data, this step is REQUIRED, even if there is only a single processed plane.
    if len(ops_paths) > 1 and ops["combined"] and ops.get("roidetect", True):
        console.echo(
            message=f"Creating combined view using {len(plane_folders)} processed planes...", level=LogLevel.INFO
        )
        io.combined(save_folder, save=True)
