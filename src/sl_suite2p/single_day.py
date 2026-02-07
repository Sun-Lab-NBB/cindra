"""Provides the high-level API for the single-day suite2p processing pipeline."""

from typing import Any
from pathlib import Path
from datetime import datetime
import contextlib

import numba
import numpy as np
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import LogLevel, console

from . import io, detection, extraction, classification
from .io.binary import BinaryFile
from .registration import register_plane
from .dataclasses import (
    RuntimeContext,
    SingleDayConfiguration,
)

# Defines constants used in this module

# Specifies the maximum number of channels in processed movie images.
_MAXIMUM_SUPPORTED_CHANNELS = 2  # At most two channels: red and green.

# Movie processing thresholds.
_MINIMUM_PROCESSING_FRAMES = 50  # The minimum number of frames in the processed movie to allow processing.
_RECOMMENDED_PROCESSING_FRAMES = 200  # The recommended number of frames in the processed movie.

_MINIMUM_REGISTRATION_METRIC_FRAMES = 1500  # The minimum number of frames required to compute registration metrics.


def _initialize_pipeline(config: SingleDayConfiguration) -> list[RuntimeContext]:
    """Initializes the single-day processing pipeline.

    This function validates the input configuration, imports the processed data by converting it fom the TIFF to
    the binary format, and initializes the output data hierarchy.

    Args:
        config: The single-day pipeline configuration.

    Returns:
        A list of RuntimeContext instances, one per each plane to be processed (or virtual plane for MROI data).

    Raises:
        ValueError: If data_path is not configured.
    """
    # Validates that data_path is configured.
    if config.file_io.data_path is None:
        message = (
            "Unable to initialize the pipeline. The data_path must be configured in the FileIO section of the "
            "configuration, but it is currently None."
        )
        console.error(message=message, error=ValueError)

    # Defaults save_path to data_path if not explicitly set.
    if config.file_io.save_path is None:
        config.file_io.save_path = config.file_io.data_path

    # Finds and converts the input data stored as one or more TIFFs to binary format and creates RuntimeContext
    # instances.
    contexts = io.convert_tiffs_to_binary(config)

    # Saves shared configuration and acquisition parameters once (using first plane's context).
    contexts[0].save_shared()

    # Saves runtime data for each plane.
    for context in contexts:
        context.save_runtime()

    return contexts


def resolve_processing_contexts(config: SingleDayConfiguration) -> list[RuntimeContext]:
    """Resolves RuntimeContext instances for all planes in the processed session.

    This function serves as the primary entry point for obtaining the runtime contexts needed by subsequent pipeline
    stages. It first checks for existing processed data in the output directory. If valid configuration and binary
    files are found, it loads and returns the existing RuntimeContext instances. If binaries are missing or invalid,
    it imports the raw data, converts it to the internal binary format, and initializes new contexts.

    Args:
        config: The single-day pipeline configuration.

    Returns:
        A list of RuntimeContext instances, one for each plane to be processed.

    Raises:
        ValueError: If save_path is not configured.
    """
    # Validates that save_path is configured (or can be derived from data_path).
    if config.file_io.save_path is None:
        if config.file_io.data_path is None:
            message = (
                "Unable to resolve processing contexts. Either save_path or data_path must be configured in the "
                "FileIO section of the configuration, but both are currently None."
            )
            console.error(message=message, error=ValueError)
        config.file_io.save_path = config.file_io.data_path

    # Statically uses 'suite2p' as the rot output directory.
    root_path = config.file_io.save_path / "suite2p"

    # Checks for existing configuration file.
    config_path = root_path / "configuration.yaml"
    if config_path.exists():
        console.echo(message=f"Found existing configuration at: {config_path}.", level=LogLevel.INFO)

        # Loads all existing contexts.
        loaded_contexts = RuntimeContext.load(root_path=root_path, plane_index=-1)

        # Validates that required binary files exist for all contexts.
        binaries_valid = True
        for context in loaded_contexts:
            # Registered binary must always exist.
            registered_path = context.runtime.io.registered_binary_path
            if registered_path is None or not registered_path.exists():
                binaries_valid = False
                break

            # Raw binary must exist if keep_movie_raw is enabled.
            if config.registration.keep_movie_raw:
                raw_path = context.runtime.io.raw_binary_path
                if raw_path is None or not raw_path.exists():
                    binaries_valid = False
                    break

        if binaries_valid:
            message = f"Loaded {len(loaded_contexts)} existing plane contexts with valid binaries."
            console.echo(message=message, level=LogLevel.SUCCESS)
            return loaded_contexts

        # Binaries are missing or invalid - need to recreate.
        console.echo(
            message="Existing binaries are missing or invalid. Recreating from TIFF files...", level=LogLevel.WARNING
        )
        return _initialize_pipeline(config)

    # No existing data - create new binaries.
    console.echo(message="No existing data found. Initializing a new pipeline...", level=LogLevel.INFO)
    return _initialize_pipeline(config)


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
    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)

    # Selects the classifier file based on the processing configuration.
    classifier_file = classification.resolve_classifier_path(custom_classifier_path=ops.get("classifier_path"))

    # Memory-maps the necessary binary files.
    n_frames, height, width = ops["frame_count"], ops["frame_height"], ops["frame_width"]
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
                roi_statistics=roi_statistics,
                plane_number=plane_number,
                frames=frames,
                channel_2_frames=frames_channel_2,
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
            if ops.get("extract_spikes", True):
                message = f"Processing plane {plane_number} activity spikes..."
                console.echo(message=message, level=LogLevel.INFO)
                timer.reset()

                # Computes delta f/f (neuropil-subtracted ROI fluorescence)
                df = extraction.preprocess(
                    roi_fluorescence=cell_fluorescence,
                    neuropil_fluorescence=neuropil_fluorescence,
                    ops=ops,
                )

                # Extracts the cell fluorescence spikes using the OASIS algorithm.
                spikes = extraction.oasis(
                    cell_fluorescence=df,
                    batch_size=ops["batch_size"],
                    time_constant=ops["tau"],
                    sampling_rate=ops["sampling_rate"],
                )
                ops["timing"]["deconvolution"] = timer.elapsed

                message = (
                    f"Plane {plane_number} spikes: computed. Time taken: {ops['timing']['deconvolution']} seconds."
                )
                console.echo(message=message, level=LogLevel.SUCCESS)

            else:
                message = (
                    f"Skipping plane {plane_number} spike deconvolution, as the 'extract_spikes' configuration "
                    f"parameter is set to False."
                )
                console.echo(message=message, level=LogLevel.WARNING)
                spikes = np.zeros_like(cell_fluorescence)

            # Saves pipeline output to disk as .npy files.
            fpath = Path(ops["output_directory"])
            if ops.get("output_directory"):
                np.save(fpath.joinpath("stat.npy"), roi_statistics)
                np.save(fpath.joinpath("F.npy"), cell_fluorescence)
                np.save(fpath.joinpath("Fneu.npy"), neuropil_fluorescence)
                np.save(fpath.joinpath("Fsub.npy"), df)
                np.save(fpath.joinpath("iscell.npy"), iscell)
                np.save(fpath.joinpath("spks.npy"), spikes)

                # If the data contains two functional channels, also saves the data for the second channel.
                if "mean_image_channel_2" in ops:
                    np.save(fpath.joinpath("F_chan2.npy"), cell_fluorescence_channel_2)
                    np.save(fpath.joinpath("Fneu_chan2.npy"), neuropil_fluorescence_channel_2)
        else:
            message = f"No ROIs found for plane {plane_number}."
            console.echo(message=message, level=LogLevel.WARNING)

    # Returns the updated ops dictionary to caller
    return ops


def save_combined_data(contexts: list[RuntimeContext]) -> None:
    """Assembles all data processed as part of the single-day suite2p pipeline into a combined dataset.

    This function combines the result of processing individual imaging planes of the target movie into a unified
    dataset. Detection images (mean images, correlation maps) and extraction data (ROI statistics, fluorescence traces)
    from all planes are spatially arranged and saved to the root suite2p directory.

    Notes:
        Assembling all data into the combined dataset is a prerequisite for running the multi-day processing pipeline.

    Args:
        contexts: A list of RuntimeContext instances, one per plane to combine. Each context must have valid runtime
            data populated by the processing pipeline.
    """
    if not contexts:
        message = "Unable to combine planes. At least one RuntimeContext must be provided."
        console.error(message=message, error=ValueError)

    # Gets the root output path from the first context's configuration.
    root_path = contexts[0].config.file_io.save_path
    if root_path is None:
        message = (
            "Unable to save combined data. The save_path must be configured in the FileIO section of the "
            "configuration, but it is currently None."
        )
        console.error(message=message, error=ValueError)

    # Combines all planes into a unified dataset.
    combined_data = io.combine_planes(contexts)

    # Saves the combined data to the root suite2p directory.
    combined_data.save(root_path / "suite2p")
    console.echo(message=f"Combined data saved to: {root_path / 'suite2p'}", level=LogLevel.SUCCESS)


def run_s2p(config: SingleDayConfiguration) -> None:
    """Executes the single-day suite2p processing pipeline.

    This function sequentially calls all steps of the suite2p single-day processing pipeline, converting raw data
    frames into extracted cell fluorescence data.

    Args:
        config: The single-day pipeline configuration.
    """
    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    console.echo(message="Initializing single-day suite2p runtime...", level=LogLevel.INFO)

    # Step 1: Ensures the processed data is converted to the internal BinaryFile format.
    contexts = resolve_processing_contexts(config)

    # Step 2: Processes each plane. Plane processing is not yet refactored to use RuntimeContext.

    # Step 3: Combines all planes into a unified dataset and saves to disk.
    save_combined_data(contexts)

    message = f"Single-day suite2p runtime: Complete. Total time: {timer.elapsed} seconds."
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
                "save_path",
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
    if ops["frame_count"] < _MINIMUM_PROCESSING_FRAMES:
        message = (
            f"Unable to process plane {plane_index}. A plane must contain at least 50 frames to be processed, but "
            f"the input plane contains only {ops['nframes']} frames."
        )
        console.error(message=message, error=ValueError)

    if ops["frame_count"] < _RECOMMENDED_PROCESSING_FRAMES:
        message = (
            f"The number of frames for plane {plane_index} is below 200, unexpected behavior may occur during "
            f"processing."
        )
        console.echo(message=message, level=LogLevel.WARNING)

    # TODO implement proper registration
    register_plane(context=None)

    # If ROI (cell) segmentation is enabled, segments (detects) cell ROIs
    if ops.get("roidetect", True):
        ops = _process_rois(
            ops=ops, plane_number=plane_index, frames_path=frames_path, frames_channel_2_path=frames_channel_2_path
        )
    else:
        message = f"Skipping plane {plane_index} cell detection (disabled via 'roidetect' parameter)."
        console.echo(message=message, level=LogLevel.WARNING)

    # Appends the overall plane processing time to the 'ops' file.
    ops["timing"]["total_plane_runtime"] = timer.elapsed

    # Caches plane processing results to disk
    if ops.get("ops_path"):
        np.save(ops["ops_path"], ops)

    message = (
        f"Plane {plane_index} processed in {ops['timing']['total_plane_runtime']} seconds. Processing results "
        f"can now be viewed in the GUI."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)
