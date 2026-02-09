"""Provides the high-level API for the single-day suite2p processing pipeline."""

import numba  # type: ignore[import-untyped]
from ataraxis_time import PrecisionTimer, TimerPrecisions, get_timestamp
from ataraxis_base_utilities import LogLevel, console

from . import io
from .detection import detect_plane_rois
from .extraction import extract_traces
from .dataclasses import (
    RuntimeContext,
    SingleDayConfiguration,
)
from .registration import register_plane

# Movie processing thresholds.
_MINIMUM_PROCESSING_FRAMES: int = 50
"""The minimum number of frames in the processed movie to allow processing."""

_RECOMMENDED_PROCESSING_FRAMES: int = 200
"""The recommended number of frames in the processed movie."""


def _initialize_pipeline(configuration: SingleDayConfiguration) -> list[RuntimeContext]:
    """Initializes the single-day processing pipeline.

    This function validates the input configuration, imports the processed data by converting it from TIFF to
    binary format, and initializes the output data hierarchy.

    Args:
        configuration: The single-day pipeline configuration.

    Returns:
        A list of RuntimeContext instances, one per each plane to be processed (or virtual plane for MROI data).

    Raises:
        ValueError: If data_path is not configured.
    """
    # Validates that data_path is configured.
    if configuration.file_io.data_path is None:
        message = (
            "Unable to initialize the single-day pipeline. The data_path must be configured in the FileIO section of "
            "the configuration, but it is currently None."
        )
        console.error(message=message, error=ValueError)

    # Defaults save_path to data_path if not explicitly set.
    if configuration.file_io.save_path is None:
        configuration.file_io.save_path = configuration.file_io.data_path

    # Finds and converts the input data stored as one or more TIFFs to binary format and creates RuntimeContext
    # instances.
    contexts = io.convert_tiffs_to_binary(configuration)

    # Saves shared configuration and acquisition parameters once (using first plane's context).
    contexts[0].save_shared()

    # Saves runtime data for each plane.
    for context in contexts:
        context.save_runtime()

    return contexts


def resolve_processing_contexts(configuration: SingleDayConfiguration) -> list[RuntimeContext]:
    """Resolves RuntimeContext instances for all planes in the processed recording.

    This function serves as the primary entry point for obtaining the runtime contexts needed by subsequent pipeline
    stages. It first checks for existing processed data in the output directory. If valid configuration and binary
    files are found, it loads and returns the existing RuntimeContext instances. If binaries are missing or invalid,
    it imports the raw data, converts it to the internal binary format, and initializes new contexts.

    Args:
        configuration: The single-day pipeline configuration.

    Returns:
        A list of RuntimeContext instances, one for each plane to be processed.

    Raises:
        ValueError: If save_path is not configured in the input SingleDayConfiguration class.
    """
    # Validates that save_path is configured (or can be derived from data_path).
    if configuration.file_io.save_path is None:
        if configuration.file_io.data_path is None:
            message = (
                "Unable to resolve processing contexts. Either save_path or data_path must be configured in the "
                "FileIO section of the configuration, but both are currently None."
            )
            console.error(message=message, error=ValueError)
        configuration.file_io.save_path = configuration.file_io.data_path

    # Statically uses 'suite2p' as the root output directory.
    root_path = configuration.file_io.save_path / "suite2p"

    # Checks for existing configuration file.
    config_path = root_path / "configuration.yaml"
    if config_path.exists():
        console.echo(message=f"Found existing configuration at: {config_path}.", level=LogLevel.INFO)

        # Loads all existing contexts. Uses plane_index=-1 to load all planes, which always returns a list.
        loaded_contexts = RuntimeContext.load(root_path=root_path, plane_index=-1)
        if not isinstance(loaded_contexts, list):
            loaded_contexts = [loaded_contexts]

        # Validates that required binary files exist for all contexts.
        binaries_valid = True
        for context in loaded_contexts:
            # Registered binary must always exist.
            registered_path = context.runtime.io.registered_binary_path
            if registered_path is None or not registered_path.exists():
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
        return _initialize_pipeline(configuration)

    # No existing data - create new binaries.
    console.echo(message="No existing data found. Initializing a new pipeline...", level=LogLevel.INFO)
    return _initialize_pipeline(configuration)


def save_combined_data(contexts: list[RuntimeContext]) -> None:
    """Assembles all data processed as part of the single-day suite2p pipeline into a combined dataset.

    This function combines the result of processing individual imaging planes of the recording into a unified
    dataset. Detection images (mean images, correlation maps) and extraction data (ROI statistics, fluorescence traces)
    from all planes are spatially arranged and saved to the root output directory.

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
    root_path = contexts[0].configuration.file_io.save_path
    if root_path is None:
        message = (
            "Unable to save combined plane data. The save_path must be configured in the FileIO section of the "
            "configuration, but it is currently None."
        )
        console.error(message=message, error=ValueError)

    # Combines all planes into a unified dataset.
    combined_data = io.combine_planes(contexts)

    # Saves the combined data to the root suite2p directory.
    combined_data.save(root_path / "suite2p")
    console.echo(message=f"Combined data saved to: {root_path / 'suite2p'}", level=LogLevel.SUCCESS)


def run_single_day_pipeline(configuration: SingleDayConfiguration) -> None:
    """Executes the single-day suite2p processing pipeline.

    This function sequentially calls all steps of the suite2p single-day processing pipeline, converting raw data
    frames into extracted cell fluorescence data.

    Args:
        configuration: The single-day pipeline configuration.
    """
    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    console.echo(message="Initializing single-day suite2p runtime...", level=LogLevel.INFO)

    # Step 1: Ensures the processed data is converted to the internal BinaryFile format.
    contexts = resolve_processing_contexts(configuration)

    # Step 2: Processes each plane sequentially.
    for plane_index in range(len(contexts)):
        process_plane(configuration=configuration, plane_index=plane_index)

    # Step 3: Combines all planes into a unified dataset and saves to disk.
    save_combined_data(contexts)

    message = f"Single-day suite2p runtime: Complete. Total time: {timer.elapsed} seconds."
    console.echo(message=message, level=LogLevel.SUCCESS)


def process_plane(configuration: SingleDayConfiguration, plane_index: int) -> None:
    """Runs the single-day suite2p pipeline on the target imaging plane's data.

    Loads the RuntimeContext for the specified plane from disk and sequentially runs registration, ROI detection, and
    trace extraction. Each sub-pipeline mutates the context in-place and records timing data. Results are persisted to
    disk via context.save_runtime() after processing completes.

    Notes:
        This function can be parallelized to process multiple planes at the same time. Many processing steps executed
        by this function are also internally parallelized by numba. Depending on the execution context, this function
        may use up to 2 x plane binary file size of RAM and use multiple CPU cores for each plane. Processing multiple
        planes in parallel may therefore require considerable memory and CPU resources.

    Args:
        configuration: The single-day pipeline configuration.
        plane_index: The index of the imaging plane to process.
    """
    # Aborts the processing early if the input plane is a flyback plane.
    if plane_index in configuration.main.ignored_flyback_planes:
        console.echo(message=f"Skipping processing the flyback plane {plane_index}.", level=LogLevel.SUCCESS)
        return

    # Validates that save_path is configured.
    if configuration.file_io.save_path is None:
        message = (
            "Unable to process the target plane. The save_path must be configured in the FileIO section of the "
            "configuration, but it is currently None."
        )
        console.error(message=message, error=ValueError)

    # Loads the RuntimeContext for the target plane from disk.
    root_path = configuration.file_io.save_path / "suite2p"
    context = RuntimeContext.load(root_path=root_path, plane_index=plane_index)
    if isinstance(context, list):
        message = (
            f"Unable to process the target plane. Expected a single RuntimeContext for plane {plane_index}, "
            f"but received a list of {len(context)} contexts."
        )
        console.error(message=message, error=TypeError)

    console.echo(message=f"Processing plane {plane_index}...", level=LogLevel.INFO)

    # Configures the maximum number of cores this function is allowed to use when parallelizing processing steps.
    numba.set_num_threads(configuration.main.parallel_workers)

    # Ensures that the plane contains enough frames for the processing to work as expected and, if not, either
    # aborts or notifies the user about the unexpected behavior possibility.
    frame_count = context.runtime.io.frame_count
    if frame_count < _MINIMUM_PROCESSING_FRAMES:
        message = (
            f"Unable to process plane {plane_index}. A plane must contain at least {_MINIMUM_PROCESSING_FRAMES} "
            f"frames to be processed, but the input plane contains only {frame_count} frames."
        )
        console.error(message=message, error=ValueError)

    if frame_count < _RECOMMENDED_PROCESSING_FRAMES:
        message = (
            f"The number of frames for plane {plane_index} is below {_RECOMMENDED_PROCESSING_FRAMES}, unexpected "
            f"behavior may occur during processing."
        )
        console.echo(message=message, level=LogLevel.WARNING)

    # Starts the overall plane processing timer.
    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    # Step 1: Registration (motion correction).
    register_plane(context=context)

    # Step 2: ROI detection (cell segmentation and classification).
    if configuration.roi_detection.enabled:
        detect_plane_rois(context=context)

        # Step 3: Trace extraction (fluorescence, deconvolution, colocalization).
        if context.runtime.extraction.roi_statistics is not None:
            extract_traces(context=context)
    else:
        message = f"Skipping plane {plane_index} ROI detection (disabled via 'roi_detection.enabled' parameter)."
        console.echo(message=message, level=LogLevel.WARNING)

    # Records the total plane processing time and the processing timestamp.
    context.runtime.timing.total_plane_time = timer.elapsed
    context.runtime.timing.date_processed = str(get_timestamp())

    # Persists the updated runtime data to disk.
    context.save_runtime()

    message = (
        f"Plane {plane_index} processed in {context.runtime.timing.total_plane_time} seconds. Processing results "
        f"can now be viewed in the GUI."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)
