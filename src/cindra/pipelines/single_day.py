"""Provides the high-level API for the single-day processing pipeline."""

import numba  # type: ignore[import-untyped]
from ataraxis_time import PrecisionTimer, TimerPrecisions, get_timestamp
from ataraxis_base_utilities import LogLevel, console

from ..io import combine_planes, convert_tiffs_to_binary, resolve_single_day_contexts
from ..detection import detect_plane_rois
from ..extraction import extract_traces
from ..dataclasses import (
    RuntimeContext,
    SingleDayConfiguration,
)
from ..registration import register_plane

# Movie processing thresholds.
_MINIMUM_PROCESSING_FRAMES: int = 50
"""The minimum number of frames in the processed movie to allow processing."""

_RECOMMENDED_PROCESSING_FRAMES: int = 200
"""The recommended minimum number of frames in the processed movie for the processing to work as expected."""


def binarize_recording(configuration: SingleDayConfiguration) -> None:
    """Converts raw TIFF recording data into the internal binary format used by the processing pipeline.

    Notes:
        This function executes the first phase of the single-day pipeline: it converts the raw recording data into the
        internal binary format and initializes the per-plane runtime data hierarchy. If valid binaries already exist at
        the output path, the conversion is skipped.

    Args:
        configuration: The single-day pipeline configuration.

    Raises:
        ValueError: If data_path is not configured.
    """
    # Validates that data_path is configured.
    if configuration.file_io.data_path is None:
        message = (
            "Unable to binarize the recording. The data_path must be configured in the FileIO section of the "
            "configuration, but it is currently None."
        )
        console.error(message=message, error=ValueError)

    # Defaults save_path to data_path if not explicitly set.
    if configuration.file_io.save_path is None:
        configuration.file_io.save_path = configuration.file_io.data_path

    # Checks for existing valid binaries to allow early return.
    root_path = configuration.file_io.save_path / "cindra"
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
            registered_path = context.runtime.io.registered_binary_path
            if registered_path is None or not registered_path.exists():
                binaries_valid = False
                break

        if binaries_valid:
            message = f"Loaded {len(loaded_contexts)} existing plane contexts with valid binaries."
            console.echo(message=message, level=LogLevel.SUCCESS)
            return

        # Binaries are missing or invalid - fall through to recreate.
        console.echo(
            message="Existing binaries are missing or invalid. Recreating from TIFF files...", level=LogLevel.WARNING
        )

    # Starts the binarization timer.
    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    # Creates RuntimeContext instances for all planes.
    contexts = resolve_single_day_contexts(configuration=configuration)

    # Converts TIFF data to binary format.
    convert_tiffs_to_binary(contexts=contexts)

    # Saves shared configuration and acquisition parameters once (using first plane's context).
    contexts[0].save_shared()

    # Records the binarization time and saves runtime data for each plane.
    for context in contexts:
        context.runtime.timing.binarization_time = timer.elapsed
        context.save_runtime()

    message = f"Binarization complete. {len(contexts)} plane(s) converted in {timer.elapsed} seconds."
    console.echo(message=message, level=LogLevel.SUCCESS)


def process_plane(configuration: SingleDayConfiguration, plane_index: int) -> None:
    """Registers, detects ROIs, and extracts fluorescence traces for the target imaging plane.

    Notes:
        This function executes the second phase of the single-day pipeline: it processes a single imaging plane through
        registration, ROI detection, and trace extraction. Multiple planes can be processed in parallel, but each
        plane may use significant memory and CPU resources.

    Args:
        configuration: The single-day pipeline configuration.
        plane_index: The index of the imaging plane to process.
    """
    # Skips flyback planes early.
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
    root_path = configuration.file_io.save_path / "cindra"
    context = RuntimeContext.load(root_path=root_path, plane_index=plane_index)
    if isinstance(context, list):
        message = (
            f"Unable to process the target plane. Expected a single RuntimeContext for plane {plane_index}, "
            f"but received a list of {len(context)} contexts."
        )
        console.error(message=message, error=TypeError)

    console.echo(message=f"Processing plane {plane_index}...", level=LogLevel.INFO)

    # Configures the maximum number of numba threads for parallelization. The worker count is already resolved to a
    # valid positive integer by the pipeline entry point.
    numba.set_num_threads(configuration.runtime.parallel_workers)

    # Validates the frame count meets minimum processing requirements.
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

    # Runs registration (motion correction).
    register_plane(context=context)

    # Runs ROI detection and trace extraction when detection is enabled.
    if configuration.roi_detection.enabled:
        detect_plane_rois(context=context)

        # Extracts fluorescence traces when ROIs were detected.
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


def save_combined_data(contexts: list[RuntimeContext]) -> None:
    """Combines processed data from all imaging planes into a unified dataset and saves it to disk.

    Notes:
        This function executes the final phase of the single-day pipeline. The combined dataset is a prerequisite for
        running the multi-day processing pipeline.

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
    combined_data = combine_planes(plane_contexts=contexts)

    # Saves the combined data to the root cindra directory.
    combined_data.save(root_path / "cindra")
    console.echo(message=f"Combined data saved to: {root_path / 'cindra'}", level=LogLevel.SUCCESS)
