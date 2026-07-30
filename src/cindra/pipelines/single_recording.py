"""Provides the high-level API for the single-recording processing pipeline."""

from pathlib import Path  # noqa: TC003 - the module does not defer annotation evaluation.

import numba
from ataraxis_time import PrecisionTimer, TimerPrecisions, get_timestamp
from ataraxis_base_utilities import LogLevel, console

from ..io import (
    combine_planes,
    convert_tiffs_to_binary,
    resolve_registration_marker_path,
    resolve_single_recording_contexts,
)
from ..detection import detect_plane_rois
from ..extraction import extract_traces
from ..dataclasses import (
    RuntimeContext,
    SingleRecordingConfiguration,
)
from ..registration import register_plane

_MINIMUM_PROCESSING_FRAMES: int = 50
"""The minimum number of frames in the processed movie to allow processing."""

_RECOMMENDED_PROCESSING_FRAMES: int = 200
"""The recommended minimum number of frames in the processed movie for the processing to work as expected."""

_BINARY_ITEM_SIZE: int = 2
"""The number of bytes one pixel occupies inside a cindra binary, which stores int16 samples."""


def binarize_recording(configuration: SingleRecordingConfiguration, *, workers: int) -> None:
    """Converts raw TIFF recording data into the internal binary format used by the processing pipeline.

    Notes:
        This function executes the first phase of the single-recording pipeline: it converts the raw recording data
        into the internal binary format and initializes the per-plane runtime data hierarchy. The conversion is
        skipped only when every plane already has a valid binary at the output path and 'repeat_binarization' is
        disabled in the FileIO configuration section. A binary left marked by an interrupted registration, or one
        whose size disagrees with its plane's recorded frame geometry, is treated as invalid and rebuilt from the
        source TIFF files even when that parameter is disabled.

    Args:
        configuration: The single-recording pipeline configuration.
        workers: The number of parallel workers allocated to this binarization job. Must be a positive integer, which
            the caller resolves before invoking this function.

    Raises:
        ValueError: If data_path or output_path is not configured, or if the discovered TIFF files do not all hold
            frames of the same shape.
        FileNotFoundError: If a plane's runtime_data.yaml was not written by an earlier bootstrap step, or if no TIFF
            files are found in the data directory.
    """
    # Validates that data_path is configured.
    if configuration.file_io.data_path is None:
        message = (
            "Unable to binarize the recording. The data_path must be configured in the FileIO section of the "
            "configuration, but it is currently None."
        )
        console.error(message=message, error=ValueError)

    # Validates that output_path is configured.
    if configuration.file_io.output_path is None:
        message = (
            "Unable to binarize the recording. The output_path must be configured in the FileIO section of the "
            "configuration, but it is currently None."
        )
        console.error(message=message, error=ValueError)

    # Checks for existing valid binaries to allow early return.
    root_path = configuration.file_io.output_path / "cindra"
    config_path = root_path / "configuration.yaml"
    acquisition_path = root_path / "acquisition_parameters.yaml"
    if config_path.exists() and acquisition_path.exists():
        console.echo(message=f"Found existing configuration at: {config_path}.", level=LogLevel.INFO)

        # Loads all existing contexts. Uses plane_index=-1 to load all planes, which always returns a list.
        loaded_contexts = RuntimeContext.load(root_path=root_path, plane_index=-1)
        if not isinstance(loaded_contexts, list):  # pragma: no cover - load with plane_index=-1 always returns a list
            loaded_contexts = [loaded_contexts]

        # Validates that required binary files exist for all contexts.
        binaries_valid = True
        marked_binaries: list[Path] = []
        for context in loaded_contexts:
            registered_path = context.runtime.io.registered_binary_path
            if registered_path is None or not registered_path.exists():
                binaries_valid = False
                break

            # An interrupted registration leaves its binary holding motion-corrected frames up to an unknown point and
            # raw frames after it, which only the marker beside the binary records.
            marked_binaries.extend(
                path
                for path in (registered_path, context.runtime.io.registered_binary_path_channel_2)
                if path is not None and resolve_registration_marker_path(binary_path=path).exists()
            )

        # Rebuilds a marked binary without waiting for the caller to request it. The frames it holds are indeterminate,
        # so re-converting from the source TIFF files is the only way to make the recording processable again.
        if marked_binaries:
            binaries_valid = False
            console.echo(
                message=(
                    f"Rebuilding {len(marked_binaries)} binary file(s) that a previous interrupted registration left "
                    f"in an indeterminate state: {sorted(str(path) for path in marked_binaries)}."
                ),
                level=LogLevel.WARNING,
            )

        # Rebuilds a binary whose size disagrees with the geometry recorded for its plane, which is what an
        # interrupted conversion leaves behind. Every later stage derives its frame count by dividing the file size by
        # the frame size, so a short binary would otherwise be processed as a silently truncated movie.
        if binaries_valid:
            malformed_binaries = _resolve_malformed_binaries(contexts=loaded_contexts)
            if malformed_binaries:
                binaries_valid = False
                console.echo(
                    message=(
                        f"Rebuilding {len(malformed_binaries)} binary file(s) whose size does not match the frame "
                        f"geometry recorded for their plane: {sorted(str(path) for path in malformed_binaries)}."
                    ),
                    level=LogLevel.WARNING,
                )

        if binaries_valid and not configuration.file_io.repeat_binarization:
            message = f"Loaded {len(loaded_contexts)} existing plane contexts with valid binaries."
            console.echo(message=message, level=LogLevel.SUCCESS)
            return

        if configuration.file_io.repeat_binarization:
            console.echo(
                message="Repeating binarization as requested by 'repeat_binarization' parameter...",
                level=LogLevel.WARNING,
            )
        else:
            # Binaries are missing or invalid, so the code below recreates them from the source TIFF files.
            console.echo(
                message="Existing binaries are missing or invalid. Recreating from TIFF files...",
                level=LogLevel.WARNING,
            )

    # Starts the binarization timer.
    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    # Creates RuntimeContext instances for all planes. The outer pipeline entry (run_single_recording_pipeline) or
    # the prepare_single_recording_batch_tool already wrote the shared configuration, acquisition parameters, and
    # per-plane runtime_data.yaml files, so this call is load-only to avoid racing against peer worker threads.
    contexts = resolve_single_recording_contexts(configuration=configuration, persist=False)

    convert_tiffs_to_binary(contexts=contexts, workers=workers)

    # Records the binarization time and saves runtime data for each plane. Each plane's runtime_data.yaml has only
    # one writer here (the single BINARIZE worker for this recording), so the per-plane save is race-free.
    for context in contexts:
        context.runtime.timing.binarization_time = timer.elapsed
        context.save_runtime()

    message = f"Binarization complete. {len(contexts)} plane(s) converted in {timer.elapsed} seconds."
    console.echo(message=message, level=LogLevel.SUCCESS)


def register_recording_plane(configuration: SingleRecordingConfiguration, plane_index: int, *, workers: int) -> None:
    """Removes motion from the target imaging plane and computes its registration quality metrics.

    Notes:
        This function executes the second phase of the single-recording pipeline: it motion-corrects a single imaging
        plane and computes the principal components used to assess the registration quality. Multiple planes can be
        registered in parallel, but each plane may use significant memory and CPU resources.

        This stage is the prerequisite for the processing stage. It writes the registration offsets, the valid pixel
        ranges, and the bad-frame mask to disk, all of which the processing stage reads back before detecting ROIs.

    Args:
        configuration: The single-recording pipeline configuration.
        plane_index: The index of the imaging plane to register.
        workers: The number of parallel workers allocated to this registration job. Must be a positive integer, which
            the caller resolves before invoking this function.
    """
    context = _resolve_plane_context(
        configuration=configuration,
        plane_index=plane_index,
        workers=workers,
        stage_action="register",
        stage_progressive="Registering",
        stage_noun="registration",
    )
    if context is None:
        return

    # Starts the overall plane registration timer.
    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    # Runs registration (motion correction) and the registration quality metrics computation.
    register_plane(context=context, workers=workers)

    # Records the total plane registration time and the allocation the stage actually used.
    context.runtime.timing.total_registration_time = timer.elapsed
    context.runtime.timing.registration_workers = workers

    context.save_runtime()

    message = (
        f"Plane {plane_index} registered in {context.runtime.timing.total_registration_time} seconds. Registration "
        f"quality can now be reviewed in the GUI."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)


def process_plane(configuration: SingleRecordingConfiguration, plane_index: int, *, workers: int) -> None:
    """Detects ROIs and extracts their fluorescence traces for the target imaging plane.

    Notes:
        This function executes the third phase of the single-recording pipeline: it discovers the ROIs of a single
        registered imaging plane and extracts, classifies, and deconvolves their fluorescence. Multiple planes can be
        processed in parallel, but each plane may use significant memory and CPU resources.

        The plane must be registered before it is processed. Detection reads the valid pixel ranges computed during
        registration, and an unregistered plane carries the (0, 0) defaults for those ranges, which silently produce a
        zero-size binned movie instead of an error.

    Args:
        configuration: The single-recording pipeline configuration.
        plane_index: The index of the imaging plane to process.
        workers: The number of parallel workers allocated to this processing job. Must be a positive integer, which the
            caller resolves before invoking this function.

    Raises:
        ValueError: If output_path is not configured, or if the plane contains fewer frames than the processing
            minimum.
        TypeError: If the runtime context loader returns multiple contexts for the target plane.
        RuntimeError: If the target plane has not been registered.
    """
    context = _resolve_plane_context(
        configuration=configuration,
        plane_index=plane_index,
        workers=workers,
        stage_action="process",
        stage_progressive="Processing",
        stage_noun="processing",
    )
    if context is None:
        return

    # Validates that the plane has been registered.
    if not context.runtime.registration.is_registered(output_path=context.runtime.io.output_path):
        message = (
            f"Unable to process plane {plane_index}. The plane must be registered before ROI detection, but no "
            f"registration data was found in memory or under the plane's output directory. Run the registration "
            f"stage for this plane before running the processing stage."
        )
        console.error(message=message, error=RuntimeError)

    # Starts the overall plane processing timer.
    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    # Runs ROI detection and trace extraction when detection is enabled.
    if configuration.roi_detection.enabled:
        detect_plane_rois(context=context, workers=workers)

        # Extracts fluorescence traces when ROIs were detected.
        # pragma justification: detection always populates ROI statistics on success or raises beforehand.
        if context.runtime.extraction.roi_statistics is not None:  # pragma: no branch
            extract_traces(context=context, workers=workers)
    else:
        message = f"Skipping plane {plane_index} ROI detection (disabled via 'roi_detection.enabled' parameter)."
        console.echo(message=message, level=LogLevel.WARNING)

    # Records the total plane processing time, the allocation the stage actually used, and the processing timestamp.
    context.runtime.timing.total_processing_time = timer.elapsed
    context.runtime.timing.processing_workers = workers
    context.runtime.timing.date_processed = str(get_timestamp())

    context.save_runtime()

    message = (
        f"Plane {plane_index} processed in {context.runtime.timing.total_processing_time} seconds. Processing results "
        f"can now be viewed in the GUI."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)


def save_combined_data(contexts: list[RuntimeContext]) -> None:
    """Combines processed data from all imaging planes into a unified dataset and saves it to disk.

    Notes:
        This function executes the final phase of the single-recording pipeline. The combined dataset is a
        prerequisite for running the multi-recording processing pipeline.

    Args:
        contexts: A list of RuntimeContext instances, one per plane to combine. Each context must have valid runtime
            data populated by the processing pipeline.
    """
    if not contexts:
        message = "Unable to combine planes. At least one RuntimeContext must be provided."
        console.error(message=message, error=ValueError)

    root_path = contexts[0].configuration.file_io.output_path
    if root_path is None:
        message = (
            "Unable to save combined plane data. The output_path must be configured in the FileIO section of the "
            "configuration, but it is currently None."
        )
        console.error(message=message, error=ValueError)

    combined_data = combine_planes(plane_contexts=contexts)

    combined_data.save(root_path=root_path / "cindra")
    console.echo(message=f"Combined data saved to: {root_path / 'cindra'}", level=LogLevel.SUCCESS)


def _resolve_malformed_binaries(contexts: list[RuntimeContext]) -> list[Path]:
    """Returns the plane binaries whose size disagrees with the frame geometry recorded for their plane.

    Notes:
        The check compares file sizes rather than file contents. The registration stage rewrites a binary in place, so
        it changes what the file holds without changing how large it is, which makes size the one property that stays
        invariant across every stage that consumes the binary. A content check would instead report every registered
        plane as malformed.

        A size mismatch is what a conversion that did not finish leaves behind. The frame count every later stage
        works from is derived by dividing the file size by the frame size, so a short file silently yields a truncated
        movie rather than an error.

        Channel 2 is allowed to hold one frame more or fewer than channel 1, because a recording whose acquisition
        stopped partway through a volume delivers a different number of frames to the two channels of one plane.

    Args:
        contexts: The loaded plane contexts to check the binaries of.

    Returns:
        The paths of every binary whose size does not match its recorded geometry.
    """
    malformed: list[Path] = []
    for context in contexts:
        io_data = context.runtime.io
        frame_bytes = io_data.frame_height * io_data.frame_width * _BINARY_ITEM_SIZE
        if frame_bytes <= 0:  # pragma: no cover - a persisted plane always records its frame dimensions
            continue

        for path, expected_frames, tolerance in (
            (io_data.registered_binary_path, io_data.frame_count, 0),
            (io_data.registered_binary_path_channel_2, io_data.frame_count, 1),
        ):
            if path is None or not path.exists():
                continue

            byte_number = path.stat().st_size
            stored_frames = byte_number // frame_bytes
            if byte_number % frame_bytes != 0 or abs(stored_frames - expected_frames) > tolerance:
                malformed.append(path)

    return malformed


def _resolve_plane_context(
    configuration: SingleRecordingConfiguration,
    plane_index: int,
    *,
    workers: int,
    stage_action: str,
    stage_progressive: str,
    stage_noun: str,
) -> RuntimeContext | None:
    """Loads and validates the runtime context for the target plane and applies the stage's worker budget.

    Notes:
        This helper carries the preamble shared by the registration and processing plane stages. It skips flyback
        planes, validates the configured output path, loads the plane's RuntimeContext from disk, applies the
        allocated worker budget to the calling thread's Numba mask, and validates the plane's frame count.

        The stage labels are parameterized because the two stages report different actions to the user.

    Args:
        configuration: The single-recording pipeline configuration.
        plane_index: The index of the imaging plane to load the runtime context for.
        workers: The number of parallel workers allocated to the calling stage. Must be a positive integer, which the
            caller resolves before invoking this function.
        stage_action: The lowercase verb naming the calling stage's action, used in error messages. Use 'register' for
            the registration stage and 'process' for the processing stage.
        stage_progressive: The capitalized progressive form of the calling stage's action, used in status messages. Use
            'Registering' for the registration stage and 'Processing' for the processing stage.
        stage_noun: The lowercase noun naming the calling stage, used in the flyback skip message. Use 'registration'
            for the registration stage and 'processing' for the processing stage.

    Returns:
        The loaded RuntimeContext for the target plane, or None if the target plane is a flyback plane that the calling
        stage must skip.

    Raises:
        ValueError: If output_path is not configured or the plane contains too few frames to be processed.
        TypeError: If the runtime context loader returns multiple contexts for the target plane.
    """
    # Skips flyback planes early.
    if plane_index in configuration.main.ignored_flyback_planes:
        message = f"Skipping the {stage_noun} of the flyback plane {plane_index}."
        console.echo(message=message, level=LogLevel.SUCCESS)
        return None

    # Validates that output_path is configured.
    if configuration.file_io.output_path is None:
        message = (
            f"Unable to {stage_action} the target plane. The output_path must be configured in the FileIO section of "
            f"the configuration, but it is currently None."
        )
        console.error(message=message, error=ValueError)

    root_path = configuration.file_io.output_path / "cindra"
    context = RuntimeContext.load(root_path=root_path, plane_index=plane_index)
    if isinstance(context, list):
        message = (
            f"Unable to {stage_action} the target plane. Expected a single RuntimeContext for plane {plane_index}, "
            f"but received a list of {len(context)} contexts."
        )
        console.error(message=message, error=TypeError)

    console.echo(message=f"{stage_progressive} plane {plane_index}...", level=LogLevel.INFO)

    # Applies the allocated worker budget to the calling thread's Numba mask. The mask is thread-local, so concurrently
    # dispatched planes can hold different budgets inside one process. The mask cannot exceed the number of cores Numba
    # detected at import time, so the requested budget is capped at that ceiling.
    numba.set_num_threads(min(workers, numba.config.NUMBA_NUM_THREADS))

    # Validates the frame count meets minimum processing requirements.
    frame_count = context.runtime.io.frame_count
    if frame_count < _MINIMUM_PROCESSING_FRAMES:
        message = (
            f"Unable to {stage_action} plane {plane_index}. A plane must contain at least "
            f"{_MINIMUM_PROCESSING_FRAMES} frames to be processed, but the input plane contains only {frame_count} "
            f"frames."
        )
        console.error(message=message, error=ValueError)

    if frame_count < _RECOMMENDED_PROCESSING_FRAMES:
        message = (
            f"The number of frames for plane {plane_index} is below {_RECOMMENDED_PROCESSING_FRAMES}, unexpected "
            f"behavior may occur during processing."
        )
        console.echo(message=message, level=LogLevel.WARNING)

    return context
