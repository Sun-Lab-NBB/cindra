"""Provides the high-level API for the single-recording processing pipeline."""

from pathlib import Path  # noqa: TC003 - the module does not defer annotation evaluation.

import numba
from natsort import natsorted
from ataraxis_time import PrecisionTimer, TimerPrecisions, get_timestamp
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]
from ataraxis_base_utilities import LogLevel, console

from ..io import (
    combine_planes,
    convert_tiffs_to_binary,
    clear_recording_selections,
    resolve_active_binary_marker,
    resolve_tiff_conversion_plan,
    resolve_single_recording_contexts,
)
from ..layout import (
    CHANNEL_2_BINARY_FILENAME,
    COMBINED_METADATA_FILENAME,
    DETECTION_DATA_DIRECTORY_NAME,
    ACQUISITION_PARAMETERS_FILENAME,
    REGISTRATION_DATA_DIRECTORY_NAME,
    SINGLE_RECORDING_CONFIGURATION_FILENAME,
    DetectionImages,
    RecordingArrays,
    RegistrationArrays,
    resolve_array_path,
    resolve_output_path,
    parse_plane_specifier,
)
from ..detection import detect_plane_rois
from ..extraction import extract_traces
from ..dataclasses import (
    TimingData,
    DetectionData,
    ExtractionData,
    RuntimeContext,
    RegistrationData,
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
        The conversion is skipped when every plane already holds the channel binaries the recording declares and
        'repeat_binarization' is disabled in the FileIO configuration section.

        An existing output that disagrees with what the recording declares is refused rather than repaired. The
        refusals cover a binary an interrupted conversion or registration left marked, a plane of a two-channel
        recording holding no second channel binary, and a binary whose size disagrees with its plane's recorded frame
        geometry. Enabling 'repeat_binarization' rebuilds the recording past all three, because that parameter is the
        caller asking for the conversion that replaces every binary those refusals name.

        The conversion consumes whole plane and channel interleave cycles, so every plane and channel of the recording
        receives the same frame count and the frames of an incomplete final cycle are discarded.

        A conversion replaces every plane binary of the recording, so it first deletes the registration, detection,
        and extraction outputs of every plane directory the output root holds, along with the recording's combined
        dataset. The rebuilt binaries hold raw frames again, which voids every offset, image, and trace measured from
        the previous binaries, and deleting the registration output is what makes the registration stage run instead
        of skipping the plane. That deletion follows the conversion plan, so a recording whose TIFF files cannot be
        converted keeps its results.

    Args:
        configuration: The single-recording pipeline configuration.
        workers: The number of parallel workers allocated to this binarization job. Must be a positive integer, which
            the caller resolves before invoking this function.

    Raises:
        ValueError: If data_path or output_path is not configured, if the discovered TIFF files do not all hold frames
            of the same shape, or if the frames they hold do not fill one complete plane and channel interleave cycle.
        RuntimeError: If a converted plane binary carries the marker of an interrupted write, if a converted plane of
            a two-channel recording holds no second channel binary, or if a binary's size disagrees with the frame
            geometry recorded for its plane.
        FileNotFoundError: If a plane's runtime_data.yaml was not written by an earlier bootstrap step, or if no TIFF
            files are found in the data directory.
    """
    if configuration.file_io.data_path is None:
        message = (
            "Unable to binarize the recording. The data_path must be configured in the FileIO section of the "
            "configuration, but it is currently None."
        )
        console.error(message=message, error=ValueError)

    if configuration.file_io.output_path is None:
        message = (
            "Unable to binarize the recording. The output_path must be configured in the FileIO section of the "
            "configuration, but it is currently None."
        )
        console.error(message=message, error=ValueError)

    # Holds the recording's existing output to what the recording declares, which either refuses the recording or
    # allows an early return. Every refusal fires before the conversion plan is resolved, so a refused recording keeps
    # every binary and every result it held on arrival.
    output_root = configuration.file_io.output_path
    root_path = resolve_output_path(output_root=output_root)
    configuration_path = root_path / SINGLE_RECORDING_CONFIGURATION_FILENAME
    acquisition_path = root_path / ACQUISITION_PARAMETERS_FILENAME
    if configuration_path.exists() and acquisition_path.exists():
        console.echo(message=f"Found existing configuration at: {configuration_path}.", level=LogLevel.INFO)

        # Loads all existing contexts. Uses plane_index=-1 to load all planes, which always returns a list.
        loaded_contexts = RuntimeContext.load(root_path=root_path, plane_index=-1)
        if not isinstance(loaded_contexts, list):  # pragma: no cover - load with plane_index=-1 always returns a list
            loaded_contexts = [loaded_contexts]

        if configuration.file_io.repeat_binarization:
            console.echo(
                message="Repeating binarization as requested by 'repeat_binarization' parameter...",
                level=LogLevel.WARNING,
            )
        else:
            # A plane holding no functional channel binary has not been converted yet, so the checks below hold the
            # converted planes alone to what the recording declares and a recording holding no binary at all converts
            # normally. A caller that enabled 'repeat_binarization' skips them, because the conversion replaces every
            # binary they would refuse.
            converted_contexts = [
                context
                for context in loaded_contexts
                if context.runtime.io.registered_binary_path is not None
                and context.runtime.io.registered_binary_path.exists()
            ]
            _validate_binaries_are_unmarked(contexts=converted_contexts)
            _validate_second_channel_binaries(contexts=converted_contexts)
            _validate_binary_sizes(contexts=converted_contexts)

            if len(converted_contexts) == len(loaded_contexts):
                message = f"Loaded {len(loaded_contexts)} existing plane contexts with valid binaries."
                console.echo(message=message, level=LogLevel.SUCCESS)
                return

            console.echo(
                message="Existing binaries are missing. Recreating them from the source TIFF files...",
                level=LogLevel.WARNING,
            )

    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    # Creates RuntimeContext instances for all planes. The outer pipeline entry (run_single_recording_pipeline) or
    # the prepare_single_recording_batch_tool already wrote the shared configuration, acquisition parameters, and
    # per-plane runtime_data.yaml files, so this call is load-only to avoid racing against peer worker processes.
    contexts = resolve_single_recording_contexts(configuration=configuration, persist=False)

    # Resolves the source files, the frame accounting, and the destination binaries before anything is deleted. The
    # resolution runs every check that can reject the recording, from TIFF files that disagree about their frame
    # shape to a plane the frame accounting leaves with no frames, so a rejected recording keeps its results.
    plan = resolve_tiff_conversion_plan(contexts=contexts, workers=workers)

    # The conversion replaces all of the recording's plane binaries from here on, so the outputs describing the
    # previous binaries go first. The per-plane runtime sections are reset alongside them, because registration reads
    # back the bidirectional correction it recorded and would otherwise skip a correction the rebuilt binary needs.
    # Each reset is persisted before the conversion starts, so an interrupted rebuild leaves every runtime record
    # agreeing with the results the sweep removed instead of still advertising them.
    _clear_downstream_data(output_root=output_root)
    for context in contexts:
        context.runtime.registration = RegistrationData()
        context.runtime.detection = DetectionData()
        context.runtime.extraction = ExtractionData()
        context.runtime.timing = TimingData()
        context.save_runtime()

    convert_tiffs_to_binary(plan=plan)

    # Records the binarization time and saves runtime data for each plane. Each plane's runtime_data.yaml has only
    # one writer here (the single BINARIZE worker for this recording), so the per-plane save is race-free.
    for context in contexts:
        context.runtime.timing.binarization_time = timer.elapsed
        context.save_runtime()

    message = f"Binarization complete. {len(contexts)} plane(s) converted in {timer.elapsed} seconds."
    console.echo(message=message, level=LogLevel.SUCCESS)


def register_recording_plane(
    configuration: SingleRecordingConfiguration, plane_index: int, *, workers: int, device: int | None = None
) -> None:
    """Removes motion from the target imaging plane and computes its registration quality metrics.

    Notes:
        The stage writes the registration offsets, the valid pixel ranges, and the bad-frame mask to disk. Multiple
        planes can be registered in parallel, but each plane may use significant memory and CPU resources.

        A plane that already holds its registration outputs is skipped unless 'repeat_registration' is enabled in the
        Registration configuration section. A skip runs no registration work, so it reports the plane as skipped and
        leaves the timing and the worker allocation an earlier run recorded in place.

    Args:
        configuration: The single-recording pipeline configuration.
        plane_index: The index of the imaging plane to register.
        workers: The number of parallel workers allocated to this registration job. Must be a positive integer, which
            the caller resolves before invoking this function.
        device: The zero-based index of the CUDA device on which this job registers the plane. Use None to register the
            plane on the host CPU.

    Raises:
        ValueError: If output_path is not configured, or if the plane contains fewer frames than the processing
            minimum.
        RuntimeError: If one of the plane's binaries carries the marker of an interrupted write.
        TypeError: If the runtime context loader returns multiple contexts for the target plane.
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

    # Resolves the skip decision from the same registration state and configuration the registration stage itself
    # reads, before the call reaches the point where it would replace either. The elapsed time of a skip measures no
    # registration work, so overwriting the recorded timing with it would erase the duration an earlier run measured.
    registration_skipped = (
        context.runtime.registration.is_registered(output_path=context.runtime.io.output_path)
        and not context.configuration.registration.repeat_registration
    )

    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()
    register_plane(context=context, workers=workers, device=device)

    if registration_skipped:
        message = (
            f"Plane {plane_index} registration skipped. The plane is already registered and "
            f"'registration.repeat_registration' is disabled, so its recorded registration time of "
            f"{context.runtime.timing.total_registration_time} seconds stands."
        )
        console.echo(message=message, level=LogLevel.SUCCESS)
        return

    # Records the allocation the stage actually used.
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
        Multiple planes can be processed in parallel, but each plane may use significant memory and CPU resources.

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

    if not context.runtime.registration.is_registered(output_path=context.runtime.io.output_path):
        message = (
            f"Unable to process plane {plane_index}. The plane must be registered before ROI detection, but no "
            f"registration data was found in memory or under the plane's output directory. Run the registration "
            f"stage for this plane before running the processing stage."
        )
        console.error(message=message, error=RuntimeError)

    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    # Confines the linear-algebra backends to the allocated worker budget for the whole stage, matching the Numba mask
    # the context resolver applied. The batch engine pins every worker process to one backend thread, so the classifier
    # fits and the trace algebra outside detection's own limited block would otherwise run single-threaded. Invoked
    # outside that engine, the same code would size those backends to the whole host rather than to the job.
    with threadpool_limits(limits=workers):
        if configuration.roi_detection.enabled:
            detect_plane_rois(context=context, workers=workers)

            # pragma justification: detection always populates ROI statistics on success or raises beforehand.
            if context.runtime.extraction.roi_statistics is not None:  # pragma: no branch
                extract_traces(context=context, workers=workers)
        else:
            message = f"Skipping plane {plane_index} ROI detection (disabled via 'roi_detection.enabled' parameter)."
            console.echo(message=message, level=LogLevel.WARNING)

    # Records the allocation the stage actually used.
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

    Args:
        contexts: The plane contexts to combine, one per imaging plane. Each must carry the runtime data that the
            processing pipeline populates.

    Raises:
        ValueError: If no context is provided, if output_path is not configured, or if no plane carries ROI statistics.
        RuntimeError: If a plane's registered binary path (or channel 2 registered binary path, when the second channel
            is functional) is not set, indicating that registration did not complete successfully.
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

    output_path = resolve_output_path(output_root=root_path)
    combined_data.save(root_path=output_path)
    console.echo(message=f"Combined data saved to: {output_path}.", level=LogLevel.SUCCESS)


def _validate_binaries_are_unmarked(contexts: list[RuntimeContext]) -> None:
    """Verifies that no interrupted write left one of the recording's plane binaries in an indeterminate state.

    Notes:
        An interrupted conversion leaves its binary holding zeros past the last frame it wrote, and an interrupted
        registration leaves one holding motion-corrected frames up to an unknown point and raw frames after it. Only
        the marker beside the binary records either state, so its suffix is what names the interrupted phase.

    Args:
        contexts: The plane contexts whose functional channel binary exists on disk.

    Raises:
        RuntimeError: If a marker sits beside any binary the given planes hold.
    """
    marked_binaries: list[str] = [
        f"'{binary_path}' marked by '{marker_path.name}'"
        for context in contexts
        for binary_path in _resolve_existing_plane_binaries(context=context)
        if (marker_path := resolve_active_binary_marker(binary_path=binary_path)) is not None
    ]

    if not marked_binaries:
        return

    report = natsorted(marked_binaries)
    message = (
        f"Unable to binarize the recording. An interrupted write left {len(marked_binaries)} plane binary file(s) "
        f"holding finished frames up to an unknown point and unfinished frames after it. The marker beside each of "
        f"them names the interrupted phase: {report}. Enable 'file_io.repeat_binarization' to rebuild the recording "
        f"from its source TIFF files, which also clears every marker."
    )
    console.error(message=message, error=RuntimeError)


def _validate_second_channel_binaries(contexts: list[RuntimeContext]) -> None:
    """Verifies that every converted plane of a two-channel recording holds its second channel binary.

    Notes:
        A binary the pipeline opens for writing is created when it does not exist. Registration would therefore fill
        the absent binary with zeros, and every stage after it would measure a black mean image, a zero trace, and a
        colocalization against nothing. The file that write leaves behind carries no marker and matches the recorded
        frame geometry exactly, so the recording reports itself healthy from then on.

        A single-channel recording holds no second channel binary at all, which is why the declared channel count
        gates the check.

    Args:
        contexts: The plane contexts whose functional channel binary exists on disk.

    Raises:
        RuntimeError: If a converted plane of a two-channel recording holds no second channel binary.
    """
    missing_binaries: list[Path] = [
        binary_path
        for context in contexts
        if (binary_path := _resolve_second_channel_binary(context=context)) is not None and not binary_path.exists()
    ]

    if not missing_binaries:
        return

    message = (
        f"Unable to binarize the recording. The acquisition parameters declare two imaging channels, but "
        f"{len(missing_binaries)} converted plane(s) hold no second channel binary: "
        f"{natsorted(str(path) for path in missing_binaries)}. Enable 'file_io.repeat_binarization' to rebuild the "
        f"recording from its source TIFF files, which writes both channels of every plane."
    )
    console.error(message=message, error=RuntimeError)


def _validate_binary_sizes(contexts: list[RuntimeContext]) -> None:
    """Verifies that every plane binary on disk is sized to the frame geometry recorded for its plane.

    Notes:
        The check compares file sizes rather than file contents. The registration stage rewrites a binary in place, so
        it changes what the file holds without changing how large it is, which makes size the one property that stays
        invariant across every stage that consumes the binary. A content check would instead report every registered
        plane as malformed.

        A size mismatch is what a binary damaged outside the pipeline leaves behind, such as one truncated by a
        partial copy or written for a plane geometry the recording no longer uses. The frame count every later stage
        works from is derived by dividing the file size by the frame size, so a short file silently yields a truncated
        movie rather than an error. A write the pipeline itself abandoned partway is caught by the marker beside the
        binary instead, because both stages that write a binary size it to its full frame count before the first
        frame lands.

        Both channels of a plane are held to the same frame count, because binarization consumes whole plane and
        channel interleave cycles and therefore writes the two channels of one plane frame for frame.

    Args:
        contexts: The plane contexts whose functional channel binary exists on disk.

    Raises:
        RuntimeError: If a binary's size disagrees with the frame geometry recorded for its plane.
    """
    malformed: list[Path] = []
    for context in contexts:
        io_data = context.runtime.io
        frame_bytes = io_data.frame_height * io_data.frame_width * _BINARY_ITEM_SIZE

        # A bootstrapped plane records a zero frame geometry until its conversion saves the measured values, and the
        # conversion persists that save before it clears the binarization marker. A run interrupted before the save
        # therefore leaves a plane holding no geometry against which to size its binary, which this skips rather than
        # misreports.
        if frame_bytes <= 0:
            continue

        expected_size = frame_bytes * io_data.frame_count
        malformed.extend(
            path for path in _resolve_existing_plane_binaries(context=context) if path.stat().st_size != expected_size
        )

    if not malformed:
        return

    message = (
        f"Unable to binarize the recording. The size of {len(malformed)} plane binary file(s) disagrees with the "
        f"frame geometry recorded for their plane: {natsorted(str(path) for path in malformed)}. Every later stage "
        f"derives its frame count by dividing the file size by the frame size, so such a binary is consumed as a "
        f"silently truncated movie. Enable 'file_io.repeat_binarization' to rebuild the recording from its source "
        f"TIFF files."
    )
    console.error(message=message, error=RuntimeError)


def _resolve_existing_plane_binaries(context: RuntimeContext) -> tuple[Path, ...]:
    """Returns the channel binaries one plane holds on disk.

    Args:
        context: The plane context whose binaries are resolved.

    Returns:
        The path of every channel binary of the plane that exists on disk.
    """
    io_data = context.runtime.io
    candidates = (io_data.registered_binary_path, io_data.registered_binary_path_channel_2)
    return tuple(path for path in candidates if path is not None and path.exists())


def _resolve_second_channel_binary(context: RuntimeContext) -> Path | None:
    """Returns the second channel binary the recording declares for one plane.

    Notes:
        The declared channel count is the authority rather than the path the plane's own runtime record carries, so a
        recording re-declared as two-channel after its planes were persisted is held to the second binary as well. Such
        a plane carries no second channel path of its own, so the layout name under the plane's output directory stands
        in for it.

    Args:
        context: The plane context whose second channel binary is resolved.

    Returns:
        The path of the plane's second channel binary, or None when the recording declares a single channel or the
        plane record carries no output path.
    """
    if context.acquisition.channel_number <= 1:
        return None

    io_data = context.runtime.io
    if io_data.registered_binary_path_channel_2 is not None:
        return io_data.registered_binary_path_channel_2
    if io_data.output_path is None:
        return None
    return io_data.output_path / CHANNEL_2_BINARY_FILENAME


def _clear_downstream_data(output_root: Path) -> None:
    """Removes every artifact the pipeline derived from the recording's previous plane binaries.

    Notes:
        The conversion replaces every plane binary of the recording, so the offsets, images, traces, and combined
        dataset that earlier runs measured describe frames that no longer exist. Removing the registration reference
        image is what makes the registration stage run again, because that image is one of the three arrays it reads
        before skipping an already registered plane, alongside both rigid offset arrays.

        The combined outputs belong to the recording rather than to one plane, and the combination stage merges every
        plane into them, so rebuilding any plane voids them. The completion marker goes first, which leaves an
        interrupted clearing reporting the recording as unfinished rather than as complete with a payload that is
        partly gone.

        The tracked multi-recording outputs of this recording stay on disk, because they belong to a dataset spanning
        other recordings. Every multi-recording stage resolves its contexts through the combined metadata removed
        above, so they stay unreachable until this recording is processed and combined again.

        The selections those datasets hold for this recording are cleared, because they are the one piece of
        multi-recording state the conversion invalidates. A selection names regions by their position in this
        recording's own region list, and the detection that rebuilds that list produces a different one, so a
        selection carried across the conversion addresses regions the recording no longer holds.

    Args:
        output_root: The output root the caller configured for the recording.
    """
    root_path = resolve_output_path(output_root=output_root)
    (root_path / COMBINED_METADATA_FILENAME).unlink(missing_ok=True)

    # Reads the plane directories off disk rather than deriving them from the contiguous range the recording's plane
    # count spans. A recording re-declared with fewer planes than an earlier run wrote keeps every directory its
    # previous geometry left behind, and each of those directories holds results measured from replaced frames.
    plane_paths = [
        entry
        for entry in root_path.iterdir()
        if entry.is_dir() and parse_plane_specifier(specifier=entry.name) is not None
    ]

    # The combination stage writes the merged result arrays and detection images into the recording's own output
    # directory under the names each plane writes into its own directory, so one sweep covers both scopes.
    for directory in (root_path, *plane_paths):
        _clear_result_arrays(directory=directory)

    clear_recording_selections(cindra_root=root_path)


def _clear_result_arrays(directory: Path) -> None:
    """Removes every result array, detection image, and registration offset file stored under one directory.

    Args:
        directory: The plane output directory, or the recording output directory into which the combination stage
            writes the merged arrays.
    """
    stale_paths = [
        resolve_array_path(root_path=directory, array=result, second_channel=second_channel)
        for result in RecordingArrays
        for second_channel in (False, True)
    ]
    stale_paths.extend(
        resolve_array_path(
            root_path=directory / DETECTION_DATA_DIRECTORY_NAME, array=image, second_channel=second_channel
        )
        for image in DetectionImages
        for second_channel in (False, True)
    )
    stale_paths.extend(
        resolve_array_path(root_path=directory / REGISTRATION_DATA_DIRECTORY_NAME, array=offsets)
        for offsets in RegistrationArrays
    )
    for stale_path in stale_paths:
        stale_path.unlink(missing_ok=True)


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
        The stage labels are parameterized because the two stages report different actions to the user.

    Args:
        configuration: The single-recording pipeline configuration.
        plane_index: The index of the imaging plane whose runtime context is loaded.
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
    if plane_index in configuration.main.ignored_flyback_planes:
        message = f"Skipping the {stage_noun} of the flyback plane {plane_index}."
        console.echo(message=message, level=LogLevel.SUCCESS)
        return None

    if configuration.file_io.output_path is None:
        message = (
            f"Unable to {stage_action} the target plane. The output_path must be configured in the FileIO section of "
            f"the configuration, but it is currently None."
        )
        console.error(message=message, error=ValueError)

    root_path = resolve_output_path(output_root=configuration.file_io.output_path)
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
