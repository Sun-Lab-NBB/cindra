"""Provides assets for importing, converting, and saving TIFF imaging data."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
from natsort import natsorted
from tifffile import TiffFile
from ataraxis_base_utilities import LogLevel, console

from .binary import (
    BinaryFile,
    clear_binarization_marker,
    clear_registration_marker,
    create_binarization_marker,
)
from .context import find_data_directory
from ..dataclasses import RuntimeContext, AcquisitionParameters  # noqa: TC001

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

TIFF_EXTENSIONS: tuple[str, ...] = ("tif", "tiff", "TIF", "TIFF")
"""The supported TIFF file extensions."""

TIFF_DECODE_CEILING: int = 4
"""The maximum number of TIFF decode threads, measured as the point where added decode threads stop shortening the
conversion. The decode pool never exceeds this value regardless of how many cores the surrounding job holds."""

_MULTIDIMENSIONAL_PROCESSING_THRESHOLD: int = 3
"""The minimum number of image dimensions considered 'multidimensional'."""

_MISMATCH_REPORT_LIMIT: int = 5
"""The maximum number of differently shaped TIFF files named individually in the frame-shape mismatch error."""


@dataclass(frozen=True, slots=True)
class TiffConversionPlan:
    """Describes every source file one TIFF to binary conversion reads and every binary it writes.

    Notes:
        Building a plan performs the whole of the conversion's input resolution, so every input error the conversion
        can raise surfaces before it touches a destination binary. A caller that discards the data the previous
        binaries produced therefore builds the plan first and deletes only once it holds one.

        Every plane of a resolved plan receives the same positive number of frames on every channel the recording
        carries, which is what lets the conversion open each destination binary without checking the count it was
        sized for.
    """

    contexts: tuple[RuntimeContext, ...]
    """The runtime context of every plane the conversion writes, in plane order."""
    tiff_files: tuple[Path, ...]
    """The source TIFF files, in conversion order."""
    total_frames: int
    """The number of frames the source files hold across every plane and channel."""
    converted_frames: int
    """The number of leading source frames the conversion reads, which spans whole interleave cycles alone."""
    batch_size: int
    """The number of frames each read decodes, rounded up to a whole plane and channel interleave cycle."""
    decode_workers: int
    """The number of threads tifffile uses to decode each batch."""
    frame_heights: tuple[int, ...]
    """The height of the frames each plane receives."""
    frame_widths: tuple[int, ...]
    """The width of the frames each plane receives."""
    channel_1_paths: tuple[Path, ...]
    """The functional channel binary each plane is written into."""
    channel_2_paths: tuple[Path, ...]
    """The second channel binary each plane is written into, empty for a single-channel recording."""
    channel_1_frame_counts: tuple[int, ...]
    """The number of functional channel frames each plane receives."""
    channel_2_frame_counts: tuple[int, ...]
    """The number of second channel frames each plane receives, empty for a single-channel recording."""


def resolve_tiff_conversion_plan(contexts: list[RuntimeContext], *, workers: int) -> TiffConversionPlan:
    """Resolves every source file the TIFF to binary conversion reads and every binary it writes.

    Discovers the recording's TIFF files, counts the frames each plane and channel receives, reads the frame geometry
    the source files hold, and names the destination binary of every plane.

    Notes:
        The allocated workers become the TIFF image decode threads, capped at TIFF_DECODE_CEILING.

        The plan budgets whole plane and channel interleave cycles alone, so every plane receives the same count on
        every channel and the frames of an incomplete final cycle are left out of the conversion.

        This resolution runs every check that can reject the recording, down to the frame accounting that leaves the
        recording without one complete interleave cycle. A caller that must discard data derived from the recording's
        previous binaries resolves a plan first, which leaves that data untouched when the recording cannot be
        converted.

    Args:
        contexts: A list of RuntimeContext instances created by resolve_single_recording_contexts(). Each
            context must have valid configuration, acquisition parameters, and IOData with binary file paths
            configured.
        workers: The number of parallel workers allocated to this binarization job. Must be a positive integer, which
            the caller resolves before invoking this function.

    Returns:
        The resolved conversion plan.

    Raises:
        ValueError: If contexts is empty, if data_path is not configured, or if a plane carries no destination binary
            path. Also raised if the discovered TIFF files do not all hold frames of the same shape, or if the frames
            those files hold do not fill one complete plane and channel interleave cycle.
        FileNotFoundError: If no TIFF files are found in the data directory.
    """
    if not contexts:
        message = "Unable to resolve the TIFF conversion plan. At least one RuntimeContext must be provided."
        console.error(message=message, error=ValueError)

    decode_workers = min(workers, TIFF_DECODE_CEILING)

    # Extracts configuration and acquisition from the first context (shared across all contexts).
    configuration = contexts[0].configuration
    acquisition = contexts[0].acquisition

    # Finds the data directory from the configuration's data_path.
    data_path = configuration.file_io.data_path
    if data_path is None:
        message = (
            "Unable to resolve the TIFF conversion plan. The data_path must be configured in the FileIO section of "
            "the configuration, but it is currently None."
        )
        console.error(message=message, error=ValueError)

    data_directory = find_data_directory(data_path=data_path)

    tiff_files = _discover_tiff_files(
        data_directory=data_directory,
        ignored_file_names=configuration.file_io.ignored_file_names,
    )

    # Extracts processing parameters.
    plane_number = acquisition.plane_number
    channel_number = acquisition.channel_number

    # Computes batch size adjusted for planes and channels.
    batch_size = configuration.registration.batch_size
    batch_size = plane_number * channel_number * math.ceil(batch_size / (plane_number * channel_number))

    # Counts the frames the source files hold across every plane and channel and reads the frame shape those files
    # hold, in one pass over each file.
    total_frames, base_height, base_width = _scan_source_frames(tiff_files=tiff_files)

    # Resolves the interleave geometry and the number of frames each plane receives on each channel. One interleave
    # cycle carries one frame of every plane on every channel, so counting whole cycles alone hands every plane and
    # channel the same count and leaves the two channels of a plane aligned frame for frame.
    interleave_stride: int = plane_number * channel_number
    plane_frame_count = _resolve_interleave_frame_count(total_frames=total_frames, interleave_stride=interleave_stride)
    _validate_interleave_frame_count(
        data_directory=data_directory,
        total_frames=total_frames,
        interleave_stride=interleave_stride,
        plane_frame_count=plane_frame_count,
    )

    converted_frames = plane_frame_count * interleave_stride
    discarded_frames = total_frames - converted_frames
    if discarded_frames > 0:
        message = (
            f"Discarding the {discarded_frames} trailing frame(s) of the recording stored in {data_directory}. The "
            f"{total_frames} frame(s) its TIFF files hold end partway through a {interleave_stride} frame plane and "
            f"channel interleave cycle, whose frames reach some planes and channels of the recording and not others."
        )
        console.echo(message=message, level=LogLevel.WARNING)

    channel_1_frame_counts: tuple[int, ...] = (plane_frame_count,) * len(contexts)
    channel_2_frame_counts: tuple[int, ...] = (plane_frame_count,) * len(contexts) if channel_number > 1 else ()

    frame_heights, frame_widths = _resolve_plane_dimensions(
        contexts=contexts,
        acquisition=acquisition,
        base_height=base_height,
        base_width=base_width,
    )

    channel_1_paths, channel_2_paths = _resolve_binary_paths(contexts=contexts)

    return TiffConversionPlan(
        contexts=tuple(contexts),
        tiff_files=tuple(tiff_files),
        total_frames=total_frames,
        converted_frames=converted_frames,
        batch_size=batch_size,
        decode_workers=decode_workers,
        frame_heights=tuple(frame_heights),
        frame_widths=tuple(frame_widths),
        channel_1_paths=channel_1_paths,
        channel_2_paths=channel_2_paths,
        channel_1_frame_counts=channel_1_frame_counts,
        channel_2_frame_counts=channel_2_frame_counts,
    )


def convert_tiffs_to_binary(plan: TiffConversionPlan) -> None:
    """Converts the TIFF files a conversion plan names into cindra binary format for all planes.

    Reads the planned source files in batches and writes the converted frames into each plane's binary files. The
    function handles both standard TIFF data and MROI (Multi-ROI) data automatically based on the acquisition
    parameters stored in the planned contexts.

    Notes:
        Modifies the planned contexts in place, populating frame dimensions, frame counts, and mean images in each
        context's runtime data, and initializing each plane's valid pixel ranges to the full frame.

        The conversion reads the leading frames the plan budgets, which span whole plane and channel interleave cycles
        alone, so the trailing frames of an incomplete final cycle are never decoded.

        Every destination binary carries a mid-write mark for the duration of the conversion and is cleared of it once
        the frame accounting agrees and the file is closed. Each binary is sized to its full frame count when it is
        opened, so an interrupted conversion leaves a correctly sized file whose tail frames are zeros. The mark is
        what makes the binarization stage refuse that file instead of consuming it.

    Args:
        plan: The conversion plan resolved by resolve_tiff_conversion_plan(), which names the source files, the
            destination binaries, and the number of frames each plane receives. Every one of those counts is
            positive, so opening a destination binary raises nothing the resolution has not already rejected.

    Raises:
        RuntimeError: If a plane receives a different number of frames than its binary file was sized for.
    """
    contexts = plan.contexts
    acquisition = contexts[0].acquisition

    # Extracts processing parameters.
    plane_number = acquisition.plane_number
    channel_number = acquisition.channel_number
    is_mroi = acquisition.is_mroi
    functional_channel_index = _resolve_functional_channel_index(context=contexts[0])
    interleave_stride: int = plane_number * channel_number
    second_channel_index = 1 - functional_channel_index

    converted_frames = plan.converted_frames
    tiff_files = plan.tiff_files
    batch_size = plan.batch_size
    decode_workers = plan.decode_workers

    channel_1_binaries, channel_2_binaries = _create_binary_files(plan=plan)

    description = "Converting MROI frames to binary" if is_mroi else "Converting frames to binary"

    # Initializes mean image accumulators and write indices for each context.
    mean_images: list[NDArray[np.float32] | None] = [None] * len(contexts)
    mean_images_channel_2: list[NDArray[np.float32] | None] = [None] * len(contexts)
    write_indices: list[int] = [0] * len(contexts)
    write_indices_channel_2: list[int] = [0] * len(contexts)

    # Tracks the position within the plane/channel interleave cycle across file boundaries. When a TIFF file ends
    # mid-cycle, the next file must continue from the correct interleave position rather than resetting to zero.
    interleave_offset: int = 0

    # Counts the frames the conversion has left to read. Capping every batch by this budget is what keeps the frames
    # of an incomplete final interleave cycle out of the binaries, whatever file boundary they fall behind.
    remaining_frames = converted_frames

    # Processes each TIFF file.
    with console.progress(total=converted_frames, description=description, unit="frames") as progress_bar:
        for tiff_file in tiff_files:
            if remaining_frames <= 0:
                break

            start_index = 0

            # Opens the file through a context manager. tifffile's TiffFile forms reference cycles with its pages, so
            # leaving the handle to garbage collection holds the file descriptor open until the next collection and
            # emits spurious ResourceWarnings for the unclosed file.
            with TiffFile(tiff_file) as tiff:
                while remaining_frames > 0:
                    frames = _read_tiff(
                        tiff=tiff,
                        start_index=start_index,
                        batch_size=min(batch_size, remaining_frames),
                        decode_workers=decode_workers,
                    )
                    if frames is None:
                        break

                    frame_count = frames.shape[0]
                    progress_bar.update(frame_count)
                    remaining_frames -= frame_count

                    # Processes each context (plane or virtual plane).
                    for context_index, context in enumerate(contexts):
                        io_data = context.runtime.io

                        # Determines the physical plane index for frame extraction. A virtual plane index enumerates
                        # every ROI and z-plane combination, so reducing it by the z-plane count recovers the physical
                        # interleave position whose frames that virtual plane receives.
                        if is_mroi:
                            physical_plane_index = (
                                io_data.plane_index if io_data.plane_index is not None else 0
                            ) % plane_number
                            roi_lines = io_data.mroi_lines
                        else:
                            physical_plane_index = context_index % plane_number
                            roi_lines = ()

                        # Writes each channel of this plane from its own interleave position, accounting for the
                        # offset previous files left behind. The two channels are written independently, because a
                        # batch shorter than the interleave cycle can cover the position of one and not the other.
                        target_position = physical_plane_index * channel_number + functional_channel_index
                        write_index, mean_image = _write_interleave_selection(
                            frames=frames,
                            first_frame_index=(target_position - interleave_offset) % interleave_stride,
                            interleave_stride=interleave_stride,
                            roi_lines=roi_lines,
                            binary=channel_1_binaries[context_index],
                            write_index=write_indices[context_index],
                            mean_image=mean_images[context_index],
                        )
                        write_indices[context_index] = write_index
                        mean_images[context_index] = mean_image

                        # Processes channel 2 if applicable.
                        if channel_number > 1:
                            target_position_channel_2 = physical_plane_index * channel_number + second_channel_index
                            write_index, mean_image = _write_interleave_selection(
                                frames=frames,
                                first_frame_index=(target_position_channel_2 - interleave_offset) % interleave_stride,
                                interleave_stride=interleave_stride,
                                roi_lines=roi_lines,
                                binary=channel_2_binaries[context_index],
                                write_index=write_indices_channel_2[context_index],
                                mean_image=mean_images_channel_2[context_index],
                            )
                            write_indices_channel_2[context_index] = write_index
                            mean_images_channel_2[context_index] = mean_image

                    start_index += frame_count

            # Updates the interleave offset for the next file based on the total frames read from this file.
            interleave_offset = (interleave_offset + start_index) % interleave_stride

    # Verifies that every plane received exactly the number of frames its binary was sized for. The conversion reads
    # whole interleave cycles, which deliver that count to every plane and channel whatever the source files hold. A
    # mismatch therefore reports a defect in the accounting rather than a recording the resolution should have rejected.
    for context_index, context in enumerate(contexts):
        expected_counts: list[tuple[str, int, int]] = [
            ("channel 1", write_indices[context_index], plan.channel_1_frame_counts[context_index])
        ]
        if channel_number > 1:
            expected_counts.append(
                ("channel 2", write_indices_channel_2[context_index], plan.channel_2_frame_counts[context_index])
            )
        for channel_name, written_frames, allocated_frames in expected_counts:
            if written_frames != allocated_frames:
                message = (
                    f"Unable to convert the recording's TIFF files to binary format. Plane "
                    f"{context.runtime.io.plane_index} received {written_frames} {channel_name} frames, but its "
                    f"binary file was sized for {allocated_frames} frames."
                )
                console.error(message=message, error=RuntimeError)

    # Closes binary files and updates runtime data in each context. Clearing each binary's mark declares its contents
    # complete, which happens only after every frame has been written and the frame accounting above has agreed.
    for context_index, context in enumerate(contexts):
        channel_1_binaries[context_index].close()
        clear_binarization_marker(binary_path=channel_1_binaries[context_index].file_path)
        if channel_number > 1:
            channel_2_binaries[context_index].close()
            clear_binarization_marker(binary_path=channel_2_binaries[context_index].file_path)

        # Divides each channel's accumulator by the frames that channel received. Every context receives at least one
        # frame on every channel, so the guard against an unpopulated mean image or a zero divisor never fails for
        # valid recordings.
        mean_image = mean_images[context_index]
        if mean_image is not None and write_indices[context_index] > 0:  # pragma: no branch
            mean_image /= write_indices[context_index]

        mean_image_channel_2 = mean_images_channel_2[context_index]
        if mean_image_channel_2 is not None and write_indices_channel_2[context_index] > 0:
            mean_image_channel_2 /= write_indices_channel_2[context_index]

        # Updates IOData with frame dimensions. The mean image is always populated because every context receives at
        # least one frame, so the dimension update always runs for valid recordings.
        io_data = context.runtime.io
        if mean_image is not None:  # pragma: no branch
            io_data.frame_height = mean_image.shape[0]
            io_data.frame_width = mean_image.shape[1]
        io_data.frame_count = write_indices[context_index]

        # Updates DetectionData with mean images.
        context.runtime.detection.mean_image = mean_image
        if channel_number > 1:
            context.runtime.detection.mean_image_channel_2 = mean_image_channel_2

        # Sets initial valid pixel ranges to full frame (registration will update these).
        context.runtime.registration.valid_y_range = (0, io_data.frame_height)
        context.runtime.registration.valid_x_range = (0, io_data.frame_width)

    message = f"Converted {converted_frames} frames across {len(tiff_files)} TIFF files to binary format."
    console.echo(message=message, level=LogLevel.SUCCESS)


def _discover_tiff_files(
    data_directory: Path,
    ignored_file_names: tuple[str, ...] = (),
) -> list[Path]:
    """Discovers TIFF files in the specified directory.

    Notes:
        Performs a non-recursive scan of the data_directory for files with valid TIFF extension aliases.

    Args:
        data_directory: The directory to scan for TIFF files. This should be the same directory that contains the
            acquisition parameters JSON file.
        ignored_file_names: A tuple of file names (without extension) to ignore. Files whose stem matches any of
            these names are excluded from the results.

    Returns:
        A list of absolute paths to TIFF files, sorted naturally by filename.

    Raises:
        ValueError: If the data_directory is not a valid directory.
        FileNotFoundError: If no TIFF files are found in the directory.
    """
    if not data_directory.is_dir():
        message = f"Unable to discover TIFF files. The path is not a directory: {data_directory}."
        console.error(message=message, error=ValueError)

    # Performs non-recursive scan for TIFF files. Uses a set to deduplicate matches on case-insensitive filesystems
    # (e.g., Windows, macOS default) where a single file is returned by multiple case-variant globs.
    discovered_paths: set[Path] = set()
    for extension in TIFF_EXTENSIONS:
        discovered_paths.update(
            file_path.resolve()
            for file_path in data_directory.glob(f"*.{extension}")
            if file_path.stem not in ignored_file_names
        )

    if not discovered_paths:
        message = f"Unable to find any TIFF files in the data directory: {data_directory}."
        console.error(message=message, error=FileNotFoundError)

    file_paths = natsorted(discovered_paths)

    message = f"Found {len(file_paths)} valid TIFF files."
    console.echo(message=message, level=LogLevel.INFO)

    return file_paths


def _read_tiff(tiff: TiffFile, start_index: int, batch_size: int, decode_workers: int) -> NDArray[np.int16] | None:
    """Reads a batch (subset) of frames stored inside the TIFF file wrapped by the input TiffFile instance.

    Args:
        tiff: The TiffFile instance that wraps the .tiff file from which to read the data.
        start_index: Index of the first frame to read.
        batch_size: Maximum number of frames to read in this batch.
        decode_workers: The number of threads tifffile uses to decode the requested frames.

    Returns:
        A 3D NumPy array with shape (frames, height, width) containing the requested frame data, or None if the
        start_index is beyond the end of the file.
    """
    tiff_length = len(tiff.pages)

    if start_index >= tiff_length:
        return None

    frames_to_read = min(tiff_length - start_index, batch_size)
    frames = (
        tiff.asarray(maxworkers=decode_workers)
        if tiff_length == 1
        else tiff.asarray(key=range(start_index, start_index + frames_to_read), maxworkers=decode_workers)
    )

    # Adds extra dimension for single-frame TIFFs to ensure 3D array.
    if len(frames.shape) < _MULTIDIMENSIONAL_PROCESSING_THRESHOLD:
        frames = np.expand_dims(frames, axis=0)

    # Converts to int16, rescaling where possible. Halves uint16 (0 to 65535) and int32 values, then clips to the
    # int16 range (-32768 to 32767) so out-of-range int32 magnitudes saturate instead of wrapping during the cast.
    # A halved uint16 spans 0 to 32767, which already lies inside the int16 range, so only int32 needs the clip.
    if frames.dtype.type in {np.uint16, np.int32}:
        halved = frames // 2
        if frames.dtype.type == np.int32:
            np.clip(halved, a_min=np.iinfo(np.int16).min, a_max=np.iinfo(np.int16).max, out=halved)
        frames = halved.astype(dtype=np.int16)
    elif frames.dtype.type != np.int16:  # pragma: no cover, rare non-standard TIFF dtype such as float
        frames = frames.astype(dtype=np.int16)

    return frames


def _write_interleave_selection(
    frames: NDArray[np.int16],
    first_frame_index: int,
    interleave_stride: int,
    roi_lines: tuple[int, ...],
    binary: BinaryFile,
    write_index: int,
    mean_image: NDArray[np.float32] | None,
) -> tuple[int, NDArray[np.float32] | None]:
    """Writes the frames one interleave position holds in a decoded batch into the destination binary.

    Notes:
        Striding the batch by the interleave period yields a view, so the selection costs no copy of the decoded
        frames. A batch shorter than the interleave cycle covers only part of the cycle and can therefore hold no
        frame of the requested position, which leaves the binary and the accumulator untouched.

    Args:
        frames: The decoded batch of source frames.
        first_frame_index: The index the requested interleave position first occupies inside the batch.
        interleave_stride: The length of the plane and channel interleave cycle.
        roi_lines: The MROI line range each selected frame is cropped to, empty for single-ROI data.
        binary: The destination binary the selected frames are written into.
        write_index: The index inside the destination binary the selected frames are written from.
        mean_image: The mean image accumulator of the written channel, or None before the first frame lands.

    Returns:
        A tuple of the index past the written frames and the mean image accumulator the written frames were added to.
    """
    selection = frames[first_frame_index::interleave_stride]
    if selection.shape[0] == 0:
        return write_index, mean_image

    # For MROI data, slices frames to extract only the ROI lines.
    if roi_lines:
        selection = selection[:, roi_lines[0] : roi_lines[-1] + 1, :]

    # Initializes mean image accumulator on the first batch that reaches this position.
    if mean_image is None:
        mean_image = np.zeros((selection.shape[1], selection.shape[2]), dtype=np.float32)

    frame_count = selection.shape[0]
    binary[write_index : write_index + frame_count] = selection
    mean_image += selection.sum(axis=0, dtype=np.float32)

    return write_index + frame_count, mean_image


def _scan_source_frames(tiff_files: list[Path]) -> tuple[int, int, int]:
    """Counts the frames the recording's TIFF files hold and resolves the frame shape those files share.

    Opens each discovered file once, adds its page count to the recording's frame total, and compares the first page
    of every file against the shape of the first file's first page.

    Notes:
        The conversion sizes every plane binary from the first file's frame shape and assumes the remaining files match
        it. A data directory can also hold TIFF files that are not part of the recording, such as an anatomical
        z-stack, and those are usually shaped differently. Such a file otherwise reaches the conversion loop, where its
        frames fail to broadcast into a binary sized for the recording. The shape error that follows names neither the
        file nor the reason, and it lands after the caller has discarded the results the previous binaries produced.

        Counting the frames and comparing their shape share one open of each source file, because both work off the
        page-offset chain tifffile discovers the first time either one asks for it.

    Args:
        tiff_files: The discovered TIFF files, in conversion order.

    Returns:
        A tuple of the number of frames the files hold, the height of the first file's first frame, and its width.

    Raises:
        ValueError: If the first TIFF file is empty, or if any file holds frames of a different shape than the first
            file.
    """
    total_frames = 0
    base_height = 0
    base_width = 0
    mismatched: list[str] = []

    for file_index, tiff_path in enumerate(tiff_files):
        with TiffFile(tiff_path) as tiff:
            page_count = len(tiff.pages)
            total_frames += page_count

            if file_index == 0 and page_count == 0:
                message = f"Unable to determine frame dimensions. The first TIFF file is empty: {tiff_path}."
                console.error(message=message, error=ValueError)

            page_shape = tiff.pages.first.shape
            frame_shape = (page_shape[-2], page_shape[-1])

        if file_index == 0:
            base_height, base_width = frame_shape
        elif frame_shape != (base_height, base_width):
            mismatched.append(f"'{tiff_path.name}' {frame_shape}")

    if mismatched:
        reported = ", ".join(mismatched[:_MISMATCH_REPORT_LIMIT])
        remainder = len(mismatched) - _MISMATCH_REPORT_LIMIT
        if remainder > 0:
            reported = f"{reported}, and {remainder} more"
        message = (
            f"Unable to determine frame dimensions. Every TIFF file in the data directory must hold frames of the "
            f"same shape, but {len(mismatched)} file(s) differ from the ({base_height}, {base_width}) frames of "
            f"'{tiff_files[0].name}': {reported}. Exclude any file that is not part of the recording, such as an "
            f"anatomical z-stack, through the 'file_io.ignored_file_names' configuration parameter."
        )
        console.error(message=message, error=ValueError)

    return total_frames, base_height, base_width


def _resolve_plane_dimensions(
    contexts: list[RuntimeContext],
    acquisition: AcquisitionParameters,
    base_height: int,
    base_width: int,
) -> tuple[list[int], list[int]]:
    """Returns the frame dimensions the binary of each plane is sized for.

    Args:
        contexts: The list of RuntimeContext instances, one per plane.
        acquisition: The acquisition parameters describing the recording setup.
        base_height: The frame height the recording's source files hold.
        base_width: The frame width the recording's source files hold.

    Returns:
        A tuple of two lists: (heights, widths) where each list has one entry per plane/context.
    """
    heights: list[int] = []
    widths: list[int] = []

    for context in contexts:
        io_data = context.runtime.io

        # For MROI data, the height is determined by the ROI line range.
        if acquisition.is_mroi and io_data.mroi_lines:
            plane_height = io_data.mroi_lines[-1] - io_data.mroi_lines[0] + 1
            plane_width = base_width
        else:
            plane_height = base_height
            plane_width = base_width

        heights.append(plane_height)
        widths.append(plane_width)

    return heights, widths


def _resolve_interleave_frame_count(total_frames: int, interleave_stride: int) -> int:
    """Returns the number of frames the recording delivers to every plane and channel interleave position.

    Notes:
        Frames cycle through the plane and channel positions in a fixed order, so one whole cycle carries exactly one
        frame of every plane on every channel. Counting whole cycles alone therefore hands every position the same
        count and discards the frames of a final incomplete cycle, which is what an acquisition stopped partway
        through a volume leaves behind. Those frames reach some planes and channels and not others, and two channels
        of one plane that hold different frame counts describe no shared moment in time.

    Args:
        total_frames: The total number of frames across every source TIFF file.
        interleave_stride: The length of the plane and channel interleave cycle.

    Returns:
        The number of frames delivered to each interleave position.
    """
    return total_frames // interleave_stride


def _validate_interleave_frame_count(
    data_directory: Path,
    total_frames: int,
    interleave_stride: int,
    plane_frame_count: int,
) -> None:
    """Verifies that the recording's source files hold at least one whole plane and channel interleave cycle.

    Notes:
        The conversion consumes whole interleave cycles, so a recording that holds fewer frames than one cycle
        delivers no frames to any plane on any channel. The conversion sizes each destination binary by that count,
        and a binary sized for no frames is rejected only when it is opened, which is after the caller has discarded
        the results the previous binaries produced.

    Args:
        data_directory: The directory holding the recording's source TIFF files.
        total_frames: The number of frames the source files hold across every plane and channel.
        interleave_stride: The length of the plane and channel interleave cycle.
        plane_frame_count: The number of frames every plane receives on every channel.

    Raises:
        ValueError: If the source files hold fewer frames than one whole interleave cycle.
    """
    if plane_frame_count == 0:
        message = (
            f"Unable to resolve the TIFF conversion plan for the recording stored in {data_directory}. The "
            f"{total_frames} frame(s) the recording's TIFF files hold do not fill one {interleave_stride} frame "
            f"plane and channel interleave cycle, so no plane receives any frames."
        )
        console.error(message=message, error=ValueError)


def _resolve_functional_channel_index(context: RuntimeContext) -> int:
    """Returns the interleave position the recording's functional channel occupies inside one plane.

    Args:
        context: Any plane context of the recording, which carries the shared configuration and acquisition
            parameters.

    Returns:
        The zero-based position of the functional channel within a plane's channel group.
    """
    if context.acquisition.channel_number == 1:
        return 0
    return 0 if context.configuration.main.first_channel_functional else 1


def _resolve_binary_paths(contexts: list[RuntimeContext]) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    """Returns the binaries the conversion writes each plane's channels into.

    Args:
        contexts: The list of RuntimeContext instances, one per plane. Each context must have IOData with binary file
            paths configured.

    Returns:
        A tuple of two tuples. The first holds the functional channel binary of every plane, and the second holds the
        second channel binary of every plane, which is empty for a single-channel recording.

    Raises:
        ValueError: If a plane carries no path for a binary the conversion writes.
    """
    has_two_channels = contexts[0].acquisition.channel_number > 1

    channel_1_paths: list[Path] = []
    channel_2_paths: list[Path] = []
    for context in contexts:
        io_data = context.runtime.io

        registered_path = io_data.registered_binary_path
        if registered_path is None:
            message = (
                f"Unable to resolve the binary file of plane {io_data.plane_index}. The registered_binary_path is not "
                f"configured in IOData."
            )
            console.error(message=message, error=ValueError)
        channel_1_paths.append(registered_path)

        if has_two_channels:
            registered_path_channel_2 = io_data.registered_binary_path_channel_2
            if registered_path_channel_2 is None:
                message = (
                    f"Unable to resolve the channel 2 binary file of plane {io_data.plane_index}. The "
                    f"registered_binary_path_channel_2 is not configured in IOData."
                )
                console.error(message=message, error=ValueError)
            channel_2_paths.append(registered_path_channel_2)

    return tuple(channel_1_paths), tuple(channel_2_paths)


def _create_binary_files(plan: TiffConversionPlan) -> tuple[list[BinaryFile], list[BinaryFile]]:
    """Creates BinaryFile instances for writing converted TIFF data for each plane.

    Notes:
        Removes each plane's previous binary and marks the replacement as being mid-binarization, which the caller
        clears once every frame has landed. This is the first step of the conversion that changes anything on disk.

    Args:
        plan: The resolved conversion plan, which names every destination binary and the frames it receives.

    Returns:
        A tuple of two lists. The first list contains BinaryFile instances for channel 1 (one per plane). The second
        list contains BinaryFile instances for channel 2 (empty if single channel).
    """
    channel_1_binary_files: list[BinaryFile] = []
    channel_2_binary_files: list[BinaryFile] = []

    # Creates BinaryFile instances for each plane based on the paths resolved into the plan.
    for context_index, channel_1_path in enumerate(plan.channel_1_paths):
        height = plan.frame_heights[context_index]
        width = plan.frame_widths[context_index]

        # Removes the previous binary so that the new one is sized by the frame count the plan resolved. BinaryFile
        # reads the frame count out of the file whenever the file already exists, so converting over a damaged binary
        # would otherwise inherit its wrong length instead of replacing it.
        channel_1_path.unlink(missing_ok=True)

        # Drops the mark an interrupted registration left behind. That mark described the binary just unlinked, and
        # the replacement built from the source TIFFs carries no motion correction for it to describe.
        clear_registration_marker(binary_path=channel_1_path)

        # Marks the destination for the duration of the conversion, which convert_tiffs_to_binary clears once every
        # frame has landed. The memory map below sizes the file to its full frame count before the first frame is
        # written. An interrupted conversion therefore leaves a correctly sized binary whose tail is zeros, which no
        # size check can tell apart from a finished one. The mark is the only record of that state.
        create_binarization_marker(binary_path=channel_1_path)

        channel_1_binary_files.append(
            BinaryFile(
                height=height,
                width=width,
                file_path=channel_1_path,
                frame_number=plan.channel_1_frame_counts[context_index],
            )
        )

        # Creates channel 2 binary file if applicable.
        if plan.channel_2_paths:
            channel_2_path = plan.channel_2_paths[context_index]
            channel_2_path.unlink(missing_ok=True)
            clear_registration_marker(binary_path=channel_2_path)
            create_binarization_marker(binary_path=channel_2_path)

            channel_2_binary_files.append(
                BinaryFile(
                    height=height,
                    width=width,
                    file_path=channel_2_path,
                    frame_number=plan.channel_2_frame_counts[context_index],
                )
            )

    return channel_1_binary_files, channel_2_binary_files
