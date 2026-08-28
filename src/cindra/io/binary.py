"""Provides assets for reading and writing image data stored in cindra binary (.bin) files."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Self
from pathlib import Path

import numpy as np
from tifffile import TiffWriter
from ataraxis_base_utilities import LogLevel, console

from ..layout import resolve_binarization_marker_name, resolve_registration_marker_name

if TYPE_CHECKING:
    from types import TracebackType

    from numpy.typing import NDArray

_INT16_MAX_VALUE: int = 2**15 - 2
"""The maximum value that can be stored in a signed 16-bit integer, minus a small buffer."""

_DEFAULT_BIN_BATCH_SIZE: int = 500
"""The default maximum batch size for frame binning operations."""

_BINARIZATION_MARKER_CONTENTS: str = (
    "The binarization stage was writing converted frames into the binary this marker sits beside when it was "
    "interrupted. The binary therefore holds converted frames up to some unknown point and unwritten frames after it. "
    "Enable 'file_io.repeat_binarization' and re-run the binarization stage for this recording to rebuild the binary "
    "from its source TIFF files.\n"
)
"""The text written into a binarization marker, so that the marker explains itself to whoever finds it on disk."""

_REGISTRATION_MARKER_CONTENTS: str = (
    "The registration stage was writing motion-corrected frames into the binary this marker sits beside when it was "
    "interrupted. The binary therefore holds corrected frames up to some unknown point and raw frames after it. "
    "Enable 'file_io.repeat_binarization' and re-run the binarization stage for this recording to rebuild the binary "
    "from its source TIFF files.\n"
)
"""The text written into a registration marker, so that the marker explains itself to whoever finds it on disk."""


def create_binarization_marker(binary_path: Path) -> None:
    """Marks a binary as being mid-binarization, which declares its contents indeterminate until the mark is cleared.

    Args:
        binary_path: The path to the binary whose conversion is about to begin.
    """
    _resolve_binarization_marker_path(binary_path=binary_path).write_text(_BINARIZATION_MARKER_CONTENTS)


def clear_binarization_marker(binary_path: Path) -> None:
    """Clears the mid-binarization mark from a binary, which declares its contents consistent again.

    Notes:
        Clearing a marker that does not exist is not an error, so the binarization stage can call this for every binary
        it finishes without first checking whether an earlier interrupted conversion left one behind.

    Args:
        binary_path: The path to the binary to clear the mark from.
    """
    _resolve_binarization_marker_path(binary_path=binary_path).unlink(missing_ok=True)


def create_registration_marker(binary_path: Path) -> None:
    """Marks a binary as being mid-registration, which declares its contents indeterminate until the mark is cleared.

    Args:
        binary_path: The path to the binary whose rewrite is about to begin.
    """
    _resolve_registration_marker_path(binary_path=binary_path).write_text(_REGISTRATION_MARKER_CONTENTS)


def clear_registration_marker(binary_path: Path) -> None:
    """Clears the mid-registration mark from a binary, which declares its contents consistent again.

    Notes:
        Clearing a marker that does not exist is not an error, so the registration stage can call this for every binary
        it finishes without first checking whether an earlier interrupted rewrite left one behind.

    Args:
        binary_path: The path to the binary to clear the mark from.
    """
    _resolve_registration_marker_path(binary_path=binary_path).unlink(missing_ok=True)


def resolve_active_binary_marker(binary_path: Path) -> Path | None:
    """Returns the path of the phase marker sitting beside a plane binary, or None when the binary carries neither.

    Notes:
        Binarization sizes a plane binary to its full frame count the moment it opens it, and registration then
        rewrites that binary in place. An interrupted run of either stage leaves a correctly sized binary holding an
        indeterminate mixture of frames, which nothing but the marker records. Both markers carry the same meaning for
        the pipeline, so every stage that consumes a binary asks this rather than testing one phase's marker.

    Args:
        binary_path: The path to the binary to look for a marker beside.

    Returns:
        The path of the marker guarding the binary, or None when no stage left one there.
    """
    for marker_path in (
        _resolve_binarization_marker_path(binary_path=binary_path),
        _resolve_registration_marker_path(binary_path=binary_path),
    ):
        if marker_path.exists():
            return marker_path
    return None


class BinaryFile:
    """Creates or opens a cindra binary (.bin) for reading and/or writing image data.

    The file behaves like a memory-mapped NumPy array and can be converted between cindra binary and NumPy array
    format at any time with minimal call API changes.

    Args:
        height: The height of each frame stored inside the file.
        width: The width of each frame stored inside the file.
        file_path: The absolute path of the file to read from or write to.
        frame_number: The total number of frames to size a newly created file for. The value is ignored when the
            file already exists, where the frame count is read from the file's size instead.
        dtype: The name of the NumPy data type to use for the pixel values stored inside the file.
        read_only: Determines whether to open an existing file in read-only mode. When enabled, the file is
            memory-mapped without write access and __setitem__ raises a PermissionError.

    Attributes:
        height: Stores the height of each frame stored inside the file.
        width: Stores the width of each frame stored inside the file.
        file_path: Stores the absolute path to the file managed by this instance.
        dtype: Stores the name of the datatype used by the file values.
        _read_only: Stores whether the file was opened in read-only mode.
        file: Stores the NumPy array instance used to memory-map the contents of the binary file.

    Raises:
        ValueError: If the number of frames is not provided when creating (writing) a new BinaryFile
            instance, or if read-only mode is requested for a file that does not exist.
        PermissionError: If __setitem__ is called on a file opened in read-only mode.
    """

    def __init__(
        self,
        height: int,
        width: int,
        file_path: str | Path,
        frame_number: int = 0,
        dtype: str = "int16",
        *,
        read_only: bool = False,
    ) -> None:
        self.height: int = height
        self.width: int = width
        self.file_path: Path = Path(file_path)
        self.dtype: str = dtype
        self._read_only: bool = read_only

        write = not self.file_path.exists()

        if write and read_only:
            message = (
                f"Unable to open the BinaryFile {file_path} in read-only mode, as the file does not exist. "
                f"Read-only mode is only supported for existing files."
            )
            console.error(message=message, error=ValueError)

        if write and frame_number == 0:
            message = (
                f"Unable to create a new cindra binary {file_path}, as the number of frames to be "
                f"written to the file is not specified (is 0). Provide a non-zero 'frame_number' argument "
                f"value to create a new BinaryFile."
            )
            console.error(message=message, error=ValueError)

        elif not write:
            frame_number = self.frame_number

        shape = (frame_number, self.height, self.width)

        if write:
            mode = "w+"
        elif read_only:
            mode = "r"
        else:
            mode = "r+"

        self.file: np.memmap[Any, np.dtype[np.int16]] = np.memmap(  # type: ignore[call-overload]
            filename=str(self.file_path),
            dtype=self.dtype,
            mode=mode,
            shape=shape,
        )

    def __repr__(self) -> str:
        """Returns a string representation of the BinaryFile instance."""
        return (
            f"BinaryFile(file_path={self.file_path}, height={self.height}, width={self.width}, "
            f"dtype={self.dtype}, read_only={self._read_only})"
        )

    def __enter__(self) -> Self:
        """Returns self to enable use as a context manager."""
        return self

    def __exit__(
        self,
        execution_type: type[BaseException] | None,
        execution_value: BaseException | None,
        execution_traceback: TracebackType | None,
    ) -> None:
        """Ensures the memory-mapped file view is closed upon termination of the context that uses the file."""
        self.close()

    def __setitem__(self, indices: slice | int | tuple[int, ...] | NDArray[Any], data: NDArray[np.int16]) -> None:
        """Sets data in the binary file at specific indices.

        If the data is not in 'int16' format, its values are capped at one below the maximum value representable by
        a 16-bit signed integer (32766) and the data is converted to an 'int16' format. The cap acts on the upper
        bound alone, because the source imaging data is non-negative and therefore already sits above the int16
        minimum.

        Args:
            indices: The indices at which to write the data.
            data: The data to be written to the specified indices.

        Raises:
            PermissionError: If the file was opened in read-only mode.
        """
        if self._read_only:
            message = f"Unable to write data to the BinaryFile {self.file_path}. The file was opened in read-only mode."
            console.error(message=message, error=PermissionError)

        if data.dtype != "int16":
            data = np.minimum(data, _INT16_MAX_VALUE).astype(dtype="int16")

        self.file[indices] = data

    def __getitem__(self, indices: slice | int | tuple[int, ...] | NDArray[Any]) -> NDArray[np.int16]:
        """Retrieves data from the binary file at the specified indices.

        Args:
            indices: The indices from which to read the data.

        Returns:
            The data read from the binary file at the specified indices.
        """
        return self.file[indices]

    @staticmethod
    def convert_numpy_file_to_binary(source_file_name: Path, destination_file_name: Path) -> None:
        """Converts a NumPy .npy file to a cindra binary.

        Args:
            source_file_name: The absolute path to the NumPy .npy file to convert to cindra binary format.
            destination_file_name: The absolute path to the cindra .bin file to create using the data from the source
                file.

        Raises:
            FileNotFoundError: If the provided NumPy file does not exist, is not a regular file, or does not use the
                .npy extension.
        """
        if not source_file_name.exists() or not source_file_name.is_file() or source_file_name.suffix != ".npy":
            message = (
                f"Unable to create the target cindra binary {destination_file_name}, as the source file "
                f"'{source_file_name}' does not exist or is not a valid NumPy file."
            )
            console.error(message=message, error=FileNotFoundError)

        if destination_file_name.suffix != ".bin":
            destination_file_name = destination_file_name.with_suffix(".bin")

        np.load(source_file_name).tofile(destination_file_name)

    def close(self) -> None:
        """Closes the memory-mapped file view."""
        self.file._mmap.close()  # type: ignore[attr-defined]

    @property
    def bytes_per_frame(self) -> int:
        """Returns the memory size, in bytes, reserved by each frame stored inside the file."""
        return int(np.dtype(self.dtype).itemsize * self.height * self.width)

    @property
    def byte_number(self) -> int:
        """Returns the total number of bytes stored in the file."""
        return self.file_path.stat().st_size

    @property
    def frame_number(self) -> int:
        """Returns the total number of frames stored in the file."""
        return int(self.byte_number // self.bytes_per_frame)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Returns the dimensions of the data in the file as (frame_number, height, width)."""
        return self.frame_number, self.height, self.width

    @property
    def size(self) -> np.int64:
        """Returns the total number of pixels (values) stored inside the file."""
        return np.prod(np.array(self.shape).astype(dtype=np.int64))

    @property
    def data(self) -> NDArray[np.int16]:
        """Returns all frames stored inside the file as a NumPy array."""
        return self.file[:]

    def subsample_movie(
        self,
        sample_count: int,
        x_range: tuple[int, int] | None = None,
        y_range: tuple[int, int] | None = None,
    ) -> NDArray[np.float32]:
        """Subsamples the movie by selecting evenly-spaced frames across the recording.

        Args:
            sample_count: The number of frames to sample from the movie. The actual number of returned frames is
                min(sample_count, frame_number).
            x_range: A tuple of (start, end) indices for cropping frames along the x-axis. Cropping is applied only
                when both x_range and y_range are provided. If set to None, no cropping (x or y) is performed.
            y_range: A tuple of (start, end) indices for cropping frames along the y-axis. Cropping is applied only
                when both x_range and y_range are provided. If set to None, no cropping (x or y) is performed.

        Returns:
            The subsampled and optionally cropped frames, shaped as (sample_count, height, width), where the leading
            dimension is capped at the file's frame count.
        """
        actual_samples = min(sample_count, self.frame_number)
        indices = np.linspace(start=0, stop=self.frame_number - 1, num=actual_samples).astype(dtype=np.intp)

        movie = self.file[indices]

        if y_range is not None and x_range is not None:
            movie = movie[:, y_range[0] : y_range[1], x_range[0] : x_range[1]]

        return movie.astype(dtype=np.float32)

    def bin_movie(
        self,
        bin_size: int,
        x_range: tuple[int, int] | None = None,
        y_range: tuple[int, int] | None = None,
        bad_frames: NDArray[np.bool_] | None = None,
        reject_threshold: float = 0.5,
    ) -> NDArray[np.float32]:
        """Bins the frames of the movie (frame sequence) stored inside the file wrapped by this instance.

        Args:
            bin_size: The size of each bin, in frames.
            x_range: A tuple of (start, end) indices for cropping frames along the x-axis, where the end index is
                exclusive. If set to None, no cropping (x or y) is performed.
            y_range: A tuple of (start, end) indices for cropping frames along the y-axis, where the end index is
                exclusive. If set to None, no cropping (x or y) is performed.
            bad_frames: A mask holding one element per frame stored inside the BinaryFile managed by this instance,
                True at each bad frame and False at each good frame.
            reject_threshold: The fraction of good frames to all frames inside the batch that must be exceeded for bad
                frames to be discarded. If the fraction of good frames in the batch does not exceed this threshold,
                then both bad and good frames are kept and binned as part of the batch processing.

        Returns:
            The binned movie, shaped as (bin_number, height, width). Each bin holds the average of bin_size frames taken
            from one batch. A batch left with bin_size or fewer frames is averaged into a single bin instead, and the
            bad frames discarded from a batch are absent from the frames its bins average.
        """
        good_frames = ~bad_frames if bad_frames is not None else np.ones(self.frame_number, dtype=np.bool_)

        # Resolves the batch size. It is capped either to the total number of good frames or the default maximum batch
        # size, whichever is smaller. A movie whose every frame is marked bad has no good frames to count, so the batch
        # floors at a single frame and the below-threshold branch bins the bad frames rather than discarding them.
        batch_size = max(1, min(int(np.sum(good_frames)), _DEFAULT_BIN_BATCH_SIZE))

        # Bins the frames in batches to reduce memory consumption.
        batches: list[NDArray[np.float32]] = []
        for batch_index in range(0, self.frame_number, batch_size):
            indices = slice(batch_index, min(batch_index + batch_size, self.frame_number))
            data = self.file[indices]

            if x_range is not None and y_range is not None:
                data = data[:, slice(*y_range), slice(*x_range)]

            good_indices = good_frames[indices]
            if np.mean(good_indices) > reject_threshold:
                data = data[good_indices]

            # If a processed data batch has more frames than bin_size, bins the data. Otherwise, averages the batch into
            # a single bin to preserve data when there are many bad frames.
            if data.shape[0] > bin_size:
                frame_number, height, width = data.shape

                # Ensures the number of frames is a multiple of bin_size for even binning. Truncates the data to a size
                # that is divisible by bin_size.
                movie = data[: (frame_number // bin_size) * bin_size]

                binned_movie = movie.reshape(-1, bin_size, height, width).astype(dtype=np.float32).mean(axis=1)
                batches.extend(binned_movie)
            elif data.shape[0] > 0:  # pragma: no branch, a batch always retains at least one frame.
                # Averages a batch holding fewer frames than bin_size into a single bin, which preserves the data of
                # a batch left short by many bad frames.
                batches.append(data.astype(dtype=np.float32).mean(axis=0))

        return np.stack(batches)

    def write_tiff(
        self,
        file_name: Path,
        frame_range: slice | None = None,
        y_range: slice | None = None,
        x_range: slice | None = None,
    ) -> None:
        """Writes the contents of the BinaryFile wrapped by this instance into a .tiff file.

        The output data is encoded into a single BigTiff stack.

        Args:
            file_name: The absolute path to the output .tiff file.
            frame_range: The frames to export. If None, exports all frames.
            y_range: The y (height) range to crop. If None, uses full height.
            x_range: The x (width) range to crop. If None, uses full width.
        """
        if file_name.suffix != ".tiff":
            file_name = file_name.with_suffix(".tiff")

        # If explicit range overrides are not provided, defaults to converting the entire file into a large .tiff
        # stack.
        frame_number, height, width = self.shape
        if frame_range is None:
            frame_range = slice(0, frame_number)
        if y_range is None:
            y_range = slice(0, height)
        if x_range is None:
            x_range = slice(0, width)

        # Converts slices to start/stop for the range() function.
        frame_start, frame_stop, _ = frame_range.indices(frame_number)

        message = (
            f"Converting a subset of {self.file_path.name} BinaryFile data into BigTiff stack... "
            f"Frame range: {frame_range}. y_range: {y_range}. x_range: {x_range}."
        )
        console.echo(message=message, level=LogLevel.INFO)

        with TiffWriter(file_name, bigtiff=True) as file:
            for index in range(frame_start, frame_stop):
                current_frame = self.file[index, y_range, x_range].astype(dtype=np.int16)
                file.write(current_frame, contiguous=True)

        message = f"BigTiff: Saved as {file_name} file."
        console.echo(message=message, level=LogLevel.SUCCESS)


class BinaryFileCombined:
    """Opens a collection of existing cindra binaries (.bin) for reading image data across planes.

    Works with multiple imaging planes, each stored inside a separate cindra binary.

    Notes:
        The combined view is capped at the shortest managed file's frame count and a warning is emitted whenever the
        binaries disagree, which keeps every combined frame backed by real data on every plane.

    Args:
        height: The height of the combined ROI, in pixels, obtained by combining all managed planes (BinaryFiles).
            This is the height of the ROI that would be drawn if all managed planes were combined into a single image.
        width: The width of the combined ROI, in pixels, obtained by combining all managed planes (BinaryFiles).
            This is the width of the ROI that would be drawn if all managed planes were combined into a single image.
        plane_heights: The height of each plane (BinaryFile) managed by this instance.
        plane_widths: The width of each plane (BinaryFile) managed by this instance.
        plane_y_coordinates: The top-left-corner pixel y-coordinate of each managed plane, relative to the original
            image from which plane data was extracted.
        plane_x_coordinates: The top-left-corner pixel x-coordinate of each managed plane, relative to the original
            image from which plane data was extracted.
        file_paths: The absolute paths to the binary files from which to read the plane data.

    Attributes:
        height: Stores the combined height of all managed planes.
        width: Stores the combined width of all managed planes.
        plane_heights: Stores the heights of each plane managed by this instance.
        plane_widths: Stores the widths of each plane managed by this instance.
        plane_y_coordinates: Stores the top-left-corner pixel y-coordinates of each plane managed by this instance.
        plane_x_coordinates: Stores the top-left-corner pixel x-coordinates of each plane managed by this instance.
        file_paths: Stores the absolute paths to the BinaryFiles for each plane managed by this instance.
        files: Stores opened (memory-mapped) BinaryFile instances for each plane managed by this instance.
        _frame_number: Stores the number of frames spanned by the combined view, which is the frame count of the
            shortest managed file.
    """

    def __init__(
        self,
        height: int,
        width: int,
        plane_heights: NDArray[np.uint16],
        plane_widths: NDArray[np.uint16],
        plane_y_coordinates: NDArray[np.int32],
        plane_x_coordinates: NDArray[np.int32],
        file_paths: list[Path] | tuple[Path, ...],
    ) -> None:
        self.height: int = height
        self.width: int = width
        self.plane_heights: NDArray[np.uint16] = plane_heights
        self.plane_widths: NDArray[np.uint16] = plane_widths
        self.plane_y_coordinates: NDArray[np.int32] = plane_y_coordinates
        self.plane_x_coordinates: NDArray[np.int32] = plane_x_coordinates
        self.file_paths: tuple[Path, ...] = tuple(Path(path) for path in file_paths)

        self.files: list[BinaryFile] = [
            BinaryFile(height=int(height), width=int(width), file_path=file_path)
            for height, width, file_path in zip(self.plane_heights, self.plane_widths, self.file_paths, strict=False)
        ]

        # Resolves the combined frame count as that of the shortest managed file. Capping the combined view there keeps
        # every combined frame backed by real data on every plane, which matches how the combination stage trims its
        # traces.
        frame_numbers = [file.frame_number for file in self.files]
        self._frame_number: int = min(frame_numbers)
        if len(set(frame_numbers)) > 1:
            console.echo(
                message=(
                    f"Capping the combined view of the plane binaries stored under root {self.file_paths[0].parent} at "
                    f"{self._frame_number} frames. The binaries hold between {self._frame_number} and "
                    f"{max(frame_numbers)} frames, so every frame past that count is backed by some planes alone."
                ),
                level=LogLevel.WARNING,
            )

    def __repr__(self) -> str:
        """Returns a string representation of the BinaryFileCombined instance."""
        return (
            f"BinaryFileCombined(height={self.height}, width={self.width}, "
            f"plane_count={len(self.files)}, frame_number={self._frame_number})"
        )

    def __enter__(self) -> Self:
        """Returns self to enable use as a context manager."""
        return self

    def __exit__(
        self,
        execution_type: type[BaseException] | None,
        execution_value: BaseException | None,
        execution_traceback: TracebackType | None,
    ) -> None:
        """Ensures the memory-mapped files are closed upon termination of the context that uses the files."""
        self.close()

    def __getitem__(self, indices: slice | int | tuple[int, ...] | NDArray[Any]) -> NDArray[np.int16]:
        """Retrieves and combines data from multiple binary files at the specified indices.

        Args:
            indices: The indices inside each plane file from which to read and combine the data.

        Returns:
            The data sampled at the specified indices from each managed plane file, using the height and width
            combined from all managed planes.
        """
        # Reads from the first plane file to determine the number of frames in the processed slice.
        first_file_data = self.files[0][indices]
        actual_frames = first_file_data.shape[0]

        data = np.zeros((actual_frames, self.height, self.width), dtype=np.int16)

        for file_index, file in enumerate(self.files):
            file_data = first_file_data if file_index == 0 else file[indices]

            # Overwrites the specific section of the combined file data with the data read from the target file. Note,
            # this assumes that planes do not overlap.
            data[
                :,
                self.plane_y_coordinates[file_index] : self.plane_y_coordinates[file_index]
                + self.plane_heights[file_index],
                self.plane_x_coordinates[file_index] : self.plane_x_coordinates[file_index]
                + self.plane_widths[file_index],
            ] = file_data

        return data

    def close(self) -> None:
        """Closes the memory-mapped file view for all managed plane files."""
        for file in self.files:
            file.close()

    @property
    def byte_number(self) -> NDArray[np.int64]:
        """Returns an array that stores the size of each managed BinaryFile, in bytes."""
        return np.array([file.byte_number for file in self.files], dtype=np.int64)

    @property
    def frame_number(self) -> int:
        """Returns the number of frames the combined view spans, which is that of the shortest managed file."""
        return self._frame_number

    @property
    def shape(self) -> tuple[int, NDArray[np.uint16], NDArray[np.uint16]]:
        """Returns the dimensions of the managed files as (frame_number, plane_heights, plane_widths), where
        frame_number is the frame count of the shortest managed file and the arrays contain per-file plane dimensions.
        """
        return self.frame_number, self.plane_heights, self.plane_widths


def _resolve_binarization_marker_path(binary_path: Path) -> Path:
    """Returns the path of the marker that flags a binary as being mid-binarization.

    Args:
        binary_path: The path to the binary the marker guards.

    Returns:
        The marker path, which sits beside the binary it guards.
    """
    return binary_path.with_name(resolve_binarization_marker_name(binary_name=binary_path.name))


def _resolve_registration_marker_path(binary_path: Path) -> Path:
    """Returns the path of the marker that flags a binary as being mid-registration.

    Args:
        binary_path: The path to the binary the marker guards.

    Returns:
        The marker path, which sits beside the binary it guards.
    """
    return binary_path.with_name(resolve_registration_marker_name(binary_name=binary_path.name))
