"""This module provides tools for reading and writing image data stored in suite2p binary (.bin) files."""

import types
from typing import Any
from pathlib import Path

import numpy as np
from tifffile import TiffWriter
from numpy.typing import NDArray
from ataraxis_base_utilities import LogLevel, console


class BinaryFile:
    """Creates or opens a Suite2p BinaryFile (.bin) for reading and/or writing image data.

    The file behaves like a memory-mapped NumPy array and can be converted between suite2p BinaryFile and
    NumPy array format at any time with minimal call API changes.

    Args:
        height: The height of each frame stored inside the file.
        width: The width of each frame stored inside the file.
        file_path: The absolute path of the file to read from or write to.
        frame_number: The total number of frames in the file.
        dtype: The data type to use for pixel values stored inside the file, specified as a NumPy datatype
            string (e.g.: "int16").

    Attributes:
        height: Stores the height of each frame stored inside the file.
        width: Stores the width of each frame stored inside the file.
        file_path: Stores the absolute path to the file managed by this instance.
        dtype: Stores the name of the datatype used by the file values.
        file: Stores the NumPy array instance used to memory-map the contents of the binary file.

    Raises:
        ValueError: If the number of frames is not provided when creating (writing) a new BinaryFile
            instance.
    """

    def __init__(
        self, height: int, width: int, file_path: str | Path, frame_number: int = 0, dtype: str = "int16"
    ) -> None:
        # Initializes class attributes using input arguments.
        self.height: int = height
        self.width: int = width
        self.file_path: Path = Path(file_path)
        self.dtype: str = dtype

        # Checks if the file exists, sets to True if it needs to be written, False if it exists.
        write = not self.file_path.exists()

        # If the file does not exist and the number of frames is not provided, raises a ValueError.
        if write and frame_number == 0:
            message = (
                f"Unable to create a new suite2p BinaryFile {file_path}, as the number of frames to be "
                f"written to the file is not specified (is 0). Provide a non-zero 'frame_number' argument "
                f"value to create a new BinaryFile."
            )
            console.error(message=message, error=ValueError)

        # If the file exists, reads the number of frames from the file metadata.
        elif not write:
            frame_number = self.frame_number

        # Determines the shape of the file using the default order of frames x height x width used by suite2p
        shape = (frame_number, self.height, self.width)

        # Resolves the memory-mapping mode used to initialize the memory-mapped NumPy array for the file's data.
        # If the file does not exist, sets the mode to 'w+'. Otherwise, sets the mode to 'r+'.
        mode = "w+" if write else "r+"

        # Creates a memory-mapped NumPy array to access and interface with the contents of the binary file.
        # noinspection PyTypeChecker
        self.file: NDArray[np.int16] = np.memmap(
            filename=self.file_path,
            dtype=self.dtype,
            mode=mode,
            shape=shape,
        )

    @staticmethod
    def convert_numpy_file_to_suite2p_binary(source_file_name: Path, destination_file_name: Path) -> None:
        """Converts a NumPy file, such as a .npz or .npy file, to a suite2p BinaryFile.

        Args:
            source_file_name: The absolute path to the NumPy .npy or .npz file to convert to suite2p BinaryFile format.
            destination_file_name: The absolute path to the suite2p .bin file to create using the data from the source
                file.

        Raises:
            FileNotFoundError: If the provided NumPy file does not exist.
        """
        if (
            not source_file_name.exists()
            or not source_file_name.is_file()
            or source_file_name.suffix not in (".npy", ".npz")
        ):
            message = (
                f"Unable to create the target suite2p BinaryFile {destination_file_name}, as the source file "
                f"'{source_file_name}' does not exist or is not a valid NumPy file."
            )
            console.error(message=message, error=FileNotFoundError)

        # Ensures that the destination file uses the .bin suffix. If the destination path points to a directory, this
        # converts the directory path to a binary file path by adding the .bin suffix.
        if destination_file_name.suffix == ".bin":
            destination_file_name.with_suffix(".bin")

        # Uses NumPy API to convert the file.
        np.load(source_file_name).tofile(destination_file_name)

    @property
    def bytes_per_frame_number(self) -> int:
        """Returns the memory size, in bytes, reserved by each frame stored inside the file."""
        return int(2 * self.height * self.width)

    @property
    def byte_number(self) -> int:
        """Returns the total number of frames stored in the file."""
        return self.file_path.stat().st_size

    @property
    def frame_number(self) -> int:
        """Returns the total number of frames stored in the file."""
        return int(self.byte_number // self.bytes_per_frame_number)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Returns the dimensions of the data in the file as a tuple of three elements.

        The first element is the number of frames. The second element is the height of each frame. The third
        element is the width of each frame.
        """
        return self.frame_number, self.height, self.width

    @property
    def size(self) -> np.int64:
        """Returns the total number of pixels (values) stored inside the file."""
        return np.prod(np.array(self.shape).astype(np.int64))

    def close(self) -> None:
        """Closes the memory-mapped file view."""
        # noinspection PyProtectedMember,PyUnresolvedReferences
        self.file._mmap.close()

    def __enter__(self) -> "BinaryFile":
        """Supports accessing the file via a context manager by returning self to caller upon entering the context."""
        return self

    def __exit__(
        self,
        execution_type: type[BaseException] | None,
        execution_value: BaseException | None,
        execution_traceback: types.TracebackType | None,
    ) -> None:
        """Ensures the memory-mapped file view is closed upon termination of the context that uses the file."""
        self.close()

    def __setitem__(self, indices: slice | int | tuple[int, ...] | NDArray[Any], data: NDArray[Any]) -> None:
        """Sets data in the binary file at specific indices.

        This method is used to assign data to the binary file at the provided indices. If the data is not in 'int16'
        format, it is restricted to the maximum value that can be represented by a 16-bit signed integer and converted
        to an 'int16'.

        Args:
            indices: A slice, integer, or iterable that specifies the indices at which to write the data.
            data: The data to be written to the specified indices.
        """
        # Checks and converts data type top int16, if needed
        if data.dtype != "int16":
            data = np.minimum(data, 2**15 - 2).astype("int16")

        # Writes data to the memory-mapped file
        self.file[indices] = data

    def __getitem__(self, indices: slice | int | tuple[int, ...] | NDArray[Any]) -> NDArray[np.int16]:
        """Retrieves data from the binary file at the specified indices.

        Args:
            indices: A slice, integer, or iterable that specifies the indices from which to read the data.

        Returns:
            A NumPy array of the data read from the binary file at the specified indices.
        """
        # Directly passes indices to the memory-mapped file
        return self.file[indices]

    @property
    def data(self) -> NDArray[np.int16]:
        """Returns all frames stored inside the file as a NumPy array."""
        return self.file[:]

    # noinspection PyTypeHints
    def bin_movie(
        self,
        bin_size: int,
        x_range: tuple[int, int] | None = None,
        y_range: tuple[int, int] | None = None,
        bad_frames: NDArray[np.bool] | None = None,
        reject_threshold: float = 0.5,
    ) -> NDArray[np.float32]:
        """Bins the frames of the movie (frame sequence) stored inside the file wrapped by this instance.

        This method groups the frames stored inside the file into bins of the size 'bin_size'. Optionally, the method
        also rejects bad frames and crops good frames according to the provided x- and y-dimension ranges.

        Args:
            bin_size: The size of each bin, in frames.
            x_range: A tuple of two elements. The first element is the minimum, and the second element is the maximum
                x-index to include in the output binned dataset. If set to None, no cropping (x or y) is performed.
            y_range: A tuple of two elements. The first element is the minimum, and the second element is the maximum
                y-index to include in the output binned dataset. If set to None, no cropping (x or y) is performed.
            bad_frames: A boolean one-dimensional NumPy array mask that has the same length as the number of frames
                stored inside the BinaryFile managed by this instance. The array should be True at each bad frame and
                False at each good frame.
            reject_threshold: The minimum fraction of good frames to all frames inside the batch for bad frames to be
                discarded. If the fraction of good frames in the batch is less than this threshold, then both bad and
                good frames are kept and binned as part of the batch processing.

        Returns:
            A 3-dimensional NumPy array that stores the binned movie. Specifically, the first dimension specifies the
            bin number (an average of bin_size frames), the second specifies the height, and the third specifies the
            width. In other words, the returned data represents an average of bin_size frames at each consecutive
            time-point.
        """
        # If 'bad_frames' is provided, creates a NumPy array that tracks which frames are good. Otherwise, considers all
        # the frames as good.
        good_frames = ~bad_frames if bad_frames is not None else np.ones(self.frame_number, dtype=bool)

        # Resolves the batch size. It is capped either to the provided maximum limit or the total number of good frames
        # if there are fewer good frames than 500.
        batch_size = min(int(np.sum(good_frames.astype(int))), 500)

        # Bins the frames in batches to reduce memory consumption
        batches: list[NDArray[np.float32]] = []  # Stores frames of each batch
        for batch_index in range(0, self.frame_number, batch_size):
            # Retrieves the frames in the processed batch.
            indices = slice(batch_index, min(batch_index + batch_size, self.frame_number))
            data = self.file[indices]

            # Crops the data if the 'x_range' and 'y_range' are provided.
            if x_range is not None and y_range is not None:
                data = data[:, slice(*y_range), slice(*x_range)]

            # If the fraction of good frames inside the batch is above the threshold, the bad frames are discarded and
            # only good frames are kept in the batch. Otherwise, keeps both good and bad frames.
            good_indices = good_frames[indices]
            if np.mean(good_indices) > reject_threshold:
                data = data[good_indices]

            # If a processed data batch has more frames than bin_size, bins the data. Otherwise, rejects the batch
            # altogether
            if data.shape[0] > bin_size:
                # Retrieves the dimensions of the data after cropping and frame rejection
                frame_number, height, width = data.shape

                # Ensures the number of frames is a multiple of bin_size for even binning. Truncates the data to a size
                # that is divisible by bin_size.
                movie = data[: (frame_number // bin_size) * bin_size]

                # Reshapes movie data into bins and computes the mean for each bin. Also casts the data to float32
                # (from int16) type.
                binned_movie = movie.reshape(-1, bin_size, height, width).astype(np.float32).mean(axis=1)
                batches.extend(binned_movie)

        # Stacks and returns the batches as a single NumPy array representing the binned movie.
        return np.stack(batches, dtype=np.float32)

    def write_tiff(
        self,
        file_name: Path,
        frame_range: slice | None = None,
        y_range: slice | None = None,
        x_range: slice | None = None,
    ) -> None:
        """Writes the contents of the BinaryFile wrapped by this instance into a .tiff file.

        This method can be used to convert a subset of the movie stored in the BinaryFile into a .tiff file for further
        analysis or visualization purposes. Note, the output data is encoded into a single BigTiff stack.

        Args:
            file_name: The absolute path to the output .tiff file.
            frame_range: Slice object specifying which frames to export. If None, exports all frames.
            y_range: Slice object specifying the y (height) range to crop. If None, uses full height.
            x_range: Slice object specifying the x (width) range to crop. If None, uses full width.
        """
        # Ensures that the file name includes the .tiff extension.
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

        # Convert slices to start/stop for range() function
        frame_start, frame_stop, _ = frame_range.indices(frame_number)

        message = (
            f"Converting a subset of {self.file_path.name} BinaryFile data into BigTiff stack... "
            f"Frame range: {frame_range}. y_range: {y_range}. x_range: {x_range}."
        )
        console.echo(message=message, level=LogLevel.INFO)

        # Iterates through the data and writes each frame to the .tiff file as an independent page.
        with TiffWriter(file_name, bigtiff=True) as file:
            # For each selected frame, extracts and crops the frame based on y_range and x_range. After extracting and
            # cropping, writes the frame to the file.
            for index in range(frame_start, frame_stop):
                current_frame = self.file[index, y_range, x_range].astype(np.int16)
                file.write(current_frame, contiguous=True)

        message = f"BigTiff: Saved as {file_name} file."
        console.echo(message=message, level=LogLevel.SUCCESS)


class BinaryFileCombined:
    """Creates or opens a set of Suite2p BinaryFiles (.bin) for reading image data across planes.

    This class allows working with multiple imaging planes, each stored inside a separate Suite2p BinaryFile.
    It provides similar functionality to the BinaryFile class but extends it to handle multiple planes.

    Args:
        plane_heights: The height of the combined ROI, in pixels, obtained by combining all managed planes
            (BinaryFiles). This is the height of the ROI that would be drawn if all managed planes were combined
            into a single image.
        width: The width of the combined ROI, in pixels, obtained by combining all managed planes
            (BinaryFiles). This is the width of the ROI that would be drawn if all managed planes were combined
            into a single image.
        plane_heights: An NumPy array that stores the heights of each plane (BinaryFile) managed by this instance.
        plane_widths: A NumPy array that stores the widths of each plane (BinaryFile) managed by this instance.
        plane_y_coordinates: A NumPy array that stores the top-left-corner pixel y-coordinate of each managed
            plane, relative to the original image from which plane data was extracted.
        plane_x_coordinates: A NumPy array that stores the top-left-corner pixel x-coordinate of each managed
            plane, relative to the original image from which plane data was extracted.
        file_paths: An iterable that stores the absolute paths to the binary files from which to read the plane data.

    Attributes:
        height: Stores the combined height of all managed planes.
        width: Stores the combined width of all managed planes.
        plane_heights: Stores the heights of each plane managed by this instance.
        plane_widths: Stores the widths of each plane managed by this instance.
        plane_y_coordinates: Stores the top-left-corner pixel y-coordinates of each plane managed by this instance.
        plane_x_coordinates: Stores the top-left-corner pixel x-coordinates of each plane managed by this instance.
        file_paths: Stores the absolute paths to the BinaryFiles for each plane managed by this instance.
        files: Stores opened (memory-mapped) BinaryFile instances for each plane managed by this instance.

    Raises:
        ValueError: If the frame count is different across two or more opened BinaryFiles.
    """

    def __init__(
        self,
        height: int,
        width: int,
        plane_heights: NDArray[np.int_],
        plane_widths: NDArray[np.int_],
        plane_y_coordinates: NDArray[np.int_],
        plane_x_coordinates: NDArray[np.int_],
        file_paths: list[str | Path] | tuple[str | Path, ...],
    ) -> None:
        # Initializes class attributes using input arguments.
        self.height: int = height
        self.width: int = width
        self.plane_heights: NDArray[np.int_] = plane_heights
        self.plane_widths: NDArray[np.int_] = plane_widths
        self.plane_y_coordinates: NDArray[np.int_] = plane_y_coordinates
        self.plane_x_coordinates: NDArray[np.int_] = plane_x_coordinates
        self.file_paths: tuple[Path, ...] = tuple([Path(file_path) for file_path in file_paths])

        # Opens BinaryFile instances for requested planes, using the input data.
        self.files = [
            BinaryFile(int(plane_heights), int(plane_widths), file_name)
            for (plane_heights, plane_widths, file_name) in zip(
                self.plane_heights, self.plane_widths, self.file_paths, strict=False
            )
        ]

        # Verifies that all opened files have the same number of frames.
        frame_numbers = [file.frame_number for file in self.files]
        if len(set(frame_numbers)) > 1:
            message = (
                f"Unable to create a new BinaryFileCombined instance from the target files "
                f"{[file.name for file in self.file_paths]} stored under root {self.file_paths[0].parent}, as the "
                f"number of frames across the files does not match."
            )
            console.error(message=message, error=ValueError)

    def __enter__(self) -> "BinaryFileCombined":
        """Supports accessing managed files via a context manager by returning self to caller upon entering the
        context.
        """
        return self

    def __exit__(
        self,
        execution_type: type[BaseException] | None,
        execution_value: BaseException | None,
        execution_traceback: types.TracebackType | None,
    ) -> None:
        """Ensures the memory-mapped files are closed upon termination of the context that uses the files."""
        self.close()

    def close(self) -> None:
        """Closes the memory-mapped file view for all managed plane files."""
        for f in self.files:
            f.close()

    @property
    def byte_number(self) -> NDArray[np.int64]:
        """Returns an array that stores the size of each managed BinaryFile, in bytes."""
        byte_number = np.zeros(len(self.files), np.int64)
        for file_index, file in enumerate(self.files):
            byte_number[file_index] = file.byte_number
        return byte_number

    @property
    def frame_number(self) -> int:
        """Returns the total number of frames stored in each of the managed files.

        This number is always the same across all managed files.
        """
        return self.files[0].frame_number

    @property
    def shape(self) -> tuple[int, NDArray[np.int_], NDArray[np.int_]]:
        """Returns the dimensions of the data stored inside managed files as a tuple of three elements.

        The first element is the total number of frames inside each file, which is the same for all files. The second
        element is an array of plane (frame) heights for each managed file. The third element is the array of plane
        (frame) widths for each managed file.
        """
        return self.frame_number, self.plane_heights, self.plane_widths

    def __getitem__(self, indices: slice | int | tuple[int, ...] | NDArray[Any]) -> NDArray[np.int16]:
        """Retrieves and combines data from multiple binary files at the specified indices.

        Args:
            indices: A slice, integer, or iterable that specifies the indices inside each plane file from which to read
                and combine the data.

        Returns:
            A NumPy array that stores the data sampled at the specified indices from each managed plane file. Note, the
            returned array uses the height and width combined from all managed planes.
        """
        # Reads from the first plane file to determine the number of frames in the processed slice
        first_file_data = self.files[0][indices]
        actual_frames = first_file_data.shape[0]

        # Initializes the combined array using the frame count from the slice
        data = np.zeros((actual_frames, self.height, self.width), dtype="int16")

        # Iterates through each file and copies the relevant data slice(s) into the combined array.
        for file_index, file in enumerate(self.files):
            # Uses the data already read from the first file
            if file_index == 0:
                file_data = first_file_data
            # Reads the requested data slice from each file
            else:
                file_data = file[indices]

            # Overwrites the specific section of the combined file data with the data read from the target file. Note,
            # this assumes that planes do not overlap.
            data[
                :,
                self.plane_y_coordinates[file_index] : self.plane_y_coordinates[file_index]
                + self.plane_heights[file_index],
                self.plane_x_coordinates[file_index] : self.plane_x_coordinates[file_index]
                + self.plane_widths[file_index],
            ] = file_data

        # Returns the combined data.
        return data
