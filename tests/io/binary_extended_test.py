"""Contains tests for extended BinaryFile and BinaryFileCombined functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
import pytest
from tifffile import TiffFile

from cindra.io.binary import BinaryFile, BinaryFileCombined

_FRAME_HEIGHT: int = 8
"""The height of each frame used in test binary files."""

_FRAME_WIDTH: int = 8
"""The width of each frame used in test binary files."""


def _create_test_binary(file_path: Path, frame_count: int, height: int, width: int) -> np.ndarray:
    """Creates a test binary file with sequential int16 data and returns the written data array.

    Args:
        file_path: The absolute path where the binary file will be created.
        frame_count: The number of frames to write into the binary file.
        height: The frame height in pixels.
        width: The frame width in pixels.

    Returns:
        The int16 data array that was written to the file.
    """
    data = np.arange(frame_count * height * width, dtype=np.int16).reshape(frame_count, height, width)
    data.tofile(file_path)
    return data


class TestConvertNumpyFileToBinary:
    """Tests for BinaryFile.convert_numpy_file_to_binary."""

    def test_converts_npy_to_binary(self, tmp_path: Path) -> None:
        """Verifies that a .npy file is correctly converted to a .bin file with matching contents."""
        source_path = tmp_path / "source.npy"
        destination_path = tmp_path / "output.bin"
        data = np.arange(100, dtype=np.float64)
        np.save(source_path, data)

        BinaryFile.convert_numpy_file_to_binary(source_file_name=source_path, destination_file_name=destination_path)

        assert destination_path.exists()
        # The binary contents should match what np.load would produce written with tofile.
        loaded = np.fromfile(destination_path, dtype=np.float64)
        np.testing.assert_array_equal(loaded, data)

    def test_nonexistent_source_raises_error(self, tmp_path: Path) -> None:
        """Verifies that a non-existent source file raises a FileNotFoundError."""
        source_path = tmp_path / "nonexistent.npy"
        destination_path = tmp_path / "output.bin"

        with pytest.raises(FileNotFoundError):
            BinaryFile.convert_numpy_file_to_binary(
                source_file_name=source_path, destination_file_name=destination_path
            )

    def test_appends_bin_suffix_if_missing(self, tmp_path: Path) -> None:
        """Verifies that the .bin suffix is appended when the destination path lacks it."""
        source_path = tmp_path / "source.npy"
        destination_path = tmp_path / "output"
        data = np.arange(50, dtype=np.float32)
        np.save(source_path, data)

        BinaryFile.convert_numpy_file_to_binary(source_file_name=source_path, destination_file_name=destination_path)

        expected_path = tmp_path / "output.bin"
        assert expected_path.exists()


class TestWriteTiff:
    """Tests for BinaryFile.write_tiff."""

    def test_writes_full_tiff(self, tmp_path: Path) -> None:
        """Verifies that binary data is correctly written to a BigTiff and can be read back."""
        frame_count = 5
        binary_path = tmp_path / "data.bin"
        data = _create_test_binary(
            file_path=binary_path, frame_count=frame_count, height=_FRAME_HEIGHT, width=_FRAME_WIDTH
        )

        tiff_path = tmp_path / "output.tiff"
        with BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=binary_path) as binary_file:
            binary_file.write_tiff(file_name=tiff_path)

        assert tiff_path.exists()
        with TiffFile(tiff_path) as tiff:
            tiff_data = tiff.asarray()

        assert tiff_data.shape == (frame_count, _FRAME_HEIGHT, _FRAME_WIDTH)
        np.testing.assert_array_equal(tiff_data, data)

    def test_writes_frame_range_subset(self, tmp_path: Path) -> None:
        """Verifies that writing a subset of frames produces a TIFF with the correct frame count."""
        frame_count = 10
        binary_path = tmp_path / "data.bin"
        data = _create_test_binary(
            file_path=binary_path, frame_count=frame_count, height=_FRAME_HEIGHT, width=_FRAME_WIDTH
        )

        tiff_path = tmp_path / "subset.tiff"
        with BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=binary_path) as binary_file:
            binary_file.write_tiff(file_name=tiff_path, frame_range=slice(2, 5))

        assert tiff_path.exists()
        with TiffFile(tiff_path) as tiff:
            tiff_data = tiff.asarray()

        assert tiff_data.shape[0] == 3
        np.testing.assert_array_equal(tiff_data, data[2:5])

    def test_appends_tiff_suffix_if_missing(self, tmp_path: Path) -> None:
        """Verifies that the .tiff suffix is appended when the output path lacks it."""
        frame_count = 3
        binary_path = tmp_path / "data.bin"
        _create_test_binary(file_path=binary_path, frame_count=frame_count, height=_FRAME_HEIGHT, width=_FRAME_WIDTH)

        output_path = tmp_path / "output"
        with BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=binary_path) as binary_file:
            binary_file.write_tiff(file_name=output_path)

        expected_path = tmp_path / "output.tiff"
        assert expected_path.exists()


class TestBinaryFileCombined:
    """Tests for the BinaryFileCombined class."""

    def test_reads_combined_frames_from_two_planes(self, tmp_path: Path) -> None:
        """Verifies that frames from two planes are correctly assembled into a combined array."""
        plane_height = 4
        plane_width = 4
        frame_count = 3

        # Creates two binary files representing two imaging planes.
        path_1 = tmp_path / "plane0.bin"
        data_1 = np.ones((frame_count, plane_height, plane_width), dtype=np.int16) * 10
        data_1.tofile(path_1)

        path_2 = tmp_path / "plane1.bin"
        data_2 = np.ones((frame_count, plane_height, plane_width), dtype=np.int16) * 20
        data_2.tofile(path_2)

        combined_height = plane_height * 2
        combined_width = plane_width

        combined = BinaryFileCombined(
            height=combined_height,
            width=combined_width,
            plane_heights=np.array([plane_height, plane_height], dtype=np.uint16),
            plane_widths=np.array([plane_width, plane_width], dtype=np.uint16),
            plane_y_coordinates=np.array([0, plane_height], dtype=np.int32),
            plane_x_coordinates=np.array([0, 0], dtype=np.int32),
            file_paths=[path_1, path_2],
        )

        result = combined[slice(0, frame_count)]
        combined.close()

        assert result.shape == (frame_count, combined_height, combined_width)
        # The top half should contain plane 1 data (10s).
        np.testing.assert_array_equal(result[:, :plane_height, :], 10)
        # The bottom half should contain plane 2 data (20s).
        np.testing.assert_array_equal(result[:, plane_height:, :], 20)

    def test_context_manager_opens_and_closes(self, tmp_path: Path) -> None:
        """Verifies that the context manager protocol correctly opens and closes file handles."""
        plane_height = 4
        plane_width = 4
        frame_count = 2

        path_1 = tmp_path / "plane0.bin"
        np.ones((frame_count, plane_height, plane_width), dtype=np.int16).tofile(path_1)

        with BinaryFileCombined(
            height=plane_height,
            width=plane_width,
            plane_heights=np.array([plane_height], dtype=np.uint16),
            plane_widths=np.array([plane_width], dtype=np.uint16),
            plane_y_coordinates=np.array([0], dtype=np.int32),
            plane_x_coordinates=np.array([0], dtype=np.int32),
            file_paths=[path_1],
        ) as combined:
            # Verifies it can read data while context manager is active. The __getitem__ with a slice
            # always returns a 3D array (frames, height, width).
            data = combined[slice(0, 1)]
            assert data.shape == (1, plane_height, plane_width)

    def test_frame_number_property(self, tmp_path: Path) -> None:
        """Verifies that the frame_number property returns the correct count."""
        plane_height = 4
        plane_width = 4
        frame_count = 7

        path_1 = tmp_path / "plane0.bin"
        np.ones((frame_count, plane_height, plane_width), dtype=np.int16).tofile(path_1)

        with BinaryFileCombined(
            height=plane_height,
            width=plane_width,
            plane_heights=np.array([plane_height], dtype=np.uint16),
            plane_widths=np.array([plane_width], dtype=np.uint16),
            plane_y_coordinates=np.array([0], dtype=np.int32),
            plane_x_coordinates=np.array([0], dtype=np.int32),
            file_paths=[path_1],
        ) as combined:
            assert combined.frame_number == frame_count

    def test_shape_property(self, tmp_path: Path) -> None:
        """Verifies that the shape property returns the correct structure."""
        plane_height = 6
        plane_width = 8
        frame_count = 5

        path_1 = tmp_path / "plane0.bin"
        np.ones((frame_count, plane_height, plane_width), dtype=np.int16).tofile(path_1)

        plane_heights = np.array([plane_height], dtype=np.uint16)
        plane_widths = np.array([plane_width], dtype=np.uint16)

        with BinaryFileCombined(
            height=plane_height,
            width=plane_width,
            plane_heights=plane_heights,
            plane_widths=plane_widths,
            plane_y_coordinates=np.array([0], dtype=np.int32),
            plane_x_coordinates=np.array([0], dtype=np.int32),
            file_paths=[path_1],
        ) as combined:
            frame_number, heights, widths = combined.shape
            assert frame_number == frame_count
            np.testing.assert_array_equal(heights, plane_heights)
            np.testing.assert_array_equal(widths, plane_widths)

    def test_raises_error_for_mismatched_frame_counts(self, tmp_path: Path) -> None:
        """Verifies that a ValueError is raised when plane binary files have different frame counts."""
        plane_height = 4
        plane_width = 4

        path_1 = tmp_path / "plane0.bin"
        np.ones((5, plane_height, plane_width), dtype=np.int16).tofile(path_1)

        path_2 = tmp_path / "plane1.bin"
        np.ones((7, plane_height, plane_width), dtype=np.int16).tofile(path_2)

        with pytest.raises(ValueError, match="Unable to create a new BinaryFileCombined"):
            BinaryFileCombined(
                height=plane_height * 2,
                width=plane_width,
                plane_heights=np.array([plane_height, plane_height], dtype=np.uint16),
                plane_widths=np.array([plane_width, plane_width], dtype=np.uint16),
                plane_y_coordinates=np.array([0, plane_height], dtype=np.int32),
                plane_x_coordinates=np.array([0, 0], dtype=np.int32),
                file_paths=[path_1, path_2],
            )

    def test_byte_number_property(self, tmp_path: Path) -> None:
        """Verifies that the byte_number property returns correct sizes for each managed file."""
        plane_height = 4
        plane_width = 4
        frame_count = 3

        path_1 = tmp_path / "plane0.bin"
        np.ones((frame_count, plane_height, plane_width), dtype=np.int16).tofile(path_1)

        path_2 = tmp_path / "plane1.bin"
        np.ones((frame_count, plane_height, plane_width), dtype=np.int16).tofile(path_2)

        combined = BinaryFileCombined(
            height=plane_height * 2,
            width=plane_width,
            plane_heights=np.array([plane_height, plane_height], dtype=np.uint16),
            plane_widths=np.array([plane_width, plane_width], dtype=np.uint16),
            plane_y_coordinates=np.array([0, plane_height], dtype=np.int32),
            plane_x_coordinates=np.array([0, 0], dtype=np.int32),
            file_paths=[path_1, path_2],
        )

        byte_numbers = combined.byte_number
        combined.close()

        # Each file has frame_count * plane_height * plane_width * 2 bytes (int16 = 2 bytes).
        expected_bytes = frame_count * plane_height * plane_width * 2
        assert byte_numbers.shape == (2,)
        assert byte_numbers[0] == expected_bytes
        assert byte_numbers[1] == expected_bytes
