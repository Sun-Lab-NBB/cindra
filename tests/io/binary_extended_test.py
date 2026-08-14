"""Contains tests for extended BinaryFile and BinaryFileCombined functionality."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray
import pytest
from tifffile import TiffFile

from cindra.io.binary import BinaryFile, BinaryFileCombined

_FRAME_HEIGHT: int = 8
"""The height of each frame used in test binary files."""

_FRAME_WIDTH: int = 8
"""The width of each frame used in test binary files."""


def _create_test_binary(file_path: Path, frame_count: int, height: int, width: int) -> NDArray[np.int16]:
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
    """Tests BinaryFile.convert_numpy_file_to_binary."""

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
    """Tests BinaryFile.write_tiff."""

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
    """Tests the BinaryFileCombined class."""

    def test_caps_the_combined_view_at_the_shortest_plane(self, tmp_path: Path) -> None:
        """Verifies that planes of unequal length combine into a view spanning the shortest plane's frames."""
        plane_extent = 4

        # Reproduces plane binaries of unequal length, which the combined view caps rather than reads past.
        long_path = tmp_path / "plane0.bin"
        short_path = tmp_path / "plane1.bin"
        _create_test_binary(file_path=long_path, frame_count=11, height=plane_extent, width=plane_extent)
        _create_test_binary(file_path=short_path, frame_count=10, height=plane_extent, width=plane_extent)

        combined = BinaryFileCombined(
            height=plane_extent * 2,
            width=plane_extent,
            plane_heights=np.array([plane_extent, plane_extent], dtype=np.uint16),
            plane_widths=np.array([plane_extent, plane_extent], dtype=np.uint16),
            plane_y_coordinates=np.array([0, plane_extent], dtype=np.int32),
            plane_x_coordinates=np.array([0, 0], dtype=np.int32),
            file_paths=[long_path, short_path],
        )

        # Every combined frame must be backed by real data on every plane, so the view spans the shorter plane and a
        # read across its full range stays inside both files.
        assert combined.frame_number == 10
        assert combined.shape[0] == 10
        assert combined[slice(0, combined.frame_number)].shape == (10, plane_extent * 2, plane_extent)

        combined.close()

    def test_representation_reports_the_capped_frame_count_and_the_plane_total(self, tmp_path: Path) -> None:
        """Verifies that the representation reports the derived combined geometry rather than any single plane's."""
        plane_extent = 4

        # The three planes hold different frame counts, so a representation reporting any one plane's count, or the
        # longest count, prints a number the combined view cannot read.
        first_path = tmp_path / "plane0.bin"
        second_path = tmp_path / "plane1.bin"
        third_path = tmp_path / "plane2.bin"
        _create_test_binary(file_path=first_path, frame_count=11, height=plane_extent, width=plane_extent)
        _create_test_binary(file_path=second_path, frame_count=6, height=plane_extent, width=plane_extent)
        _create_test_binary(file_path=third_path, frame_count=9, height=plane_extent, width=plane_extent)

        with BinaryFileCombined(
            height=plane_extent * 3,
            width=plane_extent,
            plane_heights=np.array([plane_extent] * 3, dtype=np.uint16),
            plane_widths=np.array([plane_extent] * 3, dtype=np.uint16),
            plane_y_coordinates=np.array([0, plane_extent, plane_extent * 2], dtype=np.int32),
            plane_x_coordinates=np.array([0, 0, 0], dtype=np.int32),
            file_paths=[first_path, second_path, third_path],
        ) as combined:
            representation = repr(combined)
            assert combined.frame_number == 6

        # The frame count is the shortest of 11, 6, and 9, and the plane count is the number of managed files.
        assert representation == (
            f"BinaryFileCombined(height={plane_extent * 3}, width={plane_extent}, plane_count=3, frame_number=6)"
        )

    def test_reads_combined_frames_from_two_planes(self, tmp_path: Path) -> None:
        """Verifies that frames from two planes are correctly assembled into a combined array."""
        plane_height = 4
        plane_width = 4
        frame_count = 3

        plane_0_path = tmp_path / "plane0.bin"
        plane_0_data = np.ones((frame_count, plane_height, plane_width), dtype=np.int16) * 10
        plane_0_data.tofile(plane_0_path)

        plane_1_path = tmp_path / "plane1.bin"
        plane_1_data = np.ones((frame_count, plane_height, plane_width), dtype=np.int16) * 20
        plane_1_data.tofile(plane_1_path)

        combined_height = plane_height * 2
        combined_width = plane_width

        combined = BinaryFileCombined(
            height=combined_height,
            width=combined_width,
            plane_heights=np.array([plane_height, plane_height], dtype=np.uint16),
            plane_widths=np.array([plane_width, plane_width], dtype=np.uint16),
            plane_y_coordinates=np.array([0, plane_height], dtype=np.int32),
            plane_x_coordinates=np.array([0, 0], dtype=np.int32),
            file_paths=[plane_0_path, plane_1_path],
        )

        result = combined[slice(0, frame_count)]
        combined.close()

        assert result.shape == (frame_count, combined_height, combined_width)
        # The top half holds plane 0 data (10s).
        np.testing.assert_array_equal(result[:, :plane_height, :], 10)
        # The bottom half holds plane 1 data (20s).
        np.testing.assert_array_equal(result[:, plane_height:, :], 20)

    def test_reads_combined_frames_from_horizontally_tiled_planes(self, tmp_path: Path) -> None:
        """Verifies that a plane offset along x is pasted into its own column range of the combined frame."""
        plane_height = 4
        plane_width = 4
        frame_count = 3

        plane_0_path = tmp_path / "plane0.bin"
        (np.ones((frame_count, plane_height, plane_width), dtype=np.int16) * 10).tofile(plane_0_path)

        plane_1_path = tmp_path / "plane1.bin"
        (np.ones((frame_count, plane_height, plane_width), dtype=np.int16) * 20).tofile(plane_1_path)

        # The combination stage lays planes out in a roughly square grid, which gives a two-plane recording the
        # x-offsets [0, plane_width] that the multi-recording extraction feeds back into this class.
        combined = BinaryFileCombined(
            height=plane_height,
            width=plane_width * 2,
            plane_heights=np.array([plane_height, plane_height], dtype=np.uint16),
            plane_widths=np.array([plane_width, plane_width], dtype=np.uint16),
            plane_y_coordinates=np.array([0, 0], dtype=np.int32),
            plane_x_coordinates=np.array([0, plane_width], dtype=np.int32),
            file_paths=[plane_0_path, plane_1_path],
        )

        result = combined[slice(0, frame_count)]
        combined.close()

        assert result.shape == (frame_count, plane_height, plane_width * 2)
        # The left tile holds plane 0 data (10s).
        np.testing.assert_array_equal(result[:, :, :plane_width], 10)
        # The right tile holds plane 1 data (20s).
        np.testing.assert_array_equal(result[:, :, plane_width:], 20)

    def test_context_manager_opens_and_closes(self, tmp_path: Path) -> None:
        """Verifies that the context manager protocol correctly opens and closes file handles."""
        plane_height = 4
        plane_width = 4
        frame_count = 2

        plane_0_path = tmp_path / "plane0.bin"
        np.ones((frame_count, plane_height, plane_width), dtype=np.int16).tofile(plane_0_path)

        with BinaryFileCombined(
            height=plane_height,
            width=plane_width,
            plane_heights=np.array([plane_height], dtype=np.uint16),
            plane_widths=np.array([plane_width], dtype=np.uint16),
            plane_y_coordinates=np.array([0], dtype=np.int32),
            plane_x_coordinates=np.array([0], dtype=np.int32),
            file_paths=[plane_0_path],
        ) as combined:
            # Slice indexing always returns a 3D (frames, height, width) array, so the shape check confirms the
            # plane handles are still open.
            data = combined[slice(0, 1)]
            assert data.shape == (1, plane_height, plane_width)

    def test_frame_number_property(self, tmp_path: Path) -> None:
        """Verifies that the frame_number property returns the correct count."""
        plane_height = 4
        plane_width = 4
        frame_count = 7

        plane_0_path = tmp_path / "plane0.bin"
        np.ones((frame_count, plane_height, plane_width), dtype=np.int16).tofile(plane_0_path)

        with BinaryFileCombined(
            height=plane_height,
            width=plane_width,
            plane_heights=np.array([plane_height], dtype=np.uint16),
            plane_widths=np.array([plane_width], dtype=np.uint16),
            plane_y_coordinates=np.array([0], dtype=np.int32),
            plane_x_coordinates=np.array([0], dtype=np.int32),
            file_paths=[plane_0_path],
        ) as combined:
            assert combined.frame_number == frame_count

    def test_shape_property(self, tmp_path: Path) -> None:
        """Verifies that the shape property returns the correct structure."""
        plane_height = 6
        plane_width = 8
        frame_count = 5

        plane_0_path = tmp_path / "plane0.bin"
        np.ones((frame_count, plane_height, plane_width), dtype=np.int16).tofile(plane_0_path)

        plane_heights = np.array([plane_height], dtype=np.uint16)
        plane_widths = np.array([plane_width], dtype=np.uint16)

        with BinaryFileCombined(
            height=plane_height,
            width=plane_width,
            plane_heights=plane_heights,
            plane_widths=plane_widths,
            plane_y_coordinates=np.array([0], dtype=np.int32),
            plane_x_coordinates=np.array([0], dtype=np.int32),
            file_paths=[plane_0_path],
        ) as combined:
            frame_number, heights, widths = combined.shape
            assert frame_number == frame_count
            np.testing.assert_array_equal(heights, plane_heights)
            np.testing.assert_array_equal(widths, plane_widths)

    def test_mismatched_frame_counts_resolve_to_the_shortest_plane(self, tmp_path: Path) -> None:
        """Verifies that widely differing plane frame counts resolve to the shortest plane rather than raising."""
        plane_height = 4
        plane_width = 4

        plane_0_path = tmp_path / "plane0.bin"
        np.ones((5, plane_height, plane_width), dtype=np.int16).tofile(plane_0_path)

        plane_1_path = tmp_path / "plane1.bin"
        np.ones((7, plane_height, plane_width), dtype=np.int16).tofile(plane_1_path)

        combined = BinaryFileCombined(
            height=plane_height * 2,
            width=plane_width,
            plane_heights=np.array([plane_height, plane_height], dtype=np.uint16),
            plane_widths=np.array([plane_width, plane_width], dtype=np.uint16),
            plane_y_coordinates=np.array([0, plane_height], dtype=np.int32),
            plane_x_coordinates=np.array([0, 0], dtype=np.int32),
            file_paths=[plane_0_path, plane_1_path],
        )

        assert combined.frame_number == 5

        combined.close()

    def test_byte_number_property(self, tmp_path: Path) -> None:
        """Verifies that the byte_number property returns correct sizes for each managed file."""
        plane_height = 4
        plane_width = 4
        frame_count = 3

        plane_0_path = tmp_path / "plane0.bin"
        np.ones((frame_count, plane_height, plane_width), dtype=np.int16).tofile(plane_0_path)

        plane_1_path = tmp_path / "plane1.bin"
        np.ones((frame_count, plane_height, plane_width), dtype=np.int16).tofile(plane_1_path)

        combined = BinaryFileCombined(
            height=plane_height * 2,
            width=plane_width,
            plane_heights=np.array([plane_height, plane_height], dtype=np.uint16),
            plane_widths=np.array([plane_width, plane_width], dtype=np.uint16),
            plane_y_coordinates=np.array([0, plane_height], dtype=np.int32),
            plane_x_coordinates=np.array([0, 0], dtype=np.int32),
            file_paths=[plane_0_path, plane_1_path],
        )

        byte_numbers = combined.byte_number
        combined.close()

        # Each file has frame_count * plane_height * plane_width * 2 bytes (int16 = 2 bytes).
        expected_bytes = frame_count * plane_height * plane_width * 2
        assert byte_numbers.shape == (2,)
        assert byte_numbers[0] == expected_bytes
        assert byte_numbers[1] == expected_bytes
