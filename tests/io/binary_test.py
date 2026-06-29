"""Contains tests for classes and methods provided by the binary.py module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import pytest
import tifffile

from cindra.io.binary import BinaryFile

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Test data constants.
_FRAME_HEIGHT: int = 8
"""The height of each frame used in test binary files."""

_FRAME_WIDTH: int = 8
"""The width of each frame used in test binary files."""

_FRAME_COUNT: int = 10
"""The number of frames stored in test binary files."""


def _create_test_binary(file_path: Path, frame_count: int = _FRAME_COUNT) -> NDArray[np.int16]:
    """Creates a test binary file with sequential int16 data and returns the written data array.

    Args:
        file_path: The absolute path where the binary file will be created.
        frame_count: The number of frames to write into the binary file.

    Returns:
        The int16 data array that was written to the file, with shape (frame_count, height, width).
    """
    data = np.arange(frame_count * _FRAME_HEIGHT * _FRAME_WIDTH, dtype=np.int16).reshape(
        frame_count, _FRAME_HEIGHT, _FRAME_WIDTH
    )
    data.tofile(file_path)
    return data


class TestBinaryFileInit:
    """Tests BinaryFile.__init__() constructor behavior."""

    def test_creates_new_binary_file_with_write_mode(self, tmp_path: Path) -> None:
        """Verifies that providing frame_number to the constructor creates a new writable binary file."""
        file_path = tmp_path / "test.bin"
        binary_file = BinaryFile(
            height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path, frame_number=_FRAME_COUNT
        )

        assert binary_file.frame_number == _FRAME_COUNT
        assert binary_file.height == _FRAME_HEIGHT
        assert binary_file.width == _FRAME_WIDTH
        assert file_path.exists()
        binary_file.close()

    def test_opens_existing_binary_file_in_read_mode(self, tmp_path: Path) -> None:
        """Verifies that omitting frame_number opens an existing binary file and infers the frame count."""
        file_path = tmp_path / "test.bin"
        _create_test_binary(file_path=file_path)

        binary_file = BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path)
        assert binary_file.frame_number == _FRAME_COUNT
        binary_file.close()

    def test_opens_existing_file_in_read_only_mode(self, tmp_path: Path) -> None:
        """Verifies that read_only=True opens an existing file without write access."""
        file_path = tmp_path / "test.bin"
        _create_test_binary(file_path=file_path)

        binary_file = BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path, read_only=True)
        assert binary_file.frame_number == _FRAME_COUNT
        assert binary_file._read_only
        binary_file.close()

    def test_raises_error_for_nonexistent_file_without_frame_number(self, tmp_path: Path) -> None:
        """Verifies that opening a nonexistent file without specifying frame_number raises a ValueError."""
        file_path = tmp_path / "nonexistent.bin"
        with pytest.raises(ValueError, match="Unable to"):
            BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path)

    def test_raises_error_for_read_only_nonexistent_file(self, tmp_path: Path) -> None:
        """Verifies that read_only=True raises a ValueError when the file does not exist."""
        file_path = tmp_path / "nonexistent.bin"
        with pytest.raises(ValueError, match="Unable to"):
            BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path, read_only=True)

    def test_stores_file_path_as_path_object(self, tmp_path: Path) -> None:
        """Verifies that a string file_path argument is converted to a Path instance."""
        file_path = tmp_path / "test.bin"
        _create_test_binary(file_path=file_path)

        binary_file = BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=str(file_path))
        assert isinstance(binary_file.file_path, Path)
        assert binary_file.file_path == file_path
        binary_file.close()


class TestBinaryFileProperties:
    """Tests BinaryFile computed properties."""

    def test_bytes_per_frame_returns_correct_size(self, tmp_path: Path) -> None:
        """Verifies that bytes_per_frame returns the number of bytes needed for one frame of int16 data."""
        file_path = tmp_path / "test.bin"
        _create_test_binary(file_path=file_path)

        binary_file = BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path)
        # int16 = 2 bytes per pixel, 8x8 = 64 pixels per frame.
        expected_bytes = 2 * _FRAME_HEIGHT * _FRAME_WIDTH
        assert binary_file.bytes_per_frame == expected_bytes
        binary_file.close()

    def test_byte_number_returns_total_file_size(self, tmp_path: Path) -> None:
        """Verifies that byte_number returns the total size of the binary file in bytes."""
        file_path = tmp_path / "test.bin"
        _create_test_binary(file_path=file_path)

        binary_file = BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path)
        expected_total_bytes = 2 * _FRAME_HEIGHT * _FRAME_WIDTH * _FRAME_COUNT
        assert binary_file.byte_number == expected_total_bytes
        binary_file.close()

    def test_frame_number_returns_correct_count(self, tmp_path: Path) -> None:
        """Verifies that frame_number returns the number of frames derived from file size and frame dimensions."""
        file_path = tmp_path / "test.bin"
        _create_test_binary(file_path=file_path)

        binary_file = BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path)
        assert binary_file.frame_number == _FRAME_COUNT
        binary_file.close()

    def test_shape_returns_frames_height_width_tuple(self, tmp_path: Path) -> None:
        """Verifies that shape returns a tuple of (frame_number, height, width)."""
        file_path = tmp_path / "test.bin"
        _create_test_binary(file_path=file_path)

        binary_file = BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path)
        assert binary_file.shape == (_FRAME_COUNT, _FRAME_HEIGHT, _FRAME_WIDTH)
        binary_file.close()

    def test_size_returns_total_pixel_count(self, tmp_path: Path) -> None:
        """Verifies that size returns the total number of pixel values across all frames."""
        file_path = tmp_path / "test.bin"
        _create_test_binary(file_path=file_path)

        binary_file = BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path)
        expected_size = _FRAME_COUNT * _FRAME_HEIGHT * _FRAME_WIDTH
        assert binary_file.size == expected_size
        binary_file.close()


class TestBinaryFileContextManager:
    """Tests BinaryFile context manager protocol (__enter__, __exit__, close)."""

    def test_enter_returns_self(self, tmp_path: Path) -> None:
        """Verifies that __enter__ returns the BinaryFile instance itself."""
        file_path = tmp_path / "test.bin"
        _create_test_binary(file_path=file_path)

        binary_file = BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path)
        result = binary_file.__enter__()
        assert result is binary_file
        binary_file.close()

    def test_context_manager_closes_file_on_exit(self, tmp_path: Path) -> None:
        """Verifies that the context manager properly closes the memory-mapped file upon exiting the with block."""
        file_path = tmp_path / "test.bin"
        _create_test_binary(file_path=file_path)

        with BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path) as binary_file:
            assert binary_file.frame_number == _FRAME_COUNT

        # numpy memmap exposes no public closed flag; _mmap.closed is the authoritative signal it was released.
        assert binary_file.file._mmap.closed  # type: ignore[attr-defined]

    def test_exit_closes_file_even_with_exception(self, tmp_path: Path) -> None:
        """Verifies that __exit__ closes the memory-mapped file even when an exception occurs inside the context."""
        file_path = tmp_path / "test.bin"
        _create_test_binary(file_path=file_path)

        simulated_error = RuntimeError("Simulated error")
        with (
            pytest.raises(RuntimeError),
            BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path) as binary_file,
        ):
            raise simulated_error

        # numpy memmap exposes no public closed flag; _mmap.closed confirms release even after the exception.
        assert binary_file.file._mmap.closed  # type: ignore[attr-defined]

    def test_close_method_closes_memmap(self, tmp_path: Path) -> None:
        """Verifies that close() terminates the memory-mapped file view."""
        file_path = tmp_path / "test.bin"
        _create_test_binary(file_path=file_path)

        binary_file = BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path)
        binary_file.close()

        assert binary_file.file._mmap.closed  # type: ignore[attr-defined]


class TestBinaryFileSetItem:
    """Tests BinaryFile.__setitem__() write operations."""

    def test_writes_int16_data_to_file(self, tmp_path: Path) -> None:
        """Verifies that int16 data is correctly written to the binary file at the specified frame index."""
        file_path = tmp_path / "test.bin"

        with BinaryFile(
            height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path, frame_number=_FRAME_COUNT
        ) as binary_file:
            frame_data = np.ones((_FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.int16) * 42
            binary_file[0] = frame_data
            np.testing.assert_array_equal(binary_file[0], frame_data)

    def test_writes_slice_of_frames(self, tmp_path: Path) -> None:
        """Verifies that multiple frames can be written using slice indexing."""
        file_path = tmp_path / "test.bin"

        with BinaryFile(
            height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path, frame_number=_FRAME_COUNT
        ) as binary_file:
            frame_data = np.full((3, _FRAME_HEIGHT, _FRAME_WIDTH), fill_value=100, dtype=np.int16)
            binary_file[2:5] = frame_data
            np.testing.assert_array_equal(binary_file[2:5], frame_data)

    def test_converts_non_int16_data_and_clips_values(self, tmp_path: Path) -> None:
        """Verifies that non-int16 data is clipped to the maximum int16 value and cast to int16 before writing."""
        file_path = tmp_path / "test.bin"

        with BinaryFile(
            height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path, frame_number=_FRAME_COUNT
        ) as binary_file:
            # Uses float32 data with values exceeding int16 range.
            large_value_data = np.full((_FRAME_HEIGHT, _FRAME_WIDTH), fill_value=50000.0, dtype=np.float32)
            binary_file[0] = large_value_data

            # The maximum representable int16 value used by cindra is 2**15 - 2 = 32766.
            maximum_int16_value = 2**15 - 2
            result = binary_file[0]
            assert result.dtype == np.int16
            assert np.all(result == maximum_int16_value)

    def test_raises_permission_error_on_read_only_file(self, tmp_path: Path) -> None:
        """Verifies that __setitem__ raises a PermissionError when the file is opened in read-only mode."""
        file_path = tmp_path / "test.bin"
        _create_test_binary(file_path=file_path)

        with BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path, read_only=True) as binary_file:
            frame_data = np.ones((_FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.int16)
            with pytest.raises(PermissionError):
                binary_file[0] = frame_data


class TestBinaryFileGetItemAndData:
    """Tests BinaryFile.__getitem__() and the data property."""

    def test_getitem_returns_single_frame(self, tmp_path: Path) -> None:
        """Verifies that integer indexing returns the correct single frame from the binary file."""
        file_path = tmp_path / "test.bin"
        original_data = _create_test_binary(file_path=file_path)

        with BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path) as binary_file:
            np.testing.assert_array_equal(binary_file[0], original_data[0])

    def test_getitem_returns_frame_slice(self, tmp_path: Path) -> None:
        """Verifies that slice indexing returns the correct subset of frames."""
        file_path = tmp_path / "test.bin"
        original_data = _create_test_binary(file_path=file_path)

        with BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path) as binary_file:
            np.testing.assert_array_equal(binary_file[2:5], original_data[2:5])

    def test_data_property_returns_all_frames(self, tmp_path: Path) -> None:
        """Verifies that the data property returns the complete contents of the binary file."""
        file_path = tmp_path / "test.bin"
        original_data = _create_test_binary(file_path=file_path)

        with BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path) as binary_file:
            np.testing.assert_array_equal(binary_file.data, original_data)


class TestSubsampleMovie:
    """Tests BinaryFile.subsample_movie()."""

    def test_subsamples_evenly_spaced_frames(self, tmp_path: Path) -> None:
        """Verifies that subsample_movie selects evenly-spaced frames across the recording."""
        file_path = tmp_path / "test.bin"
        _create_test_binary(file_path=file_path)

        with BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path) as binary_file:
            result = binary_file.subsample_movie(sample_count=5)

            assert result.shape[0] == 5
            assert result.shape[1] == _FRAME_HEIGHT
            assert result.shape[2] == _FRAME_WIDTH
            assert result.dtype == np.float32

    def test_caps_sample_count_to_frame_number(self, tmp_path: Path) -> None:
        """Verifies that requesting more samples than frames returns at most frame_number frames."""
        file_path = tmp_path / "test.bin"
        _create_test_binary(file_path=file_path)

        with BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path) as binary_file:
            result = binary_file.subsample_movie(sample_count=100)
            assert result.shape[0] == _FRAME_COUNT

    def test_applies_cropping_with_x_and_y_ranges(self, tmp_path: Path) -> None:
        """Verifies that providing x_range and y_range crops the subsampled frames."""
        file_path = tmp_path / "test.bin"
        _create_test_binary(file_path=file_path)

        with BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path) as binary_file:
            result = binary_file.subsample_movie(
                sample_count=5,
                x_range=(1, 5),
                y_range=(2, 6),
            )
            assert result.shape == (5, 4, 4)

    def test_no_cropping_without_ranges(self, tmp_path: Path) -> None:
        """Verifies that omitting x_range and y_range returns full-size frames."""
        file_path = tmp_path / "test.bin"
        _create_test_binary(file_path=file_path)

        with BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path) as binary_file:
            result = binary_file.subsample_movie(sample_count=5)
            assert result.shape == (5, _FRAME_HEIGHT, _FRAME_WIDTH)


class TestBinMovie:
    """Tests BinaryFile.bin_movie()."""

    def test_bins_frames_by_averaging(self, tmp_path: Path) -> None:
        """Verifies that bin_movie groups consecutive frames and averages each bin."""
        file_path = tmp_path / "test.bin"
        data = _create_test_binary(file_path=file_path)

        with BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path) as binary_file:
            result = binary_file.bin_movie(bin_size=5)

            assert result.dtype == np.float32
            # 10 frames with bin_size=5 should produce 2 bins.
            assert result.shape == (2, _FRAME_HEIGHT, _FRAME_WIDTH)
            # The arange input is deterministic, so each bin equals the exact mean of its five source frames.
            np.testing.assert_allclose(result[0], data[0:5].mean(axis=0))
            np.testing.assert_allclose(result[1], data[5:10].mean(axis=0))

    def test_bins_with_cropping(self, tmp_path: Path) -> None:
        """Verifies that bin_movie applies x_range and y_range cropping to binned frames."""
        file_path = tmp_path / "test.bin"
        data = _create_test_binary(file_path=file_path)

        with BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path) as binary_file:
            result = binary_file.bin_movie(
                bin_size=5,
                x_range=(0, 4),
                y_range=(0, 4),
            )
            assert result.shape == (2, 4, 4)
            # Cropping selects the top-left 4x4 region of each frame before averaging the 5-frame bins.
            np.testing.assert_allclose(result[0], data[0:5, 0:4, 0:4].mean(axis=0))
            np.testing.assert_allclose(result[1], data[5:10, 0:4, 0:4].mean(axis=0))

    def test_bins_with_bad_frames_rejected(self, tmp_path: Path) -> None:
        """Verifies that bin_movie excludes bad frames when the good frame fraction exceeds the threshold."""
        file_path = tmp_path / "test.bin"
        data = _create_test_binary(file_path=file_path)

        # Marks frames 0 and 1 as bad (2 out of 10, so 80% good > 50% threshold).
        bad_frames = np.zeros(_FRAME_COUNT, dtype=np.bool_)
        bad_frames[0] = True
        bad_frames[1] = True

        with BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path) as binary_file:
            result = binary_file.bin_movie(bin_size=2, bad_frames=bad_frames)

            assert result.dtype == np.float32
            # 8 good frames with bin_size=2 should produce 4 bins.
            assert result.shape[0] == 4
            # Bad frames 0 and 1 are dropped, leaving good frames 2..9 binned into pairs (2,3), (4,5), (6,7), (8,9).
            np.testing.assert_allclose(result[0], data[2:4].mean(axis=0))
            np.testing.assert_allclose(result[3], data[8:10].mean(axis=0))

    def test_bins_without_bad_frames(self, tmp_path: Path) -> None:
        """Verifies that bin_movie treats all frames as good when bad_frames is not provided."""
        file_path = tmp_path / "test.bin"
        _create_test_binary(file_path=file_path)

        with BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path) as binary_file:
            result = binary_file.bin_movie(bin_size=2)

            # 10 frames with bin_size=2 produces 5 bins.
            assert result.shape[0] == 5

    def test_bins_preserves_data_when_too_many_bad_frames(self, tmp_path: Path) -> None:
        """Verifies that bin_movie keeps all frames when the good frame fraction is below the reject threshold."""
        file_path = tmp_path / "test.bin"
        # Creates a binary with 20 frames so that batch_size is large enough to trigger the below-threshold path.
        frame_count = 20
        data = np.arange(frame_count * _FRAME_HEIGHT * _FRAME_WIDTH, dtype=np.int16).reshape(
            frame_count, _FRAME_HEIGHT, _FRAME_WIDTH
        )
        data.tofile(file_path)

        # Marks 15 out of 20 frames as bad (25% good < 50% threshold), so no frames are rejected within batches.
        bad_frames = np.ones(frame_count, dtype=np.bool_)
        bad_frames[:5] = False

        with BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path) as binary_file:
            result = binary_file.bin_movie(bin_size=2, bad_frames=bad_frames)

            assert result.dtype == np.float32
            # batch_size = min(5, 500) = 5. Each batch has 5 frames. 20 // 5 = 4 batches.
            # Mean good fraction per batch is <= 0.5, so all frames kept. 5 frames with bin_size=2 yields
            # floor(5/2) = 2 bins per batch. 4 batches * 2 bins = 8 bins total.
            assert result.shape[0] == 8

    def test_bins_small_batch_averaged_into_single_bin(self, tmp_path: Path) -> None:
        """Verifies that a batch smaller than bin_size is averaged into a single bin to preserve data."""
        file_path = tmp_path / "test.bin"
        small_frame_count = 3
        data = np.ones((small_frame_count, _FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.int16) * 10
        data.tofile(file_path)

        with BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path) as binary_file:
            # Uses bin_size larger than the frame count, triggering the single-bin averaging path.
            result = binary_file.bin_movie(bin_size=5)

            assert result.dtype == np.float32
            assert result.shape == (1, _FRAME_HEIGHT, _FRAME_WIDTH)
            # All frames have value 10, so the averaged bin should also be 10.
            np.testing.assert_allclose(result[0], 10.0)

    def test_bin_movie_averages_correctly(self, tmp_path: Path) -> None:
        """Verifies that the binned output contains the correct mean of each bin's frames."""
        file_path = tmp_path / "test.bin"
        # Creates 4 frames with known constant values for straightforward average verification.
        frame_count = 4
        data = np.empty((frame_count, _FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.int16)
        data[0] = 10
        data[1] = 20
        data[2] = 30
        data[3] = 40
        data.tofile(file_path)

        with BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path) as binary_file:
            result = binary_file.bin_movie(bin_size=2)

            # Bin 0: mean(10, 20) = 15.0, Bin 1: mean(30, 40) = 35.0.
            assert result.shape[0] == 2
            np.testing.assert_allclose(result[0], 15.0)
            np.testing.assert_allclose(result[1], 35.0)


class TestWriteTiff:
    """Tests BinaryFile.write_tiff()."""

    def test_writes_tiff_with_explicit_ranges(self, tmp_path: Path) -> None:
        """Verifies that write_tiff crops frames to the explicit y_range and x_range before saving the .tiff stack."""
        file_path = tmp_path / "test.bin"
        data = _create_test_binary(file_path=file_path, frame_count=5)
        tiff_path = tmp_path / "output.tiff"

        with BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path) as binary_file:
            binary_file.write_tiff(file_name=tiff_path, y_range=slice(1, 3), x_range=slice(2, 4))

        # The explicit ranges crop each frame to rows 1:3 and columns 2:4, yielding a (5, 2, 2) stack on read-back.
        tiff_data = tifffile.imread(tiff_path)
        assert tiff_data.shape == (5, 2, 2)
        np.testing.assert_array_equal(tiff_data, data[:, 1:3, 2:4])

    def test_writes_tiff_with_default_ranges(self, tmp_path: Path) -> None:
        """Verifies that write_tiff exports a frame subset at full frame size when y_range and x_range are omitted."""
        file_path = tmp_path / "test.bin"
        data = _create_test_binary(file_path=file_path, frame_count=5)
        tiff_path = tmp_path / "full.tiff"

        with BinaryFile(height=_FRAME_HEIGHT, width=_FRAME_WIDTH, file_path=file_path) as binary_file:
            binary_file.write_tiff(file_name=tiff_path, frame_range=slice(0, 3))

        # Omitting y_range and x_range defaults to the full frame dimensions, exporting the first three frames in full.
        tiff_data = tifffile.imread(tiff_path)
        assert tiff_data.shape == (3, _FRAME_HEIGHT, _FRAME_WIDTH)
        np.testing.assert_array_equal(tiff_data, data[0:3])
