"""Contains tests for the tiff module helper functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
import pytest
from tifffile import TiffFile, TiffWriter

from cindra.io.tiff import _read_tiff, _discover_tiff_files


class TestDiscoverTiffFiles:
    """Tests for _discover_tiff_files."""

    def test_discovers_tif_and_tiff_files(self, tmp_path: Path) -> None:
        """Verifies that both .tif and .tiff files are discovered."""
        (tmp_path / "image_001.tif").write_bytes(b"fake")
        (tmp_path / "image_002.tiff").write_bytes(b"fake")
        (tmp_path / "notes.txt").write_bytes(b"not a tiff")

        result = _discover_tiff_files(data_directory=tmp_path)

        assert len(result) == 2
        stems = {path.stem for path in result}
        assert "image_001" in stems
        assert "image_002" in stems

    def test_ignored_file_names_filters_correctly(self, tmp_path: Path) -> None:
        """Verifies that files matching ignored_file_names are excluded from results."""
        (tmp_path / "good_image.tif").write_bytes(b"fake")
        (tmp_path / "bad_image.tif").write_bytes(b"fake")
        (tmp_path / "another.tiff").write_bytes(b"fake")

        result = _discover_tiff_files(data_directory=tmp_path, ignored_file_names=("bad_image",))

        assert len(result) == 2
        stems = {path.stem for path in result}
        assert "bad_image" not in stems
        assert "good_image" in stems
        assert "another" in stems

    def test_no_tiff_files_raises_error(self, tmp_path: Path) -> None:
        """Verifies that a FileNotFoundError is raised when no TIFF files are found."""
        (tmp_path / "data.csv").write_bytes(b"not a tiff")

        with pytest.raises(FileNotFoundError):
            _discover_tiff_files(data_directory=tmp_path)

    def test_non_directory_path_raises_error(self, tmp_path: Path) -> None:
        """Verifies that a ValueError is raised when the path is not a directory."""
        file_path = tmp_path / "not_a_directory.txt"
        file_path.write_bytes(b"file content")

        with pytest.raises(ValueError, match="Unable to"):
            _discover_tiff_files(data_directory=file_path)

    def test_results_are_naturally_sorted(self, tmp_path: Path) -> None:
        """Verifies that discovered files are returned in natural sort order."""
        (tmp_path / "image_10.tif").write_bytes(b"fake")
        (tmp_path / "image_2.tif").write_bytes(b"fake")
        (tmp_path / "image_1.tif").write_bytes(b"fake")

        result = _discover_tiff_files(data_directory=tmp_path)

        stems = [path.stem for path in result]
        assert stems == ["image_1", "image_2", "image_10"]

    def test_empty_directory_raises_error(self, tmp_path: Path) -> None:
        """Verifies that an empty directory raises a FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            _discover_tiff_files(data_directory=tmp_path)


class TestReadTiff:
    """Tests for _read_tiff."""

    def test_reads_batch_of_frames(self, tmp_path: Path) -> None:
        """Verifies that a batch of frames is correctly read from a multi-frame TIFF."""
        tiff_path = tmp_path / "multi_frame.tif"
        frame_count = 10
        height = 16
        width = 16
        data = np.arange(frame_count * height * width, dtype=np.int16).reshape(frame_count, height, width)

        with TiffWriter(tiff_path) as writer:
            for frame_index in range(frame_count):
                writer.write(data[frame_index])

        with TiffFile(tiff_path) as tiff:
            result = _read_tiff(tiff=tiff, start_index=0, batch_size=5)

        assert result is not None
        assert result.shape == (5, height, width)
        assert result.dtype == np.int16

    def test_start_index_beyond_file_returns_none(self, tmp_path: Path) -> None:
        """Verifies that reading beyond the file length returns None."""
        tiff_path = tmp_path / "small.tif"
        data = np.zeros((3, 8, 8), dtype=np.int16)

        with TiffWriter(tiff_path) as writer:
            for frame_index in range(3):
                writer.write(data[frame_index])

        with TiffFile(tiff_path) as tiff:
            result = _read_tiff(tiff=tiff, start_index=100, batch_size=5)

        assert result is None

    def test_reads_partial_batch_at_end(self, tmp_path: Path) -> None:
        """Verifies that a partial batch is returned when fewer frames remain than the batch size."""
        tiff_path = tmp_path / "partial.tif"
        frame_count = 7
        height = 8
        width = 8
        data = np.ones((frame_count, height, width), dtype=np.int16) * 100

        with TiffWriter(tiff_path) as writer:
            for frame_index in range(frame_count):
                writer.write(data[frame_index])

        with TiffFile(tiff_path) as tiff:
            result = _read_tiff(tiff=tiff, start_index=5, batch_size=10)

        assert result is not None
        # Only 2 frames remain starting at index 5.
        assert result.shape[0] == 2

    def test_single_frame_tiff_returns_3d_array(self, tmp_path: Path) -> None:
        """Verifies that a single-frame TIFF produces a 3D array with shape (1, height, width)."""
        tiff_path = tmp_path / "single.tif"
        height = 12
        width = 12
        data = np.ones((height, width), dtype=np.int16) * 42

        with TiffWriter(tiff_path) as writer:
            writer.write(data)

        with TiffFile(tiff_path) as tiff:
            result = _read_tiff(tiff=tiff, start_index=0, batch_size=1)

        assert result is not None
        assert result.ndim == 3
        assert result.shape == (1, height, width)

    def test_uint16_data_is_rescaled_to_int16(self, tmp_path: Path) -> None:
        """Verifies that uint16 data is divided by 2 and converted to int16."""
        tiff_path = tmp_path / "uint16.tif"
        height = 8
        width = 8
        data = np.full((height, width), fill_value=60000, dtype=np.uint16)

        with TiffWriter(tiff_path) as writer:
            writer.write(data)

        with TiffFile(tiff_path) as tiff:
            result = _read_tiff(tiff=tiff, start_index=0, batch_size=1)

        assert result is not None
        assert result.dtype == np.int16
        # 60000 // 2 = 30000, which fits in int16.
        assert result[0, 0, 0] == 30000
