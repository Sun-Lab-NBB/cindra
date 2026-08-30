"""Contains tests for the tiff module helper functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path
import pytest
from tifffile import TiffFile, TiffWriter
from ataraxis_base_utilities import error_format

from cindra.io.tiff import _read_tiff, _discover_tiff_files, resolve_source_frame_geometry


class TestDiscoverTiffFiles:
    """Tests the suffixes, ignored stems, and natural ordering of the flat TIFF scan, and the paths it rejects."""

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

        expected_message = f"Unable to discover TIFF files. The path is not a directory: {file_path}."
        with pytest.raises(ValueError, match=error_format(message=expected_message)):
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
    """Tests the frame batches the TIFF reader returns and the int16 conversion each source dtype receives."""

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
            result = _read_tiff(tiff=tiff, start_index=0, batch_size=5, decode_workers=1)

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
            result = _read_tiff(tiff=tiff, start_index=100, batch_size=5, decode_workers=1)

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
            result = _read_tiff(tiff=tiff, start_index=5, batch_size=10, decode_workers=1)

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
            result = _read_tiff(tiff=tiff, start_index=0, batch_size=1, decode_workers=1)

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
            result = _read_tiff(tiff=tiff, start_index=0, batch_size=1, decode_workers=1)

        assert result is not None
        assert result.dtype == np.int16
        # 60000 // 2 = 30000, which fits in int16.
        assert result[0, 0, 0] == 30000

    @pytest.mark.parametrize(
        ("dtype", "boundary_values"),
        [
            (np.uint16, (0, 1, 2, 3, 65534, 65535)),
            (np.int32, (np.iinfo(np.int32).min, -65537, -3, -1, 0, 1, 65535, 65536, np.iinfo(np.int32).max)),
        ],
    )
    def test_source_dtype_is_halved_and_saturated(
        self, tmp_path: Path, dtype: type[np.generic], boundary_values: tuple[int, ...]
    ) -> None:
        """Verifies that uint16 and int32 pages are halved and saturated into the int16 range."""
        generator = np.random.default_rng(seed=42)
        pages = [np.full((4, 5), fill_value=value, dtype=dtype) for value in boundary_values]
        pages.append(
            generator.integers(np.iinfo(dtype).min, np.iinfo(dtype).max, size=(4, 5), endpoint=True).astype(dtype)
        )
        source = np.stack(pages)

        tiff_path = tmp_path / "source.tif"
        with TiffWriter(tiff_path) as writer:
            for page in pages:
                writer.write(page)

        with TiffFile(tiff_path) as tiff:
            result = _read_tiff(tiff=tiff, start_index=0, batch_size=len(pages), decode_workers=1)

        expected = np.clip(source // 2, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)
        assert result is not None
        assert result.dtype == np.int16
        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("dtype", [np.float32, np.float64])
    def test_float_pages_are_truncated_toward_zero_without_rescaling(
        self, tmp_path: Path, dtype: type[np.generic]
    ) -> None:
        """Verifies that float pages keep their magnitude and are truncated toward zero rather than rounded."""
        # Most values are chosen so that truncation and rounding disagree, and so that halving the page would change
        # the result. -0.9 truncates to 0 but rounds to -1, 5.999 truncates to 5 but rounds to 6, and 1200.7
        # truncates to 1200 but would land on 600 if the float arm were folded into the uint16 halving branch. 2.5
        # and 0.0 truncate and round alike, and 0.0 also survives a halving unchanged, so they pin the arm rather
        # than separate it.
        page_values = (-0.9, 5.999, -3.5, 2.5, 1200.7, -1200.7, 0.0)
        pages = [np.full((4, 5), fill_value=value, dtype=dtype) for value in page_values]

        tiff_path = tmp_path / f"{np.dtype(dtype).name}.tif"
        with TiffWriter(tiff_path) as writer:
            for page in pages:
                writer.write(page)

        with TiffFile(tiff_path) as tiff:
            result = _read_tiff(tiff=tiff, start_index=0, batch_size=len(pages), decode_workers=1)

        assert result is not None
        assert result.dtype == np.int16
        # The expected values are the hand-derived truncations of the page values above, which is what the C cast
        # performed by the conversion produces. np.rint would instead give [-1, 6, -4, 2, 1201, -1201, 0].
        expected_column = np.array([0, 5, -3, 2, 1200, -1200, 0], dtype=np.int16)
        np.testing.assert_array_equal(result, np.tile(expected_column[:, None, None], (1, 4, 5)))
        np.testing.assert_array_equal(result, np.trunc(np.stack(pages)).astype(np.int16))

    @pytest.mark.parametrize("dtype", [np.uint8, np.int8])
    def test_narrow_integer_pages_are_widened_without_rescaling(self, tmp_path: Path, dtype: type[np.generic]) -> None:
        """Verifies that 8-bit pages are widened to int16 verbatim rather than halved like the wider dtypes."""
        # Odd values expose a halving fold: 201 // 2 is 100, and 127 // 2 is 63, so neither survives a rescale.
        page_values = (0, 3, 127, 201) if dtype is np.uint8 else (0, 3, 127, -128)
        pages = [np.full((4, 5), fill_value=value, dtype=dtype) for value in page_values]
        source = np.stack(pages)

        tiff_path = tmp_path / f"{np.dtype(dtype).name}.tif"
        with TiffWriter(tiff_path) as writer:
            for page in pages:
                writer.write(page)

        with TiffFile(tiff_path) as tiff:
            result = _read_tiff(tiff=tiff, start_index=0, batch_size=len(pages), decode_workers=1)

        assert result is not None
        assert result.dtype == np.int16
        np.testing.assert_array_equal(result, source.astype(np.int16))
        assert [int(value) for value in result[:, 0, 0]] == [int(value) for value in page_values]


class TestSourceFrameGeometry:
    """Tests the frame shape and frame count the source scan derives, and the empty directory it rejects."""

    def test_geometry_follows_the_first_page_header(self, tmp_path: Path) -> None:
        """Verifies that the frame shape and element width are read from the first source file."""
        TestSourceFrameGeometry._write_stack(directory=tmp_path, name="frames_001.tif", pages=4, height=12, width=9)

        geometry = resolve_source_frame_geometry(data_directory=tmp_path)

        assert geometry.frame_height == 12
        assert geometry.frame_width == 9
        assert geometry.element_bytes == 2

    def test_frame_count_scales_the_first_file_over_the_whole_directory(self, tmp_path: Path) -> None:
        """Verifies that the frame count is the first file's page count taken across every file."""
        TestSourceFrameGeometry._write_stack(directory=tmp_path, name="frames_001.tif", pages=5, height=8, width=8)
        TestSourceFrameGeometry._write_stack(directory=tmp_path, name="frames_002.tif", pages=5, height=8, width=8)
        TestSourceFrameGeometry._write_stack(directory=tmp_path, name="frames_003.tif", pages=2, height=8, width=8)

        geometry = resolve_source_frame_geometry(data_directory=tmp_path)

        # The final file is short, so the product is an upper bound on the twelve frames the directory holds.
        assert geometry.frame_count == 15

    def test_ignored_stems_are_excluded_from_the_geometry(self, tmp_path: Path) -> None:
        """Verifies that an ignored file neither supplies the shape nor counts toward the frames."""
        TestSourceFrameGeometry._write_stack(directory=tmp_path, name="zstack.tif", pages=1, height=64, width=64)
        TestSourceFrameGeometry._write_stack(directory=tmp_path, name="frames_001.tif", pages=3, height=8, width=8)

        geometry = resolve_source_frame_geometry(data_directory=tmp_path, ignored_file_names=("zstack",))

        assert geometry.frame_height == 8
        assert geometry.frame_count == 3

    def test_directory_without_sources_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that a directory holding no accepted TIFF file raises."""
        message = f"Unable to find any TIFF files in the data directory: {tmp_path}."

        with pytest.raises(FileNotFoundError, match=error_format(message=message)):
            resolve_source_frame_geometry(data_directory=tmp_path)

    @staticmethod
    def _write_stack(directory: Path, name: str, pages: int, height: int, width: int) -> None:
        """Writes one TIFF file holding the requested number of int16 pages."""
        with TiffWriter(directory / name) as writer:
            for _ in range(pages):
                writer.write(np.zeros((height, width), dtype=np.int16))
