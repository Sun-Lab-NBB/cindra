"""Contains tests for methods provided by the binary.py module."""

import numpy as np
import pytest
from pathlib import Path
from tifffile import TiffFile
from sl_suite2p.io.binary import BinaryFile, BinaryFileCombined


@pytest.fixture
def small_bin(tmp_path):
    """Creates a small temporary .bin file with data."""
    height, width, frames = 10, 20, 8
    path = tmp_path / "test.bin"
    data = np.arange(frames * height * width, dtype=np.int16).reshape(frames, height, width)
    data.tofile(path)
    return height, width, frames, path, data

# BinaryFile Class
def test_convert_numpy_file_to_suite2p_binary(tmp_path):
    """Verifies that a .npy file is correctly converted to a .bin file and error path raises FileNotFoundError."""
    src = tmp_path / "data.npy"
    dst = tmp_path / "out.bin"

    arr = np.arange(12, dtype=np.int16).reshape(3, 2, 2)
    np.save(src, arr)

    # Valid conversion
    BinaryFile.convert_numpy_file_to_suite2p_binary(src, dst)
    reread = np.fromfile(dst, dtype=np.int16)
    assert np.array_equal(reread, arr.flatten())

    # Invalid source path triggers FileNotFoundError
    with pytest.raises(FileNotFoundError):
        BinaryFile.convert_numpy_file_to_suite2p_binary(tmp_path / "missing.npy", dst)

def test_binaryfile_byte_calculations(small_bin):
    """Validates bytes_per_frame_number and byte_number properties."""
    height, width, frames, path, _ = small_bin
    bf = BinaryFile(height, width, path)
    expected_bytes_per_frame = 2 * height * width
    assert bf.bytes_per_frame_number == expected_bytes_per_frame
    assert bf.byte_number == path.stat().st_size
    bf.close()

def test_open_existing_binary_reads_correctly(small_bin):
    """Verifies that BinaryFile reads an existing binary file correctly."""
    height, width, frames, path, data = small_bin

    bf = BinaryFile(height, width, path)
    assert bf.frame_number == frames
    assert bf.shape == (frames, height, width)
    assert np.array_equal(bf.data, data)
    bf.close()


def test_create_new_binary_requires_frame_number(tmp_path):
    """Verifies that BinaryFile raises ValueError if frame_number is missing when creating a new file."""
    new_path = tmp_path / "new.bin"
    # Missing frame_number
    with pytest.raises(ValueError): 
        BinaryFile(10, 20, new_path) 


def test_create_new_binary_and_write(tmp_path):
    """Creates a new BinaryFile with a given frame count and writes data correctly."""
    height, width, frames = 3, 4, 2
    path = tmp_path / "write_test.bin"
    bf = BinaryFile(height, width, path, frame_number=frames)
    assert path.exists()

    arr = np.arange(frames * height * width, dtype=np.int16).reshape(frames, height, width)
    bf[:] = arr
    bf.close()

    reread = np.memmap(path, dtype="int16", mode="r", shape=(frames, height, width))
    assert np.array_equal(reread, arr)

def test_get_set_item_and_dtype_conversion(small_bin):
    """Checks __getitem__ and __setitem__, including dtype auto-conversion."""
    height, width, frames, path, data = small_bin
    bf = BinaryFile(height, width, path)

    subset = bf[1]
    assert subset.shape == (height, width)
    assert np.array_equal(subset, data[1])

    # Data with float type should be converted to int16
    new_vals = np.ones((height, width), dtype=np.float32) * 100.9
    bf[1] = new_vals
    reread = bf[1]
    assert reread.dtype == np.int16
    assert np.all(reread == 100)
    bf.close()


def test_bin_movie_basic(small_bin):
    """Ensures bin_movie correctly bins frames by averaging."""
    height, width, frames, path, data = small_bin
    bf = BinaryFile(height, width, path)

    result = bf.bin_movie(bin_size=2)
    # Expect shape (frames // bin_size, height, width)
    expected_shape = (frames // 2, height, width)
    assert result.shape == expected_shape

    # Validate mean of first two frames
    expected_first = data[:2].mean(axis=0).astype(np.float32)
    np.testing.assert_allclose(result[0], expected_first, rtol=1e-5)
    bf.close()


def test_write_tiff_creates_valid_stack(small_bin, tmp_path):
    """Ensures write_tiff exports a valid BigTIFF stack with correct data."""
    height, width, frames, path, data = small_bin
    bf = BinaryFile(height, width, path)

    tiff_path = tmp_path / "exported.tiff"
    bf.write_tiff(tiff_path)

    with TiffFile(tiff_path) as tif:
        read_data = tif.asarray()
        assert read_data.shape == (frames, height, width)
        assert np.array_equal(read_data, data)
    bf.close()

# BinaryFileCombined Tests
@pytest.fixture
def combined_setup(tmp_path):
    """Creates a fake 2-plane setup for BinaryFileCombined testing."""
    height, width, frames = 4, 5, 2
    plane_heights = np.array([height, height], dtype=int)
    plane_widths = np.array([width, width], dtype=int)
    plane_y_coords = np.array([0, height], dtype=int)
    plane_x_coords = np.array([0, 0], dtype=int)

    paths = []
    for i in range(2):
        path = tmp_path / f"plane_{i}.bin"
        arr = np.full((frames, height, width), fill_value=i + 1, dtype=np.int16)
        arr.tofile(path)
        paths.append(path)

    total_height = height * 2
    total_width = width
    return total_height, total_width, plane_heights, plane_widths, plane_y_coords, plane_x_coords, paths


def test_combined_reads_and_stacks_correctly(combined_setup):
    """Verifies that BinaryFileCombined stitches planes correctly."""
    args = combined_setup
    bfc = BinaryFileCombined(*args)

    # Read all frames
    data = bfc[:]
    total_height, total_width, *_ = args
    assert data.shape[1:] == (total_height, total_width)
    # Upper and lower halves filled with plane-specific values
    assert np.all(data[:, : total_height // 2] == 1)
    assert np.all(data[:, total_height // 2 :] == 2)
    bfc.close()


def test_combined_shape_and_frame_number(combined_setup):
    """Verifies that BinaryFileCombined.shape and frame_number are consistent."""
    args = combined_setup
    binary_file_combined = BinaryFileCombined(*args)
    frame_number, plane_heights, plane_widths = binary_file_combined.shape
    assert isinstance(frame_number, int)
    assert np.array_equal(plane_heights, args[2])
    assert np.array_equal(plane_widths, args[3])
    binary_file_combined.close()

def test_binaryfilecombined_frame_mismatch_raises(tmp_path):
    """Verifies that BinaryFileCombined raises ValueError when frame counts differ."""
    height, width = 2, 2
    # Plane 1 - 2 frames
    plane_1 = tmp_path / "p1.bin"
    np.arange(8, dtype=np.int16).reshape(2, height, width).tofile(plane_1)
    # Plane 2 - 3 frames
    plane_2 = tmp_path / "p2.bin"
    np.arange(12, dtype=np.int16).reshape(3, height, width).tofile(plane_2)

    plane_heights = np.array([height, height])
    plane_widths = np.array([width, width])
    plane_y_coordinates = np.array([0, height])
    plane_x_coordinates = np.array([0, 0])

    with pytest.raises(ValueError):
        BinaryFileCombined(height * 2, width, plane_heights, plane_widths, plane_y_coordinates, plane_x_coordinates, [plane_1, plane_2])