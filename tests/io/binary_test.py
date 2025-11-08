"""Contains tests for methods provided by the binary.py module."""

import numpy as np
import pytest
from pathlib import Path
from tifffile import TiffFile
from sl_suite2p.io.binary import BinaryFile, BinaryFileCombined


@pytest.fixture
def small_bin(tmp_path):
    """Creates a small temporary .bin file with data."""
    height, width, frames = 4, 5, 3
    path = tmp_path / "test.bin"
    data = np.arange(frames * height * width, dtype=np.int16).reshape(frames, height, width)
    data.tofile(path)
    return height, width, frames, path, data

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
        BinaryFile(4, 5, new_path) 


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

    # Write new data with float type — should be converted to int16
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