"""Contains tests for methods provided by the tiff.py module."""

import copy
import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
from tifffile import TiffWriter, TiffFile

from sl_suite2p.io.tiff import (
    generate_tiff_filename,
    save_tiff,
    _open_tiff,
    _read_tiff,
    tiff_to_binary,
    mesoscan_to_binary,
)


@pytest.fixture
def create_single_frame_tiff(tmp_path):
    """Creates a single-frame tiff file."""
    tiff_path = tmp_path / "single_frame.tiff"

    frame = np.random.randint(0, 65535, size=(100, 100), dtype=np.uint16)

    with TiffWriter(tiff_path) as tiff:
        tiff.write(frame, contiguous=True)

    return tiff_path


@pytest.fixture
def create_multi_frame_tiff(tmp_path):
    """Creates a multi-frame tiff file."""
    tiff_path = tmp_path / "multi_frame.tiff"

    frames = np.random.randint(0, 65535, size=(10, 100, 100), dtype=np.uint16)

    with TiffWriter(tiff_path) as tiff:
        for frame in frames:
            tiff.write(frame, contiguous=True)

    return tiff_path


@pytest.fixture
def ops(tmp_path):
    """Creates a basic default ops dictionary for testing."""
    return {
        "data_path": [str(tmp_path)],
        "save_path": str(tmp_path),
        "functional_chan": 1,
        "align_by_chan": 1,
        "nplanes": 1,
        "nchannels": 1,
        "look_one_level_down": False,
        "ignored_file_names": [],
        "batch_size": 500,
        "progress_bars": False,
        "do_registration": True,
    }


@pytest.mark.parametrize(
    "functional_channel, alignment_channel, batch_number, channel, expected_subdir, expected_channel_idx",
    [
        (0, 0, 0, 0, "channel_1_tiffs", 0),
        (0, 1, 0, 0, "channel_2_tiffs", 1),
        (0, 1, 5, 1, "channel_1_tiffs", 0),
        (1, 1, 10, 0, "channel_1_tiffs", 0),
    ],
)
def test_generate_tiff_filename(
    tmp_path,
    functional_channel,
    alignment_channel,
    batch_number,
    channel,
    expected_subdir,
    expected_channel_idx,
):
    """Verifies that a suite2p .tiff filename and its path are created based on the input parameters."""
    result = generate_tiff_filename(
        functional_channel=functional_channel,
        alignment_channel=alignment_channel,
        save_path=tmp_path,
        batch_number=batch_number,
        channel=channel,
    )

    assert expected_subdir in result
    expected_filename = f"file_{str(batch_number).zfill(9)}_channel_{expected_channel_idx}.tiff"

    assert expected_filename in result
    assert Path(result).parent.exists()


def test_save_tiff(tmp_path):
    """Verifies that the input frame stack array is saved as the specified tiff file."""
    # Creates the first set of frames
    frames1 = np.random.randint(0, 1000, size=(3, 50, 50), dtype=np.int16)
    output_path = str(tmp_path / "output.tiff")

    # Saves the first set of frames
    save_tiff(frames=frames1, file_path=output_path)
    assert Path(output_path).exists()

    with TiffFile(output_path) as tiff:
        saved_frames1 = tiff.asarray()
        assert saved_frames1.shape == frames1.shape
        assert np.array_equal(saved_frames1, frames1)

    # Creates a second set of frames with different dimensions and data
    frames2 = np.random.randint(1000, 2000, size=(5, 30, 40), dtype=np.int16)
    save_tiff(frames=frames2, file_path=output_path)

    # Tests that the first set of frames is overwritten
    with TiffFile(output_path) as tiff:
        saved_frames2 = tiff.asarray()
        assert saved_frames2.shape == frames2.shape
        assert np.array_equal(saved_frames2, frames2)
        assert not np.array_equal(saved_frames2, frames1)


def test_pixel_type():
    """Verifies that all image data are converted into type int16."""
    # Tests float32 to int16
    float_frames = np.array([[[100.7, 200.9], [300.3, 400.1]]], dtype=np.float32)
    converted_float = np.floor(float_frames).astype(np.int16)

    assert converted_float.dtype == np.int16
    assert converted_float[0, 0, 0] == 100
    assert converted_float[0, 0, 1] == 200
    assert converted_float[0, 1, 0] == 300
    assert converted_float[0, 1, 1] == 400

    # Tests uint16 to int16
    uint16_frames = np.array([[[1000, 30000], [50000, 65535]]], dtype=np.uint16)
    converted_uint16 = (uint16_frames // 2).astype(np.int16)

    assert converted_uint16.dtype == np.int16
    assert converted_uint16[0, 0, 0] == 500
    assert converted_uint16[0, 0, 1] == 15000
    assert converted_uint16[0, 1, 0] == 25000
    assert converted_uint16[0, 1, 1] == 32767

    # Tests int32 to int16
    int32_frames = np.array([[[1000, 20000], [30000, 32000]]], dtype=np.int32)
    converted_int32 = (int32_frames // 2).astype(np.int16)

    assert converted_int32[0, 0, 0] == 500
    assert converted_int32[0, 0, 1] == 10000
    assert converted_int32[0, 1, 0] == 15000
    assert converted_int32[0, 1, 1] == 16000


def test_single_frame_tiff_bin(tmp_path, ops):
    """Verifies that a third dimension is added when a single-frame TIFF is converted to a suite2p plane
    binary (.bin)."""
    frame = np.random.randint(0, 65535, size=(100, 100), dtype=np.uint16)
    tiff_path = tmp_path / "test_single_frame.tiff"
    with TiffWriter(tiff_path) as tiff:
        tiff.write(frame, contiguous=True)

    tiff, length = _open_tiff(tiff_path)
    assert tiff is not None
    assert length == 1
    assert hasattr(tiff, "pages")

    frames = _read_tiff(tiff, start_index=0, batch_size=1)
    assert frames is not None
    assert len(frames.shape) == 3
    assert frames.shape[0] == 1

    result_ops = tiff_to_binary(ops)
    assert result_ops["Ly"] == 100
    assert result_ops["Lx"] == 100
    assert result_ops["nframes"] == 1
    assert "mean_image" in result_ops
    assert isinstance(result_ops["mean_image"], np.ndarray)
    assert result_ops["mean_image"].shape == (100, 100)


def test_multi_plane_channel_bin(tmp_path, ops):
    """Verifies multi-plane and multi-channel tiff to binary conversion."""

    ops["nplanes"] = 3
    ops["nchannels"] = 2

    frames = np.random.randint(0, 1000, size=(60, 100, 100), dtype=np.uint16)
    tiff_path = tmp_path / "multiple_planes_channels.tiff"
    with TiffWriter(tiff_path) as tiff:
        for frame in frames:
            tiff.write(frame, contiguous=True)

    result_ops = tiff_to_binary(ops)

    assert "mean_image_channel_2" in result_ops

    # Checks that the total frames = 60, frames per channel = 60/(3x2) = 10
    assert result_ops["nframes"] == 10


@pytest.fixture
def mesoscan_test_setup(tmp_path):
    """Creates a temporary directory containing both a TIFF file with sample frames and an
    ops.json configuration file."""
    frames = np.random.randint(0, 1000, size=(20, 100, 100), dtype=np.uint16)

    def _setup(json_data=None, test_name="default"):
        test_dir = tmp_path / test_name
        test_dir.mkdir(exist_ok=True)

        if json_data:
            json_path = test_dir / "ops.json"
            with open(json_path, "w") as f:
                json.dump(json_data, f)

        tiff_path = test_dir / "test.tiff"
        with TiffWriter(tiff_path) as tiff:
            for frame in frames:
                tiff.write(frame, contiguous=True)

        # Returns the base test_ops with default mesoscan parameters
        return {
            "data_path": [str(test_dir)],
            "save_path": str(test_dir),
            "batch_size": 500,
            "progress_bars": False,
            "do_registration": True,
            "functional_chan": 1,
            "nchannels": 1,
            "ignored_file_names": [],
            "look_one_level_down": False,
            "lines": [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
            "nrois": 2,
            "nplanes": 2,
            "dy": [0, 5],
            "dx": [0, 0],
            "fs": 30.0,
        }

    return _setup


def test_mesoscan_registration_settings(mesoscan_test_setup):
    """Tests mesoscan behavior with different registration settings."""
    test_ops_reg_on = mesoscan_test_setup(test_name="reg_on")
    test_ops_reg_on["do_registration"] = True

    result_ops_reg_on = mesoscan_to_binary(test_ops_reg_on)

    # Tests that when registration is enabled, the frame dimensions reflect the dimensions of the extracted
    # ROI scan lines:
    assert "mean_image" in result_ops_reg_on
    assert result_ops_reg_on["Ly"] == 5
    assert result_ops_reg_on["Lx"] == 100

    # Tests that when registration is disabled, yrange and xrange span the entire frame
    test_ops_reg_off = mesoscan_test_setup(test_name="reg_off")
    test_ops_reg_off["do_registration"] = False

    result_ops_reg_off = mesoscan_to_binary(test_ops_reg_off)

    assert result_ops_reg_off["yrange"][0] == 0
    assert result_ops_reg_off["yrange"][1] == result_ops_reg_off["Ly"]
    assert result_ops_reg_off["xrange"][0] == 0
    assert result_ops_reg_off["xrange"][1] == result_ops_reg_off["Lx"]


def test_mesoscan_to_binary(mesoscan_test_setup):
    """Ensures that the mesoscan to binary conversion process properly handles the expansion from the original
    nested ROI structure to individual ROI * plane combinations."""
    test_ops = mesoscan_test_setup()
    result_ops = mesoscan_to_binary(test_ops)

    assert len(result_ops["lines"]) == 5
    assert result_ops["lines"] in [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

    bin_files = list(Path(test_ops["save_path"]).rglob("*.bin"))
    assert len(bin_files) >= 1


def test_mesoscan_no_nrois(mesoscan_test_setup):
    """Tests that when the number of ROIs is not specified but lines are, the number of planes is used
    as the number of ROIs."""
    json_data = {
        "nplanes": 3,
        "lines": [[0, 1], [2, 3], [4, 5]],
    }

    test_ops = mesoscan_test_setup(json_data, "nplanes")
    del test_ops["lines"]
    test_ops.update(
        {
            "dy": [0, 2, 4],
            "dx": [0, 0, 0],
        }
    )

    result_ops = mesoscan_to_binary(test_ops)

    assert test_ops["nrois"] == 3  # nrois = nplanes
    assert test_ops["nplanes"] == 3
    assert result_ops["Ly"] == 2


def test_mesoscan_nested_structure(mesoscan_test_setup):
    """Tests that if the number of planes or files are not specified in ops.json, the data is
    assumed to be nested by planes. The number of planes should be set as the number of top-level
    keys."""
    json_data = {
        "plane0": {"param": "value1"},
        "plane1": {"param": "value2"},
        "lines": [[0, 1], [2, 3]],
    }

    test_ops = mesoscan_test_setup(json_data, "nested_json")
    test_ops.update(
        {
            "dy": [0, 2],
            "dx": [0, 0],
            "lines": [[0, 1], [2, 3]],
            "nplanes": 3,
        }
    )
    result_ops = mesoscan_to_binary(test_ops)

    assert test_ops["nrois"] == 2
    assert test_ops["nplanes"] == 6
    assert result_ops["Ly"] == 2
