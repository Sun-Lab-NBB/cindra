"""Contains tests for methods provided by the tiff.py module."""

import copy
import json
from pathlib import Path
import tempfile
import numpy as np
import pytest
from tifffile import TiffWriter, TiffFile
from sl_suite2p.configuration import RuntimeData

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
    tiff_path = tmp_path.joinpath("single_frame.tiff")

    frame = np.random.randint(0, 65535, size=(100, 100), dtype=np.uint16)

    with TiffWriter(tiff_path) as tiff:
        tiff.write(frame, contiguous=True)

    return tiff_path


@pytest.fixture
def create_multi_frame_tiff(tmp_path):
    """Creates a multi-frame tiff file."""
    tiff_path = tmp_path.joinpath("multi_frame.tiff")

    frames = np.random.randint(0, 65535, size=(10, 100, 100), dtype=np.uint16)

    with TiffWriter(tiff_path) as tiff:
        for frame in frames:
            tiff.write(frame, contiguous=True)

    return tiff_path


@pytest.fixture
def runtime_data(tmp_path):
    """Creates a basic default RuntimeData instance."""
    runtime_data = RuntimeData()

    runtime_data.configuration.file_io.data_path = [tmp_path]
    runtime_data.configuration.output.save_path = str(tmp_path)
    runtime_data.configuration.main.functional_chan = 1
    runtime_data.configuration.main.align_by_chan = 1
    runtime_data.configuration.main.nplanes = 1
    runtime_data.configuration.main.nchannels = 1
    runtime_data.configuration.file_io.ignored_file_names = []
    runtime_data.configuration.registration.batch_size = 500
    runtime_data.configuration.output.progress_bars = False
    runtime_data.configuration.registration.do_registration = True

    return runtime_data


@pytest.mark.parametrize(
    "functional_channel, alignment_channel, batch_number, channel, expected_subdir, expected_channel_idx",
    [
        (0, 0, 0, 0, "channel_1_tiffs", 0),
        (0, 1, 0, 0, "channel_2_tiffs", 1),
        (1, 1, 5, 1, "channel_2_tiffs", 1),
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
    output_path = str(tmp_path.joinpath("output.tiff"))

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


@pytest.mark.parametrize(
    "input_dtype, input_data, expected_data",
    [
        # float32 to int16
        (np.float32, [[[100.7, 200.9], [300.3, 400.1]]], [[[100, 200], [300, 400]]]),
        # uint16 to int16
        (np.uint16, [[[1000, 30000], [50000, 65535]]], [[[500, 15000], [25000, 32767]]]),
        # int16 to int16
        (np.int16, [[[10, 20], [30, 40]]], [[[10, 20], [30, 40]]]),
    ],
)
def test_read_tiff_pixel_conversion(tmp_path, input_dtype, input_data, expected_data):
    """Verifies that all image data are converted into type int16."""
    tiff_path = tmp_path.joinpath("test.tiff")
    frames = np.array(input_data, dtype=input_dtype)

    with TiffWriter(tiff_path) as tif:
        for frame in frames:
            tif.write(frame.astype(input_dtype))

    with TiffFile(tiff_path) as tif:
        result_frames = _read_tiff(tif, start_index=0, batch_size=len(frames))

    expected_frames = np.array(expected_data, dtype=np.int16)

    assert result_frames.dtype == np.int16
    assert result_frames.shape == expected_frames.shape
    np.testing.assert_array_equal(result_frames, expected_frames)


def test_single_frame_tiff_bin(tmp_path, runtime_data):
    """Verifies that a third dimension is added when a single-frame TIFF is converted to a suite2p plane
    binary (.bin), and that any extra frames read beyond the file length are truncated to match the actual
    number of frames."""
    frame = np.random.randint(0, 65535, size=(100, 100), dtype=np.uint16)
    tiff_path = tmp_path.joinpath("test_single_frame.tiff")
    with TiffWriter(tiff_path) as tiff:
        tiff.write(frame, contiguous=True)

    tiff, length = _open_tiff(tiff_path)
    assert tiff is not None and hasattr(tiff, "pages")
    assert length == 1

    # Reads more frames than available to trigger truncation logic
    frames = _read_tiff(tiff, start_index=0, batch_size=2)
    assert frames is not None and len(frames.shape) == 3
    assert frames.shape[0] == 1

    result_runtime_data = tiff_to_binary(runtime_data)
    assert result_runtime_data.data.file_io.height == 100
    assert result_runtime_data.data.file_io.width == 100
    assert result_runtime_data.data.file_io.nframes == 1
    assert isinstance(result_runtime_data.data.file_io.mean_image, np.ndarray)
    assert result_runtime_data.data.file_io.mean_image.shape == (100, 100)


def test_multi_plane_channel_bin(tmp_path, runtime_data):
    """Verifies multi-plane and multi-channel tiff to binary conversion."""
    runtime_data.configuration.main.nplanes = 3
    runtime_data.configuration.main.nchannels = 2

    frames = np.random.randint(0, 1000, size=(60, 100, 100), dtype=np.uint16)
    tiff_path = tmp_path.joinpath("multiple_planes_channels.tiff")
    with TiffWriter(tiff_path) as tiff:
        for frame in frames:
            tiff.write(frame, contiguous=True)

    result_runtime_data = tiff_to_binary(runtime_data)

    assert result_runtime_data.data.file_io.mean_image_channel_2 is not None
    # Checks that the total frames = 60, frames per channel = 60/(3x2) = 10
    assert result_runtime_data.data.file_io.nframes == 10


@pytest.fixture
def mesoscan_test_setup(tmp_path):
    """Creates a temporary directory containing both a TIFF file with sample frames and an
    ops.json configuration file."""
    frames = np.random.randint(0, 1000, size=(20, 100, 100), dtype=np.uint16)

    def _setup(json_data=None, test_name="default"):
        test_dir = tmp_path.joinpath(test_name)
        test_dir.mkdir(exist_ok=True)

        if json_data:
            json_path = test_dir.joinpath("ops.json")
            with open(json_path, "w") as f:
                json.dump(json_data, f)

        tiff_path = test_dir.joinpath("test.tiff")
        with TiffWriter(tiff_path) as tiff:
            for frame in frames:
                tiff.write(frame, contiguous=True)

        # Returns RuntimeData with default mesoscan parameters
        runtime_data = RuntimeData()
        runtime_data.configuration.file_io.data_path = [test_dir]
        runtime_data.configuration.output.save_path = str(test_dir)
        runtime_data.configuration.registration.batch_size = 500
        runtime_data.configuration.output.progress_bars = False
        runtime_data.configuration.registration.do_registration = True
        runtime_data.configuration.main.functional_chan = 1
        runtime_data.configuration.main.nchannels = 1
        runtime_data.configuration.file_io.ignored_file_names = []
        runtime_data.data.file_io.lines = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
        runtime_data.data.file_io.nrois = 2
        runtime_data.configuration.main.nplanes = 2
        runtime_data.data.file_io.dy = [0, 5]
        runtime_data.data.file_io.dx = [0, 0]
        runtime_data.configuration.main.fs = 30.0

        return runtime_data

    return _setup


def test_mesoscan_registration_settings(mesoscan_test_setup):
    """Tests mesoscan behavior with different registration settings."""
    runtime_data_reg_on = mesoscan_test_setup(test_name="reg_on")
    runtime_data_reg_on.configuration.registration.do_registration = True

    result_runtime_data_reg_on = mesoscan_to_binary(runtime_data_reg_on)

    # Tests that when registration is enabled, the frame dimensions reflect the dimensions of the extracted
    # ROI scan lines:
    assert result_runtime_data_reg_on.data.file_io.mean_image is not None
    assert result_runtime_data_reg_on.data.file_io.height == 5
    assert result_runtime_data_reg_on.data.file_io.width == 100

    # Tests that when registration is disabled, yrange and xrange span the entire frame
    runtime_data_reg_off = mesoscan_test_setup(test_name="reg_off")
    runtime_data_reg_off.configuration.registration.do_registration = False

    result_runtime_data_reg_off = mesoscan_to_binary(runtime_data_reg_off)

    assert result_runtime_data_reg_off.data.file_io.height_range[0] == 0
    assert result_runtime_data_reg_off.data.file_io.height_range[1] == result_runtime_data_reg_off.data.file_io.height
    assert result_runtime_data_reg_off.data.file_io.width_range[0] == 0
    assert result_runtime_data_reg_off.data.file_io.width_range[1] == result_runtime_data_reg_off.data.file_io.width


@pytest.mark.parametrize("channels", [1, 2])
def test_mesoscan_to_binary_channels(mesoscan_test_setup, channels):
    """Ensures that the mesoscan to binary conversion process properly handles the expansion from the original
    nested ROI structure to individual ROI * plane combinations for both single and multichannel data."""
    runtime_data = mesoscan_test_setup()
    runtime_data.configuration.main.nchannels = channels

    result_runtime_data = mesoscan_to_binary(runtime_data)

    assert len(result_runtime_data.data.file_io.lines) == 5
    assert result_runtime_data.data.file_io.lines in [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

    save_path = Path(runtime_data.configuration.output.save_path)
    bin_files = list(save_path.rglob("*.bin"))
    expected_bin_files = 4 * channels  # 2 ROIs * 2 planes * channels
    assert len(bin_files) >= expected_bin_files

    # Verifies that channel 2 data is saved correctly for multichannel tests
    if channels > 1:
        plane_dirs = list(save_path.glob("plane*"))
        for plane_dir in plane_dirs:
            plane_runtime_data = RuntimeData.from_yaml(plane_dir.joinpath("runtime_data.yaml"))
            assert plane_runtime_data.data.file_io.mean_image_channel_2 is not None
            assert plane_runtime_data.data.file_io.mean_image_channel_2.shape == (
                plane_runtime_data.data.file_io.height,
                plane_runtime_data.data.file_io.width,
            )


def test_mesoscan_no_nrois(mesoscan_test_setup):
    """Tests that when the number of ROIs is not specified but lines are, the number of planes is used
    as the number of ROIs."""
    json_data = {
        "nplanes": 3,
        "lines": [[0, 1], [2, 3], [4, 5]],
    }

    runtime_data = mesoscan_test_setup(json_data, "nplanes")
    runtime_data.data.file_io.lines = None
    runtime_data.data.file_io.dy = [0, 2, 4]
    runtime_data.data.file_io.dx = [0, 0, 0]

    result_runtime_data = mesoscan_to_binary(runtime_data)

    assert runtime_data.data.file_io.nrois == 3  # nrois = nplanes
    assert runtime_data.configuration.main.nplanes == 3
    assert result_runtime_data.data.file_io.height == 2


def test_mesoscan_nested_structure(mesoscan_test_setup):
    """Tests that if the number of planes or files are not specified in ops.json, the data is
    assumed to be nested by planes. The number of planes should be set as the number of top-level
    keys."""
    json_data = {
        "plane0": {"param": "value1"},
        "plane1": {"param": "value2"},
        "lines": [[0, 1], [2, 3]],
    }

    runtime_data = mesoscan_test_setup(json_data, "nested_json")
    runtime_data.data.file_io.dy = [0, 2]
    runtime_data.data.file_io.dx = [0, 0]
    runtime_data.configuration.main.nplanes = 3
    runtime_data.data.file_io.lines = None

    result_runtime_data = mesoscan_to_binary(runtime_data)

    assert runtime_data.data.file_io.nrois == 2
    assert runtime_data.configuration.main.nplanes == 6
    assert result_runtime_data.data.file_io.height == 2


def test_mesoscan_with_nrois_in_json(mesoscan_test_setup):
    """Tests that the parameters from the ops.json file are used directly if the number of ROIs is specified inside it."""
    json_data = {
        "nrois": 2,
        "nplanes": 2,
        "dy": [0, 5],
        "dx": [0, 0],
        "fs": 30.0,
        "lines": [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
    }

    runtime_data = mesoscan_test_setup(json_data, "with_nrois")

    # Removes lines to force reading from ops.json
    runtime_data.data.file_io.lines = None
    runtime_data.configuration.main.nchannels = 1

    result_runtime_data = mesoscan_to_binary(runtime_data)

    # Checks that the input runtime_data's nrois is taken directly from ops.json
    assert runtime_data.data.file_io.nrois == 2
    # Checks the result's nplanes (nrois * original nplanes = 2 * 2 = 4)
    assert result_runtime_data.configuration.main.nplanes == 4
