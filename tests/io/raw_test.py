"""Contains tests for classes and methods provided by the raw.py module."""

import pytest
import copy
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

from ataraxis_base_utilities import error_format
from sl_suite2p.io.raw import _RawFile, raw_to_binary


@pytest.fixture
def default_xml_fields():
    """Initializes basic XML content for ThorImage configuration."""
    return {
        "ThorImageExperiment": {
            "LSM": {
                "@pixelX": "512",
                "@pixelY": "512",
                "@channel": "1",
                "@frameRate": "10.0",
                "@widthUM": "1000.0",
                "@heightUM": "1000.0",
            },
            "Streaming": {"@frames": "100", "@zFastEnable": "0", "@flybackFrames": "0"},
            "ZStage": {"@steps": "1"},
            "ExperimentStatus": {"@value": "Completed"},
        }
    }


def _add_xml_elements(parent: ET.Element, data: dict[str, any]) -> None:
    """Recursively adds XML elements and attributes to parent element."""
    for key, value in data.items():
        if key.startswith("@"):
            parent.set(key[1:], str(value))
        elif isinstance(value, dict):
            child = ET.Element(key)
            parent.append(child)
            _add_xml_elements(child, value)
        else:
            child = ET.Element(key)
            child.text = str(value)
            parent.append(child)


def create_thorlabs_raw_file(
    file_path: Path, height: int, width: int, frames: int, channels: int = 1, recorded_planes: int = 1
):
    """Creates a Thorlabs .raw file."""
    total_elements = frames * height * width * channels * recorded_planes
    data = np.random.randint(0, 1000, total_elements, dtype=np.int16)

    with open(file_path, "wb") as f:
        f.write(data.tobytes())


def create_xml_file(xml_path: Path, xml_content: dict[str, any]) -> None:
    """Creates a Throlabs XML (.xml) companion configuration file."""
    root_data = xml_content["ThorImageExperiment"]
    root = ET.Element("ThorImageExperiment")
    _add_xml_elements(root, root_data)

    tree = ET.ElementTree(root)
    with open(xml_path, "wb") as f:
        tree.write(f, encoding="utf-8", xml_declaration=True)


def create_test_files(
    tmp_path: Path, xml_content: dict, height: int, width: int, frames: int, channels: int = 1, recorded_planes: int = 1
) -> tuple[Path, Path]:
    """Creates both raw and XML test files and returns their paths."""
    raw_file = tmp_path.joinpath("test_001.raw")
    xml_file = tmp_path.joinpath("test.xml")

    create_thorlabs_raw_file(
        file_path=raw_file,
        height=height,
        width=width,
        frames=frames,
        channels=channels,
        recorded_planes=recorded_planes,
    )
    create_xml_file(xml_path=xml_file, xml_content=xml_content)

    return raw_file, xml_file


@pytest.mark.parametrize(
    "height,width,channels,z_enable,z_steps,flyback_frames,total_frames,expected_shape,experiment_status",
    [
        (512, 512, 1, 0, 1, 0, 100, (100, 512, 512), "Completed"),
        (256, 256, 2, 0, 1, 0, 100, (200, 256, 256), "Completed"),
        (128, 128, 1, 1, 5, 1, 150, (6, 25, 128, 128), "Completed"),
        (64, 64, 2, 1, 3, 1, 120, (4, 60, 64, 64), "Completed"),
        (128, 128, 1, 0, 1, 0, 50, (75, 128, 128), "Stopped"),
    ],
)
def test_raw_file_init(
    tmp_path,
    default_xml_fields,
    height,
    width,
    channels,
    z_enable,
    z_steps,
    flyback_frames,
    total_frames,
    expected_shape,
    experiment_status,
):
    """Verifies that parameters stored in the ThorImage XML configuration file are parsed and reflected correctly in the attributes
    used to read Thorlabs RAW files."""
    xml_content = copy.deepcopy(default_xml_fields)
    xml_content["ThorImageExperiment"]["LSM"].update(
        {
            "@pixelX": str(height),
            "@pixelY": str(width),
            "@channel": str(channels),
        }
    )
    xml_content["ThorImageExperiment"]["Streaming"].update(
        {
            "@frames": str(total_frames),
            "@zFastEnable": str(z_enable),
            "@flybackFrames": str(flyback_frames),
        }
    )
    xml_content["ThorImageExperiment"]["ZStage"]["@steps"] = str(z_steps)
    xml_content["ThorImageExperiment"]["ExperimentStatus"]["@value"] = experiment_status

    # For stopped experiments, the expected shape is used to determine the actual frame count.
    if experiment_status == "Stopped":
        actual_frames = expected_shape[0]
    else:
        actual_frames = total_frames

    recorded_planes = z_steps + flyback_frames if z_enable else 1
    raw_file, xml_file = create_test_files(
        tmp_path, xml_content, height, width, actual_frames, channels, recorded_planes
    )

    raw_file_instance = _RawFile(tmp_path)

    # Tests .raw file properties
    assert raw_file_instance.path == raw_file
    assert raw_file_instance.size == raw_file.stat().st_size
    assert raw_file_instance.shape == expected_shape

    # Verifies z-stack specific attributes
    if z_enable:
        assert raw_file_instance.z_planes == z_steps
        assert raw_file_instance.recorded_planes == recorded_planes
        expected_frame_number = total_frames // recorded_planes
        assert raw_file_instance.frame_number == expected_frame_number

    else:
        assert raw_file_instance.z_planes == 1
        assert raw_file_instance.recorded_planes == 1

        if experiment_status == "Stopped":
            assert raw_file_instance.frame_number == actual_frames
        else:
            assert raw_file_instance.frame_number == total_frames


@pytest.mark.parametrize("missing_file", ["raw", "xml"])
def test_missing_file(tmp_path, default_xml_fields, missing_file):
    """Verifies that _RawFile triggers the correct error handling when either the .xml or .raw file is missing."""
    if missing_file == "xml":
        raw_file = tmp_path.joinpath("test_001.raw")
        create_thorlabs_raw_file(file_path=raw_file, height=512, width=512, frames=100)
        message = (
            f"Unable to convert Thorlabs RAW data to the Suite2p BinaryFile format. Unable to find the required "
            f".xml configuration file inside the input directory: {tmp_path}."
        )
    else:
        xml_file = tmp_path.joinpath("test.xml")
        create_xml_file(xml_path=xml_file, xml_content=default_xml_fields)
        message = (
            f"Unable to convert Thorlabs RAW data to the Suite2p BinaryFile format. Unable to find the required "
            f"'001.raw' file inside the input directory: {tmp_path}."
        )

    with pytest.raises(FileNotFoundError, match=error_format(message)):
        _RawFile(tmp_path)


@pytest.fixture
def ops(tmp_path):
    """Creates a basic default ops dictionary for testing."""
    return {
        "nplanes": 1,
        "nchannels": 1,
        "fs": 10.0,
        "save_path": tmp_path.joinpath("output"),
        "do_registration": False,
        "batch_size": 50,
        "progress_bars": False,
        "data_path": [],
    }


@pytest.mark.parametrize("channels", [1, 2])
def test_raw_to_binary(tmp_path, default_xml_fields, ops, channels):
    """Tests raw_to_binary conversion for both single and multichannel data."""
    data_dir = tmp_path.joinpath("data")
    data_dir.mkdir()

    # Modifies the .xml file based on the test dimensions and the number of channnels
    xml_content = copy.deepcopy(default_xml_fields)
    height, width = 16, 16
    frames_per_file = 10

    xml_content["ThorImageExperiment"]["LSM"].update(
        {"@pixelX": str(height), "@pixelY": str(width), "@channel": str(channels)}
    )
    xml_content["ThorImageExperiment"]["Streaming"]["@frames"] = str(frames_per_file)

    raw_file = data_dir.joinpath("experiment_001.raw")
    xml_file = data_dir.joinpath("experiment.xml")

    z_planes = 1
    recorded_planes = z_planes

    # Updates the z-stack configuration
    xml_content["ThorImageExperiment"]["Streaming"]["@zFastEnable"] = "0"
    xml_content["ThorImageExperiment"]["ZStage"]["@steps"] = str(z_planes)

    create_thorlabs_raw_file(
        raw_file, height=height, width=width, frames=frames_per_file, channels=channels, recorded_planes=recorded_planes
    )
    create_xml_file(xml_file, xml_content)

    ops = copy.deepcopy(ops)
    ops.update(
        {
            "nplanes": z_planes,
            "nchannels": channels,
            "data_path": [data_dir],
        }
    )

    result_ops = raw_to_binary(ops, override_ops_parameters=True)

    # Verifies basic output properties
    assert result_ops["Ly"] == height
    assert result_ops["Lx"] == width
    assert result_ops["nframes"] == frames_per_file

    plane_dir = tmp_path.joinpath("output").joinpath("plane0")

    # Verifies that if the data uses two functional channels, a second data file is created and the corresponding data is stored.
    if channels > 1:
        assert "mean_image_channel_2" in result_ops
        assert result_ops["mean_image_channel_2"].shape == (height, width)
        assert plane_dir.joinpath("data_chan2.bin").exists()

    # Verifies that ops.npy is updated
    saved_ops = np.load(plane_dir.joinpath("ops.npy"), allow_pickle=True)[()]
    assert saved_ops["nframes"] == frames_per_file
    assert saved_ops["mean_image"].shape == (height, width)


def test_raw_to_binary_inconsistent_config(tmp_path, default_xml_fields, ops):
    """Ensures that all _RawFile instances have the same attributes (the recording configuration is the same for
    all input files)"""
    data_dirs = []

    for i, (height, width, channels) in enumerate([(16, 16, 1), (32, 32, 2)]):
        data_dir = tmp_path.joinpath(f"data_{i}")
        data_dir.mkdir()

        xml_content = copy.deepcopy(default_xml_fields)
        xml_content["ThorImageExperiment"]["LSM"].update(
            {"@pixelX": str(height), "@pixelY": str(width), "@channel": str(channels)}
        )
        xml_content["ThorImageExperiment"]["Streaming"]["@frames"] = "10"

        raw_file = data_dir.joinpath("experiment_001.raw")
        xml_file = data_dir.joinpath("experiment.xml")

        create_thorlabs_raw_file(raw_file, height=height, width=width, frames=10, channels=channels, recorded_planes=1)
        create_xml_file(xml_file, xml_content)
        data_dirs.append(data_dir)

    ops = copy.deepcopy(ops)
    ops["data_path"] = data_dirs

    message = (
        "Unable to convert the input list of Thorlabs .raw files to Suite2P plane BinaryFiles. The recording "
        "configurations of the input .raw files do not match for at least two file instances, indicating that "
        "the files belong to separate recordings."
    )
    with pytest.raises(ValueError, match=error_format(message)):
        raw_to_binary(ops, override_ops_parameters=True)
