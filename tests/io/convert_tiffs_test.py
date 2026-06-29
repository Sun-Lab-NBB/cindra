"""Contains integration tests for the convert_tiffs_to_binary stage entry point."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Callable

import numpy as np
import pytest
from tifffile import TiffWriter

from cindra.io.tiff import (
    _create_binary_files,
    _get_frame_dimensions,
    convert_tiffs_to_binary,
)
from cindra.io.context import PARAMETERS_FILENAME
from cindra.dataclasses import (
    IOData,
    RuntimeContext,
    AcquisitionParameters,
    SingleRecordingRuntimeData,
    SingleRecordingConfiguration,
)

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

_FRAME_HEIGHT: int = 8
"""The base frame height in pixels for the synthetic TIFF inputs."""

_FRAME_WIDTH: int = 6
"""The base frame width in pixels for the synthetic TIFF inputs."""


def _write_parameters_json(directory: Path, *, plane_number: int, channel_number: int) -> None:
    """Writes a minimal cindra_parameters.json file into the given data directory."""
    directory.mkdir(parents=True, exist_ok=True)
    data = {"frame_rate": 30.0, "plane_number": plane_number, "channel_number": channel_number}
    (directory / PARAMETERS_FILENAME).write_text(json.dumps(data))


def _write_constant_tiff(file_path: Path, frame_values: list[int], height: int, width: int) -> None:
    """Writes a multi-page TIFF where page k is a constant int16 image filled with frame_values[k]."""
    with TiffWriter(file_path) as writer:
        for value in frame_values:
            writer.write(np.full((height, width), value, dtype=np.int16))


def _constant_stack(frame_values: list[int], height: int, width: int) -> NDArray[np.int16]:
    """Builds the int16 frame stack expected on disk for a sequence of constant frame values."""
    return np.stack([np.full((height, width), value, dtype=np.int16) for value in frame_values])


def _build_configuration(*, data_path: Path | None, output_path: Path) -> SingleRecordingConfiguration:
    """Builds a single-recording configuration wired to the given data and output directories."""
    configuration = SingleRecordingConfiguration()
    configuration.file_io.data_path = data_path
    configuration.file_io.output_path = output_path
    configuration.runtime.parallel_workers = 1
    configuration.runtime.display_progress_bars = False
    return configuration


def _build_context(
    *,
    output_path: Path,
    configuration: SingleRecordingConfiguration,
    acquisition: AcquisitionParameters,
    plane_index: int,
    two_channels: bool = False,
    mroi_lines: tuple[int, ...] = (),
) -> RuntimeContext:
    """Builds a RuntimeContext whose binary paths point into a fresh per-plane output directory."""
    plane_directory = output_path / "cindra" / f"plane_{plane_index}"
    plane_directory.mkdir(parents=True, exist_ok=True)
    io_data = IOData(
        output_path=plane_directory,
        plane_index=plane_index,
        mroi_lines=mroi_lines,
        registered_binary_path=plane_directory / "channel_1_data.bin",
    )
    if two_channels:
        io_data.registered_binary_path_channel_2 = plane_directory / "channel_2_data.bin"
    runtime = SingleRecordingRuntimeData(output_path=plane_directory, io=io_data)
    return RuntimeContext(configuration=configuration, acquisition=acquisition, runtime=runtime)


class TestConvertTiffsToBinary:
    """Tests convert_tiffs_to_binary."""

    def test_single_plane_single_channel_writes_exact_frames(
        self, tmp_path: Path, read_binary_movie: Callable[[Path, int, int], NDArray[np.int16]]
    ) -> None:
        """Verifies that single-plane single-channel conversion writes the TIFF frames verbatim and sets metadata."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(data_path, plane_number=1, channel_number=1)
        frame_values = [0, 1, 2, 3, 4]
        _write_constant_tiff(data_path / "recording.tif", frame_values, _FRAME_HEIGHT, _FRAME_WIDTH)

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )

        convert_tiffs_to_binary(contexts=[context])

        io_data = context.runtime.io
        assert io_data.frame_count == len(frame_values)
        assert io_data.frame_height == _FRAME_HEIGHT
        assert io_data.frame_width == _FRAME_WIDTH
        assert context.runtime.registration.valid_y_range == (0, _FRAME_HEIGHT)
        assert context.runtime.registration.valid_x_range == (0, _FRAME_WIDTH)

        binary = read_binary_movie(io_data.registered_binary_path, _FRAME_HEIGHT, _FRAME_WIDTH)
        assert np.array_equal(binary, _constant_stack(frame_values, _FRAME_HEIGHT, _FRAME_WIDTH))

        expected_mean = np.full((_FRAME_HEIGHT, _FRAME_WIDTH), np.mean(frame_values), dtype=np.float32)
        assert np.allclose(context.runtime.detection.mean_image, expected_mean)

    def test_multi_plane_interleaves_frames(
        self, tmp_path: Path, read_binary_movie: Callable[[Path, int, int], NDArray[np.int16]]
    ) -> None:
        """Verifies that two-plane conversion deinterleaves frames into even and odd plane streams."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(data_path, plane_number=2, channel_number=1)
        frame_values = [0, 1, 2, 3, 4, 5, 6, 7]
        _write_constant_tiff(data_path / "recording.tif", frame_values, _FRAME_HEIGHT, _FRAME_WIDTH)

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=2, channel_number=1)
        context_0 = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )
        context_1 = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=1
        )

        convert_tiffs_to_binary(contexts=[context_0, context_1])

        binary_0 = read_binary_movie(context_0.runtime.io.registered_binary_path, _FRAME_HEIGHT, _FRAME_WIDTH)
        binary_1 = read_binary_movie(context_1.runtime.io.registered_binary_path, _FRAME_HEIGHT, _FRAME_WIDTH)
        assert np.array_equal(binary_0, _constant_stack([0, 2, 4, 6], _FRAME_HEIGHT, _FRAME_WIDTH))
        assert np.array_equal(binary_1, _constant_stack([1, 3, 5, 7], _FRAME_HEIGHT, _FRAME_WIDTH))
        assert context_0.runtime.io.frame_count == 4
        assert context_1.runtime.io.frame_count == 4
        assert np.allclose(context_0.runtime.detection.mean_image, 3.0)
        assert np.allclose(context_1.runtime.detection.mean_image, 4.0)

    def test_two_channels_split_across_small_batches(
        self, tmp_path: Path, read_binary_movie: Callable[[Path, int, int], NDArray[np.int16]]
    ) -> None:
        """Verifies that two-channel conversion routes channels into separate binaries across multiple batches."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(data_path, plane_number=1, channel_number=2)
        frame_values = [0, 1, 2, 3, 4, 5, 6, 7]
        _write_constant_tiff(data_path / "recording.tif", frame_values, _FRAME_HEIGHT, _FRAME_WIDTH)

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        configuration.main.two_channels = True
        # A small batch size forces several read batches, exercising the mean-accumulator reuse path for both channels.
        configuration.registration.batch_size = 2
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=2)
        context = _build_context(
            output_path=output_path,
            configuration=configuration,
            acquisition=acquisition,
            plane_index=0,
            two_channels=True,
        )

        convert_tiffs_to_binary(contexts=[context])

        io_data = context.runtime.io
        binary_1 = read_binary_movie(io_data.registered_binary_path, _FRAME_HEIGHT, _FRAME_WIDTH)
        binary_2 = read_binary_movie(io_data.registered_binary_path_channel_2, _FRAME_HEIGHT, _FRAME_WIDTH)
        assert np.array_equal(binary_1, _constant_stack([0, 2, 4, 6], _FRAME_HEIGHT, _FRAME_WIDTH))
        assert np.array_equal(binary_2, _constant_stack([1, 3, 5, 7], _FRAME_HEIGHT, _FRAME_WIDTH))
        assert io_data.frame_count == 4
        assert np.allclose(context.runtime.detection.mean_image, 3.0)
        assert np.allclose(context.runtime.detection.mean_image_channel_2, 4.0)

    def test_second_channel_functional_swaps_channel_streams(
        self, tmp_path: Path, read_binary_movie: Callable[[Path, int, int], NDArray[np.int16]]
    ) -> None:
        """Verifies that disabling first_channel_functional routes the functional stream to the second interleave."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(data_path, plane_number=1, channel_number=2)
        frame_values = [0, 1, 2, 3, 4, 5, 6, 7]
        _write_constant_tiff(data_path / "recording.tif", frame_values, _FRAME_HEIGHT, _FRAME_WIDTH)

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        configuration.main.two_channels = True
        configuration.main.first_channel_functional = False
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=2)
        context = _build_context(
            output_path=output_path,
            configuration=configuration,
            acquisition=acquisition,
            plane_index=0,
            two_channels=True,
        )

        convert_tiffs_to_binary(contexts=[context])

        io_data = context.runtime.io
        binary_1 = read_binary_movie(io_data.registered_binary_path, _FRAME_HEIGHT, _FRAME_WIDTH)
        binary_2 = read_binary_movie(io_data.registered_binary_path_channel_2, _FRAME_HEIGHT, _FRAME_WIDTH)
        # With the functional channel set to the second interleave slot, the functional binary receives the odd frames.
        assert np.array_equal(binary_1, _constant_stack([1, 3, 5, 7], _FRAME_HEIGHT, _FRAME_WIDTH))
        assert np.array_equal(binary_2, _constant_stack([0, 2, 4, 6], _FRAME_HEIGHT, _FRAME_WIDTH))

    def test_mroi_single_channel_slices_roi_lines(
        self, tmp_path: Path, read_binary_movie: Callable[[Path, int, int], NDArray[np.int16]]
    ) -> None:
        """Verifies that MROI conversion crops each frame to its ROI line range before writing the binary."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(data_path, plane_number=1, channel_number=1)
        frame_values = [0, 1, 2, 3]
        _write_constant_tiff(data_path / "recording.tif", frame_values, _FRAME_HEIGHT, _FRAME_WIDTH)

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1, roi_number=2)
        mroi_lines = (2, 3, 4, 5)
        context = _build_context(
            output_path=output_path,
            configuration=configuration,
            acquisition=acquisition,
            plane_index=0,
            mroi_lines=mroi_lines,
        )

        convert_tiffs_to_binary(contexts=[context])

        io_data = context.runtime.io
        roi_height = mroi_lines[-1] - mroi_lines[0] + 1
        assert io_data.frame_height == roi_height
        assert io_data.frame_width == _FRAME_WIDTH

        binary = read_binary_movie(io_data.registered_binary_path, roi_height, _FRAME_WIDTH)
        assert np.array_equal(binary, _constant_stack(frame_values, roi_height, _FRAME_WIDTH))

    def test_mroi_two_channels_slices_both_streams(
        self, tmp_path: Path, read_binary_movie: Callable[[Path, int, int], NDArray[np.int16]]
    ) -> None:
        """Verifies that MROI two-channel conversion crops both channel streams to the ROI line range."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(data_path, plane_number=1, channel_number=2)
        frame_values = [0, 1, 2, 3, 4, 5, 6, 7]
        _write_constant_tiff(data_path / "recording.tif", frame_values, _FRAME_HEIGHT, _FRAME_WIDTH)

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        configuration.main.two_channels = True
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=2, roi_number=2)
        mroi_lines = (2, 3, 4, 5)
        context = _build_context(
            output_path=output_path,
            configuration=configuration,
            acquisition=acquisition,
            plane_index=0,
            two_channels=True,
            mroi_lines=mroi_lines,
        )

        convert_tiffs_to_binary(contexts=[context])

        io_data = context.runtime.io
        roi_height = mroi_lines[-1] - mroi_lines[0] + 1
        assert io_data.frame_height == roi_height
        binary_1 = read_binary_movie(io_data.registered_binary_path, roi_height, _FRAME_WIDTH)
        binary_2 = read_binary_movie(io_data.registered_binary_path_channel_2, roi_height, _FRAME_WIDTH)
        assert np.array_equal(binary_1, _constant_stack([0, 2, 4, 6], roi_height, _FRAME_WIDTH))
        assert np.array_equal(binary_2, _constant_stack([1, 3, 5, 7], roi_height, _FRAME_WIDTH))

    def test_multiple_files_continue_on_empty_plane_batch(
        self, tmp_path: Path, read_binary_movie: Callable[[Path, int, int], NDArray[np.int16]]
    ) -> None:
        """Verifies that a file boundary shifting the interleave can leave a plane with no frames in a batch."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(data_path, plane_number=2, channel_number=1)
        # The first file holds three frames (an odd count), shifting the interleave offset so that the trailing
        # single-frame file contributes no frames to plane 0 but one frame to plane 1.
        _write_constant_tiff(data_path / "recording_0.tif", [0, 1, 2], _FRAME_HEIGHT, _FRAME_WIDTH)
        _write_constant_tiff(data_path / "recording_1.tif", [3], _FRAME_HEIGHT, _FRAME_WIDTH)

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=2, channel_number=1)
        context_0 = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )
        context_1 = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=1
        )

        convert_tiffs_to_binary(contexts=[context_0, context_1])

        binary_0 = read_binary_movie(context_0.runtime.io.registered_binary_path, _FRAME_HEIGHT, _FRAME_WIDTH)
        binary_1 = read_binary_movie(context_1.runtime.io.registered_binary_path, _FRAME_HEIGHT, _FRAME_WIDTH)
        assert np.array_equal(binary_0, _constant_stack([0, 2], _FRAME_HEIGHT, _FRAME_WIDTH))
        assert np.array_equal(binary_1, _constant_stack([1, 3], _FRAME_HEIGHT, _FRAME_WIDTH))

    def test_single_frame_tiff(
        self, tmp_path: Path, read_binary_movie: Callable[[Path, int, int], NDArray[np.int16]]
    ) -> None:
        """Verifies that a single-page TIFF is converted into a single-frame binary."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(data_path, plane_number=1, channel_number=1)
        _write_constant_tiff(data_path / "recording.tif", [7], _FRAME_HEIGHT, _FRAME_WIDTH)

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )

        convert_tiffs_to_binary(contexts=[context])

        io_data = context.runtime.io
        assert io_data.frame_count == 1
        binary = read_binary_movie(io_data.registered_binary_path, _FRAME_HEIGHT, _FRAME_WIDTH)
        assert np.array_equal(binary, _constant_stack([7], _FRAME_HEIGHT, _FRAME_WIDTH))

    def test_ignored_file_names_are_excluded(
        self, tmp_path: Path, read_binary_movie: Callable[[Path, int, int], NDArray[np.int16]]
    ) -> None:
        """Verifies that files whose stem matches ignored_file_names are skipped during discovery."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(data_path, plane_number=1, channel_number=1)
        frame_values = [0, 1, 2, 3]
        _write_constant_tiff(data_path / "recording.tif", frame_values, _FRAME_HEIGHT, _FRAME_WIDTH)
        (data_path / "ignored.tif").write_bytes(b"not a real tiff")

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        configuration.file_io.ignored_file_names = ("ignored",)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )

        convert_tiffs_to_binary(contexts=[context])

        io_data = context.runtime.io
        assert io_data.frame_count == len(frame_values)
        binary = read_binary_movie(io_data.registered_binary_path, _FRAME_HEIGHT, _FRAME_WIDTH)
        assert np.array_equal(binary, _constant_stack(frame_values, _FRAME_HEIGHT, _FRAME_WIDTH))

    def test_empty_contexts_raises(self) -> None:
        """Verifies that providing no contexts raises a ValueError."""
        with pytest.raises(ValueError, match="At least one RuntimeContext"):
            convert_tiffs_to_binary(contexts=[])

    def test_missing_data_path_raises(self, tmp_path: Path) -> None:
        """Verifies that a configuration without a data_path raises a ValueError."""
        output_path = tmp_path / "output"
        configuration = _build_configuration(data_path=None, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )

        with pytest.raises(ValueError, match="data_path must be configured"):
            convert_tiffs_to_binary(contexts=[context])


class TestGetFrameDimensions:
    """Tests _get_frame_dimensions."""

    def test_empty_tiff_raises(self, tmp_path: Path) -> None:
        """Verifies that an empty (zero-page) first TIFF file raises a ValueError."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(data_path, plane_number=1, channel_number=1)
        empty_tiff = data_path / "empty.tif"
        with TiffWriter(empty_tiff):
            pass

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )

        with pytest.raises(ValueError, match="first TIFF file is empty"):
            _get_frame_dimensions(tiff_files=[empty_tiff], contexts=[context], acquisition=acquisition)


class TestCreateBinaryFiles:
    """Tests _create_binary_files."""

    def test_empty_contexts_raises(self) -> None:
        """Verifies that providing no contexts raises a ValueError."""
        with pytest.raises(ValueError, match="At least one RuntimeContext"):
            _create_binary_files(contexts=[], frame_heights=[], frame_widths=[], frames_per_plane=1)

    def test_missing_channel_1_path_raises(self, tmp_path: Path) -> None:
        """Verifies that a missing channel 1 binary path raises a ValueError."""
        output_path = tmp_path / "output"
        configuration = _build_configuration(data_path=tmp_path / "data", output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )
        context.runtime.io.registered_binary_path = None

        with pytest.raises(ValueError, match="registered_binary_path is not"):
            _create_binary_files(
                contexts=[context], frame_heights=[_FRAME_HEIGHT], frame_widths=[_FRAME_WIDTH], frames_per_plane=1
            )

    def test_missing_channel_2_path_raises(self, tmp_path: Path) -> None:
        """Verifies that a missing channel 2 binary path raises a ValueError for two-channel data."""
        output_path = tmp_path / "output"
        configuration = _build_configuration(data_path=tmp_path / "data", output_path=output_path)
        configuration.main.two_channels = True
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=2)
        context = _build_context(
            output_path=output_path,
            configuration=configuration,
            acquisition=acquisition,
            plane_index=0,
            two_channels=True,
        )
        context.runtime.io.registered_binary_path_channel_2 = None

        with pytest.raises(ValueError, match="registered_binary_path_channel_2 is not"):
            _create_binary_files(
                contexts=[context], frame_heights=[_FRAME_HEIGHT], frame_widths=[_FRAME_WIDTH], frames_per_plane=1
            )
