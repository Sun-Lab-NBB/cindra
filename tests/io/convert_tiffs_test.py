"""Contains integration tests for the convert_tiffs_to_binary stage entry point."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from tifffile import TiffWriter

from cindra.io.tiff import (
    _MISMATCH_REPORT_LIMIT,
    _create_binary_files,
    _get_frame_dimensions,
    convert_tiffs_to_binary,
)
from cindra.io.binary import create_registration_marker, resolve_registration_marker_path
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
    from collections.abc import Callable

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
            writer.write(np.full((height, width), fill_value=value, dtype=np.int16))


def _constant_stack(frame_values: list[int], height: int, width: int) -> NDArray[np.int16]:
    """Builds the int16 frame stack expected on disk for a sequence of constant frame values."""
    return np.stack([np.full((height, width), fill_value=value, dtype=np.int16) for value in frame_values])


def _build_configuration(*, data_path: Path | None, output_path: Path) -> SingleRecordingConfiguration:
    """Builds a single-recording configuration wired to the given data and output directories."""
    configuration = SingleRecordingConfiguration()
    configuration.file_io.data_path = data_path
    configuration.file_io.output_path = output_path
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
        _write_parameters_json(directory=data_path, plane_number=1, channel_number=1)
        frame_values = [0, 1, 2, 3, 4]
        _write_constant_tiff(
            file_path=data_path / "recording.tif", frame_values=frame_values, height=_FRAME_HEIGHT, width=_FRAME_WIDTH
        )

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )

        convert_tiffs_to_binary(contexts=[context], workers=1)

        io_data = context.runtime.io
        assert io_data.frame_count == len(frame_values)
        assert io_data.frame_height == _FRAME_HEIGHT
        assert io_data.frame_width == _FRAME_WIDTH
        assert context.runtime.registration.valid_y_range == (0, _FRAME_HEIGHT)
        assert context.runtime.registration.valid_x_range == (0, _FRAME_WIDTH)

        binary = read_binary_movie(
            file_path=io_data.registered_binary_path, frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH
        )
        assert np.array_equal(
            binary, _constant_stack(frame_values=frame_values, height=_FRAME_HEIGHT, width=_FRAME_WIDTH)
        )

        expected_mean = np.full((_FRAME_HEIGHT, _FRAME_WIDTH), fill_value=np.mean(frame_values), dtype=np.float32)
        assert np.allclose(context.runtime.detection.mean_image, expected_mean)

    def test_multi_plane_interleaves_frames(
        self, tmp_path: Path, read_binary_movie: Callable[[Path, int, int], NDArray[np.int16]]
    ) -> None:
        """Verifies that two-plane conversion deinterleaves frames into even and odd plane streams."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=2, channel_number=1)
        frame_values = [0, 1, 2, 3, 4, 5, 6, 7]
        _write_constant_tiff(
            file_path=data_path / "recording.tif", frame_values=frame_values, height=_FRAME_HEIGHT, width=_FRAME_WIDTH
        )

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=2, channel_number=1)
        context_0 = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )
        context_1 = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=1
        )

        convert_tiffs_to_binary(contexts=[context_0, context_1], workers=1)

        binary_0 = read_binary_movie(
            file_path=context_0.runtime.io.registered_binary_path, frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH
        )
        binary_1 = read_binary_movie(
            file_path=context_1.runtime.io.registered_binary_path, frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH
        )
        assert np.array_equal(
            binary_0, _constant_stack(frame_values=[0, 2, 4, 6], height=_FRAME_HEIGHT, width=_FRAME_WIDTH)
        )
        assert np.array_equal(
            binary_1, _constant_stack(frame_values=[1, 3, 5, 7], height=_FRAME_HEIGHT, width=_FRAME_WIDTH)
        )
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
        _write_parameters_json(directory=data_path, plane_number=1, channel_number=2)
        frame_values = [0, 1, 2, 3, 4, 5, 6, 7]
        _write_constant_tiff(
            file_path=data_path / "recording.tif", frame_values=frame_values, height=_FRAME_HEIGHT, width=_FRAME_WIDTH
        )

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

        convert_tiffs_to_binary(contexts=[context], workers=1)

        io_data = context.runtime.io
        binary_1 = read_binary_movie(
            file_path=io_data.registered_binary_path, frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH
        )
        binary_2 = read_binary_movie(
            file_path=io_data.registered_binary_path_channel_2, frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH
        )
        assert np.array_equal(
            binary_1, _constant_stack(frame_values=[0, 2, 4, 6], height=_FRAME_HEIGHT, width=_FRAME_WIDTH)
        )
        assert np.array_equal(
            binary_2, _constant_stack(frame_values=[1, 3, 5, 7], height=_FRAME_HEIGHT, width=_FRAME_WIDTH)
        )
        assert io_data.frame_count == 4
        assert np.allclose(context.runtime.detection.mean_image, 3.0)
        assert np.allclose(context.runtime.detection.mean_image_channel_2, 4.0)

    def test_second_channel_functional_swaps_channel_streams(
        self, tmp_path: Path, read_binary_movie: Callable[[Path, int, int], NDArray[np.int16]]
    ) -> None:
        """Verifies that disabling first_channel_functional routes the functional stream to the second interleave."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=1, channel_number=2)
        frame_values = [0, 1, 2, 3, 4, 5, 6, 7]
        _write_constant_tiff(
            file_path=data_path / "recording.tif", frame_values=frame_values, height=_FRAME_HEIGHT, width=_FRAME_WIDTH
        )

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

        convert_tiffs_to_binary(contexts=[context], workers=1)

        io_data = context.runtime.io
        binary_1 = read_binary_movie(
            file_path=io_data.registered_binary_path, frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH
        )
        binary_2 = read_binary_movie(
            file_path=io_data.registered_binary_path_channel_2, frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH
        )
        # With the functional channel set to the second interleave slot, the functional binary receives the odd frames.
        assert np.array_equal(
            binary_1, _constant_stack(frame_values=[1, 3, 5, 7], height=_FRAME_HEIGHT, width=_FRAME_WIDTH)
        )
        assert np.array_equal(
            binary_2, _constant_stack(frame_values=[0, 2, 4, 6], height=_FRAME_HEIGHT, width=_FRAME_WIDTH)
        )

    def test_mroi_single_channel_slices_roi_lines(
        self, tmp_path: Path, read_binary_movie: Callable[[Path, int, int], NDArray[np.int16]]
    ) -> None:
        """Verifies that MROI conversion crops each frame to its ROI line range before writing the binary."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=1, channel_number=1)
        frame_values = [0, 1, 2, 3]
        _write_constant_tiff(
            file_path=data_path / "recording.tif", frame_values=frame_values, height=_FRAME_HEIGHT, width=_FRAME_WIDTH
        )

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

        convert_tiffs_to_binary(contexts=[context], workers=1)

        io_data = context.runtime.io
        roi_height = mroi_lines[-1] - mroi_lines[0] + 1
        assert io_data.frame_height == roi_height
        assert io_data.frame_width == _FRAME_WIDTH

        binary = read_binary_movie(
            file_path=io_data.registered_binary_path, frame_height=roi_height, frame_width=_FRAME_WIDTH
        )
        assert np.array_equal(binary, _constant_stack(frame_values=frame_values, height=roi_height, width=_FRAME_WIDTH))

    def test_mroi_two_channels_slices_both_streams(
        self, tmp_path: Path, read_binary_movie: Callable[[Path, int, int], NDArray[np.int16]]
    ) -> None:
        """Verifies that MROI two-channel conversion crops both channel streams to the ROI line range."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=1, channel_number=2)
        frame_values = [0, 1, 2, 3, 4, 5, 6, 7]
        _write_constant_tiff(
            file_path=data_path / "recording.tif", frame_values=frame_values, height=_FRAME_HEIGHT, width=_FRAME_WIDTH
        )

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

        convert_tiffs_to_binary(contexts=[context], workers=1)

        io_data = context.runtime.io
        roi_height = mroi_lines[-1] - mroi_lines[0] + 1
        assert io_data.frame_height == roi_height
        binary_1 = read_binary_movie(
            file_path=io_data.registered_binary_path, frame_height=roi_height, frame_width=_FRAME_WIDTH
        )
        binary_2 = read_binary_movie(
            file_path=io_data.registered_binary_path_channel_2, frame_height=roi_height, frame_width=_FRAME_WIDTH
        )
        assert np.array_equal(
            binary_1, _constant_stack(frame_values=[0, 2, 4, 6], height=roi_height, width=_FRAME_WIDTH)
        )
        assert np.array_equal(
            binary_2, _constant_stack(frame_values=[1, 3, 5, 7], height=roi_height, width=_FRAME_WIDTH)
        )

    def test_multiple_files_continue_on_empty_plane_batch(
        self, tmp_path: Path, read_binary_movie: Callable[[Path, int, int], NDArray[np.int16]]
    ) -> None:
        """Verifies that a file boundary shifting the interleave can leave a plane with no frames in a batch."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=2, channel_number=1)
        # The first file holds three frames (an odd count), shifting the interleave offset so that the trailing
        # single-frame file contributes no frames to plane 0 but one frame to plane 1.
        _write_constant_tiff(
            file_path=data_path / "recording_0.tif", frame_values=[0, 1, 2], height=_FRAME_HEIGHT, width=_FRAME_WIDTH
        )
        _write_constant_tiff(
            file_path=data_path / "recording_1.tif", frame_values=[3], height=_FRAME_HEIGHT, width=_FRAME_WIDTH
        )

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=2, channel_number=1)
        context_0 = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )
        context_1 = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=1
        )

        convert_tiffs_to_binary(contexts=[context_0, context_1], workers=1)

        binary_0 = read_binary_movie(
            file_path=context_0.runtime.io.registered_binary_path, frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH
        )
        binary_1 = read_binary_movie(
            file_path=context_1.runtime.io.registered_binary_path, frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH
        )
        assert np.array_equal(binary_0, _constant_stack(frame_values=[0, 2], height=_FRAME_HEIGHT, width=_FRAME_WIDTH))
        assert np.array_equal(binary_1, _constant_stack(frame_values=[1, 3], height=_FRAME_HEIGHT, width=_FRAME_WIDTH))

    def test_interleave_selection_survives_split_batches(
        self, tmp_path: Path, read_binary_movie: Callable[[Path, int, int], NDArray[np.int16]]
    ) -> None:
        """Verifies that the interleave selection is exact when both the files and the batches split mid-cycle."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=2, channel_number=2)
        # Six pages followed by five pages carries the interleave offset across the file boundary, and the four-frame
        # batch the two planes and two channels impose splits both files mid-cycle.
        _write_constant_tiff(
            file_path=data_path / "recording_0.tif",
            frame_values=[0, 1, 2, 3, 4, 5],
            height=_FRAME_HEIGHT,
            width=_FRAME_WIDTH,
        )
        _write_constant_tiff(
            file_path=data_path / "recording_1.tif",
            frame_values=[6, 7, 8, 9, 10],
            height=_FRAME_HEIGHT,
            width=_FRAME_WIDTH,
        )

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        configuration.main.two_channels = True
        configuration.registration.batch_size = 2
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=2, channel_number=2)
        contexts = [
            _build_context(
                output_path=output_path,
                configuration=configuration,
                acquisition=acquisition,
                plane_index=plane_index,
                two_channels=True,
            )
            for plane_index in range(2)
        ]

        convert_tiffs_to_binary(contexts=contexts, workers=1)

        expected_selections = [([0, 4, 8], [1, 5, 9]), ([2, 6, 10], [3, 7])]
        for context, (channel_1_values, channel_2_values) in zip(contexts, expected_selections, strict=True):
            io_data = context.runtime.io
            binary_1 = read_binary_movie(
                file_path=io_data.registered_binary_path, frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH
            )
            binary_2 = read_binary_movie(
                file_path=io_data.registered_binary_path_channel_2,
                frame_height=_FRAME_HEIGHT,
                frame_width=_FRAME_WIDTH,
            )
            assert np.array_equal(
                binary_1, _constant_stack(frame_values=channel_1_values, height=_FRAME_HEIGHT, width=_FRAME_WIDTH)
            )
            assert np.array_equal(
                binary_2, _constant_stack(frame_values=channel_2_values, height=_FRAME_HEIGHT, width=_FRAME_WIDTH)
            )
            expected_mean = np.full(
                (_FRAME_HEIGHT, _FRAME_WIDTH), fill_value=np.mean(channel_1_values), dtype=np.float32
            )
            np.testing.assert_array_equal(context.runtime.detection.mean_image, expected_mean)

    def test_single_frame_tiff(
        self, tmp_path: Path, read_binary_movie: Callable[[Path, int, int], NDArray[np.int16]]
    ) -> None:
        """Verifies that a single-page TIFF is converted into a single-frame binary."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=1, channel_number=1)
        _write_constant_tiff(
            file_path=data_path / "recording.tif", frame_values=[7], height=_FRAME_HEIGHT, width=_FRAME_WIDTH
        )

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )

        convert_tiffs_to_binary(contexts=[context], workers=1)

        io_data = context.runtime.io
        assert io_data.frame_count == 1
        binary = read_binary_movie(
            file_path=io_data.registered_binary_path, frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH
        )
        assert np.array_equal(binary, _constant_stack(frame_values=[7], height=_FRAME_HEIGHT, width=_FRAME_WIDTH))

    def test_ignored_file_names_are_excluded(
        self, tmp_path: Path, read_binary_movie: Callable[[Path, int, int], NDArray[np.int16]]
    ) -> None:
        """Verifies that files whose stem matches ignored_file_names are skipped during discovery."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=1, channel_number=1)
        frame_values = [0, 1, 2, 3]
        _write_constant_tiff(
            file_path=data_path / "recording.tif", frame_values=frame_values, height=_FRAME_HEIGHT, width=_FRAME_WIDTH
        )
        (data_path / "ignored.tif").write_bytes(b"not a real tiff")

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        configuration.file_io.ignored_file_names = ("ignored",)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )

        convert_tiffs_to_binary(contexts=[context], workers=1)

        io_data = context.runtime.io
        assert io_data.frame_count == len(frame_values)
        binary = read_binary_movie(
            file_path=io_data.registered_binary_path, frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH
        )
        assert np.array_equal(
            binary, _constant_stack(frame_values=frame_values, height=_FRAME_HEIGHT, width=_FRAME_WIDTH)
        )

    def test_frame_count_mismatch_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifies that a binary sized for more frames than the source files deliver raises a RuntimeError."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=1, channel_number=1)
        _write_constant_tiff(
            file_path=data_path / "recording.tif", frame_values=[0, 1, 2], height=_FRAME_HEIGHT, width=_FRAME_WIDTH
        )

        # Inflates the interleave accounting by one frame, so each binary is sized for a frame the TIFF never delivers.
        # This reproduces an accounting disagreement that would otherwise leave a silently truncated binary behind.
        def _inflated_frame_count(total_frames: int, interleave_stride: int, position: int) -> int:
            """Returns an inflated per-position frame count that exceeds what the source TIFF files deliver."""
            return (total_frames // interleave_stride) + 1

        monkeypatch.setattr("cindra.io.tiff._resolve_interleave_frame_count", _inflated_frame_count)

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )

        with pytest.raises(RuntimeError, match=r"binary file\s+was sized for 4 frames"):
            convert_tiffs_to_binary(contexts=[context], workers=1)

    def test_empty_contexts_raises(self) -> None:
        """Verifies that providing no contexts raises a ValueError."""
        with pytest.raises(ValueError, match="At least one RuntimeContext"):
            convert_tiffs_to_binary(contexts=[], workers=1)

    def test_missing_data_path_raises(self, tmp_path: Path) -> None:
        """Verifies that a configuration without a data_path raises a ValueError."""
        output_path = tmp_path / "output"
        configuration = _build_configuration(data_path=None, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )

        with pytest.raises(ValueError, match="data_path must be configured"):
            convert_tiffs_to_binary(contexts=[context], workers=1)


class TestGetFrameDimensions:
    """Tests _get_frame_dimensions."""

    def test_empty_tiff_raises(self, tmp_path: Path) -> None:
        """Verifies that an empty (zero-page) first TIFF file raises a ValueError."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=1, channel_number=1)
        empty_tiff = data_path / "empty.tif"
        with TiffWriter(empty_tiff):
            pass

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )

        with pytest.raises(ValueError, match="first TIFF file is empty"):
            _get_frame_dimensions(
                tiff_files=[empty_tiff], contexts=[context], acquisition=acquisition, decode_workers=1
            )

    def test_differently_shaped_tiff_raises(self, tmp_path: Path) -> None:
        """Verifies that a TIFF holding differently shaped frames raises an error that names it and the remedy."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=1, channel_number=1)

        # Reproduces an anatomical z-stack sitting beside the recording's imaging files, which is shaped differently
        # and would otherwise fail to broadcast into a binary sized for the recording.
        recording_tiff = data_path / "mesoscope_000001.tif"
        zstack_tiff = data_path / "zstack.tif"
        _write_constant_tiff(file_path=recording_tiff, frame_values=[1, 2], height=_FRAME_HEIGHT, width=_FRAME_WIDTH)
        _write_constant_tiff(file_path=zstack_tiff, frame_values=[3], height=_FRAME_HEIGHT * 2, width=_FRAME_WIDTH * 2)

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )

        with pytest.raises(ValueError, match=r"ignored_file_names"):
            _get_frame_dimensions(
                tiff_files=[recording_tiff, zstack_tiff],
                contexts=[context],
                acquisition=acquisition,
                decode_workers=1,
            )

    def test_many_differently_shaped_tiffs_truncate_the_report(self, tmp_path: Path) -> None:
        """Verifies that more mismatched files than the report limit are summarized with a remainder count."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=1, channel_number=1)

        recording_tiff = data_path / "mesoscope_000001.tif"
        _write_constant_tiff(file_path=recording_tiff, frame_values=[1], height=_FRAME_HEIGHT, width=_FRAME_WIDTH)

        # Writes one more mismatched file than the report limit, so the message names the limit and counts the rest.
        mismatched_count = _MISMATCH_REPORT_LIMIT + 1
        mismatched_tiffs = []
        for file_index in range(mismatched_count):
            mismatched_tiff = data_path / f"zstack_{file_index}.tif"
            _write_constant_tiff(
                file_path=mismatched_tiff, frame_values=[2], height=_FRAME_HEIGHT * 2, width=_FRAME_WIDTH * 2
            )
            mismatched_tiffs.append(mismatched_tiff)

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )

        with pytest.raises(ValueError, match=r"and 1 more"):
            _get_frame_dimensions(
                tiff_files=[recording_tiff, *mismatched_tiffs],
                contexts=[context],
                acquisition=acquisition,
                decode_workers=1,
            )

    def test_uniformly_shaped_tiffs_pass(self, tmp_path: Path) -> None:
        """Verifies that files sharing a frame shape resolve dimensions without raising."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=1, channel_number=1)
        first_tiff = data_path / "mesoscope_000001.tif"
        second_tiff = data_path / "mesoscope_000002.tif"
        _write_constant_tiff(file_path=first_tiff, frame_values=[1, 2], height=_FRAME_HEIGHT, width=_FRAME_WIDTH)
        _write_constant_tiff(file_path=second_tiff, frame_values=[3, 4], height=_FRAME_HEIGHT, width=_FRAME_WIDTH)

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )

        heights, widths = _get_frame_dimensions(
            tiff_files=[first_tiff, second_tiff], contexts=[context], acquisition=acquisition, decode_workers=1
        )

        assert heights == [_FRAME_HEIGHT]
        assert widths == [_FRAME_WIDTH]


class TestCreateBinaryFiles:
    """Tests _create_binary_files."""

    def test_empty_contexts_raises(self) -> None:
        """Verifies that providing no contexts raises a ValueError."""
        with pytest.raises(ValueError, match="At least one RuntimeContext"):
            _create_binary_files(
                contexts=[], frame_heights=[], frame_widths=[], channel_1_frame_counts=[], channel_2_frame_counts=[]
            )

    def test_clears_a_stale_registration_marker(self, tmp_path: Path) -> None:
        """Verifies that rebuilding a binary clears the marker an interrupted registration left beside it."""
        output_path = tmp_path / "output"
        configuration = _build_configuration(data_path=tmp_path / "data", output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )
        binary_path = context.runtime.io.registered_binary_path
        create_registration_marker(binary_path=binary_path)
        assert resolve_registration_marker_path(binary_path=binary_path).exists()

        binaries, _ = _create_binary_files(
            contexts=[context],
            frame_heights=[_FRAME_HEIGHT],
            frame_widths=[_FRAME_WIDTH],
            channel_1_frame_counts=[1],
            channel_2_frame_counts=[1],
        )
        binaries[0].close()

        # Re-running binarization is the documented recovery path for an interrupted registration, so it must leave
        # the rebuilt binary usable rather than permanently marked.
        assert not resolve_registration_marker_path(binary_path=binary_path).exists()

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
                contexts=[context],
                frame_heights=[_FRAME_HEIGHT],
                frame_widths=[_FRAME_WIDTH],
                channel_1_frame_counts=[1],
                channel_2_frame_counts=[1],
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
                contexts=[context],
                frame_heights=[_FRAME_HEIGHT],
                frame_widths=[_FRAME_WIDTH],
                channel_1_frame_counts=[1],
                channel_2_frame_counts=[1],
            )
