"""Contains integration tests for the TIFF to binary conversion stage entry points."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from tifffile import TiffFile, TiffFrame, TiffWriter

from cindra.io.tiff import (
    _MISMATCH_REPORT_LIMIT,
    TiffConversionPlan,
    _create_binary_files,
    _resolve_binary_paths,
    _resolve_plane_dimensions,
    _scan_source_frames,
    convert_tiffs_to_binary,
    resolve_tiff_conversion_plan,
)
from cindra.io.binary import create_binary_write_marker, resolve_binary_write_marker_path
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


def _write_scanimage_tiff(file_path: Path, page_shapes: list[tuple[int, int]]) -> None:
    """Writes a non-BigTIFF file carrying the ScanImage software tag, whose pages hold the requested shapes."""
    # tifffile routes the pages of such a file through TiffPages._load_virtual_frames(), which hands back frames that
    # report the shape of the file's first page instead of their own.
    with TiffWriter(file_path, bigtiff=False) as writer:
        for page_index, (height, width) in enumerate(page_shapes):
            writer.write(
                np.full((height, width), fill_value=page_index, dtype=np.int16),
                software="SI.5.6",
                contiguous=False,
            )


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


def _build_plan(*, context: RuntimeContext) -> TiffConversionPlan:
    """Builds a conversion plan that writes a single frame into every binary the given context names."""
    channel_2_path = context.runtime.io.registered_binary_path_channel_2
    return TiffConversionPlan(
        contexts=(context,),
        tiff_files=(),
        total_frames=1,
        converted_frames=1,
        batch_size=1,
        decode_workers=1,
        frame_heights=(_FRAME_HEIGHT,),
        frame_widths=(_FRAME_WIDTH,),
        channel_1_paths=(context.runtime.io.registered_binary_path,),
        channel_2_paths=() if channel_2_path is None else (channel_2_path,),
        channel_1_frame_counts=(1,),
        channel_2_frame_counts=() if channel_2_path is None else (1,),
    )


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

        convert_tiffs_to_binary(plan=resolve_tiff_conversion_plan(contexts=[context], workers=1))

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

        convert_tiffs_to_binary(plan=resolve_tiff_conversion_plan(contexts=[context_0, context_1], workers=1))

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

        convert_tiffs_to_binary(plan=resolve_tiff_conversion_plan(contexts=[context], workers=1))

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

    def test_each_channel_mean_image_uses_its_own_frame_count(self, tmp_path: Path) -> None:
        """Verifies that each channel's mean image is divided by the frames that channel received."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=1, channel_number=2)
        tiff_path = data_path / "recording.tif"
        _write_constant_tiff(file_path=tiff_path, frame_values=[10, 4, 20], height=_FRAME_HEIGHT, width=_FRAME_WIDTH)

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        configuration.main.two_channels = True
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=2)
        context = _build_context(
            output_path=output_path,
            configuration=configuration,
            acquisition=acquisition,
            plane_index=0,
            two_channels=True,
        )

        # Budgets three frames of a two-frame interleave cycle, which hands channel 1 the frames valued 10 and 20 and
        # channel 2 the single frame valued 4. Stating both counts here rather than resolving them keeps the assertion
        # meaningful if the two channels of a plane ever stop receiving the same count.
        plan = TiffConversionPlan(
            contexts=(context,),
            tiff_files=(tiff_path,),
            total_frames=3,
            converted_frames=3,
            batch_size=4,
            decode_workers=1,
            frame_heights=(_FRAME_HEIGHT,),
            frame_widths=(_FRAME_WIDTH,),
            channel_1_paths=(context.runtime.io.registered_binary_path,),
            channel_2_paths=(context.runtime.io.registered_binary_path_channel_2,),
            channel_1_frame_counts=(2,),
            channel_2_frame_counts=(1,),
        )

        convert_tiffs_to_binary(plan=plan)

        assert context.runtime.io.frame_count == 2
        assert np.allclose(context.runtime.detection.mean_image, 15.0)
        # Dividing channel 2 by the two frames channel 1 received would halve this to 2.0.
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

        convert_tiffs_to_binary(plan=resolve_tiff_conversion_plan(contexts=[context], workers=1))

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

        convert_tiffs_to_binary(plan=resolve_tiff_conversion_plan(contexts=[context], workers=1))

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

        convert_tiffs_to_binary(plan=resolve_tiff_conversion_plan(contexts=[context], workers=1))

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

    def test_mroi_incomplete_final_volume_is_discarded_for_every_roi(
        self, tmp_path: Path, read_binary_movie: Callable[[Path, int, int], NDArray[np.int16]]
    ) -> None:
        """Verifies that MROI planes sharing a physical plane receive the same frames whatever ROI they belong to."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=2, channel_number=1)
        # Five pages over a two-plane interleave end the recording partway through a volume, which is what an
        # operator-stopped acquisition leaves behind. The trailing frame reaches the leading plane alone, so the
        # conversion discards it and every virtual plane of every ROI holds the two whole volumes that precede it.
        frame_values = [0, 1, 2, 3, 4]
        _write_constant_tiff(
            file_path=data_path / "recording.tif", frame_values=frame_values, height=_FRAME_HEIGHT, width=_FRAME_WIDTH
        )

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=2, channel_number=1, roi_number=2)
        roi_lines = ((0, 1, 2, 3), (4, 5, 6, 7))
        contexts = [
            _build_context(
                output_path=output_path,
                configuration=configuration,
                acquisition=acquisition,
                plane_index=virtual_plane_index,
                mroi_lines=roi_lines[virtual_plane_index // acquisition.plane_number],
            )
            for virtual_plane_index in range(acquisition.plane_number * acquisition.roi_number)
        ]

        convert_tiffs_to_binary(plan=resolve_tiff_conversion_plan(contexts=contexts, workers=1))

        roi_height = len(roi_lines[0])
        # Virtual planes 0 and 2 belong to physical plane 0 and virtual planes 1 and 3 belong to physical plane 1, so
        # the two members of each pair hold the same frames of the interleave cycle.
        expected_selections = [[0, 2], [1, 3], [0, 2], [1, 3]]
        for context, selection in zip(contexts, expected_selections, strict=True):
            io_data = context.runtime.io
            assert io_data.frame_count == len(selection)
            binary = read_binary_movie(
                file_path=io_data.registered_binary_path, frame_height=roi_height, frame_width=_FRAME_WIDTH
            )
            assert np.array_equal(
                binary, _constant_stack(frame_values=selection, height=roi_height, width=_FRAME_WIDTH)
            )

    def test_multiple_files_skip_a_plane_with_no_frames_in_a_batch(
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

        convert_tiffs_to_binary(plan=resolve_tiff_conversion_plan(contexts=[context_0, context_1], workers=1))

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

        convert_tiffs_to_binary(plan=resolve_tiff_conversion_plan(contexts=contexts, workers=1))

        # The eleven pages fill two whole four-frame cycles, so the conversion stops two frames into the second file.
        expected_selections = [([0, 4], [1, 5]), ([2, 6], [3, 7])]
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

    def test_incomplete_final_cycle_is_discarded(
        self, tmp_path: Path, read_binary_movie: Callable[[Path, int, int], NDArray[np.int16]]
    ) -> None:
        """Verifies that the frames of an incomplete final interleave cycle reach no plane and no channel."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=2, channel_number=2)
        # Eleven pages over a four-frame interleave cycle leave a trailing three-frame remainder, which is what an
        # acquisition stopped partway through a volume delivers.
        _write_constant_tiff(
            file_path=data_path / "recording.tif",
            frame_values=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            height=_FRAME_HEIGHT,
            width=_FRAME_WIDTH,
        )

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        configuration.main.two_channels = True
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

        plan = resolve_tiff_conversion_plan(contexts=contexts, workers=1)
        convert_tiffs_to_binary(plan=plan)

        assert plan.total_frames == 11
        assert plan.converted_frames == 8
        assert plan.channel_1_frame_counts == (2, 2)
        assert plan.channel_2_frame_counts == (2, 2)

        expected_selections = [([0, 4], [1, 5]), ([2, 6], [3, 7])]
        for context, (channel_1_values, channel_2_values) in zip(contexts, expected_selections, strict=True):
            io_data = context.runtime.io
            assert io_data.frame_count == 2
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

    def test_ragged_files_write_the_channel_a_short_batch_reaches(
        self, tmp_path: Path, read_binary_movie: Callable[[Path, int, int], NDArray[np.int16]]
    ) -> None:
        """Verifies that a batch covering channel 2 alone writes it even though channel 1 receives nothing."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=2, channel_number=2)
        # Five, four, and three pages over a four-frame interleave cycle leave the final file starting at interleave
        # position 1, so its three frames cover channel 2 of plane 0 while missing channel 1 of the same plane.
        _write_constant_tiff(
            file_path=data_path / "recording_0.tif",
            frame_values=[0, 1, 2, 3, 4],
            height=_FRAME_HEIGHT,
            width=_FRAME_WIDTH,
        )
        _write_constant_tiff(
            file_path=data_path / "recording_1.tif",
            frame_values=[5, 6, 7, 8],
            height=_FRAME_HEIGHT,
            width=_FRAME_WIDTH,
        )
        _write_constant_tiff(
            file_path=data_path / "recording_2.tif", frame_values=[9, 10, 11], height=_FRAME_HEIGHT, width=_FRAME_WIDTH
        )

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        configuration.main.two_channels = True
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

        convert_tiffs_to_binary(plan=resolve_tiff_conversion_plan(contexts=contexts, workers=1))

        expected_selections = [([0, 4, 8], [1, 5, 9]), ([2, 6, 10], [3, 7, 11])]
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

    def test_file_beyond_the_converted_budget_is_not_read(
        self, tmp_path: Path, read_binary_movie: Callable[[Path, int, int], NDArray[np.int16]]
    ) -> None:
        """Verifies that a trailing file holding only discarded frames contributes nothing to any binary."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=2, channel_number=2)
        # The first file holds one whole four-frame interleave cycle and the second holds half of the next one, so the
        # budget is exhausted before the second file is opened.
        _write_constant_tiff(
            file_path=data_path / "recording_0.tif",
            frame_values=[0, 1, 2, 3],
            height=_FRAME_HEIGHT,
            width=_FRAME_WIDTH,
        )
        _write_constant_tiff(
            file_path=data_path / "recording_1.tif", frame_values=[4, 5], height=_FRAME_HEIGHT, width=_FRAME_WIDTH
        )

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        configuration.main.two_channels = True
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

        plan = resolve_tiff_conversion_plan(contexts=contexts, workers=1)
        convert_tiffs_to_binary(plan=plan)

        assert plan.total_frames == 6
        assert plan.converted_frames == 4

        expected_selections = [([0], [1]), ([2], [3])]
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

        convert_tiffs_to_binary(plan=resolve_tiff_conversion_plan(contexts=[context], workers=1))

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

        convert_tiffs_to_binary(plan=resolve_tiff_conversion_plan(contexts=[context], workers=1))

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
        def _inflated_frame_count(total_frames: int, interleave_stride: int) -> int:
            """Returns an inflated per-position frame count that exceeds what the source TIFF files deliver."""
            return (total_frames // interleave_stride) + 1

        monkeypatch.setattr("cindra.io.tiff._resolve_interleave_frame_count", _inflated_frame_count)

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )

        with pytest.raises(RuntimeError, match=r"binary file\s+was sized for 4 frames"):
            convert_tiffs_to_binary(plan=resolve_tiff_conversion_plan(contexts=[context], workers=1))

    def test_completed_conversion_clears_the_binary_marks(self, tmp_path: Path) -> None:
        """Verifies that a conversion that writes every frame leaves both channel binaries unmarked."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=1, channel_number=2)
        _write_constant_tiff(
            file_path=data_path / "recording.tif",
            frame_values=[0, 1, 2, 3],
            height=_FRAME_HEIGHT,
            width=_FRAME_WIDTH,
        )

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        configuration.main.two_channels = True
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=2)
        context = _build_context(
            output_path=output_path,
            configuration=configuration,
            acquisition=acquisition,
            plane_index=0,
            two_channels=True,
        )

        convert_tiffs_to_binary(plan=resolve_tiff_conversion_plan(contexts=[context], workers=1))

        io_data = context.runtime.io
        assert not resolve_binary_write_marker_path(binary_path=io_data.registered_binary_path).exists()
        assert not resolve_binary_write_marker_path(binary_path=io_data.registered_binary_path_channel_2).exists()

    def test_interrupted_conversion_leaves_the_binary_marked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verifies that a conversion that fails partway leaves its full-size binary marked for rebuilding."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=1, channel_number=1)
        frame_values = [0, 1, 2, 3, 4]
        _write_constant_tiff(
            file_path=data_path / "recording.tif", frame_values=frame_values, height=_FRAME_HEIGHT, width=_FRAME_WIDTH
        )

        # Reproduces an interruption that reaches the conversion loop after the destination binary has been opened.
        def _interrupted_read(
            tiff: object, start_index: int, batch_size: int, decode_workers: int
        ) -> NDArray[np.int16] | None:
            """Fails the first read of the conversion loop, leaving the destination binary untouched."""
            raise RuntimeError("Simulated conversion interruption")

        monkeypatch.setattr("cindra.io.tiff._read_tiff", _interrupted_read)

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )

        with pytest.raises(RuntimeError, match="Simulated conversion interruption"):
            convert_tiffs_to_binary(plan=resolve_tiff_conversion_plan(contexts=[context], workers=1))

        binary_path = context.runtime.io.registered_binary_path
        # The abandoned binary holds the exact byte count a finished conversion would leave behind, with every frame
        # still zero, so the mark beside it is the only record that its contents are incomplete.
        assert binary_path.stat().st_size == len(frame_values) * _FRAME_HEIGHT * _FRAME_WIDTH * 2
        assert resolve_binary_write_marker_path(binary_path=binary_path).exists()


class TestResolveTiffConversionPlan:
    """Tests resolve_tiff_conversion_plan."""

    def test_empty_contexts_raises(self) -> None:
        """Verifies that providing no contexts raises a ValueError."""
        with pytest.raises(ValueError, match="At least one RuntimeContext"):
            resolve_tiff_conversion_plan(contexts=[], workers=1)

    def test_missing_data_path_raises(self, tmp_path: Path) -> None:
        """Verifies that a configuration without a data_path raises a ValueError."""
        output_path = tmp_path / "output"
        configuration = _build_configuration(data_path=None, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )

        with pytest.raises(ValueError, match="data_path must be configured"):
            resolve_tiff_conversion_plan(contexts=[context], workers=1)

    def test_resolution_leaves_the_previous_binary_in_place(self, tmp_path: Path) -> None:
        """Verifies that resolving a plan changes nothing on disk, which is what lets the caller delete after it."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=1, channel_number=1)
        _write_constant_tiff(
            file_path=data_path / "recording.tif", frame_values=[0, 1], height=_FRAME_HEIGHT, width=_FRAME_WIDTH
        )

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )
        binary_path = context.runtime.io.registered_binary_path
        binary_path.write_bytes(b"previous binary")

        plan = resolve_tiff_conversion_plan(contexts=[context], workers=1)

        assert plan.total_frames == 2
        assert plan.converted_frames == 2
        assert plan.channel_1_paths == (binary_path,)
        assert plan.channel_2_paths == ()
        assert binary_path.read_bytes() == b"previous binary"
        assert not resolve_binary_write_marker_path(binary_path=binary_path).exists()

    def test_recording_without_a_complete_cycle_raises(self, tmp_path: Path) -> None:
        """Verifies that a recording too short to fill one interleave cycle is rejected before a plan is returned."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=2, channel_number=1)

        # A single frame covers the first position of the two-plane interleave cycle and leaves the second empty.
        _write_constant_tiff(
            file_path=data_path / "recording.tif", frame_values=[0], height=_FRAME_HEIGHT, width=_FRAME_WIDTH
        )

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=2, channel_number=1)
        contexts = [
            _build_context(
                output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=plane_index
            )
            for plane_index in range(2)
        ]

        with pytest.raises(ValueError, match=r"do\s+not\s+fill\s+one\s+2\s+frame\s+plane\s+and\s+channel"):
            resolve_tiff_conversion_plan(contexts=contexts, workers=1)


class TestResolveBinaryPaths:
    """Tests _resolve_binary_paths."""

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
            _resolve_binary_paths(contexts=[context])

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
            _resolve_binary_paths(contexts=[context])


class TestScanSourceFrames:
    """Tests _scan_source_frames."""

    def test_empty_tiff_raises(self, tmp_path: Path) -> None:
        """Verifies that an empty (zero-page) first TIFF file raises a ValueError."""
        data_path = tmp_path / "data"
        data_path.mkdir(parents=True, exist_ok=True)
        empty_tiff = data_path / "empty.tif"
        with TiffWriter(empty_tiff):
            pass

        with pytest.raises(ValueError, match="first TIFF file is empty"):
            _scan_source_frames(tiff_files=[empty_tiff])

    def test_differently_shaped_tiff_raises(self, tmp_path: Path) -> None:
        """Verifies that a TIFF holding differently shaped frames raises an error that names it and the remedy."""
        data_path = tmp_path / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        # Reproduces an anatomical z-stack sitting beside the recording's imaging files, which is shaped differently
        # and would otherwise fail to broadcast into a binary sized for the recording.
        recording_tiff = data_path / "mesoscope_000001.tif"
        zstack_tiff = data_path / "zstack.tif"
        _write_constant_tiff(file_path=recording_tiff, frame_values=[1, 2], height=_FRAME_HEIGHT, width=_FRAME_WIDTH)
        _write_constant_tiff(file_path=zstack_tiff, frame_values=[3], height=_FRAME_HEIGHT * 2, width=_FRAME_WIDTH * 2)

        with pytest.raises(ValueError, match=r"ignored_file_names"):
            _scan_source_frames(tiff_files=[recording_tiff, zstack_tiff])

    def test_scanimage_page_shape_is_read_from_its_own_header(self, tmp_path: Path) -> None:
        """Verifies that a ScanImage classic file holding a differently shaped page is rejected."""
        data_path = tmp_path / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        recording_tiff = data_path / "mesoscope_000001.tif"
        page_shapes = [(_FRAME_HEIGHT, _FRAME_WIDTH)] * 8
        page_shapes[5] = (_FRAME_HEIGHT * 2, _FRAME_WIDTH * 2)
        _write_scanimage_tiff(file_path=recording_tiff, page_shapes=page_shapes)

        # Every page of such a file past the first inherits the shape tifffile read from the first, so the differing
        # page is invisible to any check reading the shape the page itself reports.
        with TiffFile(recording_tiff) as tiff:
            assert {page.shape for page in tiff.pages} == {(_FRAME_HEIGHT, _FRAME_WIDTH)}

        with pytest.raises(ValueError, match=r"mesoscope_000001\.tif"):
            _scan_source_frames(tiff_files=[recording_tiff])

    def test_page_without_a_readable_header_is_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verifies that a page the file holds no readable header for is accepted rather than reported."""
        data_path = tmp_path / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        recording_tiff = data_path / "mesoscope_000001.tif"
        _write_scanimage_tiff(file_path=recording_tiff, page_shapes=[(_FRAME_HEIGHT, _FRAME_WIDTH)] * 8)

        # Reproduces a ScanImage classic file addressing its pages arithmetically past the two gigabyte offset
        # ceiling, whose frames carry no header of their own to read.
        def _unreadable_header(frame: TiffFrame) -> None:
            """Fails every attempt to read a frame's own header, as a frame without one does."""
            raise ValueError("cannot return virtual frame as page")

        monkeypatch.setattr(TiffFrame, "aspage", _unreadable_header)

        total_frames, base_height, base_width = _scan_source_frames(tiff_files=[recording_tiff])

        assert total_frames > 0
        assert (base_height, base_width) == (_FRAME_HEIGHT, _FRAME_WIDTH)

    def test_counts_every_frame_and_opens_each_file_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verifies that the frame count and the shape comparison share one open of each source file."""
        data_path = tmp_path / "data"
        data_path.mkdir(parents=True, exist_ok=True)

        tiff_files = []
        for file_index in range(3):
            tiff_file = data_path / f"mesoscope_{file_index:06d}.tif"
            _write_constant_tiff(file_path=tiff_file, frame_values=[1, 2], height=_FRAME_HEIGHT, width=_FRAME_WIDTH)
            tiff_files.append(tiff_file)

        opened: list[Path] = []
        open_tiff_file = TiffFile

        def _counting_open(file_path: Path) -> TiffFile:
            """Records the file being opened and hands back the TiffFile instance the scan works through."""
            opened.append(file_path)
            return open_tiff_file(file_path)

        monkeypatch.setattr("cindra.io.tiff.TiffFile", _counting_open)

        total_frames, base_height, base_width = _scan_source_frames(tiff_files=tiff_files)

        assert opened == tiff_files
        assert total_frames == 6
        assert (base_height, base_width) == (_FRAME_HEIGHT, _FRAME_WIDTH)

    def test_file_with_differently_shaped_pages_raises(self, tmp_path: Path) -> None:
        """Verifies that a page shaped unlike its own file's first page is rejected before the conversion runs."""
        data_path = tmp_path / "data"
        output_path = tmp_path / "output"
        _write_parameters_json(directory=data_path, plane_number=1, channel_number=1)

        # Reproduces a recording file whose trailing page holds a differently shaped frame, which every check reading
        # the first page of each file alone accepts and the conversion loop then fails to broadcast.
        recording_tiff = data_path / "mesoscope_000001.tif"
        with TiffWriter(recording_tiff) as writer:
            writer.write(np.full((_FRAME_HEIGHT, _FRAME_WIDTH), fill_value=1, dtype=np.int16))
            writer.write(np.full((_FRAME_HEIGHT * 2, _FRAME_WIDTH * 2), fill_value=2, dtype=np.int16))

        configuration = _build_configuration(data_path=data_path, output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )

        with pytest.raises(ValueError, match=r"mesoscope_000001\.tif"):
            resolve_tiff_conversion_plan(contexts=[context], workers=1)

    def test_many_differently_shaped_tiffs_truncate_the_report(self, tmp_path: Path) -> None:
        """Verifies that more mismatched files than the report limit are summarized with a remainder count."""
        data_path = tmp_path / "data"
        data_path.mkdir(parents=True, exist_ok=True)

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

        with pytest.raises(ValueError, match=r"and 1 more"):
            _scan_source_frames(tiff_files=[recording_tiff, *mismatched_tiffs])


class TestResolvePlaneDimensions:
    """Tests _resolve_plane_dimensions."""

    def test_plane_receives_the_source_frame_shape(self, tmp_path: Path) -> None:
        """Verifies that a plane carrying no ROI line range is sized by the source frame shape."""
        output_path = tmp_path / "output"
        configuration = _build_configuration(data_path=tmp_path / "data", output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )

        heights, widths = _resolve_plane_dimensions(
            contexts=[context], acquisition=acquisition, base_height=_FRAME_HEIGHT, base_width=_FRAME_WIDTH
        )

        assert heights == [_FRAME_HEIGHT]
        assert widths == [_FRAME_WIDTH]


class TestCreateBinaryFiles:
    """Tests _create_binary_files."""

    def test_marks_the_binary_it_opens(self, tmp_path: Path) -> None:
        """Verifies that a freshly opened binary carries the mark that flags its contents as incomplete."""
        output_path = tmp_path / "output"
        configuration = _build_configuration(data_path=tmp_path / "data", output_path=output_path)
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        context = _build_context(
            output_path=output_path, configuration=configuration, acquisition=acquisition, plane_index=0
        )
        binary_path = context.runtime.io.registered_binary_path
        create_binary_write_marker(binary_path=binary_path)

        binaries, _ = _create_binary_files(plan=_build_plan(context=context))
        binaries[0].close()

        # The binary is sized to its full frame count the moment it is opened, so nothing but the mark separates it
        # from a finished conversion until the caller writes every frame and clears the mark.
        assert resolve_binary_write_marker_path(binary_path=binary_path).exists()

    def test_marks_both_channel_binaries_it_opens(self, tmp_path: Path) -> None:
        """Verifies that a two-channel plane carries the mark on both of the binaries the conversion opens."""
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

        channel_1_binaries, channel_2_binaries = _create_binary_files(plan=_build_plan(context=context))
        channel_1_binaries[0].close()
        channel_2_binaries[0].close()

        io_data = context.runtime.io
        assert resolve_binary_write_marker_path(binary_path=io_data.registered_binary_path).exists()
        assert resolve_binary_write_marker_path(binary_path=io_data.registered_binary_path_channel_2).exists()
