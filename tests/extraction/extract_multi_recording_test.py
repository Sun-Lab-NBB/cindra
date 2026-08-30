"""Contains integration tests for the multi-recording path of the extract_traces stage entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml
import numpy as np
import pytest
from ataraxis_base_utilities import error_format, ensure_directory_exists

from cindra.io.binary import _resolve_binarization_marker_path, _resolve_registration_marker_path
from cindra.dataclasses import (
    ROIMask,
    CombinedData,
    DetectionData,
    ROIStatistics,
    ExtractionData,
    MultiRecordingRuntimeData,
    MultiRecordingConfiguration,
    MultiRecordingRuntimeContext,
)
from cindra.extraction.extract import extract_traces

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from numpy.typing import NDArray

_CONSTANT_PIXEL_VALUE: int = 500
"""The constant channel 1 pixel intensity used to build predictable synthetic movies for exact extraction oracles."""

_CONSTANT_PIXEL_VALUE_CHANNEL_2: int = 300
"""The constant channel 2 pixel intensity used to build predictable synthetic movies for exact extraction oracles."""

_FAKE_ELAPSED_SECONDS: int = 10
"""The interval every timed section reports once the stub timer replaces PrecisionTimer."""


class TestExtractMultiRecording:
    """Tests the multi-recording dispatch path of the extract_traces stage entry point."""

    def test_constant_movie_yields_weighted_sum_traces(self, tmp_path: Path) -> None:
        """Verifies that a constant movie over normalized tracked masks yields cell traces equal to the constant."""
        frame_height = frame_width = 32
        frame_count = 12
        movie = _constant_movie(
            value=_CONSTANT_PIXEL_VALUE, frame_count=frame_count, frame_height=frame_height, frame_width=frame_width
        )

        def configure(configuration: MultiRecordingConfiguration) -> None:
            configuration.signal_extraction.extract_neuropil = False
            configuration.signal_extraction.batch_size = 5
            configuration.spike_deconvolution.extract_spikes = False

        context = _build_multi_context(
            tmp_path=tmp_path, frame_height=frame_height, frame_width=frame_width, movie=movie, configure=configure
        )
        rois = _make_roi_statistics(centers=((12, 12), (20, 20)), frame_height=frame_height, frame_width=frame_width)
        context.runtime.extraction.roi_statistics = rois

        extract_traces(context=context, workers=1)

        output_directory = context.runtime.output_path
        cell_fluorescence = _load_result(output_directory=output_directory, name="cell_fluorescence")
        neuropil_fluorescence = _load_result(output_directory=output_directory, name="neuropil_fluorescence")
        subtracted_fluorescence = _load_result(output_directory=output_directory, name="subtracted_fluorescence")
        spikes = _load_result(output_directory=output_directory, name="spikes")

        assert cell_fluorescence.shape == (len(rois), frame_count)
        np.testing.assert_allclose(cell_fluorescence, float(_CONSTANT_PIXEL_VALUE), rtol=1e-4)
        # A constant movie produces identical accumulator values for every frame, so traces are flat across time.
        assert np.all(cell_fluorescence == cell_fluorescence[:, :1])
        # Neuropil extraction is disabled, so the neuropil traces remain exact zeros.
        assert np.all(neuropil_fluorescence == 0.0)
        # Spike extraction is disabled, so subtracted fluorescence and spikes are filled with exact zeros.
        assert np.all(subtracted_fluorescence == 0.0)
        assert np.all(spikes == 0.0)

    def test_extract_spikes_runs_deconvolution(self, tmp_path: Path) -> None:
        """Verifies that enabling spike extraction runs deconvolution and yields zero spikes for a constant movie."""
        frame_height = frame_width = 32
        frame_count = 40
        movie = _constant_movie(
            value=_CONSTANT_PIXEL_VALUE, frame_count=frame_count, frame_height=frame_height, frame_width=frame_width
        )

        def configure(configuration: MultiRecordingConfiguration) -> None:
            configuration.signal_extraction.extract_neuropil = False
            configuration.spike_deconvolution.extract_spikes = True
            configuration.spike_deconvolution.baseline_window = 1.0

        context = _build_multi_context(
            tmp_path=tmp_path, frame_height=frame_height, frame_width=frame_width, movie=movie, configure=configure
        )
        context.runtime.extraction.roi_statistics = _make_roi_statistics(
            centers=((16, 16),), frame_height=frame_height, frame_width=frame_width
        )

        extract_traces(context=context, workers=1)

        output_directory = context.runtime.output_path
        subtracted_fluorescence = _load_result(output_directory=output_directory, name="subtracted_fluorescence")
        spikes = _load_result(output_directory=output_directory, name="spikes")

        assert subtracted_fluorescence.shape == (1, frame_count)
        assert spikes.shape == (1, frame_count)
        # A flat baseline subtracts to zero, leaving no detectable transients for the deconvolution to recover.
        np.testing.assert_allclose(subtracted_fluorescence, 0.0, atol=1e-3)
        np.testing.assert_allclose(spikes, 0.0, atol=1e-3)

    def test_two_channel_extraction_computes_colocalization(self, tmp_path: Path) -> None:
        """Verifies that channel 2 tracked statistics drive channel 2 extraction and spatial colocalization."""
        frame_height = frame_width = 32
        frame_count = 10
        movie = _constant_movie(
            value=_CONSTANT_PIXEL_VALUE, frame_count=frame_count, frame_height=frame_height, frame_width=frame_width
        )
        movie_channel_2 = _constant_movie(
            value=_CONSTANT_PIXEL_VALUE_CHANNEL_2,
            frame_count=frame_count,
            frame_height=frame_height,
            frame_width=frame_width,
        )

        def configure(configuration: MultiRecordingConfiguration) -> None:
            configuration.signal_extraction.extract_neuropil = False
            configuration.spike_deconvolution.extract_spikes = False

        context = _build_multi_context(
            tmp_path=tmp_path,
            frame_height=frame_height,
            frame_width=frame_width,
            movie=movie,
            movie_channel_2=movie_channel_2,
            configure=configure,
        )
        centers = ((12, 12), (20, 20))
        context.runtime.extraction.roi_statistics = _make_roi_statistics(
            centers=centers, frame_height=frame_height, frame_width=frame_width
        )
        context.runtime.extraction.roi_statistics_channel_2 = _make_roi_statistics(
            centers=centers, frame_height=frame_height, frame_width=frame_width
        )

        extract_traces(context=context, workers=1)

        output_directory = context.runtime.output_path
        cell_fluorescence = _load_result(output_directory=output_directory, name="cell_fluorescence")
        cell_fluorescence_channel_2 = _load_result(
            output_directory=output_directory, name="cell_fluorescence_channel_2"
        )
        cell_colocalization = _load_result(output_directory=output_directory, name="cell_colocalization")

        np.testing.assert_allclose(cell_fluorescence, float(_CONSTANT_PIXEL_VALUE), rtol=1e-4)
        # Channel 2 reuses the same weighted-sum kernel against its own constant movie, recovering its constant.
        np.testing.assert_allclose(cell_fluorescence_channel_2, float(_CONSTANT_PIXEL_VALUE_CHANNEL_2), rtol=1e-4)
        assert cell_colocalization.shape == (len(centers), 2)

    def test_loads_roi_statistics_from_disk(self, tmp_path: Path) -> None:
        """Verifies that extraction loads tracked ROI statistics from disk when absent from the in-memory context."""
        frame_height = frame_width = 32
        frame_count = 8
        movie = _constant_movie(
            value=_CONSTANT_PIXEL_VALUE, frame_count=frame_count, frame_height=frame_height, frame_width=frame_width
        )

        def configure(configuration: MultiRecordingConfiguration) -> None:
            configuration.signal_extraction.extract_neuropil = False
            configuration.spike_deconvolution.extract_spikes = False

        context = _build_multi_context(
            tmp_path=tmp_path, frame_height=frame_height, frame_width=frame_width, movie=movie, configure=configure
        )
        rois = _make_roi_statistics(centers=((16, 16),), frame_height=frame_height, frame_width=frame_width)

        output_directory = context.runtime.output_path
        context.runtime.extraction.roi_statistics = rois
        context.runtime.extraction.save_arrays(output_path=output_directory)

        # Clears the in-memory statistics so the entry point must reload them from the saved files.
        context.runtime.extraction.roi_statistics = None

        extract_traces(context=context, workers=1)

        cell_fluorescence = _load_result(output_directory=output_directory, name="cell_fluorescence")
        assert cell_fluorescence.shape == (len(rois), frame_count)
        np.testing.assert_allclose(cell_fluorescence, float(_CONSTANT_PIXEL_VALUE), rtol=1e-4)

    def test_missing_combined_data_raises(self, tmp_path: Path) -> None:
        """Verifies that extraction raises RuntimeError when the combined single-recording data is not loaded."""
        frame_height = frame_width = 16
        movie = _constant_movie(
            value=_CONSTANT_PIXEL_VALUE, frame_count=8, frame_height=frame_height, frame_width=frame_width
        )
        context = _build_multi_context(
            tmp_path=tmp_path, frame_height=frame_height, frame_width=frame_width, movie=movie
        )
        context.runtime.combined_data = None

        with pytest.raises(RuntimeError):
            extract_traces(context=context, workers=1)

    def test_missing_roi_statistics_raises(self, tmp_path: Path) -> None:
        """Verifies that extraction raises RuntimeError when no tracked ROI statistics are available."""
        frame_height = frame_width = 16
        movie = _constant_movie(
            value=_CONSTANT_PIXEL_VALUE, frame_count=8, frame_height=frame_height, frame_width=frame_width
        )
        context = _build_multi_context(
            tmp_path=tmp_path, frame_height=frame_height, frame_width=frame_width, movie=movie
        )
        context.runtime.extraction.roi_statistics = None

        with pytest.raises(RuntimeError):
            extract_traces(context=context, workers=1)

    @pytest.mark.parametrize(
        "resolve_marker_path", [_resolve_binarization_marker_path, _resolve_registration_marker_path]
    )
    def test_marked_plane_binary_raises(self, tmp_path: Path, resolve_marker_path: Callable[..., Path]) -> None:
        """Verifies that a plane binary either phase left marked is refused instead of read as a finished movie."""
        frame_height = frame_width = 16
        frame_count = 8
        movie = _constant_movie(
            value=_CONSTANT_PIXEL_VALUE,
            frame_count=frame_count,
            frame_height=frame_height,
            frame_width=frame_width,
        )
        context = _build_multi_context(
            tmp_path=tmp_path, frame_height=frame_height, frame_width=frame_width, movie=movie
        )
        context.runtime.extraction.roi_statistics = _make_roi_statistics(
            centers=((8, 8),), frame_height=frame_height, frame_width=frame_width
        )

        # An interrupted conversion or rewrite leaves its own phase marker beside the binary it was writing.
        binary_path = context.runtime.combined_data.registered_binary_paths[0]
        marker_path = resolve_marker_path(binary_path=binary_path)
        marker_path.touch()

        # error_format wraps the message the way the console does, so the match survives the line break the console
        # inserts between 'was' and 'interrupted' when the binary path is long enough to push it there.
        expected_message = (
            f"Unable to extract multi-recording traces for recording rec0. A previous write of the binary file "
            f"'{binary_path}' was interrupted, so the file holds finished frames up to an unknown point and "
            f"unfinished frames after it. Enable 'file_io.repeat_binarization' in that recording's single-recording "
            f"configuration and re-run its single-recording pipeline, which rebuilds the binary from its source TIFF "
            f"files and clears the marker at '{marker_path}'."
        )

        with pytest.raises(RuntimeError, match=error_format(expected_message)):
            extract_traces(context=context, workers=1)

    def test_persists_disjoint_extraction_and_deconvolution_times(self, tmp_path: Path, monkeypatch) -> None:
        """Verifies that both timing segments and their total reach the runtime data file on disk."""
        frame_height = frame_width = 16
        frame_count = 40
        movie = _constant_movie(
            value=_CONSTANT_PIXEL_VALUE,
            frame_count=frame_count,
            frame_height=frame_height,
            frame_width=frame_width,
        )

        def configure(configuration: MultiRecordingConfiguration) -> None:
            configuration.signal_extraction.extract_neuropil = False
            configuration.spike_deconvolution.extract_spikes = True
            configuration.spike_deconvolution.baseline_window = 1.0

        context = _build_multi_context(
            tmp_path=tmp_path, frame_height=frame_height, frame_width=frame_width, movie=movie, configure=configure
        )
        context.runtime.extraction.roi_statistics = _make_roi_statistics(
            centers=((8, 8),), frame_height=frame_height, frame_width=frame_width
        )

        # Every timed section reports the same fixed interval, so the whole channel segment and the deconvolution
        # segment nested inside it both measure _FAKE_ELAPSED_SECONDS and the extraction remainder is zero.
        monkeypatch.setattr("cindra.extraction.extract.PrecisionTimer", _ConstantTimer)

        extract_traces(context=context, workers=1)

        timing = yaml.safe_load((context.runtime.output_path / "multi_recording_runtime_data.yaml").read_text())[
            "timing"
        ]
        assert timing["deconvolution_time"] == _FAKE_ELAPSED_SECONDS
        assert timing["extraction_time"] == 0
        assert timing["total_extraction_time"] == _FAKE_ELAPSED_SECONDS


class _ConstantTimer:
    """Replaces PrecisionTimer with a stub reporting a fixed interval, so timing oracles do not depend on wall time."""

    def __init__(self, precision=None) -> None:
        self.precision = precision

    def reset(self) -> None:
        """Accepts the reset every timed section issues without changing the reported interval."""

    @property
    def elapsed(self) -> int:
        """Returns the fixed interval every timed section reports."""
        return _FAKE_ELAPSED_SECONDS


def _make_roi(
    center: tuple[int, int],
    *,
    frame_height: int,
    frame_width: int,
    half: int = 2,
    radius: float = 3.0,
) -> ROIStatistics:
    """Creates an ROIStatistics instance backed by a small square block of unit-weight pixels."""
    center_y, center_x = center
    y_coordinates, x_coordinates = np.mgrid[
        center_y - half : center_y + half + 1, center_x - half : center_x + half + 1
    ]
    inside = (
        (y_coordinates >= 0) & (y_coordinates < frame_height) & (x_coordinates >= 0) & (x_coordinates < frame_width)
    )
    y_pixels = y_coordinates[inside].astype(np.int32)
    x_pixels = x_coordinates[inside].astype(np.int32)
    pixel_weights = np.ones(y_pixels.size, dtype=np.float32)
    mask = ROIMask(
        y_pixels=y_pixels,
        x_pixels=x_pixels,
        pixel_weights=pixel_weights,
        centroid=center,
        frame_width=frame_width,
        radius=radius,
    )
    roi = ROIStatistics(mask=mask)
    roi.pixel_count = y_pixels.size
    return roi


def _make_roi_statistics(
    centers: tuple[tuple[int, int], ...],
    *,
    frame_height: int,
    frame_width: int,
) -> list[ROIStatistics]:
    """Creates a list of ROIStatistics instances at the requested centroids."""
    return [_make_roi(center=center, frame_height=frame_height, frame_width=frame_width) for center in centers]


def _constant_movie(value: int, *, frame_count: int, frame_height: int, frame_width: int) -> NDArray[np.int16]:
    """Creates a constant-valued synthetic movie so that weighted extraction has an exact analytic oracle."""
    return np.full((frame_count, frame_height, frame_width), fill_value=value, dtype=np.int16)


def _load_result(output_directory: Path, name: str) -> NDArray[np.float32]:
    """Loads a saved extraction result array from disk after the entry point releases its in-memory copy."""
    return np.load(output_directory / f"{name}.npy")


def _build_multi_context(
    tmp_path: Path,
    *,
    frame_height: int,
    frame_width: int,
    movie: NDArray[np.int16],
    movie_channel_2: NDArray[np.int16] | None = None,
    tau: float = 1.0,
    sampling_rate: float = 30.0,
    configure: Callable[[MultiRecordingConfiguration], None] | None = None,
) -> MultiRecordingRuntimeContext:
    """Builds a minimal single-plane MultiRecordingRuntimeContext backed by synthetic constant binary movies.

    Writes the channel 1 movie, and the optional channel 2 movie, as raw int16 binaries. Wires a single-plane
    CombinedData with the combined geometry and cached binary paths, then returns a context whose extraction
    data the caller populates with backward-transformed tracked ROI statistics.
    """
    output_directory = tmp_path / "rec0" / "cindra" / "multi_recording" / "dataset"
    binary_directory = tmp_path / "rec0" / "cindra" / "binaries"
    ensure_directory_exists(output_directory)
    ensure_directory_exists(binary_directory)

    channel_1_path = binary_directory / "channel_1_data.bin"
    movie.astype(np.int16).tofile(channel_1_path)

    channel_2_paths: tuple[Path, ...] | None = None
    if movie_channel_2 is not None:
        channel_2_path = binary_directory / "channel_2_data.bin"
        movie_channel_2.astype(np.int16).tofile(channel_2_path)
        channel_2_paths = (channel_2_path,)

    combined_data = CombinedData(
        detection=DetectionData(),
        extraction=ExtractionData(),
        plane_count=1,
        combined_height=frame_height,
        combined_width=frame_width,
        tau=tau,
        sampling_rate=sampling_rate,
        plane_heights=np.array([frame_height], dtype=np.uint16),
        plane_widths=np.array([frame_width], dtype=np.uint16),
        plane_y_offsets=np.array([0], dtype=np.int32),
        plane_x_offsets=np.array([0], dtype=np.int32),
        registered_binary_paths=(channel_1_path,),
        registered_binary_paths_channel_2=channel_2_paths,
    )

    configuration = MultiRecordingConfiguration()
    if configure is not None:
        configure(configuration)

    runtime = MultiRecordingRuntimeData()
    runtime.output_path = output_directory
    runtime.io.recording_id = "rec0"
    runtime.combined_data = combined_data

    return MultiRecordingRuntimeContext(configuration=configuration, runtime=runtime)
