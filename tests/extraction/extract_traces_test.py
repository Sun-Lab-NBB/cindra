"""Contains integration tests for the extract_traces stage entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from cindra.dataclasses import (
    ROIMask,
    ROIStatistics,
    MultiRecordingRuntimeData,
    MultiRecordingConfiguration,
    MultiRecordingRuntimeContext,
)
from cindra.extraction.extract import (
    extract_traces,
    _extract_functional_channel_2,
    _extract_structural_channel_2,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from numpy.typing import NDArray

    from cindra.dataclasses import RuntimeContext, SingleRecordingConfiguration

_CONSTANT_PIXEL_VALUE: int = 500
"""The constant pixel intensity used to build predictable synthetic movies for exact extraction oracles."""


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
    y_coordinates: list[int] = []
    x_coordinates: list[int] = []
    for delta_y in range(-half, half + 1):
        for delta_x in range(-half, half + 1):
            pixel_y = center_y + delta_y
            pixel_x = center_x + delta_x
            if 0 <= pixel_y < frame_height and 0 <= pixel_x < frame_width:
                y_coordinates.append(pixel_y)
                x_coordinates.append(pixel_x)

    y_pixels = np.array(y_coordinates, dtype=np.int32)
    x_pixels = np.array(x_coordinates, dtype=np.int32)
    pixel_weights = np.ones(len(y_coordinates), dtype=np.float32)
    mask = ROIMask(
        y_pixels=y_pixels,
        x_pixels=x_pixels,
        pixel_weights=pixel_weights,
        centroid=center,
        frame_width=frame_width,
        radius=radius,
    )
    roi = ROIStatistics(mask=mask)
    roi.pixel_count = len(y_coordinates)
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


def _load_result(plane_directory: Path, name: str) -> NDArray[np.float32]:
    """Loads a saved extraction result array from disk after the entry point releases its in-memory copy."""
    return np.load(plane_directory / f"{name}.npy")


class TestExtractTracesSingleRecording:
    """Tests the single-recording dispatch path of the extract_traces stage entry point."""

    def test_constant_movie_cell_fluorescence_matches_weighted_sum(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that a constant movie over normalized masks yields cell traces equal to the constant value."""
        frame_height = frame_width = 48
        frame_count = 12
        movie = _constant_movie(
            _CONSTANT_PIXEL_VALUE, frame_count=frame_count, frame_height=frame_height, frame_width=frame_width
        )

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.signal_extraction.extract_neuropil = False
            configuration.signal_extraction.batch_size = 5
            configuration.spike_deconvolution.extract_spikes = False

        context = single_recording_context(
            tmp_path,
            frame_height=frame_height,
            frame_width=frame_width,
            frame_count=frame_count,
            movie=movie,
            configure=configure,
        )
        rois = _make_roi_statistics(((12, 12), (30, 30)), frame_height=frame_height, frame_width=frame_width)
        context.runtime.extraction.roi_statistics = rois

        extract_traces(context=context)

        plane_directory = context.runtime.io.output_path
        cell_fluorescence = _load_result(plane_directory, "cell_fluorescence")
        neuropil_fluorescence = _load_result(plane_directory, "neuropil_fluorescence")
        subtracted_fluorescence = _load_result(plane_directory, "subtracted_fluorescence")
        spikes = _load_result(plane_directory, "spikes")

        assert cell_fluorescence.shape == (len(rois), frame_count)
        np.testing.assert_allclose(cell_fluorescence, float(_CONSTANT_PIXEL_VALUE), rtol=1e-4)
        # A constant movie produces identical accumulator values for every frame, so traces are flat across time.
        assert np.all(cell_fluorescence == cell_fluorescence[:, :1])
        # Neuropil extraction is disabled, so the neuropil traces remain exact zeros.
        assert np.all(neuropil_fluorescence == 0.0)
        # Spike extraction is disabled, so subtracted fluorescence and spikes are filled with exact zeros.
        assert np.all(subtracted_fluorescence == 0.0)
        assert np.all(spikes == 0.0)

    def test_neuropil_extraction_constant_movie(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that enabling neuropil extraction produces neuropil traces equal to the constant value."""
        frame_height = frame_width = 48
        frame_count = 10
        movie = _constant_movie(
            _CONSTANT_PIXEL_VALUE, frame_count=frame_count, frame_height=frame_height, frame_width=frame_width
        )

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.signal_extraction.extract_neuropil = True
            configuration.signal_extraction.minimum_neuropil_pixels = 10
            configuration.spike_deconvolution.extract_spikes = False

        context = single_recording_context(
            tmp_path,
            frame_height=frame_height,
            frame_width=frame_width,
            frame_count=frame_count,
            movie=movie,
            configure=configure,
        )
        context.runtime.extraction.roi_statistics = _make_roi_statistics(
            ((14, 14),), frame_height=frame_height, frame_width=frame_width
        )

        extract_traces(context=context)

        plane_directory = context.runtime.io.output_path
        cell_fluorescence = _load_result(plane_directory, "cell_fluorescence")
        neuropil_fluorescence = _load_result(plane_directory, "neuropil_fluorescence")

        np.testing.assert_allclose(cell_fluorescence, float(_CONSTANT_PIXEL_VALUE), rtol=1e-4)
        # The neuropil region averages a constant movie, recovering the same constant intensity.
        np.testing.assert_allclose(neuropil_fluorescence, float(_CONSTANT_PIXEL_VALUE), rtol=1e-4)

    def test_extract_spikes_runs_deconvolution(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that enabling spike extraction runs deconvolution and yields zero spikes for a constant movie."""
        frame_height = frame_width = 48
        frame_count = 40
        movie = _constant_movie(
            _CONSTANT_PIXEL_VALUE, frame_count=frame_count, frame_height=frame_height, frame_width=frame_width
        )

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.signal_extraction.extract_neuropil = False
            configuration.spike_deconvolution.extract_spikes = True
            configuration.spike_deconvolution.baseline_window = 1.0

        context = single_recording_context(
            tmp_path,
            frame_height=frame_height,
            frame_width=frame_width,
            frame_count=frame_count,
            movie=movie,
            configure=configure,
        )
        context.runtime.extraction.roi_statistics = _make_roi_statistics(
            ((20, 20),), frame_height=frame_height, frame_width=frame_width
        )

        extract_traces(context=context)

        plane_directory = context.runtime.io.output_path
        subtracted_fluorescence = _load_result(plane_directory, "subtracted_fluorescence")
        spikes = _load_result(plane_directory, "spikes")

        assert subtracted_fluorescence.shape == (1, frame_count)
        assert spikes.shape == (1, frame_count)
        # A flat baseline subtracts to zero, leaving no detectable transients for the deconvolution to recover.
        np.testing.assert_allclose(subtracted_fluorescence, 0.0, atol=1e-3)
        np.testing.assert_allclose(spikes, 0.0, atol=1e-3)

    def test_loads_roi_statistics_from_disk(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that extraction loads ROI statistics from disk when they are absent from the in-memory context."""
        frame_height = frame_width = 48
        frame_count = 8
        movie = _constant_movie(
            _CONSTANT_PIXEL_VALUE, frame_count=frame_count, frame_height=frame_height, frame_width=frame_width
        )

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.signal_extraction.extract_neuropil = False
            configuration.spike_deconvolution.extract_spikes = False

        context = single_recording_context(
            tmp_path,
            frame_height=frame_height,
            frame_width=frame_width,
            frame_count=frame_count,
            movie=movie,
            configure=configure,
        )
        rois = _make_roi_statistics(((16, 16),), frame_height=frame_height, frame_width=frame_width)

        plane_directory = context.runtime.io.output_path
        context.runtime.extraction.roi_statistics = rois
        context.runtime.extraction.save_arrays(output_path=plane_directory)

        # Clears the in-memory statistics so the entry point must reload them from the saved files.
        context.runtime.extraction.roi_statistics = None

        extract_traces(context=context)

        cell_fluorescence = _load_result(plane_directory, "cell_fluorescence")
        assert cell_fluorescence.shape == (len(rois), frame_count)

    def test_missing_roi_statistics_raises(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that extraction raises RuntimeError when no ROI statistics are available in memory or on disk."""
        context = single_recording_context(tmp_path, frame_height=48, frame_width=48, frame_count=8)
        context.runtime.extraction.roi_statistics = None

        with pytest.raises(RuntimeError):
            extract_traces(context=context)

    def test_missing_registered_binary_path_raises(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that extraction raises RuntimeError when the channel 1 registered binary path is unset."""
        context = single_recording_context(tmp_path, frame_height=48, frame_width=48, frame_count=8)
        context.runtime.extraction.roi_statistics = _make_roi_statistics(((16, 16),), frame_height=48, frame_width=48)
        context.runtime.io.registered_binary_path = None

        with pytest.raises(RuntimeError):
            extract_traces(context=context)


class TestExtractTracesChannel2:
    """Tests the dual-channel structural and functional channel 2 extraction paths."""

    def test_structural_channel_2_with_colocalization(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that structural channel 2 extraction reuses channel 1 masks and computes colocalization."""
        frame_height = frame_width = 48
        frame_count = 10
        movie = _constant_movie(
            _CONSTANT_PIXEL_VALUE, frame_count=frame_count, frame_height=frame_height, frame_width=frame_width
        )
        movie_channel_2 = _constant_movie(
            300, frame_count=frame_count, frame_height=frame_height, frame_width=frame_width
        )

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.signal_extraction.minimum_neuropil_pixels = 10
            configuration.spike_deconvolution.extract_spikes = False

        context = single_recording_context(
            tmp_path,
            frame_height=frame_height,
            frame_width=frame_width,
            frame_count=frame_count,
            movie=movie,
            movie_channel_2=movie_channel_2,
            configure=configure,
        )
        context.runtime.extraction.roi_statistics = _make_roi_statistics(
            ((14, 14), (30, 30)), frame_height=frame_height, frame_width=frame_width
        )
        context.runtime.detection.mean_image = np.full((frame_height, frame_width), 100.0, dtype=np.float32)
        context.runtime.detection.mean_image_channel_2 = np.full((frame_height, frame_width), 80.0, dtype=np.float32)

        extract_traces(context=context)

        plane_directory = context.runtime.io.output_path
        cell_fluorescence_channel_2 = _load_result(plane_directory, "cell_fluorescence_channel_2")
        cell_colocalization = _load_result(plane_directory, "cell_colocalization")
        corrected_structural_mean_image = _load_result(plane_directory, "corrected_structural_mean_image")

        assert cell_fluorescence_channel_2.shape == (2, frame_count)
        assert cell_colocalization.shape == (2, 2)
        assert corrected_structural_mean_image.shape == (frame_height, frame_width)

    def test_structural_channel_2_without_mean_images_skips_colocalization(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that structural channel 2 extraction skips colocalization when mean images are unavailable."""
        frame_height = frame_width = 48
        frame_count = 8
        movie = _constant_movie(
            _CONSTANT_PIXEL_VALUE, frame_count=frame_count, frame_height=frame_height, frame_width=frame_width
        )
        movie_channel_2 = _constant_movie(
            300, frame_count=frame_count, frame_height=frame_height, frame_width=frame_width
        )

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.signal_extraction.extract_neuropil = False
            configuration.spike_deconvolution.extract_spikes = False

        context = single_recording_context(
            tmp_path,
            frame_height=frame_height,
            frame_width=frame_width,
            frame_count=frame_count,
            movie=movie,
            movie_channel_2=movie_channel_2,
            configure=configure,
        )
        context.runtime.extraction.roi_statistics = _make_roi_statistics(
            ((20, 20),), frame_height=frame_height, frame_width=frame_width
        )

        extract_traces(context=context)

        plane_directory = context.runtime.io.output_path
        cell_fluorescence_channel_2 = _load_result(plane_directory, "cell_fluorescence_channel_2")
        assert cell_fluorescence_channel_2.shape == (1, frame_count)
        # The colocalization step is skipped without both mean images, so its output file is never written.
        assert not (plane_directory / "cell_colocalization.npy").exists()

    def test_functional_channel_2_with_spikes(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that functional channel 2 extraction classifies, deconvolves, and computes colocalization."""
        frame_height = frame_width = 48
        frame_count = 40
        movie = _constant_movie(
            _CONSTANT_PIXEL_VALUE, frame_count=frame_count, frame_height=frame_height, frame_width=frame_width
        )
        movie_channel_2 = _constant_movie(
            400, frame_count=frame_count, frame_height=frame_height, frame_width=frame_width
        )

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.main.second_channel_functional = True
            configuration.signal_extraction.extract_neuropil = False
            configuration.spike_deconvolution.extract_spikes = True
            configuration.spike_deconvolution.baseline_window = 1.0

        context = single_recording_context(
            tmp_path,
            frame_height=frame_height,
            frame_width=frame_width,
            frame_count=frame_count,
            movie=movie,
            movie_channel_2=movie_channel_2,
            configure=configure,
        )
        context.runtime.extraction.roi_statistics = _make_roi_statistics(
            ((14, 14),), frame_height=frame_height, frame_width=frame_width
        )
        context.runtime.extraction.roi_statistics_channel_2 = _make_roi_statistics(
            ((30, 30),), frame_height=frame_height, frame_width=frame_width
        )

        extract_traces(context=context)

        plane_directory = context.runtime.io.output_path
        cell_fluorescence_channel_2 = _load_result(plane_directory, "cell_fluorescence_channel_2")
        cell_classification_channel_2 = _load_result(plane_directory, "cell_classification_channel_2")
        spikes_channel_2 = _load_result(plane_directory, "spikes_channel_2")
        cell_colocalization = _load_result(plane_directory, "cell_colocalization")

        assert cell_fluorescence_channel_2.shape == (1, frame_count)
        assert cell_classification_channel_2.shape == (1, 2)
        assert spikes_channel_2.shape == (1, frame_count)
        assert cell_colocalization.shape == (1, 2)

    def test_functional_channel_2_without_spikes(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that functional channel 2 extraction fills zero spikes when spike extraction is disabled."""
        frame_height = frame_width = 48
        frame_count = 10
        movie = _constant_movie(
            _CONSTANT_PIXEL_VALUE, frame_count=frame_count, frame_height=frame_height, frame_width=frame_width
        )
        movie_channel_2 = _constant_movie(
            400, frame_count=frame_count, frame_height=frame_height, frame_width=frame_width
        )

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.main.second_channel_functional = True
            configuration.signal_extraction.extract_neuropil = False
            configuration.spike_deconvolution.extract_spikes = False

        context = single_recording_context(
            tmp_path,
            frame_height=frame_height,
            frame_width=frame_width,
            frame_count=frame_count,
            movie=movie,
            movie_channel_2=movie_channel_2,
            configure=configure,
        )
        context.runtime.extraction.roi_statistics = _make_roi_statistics(
            ((14, 14),), frame_height=frame_height, frame_width=frame_width
        )
        context.runtime.extraction.roi_statistics_channel_2 = _make_roi_statistics(
            ((30, 30),), frame_height=frame_height, frame_width=frame_width
        )

        extract_traces(context=context)

        plane_directory = context.runtime.io.output_path
        spikes_channel_2 = _load_result(plane_directory, "spikes_channel_2")
        subtracted_fluorescence_channel_2 = _load_result(plane_directory, "subtracted_fluorescence_channel_2")
        assert np.all(spikes_channel_2 == 0.0)
        assert np.all(subtracted_fluorescence_channel_2 == 0.0)

    def test_functional_channel_2_missing_statistics_raises(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that functional channel 2 extraction raises RuntimeError without channel 2 ROI statistics."""
        frame_height = frame_width = 48
        frame_count = 8
        movie = _constant_movie(
            _CONSTANT_PIXEL_VALUE, frame_count=frame_count, frame_height=frame_height, frame_width=frame_width
        )
        movie_channel_2 = _constant_movie(
            400, frame_count=frame_count, frame_height=frame_height, frame_width=frame_width
        )

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.main.second_channel_functional = True
            configuration.signal_extraction.extract_neuropil = False
            configuration.spike_deconvolution.extract_spikes = False

        context = single_recording_context(
            tmp_path,
            frame_height=frame_height,
            frame_width=frame_width,
            frame_count=frame_count,
            movie=movie,
            movie_channel_2=movie_channel_2,
            configure=configure,
        )
        context.runtime.extraction.roi_statistics = _make_roi_statistics(
            ((14, 14),), frame_height=frame_height, frame_width=frame_width
        )
        context.runtime.extraction.roi_statistics_channel_2 = None

        with pytest.raises(RuntimeError):
            extract_traces(context=context)


class TestExtractTracesChannel2Guards:
    """Tests the defensive channel 2 path guards in the structural and functional helpers."""

    def test_structural_helper_missing_path_raises(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that the structural helper raises RuntimeError when the channel 2 binary path is unset."""
        context = single_recording_context(tmp_path, frame_height=48, frame_width=48, frame_count=8)
        context.runtime.io.registered_binary_path_channel_2 = None
        roi_masks = ((np.array([0], dtype=np.int32), np.array([1.0], dtype=np.float32)),)

        with pytest.raises(RuntimeError):
            _extract_structural_channel_2(context=context, batch_size=5, roi_masks=roi_masks, neuropil_masks=None)

    def test_functional_helper_missing_path_raises(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that the functional helper raises RuntimeError when the channel 2 binary path is unset."""
        context = single_recording_context(tmp_path, frame_height=48, frame_width=48, frame_count=8)
        context.runtime.io.registered_binary_path_channel_2 = None

        with pytest.raises(RuntimeError):
            _extract_functional_channel_2(context=context, batch_size=5)


class TestExtractTracesDispatch:
    """Tests the runtime context dispatch performed by the extract_traces entry point."""

    def test_multi_recording_context_routes_to_multi_handler(self) -> None:
        """Verifies that a multi-recording context dispatches to the multi-recording handler and raises early."""
        context = MultiRecordingRuntimeContext(
            configuration=MultiRecordingConfiguration(),
            runtime=MultiRecordingRuntimeData(),
        )

        with pytest.raises(RuntimeError):
            extract_traces(context=context)
