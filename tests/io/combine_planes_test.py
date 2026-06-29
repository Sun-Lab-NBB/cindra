"""Contains integration tests for the combine_planes stage entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from cindra.io.combine import combine_planes
from cindra.dataclasses import ROIMask, ROIStatistics

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from numpy.typing import NDArray

    from cindra.dataclasses import RuntimeContext, SingleRecordingConfiguration

type _RoiSpecs = tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]

_FRAME_HEIGHT: int = 16
"""The synthetic plane height in pixels used by the combine integration tests."""

_FRAME_WIDTH: int = 16
"""The synthetic plane width in pixels used by the combine integration tests."""


def _make_roi_statistics(*, frame_width: int, y_pixels: tuple[int, ...], x_pixels: tuple[int, ...]) -> ROIStatistics:
    """Builds a deterministic ROIStatistics instance from explicit pixel coordinates."""
    y_array = np.array(y_pixels, dtype=np.int32)
    x_array = np.array(x_pixels, dtype=np.int32)
    mask = ROIMask(
        y_pixels=y_array,
        x_pixels=x_array,
        pixel_weights=np.ones(shape=len(y_pixels), dtype=np.float32),
        centroid=(int(np.median(y_array)), int(np.median(x_array))),
        frame_width=frame_width,
        radius=2.0,
    )
    return ROIStatistics(mask=mask, footprint=2, pixel_count=len(y_pixels))


def _make_channel_2_movie(*, frame_count: int, seed: int) -> NDArray[np.int16]:
    """Builds a synthetic int16 channel 2 movie for two-channel contexts."""
    generator = np.random.default_rng(seed=seed)
    return generator.integers(low=100, high=1000, size=(frame_count, _FRAME_HEIGHT, _FRAME_WIDTH)).astype(np.int16)


def _configure_two_channels(*, second_functional: bool) -> Callable[[SingleRecordingConfiguration], None]:
    """Returns a configuration callback that enables two channels with the given second-channel mode."""

    def _configure(configuration: SingleRecordingConfiguration) -> None:
        configuration.main.two_channels = True
        configuration.main.first_channel_functional = True
        configuration.main.second_channel_functional = second_functional

    return _configure


def _populate_channel_1(
    context: RuntimeContext,
    *,
    roi_specs: _RoiSpecs,
    frame_count: int,
    fill: float,
    seed: int,
    with_images: bool = True,
    with_max_projection: bool = False,
    with_corrected_structural: bool = False,
    with_colocalization: bool = False,
) -> None:
    """Populates channel 1 detection images and extraction traces on the given context."""
    generator = np.random.default_rng(seed=seed)
    detection = context.runtime.detection
    if with_images:
        detection.mean_image = generator.random(size=(_FRAME_HEIGHT, _FRAME_WIDTH)).astype(np.float32)
        detection.enhanced_mean_image = generator.random(size=(_FRAME_HEIGHT, _FRAME_WIDTH)).astype(np.float32)
        detection.correlation_map = generator.random(size=(_FRAME_HEIGHT, _FRAME_WIDTH)).astype(np.float32)
    if with_max_projection:
        detection.maximum_projection = generator.random(size=(_FRAME_HEIGHT, _FRAME_WIDTH)).astype(np.float32)

    width = context.runtime.io.frame_width
    rois = [_make_roi_statistics(frame_width=width, y_pixels=ys, x_pixels=xs) for ys, xs in roi_specs]
    roi_count = len(rois)
    extraction = context.runtime.extraction
    extraction.roi_statistics = rois
    extraction.cell_fluorescence = np.full(shape=(roi_count, frame_count), fill_value=fill, dtype=np.float32)
    extraction.neuropil_fluorescence = np.full(shape=(roi_count, frame_count), fill_value=fill, dtype=np.float32)
    extraction.subtracted_fluorescence = np.full(shape=(roi_count, frame_count), fill_value=fill, dtype=np.float32)
    extraction.spikes = np.full(shape=(roi_count, frame_count), fill_value=fill, dtype=np.float32)
    extraction.cell_classification = generator.random(size=(roi_count, 2)).astype(np.float32)
    if with_colocalization:
        extraction.cell_colocalization = generator.random(size=(roi_count, 2)).astype(np.float32)
    if with_corrected_structural:
        extraction.corrected_structural_mean_image = generator.random(size=(_FRAME_HEIGHT, _FRAME_WIDTH)).astype(
            np.float32
        )


def _populate_channel_2(
    context: RuntimeContext,
    *,
    roi_specs: _RoiSpecs,
    frame_count: int,
    seed: int,
    with_traces: bool = True,
    with_max_projection: bool = False,
) -> None:
    """Populates channel 2 detection images and extraction data on the given context."""
    generator = np.random.default_rng(seed=seed)
    detection = context.runtime.detection
    detection.mean_image_channel_2 = generator.random(size=(_FRAME_HEIGHT, _FRAME_WIDTH)).astype(np.float32)
    detection.enhanced_mean_image_channel_2 = generator.random(size=(_FRAME_HEIGHT, _FRAME_WIDTH)).astype(np.float32)
    detection.correlation_map_channel_2 = generator.random(size=(_FRAME_HEIGHT, _FRAME_WIDTH)).astype(np.float32)
    if with_max_projection:
        detection.maximum_projection_channel_2 = generator.random(size=(_FRAME_HEIGHT, _FRAME_WIDTH)).astype(np.float32)

    width = context.runtime.io.frame_width
    rois = [_make_roi_statistics(frame_width=width, y_pixels=ys, x_pixels=xs) for ys, xs in roi_specs]
    roi_count = len(rois)
    extraction = context.runtime.extraction
    extraction.roi_statistics_channel_2 = rois
    extraction.cell_classification_channel_2 = generator.random(size=(roi_count, 2)).astype(np.float32)
    if with_traces:
        extraction.cell_fluorescence_channel_2 = np.full(
            shape=(roi_count, frame_count), fill_value=5.0, dtype=np.float32
        )
        extraction.neuropil_fluorescence_channel_2 = np.full(
            shape=(roi_count, frame_count), fill_value=5.0, dtype=np.float32
        )
        extraction.subtracted_fluorescence_channel_2 = np.full(
            shape=(roi_count, frame_count), fill_value=5.0, dtype=np.float32
        )
        extraction.spikes_channel_2 = np.full(shape=(roi_count, frame_count), fill_value=5.0, dtype=np.float32)


class TestCombinePlanes:
    """Tests the combine_planes multi-plane combination entry point."""

    def test_single_plane_single_channel(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that a single-plane single-channel recording produces a combined dataset matching the plane."""
        context = single_recording_context(
            tmp_path / "plane_0", frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH, frame_count=8
        )
        _populate_channel_1(
            context,
            roi_specs=(((1, 2), (1, 2)), ((4, 5), (6, 7))),
            frame_count=8,
            fill=1.0,
            seed=1,
            with_max_projection=True,
            with_corrected_structural=True,
            with_colocalization=True,
        )

        combined = combine_planes(plane_contexts=[context])

        assert combined.plane_count == 1
        assert combined.combined_height == _FRAME_HEIGHT
        assert combined.combined_width == _FRAME_WIDTH
        assert combined.extraction.roi_statistics is not None
        assert len(combined.extraction.roi_statistics) == 2
        assert all(roi.plane_index == 0 for roi in combined.extraction.roi_statistics)
        assert combined.extraction.cell_fluorescence is not None
        assert combined.extraction.cell_fluorescence.shape == (2, 8)
        assert combined.detection.mean_image is not None
        assert combined.detection.mean_image.shape == (_FRAME_HEIGHT, _FRAME_WIDTH)
        assert combined.detection.maximum_projection is not None
        assert combined.extraction.corrected_structural_mean_image is not None
        assert combined.extraction.cell_colocalization is not None
        assert combined.extraction.cell_fluorescence_channel_2 is None
        assert combined.registered_binary_paths_channel_2 is None

    def test_two_planes_grid_layout_with_padding(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies grid placement, ROI offsetting, and zero-padding when planes have different frame counts."""
        plane_0 = single_recording_context(
            tmp_path / "plane_0", frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH, frame_count=8
        )
        plane_1 = single_recording_context(
            tmp_path / "plane_1", frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH, frame_count=4
        )
        _populate_channel_1(plane_0, roi_specs=(((1, 2), (1, 2)),), frame_count=8, fill=1.0, seed=2)
        _populate_channel_1(plane_1, roi_specs=(((3, 4), (5, 6)),), frame_count=4, fill=2.0, seed=3)

        combined = combine_planes(plane_contexts=[plane_0, plane_1])

        assert combined.plane_count == 2
        assert combined.combined_height == _FRAME_HEIGHT
        assert combined.combined_width == _FRAME_WIDTH * 2
        np.testing.assert_array_equal(combined.plane_x_offsets, [0, _FRAME_WIDTH])
        np.testing.assert_array_equal(combined.plane_y_offsets, [0, 0])

        rois = combined.extraction.roi_statistics
        assert rois is not None
        assert len(rois) == 2
        assert rois[0].plane_index == 0
        assert rois[1].plane_index == 1
        np.testing.assert_array_equal(rois[1].mask.x_pixels, np.array([5, 6]) + _FRAME_WIDTH)
        np.testing.assert_array_equal(rois[1].mask.y_pixels, np.array([3, 4]))

        fluorescence = combined.extraction.cell_fluorescence
        assert fluorescence is not None
        assert fluorescence.shape == (2, 8)
        np.testing.assert_allclose(fluorescence[0], np.full(shape=8, fill_value=1.0, dtype=np.float32))
        np.testing.assert_allclose(fluorescence[1, :4], np.full(shape=4, fill_value=2.0, dtype=np.float32))
        np.testing.assert_allclose(fluorescence[1, 4:], np.zeros(shape=4, dtype=np.float32))

        assert combined.extraction.cell_colocalization is None

    def test_two_functional_channels(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that two fully functional channels populate both channel datasets in the combined output."""
        configure = _configure_two_channels(second_functional=True)
        plane_0 = single_recording_context(
            tmp_path / "plane_0",
            frame_height=_FRAME_HEIGHT,
            frame_width=_FRAME_WIDTH,
            frame_count=8,
            movie_channel_2=_make_channel_2_movie(frame_count=8, seed=10),
            configure=configure,
        )
        plane_1 = single_recording_context(
            tmp_path / "plane_1",
            frame_height=_FRAME_HEIGHT,
            frame_width=_FRAME_WIDTH,
            frame_count=4,
            movie_channel_2=_make_channel_2_movie(frame_count=4, seed=11),
            configure=configure,
        )
        _populate_channel_1(
            plane_0, roi_specs=(((1, 2), (1, 2)),), frame_count=8, fill=1.0, seed=4, with_max_projection=True
        )
        _populate_channel_1(
            plane_1, roi_specs=(((3, 4), (5, 6)),), frame_count=4, fill=2.0, seed=5, with_max_projection=True
        )
        _populate_channel_2(plane_0, roi_specs=(((1, 2), (1, 2)),), frame_count=8, seed=6, with_max_projection=True)
        _populate_channel_2(plane_1, roi_specs=(((3, 4), (5, 6)),), frame_count=4, seed=7, with_max_projection=True)

        combined = combine_planes(plane_contexts=[plane_0, plane_1])

        assert combined.detection.mean_image_channel_2 is not None
        assert combined.detection.enhanced_mean_image_channel_2 is not None
        assert combined.detection.correlation_map_channel_2 is not None
        assert combined.detection.maximum_projection_channel_2 is not None
        assert combined.extraction.roi_statistics_channel_2 is not None
        assert len(combined.extraction.roi_statistics_channel_2) == 2
        channel_2_fluorescence = combined.extraction.cell_fluorescence_channel_2
        assert channel_2_fluorescence is not None
        assert channel_2_fluorescence.shape == (2, 8)
        np.testing.assert_allclose(channel_2_fluorescence[1, 4:], np.zeros(shape=4, dtype=np.float32))
        assert combined.registered_binary_paths_channel_2 is not None
        assert len(combined.registered_binary_paths_channel_2) == 2

    def test_two_channels_structural_second(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that a structural second channel copies its mean image but skips functional channel 2 outputs."""
        configure = _configure_two_channels(second_functional=False)
        context = single_recording_context(
            tmp_path / "plane_0",
            frame_height=_FRAME_HEIGHT,
            frame_width=_FRAME_WIDTH,
            frame_count=8,
            configure=configure,
        )
        _populate_channel_1(
            context, roi_specs=(((1, 2), (1, 2)),), frame_count=8, fill=1.0, seed=8, with_corrected_structural=True
        )
        context.runtime.detection.mean_image_channel_2 = np.zeros(shape=(_FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.float32)

        combined = combine_planes(plane_contexts=[context])

        assert combined.detection.mean_image_channel_2 is not None
        assert combined.detection.enhanced_mean_image_channel_2 is None
        assert combined.extraction.roi_statistics_channel_2 is None
        assert combined.extraction.cell_fluorescence_channel_2 is None
        assert combined.registered_binary_paths_channel_2 is None
        assert combined.extraction.corrected_structural_mean_image is not None

    def test_plane_without_detection_images(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that a plane lacking detection images still contributes ROIs with zeroed combined images."""
        context = single_recording_context(
            tmp_path / "plane_0", frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH, frame_count=8
        )
        _populate_channel_1(context, roi_specs=(((1, 2), (1, 2)),), frame_count=8, fill=1.0, seed=9, with_images=False)

        combined = combine_planes(plane_contexts=[context])

        assert combined.extraction.roi_statistics is not None
        assert len(combined.extraction.roi_statistics) == 1
        assert combined.detection.mean_image is not None
        np.testing.assert_array_equal(
            combined.detection.mean_image, np.zeros(shape=(_FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.float32)
        )
        assert combined.detection.correlation_map is not None
        np.testing.assert_array_equal(
            combined.detection.correlation_map, np.zeros(shape=(_FRAME_HEIGHT, _FRAME_WIDTH), dtype=np.float32)
        )

    def test_plane_missing_traces_is_skipped(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that a plane with ROI statistics but no fluorescence traces is excluded from the output."""
        plane_0 = single_recording_context(
            tmp_path / "plane_0", frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH, frame_count=8
        )
        plane_1 = single_recording_context(
            tmp_path / "plane_1", frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH, frame_count=8
        )
        _populate_channel_1(plane_0, roi_specs=(((1, 2), (1, 2)),), frame_count=8, fill=1.0, seed=12)
        # Plane 1 carries ROI statistics but no fluorescence traces, so combination skips it.
        width = plane_1.runtime.io.frame_width
        plane_1.runtime.extraction.roi_statistics = [
            _make_roi_statistics(frame_width=width, y_pixels=(3, 4), x_pixels=(5, 6))
        ]

        combined = combine_planes(plane_contexts=[plane_0, plane_1])

        assert combined.extraction.roi_statistics is not None
        assert len(combined.extraction.roi_statistics) == 1
        assert combined.extraction.roi_statistics[0].plane_index == 0

    def test_no_roi_statistics_raises_value_error(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that combination fails when no plane provides ROI statistics."""
        plane_0 = single_recording_context(
            tmp_path / "plane_0", frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH, frame_count=8
        )
        plane_1 = single_recording_context(
            tmp_path / "plane_1", frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH, frame_count=8
        )
        # A None output_path on one plane exercises the directory-name filter during logging.
        plane_1.runtime.io.output_path = None

        with pytest.raises(ValueError, match="Unable to combine plane data"):
            combine_planes(plane_contexts=[plane_0, plane_1])

    def test_missing_registered_binary_path_raises_runtime_error(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that a missing channel 1 registered binary path aborts combination."""
        context = single_recording_context(
            tmp_path / "plane_0", frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH, frame_count=8
        )
        _populate_channel_1(context, roi_specs=(((1, 2), (1, 2)),), frame_count=8, fill=1.0, seed=13)
        context.runtime.io.registered_binary_path = None

        with pytest.raises(RuntimeError, match="registered binary path is not set"):
            combine_planes(plane_contexts=[context])

    def test_missing_channel_2_binary_path_raises_runtime_error(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that a missing channel 2 registered binary path aborts combination when channel 2 is functional."""
        configure = _configure_two_channels(second_functional=True)
        context = single_recording_context(
            tmp_path / "plane_0",
            frame_height=_FRAME_HEIGHT,
            frame_width=_FRAME_WIDTH,
            frame_count=8,
            movie_channel_2=_make_channel_2_movie(frame_count=8, seed=14),
            configure=configure,
        )
        _populate_channel_1(context, roi_specs=(((1, 2), (1, 2)),), frame_count=8, fill=1.0, seed=15)
        _populate_channel_2(context, roi_specs=(((1, 2), (1, 2)),), frame_count=8, seed=16)
        context.runtime.io.registered_binary_path_channel_2 = None

        with pytest.raises(RuntimeError, match="registered binary path for channel 2 is not set"):
            combine_planes(plane_contexts=[context])

    def test_second_channel_functional_without_traces(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that channel 2 ROI statistics without traces yield no combined channel 2 fluorescence."""
        configure = _configure_two_channels(second_functional=True)
        context = single_recording_context(
            tmp_path / "plane_0",
            frame_height=_FRAME_HEIGHT,
            frame_width=_FRAME_WIDTH,
            frame_count=8,
            movie_channel_2=_make_channel_2_movie(frame_count=8, seed=17),
            configure=configure,
        )
        _populate_channel_1(context, roi_specs=(((1, 2), (1, 2)),), frame_count=8, fill=1.0, seed=18)
        _populate_channel_2(context, roi_specs=(((1, 2), (1, 2)),), frame_count=8, seed=19, with_traces=False)

        combined = combine_planes(plane_contexts=[context])

        assert combined.extraction.roi_statistics_channel_2 is not None
        assert combined.extraction.cell_fluorescence_channel_2 is None

    def test_mroi_single_z_plane(self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path) -> None:
        """Verifies that MROI offsets directly position planes and shift ROI coordinates accordingly."""
        plane_0 = single_recording_context(
            tmp_path / "plane_0", frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH, frame_count=8
        )
        plane_1 = single_recording_context(
            tmp_path / "plane_1", frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH, frame_count=8
        )
        plane_0.runtime.io.mroi_y_offset = 0
        plane_0.runtime.io.mroi_x_offset = 0
        plane_1.runtime.io.mroi_y_offset = 0
        plane_1.runtime.io.mroi_x_offset = _FRAME_WIDTH
        _populate_channel_1(plane_0, roi_specs=(((1, 2), (1, 2)),), frame_count=8, fill=1.0, seed=20)
        _populate_channel_1(plane_1, roi_specs=(((3, 4), (5, 6)),), frame_count=8, fill=2.0, seed=21)

        combined = combine_planes(plane_contexts=[plane_0, plane_1])

        np.testing.assert_array_equal(combined.plane_x_offsets, [0, _FRAME_WIDTH])
        assert combined.combined_width == _FRAME_WIDTH * 2
        rois = combined.extraction.roi_statistics
        assert rois is not None
        assert len(rois) == 2
        np.testing.assert_array_equal(rois[1].mask.x_pixels, np.array([5, 6]) + _FRAME_WIDTH)

    def test_mroi_multiple_z_planes(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that MROI recordings with repeated positions tile z-planes into a combined grid."""
        positions = ((0, 0), (0, _FRAME_WIDTH), (0, 0), (0, _FRAME_WIDTH))
        contexts: list[RuntimeContext] = []
        for index, (y_offset, x_offset) in enumerate(positions):
            context = single_recording_context(
                tmp_path / f"plane_{index}", frame_height=_FRAME_HEIGHT, frame_width=_FRAME_WIDTH, frame_count=8
            )
            context.runtime.io.mroi_y_offset = y_offset
            context.runtime.io.mroi_x_offset = x_offset
            contexts.append(context)
        # Only the first plane carries ROIs; the remaining planes are skipped during combination.
        _populate_channel_1(contexts[0], roi_specs=(((1, 2), (1, 2)),), frame_count=8, fill=1.0, seed=22)

        combined = combine_planes(plane_contexts=contexts)

        assert combined.plane_count == 4
        assert combined.combined_height == _FRAME_HEIGHT * 2
        assert combined.combined_width == _FRAME_WIDTH * 2
        rois = combined.extraction.roi_statistics
        assert rois is not None
        assert len(rois) == 1
        assert rois[0].plane_index == 0
