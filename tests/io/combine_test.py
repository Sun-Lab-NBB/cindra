"""Contains tests for the combine module helper functions."""

from __future__ import annotations

import numpy as np

from cindra.io.combine import _compute_plane_offsets
from cindra.dataclasses import (
    RuntimeContext,
    AcquisitionParameters,
    SingleRecordingRuntimeData,
    SingleRecordingConfiguration,
)


class TestComputePlaneOffsets:
    """Tests _compute_plane_offsets."""

    def test_single_plane_returns_zero_offsets(self) -> None:
        """Verifies that a single-plane recording produces zero displacements."""
        contexts = [_make_context(frame_height=64, frame_width=64)]

        y_displacement, x_displacement = _compute_plane_offsets(plane_contexts=contexts)

        assert y_displacement.shape == (1,)
        assert x_displacement.shape == (1,)
        assert y_displacement[0] == 0
        assert x_displacement[0] == 0

    def test_four_planes_grid_layout(self) -> None:
        """Verifies that four equal-size planes produce a 2x2 grid layout."""
        height = 64
        width = 64
        contexts = [_make_context(frame_height=height, frame_width=width) for _ in range(4)]

        y_displacement, x_displacement = _compute_plane_offsets(plane_contexts=contexts)

        assert y_displacement.shape == (4,)
        assert x_displacement.shape == (4,)

        assert y_displacement[0] == 0
        assert x_displacement[0] == 0

        assert np.all(y_displacement >= 0)
        assert np.all(x_displacement >= 0)
        assert np.all(y_displacement % height == 0)
        assert np.all(x_displacement % width == 0)

    def test_two_planes_non_mroi(self) -> None:
        """Verifies that two non-MROI planes are placed in a grid layout."""
        height = 32
        width = 32
        contexts = [_make_context(frame_height=height, frame_width=width) for _ in range(2)]

        y_displacement, x_displacement = _compute_plane_offsets(plane_contexts=contexts)

        assert y_displacement.shape == (2,)
        assert x_displacement.shape == (2,)

        # Two planes in a grid: they should not both be at the origin.
        offsets = np.stack([y_displacement, x_displacement], axis=1)
        unique_positions = np.unique(offsets, axis=0)
        assert len(unique_positions) == 2

    def test_mroi_single_z_plane_uses_offsets_directly(self) -> None:
        """Verifies that MROI contexts with a single z-plane per ROI use MROI offsets as displacements."""
        contexts = [
            _make_context(frame_height=32, frame_width=32, mroi_y_offset=0, mroi_x_offset=0),
            _make_context(frame_height=32, frame_width=32, mroi_y_offset=0, mroi_x_offset=100),
            _make_context(frame_height=32, frame_width=32, mroi_y_offset=0, mroi_x_offset=200),
        ]

        y_displacement, x_displacement = _compute_plane_offsets(plane_contexts=contexts)

        assert y_displacement.shape == (3,)
        assert x_displacement.shape == (3,)
        np.testing.assert_array_equal(x_displacement, [0, 100, 200])
        np.testing.assert_array_equal(y_displacement, [0, 0, 0])

    def test_mroi_multiple_z_planes_applies_two_level_tiling(self) -> None:
        """Verifies that MROI contexts with multiple z-planes per ROI tile across z-planes correctly."""
        # Two ROIs at positions (0, 0) and (0, 50), each with 2 z-planes (4 virtual planes total). The context
        # resolver lays virtual planes out ROI-major, so the two planes of ROI 0 precede the two planes of ROI 1.
        contexts = [
            _make_context(frame_height=32, frame_width=32, mroi_y_offset=0, mroi_x_offset=0),
            _make_context(frame_height=32, frame_width=32, mroi_y_offset=0, mroi_x_offset=0),
            _make_context(frame_height=32, frame_width=32, mroi_y_offset=0, mroi_x_offset=50),
            _make_context(frame_height=32, frame_width=32, mroi_y_offset=0, mroi_x_offset=50),
        ]

        y_displacement, x_displacement = _compute_plane_offsets(plane_contexts=contexts)

        assert y_displacement.shape == (4,)
        assert x_displacement.shape == (4,)

        # Every virtual plane occupies its own rectangle in the combined view, which is the property the two-level
        # tiling exists to guarantee.
        offsets = np.stack([y_displacement, x_displacement], axis=1)
        assert len(np.unique(offsets, axis=0)) == 4

        # Both planes of one ROI keep that ROI's x position, and the second z-plane of each ROI is tiled below the
        # first by one tile height.
        np.testing.assert_array_equal(x_displacement, [0, 0, 50, 50])
        np.testing.assert_array_equal(y_displacement, [0, 32, 0, 32])

    def test_mroi_three_rois_two_z_planes_pins_offsets(self) -> None:
        """Verifies the exact displacements for a six-plane recording holding three ROIs with two z-planes each."""
        # Three ROIs at x positions 0, 50 and 100, each with 2 z-planes, laid out ROI-major.
        contexts = [
            _make_context(frame_height=32, frame_width=32, mroi_y_offset=0, mroi_x_offset=x_offset)
            for x_offset in (0, 0, 50, 50, 100, 100)
        ]

        y_displacement, x_displacement = _compute_plane_offsets(plane_contexts=contexts)

        # The tile bounding box spans 132 pixels in x, so a single tile column fits the six planes into two rows.
        np.testing.assert_array_equal(x_displacement, [0, 0, 50, 50, 100, 100])
        np.testing.assert_array_equal(y_displacement, [0, 32, 0, 32, 0, 32])

        offsets = np.stack([y_displacement, x_displacement], axis=1)
        assert len(np.unique(offsets, axis=0)) == 6

    def test_mroi_two_rois_three_z_planes_wraps_tile_columns(self) -> None:
        """Verifies that z-plane tiles wrap onto a second tile row once the tile column count is exceeded."""
        # Two ROIs at x positions 0 and 50, each with 3 z-planes, laid out ROI-major.
        contexts = [
            _make_context(frame_height=32, frame_width=32, mroi_y_offset=0, mroi_x_offset=x_offset)
            for x_offset in (0, 0, 0, 50, 50, 50)
        ]

        y_displacement, x_displacement = _compute_plane_offsets(plane_contexts=contexts)

        # The tile bounding box spans 82 pixels in x and the grid holds two tile columns, so the third z-plane wraps
        # onto the second tile row while the second z-plane shifts one tile width to the right.
        np.testing.assert_array_equal(x_displacement, [0, 82, 0, 50, 132, 50])
        np.testing.assert_array_equal(y_displacement, [0, 0, 32, 0, 0, 32])

        offsets = np.stack([y_displacement, x_displacement], axis=1)
        assert len(np.unique(offsets, axis=0)) == 6

    def test_single_plane_mroi_uses_offsets_directly(self) -> None:
        """Verifies that a single-plane MROI recording keeps its base MROI offsets."""
        contexts = [_make_context(frame_height=32, frame_width=32, mroi_y_offset=10, mroi_x_offset=20)]

        y_displacement, x_displacement = _compute_plane_offsets(plane_contexts=contexts)

        # A single plane holds as many unique MROI positions as it holds planes, so the two-level tiling is skipped.
        np.testing.assert_array_equal(y_displacement, [10])
        np.testing.assert_array_equal(x_displacement, [20])

    def test_mroi_single_roi_multiple_z_planes_tiles_as_grid(self) -> None:
        """Verifies that a single-ROI recording with multiple z-planes tiles its planes into a square grid."""
        contexts = [_make_context(frame_height=32, frame_width=32, mroi_y_offset=0, mroi_x_offset=0) for _ in range(4)]

        y_displacement, x_displacement = _compute_plane_offsets(plane_contexts=contexts)

        # With one ROI position, every virtual plane index is also its z-plane index, so the four planes fill a 2x2
        # grid of tiles.
        np.testing.assert_array_equal(x_displacement, [0, 32, 0, 32])
        np.testing.assert_array_equal(y_displacement, [0, 0, 32, 32])


def _make_context(
    frame_height: int = 64,
    frame_width: int = 64,
    mroi_y_offset: int | None = None,
    mroi_x_offset: int | None = None,
) -> RuntimeContext:
    """Creates a minimal RuntimeContext with the specified IO dimensions.

    Args:
        frame_height: The frame height in pixels.
        frame_width: The frame width in pixels.
        mroi_y_offset: The optional MROI y-offset.
        mroi_x_offset: The optional MROI x-offset.

    Returns:
        The context populated with the requested frame geometry and MROI offsets.
    """
    runtime = SingleRecordingRuntimeData()
    runtime.io.frame_height = frame_height
    runtime.io.frame_width = frame_width
    runtime.io.mroi_y_offset = mroi_y_offset
    runtime.io.mroi_x_offset = mroi_x_offset
    return RuntimeContext(
        configuration=SingleRecordingConfiguration(),
        acquisition=AcquisitionParameters(frame_rate=30.0),
        runtime=runtime,
    )
