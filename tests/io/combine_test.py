"""Contains tests for the combine module helper functions."""

from __future__ import annotations

import numpy as np

from cindra.io.combine import compute_plane_offsets
from cindra.dataclasses import (
    RuntimeContext,
    AcquisitionParameters,
    SingleRecordingRuntimeData,
    SingleRecordingConfiguration,
)


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
        A RuntimeContext instance with minimal configuration.
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


class TestComputePlaneOffsets:
    """Tests for compute_plane_offsets."""

    def test_single_plane_returns_zero_offsets(self) -> None:
        """Verifies that a single-plane recording produces zero displacements."""
        contexts = [_make_context(frame_height=64, frame_width=64)]

        y_displacement, x_displacement = compute_plane_offsets(plane_contexts=contexts)

        assert y_displacement.shape == (1,)
        assert x_displacement.shape == (1,)
        assert y_displacement[0] == 0
        assert x_displacement[0] == 0

    def test_four_planes_grid_layout(self) -> None:
        """Verifies that four equal-size planes produce a 2x2 grid layout."""
        height = 64
        width = 64
        contexts = [_make_context(frame_height=height, frame_width=width) for _ in range(4)]

        y_displacement, x_displacement = compute_plane_offsets(plane_contexts=contexts)

        assert y_displacement.shape == (4,)
        assert x_displacement.shape == (4,)

        # The first plane should be at (0, 0).
        assert y_displacement[0] == 0
        assert x_displacement[0] == 0

        # All displacements should be non-negative multiples of the plane dimensions.
        assert np.all(y_displacement >= 0)
        assert np.all(x_displacement >= 0)
        assert np.all(y_displacement % height == 0)
        assert np.all(x_displacement % width == 0)

    def test_two_planes_non_mroi(self) -> None:
        """Verifies that two non-MROI planes are placed in a grid layout."""
        height = 32
        width = 32
        contexts = [_make_context(frame_height=height, frame_width=width) for _ in range(2)]

        y_displacement, x_displacement = compute_plane_offsets(plane_contexts=contexts)

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

        y_displacement, x_displacement = compute_plane_offsets(plane_contexts=contexts)

        assert y_displacement.shape == (3,)
        assert x_displacement.shape == (3,)
        np.testing.assert_array_equal(x_displacement, [0, 100, 200])
        np.testing.assert_array_equal(y_displacement, [0, 0, 0])

    def test_mroi_multiple_z_planes_applies_two_level_tiling(self) -> None:
        """Verifies that MROI contexts with multiple z-planes per ROI tile across z-planes correctly."""
        # Two ROIs at positions (0, 0) and (0, 50), each with 2 z-planes (4 virtual planes total).
        contexts = [
            _make_context(frame_height=32, frame_width=32, mroi_y_offset=0, mroi_x_offset=0),
            _make_context(frame_height=32, frame_width=32, mroi_y_offset=0, mroi_x_offset=50),
            _make_context(frame_height=32, frame_width=32, mroi_y_offset=0, mroi_x_offset=0),
            _make_context(frame_height=32, frame_width=32, mroi_y_offset=0, mroi_x_offset=50),
        ]

        y_displacement, x_displacement = compute_plane_offsets(plane_contexts=contexts)

        assert y_displacement.shape == (4,)
        assert x_displacement.shape == (4,)

        # The first z-plane pair (indices 0, 1) should have the base MROI offsets.
        # The second z-plane pair (indices 2, 3) should have the base MROI offsets plus tile offsets.
        # Within each pair, relative positions should be preserved.
        assert x_displacement[0] < x_displacement[1]
        assert x_displacement[2] < x_displacement[3]

        # The tile offset causes the second pair to be displaced from the first pair.
        assert not (y_displacement[0] == y_displacement[2] and x_displacement[0] == x_displacement[2])
