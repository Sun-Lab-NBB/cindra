"""Contains tests for the _compute_pc_extremes function provided by the metrics module."""

from __future__ import annotations

import numpy as np

from cindra.registration.metrics import _compute_pc_extremes


class TestComputePcExtremes:
    """Tests _compute_pc_extremes."""

    def test_output_shapes_two_components(self) -> None:
        """Verifies output array shapes when using two principal components."""
        frame_count = 50
        height = 16
        width = 16
        component_count = 2
        extreme_frame_count = 10

        rng = np.random.default_rng(seed=42)
        frames = rng.standard_normal(size=(frame_count, height, width)).astype(np.float32)

        pc_low, pc_high, projections = _compute_pc_extremes(
            frames=frames,
            num_extreme_frames=extreme_frame_count,
            num_components=component_count,
        )

        assert pc_low.shape == (component_count, height, width)
        assert pc_high.shape == (component_count, height, width)
        assert projections.shape == (frame_count, component_count)

    def test_output_shapes_one_component(self) -> None:
        """Verifies output array shapes when using a single principal component."""
        frame_count = 50
        height = 16
        width = 16
        component_count = 1
        extreme_frame_count = 10

        rng = np.random.default_rng(seed=42)
        frames = rng.standard_normal(size=(frame_count, height, width)).astype(np.float32)

        pc_low, pc_high, projections = _compute_pc_extremes(
            frames=frames,
            num_extreme_frames=extreme_frame_count,
            num_components=component_count,
        )

        assert pc_low.shape == (component_count, height, width)
        assert pc_high.shape == (component_count, height, width)
        assert projections.shape == (frame_count, component_count)

    def test_output_dtypes(self) -> None:
        """Verifies that pc_low and pc_high arrays have float32 dtype."""
        frame_count = 50
        height = 16
        width = 16
        component_count = 2
        extreme_frame_count = 10

        rng = np.random.default_rng(seed=42)
        frames = rng.standard_normal(size=(frame_count, height, width)).astype(np.float32)

        pc_low, pc_high, projections = _compute_pc_extremes(
            frames=frames,
            num_extreme_frames=extreme_frame_count,
            num_components=component_count,
        )

        assert pc_low.dtype == np.float32
        assert pc_high.dtype == np.float32
        assert projections.dtype == np.float32

    def test_projections_are_finite(self) -> None:
        """Verifies that all projection values are finite (no NaN or Inf)."""
        frame_count = 50
        height = 16
        width = 16
        component_count = 2
        extreme_frame_count = 10

        rng = np.random.default_rng(seed=42)
        frames = rng.standard_normal(size=(frame_count, height, width)).astype(np.float32)

        _, _, projections = _compute_pc_extremes(
            frames=frames,
            num_extreme_frames=extreme_frame_count,
            num_components=component_count,
        )

        assert np.all(np.isfinite(projections))

    def test_extreme_means_differ(self) -> None:
        """Verifies that pc_low and pc_high produce different mean images when the input contains structured signal."""
        frame_count = 50
        height = 16
        width = 16
        component_count = 1
        extreme_frame_count = 10

        # Creates frames with a temporal gradient so PCA captures a clear signal direction. Early frames are dark,
        # late frames are bright, ensuring the first PC separates low from high projections.
        rng = np.random.default_rng(seed=42)
        temporal_gradient = np.linspace(start=0.0, stop=1.0, num=frame_count, dtype=np.float32)
        frames = rng.standard_normal(size=(frame_count, height, width)).astype(np.float32)
        frames += temporal_gradient[:, np.newaxis, np.newaxis] * 10.0

        pc_low, pc_high, _ = _compute_pc_extremes(
            frames=frames,
            num_extreme_frames=extreme_frame_count,
            num_components=component_count,
        )

        # Expects the pc_low and pc_high means to differ because extreme frames come from opposite ends of the gradient.
        assert not np.allclose(pc_low[0], pc_high[0])

    def test_extreme_means_reflect_gradient_direction(self) -> None:
        """Verifies that one extreme has a higher overall intensity than the other when frames have a temporal ramp."""
        frame_count = 50
        height = 16
        width = 16
        component_count = 1
        extreme_frame_count = 10

        # Creates frames with spatial noise plus a temporal ramp. Spatial variance is required for PCA over pixels to
        # find a non-degenerate first component; without it, centering collapses the input to numerical noise and the
        # returned PC direction is arbitrary.
        rng = np.random.default_rng(seed=42)
        temporal_gradient = np.linspace(start=0.0, stop=10.0, num=frame_count, dtype=np.float32)
        frames = rng.standard_normal(size=(frame_count, height, width)).astype(np.float32)
        frames += temporal_gradient[:, np.newaxis, np.newaxis]

        pc_low, pc_high, _ = _compute_pc_extremes(
            frames=frames,
            num_extreme_frames=extreme_frame_count,
            num_components=component_count,
        )

        # Checks that the two extremes have different mean intensities; the PC sign is arbitrary, so avoid assuming an
        # ordering.
        low_mean = float(pc_low[0].mean())
        high_mean = float(pc_high[0].mean())
        assert abs(high_mean - low_mean) > 1.0
