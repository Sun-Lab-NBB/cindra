"""Contains tests for the pyramid module."""

from __future__ import annotations

import numpy as np

from cindra.registration.pyramid import ScaleSpacePyramid


class TestScaleSpacePyramid:
    """Tests for ScaleSpacePyramid."""

    def test_base_level_created(self) -> None:
        """Verifies that the pyramid has at least one level after initialization."""
        data = np.ones((32, 32), dtype=np.float32)
        pyramid = ScaleSpacePyramid(data=data, min_scale=0.0)
        assert len(pyramid._levels) == 1
        assert len(pyramid._level_scales) == 1

    def test_min_scale_zero_preserves_data(self) -> None:
        """Verifies that min_scale=0 does not smooth or downsample the data."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((32, 32)).astype(np.float32)
        pyramid = ScaleSpacePyramid(data=data, min_scale=0.0)
        np.testing.assert_array_equal(pyramid._levels[0], data)

    def test_get_scale_at_base_level(self) -> None:
        """Verifies that retrieving the base scale returns data without additional smoothing."""
        data = np.ones((32, 32), dtype=np.float32) * 10.0
        pyramid = ScaleSpacePyramid(data=data, min_scale=0.0)
        result = pyramid.get_scale(scale=0.0)
        np.testing.assert_allclose(result, 10.0, atol=1e-5)

    def test_get_scale_adds_levels(self) -> None:
        """Verifies that requesting a higher scale adds new pyramid levels."""
        data = np.ones((64, 64), dtype=np.float32)
        pyramid = ScaleSpacePyramid(data=data, min_scale=0.0)
        initial_count = len(pyramid._levels)
        _ = pyramid.get_scale(scale=4.0)
        assert len(pyramid._levels) > initial_count

    def test_get_scale_returns_finite(self) -> None:
        """Verifies that retrieved scales produce finite output."""
        rng = np.random.default_rng(42)
        data = rng.standard_normal((64, 64)).astype(np.float32)
        pyramid = ScaleSpacePyramid(data=data, min_scale=0.0)
        result = pyramid.get_scale(scale=4.0)
        assert np.isfinite(result).all()

    def test_min_scale_with_downsampling(self) -> None:
        """Verifies that a large min_scale triggers downsampling of the base level."""
        data = np.ones((64, 64), dtype=np.float32)
        pyramid = ScaleSpacePyramid(data=data, min_scale=3.0)
        # With min_scale=3.0, zoom_factor = 1/3 ≈ 0.33 < 0.9, so downsampling applies.
        base_level = pyramid._levels[0]
        assert base_level.shape[0] < 64 or base_level.shape[1] < 64

    def test_small_min_scale_no_downsampling(self) -> None:
        """Verifies that a small min_scale does not downsample the base level."""
        data = np.ones((64, 64), dtype=np.float32)
        pyramid = ScaleSpacePyramid(data=data, min_scale=0.5)
        # With min_scale=0.5, zoom_factor = 1/0.5 = 2.0 > 0.9, so no downsampling.
        base_level = pyramid._levels[0]
        assert base_level.shape == (64, 64)

    def test_uniform_image_stays_uniform(self) -> None:
        """Verifies that a uniform image remains uniform at any scale."""
        data = np.ones((32, 32), dtype=np.float32) * 5.0
        pyramid = ScaleSpacePyramid(data=data, min_scale=0.0)
        result = pyramid.get_scale(scale=3.0)
        np.testing.assert_allclose(result, 5.0, atol=0.1)
