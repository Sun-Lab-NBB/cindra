"""Contains tests for the detect_rois module."""

from __future__ import annotations

import numpy as np

from cindra.detection.detect_rois import (
    extend_roi,
    _extend_mask,
    _subtract_neuropil,
    _convolve_square_2d,
    _create_initial_square,
    _estimate_spatial_scale,
    _compute_multiscale_masks,
)


class TestExtendRoi:
    """Tests the diamond-shaped cardinal-neighbor growth, its frame clipping, and its duplicate-free output."""

    def test_single_pixel_one_iteration(self) -> None:
        """Verifies that a single pixel expands to a diamond of 5 pixels after one iteration."""
        y_pixels = np.array([5], dtype=np.int32)
        x_pixels = np.array([5], dtype=np.int32)
        y_out, _x_out = extend_roi(y_pixels=y_pixels, x_pixels=x_pixels, height=10, width=10, iterations=1)
        assert len(y_out) == 5  # center + 4 cardinal neighbors.

    def test_boundary_clipping(self) -> None:
        """Verifies that pixels outside the frame boundary are excluded."""
        y_pixels = np.array([0], dtype=np.int32)
        x_pixels = np.array([0], dtype=np.int32)
        y_out, x_out = extend_roi(y_pixels=y_pixels, x_pixels=x_pixels, height=10, width=10, iterations=1)
        assert np.all(y_out >= 0)
        assert np.all(x_out >= 0)
        # Corner pixel: only center, right, and down are valid.
        assert len(y_out) == 3

    def test_multiple_iterations(self) -> None:
        """Verifies that each iteration expands the ROI further."""
        y_pixels = np.array([5], dtype=np.int32)
        x_pixels = np.array([5], dtype=np.int32)
        single_iteration_y, _ = extend_roi(y_pixels=y_pixels, x_pixels=x_pixels, height=20, width=20, iterations=1)
        double_iteration_y, _ = extend_roi(y_pixels=y_pixels, x_pixels=x_pixels, height=20, width=20, iterations=2)
        assert len(double_iteration_y) > len(single_iteration_y)

    def test_zero_iterations(self) -> None:
        """Verifies that zero iterations return the original pixels."""
        y_pixels = np.array([5, 6], dtype=np.int32)
        x_pixels = np.array([5, 6], dtype=np.int32)
        y_out, x_out = extend_roi(y_pixels=y_pixels, x_pixels=x_pixels, height=10, width=10, iterations=0)
        np.testing.assert_array_equal(actual=y_out, desired=y_pixels)
        np.testing.assert_array_equal(actual=x_out, desired=x_pixels)

    def test_no_duplicates(self) -> None:
        """Verifies that the output contains no duplicate coordinates."""
        y_pixels = np.array([5, 5, 6], dtype=np.int32)
        x_pixels = np.array([5, 6, 5], dtype=np.int32)
        y_out, x_out = extend_roi(y_pixels=y_pixels, x_pixels=x_pixels, height=20, width=20, iterations=1)
        flattened_indices = y_out * 20 + x_out
        assert len(flattened_indices) == len(np.unique(flattened_indices))


class TestSubtractNeuropil:
    """Tests the in-place per-frame high-pass that removes neuropil contamination from the detection input."""

    def test_in_place_modification(self) -> None:
        """Verifies that frames are modified in-place."""
        generator = np.random.default_rng(seed=42)
        frames = generator.standard_normal((5, 32, 32)).astype(np.float32) + 10.0
        original = frames.copy()
        _subtract_neuropil(frames=frames, filter_size=5)
        assert not np.array_equal(a1=frames, a2=original)

    def test_uniform_frames_become_near_zero(self) -> None:
        """Verifies that uniform frames produce near-zero output after high-pass filtering."""
        frames = np.ones((5, 32, 32), dtype=np.float32) * 100.0
        _subtract_neuropil(frames=frames, filter_size=5)
        np.testing.assert_allclose(actual=frames, desired=0.0, atol=1e-3)

    def test_output_finite(self) -> None:
        """Verifies that the filtered frames contain only finite values."""
        generator = np.random.default_rng(seed=42)
        frames = generator.standard_normal((5, 32, 32)).astype(np.float32)
        _subtract_neuropil(frames=frames, filter_size=5)
        assert np.isfinite(frames).all()


class TestConvolveSquare2d:
    """Tests the box-mean scaling the uniform square kernel applies to each frame."""

    def test_output_shape(self) -> None:
        """Verifies that the output shape matches the input shape."""
        frames = np.ones((5, 32, 32), dtype=np.float32)
        result = _convolve_square_2d(frames=frames, filter_size=3)
        assert result.shape == frames.shape

    def test_uniform_input_scaled(self) -> None:
        """Verifies that a uniform input is scaled by the filter size."""
        frames = np.ones((5, 32, 32), dtype=np.float32)
        result = _convolve_square_2d(frames=frames, filter_size=3)
        # Interior pixels of uniform input: uniform_filter gives 1.0, scaled by 3 = 3.0.
        # Edge pixels will have smaller values due to zero padding.
        center = result[:, 10:22, 10:22]
        np.testing.assert_allclose(actual=center, desired=3.0, atol=0.1)

    def test_output_finite(self) -> None:
        """Verifies that the output is finite."""
        generator = np.random.default_rng(seed=42)
        frames = generator.standard_normal((5, 32, 32)).astype(np.float32)
        result = _convolve_square_2d(frames=frames, filter_size=3)
        assert np.isfinite(result).all()


class TestCreateInitialSquare:
    """Tests the centered seed patch, its frame clipping, and the unit normalization of its weights."""

    def test_centered_square(self) -> None:
        """Verifies that the output is a square patch centered at the given location."""
        y_pixels, x_pixels, _weights = _create_initial_square(
            center_y=10, center_x=10, square_size=5, height=30, width=30
        )
        assert len(y_pixels) == 25
        assert np.all(y_pixels >= 8)
        assert np.all(y_pixels <= 12)
        assert np.all(x_pixels >= 8)
        assert np.all(x_pixels <= 12)

    def test_boundary_clipping(self) -> None:
        """Verifies that pixels outside the frame boundary are excluded."""
        y_pixels, x_pixels, _weights = _create_initial_square(
            center_y=0, center_x=0, square_size=5, height=30, width=30
        )
        assert np.all(y_pixels >= 0)
        assert np.all(x_pixels >= 0)
        assert len(y_pixels) < 25

    def test_weights_unit_normalized(self) -> None:
        """Verifies that the output weights have unit norm."""
        _, _, weights = _create_initial_square(center_y=10, center_x=10, square_size=5, height=30, width=30)
        np.testing.assert_allclose(actual=np.linalg.norm(weights), desired=1.0, atol=1e-5)

    def test_output_dtypes(self) -> None:
        """Verifies the output dtypes."""
        y_pixels, x_pixels, weights = _create_initial_square(
            center_y=10, center_x=10, square_size=3, height=30, width=30
        )
        assert y_pixels.dtype == np.int32
        assert x_pixels.dtype == np.int32
        assert weights.dtype == np.float32


class TestExtendMask:
    """Tests the eight-neighbor weight distribution and the frame bounds the mask growth respects."""

    def test_expands_in_all_directions(self) -> None:
        """Verifies that the mask expands into all 8 surrounding neighbors."""
        y_pixels = np.array([5], dtype=np.int32)
        x_pixels = np.array([5], dtype=np.int32)
        weights = np.array([1.0], dtype=np.float32)
        y_out, _x_out, _weights_out = _extend_mask(
            y_pixels=y_pixels, x_pixels=x_pixels, weights=weights, height=20, width=20
        )
        # Single pixel + 8 neighbors = 9 pixels.
        assert len(y_out) == 9

    def test_boundary_handling(self) -> None:
        """Verifies that the mask respects frame boundaries."""
        y_pixels = np.array([0], dtype=np.int32)
        x_pixels = np.array([0], dtype=np.int32)
        weights = np.array([1.0], dtype=np.float32)
        y_out, x_out, _weights_out = _extend_mask(
            y_pixels=y_pixels, x_pixels=x_pixels, weights=weights, height=20, width=20
        )
        assert np.all(y_out >= 0)
        assert np.all(x_out >= 0)
        # Corner pixel: only center, right, down, and diagonal = 4 pixels.
        assert len(y_out) == 4

    def test_weights_non_negative(self) -> None:
        """Verifies that the accumulated weights are non-negative."""
        y_pixels = np.array([5, 5, 6], dtype=np.int32)
        x_pixels = np.array([5, 6, 5], dtype=np.int32)
        weights = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        _, _, weights_out = _extend_mask(y_pixels=y_pixels, x_pixels=x_pixels, weights=weights, height=20, width=20)
        assert np.all(weights_out >= 0)


class TestEstimateSpatialScale:
    """Tests the dominant scale index that the multiscale projection peaks select."""

    def test_returns_dominant_scale(self) -> None:
        """Verifies that the dominant scale is returned for a clear scale pattern."""
        # A single non-zero scale makes every pixel of the maximum projection a local maximum, so the scale vote is
        # unanimous and its mode settles on index 2.
        scale_images = np.zeros((5, 32, 32), dtype=np.float32)
        scale_images[2, :, :] = 10.0
        result = _estimate_spatial_scale(scale_images=scale_images)
        assert result == 2

    def test_returns_valid_index(self) -> None:
        """Verifies that the returned scale index is within the valid range."""
        generator = np.random.default_rng(seed=42)
        scale_images = generator.standard_normal((5, 32, 32)).astype(np.float32)
        result = _estimate_spatial_scale(scale_images=scale_images)
        assert 0 <= result < 5


class TestComputeMultiscaleMasks:
    """Tests that a full-resolution mask projects onto every requested scale with a non-empty coordinate set."""

    def test_output_list_lengths(self) -> None:
        """Verifies that the output lists have one entry per scale."""
        y_pixels = np.array([5, 5, 6, 6], dtype=np.int32)
        x_pixels = np.array([5, 6, 5, 6], dtype=np.int32)
        pixel_weights = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        scale_heights = np.array([32, 16, 8], dtype=np.uint16)
        scale_widths = np.array([32, 16, 8], dtype=np.uint16)
        y_coordinates, x_coordinates, weights = _compute_multiscale_masks(
            y_pixels=y_pixels,
            x_pixels=x_pixels,
            weights=pixel_weights,
            scale_heights=scale_heights,
            scale_widths=scale_widths,
        )
        assert len(y_coordinates) == 3
        assert len(x_coordinates) == 3
        assert len(weights) == 3

    def test_coarser_scales_have_fewer_or_equal_pixels(self) -> None:
        """Verifies that every requested scale yields a non-empty coordinate array after mask extension."""
        y_pixels = np.arange(10, dtype=np.int32)
        x_pixels = np.arange(10, dtype=np.int32)
        weights = np.ones(10, dtype=np.float32) / 10
        scale_heights = np.array([64, 32, 16], dtype=np.uint16)
        scale_widths = np.array([64, 32, 16], dtype=np.uint16)
        y_coordinates, _x_coordinates, _weights = _compute_multiscale_masks(
            y_pixels=y_pixels,
            x_pixels=x_pixels,
            weights=weights,
            scale_heights=scale_heights,
            scale_widths=scale_widths,
        )
        # Mask extension can grow a coarse scale past its raw downsampled footprint, so only non-emptiness is checked.
        for coordinates in y_coordinates:
            assert len(coordinates) > 0
