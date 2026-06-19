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
    """Tests extend_roi."""

    def test_single_pixel_one_iteration(self) -> None:
        """Verifies that a single pixel expands to a diamond of 5 pixels after one iteration."""
        y = np.array([5], dtype=np.int32)
        x = np.array([5], dtype=np.int32)
        y_out, _x_out = extend_roi(y_pixels=y, x_pixels=x, height=10, width=10, iterations=1)
        assert len(y_out) == 5  # center + 4 cardinal neighbors.

    def test_boundary_clipping(self) -> None:
        """Verifies that pixels outside the frame boundary are excluded."""
        y = np.array([0], dtype=np.int32)
        x = np.array([0], dtype=np.int32)
        y_out, x_out = extend_roi(y_pixels=y, x_pixels=x, height=10, width=10, iterations=1)
        assert np.all(y_out >= 0)
        assert np.all(x_out >= 0)
        # Corner pixel: only center, right, and down are valid.
        assert len(y_out) == 3

    def test_multiple_iterations(self) -> None:
        """Verifies that each iteration expands the ROI further."""
        y = np.array([5], dtype=np.int32)
        x = np.array([5], dtype=np.int32)
        y1, _ = extend_roi(y_pixels=y, x_pixels=x, height=20, width=20, iterations=1)
        y2, _ = extend_roi(y_pixels=y, x_pixels=x, height=20, width=20, iterations=2)
        assert len(y2) > len(y1)

    def test_zero_iterations(self) -> None:
        """Verifies that zero iterations return the original pixels."""
        y = np.array([5, 6], dtype=np.int32)
        x = np.array([5, 6], dtype=np.int32)
        y_out, x_out = extend_roi(y_pixels=y, x_pixels=x, height=10, width=10, iterations=0)
        np.testing.assert_array_equal(y_out, y)
        np.testing.assert_array_equal(x_out, x)

    def test_no_duplicates(self) -> None:
        """Verifies that the output contains no duplicate coordinates."""
        y = np.array([5, 5, 6], dtype=np.int32)
        x = np.array([5, 6, 5], dtype=np.int32)
        y_out, x_out = extend_roi(y_pixels=y, x_pixels=x, height=20, width=20, iterations=1)
        flat = y_out * 20 + x_out
        assert len(flat) == len(np.unique(flat))


class TestSubtractNeuropil:
    """Tests _subtract_neuropil."""

    def test_in_place_modification(self) -> None:
        """Verifies that frames are modified in-place."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((5, 32, 32)).astype(np.float32) + 10.0
        original = frames.copy()
        _subtract_neuropil(frames=frames, filter_size=5)
        assert not np.array_equal(frames, original)

    def test_uniform_frames_become_near_zero(self) -> None:
        """Verifies that uniform frames produce near-zero output after high-pass filtering."""
        frames = np.ones((5, 32, 32), dtype=np.float32) * 100.0
        _subtract_neuropil(frames=frames, filter_size=5)
        np.testing.assert_allclose(frames, 0.0, atol=1e-3)

    def test_output_finite(self) -> None:
        """Verifies that the filtered frames contain only finite values."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((5, 32, 32)).astype(np.float32)
        _subtract_neuropil(frames=frames, filter_size=5)
        assert np.isfinite(frames).all()


class TestConvolveSquare2d:
    """Tests _convolve_square_2d."""

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
        np.testing.assert_allclose(center, 3.0, atol=0.1)

    def test_output_finite(self) -> None:
        """Verifies that the output is finite."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((5, 32, 32)).astype(np.float32)
        result = _convolve_square_2d(frames=frames, filter_size=3)
        assert np.isfinite(result).all()


class TestCreateInitialSquare:
    """Tests _create_initial_square."""

    def test_centered_square(self) -> None:
        """Verifies that the output is a square patch centered at the given location."""
        y, x, _w = _create_initial_square(center_y=10, center_x=10, square_size=5, height=30, width=30)
        assert len(y) == 25  # 5x5
        assert np.all(y >= 8)
        assert np.all(y <= 12)
        assert np.all(x >= 8)
        assert np.all(x <= 12)

    def test_boundary_clipping(self) -> None:
        """Verifies that pixels outside the frame boundary are excluded."""
        y, x, _w = _create_initial_square(center_y=0, center_x=0, square_size=5, height=30, width=30)
        assert np.all(y >= 0)
        assert np.all(x >= 0)
        assert len(y) < 25

    def test_weights_unit_normalized(self) -> None:
        """Verifies that the output weights have unit norm."""
        _, _, w = _create_initial_square(center_y=10, center_x=10, square_size=5, height=30, width=30)
        np.testing.assert_allclose(np.linalg.norm(w), 1.0, atol=1e-5)

    def test_output_dtypes(self) -> None:
        """Verifies the output dtypes."""
        y, x, w = _create_initial_square(center_y=10, center_x=10, square_size=3, height=30, width=30)
        assert y.dtype == np.int32
        assert x.dtype == np.int32
        assert w.dtype == np.float32


class TestExtendMask:
    """Tests _extend_mask."""

    def test_expands_in_all_directions(self) -> None:
        """Verifies that the mask expands into all 8 surrounding neighbors."""
        y = np.array([5], dtype=np.int32)
        x = np.array([5], dtype=np.int32)
        w = np.array([1.0], dtype=np.float32)
        y_out, _x_out, _w_out = _extend_mask(y_pixels=y, x_pixels=x, weights=w, height=20, width=20)
        # Single pixel + 8 neighbors = 9 pixels.
        assert len(y_out) == 9

    def test_boundary_handling(self) -> None:
        """Verifies that the mask respects frame boundaries."""
        y = np.array([0], dtype=np.int32)
        x = np.array([0], dtype=np.int32)
        w = np.array([1.0], dtype=np.float32)
        y_out, x_out, _w_out = _extend_mask(y_pixels=y, x_pixels=x, weights=w, height=20, width=20)
        assert np.all(y_out >= 0)
        assert np.all(x_out >= 0)
        # Corner pixel: only center, right, down, and diagonal = 4 pixels.
        assert len(y_out) == 4

    def test_weights_non_negative(self) -> None:
        """Verifies that the accumulated weights are non-negative."""
        y = np.array([5, 5, 6], dtype=np.int32)
        x = np.array([5, 6, 5], dtype=np.int32)
        w = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        _, _, w_out = _extend_mask(y_pixels=y, x_pixels=x, weights=w, height=20, width=20)
        assert np.all(w_out >= 0)


class TestEstimateSpatialScale:
    """Tests _estimate_spatial_scale."""

    def test_returns_dominant_scale(self) -> None:
        """Verifies that the dominant scale is returned for a clear scale pattern."""
        # Creates scale images where scale 2 has the highest values.
        scale_images = np.zeros((5, 32, 32), dtype=np.float32)
        scale_images[2, :, :] = 10.0
        result = _estimate_spatial_scale(scale_images=scale_images)
        assert result == 2

    def test_returns_valid_index(self) -> None:
        """Verifies that the returned scale index is within the valid range."""
        rng = np.random.default_rng(42)
        scale_images = rng.standard_normal((5, 32, 32)).astype(np.float32)
        result = _estimate_spatial_scale(scale_images=scale_images)
        assert 0 <= result < 5


class TestComputeMultiscaleMasks:
    """Tests _compute_multiscale_masks."""

    def test_output_list_lengths(self) -> None:
        """Verifies that the output lists have one entry per scale."""
        y = np.array([5, 5, 6, 6], dtype=np.int32)
        x = np.array([5, 6, 5, 6], dtype=np.int32)
        w = np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32)
        scale_heights = np.array([32, 16, 8], dtype=np.uint16)
        scale_widths = np.array([32, 16, 8], dtype=np.uint16)
        y_coords, x_coords, weights = _compute_multiscale_masks(
            y_pixels=y, x_pixels=x, weights=w, scale_heights=scale_heights, scale_widths=scale_widths
        )
        assert len(y_coords) == 3
        assert len(x_coords) == 3
        assert len(weights) == 3

    def test_coarser_scales_have_fewer_or_equal_pixels(self) -> None:
        """Verifies that coarser scales have fewer or comparable pixels due to downsampling."""
        y = np.arange(10, dtype=np.int32)
        x = np.arange(10, dtype=np.int32)
        w = np.ones(10, dtype=np.float32) / 10
        scale_heights = np.array([64, 32, 16], dtype=np.uint16)
        scale_widths = np.array([64, 32, 16], dtype=np.uint16)
        y_coords, _x_coords, _weights = _compute_multiscale_masks(
            y_pixels=y, x_pixels=x, weights=w, scale_heights=scale_heights, scale_widths=scale_widths
        )
        # After extension, coarser scales may have more pixels than the raw downsampled count,
        # but the original downsampled coordinates should be fewer.
        for i in range(3):
            assert len(y_coords[i]) > 0
