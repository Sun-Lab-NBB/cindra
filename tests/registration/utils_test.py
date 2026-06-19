"""Contains tests for the registration utils module."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.fft import rfft2

from cindra.registration.utils import (
    NORMALIZATION_EPSILON,
    apply_mask,
    combine_rigid_offsets,
    compute_reference_fft,
    mean_centered_meshgrid,
    apply_phase_correlation,
    apply_spatial_high_pass,
    apply_spatial_smoothing,
    apply_temporal_smoothing,
    combine_nonrigid_offsets,
    compute_upsampling_kernel,
    _get_normalization_weights,
    _compute_gaussian_rbf_weights,
    compute_gaussian_frequency_filter,
)


class TestApplyPhaseCorrelation:
    """Tests apply_phase_correlation."""

    def test_output_shape(self) -> None:
        """Verifies the output shape matches the input frames shape."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((5, 32, 32)).astype(np.float32)
        reference = rng.standard_normal((32, 32)).astype(np.float32)
        kernel = compute_reference_fft(reference_image=reference)
        result = apply_phase_correlation(frames=frames, kernel=kernel, workers=1)
        assert result.shape == frames.shape

    def test_output_dtype(self) -> None:
        """Verifies the output dtype is float32."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((3, 16, 16)).astype(np.float32)
        reference = rng.standard_normal((16, 16)).astype(np.float32)
        kernel = compute_reference_fft(reference_image=reference)
        result = apply_phase_correlation(frames=frames, kernel=kernel, workers=1)
        assert result.dtype == np.float32

    def test_self_correlation_peak_at_origin(self) -> None:
        """Verifies that correlating a frame with itself produces a peak at the origin."""
        rng = np.random.default_rng(42)
        reference = rng.standard_normal((32, 32)).astype(np.float32)
        kernel = compute_reference_fft(reference_image=reference)
        kernel /= NORMALIZATION_EPSILON + np.abs(kernel)
        frames = reference[np.newaxis, :, :]
        result = apply_phase_correlation(frames=frames, kernel=kernel, workers=1)
        # Peak should be at (0, 0).
        assert result[0, 0, 0] == result[0].max()


class TestApplyMask:
    """Tests apply_mask (numba vectorized)."""

    def test_basic_computation(self) -> None:
        """Verifies the element-wise computation frames * mask + offset."""
        frames = np.ones((2, 4, 4), dtype=np.float32) * 10.0
        mask = np.ones((4, 4), dtype=np.float32) * 0.5
        offset = np.ones((4, 4), dtype=np.float32) * 2.0
        result = apply_mask(frames, mask, offset)
        expected = 10.0 * 0.5 + 2.0
        np.testing.assert_allclose(result, expected)

    def test_zero_mask(self) -> None:
        """Verifies zero mask returns only offset."""
        frames = np.ones((1, 4, 4), dtype=np.float32) * 100.0
        mask = np.zeros((4, 4), dtype=np.float32)
        offset = np.ones((4, 4), dtype=np.float32) * 5.0
        result = apply_mask(frames, mask, offset)
        np.testing.assert_allclose(result, 5.0)

    def test_identity_mask(self) -> None:
        """Verifies identity mask with zero offset returns original frames."""
        frames = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
        mask = np.ones((4, 4), dtype=np.float32)
        offset = np.zeros((4, 4), dtype=np.float32)
        result = apply_mask(frames, mask, offset)
        np.testing.assert_allclose(result, frames)

    def test_broadcasting(self) -> None:
        """Verifies broadcasting works correctly with batch dimensions."""
        frames = np.ones((3, 4, 4), dtype=np.float32) * 2.0
        mask = np.ones((4, 4), dtype=np.float32) * 3.0
        offset = np.ones((4, 4), dtype=np.float32) * 1.0
        result = apply_mask(frames, mask, offset)
        assert result.shape == (3, 4, 4)
        np.testing.assert_allclose(result, 7.0)


class TestCombineRigidOffsets:
    """Tests combine_rigid_offsets."""

    def test_concatenation(self) -> None:
        """Verifies horizontal concatenation of rigid offset batches."""
        batch1 = (
            np.array([1, 2], dtype=np.int32),
            np.array([3, 4], dtype=np.int32),
            np.array([0.9, 0.8], dtype=np.float32),
        )
        batch2 = (
            np.array([5], dtype=np.int32),
            np.array([6], dtype=np.int32),
            np.array([0.7], dtype=np.float32),
        )
        y, x, correlation = combine_rigid_offsets([batch1, batch2])
        np.testing.assert_array_equal(y, [1, 2, 5])
        np.testing.assert_array_equal(x, [3, 4, 6])
        np.testing.assert_allclose(correlation, [0.9, 0.8, 0.7])

    def test_single_batch(self) -> None:
        """Verifies combining a single batch returns the same data."""
        batch = (
            np.array([10, 20], dtype=np.int32),
            np.array([30, 40], dtype=np.int32),
            np.array([1.0, 0.5], dtype=np.float32),
        )
        y, x, _correlation = combine_rigid_offsets([batch])
        np.testing.assert_array_equal(y, [10, 20])
        np.testing.assert_array_equal(x, [30, 40])


class TestCombineNonrigidOffsets:
    """Tests combine_nonrigid_offsets."""

    def test_vertical_stacking(self) -> None:
        """Verifies vertical stacking of nonrigid offset batches."""
        batch1 = (
            np.ones((3, 4), dtype=np.float32),
            np.ones((3, 4), dtype=np.float32) * 2.0,
            np.ones((3, 4), dtype=np.float32) * 0.9,
        )
        batch2 = (
            np.ones((2, 4), dtype=np.float32) * 3.0,
            np.ones((2, 4), dtype=np.float32) * 4.0,
            np.ones((2, 4), dtype=np.float32) * 0.8,
        )
        y, x, correlation = combine_nonrigid_offsets([batch1, batch2])
        assert y.shape == (5, 4)
        assert x.shape == (5, 4)
        assert correlation.shape == (5, 4)


class TestComputeGaussianFrequencyFilter:
    """Tests compute_gaussian_frequency_filter."""

    def test_shape(self) -> None:
        """Verifies the filter shape matches rfft2 output dimensions."""
        result = compute_gaussian_frequency_filter(sigma=1.5, height=32, width=32)
        assert result.shape == (32, 32 // 2 + 1)

    def test_dtype(self) -> None:
        """Verifies the filter dtype is complex64."""
        result = compute_gaussian_frequency_filter(sigma=1.5, height=16, width=16)
        assert result.dtype == np.complex64

    def test_dc_component_near_one(self) -> None:
        """Verifies the DC component is approximately 1.0 (normalized Gaussian)."""
        result = compute_gaussian_frequency_filter(sigma=2.0, height=32, width=32)
        np.testing.assert_allclose(np.abs(result[0, 0]), 1.0, atol=1e-4)

    def test_cache_returns_same_object(self) -> None:
        """Verifies the lru_cache returns the same object for identical parameters."""
        result1 = compute_gaussian_frequency_filter(sigma=3.0, height=64, width=64)
        result2 = compute_gaussian_frequency_filter(sigma=3.0, height=64, width=64)
        assert result1 is result2


class TestApplyTemporalSmoothing:
    """Tests apply_temporal_smoothing."""

    def test_preserves_shape(self) -> None:
        """Verifies the output shape matches the input shape."""
        frames = np.ones((10, 8, 8), dtype=np.float32)
        result = apply_temporal_smoothing(frames=frames, sigma=2.0)
        assert result.shape == frames.shape

    def test_dtype(self) -> None:
        """Verifies the output dtype is float32."""
        frames = np.ones((10, 8, 8), dtype=np.float32)
        result = apply_temporal_smoothing(frames=frames, sigma=2.0)
        assert result.dtype == np.float32

    def test_constant_input_unchanged(self) -> None:
        """Verifies constant input is unchanged by smoothing."""
        frames = np.ones((10, 8, 8), dtype=np.float32) * 5.0
        result = apply_temporal_smoothing(frames=frames, sigma=2.0)
        np.testing.assert_allclose(result, 5.0, atol=1e-5)

    def test_smoothing_reduces_variation(self) -> None:
        """Verifies temporal smoothing reduces frame-to-frame variation."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((50, 8, 8)).astype(np.float32)
        result = apply_temporal_smoothing(frames=frames, sigma=5.0)
        original_std = np.std(np.diff(frames, axis=0))
        smoothed_std = np.std(np.diff(result, axis=0))
        assert smoothed_std < original_std


class TestApplySpatialSmoothing:
    """Tests apply_spatial_smoothing."""

    def test_3d_input(self) -> None:
        """Verifies correct output shape for 3D input."""
        data = np.ones((3, 20, 20), dtype=np.float32)
        result = apply_spatial_smoothing(data=data, window=4)
        assert result.shape == data.shape

    def test_2d_input(self) -> None:
        """Verifies 2D input is handled and returns 2D output."""
        data = np.ones((20, 20), dtype=np.float32)
        result = apply_spatial_smoothing(data=data, window=4)
        assert result.shape == (20, 20)

    def test_constant_input(self) -> None:
        """Verifies constant input produces constant output (after normalization)."""
        data = np.ones((2, 20, 20), dtype=np.float32) * 7.0
        result = apply_spatial_smoothing(data=data, window=4)
        # Interior values should be close to original (border effects exist at edges).
        np.testing.assert_allclose(result[0, 5:15, 5:15], 7.0, atol=0.5)

    def test_odd_window_raises_error(self) -> None:
        """Verifies odd window size raises ValueError."""
        data = np.ones((1, 20, 20), dtype=np.float32)
        with pytest.raises(ValueError, match="Unable to apply spatial smoothing"):
            apply_spatial_smoothing(data=data, window=3)

    def test_even_window_no_error(self) -> None:
        """Verifies even window size does not raise an error."""
        data = np.ones((1, 20, 20), dtype=np.float32)
        result = apply_spatial_smoothing(data=data, window=4)
        assert result is not None


class TestApplySpatialHighPass:
    """Tests apply_spatial_high_pass."""

    def test_removes_uniform_background(self) -> None:
        """Verifies the high-pass filter removes uniform spatial background."""
        data = np.ones((3, 20, 20), dtype=np.float32) * 100.0
        result = apply_spatial_high_pass(data=data, window=4)
        np.testing.assert_allclose(result, 0.0, atol=1e-3)

    def test_preserves_high_frequency(self) -> None:
        """Verifies the high-pass filter preserves high-frequency structure."""
        data = np.zeros((2, 20, 20), dtype=np.float32)
        # Adds a point source to introduce a high-frequency feature.
        data[:, 10, 10] = 100.0
        result = apply_spatial_high_pass(data=data, window=4)
        # The point source should still be prominent after filtering.
        assert result[0, 10, 10] > 50.0

    def test_2d_input(self) -> None:
        """Verifies 2D input is handled and returns 2D output."""
        data = np.ones((20, 20), dtype=np.float32) * 50.0
        result = apply_spatial_high_pass(data=data, window=4)
        assert result.shape == (20, 20)

    def test_output_shape(self) -> None:
        """Verifies the output shape matches the input shape."""
        data = np.ones((5, 16, 16), dtype=np.float32)
        result = apply_spatial_high_pass(data=data, window=4)
        assert result.shape == data.shape


class TestComputeReferenceFft:
    """Tests compute_reference_fft."""

    def test_shape(self) -> None:
        """Verifies the output shape matches rfft2 dimensions."""
        image = np.ones((32, 32), dtype=np.float32)
        result = compute_reference_fft(reference_image=image)
        assert result.shape == (32, 32 // 2 + 1)

    def test_dtype(self) -> None:
        """Verifies the output dtype is complex64."""
        image = np.ones((16, 16), dtype=np.float32)
        result = compute_reference_fft(reference_image=image)
        assert result.dtype == np.complex64

    def test_conjugate_property(self) -> None:
        """Verifies the result is the complex conjugate of the rfft2."""
        image = np.random.default_rng(42).standard_normal((16, 16)).astype(np.float32)
        result = compute_reference_fft(reference_image=image)
        expected = np.conj(rfft2(image, axes=(-2, -1))).astype(np.complex64)
        np.testing.assert_allclose(result, expected, atol=1e-5)


class TestComputeUpsamplingKernel:
    """Tests compute_upsampling_kernel."""

    def test_output_types(self) -> None:
        """Verifies the return types are (ndarray, int)."""
        kernel, upsampled_size = compute_upsampling_kernel(padding=3, subpixel=10)
        assert isinstance(kernel, np.ndarray)
        assert isinstance(upsampled_size, int)

    def test_kernel_shape(self) -> None:
        """Verifies the kernel matrix shape based on padding and subpixel parameters."""
        padding = 3
        subpixel = 10
        kernel, upsampled_size = compute_upsampling_kernel(padding=padding, subpixel=subpixel)
        low_resolution_size = (2 * padding + 1) ** 2
        assert kernel.shape[0] == low_resolution_size
        assert kernel.shape[1] == upsampled_size**2

    def test_dtype(self) -> None:
        """Verifies the kernel dtype is float32."""
        kernel, _ = compute_upsampling_kernel(padding=2, subpixel=5)
        assert kernel.dtype == np.float32

    def test_cache_returns_same_object(self) -> None:
        """Verifies the lru_cache returns the same object for identical parameters."""
        result1 = compute_upsampling_kernel(padding=4, subpixel=10)
        result2 = compute_upsampling_kernel(padding=4, subpixel=10)
        assert result1[0] is result2[0]


class TestMeanCenteredMeshgridRegistration:
    """Tests _mean_centered_meshgrid in registration/utils."""

    def test_shape(self) -> None:
        """Verifies meshgrid output shapes match input dimensions."""
        col_dist, row_dist = mean_centered_meshgrid(height=10, width=20)
        assert col_dist.shape == (10, 20)
        assert row_dist.shape == (10, 20)

    def test_center_zero_for_odd_dimensions(self) -> None:
        """Verifies center values are zero for odd dimensions."""
        col_dist, row_dist = mean_centered_meshgrid(height=11, width=11)
        np.testing.assert_allclose(row_dist[5, 5], 0.0)
        np.testing.assert_allclose(col_dist[5, 5], 0.0)

    def test_dtype(self) -> None:
        """Verifies the meshgrid dtype is float32."""
        col_dist, row_dist = mean_centered_meshgrid(height=8, width=8)
        assert col_dist.dtype == np.float32
        assert row_dist.dtype == np.float32


class TestComputeGaussianRbfWeights:
    """Tests _compute_gaussian_rbf_weights (private)."""

    def test_square_matrix_for_same_coordinates(self) -> None:
        """Verifies a square matrix is returned when source equals target."""
        coords = np.arange(-2, 3, dtype=np.float64)
        weights = _compute_gaussian_rbf_weights(source_coordinates=coords, target_coordinates=coords)
        n = len(coords)
        assert weights.shape == (n**2, n**2)

    def test_diagonal_is_one(self) -> None:
        """Verifies diagonal elements are 1.0 (zero distance)."""
        coords = np.arange(-2, 3, dtype=np.float64)
        weights = _compute_gaussian_rbf_weights(source_coordinates=coords, target_coordinates=coords)
        np.testing.assert_allclose(np.diag(weights), 1.0, atol=1e-10)

    def test_symmetry(self) -> None:
        """Verifies the weight matrix is symmetric when source equals target."""
        coords = np.arange(-2, 3, dtype=np.float64)
        weights = _compute_gaussian_rbf_weights(source_coordinates=coords, target_coordinates=coords)
        np.testing.assert_allclose(weights, weights.T, atol=1e-10)

    def test_values_in_unit_range(self) -> None:
        """Verifies all weight values are in (0, 1]."""
        coords = np.arange(-3, 4, dtype=np.float64)
        weights = _compute_gaussian_rbf_weights(source_coordinates=coords, target_coordinates=coords)
        assert weights.min() > 0
        assert weights.max() <= 1.0 + 1e-10


class TestGetNormalizationWeights:
    """Tests _get_normalization_weights (private)."""

    def test_shape(self) -> None:
        """Verifies the output shape matches (height, width)."""
        weights = _get_normalization_weights(height=20, width=20, window=4)
        assert weights.shape == (20, 20)

    def test_interior_near_one(self) -> None:
        """Verifies interior normalization weights are close to 1.0."""
        weights = _get_normalization_weights(height=30, width=30, window=4)
        # Interior values should be close to 1.0.
        np.testing.assert_allclose(weights[10:20, 10:20], 1.0, atol=0.1)

    def test_border_less_than_one(self) -> None:
        """Verifies border weights are less than 1.0 due to zero-padding."""
        weights = _get_normalization_weights(height=30, width=30, window=4)
        assert weights[0, 0] < 1.0
