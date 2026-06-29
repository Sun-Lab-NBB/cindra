"""Contains tests for the detection utils module."""

from __future__ import annotations

import numpy as np

from cindra.detection.utils import (
    downsample,
    mean_centered_meshgrid,
    _apply_gaussian_high_pass,
    compute_spatial_taper_mask,
    compute_registration_blocks,
    compute_thresholded_variance,
    _apply_rolling_mean_high_pass,
    compute_block_smoothing_kernel,
    apply_temporal_high_pass_filter,
    compute_temporal_standard_deviation,
)


class TestComputeSpatialTaperMask:
    """Tests compute_spatial_taper_mask."""

    def test_shape_and_dtype(self) -> None:
        """Verifies the taper mask has the correct shape and dtype."""
        mask = compute_spatial_taper_mask(sigma=5.0, height=64, width=128)
        assert mask.shape == (64, 128)
        assert mask.dtype == np.float32

    def test_values_in_unit_range(self) -> None:
        """Verifies taper mask values fall within [0, 1]."""
        mask = compute_spatial_taper_mask(sigma=5.0, height=100, width=100)
        assert mask.min() >= 0.0
        assert mask.max() <= 1.0

    def test_center_near_one(self) -> None:
        """Verifies the center of the taper mask is close to 1.0."""
        mask = compute_spatial_taper_mask(sigma=5.0, height=101, width=101)
        assert mask[50, 50] > 0.95

    def test_edges_near_zero(self) -> None:
        """Verifies the edges of the taper mask approach zero."""
        mask = compute_spatial_taper_mask(sigma=5.0, height=100, width=100)
        assert mask[0, 0] < 0.15
        assert mask[0, 99] < 0.15
        assert mask[99, 0] < 0.15
        assert mask[99, 99] < 0.15

    def test_symmetry(self) -> None:
        """Verifies the taper mask is symmetric around its center."""
        mask = compute_spatial_taper_mask(sigma=5.0, height=100, width=100)
        np.testing.assert_allclose(mask, np.flip(mask, axis=0), atol=1e-6)
        np.testing.assert_allclose(mask, np.flip(mask, axis=1), atol=1e-6)

    def test_sigma_controls_taper_width(self) -> None:
        """Verifies that larger sigma produces a more gradual taper that starts closer to center."""
        mask_narrow = compute_spatial_taper_mask(sigma=2.0, height=100, width=100)
        mask_wide = compute_spatial_taper_mask(sigma=10.0, height=100, width=100)
        # With wider sigma, the taper starts closer to center, so at an interior position (row 25)
        # the narrow taper is still ~1.0 while the wide taper has already begun declining.
        assert mask_narrow[25, 50] > mask_wide[25, 50]

    def test_small_image(self) -> None:
        """Verifies taper mask works for small images."""
        mask = compute_spatial_taper_mask(sigma=1.0, height=5, width=5)
        assert mask.shape == (5, 5)
        assert mask.dtype == np.float32


class TestDownsample:
    """Tests downsample."""

    def test_even_dimensions(self) -> None:
        """Verifies 2x downsampling with even spatial dimensions."""
        data = np.ones((2, 10, 12), dtype=np.float32)
        result = downsample(data=data)
        assert result.shape == (2, 5, 6)
        np.testing.assert_allclose(result, 1.0)

    def test_odd_dimensions_with_taper(self) -> None:
        """Verifies 2x downsampling with odd dimensions and edge tapering."""
        data = np.ones((1, 11, 13), dtype=np.float32)
        result = downsample(data=data, taper_edge=True)
        assert result.shape == (1, 6, 7)
        # Interior elements from complete 2x2 blocks should be 1.0.
        np.testing.assert_allclose(result[0, :5, :6], 1.0)
        # Bottom-right corner: tapered twice (0.5 * 0.5 = 0.25)
        np.testing.assert_allclose(result[0, -1, -1], 0.25)

    def test_odd_dimensions_without_taper(self) -> None:
        """Verifies 2x downsampling with odd dimensions without edge tapering."""
        data = np.ones((1, 11, 13), dtype=np.float32)
        result = downsample(data=data, taper_edge=False)
        assert result.shape == (1, 6, 7)
        np.testing.assert_allclose(result, 1.0)

    def test_averages_2x2_blocks(self) -> None:
        """Verifies that downsampling correctly averages 2x2 blocks."""
        data = np.zeros((1, 4, 4), dtype=np.float32)
        data[0, 0, 0] = 4.0
        result = downsample(data=data)
        assert result.shape == (1, 2, 2)
        np.testing.assert_allclose(result[0, 0, 0], 1.0)

    def test_odd_width_even_height(self) -> None:
        """Verifies downsampling when only width is odd."""
        data = np.ones((1, 10, 11), dtype=np.float32)
        result = downsample(data=data, taper_edge=True)
        assert result.shape == (1, 5, 6)
        np.testing.assert_allclose(result[0, :, :5], 1.0)
        np.testing.assert_allclose(result[0, :, -1], 0.5)

    def test_even_width_odd_height(self) -> None:
        """Verifies downsampling when only height is odd."""
        data = np.ones((1, 11, 10), dtype=np.float32)
        result = downsample(data=data, taper_edge=True)
        assert result.shape == (1, 6, 5)
        np.testing.assert_allclose(result[0, :5, :], 1.0)
        np.testing.assert_allclose(result[0, -1, :], 0.5)

    def test_single_row_skips_block_reshape(self) -> None:
        """Verifies downsampling a single-row input averages along width only."""
        data = np.zeros((1, 1, 4), dtype=np.float32)
        data[0, 0, :] = [1.0, 3.0, 5.0, 7.0]
        result = downsample(data=data, taper_edge=False)
        # height=1 makes even_height 0, so the main 2x2-block reshape is skipped and only the bottom-row
        # width averaging runs: [(1+3)/2, (5+7)/2] = [2.0, 6.0].
        assert result.shape == (1, 1, 2)
        np.testing.assert_allclose(result, [[[2.0, 6.0]]])

    def test_preserves_depth_dimension(self) -> None:
        """Verifies the depth (frame) dimension is preserved and each frame is downsampled independently."""
        data = np.ones((5, 10, 10), dtype=np.float32)
        result = downsample(data=data)
        assert result.shape == (5, 5, 5)
        np.testing.assert_allclose(result, 1.0)


class TestComputeTemporalStandardDeviation:
    """Tests compute_temporal_standard_deviation."""

    def test_shape(self) -> None:
        """Verifies the output shape matches spatial dimensions."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((20, 32, 32)).astype(np.float32)
        result = compute_temporal_standard_deviation(frames=frames)
        assert result.shape == (32, 32)
        assert result.dtype == np.float32

    def test_constant_frames_return_minimum(self) -> None:
        """Verifies constant frames produce the minimum threshold value."""
        frames = np.ones((10, 8, 8), dtype=np.float32) * 5.0
        result = compute_temporal_standard_deviation(frames=frames)
        np.testing.assert_allclose(result, 1e-10)

    def test_all_values_positive(self) -> None:
        """Verifies all output values are strictly positive."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((20, 16, 16)).astype(np.float32)
        result = compute_temporal_standard_deviation(frames=frames)
        assert np.all(result > 0)

    def test_higher_variation_produces_larger_values(self) -> None:
        """Verifies pixels with more temporal variation produce larger standard deviation."""
        frames = np.zeros((20, 4, 4), dtype=np.float32)
        # Left half: high variation.
        rng = np.random.default_rng(42)
        frames[:, :, :2] = rng.standard_normal((20, 4, 2)).astype(np.float32) * 10.0
        # Right half: low variation.
        frames[:, :, 2:] = rng.standard_normal((20, 4, 2)).astype(np.float32) * 0.1
        result = compute_temporal_standard_deviation(frames=frames)
        assert result[:, :2].mean() > result[:, 2:].mean()


class TestComputeThresholdedVariance:
    """Tests compute_thresholded_variance."""

    def test_shape(self) -> None:
        """Verifies the output shape matches spatial dimensions."""
        frames = np.ones((10, 16, 16), dtype=np.float32)
        result = compute_thresholded_variance(frames=frames, intensity_threshold=0.0)
        assert result.shape == (16, 16)

    def test_all_below_threshold_gives_zero(self) -> None:
        """Verifies that all-below-threshold data produces zero output."""
        frames = np.ones((10, 8, 8), dtype=np.float32) * 0.5
        result = compute_thresholded_variance(frames=frames, intensity_threshold=1.0)
        np.testing.assert_allclose(result, 0.0)

    def test_threshold_excludes_low_values(self) -> None:
        """Verifies that values below threshold are excluded from computation."""
        frames = np.array([[[1.0, -1.0], [2.0, -2.0]]], dtype=np.float32)
        result_all = compute_thresholded_variance(frames=frames, intensity_threshold=-10.0)
        result_positive = compute_thresholded_variance(frames=frames, intensity_threshold=0.0)
        assert np.all(result_positive <= result_all + 1e-6)


class TestApplyTemporalHighPassFilter:
    """Tests apply_temporal_high_pass_filter."""

    def test_gaussian_dispatch_modifies_frames(self) -> None:
        """Verifies Gaussian dispatch for small kernel sizes modifies frames."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((20, 8, 8)).astype(np.float32) + 10.0
        frames_copy = frames.copy()
        apply_temporal_high_pass_filter(frames=frames, kernel_size=5)
        assert not np.array_equal(frames, frames_copy)

    def test_rolling_dispatch_modifies_frames(self) -> None:
        """Verifies rolling mean dispatch for large kernel sizes modifies frames."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((50, 8, 8)).astype(np.float32) + 10.0
        frames_copy = frames.copy()
        apply_temporal_high_pass_filter(frames=frames, kernel_size=15)
        assert not np.array_equal(frames, frames_copy)

    def test_in_place_modification(self) -> None:
        """Verifies the filter modifies frames in-place."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((20, 8, 8)).astype(np.float32) + 100.0
        original_id = id(frames)
        apply_temporal_high_pass_filter(frames=frames, kernel_size=5)
        assert id(frames) == original_id

    def test_removes_constant_offset(self) -> None:
        """Verifies the high-pass filter removes constant temporal offset."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((30, 4, 4)).astype(np.float32) + 1000.0
        apply_temporal_high_pass_filter(frames=frames, kernel_size=5)
        # After high-pass, temporal mean should be significantly reduced.
        assert np.abs(frames.mean()) < 10.0


class TestApplyGaussianHighPass:
    """Tests _apply_gaussian_high_pass."""

    def test_in_place_subtraction(self) -> None:
        """Verifies Gaussian high-pass subtracts low-frequency content in-place."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((20, 8, 8)).astype(np.float32) + 50.0
        original = frames.copy()
        _apply_gaussian_high_pass(frames=frames, kernel_size=3)
        assert not np.array_equal(frames, original)
        # High-pass output should have near-zero mean.
        assert np.abs(frames.mean()) < 5.0


class TestApplyRollingMeanHighPass:
    """Tests _apply_rolling_mean_high_pass."""

    def test_complete_windows_no_remainder(self) -> None:
        """Verifies rolling mean with frames evenly divisible by kernel size."""
        frames = np.ones((30, 4, 4), dtype=np.float32) * 10.0
        _apply_rolling_mean_high_pass(frames=frames, kernel_size=10)
        # Constant frames minus their window mean should give zero.
        np.testing.assert_allclose(frames, 0.0, atol=1e-6)

    def test_with_remainder(self) -> None:
        """Verifies rolling mean handles remaining frames after complete windows."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((35, 4, 4)).astype(np.float32) + 100.0
        _apply_rolling_mean_high_pass(frames=frames, kernel_size=10)
        # Complete windows (30 frames): within-window mean should be ~0.
        for window_start in range(0, 30, 10):
            window = frames[window_start : window_start + 10]
            np.testing.assert_allclose(window.mean(axis=0), 0.0, atol=1e-4)
        # Remainder (5 frames): their mean should also be ~0.
        np.testing.assert_allclose(frames[30:].mean(axis=0), 0.0, atol=1e-4)


class TestComputeBlockSmoothingKernel:
    """Tests compute_block_smoothing_kernel."""

    def test_shape(self) -> None:
        """Verifies kernel matrix shape matches total block count."""
        kernel = compute_block_smoothing_kernel(x_block_count=3, y_block_count=4)
        total_blocks = 3 * 4
        assert kernel.shape == (total_blocks, total_blocks)

    def test_column_normalization(self) -> None:
        """Verifies each column of the kernel sums to 1."""
        kernel = compute_block_smoothing_kernel(x_block_count=3, y_block_count=4)
        column_sums = kernel.sum(axis=0)
        np.testing.assert_allclose(column_sums, 1.0, atol=1e-5)

    def test_single_block(self) -> None:
        """Verifies a single block produces a 1x1 identity kernel."""
        kernel = compute_block_smoothing_kernel(x_block_count=1, y_block_count=1)
        assert kernel.shape == (1, 1)
        np.testing.assert_allclose(kernel, [[1.0]])

    def test_diagonal_is_max(self) -> None:
        """Verifies diagonal elements (self-weights) are the largest in each column."""
        kernel = compute_block_smoothing_kernel(x_block_count=3, y_block_count=3)
        for column in range(kernel.shape[1]):
            assert kernel[column, column] == kernel[:, column].max()

    def test_dtype(self) -> None:
        """Verifies the kernel dtype is float32."""
        kernel = compute_block_smoothing_kernel(x_block_count=2, y_block_count=2)
        assert kernel.dtype == np.float32


class TestComputeRegistrationBlocks:
    """Tests compute_registration_blocks."""

    def test_single_block_when_smaller_than_block_size(self) -> None:
        """Verifies a single block when image is smaller than block size."""
        y_blocks, x_blocks, block_counts, actual_size, _kernel = compute_registration_blocks(
            height=64, width=64, block_size=(128, 128)
        )
        assert block_counts == (1, 1)
        assert actual_size == (64, 64)
        assert len(y_blocks) == 1
        assert len(x_blocks) == 1

    def test_multiple_blocks(self) -> None:
        """Verifies block layout with overlapping blocks."""
        y_blocks, x_blocks, block_counts, _actual_size, _kernel = compute_registration_blocks(
            height=256, width=256, block_size=(128, 128)
        )
        assert block_counts[0] >= 2
        assert block_counts[1] >= 2
        total_blocks = block_counts[0] * block_counts[1]
        assert len(y_blocks) == total_blocks
        assert len(x_blocks) == total_blocks

    def test_blocks_cover_full_image(self) -> None:
        """Verifies blocks cover the entire image area."""
        height, width = 256, 256
        y_blocks, x_blocks, _, _actual_size, _ = compute_registration_blocks(
            height=height, width=width, block_size=(128, 128)
        )
        assert y_blocks[0][0] == 0
        assert x_blocks[0][0] == 0
        # Last block should end at image boundary.
        assert y_blocks[-1][1] == height
        assert x_blocks[-1][1] == width

    def test_block_boundaries_are_valid(self) -> None:
        """Verifies all block boundaries are within image bounds."""
        height, width = 300, 400
        y_blocks, x_blocks, _, _, _ = compute_registration_blocks(height=height, width=width, block_size=(128, 128))
        for y_block in y_blocks:
            assert y_block[0] >= 0
            assert y_block[1] <= height
            assert y_block[1] > y_block[0]
        for x_block in x_blocks:
            assert x_block[0] >= 0
            assert x_block[1] <= width
            assert x_block[1] > x_block[0]

    def test_smoothing_kernel_returned(self) -> None:
        """Verifies the smoothing kernel has correct dimensions."""
        _, _, block_counts, _, kernel = compute_registration_blocks(height=256, width=256, block_size=(128, 128))
        total_blocks = block_counts[0] * block_counts[1]
        assert kernel.shape == (total_blocks, total_blocks)

    def test_block_size_equals_image_size(self) -> None:
        """Verifies single block when block size equals image size."""
        _y_blocks, _x_blocks, block_counts, actual_size, _ = compute_registration_blocks(
            height=128, width=128, block_size=(128, 128)
        )
        assert block_counts == (1, 1)
        assert actual_size == (128, 128)


class TestMeanCenteredMeshgrid:
    """Tests mean_centered_meshgrid."""

    def test_shape(self) -> None:
        """Verifies meshgrid output shapes match input dimensions."""
        column_distances, row_distances = mean_centered_meshgrid(height=10, width=20)
        assert column_distances.shape == (10, 20)
        assert row_distances.shape == (10, 20)

    def test_center_zero_for_odd_dimensions(self) -> None:
        """Verifies center values are zero for odd dimensions."""
        column_distances, row_distances = mean_centered_meshgrid(height=11, width=11)
        np.testing.assert_allclose(row_distances[5, 5], 0.0)
        np.testing.assert_allclose(column_distances[5, 5], 0.0)

    def test_symmetry(self) -> None:
        """Verifies the meshgrid is symmetric."""
        column_distances, row_distances = mean_centered_meshgrid(height=10, width=10)
        np.testing.assert_allclose(column_distances, np.flip(column_distances, axis=1))
        np.testing.assert_allclose(row_distances, np.flip(row_distances, axis=0))

    def test_dtype(self) -> None:
        """Verifies the meshgrid dtype is float32."""
        column_distances, row_distances = mean_centered_meshgrid(height=8, width=8)
        assert column_distances.dtype == np.float32
        assert row_distances.dtype == np.float32

    def test_max_distance_at_corners(self) -> None:
        """Verifies maximum distances occur at corners."""
        column_distances, row_distances = mean_centered_meshgrid(height=10, width=10)
        # Corner should have max distance from center.
        assert row_distances[0, 0] == row_distances.max()
        assert column_distances[0, 0] == column_distances.max()
