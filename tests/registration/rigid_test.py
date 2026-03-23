"""Contains tests for the rigid module."""

from __future__ import annotations

import numpy as np

from cindra.registration.rigid import (
    translate_frame,
    apply_edge_taper,
    compute_edge_taper,
    compute_rigid_offsets,
    compute_phase_correlation_kernel,
)


class TestComputeEdgeTaper:
    """Tests for compute_edge_taper."""

    def test_output_shapes(self) -> None:
        """Verifies both outputs have the correct shape."""
        reference = np.ones((64, 64), dtype=np.float32) * 100.0
        taper_mask, mean_offset = compute_edge_taper(reference_image=reference, taper_slope=5.0)
        assert taper_mask.shape == (64, 64)
        assert mean_offset.shape == (64, 64)

    def test_output_dtypes(self) -> None:
        """Verifies both outputs are float32."""
        reference = np.ones((32, 32), dtype=np.float32)
        taper_mask, mean_offset = compute_edge_taper(reference_image=reference, taper_slope=5.0)
        assert taper_mask.dtype == np.float32
        assert mean_offset.dtype == np.float32

    def test_mask_center_near_one(self) -> None:
        """Verifies the taper mask center is close to 1.0."""
        reference = np.ones((101, 101), dtype=np.float32)
        taper_mask, _ = compute_edge_taper(reference_image=reference, taper_slope=5.0)
        assert taper_mask[50, 50] > 0.95

    def test_mask_plus_offset_equals_mean(self) -> None:
        """Verifies that taper_mask * value + offset preserves the reference mean at edges."""
        reference = np.ones((64, 64), dtype=np.float32) * 100.0
        taper_mask, mean_offset = compute_edge_taper(reference_image=reference, taper_slope=5.0)
        # offset = mean * (1 - mask), so mask * mean + offset = mean everywhere
        reconstructed = taper_mask * 100.0 + mean_offset
        np.testing.assert_allclose(reconstructed, 100.0, atol=1e-4)

    def test_slope_affects_taper_width(self) -> None:
        """Verifies that different slopes produce different taper profiles."""
        reference = np.ones((100, 100), dtype=np.float32)
        mask_narrow, _ = compute_edge_taper(reference_image=reference, taper_slope=2.0)
        mask_wide, _ = compute_edge_taper(reference_image=reference, taper_slope=10.0)
        # Wider taper starts closer to center, so at an interior position the narrow
        # taper is still ~1.0 while the wide taper has already begun declining.
        assert mask_narrow[25, 50] > mask_wide[25, 50]


class TestApplyEdgeTaper:
    """Tests for apply_edge_taper."""

    def test_output_shape(self) -> None:
        """Verifies the output shape matches the input frames shape."""
        frames = np.ones((5, 32, 32), dtype=np.float32)
        mask = np.ones((32, 32), dtype=np.float32)
        offset = np.zeros((32, 32), dtype=np.float32)
        result = apply_edge_taper(frames=frames, taper_mask=mask, mean_offset=offset)
        assert result.shape == (5, 32, 32)

    def test_identity_mask_preserves_frames(self) -> None:
        """Verifies identity mask with zero offset preserves original frames."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((3, 16, 16)).astype(np.float32)
        mask = np.ones((16, 16), dtype=np.float32)
        offset = np.zeros((16, 16), dtype=np.float32)
        result = apply_edge_taper(frames=frames, taper_mask=mask, mean_offset=offset)
        np.testing.assert_allclose(result, frames, atol=1e-6)


class TestComputePhaseCorrelationKernel:
    """Tests for compute_phase_correlation_kernel."""

    def test_shape(self) -> None:
        """Verifies the kernel shape matches rfft2 output dimensions."""
        reference = np.ones((32, 32), dtype=np.float32)
        kernel = compute_phase_correlation_kernel(reference_image=reference)
        assert kernel.shape == (32, 32 // 2 + 1)

    def test_dtype(self) -> None:
        """Verifies the kernel dtype is complex64."""
        reference = np.ones((16, 16), dtype=np.float32)
        kernel = compute_phase_correlation_kernel(reference_image=reference)
        assert kernel.dtype == np.complex64

    def test_no_smoothing(self) -> None:
        """Verifies the kernel works without Gaussian smoothing."""
        rng = np.random.default_rng(42)
        reference = rng.standard_normal((32, 32)).astype(np.float32)
        kernel = compute_phase_correlation_kernel(reference_image=reference, smoothing_sigma=0.0)
        assert kernel.shape == (32, 32 // 2 + 1)

    def test_with_smoothing(self) -> None:
        """Verifies the kernel works with Gaussian smoothing."""
        rng = np.random.default_rng(42)
        reference = rng.standard_normal((32, 32)).astype(np.float32)
        kernel_no_smooth = compute_phase_correlation_kernel(reference_image=reference, smoothing_sigma=0.0)
        kernel_smooth = compute_phase_correlation_kernel(reference_image=reference, smoothing_sigma=1.5)
        # Smoothed kernel should differ from unsmoothed
        assert not np.allclose(kernel_no_smooth, kernel_smooth)

    def test_normalized_magnitude(self) -> None:
        """Verifies the kernel magnitudes are approximately normalized."""
        rng = np.random.default_rng(42)
        reference = rng.standard_normal((32, 32)).astype(np.float32)
        kernel = compute_phase_correlation_kernel(reference_image=reference, smoothing_sigma=0.0)
        magnitudes = np.abs(kernel)
        # After normalization, magnitudes should be close to 1.0 (within epsilon tolerance)
        np.testing.assert_allclose(magnitudes, 1.0, atol=0.15)


class TestComputeRigidOffsets:
    """Tests for compute_rigid_offsets."""

    def test_zero_offset_for_identical_frames(self) -> None:
        """Verifies zero offsets when frames match the reference."""
        rng = np.random.default_rng(42)
        reference = rng.standard_normal((64, 64)).astype(np.float32)
        kernel = compute_phase_correlation_kernel(reference_image=reference)
        frames = np.tile(reference, (3, 1, 1))
        y_offsets, x_offsets, _correlation = compute_rigid_offsets(
            frames=frames,
            reference_kernel=kernel,
            maximum_offset_fraction=0.5,
            temporal_smoothing_sigma=0.0,
            workers=1,
        )
        assert y_offsets.shape == (3,)
        assert x_offsets.shape == (3,)
        np.testing.assert_array_equal(y_offsets, 0)
        np.testing.assert_array_equal(x_offsets, 0)

    def test_detects_known_translation(self) -> None:
        """Verifies detection of a known rigid translation."""
        rng = np.random.default_rng(42)
        reference = rng.standard_normal((64, 64)).astype(np.float32)
        kernel = compute_phase_correlation_kernel(reference_image=reference)
        # Shift the reference by (3, -2) using roll
        shifted = np.roll(reference, shift=(3, -2), axis=(0, 1))
        frames = shifted[np.newaxis, :, :]
        y_offsets, x_offsets, _correlation = compute_rigid_offsets(
            frames=frames,
            reference_kernel=kernel,
            maximum_offset_fraction=0.5,
            temporal_smoothing_sigma=0.0,
            workers=1,
        )
        assert y_offsets[0] == 3
        assert x_offsets[0] == -2

    def test_output_dtypes(self) -> None:
        """Verifies the output dtypes are correct."""
        rng = np.random.default_rng(42)
        reference = rng.standard_normal((32, 32)).astype(np.float32)
        kernel = compute_phase_correlation_kernel(reference_image=reference)
        frames = np.tile(reference, (2, 1, 1))
        y_offsets, x_offsets, correlation = compute_rigid_offsets(
            frames=frames,
            reference_kernel=kernel,
            maximum_offset_fraction=0.5,
            temporal_smoothing_sigma=0.0,
            workers=1,
        )
        assert y_offsets.dtype == np.int32
        assert x_offsets.dtype == np.int32
        assert correlation.dtype == np.float32

    def test_with_temporal_smoothing(self) -> None:
        """Verifies offsets can be computed with temporal smoothing enabled."""
        rng = np.random.default_rng(42)
        reference = rng.standard_normal((32, 32)).astype(np.float32)
        kernel = compute_phase_correlation_kernel(reference_image=reference)
        frames = np.tile(reference, (10, 1, 1))
        y_offsets, x_offsets, _correlation = compute_rigid_offsets(
            frames=frames,
            reference_kernel=kernel,
            maximum_offset_fraction=0.5,
            temporal_smoothing_sigma=1.0,
            workers=1,
        )
        assert y_offsets.shape == (10,)
        np.testing.assert_array_equal(y_offsets, 0)
        np.testing.assert_array_equal(x_offsets, 0)

    def test_positive_correlation_for_matching_frames(self) -> None:
        """Verifies that matching frames produce high positive correlation values."""
        rng = np.random.default_rng(42)
        reference = rng.standard_normal((32, 32)).astype(np.float32)
        kernel = compute_phase_correlation_kernel(reference_image=reference)
        frames = np.tile(reference, (2, 1, 1))
        _, _, correlation = compute_rigid_offsets(
            frames=frames,
            reference_kernel=kernel,
            maximum_offset_fraction=0.5,
            temporal_smoothing_sigma=0.0,
            workers=1,
        )
        assert np.all(correlation > 0)


class TestTranslateFrame:
    """Tests for translate_frame."""

    def test_zero_offset_identity(self) -> None:
        """Verifies zero offset produces no change."""
        rng = np.random.default_rng(42)
        frame = rng.standard_normal((32, 32)).astype(np.float32)
        result = translate_frame(frame=frame, y_offset=0, x_offset=0)
        np.testing.assert_array_equal(result, frame)

    def test_known_vertical_shift(self) -> None:
        """Verifies correct vertical circular shift."""
        frame = np.zeros((8, 8), dtype=np.float32)
        frame[0, :] = 1.0
        result = translate_frame(frame=frame, y_offset=2, x_offset=0)
        # y_offset=2 means shift content upward by 2, so row 0 moves to row -2 = row 6
        np.testing.assert_allclose(result[6, :], 1.0)
        np.testing.assert_allclose(result[0, :], 0.0)

    def test_known_horizontal_shift(self) -> None:
        """Verifies correct horizontal circular shift."""
        frame = np.zeros((8, 8), dtype=np.float32)
        frame[:, 0] = 1.0
        result = translate_frame(frame=frame, y_offset=0, x_offset=3)
        # x_offset=3 means shift content left by 3, so col 0 moves to col -3 = col 5
        np.testing.assert_allclose(result[:, 5], 1.0)
        np.testing.assert_allclose(result[:, 0], 0.0)

    def test_combined_shift(self) -> None:
        """Verifies correct combined vertical and horizontal shift."""
        frame = np.zeros((8, 8), dtype=np.float32)
        frame[2, 3] = 1.0
        result = translate_frame(frame=frame, y_offset=1, x_offset=1)
        # (2, 3) should move to (2-1, 3-1) = (1, 2)
        assert result[1, 2] == 1.0

    def test_preserves_shape_and_dtype(self) -> None:
        """Verifies the output shape and dtype match the input."""
        frame = np.ones((16, 32), dtype=np.float32)
        result = translate_frame(frame=frame, y_offset=5, x_offset=-3)
        assert result.shape == (16, 32)
        assert result.dtype == np.float32

    def test_roundtrip_translation(self) -> None:
        """Verifies that applying opposite translations returns the original frame."""
        rng = np.random.default_rng(42)
        frame = rng.standard_normal((32, 32)).astype(np.float32)
        shifted = translate_frame(frame=frame, y_offset=5, x_offset=-3)
        restored = translate_frame(frame=shifted, y_offset=-5, x_offset=3)
        np.testing.assert_array_equal(restored, frame)
