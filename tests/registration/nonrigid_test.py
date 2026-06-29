"""Contains tests for the nonrigid module."""

from __future__ import annotations

import numpy as np

from cindra.detection import compute_registration_blocks
from cindra.registration.nonrigid import (
    _upsample_block_offsets,
    compute_nonrigid_offsets,
    apply_nonrigid_correction,
    compute_nonrigid_reference_data,
)


class TestComputeNonrigidReferenceData:
    """Tests compute_nonrigid_reference_data."""

    def test_output_shapes(self) -> None:
        """Verifies the output arrays have correct shapes."""
        reference = np.ones((64, 64), dtype=np.float32)
        y_blocks = [np.array([0, 32], dtype=np.int32), np.array([0, 32], dtype=np.int32)]
        x_blocks = [np.array([0, 32], dtype=np.int32), np.array([32, 64], dtype=np.int32)]

        taper, offset, kernel = compute_nonrigid_reference_data(
            reference_image=reference,
            taper_slope=5.0,
            smoothing_sigma=1.15,
            y_blocks=y_blocks,
            x_blocks=x_blocks,
        )

        block_count = 2
        block_height, block_width = 32, 32
        rfft_width = block_width // 2 + 1

        assert taper.shape == (block_count, block_height, block_width)
        assert offset.shape == (block_count, block_height, block_width)
        assert kernel.shape == (block_count, block_height, rfft_width)

    def test_output_dtypes(self) -> None:
        """Verifies the output dtypes are correct."""
        reference = np.ones((64, 64), dtype=np.float32)
        y_blocks = [np.array([0, 32], dtype=np.int32)]
        x_blocks = [np.array([0, 32], dtype=np.int32)]

        taper, offset, kernel = compute_nonrigid_reference_data(
            reference_image=reference,
            taper_slope=5.0,
            smoothing_sigma=1.15,
            y_blocks=y_blocks,
            x_blocks=x_blocks,
        )

        assert taper.dtype == np.float32
        assert offset.dtype == np.float32
        assert kernel.dtype == np.complex64

    def test_taper_mask_values_bounded(self) -> None:
        """Verifies that taper mask values are in [0, 1]."""
        reference = np.ones((64, 64), dtype=np.float32) * 100.0
        y_blocks = [np.array([0, 64], dtype=np.int32)]
        x_blocks = [np.array([0, 64], dtype=np.int32)]

        taper, _, _ = compute_nonrigid_reference_data(
            reference_image=reference,
            taper_slope=5.0,
            smoothing_sigma=1.15,
            y_blocks=y_blocks,
            x_blocks=x_blocks,
        )

        assert np.all(taper >= 0.0)
        assert np.all(taper <= 1.0)


class TestComputeNonrigidOffsets:
    """Tests compute_nonrigid_offsets."""

    def test_consistent_offsets_for_identical_frames(self) -> None:
        """Verifies consistent offsets and correct shapes when frames match the reference."""
        rng = np.random.default_rng(42)
        reference = rng.standard_normal((64, 64)).astype(np.float32)
        y_blocks, x_blocks, _block_counts, _, smoothing_kernel = compute_registration_blocks(
            height=64, width=64, block_size=(32, 32)
        )

        taper, offset, kernel = compute_nonrigid_reference_data(
            reference_image=reference,
            taper_slope=5.0,
            smoothing_sigma=1.15,
            y_blocks=y_blocks,
            x_blocks=x_blocks,
        )

        frames = np.tile(reference, (2, 1, 1))
        y_offsets, x_offsets, _correlation = compute_nonrigid_offsets(
            frames=frames,
            taper_mask=taper,
            mean_offset=offset,
            reference_kernel=kernel,
            snr_threshold=1.2,
            smoothing_kernel=smoothing_kernel,
            x_blocks=x_blocks,
            y_blocks=y_blocks,
            maximum_offset=5.0,
            workers=1,
        )

        block_count = len(y_blocks)
        assert y_offsets.shape == (2, block_count)
        assert x_offsets.shape == (2, block_count)
        # Identical frames should produce consistent offsets across frames.
        np.testing.assert_allclose(y_offsets[0], y_offsets[1], atol=1e-4)
        np.testing.assert_allclose(x_offsets[0], x_offsets[1], atol=1e-4)
        # Offsets should be small (within 1 pixel) for identical frames.
        assert np.max(np.abs(y_offsets)) < 1.0
        assert np.max(np.abs(x_offsets)) < 1.0

    def test_output_dtypes(self) -> None:
        """Verifies the output dtypes are correct."""
        rng = np.random.default_rng(42)
        reference = rng.standard_normal((64, 64)).astype(np.float32)
        y_blocks, x_blocks, _, _, smoothing_kernel = compute_registration_blocks(
            height=64, width=64, block_size=(32, 32)
        )

        taper, offset, kernel = compute_nonrigid_reference_data(
            reference_image=reference,
            taper_slope=5.0,
            smoothing_sigma=1.15,
            y_blocks=y_blocks,
            x_blocks=x_blocks,
        )

        frames = np.tile(reference, (1, 1, 1))
        y_offsets, x_offsets, correlation = compute_nonrigid_offsets(
            frames=frames,
            taper_mask=taper,
            mean_offset=offset,
            reference_kernel=kernel,
            snr_threshold=1.2,
            smoothing_kernel=smoothing_kernel,
            x_blocks=x_blocks,
            y_blocks=y_blocks,
            maximum_offset=5.0,
            workers=1,
        )

        assert y_offsets.dtype == np.float32
        assert x_offsets.dtype == np.float32
        assert correlation.dtype == np.float32


class TestApplyNonrigidCorrection:
    """Tests apply_nonrigid_correction."""

    def test_zero_offsets_preserve_frames(self) -> None:
        """Verifies that zero offsets preserve the original frames."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((2, 64, 64)).astype(np.float32)
        y_blocks, x_blocks, block_counts, _, _ = compute_registration_blocks(height=64, width=64, block_size=(32, 32))
        block_count = len(y_blocks)
        y_offsets = np.zeros((2, block_count), dtype=np.float32)
        x_offsets = np.zeros((2, block_count), dtype=np.float32)

        result = apply_nonrigid_correction(
            frames=frames,
            block_counts=block_counts,
            x_blocks=x_blocks,
            y_blocks=y_blocks,
            y_block_offsets=y_offsets,
            x_block_offsets=x_offsets,
        )

        assert result.shape == frames.shape
        np.testing.assert_allclose(result, frames, atol=1e-4)

    def test_output_shape_and_dtype(self) -> None:
        """Verifies the output shape and dtype match the input."""
        frames = np.ones((3, 64, 64), dtype=np.float32)
        y_blocks, x_blocks, block_counts, _, _ = compute_registration_blocks(height=64, width=64, block_size=(32, 32))
        block_count = len(y_blocks)
        y_offsets = np.ones((3, block_count), dtype=np.float32) * 0.5
        x_offsets = np.ones((3, block_count), dtype=np.float32) * 0.5

        result = apply_nonrigid_correction(
            frames=frames,
            block_counts=block_counts,
            x_blocks=x_blocks,
            y_blocks=y_blocks,
            y_block_offsets=y_offsets,
            x_block_offsets=x_offsets,
        )

        assert result.shape == (3, 64, 64)
        assert result.dtype == np.float32


class TestUpsampleBlockOffsets:
    """Tests _upsample_block_offsets."""

    def test_output_shape(self) -> None:
        """Verifies the output offset maps have the correct shape."""
        y_blocks, x_blocks, block_counts, _, _ = compute_registration_blocks(height=64, width=64, block_size=(32, 32))
        block_count = len(y_blocks)
        y_offsets = np.ones((2, block_count), dtype=np.float32)
        x_offsets = np.ones((2, block_count), dtype=np.float32)

        y_maps, x_maps = _upsample_block_offsets(
            width=64,
            height=64,
            block_counts=block_counts,
            x_blocks=x_blocks,
            y_blocks=y_blocks,
            y_block_offsets=y_offsets,
            x_block_offsets=x_offsets,
        )

        assert y_maps.shape == (2, 64, 64)
        assert x_maps.shape == (2, 64, 64)

    def test_uniform_offsets_preserved(self) -> None:
        """Verifies that uniform block offsets produce uniform pixel offset maps."""
        y_blocks, x_blocks, block_counts, _, _ = compute_registration_blocks(height=64, width=64, block_size=(32, 32))
        block_count = len(y_blocks)
        y_offsets = np.ones((1, block_count), dtype=np.float32) * 2.5
        x_offsets = np.ones((1, block_count), dtype=np.float32) * -1.5

        y_maps, x_maps = _upsample_block_offsets(
            width=64,
            height=64,
            block_counts=block_counts,
            x_blocks=x_blocks,
            y_blocks=y_blocks,
            y_block_offsets=y_offsets,
            x_block_offsets=x_offsets,
        )

        np.testing.assert_allclose(y_maps[0], 2.5, atol=0.1)
        np.testing.assert_allclose(x_maps[0], -1.5, atol=0.1)
