"""Contains tests for the denoise module."""

from __future__ import annotations

import numpy as np

from cindra.detection.denoise import pca_denoise, _fit_and_reconstruct_block


class TestFitAndReconstructBlock:
    """Tests _fit_and_reconstruct_block."""

    def test_output_shape(self) -> None:
        """Verifies that the reconstructed block has the same shape as the input."""
        rng = np.random.default_rng(42)
        block = rng.standard_normal((50, 100)).astype(np.float32)
        result = _fit_and_reconstruct_block(block=block, num_components=5)
        assert result.shape == block.shape

    def test_output_dtype(self) -> None:
        """Verifies that the output dtype is float32."""
        rng = np.random.default_rng(42)
        block = rng.standard_normal((50, 100)).astype(np.float32)
        result = _fit_and_reconstruct_block(block=block, num_components=5)
        assert result.dtype == np.float32

    def test_low_rank_reconstruction(self) -> None:
        """Verifies that a low-rank input is perfectly reconstructed when enough components are retained."""
        rng = np.random.default_rng(42)
        # Creates a rank-3 matrix.
        left_factor = rng.standard_normal((50, 3)).astype(np.float32)
        right_factor = rng.standard_normal((3, 100)).astype(np.float32)
        block = (left_factor @ right_factor).astype(np.float32)
        result = _fit_and_reconstruct_block(block=block, num_components=3)
        np.testing.assert_allclose(result, block, atol=1e-3)

    def test_reduces_noise(self) -> None:
        """Verifies that reconstruction with fewer components reduces noise energy."""
        rng = np.random.default_rng(42)
        signal = rng.standard_normal((50, 3)).astype(np.float32) @ rng.standard_normal((3, 100)).astype(np.float32)
        noise = rng.standard_normal((50, 100)).astype(np.float32) * 0.1
        block = (signal + noise).astype(np.float32)
        result = _fit_and_reconstruct_block(block=block, num_components=3)
        # The reconstruction should be closer to the signal than the noisy input.
        error_before = np.mean((block - signal) ** 2)
        error_after = np.mean((result - signal) ** 2)
        assert error_after < error_before

    def test_single_component(self) -> None:
        """Verifies that a single component produces a rank-1 reconstruction."""
        rng = np.random.default_rng(42)
        block = rng.standard_normal((30, 50)).astype(np.float32)
        result = _fit_and_reconstruct_block(block=block, num_components=1)
        # A rank-1 matrix has at most 1 non-zero singular value.
        singular_values = np.linalg.svd(result, compute_uv=False)
        # All singular values beyond the first should be near zero.
        np.testing.assert_allclose(singular_values[1:], 0, atol=1e-4)


class TestPcaDenoise:
    """Tests pca_denoise."""

    def test_in_place_modification(self) -> None:
        """Verifies that pca_denoise modifies frames in-place."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((20, 32, 32)).astype(np.float32)
        original = frames.copy()
        pca_denoise(frames=frames, block_size=(32, 32), component_fraction=0.5)
        assert not np.array_equal(frames, original)

    def test_output_shape_preserved(self) -> None:
        """Verifies that the output shape matches the input shape."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((20, 32, 32)).astype(np.float32)
        shape_before = frames.shape
        pca_denoise(frames=frames, block_size=(32, 32), component_fraction=0.5)
        assert frames.shape == shape_before

    def test_output_finite(self) -> None:
        """Verifies that the denoised frames contain only finite values."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((20, 32, 32)).astype(np.float32)
        pca_denoise(frames=frames, block_size=(32, 32), component_fraction=0.5)
        assert np.isfinite(frames).all()

    def test_uniform_frames_preserved(self) -> None:
        """Verifies that uniform frames remain approximately uniform after denoising."""
        frames = np.ones((20, 32, 32), dtype=np.float32) * 5.0
        pca_denoise(frames=frames, block_size=(32, 32), component_fraction=0.5)
        np.testing.assert_allclose(frames, 5.0, atol=1e-4)

    def test_parallel_workers(self) -> None:
        """Verifies that parallel execution produces finite results."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((20, 32, 32)).astype(np.float32)
        pca_denoise(frames=frames, block_size=(32, 32), component_fraction=0.5, parallel_workers=-1)
        assert np.isfinite(frames).all()

    def test_sequential_and_parallel_consistent(self) -> None:
        """Verifies that sequential and parallel execution produce identical results."""
        rng = np.random.default_rng(42)
        frames_sequential = rng.standard_normal((20, 32, 32)).astype(np.float32)
        frames_parallel = frames_sequential.copy()
        pca_denoise(frames=frames_sequential, block_size=(32, 32), component_fraction=0.5, parallel_workers=1)
        pca_denoise(frames=frames_parallel, block_size=(32, 32), component_fraction=0.5, parallel_workers=2)
        np.testing.assert_allclose(frames_sequential, frames_parallel, atol=1e-4)
