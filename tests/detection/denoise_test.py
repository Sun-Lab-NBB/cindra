"""Contains tests for the denoise module."""

from __future__ import annotations

import numpy as np
import pytest

from cindra.detection.denoise import pca_denoise, _fit_and_reconstruct_block


class TestFitAndReconstructBlock:
    """Tests _fit_and_reconstruct_block."""

    def test_output_shape(self) -> None:
        """Verifies that the reconstructed block has the same shape as the input."""
        generator = np.random.default_rng(seed=42)
        block = generator.standard_normal((50, 100)).astype(np.float32)
        result = _fit_and_reconstruct_block(block=block, component_count=5)
        assert result.shape == block.shape

    def test_output_dtype(self) -> None:
        """Verifies that the output dtype is float32."""
        generator = np.random.default_rng(seed=42)
        block = generator.standard_normal((50, 100)).astype(np.float32)
        result = _fit_and_reconstruct_block(block=block, component_count=5)
        assert result.dtype == np.float32

    def test_low_rank_reconstruction(self) -> None:
        """Verifies that a low-rank input is perfectly reconstructed when enough components are retained."""
        generator = np.random.default_rng(seed=42)
        # Creates a rank-3 matrix.
        left_factor = generator.standard_normal((50, 3)).astype(np.float32)
        right_factor = generator.standard_normal((3, 100)).astype(np.float32)
        block = (left_factor @ right_factor).astype(np.float32)
        result = _fit_and_reconstruct_block(block=block, component_count=3)
        np.testing.assert_allclose(result, block, atol=1e-3)

    def test_reduces_noise(self) -> None:
        """Verifies that reconstruction with fewer components reduces noise energy."""
        generator = np.random.default_rng(seed=42)
        signal = generator.standard_normal((50, 3)).astype(np.float32) @ generator.standard_normal((3, 100)).astype(
            np.float32
        )
        noise = generator.standard_normal((50, 100)).astype(np.float32) * 0.1
        block = (signal + noise).astype(np.float32)
        result = _fit_and_reconstruct_block(block=block, component_count=3)
        error_before = np.mean((block - signal) ** 2)
        error_after = np.mean((result - signal) ** 2)
        assert error_after < error_before

    def test_single_component(self) -> None:
        """Verifies that a single component produces a rank-1 reconstruction."""
        generator = np.random.default_rng(seed=42)
        block = generator.standard_normal((30, 50)).astype(np.float32)
        result = _fit_and_reconstruct_block(block=block, component_count=1)
        # A rank-1 matrix has at most 1 non-zero singular value.
        singular_values = np.linalg.svd(result, compute_uv=False)
        np.testing.assert_allclose(singular_values[1:], 0, atol=1e-4)


class TestPcaDenoise:
    """Tests pca_denoise."""

    def test_in_place_modification(self) -> None:
        """Verifies that pca_denoise modifies frames in-place."""
        generator = np.random.default_rng(seed=42)
        frames = generator.standard_normal((20, 32, 32)).astype(np.float32)
        original = frames.copy()
        pca_denoise(frames=frames, block_size=(32, 32), component_fraction=0.5)
        assert not np.array_equal(frames, original)

    def test_output_shape_preserved(self) -> None:
        """Verifies that the output shape matches the input shape."""
        generator = np.random.default_rng(seed=42)
        frames = generator.standard_normal((20, 32, 32)).astype(np.float32)
        shape_before = frames.shape
        pca_denoise(frames=frames, block_size=(32, 32), component_fraction=0.5)
        assert frames.shape == shape_before

    def test_output_finite(self) -> None:
        """Verifies that the denoised frames contain only finite values."""
        generator = np.random.default_rng(seed=42)
        frames = generator.standard_normal((20, 32, 32)).astype(np.float32)
        pca_denoise(frames=frames, block_size=(32, 32), component_fraction=0.5)
        assert np.isfinite(frames).all()

    def test_uniform_frames_preserved(self) -> None:
        """Verifies that uniform frames remain approximately uniform after denoising."""
        frames = np.ones((20, 32, 32), dtype=np.float32) * 5.0
        pca_denoise(frames=frames, block_size=(32, 32), component_fraction=0.5)
        np.testing.assert_allclose(frames, 5.0, atol=1e-4)

    def test_noiseless_low_rank_movie_survives_the_block_blend(self) -> None:
        """Verifies that a movie whose rank the component budget covers is returned unchanged by the blend."""
        frame_count = 20
        height = 48
        width = 48
        rows, columns = np.mgrid[0:height, 0:width]

        # Every frame is the same three smooth spatial patterns at frame-specific amplitudes, so the movie has rank
        # 3 over its pixels and so does every block of it. The block budget here is int(16 * 0.5) = 8 components,
        # which covers that rank, and PCA projects a matrix onto a basis spanning its own rows without loss.
        patterns = np.stack(
            (
                np.sin(2 * np.pi * rows / height),
                np.cos(2 * np.pi * columns / width),
                np.sin(2 * np.pi * (rows + columns) / (height + width)),
            )
        ).astype(np.float32)
        generator = np.random.default_rng(seed=3)
        amplitudes = generator.standard_normal((frame_count, 3)).astype(np.float32)
        movie = np.tensordot(amplitudes, patterns, axes=(1, 0)).astype(np.float32)
        assert np.linalg.matrix_rank(movie.reshape(frame_count, -1)) == 3

        denoised = movie.copy()
        pca_denoise(frames=denoised, block_size=(16, 16), component_fraction=0.5)

        # The 16x16 blocks overlap, so most pixels accumulate a taper-weighted sum over several reconstructions.
        # Recovering the input to six digits requires the taper to be accumulated into the normalizer exactly as it
        # is into the reconstruction, and requires the running total to be divided by that normalizer.
        np.testing.assert_allclose(denoised, movie, atol=1e-4)

    def test_parallel_workers(self) -> None:
        """Verifies that parallel execution produces finite results."""
        generator = np.random.default_rng(seed=42)
        frames = generator.standard_normal((20, 32, 32)).astype(np.float32)
        pca_denoise(frames=frames, block_size=(32, 32), component_fraction=0.5, parallel_workers=4)
        assert np.isfinite(frames).all()

    @pytest.mark.parametrize("parallel_workers", [0, -1, -2, -100])
    def test_invalid_worker_count_is_rejected(self, parallel_workers: int) -> None:
        """Verifies that every non-positive worker count raises an error."""
        frames = np.ones((20, 32, 32), dtype=np.float32)
        with pytest.raises(ValueError, match=r"must be a positive\s+integer"):
            pca_denoise(frames=frames, block_size=(32, 32), component_fraction=0.5, parallel_workers=parallel_workers)

    @pytest.mark.parametrize("parallel_workers", [2, 4, 8])
    def test_sequential_and_parallel_consistent(self, parallel_workers: int) -> None:
        """Verifies that sequential and parallel execution produce bit-identical results.

        Blocks overlap, so most pixels accumulate a float32 sum over several of them and float addition is not
        associative. Consuming the block futures in completion order rather than submission order therefore
        perturbs the low bits of the denoised movie, which this assertion catches.
        """
        generator = np.random.default_rng(seed=42)
        frames_sequential = generator.standard_normal((20, 32, 32)).astype(np.float32)
        frames_parallel = frames_sequential.copy()
        pca_denoise(frames=frames_sequential, block_size=(16, 16), component_fraction=0.5, parallel_workers=1)
        pca_denoise(
            frames=frames_parallel,
            block_size=(16, 16),
            component_fraction=0.5,
            parallel_workers=parallel_workers,
        )
        np.testing.assert_array_equal(frames_sequential, frames_parallel)
