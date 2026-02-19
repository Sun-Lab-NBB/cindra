"""Provides PCA-based denoising algorithm applied to the recording frames before ROI detection."""

from __future__ import annotations

from typing import TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ataraxis_time import PrecisionTimer, TimerPrecisions
from sklearn.decomposition import PCA
from ataraxis_base_utilities import LogLevel, console

from ..registration import compute_spatial_taper_mask, compute_registration_blocks

if TYPE_CHECKING:
    from numpy.typing import NDArray


def pca_denoise(
    frames: NDArray[np.float32],
    block_size: tuple[int, int],
    component_fraction: float,
    parallel_workers: int = 1,
) -> None:
    """Applies PCA-based denoising to movie frames in-place using overlapping spatial blocks.

    Notes:
        The movie is divided into overlapping blocks, and PCA is applied to each block independently. The denoised
        blocks are then blended together using a taper mask to ensure smooth transitions between adjacent blocks.
        This approach reduces noise while preserving spatially localized signals. When parallel_workers is greater
        than 1, PCA fitting runs concurrently across blocks using a thread pool. LAPACK's SVD implementation used by
        sklearn releases the GIL, making threading effective for this workload. The subsequent accumulation step
        remains sequential to avoid write conflicts on overlapping block regions.

    Args:
        frames: The input movie array with shape (num_frames, height, width). Modified in-place.
        block_size: The spatial dimensions (height, width) of each processing block.
        component_fraction: The fraction of PCA components to retain, relative to the smaller block dimension.
        parallel_workers: The number of parallel threads for PCA fitting. Values of -1 or 0 use all available cores.
            Defaults to 1 (sequential).
    """
    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)

    num_frames, height, width = frames.shape
    y_blocks, x_blocks, _, (block_height, block_width), _ = compute_registration_blocks(
        height=height, width=width, block_size=block_size
    )

    frame_mean = frames.mean(axis=0)
    max_components = int(min(block_height, block_width) * component_fraction)
    num_components = min(block_height * block_width, num_frames, max_components)
    taper_mask = compute_spatial_taper_mask(sigma=block_height // 4, height=block_height, width=block_width)

    normalization = np.zeros((height, width), dtype=np.float32)
    reconstruction = np.zeros_like(frames)

    # Resolves the effective worker count. Values <= 0 mean unlimited (all available cores), which
    # ThreadPoolExecutor interprets as None.
    effective_workers: int | None = None if parallel_workers <= 0 else parallel_workers

    # Extracts and centers each block for PCA.
    block_slices: list[tuple[slice, slice]] = []
    centered_blocks: list[NDArray[np.float32]] = []
    for y_block, x_block in zip(y_blocks, x_blocks, strict=True):
        y_slice = slice(y_block[0], y_block[-1])
        x_slice = slice(x_block[0], x_block[-1])
        block_slices.append((y_slice, x_slice))
        centered_blocks.append(
            frames[:, y_slice, x_slice].reshape(num_frames, -1) - frame_mean[y_slice, x_slice].ravel()
        )

    # Fits PCA and reconstructs each block. When multiple workers are available, the fitting runs in parallel across
    # blocks since each block's SVD is independent. LAPACK releases the GIL during SVD computation.
    if effective_workers is not None and effective_workers <= 1:
        reconstructed_blocks = [
            _fit_and_reconstruct_block(block=block, num_components=num_components) for block in centered_blocks
        ]
    else:
        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [
                executor.submit(_fit_and_reconstruct_block, block=block, num_components=num_components)
                for block in centered_blocks
            ]
            reconstructed_blocks = [future.result() for future in futures]

    # Accumulates the tapered reconstructions sequentially to avoid write conflicts on overlapping block regions.
    # noinspection PyUnboundLocalVariable
    for (y_slice, x_slice), block_recon in zip(block_slices, reconstructed_blocks, strict=True):
        reconstruction[:, y_slice, x_slice] += block_recon.reshape(num_frames, block_height, block_width) * taper_mask
        normalization[y_slice, x_slice] += taper_mask

    # Normalizes and restores the mean.
    reconstruction /= normalization
    reconstruction += frame_mean

    # Copies result to input array for in-place semantics.
    frames[:] = reconstruction

    message = f"PCA denoising of binned movie: complete. Time taken: {timer.elapsed} seconds."
    console.echo(message=message, level=LogLevel.SUCCESS)


def _fit_and_reconstruct_block(
    block: NDArray[np.float32],
    num_components: int,
) -> NDArray[np.float32]:
    """Fits a PCA model to a single spatial block and returns the low-rank reconstruction.

    Args:
        block: The centered block data with shape (num_frames, num_pixels).
        num_components: The number of PCA components to retain.

    Returns:
        The reconstructed block data with shape (num_frames, num_pixels).
    """
    model = PCA(n_components=num_components, random_state=0).fit(block)
    # noinspection PyUnresolvedReferences
    reconstructed: NDArray[np.float32] = ((block @ model.components_.T) @ model.components_).astype(np.float32)
    return reconstructed
