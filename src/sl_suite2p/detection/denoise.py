"""Provides PCA-based denoising algorithm applied to the recording frames before ROI detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
) -> None:
    """Applies PCA-based denoising to movie frames in-place using overlapping spatial blocks.

    Notes:
        The movie is divided into overlapping blocks, and PCA is applied to each block independently. The denoised
        blocks are then blended together using a taper mask to ensure smooth transitions between adjacent blocks.
        This approach reduces noise while preserving spatially localized signals.

    Args:
        frames: The input movie array with shape (num_frames, height, width). Modified in-place.
        block_size: The spatial dimensions (height, width) of each processing block.
        component_fraction: The fraction of PCA components to retain, relative to the smaller block dimension.
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

    # Applies PCA denoising to each block and accumulates the tapered result.
    for block_index in range(len(y_blocks)):
        y_slice = slice(y_blocks[block_index][0], y_blocks[block_index][-1])
        x_slice = slice(x_blocks[block_index][0], x_blocks[block_index][-1])

        # Extracts and centers the block for PCA.
        block = frames[:, y_slice, x_slice].reshape(num_frames, -1) - frame_mean[y_slice, x_slice].ravel()
        model = PCA(n_components=num_components, random_state=0).fit(block)

        # Reconstructs, reshapes, tapers, and accumulates in a single step.
        # noinspection PyUnresolvedReferences
        block_recon = ((block @ model.components_.T) @ model.components_).reshape(num_frames, block_height, block_width)
        reconstruction[:, y_slice, x_slice] += block_recon * taper_mask
        normalization[y_slice, x_slice] += taper_mask

    # Normalizes and restores the mean.
    reconstruction /= normalization
    reconstruction += frame_mean

    # Copies result to input array for in-place semantics.
    frames[:] = reconstruction

    message = f"PCA denoising of binned movie: complete. Time taken: {timer.elapsed} seconds."
    console.echo(message=message, level=LogLevel.SUCCESS)
