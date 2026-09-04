"""Provides the PCA-based denoising algorithm applied to the recording frames before ROI detection."""

from __future__ import annotations

from typing import TYPE_CHECKING
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from ataraxis_time import PrecisionTimer, TimerPrecisions
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from ataraxis_base_utilities import LogLevel, console

from .utils import compute_spatial_taper_mask, compute_registration_blocks

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
        This approach reduces noise while preserving spatially localized signals. Whenever more than one worker is
        available, PCA fitting runs concurrently across blocks using a thread pool. LAPACK's SVD implementation used
        by sklearn releases the GIL, making threading effective for this workload. The subsequent accumulation step
        remains sequential to avoid write conflicts on overlapping block regions.

    Args:
        frames: The input movie array with shape (num_frames, height, width). Modified in-place.
        block_size: The spatial dimensions (height, width) of each processing block.
        component_fraction: The fraction of PCA components to retain, relative to the smaller block dimension.
        parallel_workers: The number of parallel threads for PCA fitting. Must be a positive integer, which the
            caller resolves before invoking this function. Defaults to 1 (sequential).

    Raises:
        ValueError: If parallel_workers is not a positive integer.
    """
    if parallel_workers <= 0:
        message = (
            f"Unable to apply PCA denoising. The requested parallel worker count must be a positive integer, but "
            f"encountered {parallel_workers}."
        )
        console.error(message=message, error=ValueError)

    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    frame_count, height, width = frames.shape
    y_blocks, x_blocks, _, (block_height, block_width), _ = compute_registration_blocks(
        height=height, width=width, block_size=block_size
    )

    frame_mean = frames.mean(axis=0)
    maximum_components = int(min(block_height, block_width) * component_fraction)
    component_count = min(block_height * block_width, frame_count, maximum_components)
    taper_mask = compute_spatial_taper_mask(sigma=block_height // 4, height=block_height, width=block_width)

    normalization = np.zeros((height, width), dtype=np.float32)
    reconstruction = np.zeros_like(frames)

    block_slices: list[tuple[slice, slice]] = [
        (slice(y_block[0], y_block[-1]), slice(x_block[0], x_block[-1]))
        for y_block, x_block in zip(y_blocks, x_blocks, strict=True)
    ]

    def _center_and_reconstruct(block_slice: tuple[slice, slice]) -> NDArray[np.float32]:
        """Centers one block against the frame mean and returns its low-rank reconstruction."""
        y_slice, x_slice = block_slice
        centered = frames[:, y_slice, x_slice].reshape(frame_count, -1) - frame_mean[y_slice, x_slice].ravel()
        return _fit_and_reconstruct_block(block=centered, component_count=component_count)

    def _accumulate(block_slice: tuple[slice, slice], block_reconstruction: NDArray[np.float32]) -> None:
        """Adds one tapered block reconstruction into the running totals."""
        y_slice, x_slice = block_slice
        reconstruction[:, y_slice, x_slice] += (
            block_reconstruction.reshape(frame_count, block_height, block_width) * taper_mask
        )
        normalization[y_slice, x_slice] += taper_mask

    # Limits each block fit to a single BLAS thread. The worker budget is already spent on the block pool below, so
    # leaving the BLAS thread count unconstrained would multiply the two and oversubscribe the host. The limit also
    # encloses the accumulation, because the BLAS width the fits run at decides their summation order. Each block is
    # centered inside the worker that fits it, accumulated in submission order, and released once accumulated,
    # so the resident set holds the blocks still in flight rather than a reconstruction of every block.
    with threadpool_limits(limits=1):
        if parallel_workers == 1:
            for block_slice in block_slices:
                _accumulate(block_slice=block_slice, block_reconstruction=_center_and_reconstruct(block_slice))
        else:
            with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
                pending = deque(executor.submit(_center_and_reconstruct, block_slice) for block_slice in block_slices)

                # Consumes the futures in submission order rather than completion order. Blocks overlap, so most
                # pixels accumulate a float32 sum over several of them, and float addition is not associative.
                # Each future leaves the queue as its block is accumulated, because a future holds its result
                # alive after it is read, which would otherwise keep every reconstruction resident to the end.
                for block_slice in block_slices:
                    _accumulate(block_slice=block_slice, block_reconstruction=pending.popleft().result())

    reconstruction /= normalization
    reconstruction += frame_mean

    frames[:] = reconstruction

    message = f"PCA denoising of binned movie: complete. Time taken: {timer.elapsed} seconds."
    console.echo(message=message, level=LogLevel.SUCCESS)


def _fit_and_reconstruct_block(
    block: NDArray[np.float32],
    component_count: int,
) -> NDArray[np.float32]:
    """Fits a PCA model to a single spatial block and returns the low-rank reconstruction.

    Args:
        block: The centered block data with shape (num_frames, num_pixels).
        component_count: The number of PCA components to retain.

    Returns:
        The reconstructed block data with shape (num_frames, num_pixels).
    """
    # Uniform blocks have zero variance, making PCA undefined. Returns the block unchanged to avoid a
    # division-by-zero warning inside sklearn.
    if np.ptp(block) == 0.0:
        return block.copy()

    # A float32 block yields float32 components, so the projection and its back-projection stay float32 throughout.
    model = PCA(n_components=component_count, random_state=0).fit(block)
    reconstructed: NDArray[np.float32] = (block @ model.components_.T) @ model.components_
    return reconstructed
