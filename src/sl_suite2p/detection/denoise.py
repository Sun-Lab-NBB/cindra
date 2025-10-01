"""This module provides assets for denoising binned fluorescence movies for further cell activity extraction."""

import numpy as np
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
from sklearn.decomposition import PCA
from ataraxis_base_utilities import LogLevel, console

from ..registration.nonrigid import make_blocks, spatial_taper


def block_pca_denoise(
    movie: NDArray[np.float32], block_height: int, block_width: int, dimensionality_reduction_strength: float
) -> NDArray[np.float32]:
    """Denoises the input fluorescence movie using block-wise Principal Component Analysis (PCA).

    Args:
        movie: Input movie with shape (n_frames, movie_height, movie_width).
        block_height: The height of the spatial blocks into which to split the movie before passing it through the PCA.
        block_width: The width of the spatial blocks into which to split the movie before passing it through the PCA.
        dimensionality_reduction_strength: Determines the strength of PCA dimensionality reduction based on the
            smallest processed block dimension. The larger the value, the more PCA components are used, leading to lower
            dimensionality reduction.

    Returns:
        The denoised fluorescence movie.
    """
    # Initializes a timer to measure the function's execution time.
    timer = PrecisionTimer("ms")
    timer.reset()

    # Normalizes the fluorescence to support
    n_frames, movie_height, movie_width = movie.shape
    movie_mean = movie.mean(axis=0)
    mean_subtracted_movie = movie - movie_mean

    # Define block indicies
    y_block, x_block, _, block_size, _ = make_blocks(movie_height, movie_width, block_size=(block_height, block_width))
    n_blocks = len(y_block)
    block_height, block_width = block_size

    # Determine the number of PCA components
    pca_components = int(
        min(
            min(block_height * block_width, n_frames),
            min(block_height, block_width) * dimensionality_reduction_strength,
        )
    )

    taper_mask = spatial_taper(block_height // 4, block_height, block_width)
    norm_map = np.zeros((movie_height, movie_width), np.float32)
    reconstruction = np.zeros_like(mean_subtracted_movie)

    # Processes each block using PCA
    block_reconstruction = np.zeros((n_blocks, n_frames, block_height * block_width))
    for block_idx in range(n_blocks):
        x_block_start, x_block_end = x_block[block_idx][0], x_block[block_idx][-1]
        y_block_start, y_block_end = y_block[block_idx][0], y_block[block_idx][-1]

        flattened_block = mean_subtracted_movie[:, y_block_start:y_block_end, x_block_start:x_block_end].reshape(
            -1, block_height * block_width
        )

        model = PCA(n_components=pca_components, random_state=0).fit(flattened_block)
        block_reconstruction[block_idx] = (flattened_block @ model.components_.T) @ model.components_
        norm_map[y_block_start:y_block_end, x_block_start:x_block_end] += taper_mask

    # Apply taper mask
    block_reconstruction = block_reconstruction.reshape(n_blocks, n_frames, block_height, block_width)
    block_reconstruction *= taper_mask

    # Combine all the blocks after block-wise PCA
    for block_idx in range(n_blocks):
        x_block_start, x_block_end = x_block[block_idx][0], x_block[block_idx][-1]
        y_block_start, y_block_end = y_block[block_idx][0], y_block[block_idx][-1]

        reconstruction[:, y_block_start:y_block_end, x_block_start:x_block_end] += block_reconstruction[block_idx]

    # Normalize the reconstruction
    reconstruction /= norm_map

    # Add back mean image
    denoised_movie = reconstruction + movie_mean

    elapsed_time = timer.elapsed
    console.echo(
        message=f"Binned movie denoising: Complete. Time taken {elapsed_time / 1000:.2f} seconds",
        level=LogLevel.SUCCESS,
    )

    return denoised_movie
