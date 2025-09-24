"""Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu."""

import time

import numpy as np
from sklearn.decomposition import PCA

from ..registration.nonrigid import make_blocks, spatial_taper


def block_pca_denoise(movie: np.ndarray, block_size: tuple[int, int], n_comps_frac: float) -> np.ndarray:
    """
    Denoises a movie using block-wise PCA.
    
    Args:
        movie: Input movie with shape (n_frames, movie_height, movie_width).
        block_size: Size of spatial blocks (block_height, block_width).
        n_comps_frac: The fraction of block size to help determine number of PCA compopnents.

    Returns:
        The PCA denoised movie with the same size as the the input
    """

    start_time = time.time()
    n_frames, movie_height, movie_width = movie.shape

    # Center the data
    movie_mean = movie.mean(axis=0)
    movie_centered = movie -  movie_mean

    # Define block indicies
    y_block, x_block, _, block_size, _ = make_blocks(movie_height, movie_width, block_size=block_size)
    n_blocks = len(y_block)
    block_height, block_width = block_size

    # Determine the number of PCA components
    n_comps = int(min(min(block_height * block_width, n_frames), min(block_height, block_width) * n_comps_frac))

    taper_mask = spatial_taper(block_height // 4, block_height, block_width)
    norm_map = np.zeros((movie_height, movie_width), np.float32)
    reconstruction = np.zeros_like(movie_centered)

    # Block-wise PCA
    block_reconstruction = np.zeros((n_blocks, n_frames, block_height * block_width))
    for block_idx in range(n_blocks):
        x_block_start, x_block_end = x_block[block_idx][0], x_block[block_idx][-1]
        y_block_start, y_block_end = y_block[block_idx][0],  y_block[block_idx][-1]

        flattened_block = movie_centered[:, y_block_start : y_block_end, 
                                         x_block_start : x_block_end].reshape(-1, block_height * block_width)
        
        model = PCA(n_components=n_comps, random_state=0).fit(flattened_block)
        block_reconstruction[block_idx] = (flattened_block @ model.components_.T) @ model.components_
        norm_map[y_block_start : y_block_end, x_block_start : x_block_end] += taper_mask

    # Apply taper mask
    block_reconstruction = block_reconstruction.reshape(n_blocks, n_frames, block_height, block_width)
    block_reconstruction *= taper_mask

    # Combine all the blocks after block-wise PCA
    for block_idx in range(n_blocks):
        x_block_start, x_block_end = x_block[block_idx][0], x_block[block_idx][-1]
        y_block_start, y_block_end = y_block[block_idx][0],  y_block[block_idx][-1]

        reconstruction[:, y_block_start : y_block_end, 
                       x_block_start : x_block_end] += block_reconstruction[block_idx]

    # Normalize the reconstruction
    reconstruction /= norm_map
    # Add back mean image
    denoised_movie = reconstruction + movie_mean

    elapsed_time = time.time() - start_time
    print(f"Binned movie denoised (for cell detection only) in {elapsed_time:.2f} sec.")

    return denoised_movie
