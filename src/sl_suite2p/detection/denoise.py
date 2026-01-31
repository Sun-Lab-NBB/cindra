"""Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu."""

import time

import numpy as np
from sklearn.decomposition import PCA
from ataraxis_base_utilities import LogLevel, console

from ..registration import compute_spatial_taper_mask, compute_registration_blocks


def pca_denoise(mov: np.ndarray, block_size: list, n_comps_frac: float):
    t0 = time.time()
    nframes, Ly, Lx = mov.shape
    yblock, xblock, _, block_size, _ = compute_registration_blocks(height=Ly, width=Lx, block_size=tuple(block_size))

    mov_mean = mov.mean(axis=0)
    mov -= mov_mean

    nblocks = len(yblock)
    Lyb, Lxb = block_size
    n_comps = int(min(min(Lyb * Lxb, nframes), min(Lyb, Lxb) * n_comps_frac))
    maskMul = compute_spatial_taper_mask(sigma=Lyb // 4, height=Lyb, width=Lxb)
    norm = np.zeros((Ly, Lx), np.float32)
    reconstruction = np.zeros_like(mov)
    block_re = np.zeros((nblocks, nframes, Lyb * Lxb))
    for i in range(nblocks):
        block = mov[:, yblock[i][0] : yblock[i][-1], xblock[i][0] : xblock[i][-1]].reshape(-1, Lyb * Lxb)
        model = PCA(n_components=n_comps, random_state=0).fit(block)
        block_re[i] = (block @ model.components_.T) @ model.components_
        norm[yblock[i][0] : yblock[i][-1], xblock[i][0] : xblock[i][-1]] += maskMul

    block_re = block_re.reshape(nblocks, nframes, Lyb, Lxb)
    block_re *= maskMul
    for i in range(nblocks):
        reconstruction[:, yblock[i][0] : yblock[i][-1], xblock[i][0] : xblock[i][-1]] += block_re[i]
    reconstruction /= norm
    console.echo(
        message=f"PCA denoising of binned movie (for cell detection): complete. Time taken: {time.time() - t0:.2f} seconds.",
        level=LogLevel.SUCCESS,
    )
    reconstruction += mov_mean
    return reconstruction
