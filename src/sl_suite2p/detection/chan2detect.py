"""Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu."""

import numpy as np
from scipy.ndimage import gaussian_filter

from ..extraction import masks

"""
identify cells with channel 2 brightness (aka red cells)

main function is detect
takes from ops: "mean_image", "mean_image_channel_2", "Ly", "Lx"
takes from stat: "ypix", "xpix", "lam"
"""


def quadrant_mask(Ly, Lx, ny, nx, sT):
    mask = np.zeros((Ly, Lx), np.float32)
    mask[np.ix_(ny, nx)] = 1
    mask = gaussian_filter(mask, sT)
    return mask


def correct_bleedthrough(Ly, Lx, nblks, mimg, mimg2):
    # subtract bleedthrough of green into red channel
    # non-rigid regression with nblks x nblks pieces
    sT = np.round((Ly + Lx) / (nblks * 2) * 0.25)
    mask = np.zeros((Ly, Lx, nblks, nblks), np.float32)
    weights = np.zeros((nblks, nblks), np.float32)
    yb = np.linspace(0, Ly, nblks + 1).astype(int)
    xb = np.linspace(0, Lx, nblks + 1).astype(int)
    for iy in range(nblks):
        for ix in range(nblks):
            ny = np.arange(yb[iy], yb[iy + 1]).astype(int)
            nx = np.arange(xb[ix], xb[ix + 1]).astype(int)
            mask[:, :, iy, ix] = quadrant_mask(Ly, Lx, ny, nx, sT)
            x = mimg[np.ix_(ny, nx)].flatten()
            x2 = mimg2[np.ix_(ny, nx)].flatten()
            # predict chan2 from chan1
            a = (x * x2).sum() / (x * x).sum()
            weights[iy, ix] = a
    mask /= mask.sum(axis=-1).sum(axis=-1)[:, :, np.newaxis, np.newaxis]
    mask *= weights
    mask *= mimg[:, :, np.newaxis, np.newaxis]
    mimg2 -= mask.sum(axis=-1).sum(axis=-1)
    mimg2 = np.maximum(0, mimg2)
    return mimg2


def intensity_ratio(ops, stats):
    """Compute pixels in cell and in area around cell (including overlaps)
    (exclude pixels from other cells)
    """
    Ly, Lx = ops["Ly"], ops["Lx"]
    cell_masks0, neuropil_ipix = masks.create_masks(
        roi_statistics=stats, height=ops["Ly"], width=ops["Lx"], neuropil=True, ops=ops
    )
    cell_masks = np.zeros((len(stats), Ly * Lx), np.float32)
    neuropil_masks = np.zeros((len(stats), Ly * Lx), np.float32)
    for cell_mask, cell_mask0, neuropil_mask, neuropil_mask0 in zip(
        cell_masks, cell_masks0, neuropil_masks, neuropil_ipix, strict=False
    ):
        cell_mask[cell_mask0[0]] = cell_mask0[1]
        neuropil_mask[neuropil_mask0.astype(np.int64)] = 1.0 / len(neuropil_mask0)

    mimg2 = ops["mean_image_channel_2"]
    inpix = cell_masks @ mimg2.flatten()
    extpix = neuropil_masks @ mimg2.flatten()
    inpix = np.maximum(1e-3, inpix)
    redprob = inpix / (inpix + extpix)
    redcell = redprob > ops["chan2_thres"]
    return np.stack((redcell, redprob), axis=-1)


def detect(ops, stats):
    """Detects cells with channel 2 brightness (red cells).

    Args:
        ops: The pipeline options dictionary containing mean images and dimensions.
        stats: The ROI statistics from primary channel detection.

    Returns:
        A tuple containing the updated ops dictionary and the red cell statistics array.
    """
    mimg = ops["mean_image"].copy()
    mimg2 = ops["mean_image_channel_2"].copy()

    # Subtracts bleedthrough of green into red channel using non-rigid regression with nblks x nblks pieces.
    nblks = 3
    Ly, Lx = ops["Ly"], ops["Lx"]
    ops["meanImg_chan2_corrected"] = correct_bleedthrough(Ly, Lx, nblks, mimg, mimg2)

    redstats = intensity_ratio(ops, stats)

    return ops, redstats
