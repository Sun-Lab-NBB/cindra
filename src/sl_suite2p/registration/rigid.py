"""Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu."""

import numpy as np

from .utils import (
    apply_mask,
    compute_reference_fft,
    apply_phase_correlation,
    apply_temporal_smoothing,
    compute_spatial_taper_mask,
    compute_gaussian_frequency_filter,
)


def compute_masks(refImg, maskSlope) -> tuple[np.ndarray, np.ndarray]:
    """Returns maskMul and maskOffset from an image and slope parameter

    Parameters
    ----------
    refImg: Ly x Lx
        The image
    maskSlope

    Returns:
    -------
    maskMul: float arrray
    maskOffset: float array
    """
    Ly, Lx = refImg.shape
    maskMul = compute_spatial_taper_mask(sigma=maskSlope, height=Ly, width=Lx)
    maskOffset = refImg.mean() * (1.0 - maskMul)
    return maskMul.astype("float32"), maskOffset.astype("float32")


def apply_masks(data: np.ndarray, maskMul: np.ndarray, maskOffset: np.ndarray) -> np.ndarray:
    """Returns a 3D image "data", multiplied by "maskMul" and then added "maskOffet".

    Parameters
    ----------
    data: nImg x Ly x Lx
    maskMul
    maskOffset

    Returns:
    --------
    maskedData: nImg x Ly x Lx
    """
    return apply_mask(data, maskMul, maskOffset)


def phasecorr_reference(refImg: np.ndarray, smooth_sigma=None) -> np.ndarray:
    """Returns reference image fft"ed and complex conjugate and multiplied by gaussian filter in the fft domain,
    with standard deviation "smooth_sigma" computes fft"ed reference image for phasecorr.

    Parameters
    ----------
    refImg : 2D array, int16
        reference image

    Returns:
    -------
    cfRefImg : 2D array, complex64
    """
    cfRefImg = compute_reference_fft(reference_image=refImg)
    cfRefImg /= 1e-5 + np.absolute(cfRefImg)
    cfRefImg *= compute_gaussian_frequency_filter(sigma=smooth_sigma, height=cfRefImg.shape[0], width=cfRefImg.shape[1])
    return cfRefImg.astype("complex64")


def phasecorr(data, cfRefImg, maxregshift, smooth_sigma_time) -> tuple[int, int, float]:
    """Compute phase correlation between data and reference image

    Parameters
    ----------
    data : int16
        array that"s frames x Ly x Lx
    maxregshift : float
        maximum shift as a fraction of the minimum dimension of data (min(Ly,Lx) * maxregshift)
    smooth_sigma_time : float
        how many frames to smooth in time

    Returns:
    -------
    ymax : int
        shifts in y from cfRefImg to data for each frame
    xmax : int
        shifts in x from cfRefImg to data for each frame
    cmax : float
        maximum of phase correlation for each frame

    """
    min_dim = np.minimum(*data.shape[1:])  # maximum registration shift allowed
    lcorr = int(np.minimum(np.round(maxregshift * min_dim), min_dim // 2))

    data = apply_phase_correlation(frames=data, kernel=cfRefImg)
    cc = np.real(
        np.block(
            [
                [data[:, -lcorr:, -lcorr:], data[:, -lcorr:, : lcorr + 1]],
                [data[:, : lcorr + 1, -lcorr:], data[:, : lcorr + 1, : lcorr + 1]],
            ]
        )
    )

    cc = apply_temporal_smoothing(frames=cc, sigma=smooth_sigma_time) if smooth_sigma_time > 0 else cc

    ymax, xmax = np.zeros(data.shape[0], np.int32), np.zeros(data.shape[0], np.int32)
    for t in np.arange(data.shape[0]):
        ymax[t], xmax[t] = np.unravel_index(np.argmax(cc[t], axis=None), (2 * lcorr + 1, 2 * lcorr + 1))
    cmax = cc[np.arange(len(cc)), ymax, xmax]
    ymax, xmax = ymax - lcorr, xmax - lcorr

    return ymax, xmax, cmax.astype(np.float32)


def shift_frame(frame: np.ndarray, dy: int, dx: int) -> np.ndarray:
    """Returns frame, shifted by dy and dx

    Parameters
    ----------
    frame: Ly x Lx
    dy: int
        vertical shift amount
    dx: int
        horizontal shift amount

    Returns:
    -------
    frame_shifted: Ly x Lx
        The shifted frame

    """
    return np.roll(frame, (-dy, -dx), axis=(0, 1))
