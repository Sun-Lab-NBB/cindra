"""Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu."""

import os
import time

import numpy as np
from ataraxis_base_utilities import console

from . import rigid, utils


def compute_zpos(Zreg, ops, reg_file=None):
    """Compute z position of frames given z-stack Zreg

    Parameters
    ----------

    Zreg : 3D array
        size [nplanes x Ly x Lx], z-stack

    ops : dictionary
        "registered_binary_path" <- binary to register to z-stack, "smooth_sigma",
        "Ly", "Lx", "batch_size"

    Returns:
    -------
    ops_orig
    zcorr
    """
    if "registered_binary_path" not in ops:
        message = "Unable to compute z-position of frames. The 'registered_binary_path' key is not present in the ops dictionary."
        console.error(message=message, error=OSError)

    nbatch = ops["batch_size"]
    Ly = ops["frame_height"]
    Lx = ops["frame_width"]
    nbytesread = 2 * Ly * Lx * nbatch

    ops_orig = ops.copy()
    ops["nonrigid"] = False
    nplanes, zLy, zLx = Zreg.shape
    if Zreg.shape[1] > Ly or Zreg.shape[2] != Lx:
        Zreg = Zreg[:,]

    reg_file = ops["registered_binary_path"] if reg_file is None else reg_file
    nbytes = os.path.getsize(reg_file)
    nFrames = int(nbytes / (2 * Ly * Lx))

    reg_file = open(reg_file, "rb")
    refAndMasks = []
    for Z in Zreg:
        if ops["one_p_reg"]:
            Z = Z.astype(np.float32)
            Z = Z[np.newaxis, :, :]
            if ops["pre_smooth"]:
                Z = utils.apply_spatial_smoothing(Z, int(ops["pre_smooth"]))
            Z = utils.apply_spatial_high_pass(Z, int(ops["spatial_hp_reg"]))
            Z = Z.squeeze()

        maskMul, maskOffset = rigid.compute_masks(
            refImg=Z,
            maskSlope=ops["spatial_taper"] if ops["one_p_reg"] else 3 * ops["smooth_sigma"],
        )
        cfRefImag = rigid.phasecorr_reference(refImg=Z, smooth_sigma=ops["smooth_sigma"])
        cfRefImag = cfRefImag[np.newaxis, :, :]
        refAndMasks.append((maskMul, maskOffset, cfRefImag))

    zcorr = np.zeros((Zreg.shape[0], nFrames), np.float32)
    t0 = time.time()
    k = 0
    nfr = 0
    while True:
        buff = reg_file.read(nbytesread)
        data = np.frombuffer(buff, dtype=np.int16, offset=0).copy()
        if (data.size == 0) | (nfr >= ops["frame_count"]):
            break
        data = np.float32(np.reshape(data, (-1, Ly, Lx)))
        inds = np.arange(nfr, nfr + data.shape[0], 1, int)
        for z, ref in enumerate(refAndMasks):
            # Preprocessing for 1P recordings. Data is already float32 from conversion at line 77.
            if ops["one_p_reg"]:
                if ops["pre_smooth"]:
                    data = utils.apply_spatial_smoothing(data, int(ops["pre_smooth"]))
                data = utils.apply_spatial_high_pass(data, int(ops["spatial_hp_reg"]))

            maskMul, maskOffset, cfRefImg = ref
            cfRefImg = cfRefImg.squeeze()

            _, _, zcorr[z, inds] = rigid.phasecorr(
                data=rigid.apply_masks(data=data, maskMul=maskMul, maskOffset=maskOffset),
                cfRefImg=cfRefImg,
                maxregshift=ops["maxregshift"],
                smooth_sigma_time=ops["smooth_sigma_time"],
            )
            if z % 10 == 1:
                console.echo(
                    message=(
                        f"Computing z-position: {z} planes, {nfr}/{ops['nframes']} frames processed "
                        f"({time.time() - t0:.2f} seconds elapsed)."
                    )
                )
        console.echo(
            message=(
                f"Computing z-position batch: {z} planes, {nfr}/{ops['nframes']} frames processed "
                f"({time.time() - t0:.2f} seconds elapsed)."
            )
        )
        nfr += data.shape[0]
        k += 1

    reg_file.close()
    ops_orig["zcorr"] = zcorr
    return ops_orig, zcorr
