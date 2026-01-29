"""Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu."""

from os import path
from warnings import warn

from tqdm import tqdm
import numpy as np
from scipy.signal import medfilt, medfilt2d
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import LogLevel, console

from . import (
    rigid,
    utils,
    nonrigid,
    bidiphase as bidi,
)


def compute_crop(xoff: int, yoff: int, corrXY, th_badframes, badframes, maxregshift, Ly: int, Lx: int):
    """Determines how much to crop FOV based on motion

    determines badframes which are frames with large outlier shifts
    (threshold of outlier is th_badframes) and
    it excludes these badframes when computing valid ranges
    from registration in y and x

    Parameters
    __________
    xoff: int
    yoff: int
    corrXY
    th_badframes
    badframes
    maxregshift
    Ly: int
        Height of a frame
    Lx: int
        Width of a frame

    Returns:
    _______
    badframes
    yrange
    xrange
    """
    filter_window = min((len(yoff) // 2) * 2 - 1, 101)
    dx = xoff - medfilt(xoff, filter_window)
    dy = yoff - medfilt(yoff, filter_window)
    # Offset in x and y (normed by mean offset). If mean is 0 (no motion), dxy stays as zeros.
    dxy = (dx**2 + dy**2) ** 0.5
    dxy_mean = dxy.mean()
    if dxy_mean > 0:
        dxy = dxy / dxy_mean
    # phase-corr of each frame with reference (normed by median phase-corr)
    cXY = corrXY / medfilt(corrXY, filter_window)
    # exclude frames which have a large deviation and/or low correlation
    px = dxy / np.maximum(0, cXY)
    badframes = np.logical_or(px > th_badframes * 100, badframes)
    badframes = np.logical_or(abs(xoff) > (maxregshift * Lx * 0.95), badframes)
    badframes = np.logical_or(abs(yoff) > (maxregshift * Ly * 0.95), badframes)
    if badframes.mean() < 0.5:
        ymin = np.ceil(np.abs(yoff[np.logical_not(badframes)]).max())
        xmin = np.ceil(np.abs(xoff[np.logical_not(badframes)]).max())
    else:
        warn("WARNING: >50% of frames have large movements, registration likely problematic")
        ymin = np.ceil(np.abs(yoff).max())
        xmin = np.ceil(np.abs(xoff).max())
    ymax = Ly - ymin
    xmax = Lx - xmin
    yrange = [int(ymin), int(ymax)]
    xrange = [int(xmin), int(xmax)]

    return badframes, yrange, xrange


def pick_initial_reference(frames: np.ndarray, k=20) -> np.ndarray:
    """Computes the initial reference image

    the seed frame is the frame with the largest correlations with other frames;
    the average of the seed frame with its top k correlated pairs is the
    initial reference frame returned

    Parameters
    ----------
    frames : 3D array, int16
        size [frames x Ly x Lx], frames from binary
    k : int, optional
        number of top correlations to average, by default 20

    Returns:
    -------
    refImg : 2D array, int16
        size [Ly x Lx], initial reference image

    """
    nimg, Ly, Lx = frames.shape
    frames = np.reshape(frames, (nimg, -1)).astype("float32")  # flatten frames
    frames = frames - np.reshape(frames.mean(axis=1), (nimg, 1))  # subtract mean
    cc = np.matmul(frames, frames.T)  # correlation matrix (nimg x nimg)
    ndiag = np.sqrt(np.diag(cc))  # norm of each frame
    cc = cc / np.outer(ndiag, ndiag)  # normalize by norm of each frame
    CCpartsort = np.partition(cc, -(k + 1), axis=1)[:, -k:-1]  # skip the self-correlation
    bestCC = np.mean(CCpartsort, axis=1)  # mean of top k-1 correlations for each frame
    imax = np.argmax(bestCC)
    indpartsort = np.argpartition(cc[imax, :], -k)[-k:]  # top k correlations for seed frame
    refImg = np.mean(frames[indpartsort, :], axis=0)
    refImg = np.reshape(refImg, (Ly, Lx))
    return refImg


def compute_reference(frames, ops=generate_default_ops()):
    """Computes the reference image

    picks initial reference then iteratively aligns frames to create reference

    Parameters
    ----------

    ops : dictionary
        need registration options

    frames : 3D array, int16
        size [nimg_init x Ly x Lx], frames to use to create initial reference

    Returns:
    -------
    refImg : 2D array, int16
        size [Ly x Lx], initial reference image

    """
    refImg = pick_initial_reference(frames)
    if ops["one_p_reg"]:
        if ops["pre_smooth"]:
            refImg = utils.spatial_smooth(refImg, int(ops["pre_smooth"]))
            frames = utils.spatial_smooth(frames, int(ops["pre_smooth"]))
        refImg = utils.spatial_high_pass(refImg, int(ops["spatial_hp_reg"]))
        frames = utils.spatial_high_pass(frames, int(ops["spatial_hp_reg"]))

    niter = 8
    for iter in range(niter):
        # rigid registration
        ymax, xmax, cmax = rigid.phasecorr(
            data=rigid.apply_masks(
                frames,
                *rigid.compute_masks(
                    refImg=refImg,
                    maskSlope=ops["spatial_taper"] if ops["one_p_reg"] else 3 * ops["smooth_sigma"],
                ),
            ),
            cfRefImg=rigid.phasecorr_reference(
                refImg=refImg,
                smooth_sigma=ops["smooth_sigma"],
            ),
            maxregshift=ops["maxregshift"],
            smooth_sigma_time=ops["smooth_sigma_time"],
        )
        for frame, dy, dx in zip(frames, ymax, xmax, strict=False):
            frame[:] = rigid.shift_frame(frame=frame, dy=dy, dx=dx)

        nmax = max(2, int(frames.shape[0] * (1.0 + iter) / (2 * niter)))
        isort = np.argsort(-cmax)[1:nmax]
        # reset reference image
        refImg = frames[isort].mean(axis=0).astype(np.int16)
        # shift reference image to position of mean shifts
        refImg = rigid.shift_frame(
            frame=refImg, dy=int(np.round(-ymax[isort].mean())), dx=int(np.round(-xmax[isort].mean()))
        )

    return refImg


def compute_reference_masks(refImg, ops=generate_default_ops()):
    """Computes registration masks for the reference image."""
    maskMul, maskOffset = rigid.compute_masks(
        refImg=refImg,
        maskSlope=ops["spatial_taper"] if ops["one_p_reg"] else 3 * ops["smooth_sigma"],
    )
    cfRefImg = rigid.phasecorr_reference(
        refImg=refImg,
        smooth_sigma=ops["smooth_sigma"],
    )
    Ly, Lx = refImg.shape
    blocks = []
    if ops.get("nonrigid"):
        blocks = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=ops["block_size"])

        maskMulNR, maskOffsetNR, cfRefImgNR = nonrigid.phasecorr_reference(
            refImg0=refImg,
            maskSlope=ops["spatial_taper"]
            if ops["one_p_reg"]
            else 3 * ops["smooth_sigma"],  # slope of taper mask at the edges
            smooth_sigma=ops["smooth_sigma"],
            yblock=blocks[0],
            xblock=blocks[1],
        )
    else:
        maskMulNR, maskOffsetNR, cfRefImgNR = [], [], []

    return maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR, blocks


def register_frames(refAndMasks, frames, rmin=-np.inf, rmax=np.inf, bidiphase=0, ops=generate_default_ops()):
    """Registers frames to a reference image.

    Args:
        refAndMasks: Processed reference images and masks, or 2D array of reference image.
        frames: Frame data with shape (time, Ly, Lx).
        rmin: Minimum value to clip frames at.
        rmax: Maximum value to clip frames at.
        bidiphase: Bidirectional phase offset.
        ops: Registration options dictionary.

    Returns:
        Tuple of (frames, ymax, xmax, cmax, ymax1, xmax1, cmax1, zest).
    """
    if len(refAndMasks) == 7 or not isinstance(refAndMasks, np.ndarray):
        maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR, blocks = refAndMasks
    else:
        refImg = refAndMasks
        if ops.get("norm_frames", False) and "rmin" not in ops:
            rmin, rmax = np.int16(np.percentile(refImg, 1)), np.int16(np.percentile(refImg, 99))
            refImg = np.clip(refImg, rmin, rmax)
        maskMul, maskOffset, cfRefImg, maskMulNR, maskOffsetNR, cfRefImgNR, blocks = compute_reference_masks(
            refImg, ops
        )

    if bidiphase != 0:
        bidi.shift(frames, bidiphase)

    # if smoothing or filtering or clipping to compute registration shifts, make a copy of the frames
    dtype = "float32" if ops["smooth_sigma_time"] > 0 or ops["one_p_reg"] else frames.dtype
    fsmooth = frames.copy().astype(dtype) if ops["smooth_sigma_time"] > 0 or ops["one_p_reg"] else frames

    if ops["smooth_sigma_time"]:
        fsmooth = utils.temporal_smooth(data=fsmooth, sigma=ops["smooth_sigma_time"])
    else:
        fsmooth = frames

    # preprocessing for 1P recordings
    if ops["one_p_reg"]:
        if ops["pre_smooth"]:
            fsmooth = utils.spatial_smooth(fsmooth, int(ops["pre_smooth"]))
        fsmooth = utils.spatial_high_pass(fsmooth, int(ops["spatial_hp_reg"]))

    # rigid registration
    ymax, xmax, cmax = rigid.phasecorr(
        data=rigid.apply_masks(
            data=np.clip(fsmooth, rmin, rmax) if rmin > -np.inf else fsmooth, maskMul=maskMul, maskOffset=maskOffset
        ),
        cfRefImg=cfRefImg,
        maxregshift=ops["maxregshift"],
        smooth_sigma_time=ops["smooth_sigma_time"],
    )

    for frame, dy, dx in zip(frames, ymax, xmax, strict=False):
        frame[:] = rigid.shift_frame(frame=frame, dy=dy, dx=dx)

    # non-rigid registration
    if ops["nonrigid"]:
        # need to also shift smoothed/filtered data
        if ops["smooth_sigma_time"] or ops["one_p_reg"]:
            for fsm, dy, dx in zip(fsmooth, ymax, xmax, strict=False):
                fsm[:] = rigid.shift_frame(frame=fsm, dy=dy, dx=dx)

        ymax1, xmax1, cmax1 = nonrigid.phasecorr(
            data=np.clip(fsmooth, rmin, rmax) if rmin > -np.inf else fsmooth,
            maskMul=maskMulNR.squeeze(),
            maskOffset=maskOffsetNR.squeeze(),
            cfRefImg=cfRefImgNR.squeeze(),
            snr_thresh=ops["snr_thresh"],
            NRsm=blocks[-1],
            xblock=blocks[1],
            yblock=blocks[0],
            maxregshift_nr=ops["maxregshift_nr"],
        )

        frames = nonrigid.transform_data(
            data=frames,
            yblock=blocks[0],
            xblock=blocks[1],
            nblocks=blocks[2],
            ymax1=ymax1,
            xmax1=xmax1,
        )
    else:
        ymax1, xmax1, cmax1 = None, None, None

    return frames, ymax, xmax, cmax, ymax1, xmax1, cmax1, None


def shift_frames(frames, yoff, xoff, yoff1, xoff1, blocks=None, ops=generate_default_ops()):
    if ops["bidiphase"] != 0 and not ops["bidi_corrected"]:
        bidi.shift(frames, int(ops["bidiphase"]))

    for frame, dy, dx in zip(frames, yoff, xoff, strict=False):
        frame[:] = rigid.shift_frame(frame=frame, dy=dy, dx=dx)

    if ops["nonrigid"]:
        frames = nonrigid.transform_data(
            frames,
            yblock=blocks[0],
            xblock=blocks[1],
            nblocks=blocks[2],
            ymax1=yoff1,
            xmax1=xoff1,
            bilinear=ops.get("bilinear_reg", True),
        )
    return frames


def normalize_reference_image(refImg):
    """Normalizes the reference image by clipping to the 1st and 99th percentiles."""
    rmin, rmax = np.int16(np.percentile(refImg, 1)), np.int16(np.percentile(refImg, 99))
    refImg = np.clip(refImg, rmin, rmax)
    return refImg, rmin, rmax


def compute_reference_and_register_frames(
    f_align_in, plane_number: int, f_align_out=None, refImg=None, ops=generate_default_ops()
):
    """Compute reference frame, if refImg is None, and align frames in f_align_in to reference

    if f_align_out is not None, registered frames are written to f_align_out

    f_align_in, f_align_out can be a BinaryFile or any type of array that can be slice-indexed

    """
    # Initializes a timer to time processing stages
    timer = PrecisionTimer("s")

    n_frames, Ly, Lx = f_align_in.shape

    batch_size = ops["batch_size"]
    # Compute reference image and bidiphase shift
    if refImg is None:
        # grab frames
        frames = f_align_in[np.linspace(0, n_frames, 1 + np.minimum(ops["nimg_init"], n_frames), dtype=int)[:-1]]
        # compute bidiphase shift
        if ops["do_bidiphase"] and ops["bidiphase"] == 0 and not ops["bidi_corrected"]:
            bidiphase = bidi.compute(frames)
            console.echo(
                f"Plane {plane_number} estimated bidiphase offset from data: {bidiphase} pixels.",
                level=LogLevel.INFO,
            )
            ops["bidiphase"] = bidiphase
            # shift frames
            if bidiphase != 0:
                bidi.shift(frames, int(ops["bidiphase"]))

        if refImg is None:
            console.echo(f"Computing plane {plane_number} reference frame...", level=LogLevel.INFO)
            timer.reset()
            refImg = compute_reference(frames, ops=ops)
            console.echo(
                f"Plane {plane_number} reference frame: computed. Time taken: {timer.elapsed} seconds.",
                level=LogLevel.SUCCESS,
            )

    # Normalize reference image
    refImg_orig = refImg.copy()
    if ops.get("norm_frames", False):
        refImg, rmin, rmax = normalize_reference_image(refImg)
    else:
        rmin, rmax = -np.inf, np.inf

    if ops["bidiphase"] and not ops["bidi_corrected"]:
        bidiphase = int(ops["bidiphase"])
    else:
        bidiphase = 0

    refAndMasks = compute_reference_masks(refImg, ops)

    # Register frames to reference image
    mean_img = np.zeros((Ly, Lx), "float32")
    rigid_offsets, nonrigid_offsets, zpos, cmax_all = [], [], [], []

    if ops["frames_include"] != -1:
        n_frames = min(n_frames, ops["frames_include"])

    timer.reset()
    console.echo(f"Computing plane {plane_number} frame registration offsets for channel 1...", level=LogLevel.INFO)

    # Uses tqdm progress bar when sessions are processed sequentially.
    for batch_number in tqdm(
        np.arange(0, n_frames, batch_size),
        desc=f"Registering batches of {batch_size} frames",
        unit="batch",
        disable=not ops["progress_bars"],
    ):
        frames = f_align_in[batch_number : min(batch_number + batch_size, n_frames)]
        frames, ymax, xmax, cmax, ymax1, xmax1, cmax1, zest = register_frames(
            refAndMasks, frames, rmin=rmin, rmax=rmax, bidiphase=bidiphase, ops=ops
        )
        rigid_offsets.append([ymax, xmax, cmax])
        if zest is not None:
            zpos.extend(list(zest[0]))
            cmax_all.extend(list(zest[1]))
        if ops["nonrigid"]:
            nonrigid_offsets.append([ymax1, xmax1, cmax1])

        mean_img += frames.sum(axis=0) / n_frames

        if f_align_out is None:
            f_align_in[batch_number : min(batch_number + batch_size, n_frames)] = frames
        else:
            f_align_out[batch_number : min(batch_number + batch_size, n_frames)] = frames

    console.echo(
        f"Plane {plane_number} channel 1 frame registration offsets: computed. Time taken: {timer.elapsed} seconds.",
        level=LogLevel.SUCCESS,
    )

    rigid_offsets = utils.combine_offsets_across_batches(rigid_offsets, rigid=True)
    if ops["nonrigid"]:
        nonrigid_offsets = utils.combine_offsets_across_batches(nonrigid_offsets, rigid=False)

    return refImg_orig, rmin, rmax, mean_img, rigid_offsets, nonrigid_offsets, (zpos, cmax_all)


def shift_frames_and_write(
    f_alt_in,
    plane_number: int,
    f_alt_out=None,
    yoff=None,
    xoff=None,
    yoff1=None,
    xoff1=None,
    ops=generate_default_ops(),
):
    """Shift frames for alternate channel in f_alt_in and write to f_alt_out if not None (else write to f_alt_in)"""
    n_frames, Ly, Lx = f_alt_in.shape
    if yoff is None or xoff is None:
        message = "Unable to shift and write frames. No rigid registration offsets provided (yoff or xoff is None)."
        console.error(message=message, error=ValueError)
    if yoff.shape[0] != n_frames or xoff.shape[0] != n_frames:
        message = f"Unable to shift and write frames. Rigid registration offsets size mismatch: expected {n_frames} frames, but got yoff={yoff.shape[0]}, xoff={xoff.shape[0]}."
        console.error(message=message, error=ValueError)
    # Overwrite blocks if nonrigid registration is activated
    blocks = None
    if ops.get("nonrigid"):
        if yoff1 is None or xoff1 is None:
            message = "Unable to shift and write frames. Non-rigid registration is enabled but no non-rigid offsets provided (yoff1 or xoff1 is None)."
            console.error(message=message, error=ValueError)
        if yoff1.shape[0] != n_frames or xoff1.shape[0] != n_frames:
            message = f"Unable to shift and write frames. Non-rigid registration offsets size mismatch: expected {n_frames} frames, but got yoff1={yoff1.shape[0]}, xoff1={xoff1.shape[0]}."
            console.error(message=message, error=ValueError)

        blocks = nonrigid.make_blocks(Ly=Ly, Lx=Lx, block_size=ops["block_size"])

    if ops["frames_include"] != -1:
        n_frames = min(n_frames, ops["frames_include"])

    mean_img = np.zeros((Ly, Lx), "float32")
    batch_size = ops["batch_size"]
    timer = PrecisionTimer("s")
    console.echo(f"Computing plane {plane_number} frame registration offsets for channel 2...", level=LogLevel.INFO)
    timer.reset()
    for batch_number in tqdm(
        np.arange(0, n_frames, batch_size),
        desc=f"Registering batches of {batch_size} frames",
        unit="batch",
        disable=not ops["progress_bars"],
    ):
        frames = f_alt_in[batch_number : min(batch_number + batch_size, n_frames)].astype("float32")
        yoffk = yoff[batch_number : min(batch_number + batch_size, n_frames)].astype(int)
        xoffk = xoff[batch_number : min(batch_number + batch_size, n_frames)].astype(int)
        if ops.get("nonrigid"):
            yoff1k = yoff1[batch_number : min(batch_number + batch_size, n_frames)]
            xoff1k = xoff1[batch_number : min(batch_number + batch_size, n_frames)]
        else:
            yoff1k, xoff1k = None, None

        frames = shift_frames(frames, yoffk, xoffk, yoff1k, xoff1k, blocks, ops)
        mean_img += frames.sum(axis=0) / n_frames

        if f_alt_out is None:
            f_alt_in[batch_number : min(batch_number + batch_size, n_frames)] = frames
        else:
            f_alt_out[batch_number : min(batch_number + batch_size, n_frames)] = frames

    console.echo(
        f"Plane {plane_number} channel 2 frame registration offsets: computed. Time taken: {timer.elapsed} seconds.",
        level=LogLevel.SUCCESS,
    )

    return mean_img


def registration_wrapper(
    f_reg,
    plane_number: int,
    f_raw=None,
    f_reg_chan2=None,
    f_raw_chan2=None,
    refImg=None,
    align_by_chan2=False,
    ops=generate_default_ops(),
):
    """Main registration function

    if f_raw is not None, f_raw is read and registered and saved to f_reg
    if f_raw_chan2 is not None, f_raw_chan2 is read and registered and saved to f_reg_chan2

    the registration shifts are computed on chan2 if ops["functional_chan"] != ops["align_by_chan"]


    Parameters
    ----------------

    f_reg : array of registered functional frames, np.ndarray or io.BinaryFile
        n_frames x Ly x Lx

    f_raw : array of raw functional frames, np.ndarray or io.BinaryFile
        n_frames x Ly x Lx

    f_reg_chan2 : array of registered anatomical frames, np.ndarray or io.BinaryFile
        n_frames x Ly x Lx

    f_raw_chan2 : array of raw anatomical frames, np.ndarray or io.BinaryFile
        n_frames x Ly x Lx

    refImg : 2D array, int16
        size [Ly x Lx], initial reference image

    align_by_chan2: boolean
        whether you"d like to align by non-functional channel

    ops : dictionary or list of dicts
        dictionary containing input arguments for suite2p pipeline

    Returns:
    ----------------
    refImg : 2D array, int16
        size [Ly x Lx], initial reference image (if not registered)

    rmin : int
        clip frames at rmin

    rmax : int
        clip frames at rmax

    meanImg : np.ndarray,
        size [Ly x Lx], Computed Mean Image for functional channel

    rigid_offsets : Tuple of length 3,
        Rigid shifts computed between each frame and reference image. Shifts for each frame in x,y, and z directions

    nonrigid_offsets : Tuple of length 3
        Non-rigid shifts computed between each frame and reference image.

    zest : Tuple of length 2

    meanImg_chan2: np.ndarray,
        size [Ly x Lx], Computed Mean Image for non-functional channel

    badframes : np.ndarray,
        size [n_frames, ] Boolean array of frames that have large outlier shifts that may make registration problematic.

    yrange : list of length 2
        Valid ranges for registration along y-axis of frames

    xrange : list of length 2
        Valid ranges for registration along x-axis of frames

    """
    f_alt_in, f_align_out, f_alt_out = None, None, None
    if f_reg_chan2 is None or not align_by_chan2:
        if f_raw is None:
            f_align_in = f_reg
            f_alt_in = f_reg_chan2
        else:
            f_align_in = f_raw
            f_alt_in = f_raw_chan2
            f_align_out = f_reg
            f_alt_out = f_reg_chan2
    elif f_raw is None:
        f_align_in = f_reg_chan2
        f_alt_in = f_reg
    else:
        f_align_in = f_raw_chan2
        f_alt_in = f_raw
        f_align_out = f_reg_chan2
        f_alt_out = f_reg

    n_frames, Ly, Lx = f_align_in.shape
    if f_alt_in is not None and f_alt_in.shape[0] == f_align_in.shape[0]:
        nchannels = 2
        console.echo(message=f"Registering two channels for plane {plane_number}...", level=LogLevel.INFO)
    else:
        nchannels = 1
        console.echo(message=f"Registering a single channel for plane {plane_number}...", level=LogLevel.INFO)

    outputs = compute_reference_and_register_frames(
        f_align_in, plane_number=plane_number, f_align_out=f_align_out, refImg=refImg, ops=ops
    )
    refImg, rmin, rmax, mean_img, rigid_offsets, nonrigid_offsets, zest = outputs
    yoff, xoff, corrXY = rigid_offsets

    if ops["nonrigid"]:
        yoff1, xoff1, corrXY1 = nonrigid_offsets
    else:
        yoff1, xoff1, corryXY1 = None, None, None

    if nchannels > 1:
        mean_img_alt = shift_frames_and_write(
            f_alt_in=f_alt_in,
            plane_number=plane_number,
            f_alt_out=f_alt_out,
            yoff=yoff,
            xoff=xoff,
            yoff1=yoff1,
            xoff1=xoff1,
            ops=ops,
        )
    else:
        mean_img_alt = None

    if nchannels == 1 or not align_by_chan2:
        meanImg = mean_img
        if nchannels == 2:
            meanImg_chan2 = mean_img_alt
        else:
            meanImg_chan2 = None
    elif nchannels == 2:
        meanImg_chan2 = mean_img
        meanImg = mean_img_alt

    # compute valid region
    badframes = np.zeros(n_frames, "bool")
    if ops.get("data_path"):
        badfrfile = path.abspath(path.join(str(ops["data_path"]), "bad_frames.npy"))
        # Check if badframes file exists
        if path.isfile(badfrfile):
            console.echo(
                message=f"Plane {plane_number} bad frames file: exists. Path: {badfrfile}.",
                level=LogLevel.WARNING,
            )
            bf_indices = np.load(badfrfile)
            bf_indices = bf_indices.flatten().astype(int)
            # Set indices of badframes to true
            badframes[bf_indices] = True
            console.echo(message=f"Plane {plane_number} bad frames count: {badframes.sum()}.", level=LogLevel.WARNING)

    # return frames which fall outside range
    badframes, yrange, xrange = compute_crop(
        xoff=xoff,
        yoff=yoff,
        corrXY=corrXY,
        th_badframes=ops["th_badframes"],
        badframes=badframes,
        maxregshift=ops["maxregshift"],
        Ly=Ly,
        Lx=Lx,
    )

    return refImg, rmin, rmax, meanImg, rigid_offsets, nonrigid_offsets, zest, meanImg_chan2, badframes, yrange, xrange


def save_registration_outputs_to_ops(registration_outputs, ops):
    refImg, rmin, rmax, meanImg, rigid_offsets, nonrigid_offsets, zest, meanImg_chan2, badframes, yrange, xrange = (
        registration_outputs
    )
    # assign reference image and normalizers
    ops["refImg"] = refImg
    ops["rmin"], ops["rmax"] = rmin, rmax
    # assign rigid offsets to ops
    ops["yoff"], ops["xoff"], ops["corrXY"] = rigid_offsets
    # assign nonrigid offsets to ops
    if ops["nonrigid"]:
        ops["yoff1"], ops["xoff1"], ops["corrXY1"] = nonrigid_offsets
    # assign mean images
    ops["mean_image"] = meanImg
    if meanImg_chan2 is not None:
        ops["mean_image_channel_2"] = meanImg_chan2
    # assign crop computation and badframes
    ops["badframes"], ops["yrange"], ops["xrange"] = badframes, yrange, xrange
    if len(zest[0]) > 0:
        ops["zpos_registration"] = np.array(zest[0])
        ops["cmax_registration"] = np.array(zest[1])
    return ops


def create_enhanced_mean_image(ops):
    """Updates the input 'ops' dictionary to include the enhanced mean image of the processed cell activity movie.

    Args:
        ops: The dictionary that contains the processing parameters and the intermediate processing results.

    Returns:
        The input 'ops' dictionary, updated to include the computed enhanced mean image 'enhanced_mean_image' field.
    """
    # Pre-initializes the enhanced mean image array by taking the original mean image
    mean_image = ops["mean_image"].astype(np.float32)

    # Defines the parameters for enhancing the mean image.
    scale_background = 4
    minimum_intensity = -6
    maximum_intensity = 6

    # If the spatial scaling is not defined inside the 'ops' dictionary, determines the optimal spatial scaling based
    # on the diameter of the discovered cell ROI objects.
    if "spatial_scale_pixels" not in ops:
        if isinstance(ops["diameter"], int):
            cell_diameter = np.array([ops["diameter"], ops["diameter"]])
        else:
            cell_diameter = np.array(ops["diameter"])

        # If the diameter is set to 0, the CellPose algorithm has not yet established the ROI diameter. In this case,
        # defaults to using the diameter of 12 pixels.
        if cell_diameter[0] == 0:
            cell_diameter[:] = 12

        # Calculates the spatial scaling and the aspect ratio based on the resolved cell ROI diameter.
        ops["spatial_scale_pixels"] = cell_diameter[1]
        ops["aspect"] = cell_diameter[0] / cell_diameter[1]

    # Creates a median filter with a large kernel (4 * cell diameter)
    filter_height = scale_background * np.ceil(ops["spatial_scale_pixels"] * ops["aspect"]) + 1
    filter_width = scale_background * np.ceil(ops["spatial_scale_pixels"]) + 1
    filter_kernel_size = (int(filter_height), int(filter_width))

    # Uses the median filter to compute and subtract the background fluorescence from the mean image, enhancing the
    # cell objects.
    background = medfilt2d(mean_image, filter_kernel_size)
    background_removed = mean_image - background

    # Normalizes the cell contrast across the image by dividing the background-subtracted image by the local variance.
    local_variance = medfilt2d(np.absolute(background_removed), filter_kernel_size)
    normalized_image = background_removed / (1e-10 + local_variance)

    # Excludes the pixels along the border of the image, as they are typically discarded during the registration
    # process.
    y_start, y_end = ops["yrange"]
    x_start, x_end = ops["xrange"]
    roi_image = normalized_image[y_start:y_end, x_start:x_end]

    # Clips the normalized image intensities to +-6 standard deviations after normalization and scales the image to
    # reflect this range of intensities.
    scaled_roi = (roi_image - minimum_intensity) / (maximum_intensity - minimum_intensity)
    scaled_roi = np.clip(scaled_roi, 0, 1)

    # Places the enhanced mean image back into the original mean image array, replacing the pixels along the border with
    # the minimal intensity values.
    height, width = ops["Ly"], ops["Lx"]
    enhanced_image = np.full((height, width), scaled_roi.min(), dtype=np.float32)
    enhanced_image[y_start:y_end, x_start:x_end] = scaled_roi

    ops["enhanced_mean_image"] = enhanced_image

    return ops
