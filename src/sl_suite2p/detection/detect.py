"""Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu."""

from typing import Any
from pathlib import Path

import numpy as np
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import LogLevel, console

from . import sourcery, anatomical, chan2detect, sparsedetect
from .stats import roi_stats
from .denoise import pca_denoise
from ..io.binary import BinaryFile
from ..configuration import generate_default_ops
from ..classification import classify, user_classfile


def detect(ops, plane_number: int, classfile=None):
    timer = PrecisionTimer("s")

    bin_size = int(max(1, ops["nframes"] // ops["nbinned"], np.round(ops["tau"] * ops["fs"])))

    console.echo(f"Binning plane {plane_number} movie in chunks of length {bin_size}...", level=LogLevel.INFO)
    timer.reset()

    with BinaryFile(file_path=ops["reg_file"], height=ops["Ly"], width=ops["Lx"]) as f:
        mov = f.bin_movie(
            bin_size=bin_size,
            bad_frames=ops.get("badframes"),
            y_range=ops["yrange"],
            x_range=ops["xrange"],
        )

        message = (
            f"Plane {plane_number} movie: binned. Resultant movie dimensions: "
            f"{mov.shape[0]}, {mov.shape[1]}, {mov.shape[2]}. Time taken: {timer.elapsed} seconds."
        )
        console.echo(message=message, level=LogLevel.SUCCESS)

        ops, stat = detection_wrapper(f, plane_number=plane_number, mov=mov, ops=ops, classfile=classfile)

    return ops, stat


def bin_movie(f_reg, plane_number: int, bin_size, yrange=None, xrange=None, badframes=None):
    """Bin registered movie"""
    n_frames = f_reg.shape[0]
    good_frames = ~badframes if badframes is not None else np.ones(n_frames, dtype=bool)
    batch_size = min(good_frames.sum(), 500)
    Lyc = yrange[1] - yrange[0]
    Lxc = xrange[1] - xrange[0]

    # Number of binned frames is rounded down when binning frames
    num_binned_frames = n_frames // bin_size
    mov = np.zeros((num_binned_frames, Lyc, Lxc), np.float32)
    curr_bin_number = 0

    # Iterate over n_frames to maintain binning over TIME
    for k in np.arange(0, n_frames, batch_size):
        data = f_reg[k : min(k + batch_size, n_frames)]

        # exclude badframes
        good_indices = good_frames[k : min(k + batch_size, n_frames)]
        if good_indices.mean() > 0.5:
            data = data[good_indices]

        # crop to valid region
        if yrange is not None and xrange is not None:
            data = data[:, slice(*yrange), slice(*xrange)]

        # bin in time
        if data.shape[0] > bin_size:
            # Downsample by binning via reshaping and taking mean of each bin
            # only if current batch size exceeds or matches bin_size
            n_d = data.shape[0]
            data = data[: (n_d // bin_size) * bin_size]
            data = data.reshape(-1, bin_size, Lyc, Lxc).astype(np.float32).mean(axis=1)
        else:
            # Current batch size is below bin_size (could have many bad frames in this batch)
            # Downsample taking the mean of batch to get a single bin
            data = data.mean(axis=0)[np.newaxis, :, :]
        # Only fill in binned data if not exceeding the number of bins mov has
        if mov.shape[0] > curr_bin_number:
            # Fill in binned data
            n_bins = data.shape[0]
            mov[curr_bin_number : curr_bin_number + n_bins] = data
            curr_bin_number += n_bins
    return mov


def detection_wrapper(
    f_reg, plane_number: int, mov=None, yrange=None, xrange=None, ops=generate_default_ops(), classfile=None
):
    """Main detection function.

    Identifies ROIs.

    Parameters
    ----------------

    f_reg : np.ndarray or io.BinaryWFile,
            n_frames x Ly x Lx

    mov : ndarray (t x Lyc x Lxc)
                    binned movie

    yrange : list of length 2
            Range of pixels along the y-axis of mov the detection module will be run on

    xrange : list of length 2
            Range of pixels along the x-axis of mov the detection module will be run on

    ops : dictionary or list of dicts

    classfile: string (optional, default None)
            path to saved classifier

    Returns:
    ----------------
    ops : dictionary or list of dicts

    stat : dictionary "ypix", "xpix", "lam"
            Dictionary containing statistics for ROIs


    """
    timer = PrecisionTimer("s")
    n_frames, Ly, Lx = f_reg.shape
    yrange = ops.get("yrange", [0, Ly]) if yrange is None else yrange
    xrange = ops.get("xrange", [0, Lx]) if xrange is None else xrange
    ops["yrange"] = yrange
    ops["xrange"] = xrange

    if mov is None:
        bin_size = int(max(1, n_frames // ops["nbinned"], np.round(ops["tau"] * ops["fs"])))
        console.echo(f"Binning plane {plane_number} movie in chunks of length {bin_size}...", level=LogLevel.INFO)

        timer.reset()
        mov = bin_movie(
            f_reg,
            plane_number=plane_number,
            bin_size=bin_size,
            yrange=yrange,
            xrange=xrange,
            badframes=ops.get("badframes", None),
        )

        message = (
            f"Plane {plane_number} movie: binned. Resultant movie dimensions: "
            f"{mov.shape[0]}, {mov.shape[1]}, {mov.shape[2]}. Time taken: {timer.elapsed} seconds."
        )
        console.echo(message=message, level=LogLevel.SUCCESS)
    elif mov.shape[1] != yrange[-1] - yrange[0]:
        raise ValueError("mov.shape[1] is not same size as yrange")
    elif mov.shape[2] != xrange[-1] - xrange[0]:
        raise ValueError("mov.shape[2] is not same size as xrange")

    if "meanImg" not in ops:
        ops["meanImg"] = mov.mean(axis=0)
        ops["max_proj"] = mov.max(axis=0)

    if ops.get("inverted_activity", False):
        mov -= mov.min()
        mov *= -1
        mov -= mov.min()

    if ops.get("denoise", 1):
        mov = pca_denoise(mov, block_size=[ops["block_size"][0] // 2, ops["block_size"][1] // 2], n_comps_frac=0.5)

    if ops.get("anatomical_only", 0):
        source_types = ["max_proj / mean_img", "mean_img", "enhanced_mean_img", "max_proj"]
        message = (
            f"Applying Cellpose to plane {plane_number} movie to find cell masks in "
            f"{source_types[int(ops['anatomical_only']) - 1]}..."
        )
        console.echo(message=message, level=LogLevel.INFO)
        timer.reset()
        stat = anatomical.select_rois(ops=ops, mov=mov, diameter=ops.get("diameter", None))
        message = (
            f"Plane {plane_number} cell masks: discovered. Detected ROIs: {len(stat)}. "
            f"Time taken: {timer.elapsed} seconds."
        )
        console.echo(message=message, level=LogLevel.SUCCESS)
    else:
        message = f"Finding cell mask ROIs for plane {plane_number}..."
        console.echo(message=message, level=LogLevel.INFO)
        timer.reset()
        stat = select_rois(ops=ops, mov=mov, sparse_mode=ops["sparse_mode"], plane_number=plane_number)
        message = (
            f"Plane {plane_number} cell masks: discovered. Detected ROIs: {len(stat)}. "
            f"Time taken: {timer.elapsed} seconds."
        )
        console.echo(message=message, level=LogLevel.SUCCESS)

    ymin = int(yrange[0])
    xmin = int(xrange[0])
    if len(stat) > 0:
        for s in stat:
            s["ypix"] += ymin
            s["xpix"] += xmin
            s["med"][0] += ymin
            s["med"][1] += xmin

        if ops["preclassify"] > 0:
            if classfile is None:
                message = f"Applying user classifier to plane {plane_number} from {user_classfile!s}..."
                console.echo(message=message, level=LogLevel.INFO)
                classfile = user_classfile

            stat = roi_stats(
                stat,
                Ly,
                Lx,
                aspect=ops.get("aspect", None),
                diameter=ops.get("diameter", None),
                do_crop=ops.get("soma_crop", 1),
            )
            if len(stat) == 0:
                iscell = np.zeros((0, 2))
            else:
                iscell = classify(stat=stat, classfile=classfile)
            np.save(Path(ops["save_path"]).joinpath("iscell.npy"), iscell)
            ic = (iscell[:, 0] > ops["preclassify"]).flatten().astype("bool")
            stat = stat[ic]
            message = (
                f"Plane {plane_number} preclassification pass with threshold {ops['preclassify']}: complete. Removed "
                f"{(~ic).sum()} ROIs."
            )
            console.echo(message=message, level=LogLevel.SUCCESS)

        stat = roi_stats(
            stat,
            Ly,
            Lx,
            aspect=ops.get("aspect", None),
            diameter=ops.get("diameter", None),
            max_overlap=ops["max_overlap"],
            do_crop=ops.get("soma_crop", 1),
        )
        message = f"Plane {plane_number} overlapping ROI filtering: complete. Kept {len(stat)} ROIs."
        console.echo(message=message, level=LogLevel.SUCCESS)

    # if second channel, detect bright cells in the second channel
    if "meanImg_chan2" in ops:
        if "chan2_thres" not in ops:
            ops["chan2_thres"] = 0.65
        ops, redcell = chan2detect.detect(ops, stat)
        np.save(Path(ops["save_path"]).joinpath("redcell.npy"), redcell)

    return ops, stat


def select_rois(ops: dict[str, Any], mov: np.ndarray, plane_number: int, sparse_mode: bool = True):
    if sparse_mode:
        ops.update({"Lyc": mov.shape[1], "Lxc": mov.shape[2]})
        new_ops, stat = sparsedetect.sparsery(
            mov=mov,
            high_pass=ops["high_pass"],
            neuropil_high_pass=ops["spatial_hp_detect"],
            batch_size=ops["batch_size"],
            spatial_scale=ops["spatial_scale"],
            threshold_scaling=ops["threshold_scaling"],
            max_iterations=250 * ops["max_iterations"],
            percentile=ops.get("active_percentile", 0.0),
            plane_number=plane_number,
        )
        ops.update(new_ops)
    else:
        ops, stat = sourcery.sourcery(mov=mov, ops=ops, plane_number=plane_number)

    stat = np.array(stat)

    if len(stat) == 0:
        raise ValueError("no ROIs were found -- check registered binary and maybe change spatial scale")

    return stat
