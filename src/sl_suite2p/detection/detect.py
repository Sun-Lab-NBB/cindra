"""Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu."""

from typing import Any
from pathlib import Path

import numpy as np
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import LogLevel, console

from . import chan2detect, sparsedetect
from .stats import roi_stats
from .denoise import pca_denoise
from ..io.binary import BinaryFile
from ..configuration import generate_default_ops
from ..classification import classify, user_classfile


def detect(ops, plane_number: int, classfile=None):
    timer = PrecisionTimer("s")

    bin_size = int(
        max(1, ops["frame_count"] // ops["maximum_binned_frames"], np.round(ops["tau"] * ops["sampling_rate"]))
    )

    console.echo(f"Binning plane {plane_number} movie in chunks of length {bin_size}...", level=LogLevel.INFO)
    timer.reset()

    with BinaryFile(file_path=ops["registered_binary_path"], height=ops["frame_height"], width=ops["frame_width"]) as f:
        mov = f.bin_movie(
            bin_size=bin_size,
            bad_frames=ops.get("badframes"),
            y_range=ops["valid_y_range"],
            x_range=ops["valid_x_range"],
        )

        message = (
            f"Plane {plane_number} movie: binned. Resultant movie dimensions: "
            f"{mov.shape[0]}, {mov.shape[1]}, {mov.shape[2]}. Time taken: {timer.elapsed} seconds."
        )
        console.echo(message=message, level=LogLevel.SUCCESS)

        ops, stat = detection_wrapper(f, plane_number=plane_number, mov=mov, ops=ops, classfile=classfile)

    return ops, stat


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

    stat : dictionary "y_pixels", "x_pixels", "pixel_weights"
            Dictionary containing statistics for ROIs


    """
    timer = PrecisionTimer("s")
    n_frames, Ly, Lx = f_reg.shape
    yrange = ops.get("valid_y_range", [0, Ly]) if yrange is None else yrange
    xrange = ops.get("valid_x_range", [0, Lx]) if xrange is None else xrange
    ops["valid_y_range"] = yrange
    ops["valid_x_range"] = xrange

    if mov is None:
        bin_size = int(max(1, n_frames // ops["maximum_binned_frames"], np.round(ops["tau"] * ops["sampling_rate"])))
        console.echo(f"Binning plane {plane_number} movie in chunks of length {bin_size}...", level=LogLevel.INFO)

        timer.reset()
        mov = f_reg.bin_movie(
            bin_size=bin_size,
            y_range=yrange,
            x_range=xrange,
            bad_frames=ops.get("badframes", None),
        )

        message = (
            f"Plane {plane_number} movie: binned. Resultant movie dimensions: "
            f"{mov.shape[0]}, {mov.shape[1]}, {mov.shape[2]}. Time taken: {timer.elapsed} seconds."
        )
        console.echo(message=message, level=LogLevel.SUCCESS)
    elif mov.shape[1] != yrange[-1] - yrange[0]:
        message = (
            f"Unable to run ROI detection. Movie height ({mov.shape[1]}) "
            f"does not match yrange size ({yrange[-1] - yrange[0]})."
        )
        console.error(message=message, error=ValueError)
    elif mov.shape[2] != xrange[-1] - xrange[0]:
        message = (
            f"Unable to run ROI detection. Movie width ({mov.shape[2]}) "
            f"does not match xrange size ({xrange[-1] - xrange[0]})."
        )
        console.error(message=message, error=ValueError)

    if "mean_image" not in ops:
        ops["mean_image"] = mov.mean(axis=0)
        ops["maximum_projection"] = mov.max(axis=0)

    if ops.get("inverted_activity", False):
        mov -= mov.min()
        mov *= -1
        mov -= mov.min()

    if ops.get("denoise", 1):
        mov = pca_denoise(mov, block_size=[ops["block_size"][0] // 2, ops["block_size"][1] // 2], n_comps_frac=0.5)

    message = f"Finding cell mask ROIs for plane {plane_number}..."
    console.echo(message=message, level=LogLevel.INFO)
    timer.reset()
    stat = select_rois(ops=ops, mov=mov, plane_number=plane_number)
    message = (
        f"Plane {plane_number} cell masks: discovered. Detected ROIs: {len(stat)}. Time taken: {timer.elapsed} seconds."
    )
    console.echo(message=message, level=LogLevel.SUCCESS)

    ymin = int(yrange[0])
    xmin = int(xrange[0])
    if len(stat) > 0:
        for s in stat:
            s["y_pixels"] += ymin
            s["x_pixels"] += xmin
            s["centroid"][0] += ymin
            s["centroid"][1] += xmin

        if ops["preclassification_threshold"] > 0:
            if classfile is None:
                classfile = user_classfile

            message = f"Applying classifier {Path(classfile).name} to plane {plane_number}..."
            console.echo(message=message, level=LogLevel.INFO)

            stat = roi_stats(
                stat,
                Ly,
                Lx,
                aspect=ops.get("aspect_ratio", None),
                diameter=ops.get("cell_diameter", None),
                do_crop=ops.get("crop_to_soma", 1),
            )
            if len(stat) == 0:
                iscell = np.zeros((0, 2))
            else:
                iscell = classify(stat=stat, classfile=classfile)
            np.save(Path(ops["output_directory"]).joinpath("iscell.npy"), iscell)
            ic = (iscell[:, 0] > ops["preclassification_threshold"]).flatten().astype("bool")
            stat = stat[ic]
            message = (
                f"Plane {plane_number} preclassification pass with threshold {ops['preclassification_threshold']}: complete. Removed "
                f"{(~ic).sum()} ROIs."
            )
            console.echo(message=message, level=LogLevel.SUCCESS)

        stat = roi_stats(
            stat,
            Ly,
            Lx,
            aspect=ops.get("aspect_ratio", None),
            diameter=ops.get("cell_diameter", None),
            max_overlap=ops["maximum_overlap"],
            do_crop=ops.get("crop_to_soma", 1),
        )
        message = f"Plane {plane_number} overlapping ROI filtering: complete. Kept {len(stat)} ROIs."
        console.echo(message=message, level=LogLevel.SUCCESS)

    # if second channel, detect bright cells in the second channel
    if "mean_image_channel_2" in ops:
        if "chan2_thres" not in ops:
            ops["chan2_thres"] = 0.65
        ops, redcell = chan2detect.detect(ops, stat)
        np.save(Path(ops["output_directory"]).joinpath("redcell.npy"), redcell)

    return ops, stat


def select_rois(ops: dict[str, Any], mov: np.ndarray, plane_number: int):
    """Detects ROIs using sparse detection algorithm.

    Args:
        ops: Pipeline options dictionary containing detection parameters.
        mov: Binned movie array with shape (frames, height, width).
        plane_number: Index of the imaging plane being processed.

    Returns:
        Array of ROI statistics dictionaries.
    """
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

    # Sets the cell_diameter from the computed spatial_scale_pixels if not explicitly configured.
    if ops.get("cell_diameter", 0) == 0:
        ops["cell_diameter"] = ops["spatial_scale_pixels"]

    stat = np.array(stat)

    if len(stat) == 0:
        message = (
            "Unable to complete ROI detection. No ROIs found. "
            "Check the binary file and consider adjusting the spatial scale parameter."
        )
        console.error(message=message, error=ValueError)

    return stat
