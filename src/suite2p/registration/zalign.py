"""This module provides functionality to compute per-plane correlation profiles for estimating z-positions in
imaging data."""

from typing import Any
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import LogLevel, console

from . import rigid, utils


def compute_z_position(
    z_stack: NDArray[np.int16] | NDArray[np.float32], ops: dict[str, Any], file_path: Path | None = None
) -> tuple[dict[str, Any], NDArray[np.float32]]:
    """Computes per-frame z-plane correlation scores for a stack of frames relative to a provided z-stack.

    Args:
        z_stack: A 3D NumPy array storing the z-stack.
        ops: The dictionary that stores the suite2p processing parameters.
        file_path: The file path to the binary file. If not provided, uses the file path specified in ops["reg_file"].
            The default is None.

    Returns:
        A tuple of two elements. The first element is a copy of the original 'ops' dictionary augmented with the
        additional key "zcorr". The second element is a 2D NumPy array storing the computed per-plane correlation scores
        for each frame.

    Raises:
        IOError: If no binary file path is provided in either the 'file_path' argument or ops["reg_file"].
    """
    # Ensures a binary file path provided as an argument or is specified in the input 'ops' dictionary.
    if file_path is None and "reg_file" not in ops:
        message = f"No binary file path was specified in either the 'file_path' argument or the input 'ops' dictionary."
        console.error(message=message, error=IOError)
        raise IOError(message)

    # Initializes the run timer.
    timer = PrecisionTimer("s")

    # Extracts batch processing parameters for reading the binary movie.
    batch_number = ops["batch_size"]
    height = ops["Ly"]
    width = ops["Lx"]
    batch_byte_number = 2 * height * width * batch_number

    # Copies the original 'ops' dictionary to preserve original settings.
    original_ops = ops.copy()

    # Disables non-rigid registration for this computation.
    ops["nonrigid"] = False

    # Extracts the number of z-planes and the height and width of the z-stack.
    plane_number, z_height, z_width = z_stack.shape

    # TODO: how should dimension mismatches actually be handled?
    if z_height > height or z_width != width:
        z_stack = z_stack[:,]

    # Extracts the path to the registration file from the input 'ops' dictionary if 'reg_file' was not provided.
    file_path = ops["reg_file"] if file_path is None else file_path

    # Queries the number of bytes in the file.
    byte_number = file_path.stat().st_size

    # Computes the number of frames in the binary movie based on the number of bytes.
    frame_number = int(byte_number / (2 * height * width))

    # Initializes a list to store the taper masks, mask offsets, and FFT reference images for each plane in the z-stack.
    reference_and_masks = []

    # Loops over the z-planes in the z-stack.
    for z_plane in z_stack:
        # Applies pre-processing steps to the current z-plane for one-photon recordings if enabled.
        if ops["one_p_reg"]:
            # Converts the z-plane to float32.
            z_plane = z_plane.astype(np.float32)

            # Applies spatial smoothing before high-pass filtering if enabled.
            if ops["pre_smooth"]:
                z_plane = utils.spatial_smooth(z_plane, int(ops["pre_smooth"]))

            # Applies spatial high-pass filtering to enhance spatial features.
            z_plane = utils.spatial_high_pass(z_plane, ops["spatial_hp_reg"])

        # Computes the taper mask and mask offset for rigid registration.
        taper_mask, mask_offset = rigid.compute_masks(
            reference_image=z_plane,
            mask_slope=ops["spatial_taper"] if ops["one_p_reg"] else 3 * ops["smooth_sigma"],
        )

        # Computes the FFT of the reference image, applying Gaussian smoothing with the sigma specified in the input
        # 'ops' dictionary.
        fft_reference_image = rigid.fft_reference_image(reference_image=z_plane, smooth_sigma=ops["smooth_sigma"])

        # Appends a tuple of the current z-plane's taper mask, mask offset, and FFT reference image to
        # 'reference_and_masks'.
        reference_and_masks.append((taper_mask, mask_offset, fft_reference_image))

    # Initializes a NumPy array to store registration shifts per plane and frame.
    z_correlations = np.zeros((plane_number, frame_number), np.float32)

    # Resets run timer.
    timer.reset()

    # Opens the binary file for reading in binary mode.
    with open(file_path, "rb") as file:
        # Loops until z-planes from the binary file are processed.
        start_index = 0  # Determines the index from which to start reading the frames
        while True:
            # Reads a batch of frames from the binary file.
            frames = np.frombuffer(file.read(batch_byte_number), dtype=np.int16).copy()

            # If there are no more frames to read or all frames were already processes, ends the runtime.
            if (frames.size == 0) | (start_index >= ops["nframes"]):
                break

            # Reshapes the batch of frames and converts to float32.
            frames = np.reshape(frames, (-1, height, width)).astype(np.float32)

            # Determines the frame indices for this batch.
            frame_indices = np.arange(start_index, start_index + frames.shape[0], 1, int)

            # Applies pre-processing steps to the current batch of frames for one-photon recordings if enabled.
            if ops["one_p_reg"]:
                # Applies spatial smoothing before high-pass filtering if enabled.
                if ops["pre_smooth"]:
                    frames = utils.spatial_smooth(frames, int(ops["pre_smooth"]))

                # Applies spatial high-pass filtering to enhance spatial features.
                frames = utils.spatial_high_pass(frames, ops["spatial_hp_reg"])

            # Loops over the z-planes in the z-stack to compute per-plane rigid registration shifts.
            for z_plane_index, reference_and_mask in enumerate(reference_and_masks):
                # Extracts the taper mask, mask offset, and FFT reference image of the current z-plane.
                taper_mask, mask_offset, fft_reference_image = reference_and_mask

                # Performs rigid phase correlation to compute correlation scores.
                _, _, z_correlations[z_plane_index, frame_indices] = rigid.phase_correlate(
                    frames=rigid.apply_masks(frames=frames, mask_multiplier=taper_mask, mask_offset=mask_offset),
                    fft_reference_image=fft_reference_image,
                    maximum_shift=ops["maxregshift"],
                    smooth_sigma_time=ops["smooth_sigma_time"],
                )

                # Logs progress every 10 planes.
                if z_plane_index % 10 == 1:
                    console.echo(
                        f"{z_plane_index}/{plane_number} planes, {start_index}/{ops['nframes']} frames processed, "
                        f"{timer.elapsed:.2f} sec elapsed.",
                        level=LogLevel.INFO,
                    )
            # Updates frame counter.
            start_index += frames.shape[0]

    # Saves the computed z-stack correlation scores in the input 'ops' dictionary.
    original_ops["zcorr"] = z_correlations

    console.echo(f"Z-position computation complete. Time taken: {timer.elapsed:.2f} seconds.", level=LogLevel.SUCCESS)

    # Returns the original 'ops' dictionary and the computed registration correlations.
    return original_ops, z_correlations
