"""Copyright © 2023 Howard Hughes Medical Institute, Authored by Carsen Stringer and Marius Pachitariu."""

from os import path
from typing import Any, TypeAlias
from warnings import warn

from tqdm import tqdm
import numpy as np
from numpy.typing import NDArray
from scipy.signal import medfilt, medfilt2d
from ataraxis_time import PrecisionTimer
from ataraxis_base_utilities import LogLevel, console

from . import (
    rigid,
    utils,
    nonrigid,
    bidiphase as bidi,
)
from .. import io
from ..configuration import generate_default_ops

Blocks: TypeAlias = tuple[
    list[NDArray[np.int_]],  # y_blocks
    list[NDArray[np.int_]],  # x_blocks
    list[int],  # [y_block_number, x_block_number]
    tuple[int, int],  # block_size
    NDArray[np.float32],  # smoothing_kernel
]

ReferenceAndMasks: TypeAlias = tuple[
    NDArray[np.float32],  # taper_mask
    NDArray[np.float32],  # mask_offset
    NDArray[np.complex64],  # fft_reference_image
    NDArray[np.float32],  # nonrigid_taper_mask
    NDArray[np.float32],  # nonrigid_mask_offset
    NDArray[np.complex64],  # nonrigid_fft_reference_image
    Blocks,  # blocks
]

RegisteredFrames: TypeAlias = tuple[
    NDArray[np.int16] | NDArray[np.float32],  # frames
    NDArray[np.int32],  # rigid_y_shifts
    NDArray[np.int32],  # rigid_x_shifts
    NDArray[np.float32],  # rigid_correlations
    NDArray[np.float32],  # nonrigid_y_shifts
    NDArray[np.float32],  # nonrigid_x_shifts
    NDArray[np.float32],  # nonrigid_correlations
    tuple[  # z_estimates
        NDArray[np.int_],  # z_plane_indices
        NDArray[np.float32],  # z_plane_correlations
    ]
    | None,
]

# RegistrationOutput: TypeAlias = tuple[
#     NDArray[np.int16],  # reference_image
#     NDArray[np.int16],  # minimum_intensity,
#     NDArray[np.int16],  #  maximum_intensity,
#         channel_1_mean_image,
#         rigid_shifts,
#         nonrigid_shifts,
#         z_estimates,
#         channel_2_mean_image,
#         bad_frames,
#         y_range,
#         x_range,
# ]


def _compute_initial_reference(frames: NDArray[np.int16], correlated_frames_number: int = 20) -> NDArray[np.float32]:
    """Computes the initial reference image by finding the average of the seed frame with its top
    'correlated_frames_number' most correlated frames.

    The seed frame is the frame with the highest average correlation to all other frames.

    Args:
        frames: A 3D NumPy array storing the input frames.
        correlated_frames_number: The number of top correlated frames to include in the computation of the reference
            image. The default value is 20.

    Returns:
        A 2D NumPy array of shape (height, width) storing the computed initial reference image.
    """
    # Extracts the number of frames, height, and width from the input frames.
    frame_number, height, width = frames.shape

    # Flattens the frames into 2D and converts to float32 for correlation computation.
    frames = np.reshape(frames, (frame_number, -1)).astype("float32")

    # Centers the frames by subtracting the mean of each frame.
    frames = frames - np.reshape(frames.mean(axis=1), (frame_number, 1))

    # Computes the correlation matrix between frames.
    correlation_matrix = np.matmul(frames, frames.T)

    # Normalizes the correlation matrix by diving by the norms.
    correlation_matrix_norm = np.sqrt(np.diag(correlation_matrix))  # Norm of each frame
    correlation_matrix = correlation_matrix / np.outer(correlation_matrix_norm, correlation_matrix_norm)

    # Extracts the top 'correlated_frames_number' correlations (excluding self-correlation).
    top_correlations = np.partition(correlation_matrix, -(correlated_frames_number + 1), axis=1)[
        :, -correlated_frames_number:-1
    ]

    # Computes the mean of the top correlations for each frame to identify the seed frame.
    average_top_correlations = np.mean(top_correlations, axis=1)

    # Identifies the seed frame (frame with the highest average correlation with other frames) and extracts its index.
    seed_frame_index = np.argmax(average_top_correlations)

    # Extracts the indices of the top 'correlated_frame_number' most correlated frames with the seed frame.
    top_correlation_indices = np.argpartition(correlation_matrix[seed_frame_index, :], -correlated_frames_number)[
        -correlated_frames_number:
    ]  # top k correlations for seed frame

    # Computes the reference image by averaging the top correlated frames.
    reference_image = np.mean(frames[top_correlation_indices, :], axis=0)

    # Reshapes the reference image back to its 2D shape.
    reference_image = np.reshape(reference_image, (height, width))

    # Returns the initial reference image.
    return reference_image


def _compute_reference_image(
    frames: NDArray[np.int16], ops: dict[str, Any] = generate_default_ops()
) -> NDArray[np.int16]:
    """Computes the reference image used for registration by iteratively aligning subsets of frames.

    This function selects an initial reference image, applies optional spatial preprocessing  (e.g., smoothing and
    spatial filtering and tapering), and then refines the reference image through iterative rigid registration. During
    each iteration, the frames are aligned to the current reference image, and a new reference image is computed from the
    best-aligned frames.

    Args:
        frames: A 3D NumPy array storing the frames to compute the reference image from.
        ops: The dictionary that stores the suite2p processing parameters.

    Returns:
        A 2D NumPy array storing the computed reference image.
    """
    # Generates the initial reference image.
    initial_reference_image = _compute_initial_reference(frames=frames)

    # If high-pass spatial filtering is enabled, checks if Gaussian smoothing is enabled.
    if ops["one_p_reg"]:
        # If the "pre_smooth" key has a value greater than 0.0, applies spatial smoothing to the reference image and
        # frames using the "pre_smooth" configured value as the window size.
        if ops["pre_smooth"]:
            initial_reference_image = utils.spatial_smooth(
                frames=initial_reference_image, window_size=int(ops["pre_smooth"])
            )
            frames = utils.spatial_smooth(frames=frames, window_size=int(ops["pre_smooth"]))

        # Applies high-pass filtering to the reference image and frames to remove low-frequency background signals.
        initial_reference_image = utils.spatial_high_pass(initial_reference_image, int(ops["spatial_hp_reg"]))
        frames = utils.spatial_high_pass(frames, int(ops["spatial_hp_reg"]))

    # Loops over the number of iterations and performs rigid registration using phase correlation.
    iteration_number = 8  # Number of registration refinement iterations
    for iteration in range(iteration_number):
        # Calculates the most optimal y- and x-shifts and correlation values based on the input 'ops' dictionary
        # configurations.
        rigid_y_shifts, rigid_x_shifts, rigid_correlations = rigid.phase_correlate(
            # Applies the taper mask to the frames using the computed taper mask and mask offset.
            frames=rigid.apply_masks(
                frames,
                # Computes the taper mask and mask offset. Uses the value for "spatial_taper" for the mask slope if
                # high-pass spatial filtering is enabled. Otherwise, uses the value of "smooth_sigma" * 3.
                *rigid.compute_masks(
                    reference_image=initial_reference_image,
                    mask_slope=ops["spatial_taper"] if ops["one_p_reg"] else 3 * ops["smooth_sigma"],
                ),
            ),
            # Computes the Fast Fourier Transform (FFT) of the reference image and applies a Gaussian filter in the
            # Fourier domain using the configured "smooth_sigma" value.
            fft_reference_image=rigid.fft_reference_image(
                reference_image=initial_reference_image,
                smooth_sigma=ops["smooth_sigma"],
            ),
            maximum_shift=ops["maxregshift"],
            smooth_sigma_time=ops["smooth_sigma_time"],
        )

        # Applies the computed y- and x-shifts to align each frame.
        for frame, y_shift, x_shift in zip(frames, rigid_y_shifts, rigid_x_shifts):
            frame[:] = rigid.shift_frame(frame=frame, y_shift=y_shift, x_shift=x_shift)

        # Determines the number of top correlated frames to use for updating the reference image. This number increases
        # with each iteration to progressively incorporate more data. Ensures at least 2 frames are used.
        best_frames_number = max(2, int(frames.shape[0] * (1.0 + iteration) / (2 * iteration_number)))

        # Sorts the frames in descending order of the correlation values, and extracts the indices of the frames with
        # the highest correlation to the current reference image. Excludes the the top frame as it is often overfit.
        best_frames_indices = np.argsort(-rigid_correlations)[1:best_frames_number]

        # Computes the new reference image as the mean of the top correlated frames.
        reference_image: NDArray[np.int16] = frames[best_frames_indices].mean(axis=0).astype(np.int16)

        # Shifts the reference image to the position of the mean y- and x-shifts. This re-centers the reference image
        # in order to maintain alignment consistency across iterations.
        reference_image = rigid.shift_frame(
            frame=reference_image,
            y_shift=int(np.round(-rigid_y_shifts[best_frames_indices].mean())),
            x_shift=int(np.round(-rigid_x_shifts[best_frames_indices].mean())),
        )

    # Returns the final reference image.
    return reference_image


def _compute_reference_and_masks(
    reference_image: NDArray[np.int16] | list[NDArray[np.int16]],
    ops: dict[str, Any] = generate_default_ops(),
) -> ReferenceAndMasks | list[ReferenceAndMasks]:
    """Computes the registration masks and Fourier-transformed reference image for rigid and non-rigid registration.

    If a list of reference images is passed, the function is recursively called on each element.

    Args:
        reference_image: A 2D NumPy array storing the reference image used for image registration.
        ops: The dictionary that stores the suite2p processing parameters.

    Returns:
        A tuple of seven elements. The first element is a 2D NumPy array storing the mask multiplier applied to the
        frames. The second element is a 2D NumPy array storing the mask offset applied to the frames. The third element
        is a 2D NumPy array of complex values storing the Fourier-transformed reference image. If non-rigid registration
        is disabled in the 'ops' dictionary, the fourth, fifth, and sixth elements are empty NumPy arrays. Otherwise,
        the fourth element is a 4D NumPy array storing the taper mask for each block. The fifth element is a 4D NumPy
        array storing the mask offset for each block of the reference image. The sixth element is a 4D NumPy array
        storing the Fourier-transformed reference image blocks. The seventh element is a tuple of five elements storing
        the block boundaries and sizes.

        If a list of reference images is passed, the function returns a list of the seven-element tuples.

    Raises:
        TypeError: If the recursive call returns a list of ReferenceAndMasks instead of a single instance.
    """
    # If there are multiple reference images, calls this function recursively to process each reference image
    # individually.
    if isinstance(reference_image, list):
        all_reference_and_masks = []  # Initializes a list to store all of the returned data

        # Loops over the reference images.
        for image in reference_image:
            # Recursively computes the registration masks and Fourier-transformed reference image.
            reference_and_masks = _compute_reference_and_masks(reference_image=image, ops=ops)

            # If the computed registration masks and Fourier-transformed reference image is returned as a list, raises
            # a TypeError.
            if isinstance(reference_and_masks, list):
                message = f"Computed registration masks and reference image should be a single instance, not a list."
                console.error(message=message, error=TypeError)
                raise TypeError(message)

            # Appends the computed result to the list that is eventually returned.
            all_reference_and_masks.append(reference_and_masks)

        # Returns the list of the combined data.
        return all_reference_and_masks

    # Computes the spatial taper mask and mask offsets. Uses the value for "spatial_taper" for the mask slope if
    # high-pass spatial filtering is enabled. Otherwise, uses the value of "smooth_sigma" * 3.
    taper_mask, mask_offset = rigid.compute_masks(
        reference_image=reference_image,
        mask_slope=ops["spatial_taper"] if ops["one_p_reg"] else 3 * ops["smooth_sigma"],
    )

    # Computes the Fast Fourier Transform (FFT) of the reference image, optionally smoothed if "smooth_sigma" is
    # configured.
    fft_reference_image = rigid.fft_reference_image(reference_image=reference_image, smooth_sigma=ops["smooth_sigma"])

    # Extracts the height and width of the input reference image.
    height, width = reference_image.shape

    # If non-rigid registration is enabled in 'ops', computes the block masks and offsets and Fourier-transformed
    # reference image.
    if ops.get("nonrigid"):
        # Computes blocks to split the field of view (FOV).
        blocks = nonrigid.make_blocks(height=height, width=width, block_size=ops["block_size"])

        # Computes the block-wise taper masks and FFTs to prepare for non-rigid registration.
        nonrigid_taper_mask, nonrigid_mask_offset, nonrigid_fft_reference_image = nonrigid.fft_reference_image(
            reference_image=reference_image,
            mask_slope=ops["spatial_taper"]
            if ops["one_p_reg"]
            else 3 * ops["smooth_sigma"],  # slope of taper mask at the edges
            smooth_sigma=ops["smooth_sigma"],
            y_blocks=blocks[0],
            x_blocks=blocks[1],
        )
    # Otherwise, stores empty NumPy arrays in the return variables.
    else:
        nonrigid_taper_mask = np.empty((0, 0, 0, 0), dtype=np.float32)
        nonrigid_mask_offset = np.empty((0, 0, 0, 0), dtype=np.float32)
        nonrigid_fft_reference_image = np.empty((0, 0, 0, 0), dtype=np.complex64)

    # Return alls computed rigid and nonrigid masks, Fourier-transformed reference images, and block information.
    return (
        taper_mask,
        mask_offset,
        fft_reference_image,
        nonrigid_taper_mask,
        nonrigid_mask_offset,
        nonrigid_fft_reference_image,
        blocks,
    )


def _register_frames(
    reference_and_masks: ReferenceAndMasks | NDArray[np.float32],
    frames: NDArray[np.int16] | NDArray[np.float32],
    minimum_intensity: np.int16,
    maximum_intensity: np.int16,
    bidirectional_phase_offset: int = 0,
    ops: dict[str, Any] = generate_default_ops(),
) -> RegisteredFrames:
    """Registers a stack of frames to a reference image using rigid and optionally non-rigid registration.

    Args:
        reference_and_masks: Either a 2D NumPy array storing a raw reference image, or a tuple of precomputed
            registration masks and the Fourier-transformed reference image.
        frames: A 3D NumPy array storing the stack of frames to register.
        minimum_intensity: The minimum intensity value for frame clipping.
        maximum_intensity: The maximum intensity value for frame clipping.
        bidirectional_phase_offset: The estimated pixel offset along the x-axis between even and odd scan lines. The
            default is 0.
        ops: The dictionary that stores the suite2p processing parameters.

    Returns:
        A tuple of eight elements. The first element stores the registered frames. The second and third elements store
        the rigid y- and x-shifts respectively for each frame. The fourth element stores the rigid correlation values.
        The fifth and sixth elements store the non-rigid y- and x-shifts respectively for each frame. The seventh
        element stores the non-rigid correlation values. The eighth element is an instance of None.
    """
    # If the 'reference_and_masks' input is already a pre-computed tuple of reference masks and images, unpacks the
    # component directly in order to avoid recomputing the masks.
    if not isinstance(reference_and_masks, tuple) or len(reference_and_masks) == 7:
        (
            taper_mask,
            mask_offset,
            fft_reference_image,
            nonrigid_taper_mask,
            nonrigid_mask_offset,
            nonrigid_fft_reference_image,
            blocks,
        ) = reference_and_masks

    # If the input is the raw reference image, computes the registration masks and Fourier-transformed reference
    # image.
    else:
        reference_image = reference_and_masks

        # If normalization is enabled and "rmin" is not specified in the input 'ops' dictionary, estimates the
        # clipping bounds using percentiles of the reference image.
        if ops.get("norm_frames", False) and "rmin" not in ops:
            minimum_intensity, maximum_intensity = (
                np.int16(np.percentile(reference_image, 1)),
                np.int16(np.percentile(reference_image, 99)),
            )
            reference_image = np.clip(reference_image, minimum_intensity, maximum_intensity)

        # Computes the registration masks and Fourier-transformed reference image.
        (
            taper_mask,
            mask_offset,
            fft_reference_image,
            nonrigid_taper_mask,
            nonrigid_mask_offset,
            nonrigid_fft_reference_image,
            blocks,
        ) = _compute_reference_and_masks(reference_image=reference_image, ops=ops)

    # If the offset is non-zero, applies the bidirectional phase shift to the frames.
    if bidirectional_phase_offset != 0:
        bidi.shift(frames, bidirectional_phase_offset)

    # If smoothing or filtering is enabled, casts the frames to float32 and copies the frames. Otherwise, copies
    # the frames as their original data type.
    dtype = "float32" if ops["smooth_sigma_time"] > 0 or ops["one_p_reg"] else frames.dtype
    smooth_frames = frames.copy().astype(dtype) if ops["smooth_sigma_time"] > 0 or ops["one_p_reg"] else frames

    # If smoothing is enabled, applies temporal smoothing to the frames. Otherwise, directly uses the input frames.
    if ops["smooth_sigma_time"]:
        smooth_frames = utils.temporal_smooth(frames=smooth_frames, sigma=ops["smooth_sigma_time"])
    else:
        smooth_frames = frames

    # Applies additional pre-processing for one-photon recordings.
    if ops["one_p_reg"]:
        # If pre-smoothing is configured in the 'ops' dictionary, applies spatial smoothing to the frames.
        if ops["pre_smooth"]:
            smooth_frames = utils.spatial_smooth(smooth_frames, int(ops["pre_smooth"]))

        # Applies high-pass filtering to the frames to remove low-frequency background signals.
        smooth_frames = utils.spatial_high_pass(smooth_frames, int(ops["spatial_hp_reg"]))

    # Performs rigid phase correlation in order to compute the maximum y- and x-shifts and correlations between the
    # stack of frames and the reference image.
    rigid_y_shifts, rigid_x_shifts, rigid_correlations = rigid.phase_correlate(
        # Applies the rigid registration masks.
        frames=rigid.apply_masks(
            # Clips the intensity range if valid minimum and maximum intensity bounds are provided.
            frames=np.clip(smooth_frames, minimum_intensity, maximum_intensity)
            if minimum_intensity > -np.inf
            else smooth_frames,
            mask_multiplier=taper_mask,
            mask_offset=mask_offset,
        ),
        fft_reference_image=fft_reference_image,
        maximum_shift=ops["maxregshift"],
        smooth_sigma_time=ops["smooth_sigma_time"],
    )

    # Applies the computed rigid y- and x-shifts to the original stack of frames.
    for frame, y_shift, x_shift in zip(frames, rigid_y_shifts, rigid_x_shifts):
        frame[:] = rigid.shift_frame(frame=frame, y_shift=y_shift, x_shift=x_shift)

    # Performs non-rigid registration if enabled in the input 'ops' dictionary.
    if ops["nonrigid"]:
        # If smoothing or high-pass spatial filtering were enabled in the input 'ops' dictionary,
        # applies the same rigid y- and x-shifts to the smoothed frames as well.
        if ops["smooth_sigma_time"] or ops["one_p_reg"]:
            for frame, y_shift, x_shift in zip(smooth_frames, rigid_y_shifts, rigid_x_shifts):
                frame[:] = rigid.shift_frame(frame=frame, y_shift=y_shift, x_shift=x_shift)

        # Performs non-rigid phase correlation in order to compute the maximum y- and x-shifts and correlations
        # between the stack of frames and the reference image.
        nonrigid_y_shifts, nonrigid_x_shifts, nonrigid_correlations = nonrigid.phase_correlate(
            # Clips the intensity range if valid minimum and maximum intensity bounds are provided.
            frames=np.clip(smooth_frames, minimum_intensity, maximum_intensity)
            if minimum_intensity > -np.inf
            else smooth_frames,
            taper_mask=nonrigid_taper_mask.squeeze(),
            mask_offset=nonrigid_mask_offset.squeeze(),
            fft_reference_image=nonrigid_fft_reference_image.squeeze(),
            snr_threshold=ops["snr_thresh"],
            smoothing_kernel=blocks[-1],
            y_blocks=blocks[0],
            x_blocks=blocks[1],
            maximum_shift=ops["maxregshiftNR"],
        )

        # Applies the computed non-rigid y- and x-shifts to each block of each frame.
        frames = nonrigid.transform_frames(
            frames=frames,
            y_blocks=blocks[0],
            x_blocks=blocks[1],
            block_number=blocks[2],
            y_shifts=nonrigid_y_shifts,
            x_shifts=nonrigid_x_shifts,
        )

    # If non-rigid registration is disabled, stores None in the corresponding return variables.
    else:
        nonrigid_y_shifts = np.empty((0,), dtype=np.float32)
        nonrigid_x_shifts = np.empty((0,), dtype=np.float32)
        nonrigid_correlations = np.empty((0,), dtype=np.float32)

    # Returns the registered frames and all computed shifts and correlation values.
    return (
        frames,
        rigid_y_shifts,
        rigid_x_shifts,
        rigid_correlations,
        nonrigid_y_shifts,
        nonrigid_x_shifts,
        nonrigid_correlations,
        None,
    )


def _register_multi_plane_frames(
    reference_and_masks: list[ReferenceAndMasks],
    frames: NDArray[np.int16],
    minimum_intensity: NDArray[np.int16],
    maximum_intensity: NDArray[np.int16],
    bidirectional_phase_offset: int = 0,
    ops: dict[str, Any] = generate_default_ops(),
    z_plane_number: int = 1,
) -> RegisteredFrames:
    """Registers a stack of frames to a set of reference images using rigid and optionally non-rigid registration.

    Args:
        reference_and_masks: A list of tuples storing the precomputed registration masks and the Fourier-transformed
            reference images.
        frames: A 3D NumPy array storing the stack of frames to register.
        minimum_intensity: The minimum intensity value for frame clipping.
        maximum_intensity: The maximum intensity value for frame clipping.
        bidirectional_phase_offset: The estimated pixel offset along the x-axis between even and odd scan lines. The
        default is 0. ops: The dictionary that stores the suite2p processing parameters.
        z_plane_number: The number of z-planes in the reference stack. The default is 1.

    Returns:
        A tuple of eight elements. The first element stores the registered frames. The second and third elements store
        the rigid y- and x-shifts respectively for each frame. The fourth element stores the rigid correlation values.
        The fifth and sixth elements store the non-rigid y- and x-shifts respectively for each frame. The seventh
        element stores the non-rigid correlation values. The eighth element is a tuple of two elements storing TODO.
    """
    # Initializes a NumPy array to store the best correlation score per frame across all z-planes.
    frame_maximum_correlations = -np.inf * np.ones(shape=len(frames), dtype="float32")

    # Initializes a 2D NumPy array to store the correlation score of each frame with each z-plane.
    z_plane_correlations = -np.inf * np.ones(shape=(len(frames), z_plane_number), dtype="float32")

    # Initializes a NumPy array to store the index of the z-plane with the highest correlation for each frame.
    z_plane_indices = np.zeros(shape=len(frames), dtype="int")

    # Temporarily stores whether non-rigid registration was originally enabled in the 'ops' dictionary.
    run_nonrigid = ops["nonrigid"]

    # Loops over the z-planes to perform rigid registration.
    for z_plane_index in range(z_plane_number):
        ops["nonrigid"] = False  # Disables non-rigid registration

        # Registers the frames against the current z-plane's reference image.
        outputs = _register_frames(
            reference_and_masks=reference_and_masks[z_plane_index],
            frames=frames.copy(),
            minimum_intensity=minimum_intensity[z_plane_index],
            maximum_intensity=maximum_intensity[z_plane_index],
            bidirectional_phase_offset=bidirectional_phase_offset,
            ops=ops,
        )

        # Stores the correlation scores of the current iteration of registration in the z-plane-specific index.
        z_plane_correlations[:, z_plane_index] = outputs[3]

        # On the first iteration, initializes the 'best_output' array as a copy of all of the outputs except for
        # the last four outputs.
        if z_plane_index == 0:
            best_outputs = list(outputs[:-4]).copy()

        # Identifies frames with the better correlation for this z-plane.
        better_correlations_mask = frame_maximum_correlations < z_plane_correlations[:, z_plane_index]

        # Updates the best z-plane index and maximum correlation score for those frames.
        z_plane_indices[better_correlations_mask] = z_plane_index
        frame_maximum_correlations[better_correlations_mask] = z_plane_correlations[
            better_correlations_mask, z_plane_index
        ]

        # Replaces the best output values for the frames where this z-plane yielded better results.
        for best_output, output in zip(best_outputs, outputs[:-4]):
            best_output[better_correlations_mask] = output[better_correlations_mask]

    # If non-rigid registration was originally enabled in the 'ops' dictionary, re-runs registration on each frame
    # individually using its best-matching z-plane reference.
    if run_nonrigid:
        ops["nonrigid"] = True  # Re-enables non-rigid registration

        # Extracts the number of frames from the input frames.
        frame_number = frames.shape[0]

        # Loops over the frames and performs non-rigid registration using its best z-plane.
        for frame_index, best_z_plane_index in enumerate(z_plane_indices):
            outputs = _register_frames(
                reference_and_masks=reference_and_masks[best_z_plane_index],
                frames=frames[[frame_index]],
                minimum_intensity=minimum_intensity[best_z_plane_index],
                maximum_intensity=maximum_intensity[best_z_plane_index],
                bidirectional_phase_offset=bidirectional_phase_offset,
                ops=ops,
            )

            # On the first iteration, initializes the output arrays for each return value.
            if frame_index == 0:
                best_outputs = []
                for output in outputs[:-1]:
                    best_outputs.append(np.zeros(shape=(frame_number, *output.shape[1:]), dtype=output.dtype))
                    best_outputs[-1][0] = output[0]

            # Otherwise, fills in the output arrays for the current frame.
            else:
                for best_output, output in zip(best_outputs, outputs[:-1]):
                    best_output[frame_index] = output[0]

    # Unpacks the list 'best_output' into its corresponding variables.
    # TODO: WHY AM I GETTING int32 HERE ????????
    (
        frames,
        maximum_y_shifts,
        maximum_x_shifts,
        maximum_correlations,
        nonrigid_y_shifts,
        nonrigid_x_shifts,
        nonrigid_correlations,
    ) = best_outputs  # type: ignore

    # Returns the registered frames, shifts, correlations, and z-plane information.
    return (  # type: ignore
        frames,
        maximum_y_shifts,
        maximum_x_shifts,
        maximum_correlations,
        nonrigid_y_shifts,
        nonrigid_x_shifts,
        nonrigid_correlations,
        (z_plane_indices, z_plane_correlations),
    )


def _normalize_reference_image(
    reference_image: NDArray[np.int16] | list[NDArray[np.int16]],
) -> (
    tuple[NDArray[np.int16], np.int16, np.int16] | tuple[list[NDArray[np.int16]], NDArray[np.int16], NDArray[np.int16]]
):
    """Normalizes a reference image (or list of reference images) by clipping pixel values between the 1st and 99th
    percentile intensity values.

    Args:
        reference_image: A 2D NumPy array or a list of such arrays, each representing a reference image.

    Returns:
        A tuple of three elements. If the input 'reference_image' is a single image, the first element is the
        normalized (clipped) reference image. The second and third elements are the 1st and 99th percentile values used
        for normalization. If the input 'reference_image' is a list, the first element is the list of normalized images.
        The second element is a NumPy array of per-image 1st percentile values, and the third element is a NumPy array
        of per-image 99th percentile values.
    """
    # If the input 'reference_image' is a list, normalizes each image in the list individually.
    if isinstance(reference_image, list):
        # Extracts the number of z-planes.
        z_plane_number = len(reference_image)

        # Initializes NumPy arrays to store the 1st and 99th percentile intensity values for each reference image.
        minimum_intensities = np.empty(z_plane_number, dtype=np.int16)
        maximum_intensities = np.empty(z_plane_number, dtype=np.int16)

        # Loops over the reference images.
        for image_index, image in enumerate(reference_image):
            # Computes the 1st and 99th percentile intensity values for the current image.
            minimum_intensity, maximum_intensity = np.percentile(image, [1, 99])

            # Stores the computed intensity bounds to the appropriate NumPy array and casts the values to np.float32.
            minimum_intensities[image_index] = np.int16(minimum_intensity)
            maximum_intensities[image_index] = np.int16(maximum_intensity)

            # Clips the reference image to the computed percentile bounds.
            image[:] = np.clip(image, minimum_intensities[image_index], maximum_intensities[image_index])

        # Returns the normalized reference images and their per-image clipping bounds.
        return reference_image, minimum_intensities, maximum_intensities

    # If the input 'reference_image' is a single image, computes the 1st and 99th percentile intensity values for the
    # single image.
    minimum_intensity, maximum_intensity = np.percentile(reference_image, [1, 99])

    # Casts the values to np.int16.
    minimum_intensity, maximum_intensity = np.int16(minimum_intensity), np.int16(maximum_intensity)

    # Clips the reference image to the computed percentile bounds.
    reference_image = np.clip(reference_image, minimum_intensity, maximum_intensity)

    # Returns the normalized reference image and the clipping bounds.
    return reference_image, minimum_intensity, maximum_intensity


def _compute_reference_and_register_frames(
    align_input_frames: io.BinaryFile,
    plane_number: int,
    align_output_frames: io.BinaryFile | None = None,
    reference_image: NDArray[np.int16] | list[NDArray[np.int16]] | None = None,
    ops: dict[str, Any] = generate_default_ops(),
    # ) -> (
    #     tuple[
    #         NDArray[np.int16],  # original_reference_image
    #         NDArray[np.int16],  # minimum_intensity
    #         NDArray[np.int16],  # maximum_intensity
    #         NDArray[np.float32],  # mean_image
    #         NDArray[np.float32],  # combined_rigid_shifts
    #         NDArray[np.float32],  # combined_nonrigid_shifts
    #         tuple[NDArray[np.int_], NDArray[np.float32]],  # (z_plane_indices, z_plane_correlations)
    #     ]
    #     | tuple[
    #         list[NDArray[np.int16]],  # original_reference_image
    #         list[NDArray[np.int16]],  # minimum_intensity
    #         list[NDArray[np.int16]],  # maximum_intensity
    #         NDArray[np.float32],  # mean_image
    #         tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]],  # combined_rigid_shifts
    #         tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]],  # combined_nonrigid_shifts
    #         tuple[NDArray[np.int_], NDArray[np.float32]],  # (z_plane_indices, z_plane_correlations)
    #     ]
):
    """Computes the reference image (if not provided) and registers all the input frames to the reference image.

    This function handles bidirectional phase correction if needed, normalizes the reference image, computes the
    rigid and optionally non-rigid registration shifts, and writes the shifted output to a separate output binary file
    or in-place if 'align_output_frames' is not provided. This function also optionally saves the registered frames as a
    tiff file if enabled in the input 'ops' dictionary.

    Args:
        align_input_frames: The input frames to be registered. It can be a BinaryFile or any type of array that can be
            slice-indexed.
        plane_number: The index of the imaging plane to process.
        align_output_frames: The variable where the registered frames will be stored. The default is None.
        reference_image: The reference image to align frames to. If None, the reference image is computed.
        ops: The dictionary that stores the suite2p processing parameters.

    Returns:
        A tuple of seven elements. TODO
    """
    # Initializes the run timer.
    timer = PrecisionTimer("s")

    # Extracts the number of frames, height, and width of the input frames.
    frame_number, height, width = align_input_frames.shape

    # Queries the batch size from the input 'ops' dictionary.
    batch_size = ops["batch_size"]

    # If the reference image is not provided, computes the reference image and optionally the bidirectional phase
    # shift if configured in the input 'ops' dictionary.
    if reference_image is None:
        # Extracts a subset of input frames to compute the reference image.
        frames = align_input_frames[:: int(np.minimum(ops["nimg_init"], frame_number))]

        # If bidirectional phase correction is required and has not already been corrected, computes it from the input
        # frames.
        if ops["do_bidiphase"] and ops["bidiphase"] == 0 and not ops["bidi_corrected"]:
            # Computes the bidirectional phase offset.
            bidirectional_phase_offset = bidi.compute(frames)

            console.echo(
                f"Plane {plane_number} estimated bidiphase offset from data: {bidirectional_phase_offset} pixels.",
                level=LogLevel.INFO,
            )

            # Updates the 'ops' dictionary with the bidirectional phase offset.
            ops["bidiphase"] = bidirectional_phase_offset

            # If the offset is non-zero, applies the bidirectional phase shift to the frames.
            if bidirectional_phase_offset != 0:
                bidi.shift(frames, int(ops["bidiphase"]))

        console.echo(f"Computing plane {plane_number} reference frame...", level=LogLevel.INFO)

        # Resets run timer.
        timer.reset()

        # Computes the reference image.
        reference_image = _compute_reference_image(frames=frames, ops=ops)

        console.echo(
            f"Plane {plane_number} reference frame: computed. Time taken: {timer.elapsed} seconds.",
            level=LogLevel.SUCCESS,
        )

    # Extracts the number of z-planes if the 'reference_image' is a list.
    z_plane_number = len(reference_image) if isinstance(reference_image, list) else 1

    console.echo(
        f"Generated a total of {z_plane_number} reference frames for plane {plane_number}.", level=LogLevel.INFO
    )

    # Copies the original reference image before normalization.
    original_reference_image = reference_image.copy()

    # Normalizes the reference image if enabled in the 'ops' dictionary.
    if ops.get("norm_frames", False):
        reference_image, minimum_intensity, maximum_intensity = _normalize_reference_image(
            reference_image=reference_image
        )
    else:
        # Sets the default minimum and maximum intensity as the smallest and biggest possible int16 values respectively
        # if normalization is disabled.
        if z_plane_number == 1:
            minimum_intensity, maximum_intensity = np.int16(-32768), np.int16(32767)
        else:
            minimum_intensity = np.full(z_plane_number, -32768, dtype="int16")
            maximum_intensity = np.full(z_plane_number, 32767, dtype="int16")

    # Queries the bidirectional phase offset from the input 'ops' dictionary if available and if not already corrected.
    # Otherwise, defaults to 0.
    bidirectional_phase_offset = int(ops["bidiphase"]) if ops["bidiphase"] and not ops["bidi_corrected"] else 0

    # Computes the registration masks and Fourier-transformed reference images for registration.
    reference_and_masks = _compute_reference_and_masks(reference_image=reference_image, ops=ops)

    # Initializes a 2D NumPy array to store the mean image from the frames.
    mean_image = np.zeros(shape=(height, width), dtype="float32")

    # Initializes lists to store the computed rigid and non-rigid offsets and other metrics, which will eventually be
    # returned.
    rigid_shifts, nonrigid_shifts, z_plane_indices, z_plane_correlations = [], [], [], []

    # Queries the number of frames to process if provided in the input 'ops' dictionary.
    if ops["frames_include"] != -1:
        frame_number = min(frame_number, ops["frames_include"])

    # Resets the run timer.
    timer.reset()
    console.echo(f"Computing plane {plane_number} frame registration offsets for channel 1...", level=LogLevel.INFO)

    # Registers frames in batches and displays a progress bar if configured.
    for batch_number in tqdm(
        np.arange(0, frame_number, batch_size),
        desc=f"Registering batches of {batch_size} frames",
        unit="batch",
        disable=not ops["progress_bars"],
    ):
        # Slices a batch of frames from the input frames.
        frames = align_input_frames[batch_number : min(batch_number + batch_size, frame_number)]

        # TODO: need to fix whatever i was doing here when i separated the function
        # Registers the frames to the reference image and computes the rigid and optionally non-rigid shifts.
        if (
            isinstance(reference_and_masks, list)
            and isinstance(minimum_intensity, list)
            and isinstance(maximum_intensity, list)
        ):
            (
                registered_frames,
                rigid_y_shifts,
                rigid_x_shifts,
                rigid_correlations,
                nonrigid_y_shifts,
                nonrigid_x_shifts,
                nonrigid_correlations,
                z_estimates,
            ) = _register_multi_plane_frames(
                reference_and_masks=reference_and_masks,
                frames=frames,
                minimum_intensity=minimum_intensity,
                maximum_intensity=maximum_intensity,
                bidirectional_phase_offset=bidirectional_phase_offset,
                ops=ops,
                z_plane_number=z_plane_number,
            )
        else:
            raise TypeError("uh oh")

        # Appends the rigid offsets of the current batch to the return variable.
        rigid_shifts.append((rigid_y_shifts, rigid_x_shifts, rigid_correlations))

        # If z-estimates were returned, stores the data in the return variables.
        if z_estimates is not None:
            z_plane_indices.extend(list(z_estimates[0]))
            z_plane_correlations.extend(list(z_estimates[1]))

        # If computed, appends the non-rigid offsets of the current batch to the return variable.
        if ops["nonrigid"]:
            nonrigid_shifts.append((nonrigid_y_shifts, nonrigid_x_shifts, nonrigid_correlations))

        # Updates the mean image by averaging the frames.
        mean_image += registered_frames.sum(axis=0) / frame_number

        # If no output destination was specified, overwrites the input frames with the registered frames.
        if align_output_frames is None:
            align_input_frames[batch_number : min(batch_number + batch_size, frame_number)] = registered_frames
        else:
            align_output_frames[batch_number : min(batch_number + batch_size, frame_number)] = registered_frames

        # If tiff saving is enabled, saves the frames as a tiff file.
        if ops["reg_tif"] if ops["functional_chan"] == ops["align_by_chan"] else ops["reg_tif_chan2"]:
            file_name = io.generate_tiff_filename(
                functional_channel=ops["functional_chan"],
                alignment_channel=ops["align_by_chan"],
                save_path=ops["save_path"],
                batch_number=batch_number,
                channel=1,
            )
            io.save_tiff(frames=registered_frames, file_path=file_name)

    console.echo(
        f"Plane {plane_number} channel 1 frame registration offsets: computed. Time taken: {timer.elapsed} seconds.",
        level=LogLevel.SUCCESS,
    )

    # Combines all of the rigid offsets across batches.
    combined_rigid_shifts = utils.combine_shifts_across_batches(shifts=rigid_shifts, rigid=True)

    # If non-rigid registration is enabled, combines all of the non-rigid offsets across batches.
    if ops["nonrigid"]:
        combined_nonrigid_shifts = utils.combine_shifts_across_batches(shifts=nonrigid_shifts, rigid=False)

    # Returns the resulting tuple.
    return (
        original_reference_image,
        minimum_intensity,
        maximum_intensity,
        mean_image,
        combined_rigid_shifts,
        combined_nonrigid_shifts,
        (z_plane_indices, z_plane_correlations),
    )


def _shift_frames(
    frames: NDArray[np.int16] | NDArray[np.float32],
    y_shifts: NDArray[np.float32],
    x_shifts: NDArray[np.float32],
    nonrigid_y_shifts: NDArray[np.float32] | None,
    nonrigid_x_shifts: NDArray[np.float32] | None,
    blocks: Blocks | None = None,
    ops: dict[str, Any] = generate_default_ops(),
) -> NDArray[np.int16] | NDArray[np.float32]:
    """Applies rigid and optionally non-rigid shifts to a stack of frames for registration.

    This function first applies a rigid y- and x-shift to each frame in the input stack. If non-rigid registration is
    enabled in the input 'ops' dictionary, it also applies local block-based non-rigid shifts to each frame using the
    specified block layout and corresponding shifts. If bidirectional phase offset correction is enabled and not yet
    applied, it applies the correction.

    Args:
        frames: A (TODO: confirm 2D or 3D... seems to be conflicting) NumPy array storing the frames to be shifted.
        y_shifts: A NumPy array storing the number of pixels to shift along the y-axis for each frame.
        x_shifts: A NumPy array storing the number of pixels to shift along the x-axis for each frame.
        nonrigid_y_shifts: A NumPy array storing the shifts in the y-direction for each block in each frame.
        nonrigid_x_shifts: A NumPy array storing the shifts in the x-direction for each block in each frame.
        blocks: A five-element tuple storing information about the block boundaries, numbers, and sizes.
        ops: The dictionary that stores the suite2p processing parameters.

    Returns:
        A 3D NumPy array storing the frames with all specified shifts applied.
    """
    # If the offset is non-zero and has not already been corrected, applies the bidirectional phase shift to the frames.
    if ops["bidiphase"] != 0 and not ops["bidi_corrected"]:
        bidi.shift(frames, int(ops["bidiphase"]))

    # Applies the input y- and x-shifts to each frame.
    for frame, y_shift, x_shift in zip(frames, y_shifts, x_shifts):
        frame[:] = rigid.shift_frame(frame=frame, y_shift=y_shift, x_shift=x_shift)

    # If non-rigid registration is enabled in the input 'ops' dictionary, transforms the frames with the input non-rigid
    # y- and x-shifts and block information.
    if ops["nonrigid"]:
        frames = nonrigid.transform_frames(
            frames,
            y_blocks=blocks[0],
            x_blocks=blocks[1],
            block_number=blocks[2],
            y_shifts=nonrigid_y_shifts,
            x_shifts=nonrigid_x_shifts,
            bilinear=ops.get("bilinear_reg", True),
        )

    # Returns the shifted frames.
    return frames


def _shift_frames_and_write(
    alternative_input_frames: io.BinaryFile,
    plane_number: int,
    y_shifts: NDArray[np.float32],
    x_shifts: NDArray[np.float32],
    alternative_output_frames: io.BinaryFile | None = None,
    nonrigid_y_shifts: NDArray[np.float32] | None = None,
    nonrigid_x_shifts: NDArray[np.float32] | None = None,
    ops=generate_default_ops(),
) -> NDArray[np.float32]:
    """Applies rigid and optionally non-rigid shifts to a stack of frames and writes the shifted output.

    This function takes an input secondary channel frame stack, handles bidirectional phase correction if needed,
    applies the corresponding rigid and optionally non-rigid registration shifts, and writes the shifted output to a
    separate output binary file or in-place if 'alternative_output_frames' is not provided. This function also
    optionally saves the registered frames as a tiff file if enabled in the input 'ops' dictionary.

    Args:
        alternative_input_frames: The input frames to be shifted. It can be a BinaryFile or any type of array that can
            be slice-indexed.
        plane_number: The index of the imaging plane to process.
        y_shifts: A NumPy array storing the number of pixels to shift along the y-axis for each frame.
        x_shifts: A NumPy array storing the number of pixels to shift along the x-axis for each frame.
        alternative_output_frames: The variable where the shifted frames will be stored. The default is None.
        nonrigid_y_shifts: A NumPy array storing the shifts in the y-direction for each block in each frame.
        nonrigid_x_shifts: A NumPy array storing the shifts in the x-direction for each block in each frame.
        ops: The dictionary that stores the suite2p processing parameters.

    Returns:
        A 2D NumPy array storing the mean image computed over all registered frames.
    """
    # Extracts the number of frames, height, and width from the input frames.
    frame_number, height, width = alternative_input_frames.shape

    # If the number of y- or x-shifts provided does not match the number of input frames, raises a ValueError.
    if y_shifts.shape[0] != frame_number or x_shifts.shape[0] != frame_number:
        message = (
            f"Mismatch between number of input frames and rigid y- or x- shift values. Expected {frame_number} "
            f"shift values. Got {y_shifts.shape[0]} y-shifts and {x_shifts.shape[0]} x-shifts. Ensure the shift "
            f"arrays were generated using the same number of frames."
        )
        console.error(message=message, error=ValueError)
        raise ValueError(message)

    # Initializes a variable to store the blocks if non-rigid registration is enabled. Otherwise, remains as None.
    blocks = None

    # If non-rigid registration is enabled in the input 'ops' dictionary, computes the blocks used in non-rigid
    # registration.
    if ops.get("nonrigid"):
        # If the non-rigid y- or x-shifts are not provided, raises a ValueError.
        if nonrigid_y_shifts is None or nonrigid_x_shifts is None:
            message = f"Non-rigid registration is enabled in 'ops' but non-rigid y- and/or x-shifts were not provided."
            console.error(message=message, error=ValueError)
            raise ValueError(message)

        # If the number of non-rigid y- or x-shifts provided does not match the number of input frames, raises a
        # ValueError.
        elif nonrigid_y_shifts.shape[0] != frame_number or nonrigid_x_shifts.shape[0] != frame_number:
            message = (
                f"Mismatch between number of input frames and non-rigid y- or x- shift values. Expected "
                f"{frame_number} shift values. Got {nonrigid_y_shifts.shape[0]} y-shifts and "
                f"{nonrigid_x_shifts.shape[0]} x-shifts. Ensure the shift arrays were generated using the same"
                f"number of frames."
            )
            console.error(message=message, error=ValueError)
            raise ValueError(message)

        # Computes blocks to split the field of view (FOV).
        blocks = nonrigid.make_blocks(height=height, width=width, block_size=ops["block_size"])

    # Queries the number of frames to process if provided in the input 'ops' dictionary.
    if ops["frames_include"] != -1:
        frame_number = min(frame_number, ops["frames_include"])

    # Initializes a 2D NumPy array to store the mean image from the frames.
    mean_image = np.zeros(shape=(height, width), dtype="float32")

    # Queries the batch size from the input 'ops' dictionary.
    batch_size = ops["batch_size"]

    # Initializes the run timer.
    timer = PrecisionTimer("s")

    console.echo(f"Computing plane {plane_number} frame registration offsets for channel 2...", level=LogLevel.INFO)

    # Resets the run timer.
    timer.reset()

    # Registers frames in batches and displays a progress bar if configured.
    for batch_number in tqdm(
        np.arange(0, frame_number, batch_size),
        desc=f"Registering batches of {batch_size} frames",
        unit="batch",
        disable=not ops["progress_bars"],
    ):
        # Slices a batch of frames from the input frames.
        frames = alternative_input_frames[batch_number : min(batch_number + batch_size, frame_number)].astype("float32")

        # Slices the corresponding y- and x-shifts from the input arrays.
        batch_y_shifts = y_shifts[batch_number : min(batch_number + batch_size, frame_number)].astype(int)
        batch_x_shifts = x_shifts[batch_number : min(batch_number + batch_size, frame_number)].astype(int)

        # If non-rigid registration is enabled in the input 'ops' dictionary, slices the corresponding non-rigid y-
        # and x-shifts from the input arrays.
        if ops.get("nonrigid"):
            batch_nonrigid_y_shifts = nonrigid_y_shifts[batch_number : min(batch_number + batch_size, frame_number)]
            batch_nonrigid_x_shifts = nonrigid_x_shifts[batch_number : min(batch_number + batch_size, frame_number)]

        # Otherwise, stores None.
        else:
            batch_nonrigid_y_shifts, batch_nonrigid_x_shifts = None, None

        # Applies the rigid and if configured non-rigid shifts to the batch of frames.
        frames = _shift_frames(
            frames, batch_y_shifts, batch_x_shifts, batch_nonrigid_y_shifts, batch_nonrigid_x_shifts, blocks, ops
        )

        # Updates the mean image by averaging the frames.
        mean_image += frames.sum(axis=0) / frame_number

        # If no output destination was specified, overwrites the input frames with the registered frames.
        if alternative_output_frames is None:
            alternative_input_frames[batch_number : min(batch_number + batch_size, frame_number)] = frames
        else:
            alternative_output_frames[batch_number : min(batch_number + batch_size, frame_number)] = frames

        # If tiff saving is enabled, saves the frames as a tiff file.
        if ops["reg_tif_chan2"] if ops["functional_chan"] == ops["align_by_chan"] else ops["reg_tif"]:
            file_name = io.generate_tiff_filename(
                functional_channel=ops["functional_chan"],
                alignment_channel=ops["align_by_chan"],
                save_path=ops["save_path"],
                batch_number=batch_number,
                channel=0,
            )
            io.save_tiff(frames=frames, file_path=file_name)

    console.echo(
        f"Plane {plane_number} channel 2 frame registration offsets: computed. Time taken: {timer.elapsed} seconds.",
        level=LogLevel.SUCCESS,
    )

    # Returns the resulting mean image.
    return mean_image


def _compute_crop(
    y_shifts: NDArray[np.float32],
    x_shifts: NDArray[np.float32],
    correlations: NDArray[np.float32],
    bad_frames_threshold: float,
    bad_frames: NDArray[np.bool_],
    maximum_shift: float,
    height: int,
    width: int,
) -> tuple[NDArray[np.bool_], list[int], list[int]]:
    """Determines how much to crop the field of view (FOV) based on motion and image alignment confidence.

    This function identifies frames with excessive motion or low alignment confidence using shift values and
    correlation scores. It flags these as "bad" frames, then calculates how much to crop the field of view (FOV) along
    the y- and x-axes so that only the stable regions, consistently present in most good frames, are retained.

    Args:
        y_shifts: The rigid y-shifts applied to each frame during registration.
        x_shifts: The rigid x-shifts applied to each frame during registration.
        correlations: The phase correlation values between each frame and the reference image.
        bad_frames_threshold: The threshold above which a frame is considered bad due to motion/correlation.
        bad_frames: A boolean array indicating the known bad frames.
        maximum_shift: The maximum allowed registration shift as a fraction of the frame size.
        height: The height (in pixels) of the frames.
        width: The width (in pixels) of the frames.

    Returns:
        A tuple of three elements. The first element is the updated boolean array indicating which frames are too noisy
        or unstable. The second and third elements are integer lists storing the valid crop ranges along the y- and
        x-axes respectively.
    """
    # Computes the median filter window used to smooth the input shift arrays.
    filter_window = min((len(y_shifts) // 2) * 2 - 1, 101)

    # Computes per-frame y- and x-shift deviations from the smoothed shift arrays, which highlights abrupt motion
    # bursts.
    y_deviations = y_shifts - medfilt(y_shifts, filter_window)
    x_deviations = x_shifts - medfilt(x_shifts, filter_window)

    # Combines the y- and x-deviations to get the total per-frame motion magnitude.
    total_deviation = (y_deviations**2 + x_deviations**2) ** 0.5

    # Normalizes across all frames.
    total_deviation = total_deviation / total_deviation.mean()

    # Normalizes each frame's correlation score using the median.
    normalized_correlations = correlations / medfilt(correlations, filter_window)

    # Computes a per-frame penalty score (motion / correlation). Frames with high motion and/or low correlation will
    # have larger penalties.
    penalties = total_deviation / np.maximum(0, normalized_correlations)

    # Flags frames as bad if their penalty score exceeds the input threshold.
    bad_frames = np.logical_or(penalties > bad_frames_threshold * 100, bad_frames)

    # Flags frames as bad if the shift exceeds 95% of the input maximum registration shift.
    bad_frames = np.logical_or(abs(x_shifts) > (maximum_shift * width * 0.95), bad_frames)
    bad_frames = np.logical_or(abs(y_shifts) > (maximum_shift * height * 0.95), bad_frames)

    # If less than half of the frames are marked as bad, uses only the "good" frames to compute the maximum motion
    # along the y- and x-axes.
    if bad_frames.mean() < 0.5:
        minimum_y_crop = np.ceil(np.abs(y_shifts[np.logical_not(bad_frames)]).max())
        minimum_x_crop = np.ceil(np.abs(x_shifts[np.logical_not(bad_frames)]).max())

    # Otherwise, uses all of the frames to estimate the cropping margins.
    else:
        warn("WARNING: >50% of frames have large movements, registration likely problematic")
        minimum_y_crop = np.ceil(np.abs(y_shifts).max())
        minimum_x_crop = np.ceil(np.abs(x_shifts).max())

    # Computes the maximum crop bounds along the y- and x-axes.
    maximum_y_crop = height - minimum_y_crop
    maximum_x_crop = width - minimum_x_crop

    # Returns the bad frames mask and the crop ranges along the y- and x-axes.
    return bad_frames, [int(minimum_y_crop), int(maximum_y_crop)], [int(minimum_x_crop), int(maximum_x_crop)]


def registration_wrapper(
    frames: io.BinaryFile,
    plane_number: int,
    raw_frames: io.BinaryFile | None = None,
    frames_channel_2: io.BinaryFile | None = None,
    raw_frames_channel_2: io.BinaryFile | None = None,
    reference_image: NDArray[np.int16] | None = None,
    align_by_channel_2: bool = False,
    ops: dict[str, Any] = generate_default_ops(),
):
    """Applies registration (motion correction) to imaging data for a specific plane.

    This function is the main registration function that encompasses all of the sub-processes of registration. This
    function supports both single-channel and dual-channel recordings, and optionally uses channel 2 as the alignment
    basis. This function performs rigid and optionally non-rigid registration, computes alignment offsets, and
    determines the valid cropping region across all frames.

    Args:
        frames: The output binary file for writing registered frames of the channel 1 (functional) data.
        plane_number: The index of the imaging plane to process.
        raw_frames: The input binary file storing unregistered frames for channel 1.
        frames_channel_2: The output binary file for writing registered frames of channel 2.
        raw_frames_channel_2: The input binary file containing unregistered frames for channel 2.
        reference_image: The reference image to align against. If None, it is computed from the input frames.
        align_by_channel_2: Determines whether registration is performed using channel 2 as the alignment source.
        ops: The dictionary that stores the suite2p processing parameters.

    Returns:
        A tuple of eleven elements. The first element is the final reference image used for alignment. The second and
        third elements are the minimum and maximum intensities respectively used for normalizing the images after
        registration. The fourth element is a tuple of three elements storing the y- and x-shifts and correlation values
        used in rigid registration. The fifth element is a tuple of three elements storing y- and x-shifts and
        correlation values used in non-rigid registration. TODO: finish this docstring
    """
    """main registration function

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
    # Initializes file handlers to None.
    align_output_frames, alternative_input_frames, alternative_output_frames = None, None, None

    # Determines which channel to use for alignment. If channel 2 was not provided or selected, uses channel 1 frames
    # for alignment.
    if frames_channel_2 is None or not align_by_channel_2:
        # If no raw frames were provided, performs alignment directly on the input frames.
        if raw_frames is None:
            align_input_frames = frames
            alternative_input_frames = frames_channel_2
        # If raw frames were provided, uses them for alignment.
        else:
            align_input_frames = raw_frames
            alternative_input_frames = raw_frames_channel_2
            align_output_frames = frames
            alternative_output_frames = frames_channel_2
    # Uses channel 2 frames for alignment if explicitly requested.
    else:
        # If no raw frames were provided, performs alignment directly on the channel 2 frames.
        if raw_frames is None:
            align_input_frames = frames_channel_2
            alternative_input_frames = frames
        # If raw frames were provided, uses the raw channel 2 frames for alignment.
        else:
            align_input_frames = raw_frames_channel_2
            alternative_input_frames = raw_frames
            align_output_frames = frames_channel_2
            alternative_output_frames = frames

    # Extracts the number of frames, height, and width of the alignment frames.
    frame_number, height, width = align_input_frames.shape

    # Checks whether a second channel exists and is correctly sized.
    if alternative_input_frames is not None and alternative_input_frames.shape[0] == align_input_frames.shape[0]:
        channel_number = 2
        console.echo(message=f"Registering two channels for plane {plane_number}...", level=LogLevel.INFO)
    else:
        channel_number = 1
        console.echo(message=f"Registering a single channel for plane {plane_number}...", level=LogLevel.INFO)

    # Runs rigid and optionally non-rigid image registration and outputs the reference image used, intensity range,
    # mean image, registration shifts, and z-plane estimates.
    reference_image, minimum_intensity, maximum_intensity, mean_image, rigid_shifts, nonrigid_shifts, z_estimates = (
        _compute_reference_and_register_frames(
            align_input_frames=align_input_frames,
            plane_number=plane_number,
            align_output_frames=align_output_frames,
            reference_image=reference_image,
            ops=ops,
        )
    )

    # Unpacks the y- and x-shifts and alignment correlations from rigid registration results.
    y_shifts, x_shifts, correlations = rigid_shifts

    # If non-rigid registration was enabled, unpacks the y- and x-shifts as well.
    if ops["nonrigid"]:
        nonrigid_y_shifts, nonrigid_x_shifts, _ = nonrigid_shifts
    else:
        nonrigid_y_shifts, nonrigid_x_shifts = None, None

    # If a second channel exists, applies the same computed shifts to the channel 2 frames. Writes the channel 2
    # frames if an output file was specified.
    if channel_number > 1:
        alternative_mean_image = _shift_frames_and_write(
            alternative_input_frames=alternative_input_frames,
            plane_number=plane_number,
            alternative_output_frames=alternative_output_frames,
            y_shifts=y_shifts,
            x_shifts=x_shifts,
            nonrigid_y_shifts=nonrigid_y_shifts,
            nonrigid_x_shifts=nonrigid_x_shifts,
            ops=ops,
        )
    else:
        alternative_mean_image = None

    # Determines which mean image goes into which output channel, based on alignment source.
    if channel_number == 1 or not align_by_channel_2:
        channel_1_mean_image = mean_image
        if channel_number == 2:
            channel_2_mean_image = alternative_mean_image
        else:
            channel_2_mean_image = None
    elif channel_number == 2:
        channel_2_mean_image = mean_image
        channel_1_mean_image = alternative_mean_image

    # Initializes a boolean NumPy array to store the which frames are marked as "bad" (e.g, unstable or outlier motion).
    bad_frames = np.zeros(shape=frame_number, dtype="bool")

    # Checks whether external bad frame information is available on disk.
    if "data_path" in ops and len(ops["data_path"]) > 0:
        # Constructs the absolute path to the 'bad_frames.npy' file within the specified "data_path" directory.
        bad_frames_path = path.abspath(path.join(ops["data_path"][0], "bad_frames.npy"))

        # Checks if the file 'bad_frames.npy' exists.
        if path.isfile(bad_frames_path):
            message = f"Plane {plane_number} bad frames file: exists. File path: {bad_frames_path}."
            console.echo(message=message, level=LogLevel.WARNING)

            # Loads the indices of the bad frames from the discovered file.
            bad_frames_indices = np.load(bad_frames_path)
            bad_frames_indices = bad_frames_indices.flatten().astype(int)

            # Sets the indices of the bad frames to True.
            bad_frames[bad_frames_indices] = True

            message = f"Plane {plane_number} bad frames count: {bad_frames.sum()}."
            console.echo(message=message, level=LogLevel.WARNING)

    # Analyzes motion and correlation to define cropping bounds that exclude unstable or poorly registered regions.
    bad_frames, y_range, x_range = _compute_crop(
        y_shifts=y_shifts,
        x_shifts=x_shifts,
        correlations=correlations,
        bad_frames_threshold=ops["th_badframes"],
        bad_frames=bad_frames,
        maximum_shift=ops["maxregshift"],
        height=height,
        width=width,
    )

    # Returns the full registration results.
    return (
        reference_image,
        minimum_intensity,
        maximum_intensity,
        channel_1_mean_image,
        rigid_shifts,
        nonrigid_shifts,
        z_estimates,
        channel_2_mean_image,
        bad_frames,
        y_range,
        x_range,
    )


def save_registration_outputs_to_ops(registration_outputs, ops: dict[str, Any]) -> dict[str, Any]:
    """Saves the outputs of the registration process into the provided 'ops' dictionary.

    Args:
        registration_outputs: A tuple of 11 elements storing the results of the registration pipeline.
        ops: The dictionary that stores the suite2p processing parameters.

    Returns:
        The 'ops' dictionary of the first available plane to be processed augmented with additional descriptive
        parameters for the processed data. Specifically, the dictionary includes the following additional keys:
        "refImg", "rmin", "rmax", "yoff", "xoff", "corrXY", "yoff1", "xoff1", "corrXY1", "meanImg", "meanImg_chan2",
        "badframes", "yrange", "xrange", "zpos_registration", "cmax_registration".
    """
    # Unpacks the registration outputs into the appropriate variables.
    (
        reference_image,
        minimum_intensity,
        maximum_intensity,
        channel_1_mean_image,
        rigid_shifts,
        nonrigid_shifts,
        z_estimates,
        channel_2_mean_image,
        bad_frames,
        y_range,
        x_range,
    ) = registration_outputs

    # Stores the reference image and minimum and maximum intensity values in 'ops'.
    ops["refImg"] = reference_image
    ops["rmin"], ops["rmax"] = minimum_intensity, maximum_intensity

    # Stores the rigid registration shifts and correlation scores in 'ops'.
    ops["yoff"], ops["xoff"], ops["corrXY"] = rigid_shifts

    # If computed, stores the non-rigid registration shifts and correlation scores in 'ops'.
    if ops["nonrigid"]:
        ops["yoff1"], ops["xoff1"], ops["corrXY1"] = nonrigid_shifts

    # Stores the mean image computed from the registered frames.
    ops["meanImg"] = channel_1_mean_image

    # If computed, stores the mean image for the second channel.
    if channel_2_mean_image is not None:
        ops["meanImg_chan2"] = channel_2_mean_image

    # Stores bad frame indices and cropping ranges computed during registration.
    ops["badframes"], ops["yrange"], ops["xrange"] = bad_frames, y_range, x_range

    # If z-alignment results exist, stores the z-positions and corresponding maximum correlation scores.
    if len(z_estimates[0]) > 0:
        ops["zpos_registration"] = np.array(z_estimates[0])
        ops["cmax_registration"] = np.array(z_estimates[1])

    # Returns the updated 'ops' dictionary with all registration outputs saved.
    return ops


def compute_enhanced_mean_image(mean_image: NDArray[np.float32], ops: dict[str, Any]) -> NDArray[np.float32]:
    """Computes a high-pass filtered version of the mean image for enhanced visualization.

    The function applies a large-kernel median filter to remove low-frequency background, then subtracts and normalizes
    by this filtered image to emphasize fine spatial details. The result is scaled to the range [0, 1].

    Args:
        mean_image: The mean image to be enhanced.
        ops: The dictionary that stores the suite2p processing parameters.

    Returns:
        The enhanced mean image scaled to the range [0, 1].
    """
    # If the spatial scale (in pixels) is not already in 'ops', derives it from the cell ROI diameter.
    if "spatscale_pix" not in ops:
        # Ensures 'diameter' is represented as a NumPy array with separate y- and x-values.
        if isinstance(ops["diameter"], int):
            diameter = np.array([ops["diameter"], ops["diameter"]])
        else:
            diameter = np.array(ops["diameter"])

        # If the diameter is set to zero, defaults to 12 pixels for both dimensions.
        if diameter[0] == 0:
            diameter[:] = 12

        # Stores the pixel scale for the x-dimension in the input 'ops' dictionary.
        ops["spatscale_pix"] = diameter[1]

        # Stores the aspect ratio between y- and x-dimensions.
        ops["aspect"] = diameter[0] / diameter[1]

    # Computes the size of the median filter in pixels for the y- and x-dimensions. Multiplies the spatial scale by
    # the aspect ratio for the y-dimension and uses the original scale for the x-dimension. Multiplies by 4 to create a
    # large filter, then ensures the kernel size is odd by adding 1. Flattens and converts to int64.
    diameter = 4 * np.ceil(np.array([ops["spatscale_pix"] * ops["aspect"], ops["spatscale_pix"]])) + 1
    diameter = diameter.flatten().astype(np.int64)

    # Applies a 2D median filter to remove low-frequency background.
    median_filter_image = medfilt2d(mean_image, [diameter[0], diameter[1]])

    # Subtracts the median-filtered image to perform high-pass filtering.
    mean_image = mean_image - median_filter_image

    # Computes a divisor image by median filtering the absolute value of the high-pass filtered image. This normalizes
    # the local contrast to reduce intensity variation.
    divisor_image = medfilt2d(np.absolute(mean_image), [diameter[0], diameter[1]])

    # Divides the high-pass filtered image by the divisor image to normalize variance.
    mean_image = (mean_image / (1e-10 + divisor_image)).astype(np.float32)

    # Defines the intensity range for output scaling.
    minimum_intensity = -6  # Lower bound of expected intensity
    maximum_intensity = 6  # Upper bound of expected intensity
    enhanced_mean_image: NDArray[np.float32] = mean_image

    # Linearly scales the image to the [0 ,1] range.
    enhanced_mean_image = ((enhanced_mean_image - minimum_intensity) / (maximum_intensity - minimum_intensity)).astype(
        np.float32
    )

    # Clips the values to ensure the final image is strictly within [0, 1].
    enhanced_mean_image = np.maximum(0, np.minimum(1, enhanced_mean_image))

    # Returns the enhanced mean image.
    return enhanced_mean_image
