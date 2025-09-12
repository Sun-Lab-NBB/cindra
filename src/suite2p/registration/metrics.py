"""This module provides tools for computing rigid and non-rigid per-plane registration metrics from the principal
components of imaging data."""

from typing import Any

import numpy as np
from numpy.typing import NDArray
from ataraxis_time import PrecisionTimer
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA
from ataraxis_base_utilities import LogLevel, console

from . import rigid, utils, nonrigid, bidiphase


def _pc_low_high(
    frames: NDArray[np.int16], low_high_number: int, principal_component_number: int, random_state: int | None = None
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Computes the low and high scoring frames for a given principal component from a principal component analysis
    (PCA) of the input frames.

    Args:
        frames: The frames to perform PCA on.
        low_high_number: The number of lowest- and highest-scoring frames to return.
        principal_component_number: The number of principal components to analyze.
        random_state: Random seed for reproducibility when shuffling or sampling frames. The default is None.

    Returns:
        A tuple of four elements. The first and second elements are NumPy arrays storing the mean images of the lowest-
        highest-scoring frames respectively for each component. The third element is a NumPy array of singular values
        representing the variance captured by each component. The fourth element is a 2D Numpy array containing the
        projection scores of each frame along each principal component.
    """
    # Extracts the number of frames, height, and width from the input frames.
    frame_number, height, width = frames.shape

    # Flattens each frame into a 1D vector and ensures values are float32.
    frames = frames.reshape((frame_number, -1)).astype(np.float32)

    # Computes the mean image across all frames for centering.
    mean_image = frames.mean(axis=0)

    # Centers the data by subtracting the mean image from each frame vector.
    frames -= mean_image

    # Performs principal component analysis (PCA) on the transposed data so that the components represent spatial
    # patterns.
    pca = PCA(n_components=principal_component_number, random_state=random_state).fit(frames.T)

    # Transposes the PCA components so that the rows correspond to frames and columns to principal components.
    frame_scores = pca.components_.T

    # Retrieves the singular values, which indicates the magnitude of variance captured by each component.
    singular_values = pca.singular_values_

    # Adds the mean image back to the frame vectors to restore the original intensity scale.
    frames += mean_image

    # Reshapes the frame vectors back to (number of frames, height, width) for spatial processing.
    frames = np.transpose(np.reshape(frames, (-1, height, width)), (1, 2, 0))

    # Initializes NumPy arrays to store the average of the lowest- and highest-scoring frames per component.
    pc_low_mean_images = np.zeros((principal_component_number, height, width), np.float32)
    pc_high_mean_images = np.zeros((principal_component_number, height, width), np.float32)

    # Sorts the frame indices for each principal component based on projection scores.
    sorted_frame_indices = np.argsort(frame_scores, axis=0)

    # Loops over the principal components.
    for principal_component_index in range(principal_component_number):
        # Computes the mean image of the lowest-scoring frames for the current component.
        pc_low_mean_images[principal_component_index] = frames[
            :, :, sorted_frame_indices[:low_high_number, principal_component_index]
        ].mean(axis=-1)

        # Computes the mean image of the highest-scoring frames for the current component.
        pc_high_mean_images[principal_component_index] = frames[
            :, :, sorted_frame_indices[-low_high_number:, principal_component_index]
        ].mean(axis=-1)

    # Returns the low mean images, high mean images, singular values, and frame scores.
    return pc_low_mean_images, pc_high_mean_images, singular_values, frame_scores


def _pc_register(
    pc_low_mean_images: NDArray[np.float32],
    pc_high_mean_images: NDArray[np.float32],
    bidirectional_corrected: bool,
    spatial_high_pass: int,
    pre_smoothing: float,
    smooth_sigma: float = 1.15,
    smooth_sigma_time: float = 0,
    block_size: tuple[int, int] = (128, 128),
    maximum_shift: float = 0.1,
    nonrigid_maximum_shift: float = 10,
    one_photon_registration: bool = False,
    snr_threshold: float = 1.25,
    is_nonrigid: bool = True,
    bidirectional_phase_offset: int = 0,
    spatial_taper: float = 50.0,
) -> NDArray[np.float64]:
    """Registers low- and high-scoring mean principal component images using rigid and optionally non-rigid
    registration.

    This function computes registration shifts and returns summary metrics describing rigid and non-rigid alignment
    for each principal component.

    Args:
        pc_low_mean_images: The 3D NumPy array storing the mean images of low-scoring frames per principal component.
        pc_high_mean_images: The 3D NumPy array storing the mean images of high-scoring frames per principal component.
        bidirectional_corrected: Tracks whether bidirectional phase correction has been applied to the registered
            data set.
        spatial_high_pass: The spatial high-pass filter window size, in pixels.
        pre_smoothing: The standard deviation for Gaussian smoothing applied before spatial high-pass filtering.
        smooth_sigma: The standard deviation (in pixels) of the Gaussian filter used to smooth the phase correlation
            between the reference image and the current frame.
        smooth_sigma_time: The standard deviation (in frames) of the Gaussian used to temporally smooth the data before
            computing phase correlation.
        block_size: The block size, in pixels, for non-rigid registration, defining the dimensions of subregions used
            in the correction.
        maximum_shift: The maximum allowed registration shift as a fraction of the frame size.
        nonrigid_maximum_shift: The maximum allowed shift, in pixels, for each block relative to the rigid registration
            shift.
        one_photon_registration: Determines whether to perform high-pass spatial filtering and tapering to improve
            one-photon image registration.
        snr_threshold: The signal-to-noise ratio threshold.
        is_nonrigid: Determines whether to perform non-rigid registration.
        bidirectional_phase_offset: The bidirectional phase offset for line scanning experiments.
        spatial_taper: The number of pixels to ignore at the image edges to reduce edge artifacts during registration.

    Returns:
        A 2D NumPy array where each row corresponds to a principal component. The first and second columns correspond
        to the mean rigid and non-rigid registration shift magnitudes respectively. The third column is the maximum
        non-rigid registration shift magnitude.
    """
    # Extracts the number of principal components and the height and width of the images.
    principal_component_number, height, width = pc_low_mean_images.shape

    # Computes blocks and smoothing kernels for non-rigid registration.
    y_blocks, x_blocks, _, block_size, nonrigid_smoothing_kernel = nonrigid.make_blocks(
        height=height, width=width, block_size=block_size
    )

    # Converts the maximum allowed non-rigid shift to a NumPy array for consistent handling later.
    # nonrigid_maximum_shift = np.array(nonrigid_maximum_shift)

    # Initializes an array to store the registration metrics for each principal component.
    registration_metrics = np.zeros((principal_component_number, 3))

    # Loops over the principal components.
    for principal_component_index in range(principal_component_number):
        # Selects the low mean image as the reference image for registration.
        reference_image = pc_low_mean_images[principal_component_index]

        # Extracts the corresponding high mean image and adds a new axis to match the expected input shape.
        image = pc_high_mean_images[principal_component_index][np.newaxis, :, :]

        # Applies pre-processing steps to one-photon recordings if enabled.
        if one_photon_registration:
            # Converts the reference image to float32.
            processed_reference_image = reference_image.astype(np.float32)

            # Applies spatial smoothing before high-pass filtering if enabled.
            if pre_smoothing:
                processed_reference_image = utils.spatial_smooth(processed_reference_image, int(pre_smoothing))

            # Applies spatial high-pass filtering to enhance spatial features.
            reference_image = utils.spatial_high_pass(processed_reference_image, spatial_high_pass)

        # Computes intensity clipping bounds based on the 1st and 99th percentiles to reduce outlier effects.
        minimum_intensity, maximum_intensity = (
            np.int16(np.percentile(reference_image, 1)),
            np.int16(np.percentile(reference_image, 99)),
        )

        # Clips the intensities of the reference image within the computed bounds.
        reference_image = np.clip(reference_image, minimum_intensity, maximum_intensity)

        # Computes the taper mask and mask offset used during rigid registration.
        taper_mask, mask_offset = rigid.compute_masks(
            reference_image=reference_image, mask_slope=spatial_taper if one_photon_registration else 3 * smooth_sigma
        )

        # Computes the FFT of the reference image, applying Gaussian smoothing with the specified sigma.
        fft_reference_image = rigid.fft_reference_image(
            reference_image=reference_image,
            smooth_sigma=smooth_sigma,
        )

        # Adds a new axis to the FFT reference image to match expected input shape for registration.
        fft_reference_image = fft_reference_image[np.newaxis, :, :]

        # If non-rigid registration is enabled, prepares the non-rigid taper mask, mask offset, and FFT reference image.
        if is_nonrigid:
            # Defines the mask slope for non-rigid tapering, adjusting if one-photon registration is enabled.
            nonrigid_mask_slope = spatial_taper if one_photon_registration else 3 * smooth_sigma

            # Computes the non-rigid taper mask, mask offset, and FFT reference image using blocks for non-rigid
            # registration.
            nonrigid_taper_mask, nonrigid_mask_offset, nonrigid_fft_reference_image = nonrigid.fft_reference_image(
                reference_image=reference_image,
                mask_slope=nonrigid_mask_slope,
                smooth_sigma=smooth_sigma,
                y_blocks=y_blocks,
                x_blocks=x_blocks,
            )

        # Applies bidirectional phase offset correction if provided and not already corrected.
        if bidirectional_phase_offset and not bidirectional_corrected:
            bidiphase.shift(image, bidirectional_phase_offset)

        # Prepares the high mean image for registration by converting to float32.
        processed_image = image.astype(np.float32)

        # Applies pre-processing steps to one-photon recordings if enabled.
        if one_photon_registration:
            # Applies spatial smoothing before high-pass filtering if enabled.
            if pre_smoothing:
                processed_image = utils.spatial_smooth(processed_image, int(pre_smoothing))

            # Applies spatial high-pass filtering to enhance spatial features.
            processed_image = utils.spatial_high_pass(processed_image, spatial_high_pass)[np.newaxis, :]

        # Clips the intensities of the processed image to the same intensity bounds as the reference.
        processed_image = np.clip(processed_image, minimum_intensity, maximum_intensity)

        # Performs rigid registration by phase correlation between the processed image and FFT reference image.
        rigid_y_shifts, rigid_x_shifts, _ = rigid.phase_correlate(
            frames=rigid.apply_masks(frames=processed_image, mask_multiplier=taper_mask, mask_offset=mask_offset),
            fft_reference_image=fft_reference_image.squeeze(),
            maximum_shift=maximum_shift,
            smooth_sigma_time=0,
        )

        # Applies the computed rigid y- and x-shifts to each frame in the high mean image.
        for frame, y_shift, x_shift in zip(image, rigid_y_shifts.flatten(), rigid_x_shifts.flatten()):
            frame[:] = rigid.shift_frame(frame=frame, y_shift=y_shift, x_shift=x_shift)

        # If non-rigid registration is enabled, performs further non-rigid alignment.
        if is_nonrigid:
            # Applies temporal smoothing if provided.
            if smooth_sigma_time > 0:
                processed_image = gaussian_filter1d(processed_image, sigma=smooth_sigma_time, axis=0)

            # Performs non-rigid phase correlation to compute the non-rigid y- and x-shifts.
            (
                nonrigid_y_shifts,
                nonrigid_x_shifts,
                _,
            ) = nonrigid.phase_correlate(
                frames=processed_image,
                taper_mask=nonrigid_taper_mask.squeeze(),
                mask_offset=nonrigid_mask_offset.squeeze(),
                fft_reference_image=nonrigid_fft_reference_image.squeeze(),
                snr_threshold=snr_threshold,
                smoothing_kernel=nonrigid_smoothing_kernel,
                y_blocks=y_blocks,
                x_blocks=x_blocks,
                maximum_shift=nonrigid_maximum_shift,
            )

            # Records the average magnitude of the non-rigid shifts for the current principal component.
            registration_metrics[principal_component_index, 1] = np.mean(
                (nonrigid_y_shifts**2 + nonrigid_x_shifts**2) ** 0.5
            )

            # Records the average magnitude of the rigid shifts for the current principal component.
            registration_metrics[principal_component_index, 0] = np.mean(
                (rigid_y_shifts[0] ** 2 + rigid_x_shifts[0] ** 2) ** 0.5
            )

            # Records the maximum magnitude of the non-rigid shifts for the current principal component.
            registration_metrics[principal_component_index, 2] = np.amax(
                (nonrigid_y_shifts**2 + nonrigid_x_shifts**2) ** 0.5
            )

    # Returns an array of registration metrics summarizing the rigid and non-rigid shift magnitudes.
    return registration_metrics


def get_pc_metrics(frames: NDArray[np.int16], ops: dict[str, Any], plane_number: int) -> dict[str, Any]:
    """Computes rigid and optionally non-rigid registration metrics from the top principal components (PCs) of a
    registered movie, storing the results in the input 'ops' dictionary.

    Args:
        frames: The frames of a binned movie to compute the registration metrics of.
        ops: The dictionary that stores the suite2p processing parameters.
        plane_number: The index of the imaging plane to process.

    Returns:
        The input 'ops' dictionary augmented with additional descriptive parameters for the processed data.
        Specifically, the dictionary includes the following additional keys: "recPC", "tPC", "regDX".
    """
    # Initializes the runtime timer.
    timer = PrecisionTimer("s")

    # Determines how many principal components to compute.
    # Note, the default has been revised by the Sun lab from 30 to 10, to match the 'registration_metrics' module
    # behavior.
    principal_component_number = ops["reg_metric_n_pc"] if "reg_metric_n_pc" in ops else 10

    console.echo(
        message=f"Computing {principal_component_number} Principal Components (PCs) for plane {plane_number}...",
        level=LogLevel.INFO,
    )

    # Resets run timer.
    timer.reset()

    # Computes the lowest- and highest-scoring mean images for each principal component, along with the time courses
    # of each of the principal components, which are stored under the "tPC" key in the input 'ops' dictionary.
    pc_low_mean_images, pc_high_mean_images, _, ops["tPC"] = _pc_low_high(
        frames=frames,
        low_high_number=np.minimum(300, int(ops["nframes"] / 2)),
        principal_component_number=principal_component_number,
        random_state=None,
    )

    console.echo(
        message=f"Plane {plane_number} PCs: computed. Time taken: {timer.elapsed} seconds.", level=LogLevel.SUCCESS
    )

    # Stores the low and high mean images for principal components in the input 'ops' dictionary.
    ops["regPC"] = np.concatenate(
        (pc_low_mean_images[np.newaxis, :, :, :], pc_high_mean_images[np.newaxis, :, :, :]), axis=0
    )

    console.echo(
        message=f"Restring top and bottom of each PC to each-other for plane {plane_number}...", level=LogLevel.INFO
    )

    # Resets run timer.
    timer.reset()

    # Performs rigid and optionally non-rigid registration between the low and high PC mean images and stores the
    # per-PC metrics under the "regDX" key in the input 'ops' dictionary.
    ops["regDX"] = _pc_register(
        pc_low_mean_images,
        pc_high_mean_images,
        spatial_high_pass=ops["spatial_hp_reg"],
        pre_smoothing=ops["pre_smooth"],
        bidirectional_corrected=ops["bidi_corrected"],
        smooth_sigma=ops["smooth_sigma"] if "smooth_sigma" in ops else 1.15,
        smooth_sigma_time=ops["smooth_sigma_time"],
        block_size=ops["block_size"] if "block_size" in ops else (128, 128),
        maximum_shift=ops["maxregshift"] if "maxregshift" in ops else 0.1,
        nonrigid_maximum_shift=ops["maxregshiftNR"] if "maxregshiftNR" in ops else 5,
        one_photon_registration=ops["one_p_reg"] if "one_p_reg" in ops else False,
        snr_threshold=ops["snr_thresh"],
        is_nonrigid=ops["nonrigid"],
        bidirectional_phase_offset=ops["bidiphase"],
        spatial_taper=ops["spatial_taper"],
    )

    console.echo(
        message=f"Plane {plane_number} PC registration: complete. Time taken: {timer.elapsed} seconds.",
        level=LogLevel.SUCCESS,
    )

    # Returns the modified 'ops' dictionary.
    return ops
