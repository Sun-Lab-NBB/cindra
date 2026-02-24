"""Provides the algorithm for computing the registration quality metrics using principal component analysis."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from ataraxis_time import PrecisionTimer, TimerPrecisions
from sklearn.decomposition import PCA  # type: ignore[import-untyped]
from ataraxis_base_utilities import LogLevel, console

from ..io import BinaryFile
from .rigid import (
    shift_frame,
    apply_edge_taper,
    compute_edge_taper,
    compute_rigid_shifts,
    compute_phase_correlation_kernel,
)
from .utils import apply_spatial_high_pass, apply_spatial_smoothing
from .nonrigid import compute_nonrigid_shifts, compute_nonrigid_reference_data
from ..detection import compute_registration_blocks
from .bidiphase_correction import apply_bidirectional_phase_correction

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..dataclasses import RuntimeContext

# Frame subsampling parameters for PC metrics computation. These control how many frames are sampled from the
# recording to reduce memory overhead while maintaining statistical representativeness.
_MINIMUM_SAMPLE_COUNT: int = 2000
_MAXIMUM_SAMPLE_COUNT: int = 5000
_MAXIMUM_HEIGHT_FOR_LARGE_SAMPLE: int = 700
_MAXIMUM_WIDTH_FOR_LARGE_SAMPLE: int = 700


def compute_pc_metrics(context: RuntimeContext) -> None:
    """Computes registration quality metrics using principal component analysis.

    Evaluates registration quality by computing principal components of the registered frames and measuring how well
    frames at opposite ends of each PC align. If registration is successful, PC-based groupings should show minimal
    spatial displacement. Large displacements indicate residual motion or registration artifacts.

    Notes:
        This function processes frames from a single processing plane at a time. For multi-plane recordings, call this
        function separately for each plane's RuntimeContext.

        The function reads frames from the registered binary file, subsamples them to reduce memory overhead, and
        computes PCA-based metrics. The subsampling selects evenly-spaced frames across the recording to maintain
        statistical representativeness while limiting memory usage.

        The computed metrics are stored in context.runtime.registration. The principal_component_extreme_images field
        contains mean images from low and high PC projections. The principal_component_projections field contains PC
        projection values for each sampled frame. The principal_component_shift_metrics field contains registration
        shift metrics computed by aligning PC extremes.

    Args:
        context: The runtime context containing pipeline configuration and runtime data for the current plane. Modified
            in-place to store the computed metrics.

    Raises:
        FileNotFoundError: If the registered binary file does not exist at the specified path.
        ValueError: If the registered binary path is not set in the runtime context.
    """
    # Extracts IO parameters from runtime context.
    registered_binary_path = context.runtime.io.registered_binary_path
    if registered_binary_path is None:
        message = (
            "Unable to compute the registration quality metrics. The input RuntimeContext instance does not contain "
            "the path to the plane's registered binary file."
        )
        console.error(message=message, error=ValueError)

    if not registered_binary_path.exists():
        message = (
            f"Unable to compute the registration quality metrics. The registered binary file does not exist at the "
            f"specified path: {registered_binary_path}."
        )
        console.error(message=message, error=FileNotFoundError)

    frame_height = context.runtime.io.frame_height
    frame_width = context.runtime.io.frame_width
    frame_count = context.runtime.io.frame_count
    plane_index = context.runtime.io.plane_index

    # Extracts valid pixel ranges from registration data.
    valid_y_range = context.runtime.registration.valid_y_range
    valid_x_range = context.runtime.registration.valid_x_range

    # Extracts registration configuration parameters.
    num_components = context.configuration.registration.registration_metric_principal_components
    spatial_smoothing_sigma = context.configuration.registration.spatial_smoothing_sigma
    maximum_shift_fraction = context.configuration.registration.maximum_shift_fraction
    parallel_workers = context.configuration.runtime.parallel_workers

    # Extracts non-rigid registration parameters.
    nonrigid_enabled = context.configuration.non_rigid_registration.enabled
    block_size = context.configuration.non_rigid_registration.block_size
    snr_threshold = context.configuration.non_rigid_registration.signal_to_noise_threshold
    maximum_nonrigid_shift = context.configuration.non_rigid_registration.maximum_block_shift

    # Extracts one-photon registration parameters.
    one_photon_mode = context.configuration.one_photon_registration.enabled
    pre_smoothing_sigma = context.configuration.one_photon_registration.pre_smoothing_sigma
    spatial_highpass_window = context.configuration.one_photon_registration.spatial_highpass_window
    edge_taper_pixels = context.configuration.one_photon_registration.edge_taper_pixels

    # Extracts registration state from runtime data.
    bidirectional_phase_offset = context.runtime.registration.bidirectional_phase_offset
    bidirectional_corrected = context.runtime.registration.bidirectional_phase_corrected

    # Computes edge taper slope based on imaging mode.
    edge_taper_slope = edge_taper_pixels if one_photon_mode else 3.0 * spatial_smoothing_sigma

    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    console.echo(
        message=(
            f"Computing {num_components} Principal Components (PCs) for plane {plane_index} to assess registration "
            f"quality..."
        ),
        level=LogLevel.INFO,
    )
    timer.reset()

    # Determines the number of frames to sample based on recording dimensions. Uses fewer samples for larger
    # recordings or recordings with many frames to manage memory usage.
    use_small_sample = (
        frame_count < _MAXIMUM_SAMPLE_COUNT
        or frame_height > _MAXIMUM_HEIGHT_FOR_LARGE_SAMPLE
        or frame_width > _MAXIMUM_WIDTH_FOR_LARGE_SAMPLE
    )
    sample_count = min(
        _MINIMUM_SAMPLE_COUNT if use_small_sample else _MAXIMUM_SAMPLE_COUNT,
        frame_count,
    )

    # Reads and subsamples frames from the registered binary file.
    with BinaryFile(height=frame_height, width=frame_width, file_path=registered_binary_path) as binary_file:
        frames = binary_file.subsample_movie(
            sample_count=sample_count,
            y_range=(valid_y_range[0], valid_y_range[1]),
            x_range=(valid_x_range[0], valid_x_range[1]),
        )

    # Determines the extreme images for each requested PC and averages them into representative low / high projection
    # images.
    num_extreme_frames = min(300, frames.shape[0] // 2)
    pc_low, pc_high, principal_component_projections = _compute_pc_extremes(
        frames=frames,
        num_extreme_frames=num_extreme_frames,
        num_components=num_components,
    )

    console.echo(
        message=f"Plane {plane_index} Principal Component images: computed. Time taken: {timer.elapsed} seconds.",
        level=LogLevel.SUCCESS,
    )

    # Stores PC extreme images in the runtime context. Stacks low and high projections along axis 0.
    principal_component_extreme_images = np.stack((pc_low, pc_high), axis=0)

    console.echo(
        message=(
            f"Registering top and bottom projection images of each Principal Component for plane {plane_index} to "
            f"each-other..."
        ),
        level=LogLevel.INFO,
    )
    timer.reset()

    # Computes registration metrics by aligning PC extremes.
    principal_component_shift_metrics = _register_pc_extremes(
        pc_low=pc_low,
        pc_high=pc_high,
        spatial_highpass_window=spatial_highpass_window,
        pre_smoothing_window=int(pre_smoothing_sigma) if pre_smoothing_sigma > 0 else None,
        bidirectional_corrected=bidirectional_corrected,
        smoothing_sigma=spatial_smoothing_sigma,
        block_size=block_size,
        maximum_shift_fraction=maximum_shift_fraction,
        maximum_nonrigid_shift=maximum_nonrigid_shift,
        one_photon_mode=one_photon_mode,
        snr_threshold=snr_threshold,
        nonrigid_enabled=nonrigid_enabled,
        bidirectional_phase_offset=bidirectional_phase_offset,
        edge_taper_slope=edge_taper_slope,
        workers=parallel_workers,
    )

    console.echo(
        message=(
            f"Plane {plane_index} Principal Component projection image registration: complete. Time taken: "
            f"{timer.elapsed} seconds."
        ),
        level=LogLevel.SUCCESS,
    )

    # Stores the computed metrics in the runtime context.
    context.runtime.registration.principal_component_extreme_images = principal_component_extreme_images
    context.runtime.registration.principal_component_projections = principal_component_projections
    context.runtime.registration.principal_component_shift_metrics = principal_component_shift_metrics


def _compute_pc_extremes(
    frames: NDArray[np.float32],
    num_extreme_frames: int,
    num_components: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Computes mean images from frames at extreme ends of each principal component.

    Performs PCA on the input frames and identifies frames with the highest and lowest projections onto each component.
    The mean of these extreme frames provides representative images for each PC direction, which can be used to
    visually assess registration quality.

    Args:
        frames: The input frames with shape (num_frames, height, width).
        num_extreme_frames: The number of frames to average at each extreme of each PC.
        num_components: The number of principal components to compute.

    Returns:
        A tuple of (pc_low, pc_high, projections). The pc_low array with shape (num_components, height, width)
        contains mean images from frames with the lowest PC projections. The pc_high array has the same shape and
        contains mean images from the highest PC projections. The projections array with shape
        (num_frames, num_components) contains the PC projection values for each frame.
    """
    num_frames, height, width = frames.shape

    # Reshapes frames to 2D for PCA and centers the data. Uses a view to avoid copying when possible.
    frames_flat = frames.reshape((num_frames, -1))
    mean_image = frames_flat.mean(axis=0)
    frames_centered = frames_flat - mean_image

    # Fits PCA on transposed data to get frame-wise projections.
    pca = PCA(n_components=num_components).fit(frames_centered.T)
    # noinspection PyUnresolvedReferences
    projections: NDArray[np.float32] = pca.components_.T.astype(np.float32)

    # Pre-computes sorted indices for all components at once.
    sorted_indices = np.argsort(projections, axis=0)

    # Computes mean images from extreme frames for each PC. Indexes directly into the original frames array to avoid
    # creating a transposed copy of the entire dataset.
    pc_low = np.empty((num_components, height, width), dtype=np.float32)
    pc_high = np.empty((num_components, height, width), dtype=np.float32)

    for component_index in range(num_components):
        low_indices = sorted_indices[:num_extreme_frames, component_index]
        high_indices = sorted_indices[-num_extreme_frames:, component_index]
        pc_low[component_index] = frames[low_indices].mean(axis=0)
        pc_high[component_index] = frames[high_indices].mean(axis=0)

    return pc_low, pc_high, projections


def _register_pc_extremes(
    pc_low: NDArray[np.float32],
    pc_high: NDArray[np.float32],
    bidirectional_corrected: bool,
    spatial_highpass_window: int | None = None,
    pre_smoothing_window: int | None = None,
    smoothing_sigma: float = 1.15,
    block_size: tuple[int, int] = (128, 128),
    maximum_shift_fraction: float = 0.1,
    maximum_nonrigid_shift: float = 5.0,
    one_photon_mode: bool = False,
    snr_threshold: float = 1.2,
    nonrigid_enabled: bool = True,
    bidirectional_phase_offset: int = 0,
    edge_taper_slope: float = 40.0,
    workers: int = -1,
) -> NDArray[np.float32]:
    """Registers images at the extreme ends of each principal components to each-other to measure registration quality.

    Attempts to align the high-projection PC images to the low-projection PC images using rigid and optionally
    nonrigid registration. The magnitude of the required shifts indicates how much residual motion remains after
    registration. Large shifts suggest poor registration.

    Args:
        pc_low: The mean images from low PC projections with shape (num_components, height, width).
        pc_high: The mean images from high PC projections with shape (num_components, height, width).
        bidirectional_corrected: Determines whether bidirectional phase correction has already been applied.
        spatial_highpass_window: The window size for spatial high-pass filtering in one-photon mode.
        pre_smoothing_window: The window size for spatial smoothing before high-pass filtering.
        smoothing_sigma: The standard deviation for Gaussian smoothing of the phase correlation surface.
        block_size: The target block size as (height, width) for nonrigid registration.
        maximum_shift_fraction: The maximum allowed rigid shift as a fraction of the minimum dimension.
        maximum_nonrigid_shift: The maximum allowed nonrigid shift in pixels.
        one_photon_mode: Determines whether to apply one-photon preprocessing.
        snr_threshold: The SNR threshold for adaptive smoothing during nonrigid registration.
        nonrigid_enabled: Determines whether to compute nonrigid registration metrics.
        bidirectional_phase_offset: The bidirectional phase offset in pixels.
        edge_taper_slope: Controls the steepness of the edge taper for phase correlation.
        workers: The number of parallel workers for FFT computation. Use -1 for all available cores.

    Returns:
        A 2D array with shape (num_components, 3) containing registration metrics. Column 0 contains the mean rigid
        shift magnitude in pixels. Column 1 contains the mean nonrigid shift magnitude. Column 2 contains the maximum
        nonrigid shift magnitude. When nonrigid registration is disabled, columns 1 and 2 are zero.
    """
    num_components, height, width = pc_low.shape

    # Computes constants that do not change across components.
    taper_slope = edge_taper_slope if one_photon_mode else 3.0 * smoothing_sigma

    # Computes registration blocks for nonrigid processing.
    y_blocks, x_blocks, _, _, smoothing_kernel = compute_registration_blocks(
        height=height,
        width=width,
        block_size=block_size,
    )

    metrics = np.zeros((num_components, 3), dtype=np.float32)

    for component_index in range(num_components):
        reference_image = pc_low[component_index]
        target_frame = pc_high[component_index].copy()

        # Applies one-photon preprocessing to reference image.
        if one_photon_mode and spatial_highpass_window is not None:
            if pre_smoothing_window:
                reference_image = apply_spatial_smoothing(data=reference_image, window=pre_smoothing_window)
            reference_image = apply_spatial_high_pass(data=reference_image, window=spatial_highpass_window)

        # Clips reference image to 1st-99th percentile range.
        intensity_min = np.percentile(reference_image, q=1)
        intensity_max = np.percentile(reference_image, q=99)
        reference_image = np.clip(reference_image, intensity_min, intensity_max)

        # Computes edge taper and phase correlation kernel for rigid registration.
        taper_mask, mean_offset = compute_edge_taper(reference_image=reference_image, taper_slope=taper_slope)
        reference_kernel = compute_phase_correlation_kernel(
            reference_image=reference_image,
            smoothing_sigma=smoothing_sigma,
        )

        # Applies bidirectional phase correction to target if needed.
        if bidirectional_phase_offset and not bidirectional_corrected:
            target_batch = target_frame[np.newaxis, :, :]
            apply_bidirectional_phase_correction(
                frames=target_batch,
                bidirectional_phase_offset=bidirectional_phase_offset,
            )
            target_frame = target_batch[0]

        # Applies one-photon preprocessing to target frame.
        if one_photon_mode and spatial_highpass_window is not None:
            if pre_smoothing_window:
                target_frame = apply_spatial_smoothing(data=target_frame, window=pre_smoothing_window)
            target_frame = apply_spatial_high_pass(data=target_frame, window=spatial_highpass_window)

        # Clips target frame in-place and adds batch dimension.
        np.clip(target_frame, intensity_min, intensity_max, out=target_frame)
        preprocessed_target = target_frame[np.newaxis, :, :]

        # Computes rigid registration shifts.
        tapered_target = apply_edge_taper(frames=preprocessed_target, taper_mask=taper_mask, mean_offset=mean_offset)
        y_shifts, x_shifts, _ = compute_rigid_shifts(
            frames=tapered_target,
            reference_kernel=reference_kernel,
            maximum_shift_fraction=maximum_shift_fraction,
            temporal_smoothing_sigma=0,
            workers=workers,
        )

        # Records rigid shift magnitude.
        rigid_magnitude = float(np.sqrt(y_shifts[0] ** 2 + x_shifts[0] ** 2))
        metrics[component_index, 0] = rigid_magnitude

        # Computes nonrigid registration metrics if enabled.
        if nonrigid_enabled:
            # Applies rigid shift to target before nonrigid registration.
            shifted_target = shift_frame(
                frame=preprocessed_target[0],
                y_shift=int(y_shifts[0]),
                x_shift=int(x_shifts[0]),
            )[np.newaxis, :, :]

            # Prepares nonrigid reference data.
            nonrigid_taper, nonrigid_offset, nonrigid_kernel = compute_nonrigid_reference_data(
                reference_image=reference_image,
                taper_slope=taper_slope,
                smoothing_sigma=smoothing_sigma,
                y_blocks=y_blocks,
                x_blocks=x_blocks,
            )

            y_nonrigid, x_nonrigid, _ = compute_nonrigid_shifts(
                frames=shifted_target,
                taper_mask=nonrigid_taper,
                mean_offset=nonrigid_offset,
                reference_kernel=nonrigid_kernel,
                snr_threshold=snr_threshold,
                smoothing_kernel=smoothing_kernel,
                x_blocks=x_blocks,
                y_blocks=y_blocks,
                maximum_shift=maximum_nonrigid_shift,
                workers=workers,
            )

            # Computes nonrigid shift magnitudes.
            nonrigid_magnitudes = np.sqrt(y_nonrigid**2 + x_nonrigid**2)
            metrics[component_index, 1] = float(np.mean(nonrigid_magnitudes))
            metrics[component_index, 2] = float(np.amax(nonrigid_magnitudes))

    return metrics
