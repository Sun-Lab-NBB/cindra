"""Provides frame registration (motion correction) entry point for the single-day cindra processing pipeline."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
from scipy.signal import medfilt
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_base_utilities import LogLevel, console

from ..io import BinaryFile
from .rigid import (
    shift_frame,
    apply_edge_taper,
    compute_edge_taper,
    compute_rigid_shifts,
    compute_phase_correlation_kernel,
)
from .utils import (
    combine_rigid_offsets,
    apply_spatial_high_pass,
    apply_spatial_smoothing,
    combine_nonrigid_offsets,
)
from .metrics import compute_pc_metrics
from .nonrigid import (
    compute_nonrigid_shifts,
    apply_nonrigid_correction,
    compute_nonrigid_reference_data,
)
from ..detection import compute_registration_blocks
from .bidiphase_correction import compute_bidirectional_phase_offset, apply_bidirectional_phase_correction

_MINIMUM_REGISTRATION_METRIC_FRAMES: int = 1500
"""The minimum number of frames required to compute registration quality metrics."""

_BAD_FRAME_FRACTION_THRESHOLD: float = 0.5
"""The threshold fraction of bad frames above which registration is considered failed."""

_MAXIMUM_MEDIAN_FILTER_WINDOW: int = 101
"""The maximum median filter window size for offset time series smoothing."""

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ..dataclasses import RuntimeContext

    RegistrationBlocks = tuple[
        list[NDArray[np.int32]], list[NDArray[np.int32]], tuple[int, int], tuple[int, int], NDArray[np.float32]
    ]
    """The type alias for the registration block structure returned by compute_registration_blocks. Contains y_blocks,
    x_blocks, block_counts, actual_block_size, and smoothing_kernel."""


def register_plane(context: RuntimeContext) -> None:
    """Registers (motion-corrects) all frames for a single imaging plane specified by the input runtime context.

    This function is the primary entry point for frame registration. It computes registration offsets from the alignment
    channel (determined by config.registration.align_by_first_channel), then applies those offsets to both channels.
    If two-step registration is enabled, a refinement pass is performed using the mean of registered frames as the
    reference.

    All configuration is read from context.configuration, file paths from context.runtime.io, and results are stored in
    context.runtime.registration, context.runtime.detection, and context.runtime.timing.

    Args:
        context: The RuntimeContext containing configuration, file paths, and mutable runtime data structures. Modified
            in-place to store registration outputs including reference image, offsets, mean images, and timing data.
    """
    config = context.configuration
    io_data = context.runtime.io
    registration_data = context.runtime.registration
    plane_index = io_data.plane_index if io_data.plane_index is not None else 0

    # Checks if registration should be skipped (already registered and not forcing re-registration).
    if registration_data.is_registered() and not config.registration.repeat_registration:
        console.echo(
            message=(
                f"Plane {plane_index} registration: skipped. The plane is already registered and re-registration is "
                f"disabled."
            ),
            level=LogLevel.INFO,
        )
        return

    # Clears existing registration data if re-registering.
    if registration_data.is_registered():
        console.echo(
            message=(
                f"Plane {plane_index} registration: forced. Clearing existing data and re-running the registration."
            ),
            level=LogLevel.INFO,
        )
        registration_data.clear()

    # Determines channel configuration.
    has_second_channel = io_data.registered_binary_path_channel_2 is not None

    if has_second_channel:
        alignment_channel = "channel 1" if config.registration.align_by_first_channel else "channel 2"
        console.echo(
            message=f"Registering plane {plane_index} (two channels, aligning by {alignment_channel})...",
            level=LogLevel.INFO,
        )
    else:
        console.echo(message=f"Registering plane {plane_index} (single channel)...", level=LogLevel.INFO)

    # Starts timing for first registration step.
    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()

    # Computes registration offsets from the alignment channel and applies them.
    _register_alignment_channel(context)

    # Applies the same registration offsets to the secondary channel if present.
    if has_second_channel:
        _register_secondary_channel(context)

    # Records first registration step timing.
    context.runtime.timing.registration_time = timer.elapsed
    console.echo(
        message=f"Plane {plane_index} registration step 1: complete. Time taken: {timer.elapsed} seconds.",
        level=LogLevel.SUCCESS,
    )

    # Performs two-step registration refinement if enabled. The second step re-registers the already-registered frames
    # using a new reference computed from the first-step results, which improves alignment for noisy data.
    if config.registration.two_step_registration:
        console.echo(message=f"Running plane {plane_index} two-step registration refinement...", level=LogLevel.INFO)
        timer.reset()

        # Re-runs registration (computes new reference from already-registered frames).
        _register_alignment_channel(context)

        # Re-applies shifts to the secondary channel if present.
        if has_second_channel:
            _register_secondary_channel(context)

        # Records two-step registration timing.
        context.runtime.timing.two_step_registration_time = int(timer.elapsed)
        console.echo(
            message=f"Plane {plane_index} registration step 2: complete. Time taken: {timer.elapsed} seconds.",
            level=LogLevel.SUCCESS,
        )

    # Loads bad frames from file if present.
    num_frames = io_data.frame_count
    bad_frames = np.zeros(num_frames, dtype=np.bool_)
    data_path = config.file_io.data_path
    if data_path is not None:
        bad_frames_file = data_path / "bad_frames.npy"
        if bad_frames_file.exists():
            console.echo(
                message=f"Plane {plane_index} bad frames file: exists. Path: {bad_frames_file}.",
                level=LogLevel.WARNING,
            )
            bad_frame_indices = np.load(bad_frames_file)
            bad_frame_indices = bad_frame_indices.flatten().astype(int)
            bad_frames[bad_frame_indices] = True
            console.echo(
                message=f"Plane {plane_index} bad frames count: {bad_frames.sum()}.",
                level=LogLevel.WARNING,
            )

    # Computes valid region from registration shifts.
    registration_data = context.runtime.registration
    height, width = io_data.frame_height, io_data.frame_width

    # Extracts offsets for crop computation. Fallback assignments are for type checker only; these are always present
    # after _register_alignment_channel. Uses np.empty to avoid initialization overhead.
    y_offsets = (
        registration_data.rigid_y_offsets
        if registration_data.rigid_y_offsets is not None
        else np.empty(1, dtype=np.int32)
    )
    x_offsets = (
        registration_data.rigid_x_offsets
        if registration_data.rigid_x_offsets is not None
        else np.empty(1, dtype=np.int32)
    )
    correlations = (
        registration_data.rigid_correlations
        if registration_data.rigid_correlations is not None
        else np.empty(1, dtype=np.float32)
    )

    computed_bad_frames, valid_y_range, valid_x_range = _compute_crop(
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        correlations=correlations,
        bad_frame_threshold=config.registration.bad_frame_threshold,
        bad_frames=bad_frames,
        maximum_shift_fraction=config.registration.maximum_shift_fraction,
        frame_height=height,
        frame_width=width,
    )

    # Stores valid ranges and bad frames in context.
    registration_data.valid_y_range = valid_y_range
    registration_data.valid_x_range = valid_x_range
    registration_data.bad_frames = computed_bad_frames

    # Persists registration results to disk before the optional metrics computation step, so that registration offsets
    # and valid ranges are not lost if the metrics computation fails.
    context.save_runtime()

    # Computes registration quality metrics if enabled and recording has enough frames.
    num_principal_components = config.registration.registration_metric_principal_components
    if num_principal_components > 0 and num_frames >= _MINIMUM_REGISTRATION_METRIC_FRAMES:
        timer.reset()
        compute_pc_metrics(context)
        context.runtime.timing.registration_metrics_time = int(timer.elapsed)
        console.echo(
            message=(
                f"Plane {plane_index} registration metrics processing: complete. Time taken: {timer.elapsed} seconds."
            ),
            level=LogLevel.SUCCESS,
        )
    elif num_principal_components > 0:
        console.echo(
            message=(
                f"Skipping plane {plane_index} registration quality metrics computation. Recording has {num_frames} "
                f"frames, but at least {_MINIMUM_REGISTRATION_METRIC_FRAMES} are required."
            ),
            level=LogLevel.INFO,
        )

    # Persists the final registration state (including metrics if computed) to disk.
    context.save_runtime()

    # Releases registration arrays to free memory. The has_registration_data flag survives re-serialization.
    context.runtime.registration.release_arrays()


@dataclass(frozen=True, slots=True)
class _ReferenceData:
    """Stores precomputed reference data for phase correlation registration.

    Stores taper masks, mean offsets, and FFT kernels computed from the reference image. These are reused across all
    frame batches during registration.
    """

    taper_mask: NDArray[np.float32]
    """The edge taper mask with shape (height, width) for rigid registration."""
    mean_offset: NDArray[np.float32]
    """The mean intensity offset with shape (height, width) for rigid registration."""
    reference_kernel: NDArray[np.complex64]
    """The phase correlation kernel with shape (fft_height, fft_width) for rigid registration."""
    taper_mask_nonrigid: NDArray[np.float32] | None
    """Per-block taper masks with shape (num_blocks, block_height, block_width), or None if nonrigid is disabled."""
    mean_offset_nonrigid: NDArray[np.float32] | None
    """Per-block mean offsets with shape (num_blocks, block_height, block_width), or None if nonrigid is disabled."""
    reference_kernel_nonrigid: NDArray[np.complex64] | None
    """Per-block FFT kernels with shape (num_blocks, block_height, rfft_width), or None if nonrigid is disabled."""
    blocks: RegistrationBlocks | None
    """The registration block structure from compute_registration_blocks, or None if nonrigid is disabled."""


@dataclass(frozen=True, slots=True)
class _BatchRegistrationResult:
    """Stores the output from registering a single batch of frames."""

    frames: NDArray[np.float32]
    """The registered frames with shape (batch_size, height, width)."""
    y_shifts: NDArray[np.int32]
    """The y-direction rigid pixel offsets with shape (batch_size,)."""
    x_shifts: NDArray[np.int32]
    """The x-direction rigid pixel offsets with shape (batch_size,)."""
    correlations: NDArray[np.float32]
    """The phase correlation peak values with shape (batch_size,)."""
    y_shifts_nonrigid: NDArray[np.float32] | None
    """The y-direction nonrigid subpixel offsets with shape (batch_size, num_blocks), or None."""
    x_shifts_nonrigid: NDArray[np.float32] | None
    """The x-direction nonrigid subpixel offsets with shape (batch_size, num_blocks), or None."""
    correlations_nonrigid: NDArray[np.float32] | None
    """The nonrigid correlation values with shape (batch_size, num_blocks), or None."""


def _compute_crop(
    x_offsets: NDArray[np.int32],
    y_offsets: NDArray[np.int32],
    correlations: NDArray[np.float32],
    bad_frame_threshold: float,
    bad_frames: NDArray[np.bool_],
    maximum_shift_fraction: float,
    frame_height: int,
    frame_width: int,
) -> tuple[NDArray[np.bool_], tuple[int, int], tuple[int, int]]:
    """Computes the valid pixel region after registration by analyzing frame shifts.

    After registration, frames that shifted significantly will have undefined pixels at their edges. This function
    determines which pixel region is valid across all frames by finding the maximum shift magnitude. It also
    identifies bad frames that have abnormally large shifts or poor correlation quality, excluding them from the
    valid region calculation to prevent a few outlier frames from unnecessarily shrinking the usable field of view.

    Args:
        x_offsets: The x-direction rigid pixel offsets with shape (num_frames,).
        y_offsets: The y-direction rigid pixel offsets with shape (num_frames,).
        correlations: The phase correlation peak values with shape (num_frames,) indicating registration quality.
        bad_frame_threshold: The threshold multiplier for identifying outlier frames based on shift deviation
            relative to correlation quality.
        bad_frames: A boolean array with shape (num_frames,) of frames already marked as bad from external sources.
        maximum_shift_fraction: The maximum allowed shift as a fraction of the minimum spatial dimension. Frames
            exceeding this threshold are flagged as bad.
        frame_height: The height of each frame in pixels.
        frame_width: The width of each frame in pixels.

    Returns:
        A tuple containing the updated bad_frames boolean array with outliers marked, the valid y-range as
        (y_min, y_max) defining usable rows, and the valid x-range as (x_min, x_max) defining usable columns.
    """
    # Computes median filter window: largest odd number up to array length, capped at maximum.
    # This extracts a smooth baseline trend from the offset time series.
    filter_window = min((len(y_offsets) // 2) * 2 - 1, _MAXIMUM_MEDIAN_FILTER_WINDOW)

    # Subtracts baseline to isolate high-frequency deviations (sudden jumps indicate bad frames). Casts the medfilt
    # output to float32 to prevent float64 promotion of the entire downstream chain.
    delta_x = x_offsets - medfilt(x_offsets, kernel_size=filter_window).astype(np.float32)
    delta_y = y_offsets - medfilt(y_offsets, kernel_size=filter_window).astype(np.float32)

    # Computes offset magnitude normalized by mean offset. If mean is 0 (no motion), delta_xy stays as zeros.
    delta_xy = np.hypot(delta_x, delta_y)
    delta_xy_mean = delta_xy.mean()
    if delta_xy_mean > 0:
        delta_xy = delta_xy / delta_xy_mean

    # Normalizes phase correlation relative to local median to detect quality drops.
    correlation_normalized = correlations / medfilt(correlations, kernel_size=filter_window).astype(np.float32)

    # Combines deviation and correlation metrics: bad frames have large shifts AND/OR poor correlation.
    outlier_metric = delta_xy / np.maximum(0, correlation_normalized)
    x_threshold = maximum_shift_fraction * frame_width * 0.95
    y_threshold = maximum_shift_fraction * frame_height * 0.95
    bad_frames = (
        bad_frames
        | (outlier_metric > bad_frame_threshold * 100)
        | (np.abs(x_offsets) > x_threshold)
        | (np.abs(y_offsets) > y_threshold)
    )

    # Computes valid region from good frames only (excludes outliers from shrinking the FOV).
    # If >50% are bad, falls back to using all frames and warns about registration failure.
    if bad_frames.mean() < _BAD_FRAME_FRACTION_THRESHOLD:
        y_min = np.ceil(np.abs(y_offsets[~bad_frames]).max())
        x_min = np.ceil(np.abs(x_offsets[~bad_frames]).max())
    else:
        console.echo(
            message=(
                "WARNING: >50% of frames have large movements, suggesting that registration has failed to correct "
                "motion artifacts."
            ),
            level=LogLevel.WARNING,
        )
        y_min = np.ceil(np.abs(y_offsets).max())
        x_min = np.ceil(np.abs(x_offsets).max())

    # Valid region is the interior rectangle after accounting for maximum shifts in each direction.
    y_max = frame_height - y_min
    x_max = frame_width - x_min
    valid_y_range = (int(y_min), int(y_max))
    valid_x_range = (int(x_min), int(x_max))

    return bad_frames, valid_y_range, valid_x_range


def _pick_initial_reference(frames: NDArray[np.float32], top_correlations: int = 20) -> NDArray[np.float32]:
    """Computes the initial reference image from a set of frames.

    Identifies the seed frame as the frame with the largest correlations with other frames, then averages the seed
    frame with its top k correlated pairs to produce the initial reference.

    Args:
        frames: The processed recording's frames with shape (num_frames, height, width).
        top_correlations: The number of top frame correlations to average.

    Returns:
        The initial reference image with shape (height, width).
    """
    num_frames, height, width = frames.shape

    # Flattens frames and subtracts mean for correlation computation.
    frames_flat = frames.reshape(num_frames, -1)
    frames_flat -= frames_flat.mean(axis=1, keepdims=True)

    # Normalizes frames and computes correlation matrix.
    frame_norms = np.linalg.norm(frames_flat, axis=1, keepdims=True)
    frames_normalized = frames_flat / frame_norms
    correlation_matrix = frames_normalized @ frames_normalized.T

    # Finds the frame with the highest mean correlation to other frames (excluding self-correlation).
    top_correlations_per_frame = np.partition(correlation_matrix, kth=-(top_correlations + 1), axis=1)[
        :, -top_correlations:-1
    ]
    mean_top_correlations = np.mean(top_correlations_per_frame, axis=1)
    seed_index = np.argmax(mean_top_correlations)

    # Averages the seed frame with its top correlated frames. Uses mean-subtracted frames intentionally—this initial
    # reference only bootstraps iterative refinement in _compute_reference, which replaces it with original frames.
    top_indices = np.argpartition(correlation_matrix[seed_index, :], kth=-top_correlations)[-top_correlations:]
    reference_image = np.mean(frames_flat[top_indices, :], axis=0)

    return np.reshape(reference_image, shape=(int(height), int(width)))


def _compute_reference(
    frames: NDArray[np.float32],
    one_photon_enabled: bool,
    pre_smoothing_sigma: float,
    spatial_highpass_window: int,
    edge_taper_pixels: float,
    spatial_smoothing_sigma: float,
    maximum_shift_fraction: float,
    temporal_smoothing_sigma: float,
    workers: int,
) -> NDArray[np.float32]:
    """Computes the reference image through iterative alignment.

    Selects an initial reference by finding the frame most correlated with other frames, then refines it through
    8 iterations of rigid registration. In each iteration, all frames are aligned to the current reference using
    phase correlation, then the reference is updated to be the mean of the top-correlated frames. This progressive
    refinement produces a sharp, low-noise reference that represents the stable structure across frames.

    Args:
        frames: The frames to use for reference computation with shape (num_frames, height, width).
        one_photon_enabled: Determines whether to apply one-photon preprocessing, which includes spatial smoothing
            followed by high-pass filtering.
        pre_smoothing_sigma: The standard deviation of Gaussian smoothing applied before high-pass filtering.
        spatial_highpass_window: The window size for the spatial high-pass filter that removes low-frequency background.
        edge_taper_pixels: Controls the steepness of the edge taper falloff. Larger values produce a more gradual
            taper that suppresses border artifacts during phase correlation.
        spatial_smoothing_sigma: The standard deviation of Gaussian smoothing applied to phase correlation maps.
        maximum_shift_fraction: The maximum allowed shift as a fraction of the minimum spatial dimension.
            The search window is limited to min(height, width) * maximum_shift_fraction pixels.
        temporal_smoothing_sigma: The standard deviation for temporal Gaussian smoothing of correlation maps.
            If 0, no smoothing is applied.
        workers: The number of parallel workers for FFT computation. Use -1 for all available cores.

    Returns:
        The computed reference image with shape (height, width).
    """
    # Selects the initial reference by averaging together the most stable frames.
    reference_image = _pick_initial_reference(frames)

    # Applies one-photon preprocessing.
    if one_photon_enabled:
        if pre_smoothing_sigma > 0:
            reference_image = apply_spatial_smoothing(data=reference_image, window=int(pre_smoothing_sigma))
            frames = apply_spatial_smoothing(data=frames, window=int(pre_smoothing_sigma))
        reference_image = apply_spatial_high_pass(data=reference_image, window=spatial_highpass_window)
        frames = apply_spatial_high_pass(data=frames, window=spatial_highpass_window)

    # Computes taper slope based on registration mode.
    taper_slope = edge_taper_pixels if one_photon_enabled else 3 * spatial_smoothing_sigma

    # Iteratively refines the reference image. 8 iterations is empirically tuned from original suite2p;
    # each iteration includes progressively more frames, converging to ~50% of frames by the final iteration.
    num_iterations = 8
    for iteration in range(num_iterations):
        # Prepares edge taper mask and phase correlation kernel for current reference.
        taper_mask, mean_offset = compute_edge_taper(
            reference_image=reference_image,
            taper_slope=taper_slope,
        )

        # Computes rigid registration shifts via phase correlation.
        y_shifts, x_shifts, correlations = compute_rigid_shifts(
            frames=apply_edge_taper(frames=frames, taper_mask=taper_mask, mean_offset=mean_offset),
            reference_kernel=compute_phase_correlation_kernel(
                reference_image=reference_image,
                smoothing_sigma=spatial_smoothing_sigma,
            ),
            maximum_shift_fraction=maximum_shift_fraction,
            temporal_smoothing_sigma=temporal_smoothing_sigma,
            workers=workers,
        )

        # Applies computed shifts to align all frames to current reference.
        for frame, y_shift, x_shift in zip(frames, y_shifts, x_shifts, strict=False):
            frame[:] = shift_frame(frame=frame, y_shift=y_shift, x_shift=x_shift)

        # Selects top-correlated frames for next reference, excluding rank 0 (the frame most correlated with the
        # current reference) to prevent self-reinforcing bias in the iterative refinement.
        # Number of frames increases each iteration: ~6% at iter 0, ~31% at iter 4, ~50% at iter 7.
        num_frames_for_reference = max(2, int(frames.shape[0] * (1.0 + iteration) / (2 * num_iterations)))
        sorted_indices = np.argsort(-correlations)[1:num_frames_for_reference]

        # Updates reference as the mean of the best-aligned frames. Input frames are float32, mean preserves dtype.
        reference_image = frames[sorted_indices].mean(axis=0)

        # Centers the reference by reversing the mean shift of selected frames.
        reference_image = shift_frame(
            frame=reference_image,
            y_shift=int(np.round(-y_shifts[sorted_indices].mean())),
            x_shift=int(np.round(-x_shifts[sorted_indices].mean())),
        )

    return reference_image


def _register_frames_batch(
    reference_data: _ReferenceData,
    frames: NDArray[np.float32],
    normalization_minimum: float,
    normalization_maximum: float,
    bidirectional_phase_offset: int,
    one_photon_enabled: bool,
    pre_smoothing_sigma: float,
    spatial_highpass_window: int,
    temporal_smoothing_sigma: float,
    maximum_shift_fraction: float,
    nonrigid_enabled: bool,
    signal_to_noise_threshold: float,
    maximum_block_shift: float,
    workers: int,
) -> _BatchRegistrationResult:
    """Registers the input batch of frames to the reference image using rigid and optionally nonrigid phase correlation.

    Args:
        reference_data: Precomputed reference data containing taper masks, mean offsets, and FFT kernels for both
            rigid and nonrigid registration.
        frames: The batch of frames with shape (batch_size, height, width) sampled from the processed recording.
        normalization_minimum: The minimum intensity value for clipping frames before correlation.
        normalization_maximum: The maximum intensity value for clipping frames before correlation.
        bidirectional_phase_offset: The pixel offset to correct bidirectional scanning artifacts.
        one_photon_enabled: Determines whether to apply one-photon preprocessing, which includes spatial smoothing
            followed by high-pass filtering.
        pre_smoothing_sigma: The standard deviation of Gaussian smoothing applied before high-pass filtering.
        spatial_highpass_window: The window size for the spatial high-pass filter that removes low-frequency background.
        temporal_smoothing_sigma: The standard deviation for temporal Gaussian smoothing of correlation maps.
            If 0, no smoothing is applied.
        maximum_shift_fraction: The maximum allowed shift as a fraction of the minimum spatial dimension.
            The search window is limited to min(height, width) * maximum_shift_fraction pixels.
        nonrigid_enabled: Determines whether to apply nonrigid (piecewise) registration after rigid alignment.
        signal_to_noise_threshold: The SNR threshold below which additional smoothing is applied to correlation
            peaks. Higher values apply more smoothing; typical values range from 1.0 to 1.5.
        maximum_block_shift: The maximum allowed shift for nonrigid blocks in pixels.
        workers: The number of parallel workers for FFT computation. Use -1 for all available cores.

    Returns:
        A _BatchRegistrationResult containing registered frames and computed shifts. Nonrigid arrays are None if
        nonrigid registration is disabled.
    """
    # Corrects bidirectional scanning artifacts if offset is non-zero.
    if bidirectional_phase_offset != 0:
        apply_bidirectional_phase_correction(frames=frames, bidirectional_phase_offset=bidirectional_phase_offset)

    # Creates a working copy for correlation computation; original frames are shifted separately. Temporal smoothing
    # is only applied to the correlation maps inside compute_rigid_shifts, not to the raw frames here.
    frames_smooth = frames.copy()

    # Applies one-photon preprocessing: spatial smoothing followed by high-pass filtering.
    if one_photon_enabled:
        if pre_smoothing_sigma > 0:
            frames_smooth = apply_spatial_smoothing(data=frames_smooth, window=int(pre_smoothing_sigma))
        frames_smooth = apply_spatial_high_pass(data=frames_smooth, window=spatial_highpass_window)

    # Clips intensity range to reduce influence of outlier pixels on correlation.
    frames_for_correlation = (
        np.clip(frames_smooth, normalization_minimum, normalization_maximum)
        if normalization_minimum > -np.inf
        else frames_smooth
    )

    # Phase 1: Rigid registration - computes whole-frame translation shifts.
    y_shifts, x_shifts, correlations = compute_rigid_shifts(
        frames=apply_edge_taper(
            frames=frames_for_correlation,
            taper_mask=reference_data.taper_mask,
            mean_offset=reference_data.mean_offset,
        ),
        reference_kernel=reference_data.reference_kernel,
        maximum_shift_fraction=maximum_shift_fraction,
        temporal_smoothing_sigma=temporal_smoothing_sigma,
        workers=workers,
    )

    # Applies rigid shifts to original (unsmoothed) frames.
    for frame, y_shift, x_shift in zip(frames, y_shifts, x_shifts, strict=False):
        frame[:] = shift_frame(frame=frame, y_shift=y_shift, x_shift=x_shift)

    # Phase 2: Nonrigid registration - computes per-block subpixel shifts to correct local deformations.
    if nonrigid_enabled:
        # Extracts nonrigid reference data. Fallback assignments are for type checker only; these are guaranteed to be
        # present when nonrigid_enabled is True.
        blocks = (
            reference_data.blocks
            if reference_data.blocks is not None
            else ([], [], (0, 0), (0, 0), np.empty(0, dtype=np.float32))
        )
        taper_mask_nonrigid = (
            reference_data.taper_mask_nonrigid
            if reference_data.taper_mask_nonrigid is not None
            else np.empty((0, 0, 0), dtype=np.float32)
        )
        mean_offset_nonrigid = (
            reference_data.mean_offset_nonrigid
            if reference_data.mean_offset_nonrigid is not None
            else np.empty((0, 0, 0), dtype=np.float32)
        )
        reference_kernel_nonrigid = (
            reference_data.reference_kernel_nonrigid
            if reference_data.reference_kernel_nonrigid is not None
            else np.empty((0, 0, 0), dtype=np.complex64)
        )

        # Applies rigid shifts to the smoothed working copy so nonrigid phase correlation operates on pre-aligned data.
        # Without this, the per-block shifts would capture both global translation and local deformation,
        # double-counting
        # the rigid component that was already corrected on the original frames.
        for frame_smooth, y_shift, x_shift in zip(frames_smooth, y_shifts, x_shifts, strict=False):
            frame_smooth[:] = shift_frame(frame=frame_smooth, y_shift=y_shift, x_shift=x_shift)

        # Re-clips intensity range after rigid shift for nonrigid correlation.
        frames_for_correlation = (
            np.clip(frames_smooth, normalization_minimum, normalization_maximum)
            if normalization_minimum > -np.inf
            else frames_smooth
        )

        # Computes block-wise subpixel shifts using phase correlation on each block.
        y_shifts_nonrigid, x_shifts_nonrigid, correlations_nonrigid = compute_nonrigid_shifts(
            frames=frames_for_correlation,
            taper_mask=taper_mask_nonrigid,
            mean_offset=mean_offset_nonrigid,
            reference_kernel=reference_kernel_nonrigid,
            snr_threshold=signal_to_noise_threshold,
            smoothing_kernel=blocks[-1],
            x_blocks=blocks[1],
            y_blocks=blocks[0],
            maximum_shift=maximum_block_shift,
            workers=workers,
        )

        # Applies nonrigid warping to original frames using computed block shifts.
        frames = apply_nonrigid_correction(
            frames=frames,
            y_blocks=blocks[0],
            x_blocks=blocks[1],
            block_counts=blocks[2],
            y_block_shifts=y_shifts_nonrigid,
            x_block_shifts=x_shifts_nonrigid,
        )
    else:
        y_shifts_nonrigid, x_shifts_nonrigid, correlations_nonrigid = None, None, None

    return _BatchRegistrationResult(
        frames=frames,
        y_shifts=y_shifts,
        x_shifts=x_shifts,
        correlations=correlations,
        y_shifts_nonrigid=y_shifts_nonrigid,
        x_shifts_nonrigid=x_shifts_nonrigid,
        correlations_nonrigid=correlations_nonrigid,
    )


def _shift_frames_batch(
    frames: NDArray[np.float32],
    y_offsets: NDArray[np.int32],
    x_offsets: NDArray[np.int32],
    y_offsets_nonrigid: NDArray[np.float32] | None,
    x_offsets_nonrigid: NDArray[np.float32] | None,
    blocks: RegistrationBlocks | None,
    bidirectional_phase_offset: int,
    bidirectional_phase_corrected: bool,
    nonrigid_enabled: bool,
) -> NDArray[np.float32]:
    """Applies precomputed registration shifts to a batch of frames.

    Used to register the second channel using shifts computed from the first channel, avoiding redundant shift
    computation.

    Args:
        frames: The batch of frames with shape (batch_size, height, width).
        y_offsets: The y-direction rigid pixel offsets with shape (batch_size,).
        x_offsets: The x-direction rigid pixel offsets with shape (batch_size,).
        y_offsets_nonrigid: The y-direction nonrigid subpixel offsets with shape (batch_size, num_blocks), or None.
        x_offsets_nonrigid: The x-direction nonrigid subpixel offsets with shape (batch_size, num_blocks), or None.
        blocks: The registration block information tuple containing (y_blocks, x_blocks, block_counts,
            actual_block_size, smoothing_kernel) from compute_registration_blocks, or None if nonrigid is disabled.
        bidirectional_phase_offset: The pixel offset to correct bidirectional scanning artifacts.
        bidirectional_phase_corrected: Determines whether bidirectional correction was already applied to input frames.
        nonrigid_enabled: Determines whether to apply nonrigid (piecewise) registration after rigid alignment.

    Returns:
        The shifted frames with shape (batch_size, height, width).
    """
    # Corrects bidirectional scanning artifact if not already applied.
    if bidirectional_phase_offset != 0 and not bidirectional_phase_corrected:
        apply_bidirectional_phase_correction(
            frames=frames,
            bidirectional_phase_offset=bidirectional_phase_offset,
        )

    # Applies rigid (whole-frame) shifts.
    for frame, y_offset, x_offset in zip(frames, y_offsets, x_offsets, strict=False):
        frame[:] = shift_frame(frame=frame, y_shift=y_offset, x_shift=x_offset)

    # Applies nonrigid (per-block) warping if enabled. Fallback assignments are for type checker only; these are
    # guaranteed to be present when nonrigid_enabled is True.
    if nonrigid_enabled:
        _blocks = blocks if blocks is not None else ([], [], (0, 0), (0, 0), np.empty(0))
        _y_nr = y_offsets_nonrigid if y_offsets_nonrigid is not None else np.empty((0, 0), dtype=np.float32)
        _x_nr = x_offsets_nonrigid if x_offsets_nonrigid is not None else np.empty((0, 0), dtype=np.float32)
        frames = apply_nonrigid_correction(
            frames=frames,
            y_blocks=_blocks[0],
            x_blocks=_blocks[1],
            block_counts=_blocks[2],
            y_block_shifts=_y_nr,
            x_block_shifts=_x_nr,
        )

    return frames


def _register_alignment_channel(context: RuntimeContext) -> None:
    """Computes registration offsets from the alignment channel and applies them to that channel's frames.

    The alignment channel is determined by config.registration.align_by_first_channel. If True, channel 1 is used;
    if False, channel 2 is used. This function computes the reference image, calculates rigid and optionally nonrigid
    registration offsets, and applies them to all frames. Results are stored in context.runtime.registration and the
    mean image is stored in the appropriate detection field.

    Args:
        context: The RuntimeContext containing configuration, acquisition parameters, and runtime data.
    """
    # Extracts configuration parameters.
    config = context.configuration
    align_by_first_channel = config.registration.align_by_first_channel
    one_photon_enabled = config.one_photon_registration.enabled
    pre_smoothing_sigma = config.one_photon_registration.pre_smoothing_sigma
    spatial_highpass_window = config.one_photon_registration.spatial_highpass_window
    edge_taper_pixels = config.one_photon_registration.edge_taper_pixels
    spatial_smoothing_sigma = config.registration.spatial_smoothing_sigma
    temporal_smoothing_sigma = config.registration.temporal_smoothing_sigma
    maximum_shift_fraction = config.registration.maximum_shift_fraction
    normalize_frames = config.registration.normalize_frames
    batch_size = config.registration.batch_size
    reference_frame_count = config.registration.reference_frame_count
    nonrigid_enabled = config.nonrigid_registration.enabled
    block_size = config.nonrigid_registration.block_size
    signal_to_noise_threshold = config.nonrigid_registration.signal_to_noise_threshold
    maximum_block_shift = config.nonrigid_registration.maximum_block_shift
    parallel_workers = config.runtime.parallel_workers
    enable_bidiphase_computation = config.registration.compute_bidirectional_phase_offset
    initial_bidirectional_phase_offset = config.registration.bidirectional_phase_offset_override

    # Extracts runtime IO data.
    io_data = context.runtime.io
    plane_index = io_data.plane_index if io_data.plane_index is not None else 0
    height, width, num_frames = io_data.frame_height, io_data.frame_width, io_data.frame_count
    bidirectional_phase_corrected = context.runtime.registration.bidirectional_phase_corrected

    # Selects channel paths based on alignment configuration.
    if align_by_first_channel:
        binary_path = io_data.registered_binary_path
        channel_label = "channel 1"
    else:
        binary_path = io_data.registered_binary_path_channel_2
        channel_label = "channel 2"

    # Validates binary path exists.
    if binary_path is None:
        console.error(
            message=(
                f"Unable to register {channel_label} frames for plane {plane_index}. The plane's RuntimeContext "
                f"instance does not contain the path to the plane's {channel_label} binary file."
            ),
            error=ValueError,
        )

    # Opens BinaryFile and performs registration in-place.
    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()
    with BinaryFile(height=height, width=width, file_path=binary_path, frame_number=num_frames) as frames_file:
        # Tracks the bidirectional phase offset (may be updated from data).
        bidirectional_phase_offset = initial_bidirectional_phase_offset

        # Samples frames evenly across the recording and converts to float32 for processing.
        sample_indices = np.linspace(0, num_frames, 1 + np.minimum(reference_frame_count, num_frames), dtype=int)[:-1]
        frames = frames_file[sample_indices].astype(np.float32)

        # Computes bidiphase shift if enabled and not already set.
        if enable_bidiphase_computation and bidirectional_phase_offset == 0 and not bidirectional_phase_corrected:
            bidirectional_phase_offset = compute_bidirectional_phase_offset(frames=frames)
            console.echo(
                message=(
                    f"Plane {plane_index} estimated bidirectional phase offset: {bidirectional_phase_offset} pixels."
                ),
                level=LogLevel.INFO,
            )

            # Applies bidirectional phase correction to the sampled frames.
            if bidirectional_phase_offset != 0:
                apply_bidirectional_phase_correction(
                    frames=frames,
                    bidirectional_phase_offset=bidirectional_phase_offset,
                )

        console.echo(message=f"Computing plane {plane_index} reference frame...", level=LogLevel.INFO)
        timer.reset()
        reference_image = _compute_reference(
            frames=frames,
            one_photon_enabled=one_photon_enabled,
            pre_smoothing_sigma=pre_smoothing_sigma,
            spatial_highpass_window=spatial_highpass_window,
            edge_taper_pixels=edge_taper_pixels,
            spatial_smoothing_sigma=spatial_smoothing_sigma,
            maximum_shift_fraction=maximum_shift_fraction,
            temporal_smoothing_sigma=temporal_smoothing_sigma,
            workers=parallel_workers,
        )
        console.echo(
            message=f"Plane {plane_index} reference frame: computed. Time taken: {timer.elapsed} seconds.",
            level=LogLevel.SUCCESS,
        )

        # Normalizes reference image by clipping to the 1st and 99th percentiles.
        reference_original = reference_image.copy()
        if normalize_frames:
            normalization_minimum = float(np.percentile(reference_image, 1))
            normalization_maximum = float(np.percentile(reference_image, 99))
            reference_image = np.clip(reference_image, normalization_minimum, normalization_maximum)
        else:
            normalization_minimum, normalization_maximum = -np.inf, np.inf

        # Determines bidiphase for frame registration.
        if bidirectional_phase_offset != 0 and not bidirectional_phase_corrected:
            bidiphase_for_registration = bidirectional_phase_offset
        else:
            bidiphase_for_registration = 0

        # Computes registration masks for the reference image.
        taper_slope = edge_taper_pixels if one_photon_enabled else 3 * spatial_smoothing_sigma

        taper_mask, mean_offset = compute_edge_taper(
            reference_image=reference_image,
            taper_slope=taper_slope,
        )
        reference_kernel = compute_phase_correlation_kernel(
            reference_image=reference_image,
            smoothing_sigma=spatial_smoothing_sigma,
        )

        # Computes nonrigid reference data if enabled.
        if nonrigid_enabled:
            blocks = compute_registration_blocks(height=height, width=width, block_size=block_size)
            taper_mask_nonrigid, mean_offset_nonrigid, reference_kernel_nonrigid = compute_nonrigid_reference_data(
                reference_image=reference_image,
                taper_slope=taper_slope,
                smoothing_sigma=spatial_smoothing_sigma,
                y_blocks=blocks[0],
                x_blocks=blocks[1],
            )
        else:
            blocks = None
            taper_mask_nonrigid, mean_offset_nonrigid, reference_kernel_nonrigid = None, None, None

        # Packages the compute reference data into a helper dictionary before applying registration offsets to batches
        # of frames.
        reference_data = _ReferenceData(
            taper_mask=taper_mask,
            mean_offset=mean_offset,
            reference_kernel=reference_kernel,
            taper_mask_nonrigid=taper_mask_nonrigid,
            mean_offset_nonrigid=mean_offset_nonrigid,
            reference_kernel_nonrigid=reference_kernel_nonrigid,
            blocks=blocks,
        )

        # Registers frames to the reference image.
        mean_image = np.zeros((height, width), dtype=np.float32)
        rigid_offsets_batches: list[tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]] = []
        nonrigid_offsets_batches: list[tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]] = []

        timer.reset()
        console.echo(
            message=f"Computing and applying plane {plane_index} registration offsets for {channel_label}...",
            level=LogLevel.INFO,
        )

        for batch_start_np in console.track(
            np.arange(0, num_frames, batch_size),
            description=f"Registering batches of {batch_size} frames",
            unit="batch",
        ):
            batch_start = int(batch_start_np)
            batch_end = min(batch_start + batch_size, num_frames)
            frames = frames_file[batch_start:batch_end].astype(np.float32)

            batch_result = _register_frames_batch(
                reference_data=reference_data,
                frames=frames,
                normalization_minimum=normalization_minimum,
                normalization_maximum=normalization_maximum,
                bidirectional_phase_offset=bidiphase_for_registration,
                one_photon_enabled=one_photon_enabled,
                pre_smoothing_sigma=pre_smoothing_sigma,
                spatial_highpass_window=spatial_highpass_window,
                temporal_smoothing_sigma=temporal_smoothing_sigma,
                maximum_shift_fraction=maximum_shift_fraction,
                nonrigid_enabled=nonrigid_enabled,
                signal_to_noise_threshold=signal_to_noise_threshold,
                maximum_block_shift=maximum_block_shift,
                workers=parallel_workers,
            )

            rigid_offsets_batches.append((batch_result.y_shifts, batch_result.x_shifts, batch_result.correlations))
            if nonrigid_enabled:
                # Fallback assignments are for type checker only; guaranteed present when nonrigid_enabled is True.
                y_shifts_nonrigid = (
                    batch_result.y_shifts_nonrigid
                    if batch_result.y_shifts_nonrigid is not None
                    else np.empty((0, 0), dtype=np.float32)
                )
                x_shifts_nonrigid = (
                    batch_result.x_shifts_nonrigid
                    if batch_result.x_shifts_nonrigid is not None
                    else np.empty((0, 0), dtype=np.float32)
                )
                correlations_nonrigid = (
                    batch_result.correlations_nonrigid
                    if batch_result.correlations_nonrigid is not None
                    else np.empty((0, 0), dtype=np.float32)
                )
                nonrigid_offsets_batches.append((y_shifts_nonrigid, x_shifts_nonrigid, correlations_nonrigid))

            mean_image += batch_result.frames.sum(axis=0)

            # Converts back to int16 for BinaryFile storage and writes in-place.
            frames_int16 = np.clip(batch_result.frames, -32768, 32767).astype(np.int16)
            frames_file[batch_start:batch_end] = frames_int16

        # Normalizes accumulated sum to get mean image.
        mean_image /= num_frames

        console.echo(
            message=(
                f"Plane {plane_index} {channel_label} registration offsets: computed and applied. "
                f"Time taken: {timer.elapsed} seconds."
            ),
            level=LogLevel.SUCCESS,
        )

    # Combines batch results into full arrays.
    rigid_y_offsets, rigid_x_offsets, rigid_correlations = combine_rigid_offsets(rigid_offsets_batches)
    if nonrigid_enabled:
        nonrigid_y_offsets, nonrigid_x_offsets, nonrigid_correlations = combine_nonrigid_offsets(
            nonrigid_offsets_batches
        )
    else:
        nonrigid_y_offsets, nonrigid_x_offsets, nonrigid_correlations = None, None, None

    # Stores results in context.
    registration_data = context.runtime.registration
    registration_data.reference_image = reference_original
    registration_data.normalization_minimum = int(normalization_minimum) if normalization_minimum > -np.inf else 0
    registration_data.normalization_maximum = int(normalization_maximum) if normalization_maximum < np.inf else 0
    registration_data.bidirectional_phase_offset = bidirectional_phase_offset
    registration_data.bidirectional_phase_corrected = bidirectional_phase_offset != 0
    registration_data.rigid_y_offsets = rigid_y_offsets
    registration_data.rigid_x_offsets = rigid_x_offsets
    registration_data.rigid_correlations = rigid_correlations
    if nonrigid_enabled:
        registration_data.nonrigid_y_offsets = nonrigid_y_offsets
        registration_data.nonrigid_x_offsets = nonrigid_x_offsets
        registration_data.nonrigid_correlations = nonrigid_correlations

    # Stores mean image in the appropriate field based on which channel was aligned.
    if align_by_first_channel:
        context.runtime.detection.mean_image = mean_image
    else:
        context.runtime.detection.mean_image_channel_2 = mean_image


def _register_secondary_channel(context: RuntimeContext) -> None:
    """Applies precomputed registration offsets to the secondary (non-alignment) channel's frames.

    The secondary channel is the opposite of the alignment channel. If align_by_first_channel is True, this function
    processes channel 2; if False, it processes channel 1. Registration offsets are read from
    context.runtime.registration (computed by _register_alignment_channel) and applied to all frames.

    Args:
        context: The RuntimeContext containing configuration, acquisition parameters, and runtime data.
    """
    # Extracts configuration parameters.
    config = context.configuration
    align_by_first_channel = config.registration.align_by_first_channel
    nonrigid_enabled = config.nonrigid_registration.enabled
    block_size = config.nonrigid_registration.block_size
    batch_size = config.registration.batch_size

    # Extracts runtime IO data.
    io_data = context.runtime.io
    plane_index = io_data.plane_index if io_data.plane_index is not None else 0
    height, width, num_frames = io_data.frame_height, io_data.frame_width, io_data.frame_count

    # Extracts registration data (offsets computed from alignment channel).
    registration_data = context.runtime.registration
    bidirectional_phase_offset = registration_data.bidirectional_phase_offset
    bidirectional_phase_corrected = registration_data.bidirectional_phase_corrected

    # Extracts rigid offsets and converts to int32 for shift operations. Fallback to empty arrays is for type narrowing
    # only; offsets are always present since _register_alignment_channel populates them before this function is called.
    y_offsets = registration_data.rigid_y_offsets if registration_data.rigid_y_offsets is not None else np.empty(0)
    x_offsets = registration_data.rigid_x_offsets if registration_data.rigid_x_offsets is not None else np.empty(0)
    y_offsets_int = y_offsets.astype(np.int32)
    x_offsets_int = x_offsets.astype(np.int32)

    # Extracts nonrigid offsets if enabled.
    y_offsets_nonrigid = registration_data.nonrigid_y_offsets if nonrigid_enabled else None
    x_offsets_nonrigid = registration_data.nonrigid_x_offsets if nonrigid_enabled else None

    # Selects channel paths based on alignment configuration (uses the opposite channel from alignment).
    if align_by_first_channel:
        binary_path = io_data.registered_binary_path_channel_2
        channel_label = "channel 2"
    else:
        binary_path = io_data.registered_binary_path
        channel_label = "channel 1"

    # Validates binary path exists.
    if binary_path is None:
        console.error(
            message=(
                f"Unable to register {channel_label} frames for plane {plane_index}. The plane's RuntimeContext "
                f"instance does not contain the path to the plane's {channel_label} binary file."
            ),
            error=ValueError,
        )

    # Computes block structure if nonrigid is enabled.
    blocks = None
    if nonrigid_enabled:
        blocks = compute_registration_blocks(height=height, width=width, block_size=block_size)

    # Opens BinaryFile and performs registration in-place.
    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()
    mean_image = np.zeros((height, width), dtype=np.float32)

    with BinaryFile(height=height, width=width, file_path=binary_path, frame_number=num_frames) as frames_file:
        console.echo(
            message=f"Applying plane {plane_index} registration offsets to {channel_label}...",
            level=LogLevel.INFO,
        )
        timer.reset()

        # Prepares nonrigid offset arrays outside the loop. Fallback to empty arrays is for type narrowing only;
        # offsets are always present when nonrigid_enabled is True.
        if nonrigid_enabled:
            nonrigid_y_offsets_full = (
                y_offsets_nonrigid if y_offsets_nonrigid is not None else np.empty((0, 0), dtype=np.float32)
            )
            nonrigid_x_offsets_full = (
                x_offsets_nonrigid if x_offsets_nonrigid is not None else np.empty((0, 0), dtype=np.float32)
            )

        # Processes frames in batches to limit memory usage.
        for batch_start_np in console.track(
            np.arange(0, num_frames, batch_size),
            description=f"Registering batches of {batch_size} frames",
            unit="batch",
        ):
            batch_start = int(batch_start_np)
            batch_end = min(batch_start + batch_size, num_frames)

            # Loads batch and extracts corresponding offsets.
            frames = frames_file[batch_start:batch_end].astype(np.float32)
            y_offsets_batch = y_offsets_int[batch_start:batch_end]
            x_offsets_batch = x_offsets_int[batch_start:batch_end]

            if nonrigid_enabled:
                y_offsets_nonrigid_batch = nonrigid_y_offsets_full[batch_start:batch_end]
                x_offsets_nonrigid_batch = nonrigid_x_offsets_full[batch_start:batch_end]
            else:
                y_offsets_nonrigid_batch, x_offsets_nonrigid_batch = None, None

            # Applies precomputed shifts (rigid + nonrigid if enabled).
            frames = _shift_frames_batch(
                frames=frames,
                y_offsets=y_offsets_batch,
                x_offsets=x_offsets_batch,
                y_offsets_nonrigid=y_offsets_nonrigid_batch,
                x_offsets_nonrigid=x_offsets_nonrigid_batch,
                blocks=blocks,
                bidirectional_phase_offset=bidirectional_phase_offset,
                bidirectional_phase_corrected=bidirectional_phase_corrected,
                nonrigid_enabled=nonrigid_enabled,
            )

            # Accumulates frame sum for mean image computation.
            mean_image += frames.sum(axis=0)

            # Converts back to int16 for BinaryFile storage and writes in-place.
            frames_int16 = np.clip(frames, -32768, 32767).astype(np.int16)
            frames_file[batch_start:batch_end] = frames_int16

        # Normalizes accumulated sum to get mean image.
        mean_image /= num_frames

        console.echo(
            message=(
                f"Plane {plane_index} {channel_label} registration offsets: applied. "
                f"Time taken: {timer.elapsed} seconds."
            ),
            level=LogLevel.SUCCESS,
        )

    # Stores mean image in the appropriate field based on which channel was processed.
    if align_by_first_channel:
        context.runtime.detection.mean_image_channel_2 = mean_image
    else:
        context.runtime.detection.mean_image = mean_image
