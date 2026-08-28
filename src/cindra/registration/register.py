"""Provides the frame registration (motion correction) entry point for the single-recording cindra processing
pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numba
import numpy as np
from scipy.signal import medfilt
from ataraxis_time import PrecisionTimer, TimerPrecisions
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]
from ataraxis_base_utilities import LogLevel, console

from ..io import (
    BinaryFile,
    clear_registration_marker,
    create_registration_marker,
    resolve_active_binary_marker,
)
from .gpu import GpuRegistrationBackend
from .batch import ReferenceData, RegistrationBlocks, BatchRegistrationResult
from .rigid import (
    translate_frame,
    apply_edge_taper,
    compute_edge_taper,
    compute_rigid_offsets,
    compute_phase_correlation_kernel,
)
from .utils import (
    combine_rigid_offsets,
    apply_spatial_high_pass,
    apply_spatial_smoothing,
    combine_nonrigid_offsets,
)
from ..layout import RegistrationArrays
from .metrics import compute_pc_metrics
from .nonrigid import (
    compute_nonrigid_offsets,
    apply_nonrigid_correction,
    compute_nonrigid_reference_data,
)
from ..detection import compute_registration_blocks
from .bidirectional_phase_correction import compute_bidirectional_phase_offset, apply_bidirectional_phase_correction

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Iterator

    from numpy.typing import NDArray

    from ..dataclasses import IOData, RuntimeContext

_MINIMUM_REGISTRATION_METRIC_FRAMES: int = 1500
"""The minimum number of frames required to compute registration quality metrics."""

_BAD_FRAME_FRACTION_THRESHOLD: float = 0.5
"""The threshold fraction of bad frames at or above which registration is considered failed."""

_MAXIMUM_MEDIAN_FILTER_WINDOW: int = 101
"""The maximum median filter window size for offset time series smoothing."""

_OFFSET_MARGIN_FRACTION: float = 0.95
"""The fraction of the maximum allowed offset above which a frame is flagged as bad."""

_OUTLIER_METRIC_SCALE: int = 100
"""The scaling factor applied to the bad frame threshold before it is compared against the outlier metric."""


def register_plane(context: RuntimeContext, *, workers: int, device: int | None = None) -> None:
    """Registers (motion-corrects) all frames for a single imaging plane specified by the input runtime context.

    Computes registration offsets from the alignment channel (determined by
    config.registration.align_by_first_channel), then applies those offsets to both channels. If two-step registration
    is enabled, a refinement pass is performed using the mean of registered frames as the reference.

    All configuration is read from context.configuration, file paths from context.runtime.io, and results are stored in
    context.runtime.registration, context.runtime.detection, and context.runtime.timing. The registered frames are
    written back into the plane's channel binaries in place. The runtime data is persisted with context.save_runtime()
    before returning, after which the registration arrays are released from memory, so consumers re-acquire them with
    memory_map_arrays() or load_arrays().

    Notes:
        The worker count drives both the FFT thread pool used by phase correlation and the Numba thread mask used by
        the nonrigid warping kernels. The Numba mask is thread-local, so concurrently dispatched planes can hold
        different worker budgets inside a single process.

        The device argument selects where both channels are registered, running the pass on a CUDA device when the
        caller names one and on the host CPU otherwise. Only the alignment channel resolves offsets against a
        reference, and the secondary channel receives those same offsets, so the two channels hold the same correction.
        The secondary channel reuses the device state the alignment pass built.

        A '<binary>.registering' marker guards every one of the plane's channel binaries for the whole registration.
        Only the alignment channel is registered against a reference, and the secondary channel receives the offsets
        computed from that channel. The two binaries therefore agree about whether motion has been removed only once
        both rewrites have finished. The markers are cleared after the registration outputs that describe those
        rewrites are persisted, and an interrupted run leaves them behind for the binarization stage to act on.

    Args:
        context: The RuntimeContext containing configuration, file paths, and mutable runtime data structures. Modified
            in-place to store registration outputs including reference image, offsets, mean images, and timing data.
        workers: The number of parallel workers allocated to this registration job. Must be a positive integer.
        device: The zero-based index of the CUDA device to register this plane on. Use None to register the plane on
            the host CPU.
    """
    # The Numba thread mask is thread-local and cannot exceed the core count Numba detected at import time.
    numba.set_num_threads(min(workers, numba.config.NUMBA_NUM_THREADS))

    config = context.configuration
    io_data = context.runtime.io
    registration_data = context.runtime.registration
    plane_index = io_data.plane_index if io_data.plane_index is not None else 0

    # Refuses to consume a binary that an interrupted conversion or registration left in an indeterminate state. This
    # check precedes the skip and re-registration branches below, because neither the registration outputs nor their
    # absence reveal that the binary itself holds a mixture of finished and unfinished frames.
    _validate_binaries_are_not_mid_write(io_data=io_data, plane_index=plane_index)

    if registration_data.is_registered(output_path=io_data.output_path) and not config.registration.repeat_registration:
        console.echo(
            message=(
                f"Plane {plane_index} registration: skipped. The plane is already registered and re-registration is "
                f"disabled."
            ),
            level=LogLevel.INFO,
        )
        return

    # Holds the native thread pools to the stage's own worker budget for the rest of the registration. The
    # linear-algebra routines the phase correlation dispatches otherwise open one thread per core through the
    # underlying BLAS and OpenMP libraries, which oversubscribes the host when several planes register at once.
    # The Numba mask above bounds only the kernels Numba compiles, so it does not reach these. The scipy FFT pool
    # is reached by neither, so every transform names the stage budget through its own 'workers' argument.
    with threadpool_limits(limits=workers):
        if registration_data.is_registered(output_path=io_data.output_path):
            console.echo(
                message=(
                    f"Plane {plane_index} registration: forced. Clearing existing data and re-running the registration."
                ),
                level=LogLevel.INFO,
            )
            registration_data.clear()

        has_second_channel = io_data.registered_binary_path_channel_2 is not None

        if has_second_channel:
            alignment_channel = "channel 1" if config.registration.align_by_first_channel else "channel 2"
            console.echo(
                message=f"Registering plane {plane_index} (two channels, aligning by {alignment_channel})...",
                level=LogLevel.INFO,
            )
        else:
            console.echo(message=f"Registering plane {plane_index} (single channel)...", level=LogLevel.INFO)

        timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
        timer.reset()

        gpu_backend = _register_alignment_channel(context=context, workers=workers, device=device)
        try:
            # Applies the same registration offsets to the secondary channel if present. The secondary binary has not
            # been bidirectionally corrected at this point, so the correction travels with the offsets.
            if has_second_channel:
                _register_secondary_channel(
                    context=context, bidirectional_phase_corrected=False, gpu_backend=gpu_backend
                )
        finally:
            # The device memory this pass held returns before the next plane is dispatched, rather than staying in the
            # CuPy pool for the lifetime of the worker process that ran it.
            if gpu_backend is not None:
                gpu_backend.release()

        context.runtime.timing.registration_time = timer.elapsed
        console.echo(
            message=f"Plane {plane_index} registration step 1: complete. Time taken: {timer.elapsed} seconds.",
            level=LogLevel.SUCCESS,
        )

        # Performs two-step registration refinement if enabled. The second step re-registers the already-registered
        # frames using a new reference computed from the first-step results, which improves alignment for noisy data.
        if config.registration.two_step_registration:
            console.echo(
                message=f"Running plane {plane_index} two-step registration refinement...", level=LogLevel.INFO
            )
            timer.reset()

            # Re-runs registration (computes new reference from already-registered frames).
            gpu_backend = _register_alignment_channel(context=context, workers=workers, device=device)
            try:
                # Re-applies offsets to the secondary channel if present. The step-1 pass above already applied the
                # bidirectional correction to that binary, so this pass carries the rigid offsets alone.
                if has_second_channel:
                    _register_secondary_channel(
                        context=context, bidirectional_phase_corrected=True, gpu_backend=gpu_backend
                    )
            finally:
                if gpu_backend is not None:
                    gpu_backend.release()

            context.runtime.timing.two_step_registration_time = int(timer.elapsed)
            console.echo(
                message=f"Plane {plane_index} registration step 2: complete. Time taken: {timer.elapsed} seconds.",
                level=LogLevel.SUCCESS,
            )

        frame_count = io_data.frame_count
        bad_frames = np.zeros(frame_count, dtype=np.bool_)
        data_path = config.file_io.data_path
        if data_path is not None:
            bad_frames_file = data_path / RegistrationArrays.BAD_FRAMES
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

        registration_data = context.runtime.registration
        height, width = io_data.frame_height, io_data.frame_width

        # Extracts offsets for crop computation. Fallback assignments are for the type checker only. These are always
        # present after _register_alignment_channel. Uses np.empty to avoid initialization overhead.
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
            maximum_offset_fraction=config.registration.maximum_offset_fraction,
            frame_height=height,
            frame_width=width,
        )

        registration_data.valid_y_range = valid_y_range
        registration_data.valid_x_range = valid_x_range
        registration_data.bad_frames = computed_bad_frames

        # Persists registration results to disk before the optional metrics computation step, so that registration
        # offsets and valid ranges are not lost if the metrics computation fails.
        context.save_runtime()

        # Clears the markers that guarded the in-place rewrites. Both of the plane's binaries now carry the same
        # correction and the registration outputs that describe it are on disk, so the plane is consistent again.
        for binary_path in _resolve_plane_binary_paths(io_data=io_data):
            clear_registration_marker(binary_path=binary_path)

        principal_component_count = config.registration.registration_metric_principal_components
        # The >=1500-frame metrics path is impractical on synthetic data, and compute_pc_metrics is covered directly.
        if principal_component_count > 0 and frame_count >= _MINIMUM_REGISTRATION_METRIC_FRAMES:
            timer.reset()
            compute_pc_metrics(context=context, workers=workers)
            context.runtime.timing.registration_metrics_time = int(timer.elapsed)
            console.echo(
                message=(
                    f"Plane {plane_index} registration metrics processing: complete. "
                    f"Time taken: {timer.elapsed} seconds."
                ),
                level=LogLevel.SUCCESS,
            )
        elif principal_component_count > 0:
            console.echo(
                message=(
                    f"Skipping plane {plane_index} registration quality metrics computation. "
                    f"Recording has {frame_count} "
                    f"frames, but at least {_MINIMUM_REGISTRATION_METRIC_FRAMES} are required."
                ),
                level=LogLevel.INFO,
            )

        # Persists the final registration state (including metrics if computed) to disk.
        context.save_runtime()

        # Releases registration arrays to free memory. Arrays remain on disk for subsequent pipeline phases.
        context.runtime.registration.release_arrays()


def _compute_crop(
    x_offsets: NDArray[np.int32],
    y_offsets: NDArray[np.int32],
    correlations: NDArray[np.float32],
    bad_frame_threshold: float,
    bad_frames: NDArray[np.bool_],
    maximum_offset_fraction: float,
    frame_height: int,
    frame_width: int,
) -> tuple[NDArray[np.bool_], tuple[int, int], tuple[int, int]]:
    """Computes the valid pixel region after registration by analyzing frame offsets.

    After registration, frames that shifted significantly will have undefined pixels at their edges. This function
    determines which pixel region is valid across all frames by finding the maximum offset magnitude. It also
    identifies bad frames that have abnormally large offsets or poor correlation quality, excluding them from the
    valid region calculation to prevent a few outlier frames from unnecessarily shrinking the usable field of view.

    Args:
        x_offsets: The x-direction rigid pixel offsets with shape (num_frames,).
        y_offsets: The y-direction rigid pixel offsets with shape (num_frames,).
        correlations: The phase correlation peak values with shape (num_frames,) indicating registration quality.
        bad_frame_threshold: The threshold multiplier for identifying outlier frames based on offset deviation
            relative to correlation quality.
        bad_frames: A boolean array with shape (num_frames,) of frames already marked as bad from external sources.
        maximum_offset_fraction: The maximum allowed offset as a fraction of the frame size. Frames whose x or y
            offset exceeds 95% of this fraction of the matching frame dimension are flagged as bad.
        frame_height: The height of each frame in pixels.
        frame_width: The width of each frame in pixels.

    Returns:
        A tuple containing the updated bad_frames boolean array with outliers marked, the valid y-range as
        (y_min, y_max) defining usable rows, and the valid x-range as (x_min, x_max) defining usable columns.
    """
    # Computes median filter window: largest odd number below the array length, capped at maximum.
    # This extracts a smooth baseline trend from the offset time series.
    filter_window = min((len(y_offsets) // 2) * 2 - 1, _MAXIMUM_MEDIAN_FILTER_WINDOW)

    # Subtracts baseline to isolate high-frequency deviations (sudden jumps indicate bad frames). Casts the medfilt
    # output to float32 to prevent float64 promotion of the entire downstream chain.
    delta_x = x_offsets - medfilt(volume=x_offsets, kernel_size=filter_window).astype(np.float32)
    delta_y = y_offsets - medfilt(volume=y_offsets, kernel_size=filter_window).astype(np.float32)

    # Computes offset magnitude normalized by mean offset. If mean is 0 (no motion), delta_xy stays as zeros.
    delta_xy = np.hypot(delta_x, delta_y)
    delta_xy_mean = delta_xy.mean()
    if delta_xy_mean > 0:
        delta_xy = delta_xy / delta_xy_mean

    # Normalizes phase correlation relative to local median to detect quality drops.
    correlation_normalized = correlations / medfilt(volume=correlations, kernel_size=filter_window).astype(np.float32)

    # Combines deviation and correlation metrics: bad frames have large offsets AND/OR poor correlation.
    outlier_metric = delta_xy / np.maximum(0, correlation_normalized)
    x_threshold = maximum_offset_fraction * frame_width * _OFFSET_MARGIN_FRACTION
    y_threshold = maximum_offset_fraction * frame_height * _OFFSET_MARGIN_FRACTION
    bad_frames = (
        bad_frames
        | (outlier_metric > bad_frame_threshold * _OUTLIER_METRIC_SCALE)
        | (np.abs(x_offsets) > x_threshold)
        | (np.abs(y_offsets) > y_threshold)
    )

    # Computes valid region from good frames only (excludes outliers from shrinking the FOV).
    # If at least 50% are bad, falls back to using all frames and warns about registration failure.
    if bad_frames.mean() < _BAD_FRAME_FRACTION_THRESHOLD:
        y_min = np.ceil(np.abs(y_offsets[~bad_frames]).max())
        x_min = np.ceil(np.abs(x_offsets[~bad_frames]).max())
    else:
        console.echo(
            message=(
                "WARNING: at least 50% of frames have large movements, suggesting that registration has failed "
                "to correct motion artifacts."
            ),
            level=LogLevel.WARNING,
        )
        y_min = np.ceil(np.abs(y_offsets).max())
        x_min = np.ceil(np.abs(x_offsets).max())

    # Valid region is the interior rectangle after accounting for maximum offsets in each direction.
    y_max = frame_height - y_min
    x_max = frame_width - x_min
    valid_y_range = (int(y_min), int(y_max))
    valid_x_range = (int(x_min), int(x_max))

    return bad_frames, valid_y_range, valid_x_range


def _pick_initial_reference(frames: NDArray[np.float32], top_correlations: int = 20) -> NDArray[np.float32]:
    """Computes the initial reference image from a set of frames.

    Identifies the seed frame as the frame with the highest mean correlation against its most-correlated partners,
    then averages the top_correlations most-correlated frames to produce the initial reference.

    Args:
        frames: The processed recording's frames with shape (num_frames, height, width). The mean subtraction used by
            the correlation computation is applied to an internal working copy, so the input array retains its
            original intensity scale.
        top_correlations: The number of top frame correlations to average.

    Returns:
        The initial reference image with shape (height, width), on a mean-subtracted intensity scale. The first
        refinement iteration in _compute_reference recomputes the reference as a mean of the frames it aligns, so the
        final reference carries the intensity scale of those frames.
    """
    frame_count, height, width = frames.shape

    # Flattens frames and subtracts the per-frame mean for correlation computation. The subtraction allocates a
    # working copy, so the caller's frames stay on their original intensity scale. _compute_reference refines the
    # reference from those same frames, and register_plane derives the percentile clip bounds used during frame
    # registration from the resulting reference.
    frames_flat = frames.reshape(frame_count, -1)
    frames_flat = frames_flat - frames_flat.mean(axis=1, keepdims=True)

    frame_norms = np.linalg.norm(frames_flat, axis=1, keepdims=True)
    frames_normalized = frames_flat / frame_norms
    correlation_matrix = frames_normalized @ frames_normalized.T

    # Finds the frame with the highest mean correlation to other frames (excluding self-correlation).
    top_correlations_per_frame = np.partition(correlation_matrix, kth=-(top_correlations + 1), axis=1)[
        :, -top_correlations:-1
    ]
    mean_top_correlations = np.mean(top_correlations_per_frame, axis=1)
    seed_index = np.argmax(a=mean_top_correlations)

    # Averages the seed frame with its top correlated frames. The mean-subtracted working copy is used here, because
    # the first refinement iteration in _compute_reference recomputes the reference as a mean of the frames it aligns,
    # which carry their own intensity scale.
    top_indices = np.argpartition(correlation_matrix[seed_index, :], kth=-top_correlations)[-top_correlations:]
    reference_image = np.mean(frames_flat[top_indices, :], axis=0)

    return np.reshape(reference_image, shape=(int(height), int(width)))


def _compute_reference(
    frames: NDArray[np.float32],
    pre_smoothing_sigma: float,
    spatial_highpass_window: int,
    edge_taper_pixels: float,
    spatial_smoothing_sigma: float,
    maximum_offset_fraction: float,
    temporal_smoothing_sigma: float,
    workers: int,
    *,
    one_photon_enabled: bool,
) -> NDArray[np.float32]:
    """Computes the reference image through iterative alignment.

    Selects an initial reference by finding the frame most correlated with other frames, then refines it through
    8 iterations of rigid registration. In each iteration, all frames are aligned to the current reference using
    phase correlation, then the reference is updated to be the mean of the top-correlated frames. This progressive
    refinement produces a sharp, low-noise reference that represents the stable structure across frames.

    Args:
        frames: The frames to use for reference computation with shape (num_frames, height, width). Modified in-place
            by the iterative refinement, which overwrites each frame with its aligned version, unless one-photon
            preprocessing replaces the working array with a filtered copy.
        pre_smoothing_sigma: The sliding-window (box) smoothing size, in pixels, applied before high-pass filtering.
            Cast to an integer and passed to apply_spatial_smoothing, which requires a positive even window.
        spatial_highpass_window: The window size for the spatial high-pass filter that removes low-frequency background.
        edge_taper_pixels: Controls the steepness of the edge taper falloff. Larger values produce a more gradual
            taper that suppresses border artifacts during phase correlation.
        spatial_smoothing_sigma: The standard deviation of Gaussian smoothing applied to phase correlation maps.
        maximum_offset_fraction: The maximum allowed offset as a fraction of the minimum spatial dimension.
            The search window is limited to min(height, width) * maximum_offset_fraction pixels.
        temporal_smoothing_sigma: The standard deviation for temporal Gaussian smoothing of correlation maps.
            If 0, no smoothing is applied.
        workers: The number of parallel workers for FFT computation. Use -1 for all available cores.
        one_photon_enabled: Determines whether to apply one-photon preprocessing, which includes spatial smoothing
            followed by high-pass filtering.

    Returns:
        The computed reference image with shape (height, width).
    """
    # Selects the initial reference by averaging together the most stable frames.
    reference_image = _pick_initial_reference(frames=frames)

    if one_photon_enabled:
        if pre_smoothing_sigma > 0:
            reference_image = apply_spatial_smoothing(data=reference_image, window=int(pre_smoothing_sigma))
            frames = apply_spatial_smoothing(data=frames, window=int(pre_smoothing_sigma))
        reference_image = apply_spatial_high_pass(data=reference_image, window=spatial_highpass_window)
        frames = apply_spatial_high_pass(data=frames, window=spatial_highpass_window)

    taper_slope = edge_taper_pixels if one_photon_enabled else 3 * spatial_smoothing_sigma

    # Iteratively refines the reference image. The iteration count of 8 is empirically tuned from the original suite2p
    # implementation. Each iteration includes progressively more frames, converging to ~50% of frames by the final
    # iteration.
    iteration_count = 8
    for iteration in range(iteration_count):
        taper_mask, mean_offset = compute_edge_taper(
            reference_image=reference_image,
            taper_slope=taper_slope,
        )

        y_offsets, x_offsets, correlations = compute_rigid_offsets(
            frames=apply_edge_taper(frames=frames, taper_mask=taper_mask, mean_offset=mean_offset),
            reference_kernel=compute_phase_correlation_kernel(
                reference_image=reference_image,
                smoothing_sigma=spatial_smoothing_sigma,
            ),
            maximum_offset_fraction=maximum_offset_fraction,
            temporal_smoothing_sigma=temporal_smoothing_sigma,
            workers=workers,
        )

        for frame, y_offset, x_offset in zip(frames, y_offsets, x_offsets, strict=False):
            frame[:] = translate_frame(frame=frame, y_offset=y_offset, x_offset=x_offset)

        # Selects top-correlated frames for next reference, excluding rank 0 (the frame most correlated with the
        # current reference) to prevent self-reinforcing bias in the iterative refinement.
        # Number of frames increases each iteration: ~6% at iter 0, ~31% at iter 4, ~50% at iter 7.
        selected_frame_count = max(2, int(frames.shape[0] * (1.0 + iteration) / (2 * iteration_count)))
        sorted_indices = np.argsort(-correlations)[1:selected_frame_count]

        # Updates reference as the mean of the best-aligned frames. Input frames are float32, mean preserves dtype.
        reference_image = frames[sorted_indices].mean(axis=0)

        # Centers the reference by reversing the mean offset of selected frames.
        reference_image = translate_frame(
            frame=reference_image,
            y_offset=int(np.round(-y_offsets[sorted_indices].mean())),
            x_offset=int(np.round(-x_offsets[sorted_indices].mean())),
        )

    return reference_image


def _register_frames_batch(
    reference_data: ReferenceData,
    frames: NDArray[np.float32],
    normalization_minimum: float,
    normalization_maximum: float,
    bidirectional_phase_offset: int,
    pre_smoothing_sigma: float,
    spatial_highpass_window: int,
    temporal_smoothing_sigma: float,
    maximum_offset_fraction: float,
    signal_to_noise_threshold: float,
    maximum_block_offset: float,
    workers: int,
    *,
    one_photon_enabled: bool,
    nonrigid_enabled: bool,
) -> BatchRegistrationResult:
    """Registers the input batch of frames to the reference image using rigid and optionally nonrigid phase correlation.

    Args:
        reference_data: Precomputed reference data containing taper masks, mean offsets, and FFT kernels for both
            rigid and nonrigid registration.
        frames: The batch of frames with shape (batch_size, height, width) sampled from the processed recording.
        normalization_minimum: The minimum intensity value for clipping frames before correlation.
        normalization_maximum: The maximum intensity value for clipping frames before correlation.
        bidirectional_phase_offset: The pixel offset to correct bidirectional scanning artifacts.
        pre_smoothing_sigma: The sliding-window (box) smoothing size, in pixels, applied before high-pass filtering.
            Cast to an integer and passed to apply_spatial_smoothing, which requires a positive even window.
        spatial_highpass_window: The window size for the spatial high-pass filter that removes low-frequency background.
        temporal_smoothing_sigma: The standard deviation for temporal Gaussian smoothing of correlation maps.
            If 0, no smoothing is applied.
        maximum_offset_fraction: The maximum allowed offset as a fraction of the minimum spatial dimension.
            The search window is limited to min(height, width) * maximum_offset_fraction pixels.
        signal_to_noise_threshold: The SNR threshold below which additional smoothing is applied to correlation
            peaks. Higher values apply more smoothing. Typical values range from 1.0 to 1.5.
        maximum_block_offset: The maximum allowed offset for nonrigid blocks in pixels.
        workers: The number of parallel workers for FFT computation. Use -1 for all available cores.
        one_photon_enabled: Determines whether to apply one-photon preprocessing, which includes spatial smoothing
            followed by high-pass filtering.
        nonrigid_enabled: Determines whether to apply nonrigid (piecewise) registration after rigid alignment.

    Returns:
        The registered frames together with the per-frame rigid offsets and phase correlation peaks. The nonrigid
        offsets and correlations are None when nonrigid registration is disabled.
    """
    if bidirectional_phase_offset != 0:
        apply_bidirectional_phase_correction(frames=frames, bidirectional_phase_offset=bidirectional_phase_offset)

    # Holds a working copy for correlation computation only when one-photon preprocessing replaces its contents. On
    # the two-photon path the smoothed frames stay equal to the registered frames through every step below, so the
    # two names share one buffer and the rigid shift is applied to it once.
    frames_smooth = frames.copy() if one_photon_enabled else frames

    if one_photon_enabled:
        if pre_smoothing_sigma > 0:
            frames_smooth = apply_spatial_smoothing(data=frames_smooth, window=int(pre_smoothing_sigma))
        frames_smooth = apply_spatial_high_pass(data=frames_smooth, window=spatial_highpass_window)

    # Clips intensity range to reduce influence of outlier pixels on correlation.
    frames_for_correlation = (
        np.clip(a=frames_smooth, a_min=normalization_minimum, a_max=normalization_maximum)
        if normalization_minimum > -np.inf
        else frames_smooth
    )

    # Phase 1: rigid registration, which computes whole-frame translation offsets.
    y_offsets, x_offsets, correlations = compute_rigid_offsets(
        frames=apply_edge_taper(
            frames=frames_for_correlation,
            taper_mask=reference_data.taper_mask,
            mean_offset=reference_data.mean_offset,
        ),
        reference_kernel=reference_data.reference_kernel,
        maximum_offset_fraction=maximum_offset_fraction,
        temporal_smoothing_sigma=temporal_smoothing_sigma,
        workers=workers,
    )

    # Applies rigid offsets to original (unsmoothed) frames.
    for frame, y_offset, x_offset in zip(frames, y_offsets, x_offsets, strict=False):
        frame[:] = translate_frame(frame=frame, y_offset=y_offset, x_offset=x_offset)

    # Phase 2: nonrigid registration, which computes per-block subpixel offsets to correct local deformations.
    if nonrigid_enabled:
        # Extracts nonrigid reference data. Fallback assignments are for the type checker only. These are guaranteed
        # to be present when nonrigid_enabled is True.
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

        # Applies rigid offsets to the smoothed working copy so nonrigid phase correlation operates on pre-aligned
        # data. Without this, the per-block offsets would capture both global translation and local deformation,
        # double-counting the rigid component that was already corrected on the original frames. The two-photon path
        # shares one buffer between the two names, where the shift above already covers it.
        if one_photon_enabled:
            for frame_smooth, y_offset, x_offset in zip(frames_smooth, y_offsets, x_offsets, strict=False):
                frame_smooth[:] = translate_frame(frame=frame_smooth, y_offset=y_offset, x_offset=x_offset)

        # Re-clips intensity range after rigid offset for nonrigid correlation.
        frames_for_correlation = (
            np.clip(a=frames_smooth, a_min=normalization_minimum, a_max=normalization_maximum)
            if normalization_minimum > -np.inf
            else frames_smooth
        )

        y_offsets_nonrigid, x_offsets_nonrigid, correlations_nonrigid = compute_nonrigid_offsets(
            frames=frames_for_correlation,
            taper_mask=taper_mask_nonrigid,
            mean_offset=mean_offset_nonrigid,
            reference_kernel=reference_kernel_nonrigid,
            snr_threshold=signal_to_noise_threshold,
            smoothing_kernel=blocks[-1],
            x_blocks=blocks[1],
            y_blocks=blocks[0],
            maximum_offset=maximum_block_offset,
            workers=workers,
        )

        # Applies nonrigid warping to original frames using computed block offsets.
        frames = apply_nonrigid_correction(
            frames=frames,
            y_blocks=blocks[0],
            x_blocks=blocks[1],
            block_counts=blocks[2],
            y_block_offsets=y_offsets_nonrigid,
            x_block_offsets=x_offsets_nonrigid,
        )
    else:
        y_offsets_nonrigid, x_offsets_nonrigid, correlations_nonrigid = None, None, None

    return BatchRegistrationResult(
        frames=frames,
        y_offsets=y_offsets,
        x_offsets=x_offsets,
        correlations=correlations,
        y_offsets_nonrigid=y_offsets_nonrigid,
        x_offsets_nonrigid=x_offsets_nonrigid,
        correlations_nonrigid=correlations_nonrigid,
    )


def _apply_precomputed_offsets_batch(
    frames: NDArray[np.float32],
    y_offsets: NDArray[np.int32],
    x_offsets: NDArray[np.int32],
    y_offsets_nonrigid: NDArray[np.float32] | None,
    x_offsets_nonrigid: NDArray[np.float32] | None,
    blocks: RegistrationBlocks | None,
    bidirectional_phase_offset: int,
    *,
    bidirectional_phase_corrected: bool,
    nonrigid_enabled: bool,
) -> NDArray[np.float32]:
    """Applies precomputed registration offsets to a batch of frames.

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
    if bidirectional_phase_offset != 0 and not bidirectional_phase_corrected:
        apply_bidirectional_phase_correction(
            frames=frames,
            bidirectional_phase_offset=bidirectional_phase_offset,
        )

    for frame, y_offset, x_offset in zip(frames, y_offsets, x_offsets, strict=False):
        frame[:] = translate_frame(frame=frame, y_offset=y_offset, x_offset=x_offset)

    # Applies nonrigid (per-block) warping if enabled. Fallback assignments are for the type checker only. These are
    # guaranteed to be present when nonrigid_enabled is True.
    if nonrigid_enabled:
        _blocks = blocks if blocks is not None else ([], [], (0, 0), (0, 0), np.empty(0, dtype=np.float32))
        _y_offsets_nonrigid = (
            y_offsets_nonrigid if y_offsets_nonrigid is not None else np.empty((0, 0), dtype=np.float32)
        )
        _x_offsets_nonrigid = (
            x_offsets_nonrigid if x_offsets_nonrigid is not None else np.empty((0, 0), dtype=np.float32)
        )
        frames = apply_nonrigid_correction(
            frames=frames,
            y_blocks=_blocks[0],
            x_blocks=_blocks[1],
            block_counts=_blocks[2],
            y_block_offsets=_y_offsets_nonrigid,
            x_block_offsets=_x_offsets_nonrigid,
        )

    return frames


def _resolve_plane_binary_paths(io_data: IOData) -> tuple[Path, ...]:
    """Resolves the paths of every channel binary the registration stage rewrites for one plane.

    Args:
        io_data: The plane's IOData, which holds the paths of the binaries the registration stage rewrites.

    Returns:
        A tuple of the plane's channel binary paths, which holds a single path when the plane has one channel.
    """
    return tuple(
        binary_path
        for binary_path in (io_data.registered_binary_path, io_data.registered_binary_path_channel_2)
        if binary_path is not None
    )


def _validate_binaries_are_not_mid_write(io_data: IOData, plane_index: int) -> None:
    """Verifies that no interrupted write left the plane's binaries in an indeterminate state.

    Notes:
        The binarization stage fills its output binaries and the registration stage rewrites them in place, so a run of
        either that dies partway leaves finished frames up to some unknown point and unfinished frames after it. On a
        two-channel plane registration can also leave one binary fully corrected while the other is untouched.
        Registering such a plane computes its offsets and its valid crop region from a movie whose frames disagree
        about what they hold, and both the resulting traces and the reported registration quality look ordinary.
        Failing here converts that silent corruption into an actionable error. Either phase's marker fails the plane,
        because both describe a binary whose contents are indeterminate.

    Args:
        io_data: The plane's IOData, which holds the paths of the binaries the registration stage rewrites.
        plane_index: The index of the plane, used to identify the plane in the error message.

    Raises:
        RuntimeError: If a marker shows that a previous write of one of the plane's binaries was interrupted.
    """
    for binary_path in _resolve_plane_binary_paths(io_data=io_data):
        marker_path = resolve_active_binary_marker(binary_path=binary_path)
        if marker_path is not None:
            message = (
                f"Unable to register plane {plane_index}. A previous write of the binary file "
                f"'{binary_path}' was interrupted, so the file holds finished frames up to an unknown point and "
                f"unfinished frames after it. Enable 'file_io.repeat_binarization' and re-run the binarization stage "
                f"to rebuild the binary from its source TIFF files, which also clears the marker at '{marker_path}'."
            )
            console.error(message=message, error=RuntimeError)


def _register_alignment_channel(
    context: RuntimeContext, *, workers: int, device: int | None = None
) -> GpuRegistrationBackend | None:
    """Computes registration offsets from the alignment channel and applies them to that channel's frames.

    Computes the reference image, calculates rigid and optionally nonrigid registration offsets, and applies them to all
    frames. Results are stored in context.runtime.registration and the mean image is stored in the appropriate detection
    field. Every channel binary of the plane is marked with a '<binary>.registering' file before the in-place rewrite
    begins, and register_plane clears those markers once the registration outputs are on disk.

    Args:
        context: The RuntimeContext containing configuration, acquisition parameters, and runtime data.
        workers: The number of parallel workers to use for the phase correlation FFT computations.
        device: The zero-based index of the CUDA device to register this plane on. Use None to register the plane on
            the host CPU.

    Returns:
        The device backend this pass registered through, which the secondary channel reuses, or None when the pass
        ran on the host CPU.
    """
    config = context.configuration
    align_by_first_channel = config.registration.align_by_first_channel
    one_photon_enabled = config.one_photon_registration.enabled
    pre_smoothing_sigma = config.one_photon_registration.pre_smoothing_sigma
    spatial_highpass_window = config.one_photon_registration.spatial_highpass_window
    edge_taper_pixels = config.one_photon_registration.edge_taper_pixels
    spatial_smoothing_sigma = config.registration.spatial_smoothing_sigma
    temporal_smoothing_sigma = config.registration.temporal_smoothing_sigma
    maximum_offset_fraction = config.registration.maximum_offset_fraction
    normalize_frames = config.registration.normalize_frames
    reference_frame_count = config.registration.reference_frame_count
    gpu_enabled = device is not None

    # Registration on a CUDA device bounds its batch by the device memory budget rather than the host RAM budget, so
    # it reads its own size when one is configured. A zero keeps the device on the shared size.
    batch_size = config.registration.batch_size
    if gpu_enabled and config.registration.gpu_batch_size > 0:
        batch_size = config.registration.gpu_batch_size
    nonrigid_enabled = config.nonrigid_registration.enabled
    block_size = config.nonrigid_registration.block_size
    signal_to_noise_threshold = config.nonrigid_registration.signal_to_noise_threshold
    maximum_block_offset = config.nonrigid_registration.maximum_block_offset
    enable_bidirectional_phase_computation = config.registration.compute_bidirectional_phase_offset
    initial_bidirectional_phase_offset = config.registration.bidirectional_phase_offset_override

    io_data = context.runtime.io
    plane_index = io_data.plane_index if io_data.plane_index is not None else 0
    height, width, frame_count = io_data.frame_height, io_data.frame_width, io_data.frame_count
    bidirectional_phase_corrected = context.runtime.registration.bidirectional_phase_corrected
    recorded_bidirectional_phase_offset = context.runtime.registration.bidirectional_phase_offset

    if align_by_first_channel:
        binary_path = io_data.registered_binary_path
        channel_label = "channel 1"
    else:
        binary_path = io_data.registered_binary_path_channel_2
        channel_label = "channel 2"

    if binary_path is None:
        message = (
            f"Unable to register {channel_label} frames for plane {plane_index}. The plane's RuntimeContext "
            f"instance does not contain the path to the plane's {channel_label} binary file."
        )
        console.error(message=message, error=ValueError)

    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()
    with BinaryFile(height=height, width=width, file_path=binary_path, frame_number=frame_count) as frames_file:
        # Tracks the bidirectional phase offset (may be updated from data). The configuration override takes
        # precedence, and the offset an earlier pass recorded fills in behind it, so the two-step refinement pass
        # republishes the offset the plane's binaries actually carry instead of overwriting it with a zero.
        bidirectional_phase_offset = initial_bidirectional_phase_offset or recorded_bidirectional_phase_offset

        sample_indices = np.linspace(
            start=0, stop=frame_count, num=1 + np.minimum(reference_frame_count, frame_count), dtype=int
        )[:-1]
        frames = frames_file[sample_indices].astype(np.float32)

        if (
            enable_bidirectional_phase_computation
            and bidirectional_phase_offset == 0
            and not bidirectional_phase_corrected
        ):
            bidirectional_phase_offset = compute_bidirectional_phase_offset(frames=frames, workers=workers)
            console.echo(
                message=(
                    f"Plane {plane_index} estimated bidirectional phase offset: {bidirectional_phase_offset} pixels."
                ),
                level=LogLevel.INFO,
            )

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
            maximum_offset_fraction=maximum_offset_fraction,
            temporal_smoothing_sigma=temporal_smoothing_sigma,
            workers=workers,
        )
        console.echo(
            message=f"Plane {plane_index} reference frame: computed. Time taken: {timer.elapsed} seconds.",
            level=LogLevel.SUCCESS,
        )

        reference_original = reference_image.copy()
        if normalize_frames:
            normalization_minimum = float(np.percentile(a=reference_image, q=1))
            normalization_maximum = float(np.percentile(a=reference_image, q=99))
            reference_image = np.clip(a=reference_image, a_min=normalization_minimum, a_max=normalization_maximum)
        else:
            normalization_minimum, normalization_maximum = -np.inf, np.inf

        if bidirectional_phase_offset != 0 and not bidirectional_phase_corrected:
            bidirectional_phase_for_registration = bidirectional_phase_offset
        else:
            bidirectional_phase_for_registration = 0

        taper_slope = edge_taper_pixels if one_photon_enabled else 3 * spatial_smoothing_sigma

        taper_mask, mean_offset = compute_edge_taper(
            reference_image=reference_image,
            taper_slope=taper_slope,
        )
        reference_kernel = compute_phase_correlation_kernel(
            reference_image=reference_image,
            smoothing_sigma=spatial_smoothing_sigma,
        )

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

        reference_data = ReferenceData(
            taper_mask=taper_mask,
            mean_offset=mean_offset,
            reference_kernel=reference_kernel,
            taper_mask_nonrigid=taper_mask_nonrigid,
            mean_offset_nonrigid=mean_offset_nonrigid,
            reference_kernel_nonrigid=reference_kernel_nonrigid,
            blocks=blocks,
        )

        # The device backend uploads the reference data once and holds it on the device for every batch of this pass,
        # so it is built here rather than per batch. The device the batch engine assigned reaches it through the plane
        # entry point.
        gpu_backend = (
            GpuRegistrationBackend(reference_data=reference_data, device=device) if device is not None else None
        )

        mean_image = np.zeros((height, width), dtype=np.float32)
        rigid_offsets_batches: list[tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]] = []
        nonrigid_offsets_batches: list[tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]] = []

        timer.reset()
        console.echo(
            message=f"Computing and applying plane {plane_index} registration offsets for {channel_label}...",
            level=LogLevel.INFO,
        )

        # Marks every binary of the plane for the duration of the rewrites that follow. Until this loop completes, the
        # aligned file holds a mixture of corrected and raw frames that nothing else on disk would reveal. The
        # secondary channel is marked here as well, because it receives the offsets computed here rather than its own,
        # so it stays raw until _register_secondary_channel rewrites it. register_plane clears both markers once the
        # registration outputs that describe the rewrites are on disk.
        for plane_binary_path in _resolve_plane_binary_paths(io_data=io_data):
            create_registration_marker(binary_path=plane_binary_path)

        batch_starts = np.arange(0, frame_count, batch_size)

        def read_batches() -> Iterator[NDArray[np.int16]]:
            """Reads every batch of the pass off the plane binary in the width the binary stores."""
            for start_index in batch_starts:
                start = int(start_index)
                yield frames_file[start : min(start + batch_size, frame_count)]

        # The device path and the host path differ in where each batch is registered rather than in what the pass does
        # with the result, so both present the same stream of batch results to the loop below. The device consumes the
        # stored width directly and widens it there.
        def register_on_host() -> Iterator[BatchRegistrationResult]:
            """Registers every batch of the pass through the host kernels."""
            for batch in read_batches():
                yield _register_frames_batch(
                    reference_data=reference_data,
                    frames=batch.astype(np.float32),
                    normalization_minimum=normalization_minimum,
                    normalization_maximum=normalization_maximum,
                    bidirectional_phase_offset=bidirectional_phase_for_registration,
                    pre_smoothing_sigma=pre_smoothing_sigma,
                    spatial_highpass_window=spatial_highpass_window,
                    temporal_smoothing_sigma=temporal_smoothing_sigma,
                    maximum_offset_fraction=maximum_offset_fraction,
                    signal_to_noise_threshold=signal_to_noise_threshold,
                    maximum_block_offset=maximum_block_offset,
                    workers=workers,
                    one_photon_enabled=one_photon_enabled,
                    nonrigid_enabled=nonrigid_enabled,
                )

        batch_results: Iterator[BatchRegistrationResult]
        if gpu_backend is not None:
            batch_results = gpu_backend.register_batches(
                read_batches(),
                normalization_minimum=normalization_minimum,
                normalization_maximum=normalization_maximum,
                bidirectional_phase_offset=bidirectional_phase_for_registration,
                pre_smoothing_sigma=pre_smoothing_sigma,
                spatial_highpass_window=spatial_highpass_window,
                temporal_smoothing_sigma=temporal_smoothing_sigma,
                maximum_offset_fraction=maximum_offset_fraction,
                signal_to_noise_threshold=signal_to_noise_threshold,
                maximum_block_offset=maximum_block_offset,
                one_photon_enabled=one_photon_enabled,
                nonrigid_enabled=nonrigid_enabled,
            )
        else:
            batch_results = register_on_host()

        for batch_start_np, batch_result in console.track(
            zip(batch_starts, batch_results, strict=True),
            description=f"Registering batches of {batch_size} frames",
            total=len(batch_starts),
            unit="batch",
        ):
            batch_start = int(batch_start_np)
            batch_end = min(batch_start + batch_size, frame_count)

            rigid_offsets_batches.append((batch_result.y_offsets, batch_result.x_offsets, batch_result.correlations))
            if nonrigid_enabled:
                # Fallback assignments are for the type checker only. They are guaranteed present when
                # nonrigid_enabled is True.
                y_offsets_nonrigid = (
                    batch_result.y_offsets_nonrigid
                    if batch_result.y_offsets_nonrigid is not None
                    else np.empty((0, 0), dtype=np.float32)
                )
                x_offsets_nonrigid = (
                    batch_result.x_offsets_nonrigid
                    if batch_result.x_offsets_nonrigid is not None
                    else np.empty((0, 0), dtype=np.float32)
                )
                correlations_nonrigid = (
                    batch_result.correlations_nonrigid
                    if batch_result.correlations_nonrigid is not None
                    else np.empty((0, 0), dtype=np.float32)
                )
                nonrigid_offsets_batches.append((y_offsets_nonrigid, x_offsets_nonrigid, correlations_nonrigid))

            # The mean image is measured before the frames are narrowed to the storage width, so a backend that
            # narrows them itself carries the sum it took ahead of that narrowing.
            if batch_result.frame_sum is not None:
                mean_image += batch_result.frame_sum
                frames_file[batch_start:batch_end] = cast("NDArray[np.int16]", batch_result.frames)
            else:
                mean_image += batch_result.frames.sum(axis=0)

                # Converts back to int16 for BinaryFile storage and writes in-place. The clip writes through the batch
                # buffer, which the loop discards on the next iteration, so the narrowing needs one destination rather
                # than a clipped copy followed by a converted copy.
                np.clip(
                    a=batch_result.frames,
                    a_min=np.iinfo(np.int16).min,
                    a_max=np.iinfo(np.int16).max,
                    out=batch_result.frames,
                )
                frames_file[batch_start:batch_end] = batch_result.frames.astype(dtype=np.int16)

        # Flushes the rewritten frames. The marker stays in place, because a two-channel plane is consistent only once
        # the secondary channel has received the same offsets.
        frames_file.file.flush()

        mean_image /= frame_count

        console.echo(
            message=(
                f"Plane {plane_index} {channel_label} registration offsets: computed and applied. "
                f"Time taken: {timer.elapsed} seconds."
            ),
            level=LogLevel.SUCCESS,
        )

    rigid_y_offsets, rigid_x_offsets, rigid_correlations = combine_rigid_offsets(offset_list=rigid_offsets_batches)
    if nonrigid_enabled:
        nonrigid_y_offsets, nonrigid_x_offsets, nonrigid_correlations = combine_nonrigid_offsets(
            offset_list=nonrigid_offsets_batches
        )
    else:
        nonrigid_y_offsets, nonrigid_x_offsets, nonrigid_correlations = None, None, None

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

    if align_by_first_channel:
        context.runtime.detection.mean_image = mean_image
    else:
        context.runtime.detection.mean_image_channel_2 = mean_image

    return gpu_backend


def _register_secondary_channel(
    context: RuntimeContext, *, bidirectional_phase_corrected: bool, gpu_backend: GpuRegistrationBackend | None
) -> None:
    """Applies precomputed registration offsets to the secondary (non-alignment) channel's frames.

    Registration offsets are read from context.runtime.registration (computed by _register_alignment_channel) and
    applied to all frames, which are rewritten in place in the channel's binary under the '<binary>.registering' marker
    _register_alignment_channel created. The resulting mean image is stored in the matching detection field.

    Notes:
        The offsets this applies were measured on the alignment channel and are indexed frame for frame against this
        one. Binarization writes both channels of a plane over the same interleave cycles, so the count the plane
        records bounds the rewrite, the mean image, and the offset arrays alike.

    Args:
        context: The RuntimeContext containing configuration, acquisition parameters, and runtime data.
        bidirectional_phase_corrected: Determines whether this channel's binary already carries the bidirectional
            phase correction, which holds for the two-step refinement pass that follows a completed first pass. The
            flag stored in context.runtime.registration tracks the alignment channel rather than this one.
        gpu_backend: The device backend the alignment pass registered through, whose device-resident reference data
            this channel reuses, or None when the alignment pass ran on the host CPU.
    """
    config = context.configuration
    align_by_first_channel = config.registration.align_by_first_channel
    nonrigid_enabled = config.nonrigid_registration.enabled
    block_size = config.nonrigid_registration.block_size
    batch_size = config.registration.batch_size

    io_data = context.runtime.io
    plane_index = io_data.plane_index if io_data.plane_index is not None else 0
    height, width, frame_count = io_data.frame_height, io_data.frame_width, io_data.frame_count

    # Extracts registration data (offsets computed from alignment channel).
    registration_data = context.runtime.registration
    bidirectional_phase_offset = registration_data.bidirectional_phase_offset

    # Extracts rigid offsets and converts to int32 for translation operations. Fallback to empty arrays is for type
    # narrowing only. Offsets are always present since _register_alignment_channel populates them before this is called.
    y_offsets = (
        registration_data.rigid_y_offsets
        if registration_data.rigid_y_offsets is not None
        else np.empty(0, dtype=np.int32)
    )
    x_offsets = (
        registration_data.rigid_x_offsets
        if registration_data.rigid_x_offsets is not None
        else np.empty(0, dtype=np.int32)
    )
    y_offsets_int = y_offsets.astype(np.int32)
    x_offsets_int = x_offsets.astype(np.int32)

    y_offsets_nonrigid = registration_data.nonrigid_y_offsets if nonrigid_enabled else None
    x_offsets_nonrigid = registration_data.nonrigid_x_offsets if nonrigid_enabled else None

    # Selects channel paths based on alignment configuration (uses the opposite channel from alignment).
    if align_by_first_channel:
        binary_path = io_data.registered_binary_path_channel_2
        channel_label = "channel 2"
    else:
        binary_path = io_data.registered_binary_path
        channel_label = "channel 1"

    if binary_path is None:
        message = (
            f"Unable to register {channel_label} frames for plane {plane_index}. The plane's RuntimeContext "
            f"instance does not contain the path to the plane's {channel_label} binary file."
        )
        console.error(message=message, error=ValueError)

    blocks = None
    if nonrigid_enabled:
        blocks = compute_registration_blocks(height=height, width=width, block_size=block_size)

    timer = PrecisionTimer(precision=TimerPrecisions.SECOND)
    timer.reset()
    mean_image = np.zeros((height, width), dtype=np.float32)

    with BinaryFile(height=height, width=width, file_path=binary_path, frame_number=frame_count) as frames_file:
        console.echo(
            message=f"Applying plane {plane_index} registration offsets to {channel_label}...",
            level=LogLevel.INFO,
        )
        timer.reset()

        # Prepares nonrigid offset arrays outside the loop. Fallback to empty arrays is for type narrowing only.
        # Offsets are always present when nonrigid_enabled is True.
        if nonrigid_enabled:
            nonrigid_y_offsets_full = (
                y_offsets_nonrigid if y_offsets_nonrigid is not None else np.empty((0, 0), dtype=np.float32)
            )
            nonrigid_x_offsets_full = (
                x_offsets_nonrigid if x_offsets_nonrigid is not None else np.empty((0, 0), dtype=np.float32)
            )

        # Processes frames in batches to limit memory usage.
        for batch_start_np in console.track(
            np.arange(0, frame_count, batch_size),
            description=f"Registering batches of {batch_size} frames",
            unit="batch",
        ):
            batch_start = int(batch_start_np)
            batch_end = min(batch_start + batch_size, frame_count)

            y_offsets_batch = y_offsets_int[batch_start:batch_end]
            x_offsets_batch = x_offsets_int[batch_start:batch_end]

            if nonrigid_enabled:
                y_offsets_nonrigid_batch = nonrigid_y_offsets_full[batch_start:batch_end]
                x_offsets_nonrigid_batch = nonrigid_x_offsets_full[batch_start:batch_end]
            else:
                y_offsets_nonrigid_batch, x_offsets_nonrigid_batch = None, None

            if gpu_backend is not None:
                # The backend consumes the stored width and narrows the result itself, so it carries the sum it took
                # ahead of that narrowing.
                narrowed_frames, frame_sum = gpu_backend.apply_precomputed_offsets(
                    frames=frames_file[batch_start:batch_end],
                    y_offsets=y_offsets_batch,
                    x_offsets=x_offsets_batch,
                    y_offsets_nonrigid=y_offsets_nonrigid_batch,
                    x_offsets_nonrigid=x_offsets_nonrigid_batch,
                    bidirectional_phase_offset=bidirectional_phase_offset,
                    bidirectional_phase_corrected=bidirectional_phase_corrected,
                    nonrigid_enabled=nonrigid_enabled,
                )
                mean_image += cast("NDArray[np.float32]", frame_sum)
                frames_file[batch_start:batch_end] = cast("NDArray[np.int16]", narrowed_frames)
                continue

            frames = _apply_precomputed_offsets_batch(
                frames=frames_file[batch_start:batch_end].astype(np.float32),
                y_offsets=y_offsets_batch,
                x_offsets=x_offsets_batch,
                y_offsets_nonrigid=y_offsets_nonrigid_batch,
                x_offsets_nonrigid=x_offsets_nonrigid_batch,
                blocks=blocks,
                bidirectional_phase_offset=bidirectional_phase_offset,
                bidirectional_phase_corrected=bidirectional_phase_corrected,
                nonrigid_enabled=nonrigid_enabled,
            )

            mean_image += frames.sum(axis=0)

            # Converts back to int16 for BinaryFile storage and writes in-place. The clip writes through the batch
            # buffer, which the loop discards on the next iteration, so the narrowing needs one destination rather
            # than a clipped copy followed by a converted copy.
            np.clip(a=frames, a_min=np.iinfo(np.int16).min, a_max=np.iinfo(np.int16).max, out=frames)
            frames_file[batch_start:batch_end] = frames.astype(dtype=np.int16)

        # Flushes the rewritten frames. Both of the plane's binaries now carry the same correction, and register_plane
        # clears their markers once the registration outputs that describe it are on disk.
        frames_file.file.flush()

        mean_image /= frame_count

        console.echo(
            message=(
                f"Plane {plane_index} {channel_label} registration offsets: applied. "
                f"Time taken: {timer.elapsed} seconds."
            ),
            level=LogLevel.SUCCESS,
        )

    if align_by_first_channel:
        context.runtime.detection.mean_image_channel_2 = mean_image
    else:
        context.runtime.detection.mean_image = mean_image
