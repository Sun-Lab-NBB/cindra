"""Provides the OASIS spike deconvolution algorithm and baseline correction for fluorescence traces."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba import njit, prange  # type: ignore[import-untyped]
import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter1d, minimum_filter1d
from ataraxis_base_utilities import console

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_delta_fluorescence(
    cell_fluorescence: NDArray[np.float32],
    neuropil_fluorescence: NDArray[np.float32],
    neuropil_coefficient: float,
    baseline_method: str,
    baseline_window: float,
    baseline_sigma: float,
    baseline_percentile: float,
    sampling_rate: float,
) -> NDArray[np.float32]:
    """Computes the baseline-corrected delta fluorescence (ΔF) for each ROI from raw fluorescence traces.

    Notes:
        The correction is applied in two stages. First, the scaled neuropil fluorescence is subtracted from each ROI
        trace to remove contamination from surrounding tissue. Second, a baseline representing the resting fluorescence
        level is estimated and subtracted, isolating activity-dependent transients suitable for spike deconvolution.

    Args:
        cell_fluorescence: The ROI fluorescence traces with shape (roi_count, frame_count).
        neuropil_fluorescence: The surrounding neuropil fluorescence traces with shape (roi_count, frame_count).
        neuropil_coefficient: The scaling factor applied to neuropil fluorescence before subtracting it from the ROI
            fluorescence. The corrected signal is computed as F_corrected = F_roi - coefficient * F_neuropil.
        baseline_method: The method for computing baseline fluorescence to subtract before deconvolution. Must be
            'maximin', 'constant', or 'constant_percentile'.
        baseline_window: The size of the sliding window, in seconds, for the 'maximin' baseline method. The minimum
            and maximum filters operate over this window to track slow baseline drifts while ignoring fast transients.
        baseline_sigma: The standard deviation, in frames, of the Gaussian filter applied before baseline
            computation. Used by both 'maximin' and 'constant' methods to smooth the trace before finding minima.
        baseline_percentile: The percentile of trace activity used as baseline for the 'constant_percentile' method.
            Lower values select points near the trace minimum, providing a robust estimate that ignores outliers.
        sampling_rate: The imaging sampling rate for the processed recording plane, in Hz.

    Returns:
        The neuropil-and-baseline-corrected delta fluorescence traces with shape (roi_count, frame_count).
    """
    # Subtracts the scaled neuropil fluorescence from the ROI fluorescence. Casts the coefficient to float32 to
    # prevent Python's native float64 from promoting the entire computation chain to double precision.
    subtracted = cell_fluorescence - np.float32(neuropil_coefficient) * neuropil_fluorescence

    # Converts the baseline window from seconds to frames using the acquisition sampling rate. Forces the window to be
    # odd for symmetric min/max filtering.
    window_frames = int(baseline_window * sampling_rate)
    if window_frames % 2 == 0:
        window_frames += 1

    # Uses the requested method to calculate the baseline for the neuropil-subtracted fluorescence traces.
    if baseline_method == "maximin":
        # Uses truncate=3.0 to match the original suite2p's 3-sigma FIR Gaussian kernel, and mode='nearest'
        # (replicate edge values) for the min/max filters to match the original's boundary handling.
        baseline = gaussian_filter(input=subtracted, sigma=[0.0, baseline_sigma], truncate=3.0).astype(np.float32)
        baseline = minimum_filter1d(input=baseline, size=window_frames, axis=1, mode="nearest")
        baseline = maximum_filter1d(input=baseline, size=window_frames, axis=1, mode="nearest")
    elif baseline_method == "constant":
        baseline = gaussian_filter(input=subtracted, sigma=[0.0, baseline_sigma]).astype(np.float32)
        baseline = np.amin(a=baseline)
    elif baseline_method == "constant_percentile":
        baseline = np.percentile(a=subtracted, q=baseline_percentile, axis=1, keepdims=True).astype(np.float32)
    else:
        message = (
            f"Unable to compute delta fluorescence for spike deconvolution. The baseline computation "
            f"method must be 'maximin', 'constant', or 'constant_percentile', but got '{baseline_method}'."
        )
        console.error(message=message, error=ValueError)

    # Subtracts the computed baseline fluorescence from the neuropil-subtracted trace.
    subtracted -= baseline
    return subtracted


def apply_oasis_deconvolution(
    cell_fluorescence: NDArray[np.float32],
    batch_size: int,
    time_constant: float,
    sampling_rate: float,
) -> NDArray[np.float32]:
    """Applies the OASIS spike deconvolution algorithm to baseline-corrected ROI fluorescence traces.

    Notes:
        The ROIs are processed in batches to limit peak memory usage, as the OASIS algorithm requires four workspace
        arrays per batch, each with shape (batch_count, frame_count). Within each batch, the Numba-parallelized
        kernel deconvolves all ROIs concurrently and writes the inferred spike amplitudes directly into the
        corresponding slice of the output array.

    Args:
        cell_fluorescence: The baseline-corrected fluorescence traces with shape (roi_count, frame_count).
        batch_size: The number of ROIs to process per batch.
        time_constant: The exponential decay time constant of the calcium indicator, in seconds.
        sampling_rate: The imaging sampling rate for the processed recording plane, in Hz.

    Returns:
        The deconvolved spike traces with shape (roi_count, frame_count).
    """
    roi_count, frame_count = cell_fluorescence.shape
    cell_fluorescence = np.ascontiguousarray(cell_fluorescence, dtype=np.float32)
    spike_traces: NDArray[np.float32] = np.zeros((roi_count, frame_count), dtype=np.float32)

    # Runs the OASIS algorithm on all detected ROIs in batches.
    for start_index in range(0, roi_count, batch_size):
        # Initializes worker arrays for the current batch.
        end_index = min(start_index + batch_size, roi_count)
        batch_count = end_index - start_index
        pool_amplitude = np.empty((batch_count, frame_count), dtype=np.float32)
        pool_weight = np.empty((batch_count, frame_count), dtype=np.float32)
        pool_start_frame = np.empty((batch_count, frame_count), dtype=np.int32)
        pool_length = np.empty((batch_count, frame_count), dtype=np.float32)

        # Runs the OASIS algorithm that modifies spike_traces in-place.
        _oasis_matrix(
            cell_fluorescence=cell_fluorescence[start_index:end_index],
            pool_amplitude=pool_amplitude,
            pool_weight=pool_weight,
            pool_start_frame=pool_start_frame,
            pool_length=pool_length,
            spike_trace=spike_traces[start_index:end_index],
            time_constant=time_constant,
            sampling_rate=sampling_rate,
        )

    return spike_traces


@njit(cache=True, parallel=True)
def _oasis_matrix(
    cell_fluorescence: NDArray[np.float32],
    pool_amplitude: NDArray[np.float32],
    pool_weight: NDArray[np.float32],
    pool_start_frame: NDArray[np.int32],
    pool_length: NDArray[np.float32],
    spike_trace: NDArray[np.float32],
    time_constant: float,
    sampling_rate: float,
) -> None:
    """Performs spike deconvolution on all ROI fluorescence traces in parallel using the OASIS algorithm.

    Notes:
        This implements the unconstrained non-negative AR(1) OASIS solver from Friedrich et al. (2017). The algorithm
        models calcium fluorescence as a series of exponentially decaying "pools", where each pool represents a
        contiguous trace segment governed by a single initial amplitude and a shared decay rate. For each new time
        point, a single-frame pool is created and the algorithm checks backward through adjacent pools: if a previous
        pool's decayed value exceeds the current pool's amplitude, the implied spike between them would be negative,
        violating the non-negativity constraint. Merging resolves this by computing the optimal shared amplitude for the
        combined segment via weighted least squares. After processing all time-points, spike amplitudes are extracted as
        the discontinuities between consecutive pools, where each spike magnitude equals the difference between the
        current pool's amplitude and the predicted (decayed) value from the preceding pool.

    Args:
        cell_fluorescence: The batch of ROI fluorescence signals with shape (roi_count, frame_count).
        pool_amplitude: The workspace array that stores the optimal initial fluorescence amplitude for each pool, with
            shape (roi_count, frame_count). Each value represents the best-fit starting amplitude of a contiguous
            trace segment under the AR(1) decay model.
        pool_weight: The workspace array that stores the accumulated squared-decay normalization factor for each pool,
            with shape (roi_count, frame_count). These factors serve as denominators in the weighted least-squares
            update when adjacent pools are merged.
        pool_start_frame: The workspace array that stores the starting frame index of each pool, with shape
            (roi_count, frame_count).
        pool_length: The workspace array that stores the duration of each pool in frames, with shape
            (roi_count, frame_count).
        spike_trace: The output array that stores the deconvolved spike trace with shape (roi_count, frame_count).
            Nonzero values at pool boundaries indicate inferred spike events, with the magnitude representing the
            estimated spike amplitude.
        time_constant: The exponential decay time constant of the calcium indicator, in seconds.
        sampling_rate: The imaging sampling rate for the processed plane, in Hz.
    """
    # Converts the time constant from seconds to a per-frame negative exponent so that exp(decay_constant * n)
    # gives the calcium decay factor over n frames.
    decay_constant = -1.0 / (time_constant * sampling_rate)

    for roi_index in prange(cell_fluorescence.shape[0]):
        trace_length = cell_fluorescence[roi_index].shape[0]
        pool_index = 0

        for time_index in range(trace_length):
            # Creates a new single-frame pool for the current time point. Each pool starts with unit weight,
            # unit length, and an amplitude equal to the observed fluorescence value.
            pool_amplitude[roi_index, pool_index] = cell_fluorescence[roi_index, time_index]
            pool_weight[roi_index, pool_index] = 1
            pool_start_frame[roi_index, pool_index] = time_index
            pool_length[roi_index, pool_index] = 1

            # Walks backward through pools to enforce the non-negativity constraint on spikes. If a previous
            # pool's value, decayed forward to the current pool's start, exceeds the current pool's amplitude,
            # the implied spike between them would be negative. Merging resolves this by finding the optimal
            # shared amplitude for the combined segment.
            current_index = pool_index
            for _ in range(pool_index, 0, -1):
                if current_index == 0:
                    break

                # Computes the decay factor once and reuses it for the condition check, the weighted average
                # update, and the squared decay term (via multiplication instead of a third exp() call).
                previous_pool_decay = np.exp(decay_constant * pool_length[roi_index, current_index - 1])
                predicted = pool_amplitude[roi_index, current_index - 1] * previous_pool_decay
                if predicted <= pool_amplitude[roi_index, current_index]:
                    break

                # Merges the current pool into the previous one using weighted least squares. The new weight
                # accumulates the squared-decay contribution from the current pool, and the merged amplitude is
                # recomputed as the weighted average of both pools' contributions.
                decay_squared = previous_pool_decay * previous_pool_decay
                new_pool_weight = (
                    pool_weight[roi_index, current_index - 1] + pool_weight[roi_index, current_index] * decay_squared
                )

                pool_amplitude[roi_index, current_index - 1] = (
                    pool_amplitude[roi_index, current_index - 1] * pool_weight[roi_index, current_index - 1]
                    + pool_amplitude[roi_index, current_index]
                    * pool_weight[roi_index, current_index]
                    * previous_pool_decay
                ) / new_pool_weight

                pool_weight[roi_index, current_index - 1] = new_pool_weight
                pool_length[roi_index, current_index - 1] += pool_length[roi_index, current_index]
                current_index -= 1

            pool_index = current_index + 1

        # Extracts spike amplitudes as the discontinuity between each consecutive pair of pools. The spike at a
        # pool boundary equals the current pool's amplitude minus the previous pool's value decayed to that point.
        for pool in range(1, pool_index):
            spike_trace[roi_index, pool_start_frame[roi_index, pool]] = pool_amplitude[
                roi_index, pool
            ] - pool_amplitude[roi_index, pool - 1] * np.exp(decay_constant * pool_length[roi_index, pool - 1])
