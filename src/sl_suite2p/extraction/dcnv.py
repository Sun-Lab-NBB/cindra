"""This module provides utilities to deconvolve spike_amplitude from neuropil-corrected fluorescence traces."""

import platform

from numba import njit, config, prange
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, maximum_filter1d, minimum_filter1d

if platform.system() == "Darwin":
    config.THREADING_LAYER = "omp"
else:
    config.THREADING_LAYER = "tbb"


@njit(["float32[:], float32[:], float32[:], int64[:], float32[:], float32[:], float32, float32"], cache=True)
def oasis_trace(
    cell_fluorescence: NDArray[np.float32],
    average_fluorescence: NDArray[np.float32],
    pool_weight: NDArray[np.float32],
    pool_start_time: NDArray[np.int64],
    pool_length: NDArray[np.float32],
    spike_amplitude: NDArray[np.float32],
    time_constant: float,
    sampling_rate: float,
) -> NDArray[np.float32]:
    """Performs spike deconvolution on a single fluorescence trace using the OASIS algorithm.

    It then checks whether the exponential decay constraint is violated between consecutive pools
    and merges any violating pools into a single segment until the decay constraint is satisfied.
    Once the calcium trace is rebuilt into a monotonically decreasing sequence, spike_amplitude are identified
    as time points where the reconstructed calcium concentration exceeds the fluorescence from the
    previous pool.

    Args:
        cell_fluorescence: The array representing the raw fluorescence signal over time for a single neuron
        average_fluorescence: The array that stores the mean fluorescence value of each pool during deconvolution
        pool_weight: The array that stores the weighting coefficients used to compute the weighted average when
                     two or more pools are merged.
        pool_start_time: The array that stores the starting frame index of each pool.
        pool_length: The array that stores duration of each pool in frames.
        spike_amplitude: The array that stores the spike amplitudes. Nonzero values at pool start times indicate
                         likely spike events.
        time_constant: The timescale of the calcium indicator in seconds, used for the deconvolution kernel.
        sampling_rate: The sampling rate per plane.
    """
    trace_length = cell_fluorescence.shape[0]
    decay_constant = -1.0 / (time_constant * sampling_rate)
    pool_index = 0

    for time_index in range(trace_length):
        # Initializes a new pool to represent a possible calcium segment
        average_fluorescence[pool_index] = cell_fluorescence[time_index]
        pool_weight[pool_index] = 1
        pool_start_time[pool_index] = time_index
        pool_length[pool_index] = 1

        # Checks and corrects exponential decay violations by merging pools
        current_idx = pool_index
        for _ in range(pool_index, 0, -1):
            if (
                current_idx > 0
                and average_fluorescence[current_idx - 1] * np.exp(decay_constant * pool_length[current_idx - 1])
                > average_fluorescence[current_idx]
            ):
                # Merges the current pool with the previous one
                prev_pool_decay = np.exp(decay_constant * pool_length[current_idx - 1])
                decay_squared = np.exp(2 * decay_constant * pool_length[current_idx - 1])
                new_pool_weight = pool_weight[current_idx - 1] + pool_weight[current_idx] * decay_squared

                # Updates the merged pool's average value using a weighted average
                average_fluorescence[current_idx - 1] = (
                    average_fluorescence[current_idx - 1] * pool_weight[current_idx - 1]
                    + average_fluorescence[current_idx] * pool_weight[current_idx] * prev_pool_decay
                ) / new_pool_weight

                pool_weight[current_idx - 1] = new_pool_weight
                pool_length[current_idx - 1] += pool_length[current_idx]
                current_idx -= 1

        pool_index = current_idx + 1

    spike_amplitude[pool_start_time[1:pool_index]] = average_fluorescence[1:pool_index] - average_fluorescence[
        : pool_index - 1
    ] * np.exp(decay_constant * pool_length[: pool_index - 1])

    return spike_amplitude


@njit(
    ["float32[:,:], float32[:,:], float32[:,:], int64[:,:], float32[:,:], float32[:,:], float32, float32"],
    parallel=True,
    cache=True,
)
def oasis_matrix(
    cell_fluorescence: NDArray[np.float32],
    average_fluorescence: NDArray[np.float32],
    pool_weight: NDArray[np.float32],
    pool_start_time: NDArray[np.int64],
    pool_length: NDArray[np.float32],
    spike_amplitude: NDArray[np.float32],
    time_constant: float,
    sampling_rate: float,
) -> NDArray[np.float32]:
    """Applies the 'oasis_trace' function to all neurons in order for spike deconvolution
    to be performed in parallel across multiple cell fluorescence traces.

    Args:
        cell_fluorescence: The array representing the raw fluorescence signal over time for a single neuron.
        average_fluorescence: The array that stores the mean fluorescence value of each pool during deconvolution
        pool_weight: The array that stores the weights used to compute the weighted average when merging two or more
            pools of cells.
        pool_start_time: The array that stores the starting frame index of each pool.
        pool_length: The array that stores the duration of each pool in frames.
        spike_amplitude: The array that stores the spike amplitudes. Nonzero values at pool start times indicate
                         likely spike events.
        time_constant: The timescale of the calcium indicator in seconds, used for the deconvolution kernel.
        sampling_rate: The sampling rate per plane.
    """
    for i in prange(cell_fluorescence.shape[0]):
        oasis_trace(
            cell_fluorescence=cell_fluorescence[i],
            average_fluorescence=average_fluorescence[i],
            pool_weight=pool_weight[i],
            pool_start_time=pool_start_time[i],
            pool_length=pool_length[i],
            spike_amplitude=spike_amplitude[i],
            time_constant=time_constant,
            sampling_rate=sampling_rate,
        )


def oasis(
    cell_fluorescence: NDArray[np.float32], batch_size: int, time_constant: float, sampling_rate: float
) -> NDArray[np.float32]:
    """Computes non-negative deconvolution of calcium fluorescence traces to estimate spike activity.

    Args:
        cell_fluorescence: The ROI fluorescence traces after neuropil subtraction used for baseline correction.
        batch_size: The number of frames processed per batch.
        time_constant: The timescale of the calcium indicator in seconds, used for the deconvolution kernel.
        sampling_rate: The sampling rate of the imaging data per plane.
    """
    n_neurons, n_timepoints = cell_fluorescence.shape
    cell_fluorescence = cell_fluorescence.astype(np.float32)
    spike_traces = np.zeros((n_neurons, n_timepoints), dtype=np.float32)

    for start_index in range(0, n_neurons, batch_size):
        end_index = start_index + batch_size
        frame_batch = cell_fluorescence[start_index:end_index]
        average_fluorescence = np.zeros((frame_batch.shape[0], n_timepoints), dtype=np.float32)
        pool_weight = np.zeros((frame_batch.shape[0], n_timepoints), dtype=np.float32)
        pool_start_time = np.zeros((frame_batch.shape[0], n_timepoints), dtype=np.int64)
        pool_length = np.zeros((frame_batch.shape[0], n_timepoints), dtype=np.float32)
        spike_amplitude = np.zeros((frame_batch.shape[0], n_timepoints), dtype=np.float32)

        oasis_matrix(
            cell_fluorescence=frame_batch,
            average_fluorescence=average_fluorescence,
            pool_weight=pool_weight,
            pool_start_time=pool_start_time,
            pool_length=pool_length,
            spike_amplitude=spike_amplitude,
            time_constant=time_constant,
            sampling_rate=sampling_rate,
        )

        spike_traces[start_index:end_index] = spike_amplitude

    return spike_traces


def preprocess(
    cell_fluorescence: NDArray[np.float32],
    baseline: str,
    win_baseline: float,
    sig_baseline: float,
    sampling_rate: float,
    prctile_baseline: float = 8,
) -> NDArray[np.float32]:
    """Preprocesses fluorescence traces for spike deconvolution by performing baseline subtraction
    using the specified method and window size.

    Args:
        cell_fluorescence: The ROI fluorescence traces after neuropil subtraction used for baseline correction.
        baseline: The method for computing the baseline of each trace (maximin, constant, or constant_prctile).
        win_baseline: The time window (in seconds) used for the max/min filters.
        sig_baseline: The width of the Gaussian filter in frames.
        sampling_rate: The sampling rate of the imaging data per plane.
        prctile_baseline: The percentile of trace to use when baseline is constant_prctile.
    """
    win_frames = int(win_baseline * sampling_rate)
    baseline_trace = 0.0

    if baseline == "maximin":
        baseline_trace = gaussian_filter(cell_fluorescence, [0.0, sig_baseline])
        baseline_trace = minimum_filter1d(baseline_trace, win_frames, axis=1)
        baseline_trace = maximum_filter1d(baseline_trace, win_frames, axis=1)

    elif baseline == "constant":
        smoothed_fluorescence = gaussian_filter(cell_fluorescence, [0.0, sig_baseline])
        baseline_trace = np.amin(smoothed_fluorescence)

    elif baseline == "constant_prctile":
        baseline_trace = np.percentile(cell_fluorescence, prctile_baseline, axis=1)
        baseline_trace = np.expand_dims(baseline_trace, axis=1)

    return cell_fluorescence - baseline_trace
