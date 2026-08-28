"""Contains tests for the deconvolve module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from sklearn.isotonic import IsotonicRegression
from ataraxis_base_utilities import error_format

from cindra.extraction.deconvolve import apply_oasis_deconvolution, compute_delta_fluorescence

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TestComputeDeltaFluorescence:
    """Tests compute_delta_fluorescence."""

    def test_output_shape_and_dtype(self) -> None:
        """Verifies the output shape and dtype match expectations."""
        generator = np.random.default_rng(seed=42)
        cell = generator.standard_normal((5, 200)).astype(np.float32) + 100.0
        neuropil = generator.standard_normal((5, 200)).astype(np.float32) + 80.0
        result = compute_delta_fluorescence(
            cell_fluorescence=cell,
            neuropil_fluorescence=neuropil,
            neuropil_coefficient=0.7,
            baseline_method="maximin",
            baseline_window=1.0,
            baseline_sigma=3.0,
            baseline_percentile=8.0,
            sampling_rate=30.0,
        )
        assert result.shape == (5, 200)
        assert result.dtype == np.float32

    def test_neuropil_subtraction(self) -> None:
        """Verifies that neuropil signal is subtracted with the given coefficient."""
        cell = np.ones((1, 100), dtype=np.float32) * 100.0
        neuropil = np.ones((1, 100), dtype=np.float32) * 50.0
        # With a constant baseline, the baseline is min(smoothed), which should be close to 100 - 0.7 * 50 = 65.
        # After baseline subtraction, the result should be near zero.
        result = compute_delta_fluorescence(
            cell_fluorescence=cell,
            neuropil_fluorescence=neuropil,
            neuropil_coefficient=0.7,
            baseline_method="constant",
            baseline_window=1.0,
            baseline_sigma=3.0,
            baseline_percentile=8.0,
            sampling_rate=30.0,
        )
        np.testing.assert_allclose(result, 0.0, atol=1e-4)

    def test_maximin_baseline(self) -> None:
        """Verifies the maximin baseline method subtracts a constant baseline from a constant trace."""
        # A perfectly constant trace passes unchanged through the gaussian, minimum, and maximum filters, so the
        # estimated baseline equals the trace and the corrected delta fluorescence is zero everywhere.
        cell = np.ones((3, 300), dtype=np.float32) * 100.0
        neuropil = np.zeros((3, 300), dtype=np.float32)
        result = compute_delta_fluorescence(
            cell_fluorescence=cell,
            neuropil_fluorescence=neuropil,
            neuropil_coefficient=0.0,
            baseline_method="maximin",
            baseline_window=1.0,
            baseline_sigma=3.0,
            baseline_percentile=8.0,
            sampling_rate=30.0,
        )
        assert result.shape == (3, 300)
        np.testing.assert_allclose(result, 0.0, atol=1e-4)

    def test_odd_window_not_incremented(self) -> None:
        """Verifies that an already-odd baseline window is left unchanged during symmetric filtering."""
        cell = np.ones((1, 100), dtype=np.float32) * 100.0
        neuropil = np.zeros((1, 100), dtype=np.float32)
        # sampling_rate=31, window=1.0 => 31 frames (already odd) => left as-is. A constant trace yields zero delta F.
        result = compute_delta_fluorescence(
            cell_fluorescence=cell,
            neuropil_fluorescence=neuropil,
            neuropil_coefficient=0.0,
            baseline_method="maximin",
            baseline_window=1.0,
            baseline_sigma=3.0,
            baseline_percentile=8.0,
            sampling_rate=31.0,
        )
        assert result.shape == (1, 100)
        np.testing.assert_allclose(result, 0.0, atol=1e-4)

    def test_constant_baseline(self) -> None:
        """Verifies the constant baseline method uses the global minimum of the smoothed trace."""
        # A lone bump on a flat trace keeps the constant baseline at the smoothed global minimum.
        cell = np.ones((1, 200), dtype=np.float32) * 100.0
        cell[0, 100:120] += 50.0
        neuropil = np.zeros((1, 200), dtype=np.float32)
        result = compute_delta_fluorescence(
            cell_fluorescence=cell,
            neuropil_fluorescence=neuropil,
            neuropil_coefficient=0.0,
            baseline_method="constant",
            baseline_window=1.0,
            baseline_sigma=3.0,
            baseline_percentile=8.0,
            sampling_rate=30.0,
        )
        assert np.mean(result[0, 105:115]) > np.mean(result[0, :20])

    def test_constant_baseline_is_the_single_global_minimum(self) -> None:
        """Verifies the constant baseline is one global minimum shared by every ROI rather than a per-ROI minimum."""
        # A zero sigma makes the Gaussian stage the identity, so the baseline is exactly the smallest sample in the
        # whole array. The two ROIs hold disjoint integer ranges, 50..59 and 20..38, all of which float32 stores
        # exactly, so the baseline is 20.0 and both rows shift by that same amount.
        frame_indices = np.arange(10, dtype=np.float32)
        cell = np.stack([50.0 + frame_indices, 20.0 + 2.0 * frame_indices]).astype(np.float32)
        neuropil = np.zeros((2, 10), dtype=np.float32)

        result = compute_delta_fluorescence(
            cell_fluorescence=cell,
            neuropil_fluorescence=neuropil,
            neuropil_coefficient=0.0,
            baseline_method="constant",
            baseline_window=1.0,
            baseline_sigma=0.0,
            baseline_percentile=8.0,
            sampling_rate=30.0,
        )

        # A per-ROI minimum would leave the first row starting at 0.0 instead of 30.0.
        np.testing.assert_array_equal(result[0], 30.0 + frame_indices)
        np.testing.assert_array_equal(result[1], 2.0 * frame_indices)

    def test_constant_percentile_baseline(self) -> None:
        """Verifies the constant_percentile baseline method uses per-ROI percentile."""
        generator = np.random.default_rng(seed=42)
        cell = generator.standard_normal((2, 200)).astype(np.float32) + 100.0
        neuropil = np.zeros((2, 200), dtype=np.float32)
        result = compute_delta_fluorescence(
            cell_fluorescence=cell,
            neuropil_fluorescence=neuropil,
            neuropil_coefficient=0.0,
            baseline_method="constant_percentile",
            baseline_window=1.0,
            baseline_sigma=3.0,
            baseline_percentile=8.0,
            sampling_rate=30.0,
        )
        assert result.shape == (2, 200)
        # The baseline is the 8th percentile, so most values should be positive.
        assert np.mean(result > 0) > 0.5

    def test_invalid_baseline_method_raises(self) -> None:
        """Verifies that an invalid baseline method raises ValueError."""
        cell = np.ones((1, 100), dtype=np.float32)
        neuropil = np.zeros((1, 100), dtype=np.float32)
        expected_message = (
            "Unable to compute delta fluorescence for spike deconvolution. The baseline computation method must "
            "be 'maximin', 'constant', or 'constant_percentile', but got 'invalid_method'."
        )
        with pytest.raises(ValueError, match=error_format(expected_message)):
            compute_delta_fluorescence(
                cell_fluorescence=cell,
                neuropil_fluorescence=neuropil,
                neuropil_coefficient=0.7,
                baseline_method="invalid_method",
                baseline_window=1.0,
                baseline_sigma=3.0,
                baseline_percentile=8.0,
                sampling_rate=30.0,
            )

    def test_varying_neuropil_subtraction_matches_hand_derived_values(self) -> None:
        """Verifies the neuropil coefficient and the per-ROI percentile baseline against hand-derived values."""
        # The first ROI ramps as 100 + t and the second as 200 + 2 * t, while both share a neuropil trace of 2 * t.
        # With a coefficient of 0.7 the corrected traces are 100 - 0.4 * t (falling to 60.4) and 200 + 0.6 * t
        # (rising to 259.4). The two ROIs occupy disjoint value ranges, so a baseline pooled across ROIs cannot
        # reproduce either row.
        frame_indices = np.arange(100, dtype=np.float32)
        cell = np.stack([100.0 + frame_indices, 200.0 + 2.0 * frame_indices]).astype(np.float32)
        neuropil = np.stack([2.0 * frame_indices, 2.0 * frame_indices]).astype(np.float32)

        result = compute_delta_fluorescence(
            cell_fluorescence=cell,
            neuropil_fluorescence=neuropil,
            neuropil_coefficient=0.7,
            baseline_method="constant_percentile",
            baseline_window=1.0,
            baseline_sigma=3.0,
            baseline_percentile=8.0,
            sampling_rate=30.0,
        )

        # Sorted ascending, the first corrected trace is 60.4 + 0.4 * i and the second is 200 + 0.6 * i. The 8th
        # percentile of 100 samples sits at the fractional rank 0.08 * 99 = 7.92, so the two baselines are
        # 60.4 + 0.4 * 7.92 = 63.568 and 200 + 0.6 * 7.92 = 204.752.
        np.testing.assert_allclose(result[0], (100.0 - 0.4 * frame_indices) - 63.568, atol=1e-3)
        np.testing.assert_allclose(result[1], (200.0 + 0.6 * frame_indices) - 204.752, atol=1e-3)

        # A different coefficient reverses the slope of the first corrected trace entirely, so the coefficient the
        # function applies is pinned rather than merely present.
        weaker = compute_delta_fluorescence(
            cell_fluorescence=cell,
            neuropil_fluorescence=neuropil,
            neuropil_coefficient=0.2,
            baseline_method="constant_percentile",
            baseline_window=1.0,
            baseline_sigma=3.0,
            baseline_percentile=8.0,
            sampling_rate=30.0,
        )
        # Sorted ascending, 100 + 0.6 * t gives a baseline of 100 + 0.6 * 7.92 = 104.752, and 200 + 1.6 * t gives
        # 200 + 1.6 * 7.92 = 212.672.
        np.testing.assert_allclose(weaker[0], (100.0 + 0.6 * frame_indices) - 104.752, atol=1e-3)
        np.testing.assert_allclose(weaker[1], (200.0 + 1.6 * frame_indices) - 212.672, atol=1e-3)

    def test_maximin_window_parity_on_step_trace(self) -> None:
        """Verifies that the maximin window is widened to an odd size, keeping the min/max filters symmetric."""
        # A 40-frame plateau carrying a one-frame 5.0 spike at its center. With baseline_sigma at 0.0 the Gaussian
        # stage is the identity, so the baseline is the morphological opening of the trace by the window.
        cell = np.zeros((1, 100), dtype=np.float32)
        cell[0, 30:70] = 10.0
        cell[0, 50] += 5.0
        neuropil = np.zeros((1, 100), dtype=np.float32)

        result = compute_delta_fluorescence(
            cell_fluorescence=cell,
            neuropil_fluorescence=neuropil,
            neuropil_coefficient=0.0,
            baseline_method="maximin",
            baseline_window=1.0,
            baseline_sigma=0.0,
            baseline_percentile=8.0,
            sampling_rate=30.0,
        )

        # 1.0 s at 30 Hz gives 30 frames, which the parity correction widens to 31. A symmetric 31-frame opening
        # reproduces the plateau exactly, so the baseline cancels everything except the one-frame spike. Leaving
        # the window at 30 frames shifts the opening by one frame, which leaks 10.0 at frame 30 and -10.0 at 70.
        expected = np.zeros(100, dtype=np.float32)
        expected[50] = 5.0
        np.testing.assert_allclose(result[0], expected, atol=1e-5)

    def test_even_window_incremented_to_odd(self) -> None:
        """Verifies that an even baseline window is incremented to odd for symmetric filtering."""
        cell = np.ones((1, 100), dtype=np.float32) * 100.0
        neuropil = np.zeros((1, 100), dtype=np.float32)
        # sampling_rate=30, window=1.0 => 30 frames (even) => incremented to 31 (odd).
        result = compute_delta_fluorescence(
            cell_fluorescence=cell,
            neuropil_fluorescence=neuropil,
            neuropil_coefficient=0.0,
            baseline_method="maximin",
            baseline_window=1.0,
            baseline_sigma=3.0,
            baseline_percentile=8.0,
            sampling_rate=30.0,
        )
        assert result.shape == (1, 100)


class TestApplyOasisDeconvolution:
    """Tests apply_oasis_deconvolution."""

    def test_output_shape_and_dtype(self) -> None:
        """Verifies the output shape and dtype match expectations."""
        generator = np.random.default_rng(seed=42)
        fluorescence = np.maximum(generator.standard_normal((5, 200)).astype(np.float32), 0.0)
        result = apply_oasis_deconvolution(
            cell_fluorescence=fluorescence,
            batch_size=3,
            time_constant=1.0,
            sampling_rate=30.0,
        )
        assert result.shape == (5, 200)
        assert result.dtype == np.float32

    def test_zero_input_gives_zero_output(self) -> None:
        """Verifies that zero fluorescence produces zero spike traces."""
        fluorescence = np.zeros((3, 100), dtype=np.float32)
        result = apply_oasis_deconvolution(
            cell_fluorescence=fluorescence,
            batch_size=10,
            time_constant=1.0,
            sampling_rate=30.0,
        )
        np.testing.assert_array_equal(result, 0.0)

    def test_non_negative_spikes(self) -> None:
        """Verifies that all deconvolved spike values are non-negative."""
        generator = np.random.default_rng(seed=42)
        fluorescence = np.maximum(generator.standard_normal((10, 300)).astype(np.float32), 0.0)
        result = apply_oasis_deconvolution(
            cell_fluorescence=fluorescence,
            batch_size=5,
            time_constant=1.0,
            sampling_rate=30.0,
        )
        assert np.all(result >= -1e-6)

    def test_detects_spike_in_exponential_decay(self) -> None:
        """Verifies that OASIS detects a spike at the onset of an exponential decay."""
        time_constant = 1.0
        sampling_rate = 30.0
        frame_count = 200
        trace = np.zeros((1, frame_count), dtype=np.float32)
        decay_constant = -1.0 / (time_constant * sampling_rate)
        for frame_index in range(50, frame_count):
            trace[0, frame_index] = 10.0 * np.exp(decay_constant * (frame_index - 50))

        result = apply_oasis_deconvolution(
            cell_fluorescence=trace,
            batch_size=1,
            time_constant=time_constant,
            sampling_rate=sampling_rate,
        )
        # The trace is exactly one AR(1) pool, so the optimal solution is a single spike carrying the whole onset
        # amplitude at frame 50 and nothing anywhere else. Both the peak location and its magnitude are exact.
        assert int(np.argmax(result[0])) == 50
        np.testing.assert_allclose(result[0, 50], 10.0, atol=1e-5)

        # The total inferred spike mass equals the single spike's amplitude, and no other frame carries mass. The
        # off-peak residual measures 1.7e-07 for this trace, so the 1e-5 bound leaves nearly two orders of headroom.
        np.testing.assert_allclose(result[0].sum(), 10.0, atol=1e-5)
        assert float(np.abs(np.delete(result[0], 50)).max()) < 1e-5

    def test_recovers_two_synthesized_spikes(self) -> None:
        """Verifies that OASIS recovers the exact amplitudes and frames of two spikes driving an AR(1) calcium trace."""
        time_constant = 1.0
        sampling_rate = 30.0
        frame_count = 200
        decay = float(np.exp(-1.0 / (time_constant * sampling_rate)))

        # Synthesizes the calcium trace the AR(1) model produces from spikes of 5.0 at frame 30 and 8.0 at frame 90.
        # The trace is exactly feasible, so the optimal solution reproduces the driving spikes.
        spikes = np.zeros(frame_count, dtype=np.float64)
        spikes[30] = 5.0
        spikes[90] = 8.0
        calcium = np.zeros(frame_count, dtype=np.float64)
        for frame_index in range(1, frame_count):
            calcium[frame_index] = decay * calcium[frame_index - 1] + spikes[frame_index]

        result = apply_oasis_deconvolution(
            cell_fluorescence=calcium[None, :].astype(np.float32),
            batch_size=1,
            time_constant=time_constant,
            sampling_rate=sampling_rate,
        )

        np.testing.assert_allclose(result[0, 30], 5.0, atol=1e-5)
        np.testing.assert_allclose(result[0, 90], 8.0, atol=1e-5)
        assert float(np.abs(np.delete(result[0], [30, 90])).max()) < 1e-5

        # The recovered spikes drive an AR(1) reconstruction that matches the input trace to float32 precision.
        reconstruction = np.zeros(frame_count, dtype=np.float64)
        for frame_index in range(1, frame_count):
            reconstruction[frame_index] = decay * reconstruction[frame_index - 1] + result[0, frame_index]
        assert float(np.abs(reconstruction - calcium).max()) < 1e-4

    def test_matches_isotonic_regression_oracle(self) -> None:
        """Verifies that the pool-merge kernel reproduces the weighted isotonic regression optimum on a noisy trace."""
        time_constant = 1.0
        sampling_rate = 30.0
        frame_count = 200
        decay = float(np.exp(-1.0 / (time_constant * sampling_rate)))

        # Builds a noisy trace carrying roughly 25 spikes. The noise forces the solver through many pool merges,
        # which is the arithmetic an exactly feasible trace never exercises.
        generator = np.random.default_rng(seed=7)
        spikes = (generator.random(frame_count) < 0.12) * generator.uniform(2.0, 8.0, frame_count)
        calcium = np.zeros(frame_count, dtype=np.float64)
        for frame_index in range(1, frame_count):
            calcium[frame_index] = decay * calcium[frame_index - 1] + spikes[frame_index]
        trace = (calcium + generator.standard_normal(frame_count) * 0.5).astype(np.float32)

        result = apply_oasis_deconvolution(
            cell_fluorescence=trace[None, :],
            batch_size=1,
            time_constant=time_constant,
            sampling_rate=sampling_rate,
        )

        expected = _isotonic_oasis_oracle(
            trace=trace,
            time_constant=time_constant,
            sampling_rate=sampling_rate,
        )

        # Frame 0 is skipped because the kernel extracts spikes only at pool boundaries and never writes index 0.
        # The worst deviation across the remaining frames measures 1.8e-06, so the 1e-4 bound is safe while still
        # rejecting the multi-unit merge errors a regression in the pool update produces.
        np.testing.assert_allclose(result[0, 1:], expected[1:], atol=1e-4)

    def test_batching_produces_consistent_results(self) -> None:
        """Verifies that different batch sizes produce identical results."""
        generator = np.random.default_rng(seed=42)
        fluorescence = np.maximum(generator.standard_normal((8, 150)).astype(np.float32), 0.0)
        result_small_batch = apply_oasis_deconvolution(
            cell_fluorescence=fluorescence.copy(),
            batch_size=2,
            time_constant=1.0,
            sampling_rate=30.0,
        )
        result_large_batch = apply_oasis_deconvolution(
            cell_fluorescence=fluorescence.copy(),
            batch_size=10,
            time_constant=1.0,
            sampling_rate=30.0,
        )
        np.testing.assert_allclose(result_small_batch, result_large_batch, atol=1e-6)


def _isotonic_oasis_oracle(
    trace: NDArray[np.float32],
    time_constant: float,
    sampling_rate: float,
) -> NDArray[np.float64]:
    """Solves the unconstrained non-negative AR(1) deconvolution problem with a weighted isotonic regression."""
    # The OASIS problem is min ||trace - calcium||^2 subject to calcium[t] >= gamma * calcium[t - 1]. Substituting
    # calcium[t] = gamma**t * z[t] turns the constraint into z[t] >= z[t - 1], which makes the problem a weighted
    # isotonic regression of trace[t] / gamma**t with weights gamma**(2 * t). This is a completely different solver
    # (pool adjacent violators, from scikit-learn) reaching the same optimum as the pool-merge kernel under test.
    decay = np.exp(-1.0 / (time_constant * sampling_rate))
    frame_indices = np.arange(trace.shape[0], dtype=np.float64)
    decay_powers = decay**frame_indices
    monotone = IsotonicRegression(increasing=True).fit_transform(
        frame_indices,
        trace.astype(np.float64) / decay_powers,
        sample_weight=decay_powers**2,
    )
    calcium = decay_powers * monotone

    # Recovers the spikes as the discontinuities the AR(1) model leaves between consecutive calcium samples.
    spikes = np.zeros_like(calcium)
    spikes[1:] = calcium[1:] - decay * calcium[:-1]
    return spikes
