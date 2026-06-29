"""Contains tests for the deconvolve module."""

from __future__ import annotations

import numpy as np
import pytest

from cindra.extraction.deconvolve import apply_oasis_deconvolution, compute_delta_fluorescence


class TestComputeDeltaFluorescence:
    """Tests compute_delta_fluorescence."""

    def test_output_shape_and_dtype(self) -> None:
        """Verifies the output shape and dtype match expectations."""
        rng = np.random.default_rng(42)
        cell = rng.standard_normal((5, 200)).astype(np.float32) + 100.0
        neuropil = rng.standard_normal((5, 200)).astype(np.float32) + 80.0
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
        # With constant baseline, baseline = min(smoothed) which should be close to 100 - 0.7*50 = 65.
        # After baseline subtraction, result should be near zero.
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
        """Verifies the maximin baseline method runs and produces reasonable output."""
        rng = np.random.default_rng(42)
        cell = rng.standard_normal((3, 300)).astype(np.float32) + 200.0
        neuropil = rng.standard_normal((3, 300)).astype(np.float32) + 150.0
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
        assert result.shape == (3, 300)
        assert np.isfinite(result).all()

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
        # The bump region should have positive delta F.
        assert np.mean(result[0, 105:115]) > np.mean(result[0, :20])

    def test_constant_percentile_baseline(self) -> None:
        """Verifies the constant_percentile baseline method uses per-ROI percentile."""
        rng = np.random.default_rng(42)
        cell = rng.standard_normal((2, 200)).astype(np.float32) + 100.0
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
        # Baseline is 8th percentile, so most values should be positive.
        assert np.mean(result > 0) > 0.5

    def test_invalid_baseline_method_raises(self) -> None:
        """Verifies that an invalid baseline method raises ValueError."""
        cell = np.ones((1, 100), dtype=np.float32)
        neuropil = np.zeros((1, 100), dtype=np.float32)
        with pytest.raises(ValueError, match="Unable to compute delta fluorescence"):
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

    def test_even_window_incremented_to_odd(self) -> None:
        """Verifies that an even baseline window is incremented to odd for symmetric filtering."""
        cell = np.ones((1, 100), dtype=np.float32) * 100.0
        neuropil = np.zeros((1, 100), dtype=np.float32)
        # sampling_rate=30, window=1.0 => 30 frames (even) => 31 (odd). Should not error.
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
        rng = np.random.default_rng(42)
        fluorescence = np.maximum(rng.standard_normal((5, 200)).astype(np.float32), 0.0)
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
        rng = np.random.default_rng(42)
        fluorescence = np.maximum(rng.standard_normal((10, 300)).astype(np.float32), 0.0)
        result = apply_oasis_deconvolution(
            cell_fluorescence=fluorescence,
            batch_size=5,
            time_constant=1.0,
            sampling_rate=30.0,
        )
        assert np.all(result >= -1e-6)

    def test_detects_spike_in_exponential_decay(self) -> None:
        """Verifies that OASIS detects a spike at the onset of an exponential decay."""
        # An isolated onset followed by exponential decay should localize the deconvolved spike at frame 50.
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
        # The largest spike should be at the onset of the decay (frame 50).
        spike_frame = np.argmax(result[0])
        assert spike_frame == 50

    def test_batching_produces_consistent_results(self) -> None:
        """Verifies that different batch sizes produce identical results."""
        rng = np.random.default_rng(42)
        fluorescence = np.maximum(rng.standard_normal((8, 150)).astype(np.float32), 0.0)
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
