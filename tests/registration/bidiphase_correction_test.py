"""Contains tests for the bidiphase_correction module."""

from __future__ import annotations

import numpy as np
import pytest

from cindra.registration.bidiphase_correction import (
    compute_bidirectional_phase_offset,
    apply_bidirectional_phase_correction,
)


class TestComputeBidirectionalPhaseOffset:
    """Tests compute_bidirectional_phase_offset."""

    def test_zero_offset_for_aligned_frames(self) -> None:
        """Verifies zero offset is returned when odd and even lines are aligned."""
        # Uses a smooth, structured pattern so odd/even line correlation is unambiguous.
        sample_positions = np.linspace(0, 4 * np.pi, 128, dtype=np.float32)
        pattern = np.sin(sample_positions)
        frames = np.tile(pattern, (20, 64, 1))
        offset = compute_bidirectional_phase_offset(frames=frames)
        assert offset == 0

    @pytest.mark.parametrize("shift", [3, 5, -3, -5])
    def test_detects_known_offset(self, shift: int) -> None:
        """Verifies detection of a known horizontal offset between odd and even lines."""
        width = 256
        sample_positions = np.arange(width, dtype=np.float32)
        # Uses a broadband multi-frequency pattern for unambiguous correlation.
        base_pattern = (
            np.sin(sample_positions * 0.1) + np.sin(sample_positions * 0.27) + np.sin(sample_positions * 0.53)
        )
        frames = np.zeros((30, 64, width), dtype=np.float32)
        frames[:, ::2, :] = base_pattern
        # np.roll applies a circular shift so the correlation has no edge artifacts.
        frames[:, 1::2, :] = np.roll(base_pattern, shift=shift)
        offset = compute_bidirectional_phase_offset(frames=frames)
        # Compares against the negative applied shift, which is the returned correction offset.
        assert abs(offset - (-shift)) <= 1

    def test_returns_int(self) -> None:
        """Verifies the return type is a Python int."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((5, 32, 32)).astype(np.float32)
        offset = compute_bidirectional_phase_offset(frames=frames)
        assert isinstance(offset, int)

    def test_odd_height_frames(self) -> None:
        """Verifies the function handles frames with odd height."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((5, 33, 64)).astype(np.float32)
        offset = compute_bidirectional_phase_offset(frames=frames)
        assert isinstance(offset, int)


class TestApplyBidirectionalPhaseCorrection:
    """Tests apply_bidirectional_phase_correction."""

    def test_zero_offset_no_change(self) -> None:
        """Verifies zero offset produces no change to frames."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((5, 32, 32)).astype(np.float32)
        original = frames.copy()
        apply_bidirectional_phase_correction(frames=frames, bidirectional_phase_offset=0)
        np.testing.assert_array_equal(frames, original)

    def test_positive_offset_shifts_odd_lines_right(self) -> None:
        """Verifies positive offset shifts odd lines to the right and zeros the left border."""
        frames = np.ones((1, 4, 10), dtype=np.float32)
        frames[0, 1, :] = np.arange(10, dtype=np.float32)  # Populates the odd line.
        apply_bidirectional_phase_correction(frames=frames, bidirectional_phase_offset=3)
        # Confirms the odd line shifted right by 3 with the left border zeroed.
        np.testing.assert_allclose(frames[0, 1, :3], 0.0)
        np.testing.assert_allclose(frames[0, 1, 3:], np.arange(7, dtype=np.float32))

    def test_negative_offset_shifts_odd_lines_left(self) -> None:
        """Verifies negative offset shifts odd lines to the left and zeros the right border."""
        frames = np.ones((1, 4, 10), dtype=np.float32)
        frames[0, 1, :] = np.arange(10, dtype=np.float32)  # Populates the odd line.
        apply_bidirectional_phase_correction(frames=frames, bidirectional_phase_offset=-3)
        # Confirms the odd line shifted left by 3 with the right border zeroed.
        np.testing.assert_allclose(frames[0, 1, :7], np.arange(3, 10, dtype=np.float32))
        np.testing.assert_allclose(frames[0, 1, 7:], 0.0)

    def test_in_place_modification(self) -> None:
        """Verifies the correction is applied in-place."""
        frames = np.ones((1, 4, 10), dtype=np.float32)
        original_id = id(frames)
        apply_bidirectional_phase_correction(frames=frames, bidirectional_phase_offset=2)
        assert id(frames) == original_id

    def test_even_lines_unchanged(self) -> None:
        """Verifies even lines are not modified by the correction."""
        rng = np.random.default_rng(42)
        frames = rng.standard_normal((2, 10, 20)).astype(np.float32)
        even_lines = frames[:, ::2, :].copy()
        apply_bidirectional_phase_correction(frames=frames, bidirectional_phase_offset=3)
        np.testing.assert_array_equal(frames[:, ::2, :], even_lines)

    def test_multiple_frames(self) -> None:
        """Verifies the correction is applied consistently to all frames."""
        frames = np.ones((3, 6, 10), dtype=np.float32)
        apply_bidirectional_phase_correction(frames=frames, bidirectional_phase_offset=2)
        # Confirms all frames are corrected identically.
        np.testing.assert_array_equal(frames[0], frames[1])
        np.testing.assert_array_equal(frames[1], frames[2])
        # Confirms the left border of odd lines is zeroed.
        np.testing.assert_allclose(frames[:, 1::2, :2], 0.0)
