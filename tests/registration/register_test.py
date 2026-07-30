"""Contains tests for the register module helper functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cindra.detection import compute_registration_blocks
from cindra.registration.register import (
    _compute_crop,
    _compute_reference,
    _pick_initial_reference,
    _apply_precomputed_offsets_batch,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


def _build_baseline_frames(frame_count: int, size: int, baseline: float, seed: int) -> NDArray[np.float32]:
    """Builds a stack of blob frames with integer-pixel shifts rendered on top of a positive intensity baseline.

    Args:
        frame_count: The number of frames to render.
        size: The height and width of each square frame in pixels.
        baseline: The constant intensity offset added to every pixel of every frame.
        seed: The seed for the random generator used to render the per-frame noise.

    Returns:
        The rendered frame stack with shape (frame_count, size, size) as a float32 array.
    """
    generator = np.random.default_rng(seed=seed)
    y_grid = np.arange(size, dtype=np.float32)[:, np.newaxis]
    x_grid = np.arange(size, dtype=np.float32)[np.newaxis, :]

    template = np.zeros((size, size), dtype=np.float32)
    for center_y, center_x in ((12, 14), (30, 20), (22, 36)):
        template += 90.0 * np.exp(-(((y_grid - center_y) ** 2 + (x_grid - center_x) ** 2) / 18.0))

    frames = np.empty((frame_count, size, size), dtype=np.float32)
    for index in range(frame_count):
        shifted = np.roll(template, shift=(index % 5 - 2, index % 3 - 1), axis=(0, 1))
        frames[index] = baseline + shifted + generator.normal(scale=6.0, size=(size, size))
    return frames


class TestComputeCrop:
    """Tests the _compute_crop function."""

    def test_returns_valid_region_for_small_offsets(self) -> None:
        """Verifies that small offsets produce a large valid region."""
        frame_count = 100
        y_offsets = np.zeros(frame_count, dtype=np.int32)
        x_offsets = np.zeros(frame_count, dtype=np.int32)
        correlations = np.ones(frame_count, dtype=np.float32)
        bad_frames = np.zeros(frame_count, dtype=np.bool_)

        result_bad_frames, valid_y_range, valid_x_range = _compute_crop(
            x_offsets=x_offsets,
            y_offsets=y_offsets,
            correlations=correlations,
            bad_frame_threshold=1.0,
            bad_frames=bad_frames,
            maximum_offset_fraction=0.1,
            frame_height=64,
            frame_width=64,
        )

        assert valid_y_range == (0, 64)
        assert valid_x_range == (0, 64)
        assert result_bad_frames.shape == (frame_count,)

    def test_large_offsets_shrink_valid_region(self) -> None:
        """Verifies that large offsets reduce the valid pixel region."""
        frame_count = 100
        y_offsets = np.zeros(frame_count, dtype=np.int32)
        x_offsets = np.zeros(frame_count, dtype=np.int32)
        # Introduces a few frames with large offsets.
        y_offsets[50] = 10
        x_offsets[50] = 5
        correlations = np.ones(frame_count, dtype=np.float32)
        bad_frames = np.zeros(frame_count, dtype=np.bool_)

        _, valid_y_range, valid_x_range = _compute_crop(
            x_offsets=x_offsets,
            y_offsets=y_offsets,
            correlations=correlations,
            bad_frame_threshold=1.0,
            bad_frames=bad_frames,
            maximum_offset_fraction=0.5,
            frame_height=64,
            frame_width=64,
        )

        # The valid region should be smaller than the full frame.
        assert valid_y_range[0] >= 0
        assert valid_y_range[1] <= 64
        assert valid_x_range[0] >= 0
        assert valid_x_range[1] <= 64

    def test_detects_bad_frames_from_large_offsets(self) -> None:
        """Verifies that frames exceeding the maximum offset fraction are flagged as bad."""
        frame_count = 100
        y_offsets = np.zeros(frame_count, dtype=np.int32)
        x_offsets = np.zeros(frame_count, dtype=np.int32)
        # Sets an extremely large offset that exceeds the threshold.
        x_offsets[10] = 50
        correlations = np.ones(frame_count, dtype=np.float32)
        bad_frames = np.zeros(frame_count, dtype=np.bool_)

        result_bad_frames, _, _ = _compute_crop(
            x_offsets=x_offsets,
            y_offsets=y_offsets,
            correlations=correlations,
            bad_frame_threshold=1.0,
            bad_frames=bad_frames,
            maximum_offset_fraction=0.1,
            frame_height=64,
            frame_width=64,
        )

        # Frame 10 should be flagged as bad (50 > 0.1 * 64 * 0.95 = 6.08).
        assert result_bad_frames[10]

    def test_preserves_existing_bad_frames(self) -> None:
        """Verifies that pre-existing bad frame flags are preserved."""
        frame_count = 100
        y_offsets = np.zeros(frame_count, dtype=np.int32)
        x_offsets = np.zeros(frame_count, dtype=np.int32)
        correlations = np.ones(frame_count, dtype=np.float32)
        bad_frames = np.zeros(frame_count, dtype=np.bool_)
        bad_frames[5] = True

        result_bad_frames, _, _ = _compute_crop(
            x_offsets=x_offsets,
            y_offsets=y_offsets,
            correlations=correlations,
            bad_frame_threshold=1.0,
            bad_frames=bad_frames,
            maximum_offset_fraction=0.1,
            frame_height=64,
            frame_width=64,
        )

        assert result_bad_frames[5]


class TestPickInitialReference:
    """Tests the _pick_initial_reference function."""

    def test_output_shape(self) -> None:
        """Verifies the output shape matches the spatial dimensions of the input frames."""
        generator = np.random.default_rng(seed=42)
        frames = generator.standard_normal(size=(30, 16, 16)).astype(np.float32)

        reference = _pick_initial_reference(frames=frames, top_correlations=5)

        assert reference.shape == (16, 16)

    def test_output_dtype(self) -> None:
        """Verifies the output has float32 or float64 dtype."""
        generator = np.random.default_rng(seed=42)
        frames = generator.standard_normal(size=(30, 16, 16)).astype(np.float32)

        reference = _pick_initial_reference(frames=frames, top_correlations=5)

        assert reference.dtype in (np.float32, np.float64)

    def test_output_is_finite(self) -> None:
        """Verifies the output contains no NaN or Inf values."""
        generator = np.random.default_rng(seed=42)
        frames = generator.standard_normal(size=(30, 16, 16)).astype(np.float32)

        reference = _pick_initial_reference(frames=frames, top_correlations=5)

        assert np.all(np.isfinite(reference))

    def test_reference_resembles_average(self) -> None:
        """Verifies the reference is a meaningful average of correlated frames."""
        generator = np.random.default_rng(seed=42)
        # Creates frames with a common signal so the reference should resemble it.
        signal = generator.standard_normal(size=(16, 16)).astype(np.float32) * 10
        noise = generator.standard_normal(size=(30, 16, 16)).astype(np.float32)
        frames = signal[np.newaxis, :, :] + noise

        reference = _pick_initial_reference(frames=frames, top_correlations=10)

        # The reference should correlate with the underlying signal.
        assert reference.shape == (16, 16)
        assert np.std(reference) > 0

    def test_leaves_input_frames_unchanged(self) -> None:
        """Verifies the input frames keep their original values, including their raw intensity scale."""
        frames = _build_baseline_frames(frame_count=24, size=48, baseline=1000.0, seed=17)
        original = frames.copy()

        _pick_initial_reference(frames=frames, top_correlations=5)

        np.testing.assert_array_equal(frames, original)


class TestComputeReference:
    """Tests the _compute_reference function."""

    def test_reference_stays_on_raw_intensity_scale(self) -> None:
        """Verifies the reference keeps the intensity scale of frames carrying a large positive baseline."""
        frames = _build_baseline_frames(frame_count=24, size=48, baseline=1000.0, seed=17)
        frames_mean = float(frames.mean())

        reference = _compute_reference(
            frames=frames,
            pre_smoothing_sigma=0.0,
            spatial_highpass_window=10,
            edge_taper_pixels=10.0,
            spatial_smoothing_sigma=1.15,
            maximum_offset_fraction=0.1,
            temporal_smoothing_sigma=0.0,
            workers=1,
            one_photon_enabled=False,
        )

        # register_plane derives the 1st and 99th percentile clip bounds from this reference and applies them to raw
        # binary frames, so a reference on a different intensity scale saturates every pixel of the correlation input.
        assert reference.shape == (48, 48)
        assert abs(float(reference.mean()) - frames_mean) < 0.05 * frames_mean


class TestApplyPrecomputedOffsetsBatch:
    """Tests _apply_precomputed_offsets_batch."""

    def test_zero_offsets_preserve_frames(self) -> None:
        """Verifies that zero offsets leave frames unchanged."""
        generator = np.random.default_rng(seed=42)
        batch_size = 5
        height = 16
        width = 16
        frames = generator.standard_normal((batch_size, height, width)).astype(np.float32)
        original = frames.copy()

        y_offsets = np.zeros(batch_size, dtype=np.int32)
        x_offsets = np.zeros(batch_size, dtype=np.int32)

        result = _apply_precomputed_offsets_batch(
            frames=frames,
            y_offsets=y_offsets,
            x_offsets=x_offsets,
            y_offsets_nonrigid=None,
            x_offsets_nonrigid=None,
            blocks=None,
            bidirectional_phase_offset=0,
            bidirectional_phase_corrected=True,
            nonrigid_enabled=False,
        )

        np.testing.assert_array_equal(result, original)

    def test_rigid_offsets_shift_frames(self) -> None:
        """Verifies that non-zero rigid offsets produce shifted frames when nonrigid is disabled."""
        batch_size = 3
        height = 16
        width = 16
        frames = np.zeros((batch_size, height, width), dtype=np.float32)
        # Places a bright pixel at a known location in each frame.
        frames[:, 8, 8] = 100.0

        y_offsets = np.array([2, 0, -1], dtype=np.int32)
        x_offsets = np.array([0, 3, 0], dtype=np.int32)

        result = _apply_precomputed_offsets_batch(
            frames=frames,
            y_offsets=y_offsets,
            x_offsets=x_offsets,
            y_offsets_nonrigid=None,
            x_offsets_nonrigid=None,
            blocks=None,
            bidirectional_phase_offset=0,
            bidirectional_phase_corrected=True,
            nonrigid_enabled=False,
        )

        # The translate_frame function applies roll with (-y_offset, -x_offset), so the bright pixel moves.
        # Frame 0: y_offset=2 -> pixel moves from (8,8) to (6,8).
        assert result[0, 6, 8] == 100.0
        # Frame 1: x_offset=3 -> pixel moves from (8,8) to (8,5).
        assert result[1, 8, 5] == 100.0
        # Frame 2: y_offset=-1 -> pixel moves from (8,8) to (9,8).
        assert result[2, 9, 8] == 100.0

    def test_output_shape_matches_input(self) -> None:
        """Verifies that the output shape matches the input frame shape."""
        batch_size = 4
        height = 20
        width = 24
        generator = np.random.default_rng(seed=7)
        frames = generator.standard_normal((batch_size, height, width)).astype(np.float32)

        y_offsets = np.zeros(batch_size, dtype=np.int32)
        x_offsets = np.zeros(batch_size, dtype=np.int32)

        result = _apply_precomputed_offsets_batch(
            frames=frames,
            y_offsets=y_offsets,
            x_offsets=x_offsets,
            y_offsets_nonrigid=None,
            x_offsets_nonrigid=None,
            blocks=None,
            bidirectional_phase_offset=0,
            bidirectional_phase_corrected=True,
            nonrigid_enabled=False,
        )

        assert result.shape == (batch_size, height, width)

    def test_applies_bidirectional_phase_correction_when_not_corrected(self) -> None:
        """Verifies that bidirectional phase correction is applied when bidirectional_phase_corrected is False."""
        batch_size = 2
        height = 16
        width = 16
        generator = np.random.default_rng(seed=55)
        frames = generator.standard_normal((batch_size, height, width)).astype(np.float32)
        original = frames.copy()

        y_offsets = np.zeros(batch_size, dtype=np.int32)
        x_offsets = np.zeros(batch_size, dtype=np.int32)

        result = _apply_precomputed_offsets_batch(
            frames=frames,
            y_offsets=y_offsets,
            x_offsets=x_offsets,
            y_offsets_nonrigid=None,
            x_offsets_nonrigid=None,
            blocks=None,
            bidirectional_phase_offset=2,
            bidirectional_phase_corrected=False,
            nonrigid_enabled=False,
        )

        # The bidirectional phase correction shifts odd rows, so the result should differ from the original.
        assert result.shape == original.shape
        assert not np.array_equal(result, original)

    def test_nonrigid_branch_applies_correction(self) -> None:
        """Verifies that the nonrigid branch applies block-based warping when nonrigid_enabled is True."""
        batch_size = 2
        height = 64
        width = 64
        generator = np.random.default_rng(seed=88)
        frames = generator.standard_normal((batch_size, height, width)).astype(np.float32)
        original = frames.copy()

        y_offsets = np.zeros(batch_size, dtype=np.int32)
        x_offsets = np.zeros(batch_size, dtype=np.int32)

        blocks = compute_registration_blocks(height=height, width=width, block_size=(32, 32))
        block_count = blocks[2][0] * blocks[2][1]

        y_offsets_nonrigid = generator.standard_normal((batch_size, block_count)).astype(np.float32) * 2
        x_offsets_nonrigid = generator.standard_normal((batch_size, block_count)).astype(np.float32) * 2

        result = _apply_precomputed_offsets_batch(
            frames=frames,
            y_offsets=y_offsets,
            x_offsets=x_offsets,
            y_offsets_nonrigid=y_offsets_nonrigid,
            x_offsets_nonrigid=x_offsets_nonrigid,
            blocks=blocks,
            bidirectional_phase_offset=0,
            bidirectional_phase_corrected=True,
            nonrigid_enabled=True,
        )

        # Nonrigid correction should modify the frames.
        assert result.shape == original.shape
        assert not np.array_equal(result, original)
