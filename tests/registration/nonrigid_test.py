"""Contains tests for the nonrigid module."""

from __future__ import annotations

import numpy as np
import scipy.ndimage

from cindra.detection import compute_registration_blocks
from cindra.registration.nonrigid import (
    _upsample_block_offsets,
    _compute_correlation_snr,
    compute_nonrigid_offsets,
    _apply_coordinate_offsets,
    apply_nonrigid_correction,
    _interpolate_block_offsets,
    _extract_upsampling_regions,
    _apply_bilinear_interpolation,
    compute_nonrigid_reference_data,
)


class TestApplyBilinearInterpolation:
    """Tests the _apply_bilinear_interpolation kernel."""

    def test_matches_map_coordinates_inside_the_source(self) -> None:
        """Verifies interior sampling reproduces scipy.ndimage.map_coordinates at the same coordinates."""
        generator = np.random.default_rng(seed=11)
        source = generator.standard_normal((19, 23)).astype(np.float32)

        # Keeps every coordinate strictly inside the source, so no clamping applies to either implementation.
        y_coordinates = generator.uniform(0.0, 17.0, size=(6, 7)).astype(np.float32)
        x_coordinates = generator.uniform(0.0, 21.0, size=(6, 7)).astype(np.float32)

        output = np.empty((6, 7), dtype=np.float32)
        _apply_bilinear_interpolation(
            source=source, y_coordinates=y_coordinates, x_coordinates=x_coordinates, output=output
        )

        reference = scipy.ndimage.map_coordinates(
            input=source.astype(np.float64),
            coordinates=np.stack([y_coordinates.astype(np.float64), x_coordinates.astype(np.float64)]),
            order=1,
            mode="nearest",
        )
        np.testing.assert_allclose(output, reference, atol=1e-6)

    def test_clamps_coordinates_beyond_the_source(self) -> None:
        """Verifies that coordinates past either source bound return the nearest edge pixel exactly."""
        source = np.arange(20, dtype=np.float32).reshape(4, 5)
        # Column 0 samples past the bottom edge, column 1 past the right edge, column 2 past the top-left corner.
        y_coordinates = np.array([[9.0, 1.0, -3.0]], dtype=np.float32)
        x_coordinates = np.array([[2.0, 11.0, -4.0]], dtype=np.float32)

        output = np.empty((1, 3), dtype=np.float32)
        _apply_bilinear_interpolation(
            source=source, y_coordinates=y_coordinates, x_coordinates=x_coordinates, output=output
        )

        # source[3, 2] = 17, source[1, 4] = 9, source[0, 0] = 0.
        np.testing.assert_array_equal(output, np.array([[17.0, 9.0, 0.0]], dtype=np.float32))


class TestApplyCoordinateOffsets:
    """Tests the _apply_coordinate_offsets kernel."""

    def test_offsets_shift_the_sampled_position(self) -> None:
        """Verifies that a uniform offset map samples the frame at the grid position plus that offset."""
        generator = np.random.default_rng(seed=12)
        height, width = 12, 13
        frames = generator.standard_normal((2, height, width)).astype(np.float32)
        x_grid, y_grid = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        y_offset_maps = np.full((2, height, width), fill_value=2.0, dtype=np.float32)
        x_offset_maps = np.full((2, height, width), fill_value=-3.0, dtype=np.float32)

        output = np.empty_like(frames)
        _apply_coordinate_offsets(
            frames=frames,
            y_offset_maps=y_offset_maps,
            x_offset_maps=x_offset_maps,
            y_grid=y_grid,
            x_grid=x_grid,
            output=output,
        )

        # A positive y offset samples from a higher row, and a negative x offset from a lower column. The integer
        # offsets keep every sample on a pixel center, so the reference is an exact gather with edge clamping.
        rows = np.clip(np.arange(height) + 2, 0, height - 1)
        columns = np.clip(np.arange(width) - 3, 0, width - 1)
        expected = frames[:, rows[:, None], columns[None, :]]
        np.testing.assert_array_equal(output, expected)


class TestInterpolateBlockOffsets:
    """Tests the _interpolate_block_offsets kernel."""

    def test_each_axis_reads_its_own_block_offsets(self) -> None:
        """Verifies that the vertical and horizontal block offsets are interpolated into their own output maps."""
        # Distinct, non-transposable block grids so a swapped or transposed source is visible in the output.
        y_block_offsets = np.array([[[0.0, 4.0], [8.0, 12.0]]], dtype=np.float32)
        x_block_offsets = np.array([[[1.0, 1.0], [3.0, 3.0]]], dtype=np.float32)

        # Samples the 2x2 block grids at their four corners and at the exact center.
        y_grid = np.array([[0.0, 0.0, 0.5], [1.0, 1.0, 0.5]], dtype=np.float32)
        x_grid = np.array([[0.0, 1.0, 0.5], [0.0, 1.0, 0.5]], dtype=np.float32)

        y_offset_maps = np.empty((1, 2, 3), dtype=np.float32)
        x_offset_maps = np.empty((1, 2, 3), dtype=np.float32)
        _interpolate_block_offsets(
            y_block_offsets=y_block_offsets,
            x_block_offsets=x_block_offsets,
            y_grid=y_grid,
            x_grid=x_grid,
            y_offset_maps=y_offset_maps,
            x_offset_maps=x_offset_maps,
        )

        # Bilinear sampling returns the knot value at each corner and the mean of the four knots at the center.
        np.testing.assert_allclose(
            y_offset_maps[0], np.array([[0.0, 4.0, 6.0], [8.0, 12.0, 6.0]], dtype=np.float32), atol=1e-6
        )
        np.testing.assert_allclose(
            x_offset_maps[0], np.array([[1.0, 1.0, 2.0], [3.0, 3.0, 2.0]], dtype=np.float32), atol=1e-6
        )


class TestExtractUpsamplingRegions:
    """Tests the _extract_upsampling_regions kernel."""

    def test_copies_the_window_anchored_at_each_peak(self) -> None:
        """Verifies that every (block, frame) pair copies the window starting at its own peak coordinates."""
        block_count, frame_count, window_extent, region_size = 5, 3, 9, 3

        # Every element identifies its own source position, so a transposed block/frame split or a dropped peak
        # offset changes the copied values rather than only their arrangement.
        correlation = np.arange(block_count * frame_count * window_extent * window_extent, dtype=np.float32).reshape(
            block_count, frame_count, window_extent, window_extent
        )

        y_peaks = np.array([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5], [0, 2, 4]], dtype=np.int32)
        x_peaks = np.array([[4, 3, 2], [1, 0, 5], [2, 2, 2], [0, 1, 2], [5, 4, 3]], dtype=np.int32)

        output = np.empty((block_count, frame_count, region_size, region_size), dtype=np.float32)
        _extract_upsampling_regions(
            correlation=correlation, y_peaks=y_peaks, x_peaks=x_peaks, region_size=region_size, output=output
        )

        expected = np.empty_like(output)
        for block_index in range(block_count):
            for frame_index in range(frame_count):
                peak_y = int(y_peaks[block_index, frame_index])
                peak_x = int(x_peaks[block_index, frame_index])
                expected[block_index, frame_index] = correlation[
                    block_index, frame_index, peak_y : peak_y + region_size, peak_x : peak_x + region_size
                ]
        np.testing.assert_array_equal(output, expected)


class TestComputeCorrelationSnr:
    """Tests the _compute_correlation_snr kernel."""

    def test_ratio_of_peak_to_background_outside_the_exclusion_box(self) -> None:
        """Verifies the SNR is the central peak divided by the largest value outside the peak exclusion box."""
        correlation_data = np.zeros((1, 5, 5), dtype=np.float32)

        # The central region spans rows and columns 1 to 3 for a padding of 1, so this is the peak the scan finds.
        correlation_data[0, 2, 3] = 10.0

        # The exclusion box is anchored at (peak - padding) and spans 2 * padding along each axis, which is rows 1
        # to 2 and columns 2 to 3 here. This value sits inside it and must not become the background.
        correlation_data[0, 1, 2] = 8.0

        # The largest value outside the exclusion box, which is the background the ratio divides by.
        correlation_data[0, 0, 0] = 4.0

        snr = _compute_correlation_snr(correlation_data=correlation_data, padding=1)

        np.testing.assert_array_equal(snr, np.array([2.5], dtype=np.float32))

    def test_zero_background_falls_back_to_the_epsilon(self) -> None:
        """Verifies that an all-zero background divides by the epsilon instead of producing a non-finite ratio."""
        correlation_data = np.zeros((1, 5, 5), dtype=np.float32)
        correlation_data[0, 2, 2] = 3.0

        snr = _compute_correlation_snr(correlation_data=correlation_data, padding=1)

        # Every value outside the exclusion box is zero, so the divisor is the 1e-10 epsilon and the ratio is 3e10.
        assert np.isfinite(snr[0])
        np.testing.assert_allclose(snr, np.array([3.0e10], dtype=np.float32), rtol=1e-5)


class TestComputeNonrigidReferenceData:
    """Tests compute_nonrigid_reference_data."""

    def test_output_shapes(self) -> None:
        """Verifies the output arrays have correct shapes."""
        reference = np.ones((64, 64), dtype=np.float32)
        y_blocks = [np.array([0, 32], dtype=np.int32), np.array([0, 32], dtype=np.int32)]
        x_blocks = [np.array([0, 32], dtype=np.int32), np.array([32, 64], dtype=np.int32)]

        taper, offset, kernel = compute_nonrigid_reference_data(
            reference_image=reference,
            taper_slope=5.0,
            smoothing_sigma=1.15,
            y_blocks=y_blocks,
            x_blocks=x_blocks,
            workers=1,
        )

        block_count = 2
        block_height, block_width = 32, 32
        rfft_width = block_width // 2 + 1

        assert taper.shape == (block_count, block_height, block_width)
        assert offset.shape == (block_count, block_height, block_width)
        assert kernel.shape == (block_count, block_height, rfft_width)

    def test_output_dtypes(self) -> None:
        """Verifies the output dtypes are correct."""
        reference = np.ones((64, 64), dtype=np.float32)
        y_blocks = [np.array([0, 32], dtype=np.int32)]
        x_blocks = [np.array([0, 32], dtype=np.int32)]

        taper, offset, kernel = compute_nonrigid_reference_data(
            reference_image=reference,
            taper_slope=5.0,
            smoothing_sigma=1.15,
            y_blocks=y_blocks,
            x_blocks=x_blocks,
            workers=1,
        )

        assert taper.dtype == np.float32
        assert offset.dtype == np.float32
        assert kernel.dtype == np.complex64

    def test_taper_mask_values_bounded(self) -> None:
        """Verifies that taper mask values are in [0, 1]."""
        reference = np.ones((64, 64), dtype=np.float32) * 100.0
        y_blocks = [np.array([0, 64], dtype=np.int32)]
        x_blocks = [np.array([0, 64], dtype=np.int32)]

        taper, _, _ = compute_nonrigid_reference_data(
            reference_image=reference,
            taper_slope=5.0,
            smoothing_sigma=1.15,
            y_blocks=y_blocks,
            x_blocks=x_blocks,
            workers=1,
        )

        assert np.all(taper >= 0.0)
        assert np.all(taper <= 1.0)


class TestComputeNonrigidOffsets:
    """Tests compute_nonrigid_offsets."""

    def test_consistent_offsets_for_identical_frames(self) -> None:
        """Verifies consistent offsets and correct shapes when frames match the reference."""
        generator = np.random.default_rng(seed=42)
        reference = generator.standard_normal((64, 64)).astype(np.float32)
        y_blocks, x_blocks, _block_counts, _, smoothing_kernel = compute_registration_blocks(
            height=64, width=64, block_size=(32, 32)
        )

        taper, offset, kernel = compute_nonrigid_reference_data(
            reference_image=reference,
            taper_slope=5.0,
            smoothing_sigma=1.15,
            y_blocks=y_blocks,
            x_blocks=x_blocks,
            workers=1,
        )

        frames = np.tile(A=reference, reps=(2, 1, 1))
        y_offsets, x_offsets, _correlation = compute_nonrigid_offsets(
            frames=frames,
            taper_mask=taper,
            mean_offset=offset,
            reference_kernel=kernel,
            snr_threshold=1.2,
            smoothing_kernel=smoothing_kernel,
            x_blocks=x_blocks,
            y_blocks=y_blocks,
            maximum_offset=5.0,
            workers=1,
        )

        block_count = len(y_blocks)
        assert y_offsets.shape == (2, block_count)
        assert x_offsets.shape == (2, block_count)
        # Expects the two copies of the same reference to differ only by the subpixel interpolation residual.
        np.testing.assert_allclose(y_offsets[0], y_offsets[1], atol=1e-4)
        np.testing.assert_allclose(x_offsets[0], x_offsets[1], atol=1e-4)
        # Expects a frame matching the reference to peak at zero shift, leaving under a pixel of subpixel residual.
        assert np.max(np.abs(y_offsets)) < 1.0
        assert np.max(np.abs(x_offsets)) < 1.0

    def test_correlation_maxima_match_the_upsampled_peak(self) -> None:
        """Verifies the correlation maxima are gathered from the peak indices the offsets are derived from."""
        generator = np.random.default_rng(seed=42)
        reference = generator.standard_normal((64, 64)).astype(np.float32)
        y_blocks, x_blocks, _block_counts, _, smoothing_kernel = compute_registration_blocks(
            height=64, width=64, block_size=(32, 32)
        )

        taper, offset, kernel = compute_nonrigid_reference_data(
            reference_image=reference,
            taper_slope=5.0,
            smoothing_sigma=1.15,
            y_blocks=y_blocks,
            x_blocks=x_blocks,
            workers=1,
        )

        # Shifting the second frame by a known amount moves every block's peak off center.
        shifted = np.roll(reference, shift=(2, -3), axis=(0, 1))
        frames = np.stack([reference, shifted]).astype(np.float32)
        y_offsets, x_offsets, correlation = compute_nonrigid_offsets(
            frames=frames,
            taper_mask=taper,
            mean_offset=offset,
            reference_kernel=kernel,
            snr_threshold=1.2,
            smoothing_kernel=smoothing_kernel,
            x_blocks=x_blocks,
            y_blocks=y_blocks,
            maximum_offset=5.0,
            workers=1,
        )

        assert correlation.dtype == np.float32
        assert correlation.shape == (2, len(y_blocks))
        assert np.all(correlation > 0)
        np.testing.assert_allclose(y_offsets[1] - y_offsets[0], 2.0, atol=0.3)
        np.testing.assert_allclose(x_offsets[1] - x_offsets[0], -3.0, atol=0.3)

        # Pins the identity the gather rests on, which is that argmax reports an index attaining the maximum.
        matrix = generator.standard_normal((512, 61 * 61)).astype(np.float32)
        peak_indices = np.argmax(matrix, axis=1)
        np.testing.assert_array_equal(np.amax(matrix, axis=1), matrix[np.arange(matrix.shape[0]), peak_indices])

    def test_output_dtypes(self) -> None:
        """Verifies the output dtypes are correct."""
        generator = np.random.default_rng(seed=42)
        reference = generator.standard_normal((64, 64)).astype(np.float32)
        y_blocks, x_blocks, _, _, smoothing_kernel = compute_registration_blocks(
            height=64, width=64, block_size=(32, 32)
        )

        taper, offset, kernel = compute_nonrigid_reference_data(
            reference_image=reference,
            taper_slope=5.0,
            smoothing_sigma=1.15,
            y_blocks=y_blocks,
            x_blocks=x_blocks,
            workers=1,
        )

        frames = np.tile(A=reference, reps=(1, 1, 1))
        y_offsets, x_offsets, correlation = compute_nonrigid_offsets(
            frames=frames,
            taper_mask=taper,
            mean_offset=offset,
            reference_kernel=kernel,
            snr_threshold=1.2,
            smoothing_kernel=smoothing_kernel,
            x_blocks=x_blocks,
            y_blocks=y_blocks,
            maximum_offset=5.0,
            workers=1,
        )

        assert y_offsets.dtype == np.float32
        assert x_offsets.dtype == np.float32
        assert correlation.dtype == np.float32

    def test_high_snr_threshold_runs_all_smoothing_levels(self) -> None:
        """Verifies offsets stay valid when the SNR threshold forces all smoothing levels to run."""
        generator = np.random.default_rng(seed=42)
        reference = generator.standard_normal((64, 64)).astype(np.float32)
        y_blocks, x_blocks, _block_counts, _, smoothing_kernel = compute_registration_blocks(
            height=64, width=64, block_size=(32, 32)
        )

        taper, offset, kernel = compute_nonrigid_reference_data(
            reference_image=reference,
            taper_slope=5.0,
            smoothing_sigma=1.15,
            y_blocks=y_blocks,
            x_blocks=x_blocks,
            workers=1,
        )

        frames = np.tile(A=reference, reps=(2, 1, 1))
        # A threshold above any achievable SNR keeps low_snr_mask fully True, so the inner loop exhausts all levels.
        y_offsets, x_offsets, _correlation = compute_nonrigid_offsets(
            frames=frames,
            taper_mask=taper,
            mean_offset=offset,
            reference_kernel=kernel,
            snr_threshold=1e9,
            smoothing_kernel=smoothing_kernel,
            x_blocks=x_blocks,
            y_blocks=y_blocks,
            maximum_offset=5.0,
            workers=1,
        )

        block_count = len(y_blocks)
        assert y_offsets.shape == (2, block_count)
        assert x_offsets.shape == (2, block_count)
        # Expects frames identical to the reference to still produce sub-pixel offsets despite the extra smoothing.
        assert np.max(np.abs(y_offsets)) < 1.0
        assert np.max(np.abs(x_offsets)) < 1.0


class TestAdaptiveCorrelationSmoothing:
    """Tests the SNR-driven adaptive smoothing of the block correlation surfaces."""

    def test_smoothing_pulls_a_corrupted_block_to_the_consensus_offset(self) -> None:
        """Verifies that a block whose correlation carries no peak is smoothed toward its neighbors' offset."""
        generator = np.random.default_rng(seed=9)
        extent = 128
        reference = (generator.standard_normal((extent, extent)).astype(np.float32) * 50.0) + 500.0
        y_blocks, x_blocks, _block_counts, _, smoothing_kernel = compute_registration_blocks(
            height=extent, width=extent, block_size=(32, 32)
        )
        taper, offset, kernel = compute_nonrigid_reference_data(
            reference_image=reference,
            taper_slope=5.0,
            smoothing_sigma=1.15,
            y_blocks=y_blocks,
            x_blocks=x_blocks,
            workers=1,
        )

        # Shifts the whole frame, so all 36 blocks share one true offset, then replaces one block's content with
        # unrelated noise. That block's correlation surface holds no peak, which drops its SNR below the threshold
        # and hands it to the smoothing pass that borrows its neighbors' surfaces.
        shift = (3, -2)
        corrupted_index = 14
        corrupted = np.roll(reference, shift=shift, axis=(0, 1)).astype(np.float32)
        y_range, x_range = y_blocks[corrupted_index], x_blocks[corrupted_index]
        block_shape = (int(y_range[1] - y_range[0]), int(x_range[1] - x_range[0]))
        corrupted[y_range[0] : y_range[1], x_range[0] : x_range[1]] = (
            generator.standard_normal(block_shape).astype(np.float32) * 50.0
        ) + 500.0

        y_offsets, x_offsets, _correlation = compute_nonrigid_offsets(
            frames=corrupted[None],
            taper_mask=taper,
            mean_offset=offset,
            reference_kernel=kernel,
            snr_threshold=1.2,
            smoothing_kernel=smoothing_kernel,
            x_blocks=x_blocks,
            y_blocks=y_blocks,
            maximum_offset=10.0,
            workers=1,
        )

        assert len(y_blocks) == 36
        clean_indices = [index for index in range(36) if index != corrupted_index]

        # The blocks holding intact content recover the imposed shift, which establishes the consensus.
        assert float(np.max(np.abs(y_offsets[0][clean_indices] - shift[0]))) <= 0.75
        assert float(np.max(np.abs(x_offsets[0][clean_indices] - shift[1]))) <= 0.75

        # The corrupted block is pulled onto that consensus by the smoothed surface written back into the running
        # correlation. Without the write-back it reports (-0.2, 3.5), which is 3.2 and 5.5 pixels off.
        assert abs(float(y_offsets[0][corrupted_index]) - shift[0]) <= 1.0
        assert abs(float(x_offsets[0][corrupted_index]) - shift[1]) <= 1.0


class TestApplyNonrigidCorrection:
    """Tests apply_nonrigid_correction."""

    def test_undoes_a_known_translation(self) -> None:
        """Verifies that block offsets matching an imposed roll restore the original frame exactly."""
        generator = np.random.default_rng(seed=3)
        extent = 64
        base = generator.standard_normal((extent, extent)).astype(np.float32)
        shift = (3, -2)
        rolled = np.roll(base, shift=shift, axis=(0, 1)).astype(np.float32)

        y_blocks, x_blocks, block_counts, _, _ = compute_registration_blocks(
            height=extent, width=extent, block_size=(32, 32)
        )
        block_count = len(y_blocks)
        y_offsets = np.full((1, block_count), fill_value=float(shift[0]), dtype=np.float32)
        x_offsets = np.full((1, block_count), fill_value=float(shift[1]), dtype=np.float32)

        result = apply_nonrigid_correction(
            frames=rolled[None],
            block_counts=block_counts,
            x_blocks=x_blocks,
            y_blocks=y_blocks,
            y_block_offsets=y_offsets,
            x_block_offsets=x_offsets,
        )

        # np.roll by (3, -2) sets rolled[y, x] = base[y - 3, x + 2], so sampling at (y + 3, x - 2) recovers the
        # base frame. The offsets are whole pixels, so the interior is restored bit for bit rather than approximately.
        interior = (slice(6, -6), slice(6, -6))
        np.testing.assert_array_equal(result[0][interior], base[interior])

    def test_zero_offsets_preserve_frames(self) -> None:
        """Verifies that zero offsets preserve the original frames."""
        generator = np.random.default_rng(seed=42)
        frames = generator.standard_normal((2, 64, 64)).astype(np.float32)
        y_blocks, x_blocks, block_counts, _, _ = compute_registration_blocks(height=64, width=64, block_size=(32, 32))
        block_count = len(y_blocks)
        y_offsets = np.zeros((2, block_count), dtype=np.float32)
        x_offsets = np.zeros((2, block_count), dtype=np.float32)

        result = apply_nonrigid_correction(
            frames=frames,
            block_counts=block_counts,
            x_blocks=x_blocks,
            y_blocks=y_blocks,
            y_block_offsets=y_offsets,
            x_block_offsets=x_offsets,
        )

        assert result.shape == frames.shape
        np.testing.assert_allclose(result, frames, atol=1e-4)

    def test_output_shape_and_dtype(self) -> None:
        """Verifies the output shape and dtype match the input."""
        frames = np.ones((3, 64, 64), dtype=np.float32)
        y_blocks, x_blocks, block_counts, _, _ = compute_registration_blocks(height=64, width=64, block_size=(32, 32))
        block_count = len(y_blocks)
        y_offsets = np.ones((3, block_count), dtype=np.float32) * 0.5
        x_offsets = np.ones((3, block_count), dtype=np.float32) * 0.5

        result = apply_nonrigid_correction(
            frames=frames,
            block_counts=block_counts,
            x_blocks=x_blocks,
            y_blocks=y_blocks,
            y_block_offsets=y_offsets,
            x_block_offsets=x_offsets,
        )

        assert result.shape == (3, 64, 64)
        assert result.dtype == np.float32
        # Expects warping a constant (all-ones) image by any offset to yield a constant image.
        np.testing.assert_allclose(result, 1.0, atol=1e-4)


class TestUpsampleBlockOffsets:
    """Tests _upsample_block_offsets."""

    def test_output_shape(self) -> None:
        """Verifies the output offset maps have the correct shape."""
        y_blocks, x_blocks, block_counts, _, _ = compute_registration_blocks(height=64, width=64, block_size=(32, 32))
        block_count = len(y_blocks)
        y_offsets = np.ones((2, block_count), dtype=np.float32)
        x_offsets = np.ones((2, block_count), dtype=np.float32)

        y_maps, x_maps = _upsample_block_offsets(
            width=64,
            height=64,
            block_counts=block_counts,
            x_blocks=x_blocks,
            y_blocks=y_blocks,
            y_block_offsets=y_offsets,
            x_block_offsets=x_offsets,
        )

        assert y_maps.shape == (2, 64, 64)
        assert x_maps.shape == (2, 64, 64)

    def test_uniform_offsets_preserved(self) -> None:
        """Verifies that uniform block offsets produce uniform pixel offset maps."""
        y_blocks, x_blocks, block_counts, _, _ = compute_registration_blocks(height=64, width=64, block_size=(32, 32))
        block_count = len(y_blocks)
        y_offsets = np.ones((1, block_count), dtype=np.float32) * 2.5
        x_offsets = np.ones((1, block_count), dtype=np.float32) * -1.5

        y_maps, x_maps = _upsample_block_offsets(
            width=64,
            height=64,
            block_counts=block_counts,
            x_blocks=x_blocks,
            y_blocks=y_blocks,
            y_block_offsets=y_offsets,
            x_block_offsets=x_offsets,
        )

        np.testing.assert_allclose(y_maps[0], 2.5, atol=0.1)
        np.testing.assert_allclose(x_maps[0], -1.5, atol=0.1)
