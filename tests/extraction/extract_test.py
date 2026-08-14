"""Contains tests for the extraction kernels and the _update_roi_extraction_statistics function provided by the
extract module.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from cindra.extraction.extract import (
    _extract_cell_fluorescence,
    _extract_neuropil_fluorescence,
    _update_roi_extraction_statistics,
)
from cindra.dataclasses.single_recording_data import ROIMask, ROIStatistics


def _make_roi_statistics(count: int) -> list[ROIStatistics]:
    """Creates a list of minimal ROIStatistics instances backed by single-pixel ROIMask data."""
    roi_list: list[ROIStatistics] = []
    for index in range(count):
        mask = ROIMask(
            y_pixels=np.array([index], dtype=np.int32),
            x_pixels=np.array([index], dtype=np.int32),
            pixel_weights=np.array([1.0], dtype=np.float32),
            centroid=(index, index),
            frame_width=64,
        )
        roi_list.append(ROIStatistics(mask=mask))
    return roi_list


class TestExtractCellFluorescence:
    """Tests _extract_cell_fluorescence."""

    def test_ragged_masks_with_non_uniform_weights(self) -> None:
        """Verifies the weighted gather against a hand-written reference over ragged masks and uneven weights."""
        generator = np.random.default_rng(seed=11)
        frame_count = 7
        pixel_count = 40
        data = generator.standard_normal((frame_count, pixel_count)).astype(np.float32)

        # Three ROIs holding 3, 7, and 5 mask pixels. The sizes differ, the pixel indices are scattered rather than
        # contiguous, and no two ROIs share a weight, so reading the wrong offset or rebasing the weight index onto
        # the mask start changes the result rather than cancelling out.
        masks = (
            np.array([3, 17, 28], dtype=np.int32),
            np.array([0, 1, 2, 5, 9, 31, 39], dtype=np.int32),
            np.array([12, 14, 20, 33, 34], dtype=np.int32),
        )
        weights = (
            np.array([0.2, 0.5, 0.3], dtype=np.float32),
            np.array([0.05, 0.1, 0.15, 0.2, 0.25, 0.15, 0.1], dtype=np.float32),
            np.array([0.4, 0.05, 0.35, 0.1, 0.1], dtype=np.float32),
        )
        flat_roi_masks = np.concatenate(masks).astype(np.int32)
        flat_lambda_weights = np.concatenate(weights).astype(np.float32)
        mask_offsets = np.array([0, 3, 10, 15], dtype=np.int32)

        output_prototype = np.zeros((3, frame_count), dtype=np.float32)
        result = _extract_cell_fluorescence(
            output_prototype=output_prototype,
            data=data,
            flat_roi_masks=flat_roi_masks,
            flat_lambda_weights=flat_lambda_weights,
            mask_offsets=mask_offsets,
        )

        # Independently accumulates the same weighted sums with an explicit Python loop over each ROI's own mask.
        expected = np.zeros((3, frame_count), dtype=np.float32)
        for cell_index in range(3):
            for frame_index in range(frame_count):
                accumulator = np.float32(0.0)
                for pixel, weight in zip(masks[cell_index], weights[cell_index], strict=True):
                    accumulator += data[frame_index, pixel] * weight
                expected[cell_index, frame_index] = accumulator

        np.testing.assert_allclose(result, expected, rtol=1e-6, atol=1e-7)
        # The kernel fills and returns the caller's buffer rather than allocating a new one.
        assert result is output_prototype

    def test_empty_mask_yields_zero_trace(self) -> None:
        """Verifies that an ROI whose mask holds no pixels reports a zero fluorescence trace."""
        data = np.full((4, 10), 5.0, dtype=np.float32)
        # The second ROI spans an empty offset range, while the first and third carry one pixel each.
        flat_roi_masks = np.array([2, 7], dtype=np.int32)
        flat_lambda_weights = np.array([1.0, 1.0], dtype=np.float32)
        mask_offsets = np.array([0, 1, 1, 2], dtype=np.int32)

        result = _extract_cell_fluorescence(
            output_prototype=np.empty((3, 4), dtype=np.float32),
            data=data,
            flat_roi_masks=flat_roi_masks,
            flat_lambda_weights=flat_lambda_weights,
            mask_offsets=mask_offsets,
        )

        np.testing.assert_array_equal(result[0], np.full(4, 5.0, dtype=np.float32))
        np.testing.assert_array_equal(result[1], np.zeros(4, dtype=np.float32))
        np.testing.assert_array_equal(result[2], np.full(4, 5.0, dtype=np.float32))


class TestExtractNeuropilFluorescence:
    """Tests _extract_neuropil_fluorescence."""

    def test_ragged_masks_average_against_reference(self) -> None:
        """Verifies the neuropil average against a hand-written reference over ragged masks."""
        generator = np.random.default_rng(seed=23)
        frame_count = 6
        pixel_count = 50
        data = generator.standard_normal((frame_count, pixel_count)).astype(np.float32)

        # Three neuropil masks of 4, 9, and 6 pixels. The sizes are deliberately not powers of two and differ per
        # ROI, so dropping the per-ROI offset or reusing one pixel count for every ROI changes the averages.
        masks = (
            np.array([1, 4, 8, 11], dtype=np.int32),
            np.array([13, 15, 16, 19, 22, 27, 31, 36, 41], dtype=np.int32),
            np.array([2, 6, 24, 33, 45, 49], dtype=np.int32),
        )
        flat_neuropil_masks = np.concatenate(masks).astype(np.int32)
        mask_offsets = np.array([0, 4, 13, 19], dtype=np.int32)
        neuropil_pixel_count = np.array([4, 9, 6], dtype=np.int32)

        output_prototype = np.zeros((3, frame_count), dtype=np.float32)
        result = _extract_neuropil_fluorescence(
            output_prototype=output_prototype,
            data=data,
            flat_neuropil_masks=flat_neuropil_masks,
            mask_offsets=mask_offsets,
            neuropil_pixel_count=neuropil_pixel_count,
        )

        expected = np.zeros((3, frame_count), dtype=np.float32)
        for cell_index in range(3):
            for frame_index in range(frame_count):
                expected[cell_index, frame_index] = np.mean(data[frame_index, masks[cell_index]])

        # The kernel accumulates sequentially and multiplies by a reciprocal, while np.mean sums pairwise and
        # divides, so the two agree to about 1.5e-05 relative at these mask sizes.
        np.testing.assert_allclose(result, expected, rtol=1e-4)
        assert result is output_prototype

    def test_empty_neuropil_mask_yields_zero_rather_than_nan(self) -> None:
        """Verifies that an ROI whose neuropil mask holds no pixels reports zeros instead of NaN."""
        data = np.full((5, 12), 3.0, dtype=np.float32)
        flat_neuropil_masks = np.array([0, 1, 2], dtype=np.int32)
        mask_offsets = np.array([0, 0, 3], dtype=np.int32)
        neuropil_pixel_count = np.array([0, 3], dtype=np.int32)

        result = _extract_neuropil_fluorescence(
            output_prototype=np.empty((2, 5), dtype=np.float32),
            data=data,
            flat_neuropil_masks=flat_neuropil_masks,
            mask_offsets=mask_offsets,
            neuropil_pixel_count=neuropil_pixel_count,
        )

        np.testing.assert_array_equal(result[0], np.zeros(5, dtype=np.float32))
        np.testing.assert_array_equal(result[1], np.full(5, 3.0, dtype=np.float32))


class TestUpdateRoiExtractionStatistics:
    """Tests _update_roi_extraction_statistics."""

    def test_skewness_set_on_all_rois(self) -> None:
        """Verifies that skewness is set on every ROIStatistics instance."""
        roi_count = 4
        frame_count = 200
        generator = np.random.default_rng(seed=42)
        roi_statistics = _make_roi_statistics(count=roi_count)
        cell_fluorescence = generator.standard_normal((roi_count, frame_count)).astype(np.float32) + 100.0
        neuropil_fluorescence = generator.standard_normal((roi_count, frame_count)).astype(np.float32) + 80.0

        _update_roi_extraction_statistics(
            roi_statistics=roi_statistics,
            cell_fluorescence=cell_fluorescence,
            neuropil_fluorescence=neuropil_fluorescence,
            neuropil_coefficient=0.7,
        )

        for roi in roi_statistics:
            assert roi.skewness is not None
            assert isinstance(roi.skewness, float)
            assert np.isfinite(roi.skewness)

    def test_zero_neuropil_gives_plain_skewness(self) -> None:
        """Verifies that zero neuropil fluorescence produces skewness equal to the plain cell trace skewness."""
        roi_count = 3
        frame_count = 300
        generator = np.random.default_rng(seed=42)
        roi_statistics = _make_roi_statistics(count=roi_count)
        cell_fluorescence = generator.standard_normal((roi_count, frame_count)).astype(np.float32) + 50.0
        neuropil_fluorescence = np.zeros((roi_count, frame_count), dtype=np.float32)

        _update_roi_extraction_statistics(
            roi_statistics=roi_statistics,
            cell_fluorescence=cell_fluorescence,
            neuropil_fluorescence=neuropil_fluorescence,
            neuropil_coefficient=0.7,
        )

        # Equals the plain cell skewness because corrected equals cell when neuropil is zero.
        expected_skewness = np.asarray(stats.skew(a=cell_fluorescence, axis=1))
        for roi, expected in zip(roi_statistics, expected_skewness, strict=True):
            assert roi.skewness is not None
            np.testing.assert_allclose(roi.skewness, float(expected), atol=1e-5)

    def test_nonzero_neuropil_coefficient_changes_skewness(self) -> None:
        """Verifies that a non-zero neuropil coefficient produces different skewness than the raw cell trace."""
        roi_count = 3
        frame_count = 300
        generator = np.random.default_rng(seed=42)
        roi_statistics_corrected = _make_roi_statistics(count=roi_count)
        roi_statistics_uncorrected = _make_roi_statistics(count=roi_count)
        cell_fluorescence = generator.standard_normal((roi_count, frame_count)).astype(np.float32) + 100.0
        neuropil_fluorescence = generator.standard_normal((roi_count, frame_count)).astype(np.float32) + 80.0

        _update_roi_extraction_statistics(
            roi_statistics=roi_statistics_corrected,
            cell_fluorescence=cell_fluorescence,
            neuropil_fluorescence=neuropil_fluorescence,
            neuropil_coefficient=0.7,
        )

        _update_roi_extraction_statistics(
            roi_statistics=roi_statistics_uncorrected,
            cell_fluorescence=cell_fluorescence,
            neuropil_fluorescence=np.zeros_like(neuropil_fluorescence),
            neuropil_coefficient=0.0,
        )

        # Expects at least one ROI to differ in skewness between the corrected and uncorrected runs.
        differences_found = False
        for corrected, uncorrected in zip(roi_statistics_corrected, roi_statistics_uncorrected, strict=True):
            assert corrected.skewness is not None
            assert uncorrected.skewness is not None
            if abs(corrected.skewness - uncorrected.skewness) > 1e-5:
                differences_found = True
        assert differences_found

    def test_updates_in_place(self) -> None:
        """Verifies that skewness values are written in place onto the ROIStatistics instances passed in."""
        roi_count = 2
        frame_count = 100
        generator = np.random.default_rng(seed=42)
        roi_statistics = _make_roi_statistics(count=roi_count)

        for roi in roi_statistics:
            assert roi.skewness is None

        _update_roi_extraction_statistics(
            roi_statistics=roi_statistics,
            cell_fluorescence=generator.standard_normal((roi_count, frame_count)).astype(np.float32) + 50.0,
            neuropil_fluorescence=generator.standard_normal((roi_count, frame_count)).astype(np.float32) + 30.0,
            neuropil_coefficient=0.5,
        )

        for roi in roi_statistics:
            assert roi.skewness is not None

    def test_single_roi(self) -> None:
        """Verifies correct behavior with a single ROI."""
        roi_statistics = _make_roi_statistics(count=1)
        cell_fluorescence = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=np.float32)
        neuropil_fluorescence = np.zeros((1, 8), dtype=np.float32)

        _update_roi_extraction_statistics(
            roi_statistics=roi_statistics,
            cell_fluorescence=cell_fluorescence,
            neuropil_fluorescence=neuropil_fluorescence,
            neuropil_coefficient=0.0,
        )

        expected_skewness = float(stats.skew(a=cell_fluorescence[0]))
        assert roi_statistics[0].skewness is not None
        np.testing.assert_allclose(roi_statistics[0].skewness, expected_skewness, atol=1e-5)

    def test_neuropil_coefficient_scaling(self) -> None:
        """Verifies that the neuropil coefficient correctly scales the neuropil subtraction."""
        roi_count = 2
        frame_count = 200
        generator = np.random.default_rng(seed=42)
        cell_fluorescence = generator.standard_normal((roi_count, frame_count)).astype(np.float32) + 100.0
        neuropil_fluorescence = generator.standard_normal((roi_count, frame_count)).astype(np.float32) + 80.0

        neuropil_coefficient = 0.7
        expected_corrected = cell_fluorescence - np.float32(neuropil_coefficient) * neuropil_fluorescence
        expected_skewness = np.asarray(stats.skew(a=expected_corrected, axis=1))

        roi_statistics = _make_roi_statistics(count=roi_count)
        _update_roi_extraction_statistics(
            roi_statistics=roi_statistics,
            cell_fluorescence=cell_fluorescence,
            neuropil_fluorescence=neuropil_fluorescence,
            neuropil_coefficient=neuropil_coefficient,
        )

        for roi, expected in zip(roi_statistics, expected_skewness, strict=True):
            assert roi.skewness is not None
            np.testing.assert_allclose(roi.skewness, float(expected), atol=1e-5)
