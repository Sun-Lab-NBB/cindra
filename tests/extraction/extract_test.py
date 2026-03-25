"""Contains tests for the _update_roi_extraction_statistics function provided by the extract module."""

from __future__ import annotations

import numpy as np
from scipy import stats

from cindra.extraction.extract import _update_roi_extraction_statistics
from cindra.dataclasses.single_recording_data import ROIMask, ROIStatistics


def _make_roi_statistics(count: int) -> list[ROIStatistics]:
    """Creates a list of minimal ROIStatistics instances for testing.

    Args:
        count: The number of ROIStatistics instances to create.

    Returns:
        A list of ROIStatistics instances with default fields and minimal ROIMask data.
    """
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


class TestUpdateRoiExtractionStatistics:
    """Tests for _update_roi_extraction_statistics."""

    def test_skewness_set_on_all_rois(self) -> None:
        """Verifies that skewness is set on every ROIStatistics instance."""
        roi_count = 4
        frame_count = 200
        rng = np.random.default_rng(42)
        roi_statistics = _make_roi_statistics(count=roi_count)
        cell_fluorescence = rng.standard_normal((roi_count, frame_count)).astype(np.float32) + 100.0
        neuropil_fluorescence = rng.standard_normal((roi_count, frame_count)).astype(np.float32) + 80.0

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
        rng = np.random.default_rng(42)
        roi_statistics = _make_roi_statistics(count=roi_count)
        cell_fluorescence = rng.standard_normal((roi_count, frame_count)).astype(np.float32) + 50.0
        neuropil_fluorescence = np.zeros((roi_count, frame_count), dtype=np.float32)

        _update_roi_extraction_statistics(
            roi_statistics=roi_statistics,
            cell_fluorescence=cell_fluorescence,
            neuropil_fluorescence=neuropil_fluorescence,
            neuropil_coefficient=0.7,
        )

        # With zero neuropil, corrected = cell - 0.7 * 0 = cell, so skewness should match scipy.stats.skew of cell.
        expected_skewness = np.asarray(stats.skew(a=cell_fluorescence, axis=1))
        for roi, expected in zip(roi_statistics, expected_skewness, strict=True):
            assert roi.skewness is not None
            np.testing.assert_allclose(roi.skewness, float(expected), atol=1e-5)

    def test_nonzero_neuropil_coefficient_changes_skewness(self) -> None:
        """Verifies that a non-zero neuropil coefficient produces different skewness than the raw cell trace."""
        roi_count = 3
        frame_count = 300
        rng = np.random.default_rng(42)
        roi_statistics_corrected = _make_roi_statistics(count=roi_count)
        roi_statistics_uncorrected = _make_roi_statistics(count=roi_count)
        cell_fluorescence = rng.standard_normal((roi_count, frame_count)).astype(np.float32) + 100.0
        neuropil_fluorescence = rng.standard_normal((roi_count, frame_count)).astype(np.float32) + 80.0

        # Computes skewness with neuropil correction.
        _update_roi_extraction_statistics(
            roi_statistics=roi_statistics_corrected,
            cell_fluorescence=cell_fluorescence,
            neuropil_fluorescence=neuropil_fluorescence,
            neuropil_coefficient=0.7,
        )

        # Computes skewness without neuropil correction.
        _update_roi_extraction_statistics(
            roi_statistics=roi_statistics_uncorrected,
            cell_fluorescence=cell_fluorescence,
            neuropil_fluorescence=np.zeros_like(neuropil_fluorescence),
            neuropil_coefficient=0.0,
        )

        # At least one ROI should have different skewness values between corrected and uncorrected.
        differences_found = False
        for corrected, uncorrected in zip(roi_statistics_corrected, roi_statistics_uncorrected, strict=True):
            assert corrected.skewness is not None
            assert uncorrected.skewness is not None
            if abs(corrected.skewness - uncorrected.skewness) > 1e-5:
                differences_found = True
        assert differences_found

    def test_updates_in_place(self) -> None:
        """Verifies that skewness values are written to the existing ROIStatistics instances, not copies."""
        roi_count = 2
        frame_count = 100
        rng = np.random.default_rng(42)
        roi_statistics = _make_roi_statistics(count=roi_count)

        # Confirms skewness starts as None.
        for roi in roi_statistics:
            assert roi.skewness is None

        _update_roi_extraction_statistics(
            roi_statistics=roi_statistics,
            cell_fluorescence=rng.standard_normal((roi_count, frame_count)).astype(np.float32) + 50.0,
            neuropil_fluorescence=rng.standard_normal((roi_count, frame_count)).astype(np.float32) + 30.0,
            neuropil_coefficient=0.5,
        )

        # Confirms skewness is now set.
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
        rng = np.random.default_rng(42)
        cell_fluorescence = rng.standard_normal((roi_count, frame_count)).astype(np.float32) + 100.0
        neuropil_fluorescence = rng.standard_normal((roi_count, frame_count)).astype(np.float32) + 80.0

        # Computes expected corrected trace manually.
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
