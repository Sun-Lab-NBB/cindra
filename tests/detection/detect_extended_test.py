"""Contains tests for extended detect module helper functions."""

from __future__ import annotations

import numpy as np

from cindra.dataclasses import ROIMask, ROIStatistics
from cindra.detection.detect import _apply_preclassification
from cindra.classification.classify import classify


class TestApplyPreclassification:
    """Tests the probability threshold that decides which detected ROIs survive the preclassification filter."""

    def test_threshold_zero_keeps_all_rois(self) -> None:
        """Verifies that a threshold of 0.0 keeps all ROIs regardless of classifier output."""
        roi_statistics = [
            _make_circular_roi(centroid=(20, 20), radius=5),
            _make_circular_roi(centroid=(40, 40), radius=5),
            _make_circular_roi(centroid=(30, 30), radius=5),
        ]

        result = _apply_preclassification(
            roi_statistics=roi_statistics,
            frame_height=64,
            frame_width=64,
            preclassification_threshold=0.0,
            crop_to_soma=False,
            custom_classifier_path=None,
            plane_index=0,
            channel_label="channel 1",
            diameter=10,
        )

        assert len(result) == 3

    def test_threshold_one_keeps_none(self) -> None:
        """Verifies that a threshold of 1.0 removes all ROIs since no probability can exceed 1.0."""
        roi_statistics = [
            _make_circular_roi(centroid=(20, 20), radius=5),
            _make_circular_roi(centroid=(40, 40), radius=5),
        ]

        result = _apply_preclassification(
            roi_statistics=roi_statistics,
            frame_height=64,
            frame_width=64,
            preclassification_threshold=1.0,
            crop_to_soma=False,
            custom_classifier_path=None,
            plane_index=0,
            channel_label="channel 1",
            diameter=10,
        )

        assert not result

    def test_threshold_keeps_the_compact_rois_and_drops_the_lines(self) -> None:
        """Verifies that the pass keeps exactly the ROIs whose classifier probability clears the threshold."""
        # The preclassifier scores compactness and normalized pixel count alone. A one-pixel-wide line, whose mean
        # radius is four times the radius its pixel count would occupy if it were compact, therefore scores far
        # below a disk of the same pixel budget. Interleaving the two shapes means a pass that reversed, inverted,
        # or misaligned its keep mask selects a different set rather than the same one.
        compact_roi = _make_circular_roi(centroid=(20, 20), radius=3)
        long_line_roi = _make_line_roi(centroid=(30, 10), length=40)
        small_compact_roi = _make_circular_roi(centroid=(10, 50), radius=2)
        short_line_roi = _make_line_roi(centroid=(60, 2), length=12)
        roi_statistics = [compact_roi, long_line_roi, small_compact_roi, short_line_roi]

        # The threshold sits below the probability of both disks and above the probability of both lines, so the
        # kept set is a strict, non-empty subset in which the two shapes are separated.
        threshold = 0.1
        result = _apply_preclassification(
            roi_statistics=roi_statistics,
            frame_height=64,
            frame_width=64,
            preclassification_threshold=threshold,
            crop_to_soma=False,
            custom_classifier_path=None,
            plane_index=0,
            channel_label="channel 1",
            diameter=10,
        )

        assert [id(roi) for roi in result] == [id(compact_roi), id(small_compact_roi)]

        # The pass computes the statistics on the same objects, so re-running the classifier over them reproduces
        # the scores on which it filtered. The kept set must be exactly the ROIs whose cell probability, which is the
        # second classifier column, exceeds the threshold, in the order the input list holds them.
        classifications = classify(roi_statistics=roi_statistics, custom_classifier_path=None, preclassification=True)
        expected_ids = [
            id(roi)
            for roi, probability in zip(roi_statistics, classifications[:, 1], strict=True)
            if probability > threshold
        ]
        assert [id(roi) for roi in result] == expected_ids

        # The first column carries the binary cell flag rather than the probability, and at this threshold it
        # selects a different set. The assertions above therefore pin the column the filter reads.
        flag_ids = [
            id(roi) for roi, flag in zip(roi_statistics, classifications[:, 0], strict=True) if flag > threshold
        ]
        assert flag_ids != expected_ids

    def test_returns_subset_of_original_rois(self) -> None:
        """Verifies that the result is a subset of the original ROI list."""
        roi_statistics = [
            _make_circular_roi(centroid=(20, 20), radius=5),
            _make_circular_roi(centroid=(40, 40), radius=6),
        ]

        result = _apply_preclassification(
            roi_statistics=roi_statistics,
            frame_height=64,
            frame_width=64,
            preclassification_threshold=0.5,
            crop_to_soma=False,
            custom_classifier_path=None,
            plane_index=0,
            channel_label="channel 1",
            diameter=10,
        )

        # Confirms every returned ROI originates from the original list, compared by identity.
        assert len(result) <= len(roi_statistics)
        original_ids = {id(roi) for roi in roi_statistics}
        for roi in result:
            assert id(roi) in original_ids


def _make_circular_roi(
    centroid: tuple[int, int],
    radius: int = 5,
    frame_width: int = 64,
) -> ROIStatistics:
    """Creates an ROIStatistics instance with a circular mask."""
    row_offsets, column_offsets = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    inside = row_offsets**2 + column_offsets**2 <= radius**2
    y_array = (centroid[0] + row_offsets[inside]).astype(np.int32)
    x_array = (centroid[1] + column_offsets[inside]).astype(np.int32)
    pixel_weights = np.ones(y_array.size, dtype=np.float32)
    pixel_weights /= np.linalg.norm(pixel_weights)
    mask = ROIMask(
        y_pixels=y_array,
        x_pixels=x_array,
        pixel_weights=pixel_weights,
        centroid=centroid,
        frame_width=frame_width,
        radius=float(radius),
    )
    return ROIStatistics(mask=mask)


def _make_line_roi(
    centroid: tuple[int, int],
    length: int,
    frame_width: int = 64,
) -> ROIStatistics:
    """Creates an ROIStatistics instance with a one-pixel-wide horizontal mask."""
    y_array = np.full(length, fill_value=centroid[0], dtype=np.int32)
    x_array = np.arange(centroid[1], centroid[1] + length, dtype=np.int32)
    pixel_weights = np.ones(length, dtype=np.float32)
    pixel_weights /= np.linalg.norm(pixel_weights)
    mask = ROIMask(
        y_pixels=y_array,
        x_pixels=x_array,
        pixel_weights=pixel_weights,
        centroid=(centroid[0], int(np.median(x_array))),
        frame_width=frame_width,
        radius=float(length) / 2.0,
    )
    return ROIStatistics(mask=mask)
