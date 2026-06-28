"""Contains tests for extended detect module helper functions."""

from __future__ import annotations

import numpy as np

from cindra.dataclasses import ROIMask, ROIStatistics
from cindra.detection.detect import _apply_preclassification


def _make_circular_roi(
    centroid: tuple[int, int],
    radius: int = 5,
    frame_width: int = 64,
) -> ROIStatistics:
    """Creates an ROIStatistics instance with a circular mask for testing."""
    y_pixels = []
    x_pixels = []
    for delta_y in range(-radius, radius + 1):
        for delta_x in range(-radius, radius + 1):
            if delta_y**2 + delta_x**2 <= radius**2:
                y_pixels.append(centroid[0] + delta_y)
                x_pixels.append(centroid[1] + delta_x)
    y_array = np.array(y_pixels, dtype=np.int32)
    x_array = np.array(x_pixels, dtype=np.int32)
    pixel_weights = np.ones(len(y_pixels), dtype=np.float32)
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


class TestApplyPreclassification:
    """Tests _apply_preclassification."""

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

        assert len(result) == 0

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
