"""Contains tests for the select module."""

from __future__ import annotations

import numpy as np

from cindra.io.select import _filter_channel_rois
from cindra.dataclasses import ROIMask, ROIStatistics


def _make_roi(
    centroid: tuple[int, int] = (10, 10),
    pixel_count: int = 50,
) -> ROIStatistics:
    """Creates a minimal ROIStatistics instance for testing."""
    y = np.arange(pixel_count, dtype=np.int32) % 10
    x = np.arange(pixel_count, dtype=np.int32) // 10
    mask = ROIMask(
        y_pixels=y,
        x_pixels=x,
        pixel_weights=np.ones(pixel_count, dtype=np.float32),
        centroid=centroid,
        frame_width=100,
    )
    roi = ROIStatistics(mask=mask)
    roi.pixel_count = pixel_count
    return roi


class TestFilterChannelRois:
    """Tests for _filter_channel_rois."""

    def test_all_pass(self) -> None:
        """Verifies that all ROIs pass when no filters are restrictive."""
        rois = [_make_roi() for _ in range(3)]
        classification = np.ones((3, 2), dtype=np.float32)
        result = _filter_channel_rois(
            roi_statistics=rois,
            cell_classification=classification,
            mroi_region_borders=(),
            probability_threshold=0.0,
            maximum_size=10000,
            region_margin=0,
        )
        assert result == (0, 1, 2)

    def test_probability_filter(self) -> None:
        """Verifies that ROIs below the probability threshold are excluded."""
        rois = [_make_roi() for _ in range(3)]
        classification = np.array([[0.9, 1.0], [0.3, 0.0], [0.8, 1.0]], dtype=np.float32)
        result = _filter_channel_rois(
            roi_statistics=rois,
            cell_classification=classification,
            mroi_region_borders=(),
            probability_threshold=0.5,
            maximum_size=10000,
            region_margin=0,
        )
        assert result == (0, 2)

    def test_size_filter(self) -> None:
        """Verifies that ROIs exceeding maximum size are excluded while the boundary value is retained."""
        rois = [_make_roi(pixel_count=10), _make_roi(pixel_count=100), _make_roi(pixel_count=150)]
        classification = np.ones((3, 2), dtype=np.float32)
        result = _filter_channel_rois(
            roi_statistics=rois,
            cell_classification=classification,
            mroi_region_borders=(),
            probability_threshold=0.0,
            maximum_size=100,
            region_margin=0,
        )
        # maximum_size is inclusive: only pixel_count > maximum_size is excluded, so 10 and the boundary 100 pass.
        assert result == (0, 1)

    def test_mroi_border_filter(self) -> None:
        """Verifies that ROIs near MROI region borders are excluded."""
        rois = [
            _make_roi(centroid=(10, 50)),  # Near border at x=50.
            _make_roi(centroid=(10, 100)),  # Far from border.
        ]
        classification = np.ones((2, 2), dtype=np.float32)
        result = _filter_channel_rois(
            roi_statistics=rois,
            cell_classification=classification,
            mroi_region_borders=(50,),
            probability_threshold=0.0,
            maximum_size=10000,
            region_margin=10,
        )
        # ROI at x=50 is exactly at the border (distance=0 < 10), excluded.
        assert 0 not in result
        assert 1 in result

    def test_no_mroi_borders(self) -> None:
        """Verifies that empty MROI borders disable the border filter."""
        rois = [_make_roi(centroid=(10, 0))]
        classification = np.ones((1, 2), dtype=np.float32)
        result = _filter_channel_rois(
            roi_statistics=rois,
            cell_classification=classification,
            mroi_region_borders=(),
            probability_threshold=0.0,
            maximum_size=10000,
            region_margin=100,
        )
        assert result == (0,)

    def test_empty_rois(self) -> None:
        """Verifies that empty input produces empty output."""
        classification = np.empty((0, 2), dtype=np.float32)
        result = _filter_channel_rois(
            roi_statistics=[],
            cell_classification=classification,
            mroi_region_borders=(),
            probability_threshold=0.0,
            maximum_size=10000,
            region_margin=0,
        )
        assert result == ()
