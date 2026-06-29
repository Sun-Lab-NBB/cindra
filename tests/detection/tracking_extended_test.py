"""Contains tests for extended tracking module helper functions."""

from __future__ import annotations

import numpy as np

from cindra.dataclasses import ROIMask
from cindra.detection.tracking import _collect_bin_rois


def _make_roi_mask(centroid: tuple[int, int], pixel_count: int = 5, cluster_id: int = 0) -> ROIMask:
    """Creates a minimal ROIMask instance for testing."""
    y_pixels = np.full(pixel_count, fill_value=centroid[0], dtype=np.int32)
    x_pixels = np.full(pixel_count, fill_value=centroid[1], dtype=np.int32)
    return ROIMask(
        y_pixels=y_pixels,
        x_pixels=x_pixels,
        pixel_weights=np.ones(pixel_count, dtype=np.float32),
        centroid=centroid,
        frame_width=200,
        cluster_id=cluster_id,
    )


class TestCollectBinRois:
    """Tests _collect_bin_rois."""

    def test_collects_rois_within_bin(self) -> None:
        """Verifies that ROIs within the bin region are correctly collected."""
        grid_roi_size = 20
        roi_1 = _make_roi_mask(centroid=(30, 30))
        roi_2 = _make_roi_mask(centroid=(50, 50))
        roi_3 = _make_roi_mask(centroid=(35, 35))

        # ROIs that share a grid cell are grouped so the bin collector can scan neighboring cells.
        roi_grid: dict[tuple[int, int], list[tuple[ROIMask, int]]] = {}
        for roi, recording_index in [(roi_1, 0), (roi_2, 1), (roi_3, 2)]:
            grid_key = (roi.centroid[0] // grid_roi_size, roi.centroid[1] // grid_roi_size)
            roi_grid.setdefault(grid_key, []).append((roi, recording_index))

        collected_rois, collected_recordings = _collect_bin_rois(
            roi_grid=roi_grid,
            bin_origin_y=20,
            bin_origin_x=20,
            bin_height=40,
            bin_width=40,
            overlap_margin=0,
            grid_roi_size=grid_roi_size,
        )

        # All three ROIs fall within the bin region (20, 60) x (20, 60) (strict inequality).
        assert len(collected_rois) == 3
        assert len(collected_recordings) == 3

    def test_bin_with_no_nearby_rois_returns_empty(self) -> None:
        """Verifies that a bin with no nearby ROIs returns empty lists."""
        grid_roi_size = 20
        roi_1 = _make_roi_mask(centroid=(10, 10))

        roi_grid: dict[tuple[int, int], list[tuple[ROIMask, int]]] = {
            (10 // grid_roi_size, 10 // grid_roi_size): [(roi_1, 0)],
        }

        collected_rois, collected_recordings = _collect_bin_rois(
            roi_grid=roi_grid,
            bin_origin_y=200,
            bin_origin_x=200,
            bin_height=40,
            bin_width=40,
            overlap_margin=0,
            grid_roi_size=grid_roi_size,
        )

        assert len(collected_rois) == 0
        assert len(collected_recordings) == 0

    def test_overlap_margin_extends_search_region(self) -> None:
        """Verifies that the overlap margin extends the search region beyond the bin boundaries."""
        grid_roi_size = 20
        # Places an ROI just outside the bin boundary but within the overlap margin.
        roi_outside = _make_roi_mask(centroid=(15, 15))

        roi_grid: dict[tuple[int, int], list[tuple[ROIMask, int]]] = {
            (15 // grid_roi_size, 15 // grid_roi_size): [(roi_outside, 0)],
        }

        # Without overlap margin, the bin [20, 60) x [20, 60) should not capture the ROI at (15, 15).
        collected_rois_no_margin, _ = _collect_bin_rois(
            roi_grid=roi_grid,
            bin_origin_y=20,
            bin_origin_x=20,
            bin_height=40,
            bin_width=40,
            overlap_margin=0,
            grid_roi_size=grid_roi_size,
        )
        assert len(collected_rois_no_margin) == 0

        # With overlap margin of 10, the search region becomes [10, 70) x [10, 70), capturing the ROI.
        collected_rois_with_margin, collected_recordings = _collect_bin_rois(
            roi_grid=roi_grid,
            bin_origin_y=20,
            bin_origin_x=20,
            bin_height=40,
            bin_width=40,
            overlap_margin=10,
            grid_roi_size=grid_roi_size,
        )
        assert len(collected_rois_with_margin) == 1
        assert collected_recordings == [0]

    def test_skips_already_clustered_rois(self) -> None:
        """Verifies that ROIs with a non-zero cluster_id are excluded from collection."""
        grid_roi_size = 20
        roi_unclustered = _make_roi_mask(centroid=(30, 30), cluster_id=0)
        roi_clustered = _make_roi_mask(centroid=(45, 45), cluster_id=5)

        # The two ROIs are placed in different cells to avoid overwriting each other in the grid.
        roi_grid: dict[tuple[int, int], list[tuple[ROIMask, int]]] = {}
        for roi, recording_index in [(roi_unclustered, 0), (roi_clustered, 1)]:
            grid_key = (roi.centroid[0] // grid_roi_size, roi.centroid[1] // grid_roi_size)
            roi_grid.setdefault(grid_key, []).append((roi, recording_index))

        collected_rois, collected_recordings = _collect_bin_rois(
            roi_grid=roi_grid,
            bin_origin_y=20,
            bin_origin_x=20,
            bin_height=40,
            bin_width=40,
            overlap_margin=0,
            grid_roi_size=grid_roi_size,
        )

        # Only the unclustered ROI should be collected.
        assert len(collected_rois) == 1
        assert collected_recordings == [0]
