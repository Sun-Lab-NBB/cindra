"""Contains tests for the tracking module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from cindra.dataclasses import ROIMask
from cindra.detection.tracking import (
    _build_roi_grid,
    _compute_overlap,
    _filter_templates,
    _cluster_rois_in_bin,
    _create_template_roi,
    _compute_condensed_index,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _make_mask(
    y_pixels: Sequence[int],
    x_pixels: Sequence[int],
    weights: Sequence[float],
    frame_width: int,
    centroid: tuple[int, int] | None = None,
    radius: float = 5.0,
    cluster_id: int = 0,
) -> ROIMask:
    """Creates a minimal ROIMask instance for testing."""
    y = np.array(y_pixels, dtype=np.int32)
    x = np.array(x_pixels, dtype=np.int32)
    weight_array = np.array(weights, dtype=np.float32)
    if centroid is None:
        centroid = (int(np.median(y)), int(np.median(x)))
    return ROIMask(
        y_pixels=y,
        x_pixels=x,
        pixel_weights=weight_array,
        centroid=centroid,
        frame_width=frame_width,
        radius=radius,
        cluster_id=cluster_id,
    )


class TestComputeOverlap:
    """Tests for _compute_overlap."""

    def test_no_overlap(self) -> None:
        """Verifies that non-overlapping ROIs have all-False overlap masks."""
        roi1 = _make_mask(y_pixels=[0, 0], x_pixels=[0, 1], weights=[1.0, 1.0], frame_width=20)
        roi2 = _make_mask(y_pixels=[10, 10], x_pixels=[10, 11], weights=[1.0, 1.0], frame_width=20)
        _compute_overlap(rois=[roi1, roi2])
        assert not np.any(roi1.overlap_mask)
        assert not np.any(roi2.overlap_mask)

    def test_full_overlap(self) -> None:
        """Verifies that identical ROIs have all-True overlap masks."""
        roi1 = _make_mask(y_pixels=[5, 5], x_pixels=[5, 6], weights=[1.0, 1.0], frame_width=20)
        roi2 = _make_mask(y_pixels=[5, 5], x_pixels=[5, 6], weights=[1.0, 1.0], frame_width=20)
        _compute_overlap(rois=[roi1, roi2])
        assert np.all(roi1.overlap_mask)
        assert np.all(roi2.overlap_mask)

    def test_partial_overlap(self) -> None:
        """Verifies that partially overlapping ROIs have correct overlap masks."""
        roi1 = _make_mask(y_pixels=[5, 5], x_pixels=[5, 6], weights=[1.0, 1.0], frame_width=20)
        roi2 = _make_mask(y_pixels=[5, 5], x_pixels=[6, 7], weights=[1.0, 1.0], frame_width=20)
        _compute_overlap(rois=[roi1, roi2])
        # roi1 pixel (5,6) overlaps, pixel (5,5) does not.
        assert roi1.overlap_mask[1]  # (5,6) is overlapping
        assert not roi1.overlap_mask[0]  # (5,5) is not

    def test_empty_list(self) -> None:
        """Verifies that an empty list is handled without error."""
        _compute_overlap(rois=[])


class TestComputeCondensedIndex:
    """Tests for _compute_condensed_index."""

    def test_known_values(self) -> None:
        """Verifies correct condensed indices for known square matrix positions."""
        # For a 4x4 matrix, condensed form has 6 elements.
        # Position (1,0) -> condensed index 0
        assert _compute_condensed_index(row_index=1, column_index=0, matrix_size=4) == 0
        # Position (2,0) -> condensed index 1
        assert _compute_condensed_index(row_index=2, column_index=0, matrix_size=4) == 1
        # Position (3,0) -> condensed index 2
        assert _compute_condensed_index(row_index=3, column_index=0, matrix_size=4) == 2
        # Position (2,1) -> condensed index 3
        assert _compute_condensed_index(row_index=2, column_index=1, matrix_size=4) == 3

    def test_symmetric(self) -> None:
        """Verifies that swapped indices produce the same condensed index."""
        idx_a = _compute_condensed_index(row_index=3, column_index=1, matrix_size=5)
        idx_b = _compute_condensed_index(row_index=1, column_index=3, matrix_size=5)
        assert idx_a == idx_b

    def test_diagonal_raises(self) -> None:
        """Verifies that diagonal elements raise ValueError."""
        with pytest.raises(ValueError, match="Unable to convert matrix indices"):
            _compute_condensed_index(row_index=2, column_index=2, matrix_size=5)


class TestBuildRoiGrid:
    """Tests for _build_roi_grid."""

    def test_single_roi(self) -> None:
        """Verifies that a single ROI is placed in the correct grid cell."""
        roi = _make_mask(y_pixels=[25, 25], x_pixels=[30, 31], weights=[1.0, 1.0], frame_width=100, centroid=(25, 30))
        grid = _build_roi_grid(rois=[roi], recordings=[0], grid_size=50)
        assert (0, 0) in grid
        assert len(grid[(0, 0)]) == 1

    def test_multiple_cells(self) -> None:
        """Verifies that ROIs in different spatial locations map to different grid cells."""
        roi1 = _make_mask(y_pixels=[10], x_pixels=[10], weights=[1.0], frame_width=100, centroid=(10, 10))
        roi2 = _make_mask(y_pixels=[60], x_pixels=[60], weights=[1.0], frame_width=100, centroid=(60, 60))
        grid = _build_roi_grid(rois=[roi1, roi2], recordings=[0, 1], grid_size=50)
        assert (0, 0) in grid
        assert (1, 1) in grid
        assert len(grid[(0, 0)]) == 1
        assert len(grid[(1, 1)]) == 1


class TestCreateTemplateRoi:
    """Tests for _create_template_roi."""

    def test_identical_rois(self) -> None:
        """Verifies that identical ROIs produce a template with the same pixels."""
        roi1 = _make_mask(y_pixels=[5, 5, 6, 6], x_pixels=[5, 6, 5, 6], weights=[1.0] * 4, frame_width=20)
        roi2 = _make_mask(y_pixels=[5, 5, 6, 6], x_pixels=[5, 6, 5, 6], weights=[1.0] * 4, frame_width=20)
        template = _create_template_roi(
            cluster_rois=[roi1, roi2], cluster_id=1, image_shape=(20, 20), pixel_prevalence=50
        )
        assert template is not None
        assert template.cluster_id == 1
        assert template.recording_count == 2
        assert len(template.y_pixels) == 4

    def test_no_surviving_pixels(self) -> None:
        """Verifies that None is returned when no pixels meet the prevalence threshold."""
        roi1 = _make_mask(y_pixels=[5], x_pixels=[5], weights=[1.0], frame_width=20)
        roi2 = _make_mask(y_pixels=[10], x_pixels=[10], weights=[1.0], frame_width=20)
        # 100% prevalence means a pixel must appear in ALL ROIs.
        template = _create_template_roi(
            cluster_rois=[roi1, roi2], cluster_id=1, image_shape=(20, 20), pixel_prevalence=100
        )
        # Each pixel appears in only 1/2 = 50% of ROIs, below 100% threshold.
        assert template is None

    def test_weights_averaged(self) -> None:
        """Verifies that template weights are averaged across contributing ROIs."""
        roi1 = _make_mask(y_pixels=[5], x_pixels=[5], weights=[2.0], frame_width=20)
        roi2 = _make_mask(y_pixels=[5], x_pixels=[5], weights=[4.0], frame_width=20)
        template = _create_template_roi(
            cluster_rois=[roi1, roi2], cluster_id=1, image_shape=(20, 20), pixel_prevalence=0
        )
        assert template is not None
        np.testing.assert_allclose(template.pixel_weights[0], 3.0, atol=1e-5)

    def test_radius_averaged(self) -> None:
        """Verifies that the template radius is the mean of input radii."""
        roi1 = _make_mask(y_pixels=[5], x_pixels=[5], weights=[1.0], frame_width=20, radius=4.0)
        roi2 = _make_mask(y_pixels=[5], x_pixels=[5], weights=[1.0], frame_width=20, radius=6.0)
        template = _create_template_roi(
            cluster_rois=[roi1, roi2], cluster_id=1, image_shape=(20, 20), pixel_prevalence=0
        )
        assert template is not None
        np.testing.assert_allclose(template.radius, 5.0, atol=1e-5)


class TestClusterRoisInBin:
    """Tests for _cluster_rois_in_bin."""

    def test_empty_input(self) -> None:
        """Verifies that empty input returns empty output."""
        result = _cluster_rois_in_bin(rois=[], roi_recordings=[], threshold=0.5, maximum_distance=50)
        assert result == []

    def test_identical_rois_from_different_recordings(self) -> None:
        """Verifies that identical ROIs from different recordings are clustered together."""
        roi1 = _make_mask(y_pixels=[5, 5, 6, 6], x_pixels=[5, 6, 5, 6], weights=[1.0] * 4, frame_width=20)
        roi2 = _make_mask(y_pixels=[5, 5, 6, 6], x_pixels=[5, 6, 5, 6], weights=[1.0] * 4, frame_width=20)
        result = _cluster_rois_in_bin(rois=[roi1, roi2], roi_recordings=[0, 1], threshold=0.5, maximum_distance=50)
        assert len(result) > 0
        # The two ROIs should be in the same cluster.
        total_rois = sum(len(rois) for rois, _ in result)
        assert total_rois == 2

    def test_distant_rois_not_clustered(self) -> None:
        """Verifies that spatially distant ROIs are not clustered together."""
        roi1 = _make_mask(y_pixels=[5], x_pixels=[5], weights=[1.0], frame_width=200, centroid=(5, 5))
        roi2 = _make_mask(y_pixels=[100], x_pixels=[100], weights=[1.0], frame_width=200, centroid=(100, 100))
        result = _cluster_rois_in_bin(rois=[roi1, roi2], roi_recordings=[0, 1], threshold=0.5, maximum_distance=10)
        # No candidates within distance threshold.
        assert result == []

    def test_same_recording_not_clustered(self) -> None:
        """Verifies that ROIs from the same recording are not clustered."""
        roi1 = _make_mask(y_pixels=[5, 5], x_pixels=[5, 6], weights=[1.0, 1.0], frame_width=20)
        roi2 = _make_mask(y_pixels=[5, 5], x_pixels=[5, 6], weights=[1.0, 1.0], frame_width=20)
        result = _cluster_rois_in_bin(rois=[roi1, roi2], roi_recordings=[0, 0], threshold=0.5, maximum_distance=50)
        # Both are from recording 0, so no valid cross-recording pairs.
        assert result == []


class TestFilterTemplates:
    """Tests for _filter_templates."""

    def test_keeps_large_masks(self) -> None:
        """Verifies that masks with enough non-overlapping pixels are kept."""
        mask = _make_mask(y_pixels=[5, 5, 6, 6], x_pixels=[5, 6, 5, 6], weights=[1.0] * 4, frame_width=20)
        mask.overlap_mask = np.array([False, False, False, False], dtype=np.bool_)
        result = _filter_templates(template_masks=[mask], minimum_size=2)
        assert len(result) == 1

    def test_removes_small_masks(self) -> None:
        """Verifies that masks with too few non-overlapping pixels are removed."""
        mask = _make_mask(y_pixels=[5, 5, 6, 6], x_pixels=[5, 6, 5, 6], weights=[1.0] * 4, frame_width=20)
        mask.overlap_mask = np.array([True, True, True, False], dtype=np.bool_)
        result = _filter_templates(template_masks=[mask], minimum_size=2)
        assert len(result) == 0

    def test_none_overlap_mask_kept(self) -> None:
        """Verifies that masks without overlap information are always kept."""
        mask = _make_mask(y_pixels=[5], x_pixels=[5], weights=[1.0], frame_width=20)
        result = _filter_templates(template_masks=[mask], minimum_size=100)
        assert len(result) == 1
