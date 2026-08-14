"""Contains tests for the tracking module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ataraxis_base_utilities import error_format

from cindra.dataclasses import ROIMask
from cindra.detection.tracking import (
    _build_roi_grid,
    _compute_overlap,
    _filter_templates,
    _cluster_rois_in_bin,
    _count_shared_pixels,
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
    y_array = np.array(y_pixels, dtype=np.int32)
    x_array = np.array(x_pixels, dtype=np.int32)
    weight_array = np.array(weights, dtype=np.float32)
    if centroid is None:
        centroid = (int(np.median(y_array)), int(np.median(x_array)))
    return ROIMask(
        y_pixels=y_array,
        x_pixels=x_array,
        pixel_weights=weight_array,
        centroid=centroid,
        frame_width=frame_width,
        radius=radius,
        cluster_id=cluster_id,
    )


def _make_block_mask(first_row: int, first_column: int, height: int, width: int, frame_width: int = 40) -> ROIMask:
    """Creates a solid rectangular ROIMask whose pixel count and overlaps are exactly known."""
    rows, columns = np.mgrid[first_row : first_row + height, first_column : first_column + width]
    return _make_mask(
        y_pixels=rows.ravel().tolist(),
        x_pixels=columns.ravel().tolist(),
        weights=[1.0] * (height * width),
        frame_width=frame_width,
    )


class TestComputeOverlap:
    """Tests _compute_overlap."""

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
        assert roi1.overlap_mask[1]
        assert not roi1.overlap_mask[0]

    def test_empty_list(self) -> None:
        """Verifies that an empty list is handled without error."""
        _compute_overlap(rois=[])


class TestComputeCondensedIndex:
    """Tests _compute_condensed_index."""

    def test_known_values(self) -> None:
        """Verifies correct condensed indices for known square matrix positions."""
        # For a 4x4 matrix, condensed form has 6 elements.
        assert _compute_condensed_index(row_index=1, column_index=0, matrix_size=4) == 0
        assert _compute_condensed_index(row_index=2, column_index=0, matrix_size=4) == 1
        assert _compute_condensed_index(row_index=3, column_index=0, matrix_size=4) == 2
        assert _compute_condensed_index(row_index=2, column_index=1, matrix_size=4) == 3

    def test_symmetric(self) -> None:
        """Verifies that swapped indices produce the same condensed index."""
        index_a = _compute_condensed_index(row_index=3, column_index=1, matrix_size=5)
        index_b = _compute_condensed_index(row_index=1, column_index=3, matrix_size=5)
        assert index_a == index_b

    def test_diagonal_raises(self) -> None:
        """Verifies that diagonal elements raise ValueError."""
        expected_message = "Unable to convert matrix indices to condensed form. Diagonal elements are not allowed."
        with pytest.raises(ValueError, match=error_format(expected_message)):
            _compute_condensed_index(row_index=2, column_index=2, matrix_size=5)


class TestBuildRoiGrid:
    """Tests _build_roi_grid."""

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
    """Tests _create_template_roi."""

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


class TestCountSharedPixels:
    """Tests _count_shared_pixels."""

    def test_matches_the_set_intersection_over_unique_inputs(self) -> None:
        """Verifies the merge count equals the set intersection size for ascending, duplicate-free inputs."""
        generator = np.random.default_rng(seed=7)

        # The two arrays are drawn at deliberately unequal sizes from an index range narrow enough that they must
        # share pixels. A kernel that advanced only one cursor, or that returned a constant, would agree with the
        # oracle on sparse disjoint draws, so every case below is checked to carry a non-empty intersection.
        for first_size, second_size in ((7, 31), (31, 7), (12, 12), (5, 40), (40, 5)):
            first = np.unique(generator.integers(0, 24, size=first_size)).astype(np.int32)
            second = np.unique(generator.integers(0, 24, size=second_size)).astype(np.int32)
            expected = np.intersect1d(first, second, assume_unique=True).shape[0]
            assert expected > 0
            assert _count_shared_pixels(first_pixels=first, second_pixels=second) == expected

    def test_mismatched_cursors_reach_the_exact_shared_count(self) -> None:
        """Verifies that lists matching only after a long mismatch run report the exact shared count."""
        # The two lists interleave without ever matching, so the merge must advance each cursor in turn and end at
        # zero rather than stalling on the first pair or counting a mismatch.
        assert (
            _count_shared_pixels(
                first_pixels=np.arange(0, 200, 2, dtype=np.int32),
                second_pixels=np.arange(1, 200, 2, dtype=np.int32),
            )
            == 0
        )

        # The single shared index sits at the end of the longer list, so reaching it takes 99 advances of the
        # cursor standing on the smaller value. Advancing the other cursor on a mismatch instead runs the short
        # list out immediately and reports 0, which the zero case above cannot distinguish.
        ascending_range = np.arange(100, dtype=np.int32)
        final_index = np.array([99], dtype=np.int32)
        assert _count_shared_pixels(first_pixels=ascending_range, second_pixels=final_index) == 1
        assert _count_shared_pixels(first_pixels=final_index, second_pixels=ascending_range) == 1

    def test_empty_input_shares_nothing(self) -> None:
        """Verifies that an empty pixel list yields a zero count rather than raising."""
        populated = np.arange(10, dtype=np.int32)
        empty = np.array([], dtype=np.int32)
        assert _count_shared_pixels(first_pixels=populated, second_pixels=empty) == 0
        assert _count_shared_pixels(first_pixels=empty, second_pixels=populated) == 0

    def test_identical_lists_share_every_pixel(self) -> None:
        """Verifies that two identical pixel lists report the full count."""
        pixels = np.arange(0, 200, 3, dtype=np.int32)
        assert _count_shared_pixels(first_pixels=pixels, second_pixels=pixels) == pixels.size


class TestClusterRoisInBin:
    """Tests _cluster_rois_in_bin."""

    def test_empty_input(self) -> None:
        """Verifies that empty input returns empty output."""
        result = _cluster_rois_in_bin(rois=[], roi_recordings=[], threshold=0.5, maximum_distance=50)
        assert result == []

    def test_identical_rois_from_different_recordings(self) -> None:
        """Verifies that identical ROIs from different recordings are clustered together."""
        roi1 = _make_mask(y_pixels=[5, 5, 6, 6], x_pixels=[5, 6, 5, 6], weights=[1.0] * 4, frame_width=20)
        roi2 = _make_mask(y_pixels=[5, 5, 6, 6], x_pixels=[5, 6, 5, 6], weights=[1.0] * 4, frame_width=20)
        result = _cluster_rois_in_bin(rois=[roi1, roi2], roi_recordings=[0, 1], threshold=0.5, maximum_distance=50)
        assert result
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

    @pytest.mark.parametrize(("threshold", "expected_cluster_sizes"), [(0.7, [2]), (0.6, [1, 1])])
    def test_threshold_brackets_a_two_thirds_jaccard_distance(
        self, threshold: float, expected_cluster_sizes: list[int]
    ) -> None:
        """Verifies that the clustering threshold decides a pair sitting at a Jaccard distance of two thirds."""
        # Two 6x6 blocks offset by three rows share their three middle rows: the intersection is 3 * 6 = 18 pixels
        # and the union is 36 + 36 - 18 = 54, so the Jaccard distance is exactly 1 - 18 / 54 = 2 / 3. A threshold
        # just above that value must merge the pair and one just below must keep it apart. The overlap is
        # deliberately not one half of the union: at exactly one half the similarity and the distance coincide, so
        # a kernel reporting the similarity in place of the distance would decide both thresholds identically.
        first = _make_block_mask(first_row=10, first_column=10, height=6, width=6)
        second = _make_block_mask(first_row=13, first_column=10, height=6, width=6)

        result = _cluster_rois_in_bin(
            rois=[first, second], roi_recordings=[0, 1], threshold=threshold, maximum_distance=50
        )

        assert sorted(len(cluster_rois) for cluster_rois, _ in result) == expected_cluster_sizes

    @pytest.mark.parametrize(("threshold", "expected_cluster_sizes"), [(0.8, [2]), (0.7, [1, 1])])
    def test_contained_roi_distance_uses_the_union_denominator(
        self, threshold: float, expected_cluster_sizes: list[int]
    ) -> None:
        """Verifies that a fully contained ROI is scored against the union rather than the smaller pixel count."""
        # The 3x3 block sits wholly inside the 6x6 block, so the intersection is 9 and the union is 36 + 9 - 9 = 36.
        # The Jaccard distance is therefore 1 - 9 / 36 = 0.75, and a threshold of 0.7 separates the pair. Dividing
        # by the smaller count instead would yield a distance of 0.0 and merge the pair at every threshold.
        container = _make_block_mask(first_row=10, first_column=10, height=6, width=6)
        contained = _make_block_mask(first_row=11, first_column=11, height=3, width=3)

        result = _cluster_rois_in_bin(
            rois=[container, contained], roi_recordings=[0, 1], threshold=threshold, maximum_distance=50
        )

        assert sorted(len(cluster_rois) for cluster_rois, _ in result) == expected_cluster_sizes

    def test_same_recording_not_clustered(self) -> None:
        """Verifies that ROIs from the same recording are not clustered."""
        roi1 = _make_mask(y_pixels=[5, 5], x_pixels=[5, 6], weights=[1.0, 1.0], frame_width=20)
        roi2 = _make_mask(y_pixels=[5, 5], x_pixels=[5, 6], weights=[1.0, 1.0], frame_width=20)
        result = _cluster_rois_in_bin(rois=[roi1, roi2], roi_recordings=[0, 0], threshold=0.5, maximum_distance=50)
        # Both are from recording 0, so no valid cross-recording pairs.
        assert result == []


class TestFilterTemplates:
    """Tests _filter_templates."""

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
        assert not result

    def test_none_overlap_mask_kept(self) -> None:
        """Verifies that masks without overlap information are always kept."""
        mask = _make_mask(y_pixels=[5], x_pixels=[5], weights=[1.0], frame_width=20)
        result = _filter_templates(template_masks=[mask], minimum_size=100)
        assert len(result) == 1
