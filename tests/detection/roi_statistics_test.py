"""Contains tests for the roi_statistics module."""

import numpy as np
import pytest

from cindra.dataclasses import ROIMask, ROIStatistics
from cindra.detection.roi_statistics import (
    _ROI,
    _EllipseData,
    compute_roi_statistics,
    _compute_distance_kernel,
    estimate_diameter_from_rois,
    compute_median_pixel_position,
)


def _make_mask(y_pixels, x_pixels, weights, frame_width, radius=5.0, centroid=None):
    """Creates a minimal ROIMask for testing."""
    y = np.array(y_pixels, dtype=np.int32)
    x = np.array(x_pixels, dtype=np.int32)
    w = np.array(weights, dtype=np.float32)
    if centroid is None:
        centroid = (int(np.median(y)), int(np.median(x)))
    return ROIMask(
        y_pixels=y,
        x_pixels=x,
        pixel_weights=w,
        centroid=centroid,
        frame_width=frame_width,
        radius=radius,
    )


def _make_circular_roi_stats(center_y, center_x, radius, frame_height, frame_width):
    """Creates a circular ROIStatistics instance."""
    y_coords, x_coords = np.mgrid[0:frame_height, 0:frame_width]
    distance = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
    inside = distance <= radius
    y_pixels = y_coords[inside].astype(np.int32)
    x_pixels = x_coords[inside].astype(np.int32)
    weights = np.maximum(0, 1.0 - distance[inside] / radius).astype(np.float32)
    mask = ROIMask(
        y_pixels=y_pixels,
        x_pixels=x_pixels,
        pixel_weights=weights,
        centroid=(center_y, center_x),
        frame_width=frame_width,
        radius=float(radius),
    )
    return ROIStatistics(mask=mask)


class TestEstimateDiameterFromRois:
    """Tests for estimate_diameter_from_rois."""

    def test_empty_list_returns_default(self):
        """Verifies that an empty ROI list returns the default diameter."""
        result = estimate_diameter_from_rois(rois=[], default_diameter=10)
        assert result == 10

    def test_single_roi(self):
        """Verifies the estimated diameter for a single ROI with known pixel count."""
        mask = _make_mask(
            y_pixels=list(range(10)) * 10,
            x_pixels=[i for i in range(10) for _ in range(10)],
            weights=[1.0] * 100,
            frame_width=20,
        )
        result = estimate_diameter_from_rois(rois=[mask])
        # 100 pixels -> diameter = 2 * sqrt(100 / pi) ≈ 11.28 -> int = 11
        expected = int(2 * np.sqrt(100 / np.pi))
        assert result == expected

    def test_multiple_rois_uses_median(self):
        """Verifies that the median pixel count is used when multiple ROIs are given."""
        masks = []
        for count in [50, 100, 200]:
            y = np.arange(count, dtype=np.int32) % 20
            x = np.arange(count, dtype=np.int32) // 20
            masks.append(
                ROIMask(
                    y_pixels=y,
                    x_pixels=x,
                    pixel_weights=np.ones(count, dtype=np.float32),
                    centroid=(10, 5),
                    frame_width=20,
                )
            )
        result = estimate_diameter_from_rois(rois=masks)
        expected = int(2 * np.sqrt(100 / np.pi))
        assert result == expected

    def test_minimum_diameter_is_one(self):
        """Verifies that the minimum returned diameter is 1."""
        mask = _make_mask(y_pixels=[0], x_pixels=[0], weights=[1.0], frame_width=10)
        result = estimate_diameter_from_rois(rois=[mask])
        assert result >= 1


class TestComputeMedianPixelPosition:
    """Tests for compute_median_pixel_position."""

    def test_single_pixel(self):
        """Verifies that a single pixel returns itself."""
        y = np.array([5], dtype=np.int32)
        x = np.array([10], dtype=np.int32)
        result = compute_median_pixel_position(y_pixels=y, x_pixels=x)
        assert result == (5, 10)

    def test_symmetric_pixels(self):
        """Verifies that the center pixel is returned for a symmetric layout."""
        y = np.array([0, 1, 2], dtype=np.int32)
        x = np.array([0, 1, 2], dtype=np.int32)
        result = compute_median_pixel_position(y_pixels=y, x_pixels=x)
        assert result == (1, 1)

    def test_returns_actual_pixel(self):
        """Verifies that the result is an actual pixel from the input arrays."""
        y = np.array([0, 5, 10], dtype=np.int32)
        x = np.array([0, 3, 8], dtype=np.int32)
        result_y, result_x = compute_median_pixel_position(y_pixels=y, x_pixels=x)
        assert result_y in y
        assert result_x in x


class TestComputeDistanceKernel:
    """Tests for _compute_distance_kernel."""

    def test_shape(self):
        """Verifies the output kernel has the correct shape."""
        kernel = _compute_distance_kernel(radius=5)
        assert kernel.shape == (11, 11)

    def test_center_is_zero(self):
        """Verifies that the center pixel has zero distance."""
        kernel = _compute_distance_kernel(radius=5)
        assert kernel[5, 5] == 0.0

    def test_corner_distance(self):
        """Verifies the corner distance matches the expected Euclidean distance."""
        kernel = _compute_distance_kernel(radius=3)
        expected = np.sqrt(3**2 + 3**2)
        np.testing.assert_allclose(kernel[0, 0], expected, atol=1e-5)

    def test_symmetry(self):
        """Verifies that the distance kernel is symmetric."""
        kernel = _compute_distance_kernel(radius=5)
        np.testing.assert_array_equal(kernel, kernel[::-1, :])
        np.testing.assert_array_equal(kernel, kernel[:, ::-1])


class TestROI:
    """Tests for the _ROI wrapper class."""

    def test_pixel_count(self):
        """Verifies the total pixel count."""
        mask = _make_mask(y_pixels=[5, 5, 6, 6], x_pixels=[5, 6, 5, 6], weights=[1.0] * 4, frame_width=20)
        roi_stats = ROIStatistics(mask=mask)
        roi = _ROI(data=roi_stats, diameter=10)
        assert roi.pixel_count == 4

    def test_soma_mask_all_true_for_small_roi(self):
        """Verifies that a small ROI returns an all-True soma mask."""
        mask = _make_mask(y_pixels=[5, 5], x_pixels=[5, 6], weights=[1.0, 1.0], frame_width=20)
        roi_stats = ROIStatistics(mask=mask)
        roi = _ROI(data=roi_stats, diameter=10)
        assert roi.soma_mask.all()

    def test_soma_mask_cached(self):
        """Verifies that the soma mask is cached after first access."""
        mask = _make_mask(y_pixels=[5, 5, 6, 6], x_pixels=[5, 6, 5, 6], weights=[1.0] * 4, frame_width=20)
        roi_stats = ROIStatistics(mask=mask)
        roi = _ROI(data=roi_stats, diameter=10)
        first = roi.soma_mask
        second = roi.soma_mask
        assert first is second

    def test_no_crop_returns_all_true(self):
        """Verifies that crop=False returns an all-True soma mask."""
        roi_stats = _make_circular_roi_stats(center_y=25, center_x=25, radius=8, frame_height=50, frame_width=50)
        roi = _ROI(data=roi_stats, diameter=10, crop=False)
        assert roi.soma_mask.all()

    def test_compactness_for_compact_roi(self):
        """Verifies that a compact circular ROI has a compactness near 1.0."""
        roi_stats = _make_circular_roi_stats(center_y=25, center_x=25, radius=5, frame_height=50, frame_width=50)
        roi = _ROI(data=roi_stats, diameter=10)
        # Compactness is max(1.0, ...) so it must be >= 1.0.
        assert roi.compactness >= 1.0

    def test_mean_radius_positive(self):
        """Verifies that mean_radius is positive for a non-trivial ROI."""
        roi_stats = _make_circular_roi_stats(center_y=25, center_x=25, radius=5, frame_height=50, frame_width=50)
        roi = _ROI(data=roi_stats, diameter=10)
        assert roi.mean_radius > 0.0

    def test_baseline_mean_radius_cached(self):
        """Verifies that the baseline cache is populated after first access."""
        roi_stats = _make_circular_roi_stats(center_y=25, center_x=25, radius=5, frame_height=50, frame_width=50)
        roi = _ROI(data=roi_stats, diameter=10)
        _ = roi.baseline_mean_radius
        assert 10 in _ROI._baseline_cache

    def test_mismatched_arrays_raises(self):
        """Verifies that mismatched pixel array shapes raise TypeError."""
        mask = ROIMask(
            y_pixels=np.array([5, 5], dtype=np.int32),
            x_pixels=np.array([5, 6, 7], dtype=np.int32),
            pixel_weights=np.array([1.0, 1.0], dtype=np.float32),
            centroid=(5, 5),
            frame_width=20,
        )
        roi_stats = ROIStatistics(mask=mask)
        with pytest.raises(TypeError):
            _ROI(data=roi_stats, diameter=10)

    def test_solidity_small_roi(self):
        """Verifies that a small ROI returns solidity based on the default area."""
        mask = _make_mask(y_pixels=[5, 5], x_pixels=[5, 6], weights=[1.0, 1.0], frame_width=20)
        roi_stats = ROIStatistics(mask=mask)
        roi = _ROI(data=roi_stats, diameter=10)
        assert roi.solidity == 2 / 10.0

    def test_solidity_large_roi(self):
        """Verifies that a large circular ROI has solidity close to 1."""
        roi_stats = _make_circular_roi_stats(center_y=25, center_x=25, radius=8, frame_height=50, frame_width=50)
        roi = _ROI(data=roi_stats, diameter=16)
        # A disk has solidity ~1.0 (all pixels inside the convex hull).
        assert roi.solidity > 0.8

    def test_fit_ellipse_returns_ellipse_data(self):
        """Verifies that fit_ellipse returns a valid _EllipseData instance."""
        roi_stats = _make_circular_roi_stats(center_y=25, center_x=25, radius=5, frame_height=50, frame_width=50)
        roi = _ROI(data=roi_stats, diameter=10)
        ellipse = roi.fit_ellipse(y_scale=10, x_scale=10)
        assert isinstance(ellipse, _EllipseData)
        assert ellipse.radius > 0
        # A circular ROI should have an aspect ratio close to 1.
        assert 0.5 < ellipse.aspect_ratio < 1.5

    def test_get_overlap_mask(self):
        """Verifies that get_overlap_mask correctly identifies overlapping pixels."""
        mask = _make_mask(y_pixels=[5, 5, 6, 6], x_pixels=[5, 6, 5, 6], weights=[1.0] * 4, frame_width=20)
        roi_stats = ROIStatistics(mask=mask)
        roi = _ROI(data=roi_stats, diameter=10)
        overlap_image = np.ones((20, 20), dtype=np.uint16)
        overlap_image[5, 5] = 2
        result = roi.get_overlap_mask(overlap_count_image=overlap_image)
        assert result[0]  # pixel (5,5) overlaps
        assert not result[1]  # pixel (5,6) does not

    def test_get_overlap_count_image(self):
        """Verifies that the overlap count image has correct counts."""
        mask1 = _make_mask(y_pixels=[5, 5], x_pixels=[5, 6], weights=[1.0, 1.0], frame_width=20)
        mask2 = _make_mask(y_pixels=[5, 6], x_pixels=[6, 6], weights=[1.0, 1.0], frame_width=20)
        roi1 = _ROI(data=ROIStatistics(mask=mask1), diameter=10)
        roi2 = _ROI(data=ROIStatistics(mask=mask2), diameter=10)
        overlap = _ROI.get_overlap_count_image(rois=[roi1, roi2], height=20, width=20)
        assert overlap[5, 6] == 2  # Shared pixel.
        assert overlap[5, 5] == 1  # Only roi1.
        assert overlap[0, 0] == 0  # No ROI.

    def test_remove_overlapping_rois(self):
        """Verifies that ROIs exceeding the overlap threshold are flagged for removal."""
        # Two identical ROIs: all pixels overlap with count 2.
        mask = _make_mask(y_pixels=[5, 5, 6, 6], x_pixels=[5, 6, 5, 6], weights=[1.0] * 4, frame_width=20)
        roi1 = _ROI(data=ROIStatistics(mask=mask), diameter=10)
        roi2 = _ROI(data=ROIStatistics(mask=mask), diameter=10)
        overlap_image = _ROI.get_overlap_count_image(rois=[roi1, roi2], height=20, width=20)
        keep_flags = _ROI.remove_overlapping_rois(
            rois=[roi1, roi2], overlap_image=overlap_image, maximum_overlap_fraction=0.5
        )
        # At least one should be removed since all pixels overlap.
        assert not all(keep_flags)


class TestEllipseData:
    """Tests for _EllipseData properties."""

    def _make_ellipse(self, radii=(5.0, 3.0)):
        """Creates a minimal _EllipseData."""
        return _EllipseData(
            centroid=np.array([10.0, 10.0], dtype=np.float32),
            covariance=np.eye(2, dtype=np.float32),
            radii=radii,
            boundary_points=np.zeros((100, 2), dtype=np.float32),
            y_scale=10,
            x_scale=10,
        )

    def test_area(self):
        """Verifies the ellipse area formula."""
        ellipse = self._make_ellipse(radii=(5.0, 5.0))
        expected = (5.0 * 5.0) ** 0.5 * np.pi
        np.testing.assert_allclose(ellipse.area, expected, atol=1e-4)

    def test_radius_scales_by_mean(self):
        """Verifies that the effective radius is scaled by the mean of y_scale and x_scale."""
        ellipse = self._make_ellipse(radii=(5.0, 3.0))
        expected = 5.0 * np.mean([10, 10])
        np.testing.assert_allclose(ellipse.radius, expected, atol=1e-4)

    def test_aspect_ratio_circular(self):
        """Verifies that equal radii produce an aspect ratio of ~1."""
        ellipse = self._make_ellipse(radii=(5.0, 5.0))
        np.testing.assert_allclose(ellipse.aspect_ratio, 1.0, atol=0.01)

    def test_aspect_ratio_bounded(self):
        """Verifies that the aspect ratio is bounded between 0 and 2."""
        ellipse = self._make_ellipse(radii=(10.0, 0.01))
        assert 0 < ellipse.aspect_ratio <= 2.0


class TestComputeRoiStatistics:
    """Tests for compute_roi_statistics."""

    def test_empty_list_raises(self):
        """Verifies that an empty ROI list raises ValueError."""
        with pytest.raises(ValueError, match="Unable to compute ROI statistics"):
            compute_roi_statistics(rois=[], frame_height=50, frame_width=50)

    def test_updates_compactness_in_place(self):
        """Verifies that compute_roi_statistics sets compactness on each ROI."""
        roi = _make_circular_roi_stats(center_y=25, center_x=25, radius=5, frame_height=50, frame_width=50)
        rois = [roi]
        compute_roi_statistics(rois=rois, frame_height=50, frame_width=50, diameter=10)
        assert roi.compactness > 0

    def test_updates_pixel_count(self):
        """Verifies that compute_roi_statistics sets pixel_count on each ROI."""
        roi = _make_circular_roi_stats(center_y=25, center_x=25, radius=5, frame_height=50, frame_width=50)
        rois = [roi]
        compute_roi_statistics(rois=rois, frame_height=50, frame_width=50, diameter=10)
        assert roi.pixel_count > 0

    def test_lightweight_skips_solidity(self):
        """Verifies that lightweight mode leaves solidity at the default value."""
        roi = _make_circular_roi_stats(center_y=25, center_x=25, radius=5, frame_height=50, frame_width=50)
        rois = [roi]
        compute_roi_statistics(rois=rois, frame_height=50, frame_width=50, diameter=10, lightweight=True)
        assert roi.solidity == 0.0  # Default, not computed.

    def test_full_mode_sets_solidity(self):
        """Verifies that full mode computes solidity."""
        roi = _make_circular_roi_stats(center_y=25, center_x=25, radius=5, frame_height=50, frame_width=50)
        rois = [roi]
        compute_roi_statistics(rois=rois, frame_height=50, frame_width=50, diameter=10)
        assert roi.solidity > 0

    def test_initializes_missing_centroid(self):
        """Verifies that compute_roi_statistics initializes centroids when missing."""
        mask = _make_mask(
            y_pixels=[5, 5, 6, 6],
            x_pixels=[5, 6, 5, 6],
            weights=[1.0] * 4,
            frame_width=20,
            centroid=(0, 0),
        )
        roi = ROIStatistics(mask=mask)
        rois = [roi]
        compute_roi_statistics(rois=rois, frame_height=20, frame_width=20, diameter=10, lightweight=True)
        assert roi.mask.centroid != (0, 0)

    def test_normalized_pixel_count_set(self):
        """Verifies that normalized_pixel_count is set after computation."""
        roi = _make_circular_roi_stats(center_y=25, center_x=25, radius=5, frame_height=50, frame_width=50)
        rois = [roi]
        compute_roi_statistics(rois=rois, frame_height=50, frame_width=50, diameter=10, lightweight=True)
        assert roi.normalized_pixel_count > 0

    def test_overlap_removal(self):
        """Verifies that overlapping ROIs are removed when maximum_overlap_fraction is set."""
        roi1 = _make_circular_roi_stats(center_y=25, center_x=25, radius=5, frame_height=50, frame_width=50)
        roi2 = _make_circular_roi_stats(center_y=25, center_x=25, radius=5, frame_height=50, frame_width=50)
        rois = [roi1, roi2]
        compute_roi_statistics(rois=rois, frame_height=50, frame_width=50, diameter=10, maximum_overlap_fraction=0.5)
        # At least one ROI should be removed due to complete overlap.
        assert len(rois) < 2
