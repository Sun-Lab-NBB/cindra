"""Contains tests for the masks module."""

import numpy as np

from cindra.dataclasses import ROIMask, ROIStatistics
from cindra.extraction.masks import (
    create_masks,
    _create_roi_masks,
    _create_roi_pixels,
    _create_neuropil_masks,
)


def _make_roi(y_pixels, x_pixels, weights, frame_width, radius=5.0, overlap_mask=None):
    """Creates a minimal ROIStatistics instance for testing."""
    mask = ROIMask(
        y_pixels=np.array(y_pixels, dtype=np.int32),
        x_pixels=np.array(x_pixels, dtype=np.int32),
        pixel_weights=np.array(weights, dtype=np.float32),
        centroid=(int(np.median(y_pixels)), int(np.median(x_pixels))),
        frame_width=frame_width,
        radius=radius,
        overlap_mask=overlap_mask,
    )
    return ROIStatistics(mask=mask)


def _make_circular_roi(center_y, center_x, radius, frame_height, frame_width):
    """Creates a circular ROI for testing."""
    y_coords, x_coords = np.mgrid[0:frame_height, 0:frame_width]
    distance = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
    inside = distance <= radius
    y_pixels = y_coords[inside].astype(np.int32)
    x_pixels = x_coords[inside].astype(np.int32)
    weights = np.maximum(0, 1.0 - distance[inside] / radius).astype(np.float32)
    return _make_roi(y_pixels, x_pixels, weights, frame_width, radius=float(radius))


class TestCreateMasks:
    """Tests for create_masks."""

    def test_without_neuropil(self):
        """Verifies that neuropil=False produces None neuropil masks."""
        roi = _make_circular_roi(center_y=25, center_x=25, radius=5, frame_height=50, frame_width=50)
        result = create_masks(roi_statistics=[roi], height=50, width=50, neuropil=False, include_overlap=True)

        assert len(result) == 1
        indices, _weights, neuropil_indices = result[0]
        assert neuropil_indices is None
        assert indices.dtype == np.int32

    def test_with_neuropil(self):
        """Verifies that neuropil=True produces neuropil masks."""
        roi = _make_circular_roi(center_y=25, center_x=25, radius=5, frame_height=50, frame_width=50)
        result = create_masks(
            roi_statistics=[roi],
            height=50,
            width=50,
            neuropil=True,
            include_overlap=True,
            minimum_neuropil_pixels=50,
        )

        assert len(result) == 1
        _indices, _weights, neuropil_indices = result[0]
        assert neuropil_indices is not None
        assert neuropil_indices.dtype == np.int32
        assert neuropil_indices.size > 0

    def test_multiple_rois(self):
        """Verifies that masks are created for each ROI."""
        roi_1 = _make_circular_roi(center_y=15, center_x=15, radius=4, frame_height=50, frame_width=50)
        roi_2 = _make_circular_roi(center_y=35, center_x=35, radius=4, frame_height=50, frame_width=50)
        result = create_masks(roi_statistics=[roi_1, roi_2], height=50, width=50, neuropil=False, include_overlap=True)
        assert len(result) == 2


class TestCreateRoiPixels:
    """Tests for _create_roi_pixels."""

    def test_all_roi_pixels_marked(self):
        """Verifies that ROI pixel positions are marked True in the output mask."""
        roi = _make_roi(
            y_pixels=[5, 5, 6, 6],
            x_pixels=[5, 6, 5, 6],
            weights=[1.0, 1.0, 1.0, 1.0],
            frame_width=20,
        )
        pixel_mask = _create_roi_pixels(roi_statistics=[roi], height=20, width=20, cell_probability_percentile=0)
        assert pixel_mask[5, 5]
        assert pixel_mask[5, 6]
        assert pixel_mask[6, 5]
        assert pixel_mask[6, 6]

    def test_empty_outside_roi(self):
        """Verifies that pixels outside ROIs are False."""
        roi = _make_roi(
            y_pixels=[5, 5],
            x_pixels=[5, 6],
            weights=[1.0, 1.0],
            frame_width=20,
        )
        pixel_mask = _create_roi_pixels(roi_statistics=[roi], height=20, width=20, cell_probability_percentile=0)
        assert not pixel_mask[0, 0]
        assert not pixel_mask[19, 19]

    def test_with_percentile_filtering(self):
        """Verifies that percentile filtering produces a valid boolean mask."""
        roi = _make_circular_roi(center_y=15, center_x=15, radius=5, frame_height=30, frame_width=30)
        pixel_mask = _create_roi_pixels(roi_statistics=[roi], height=30, width=30, cell_probability_percentile=50)
        assert pixel_mask.dtype == np.bool_
        assert pixel_mask.shape == (30, 30)
        # Some pixels should be marked as ROI.
        assert pixel_mask.any()


class TestCreateRoiMasks:
    """Tests for _create_roi_masks."""

    def test_includes_all_pixels_with_overlap(self):
        """Verifies that all ROI pixels are included when include_overlap=True."""
        roi = _make_roi(
            y_pixels=[5, 5, 6, 6],
            x_pixels=[5, 6, 5, 6],
            weights=[1.0, 2.0, 3.0, 4.0],
            frame_width=20,
        )
        masks = _create_roi_masks(roi_statistics=[roi], width=20, include_overlap=True)
        assert len(masks) == 1
        indices, _weights = masks[0]
        assert len(indices) == 4

    def test_excludes_overlap_pixels(self):
        """Verifies that overlapping pixels are excluded when include_overlap=False."""
        overlap = np.array([True, False, False, True], dtype=np.bool_)
        roi = _make_roi(
            y_pixels=[5, 5, 6, 6],
            x_pixels=[5, 6, 5, 6],
            weights=[1.0, 2.0, 3.0, 4.0],
            frame_width=20,
            overlap_mask=overlap,
        )
        masks = _create_roi_masks(roi_statistics=[roi], width=20, include_overlap=False)
        indices, _weights = masks[0]
        # Only 2 non-overlapping pixels should remain.
        assert len(indices) == 2

    def test_weights_sum_to_one(self):
        """Verifies that the normalized weights sum to 1.0."""
        roi = _make_roi(
            y_pixels=[5, 5, 6, 6],
            x_pixels=[5, 6, 5, 6],
            weights=[1.0, 2.0, 3.0, 4.0],
            frame_width=20,
        )
        masks = _create_roi_masks(roi_statistics=[roi], width=20, include_overlap=True)
        _, weights = masks[0]
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-6)

    def test_empty_weights_after_overlap_exclusion(self):
        """Verifies that fully overlapping ROIs produce empty weight arrays."""
        overlap = np.array([True, True], dtype=np.bool_)
        roi = _make_roi(
            y_pixels=[5, 5],
            x_pixels=[5, 6],
            weights=[1.0, 2.0],
            frame_width=20,
            overlap_mask=overlap,
        )
        masks = _create_roi_masks(roi_statistics=[roi], width=20, include_overlap=False)
        indices, weights = masks[0]
        assert len(indices) == 0
        assert len(weights) == 0


class TestCreateNeuropilMasks:
    """Tests for _create_neuropil_masks."""

    def test_neuropil_does_not_overlap_roi(self):
        """Verifies that neuropil masks do not overlap with the ROI pixel region."""
        roi = _make_circular_roi(center_y=25, center_x=25, radius=5, frame_height=50, frame_width=50)
        neuropil_masks = _create_neuropil_masks(
            roi_statistics=[roi],
            height=50,
            width=50,
            inner_neuropil_border_radius=2,
            minimum_neuropil_size=50,
            cell_probability_percentile=0,
        )

        assert len(neuropil_masks) == 1
        neuropil_flat = neuropil_masks[0]

        # Creates a flat index set for the ROI pixels.
        roi_flat = set((roi.mask.y_pixels * 50 + roi.mask.x_pixels).tolist())
        neuropil_set = set(neuropil_flat.tolist())
        assert len(roi_flat & neuropil_set) == 0

    def test_minimum_neuropil_size(self):
        """Verifies that neuropil masks have at least the minimum requested size."""
        roi = _make_circular_roi(center_y=25, center_x=25, radius=3, frame_height=50, frame_width=50)
        min_size = 100
        neuropil_masks = _create_neuropil_masks(
            roi_statistics=[roi],
            height=50,
            width=50,
            inner_neuropil_border_radius=2,
            minimum_neuropil_size=min_size,
            cell_probability_percentile=0,
        )
        assert neuropil_masks[0].size >= min_size

    def test_cached_masks_returned(self):
        """Verifies that cached neuropil masks are returned on second call."""
        roi = _make_circular_roi(center_y=25, center_x=25, radius=3, frame_height=50, frame_width=50)
        # First call computes and caches.
        masks_first = _create_neuropil_masks(
            roi_statistics=[roi],
            height=50,
            width=50,
            inner_neuropil_border_radius=2,
            minimum_neuropil_size=50,
            cell_probability_percentile=0,
        )
        # Second call should return cached.
        masks_second = _create_neuropil_masks(
            roi_statistics=[roi],
            height=50,
            width=50,
            inner_neuropil_border_radius=2,
            minimum_neuropil_size=50,
            cell_probability_percentile=0,
        )
        np.testing.assert_array_equal(masks_first[0], masks_second[0])

    def test_recompute_overrides_cache(self):
        """Verifies that recompute=True forces recomputation even with cached masks."""
        roi = _make_circular_roi(center_y=25, center_x=25, radius=3, frame_height=50, frame_width=50)
        # First call computes and caches.
        _create_neuropil_masks(
            roi_statistics=[roi],
            height=50,
            width=50,
            inner_neuropil_border_radius=2,
            minimum_neuropil_size=50,
            cell_probability_percentile=0,
        )
        # Second call with recompute should succeed without error.
        masks = _create_neuropil_masks(
            roi_statistics=[roi],
            height=50,
            width=50,
            inner_neuropil_border_radius=2,
            minimum_neuropil_size=50,
            cell_probability_percentile=0,
            recompute=True,
        )
        assert masks[0].size > 0
