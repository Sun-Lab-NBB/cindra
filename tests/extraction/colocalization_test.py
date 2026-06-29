"""Contains tests for the colocalization module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

from cindra.dataclasses import ROIMask, ROIStatistics
from cindra.extraction.colocalization import (
    _correct_bleedthrough,
    _build_sparse_roi_masks,
    _compute_overlap_matrix,
    compute_spatial_colocalization,
    compute_intensity_colocalization,
)


def _make_roi(
    y_pixels: Sequence[int],
    x_pixels: Sequence[int],
    weights: Sequence[float],
    frame_width: int,
    radius: float = 5.0,
    overlap_mask: NDArray[np.bool_] | None = None,
) -> ROIStatistics:
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


def _make_circular_roi(
    center_y: int,
    center_x: int,
    radius: int,
    frame_height: int,
    frame_width: int,
) -> ROIStatistics:
    """Creates a circular ROI for testing."""
    y_coordinates, x_coordinates = np.mgrid[0:frame_height, 0:frame_width]
    distance = np.sqrt((y_coordinates - center_y) ** 2 + (x_coordinates - center_x) ** 2)
    inside = distance <= radius
    y_pixels = y_coordinates[inside].astype(np.int32)
    x_pixels = x_coordinates[inside].astype(np.int32)
    weights = np.maximum(0, 1.0 - distance[inside] / radius).astype(np.float32)
    return _make_roi(
        y_pixels=y_pixels, x_pixels=x_pixels, weights=weights, frame_width=frame_width, radius=float(radius)
    )


class TestCorrectBleedthrough:
    """Tests _correct_bleedthrough."""

    def test_output_non_negative(self) -> None:
        """Verifies that the corrected image has no negative values."""
        rng = np.random.default_rng(42)
        functional = rng.standard_normal((30, 30)).astype(np.float32) + 100.0
        structural = rng.standard_normal((30, 30)).astype(np.float32) + 50.0
        corrected = _correct_bleedthrough(
            functional_mean_image=functional,
            structural_mean_image=structural,
        )
        assert np.all(corrected >= 0)

    def test_output_shape_and_dtype(self) -> None:
        """Verifies the output shape and dtype match expectations."""
        functional = np.ones((30, 30), dtype=np.float32) * 100.0
        structural = np.ones((30, 30), dtype=np.float32) * 50.0
        corrected = _correct_bleedthrough(
            functional_mean_image=functional,
            structural_mean_image=structural,
        )
        assert corrected.shape == (30, 30)
        assert corrected.dtype == np.float32
        # Uniform functional and structural images regress to a perfectly predicted bleedthrough, so the corrected
        # structural image is zero everywhere.
        np.testing.assert_allclose(corrected, 0.0, atol=1e-4)

    def test_zero_functional_no_correction(self) -> None:
        """Verifies that zero functional image produces no bleedthrough correction."""
        functional = np.zeros((30, 30), dtype=np.float32)
        structural = np.ones((30, 30), dtype=np.float32) * 50.0
        corrected = _correct_bleedthrough(
            functional_mean_image=functional,
            structural_mean_image=structural,
        )
        np.testing.assert_allclose(corrected, 50.0, atol=1e-4)


class TestBuildSparseRoiMasks:
    """Tests _build_sparse_roi_masks."""

    def test_shape(self) -> None:
        """Verifies the sparse matrix has correct shape."""
        roi_1 = _make_roi(y_pixels=[5, 5], x_pixels=[5, 6], weights=[1.0, 1.0], frame_width=20)
        roi_2 = _make_roi(y_pixels=[10, 10], x_pixels=[10, 11], weights=[1.0, 1.0], frame_width=20)
        sparse = _build_sparse_roi_masks(rois=[roi_1, roi_2], frame_height=20, frame_width=20)
        assert sparse.shape == (2, 400)

    def test_correct_pixel_positions(self) -> None:
        """Verifies that the sparse matrix has ones at the correct flat indices."""
        roi = _make_roi(y_pixels=[2, 2], x_pixels=[3, 4], weights=[1.0, 1.0], frame_width=10)
        sparse = _build_sparse_roi_masks(rois=[roi], frame_height=10, frame_width=10)
        dense = sparse.toarray()
        assert dense[0, 2 * 10 + 3] == 1.0
        assert dense[0, 2 * 10 + 4] == 1.0
        assert dense[0, 0] == 0.0

    def test_binary_clipping(self) -> None:
        """Verifies that duplicate pixel coordinates are clipped to binary values."""
        # Creates ROI with duplicate pixels to trigger the clipping path.
        roi = _make_roi(y_pixels=[5, 5, 5], x_pixels=[5, 5, 6], weights=[1.0, 1.0, 1.0], frame_width=20)
        sparse = _build_sparse_roi_masks(rois=[roi], frame_height=20, frame_width=20)
        assert np.all(sparse.data <= 1.0)


class TestComputeOverlapMatrix:
    """Tests _compute_overlap_matrix."""

    def test_identical_rois_full_overlap(self) -> None:
        """Verifies that identical ROIs produce an overlap of 1.0."""
        roi = _make_roi(y_pixels=[5, 5, 6, 6], x_pixels=[5, 6, 5, 6], weights=[1.0] * 4, frame_width=20)
        overlap = _compute_overlap_matrix(
            rois_channel_1=[roi],
            rois_channel_2=[roi],
            frame_height=20,
            frame_width=20,
        )
        np.testing.assert_allclose(overlap[0, 0], 1.0, atol=1e-6)

    def test_non_overlapping_rois(self) -> None:
        """Verifies that non-overlapping ROIs produce zero overlap."""
        roi_1 = _make_roi(y_pixels=[2, 2], x_pixels=[2, 3], weights=[1.0, 1.0], frame_width=20)
        roi_2 = _make_roi(y_pixels=[15, 15], x_pixels=[15, 16], weights=[1.0, 1.0], frame_width=20)
        overlap = _compute_overlap_matrix(
            rois_channel_1=[roi_1],
            rois_channel_2=[roi_2],
            frame_height=20,
            frame_width=20,
        )
        assert overlap[0, 0] == 0.0

    def test_empty_rois(self) -> None:
        """Verifies correct shape for empty ROI lists."""
        overlap = _compute_overlap_matrix(
            rois_channel_1=[],
            rois_channel_2=[],
            frame_height=20,
            frame_width=20,
        )
        assert overlap.shape == (0, 0)


class TestComputeSpatialColocalization:
    """Tests compute_spatial_colocalization."""

    def test_empty_channel_1(self) -> None:
        """Verifies correct handling of empty channel 1."""
        result = compute_spatial_colocalization(
            rois_channel_1=[],
            rois_channel_2=[_make_roi(y_pixels=[5], x_pixels=[5], weights=[1.0], frame_width=20)],
            frame_height=20,
            frame_width=20,
            colocalization_threshold=0.3,
        )
        assert result.shape == (0, 2)

    def test_empty_channel_2(self) -> None:
        """Verifies correct handling of empty channel 2."""
        roi = _make_roi(y_pixels=[5], x_pixels=[5], weights=[1.0], frame_width=20)
        result = compute_spatial_colocalization(
            rois_channel_1=[roi],
            rois_channel_2=[],
            frame_height=20,
            frame_width=20,
            colocalization_threshold=0.3,
        )
        assert result.shape == (1, 2)
        assert result[0, 0] == -1  # No match.
        assert result[0, 1] == 0.0

    def test_identical_rois_match(self) -> None:
        """Verifies that identical ROIs are matched with high overlap."""
        roi = _make_roi(
            y_pixels=[5, 5, 6, 6],
            x_pixels=[5, 6, 5, 6],
            weights=[1.0] * 4,
            frame_width=20,
        )
        result = compute_spatial_colocalization(
            rois_channel_1=[roi],
            rois_channel_2=[roi],
            frame_height=20,
            frame_width=20,
            colocalization_threshold=0.3,
        )
        assert result[0, 0] == 0  # Matched to channel 2 ROI index 0.
        assert result[0, 1] > 0.9

    def test_non_overlapping_rois_no_match(self) -> None:
        """Verifies that non-overlapping ROIs are not matched."""
        roi_1 = _make_roi(y_pixels=[2, 2], x_pixels=[2, 3], weights=[1.0, 1.0], frame_width=20)
        roi_2 = _make_roi(y_pixels=[15, 15], x_pixels=[15, 16], weights=[1.0, 1.0], frame_width=20)
        result = compute_spatial_colocalization(
            rois_channel_1=[roi_1],
            rois_channel_2=[roi_2],
            frame_height=20,
            frame_width=20,
            colocalization_threshold=0.3,
        )
        assert result[0, 0] == -1

    def test_mutual_best_match_enforced(self) -> None:
        """Verifies that mutual best-match constraint is enforced."""
        # Two channel 1 ROIs both closest to one channel 2 ROI—only one should match.
        roi_1a = _make_roi(y_pixels=[5, 5, 6, 6], x_pixels=[5, 6, 5, 6], weights=[1.0] * 4, frame_width=20)
        roi_1b = _make_roi(y_pixels=[5, 5], x_pixels=[5, 6], weights=[1.0, 1.0], frame_width=20)
        roi_2 = _make_roi(y_pixels=[5, 5, 6, 6], x_pixels=[5, 6, 5, 6], weights=[1.0] * 4, frame_width=20)
        result = compute_spatial_colocalization(
            rois_channel_1=[roi_1a, roi_1b],
            rois_channel_2=[roi_2],
            frame_height=20,
            frame_width=20,
            colocalization_threshold=0.3,
        )
        # Only one of the two should be matched (the one with higher mutual overlap).
        matched_count = np.sum(result[:, 0] >= 0)
        assert matched_count <= 1


class TestComputeIntensityColocalization:
    """Tests compute_intensity_colocalization."""

    def test_empty_rois(self) -> None:
        """Verifies correct handling of empty ROI list."""
        functional = np.ones((30, 30), dtype=np.float32) * 100.0
        structural = np.ones((30, 30), dtype=np.float32) * 50.0
        result, corrected = compute_intensity_colocalization(
            rois=[],
            functional_mean_image=functional,
            structural_mean_image=structural,
            frame_height=30,
            frame_width=30,
            colocalization_threshold=0.5,
            allow_overlap=True,
            cell_probability_percentile=50,
            inner_neuropil_border_radius=2,
            minimum_neuropil_pixels=50,
        )
        assert result.shape == (0, 2)
        assert corrected.shape == (30, 30)

    def test_output_shape_and_dtype(self) -> None:
        """Verifies the output shape and dtype for intensity colocalization."""
        roi = _make_circular_roi(center_y=15, center_x=15, radius=4, frame_height=30, frame_width=30)
        functional = np.ones((30, 30), dtype=np.float32) * 100.0
        structural = np.ones((30, 30), dtype=np.float32) * 50.0
        result, corrected = compute_intensity_colocalization(
            rois=[roi],
            functional_mean_image=functional,
            structural_mean_image=structural,
            frame_height=30,
            frame_width=30,
            colocalization_threshold=0.5,
            allow_overlap=True,
            cell_probability_percentile=0,
            inner_neuropil_border_radius=2,
            minimum_neuropil_pixels=50,
        )
        assert result.shape == (1, 2)
        assert result.dtype == np.float32
        assert corrected.shape == (30, 30)
        assert corrected.dtype == np.float32
        # Uniform inputs zero the corrected structural image, so the neuropil intensity is zero and the
        # inside-to-total ratio floors to 1.0.
        np.testing.assert_allclose(corrected, 0.0, atol=1e-4)
        assert result[0, 1] == pytest.approx(1.0, abs=1e-5)

    def test_frame_filling_roi_empty_neuropil(self) -> None:
        """Verifies that a frame-filling ROI yields an empty neuropil region and zero outside intensity."""
        # An ROI covering every pixel leaves no background, so its neuropil mask is empty and the neuropil intensity
        # branch is skipped, leaving the outside intensity at zero.
        y_coordinates, x_coordinates = np.mgrid[0:8, 0:8]
        roi = _make_roi(
            y_pixels=y_coordinates.ravel().tolist(),
            x_pixels=x_coordinates.ravel().tolist(),
            weights=[1.0] * 64,
            frame_width=8,
        )
        functional = np.ones((8, 8), dtype=np.float32) * 10.0
        structural = np.ones((8, 8), dtype=np.float32) * 50.0
        result, _ = compute_intensity_colocalization(
            rois=[roi],
            functional_mean_image=functional,
            structural_mean_image=structural,
            frame_height=8,
            frame_width=8,
            colocalization_threshold=0.5,
            allow_overlap=True,
            cell_probability_percentile=0,
            inner_neuropil_border_radius=2,
            minimum_neuropil_pixels=50,
        )
        # With no neuropil pixels the outside intensity stays zero, so the inside-to-total ratio floors to 1.0.
        assert result[0, 1] == pytest.approx(1.0, abs=1e-5)

    def test_bright_roi_colocalized(self) -> None:
        """Verifies that an ROI bright in the structural channel is detected as colocalized."""
        roi = _make_circular_roi(center_y=20, center_x=20, radius=4, frame_height=40, frame_width=40)
        functional = np.ones((40, 40), dtype=np.float32) * 10.0
        # Structural image has high intensity inside the ROI region.
        structural = np.ones((40, 40), dtype=np.float32) * 10.0
        structural[16:25, 16:25] = 200.0
        result, _ = compute_intensity_colocalization(
            rois=[roi],
            functional_mean_image=functional,
            structural_mean_image=structural,
            frame_height=40,
            frame_width=40,
            colocalization_threshold=0.3,
            allow_overlap=True,
            cell_probability_percentile=0,
            inner_neuropil_border_radius=2,
            minimum_neuropil_pixels=50,
        )
        # Probability (column 1) should be high since ROI region is bright.
        assert result[0, 1] > 0.5
