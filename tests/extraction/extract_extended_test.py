"""Contains tests for extended extract module helper functions."""

from __future__ import annotations

import numpy as np

from cindra.dataclasses import ROIMask, ROIStatistics
from cindra.extraction.extract import _create_and_unpack_masks


def _make_circular_roi(
    centroid: tuple[int, int],
    radius: int = 5,
    frame_width: int = 64,
) -> ROIStatistics:
    """Creates an ROIStatistics instance with a circular mask.

    Args:
        centroid: The (y, x) centroid position.
        radius: The radius of the circular mask in pixels.
        frame_width: The width of the frame in pixels.

    Returns:
        An ROIStatistics instance with circular pixel coordinates.
    """
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
    roi = ROIStatistics(mask=mask)
    roi.pixel_count = len(y_pixels)
    return roi


class TestCreateAndUnpackMasks:
    """Tests _create_and_unpack_masks."""

    def test_with_neuropil_extraction(self) -> None:
        """Verifies that mask creation with neuropil extraction produces both cell and neuropil masks."""
        roi_statistics = [
            _make_circular_roi(centroid=(20, 20), radius=4),
            _make_circular_roi(centroid=(40, 40), radius=4),
        ]

        roi_masks, neuropil_masks = _create_and_unpack_masks(
            roi_statistics=roi_statistics,
            frame_height=64,
            frame_width=64,
            extract_neuropil=True,
            allow_overlap=True,
            cell_probability_percentile=50,
            inner_neuropil_border_radius=2,
            minimum_neuropil_pixels=10,
            channel_label="channel 1",
        )

        assert len(roi_masks) == 2
        for indices, weights in roi_masks:
            assert len(indices) > 0
            assert len(weights) > 0
            assert indices.dtype == np.int32
            assert weights.dtype == np.float32

        assert neuropil_masks is not None
        assert len(neuropil_masks) == 2
        for neuropil_indices in neuropil_masks:
            assert len(neuropil_indices) > 0

    def test_without_neuropil_extraction(self) -> None:
        """Verifies that mask creation without neuropil extraction returns None for neuropil masks."""
        roi_statistics = [
            _make_circular_roi(centroid=(30, 30), radius=4),
        ]

        roi_masks, neuropil_masks = _create_and_unpack_masks(
            roi_statistics=roi_statistics,
            frame_height=64,
            frame_width=64,
            extract_neuropil=False,
            allow_overlap=True,
            cell_probability_percentile=50,
            inner_neuropil_border_radius=2,
            minimum_neuropil_pixels=10,
            channel_label="channel 1",
        )

        assert len(roi_masks) == 1
        assert len(roi_masks[0][0]) > 0

        assert neuropil_masks is None

    def test_multiple_rois_produce_matching_mask_count(self) -> None:
        """Verifies that the number of cell masks matches the number of input ROIs."""
        roi_count = 5
        roi_statistics = [_make_circular_roi(centroid=(10 + i * 10, 10 + i * 10), radius=3) for i in range(roi_count)]

        roi_masks, _ = _create_and_unpack_masks(
            roi_statistics=roi_statistics,
            frame_height=64,
            frame_width=64,
            extract_neuropil=False,
            allow_overlap=True,
            cell_probability_percentile=50,
            inner_neuropil_border_radius=2,
            minimum_neuropil_pixels=10,
            channel_label="channel 1",
        )

        assert len(roi_masks) == roi_count
