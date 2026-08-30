"""Contains tests for extended extract module helper functions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cindra.io import BinaryFile
from cindra.dataclasses import ROIMask, ROIStatistics
from cindra.extraction.extract import _create_and_unpack_masks, _extract_fluorescence_traces

if TYPE_CHECKING:
    from pathlib import Path


class TestCreateAndUnpackMasks:
    """Tests the cell and neuropil mask arrays unpacked for extraction, with and without the neuropil pass."""

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
            assert indices.size > 0
            assert weights.size > 0
            assert indices.dtype == np.int32
            assert weights.dtype == np.float32

        assert neuropil_masks is not None
        assert len(neuropil_masks) == 2
        for neuropil_indices in neuropil_masks:
            assert neuropil_indices.size > 0

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
        assert roi_masks[0][0].size > 0

        assert neuropil_masks is None

    def test_multiple_rois_produce_matching_mask_count(self) -> None:
        """Verifies that the number of cell masks matches the number of input ROIs."""
        roi_count = 5
        roi_statistics = [
            _make_circular_roi(centroid=(10 + index * 10, 10 + index * 10), radius=3) for index in range(roi_count)
        ]

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


class TestExtractFluorescenceTraces:
    """Tests the finite zero trace an ROI reports when its neuropil mask holds no pixels."""

    def test_empty_neuropil_mask_yields_zero_trace(self, tmp_path: Path) -> None:
        """Verifies that an ROI whose neuropil mask holds no pixels reports a finite zero neuropil trace."""
        frame_height = frame_width = 8
        frame_count = 4
        pixel_value = 100
        binary_path = tmp_path / "channel_1_data.bin"
        np.full((frame_count, frame_height, frame_width), fill_value=pixel_value, dtype=np.int16).tofile(binary_path)

        # The first ROI carries a populated neuropil mask and the second carries an empty one, so the two arms of the
        # pixel count guard are exercised inside one kernel dispatch.
        roi_masks = (
            (np.array([0, 1], dtype=np.int32), np.array([0.5, 0.5], dtype=np.float32)),
            (np.array([8, 9], dtype=np.int32), np.array([0.5, 0.5], dtype=np.float32)),
        )
        neuropil_masks = (np.array([16, 17, 18], dtype=np.int32), np.array([], dtype=np.int32))

        with BinaryFile(
            height=frame_height, width=frame_width, file_path=binary_path, frame_number=frame_count
        ) as binary:
            fluorescence, neuropil_fluorescence = _extract_fluorescence_traces(
                frames=binary,
                roi_masks=roi_masks,
                neuropil_masks=neuropil_masks,
                batch_size=frame_count,
                channel_label="channel 1",
            )

        assert np.all(np.isfinite(neuropil_fluorescence))
        # The populated mask averages the constant movie, while the empty mask reports zero rather than a NaN.
        np.testing.assert_allclose(neuropil_fluorescence[0], float(pixel_value), rtol=1e-5)
        np.testing.assert_array_equal(neuropil_fluorescence[1], np.zeros(frame_count, dtype=np.float32))
        np.testing.assert_allclose(fluorescence, float(pixel_value), rtol=1e-5)


def _make_circular_roi(
    centroid: tuple[int, int],
    radius: int = 5,
    frame_width: int = 64,
) -> ROIStatistics:
    """Creates an ROIStatistics instance backed by a circular mask with L2-normalized pixel weights."""
    y_coordinates, x_coordinates = np.mgrid[
        centroid[0] - radius : centroid[0] + radius + 1, centroid[1] - radius : centroid[1] + radius + 1
    ]
    inside = (y_coordinates - centroid[0]) ** 2 + (x_coordinates - centroid[1]) ** 2 <= radius**2
    y_array = y_coordinates[inside].astype(np.int32)
    x_array = x_coordinates[inside].astype(np.int32)
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
    roi = ROIStatistics(mask=mask)
    roi.pixel_count = y_array.size
    return roi
