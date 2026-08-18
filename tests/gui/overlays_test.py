"""Contains tests for the ROI overlay rendering helpers used by the GUI viewers."""

from __future__ import annotations

import numpy as np
import pytest

from cindra.gui.overlays import update_correlation_masks
from cindra.gui.constants import ROI_CONFIG, ROIColorMode
from cindra.gui.data_models import ColorArrays, ROIIndexMaps

_FRAME_SIZE: int = 6
"""The synthetic frame height and width used by the overlay fixtures."""


class TestUpdateCorrelationMasks:
    """Tests the activity correlation coloring helper."""

    def test_single_temporal_bin_colors_every_roi_once(self) -> None:
        """Verifies that a single-bin trace yields one correlation per ROI instead of a square matrix."""
        color_arrays = _make_color_arrays(roi_count=4)
        binned_fluorescence = np.array([[1.0], [2.0], [3.0], [4.0]], dtype=np.float32)

        update_correlation_masks(
            color_arrays=color_arrays,
            roi_maps=_make_roi_maps(),
            binned_fluorescence=binned_fluorescence,
            fluorescence_standard_deviation=np.ones(4, dtype=np.float32),
            selected_indices=[1],
            colormap="hsv",
        )

        assert color_arrays.colors[ROIColorMode.CORRELATIONS].shape == (4, 3)
        assert color_arrays.normalized_statistics[ROIColorMode.CORRELATIONS].shape == (4,)
        assert color_arrays.normalized_statistics[ROIColorMode.CORRELATIONS] == pytest.approx(
            [0.0, 0.5, 2.0 / 3.0, 1.0]
        )
        assert color_arrays.colorbar[ROIColorMode.CORRELATIONS] == pytest.approx([1.0, 2.5, 4.0])

    def test_correlated_roi_receives_the_highest_normalized_value(self) -> None:
        """Verifies that the ROI tracking the reference template ranks above the anti-correlated ROI."""
        color_arrays = _make_color_arrays(roi_count=3)
        binned_fluorescence = np.array(
            [[1.0, -1.0, 1.0, -1.0], [2.0, -2.0, 2.0, -2.0], [-1.0, 1.0, -1.0, 1.0]], dtype=np.float32
        )

        update_correlation_masks(
            color_arrays=color_arrays,
            roi_maps=_make_roi_maps(),
            binned_fluorescence=binned_fluorescence,
            fluorescence_standard_deviation=np.array([1.0, 2.0, 1.0], dtype=np.float32),
            selected_indices=[0],
            colormap="hsv",
        )

        normalized = color_arrays.normalized_statistics[ROIColorMode.CORRELATIONS]
        assert normalized.shape == (3,)
        assert normalized[1] == pytest.approx(1.0)
        assert normalized[2] == pytest.approx(0.0)
        assert color_arrays.colorbar[ROIColorMode.CORRELATIONS] == pytest.approx([-1.0, 0.0, 1.0])


def _make_color_arrays(roi_count: int) -> ColorArrays:
    """Builds the zero-initialized color arrays the overlay helpers mutate in place."""
    color_count = len(ROIColorMode)
    return ColorArrays(
        colors=np.zeros((color_count, roi_count, 3), dtype=np.uint8),
        normalized_statistics=np.zeros((color_count, roi_count), dtype=np.float32),
        colorbar=[[0.0, 0.5, 1.0] for _ in range(color_count)],
        rgb=np.zeros((color_count, _FRAME_SIZE, _FRAME_SIZE, 4), dtype=np.uint8),
        random_hues=np.zeros(roi_count, dtype=np.float32),
    )


def _make_roi_maps() -> ROIIndexMaps:
    """Builds ROI index maps that attribute every pixel to the first ROI."""
    return ROIIndexMaps(
        roi_presence=np.zeros((_FRAME_SIZE, _FRAME_SIZE), dtype=bool),
        roi_indices=np.zeros((ROI_CONFIG.overlap_layers, _FRAME_SIZE, _FRAME_SIZE), dtype=np.int32),
    )
