"""Contains tests for the register_recordings module helper functions."""

from __future__ import annotations

import numpy as np

from cindra.dataclasses import ROIMask
from cindra.registration.deformation import Deformation
from cindra.registration.register_recordings import _warp_mask_pixels, _forward_deform_masks, _backward_deform_masks


class TestWarpMaskPixels:
    """Tests the pixel positions and weights an ROI mask carries after a deformation warps it."""

    def test_identity_deformation_preserves_pixels(self) -> None:
        """Verifies that a zero-displacement deformation preserves pixel positions approximately."""
        height = 128
        width = 128
        mask = _make_roi_mask(centroid=(50, 50), radius=5, frame_width=width)

        field_y = np.zeros((height, width), dtype=np.float32)
        field_x = np.zeros((height, width), dtype=np.float32)
        deformation = Deformation(field_y=field_y, field_x=field_x)

        new_y, _new_x, new_weights, new_centroid = _warp_mask_pixels(mask=mask, deformation=deformation)

        assert abs(len(new_y) - len(mask.y_pixels)) <= 2
        assert abs(new_centroid[0] - 50) <= 2
        assert abs(new_centroid[1] - 50) <= 2
        assert new_weights.dtype == np.float32

    def test_translation_deformation_shifts_pixels(self) -> None:
        """Verifies that a uniform translation deformation returns non-empty warped pixels with float32 weights."""
        height = 128
        width = 128
        mask = _make_roi_mask(centroid=(60, 60), radius=4, frame_width=width)

        field_y = np.full((height, width), fill_value=5.0, dtype=np.float32)
        field_x = np.full((height, width), fill_value=3.0, dtype=np.float32)
        deformation = Deformation(field_y=field_y, field_x=field_x)

        new_y, new_x, new_weights, _new_centroid = _warp_mask_pixels(mask=mask, deformation=deformation)

        assert new_y.size
        assert new_x.size
        assert new_weights.dtype == np.float32


class TestForwardDeformMasks:
    """Tests the mask count and derived radius that survive the push into the common deformed space."""

    def test_identity_deformation_preserves_mask_count(self) -> None:
        """Verifies that identity deformation preserves the number of masks and leaves each one with pixels."""
        height = 128
        width = 128
        masks = [
            _make_roi_mask(centroid=(40, 40), radius=4, frame_width=width),
            _make_roi_mask(centroid=(80, 80), radius=4, frame_width=width),
        ]

        field_y = np.zeros((height, width), dtype=np.float32)
        field_x = np.zeros((height, width), dtype=np.float32)
        deformation = Deformation(field_y=field_y, field_x=field_x)

        result = _forward_deform_masks(masks=masks, deformation=deformation, frame_width=width)

        assert len(result) == 2
        for transformed_mask in result:
            assert transformed_mask.y_pixels.size
            assert transformed_mask.x_pixels.size
            assert transformed_mask.frame_width == width

    def test_output_masks_have_valid_radius(self) -> None:
        """Verifies that output ROIMask instances have a computed radius based on pixel count."""
        height = 128
        width = 128
        masks = [_make_roi_mask(centroid=(50, 50), radius=5, frame_width=width)]

        field_y = np.zeros((height, width), dtype=np.float32)
        field_x = np.zeros((height, width), dtype=np.float32)
        deformation = Deformation(field_y=field_y, field_x=field_x)

        result = _forward_deform_masks(masks=masks, deformation=deformation, frame_width=width)

        assert result[0].radius > 0


class TestBackwardDeformMasks:
    """Tests the per-mask ROI statistics the pull back into a recording's own space produces."""

    def test_identity_deformation_returns_roi_statistics(self) -> None:
        """Verifies that identity deformation returns ROIStatistics with computed fields."""
        height = 128
        width = 128
        masks = [
            _make_roi_mask(centroid=(50, 50), radius=5, frame_width=width, cluster_id=1, recording_count=3),
        ]

        field_y = np.zeros((height, width), dtype=np.float32)
        field_x = np.zeros((height, width), dtype=np.float32)
        deformation = Deformation(field_y=field_y, field_x=field_x)

        result = _backward_deform_masks(
            masks=masks,
            deformation=deformation,
            frame_height=height,
            frame_width=width,
            roi_diameter=10,
        )

        assert len(result) == 1
        roi_statistics = result[0]

        assert roi_statistics.mask.y_pixels.size
        assert roi_statistics.mask.frame_width == width
        assert roi_statistics.mask.cluster_id == 1
        assert roi_statistics.mask.recording_count == 3

        assert roi_statistics.pixel_count > 0

        # Tracked ROIs bypass multi-scale detection, so the backward projection zeroes their footprint.
        assert roi_statistics.footprint == 0

    def test_multiple_masks_produce_matching_statistics_count(self) -> None:
        """Verifies that backward deformation of multiple masks returns one ROIStatistics per mask."""
        height = 128
        width = 128
        masks = [
            _make_roi_mask(centroid=(30, 30), radius=4, frame_width=width),
            _make_roi_mask(centroid=(80, 80), radius=4, frame_width=width),
            _make_roi_mask(centroid=(50, 60), radius=3, frame_width=width),
        ]

        field_y = np.zeros((height, width), dtype=np.float32)
        field_x = np.zeros((height, width), dtype=np.float32)
        deformation = Deformation(field_y=field_y, field_x=field_x)

        result = _backward_deform_masks(
            masks=masks,
            deformation=deformation,
            frame_height=height,
            frame_width=width,
            roi_diameter=10,
        )

        assert len(result) == 3
        for roi_statistics in result:
            assert roi_statistics.footprint == 0


def _make_roi_mask(
    centroid: tuple[int, int] = (50, 50),
    radius: int = 5,
    frame_width: int = 128,
    cluster_id: int = 0,
    recording_count: int = 1,
) -> ROIMask:
    """Creates a filled circular ROIMask centered on the given centroid."""
    delta_y, delta_x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    inside_circle = delta_y**2 + delta_x**2 <= radius**2
    y_array = (centroid[0] + delta_y[inside_circle]).astype(np.int32)
    x_array = (centroid[1] + delta_x[inside_circle]).astype(np.int32)
    pixel_weights = np.ones(y_array.size, dtype=np.float32)
    pixel_weights /= np.linalg.norm(pixel_weights)
    return ROIMask(
        y_pixels=y_array,
        x_pixels=x_array,
        pixel_weights=pixel_weights,
        centroid=centroid,
        frame_width=frame_width,
        radius=float(radius),
        cluster_id=cluster_id,
        recording_count=recording_count,
    )
