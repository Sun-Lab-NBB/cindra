"""Contains tests for the _create_enhanced_mean_image function provided by the detect module."""

from __future__ import annotations

import numpy as np

from cindra.detection.detect import _create_enhanced_mean_image


class TestCreateEnhancedMeanImage:
    """Tests _create_enhanced_mean_image."""

    def test_output_shape_matches_full_frame(self) -> None:
        """Verifies the output shape matches (frame_height, frame_width)."""
        frame_height = 64
        frame_width = 64
        valid_y_range = (4, 60)
        valid_x_range = (4, 60)
        mean_image = np.random.default_rng(42).standard_normal((56, 56)).astype(np.float32) + 100.0

        result = _create_enhanced_mean_image(
            mean_image=mean_image,
            roi_diameter=12,
            valid_y_range=valid_y_range,
            valid_x_range=valid_x_range,
            frame_height=frame_height,
            frame_width=frame_width,
        )

        assert result.shape == (frame_height, frame_width)
        assert result.dtype == np.float32

    def test_valid_region_values_in_unit_range(self) -> None:
        """Verifies output values inside the valid region are in the [0, 1] range."""
        valid_y_range = (2, 30)
        valid_x_range = (2, 30)
        mean_image = np.random.default_rng(42).standard_normal((28, 28)).astype(np.float32) + 50.0

        result = _create_enhanced_mean_image(
            mean_image=mean_image,
            roi_diameter=10,
            valid_y_range=valid_y_range,
            valid_x_range=valid_x_range,
            frame_height=32,
            frame_width=32,
        )

        valid_region = result[valid_y_range[0] : valid_y_range[1], valid_x_range[0] : valid_x_range[1]]
        assert np.all(valid_region >= 0.0)
        assert np.all(valid_region <= 1.0)

    def test_border_filled_with_minimum_value(self) -> None:
        """Verifies border regions outside the valid range are filled with the minimum value of the enhanced
        interior.
        """
        frame_height = 48
        frame_width = 48
        valid_y_range = (8, 40)
        valid_x_range = (8, 40)
        mean_image = np.random.default_rng(42).standard_normal((32, 32)).astype(np.float32) + 100.0

        result = _create_enhanced_mean_image(
            mean_image=mean_image,
            roi_diameter=12,
            valid_y_range=valid_y_range,
            valid_x_range=valid_x_range,
            frame_height=frame_height,
            frame_width=frame_width,
        )

        valid_region = result[valid_y_range[0] : valid_y_range[1], valid_x_range[0] : valid_x_range[1]]
        minimum_interior_value = valid_region.min()

        top_border = result[: valid_y_range[0], :]
        np.testing.assert_allclose(top_border, minimum_interior_value)

        bottom_border = result[valid_y_range[1] :, :]
        np.testing.assert_allclose(bottom_border, minimum_interior_value)

        left_border = result[valid_y_range[0] : valid_y_range[1], : valid_x_range[0]]
        np.testing.assert_allclose(left_border, minimum_interior_value)

        right_border = result[valid_y_range[0] : valid_y_range[1], valid_x_range[1] :]
        np.testing.assert_allclose(right_border, minimum_interior_value)

    def test_cropped_valid_range(self) -> None:
        """Verifies correct behavior when the valid range crops a significant portion of the frame."""
        frame_height = 64
        frame_width = 64
        valid_y_range = (16, 48)
        valid_x_range = (16, 48)
        mean_image = np.random.default_rng(42).standard_normal((32, 32)).astype(np.float32) + 200.0

        result = _create_enhanced_mean_image(
            mean_image=mean_image,
            roi_diameter=8,
            valid_y_range=valid_y_range,
            valid_x_range=valid_x_range,
            frame_height=frame_height,
            frame_width=frame_width,
        )

        assert result.shape == (frame_height, frame_width)

        # The interior should contain valid enhanced values.
        valid_region = result[valid_y_range[0] : valid_y_range[1], valid_x_range[0] : valid_x_range[1]]
        assert np.all(np.isfinite(valid_region))
        assert np.all(valid_region >= 0.0)
        assert np.all(valid_region <= 1.0)

    def test_uniform_mean_image_produces_uniform_output(self) -> None:
        """Verifies that a uniform mean image produces a constant enhanced output since background subtraction
        removes all variation.
        """
        valid_y_range = (2, 30)
        valid_x_range = (2, 30)
        mean_image = np.ones((28, 28), dtype=np.float32) * 100.0

        result = _create_enhanced_mean_image(
            mean_image=mean_image,
            roi_diameter=10,
            valid_y_range=valid_y_range,
            valid_x_range=valid_x_range,
            frame_height=32,
            frame_width=32,
        )

        valid_region = result[valid_y_range[0] : valid_y_range[1], valid_x_range[0] : valid_x_range[1]]

        # A uniform image has zero variation after background subtraction, so the normalized result should be
        # constant across the valid region.
        assert np.ptp(valid_region) < 1e-5

    def test_zero_roi_diameter_uses_default(self) -> None:
        """Verifies that a zero ROI diameter falls back to the default cell diameter without error."""
        mean_image = np.random.default_rng(42).standard_normal((28, 28)).astype(np.float32) + 50.0

        result = _create_enhanced_mean_image(
            mean_image=mean_image,
            roi_diameter=0,
            valid_y_range=(2, 30),
            valid_x_range=(2, 30),
            frame_height=32,
            frame_width=32,
        )

        assert result.shape == (32, 32)
        assert result.dtype == np.float32
        assert np.all(np.isfinite(result))

    def test_output_is_finite(self) -> None:
        """Verifies that all output values are finite (no NaN or infinity)."""
        rng = np.random.default_rng(99)
        mean_image = rng.standard_normal((56, 56)).astype(np.float32) + 100.0

        result = _create_enhanced_mean_image(
            mean_image=mean_image,
            roi_diameter=15,
            valid_y_range=(4, 60),
            valid_x_range=(4, 60),
            frame_height=64,
            frame_width=64,
        )

        assert np.all(np.isfinite(result))
