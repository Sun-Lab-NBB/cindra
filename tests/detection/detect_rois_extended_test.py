"""Contains tests for extended detect_rois module helper functions."""

from __future__ import annotations

import numpy as np
from numpy.linalg import norm

from cindra.detection.detect_rois import _find_best_scale, _extend_iteratively, _check_split_components


class TestCheckSplitComponents:
    """Tests _check_split_components."""

    def test_two_component_signal_yields_high_variance_ratio(self) -> None:
        """Verifies that data with two distinct spatial components produces a variance ratio above 1."""
        rng = np.random.default_rng(seed=42)
        pixel_count = 20
        frame_count = 100

        # Creates two distinct temporal signals assigned to different pixel subsets.
        component_1_temporal = np.abs(rng.standard_normal(frame_count).astype(np.float32)) * 5
        component_2_temporal = np.abs(rng.standard_normal(frame_count).astype(np.float32)) * 5

        pixel_frames = np.zeros((frame_count, pixel_count), dtype=np.float32)
        pixel_frames[:, :10] = component_1_temporal[:, np.newaxis]
        pixel_frames[:, 10:] = component_2_temporal[:, np.newaxis]

        weights = np.ones(pixel_count, dtype=np.float32)
        weights /= norm(weights)

        variance_ratio, (spatial_weights, temporal_projections, active_mask) = _check_split_components(
            pixel_frames=pixel_frames.copy(),
            weights=weights,
            intensity_threshold=0.1,
        )

        assert variance_ratio > 1.0
        assert spatial_weights.shape == (pixel_count,)
        assert active_mask.dtype == np.bool_
        assert temporal_projections.ndim == 1

    def test_single_component_signal_yields_ratio_near_one(self) -> None:
        """Verifies that data with a single spatial component produces a variance ratio near 1."""
        rng = np.random.default_rng(seed=99)
        pixel_count = 15
        frame_count = 80

        # Creates a single-component signal where all pixels share the same temporal trace.
        temporal_signal = np.abs(rng.standard_normal(frame_count).astype(np.float32)) * 10
        pixel_frames = temporal_signal[:, np.newaxis] * np.ones((1, pixel_count), dtype=np.float32)

        weights = np.ones(pixel_count, dtype=np.float32)
        weights /= norm(weights)

        variance_ratio, _ = _check_split_components(
            pixel_frames=pixel_frames.copy(),
            weights=weights,
            intensity_threshold=0.1,
        )

        # Single-component data should produce a ratio near 1 (the two-component model should not explain
        # significantly more variance than the single-component model).
        assert variance_ratio < 1.5

    def test_returns_valid_active_mask_and_projections(self) -> None:
        """Verifies that the returned active mask and temporal projections have consistent shapes."""
        rng = np.random.default_rng(seed=7)
        pixel_count = 12
        frame_count = 60

        pixel_frames = np.abs(rng.standard_normal((frame_count, pixel_count)).astype(np.float32)) * 3
        weights = np.ones(pixel_count, dtype=np.float32)
        weights /= norm(weights)

        _, (spatial_weights, temporal_projections, active_mask) = _check_split_components(
            pixel_frames=pixel_frames.copy(),
            weights=weights,
            intensity_threshold=0.1,
        )

        # The number of temporal projections should match the number of active frames.
        assert temporal_projections.shape[0] == active_mask.sum()
        assert spatial_weights.shape[0] == pixel_count


class TestExtendIteratively:
    """Tests _extend_iteratively."""

    def test_bright_center_extends_outward(self) -> None:
        """Verifies that a bright center pixel in a small frame extends outward into neighboring pixels."""
        height = 16
        width = 16
        frame_count = 20

        # Creates frames with a bright Gaussian-like center blob.
        frames_2d = np.zeros((frame_count, height, width), dtype=np.float32)
        center_y, center_x = 8, 8
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                distance = np.sqrt(dy**2 + dx**2)
                if center_y + dy < height and center_x + dx < width:
                    frames_2d[:, center_y + dy, center_x + dx] = max(0, 5.0 - distance)

        # Flattens frames to (frame_count, height * width) as expected by _extend_iteratively.
        frames = frames_2d.reshape(frame_count, height * width)

        y_pixels = np.array([center_y], dtype=np.int32)
        x_pixels = np.array([center_x], dtype=np.int32)
        active_frame_indices = np.arange(frame_count, dtype=np.intp)

        extended_y, extended_x, extended_weights = _extend_iteratively(
            y_pixels=y_pixels,
            x_pixels=x_pixels,
            frames=frames,
            height=height,
            width=width,
            active_frame_indices=active_frame_indices,
        )

        # The extension should have grown beyond the initial single pixel.
        assert len(extended_y) > 1
        assert len(extended_x) > 1
        assert len(extended_weights) == len(extended_y)

        # The weights should be normalized.
        assert np.isclose(norm(extended_weights), 1.0, atol=1e-5)

        # The extended pixels should be near the center.
        assert np.all(np.abs(extended_y - center_y) <= 5)
        assert np.all(np.abs(extended_x - center_x) <= 5)

    def test_returns_normalized_weights(self) -> None:
        """Verifies that the returned weights are unit-normalized."""
        height = 12
        width = 12
        frame_count = 15

        rng = np.random.default_rng(seed=55)
        frames_2d = np.zeros((frame_count, height, width), dtype=np.float32)
        # Creates a small bright region in the center.
        frames_2d[:, 4:8, 4:8] = rng.uniform(low=3.0, high=10.0, size=(frame_count, 4, 4)).astype(np.float32)
        frames = frames_2d.reshape(frame_count, height * width)

        y_pixels = np.array([6], dtype=np.int32)
        x_pixels = np.array([6], dtype=np.int32)
        active_frame_indices = np.arange(frame_count, dtype=np.intp)

        _, _, extended_weights = _extend_iteratively(
            y_pixels=y_pixels,
            x_pixels=x_pixels,
            frames=frames,
            height=height,
            width=width,
            active_frame_indices=active_frame_indices,
        )

        assert np.isclose(norm(extended_weights), 1.0, atol=1e-5)


class TestFindBestScale:
    """Tests _find_best_scale."""

    def test_returns_positive_scale_for_structured_images(self) -> None:
        """Verifies that structured scale images produce a positive scale index."""
        rng = np.random.default_rng(seed=123)
        scale_count = 5
        height = 64
        width = 64

        # Creates scale images where one scale has the strongest signal.
        scale_images = rng.standard_normal((scale_count, height, width)).astype(np.float32)
        # Makes scale 2 dominant by adding a strong signal.
        scale_images[2] += 10.0

        result = _find_best_scale(scale_images=scale_images)

        assert result >= 1

    def test_zero_images_returns_default_scale(self) -> None:
        """Verifies that all-zero scale images return the default minimum spatial scale of 1."""
        scale_count = 4
        height = 32
        width = 32

        scale_images = np.zeros((scale_count, height, width), dtype=np.float32)

        result = _find_best_scale(scale_images=scale_images)

        assert result == 1
