"""Contains tests for the helper functions provided by the metrics module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cindra.registration.metrics import _compute_pc_extremes, _register_pc_extremes

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TestComputePcExtremes:
    """Tests _compute_pc_extremes."""

    def test_output_shapes_two_components(self) -> None:
        """Verifies output array shapes when using two principal components."""
        frame_count = 50
        height = 16
        width = 16
        component_count = 2
        extreme_frame_count = 10

        generator = np.random.default_rng(seed=42)
        frames = generator.standard_normal(size=(frame_count, height, width)).astype(np.float32)

        pc_low, pc_high, projections = _compute_pc_extremes(
            frames=frames,
            num_extreme_frames=extreme_frame_count,
            num_components=component_count,
        )

        assert pc_low.shape == (component_count, height, width)
        assert pc_high.shape == (component_count, height, width)
        assert projections.shape == (frame_count, component_count)

    def test_output_shapes_one_component(self) -> None:
        """Verifies output array shapes when using a single principal component."""
        frame_count = 50
        height = 16
        width = 16
        component_count = 1
        extreme_frame_count = 10

        generator = np.random.default_rng(seed=42)
        frames = generator.standard_normal(size=(frame_count, height, width)).astype(np.float32)

        pc_low, pc_high, projections = _compute_pc_extremes(
            frames=frames,
            num_extreme_frames=extreme_frame_count,
            num_components=component_count,
        )

        assert pc_low.shape == (component_count, height, width)
        assert pc_high.shape == (component_count, height, width)
        assert projections.shape == (frame_count, component_count)

    def test_output_dtypes(self) -> None:
        """Verifies that the pc_low, pc_high, and projections arrays all have float32 dtype."""
        frame_count = 50
        height = 16
        width = 16
        component_count = 2
        extreme_frame_count = 10

        generator = np.random.default_rng(seed=42)
        frames = generator.standard_normal(size=(frame_count, height, width)).astype(np.float32)

        pc_low, pc_high, projections = _compute_pc_extremes(
            frames=frames,
            num_extreme_frames=extreme_frame_count,
            num_components=component_count,
        )

        assert pc_low.dtype == np.float32
        assert pc_high.dtype == np.float32
        assert projections.dtype == np.float32

    def test_projections_are_finite(self) -> None:
        """Verifies that all projection values are finite (no NaN or Inf)."""
        frame_count = 50
        height = 16
        width = 16
        component_count = 2
        extreme_frame_count = 10

        generator = np.random.default_rng(seed=42)
        frames = generator.standard_normal(size=(frame_count, height, width)).astype(np.float32)

        _, _, projections = _compute_pc_extremes(
            frames=frames,
            num_extreme_frames=extreme_frame_count,
            num_components=component_count,
        )

        assert np.all(np.isfinite(projections))

    def test_extreme_means_differ(self) -> None:
        """Verifies that pc_low and pc_high produce different mean images when the input contains structured signal."""
        frame_count = 50
        height = 16
        width = 16
        component_count = 1
        extreme_frame_count = 10

        # Creates frames with a temporal gradient so PCA captures a clear signal direction. Early frames are dark,
        # late frames are bright, ensuring the first PC separates low from high projections.
        generator = np.random.default_rng(seed=42)
        temporal_gradient = np.linspace(start=0.0, stop=1.0, num=frame_count, dtype=np.float32)
        frames = generator.standard_normal(size=(frame_count, height, width)).astype(np.float32)
        frames += temporal_gradient[:, np.newaxis, np.newaxis] * 10.0

        pc_low, pc_high, _ = _compute_pc_extremes(
            frames=frames,
            num_extreme_frames=extreme_frame_count,
            num_components=component_count,
        )

        # Expects the pc_low and pc_high means to differ because extreme frames come from opposite ends of the gradient.
        assert not np.allclose(pc_low[0], pc_high[0])

    def test_extreme_means_reflect_gradient_direction(self) -> None:
        """Verifies that one extreme has a higher overall intensity than the other when frames have a temporal ramp."""
        frame_count = 50
        height = 16
        width = 16
        component_count = 1
        extreme_frame_count = 10

        # Creates frames with spatial noise plus a temporal ramp. Spatial variance is required for PCA over pixels to
        # find a non-degenerate first component. Without it, centering collapses the input to numerical noise and the
        # returned PC direction is arbitrary.
        generator = np.random.default_rng(seed=42)
        temporal_gradient = np.linspace(start=0.0, stop=10.0, num=frame_count, dtype=np.float32)
        frames = generator.standard_normal(size=(frame_count, height, width)).astype(np.float32)
        frames += temporal_gradient[:, np.newaxis, np.newaxis]

        pc_low, pc_high, _ = _compute_pc_extremes(
            frames=frames,
            num_extreme_frames=extreme_frame_count,
            num_components=component_count,
        )

        # Checks that the two extremes have different mean intensities. The PC sign is arbitrary, so avoid assuming an
        # ordering.
        low_mean = float(pc_low[0].mean())
        high_mean = float(pc_high[0].mean())
        assert abs(high_mean - low_mean) > 1.0


_PC_BLOB_CENTERS: tuple[tuple[int, int], ...] = ((30, 34), (72, 46), (48, 88), (94, 92))
"""Blob centers for the 128x128 synthetic principal-component extreme images."""

_PC_SHIFT_Y: int = 3
"""The vertical translation, in pixels, planted between the low and high principal-component extreme images."""

_PC_SHIFT_X: int = -4
"""The horizontal translation, in pixels, planted between the low and high principal-component extreme images."""

_PC_EDGE_TAPER_SLOPE: float = 10.0
"""The edge taper falloff used by the principal-component tests, which keeps the taper off the blobs it measures."""

_PC_HIGHPASS_WINDOW: int = 20
"""The spatial high-pass window used by the principal-component tests."""


def _build_pc_extreme_pair() -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Builds a low and a high principal-component image separated by an exact integer translation.

    The blobs translate between the two images while a linear illumination ramp stays where it is. The ramp is far
    brighter than the blobs, and a box mean over a symmetric window reproduces a linear ramp exactly, so the spatial
    high-pass filter annihilates it while leaving the blobs intact. A correlation that skips that filter is instead
    dominated by the ramp, which is identical in both images and therefore pulls the peak toward a zero offset.

    Returns:
        A tuple of the low and high extreme images, each with shape (1, 128, 128).
    """
    rows, columns = np.mgrid[0:128, 0:128]
    blobs = np.zeros((128, 128), dtype=np.float64)
    for center_row, center_column in _PC_BLOB_CENTERS:
        blobs += 900.0 * np.exp(-(((rows - center_row) ** 2 + (columns - center_column) ** 2) / (2.0 * 4.0**2)))
    illumination = 50.0 * (rows + 0.5 * columns)

    pc_low = (blobs + illumination).astype(np.float32)[np.newaxis, :, :]
    shifted = np.roll(blobs, shift=(_PC_SHIFT_Y, _PC_SHIFT_X), axis=(0, 1))
    pc_high = (shifted + illumination).astype(np.float32)[np.newaxis, :, :]
    return pc_low, pc_high


class TestRegisterPcExtremes:
    """Tests _register_pc_extremes."""

    def test_recovers_planted_shift_without_pre_smoothing(self) -> None:
        """Verifies that one-photon mode without pre-smoothing measures the exact translation between PC extremes."""
        pc_low, pc_high = _build_pc_extreme_pair()

        metrics = _register_pc_extremes(
            pc_low=pc_low,
            pc_high=pc_high,
            bidirectional_corrected=True,
            spatial_highpass_window=_PC_HIGHPASS_WINDOW,
            pre_smoothing_window=None,
            one_photon_mode=True,
            nonrigid_enabled=False,
            edge_taper_slope=_PC_EDGE_TAPER_SLOPE,
            workers=1,
        )

        # The planted translation is exactly (3, -4) pixels, whose magnitude is the 3-4-5 right triangle's hypotenuse.
        # Both images have to be high-passed for the peak to land there, which is what this asserts.
        assert metrics.shape == (1, 3)
        assert float(metrics[0, 0]) == 5.0

        # Both nonrigid columns stay at their zero fill when nonrigid registration is disabled.
        assert float(metrics[0, 1]) == 0.0
        assert float(metrics[0, 2]) == 0.0

    def test_two_photon_mode_is_captured_by_the_illumination_ramp(self) -> None:
        """Verifies that the same PC extremes measure a near-zero shift when one-photon preprocessing is skipped."""
        pc_low, pc_high = _build_pc_extreme_pair()

        metrics = _register_pc_extremes(
            pc_low=pc_low,
            pc_high=pc_high,
            bidirectional_corrected=True,
            spatial_highpass_window=_PC_HIGHPASS_WINDOW,
            pre_smoothing_window=None,
            one_photon_mode=False,
            nonrigid_enabled=False,
            edge_taper_slope=_PC_EDGE_TAPER_SLOPE,
            workers=1,
        )

        # Without the high-pass the shared illumination ramp dominates both images, and the ramp does not move between
        # them, so the correlation peak has to stay inside the one-pixel quantum around the origin instead of landing
        # on the five-pixel translation the blobs carry. This is the control that gives the one-photon assertion above
        # its meaning: the recovered shift comes from the filtering, not from the blobs being trivially findable.
        assert float(metrics[0, 0]) <= 1.0
