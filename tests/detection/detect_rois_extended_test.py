"""Contains tests for extended detect_rois module helper functions."""

from __future__ import annotations

from typing import TYPE_CHECKING
import warnings

import numpy as np
from numpy.linalg import norm

from cindra.detection import detect_rois as detect_rois_module
from cindra.detection.detect_rois import (
    _find_best_scale,
    _extend_iteratively,
    detect_rois_in_frames,
    _check_split_components,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from cindra.dataclasses import ROIStatistics

_ITERATION_LIMIT_CENTERS: tuple[tuple[int, int], ...] = ((12, 12), (30, 18), (20, 34), (36, 36))
"""The planted blob centroids of the iteration-limit movie, spaced far enough apart to yield distinct ROIs."""

_ITERATION_LIMIT_TOLERANCE: float = 8.0
"""The maximum distance in pixels between a detected centroid and the planted blob center it must land on."""

_SPLIT_SOURCE_CENTERS: tuple[tuple[int, int], ...] = ((32, 31), (32, 34))
"""The planted centers of the two overlapping sources the split movie holds, three pixels apart on the same row."""

_SPLIT_SOURCE_MIDPOINT: tuple[float, float] = (32.0, 32.5)
"""The midpoint of the two split-movie sources, which is where a mask covering both of them centers its weight."""

_SPLIT_FRAME_COUNT: int = 120
"""The frame count of the split movie, long enough for each source to fire without ever coinciding with the other."""

_SPLIT_SOURCE_PERIODS: tuple[int, int] = (4, 8)
"""The firing periods of the two split-movie sources, which give the first source twice the energy of the second."""

_SPLIT_SOURCE_AMPLITUDE: float = 600.0
"""The peak intensity each split-movie source adds to the frames it fires on."""

_SPLIT_TAIL_ROW: int = 35
"""The first row of the optional process trailing the split movie's first source, three pixels below its center."""

_SPLIT_TAIL_LENGTH: int = 20
"""The length in pixels of the trailing process, chosen so that the retained mask's spatial mean and median differ."""

_SPLIT_TAIL_STRENGTH: float = 0.25
"""The intensity of the trailing process as a fraction of the source peak, high enough to survive the weight trim."""


class TestCheckSplitComponents:
    """Tests _check_split_components."""

    def test_two_component_signal_yields_high_variance_ratio(self) -> None:
        """Verifies that data with two distinct spatial components produces a variance ratio above 1."""
        generator = np.random.default_rng(seed=42)
        pixel_count = 20
        frame_count = 100

        # Creates two distinct temporal signals assigned to different pixel subsets.
        component_1_temporal = np.abs(generator.standard_normal(frame_count).astype(np.float32)) * 5
        component_2_temporal = np.abs(generator.standard_normal(frame_count).astype(np.float32)) * 5

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
        generator = np.random.default_rng(seed=99)
        pixel_count = 15
        frame_count = 80

        # Creates a single-component signal where all pixels share the same temporal trace.
        temporal_signal = np.abs(generator.standard_normal(frame_count).astype(np.float32)) * 10
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
        generator = np.random.default_rng(seed=7)
        pixel_count = 12
        frame_count = 60

        pixel_frames = np.abs(generator.standard_normal((frame_count, pixel_count)).astype(np.float32)) * 3
        weights = np.ones(pixel_count, dtype=np.float32)
        weights /= norm(weights)

        _, (spatial_weights, temporal_projections, active_mask) = _check_split_components(
            pixel_frames=pixel_frames.copy(),
            weights=weights,
            intensity_threshold=0.1,
        )

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
        for delta_y in range(-3, 4):
            for delta_x in range(-3, 4):
                distance = np.sqrt(delta_y**2 + delta_x**2)
                if center_y + delta_y < height and center_x + delta_x < width:
                    frames_2d[:, center_y + delta_y, center_x + delta_x] = max(0, 5.0 - distance)

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

        assert len(extended_y) > 1
        assert len(extended_x) > 1
        assert len(extended_weights) == len(extended_y)

        assert np.isclose(norm(extended_weights), 1.0, atol=1e-5)

        assert np.all(np.abs(extended_y - center_y) <= 5)
        assert np.all(np.abs(extended_x - center_x) <= 5)

    def test_retained_pixel_set_matches_the_weight_fraction(self) -> None:
        """Verifies that growth retains exactly the pixels whose activity clears the mask retention fraction."""
        height = 16
        width = 16
        frame_count = 20
        center_y, center_x = 8, 8
        blob_radius = 3

        # Every pixel of the 7x7 block carries the activity 5.0 minus its Euclidean distance from the center, so the
        # retention fraction alone decides how far the grown mask reaches. The four block corners sit 4.24 pixels out
        # and carry 0.76, which falls under the 0.2 fraction of the 5.0 peak, while every other block pixel clears it.
        frames_2d = np.zeros((frame_count, height, width), dtype=np.float32)
        for delta_y in range(-blob_radius, blob_radius + 1):
            for delta_x in range(-blob_radius, blob_radius + 1):
                distance = float(np.sqrt(delta_y**2 + delta_x**2))
                frames_2d[:, center_y + delta_y, center_x + delta_x] = max(0.0, 5.0 - distance)
        frames = frames_2d.reshape(frame_count, height * width)

        extended_y, extended_x, extended_weights = _extend_iteratively(
            y_pixels=np.array([center_y], dtype=np.int32),
            x_pixels=np.array([center_x], dtype=np.int32),
            frames=frames,
            height=height,
            width=width,
            active_frame_indices=np.arange(frame_count, dtype=np.intp),
        )

        block = {
            (center_y + delta_y, center_x + delta_x)
            for delta_y in range(-blob_radius, blob_radius + 1)
            for delta_x in range(-blob_radius, blob_radius + 1)
        }
        corners = {
            (center_y + delta_y, center_x + delta_x)
            for delta_y in (-blob_radius, blob_radius)
            for delta_x in (-blob_radius, blob_radius)
        }
        assert len(extended_y) == 45
        assert set(zip(extended_y.tolist(), extended_x.tolist(), strict=True)) == block - corners

        # The brightest retained pixel is the seed center, and the weights stay unit-normalized after the trim.
        peak_index = int(extended_weights.argmax())
        assert (int(extended_y[peak_index]), int(extended_x[peak_index])) == (center_y, center_x)
        assert np.isclose(norm(extended_weights), 1.0, atol=1e-5)

    def test_all_negative_residual_returns_the_untrimmed_dilation(self) -> None:
        """Verifies that a residual holding no positive pixel returns the raw 5-pixel dilation of the seed."""
        height = 10
        width = 10
        frame_count = 4

        # Every pixel of every frame carries the same negative residual, which the pipeline reaches once an earlier
        # ROI subtraction overshoots. The mean over the active frames is therefore -1 everywhere, the retention
        # bound collapses to max(0, -0.2) = 0, and no pixel clears it, so growth breaks before the trim runs.
        frames = np.full((frame_count, height * width), fill_value=-1.0, dtype=np.float32)

        extended_y, extended_x, extended_weights = _extend_iteratively(
            y_pixels=np.array([5], dtype=np.int32),
            x_pixels=np.array([5], dtype=np.int32),
            frames=frames,
            height=height,
            width=width,
            active_frame_indices=np.arange(frame_count, dtype=np.intp),
        )

        # The break precedes the trim, so the returned mask is exactly one cardinal dilation of the seed pixel.
        assert sorted(zip(extended_y.tolist(), extended_x.tolist(), strict=True)) == [
            (4, 5),
            (5, 4),
            (5, 5),
            (5, 6),
            (6, 5),
        ]

        # The five untrimmed weights are the untouched -1 means, unit-normalized to -1 / sqrt(5) each.
        expected_weights = np.full(5, fill_value=-1.0) / np.linalg.norm(np.full(5, fill_value=-1.0))
        np.testing.assert_allclose(extended_weights, expected_weights, rtol=1e-6)

    def test_returns_normalized_weights(self) -> None:
        """Verifies that the returned weights are unit-normalized."""
        height = 12
        width = 12
        frame_count = 15

        generator = np.random.default_rng(seed=55)
        frames_2d = np.zeros((frame_count, height, width), dtype=np.float32)
        # Creates a small bright region in the center.
        frames_2d[:, 4:8, 4:8] = generator.uniform(low=3.0, high=10.0, size=(frame_count, 4, 4)).astype(np.float32)
        frames = frames_2d.reshape(frame_count, height * width)

        y_pixels = np.array([6], dtype=np.int32)
        x_pixels = np.array([6], dtype=np.int32)
        active_frame_indices = np.arange(frame_count, dtype=np.intp)

        extended_y, extended_x, extended_weights = _extend_iteratively(
            y_pixels=y_pixels,
            x_pixels=x_pixels,
            frames=frames,
            height=height,
            width=width,
            active_frame_indices=active_frame_indices,
        )

        assert np.isclose(norm(extended_weights), 1.0, atol=1e-5)

        assert len(extended_y) > 1
        assert np.all((extended_y >= 3) & (extended_y <= 8))
        assert np.all((extended_x >= 3) & (extended_x <= 8))

    def test_grows_past_pixel_cap_exits_via_while_condition(self) -> None:
        """Verifies that a uniformly bright frame grows the ROI past the 10000-pixel cap."""
        height = 120
        width = 120
        frame_count = 4

        # Every pixel shares an identical intensity, so the ROI grows monotonically each iteration and never
        # stops via the no-growth break, eventually crossing the 10000-pixel safety cap.
        frames = np.ones((frame_count, height * width), dtype=np.float32)

        y_pixels = np.array([60], dtype=np.int32)
        x_pixels = np.array([60], dtype=np.int32)
        active_frame_indices = np.arange(frame_count, dtype=np.intp)

        extended_y, extended_x, extended_weights = _extend_iteratively(
            y_pixels=y_pixels,
            x_pixels=x_pixels,
            frames=frames,
            height=height,
            width=width,
            active_frame_indices=active_frame_indices,
        )

        # The ROI grows past the 10000-pixel cap, so the while condition (not the break) terminates the loop.
        assert len(extended_y) > 10000
        assert len(extended_x) == len(extended_y)
        assert np.isclose(norm(extended_weights), 1.0, atol=1e-5)

    def test_zero_residual_returns_finite_weights(self) -> None:
        """Verifies that a uniformly zero residual yields finite zero weights rather than a zero-norm division."""
        height = 16
        width = 16
        frame_count = 4

        # A residual that is exactly zero everywhere drives every retention weight to zero, which empties the
        # retention mask and returns the pre-mask weights. Normalizing those by their own zero norm would divide
        # zero by zero, so the guard leaves them unnormalized.
        frames = np.zeros((frame_count, height * width), dtype=np.float32)

        y_pixels = np.array([7, 7, 8, 8], dtype=np.int32)
        x_pixels = np.array([7, 8, 7, 8], dtype=np.int32)
        active_frame_indices = np.arange(frame_count, dtype=np.intp)

        with warnings.catch_warnings():
            # A zero-norm division raises this as a RuntimeWarning, so promoting it to an error fails the test on
            # the exact operation the guard exists to avoid.
            warnings.simplefilter("error", RuntimeWarning)
            extended_y, extended_x, extended_weights = _extend_iteratively(
                y_pixels=y_pixels,
                x_pixels=x_pixels,
                frames=frames,
                height=height,
                width=width,
                active_frame_indices=active_frame_indices,
            )

        # Every weight stays exactly zero, which keeps the caller's projection finite so that it rejects the ROI on
        # its activity threshold instead of on a NaN comparison that is False for every frame.
        assert np.isfinite(extended_weights).all()
        np.testing.assert_array_equal(extended_weights, np.zeros(extended_weights.size, dtype=np.float32))
        assert extended_weights.size == extended_y.size == extended_x.size

    def test_negative_residual_normalizes_weights(self) -> None:
        """Verifies that a uniformly negative residual still normalizes, so the zero guard does not over-trigger."""
        height = 16
        width = 16
        frame_count = 4

        # A negative residual empties the retention mask exactly as the zero residual does, but its weights carry a
        # non-zero norm, so the guard must leave the normalization in place for this arm.
        frames = np.full((frame_count, height * width), fill_value=-1.0, dtype=np.float32)

        extended_y, _, extended_weights = _extend_iteratively(
            y_pixels=np.array([7, 7, 8, 8], dtype=np.int32),
            x_pixels=np.array([7, 8, 7, 8], dtype=np.int32),
            frames=frames,
            height=height,
            width=width,
            active_frame_indices=np.arange(frame_count, dtype=np.intp),
        )

        # Twelve pixels each holding -1 normalize to -1/sqrt(12), so the norm is unity and every weight is negative.
        assert extended_y.size == 12
        assert np.isclose(norm(extended_weights), 1.0, atol=1e-6)
        np.testing.assert_allclose(extended_weights, np.full(12, -1.0 / np.sqrt(12.0)), rtol=1e-6)


class TestFindBestScale:
    """Tests _find_best_scale."""

    def test_returns_positive_scale_for_structured_images(self) -> None:
        """Verifies that structured scale images produce a positive scale index."""
        generator = np.random.default_rng(seed=123)
        scale_count = 5
        height = 64
        width = 64

        # Creates scale images where one scale has the strongest signal.
        scale_images = generator.standard_normal((scale_count, height, width)).astype(np.float32)
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


class TestDetectRoisInFrames:
    """Tests detect_rois_in_frames."""

    def test_iteration_limit_truncates_the_unlimited_detection_sequence(self) -> None:
        """Verifies that a movie holding more activity than the iteration limit returns exactly the limit."""
        movie = _build_flickering_movie(centers=_ITERATION_LIMIT_CENTERS)

        # The unlimited pass exhausts the movie's activity, establishing the full detection sequence the limited
        # passes must truncate. It finds strictly more ROIs than either limit, so both limits bind.
        unlimited = _detect(frames=movie, maximum_iterations=25)
        unlimited_centroids = [roi.mask.centroid for roi in unlimited]
        assert len(unlimited) > 3

        for limit in (2, 3):
            limited = _detect(frames=movie, maximum_iterations=limit)

            # Detection is deterministic and each ROI is subtracted from the residual before the next is drawn, so
            # a limited pass must reproduce the unlimited pass's leading ROIs and stop, holding exactly 'limit' of
            # them. A loop running one step long or one step short changes this count.
            assert len(limited) == limit
            assert [roi.mask.centroid for roi in limited] == unlimited_centroids[:limit]

            # Every ROI the truncated run reports is fully formed: its three mask arrays agree in length, its
            # weights are finite, and its pixels lie inside the frame and on a planted blob.
            for roi in limited:
                pixel_count = roi.mask.y_pixels.size
                assert roi.mask.x_pixels.size == pixel_count
                assert roi.mask.pixel_weights.size == pixel_count
                assert pixel_count > 0
                assert np.isfinite(roi.mask.pixel_weights).all()
                assert int(roi.mask.y_pixels.min()) >= 0
                assert int(roi.mask.y_pixels.max()) < 48
                assert int(roi.mask.x_pixels.min()) >= 0
                assert int(roi.mask.x_pixels.max()) < 48
                distance = min(
                    float(np.hypot(roi.mask.centroid[0] - center[0], roi.mask.centroid[1] - center[1]))
                    for center in _ITERATION_LIMIT_CENTERS
                )
                assert distance <= _ITERATION_LIMIT_TOLERANCE

        # The unlimited pass stops on its own once the residual falls under the threshold, so its final ROI is not
        # the one a longer budget would have added. Reaching the iteration limit and exhausting the activity are
        # therefore genuinely different outcomes rather than the same sequence cut at different points.
        assert len(_detect(frames=movie, maximum_iterations=len(unlimited) + 5)) == len(unlimited)


class TestDetectRoisInFramesDeadPeak:
    """Tests the dead-peak suppression of detect_rois_in_frames."""

    def test_abandoned_peaks_are_suppressed_so_the_loop_advances(self, monkeypatch) -> None:
        """Verifies that every abandoned peak is cleared, so no two iterations select the same dead location."""
        movie = _build_flickering_movie(centers=_ITERATION_LIMIT_CENTERS)
        iteration_limit = 12
        observed_seeds: list[tuple[int, int]] = []

        # Every extension now returns a mask the caller cannot use, so each iteration abandons its peak. This is the
        # state the suppression exists for, and it cannot be produced by any movie, because a peak only clears the
        # detection threshold when the residual under it still holds activity for the extension to find.
        monkeypatch.setattr(
            detect_rois_module, "_extend_iteratively", self._stub_extension(observed_seeds=observed_seeds)
        )

        roi_statistics = _detect(frames=movie, maximum_iterations=iteration_limit)

        # No ROI survives, because the caller rejects every mask the stub returns.
        assert roi_statistics == []

        # The run reaches at least the four planted blobs before the suppression exhausts the remaining activity,
        # so it worked through the movie rather than stopping at the first dead end.
        assert len(_ITERATION_LIMIT_CENTERS) <= len(observed_seeds) <= iteration_limit

        # Every iteration selected a location no earlier iteration had already abandoned. This is the assertion the
        # suppression carries. Without it the variance maps keep the abandoned peak as their maximum, so the next
        # iteration selects the identical seed and the run repeats one location until the budget is exhausted.
        assert len(set(observed_seeds)) == len(observed_seeds)

    @staticmethod
    def _stub_extension(observed_seeds: list[tuple[int, int]]) -> Callable[..., object]:
        """Returns an extension stub that yields zero weights for every seed and records the seed it was given."""

        def extension(**arguments: object) -> object:
            y_pixels = np.asarray(arguments["y_pixels"])
            x_pixels = np.asarray(arguments["x_pixels"])
            observed_seeds.append((round(float(y_pixels.mean())), round(float(x_pixels.mean()))))

            # Reproduces the state a fully explained residual leaves behind. The mask survives, but every weight is
            # zero, so the caller's projection clears no frame and the ROI contributes nothing.
            return y_pixels, x_pixels, np.zeros(y_pixels.size, dtype=np.float32)

        return extension


class TestDetectRoisInFramesSplit:
    """Tests the two-component split branch of detect_rois_in_frames."""

    def test_two_component_roi_is_split_onto_the_more_active_source(self) -> None:
        """Verifies that an ROI seeded over two independent overlapping sources is reduced to the dominant one."""
        movie, activities = _build_split_movie()

        roi_statistics = _detect(frames=movie, maximum_iterations=2)
        assert len(roi_statistics) == 2

        # The seed square covers both sources, so the ROI the first iteration grows spans the pair. Correlating its
        # trace against the two planted time courses reports which sources it actually follows, and the two courses
        # never overlap in time, so a mask spanning the pair follows both equally.
        split_roi = roi_statistics[0]
        trace = self._project(movie=movie, roi=split_roi)
        correlations = [float(np.corrcoef(trace, activity)[0, 1]) for activity in activities]
        dominant_index = int(np.argmax(correlations))

        # The retained component tracks one source and only weakly follows the other. A mask that kept both sources
        # splits its correlation evenly between them and clears neither bound.
        assert max(correlations) > 0.75
        assert min(correlations) < 0.45
        assert max(correlations) - min(correlations) > 0.4

        # The source it retains is the one carrying the most energy, which here is the one firing twice as often,
        # because the split keeps the component whose temporal projection explains the most variance.
        assert dominant_index == int(np.argmax(activities.sum(axis=1)))

        # The weight-trimmed mask therefore sits over the source it tracks rather than between the two, which the
        # midpoint distance measures directly: a mask covering the pair balances its weight on the midpoint.
        center_of_mass = self._center_of_mass(roi=split_roi)
        dominant_center = _SPLIT_SOURCE_CENTERS[dominant_index]
        source_distance = float(
            np.hypot(center_of_mass[0] - dominant_center[0], center_of_mass[1] - dominant_center[1])
        )
        midpoint_distance = float(
            np.hypot(center_of_mass[0] - _SPLIT_SOURCE_MIDPOINT[0], center_of_mass[1] - _SPLIT_SOURCE_MIDPOINT[1])
        )
        assert source_distance < 1.0
        assert midpoint_distance > 1.5

        # The trim also bounds the mask's reach: no pixel of it crosses the center column of the source it discarded.
        discarded_center = _SPLIT_SOURCE_CENTERS[1 - dominant_index]
        if dominant_center[1] < discarded_center[1]:
            assert int(split_roi.mask.x_pixels.max()) <= discarded_center[1]
        else:
            assert int(split_roi.mask.x_pixels.min()) >= discarded_center[1]

        # The split re-centers the ROI onto the mask pixel nearest the retained component's spatial median, replacing
        # the coarse-scale peak the iteration started from. That peak is not a minimizer of this distance, so the
        # reported centroid pins the recentring step rather than the seed it was drawn from.
        y_pixels = split_roi.mask.y_pixels.astype(np.float64)
        x_pixels = split_roi.mask.x_pixels.astype(np.float64)
        distances, centroid_distance = self._squared_distances(
            roi=split_roi, anchor=(float(np.median(y_pixels)), float(np.median(x_pixels)))
        )
        assert centroid_distance == float(distances.min())

        # The second iteration recovers the source the split discarded, so the pair ends up as two distinct ROIs.
        second_trace = self._project(movie=movie, roi=roi_statistics[1])
        second_correlations = [float(np.corrcoef(second_trace, activity)[0, 1]) for activity in activities]
        assert int(np.argmax(second_correlations)) == 1 - dominant_index
        assert max(second_correlations) > 0.75

    def test_split_centroid_follows_the_spatial_median_rather_than_the_mean(self) -> None:
        """Verifies that a skewed retained mask re-centers on its spatial median and not on its spatial mean."""
        movie, activities = _build_split_movie(tail_length=_SPLIT_TAIL_LENGTH)

        split_roi = _detect(frames=movie, maximum_iterations=1)[0]

        # The retained component still follows the source the trailing process belongs to, which is the more active
        # of the two, so the skew comes from that source's own geometry rather than from a wrongly kept component.
        trace = self._project(movie=movie, roi=split_roi)
        correlations = [float(np.corrcoef(trace, activity)[0, 1]) for activity in activities]
        assert int(np.argmax(correlations)) == int(np.argmax(activities.sum(axis=1)))
        assert max(correlations) > 0.75

        # The trailing process drags the mask's spatial mean below its spatial median, so the two anchors no longer
        # select the same pixel. Without this gap the two recentring rules would be indistinguishable at the output.
        y_pixels = split_roi.mask.y_pixels.astype(np.float64)
        x_pixels = split_roi.mask.x_pixels.astype(np.float64)
        median_anchor = (float(np.median(y_pixels)), float(np.median(x_pixels)))
        mean_anchor = (float(y_pixels.mean()), float(x_pixels.mean()))
        median_distances, centroid_to_median = self._squared_distances(roi=split_roi, anchor=median_anchor)
        mean_distances, centroid_to_mean = self._squared_distances(roi=split_roi, anchor=mean_anchor)
        assert int(np.argmin(median_distances)) != int(np.argmin(mean_distances))

        # The reported centroid is a minimizer of the distance to the median and is not one of the distance to the
        # mean, which is exactly what re-centering on the median rather than on the mean produces.
        assert centroid_to_median == float(median_distances.min())
        assert centroid_to_mean > float(mean_distances.min())

    @staticmethod
    def _project(movie: NDArray[np.float32], roi: ROIStatistics) -> NDArray[np.float64]:
        """Projects the raw movie onto an ROI mask, returning the fluorescence trace that mask reports."""
        y_pixels = roi.mask.y_pixels.astype(np.intp)
        x_pixels = roi.mask.x_pixels.astype(np.intp)
        return np.asarray(movie[:, y_pixels, x_pixels], dtype=np.float64) @ roi.mask.pixel_weights.astype(np.float64)

    @staticmethod
    def _center_of_mass(roi: ROIStatistics) -> tuple[float, float]:
        """Returns the weight-weighted center of an ROI mask."""
        weights = roi.mask.pixel_weights.astype(np.float64)
        total = weights.sum()
        return (
            float((weights * roi.mask.y_pixels).sum() / total),
            float((weights * roi.mask.x_pixels).sum() / total),
        )

    @staticmethod
    def _squared_distances(roi: ROIStatistics, anchor: tuple[float, float]) -> tuple[NDArray[np.float64], float]:
        """Returns every mask pixel's squared distance to the anchor and the reported centroid's own distance."""
        y_pixels = roi.mask.y_pixels.astype(np.float64)
        x_pixels = roi.mask.x_pixels.astype(np.float64)
        distances = (y_pixels - anchor[0]) ** 2 + (x_pixels - anchor[1]) ** 2
        centroid_distance = float((roi.mask.centroid[0] - anchor[0]) ** 2 + (roi.mask.centroid[1] - anchor[1]) ** 2)
        return distances, centroid_distance


def _build_flickering_movie(
    centers: tuple[tuple[int, int], ...],
    frame_count: int = 60,
    height: int = 48,
    width: int = 48,
    seed: int = 7,
) -> NDArray[np.float32]:
    """Builds a movie whose fixed Gaussian blobs flicker independently, planting one detectable ROI per center."""
    generator = np.random.default_rng(seed=seed)
    movie = np.full((frame_count, height, width), fill_value=100.0, dtype=np.float64)
    rows, columns = np.mgrid[0:height, 0:width]
    for center in centers:
        blob = np.exp(-(((rows - center[0]) ** 2 + (columns - center[1]) ** 2) / (2 * 3.0**2)))
        activity = (generator.random(frame_count) < 0.25).astype(np.float64)
        movie += 1500.0 * activity[:, np.newaxis, np.newaxis] * blob[np.newaxis, :, :]
    return movie.astype(np.float32)


def _build_split_movie(seed: int = 1, tail_length: int = 0) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    """Builds a movie holding two overlapping sources that never fire together, planting one splittable ROI."""
    generator = np.random.default_rng(seed=seed)
    height = width = 64

    # The first source fires on every fourth frame and the second on every eighth, offset by three frames, so the two
    # time courses never coincide and the first source carries twice the energy of the second. A single spatial
    # component cannot explain both, which drives the variance ratio the split branch tests above its threshold, and
    # the energy gap between them decides which of the two components that branch retains.
    activities = np.zeros((2, _SPLIT_FRAME_COUNT), dtype=np.float64)
    activities[0, 0 :: _SPLIT_SOURCE_PERIODS[0]] = 1.0
    activities[1, 3 :: _SPLIT_SOURCE_PERIODS[1]] = 1.0

    # The sensor noise is what keeps the per-pixel temporal standard deviation the detector divides by well defined
    # away from the sources. Without it the quotient amplifies float rounding into frame-wide structure, and the
    # detector grows masks over a quarter of the frame instead of over the planted sources.
    movie = np.full((_SPLIT_FRAME_COUNT, height, width), fill_value=100.0, dtype=np.float64)
    movie += generator.normal(loc=0.0, scale=15.0, size=movie.shape)
    rows, columns = np.mgrid[0:height, 0:width]
    for center, activity in zip(_SPLIT_SOURCE_CENTERS, activities, strict=True):
        blob = np.exp(-(((rows - center[0]) ** 2 + (columns - center[1]) ** 2) / (2 * 2.5**2)))
        movie += _SPLIT_SOURCE_AMPLITUDE * activity[:, np.newaxis, np.newaxis] * blob[np.newaxis, :, :]

    # Plants a one-pixel-wide process trailing the first source below its center, carrying that source's own time
    # course. The split therefore keeps it, and it skews the retained mask far enough down that the mask's spatial
    # mean and its spatial median no longer round onto the same pixel.
    if tail_length > 0:
        tail = np.zeros((height, width), dtype=np.float64)
        tail[_SPLIT_TAIL_ROW : _SPLIT_TAIL_ROW + tail_length, _SPLIT_SOURCE_CENTERS[0][1]] = _SPLIT_TAIL_STRENGTH
        movie += _SPLIT_SOURCE_AMPLITUDE * activities[0][:, np.newaxis, np.newaxis] * tail[np.newaxis, :, :]
    return movie.astype(np.float32), activities


def _detect(frames: NDArray[np.float32], maximum_iterations: int) -> list[ROIStatistics]:
    """Runs the sparse detector over a private copy of the movie and returns the detected ROI statistics."""
    _, _, _, roi_statistics = detect_rois_in_frames(
        frames=frames.copy(),
        temporal_highpass_window=30,
        spatial_highpass_window=25,
        threshold_scaling=0.5,
        maximum_iterations=maximum_iterations,
        plane_index=0,
    )
    return roi_statistics
