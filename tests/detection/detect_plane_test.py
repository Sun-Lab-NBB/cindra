"""Contains integration tests for the detect_plane_rois stage entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import numpy as np
import pytest

from cindra.detection import detect_plane_rois

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

    from cindra.dataclasses import RuntimeContext, SingleRecordingConfiguration

_FRAME_HEIGHT: int = 48
"""The synthetic frame height in pixels used by the detection integration movies."""

_FRAME_WIDTH: int = 48
"""The synthetic frame width in pixels used by the detection integration movies."""

_FRAME_COUNT: int = 60
"""The synthetic frame count, large enough to give the binned movie ample temporal samples for detection."""

_BLOB_CENTERS: tuple[tuple[int, int], ...] = ((12, 12), (30, 18), (20, 34), (36, 36))
"""The planted blob centroids, spaced far enough apart that each maps to a distinct, unambiguous ROI."""

_BLOB_SIGMA: float = 3.0
"""The Gaussian blob radius in pixels, matching a compact soma-sized source for detection."""

_BLOB_AMPLITUDE: float = 1500.0
"""The peak blob intensity added on top of the flat background during active frames."""

_BACKGROUND_LEVEL: float = 100.0
"""The flat baseline intensity of the synthetic field outside of active blob frames."""

_ACTIVE_FRAME_FRACTION: float = 0.25
"""The probability that any given blob is active (bright) in a given frame, producing localized temporal variance."""

_CENTROID_TOLERANCE: float = 8.0
"""The maximum allowed distance in pixels between a detected centroid and its matching planted blob center."""


def _build_flickering_movie(
    blob_builder: Callable[..., NDArray[np.float64]],
    *,
    centers: tuple[tuple[int, int], ...],
    frame_count: int,
    seed: int,
) -> NDArray[np.int16]:
    """Builds a synthetic movie whose spatially fixed Gaussian blobs flicker independently across frames.

    Notes:
        Detection keys on temporal variance, normalizing each pixel by its temporal standard deviation before searching
        for spatially coherent activity. A movie of identical frames therefore yields no detectable ROIs, so each blob
        is switched on in a random subset of frames to plant localized, temporally coherent signal.
    """
    generator = np.random.default_rng(seed=seed)
    movie = np.full((frame_count, _FRAME_HEIGHT, _FRAME_WIDTH), _BACKGROUND_LEVEL, dtype=np.float64)
    for center in centers:
        blob = blob_builder(
            height=_FRAME_HEIGHT,
            width=_FRAME_WIDTH,
            centers=(center,),
            sigma=_BLOB_SIGMA,
            amplitude=1.0,
            background=0.0,
        )
        activity = (generator.random(frame_count) < _ACTIVE_FRAME_FRACTION).astype(np.float64)
        movie += _BLOB_AMPLITUDE * activity[:, np.newaxis, np.newaxis] * blob[np.newaxis, :, :]
    return movie.astype(np.int16)


def _minimum_centroid_distance(centroid: tuple[int, int], centers: tuple[tuple[int, int], ...]) -> float:
    """Returns the distance from the centroid to the nearest planted blob center."""
    return min(float(np.hypot(centroid[0] - center[0], centroid[1] - center[1])) for center in centers)


def _permissive_detection(configuration: SingleRecordingConfiguration) -> None:
    """Configures a permissive, deterministic detection pass that keeps every blob it finds."""
    configuration.roi_detection.denoise = False
    configuration.roi_detection.preclassification_threshold = 0.0
    configuration.roi_detection.crop_to_soma = False
    configuration.roi_detection.threshold_scaling = 0.5
    configuration.main.tau = 0.01


class TestDetectPlaneRois:
    """Tests detect_plane_rois."""

    def test_detects_planted_blobs(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies detection recovers one ROI per planted blob and writes the detection projections to disk."""
        movie = _build_flickering_movie(gaussian_blob_image, centers=_BLOB_CENTERS, frame_count=_FRAME_COUNT, seed=7)
        context = single_recording_context(
            tmp_path, frame_count=_FRAME_COUNT, movie=movie, configure=_permissive_detection
        )
        context.runtime.registration.valid_y_range = (0, _FRAME_HEIGHT)
        context.runtime.registration.valid_x_range = (0, _FRAME_WIDTH)
        context.runtime.registration.bad_frames = np.zeros(_FRAME_COUNT, dtype=np.bool_)

        detect_plane_rois(context=context)

        roi_statistics = context.runtime.extraction.roi_statistics
        assert roi_statistics is not None
        assert 1 <= len(roi_statistics) <= 2 * len(_BLOB_CENTERS)

        # Every detected ROI lands on a planted blob, so detection produced no spurious centroids.
        for roi in roi_statistics:
            assert _minimum_centroid_distance(roi.mask.centroid, _BLOB_CENTERS) <= _CENTROID_TOLERANCE

        # Every planted blob is recovered by at least one detected ROI.
        centroids = tuple(roi.mask.centroid for roi in roi_statistics)
        for center in _BLOB_CENTERS:
            assert _minimum_centroid_distance(center, centroids) <= _CENTROID_TOLERANCE

        assert context.runtime.detection.roi_diameter > 0
        detection_directory = tmp_path / "output" / "cindra" / "plane_0" / "detection_data"
        assert (detection_directory / "mean_image.npy").exists()
        assert (detection_directory / "enhanced_mean_image.npy").exists()
        assert (detection_directory / "maximum_projection.npy").exists()
        assert (detection_directory / "correlation_map.npy").exists()

    def test_reads_bad_frames_from_disk(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies detection memory-maps bad_frames from disk when the in-memory array is absent."""
        movie = _build_flickering_movie(gaussian_blob_image, centers=_BLOB_CENTERS, frame_count=_FRAME_COUNT, seed=7)
        context = single_recording_context(
            tmp_path, frame_count=_FRAME_COUNT, movie=movie, configure=_permissive_detection
        )
        context.runtime.registration.valid_y_range = (0, _FRAME_HEIGHT)
        context.runtime.registration.valid_x_range = (0, _FRAME_WIDTH)

        # Persists bad_frames to disk and leaves the in-memory field unset to force the memory-map branch.
        registration_directory = tmp_path / "output" / "cindra" / "plane_0" / "registration_data"
        registration_directory.mkdir(parents=True, exist_ok=True)
        np.save(registration_directory / "bad_frames.npy", np.zeros(_FRAME_COUNT, dtype=np.bool_))
        context.runtime.registration.bad_frames = None

        detect_plane_rois(context=context)

        assert context.runtime.extraction.roi_statistics is not None
        assert len(context.runtime.extraction.roi_statistics) >= 1

    def test_detects_second_channel(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies detection runs independently on the second channel when both channels are functional."""
        movie = _build_flickering_movie(gaussian_blob_image, centers=_BLOB_CENTERS, frame_count=_FRAME_COUNT, seed=7)
        movie_channel_2 = _build_flickering_movie(
            gaussian_blob_image, centers=_BLOB_CENTERS, frame_count=_FRAME_COUNT, seed=21
        )

        def configure(configuration: SingleRecordingConfiguration) -> None:
            _permissive_detection(configuration)
            configuration.main.first_channel_functional = True
            configuration.main.second_channel_functional = True

        context = single_recording_context(
            tmp_path,
            frame_count=_FRAME_COUNT,
            movie=movie,
            movie_channel_2=movie_channel_2,
            configure=configure,
        )
        context.runtime.registration.valid_y_range = (0, _FRAME_HEIGHT)
        context.runtime.registration.valid_x_range = (0, _FRAME_WIDTH)
        context.runtime.registration.bad_frames = np.zeros(_FRAME_COUNT, dtype=np.bool_)

        detect_plane_rois(context=context)

        assert context.runtime.extraction.roi_statistics is not None
        assert len(context.runtime.extraction.roi_statistics) >= 1
        assert context.runtime.extraction.roi_statistics_channel_2 is not None
        assert len(context.runtime.extraction.roi_statistics_channel_2) >= 1

        detection_directory = tmp_path / "output" / "cindra" / "plane_0" / "detection_data"
        assert (detection_directory / "mean_image_channel_2.npy").exists()

    def test_denoise_path(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies detection still recovers ROIs when PCA denoising of the binned movie is enabled."""
        movie = _build_flickering_movie(gaussian_blob_image, centers=_BLOB_CENTERS, frame_count=_FRAME_COUNT, seed=7)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            _permissive_detection(configuration)
            configuration.roi_detection.denoise = True

        context = single_recording_context(tmp_path, frame_count=_FRAME_COUNT, movie=movie, configure=configure)
        context.runtime.registration.valid_y_range = (0, _FRAME_HEIGHT)
        context.runtime.registration.valid_x_range = (0, _FRAME_WIDTH)
        context.runtime.registration.bad_frames = np.zeros(_FRAME_COUNT, dtype=np.bool_)

        detect_plane_rois(context=context)

        assert context.runtime.extraction.roi_statistics is not None
        assert len(context.runtime.extraction.roi_statistics) >= 1

    def test_preclassification_path(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies detection applies the preclassification filter when its threshold is above zero."""
        movie = _build_flickering_movie(gaussian_blob_image, centers=_BLOB_CENTERS, frame_count=_FRAME_COUNT, seed=7)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            _permissive_detection(configuration)
            configuration.roi_detection.preclassification_threshold = 0.5

        context = single_recording_context(tmp_path, frame_count=_FRAME_COUNT, movie=movie, configure=configure)
        context.runtime.registration.valid_y_range = (0, _FRAME_HEIGHT)
        context.runtime.registration.valid_x_range = (0, _FRAME_WIDTH)
        context.runtime.registration.bad_frames = np.zeros(_FRAME_COUNT, dtype=np.bool_)

        detect_plane_rois(context=context)

        assert context.runtime.extraction.roi_statistics is not None
        assert len(context.runtime.extraction.roi_statistics) >= 1

    def test_no_rois_raises(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
    ) -> None:
        """Verifies detection raises ValueError when the binned movie contains no detectable activity."""
        movie = np.full((_FRAME_COUNT, _FRAME_HEIGHT, _FRAME_WIDTH), int(_BACKGROUND_LEVEL), dtype=np.int16)
        context = single_recording_context(
            tmp_path, frame_count=_FRAME_COUNT, movie=movie, configure=_permissive_detection
        )
        context.runtime.registration.valid_y_range = (0, _FRAME_HEIGHT)
        context.runtime.registration.valid_x_range = (0, _FRAME_WIDTH)
        context.runtime.registration.bad_frames = np.zeros(_FRAME_COUNT, dtype=np.bool_)

        with pytest.raises(ValueError, match="No ROIs found"):
            detect_plane_rois(context=context)

    def test_missing_binary_path_raises(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies detection raises RuntimeError when the channel 1 registered binary path is unset."""
        movie = _build_flickering_movie(gaussian_blob_image, centers=_BLOB_CENTERS, frame_count=_FRAME_COUNT, seed=7)
        context = single_recording_context(
            tmp_path, frame_count=_FRAME_COUNT, movie=movie, configure=_permissive_detection
        )
        context.runtime.registration.valid_y_range = (0, _FRAME_HEIGHT)
        context.runtime.registration.valid_x_range = (0, _FRAME_WIDTH)
        context.runtime.registration.bad_frames = np.zeros(_FRAME_COUNT, dtype=np.bool_)
        context.runtime.io.registered_binary_path = None

        with pytest.raises(RuntimeError, match="registered binary file path is not set"):
            detect_plane_rois(context=context)
