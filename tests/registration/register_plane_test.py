"""Contains integration tests for the register_plane stage entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from cindra.registration import register_plane

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from numpy.typing import NDArray

    from cindra.dataclasses import RuntimeContext, SingleRecordingConfiguration

_BLOB_CENTERS: tuple[tuple[int, int], ...] = ((32, 32), (76, 44), (50, 90), (96, 96))
"""Blob centers for a 128x128 synthetic frame with distinct, well-separated structure for phase correlation."""

_SECONDARY_BLOB_CENTERS: tuple[tuple[int, int], ...] = ((40, 28), (68, 80), (96, 40), (24, 100))
"""Blob centers for a distinct second-channel synthetic frame."""


def _static_blob_movie(
    gaussian_blob_image: Callable[..., NDArray[np.float64]],
    frame_count: int = 30,
    centers: tuple[tuple[int, int], ...] = _BLOB_CENTERS,
) -> NDArray[np.int16]:
    """Builds a motion-free structured movie that registers trivially, exercising the registration code paths."""
    base = gaussian_blob_image(height=128, width=128, centers=centers, sigma=4.0, amplitude=2000.0).astype(np.int16)
    return np.broadcast_to(base, (frame_count, 128, 128)).copy()


class TestRegisterPlane:
    """Tests register_plane."""

    def test_reduces_inter_frame_variance(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
        read_binary_movie: Callable[[Path, int, int], NDArray[np.int16]],
    ) -> None:
        """Verifies that register_plane corrects planted motion, sharply reducing the interior inter-frame variance."""
        base = gaussian_blob_image(height=128, width=128, centers=_BLOB_CENTERS, sigma=4.0, amplitude=2000.0).astype(
            np.int16
        )
        shifts_y = np.tile([0, 2, -2, 1, -1, 3], 5)
        shifts_x = np.tile([0, -2, 2, -1, 1, -3], 5)
        frame_count = shifts_y.size
        movie = np.empty((frame_count, 128, 128), dtype=np.int16)
        for index in range(frame_count):
            movie[index] = np.roll(base, shift=(int(shifts_y[index]), int(shifts_x[index])), axis=(0, 1))

        interior = (slice(16, 112), slice(16, 112))
        unregistered_std = movie[:, interior[0], interior[1]].astype(np.float64).std(axis=0).mean()

        context = single_recording_context(
            tmp_path, frame_height=128, frame_width=128, frame_count=frame_count, movie=movie
        )

        register_plane(context=context)

        binary_path = tmp_path / "output" / "cindra" / "plane_0" / "channel_1_data.bin"
        registered = read_binary_movie(binary_path, 128, 128)
        registered_std = registered[:, interior[0], interior[1]].astype(np.float64).std(axis=0).mean()
        # Correcting the planted motion collapses the per-pixel temporal spread across frames in the interior region.
        assert registered_std < 0.5 * unregistered_std

    def test_records_offsets_and_writes_outputs(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that register_plane persists rigid offsets, a reference image, a mean image, and crop ranges."""
        base = gaussian_blob_image(height=128, width=128, centers=_BLOB_CENTERS, sigma=4.0, amplitude=2000.0).astype(
            np.int16
        )
        movie = np.broadcast_to(base, (30, 128, 128)).copy()
        context = single_recording_context(tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie)

        register_plane(context=context)

        plane_directory = tmp_path / "output" / "cindra" / "plane_0"
        registration_directory = plane_directory / "registration_data"
        recovered_y = np.load(registration_directory / "rigid_y_offsets.npy")
        reference_image = np.load(registration_directory / "reference_image.npy")
        assert recovered_y.shape == (30,)
        assert recovered_y.dtype == np.int32
        assert reference_image.shape == (128, 128)
        assert (plane_directory / "detection_data" / "mean_image.npy").exists()
        assert context.runtime.registration.valid_y_range is not None
        assert context.runtime.registration.valid_x_range is not None

    def test_skips_when_already_registered(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
    ) -> None:
        """Verifies that register_plane returns early when the plane is registered and re-registration is disabled."""
        context = single_recording_context(tmp_path)

        # Plants a reference image on disk so that the plane reports as already registered.
        registration_directory = tmp_path / "output" / "cindra" / "plane_0" / "registration_data"
        registration_directory.mkdir(parents=True, exist_ok=True)
        np.save(registration_directory / "reference_image.npy", np.zeros((48, 48), dtype=np.float32))

        register_plane(context=context)

        # The early return happens before any registration work, so no mean image is produced.
        assert context.runtime.detection.mean_image is None

    def test_forced_reregistration_clears_existing(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that register_plane clears existing data and re-runs when re-registration is forced."""
        movie = _static_blob_movie(gaussian_blob_image)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.repeat_registration = True

        context = single_recording_context(
            tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie, configure=configure
        )

        # Plants a reference image on disk so that the plane reports as already registered before the forced run.
        registration_directory = tmp_path / "output" / "cindra" / "plane_0" / "registration_data"
        registration_directory.mkdir(parents=True, exist_ok=True)
        np.save(registration_directory / "reference_image.npy", np.zeros((128, 128), dtype=np.float32))

        register_plane(context=context)

        # The forced re-registration runs the full pipeline, producing a fresh mean image.
        assert context.runtime.detection.mean_image is not None

    def test_registers_two_channels(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that register_plane registers both channels and produces a mean image for each."""
        movie = _static_blob_movie(gaussian_blob_image)
        movie_channel_2 = _static_blob_movie(gaussian_blob_image, centers=_SECONDARY_BLOB_CENTERS)

        context = single_recording_context(
            tmp_path,
            frame_height=128,
            frame_width=128,
            frame_count=30,
            movie=movie,
            movie_channel_2=movie_channel_2,
        )

        register_plane(context=context)

        assert context.runtime.detection.mean_image is not None
        assert context.runtime.detection.mean_image_channel_2 is not None

    def test_aligns_by_second_channel(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that register_plane can align by the second channel and apply offsets to the first channel."""
        movie = _static_blob_movie(gaussian_blob_image)
        movie_channel_2 = _static_blob_movie(gaussian_blob_image, centers=_SECONDARY_BLOB_CENTERS)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.align_by_first_channel = False

        context = single_recording_context(
            tmp_path,
            frame_height=128,
            frame_width=128,
            frame_count=30,
            movie=movie,
            movie_channel_2=movie_channel_2,
            configure=configure,
        )

        register_plane(context=context)

        # Both mean images are produced regardless of which channel drives the alignment.
        assert context.runtime.detection.mean_image is not None
        assert context.runtime.detection.mean_image_channel_2 is not None

    def test_aligns_by_missing_second_channel_raises(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that aligning by the second channel without one present raises a ValueError."""
        movie = _static_blob_movie(gaussian_blob_image)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.align_by_first_channel = False

        context = single_recording_context(
            tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie, configure=configure
        )

        with pytest.raises(ValueError, match="Unable to register channel 2 frames"):
            register_plane(context=context)

    def test_two_step_registration_refines(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that register_plane completes the two-step refinement when it is enabled."""
        movie = _static_blob_movie(gaussian_blob_image)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.two_step_registration = True

        context = single_recording_context(
            tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie, configure=configure
        )

        register_plane(context=context)

        # The refinement pass re-registers the frames and still produces a mean image.
        assert context.runtime.detection.mean_image is not None
        registration_directory = tmp_path / "output" / "cindra" / "plane_0" / "registration_data"
        assert (registration_directory / "rigid_y_offsets.npy").exists()

    def test_loads_bad_frames_from_file(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that register_plane marks frames listed in a bad_frames file as bad."""
        movie = _static_blob_movie(gaussian_blob_image)
        data_directory = tmp_path / "raw"
        data_directory.mkdir(parents=True, exist_ok=True)
        np.save(data_directory / "bad_frames.npy", np.array([2, 5], dtype=np.int64))

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.file_io.data_path = data_directory

        context = single_recording_context(
            tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie, configure=configure
        )

        register_plane(context=context)

        # The in-memory arrays are released after registration, so the bad-frame mask is read back from disk.
        registration_directory = tmp_path / "output" / "cindra" / "plane_0" / "registration_data"
        bad_frames = np.load(registration_directory / "bad_frames.npy")
        assert bool(bad_frames[2])
        assert bool(bad_frames[5])

    def test_handles_absent_bad_frames_file(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that register_plane proceeds normally when a data path is set but no bad_frames file exists."""
        movie = _static_blob_movie(gaussian_blob_image)
        data_directory = tmp_path / "raw"
        data_directory.mkdir(parents=True, exist_ok=True)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.file_io.data_path = data_directory

        context = single_recording_context(
            tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie, configure=configure
        )

        register_plane(context=context)

        # No frames are flagged from disk, and the motion-free movie yields no offset-based outliers.
        registration_directory = tmp_path / "output" / "cindra" / "plane_0" / "registration_data"
        bad_frames = np.load(registration_directory / "bad_frames.npy")
        assert not bool(bad_frames.any())

    def test_nonrigid_two_channel_registration(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that register_plane runs nonrigid registration across both channels and stores block offsets."""
        movie = _static_blob_movie(gaussian_blob_image)
        movie_channel_2 = _static_blob_movie(gaussian_blob_image, centers=_SECONDARY_BLOB_CENTERS)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.nonrigid_registration.enabled = True
            configuration.nonrigid_registration.block_size = (32, 32)

        context = single_recording_context(
            tmp_path,
            frame_height=128,
            frame_width=128,
            frame_count=30,
            movie=movie,
            movie_channel_2=movie_channel_2,
            configure=configure,
        )

        register_plane(context=context)

        registration_directory = tmp_path / "output" / "cindra" / "plane_0" / "registration_data"
        assert (registration_directory / "nonrigid_y_offsets.npy").exists()
        assert context.runtime.detection.mean_image is not None
        assert context.runtime.detection.mean_image_channel_2 is not None

    def test_skips_metrics_for_short_recording(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that registration metrics are skipped when the recording has too few frames."""
        movie = _static_blob_movie(gaussian_blob_image)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.registration_metric_principal_components = 3

        context = single_recording_context(
            tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie, configure=configure
        )

        register_plane(context=context)

        # The recording has far fewer than the minimum frames, so metrics are not computed.
        assert context.runtime.registration.principal_component_extreme_images is None

    def test_disables_frame_normalization(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that register_plane completes with frame normalization disabled and stores sentinel bounds."""
        movie = _static_blob_movie(gaussian_blob_image)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.normalize_frames = False

        context = single_recording_context(
            tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie, configure=configure
        )

        register_plane(context=context)

        # With normalization disabled, the unbounded clip range is stored as the zero sentinel.
        assert context.runtime.registration.normalization_minimum == 0
        assert context.runtime.registration.normalization_maximum == 0

    def test_estimates_bidirectional_phase_offset(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that register_plane estimates and applies a non-zero bidirectional phase offset from the data."""
        movie = _static_blob_movie(gaussian_blob_image)
        # Plants a bidirectional scanning artifact by shifting odd lines horizontally.
        movie[:, 1::2, :] = np.roll(movie[:, 1::2, :], shift=4, axis=2)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.compute_bidirectional_phase_offset = True

        context = single_recording_context(
            tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie, configure=configure
        )

        register_plane(context=context)

        assert context.runtime.registration.bidirectional_phase_offset != 0
        assert context.runtime.registration.bidirectional_phase_corrected

    def test_estimates_zero_bidirectional_phase_offset(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that register_plane estimates a zero bidirectional phase offset for artifact-free data."""
        movie = _static_blob_movie(gaussian_blob_image)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.compute_bidirectional_phase_offset = True

        context = single_recording_context(
            tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie, configure=configure
        )

        register_plane(context=context)

        # The artifact-free movie yields no offset, leaving the bidirectional correction untriggered.
        assert context.runtime.registration.bidirectional_phase_offset == 0
        assert not context.runtime.registration.bidirectional_phase_corrected
