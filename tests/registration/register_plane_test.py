"""Contains integration tests for the register_plane stage entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ataraxis_base_utilities import error_format

from cindra.io.binary import (
    create_binarization_marker,
    _resolve_binarization_marker_path,
    _resolve_registration_marker_path,
)
from cindra.registration import register_plane
from cindra.registration.rigid import translate_frame
from cindra.registration.register import (
    _MINIMUM_REGISTRATION_METRIC_FRAMES,
    _register_frames_batch,
    _register_alignment_channel,
    _register_secondary_channel,
)
from cindra.registration.bidirectional_phase_correction import apply_bidirectional_phase_correction

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from numpy.typing import NDArray

    from cindra.dataclasses import RuntimeContext, SingleRecordingConfiguration

_BLOB_CENTERS: tuple[tuple[int, int], ...] = ((32, 32), (76, 44), (50, 90), (96, 96))
"""Blob centers for a 128x128 synthetic frame with distinct, well-separated structure for phase correlation."""

_SECONDARY_BLOB_CENTERS: tuple[tuple[int, int], ...] = ((40, 28), (68, 80), (96, 40), (24, 100))
"""Blob centers for a distinct second-channel synthetic frame."""

_MOTION_SHIFTS_Y: tuple[int, ...] = (0, 3, -3, 2, -2, 1, -1, 3, -3, 0, 2, -2, 1, -1, 3, -3, 0, 2, -2, 1, -1, 3, -3, 0)
"""The planted per-frame vertical translation, in pixels, of the motion-carrying synthetic movies."""

_MOTION_SHIFTS_X: tuple[int, ...] = (0, 2, 2, -3, 1, -1, 3, 0, -2, 3, -3, 1, 2, -1, -3, 0, 1, 3, -2, 2, -1, -3, 0, 1)
"""The planted per-frame horizontal translation, in pixels, of the motion-carrying synthetic movies."""


class TestRegisterPlane:
    """Tests the per-plane motion correction end to end, from its channel and refinement passes to its markers."""

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
        unregistered_standard_deviation = movie[:, interior[0], interior[1]].astype(np.float64).std(axis=0).mean()

        context = single_recording_context(
            tmp_path=tmp_path, frame_height=128, frame_width=128, frame_count=frame_count, movie=movie
        )

        register_plane(context=context, workers=1)

        binary_path = tmp_path / "output" / "cindra" / "plane_0" / "channel_1_data.bin"
        registered = read_binary_movie(file_path=binary_path, frame_height=128, frame_width=128)
        registered_standard_deviation = registered[:, interior[0], interior[1]].astype(np.float64).std(axis=0).mean()
        # Correcting the planted motion collapses the per-pixel temporal spread across frames in the interior region.
        assert registered_standard_deviation < 0.5 * unregistered_standard_deviation

    def test_records_offsets_and_writes_outputs(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that the persisted rigid offsets carry each frame's planted shift onto the reference's position."""
        movie = _build_shifted_blob_movie(gaussian_blob_image=gaussian_blob_image)
        base_image = gaussian_blob_image(height=128, width=128, centers=_BLOB_CENTERS, sigma=4.0, amplitude=2000.0)
        frame_count = len(_MOTION_SHIFTS_Y)
        planted_y = np.array(_MOTION_SHIFTS_Y, dtype=np.int32)
        planted_x = np.array(_MOTION_SHIFTS_X, dtype=np.int32)

        context = single_recording_context(
            tmp_path=tmp_path, frame_height=128, frame_width=128, frame_count=frame_count, movie=movie
        )

        register_plane(context=context, workers=1)

        plane_directory = tmp_path / "output" / "cindra" / "plane_0"
        registration_directory = plane_directory / "registration_data"
        recovered_y = np.load(registration_directory / "rigid_y_offsets.npy")
        recovered_x = np.load(registration_directory / "rigid_x_offsets.npy")
        reference_image = np.load(registration_directory / "reference_image.npy")
        assert recovered_y.shape == (frame_count,)
        assert recovered_y.dtype == np.int32
        assert reference_image.shape == (128, 128)
        assert (plane_directory / "detection_data" / "mean_image.npy").exists()

        # The offsets are measured against the refinement loop's converged image, so their absolute scale is fixed by
        # where that image's content sits rather than by the planted shifts alone. That position is measured here from
        # the saved reference image itself, by cross-correlating it against the unshifted blob pattern, which uses none
        # of the offsets under test. The refinement recenters on the frames it averages and rounds that recentering to
        # whole pixels once per iteration, so the position wanders a few pixels away from the +-3 pixel band the planted
        # shifts span. Six pixels bounds that wander while staying well inside the 13-pixel search radius, which is the
        # constant a dropped recentering term would add to every offset.
        reference_shift_y, reference_shift_x = _measure_content_shift(image=reference_image, template=base_image)
        assert abs(reference_shift_y) <= 6
        assert abs(reference_shift_x) <= 6

        # With that position known, every offset is pinned absolutely: the offset recorded for a frame has to be the
        # displacement between that frame's planted content and the reference's content. The bound is one pixel rather
        # than zero, because the reference is a mean of frames and therefore sits between integer pixels, which lets
        # the correlation peak of any one frame round either way.
        assert int(np.absolute(recovered_y - (planted_y - reference_shift_y)).max()) <= 1
        assert int(np.absolute(recovered_x - (planted_x - reference_shift_x)).max()) <= 1

        # The crop is the interior rectangle the largest surviving offset of each axis leaves behind.
        bad_frames = np.load(registration_directory / "bad_frames.npy")
        expected_y_margin = int(np.abs(recovered_y[~bad_frames]).max())
        expected_x_margin = int(np.abs(recovered_x[~bad_frames]).max())
        assert context.runtime.registration.valid_y_range == (expected_y_margin, 128 - expected_y_margin)
        assert context.runtime.registration.valid_x_range == (expected_x_margin, 128 - expected_x_margin)

    def test_skips_when_already_registered(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
    ) -> None:
        """Verifies that register_plane returns early when the plane is registered and re-registration is disabled."""
        context = single_recording_context(tmp_path=tmp_path)

        # Plants the full registration output set on disk so that the plane reports as already registered. A subset of
        # it reads as an interrupted save, which the plane re-registers rather than skips.
        registration_directory = tmp_path / "output" / "cindra" / "plane_0" / "registration_data"
        registration_directory.mkdir(parents=True, exist_ok=True)
        np.save(registration_directory / "reference_image.npy", np.zeros((48, 48), dtype=np.float32))
        np.save(registration_directory / "rigid_y_offsets.npy", np.zeros(40, dtype=np.int32))
        np.save(registration_directory / "rigid_x_offsets.npy", np.zeros(40, dtype=np.int32))

        register_plane(context=context, workers=1)

        # The early return happens before any registration work, so no mean image is produced.
        assert context.runtime.detection.mean_image is None

    def test_forced_reregistration_clears_existing(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that register_plane clears existing data and re-runs when re-registration is forced."""
        movie = _build_static_blob_movie(gaussian_blob_image=gaussian_blob_image)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.repeat_registration = True

        context = single_recording_context(
            tmp_path=tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie, configure=configure
        )

        # Plants the full registration output set on disk so that the plane reports as already registered before the
        # forced run, which is what carries the run into the branch that clears the existing data.
        registration_directory = tmp_path / "output" / "cindra" / "plane_0" / "registration_data"
        registration_directory.mkdir(parents=True, exist_ok=True)
        np.save(registration_directory / "reference_image.npy", np.zeros((128, 128), dtype=np.float32))
        np.save(registration_directory / "rigid_y_offsets.npy", np.zeros(30, dtype=np.int32))
        np.save(registration_directory / "rigid_x_offsets.npy", np.zeros(30, dtype=np.int32))

        register_plane(context=context, workers=1)

        # The forced re-registration runs the full pipeline, producing a fresh mean image.
        assert context.runtime.detection.mean_image is not None

    def test_registers_two_channels(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
        read_binary_movie: Callable[..., NDArray[np.int16]],
    ) -> None:
        """Verifies that the second channel is rebuilt from its raw frames and the offsets the first channel yielded."""
        movie = _build_shifted_blob_movie(gaussian_blob_image=gaussian_blob_image)
        movie_channel_2 = _build_shifted_blob_movie(
            gaussian_blob_image=gaussian_blob_image, centers=_SECONDARY_BLOB_CENTERS
        )
        raw_channel_2 = movie_channel_2.copy()

        context = single_recording_context(
            tmp_path=tmp_path,
            frame_height=128,
            frame_width=128,
            frame_count=len(_MOTION_SHIFTS_Y),
            movie=movie,
            movie_channel_2=movie_channel_2,
        )

        register_plane(context=context, workers=1)

        assert context.runtime.detection.mean_image is not None
        assert context.runtime.detection.mean_image_channel_2 is not None

        registration_directory = tmp_path / "output" / "cindra" / "plane_0" / "registration_data"
        y_offsets = np.load(registration_directory / "rigid_y_offsets.npy")
        x_offsets = np.load(registration_directory / "rigid_x_offsets.npy")

        # The planted motion spans six pixels on each axis, so the offsets the alignment channel produced cannot be a
        # constant. A motion-free movie would let the reconstruction below pass without moving a single frame.
        assert int(np.ptp(y_offsets)) >= 4
        assert int(np.ptp(x_offsets)) >= 4

        expected = raw_channel_2.astype(np.float32)
        for index in range(expected.shape[0]):
            expected[index] = translate_frame(
                frame=expected[index], y_offset=int(y_offsets[index]), x_offset=int(x_offsets[index])
            )
        registered = read_binary_movie(
            file_path=tmp_path / "output" / "cindra" / "plane_0" / "channel_2_data.bin",
            frame_height=128,
            frame_width=128,
        )
        np.testing.assert_array_equal(registered, expected.astype(np.int16))

    def test_aligns_by_second_channel(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
        read_binary_movie: Callable[..., NDArray[np.int16]],
    ) -> None:
        """Verifies that aligning by channel 2 rebuilds channel 1 from its raw frames and the channel-2 offsets."""
        movie = _build_shifted_blob_movie(gaussian_blob_image=gaussian_blob_image)
        movie_channel_2 = _build_shifted_blob_movie(
            gaussian_blob_image=gaussian_blob_image, centers=_SECONDARY_BLOB_CENTERS
        )
        raw_channel_1 = movie.copy()

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.align_by_first_channel = False

        context = single_recording_context(
            tmp_path=tmp_path,
            frame_height=128,
            frame_width=128,
            frame_count=len(_MOTION_SHIFTS_Y),
            movie=movie,
            movie_channel_2=movie_channel_2,
            configure=configure,
        )

        register_plane(context=context, workers=1)

        # Both mean images are produced regardless of which channel drives the alignment.
        assert context.runtime.detection.mean_image is not None
        assert context.runtime.detection.mean_image_channel_2 is not None

        registration_directory = tmp_path / "output" / "cindra" / "plane_0" / "registration_data"
        y_offsets = np.load(registration_directory / "rigid_y_offsets.npy")
        x_offsets = np.load(registration_directory / "rigid_x_offsets.npy")

        # The offsets were measured on channel 2, whose blobs sit at different positions than channel 1's, so a run
        # that measured them on channel 1 instead would not reproduce this reconstruction. They also cannot be a
        # constant, which a motion-free movie would let the reconstruction below satisfy without moving a frame.
        assert int(np.ptp(y_offsets)) >= 4
        assert int(np.ptp(x_offsets)) >= 4

        expected = raw_channel_1.astype(np.float32)
        for index in range(expected.shape[0]):
            expected[index] = translate_frame(
                frame=expected[index], y_offset=int(y_offsets[index]), x_offset=int(x_offsets[index])
            )
        registered = read_binary_movie(
            file_path=tmp_path / "output" / "cindra" / "plane_0" / "channel_1_data.bin",
            frame_height=128,
            frame_width=128,
        )
        np.testing.assert_array_equal(registered, expected.astype(np.int16))

    def test_aligns_by_missing_second_channel_raises(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that aligning by the second channel without one present raises a ValueError."""
        movie = _build_static_blob_movie(gaussian_blob_image=gaussian_blob_image)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.align_by_first_channel = False

        context = single_recording_context(
            tmp_path=tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie, configure=configure
        )

        expected_message = (
            "Unable to register channel 2 frames for plane 0. The plane's RuntimeContext instance does not contain "
            "the path to the plane's channel 2 binary file."
        )
        with pytest.raises(ValueError, match=error_format(expected_message)):
            register_plane(context=context, workers=1)

    def test_two_step_registration_refines(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
        read_binary_movie: Callable[..., NDArray[np.int16]],
    ) -> None:
        """Verifies that the refinement pass measures residuals on registered frames and holds the alignment."""
        movie = _build_shifted_blob_movie(gaussian_blob_image=gaussian_blob_image)
        frame_count = len(_MOTION_SHIFTS_Y)
        unregistered_spread = _measure_alignment_spread(movie=movie)
        results: dict[bool, tuple[float, int, int]] = {}

        for two_step_enabled in (False, True):

            def configure(configuration: SingleRecordingConfiguration, enabled: bool = two_step_enabled) -> None:
                configuration.registration.two_step_registration = enabled

            run_directory = tmp_path / f"two_step_{two_step_enabled}"
            context = single_recording_context(
                tmp_path=run_directory,
                frame_height=128,
                frame_width=128,
                frame_count=frame_count,
                movie=movie.copy(),
                configure=configure,
            )
            register_plane(context=context, workers=1)

            registration_directory = run_directory / "output" / "cindra" / "plane_0" / "registration_data"
            y_offsets = np.load(registration_directory / "rigid_y_offsets.npy")
            x_offsets = np.load(registration_directory / "rigid_x_offsets.npy")
            registered = read_binary_movie(
                file_path=run_directory / "output" / "cindra" / "plane_0" / "channel_1_data.bin",
                frame_height=128,
                frame_width=128,
            )
            results[two_step_enabled] = (
                _measure_alignment_spread(movie=registered),
                int(np.ptp(y_offsets)),
                int(np.ptp(x_offsets)),
            )
            assert context.runtime.detection.mean_image is not None

        single_pass_spread, single_pass_y_span, single_pass_x_span = results[False]
        two_step_spread, two_step_y_span, two_step_x_span = results[True]

        # The single pass measures the planted motion itself, which spans six pixels on each axis. The refinement pass
        # runs on the frames the first pass already corrected, so what it records is a residual that no longer spans
        # the planted range. This is what proves the second pass consumed registered rather than raw frames.
        assert single_pass_y_span >= 6
        assert single_pass_x_span >= 6
        assert two_step_y_span <= 1
        assert two_step_x_span <= 1

        # Both runs remove the planted motion, and the refinement leaves no more behind than the pass it refines. The
        # claim is an upper bound rather than a strict improvement, because the first pass already aligns this movie to
        # within floating point noise, which leaves the refinement no room for improvement.
        assert single_pass_spread < 0.5 * unregistered_spread
        assert two_step_spread <= single_pass_spread

    def test_loads_bad_frames_from_file(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that register_plane marks frames listed in a bad_frames file as bad."""
        movie = _build_static_blob_movie(gaussian_blob_image=gaussian_blob_image)
        data_directory = tmp_path / "raw"
        data_directory.mkdir(parents=True, exist_ok=True)
        np.save(data_directory / "bad_frames.npy", np.array([2, 5], dtype=np.int64))

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.file_io.data_path = data_directory

        context = single_recording_context(
            tmp_path=tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie, configure=configure
        )

        register_plane(context=context, workers=1)

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
        movie = _build_static_blob_movie(gaussian_blob_image=gaussian_blob_image)
        data_directory = tmp_path / "raw"
        data_directory.mkdir(parents=True, exist_ok=True)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.file_io.data_path = data_directory

        context = single_recording_context(
            tmp_path=tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie, configure=configure
        )

        register_plane(context=context, workers=1)

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
        movie = _build_static_blob_movie(gaussian_blob_image=gaussian_blob_image)
        movie_channel_2 = _build_static_blob_movie(
            gaussian_blob_image=gaussian_blob_image, centers=_SECONDARY_BLOB_CENTERS
        )

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.nonrigid_registration.enabled = True
            configuration.nonrigid_registration.block_size = (32, 32)

        context = single_recording_context(
            tmp_path=tmp_path,
            frame_height=128,
            frame_width=128,
            frame_count=30,
            movie=movie,
            movie_channel_2=movie_channel_2,
            configure=configure,
        )

        register_plane(context=context, workers=1)

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
        movie = _build_static_blob_movie(gaussian_blob_image=gaussian_blob_image)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.registration_metric_principal_components = 3

        context = single_recording_context(
            tmp_path=tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie, configure=configure
        )

        register_plane(context=context, workers=1)

        # register_plane releases its registration arrays before returning, so every in-memory metric field reads as
        # None whether or not the metrics ran. Only the absence of the saved arrays proves the dispatch was skipped.
        registration_directory = tmp_path / "output" / "cindra" / "plane_0" / "registration_data"
        assert not (registration_directory / "principal_component_extreme_images.npy").exists()
        assert not (registration_directory / "principal_component_projections.npy").exists()
        assert not (registration_directory / "principal_component_shift_metrics.npy").exists()

        # The offsets of the same run are on disk, so the missing metric files are not the result of an absent save.
        assert (registration_directory / "rigid_y_offsets.npy").exists()

    def test_disables_frame_normalization(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that register_plane completes with frame normalization disabled and stores sentinel bounds."""
        movie = _build_static_blob_movie(gaussian_blob_image=gaussian_blob_image)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.normalize_frames = False

        context = single_recording_context(
            tmp_path=tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie, configure=configure
        )

        register_plane(context=context, workers=1)

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
        movie = _build_static_blob_movie(gaussian_blob_image=gaussian_blob_image)
        # Plants a bidirectional scanning artifact by shifting odd lines horizontally.
        movie[:, 1::2, :] = np.roll(movie[:, 1::2, :], shift=4, axis=2)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.compute_bidirectional_phase_offset = True

        context = single_recording_context(
            tmp_path=tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie, configure=configure
        )

        register_plane(context=context, workers=1)

        assert context.runtime.registration.bidirectional_phase_offset != 0
        assert context.runtime.registration.bidirectional_phase_corrected

    def test_configured_bidirectional_offset_corrects_the_reference_sample(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that a configured bidirectional offset corrects the sample frames that produce the reference."""
        movie = _build_static_blob_movie(gaussian_blob_image=gaussian_blob_image)
        # Plants the same artifact the override describes, so a correctly corrected plane holds still.
        movie[:, 1::2, :] = np.roll(movie[:, 1::2, :], shift=4, axis=2)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            # The override supplies the offset directly, so the estimation branch never runs.
            configuration.registration.compute_bidirectional_phase_offset = False
            configuration.registration.bidirectional_phase_offset_override = 4

        context = single_recording_context(
            tmp_path=tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie, configure=configure
        )

        register_plane(context=context, workers=1)

        # A reference built from uncorrected frames disagrees with every corrected batch matched against it,
        # which biases the offsets of a movie that does not move.
        registration_directory = tmp_path / "output" / "cindra" / "plane_0" / "registration_data"
        y_offsets = np.load(registration_directory / "rigid_y_offsets.npy")
        x_offsets = np.load(registration_directory / "rigid_x_offsets.npy")

        assert np.all(np.abs(y_offsets) <= 1)
        assert np.all(np.abs(x_offsets) <= 1)
        assert context.runtime.registration.bidirectional_phase_offset == 4

    def test_estimates_zero_bidirectional_phase_offset(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that register_plane estimates a zero bidirectional phase offset for artifact-free data."""
        movie = _build_static_blob_movie(gaussian_blob_image=gaussian_blob_image)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.compute_bidirectional_phase_offset = True

        context = single_recording_context(
            tmp_path=tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie, configure=configure
        )

        register_plane(context=context, workers=1)

        # The artifact-free movie yields no offset, leaving the bidirectional correction untriggered.
        assert context.runtime.registration.bidirectional_phase_offset == 0
        assert not context.runtime.registration.bidirectional_phase_corrected

    def test_two_step_registration_preserves_bidirectional_record(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that the refinement pass keeps the bidirectional offset the first pass applied to the binary."""
        movie = _build_static_blob_movie(gaussian_blob_image=gaussian_blob_image)
        # Plants a bidirectional scanning artifact by shifting odd lines horizontally.
        movie[:, 1::2, :] = np.roll(movie[:, 1::2, :], shift=4, axis=2)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.compute_bidirectional_phase_offset = True
            configuration.registration.two_step_registration = True

        context = single_recording_context(
            tmp_path=tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie, configure=configure
        )

        register_plane(context=context, workers=1)

        # The refinement pass carries a zero working offset so that it does not correct the binary a second time,
        # which must not overwrite the record of the offset the binary already carries.
        assert context.runtime.registration.bidirectional_phase_offset != 0
        assert context.runtime.registration.bidirectional_phase_corrected

    def test_leaves_no_registration_marker(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that a completed registration clears the marker that guards its in-place rewrite."""
        movie = _build_static_blob_movie(gaussian_blob_image=gaussian_blob_image)
        context = single_recording_context(
            tmp_path=tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie
        )
        binary_path = context.runtime.io.registered_binary_path

        register_plane(context=context, workers=1)

        assert binary_path.exists()
        assert not _resolve_registration_marker_path(binary_path=binary_path).exists()

    def test_interrupted_registration_leaves_a_marker(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verifies that a failure partway through the batch loop leaves the binary marked as mid-registration."""
        context = _make_interrupted_registration_context(
            tmp_path=tmp_path,
            single_recording_context=single_recording_context,
            gaussian_blob_image=gaussian_blob_image,
            monkeypatch=monkeypatch,
        )
        binary_path = context.runtime.io.registered_binary_path

        expected_message = "Unable to register the frame batch. Simulated mid-loop failure."
        with pytest.raises(RuntimeError, match=error_format(expected_message)):
            register_plane(context=context, workers=1)

        # The binary now holds corrected frames up to the failure point and raw frames after it, which only the
        # marker records.
        assert _resolve_registration_marker_path(binary_path=binary_path).exists()

    def test_marker_blocks_a_later_registration(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verifies that registering a binary an interrupted run left behind raises instead of silently proceeding."""
        context = _make_interrupted_registration_context(
            tmp_path=tmp_path,
            single_recording_context=single_recording_context,
            gaussian_blob_image=gaussian_blob_image,
            monkeypatch=monkeypatch,
        )

        expected_message = "Unable to register the frame batch. Simulated mid-loop failure."
        with pytest.raises(RuntimeError, match=error_format(expected_message)):
            register_plane(context=context, workers=1)

        # Restores the real batch function, so the retry fails on the marker rather than on the injected error.
        monkeypatch.setattr("cindra.registration.register._register_frames_batch", _register_frames_batch)

        binary_path = context.runtime.io.registered_binary_path
        marker_path = _resolve_registration_marker_path(binary_path=binary_path)
        expected_message = (
            f"Unable to register plane {context.runtime.io.plane_index}. A previous write of the binary file "
            f"'{binary_path}' was interrupted, so the file holds finished frames up to an unknown point and "
            f"unfinished frames after it. Enable 'file_io.repeat_binarization' and re-run the binarization stage "
            f"to rebuild the binary from its source TIFF files, which also clears the marker at '{marker_path}'."
        )
        with pytest.raises(RuntimeError, match=error_format(expected_message)):
            register_plane(context=context, workers=1)

    def test_binarization_marker_blocks_a_registration(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that a binary an interrupted conversion left marked refuses to register."""
        movie = _build_static_blob_movie(gaussian_blob_image=gaussian_blob_image)
        context = single_recording_context(
            tmp_path=tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie
        )
        create_binarization_marker(binary_path=context.runtime.io.registered_binary_path)

        binary_path = context.runtime.io.registered_binary_path
        marker_path = _resolve_binarization_marker_path(binary_path=binary_path)
        expected_message = (
            f"Unable to register plane {context.runtime.io.plane_index}. A previous write of the binary file "
            f"'{binary_path}' was interrupted, so the file holds finished frames up to an unknown point and "
            f"unfinished frames after it. Enable 'file_io.repeat_binarization' and re-run the binarization stage "
            f"to rebuild the binary from its source TIFF files, which also clears the marker at '{marker_path}'."
        )
        with pytest.raises(RuntimeError, match=error_format(expected_message)):
            register_plane(context=context, workers=1)

    def test_interrupted_second_channel_leaves_both_markers(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verifies that a failure between the two channel rewrites leaves both of the plane's binaries marked."""
        context = _make_interrupted_second_channel_context(
            tmp_path=tmp_path,
            single_recording_context=single_recording_context,
            gaussian_blob_image=gaussian_blob_image,
            monkeypatch=monkeypatch,
        )
        binary_path = context.runtime.io.registered_binary_path
        binary_path_channel_2 = context.runtime.io.registered_binary_path_channel_2

        expected_message = "Unable to register the secondary channel. Simulated inter-channel failure."
        with pytest.raises(RuntimeError, match=error_format(expected_message)):
            register_plane(context=context, workers=1)

        # Channel 1 now holds motion-corrected frames while channel 2 is still raw, and only the markers record that
        # the two binaries disagree about whether motion has been removed.
        assert _resolve_registration_marker_path(binary_path=binary_path).exists()
        assert _resolve_registration_marker_path(binary_path=binary_path_channel_2).exists()

    def test_interrupted_second_channel_blocks_a_later_registration(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verifies that a plane left with one registered channel refuses to register again."""
        context = _make_interrupted_second_channel_context(
            tmp_path=tmp_path,
            single_recording_context=single_recording_context,
            gaussian_blob_image=gaussian_blob_image,
            monkeypatch=monkeypatch,
        )

        expected_message = "Unable to register the secondary channel. Simulated inter-channel failure."
        with pytest.raises(RuntimeError, match=error_format(expected_message)):
            register_plane(context=context, workers=1)

        # Restores the real secondary-channel pass, so the retry fails on the marker rather than the injected error.
        monkeypatch.setattr("cindra.registration.register._register_secondary_channel", _register_secondary_channel)

        binary_path = context.runtime.io.registered_binary_path
        marker_path = _resolve_registration_marker_path(binary_path=binary_path)
        expected_message = (
            f"Unable to register plane {context.runtime.io.plane_index}. A previous write of the binary file "
            f"'{binary_path}' was interrupted, so the file holds finished frames up to an unknown point and "
            f"unfinished frames after it. Enable 'file_io.repeat_binarization' and re-run the binarization stage "
            f"to rebuild the binary from its source TIFF files, which also clears the marker at '{marker_path}'."
        )
        with pytest.raises(RuntimeError, match=error_format(expected_message)):
            register_plane(context=context, workers=1)

    def test_applies_bidirectional_correction_to_second_channel(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
        read_binary_movie: Callable[..., NDArray[np.int16]],
    ) -> None:
        """Verifies that the secondary channel receives the bidirectional phase correction, not the offsets alone."""
        movie = _build_static_blob_movie(gaussian_blob_image=gaussian_blob_image)
        movie_channel_2 = _build_static_blob_movie(
            gaussian_blob_image=gaussian_blob_image, centers=_SECONDARY_BLOB_CENTERS
        )
        raw_channel_2 = movie_channel_2.copy()

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.bidirectional_phase_offset_override = 3

        context = single_recording_context(
            tmp_path=tmp_path,
            frame_height=128,
            frame_width=128,
            frame_count=30,
            movie=movie,
            movie_channel_2=movie_channel_2,
            configure=configure,
        )

        register_plane(context=context, workers=1)

        # Rebuilds the state the channel-2 binary has to hold: the odd lines shifted by the configured offset, then
        # the rigid offsets the alignment channel computed.
        registration_directory = tmp_path / "output" / "cindra" / "plane_0" / "registration_data"
        y_offsets = np.load(registration_directory / "rigid_y_offsets.npy")
        x_offsets = np.load(registration_directory / "rigid_x_offsets.npy")
        expected = raw_channel_2.astype(np.float32)
        apply_bidirectional_phase_correction(frames=expected, bidirectional_phase_offset=3)
        for index in range(expected.shape[0]):
            expected[index] = translate_frame(
                frame=expected[index], y_offset=int(y_offsets[index]), x_offset=int(x_offsets[index])
            )

        registered = read_binary_movie(
            file_path=tmp_path / "output" / "cindra" / "plane_0" / "channel_2_data.bin",
            frame_height=128,
            frame_width=128,
        )
        np.testing.assert_array_equal(registered, expected.astype(np.int16))

    @pytest.mark.parametrize("pre_smoothing_sigma", [0.0, 2.0])
    def test_one_photon_registration_pre_aligns_the_nonrigid_input(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
        pre_smoothing_sigma: float,
    ) -> None:
        """Verifies that one-photon nonrigid registration measures residuals rather than the rigid motion again."""
        movie = _build_shifted_blob_movie(gaussian_blob_image=gaussian_blob_image, illumination_amplitude=3000.0)
        planted_y = np.array(_MOTION_SHIFTS_Y, dtype=np.int32)
        planted_x = np.array(_MOTION_SHIFTS_X, dtype=np.int32)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.one_photon_registration.enabled = True
            # A zero sigma skips the box pre-smoothing and leaves the high-pass filter to run on its own.
            configuration.one_photon_registration.pre_smoothing_sigma = pre_smoothing_sigma
            configuration.one_photon_registration.spatial_highpass_window = 20
            configuration.nonrigid_registration.enabled = True
            configuration.nonrigid_registration.block_size = (32, 32)

        context = single_recording_context(
            tmp_path=tmp_path,
            frame_height=128,
            frame_width=128,
            frame_count=len(_MOTION_SHIFTS_Y),
            movie=movie,
            configure=configure,
        )

        register_plane(context=context, workers=1)

        registration_directory = tmp_path / "output" / "cindra" / "plane_0" / "registration_data"
        y_offsets = np.load(registration_directory / "rigid_y_offsets.npy")
        x_offsets = np.load(registration_directory / "rigid_x_offsets.npy")

        # The one-photon path high-passes the static illumination gradient away before correlating, so the rigid stage
        # still recovers the planted motion up to the constant its reference frame contributes.
        assert int(np.ptp(y_offsets - planted_y)) <= 1
        assert int(np.ptp(x_offsets - planted_x)) <= 1

        nonrigid_y = np.load(registration_directory / "nonrigid_y_offsets.npy")
        nonrigid_x = np.load(registration_directory / "nonrigid_x_offsets.npy")
        magnitudes = np.hypot(nonrigid_y.astype(np.float64), nonrigid_x.astype(np.float64))

        # The nonrigid stage runs on the smoothed working copy, which the rigid offsets are applied to before the
        # per-block correlation. With that pre-shift in place the blocks see pre-aligned data and report only the
        # local residual, which averages under a pixel. Dropping the pre-shift makes every block re-measure the rigid
        # translation that was already corrected on the frames themselves, which lifts the same average past three
        # pixels. The assertion is a bound rather than an equality, because the per-block residual depends on the
        # block grid the frame geometry produces, while the separation between the two regimes is what matters.
        assert float(magnitudes.mean()) < 1.5

    def test_one_photon_reference_drops_structure_below_the_smoothing_window(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that the one-photon reference is built from pre-smoothed frames, not from the raw ones."""
        movie = _build_shifted_blob_movie(
            gaussian_blob_image=gaussian_blob_image, illumination_amplitude=3000.0, checkerboard_amplitude=600.0
        )
        planted_y = np.array(_MOTION_SHIFTS_Y, dtype=np.int32)
        planted_x = np.array(_MOTION_SHIFTS_X, dtype=np.int32)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.one_photon_registration.enabled = True
            # A four-pixel box window, which spans two full periods of the checkerboard the movie carries.
            configuration.one_photon_registration.pre_smoothing_sigma = 4.0
            configuration.one_photon_registration.spatial_highpass_window = 20

        context = single_recording_context(
            tmp_path=tmp_path,
            frame_height=128,
            frame_width=128,
            frame_count=len(_MOTION_SHIFTS_Y),
            movie=movie,
            configure=configure,
        )

        register_plane(context=context, workers=1)

        registration_directory = tmp_path / "output" / "cindra" / "plane_0" / "registration_data"
        reference_image = np.load(registration_directory / "reference_image.npy")
        y_offsets = np.load(registration_directory / "rigid_y_offsets.npy")
        x_offsets = np.load(registration_directory / "rigid_x_offsets.npy")

        # The run has to have registered the movie for the reference below to mean anything, which the planted motion
        # the offsets reproduce establishes.
        assert int(np.ptp(y_offsets - planted_y)) <= 1
        assert int(np.ptp(x_offsets - planted_x)) <= 1

        # A box mean of an even-width window over a single-pixel checkerboard is identically zero, and neither the
        # high-pass filter that follows the smoothing nor the whole-pixel translations the refinement applies can
        # reintroduce a frequency the smoothing removed. A reference averaged from pre-smoothed frames therefore
        # carries none of the pattern. A reference averaged from raw frames keeps it, because the checkerboard rides
        # along with the tissue and so lands in a common phase across the frames the averaging selects. It keeps it
        # as well because the high-pass filter preserves rather than removes the frame's finest structure. The bound
        # is expressed against the amplitude the movie itself carries, and is placed between the two regimes on a
        # logarithmic scale. A tenth of a percent of the planted amplitude sits several hundred times above the
        # float32 rounding residue a correctly smoothed reference leaves behind, and several hundred times below the
        # fraction an unsmoothed reference retains. That placement also rejects a reference that keeps only a small
        # part of the pattern.
        planted_amplitude = _measure_checkerboard_amplitude(image=movie[0])
        assert planted_amplitude > 500.0
        assert _measure_checkerboard_amplitude(image=reference_image) < 0.001 * planted_amplitude

    def test_two_step_refinement_reapplies_offsets_to_the_secondary_channel(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
        read_binary_movie: Callable[..., NDArray[np.int16]],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verifies that the secondary binary holds both passes' offsets and exactly one bidirectional correction."""
        movie = _build_shifted_blob_movie(gaussian_blob_image=gaussian_blob_image)
        movie_channel_2 = _build_shifted_blob_movie(
            gaussian_blob_image=gaussian_blob_image, centers=_SECONDARY_BLOB_CENTERS
        )
        raw_channel_2 = movie_channel_2.copy()

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.two_step_registration = True
            configuration.registration.bidirectional_phase_offset_override = 3

        context = single_recording_context(
            tmp_path=tmp_path,
            frame_height=128,
            frame_width=128,
            frame_count=len(_MOTION_SHIFTS_Y),
            movie=movie,
            movie_channel_2=movie_channel_2,
            configure=configure,
        )

        pass_offsets: list[tuple[NDArray[np.int32], NDArray[np.int32]]] = []

        def record_pass_offsets(*, context: RuntimeContext, workers: int, device: int | None = None) -> None:
            """Runs the real alignment pass and keeps the offsets it measured, which the next pass overwrites."""
            _register_alignment_channel(context=context, workers=workers, device=device)
            registration_data = context.runtime.registration
            assert registration_data.rigid_y_offsets is not None
            assert registration_data.rigid_x_offsets is not None
            pass_offsets.append((registration_data.rigid_y_offsets.copy(), registration_data.rigid_x_offsets.copy()))

        monkeypatch.setattr("cindra.registration.register._register_alignment_channel", record_pass_offsets)

        register_plane(context=context, workers=1)

        # Only a run that measured the alignment channel twice can leave the secondary binary carrying two rounds of
        # offsets, which is what the reconstruction below rebuilds.
        assert len(pass_offsets) == 2
        assert int(np.ptp(pass_offsets[0][0])) >= 4
        assert int(np.ptp(pass_offsets[0][1])) >= 4

        # Rebuilds what the secondary binary has to hold, step for step: the bidirectional correction the step-1 pass
        # carried into it exactly once, then each pass's rigid offsets, with the int16 round trip the binary imposes
        # between the two passes. A refinement pass that declared the binary uncorrected would shift its odd lines a
        # second time, and one that skipped the secondary rewrite would leave the step-2 offsets out entirely.
        expected = raw_channel_2.astype(np.float32)
        apply_bidirectional_phase_correction(frames=expected, bidirectional_phase_offset=3)
        for pass_index, (y_offsets, x_offsets) in enumerate(pass_offsets):
            for index in range(expected.shape[0]):
                expected[index] = translate_frame(
                    frame=expected[index], y_offset=int(y_offsets[index]), x_offset=int(x_offsets[index])
                )
            if pass_index == 0:
                expected = np.clip(expected, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)
                expected = expected.astype(np.float32)

        registered = read_binary_movie(
            file_path=tmp_path / "output" / "cindra" / "plane_0" / "channel_2_data.bin",
            frame_height=128,
            frame_width=128,
        )
        np.testing.assert_array_equal(registered, expected.astype(np.int16))

    def test_secondary_channel_without_a_first_channel_binary_raises(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that applying channel-2 offsets without a channel-1 binary path raises the exact ValueError."""
        movie = _build_static_blob_movie(gaussian_blob_image=gaussian_blob_image)
        movie_channel_2 = _build_static_blob_movie(
            gaussian_blob_image=gaussian_blob_image, centers=_SECONDARY_BLOB_CENTERS
        )

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.align_by_first_channel = False

        context = single_recording_context(
            tmp_path=tmp_path,
            frame_height=128,
            frame_width=128,
            frame_count=30,
            movie=movie,
            movie_channel_2=movie_channel_2,
            configure=configure,
        )
        # Aligning by channel 2 makes channel 1 the secondary channel, so dropping its path is what this branch guards.
        # The alignment channel still registers normally, and the refusal lands on the pass that follows it.
        context.runtime.io.registered_binary_path = None

        expected_message = (
            "Unable to register channel 1 frames for plane 0. The plane's RuntimeContext instance does not contain "
            "the path to the plane's channel 1 binary file."
        )
        with pytest.raises(ValueError, match=error_format(expected_message)):
            register_plane(context=context, workers=1)

    def test_computes_metrics_at_the_minimum_frame_count(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that a recording holding exactly the minimum frame count dispatches the quality metrics."""
        frame_count = _MINIMUM_REGISTRATION_METRIC_FRAMES
        base = gaussian_blob_image(height=48, width=48, sigma=3.0, amplitude=1200.0, background=500.0)
        generator = np.random.default_rng(seed=7)
        # Modulates the blob amplitude across the recording, so the principal component analysis finds a direction
        # rather than collapsing onto numerical noise.
        scales = np.linspace(start=0.8, stop=1.2, num=frame_count)
        movie = (base[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis]).astype(np.int16)
        movie += generator.integers(low=-3, high=3, size=movie.shape).astype(np.int16)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.registration_metric_principal_components = 1
            configuration.registration.batch_size = 500

        context = single_recording_context(
            tmp_path=tmp_path,
            frame_height=48,
            frame_width=48,
            frame_count=frame_count,
            movie=movie,
            configure=configure,
        )

        register_plane(context=context, workers=1)

        # The frame count sits exactly on the inclusive minimum, so the metrics run and their arrays reach disk.
        registration_directory = tmp_path / "output" / "cindra" / "plane_0" / "registration_data"
        shift_metrics = np.load(registration_directory / "principal_component_shift_metrics.npy")
        extreme_images = np.load(registration_directory / "principal_component_extreme_images.npy")
        projections = np.load(registration_directory / "principal_component_projections.npy")
        assert shift_metrics.shape == (1, 3)
        assert projections.shape == (frame_count, 1)
        assert extreme_images.shape[0] == 2
        assert extreme_images.shape[1] == 1

        # The dominant structured temporal direction in the movie is the planted amplitude ramp, so the single
        # component the analysis returns has to track that ramp. Its sign is arbitrary, which is why the correlation
        # is taken in magnitude, and it is asserted against the ramp this test planted rather than against a recorded
        # output. The bound leaves room for the second direction registration contributes, because this movie's low
        # per-frame contrast resolves a minority of its frames one pixel away from the rest.
        correlation = float(np.corrcoef(projections[:, 0], scales)[0, 1])
        assert abs(correlation) > 0.95

        # The two extreme images are the means of the frames at the two ends of that same component, so the brighter of
        # them is the end the ramp's bright frames reach. Swapping the two ends flips this sign.
        intensity_difference = float(extreme_images[1, 0].mean() - extreme_images[0, 0].mean())
        assert np.sign(intensity_difference) == np.sign(correlation)

        # The movie carries no translation at all, so what aligning the two extremes of the component reports is the
        # residual registration leaves behind. That residual is bounded by the diagonal of the one-pixel quantum the
        # integer correlation peak imposes on each axis, rather than by a larger shift.
        assert float(shift_metrics[0, 0]) <= float(np.sqrt(2.0))

        # Both nonrigid columns stay at their zero fill, because the fixture leaves nonrigid registration disabled.
        assert float(shift_metrics[0, 1]) == 0.0
        assert float(shift_metrics[0, 2]) == 0.0

    def test_skips_metrics_one_frame_below_the_minimum(
        self,
        tmp_path: Path,
        single_recording_context: Callable[..., RuntimeContext],
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
    ) -> None:
        """Verifies that one frame below the inclusive minimum frame count skips the quality metrics."""
        frame_count = _MINIMUM_REGISTRATION_METRIC_FRAMES - 1
        base = gaussian_blob_image(height=48, width=48, sigma=3.0, amplitude=1200.0, background=500.0)
        generator = np.random.default_rng(seed=7)
        scales = np.linspace(start=0.8, stop=1.2, num=frame_count)
        movie = (base[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis]).astype(np.int16)
        movie += generator.integers(low=-20, high=20, size=movie.shape).astype(np.int16)

        def configure(configuration: SingleRecordingConfiguration) -> None:
            configuration.registration.registration_metric_principal_components = 1
            configuration.registration.batch_size = 500

        context = single_recording_context(
            tmp_path=tmp_path,
            frame_height=48,
            frame_width=48,
            frame_count=frame_count,
            movie=movie,
            configure=configure,
        )

        register_plane(context=context, workers=1)

        # One frame short of the minimum falls on the skip side of the same comparison the test above pins from the
        # other direction, which fixes the threshold at exactly the declared frame count.
        registration_directory = tmp_path / "output" / "cindra" / "plane_0" / "registration_data"
        assert not (registration_directory / "principal_component_shift_metrics.npy").exists()
        assert (registration_directory / "rigid_y_offsets.npy").exists()


def _build_checkerboard(amplitude: float) -> NDArray[np.float64]:
    """Builds a 128x128 single-pixel checkerboard of the given amplitude, the finest pattern the frame grid holds."""
    rows, columns = np.mgrid[0:128, 0:128]
    return amplitude * ((-1.0) ** (rows + columns))


def _measure_checkerboard_amplitude(image: NDArray[np.float64]) -> float:
    """Measures how much single-pixel checkerboard the image carries, as the magnitude of its projection onto one."""
    return float(np.absolute((image.astype(np.float64) * _build_checkerboard(amplitude=1.0)).mean()))


def _measure_content_shift(image: NDArray[np.float64], template: NDArray[np.float64]) -> tuple[int, int]:
    """Measures the integer circular shift, in pixels, that carries the template's content onto the image's content.

    The measurement is a plain mean-subtracted FFT cross-correlation computed here rather than through any pipeline
    helper, so the position it reports is independent of the offsets the pipeline itself recorded.
    """
    height, width = image.shape
    image_spectrum = np.fft.rfft2(image.astype(np.float64) - image.mean())
    template_spectrum = np.fft.rfft2(template.astype(np.float64) - template.mean())
    correlation = np.fft.irfft2(image_spectrum * np.conjugate(template_spectrum), s=(height, width))
    peak_row, peak_column = np.unravel_index(np.argmax(correlation), correlation.shape)
    shift_y = int(peak_row) - height if int(peak_row) > height // 2 else int(peak_row)
    shift_x = int(peak_column) - width if int(peak_column) > width // 2 else int(peak_column)
    return shift_y, shift_x


def _build_shifted_blob_movie(
    gaussian_blob_image: Callable[..., NDArray[np.float64]],
    centers: tuple[tuple[int, int], ...] = _BLOB_CENTERS,
    illumination_amplitude: float = 0.0,
    checkerboard_amplitude: float = 0.0,
) -> NDArray[np.int16]:
    """Builds a movie whose blobs translate by the planted per-frame shifts on a static illumination background."""
    base = gaussian_blob_image(height=128, width=128, centers=centers, sigma=4.0, amplitude=2000.0)

    # A single-pixel checkerboard riding along with the tissue, standing in for the pixel-scale detector structure the
    # one-photon pre-smoothing exists to average away before the high-pass filter keeps it.
    base = base + _build_checkerboard(amplitude=checkerboard_amplitude)

    # A broad, static, off-center illumination gradient of the kind one-photon preprocessing exists to remove. It does
    # not move with the tissue, so a registration that fails to high-pass it away is pulled toward a zero offset.
    rows, columns = np.mgrid[0:128, 0:128]
    illumination = illumination_amplitude * np.exp(-(((rows - 8) ** 2 + (columns - 8) ** 2) / (2.0 * 80.0**2)))

    movie = np.empty((len(_MOTION_SHIFTS_Y), 128, 128), dtype=np.int16)
    for index, (shift_y, shift_x) in enumerate(zip(_MOTION_SHIFTS_Y, _MOTION_SHIFTS_X, strict=True)):
        movie[index] = (np.roll(base, shift=(shift_y, shift_x), axis=(0, 1)) + illumination).astype(np.int16)
    return movie


def _measure_alignment_spread(movie: NDArray[np.int16]) -> float:
    """Measures how much the intensity centroid of the movie's interior wanders across frames, in pixels.

    The synthetic movies carry four bright blobs on a flat background that translate together, so the background-
    subtracted intensity centroid tracks the frame's translation. Its spread across frames is therefore the residual
    misalignment left after registration, measured without reference to any offset the pipeline itself reported.
    """
    # Restricts the measurement to the interior, so the wrap-around edges the translation introduces stay out of it.
    interior = movie[:, 16:112, 16:112].astype(np.float64)
    weights = np.clip(interior - np.median(interior), 0.0, None)
    rows = np.arange(interior.shape[1], dtype=np.float64)[np.newaxis, :, np.newaxis]
    columns = np.arange(interior.shape[2], dtype=np.float64)[np.newaxis, np.newaxis, :]
    totals = weights.sum(axis=(1, 2))
    centroid_y = (weights * rows).sum(axis=(1, 2)) / totals
    centroid_x = (weights * columns).sum(axis=(1, 2)) / totals
    return float(np.hypot(centroid_y.std(), centroid_x.std()))


def _build_static_blob_movie(
    gaussian_blob_image: Callable[..., NDArray[np.float64]],
    frame_count: int = 30,
    centers: tuple[tuple[int, int], ...] = _BLOB_CENTERS,
) -> NDArray[np.int16]:
    """Builds a motion-free structured movie that registers trivially, exercising the registration code paths."""
    base = gaussian_blob_image(height=128, width=128, centers=centers, sigma=4.0, amplitude=2000.0).astype(np.int16)
    return np.broadcast_to(base, (frame_count, 128, 128)).copy()


def _make_interrupted_registration_context(
    tmp_path: Path,
    single_recording_context: Callable[..., RuntimeContext],
    gaussian_blob_image: Callable[..., NDArray[np.float64]],
    monkeypatch: pytest.MonkeyPatch,
) -> RuntimeContext:
    """Builds a context whose registration fails after its first batch, leaving a partially rewritten binary."""
    movie = _build_static_blob_movie(gaussian_blob_image=gaussian_blob_image)

    def configure(configuration: SingleRecordingConfiguration) -> None:
        # Splits the movie across several batches, so the injected failure lands after at least one batch has already
        # been written into the binary.
        configuration.registration.batch_size = 10

    context = single_recording_context(
        tmp_path=tmp_path, frame_height=128, frame_width=128, frame_count=30, movie=movie, configure=configure
    )

    completed_batches = 0

    def fail_after_first_batch(**keyword_arguments: object) -> object:
        """Runs the first batch normally and raises on every batch after it."""
        nonlocal completed_batches
        completed_batches += 1
        if completed_batches > 1:
            message = "Unable to register the frame batch. Simulated mid-loop failure."
            raise RuntimeError(message)
        return _register_frames_batch(**keyword_arguments)  # type: ignore[arg-type]  # The kwargs are typed as object.

    monkeypatch.setattr("cindra.registration.register._register_frames_batch", fail_after_first_batch)
    return context


def _make_interrupted_second_channel_context(
    tmp_path: Path,
    single_recording_context: Callable[..., RuntimeContext],
    gaussian_blob_image: Callable[..., NDArray[np.float64]],
    monkeypatch: pytest.MonkeyPatch,
) -> RuntimeContext:
    """Builds a two-channel context whose registration fails once the alignment channel has been fully rewritten."""
    movie = _build_static_blob_movie(gaussian_blob_image=gaussian_blob_image)
    movie_channel_2 = _build_static_blob_movie(gaussian_blob_image=gaussian_blob_image, centers=_SECONDARY_BLOB_CENTERS)
    context = single_recording_context(
        tmp_path=tmp_path,
        frame_height=128,
        frame_width=128,
        frame_count=30,
        movie=movie,
        movie_channel_2=movie_channel_2,
    )

    def fail_before_secondary_rewrite(**keyword_arguments: object) -> None:
        """Raises in place of the secondary-channel rewrite, leaving channel 1 registered and channel 2 raw."""
        message = "Unable to register the secondary channel. Simulated inter-channel failure."
        raise RuntimeError(message)

    monkeypatch.setattr("cindra.registration.register._register_secondary_channel", fail_before_secondary_rewrite)
    return context
