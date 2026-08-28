"""Contains tests for the CuPy GPU registration backend and for the device selection register_plane performs."""

from __future__ import annotations

import numpy as np
import pytest
from ataraxis_base_utilities import error_format

from cindra.detection import compute_registration_blocks
from cindra.registration import gpu as gpu_module
from cindra.registration.gpu import (
    _GPU_REMEDY,
    _TF32_VARIABLE,
    GpuRegistrationBackend,
    _require_gpu_runtime,
    _verify_tf32_disabled,
)
from cindra.orchestration.gpu import resolve_gpu_devices
from cindra.registration.batch import ReferenceData
from cindra.registration.rigid import compute_edge_taper, compute_phase_correlation_kernel
from cindra.registration.nonrigid import compute_nonrigid_reference_data
from cindra.registration.register import register_plane, _register_frames_batch, _apply_precomputed_offsets_batch

_SMOOTHING_SIGMA = 1.15
_BLOCK_SIZE = (64, 64)

# Every worker process that reaches a device builds its own CUDA context, which costs hundreds of megabytes of device
# memory apiece. Holding the device tests to one worker keeps a wide parallel run from filling the card with contexts,
# so this group name is shared with every other module that reaches a device.
pytestmark = [
    pytest.mark.xdist_group(name="cuda_device"),
    pytest.mark.skipif(not resolve_gpu_devices().available, reason="the host exposes no usable CUDA device"),
]


def _build_shifted_movie(frame_count, height, width, maximum_shift=5, seed=3):
    """Builds a textured field translated by known integer offsets, together with its unshifted reference."""
    generator = np.random.default_rng(seed)
    field = generator.random((height * 2, width * 2), dtype=np.float32)
    for _ in range(3):
        field = (field + np.roll(field, 1, axis=0) + np.roll(field, 1, axis=1)) / 3.0
    field = ((field - field.min()) / (np.ptp(field) + 1e-9) * 2000.0).astype(np.float32)

    shifts_y = generator.integers(-maximum_shift, maximum_shift + 1, frame_count)
    shifts_x = generator.integers(-maximum_shift, maximum_shift + 1, frame_count)
    top, left = height // 2, width // 2
    movie = np.stack(
        [
            field[
                top + shifts_y[index] : top + shifts_y[index] + height,
                left + shifts_x[index] : left + shifts_x[index] + width,
            ]
            for index in range(frame_count)
        ]
    )
    return movie.astype(np.int16), field[top : top + height, left : left + width].copy()


def _build_reference_data(reference_image, *, nonrigid):
    """Builds the reference data both backends consume, through the CPU helpers the pipeline uses."""
    taper_slope = 3 * _SMOOTHING_SIGMA
    taper_mask, mean_offset = compute_edge_taper(reference_image=reference_image, taper_slope=taper_slope)
    kernel = compute_phase_correlation_kernel(reference_image=reference_image, smoothing_sigma=_SMOOTHING_SIGMA)
    if not nonrigid:
        return ReferenceData(taper_mask, mean_offset, kernel, None, None, None, None)
    height, width = reference_image.shape
    blocks = compute_registration_blocks(height=height, width=width, block_size=_BLOCK_SIZE)
    taper_nonrigid, offset_nonrigid, kernel_nonrigid = compute_nonrigid_reference_data(
        reference_image=reference_image,
        taper_slope=taper_slope,
        smoothing_sigma=_SMOOTHING_SIGMA,
        y_blocks=blocks[0],
        x_blocks=blocks[1],
    )
    return ReferenceData(taper_mask, mean_offset, kernel, taper_nonrigid, offset_nonrigid, kernel_nonrigid, blocks)


_BATCH_PARAMETERS = {
    "normalization_minimum": -np.inf,
    "normalization_maximum": np.inf,
    "bidirectional_phase_offset": 0,
    "pre_smoothing_sigma": 0.0,
    "spatial_highpass_window": 42,
    "temporal_smoothing_sigma": 0.0,
    "maximum_offset_fraction": 0.1,
    "signal_to_noise_threshold": 1.2,
    "maximum_block_offset": 5.0,
    "one_photon_enabled": False,
}


def _register_single_batch(backend, frames, *, nonrigid_enabled, **parameters):
    """Registers one batch through the streaming entry point and returns the single result it yields."""
    return next(backend.register_batches(batches=(frames,), nonrigid_enabled=nonrigid_enabled, **parameters))


class TestBufferReuseContract:
    """Tests which returned arrays survive being held while later batches are produced."""

    def test_accumulated_outputs_survive_later_batches(self):
        """Verifies the offsets and the frame sum stay valid while the whole pass accumulates them.

        The registration pass appends every batch's offsets to a list and combines them once the loop ends, so an
        offset array backed by a staging buffer the next batch refills would corrupt the recorded offsets.
        """
        movie, reference = _build_shifted_movie(frame_count=400, height=128, width=128)
        backend = GpuRegistrationBackend(reference_data=_build_reference_data(reference, nonrigid=True), device=0)

        def batches():
            for start in range(0, 400, 100):
                yield movie[start : start + 100]

        results, snapshots = [], []
        for result in backend.register_batches(batches(), nonrigid_enabled=True, **_BATCH_PARAMETERS):
            results.append(result)
            snapshots.append(
                (
                    result.y_offsets.copy(),
                    result.x_offsets.copy(),
                    result.correlations.copy(),
                    result.y_offsets_nonrigid.copy(),
                    result.frame_sum.copy(),
                )
            )

        assert len(results) == 4
        for result, snapshot in zip(results, snapshots, strict=True):
            np.testing.assert_array_equal(result.y_offsets, snapshot[0])
            np.testing.assert_array_equal(result.x_offsets, snapshot[1])
            np.testing.assert_array_equal(result.correlations, snapshot[2])
            np.testing.assert_array_equal(result.y_offsets_nonrigid, snapshot[3])
            np.testing.assert_array_equal(result.frame_sum, snapshot[4])


class TestDeviceParity:
    """Tests that the device backend resolves the offsets the host backend resolves."""

    @pytest.mark.parametrize("nonrigid", [False, True])
    @pytest.mark.parametrize(("height", "width"), [(128, 128), (128, 127)])
    def test_offsets_match_the_cpu_backend(self, height, width, *, nonrigid):
        """Verifies the device resolves the same quantized offsets the host resolves, at even and odd widths."""
        movie, reference = _build_shifted_movie(frame_count=64, height=height, width=width)
        reference_data = _build_reference_data(reference, nonrigid=nonrigid)

        host = _register_frames_batch(
            reference_data=reference_data,
            frames=movie.astype(np.float32),
            workers=1,
            nonrigid_enabled=nonrigid,
            **_BATCH_PARAMETERS,
        )
        backend = GpuRegistrationBackend(reference_data=reference_data, device=0)
        device = _register_single_batch(
            backend=backend, frames=movie.astype(np.float32), nonrigid_enabled=nonrigid, **_BATCH_PARAMETERS
        )

        np.testing.assert_array_equal(device.y_offsets, host.y_offsets)
        np.testing.assert_array_equal(device.x_offsets, host.x_offsets)
        if nonrigid:
            np.testing.assert_array_equal(device.y_offsets_nonrigid, host.y_offsets_nonrigid)
            np.testing.assert_array_equal(device.x_offsets_nonrigid, host.x_offsets_nonrigid)

    def test_narrow_input_resolves_the_same_offsets_as_a_wide_one(self):
        """Verifies a batch supplied in the stored width resolves what the widened batch resolves."""
        movie, reference = _build_shifted_movie(frame_count=64, height=128, width=128)
        backend = GpuRegistrationBackend(reference_data=_build_reference_data(reference, nonrigid=False), device=0)

        narrow = _register_single_batch(backend=backend, frames=movie, nonrigid_enabled=False, **_BATCH_PARAMETERS)
        wide = _register_single_batch(
            backend=backend, frames=movie.astype(np.float32), nonrigid_enabled=False, **_BATCH_PARAMETERS
        )

        np.testing.assert_array_equal(narrow.y_offsets, wide.y_offsets)
        np.testing.assert_array_equal(narrow.x_offsets, wide.x_offsets)
        assert narrow.frames.dtype == np.int16
        assert wide.frames.dtype == np.float32
        assert narrow.frame_sum is not None
        assert wide.frame_sum is None


class TestDeviceSelection:
    """Tests that register_plane routes the pass to the resource the device argument names."""

    def test_device_pass_writes_the_binary_the_host_pass_writes(
        self, tmp_path, single_recording_context, read_binary_movie
    ):
        """Verifies a plane registered on the device holds the offsets and the binary the host pass produces."""
        movie, _ = _build_shifted_movie(frame_count=64, height=128, width=128)

        outputs = {}
        for label, device in (("host", None), ("device", 0)):
            root = tmp_path / label

            def configure(configuration):
                configuration.registration.batch_size = 20

            root.mkdir()
            context = single_recording_context(
                tmp_path=root, frame_height=128, frame_width=128, frame_count=64, movie=movie, configure=configure
            )
            register_plane(context=context, workers=1, device=device)

            # register_plane releases the registration arrays once it has persisted them, so the recorded offsets are
            # read back off disk rather than off the context.
            registration_directory = root / "output" / "cindra" / "plane_0" / "registration_data"
            outputs[label] = (
                np.load(registration_directory / "rigid_y_offsets.npy"),
                np.load(registration_directory / "rigid_x_offsets.npy"),
                read_binary_movie(context.runtime.io.registered_binary_path, 128, 128).copy(),
                context.runtime.detection.mean_image.copy(),
            )

        cpu, gpu = outputs["host"], outputs["device"]
        np.testing.assert_array_equal(gpu[0], cpu[0])
        np.testing.assert_array_equal(gpu[1], cpu[1])
        np.testing.assert_array_equal(gpu[2], cpu[2])
        np.testing.assert_allclose(gpu[3], cpu[3], rtol=1e-5, atol=1e-3)

    def test_device_pass_registers_both_channels_like_the_host_pass(
        self, tmp_path, single_recording_context, read_binary_movie
    ):
        """Verifies a two-channel plane registered on the device holds the binaries the host pass produces."""
        movie, _ = _build_shifted_movie(frame_count=64, height=128, width=128)
        movie_channel_2 = (movie // 2).astype(np.int16)

        outputs = {}
        for label, device in (("host", None), ("device", 0)):
            root = tmp_path / label

            def configure(configuration):
                configuration.registration.batch_size = 20
                configuration.nonrigid_registration.enabled = True
                configuration.nonrigid_registration.block_size = _BLOCK_SIZE

            root.mkdir()
            context = single_recording_context(
                tmp_path=root,
                frame_height=128,
                frame_width=128,
                frame_count=64,
                movie=movie,
                movie_channel_2=movie_channel_2,
                configure=configure,
            )
            register_plane(context=context, workers=1, device=device)
            outputs[label] = (
                read_binary_movie(context.runtime.io.registered_binary_path, 128, 128).copy(),
                read_binary_movie(context.runtime.io.registered_binary_path_channel_2, 128, 128).copy(),
                context.runtime.detection.mean_image.copy(),
            )

        cpu, gpu = outputs["host"], outputs["device"]

        # The alignment channel resolves the same offsets on both backends, so its binary matches exactly. The
        # secondary channel reaches its frames through the bilinear warp, whose interpolation weights the host forms
        # in double precision, so a small share of its samples lands one storage unit apart.
        np.testing.assert_array_equal(gpu[0], cpu[0])
        disagreement = np.mean(gpu[1] != cpu[1])
        assert disagreement < 0.01, f"secondary channel disagreement {disagreement:.4%} exceeds one percent"
        assert np.abs(gpu[1].astype(np.int32) - cpu[1].astype(np.int32)).max() <= 1
        np.testing.assert_allclose(gpu[2], cpu[2], rtol=1e-4, atol=0.05)


def _parameters(**overrides):
    """Builds the per-pass parameter set with the named overrides applied."""
    return {**_BATCH_PARAMETERS, **overrides}


class TestPreprocessingPaths:
    """Tests the optional preprocessing stages against the host kernels that define them."""

    def test_one_photon_preprocessing_matches_the_cpu_backend(self):
        """Verifies the device reproduces the spatial smoothing and high-pass filtering of the one-photon path."""
        movie, reference = _build_shifted_movie(frame_count=32, height=128, width=128)
        reference_data = _build_reference_data(reference, nonrigid=True)
        parameters = _parameters(one_photon_enabled=True, pre_smoothing_sigma=4.0, spatial_highpass_window=42)

        host = _register_frames_batch(
            reference_data=reference_data,
            frames=movie.astype(np.float32),
            workers=1,
            nonrigid_enabled=True,
            **parameters,
        )
        backend = GpuRegistrationBackend(reference_data=reference_data, device=0)
        device = _register_single_batch(
            backend=backend, frames=movie.astype(np.float32), nonrigid_enabled=True, **parameters
        )

        np.testing.assert_array_equal(device.y_offsets, host.y_offsets)
        np.testing.assert_array_equal(device.x_offsets, host.x_offsets)

    def test_temporal_smoothing_matches_the_cpu_backend(self):
        """Verifies the device reproduces the temporal smoothing applied to the correlation maps."""
        movie, reference = _build_shifted_movie(frame_count=32, height=128, width=128)
        reference_data = _build_reference_data(reference, nonrigid=False)
        parameters = _parameters(temporal_smoothing_sigma=1.5)

        host = _register_frames_batch(
            reference_data=reference_data,
            frames=movie.astype(np.float32),
            workers=1,
            nonrigid_enabled=False,
            **parameters,
        )
        backend = GpuRegistrationBackend(reference_data=reference_data, device=0)
        device = _register_single_batch(
            backend=backend, frames=movie.astype(np.float32), nonrigid_enabled=False, **parameters
        )

        np.testing.assert_array_equal(device.y_offsets, host.y_offsets)
        np.testing.assert_array_equal(device.x_offsets, host.x_offsets)

    @pytest.mark.parametrize("offset", [3, -3])
    def test_bidirectional_correction_matches_the_cpu_backend(self, offset):
        """Verifies the device reproduces the bidirectional scan correction for either shift direction."""
        movie, reference = _build_shifted_movie(frame_count=32, height=128, width=128)
        reference_data = _build_reference_data(reference, nonrigid=False)
        parameters = _parameters(bidirectional_phase_offset=offset)

        host = _register_frames_batch(
            reference_data=reference_data,
            frames=movie.astype(np.float32),
            workers=1,
            nonrigid_enabled=False,
            **parameters,
        )
        backend = GpuRegistrationBackend(reference_data=reference_data, device=0)
        device = _register_single_batch(
            backend=backend, frames=movie.astype(np.float32), nonrigid_enabled=False, **parameters
        )

        np.testing.assert_array_equal(device.y_offsets, host.y_offsets)
        np.testing.assert_array_equal(device.frames, host.frames)

    def test_low_signal_to_noise_smoothing_matches_the_cpu_backend(self):
        """Verifies the device reproduces the progressive smoothing applied to low-quality correlation peaks."""
        generator = np.random.default_rng(19)
        movie = generator.integers(low=100, high=1000, size=(32, 128, 128)).astype(np.int16)
        reference = movie[0].astype(np.float32)
        reference_data = _build_reference_data(reference, nonrigid=True)

        # A threshold no block clears drives every block through all three smoothing levels.
        parameters = _parameters(signal_to_noise_threshold=1e6)

        host = _register_frames_batch(
            reference_data=reference_data,
            frames=movie.astype(np.float32),
            workers=1,
            nonrigid_enabled=True,
            **parameters,
        )
        backend = GpuRegistrationBackend(reference_data=reference_data, device=0)
        device = _register_single_batch(
            backend=backend, frames=movie.astype(np.float32), nonrigid_enabled=True, **parameters
        )

        np.testing.assert_array_equal(device.y_offsets_nonrigid, host.y_offsets_nonrigid)
        np.testing.assert_array_equal(device.x_offsets_nonrigid, host.x_offsets_nonrigid)


class TestPrecomputedOffsets:
    """Tests the entry point the secondary channel registers through."""

    @pytest.mark.parametrize("nonrigid", [False, True])
    def test_application_matches_the_cpu_backend(self, *, nonrigid):
        """Verifies the device applies precomputed offsets the way the host kernels apply them."""
        movie, reference = _build_shifted_movie(frame_count=32, height=128, width=128)
        reference_data = _build_reference_data(reference, nonrigid=nonrigid)
        generator = np.random.default_rng(5)
        y_offsets = generator.integers(-4, 5, 32).astype(np.int32)
        x_offsets = generator.integers(-4, 5, 32).astype(np.int32)
        block_count = 0 if reference_data.blocks is None else len(reference_data.blocks[0])
        y_nonrigid = generator.uniform(-2, 2, (32, block_count)).astype(np.float32) if nonrigid else None
        x_nonrigid = generator.uniform(-2, 2, (32, block_count)).astype(np.float32) if nonrigid else None

        host = _apply_precomputed_offsets_batch(
            frames=movie.astype(np.float32),
            y_offsets=y_offsets,
            x_offsets=x_offsets,
            y_offsets_nonrigid=y_nonrigid,
            x_offsets_nonrigid=x_nonrigid,
            blocks=reference_data.blocks,
            bidirectional_phase_offset=2,
            bidirectional_phase_corrected=False,
            nonrigid_enabled=nonrigid,
        )
        backend = GpuRegistrationBackend(reference_data=reference_data, device=0)
        device, frame_sum = backend.apply_precomputed_offsets(
            frames=movie.astype(np.float32),
            y_offsets=y_offsets,
            x_offsets=x_offsets,
            y_offsets_nonrigid=y_nonrigid,
            x_offsets_nonrigid=x_nonrigid,
            bidirectional_phase_offset=2,
            bidirectional_phase_corrected=False,
            nonrigid_enabled=nonrigid,
        )

        # The host bilinear kernel forms its interpolation weights in double precision, because subtracting an integer
        # index from a float32 coordinate promotes in Numba, while the device forms them in single precision. The warp
        # therefore agrees to single-precision rounding rather than exactly.
        assert frame_sum is None
        np.testing.assert_allclose(device, host, rtol=1e-4, atol=0.05)

    def test_missing_block_offsets_are_refused(self):
        """Verifies that enabling nonrigid application without block offsets reports the missing offsets."""
        movie, reference = _build_shifted_movie(frame_count=8, height=128, width=128)
        backend = GpuRegistrationBackend(reference_data=_build_reference_data(reference, nonrigid=True), device=0)
        expected_message = (
            "Unable to apply precomputed registration offsets on the GPU backend for device 0. Nonrigid registration "
            "is enabled, but the caller supplied no nonrigid block offsets."
        )
        with pytest.raises(ValueError, match=error_format(message=expected_message)):
            backend.apply_precomputed_offsets(
                frames=movie.astype(np.float32),
                y_offsets=np.zeros(8, dtype=np.int32),
                x_offsets=np.zeros(8, dtype=np.int32),
                y_offsets_nonrigid=None,
                x_offsets_nonrigid=None,
                bidirectional_phase_offset=0,
                bidirectional_phase_corrected=True,
                nonrigid_enabled=True,
            )


class TestBackendGuards:
    """Tests the refusals that keep an unusable request off the device."""

    def test_repr_names_the_device_and_the_geometry(self):
        """Verifies the representation carries the device index and the frame geometry."""
        _, reference = _build_shifted_movie(frame_count=4, height=128, width=128)
        backend = GpuRegistrationBackend(reference_data=_build_reference_data(reference, nonrigid=False), device=0)
        assert repr(backend) == "GpuRegistrationBackend(device=0, frame_height=128, frame_width=128, nonrigid=False)"

    def test_unsupported_batch_dtype_is_refused(self):
        """Verifies a batch carrying neither storage nor arithmetic width reports the widths the backend accepts."""
        movie, reference = _build_shifted_movie(frame_count=8, height=128, width=128)
        backend = GpuRegistrationBackend(reference_data=_build_reference_data(reference, nonrigid=False), device=0)
        expected_message = (
            "Unable to stage a frame batch for the GPU registration backend on device 0. The batch dtype must be "
            "int16 or float32, but got float64."
        )
        with pytest.raises(ValueError, match=error_format(message=expected_message)):
            _register_single_batch(
                backend=backend, frames=movie.astype(np.float64), nonrigid_enabled=False, **_BATCH_PARAMETERS
            )

    def test_nonrigid_without_block_reference_data_is_refused(self):
        """Verifies that requesting nonrigid registration against a rigid reference reports the missing blocks."""
        movie, reference = _build_shifted_movie(frame_count=8, height=128, width=128)
        backend = GpuRegistrationBackend(reference_data=_build_reference_data(reference, nonrigid=False), device=0)
        expected_message = (
            "Unable to run nonrigid registration on the GPU backend for device 0. The reference data the backend "
            "holds carries no block structure, so nonrigid registration was requested with a reference computed for "
            "rigid registration alone."
        )
        with pytest.raises(ValueError, match=error_format(message=expected_message)):
            _register_single_batch(
                backend=backend, frames=movie.astype(np.float32), nonrigid_enabled=True, **_BATCH_PARAMETERS
            )

    def test_odd_smoothing_window_is_refused(self):
        """Verifies the spatial smoothing refuses a window the integral-image differencing cannot express."""
        movie, reference = _build_shifted_movie(frame_count=8, height=128, width=128)
        backend = GpuRegistrationBackend(reference_data=_build_reference_data(reference, nonrigid=False), device=0)
        expected_message = (
            "Unable to apply spatial smoothing on the GPU backend. Filter window must be a positive even integer, "
            "but got 41."
        )
        parameters = _parameters(one_photon_enabled=True, pre_smoothing_sigma=41.0)
        with pytest.raises(ValueError, match=error_format(message=expected_message)):
            _register_single_batch(
                backend=backend, frames=movie.astype(np.float32), nonrigid_enabled=False, **parameters
            )

    def test_absent_runtime_is_refused(self, monkeypatch):
        """Verifies the backend refuses to initialize while the CuPy distribution is absent."""
        monkeypatch.setattr(gpu_module, "cupy", None)
        expected_message = (
            f"Unable to initialize the GPU registration backend. The CuPy distribution is not installed, so no CUDA "
            f"device is reachable. {_GPU_REMEDY} Omit the device argument to run the stage on the host CPU instead."
        )
        with pytest.raises(RuntimeError, match=error_format(message=expected_message)):
            _require_gpu_runtime()

    def test_reduced_precision_matrix_mode_is_refused(self, monkeypatch):
        """Verifies the backend refuses a device whose cuBLAS handle allows reduced-precision multiplication."""
        monkeypatch.setattr(gpu_module.cupy.cuda.cublas, "getMathMode", lambda handle: 1)
        expected_message = (
            f"Unable to initialize the GPU registration backend. The cuBLAS handle of the selected device reports "
            f"math mode 1 rather than the default single-precision mode "
            f"{int(gpu_module.cupy.cuda.cublas.CUBLAS_DEFAULT_MATH)}, so the nonrigid subpixel upsampling would run "
            f"at reduced precision and shift the reported correlation peak by a whole 0.1 pixel quantum. Set the "
            f"'{_TF32_VARIABLE}' environment variable to 0 before starting the process."
        )
        with pytest.raises(RuntimeError, match=error_format(message=expected_message)):
            _verify_tf32_disabled()


class TestRemainingBranches:
    """Tests the branches the ordinary registration passes do not reach."""

    def test_one_photon_without_pre_smoothing_matches_the_cpu_backend(self):
        """Verifies the one-photon path runs the high-pass filter alone when no pre-smoothing is configured."""
        movie, reference = _build_shifted_movie(frame_count=16, height=128, width=128)
        reference_data = _build_reference_data(reference, nonrigid=False)
        parameters = _parameters(one_photon_enabled=True, pre_smoothing_sigma=0.0)

        host = _register_frames_batch(
            reference_data=reference_data,
            frames=movie.astype(np.float32),
            workers=1,
            nonrigid_enabled=False,
            **parameters,
        )
        backend = GpuRegistrationBackend(reference_data=reference_data, device=0)
        device = _register_single_batch(
            backend=backend, frames=movie.astype(np.float32), nonrigid_enabled=False, **parameters
        )

        np.testing.assert_array_equal(device.y_offsets, host.y_offsets)

    def test_normalization_weights_are_reused_across_batches(self):
        """Verifies the high-pass normalization weights are derived once and reused for the rest of the pass."""
        movie, reference = _build_shifted_movie(frame_count=32, height=128, width=128)
        backend = GpuRegistrationBackend(reference_data=_build_reference_data(reference, nonrigid=False), device=0)
        parameters = _parameters(one_photon_enabled=True, pre_smoothing_sigma=0.0)

        results = list(backend.register_batches((movie[:16], movie[16:]), nonrigid_enabled=False, **parameters))

        assert len(results) == 2
        assert len(backend._normalization_weights) == 1

    def test_zero_bidirectional_offset_leaves_the_frames_untouched(self):
        """Verifies the bidirectional correction returns without writing when no offset is configured."""
        frames = gpu_module.cupy.asarray(np.arange(2 * 4 * 4, dtype=np.float32).reshape(2, 4, 4))
        expected = frames.copy()
        GpuRegistrationBackend._apply_bidirectional_phase_correction(frames=frames, bidirectional_phase_offset=0)
        assert bool((frames == expected).all())


class TestConfiguredBatchSize:
    """Tests the device batch size the configuration names for a pass running on a CUDA device."""

    def test_gpu_batch_size_overrides_the_shared_batch_size(self, tmp_path, single_recording_context):
        """Verifies a pass naming a device reads its own batch size while the shared one bounds the host pass."""
        movie, _ = _build_shifted_movie(frame_count=48, height=128, width=128)

        def configure(configuration):
            configuration.registration.batch_size = 100
            configuration.registration.gpu_batch_size = 16

        context = single_recording_context(
            tmp_path=tmp_path, frame_height=128, frame_width=128, frame_count=48, movie=movie, configure=configure
        )
        register_plane(context=context, workers=1, device=0)

        offsets = np.load(tmp_path / "output" / "cindra" / "plane_0" / "registration_data" / "rigid_y_offsets.npy")
        assert offsets.shape == (48,)

    def test_zero_gpu_batch_size_keeps_the_shared_batch_size(self, tmp_path, single_recording_context):
        """Verifies a device pass configured with no device batch stages the shared batch instead."""
        movie, _ = _build_shifted_movie(frame_count=48, height=128, width=128)

        def configure(configuration):
            configuration.registration.batch_size = 20
            configuration.registration.gpu_batch_size = 0

        context = single_recording_context(
            tmp_path=tmp_path, frame_height=128, frame_width=128, frame_count=48, movie=movie, configure=configure
        )
        register_plane(context=context, workers=1, device=0)

        offsets = np.load(tmp_path / "output" / "cindra" / "plane_0" / "registration_data" / "rigid_y_offsets.npy")
        assert offsets.shape == (48,)
