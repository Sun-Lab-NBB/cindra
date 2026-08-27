"""Contains tests for the CUDA device discovery and verification assets."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from ataraxis_base_utilities import error_format

from cindra.orchestration import (
    GPU_REMEDY,
    GpuStatus,
    GpuSummary,
    gpu as gpu_module,
    resolve_gpu_devices,
    resolve_device_budget,
)
from cindra.orchestration.gpu import GpuDevice, _describe_devices, verify_gpu_runtime, _probe_device_transform


def _make_cupy(device_count=2, properties=None, probe_error=None, count_error=None, name=b"NVIDIA RTX A6000"):
    """Builds a stand-in for the CuPy binding that answers the calls the discovery makes."""
    recorded = {"transforms": 0, "synchronized": 0}

    def get_device_count():
        if count_error is not None:
            raise count_error
        return device_count

    def get_device_properties(index):
        if properties is not None:
            return properties(index)
        return {"name": name, "totalGlobalMem": (index + 1) * 1024**3, "major": 8, "minor": 6}

    def rfft2(array, axes):
        if probe_error is not None:
            raise probe_error
        recorded["transforms"] += 1
        return array

    def synchronize():
        recorded["synchronized"] += 1

    binding = SimpleNamespace(
        float32="float32",
        zeros=lambda shape, dtype: SimpleNamespace(shape=shape, dtype=dtype),
        fft=SimpleNamespace(rfft2=rfft2),
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(getDeviceCount=get_device_count, getDeviceProperties=get_device_properties),
            Stream=SimpleNamespace(null=SimpleNamespace(synchronize=synchronize)),
        ),
    )
    binding.recorded = recorded
    return binding


class TestGpuSummary:
    """Tests the reporting surface of the device resolution summary."""

    def test_available_summary_reports_every_device(self):
        """Verifies that an available summary names each device with its index and memory."""
        summary = GpuSummary(
            status=GpuStatus.AVAILABLE,
            devices=(GpuDevice(index=0, name="A6000", total_memory_mb=48538, compute_capability="8.6"),),
            detail="",
        )
        assert summary.available
        assert summary.remedy == ""
        assert "1 CUDA device(s) available" in summary.describe()
        assert "0: A6000 (48538 MB)" in summary.describe()

    @pytest.mark.parametrize(
        ("status", "expects_remedy"),
        [
            (GpuStatus.RUNTIME_MISSING, True),
            (GpuStatus.LIBRARIES_MISSING, True),
            (GpuStatus.NO_DEVICES, True),
            (GpuStatus.UNSUPPORTED_PLATFORM, False),
        ],
    )
    def test_unavailable_summary_names_the_remedy_only_where_one_applies(self, status, expects_remedy):
        """Verifies that every unusable outcome except the unsupported platform names the resolving command."""
        summary = GpuSummary(status=status, devices=(), detail="reason")
        assert not summary.available
        assert summary.remedy == (GPU_REMEDY if expects_remedy else "")
        assert summary.describe() == "no usable CUDA device: reason"


class TestDeviceResolution:
    """Tests the device resolution against a stand-in for the CuPy binding."""

    def test_resolution_reports_every_device_the_runtime_exposes(self, monkeypatch):
        """Verifies that the resolution describes each device the runtime reports."""
        monkeypatch.setattr(gpu_module.sys, "platform", "linux")
        binding = _make_cupy(device_count=2)
        monkeypatch.setattr(gpu_module, "cupy", binding)

        summary = resolve_gpu_devices()

        assert summary.status == GpuStatus.AVAILABLE
        assert summary.detail == ""
        assert [device.index for device in summary.devices] == [0, 1]
        assert summary.devices[0].name == "NVIDIA RTX A6000"
        assert summary.devices[0].total_memory_mb == 1024
        assert summary.devices[1].total_memory_mb == 2048
        assert summary.devices[0].compute_capability == "8.6"

    def test_resolution_probes_the_runtime_with_a_transform(self, monkeypatch):
        """Verifies that the resolution runs a transform, because CuPy resolves cuFFT on first use."""
        monkeypatch.setattr(gpu_module.sys, "platform", "linux")
        binding = _make_cupy()
        monkeypatch.setattr(gpu_module, "cupy", binding)

        resolve_gpu_devices()

        assert binding.recorded["transforms"] == 1
        assert binding.recorded["synchronized"] == 1

    def test_resolution_decodes_a_byte_string_device_name(self, monkeypatch):
        """Verifies that a device name the runtime reports as bytes reaches the caller as text."""
        monkeypatch.setattr(gpu_module.sys, "platform", "linux")
        monkeypatch.setattr(gpu_module, "cupy", _make_cupy(device_count=1, name="A6000"))

        assert resolve_gpu_devices().devices[0].name == "A6000"

    def test_resolution_refuses_macos(self, monkeypatch):
        """Verifies that macOS resolves to the unsupported platform outcome, since no CuPy wheel exists there."""
        monkeypatch.setattr(gpu_module.sys, "platform", "darwin")

        summary = resolve_gpu_devices()

        assert summary.status == GpuStatus.UNSUPPORTED_PLATFORM
        assert "no macOS wheel" in summary.detail

    def test_resolution_reports_a_missing_runtime(self, monkeypatch):
        """Verifies that an absent CuPy distribution resolves to the missing runtime outcome."""
        monkeypatch.setattr(gpu_module.sys, "platform", "linux")
        monkeypatch.setattr(gpu_module, "cupy", None)

        summary = resolve_gpu_devices()

        assert summary.status == GpuStatus.RUNTIME_MISSING
        assert summary.devices == ()

    def test_resolution_reports_a_raising_device_count_as_no_devices(self, monkeypatch):
        """Verifies that the device count query raising resolves to the no-device outcome carrying its reason."""
        monkeypatch.setattr(gpu_module.sys, "platform", "linux")
        monkeypatch.setattr(gpu_module, "cupy", _make_cupy(count_error=RuntimeError("cudaErrorNoDevice")))

        summary = resolve_gpu_devices()

        assert summary.status == GpuStatus.NO_DEVICES
        assert summary.detail == "cudaErrorNoDevice"

    def test_resolution_reports_a_zero_device_count(self, monkeypatch):
        """Verifies that a runtime reporting no device resolves to the no-device outcome."""
        monkeypatch.setattr(gpu_module.sys, "platform", "linux")
        monkeypatch.setattr(gpu_module, "cupy", _make_cupy(device_count=0))

        summary = resolve_gpu_devices()

        assert summary.status == GpuStatus.NO_DEVICES
        assert "reports no device" in summary.detail

    def test_resolution_reports_a_failing_probe_as_missing_libraries(self, monkeypatch):
        """Verifies that a transform failing on a present device resolves to the missing libraries outcome."""
        monkeypatch.setattr(gpu_module.sys, "platform", "linux")
        monkeypatch.setattr(gpu_module, "cupy", _make_cupy(probe_error=OSError("libcufft.so not found")))

        summary = resolve_gpu_devices()

        assert summary.status == GpuStatus.LIBRARIES_MISSING
        assert "libcufft" in summary.detail


class TestDeviceBudget:
    """Tests the device budget the batch engine schedules GPU registration jobs across."""

    def test_budget_counts_every_usable_device(self, monkeypatch):
        """Verifies that the budget reports one entry per usable device."""
        monkeypatch.setattr(gpu_module.sys, "platform", "linux")
        monkeypatch.setattr(gpu_module, "cupy", _make_cupy(device_count=3))
        assert resolve_device_budget() == 3

    def test_budget_is_zero_when_no_device_is_usable(self, monkeypatch):
        """Verifies that the budget reports zero when the runtime reaches no device."""
        monkeypatch.setattr(gpu_module.sys, "platform", "linux")
        monkeypatch.setattr(gpu_module, "cupy", None)
        assert resolve_device_budget() == 0


class TestRuntimeVerification:
    """Tests the gate every entry point calls before dispatching a GPU registration job."""

    def test_verification_returns_on_a_usable_runtime(self, monkeypatch):
        """Verifies that the gate returns without raising when a device is usable."""
        monkeypatch.setattr(gpu_module.sys, "platform", "linux")
        monkeypatch.setattr(gpu_module, "cupy", _make_cupy())
        assert verify_gpu_runtime() is None

    def test_verification_refuses_an_unusable_runtime(self, monkeypatch):
        """Verifies that the gate errors with the reason, the remedy, and the CPU backend fallback."""
        monkeypatch.setattr(gpu_module.sys, "platform", "linux")
        monkeypatch.setattr(gpu_module, "cupy", None)

        expected_message = (
            "Unable to run the registration stage on the GPU backend. The host exposes no usable CUDA device: the "
            f"CuPy distribution is not installed. {GPU_REMEDY} Set 'registration.backend' to 'cpu' to run the stage "
            "on the CPU backend instead."
        )
        with pytest.raises(RuntimeError, match=error_format(message=expected_message)):
            verify_gpu_runtime()


class TestDeviceHelpers:
    """Tests the private helpers the resolution builds its summary from."""

    def test_description_reads_every_reported_device(self, monkeypatch):
        """Verifies that the descriptor reads one entry per device index."""
        monkeypatch.setattr(gpu_module, "cupy", _make_cupy(device_count=2))
        devices = _describe_devices(device_count=2)
        assert len(devices) == 2
        assert devices[1].index == 1

    def test_probe_transforms_a_square_array(self, monkeypatch):
        """Verifies that the probe allocates a square array and transforms it."""
        binding = _make_cupy()
        monkeypatch.setattr(gpu_module, "cupy", binding)
        _probe_device_transform()
        assert binding.recorded["transforms"] == 1
