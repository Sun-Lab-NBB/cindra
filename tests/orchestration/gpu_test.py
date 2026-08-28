"""Contains tests for the CUDA device discovery and verification assets."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from ataraxis_base_utilities import error_format

from cindra.orchestration import (
    GPU_REMEDY,
    ALL_DEVICES_REQUEST,
    GpuStatus,
    GpuSummary,
    gpu as gpu_module,
    resolve_gpu_devices,
)
from cindra.orchestration.gpu import (
    GpuDevice,
    _describe_devices,
    verify_gpu_runtime,
    resolve_device_budget,
    _probe_device_transform,
)


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

    def test_resolution_passes_a_text_device_name_through(self, monkeypatch):
        """Verifies that a device name the runtime already reports as text reaches the caller unchanged."""
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

    def test_resolution_probes_the_named_device(self, monkeypatch):
        """Verifies that a named index transforms the probe array inside that device's context."""
        monkeypatch.setattr(gpu_module.sys, "platform", "linux")
        binding = _make_cupy(device_count=2)
        monkeypatch.setattr(gpu_module, "cupy", binding)

        summary = resolve_gpu_devices(device=1)

        assert summary.status == GpuStatus.AVAILABLE
        assert binding.recorded["entered_devices"] == [1]
        assert binding.recorded["transforms"] == 1

    @pytest.mark.parametrize("device", [-1, 2, 9])
    def test_resolution_falls_back_to_the_default_device_for_an_absent_index(self, monkeypatch, device):
        """Verifies that an index outside the reported range probes the default device rather than raising."""
        monkeypatch.setattr(gpu_module.sys, "platform", "linux")
        binding = _make_cupy(device_count=2)
        monkeypatch.setattr(gpu_module, "cupy", binding)

        summary = resolve_gpu_devices(device=device)

        assert summary.status == GpuStatus.AVAILABLE
        assert binding.recorded["entered_devices"] == []
        assert binding.recorded["transforms"] == 1

    def test_resolution_reports_a_failing_probe_as_missing_libraries(self, monkeypatch):
        """Verifies that a transform failing on a present device resolves to the missing libraries outcome."""
        monkeypatch.setattr(gpu_module.sys, "platform", "linux")
        monkeypatch.setattr(gpu_module, "cupy", _make_cupy(probe_error=OSError("libcufft.so not found")))

        summary = resolve_gpu_devices()

        assert summary.status == GpuStatus.LIBRARIES_MISSING
        assert "libcufft" in summary.detail


class TestDeviceBudget:
    """Tests the device budget across which the batch engine schedules GPU registration jobs."""

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
    """Tests the gate every entry point calls before dispatching a registration job onto a CUDA device."""

    def test_verification_returns_on_a_usable_runtime(self, monkeypatch):
        """Verifies that the gate returns without raising when a device is usable."""
        monkeypatch.setattr(gpu_module.sys, "platform", "linux")
        monkeypatch.setattr(gpu_module, "cupy", _make_cupy())
        assert verify_gpu_runtime() is None

    @pytest.mark.parametrize("device", [0, 1])
    def test_verification_accepts_an_index_the_host_exposes(self, monkeypatch, device):
        """Verifies that the gate returns for every device index the host reports."""
        monkeypatch.setattr(gpu_module.sys, "platform", "linux")
        monkeypatch.setattr(gpu_module, "cupy", _make_cupy(device_count=2))
        assert verify_gpu_runtime(device=device) is None

    def test_verification_refuses_an_unusable_runtime(self, monkeypatch):
        """Verifies that the gate errors with the reason, the remedy, and the host CPU fallback."""
        monkeypatch.setattr(gpu_module.sys, "platform", "linux")
        monkeypatch.setattr(gpu_module, "cupy", None)

        expected_message = (
            "Unable to run the registration stage on a CUDA device. The host exposes no usable CUDA device: the "
            f"CuPy distribution is not installed. {GPU_REMEDY} Omit the registration device argument to run the "
            "stage on the host CPU instead."
        )
        with pytest.raises(RuntimeError, match=error_format(message=expected_message)):
            verify_gpu_runtime()

    def test_verification_refuses_an_index_the_host_does_not_expose(self, monkeypatch):
        """Verifies that a usable runtime still refuses an index no device on the host carries."""
        monkeypatch.setattr(gpu_module.sys, "platform", "linux")
        monkeypatch.setattr(gpu_module, "cupy", _make_cupy(device_count=2))

        expected_message = (
            "Unable to run the registration stage on CUDA device 5. The host exposes no device carrying that index. "
            "Available device indices: [0, 1]."
        )
        with pytest.raises(ValueError, match=error_format(message=expected_message)):
            verify_gpu_runtime(device=5)

    def test_unusable_runtime_is_reported_before_the_named_index(self, monkeypatch):
        """Verifies that a host reaching no device reports the installation rather than the index it was given."""
        monkeypatch.setattr(gpu_module.sys, "platform", "linux")
        monkeypatch.setattr(gpu_module, "cupy", None)

        with pytest.raises(RuntimeError, match=error_format(message="The host exposes no usable CUDA device")):
            verify_gpu_runtime(device=5)


class TestAllDevicesRequest:
    """Tests the sentinel the session device list carries to name every device the host exposes."""

    def test_sentinel_is_the_negative_index_no_device_carries(self):
        """Verifies that the all-devices request mirrors the all-cores request the allocator declares."""
        assert ALL_DEVICES_REQUEST == -1


class TestDeviceHelpers:
    """Tests the private helpers from which the resolution builds its summary."""

    def test_description_reads_every_reported_device(self, monkeypatch):
        """Verifies that the descriptor reads one entry per device index."""
        monkeypatch.setattr(gpu_module, "cupy", _make_cupy(device_count=2))
        devices = _describe_devices(device_count=2)
        assert len(devices) == 2
        assert devices[1].index == 1

    def test_probe_transforms_a_square_array_on_the_default_device(self, monkeypatch):
        """Verifies that the probe allocates a square array and transforms it without naming a device."""
        binding = _make_cupy()
        monkeypatch.setattr(gpu_module, "cupy", binding)
        _probe_device_transform(device=None)
        assert binding.recorded["transforms"] == 1
        assert binding.recorded["entered_devices"] == []

    def test_probe_transforms_inside_the_named_device_context(self, monkeypatch):
        """Verifies that a named index transforms the probe array inside that device's own context."""
        binding = _make_cupy(device_count=2)
        monkeypatch.setattr(gpu_module, "cupy", binding)
        _probe_device_transform(device=1)
        assert binding.recorded["transforms"] == 1
        assert binding.recorded["entered_devices"] == [1]


class _DeviceContext:
    """Stands in for the CuPy device context manager, recording the index inside which a probe transforms."""

    def __init__(self, index, recorded):
        self.index = index
        self._recorded = recorded

    def __enter__(self):
        self._recorded["entered_devices"].append(self.index)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        return False


def _make_cupy(device_count=2, properties=None, probe_error=None, count_error=None, name=b"NVIDIA RTX A6000"):
    """Builds a stand-in for the CuPy binding that answers the calls the discovery makes."""
    recorded = {"transforms": 0, "synchronized": 0, "entered_devices": []}

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
            Device=lambda index: _DeviceContext(index=index, recorded=recorded),
        ),
    )
    binding.recorded = recorded
    return binding
