"""Provides the CUDA device discovery and verification that gate the GPU registration backend."""

import sys
from enum import StrEnum
from dataclasses import dataclass

from ataraxis_base_utilities import console

try:
    import cupy
except ImportError:  # pragma: no cover
    # The CuPy distribution publishes no macOS wheel, so the dependency marker excludes darwin and this module has to
    # stay importable there. Guarding the import also covers a Linux or Windows host whose installation was trimmed.
    # Every entry point that reaches a device calls verify_gpu_runtime() before touching the name below, and
    # resolve_gpu_devices() reports RUNTIME_MISSING while the name is None. Only one of the two branches runs on any
    # single host, so the fallback stays out of coverage measurement.
    cupy = None

GPU_REMEDY: str = (
    "Install the CuPy build matching the CUDA version the local driver runs, as 'cupy-cuda13x[ctk]' for CUDA 13 or "
    "'cupy-cuda12x[ctk]' for CUDA 12, and run 'cindra gpu' to report what the host exposes."
)
"""The remedy an unusable GPU runtime reports, named by every message this module produces.

Notes:
    The 'ctk' extra carries the CUDA math libraries CuPy resolves on first use. A bare CuPy installation imports and
    reports its devices, then raises at the first transform, so the extra is what separates a reachable device from an
    importable module.
"""

_BYTES_PER_MEGABYTE: int = 1024**2
"""The number of bytes in one megabyte, used to convert the device memory counter into a reported size."""

_PROBE_DIMENSION: int = 8
"""The side length of the square array the runtime probe transforms.

Notes:
    The probe runs a real FFT rather than an allocation alone, because CuPy resolves the cuFFT shared library on first
    use rather than at import. A host holding the CuPy distribution without the CUDA math libraries therefore allocates
    successfully and fails at the first transform the registration backend performs.
"""


class GpuStatus(StrEnum):
    """Defines the outcome of a request to resolve the CUDA devices the GPU registration backend runs on."""

    AVAILABLE = "available"
    """At least one device is present and the runtime performs a transform on it."""
    UNSUPPORTED_PLATFORM = "unsupported_platform"
    """The host runs macOS, for which the CuPy distribution publishes no wheel."""
    RUNTIME_MISSING = "runtime_missing"
    """The CuPy distribution is absent, so no device is reachable."""
    LIBRARIES_MISSING = "libraries_missing"
    """The CuPy distribution is present, and the CUDA math libraries it loads on first use are absent."""
    NO_DEVICES = "no_devices"
    """The CUDA runtime loads, and it reports no device the driver exposes."""


@dataclass(frozen=True, slots=True)
class GpuDevice:
    """Describes one CUDA device the host exposes."""

    index: int
    """The zero-based device index the registration backend selects the device by."""
    name: str
    """The marketing name the driver reports for the device."""
    total_memory_mb: int
    """The total device memory, in megabytes."""
    compute_capability: str
    """The compute capability of the device, written as 'major.minor'."""


@dataclass(frozen=True, slots=True)
class GpuSummary:
    """Summarizes the CUDA devices the host exposes and whether the GPU registration backend runs on them."""

    status: GpuStatus
    """The outcome of the resolution."""
    devices: tuple[GpuDevice, ...]
    """The devices the backend runs on, empty for every outcome other than AVAILABLE."""
    detail: str
    """The reason the runtime is unusable, empty when the status is AVAILABLE."""

    @property
    def available(self) -> bool:
        """Returns True when the GPU registration backend runs on this host."""
        return self.status == GpuStatus.AVAILABLE

    @property
    def remedy(self) -> str:
        """Returns the command that resolves the runtime, empty when the runtime is already usable."""
        if self.available or self.status == GpuStatus.UNSUPPORTED_PLATFORM:
            return ""
        return GPU_REMEDY

    def describe(self) -> str:
        """Builds a one-line human-readable summary of what the resolution found.

        Returns:
            A compact description of the outcome.
        """
        if self.available:
            names = ", ".join(f"{device.index}: {device.name} ({device.total_memory_mb} MB)" for device in self.devices)
            return f"{len(self.devices)} CUDA device(s) available. {names}."
        return f"no usable CUDA device: {self.detail}"


def resolve_gpu_devices() -> GpuSummary:
    """Resolves the CUDA devices the GPU registration backend runs on.

    Probes the runtime by transforming a small array on the first device, because CuPy resolves the cuFFT shared
    library on first use rather than at import.

    Returns:
        The summary of the devices found and of the reason no device is usable.
    """
    if sys.platform == "darwin":
        return GpuSummary(
            status=GpuStatus.UNSUPPORTED_PLATFORM,
            devices=(),
            detail="the CuPy distribution publishes no macOS wheel, so registration runs on the CPU backend",
        )

    if cupy is None:
        return GpuSummary(
            status=GpuStatus.RUNTIME_MISSING,
            devices=(),
            detail="the CuPy distribution is not installed",
        )

    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
    except Exception as error:
        # getDeviceCount raises CUDARuntimeError rather than returning zero when the driver exposes no device, and it
        # raises a loader error when the CUDA runtime library itself is absent. Both reach the caller as one status,
        # because the same command resolves them.
        return GpuSummary(status=GpuStatus.NO_DEVICES, devices=(), detail=str(error))

    if device_count == 0:
        return GpuSummary(
            status=GpuStatus.NO_DEVICES, devices=(), detail="the CUDA runtime reports no device the driver exposes"
        )

    try:
        devices = _describe_devices(device_count=device_count)
        _probe_device_transform()
    except Exception as error:
        return GpuSummary(status=GpuStatus.LIBRARIES_MISSING, devices=(), detail=str(error))

    return GpuSummary(status=GpuStatus.AVAILABLE, devices=devices, detail="")


def verify_gpu_runtime() -> None:
    """Verifies that the GPU registration backend runs on this host, aborting the caller when it does not.

    Raises:
        RuntimeError: If no CUDA device is usable.
    """
    summary = resolve_gpu_devices()
    if summary.available:
        return

    message = (
        f"Unable to run the registration stage on the GPU backend. The host exposes no usable CUDA device: "
        f"{summary.detail}. {summary.remedy} Set 'registration.backend' to 'cpu' to run the stage on the CPU backend "
        f"instead."
    )
    console.error(message=message, error=RuntimeError)


def resolve_device_budget() -> int:
    """Resolves the number of CUDA devices the batch engine schedules GPU registration jobs across.

    Returns:
        The count of usable devices, which is zero when no device is usable.
    """
    return len(resolve_gpu_devices().devices)


def _describe_devices(device_count: int) -> tuple[GpuDevice, ...]:
    """Reads the properties of every CUDA device the runtime reports.

    Args:
        device_count: The number of devices the runtime reports.

    Returns:
        One descriptor per device, ordered by device index.
    """
    devices = []
    for index in range(device_count):
        properties = cupy.cuda.runtime.getDeviceProperties(index)
        name = properties["name"]
        devices.append(
            GpuDevice(
                index=index,
                name=name.decode() if isinstance(name, bytes) else str(name),
                total_memory_mb=int(properties["totalGlobalMem"]) // _BYTES_PER_MEGABYTE,
                compute_capability=f"{properties['major']}.{properties['minor']}",
            )
        )
    return tuple(devices)


def _probe_device_transform() -> None:
    """Transforms a small array on the first device to resolve the CUDA math libraries the backend loads."""
    probe = cupy.zeros((_PROBE_DIMENSION, _PROBE_DIMENSION), dtype=cupy.float32)
    cupy.fft.rfft2(probe, axes=(-2, -1))
    cupy.cuda.Stream.null.synchronize()
