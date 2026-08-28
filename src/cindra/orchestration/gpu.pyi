from enum import StrEnum
from dataclasses import dataclass

GPU_REMEDY: str
ALL_DEVICES_REQUEST: int
_BYTES_PER_MEGABYTE: int
_PROBE_DIMENSION: int

class GpuStatus(StrEnum):
    AVAILABLE = "available"
    UNSUPPORTED_PLATFORM = "unsupported_platform"
    RUNTIME_MISSING = "runtime_missing"
    LIBRARIES_MISSING = "libraries_missing"
    NO_DEVICES = "no_devices"

@dataclass(frozen=True, slots=True)
class GpuDevice:
    index: int
    name: str
    total_memory_mb: int
    compute_capability: str

@dataclass(frozen=True, slots=True)
class GpuSummary:
    status: GpuStatus
    devices: tuple[GpuDevice, ...]
    detail: str
    @property
    def available(self) -> bool: ...
    @property
    def remedy(self) -> str: ...
    def describe(self) -> str: ...

def resolve_gpu_devices(device: int | None = None) -> GpuSummary: ...
def verify_gpu_runtime(device: int | None = None) -> None: ...
def resolve_device_budget() -> int: ...
def _describe_devices(device_count: int) -> tuple[GpuDevice, ...]: ...
def _probe_device_transform(device: int | None) -> None: ...
def _transform_probe_array() -> None: ...
