from dataclasses import field, dataclass
from collections.abc import Iterable, Iterator

import cupy
import numpy as np
from numpy.typing import NDArray as NDArray

from .batch import (
    ReferenceData as ReferenceData,
    BatchRegistrationResult as BatchRegistrationResult,
)
from .utils import (
    NORMALIZATION_EPSILON as NORMALIZATION_EPSILON,
    compute_upsampling_kernel as compute_upsampling_kernel,
)

_GPU_REMEDY: str
_TF32_VARIABLE: str
_MINIMUM_CORRELATION_RADIUS: int
_SNR_EPSILON: float
_SUBPIXEL_FACTOR: int
_UPSAMPLING_PADDING: int
_CORRELATION_BATCH_SIZE: int
_PIPELINE_DEPTH: int
_INT16_MINIMUM: int
_INT16_MAXIMUM: int

@dataclass(slots=True)
class _StagingSlot:
    host_buffers: dict[tuple[tuple[int, ...], str], NDArray[np.int16] | NDArray[np.float32]] = field(
        default_factory=dict
    )
    device_buffers: dict[tuple[tuple[int, ...], str], cupy.ndarray] = field(default_factory=dict)

@dataclass(frozen=True, slots=True)
class _DeviceBatchResult:
    frames: cupy.ndarray
    y_offsets: cupy.ndarray
    x_offsets: cupy.ndarray
    correlations: cupy.ndarray
    y_offsets_nonrigid: cupy.ndarray | None
    x_offsets_nonrigid: cupy.ndarray | None
    correlations_nonrigid: cupy.ndarray | None

@dataclass(frozen=True, slots=True)
class _NonrigidDeviceData:
    taper_mask: cupy.ndarray
    mean_offset: cupy.ndarray
    reference_kernel: cupy.ndarray
    block_rows: cupy.ndarray
    block_columns: cupy.ndarray
    smoothing_kernel: cupy.ndarray
    upsampling_kernel: cupy.ndarray
    upsampled_size: int
    block_counts: tuple[int, int]
    block_row_grid: cupy.ndarray
    block_column_grid: cupy.ndarray
    pixel_row_grid: cupy.ndarray
    pixel_column_grid: cupy.ndarray

class GpuRegistrationBackend:
    _device: int
    _frame_height: int
    _frame_width: int
    _normalization_weights: dict[int, cupy.ndarray]
    _input_slots: tuple[_StagingSlot, ...]
    _output_slots: tuple[_StagingSlot, ...]
    _staging_slot: int
    _compute_stream: cupy.cuda.Stream
    _transfer_stream: cupy.cuda.Stream
    _upload_events: tuple[cupy.cuda.Event, ...]
    _taper_mask: cupy.ndarray
    _mean_offset: cupy.ndarray
    _reference_kernel: cupy.ndarray
    _nonrigid_data: _NonrigidDeviceData | None
    def __init__(self, reference_data: ReferenceData, device: int) -> None: ...
    def __repr__(self) -> str: ...
    def register_batches(
        self,
        batches: Iterable[NDArray[np.int16] | NDArray[np.float32]],
        normalization_minimum: float,
        normalization_maximum: float,
        bidirectional_phase_offset: int,
        pre_smoothing_sigma: float,
        spatial_highpass_window: int,
        temporal_smoothing_sigma: float,
        maximum_offset_fraction: float,
        signal_to_noise_threshold: float,
        maximum_block_offset: float,
        *,
        one_photon_enabled: bool,
        nonrigid_enabled: bool,
    ) -> Iterator[BatchRegistrationResult]: ...
    def apply_precomputed_offsets(
        self,
        frames: NDArray[np.int16] | NDArray[np.float32],
        y_offsets: NDArray[np.int32],
        x_offsets: NDArray[np.int32],
        y_offsets_nonrigid: NDArray[np.float32] | None,
        x_offsets_nonrigid: NDArray[np.float32] | None,
        bidirectional_phase_offset: int,
        *,
        bidirectional_phase_corrected: bool,
        nonrigid_enabled: bool,
    ) -> tuple[NDArray[np.int16] | NDArray[np.float32], NDArray[np.float32] | None]: ...
    def _upload_batch(self, frames: NDArray[np.int16] | NDArray[np.float32], slot: int) -> cupy.ndarray: ...
    def _register_device_batch(
        self,
        staged_frames: cupy.ndarray,
        normalization_minimum: float,
        normalization_maximum: float,
        bidirectional_phase_offset: int,
        pre_smoothing_sigma: float,
        spatial_highpass_window: int,
        temporal_smoothing_sigma: float,
        maximum_offset_fraction: float,
        signal_to_noise_threshold: float,
        maximum_block_offset: float,
        *,
        one_photon_enabled: bool,
        nonrigid_enabled: bool,
    ) -> _DeviceBatchResult: ...
    def _download_batch_result(
        self, device_result: _DeviceBatchResult, slot: int, *, narrow_frames: bool
    ) -> BatchRegistrationResult: ...
    def _download_frames(
        self, device_frames: cupy.ndarray, slot: int, *, narrow_frames: bool
    ) -> tuple[NDArray[np.int16] | NDArray[np.float32], NDArray[np.float32] | None]: ...
    def _resolve_nonrigid_data(self) -> _NonrigidDeviceData: ...
    def _compute_rigid_offsets(
        self,
        frames: cupy.ndarray,
        reference_kernel: cupy.ndarray,
        maximum_offset_fraction: float,
        temporal_smoothing_sigma: float,
    ) -> tuple[cupy.ndarray, cupy.ndarray, cupy.ndarray]: ...
    def _compute_nonrigid_offsets(
        self, nonrigid_data: _NonrigidDeviceData, frames: cupy.ndarray, snr_threshold: float, maximum_offset: float
    ) -> tuple[cupy.ndarray, cupy.ndarray, cupy.ndarray]: ...
    def _locate_subpixel_peaks(
        self, nonrigid_data: _NonrigidDeviceData, smoothed_correlation: cupy.ndarray, correlation_radius: int
    ) -> tuple[cupy.ndarray, cupy.ndarray, cupy.ndarray]: ...
    def _apply_nonrigid_correction(
        self,
        nonrigid_data: _NonrigidDeviceData,
        frames: cupy.ndarray,
        y_block_offsets: cupy.ndarray,
        x_block_offsets: cupy.ndarray,
    ) -> cupy.ndarray: ...
    def _apply_spatial_high_pass(self, data: cupy.ndarray, window: int) -> cupy.ndarray: ...
    def _resolve_normalization_weights(self, window: int) -> cupy.ndarray: ...
    @staticmethod
    def _apply_mask(frames: cupy.ndarray, mask: cupy.ndarray, offset: cupy.ndarray) -> cupy.ndarray: ...
    @staticmethod
    def _apply_phase_correlation(frames: cupy.ndarray, kernel: cupy.ndarray) -> cupy.ndarray: ...
    @staticmethod
    def _apply_temporal_smoothing(frames: cupy.ndarray, sigma: float) -> cupy.ndarray: ...
    @staticmethod
    def _apply_spatial_smoothing(data: cupy.ndarray, window: int) -> cupy.ndarray: ...
    @staticmethod
    def _clip_intensities(
        frames: cupy.ndarray, normalization_minimum: float, normalization_maximum: float
    ) -> cupy.ndarray: ...
    @staticmethod
    def _translate_frames(frames: cupy.ndarray, y_offsets: cupy.ndarray, x_offsets: cupy.ndarray) -> cupy.ndarray: ...
    @staticmethod
    def _apply_bidirectional_phase_correction(frames: cupy.ndarray, bidirectional_phase_offset: int) -> None: ...
    @staticmethod
    def _rearrange_correlation_quadrants(correlation_data: cupy.ndarray, radius: int) -> cupy.ndarray: ...
    @staticmethod
    def _sample_bilinear(
        source: cupy.ndarray, y_coordinates: cupy.ndarray, x_coordinates: cupy.ndarray
    ) -> cupy.ndarray: ...
    @staticmethod
    def _compute_correlation_snr(correlation_data: cupy.ndarray, padding: int) -> cupy.ndarray: ...

def _require_gpu_runtime() -> None: ...
def _verify_tf32_disabled() -> None: ...
def _upload_nonrigid_data(
    reference_data: ReferenceData, frame_height: int, frame_width: int
) -> _NonrigidDeviceData | None: ...
def _resolve_pinned_buffer(
    slot: _StagingSlot, shape: tuple[int, ...], dtype: np.dtype[np.int16] | np.dtype[np.float32]
) -> NDArray[np.int16] | NDArray[np.float32]: ...
def _resolve_device_buffer(
    slot: _StagingSlot, shape: tuple[int, ...], dtype: np.dtype[np.int16] | np.dtype[np.float32]
) -> cupy.ndarray: ...
