from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..io import BinaryFile as BinaryFile
from .rigid import (
    translate_frame as translate_frame,
    apply_edge_taper as apply_edge_taper,
    compute_edge_taper as compute_edge_taper,
    compute_rigid_offsets as compute_rigid_offsets,
    compute_phase_correlation_kernel as compute_phase_correlation_kernel,
)
from .utils import (
    combine_rigid_offsets as combine_rigid_offsets,
    apply_spatial_high_pass as apply_spatial_high_pass,
    apply_spatial_smoothing as apply_spatial_smoothing,
    combine_nonrigid_offsets as combine_nonrigid_offsets,
)
from .metrics import compute_pc_metrics as compute_pc_metrics
from .nonrigid import (
    compute_nonrigid_offsets as compute_nonrigid_offsets,
    apply_nonrigid_correction as apply_nonrigid_correction,
    compute_nonrigid_reference_data as compute_nonrigid_reference_data,
)
from ..detection import compute_registration_blocks as compute_registration_blocks
from ..dataclasses import RuntimeContext as RuntimeContext
from .bidiphase_correction import (
    compute_bidirectional_phase_offset as compute_bidirectional_phase_offset,
    apply_bidirectional_phase_correction as apply_bidirectional_phase_correction,
)

_MINIMUM_REGISTRATION_METRIC_FRAMES: int
_BAD_FRAME_FRACTION_THRESHOLD: float
_MAXIMUM_MEDIAN_FILTER_WINDOW: int
type RegistrationBlocks = tuple[
    list[NDArray[np.int32]], list[NDArray[np.int32]], tuple[int, int], tuple[int, int], NDArray[np.float32]
]

def register_plane(context: RuntimeContext) -> None: ...

@dataclass(frozen=True, slots=True)
class _ReferenceData:
    taper_mask: NDArray[np.float32]
    mean_offset: NDArray[np.float32]
    reference_kernel: NDArray[np.complex64]
    taper_mask_nonrigid: NDArray[np.float32] | None
    mean_offset_nonrigid: NDArray[np.float32] | None
    reference_kernel_nonrigid: NDArray[np.complex64] | None
    blocks: RegistrationBlocks | None

@dataclass(frozen=True, slots=True)
class _BatchRegistrationResult:
    frames: NDArray[np.float32]
    y_offsets: NDArray[np.int32]
    x_offsets: NDArray[np.int32]
    correlations: NDArray[np.float32]
    y_offsets_nonrigid: NDArray[np.float32] | None
    x_offsets_nonrigid: NDArray[np.float32] | None
    correlations_nonrigid: NDArray[np.float32] | None

def _compute_crop(
    x_offsets: NDArray[np.int32],
    y_offsets: NDArray[np.int32],
    correlations: NDArray[np.float32],
    bad_frame_threshold: float,
    bad_frames: NDArray[np.bool_],
    maximum_offset_fraction: float,
    frame_height: int,
    frame_width: int,
) -> tuple[NDArray[np.bool_], tuple[int, int], tuple[int, int]]: ...
def _pick_initial_reference(frames: NDArray[np.float32], top_correlations: int = 20) -> NDArray[np.float32]: ...
def _compute_reference(
    frames: NDArray[np.float32],
    one_photon_enabled: bool,
    pre_smoothing_sigma: float,
    spatial_highpass_window: int,
    edge_taper_pixels: float,
    spatial_smoothing_sigma: float,
    maximum_offset_fraction: float,
    temporal_smoothing_sigma: float,
    workers: int,
) -> NDArray[np.float32]: ...
def _register_frames_batch(
    reference_data: _ReferenceData,
    frames: NDArray[np.float32],
    normalization_minimum: float,
    normalization_maximum: float,
    bidirectional_phase_offset: int,
    one_photon_enabled: bool,
    pre_smoothing_sigma: float,
    spatial_highpass_window: int,
    temporal_smoothing_sigma: float,
    maximum_offset_fraction: float,
    nonrigid_enabled: bool,
    signal_to_noise_threshold: float,
    maximum_block_offset: float,
    workers: int,
) -> _BatchRegistrationResult: ...
def _apply_precomputed_offsets_batch(
    frames: NDArray[np.float32],
    y_offsets: NDArray[np.int32],
    x_offsets: NDArray[np.int32],
    y_offsets_nonrigid: NDArray[np.float32] | None,
    x_offsets_nonrigid: NDArray[np.float32] | None,
    blocks: RegistrationBlocks | None,
    bidirectional_phase_offset: int,
    bidirectional_phase_corrected: bool,
    nonrigid_enabled: bool,
) -> NDArray[np.float32]: ...
def _register_alignment_channel(context: RuntimeContext) -> None: ...
def _register_secondary_channel(context: RuntimeContext) -> None: ...
