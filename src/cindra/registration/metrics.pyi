import numpy as np
from numpy.typing import NDArray as NDArray

from ..io import BinaryFile as BinaryFile
from .rigid import (
    translate_frame as translate_frame,
    apply_edge_taper as apply_edge_taper,
    compute_edge_taper as compute_edge_taper,
    compute_rigid_offsets as compute_rigid_offsets,
    compute_phase_correlation_kernel as compute_phase_correlation_kernel,
)
from .utils import (
    apply_spatial_high_pass as apply_spatial_high_pass,
    apply_spatial_smoothing as apply_spatial_smoothing,
)
from .nonrigid import (
    compute_nonrigid_offsets as compute_nonrigid_offsets,
    compute_nonrigid_reference_data as compute_nonrigid_reference_data,
)
from ..detection import compute_registration_blocks as compute_registration_blocks
from ..dataclasses import RuntimeContext as RuntimeContext
from .bidiphase_correction import apply_bidirectional_phase_correction as apply_bidirectional_phase_correction

_MINIMUM_SAMPLE_COUNT: int
_MAXIMUM_SAMPLE_COUNT: int
_MAXIMUM_HEIGHT_FOR_LARGE_SAMPLE: int
_MAXIMUM_WIDTH_FOR_LARGE_SAMPLE: int

def compute_pc_metrics(context: RuntimeContext) -> None: ...
def _compute_pc_extremes(
    frames: NDArray[np.float32], num_extreme_frames: int, num_components: int
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]: ...
def _register_pc_extremes(
    pc_low: NDArray[np.float32],
    pc_high: NDArray[np.float32],
    *,
    bidirectional_corrected: bool,
    spatial_highpass_window: int | None = None,
    pre_smoothing_window: int | None = None,
    smoothing_sigma: float = 1.15,
    block_size: tuple[int, int] = (128, 128),
    maximum_offset_fraction: float = 0.1,
    maximum_nonrigid_offset: float = 5.0,
    one_photon_mode: bool = False,
    snr_threshold: float = 1.2,
    nonrigid_enabled: bool = True,
    bidirectional_phase_offset: int = 0,
    edge_taper_slope: float = 40.0,
    workers: int = -1,
) -> NDArray[np.float32]: ...
