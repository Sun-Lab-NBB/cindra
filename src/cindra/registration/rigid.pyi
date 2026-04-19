import numpy as np
from numpy.typing import NDArray as NDArray

from .utils import (
    NORMALIZATION_EPSILON as NORMALIZATION_EPSILON,
    apply_mask as apply_mask,
    compute_reference_fft as compute_reference_fft,
    apply_phase_correlation as apply_phase_correlation,
    apply_temporal_smoothing as apply_temporal_smoothing,
    compute_gaussian_frequency_filter as compute_gaussian_frequency_filter,
)
from ..detection import compute_spatial_taper_mask as compute_spatial_taper_mask

def compute_edge_taper(
    reference_image: NDArray[np.float32], taper_slope: float
) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
def apply_edge_taper(
    frames: NDArray[np.float32], taper_mask: NDArray[np.float32], mean_offset: NDArray[np.float32]
) -> NDArray[np.float32]: ...
def compute_phase_correlation_kernel(
    reference_image: NDArray[np.float32], smoothing_sigma: float = 0.0
) -> NDArray[np.complex64]: ...
def compute_rigid_offsets(
    frames: NDArray[np.float32],
    reference_kernel: NDArray[np.complex64],
    maximum_offset_fraction: float,
    temporal_smoothing_sigma: float,
    workers: int,
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]: ...
def translate_frame(frame: NDArray[np.float32], y_offset: int, x_offset: int) -> NDArray[np.float32]: ...
