import numpy as np
from numpy.typing import NDArray as NDArray

from .utils import (
    NORMALIZATION_EPSILON as NORMALIZATION_EPSILON,
    apply_mask as apply_mask,
    apply_phase_correlation as apply_phase_correlation,
    compute_upsampling_kernel as compute_upsampling_kernel,
    compute_gaussian_frequency_filter as compute_gaussian_frequency_filter,
)
from ..detection import compute_spatial_taper_mask as compute_spatial_taper_mask

_SNR_EPSILON: float
_SUBPIXEL_FACTOR: int
_UPSAMPLING_PADDING: int
_CORRELATION_BATCH_SIZE: int

def compute_nonrigid_reference_data(
    reference_image: NDArray[np.float32],
    taper_slope: float,
    smoothing_sigma: float,
    y_blocks: list[NDArray[np.int32]],
    x_blocks: list[NDArray[np.int32]],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.complex64]]: ...
def compute_nonrigid_offsets(
    frames: NDArray[np.float32],
    taper_mask: NDArray[np.float32],
    mean_offset: NDArray[np.float32],
    reference_kernel: NDArray[np.complex64],
    snr_threshold: float,
    smoothing_kernel: NDArray[np.float32],
    x_blocks: list[NDArray[np.int32]],
    y_blocks: list[NDArray[np.int32]],
    maximum_offset: float,
    workers: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]: ...
def apply_nonrigid_correction(
    frames: NDArray[np.float32],
    block_counts: tuple[int, int],
    x_blocks: list[NDArray[np.int32]],
    y_blocks: list[NDArray[np.int32]],
    y_block_offsets: NDArray[np.float32],
    x_block_offsets: NDArray[np.float32],
) -> NDArray[np.float32]: ...
def _compute_correlation_snr(correlation_data: NDArray[np.float32], padding: int) -> NDArray[np.float32]: ...
def _apply_bilinear_interpolation(
    source: NDArray[np.float32],
    y_coordinates: NDArray[np.float32],
    x_coordinates: NDArray[np.float32],
    output: NDArray[np.float32],
) -> None: ...
def _apply_coordinate_offsets(
    frames: NDArray[np.float32],
    y_offset_maps: NDArray[np.float32],
    x_offset_maps: NDArray[np.float32],
    y_grid: NDArray[np.float32],
    x_grid: NDArray[np.float32],
    output: NDArray[np.float32],
) -> None: ...
def _interpolate_block_offsets(
    y_block_offsets: NDArray[np.float32],
    x_block_offsets: NDArray[np.float32],
    y_grid: NDArray[np.float32],
    x_grid: NDArray[np.float32],
    y_offset_maps: NDArray[np.float32],
    x_offset_maps: NDArray[np.float32],
) -> None: ...
def _extract_upsampling_regions(
    correlation: NDArray[np.float32],
    y_peaks: NDArray[np.int32],
    x_peaks: NDArray[np.int32],
    region_size: int,
    output: NDArray[np.float32],
) -> None: ...
def _upsample_block_offsets(
    width: int,
    height: int,
    block_counts: tuple[int, int],
    x_blocks: list[NDArray[np.int32]],
    y_blocks: list[NDArray[np.int32]],
    y_block_offsets: NDArray[np.float32],
    x_block_offsets: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
