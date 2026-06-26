import numpy as np
from numpy.typing import NDArray as NDArray

from .utils import (
    downsample as downsample,
    compute_thresholded_variance as compute_thresholded_variance,
    apply_temporal_high_pass_filter as apply_temporal_high_pass_filter,
    compute_temporal_standard_deviation as compute_temporal_standard_deviation,
)
from ..dataclasses import (
    ROIMask as ROIMask,
    ROIStatistics as ROIStatistics,
)

_MINIMUM_WEIGHT_FRACTION: float
_MINIMUM_SPATIAL_SCALE: int
_NORMALIZATION_EPSILON: float

def extend_roi(
    y_pixels: NDArray[np.int32], x_pixels: NDArray[np.int32], height: int, width: int, iterations: int = 1
) -> tuple[NDArray[np.int32], NDArray[np.int32]]: ...
def detect(
    frames: NDArray[np.float32],
    temporal_highpass_window: int,
    spatial_highpass_window: int,
    threshold_scaling: float,
    maximum_iterations: int,
    plane_index: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], int, list[ROIStatistics]]: ...
def _subtract_neuropil(frames: NDArray[np.float32], filter_size: int) -> None: ...
def _convolve_square_2d(frames: NDArray[np.float32], filter_size: int) -> NDArray[np.float32]: ...
def _create_initial_square(
    center_y: int, center_x: int, square_size: int, height: int, width: int
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]: ...
def _check_split_components(
    pixel_frames: NDArray[np.float32], weights: NDArray[np.float32], intensity_threshold: float
) -> tuple[float, tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.bool_]]]: ...
def _extend_mask(
    y_pixels: NDArray[np.int32], x_pixels: NDArray[np.int32], weights: NDArray[np.float32], height: int, width: int
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]: ...
def _estimate_spatial_scale(scale_images: NDArray[np.float32]) -> int: ...
def _compute_multiscale_masks(
    y_pixels: NDArray[np.int32],
    x_pixels: NDArray[np.int32],
    weights: NDArray[np.float32],
    scale_heights: NDArray[np.uint16],
    scale_widths: NDArray[np.uint16],
) -> tuple[list[NDArray[np.int32]], list[NDArray[np.int32]], list[NDArray[np.float32]]]: ...
def _extend_iteratively(
    y_pixels: NDArray[np.int32],
    x_pixels: NDArray[np.int32],
    frames: NDArray[np.float32],
    height: int,
    width: int,
    active_frame_indices: NDArray[np.intp],
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32]]: ...
def _find_best_scale(scale_images: NDArray[np.float32]) -> int: ...
