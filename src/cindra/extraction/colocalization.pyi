import numpy as np
from numpy.typing import NDArray as NDArray
from scipy.sparse import csr_matrix

from .masks import create_masks as create_masks
from ..dataclasses import ROIStatistics as ROIStatistics

_BLOCK_COUNT: int
_SMOOTHING_FRACTION: float
_INTENSITY_EPSILON: float

def compute_intensity_colocalization(
    rois: list[ROIStatistics],
    functional_mean_image: NDArray[np.float32],
    structural_mean_image: NDArray[np.float32],
    frame_height: int,
    frame_width: int,
    colocalization_threshold: float,
    allow_overlap: bool,
    cell_probability_percentile: int,
    inner_neuropil_border_radius: int,
    minimum_neuropil_pixels: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
def compute_spatial_colocalization(
    rois_channel_1: list[ROIStatistics],
    rois_channel_2: list[ROIStatistics],
    frame_height: int,
    frame_width: int,
    colocalization_threshold: float,
) -> NDArray[np.float32]: ...
def _correct_bleedthrough(
    functional_mean_image: NDArray[np.float32], structural_mean_image: NDArray[np.float32]
) -> NDArray[np.float32]: ...
def _build_sparse_roi_masks(rois: list[ROIStatistics], frame_height: int, frame_width: int) -> csr_matrix: ...
def _compute_overlap_matrix(
    rois_channel_1: list[ROIStatistics], rois_channel_2: list[ROIStatistics], frame_height: int, frame_width: int
) -> NDArray[np.float32]: ...
