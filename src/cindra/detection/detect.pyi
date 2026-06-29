from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ..io import BinaryFile as BinaryFile
from .denoise import pca_denoise as pca_denoise
from .detect_rois import detect as detect
from ..dataclasses import (
    ROIDetection as ROIDetection,
    ROIStatistics as ROIStatistics,
    RuntimeContext as RuntimeContext,
)
from .roi_statistics import compute_roi_statistics as compute_roi_statistics
from ..classification import classify as classify

_ITERATION_MULTIPLIER: int
_BACKGROUND_SCALE: int
_ENHANCED_MINIMUM_INTENSITY: float
_ENHANCED_MAXIMUM_INTENSITY: float
_VARIANCE_EPSILON: float
_DEFAULT_CELL_DIAMETER: int
type _ChannelDetectionResult = tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], int, list[ROIStatistics]
]

def detect_plane_rois(context: RuntimeContext) -> None: ...
def _create_enhanced_mean_image(
    mean_image: NDArray[np.float32],
    roi_diameter: int,
    valid_y_range: tuple[int, int],
    valid_x_range: tuple[int, int],
    frame_height: int,
    frame_width: int,
) -> NDArray[np.float32]: ...
def _apply_preclassification(
    roi_statistics: list[ROIStatistics],
    frame_height: int,
    frame_width: int,
    preclassification_threshold: float,
    crop_to_soma: bool,
    custom_classifier_path: Path | None,
    plane_index: int,
    channel_label: str,
    diameter: int = 10,
) -> list[ROIStatistics]: ...
def _detect_channel(
    binary_path: Path,
    frame_height: int,
    frame_width: int,
    frame_count: int,
    bin_size: int,
    valid_y_range: tuple[int, int],
    valid_x_range: tuple[int, int],
    bad_frames: NDArray[np.bool_] | None,
    detection_config: ROIDetection,
    nonrigid_block_size: tuple[int, int],
    parallel_workers: int,
    custom_classifier_path: Path | None,
    plane_index: int,
    channel_label: str,
) -> _ChannelDetectionResult: ...
