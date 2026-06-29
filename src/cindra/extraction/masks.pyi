import numpy as np
from numpy.typing import NDArray as NDArray

from ..detection import extend_roi as extend_roi
from ..dataclasses import ROIStatistics as ROIStatistics

_NEUROPIL_EXPANSION_STEP: int
_MAXIMUM_NEUROPIL_EXPANSION_ITERATIONS: int
_RADIUS_TO_NEIGHBORHOOD_SCALE: int

def create_masks(
    roi_statistics: list[ROIStatistics],
    height: int,
    width: int,
    neuropil: bool,
    include_overlap: bool,
    cell_probability_percentile: int = 50,
    inner_neuropil_border_radius: int = 2,
    minimum_neuropil_pixels: int = 350,
) -> tuple[tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.int32] | None], ...]: ...
def _create_roi_masks(
    roi_statistics: list[ROIStatistics], width: int, include_overlap: bool
) -> tuple[tuple[NDArray[np.int32], NDArray[np.float32]], ...]: ...
def _create_neuropil_masks(
    roi_statistics: list[ROIStatistics],
    height: int,
    width: int,
    inner_neuropil_border_radius: int,
    minimum_neuropil_pixels: int,
    cell_probability_percentile: int,
    recompute: bool = False,
) -> tuple[NDArray[np.int32], ...]: ...
def _create_roi_pixels(
    roi_statistics: list[ROIStatistics], height: int, width: int, cell_probability_percentile: int
) -> NDArray[np.bool_]: ...
