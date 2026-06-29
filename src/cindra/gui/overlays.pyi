import numpy as np
import pyqtgraph as pg
from numpy.typing import NDArray as NDArray

from .styles import (
    FONTS as FONTS,
    COLORS as COLORS,
    ROI_STYLE as ROI_STYLE,
)
from .constants import (
    ROI_CONFIG as ROI_CONFIG,
    COMMON_CONFIG as COMMON_CONFIG,
    ROIColorMode as ROIColorMode,
    BackgroundView as BackgroundView,
)
from .data_models import (
    ColorArrays as ColorArrays,
    ROIIndexMaps as ROIIndexMaps,
    ColorbarWidgets as ColorbarWidgets,
)
from ..dataclasses import ROIStatistics as ROIStatistics

_STATISTIC_FIELD_MAP: dict[int, str]

def build_views(
    frame_height: int,
    frame_width: int,
    *,
    mean_image: NDArray[np.float32],
    enhanced_mean_image: NDArray[np.float32],
    correlation_map: NDArray[np.float32],
    maximum_projection: NDArray[np.float32],
    corrected_structural_mean_image: NDArray[np.float32],
    channel_2: bool = False,
    channel_2_mean_image: NDArray[np.float32],
    channel_2_enhanced_mean_image: NDArray[np.float32],
    channel_2_correlation_map: NDArray[np.float32],
    channel_2_maximum_projection: NDArray[np.float32],
    valid_y_range: tuple[int, int] | None = None,
    valid_x_range: tuple[int, int] | None = None,
) -> NDArray[np.uint8]: ...
def display_views(view: pg.ImageItem, views: NDArray[np.uint8], view_index: int) -> None: ...
def compute_colors(
    roi_statistics: list[ROIStatistics],
    frame_height: int,
    frame_width: int,
    cell_classification: NDArray[np.float32],
    cell_colocalization: NDArray[np.float32],
    roi_colormap: str,
    colocalization_threshold: float,
    classifier_threshold: float = 0.5,
    *,
    two_channels: bool = False,
) -> ColorArrays: ...
def initialize_roi_maps(
    roi_statistics: list[ROIStatistics], frame_height: int, frame_width: int, color_arrays: ColorArrays
) -> ROIIndexMaps: ...
def draw_masks(
    roi_statistics: list[ROIStatistics],
    frame_height: int,
    frame_width: int,
    color_arrays: ColorArrays,
    roi_maps: ROIIndexMaps,
    *,
    roi_color_mode: int,
    background_view: int,
    roi_opacity: int,
    selected_roi_indices: list[int],
) -> NDArray[np.uint8]: ...
def display_masks(overlay_item: pg.ImageItem, mask: NDArray[np.uint8]) -> None: ...
def render_colorbar(
    roi_color_mode: int, color_arrays: ColorArrays, colorbar_widgets: ColorbarWidgets, colorbar_image: NDArray[np.uint8]
) -> None: ...
def draw_colorbar(colormap: str = "hsv") -> NDArray[np.uint8]: ...
def update_colormap(color_arrays: ColorArrays, roi_maps: ROIIndexMaps, colormap: str) -> NDArray[np.uint8]: ...
def update_correlation_masks(
    color_arrays: ColorArrays,
    roi_maps: ROIIndexMaps,
    binned_fluorescence: NDArray[np.float32],
    fluorescence_standard_deviation: NDArray[np.float32],
    selected_indices: list[int],
    colormap: str,
) -> None: ...
def flip_rois(
    roi_statistics: list[ROIStatistics],
    cell_classification: NDArray[np.float32],
    color_arrays: ColorArrays,
    roi_maps: ROIIndexMaps,
    selected_roi_indices: list[int],
    colormap: str,
) -> None: ...
def recompute_binary_classification(
    cell_classification: NDArray[np.float32],
    color_arrays: ColorArrays,
    roi_maps: ROIIndexMaps,
    colormap: str,
    threshold: float | None = None,
) -> None: ...
def normalize_percentile(image: NDArray[np.float32], frame_height: int, frame_width: int) -> NDArray[np.float32]: ...
def _update_rgb_masks(
    color_arrays: ColorArrays, roi_maps: ROIIndexMaps, color: NDArray[np.uint8], color_index: int
) -> None: ...
def _build_single_view(
    view_index: int,
    frame_height: int,
    frame_width: int,
    mean_image: NDArray[np.float32],
    enhanced_mean_image: NDArray[np.float32],
    correlation_map: NDArray[np.float32],
    maximum_projection: NDArray[np.float32],
    corrected_structural_mean_image: NDArray[np.float32],
    channel_2: bool,
    channel_2_mean_image: NDArray[np.float32],
    channel_2_enhanced_mean_image: NDArray[np.float32],
    channel_2_correlation_map: NDArray[np.float32],
    channel_2_maximum_projection: NDArray[np.float32],
    valid_y_range: tuple[int, int] | None,
    valid_x_range: tuple[int, int] | None,
) -> NDArray[np.float32]: ...
def _place_in_valid_region(
    image: NDArray[np.float32],
    frame_height: int,
    frame_width: int,
    valid_y_range: tuple[int, int] | None,
    valid_x_range: tuple[int, int] | None,
) -> NDArray[np.float32]: ...
def _convert_hues_to_rgb(hues: NDArray[np.float32]) -> NDArray[np.uint8]: ...
def _apply_colormap(values: NDArray[np.float32], colormap: str = "hsv") -> NDArray[np.uint8]: ...
def _apply_hsv_colormap(values: NDArray[np.float32]) -> NDArray[np.uint8]: ...
def _classification_endpoint_colors(colormap: str) -> tuple[NDArray[np.uint8], NDArray[np.uint8]]: ...
def _flip_roi(
    roi_maps: ROIIndexMaps,
    color_arrays: ColorArrays,
    roi_statistics: list[ROIStatistics],
    cell_classification_labels: NDArray[np.float32],
    roi_index: int,
    cell_color: NDArray[np.uint8],
    non_cell_color: NDArray[np.uint8],
) -> None: ...
def _highlight_selected_roi(
    overlay: NDArray[np.uint8],
    y_pixels: NDArray[np.int32],
    x_pixels: NDArray[np.int32],
    brightness: NDArray[np.float32],
) -> NDArray[np.uint8]: ...
def _highlight_selected_circle(
    overlay: NDArray[np.uint8], y_circle: NDArray[np.int32], x_circle: NDArray[np.int32], color: NDArray[np.uint8]
) -> NDArray[np.uint8]: ...
