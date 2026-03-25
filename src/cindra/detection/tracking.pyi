from ..dataclasses import (
    ROIMask as ROIMask,
    MultiRecordingRuntimeContext as MultiRecordingRuntimeContext,
)
from .roi_statistics import estimate_diameter_from_rois as estimate_diameter_from_rois

_DEFAULT_JACCARD_DISTANCE: float

def track_rois_across_recordings(contexts: list[MultiRecordingRuntimeContext]) -> None: ...
def _compute_overlap(rois: list[ROIMask]) -> None: ...
def _compute_condensed_index(row_index: int, column_index: int, matrix_size: int) -> int: ...
def _cluster_rois_in_bin(
    rois: list[ROIMask], roi_recordings: list[int], threshold: float, maximum_distance: int
) -> list[tuple[list[ROIMask], list[int]]]: ...
def _create_template_roi(
    cluster_rois: list[ROIMask], cluster_id: int, image_shape: tuple[int, int], pixel_prevalence: int
) -> ROIMask | None: ...
def _collect_recording_rois(
    contexts: list[MultiRecordingRuntimeContext], channel_2: bool
) -> tuple[list[ROIMask], list[int]]: ...
def _build_roi_grid(
    rois: list[ROIMask], recordings: list[int], grid_size: int
) -> dict[tuple[int, int], list[tuple[ROIMask, int]]]: ...
def _collect_bin_rois(
    roi_grid: dict[tuple[int, int], list[tuple[ROIMask, int]]],
    bin_origin_y: int,
    bin_origin_x: int,
    bin_height: int,
    bin_width: int,
    overlap_margin: int,
    grid_roi_size: int,
) -> tuple[list[ROIMask], list[int]]: ...
def _filter_templates(template_masks: list[ROIMask], minimum_size: int) -> list[ROIMask]: ...
def _track_channel_rois(contexts: list[MultiRecordingRuntimeContext], channel_2: bool) -> None: ...
