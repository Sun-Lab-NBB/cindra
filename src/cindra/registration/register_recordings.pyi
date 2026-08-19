import numpy as np
from numpy.typing import NDArray as NDArray

from ..layout import (
    RecordingArrays as RecordingArrays,
    resolve_array_name as resolve_array_name,
)
from ..detection import compute_roi_statistics as compute_roi_statistics
from .deformation import Deformation as Deformation
from ..dataclasses import (
    ROIMask as ROIMask,
    ROIStatistics as ROIStatistics,
    ReferenceImageType as ReferenceImageType,
    MultiRecordingRuntimeContext as MultiRecordingRuntimeContext,
)
from .diffeomorphic import DiffeomorphicDemonsRegistration as DiffeomorphicDemonsRegistration

def register_recordings(contexts: list[MultiRecordingRuntimeContext], *, workers: int) -> None: ...
def project_templates_to_recordings(contexts: list[MultiRecordingRuntimeContext], *, workers: int) -> None: ...
def _resolve_pool_allocation(workers: int, task_count: int) -> tuple[int, int]: ...
def _warp_mask_pixels(
    mask: ROIMask, deformation: Deformation
) -> tuple[NDArray[np.int32], NDArray[np.int32], NDArray[np.float32], tuple[int, int]]: ...
def _forward_deform_masks(masks: list[ROIMask], deformation: Deformation, frame_width: int) -> list[ROIMask]: ...
def _backward_deform_masks(
    masks: list[ROIMask], deformation: Deformation, frame_height: int, frame_width: int, roi_diameter: int
) -> list[ROIStatistics]: ...
def _apply_forward_deformation(context: MultiRecordingRuntimeContext, deformation: Deformation) -> None: ...
def _apply_backward_deformation(context: MultiRecordingRuntimeContext) -> None: ...
