import numpy as np
from numpy.typing import NDArray as NDArray

from ..dataclasses import (
    CombinedData as CombinedData,
    DetectionData as DetectionData,
    ROIStatistics as ROIStatistics,
    ExtractionData as ExtractionData,
    RuntimeContext as RuntimeContext,
)

def combine_planes(plane_contexts: list[RuntimeContext]) -> CombinedData: ...
def _compute_plane_offsets(plane_contexts: list[RuntimeContext]) -> tuple[NDArray[np.int32], NDArray[np.int32]]: ...
def _resolve_plane_frame_count(context: RuntimeContext) -> int | None: ...
