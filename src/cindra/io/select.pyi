from pathlib import Path

import numpy as np
from numpy.typing import NDArray as NDArray

from ..layout import (
    MULTI_RECORDING_DIRECTORY_NAME as MULTI_RECORDING_DIRECTORY_NAME,
    MULTI_RECORDING_RUNTIME_DATA_FILENAME as MULTI_RECORDING_RUNTIME_DATA_FILENAME,
)
from ..dataclasses import (
    ROIStatistics as ROIStatistics,
    MultiRecordingRuntimeData as MultiRecordingRuntimeData,
    MultiRecordingConfiguration as MultiRecordingConfiguration,
    MultiRecordingRuntimeContext as MultiRecordingRuntimeContext,
)

def select_recording_rois(contexts: list[MultiRecordingRuntimeContext]) -> None: ...
def clear_dataset_selection(dataset_path: Path) -> bool: ...
def clear_recording_selections(cindra_root: Path) -> int: ...
def _filter_channel_rois(
    roi_statistics: list[ROIStatistics],
    cell_classification: NDArray[np.float32],
    mroi_region_borders: tuple[int, ...],
    probability_threshold: float,
    maximum_size: int,
    region_margin: int,
) -> tuple[int, ...]: ...
def _filter_rois(runtime: MultiRecordingRuntimeData, configuration: MultiRecordingConfiguration) -> tuple[int, int]: ...
