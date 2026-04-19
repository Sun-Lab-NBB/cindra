import numpy as np
from numpy.typing import NDArray as NDArray

from ..io import (
    BinaryFile as BinaryFile,
    BinaryFileCombined as BinaryFileCombined,
)
from .masks import create_masks as create_masks
from .deconvolve import (
    apply_oasis_deconvolution as apply_oasis_deconvolution,
    compute_delta_fluorescence as compute_delta_fluorescence,
)
from ..dataclasses import (
    ROIStatistics as ROIStatistics,
    RuntimeContext as RuntimeContext,
    SignalExtraction as SignalExtraction,
    SpikeDeconvolution as SpikeDeconvolution,
    MultiRecordingRuntimeContext as MultiRecordingRuntimeContext,
)
from .colocalization import (
    compute_spatial_colocalization as compute_spatial_colocalization,
    compute_intensity_colocalization as compute_intensity_colocalization,
)
from ..classification import classify as classify

def extract_traces(context: RuntimeContext | MultiRecordingRuntimeContext) -> None: ...
def _extract_cell_fluorescence(
    output_prototype: NDArray[np.float32],
    data: NDArray[np.float32],
    flat_roi_masks: NDArray[np.int32],
    flat_lambda_weights: NDArray[np.float32],
    mask_offsets: NDArray[np.int32],
) -> NDArray[np.float32]: ...
def _extract_neuropil_fluorescence(
    output_prototype: NDArray[np.float32],
    data: NDArray[np.float32],
    flat_neuropil_masks: NDArray[np.int32],
    mask_offsets: NDArray[np.int32],
    neuropil_pixel_count: NDArray[np.int32],
) -> NDArray[np.float32]: ...
def _create_and_unpack_masks(
    roi_statistics: list[ROIStatistics],
    frame_height: int,
    frame_width: int,
    extract_neuropil: bool,
    allow_overlap: bool,
    cell_probability_percentile: int,
    inner_neuropil_border_radius: int,
    minimum_neuropil_pixels: int,
    channel_label: str,
) -> tuple[tuple[tuple[NDArray[np.int32], NDArray[np.float32]], ...], tuple[NDArray[np.int32], ...] | None]: ...
def _extract_fluorescence_traces(
    frames: BinaryFile | BinaryFileCombined,
    roi_masks: tuple[tuple[NDArray[np.int32], NDArray[np.float32]], ...],
    neuropil_masks: tuple[NDArray[np.int32], ...] | None,
    batch_size: int,
    channel_label: str,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
def _update_roi_extraction_statistics(
    roi_statistics: list[ROIStatistics],
    cell_fluorescence: NDArray[np.float32],
    neuropil_fluorescence: NDArray[np.float32],
    neuropil_coefficient: float,
) -> None: ...
def _extract_single_recording(context: RuntimeContext) -> None: ...
def _extract_structural_channel_2(
    context: RuntimeContext,
    batch_size: int,
    roi_masks: tuple[tuple[NDArray[np.int32], NDArray[np.float32]], ...],
    neuropil_masks: tuple[NDArray[np.int32], ...] | None,
) -> None: ...
def _extract_functional_channel_2(context: RuntimeContext, batch_size: int) -> None: ...
def _extract_multi_recording_channel(
    frames: BinaryFileCombined,
    roi_statistics: list[ROIStatistics],
    extraction_config: SignalExtraction,
    deconvolution_config: SpikeDeconvolution,
    channel_label: str,
    tau: float,
    sampling_rate: float,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]: ...
def _extract_multi_recording(context: MultiRecordingRuntimeContext) -> None: ...
