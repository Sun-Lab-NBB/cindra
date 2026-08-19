from typing import Any
from pathlib import Path
from dataclasses import field, dataclass

import numpy as np
from numpy.typing import NDArray as NDArray

from ..layout import (
    OUTPUT_DIRECTORY_NAME as OUTPUT_DIRECTORY_NAME,
    PLANE_SPECIFIER_PREFIX as PLANE_SPECIFIER_PREFIX,
    DEFORMED_MASKS_FILENAME as DEFORMED_MASKS_FILENAME,
    CHANNEL_1_BINARY_FILENAME as CHANNEL_1_BINARY_FILENAME,
    CHANNEL_2_BINARY_FILENAME as CHANNEL_2_BINARY_FILENAME,
    COMBINED_METADATA_FILENAME as COMBINED_METADATA_FILENAME,
    DETECTION_DATA_DIRECTORY_NAME as DETECTION_DATA_DIRECTORY_NAME,
    MULTI_RECORDING_DIRECTORY_NAME as MULTI_RECORDING_DIRECTORY_NAME,
    ACQUISITION_PARAMETERS_FILENAME as ACQUISITION_PARAMETERS_FILENAME,
    REGISTRATION_DATA_DIRECTORY_NAME as REGISTRATION_DATA_DIRECTORY_NAME,
    TRACKING_TEMPLATE_MASKS_FILENAME as TRACKING_TEMPLATE_MASKS_FILENAME,
    MULTI_RECORDING_ARRAYS_DIRECTORY_NAME as MULTI_RECORDING_ARRAYS_DIRECTORY_NAME,
    MULTI_RECORDING_RUNTIME_DATA_FILENAME as MULTI_RECORDING_RUNTIME_DATA_FILENAME,
    MULTI_RECORDING_CONFIGURATION_FILENAME as MULTI_RECORDING_CONFIGURATION_FILENAME,
    SINGLE_RECORDING_RUNTIME_DATA_FILENAME as SINGLE_RECORDING_RUNTIME_DATA_FILENAME,
    SINGLE_RECORDING_CONFIGURATION_FILENAME as SINGLE_RECORDING_CONFIGURATION_FILENAME,
    DetectionImages as DetectionImages,
    RecordingArrays as RecordingArrays,
    RegistrationArrays as RegistrationArrays,
    MultiRecordingArrays as MultiRecordingArrays,
    resolve_array_name as resolve_array_name,
    parse_plane_specifier as parse_plane_specifier,
    resolve_channel_2_name as resolve_channel_2_name,
)
from ..dataclasses import SingleRecordingConfiguration as SingleRecordingConfiguration
from .mcp_instance import mcp as mcp

_MAX_TRACE_ROIS: int
_MAX_STATS_ROIS: int
_MAX_TEMPLATE_ENTRIES: int
_CELL_LABEL_THRESHOLD: float
_ARRAY_SUMMARY_CHUNK_ELEMENTS: int

@dataclass(slots=True)
class _VerificationState:
    total_checks: int = ...
    passed: int = ...
    missing: list[str] = field(default_factory=list)
    optional_absent: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

def verify_single_recording_output_tool(output_root: str) -> dict[str, object]: ...
def verify_multi_recording_output_tool(output_root: str, dataset: str) -> dict[str, object]: ...
def query_single_recording_metadata_tool(output_root: str) -> dict[str, object]: ...
def query_registration_quality_tool(output_root: str, plane_index: int = 0) -> dict[str, object]: ...
def query_detection_summary_tool(output_root: str, plane_index: int = -1) -> dict[str, object]: ...
def query_roi_statistics_tool(
    output_root: str,
    roi_indices: list[int] | None = None,
    sort_by: str | None = None,
    top_n: int | None = None,
    plane_index: int = -1,
    dataset: str | None = None,
    recording_index: int | None = None,
) -> dict[str, object]: ...
def query_traces_tool(
    output_root: str,
    roi_indices: list[int],
    trace_type: str = "corrected",
    downsample_factor: int = 1,
    plane_index: int = -1,
    dataset: str | None = None,
    recording_index: int | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict[str, object]: ...
def query_multi_recording_overview_tool(output_root: str, dataset: str) -> dict[str, object]: ...
def query_multi_recording_registration_quality_tool(output_root: str, dataset: str) -> dict[str, object]: ...
def query_multi_recording_tracking_summary_tool(output_root: str, dataset: str) -> dict[str, object]: ...
def query_cross_recording_traces_tool(
    output_root: str,
    dataset: str,
    roi_indices: list[int],
    trace_type: str = "corrected",
    downsample_factor: int = 1,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict[str, object]: ...
def _resolve_multi_recording_data_path(
    cindra_root: Path, dataset: str, recording_index: int | None
) -> tuple[Path | None, str | None, str | None]: ...
def _build_roi_statistics_entries(
    statistics_data: np.lib.npyio.NpzFile,
    masks_data: np.lib.npyio.NpzFile,
    roi_indices: list[int] | None,
    *,
    include_plane_index: bool,
) -> tuple[list[tuple[int, dict[str, Any]]], int]: ...
def _sort_and_cap_entries(
    entries: list[tuple[int, dict[str, Any]]], sort_by: str | None, top_n: int | None
) -> tuple[list[tuple[int, dict[str, Any]]], str | None]: ...
def _find_cindra_root(output_root: str) -> tuple[Path | None, str | None]: ...
def _find_multi_recording_root(cindra_root: Path, dataset: str) -> tuple[Path | None, str | None]: ...
def _resolve_data_path(cindra_root: Path, plane_index: int) -> tuple[Path | None, str | None]: ...
def _array_summary(array: NDArray[np.float32]) -> dict[str, object]: ...
def _load_yaml(file_path: Path) -> dict[str, Any] | None: ...
def _resolve_flyback_planes(cindra_root: Path) -> frozenset[int]: ...
def _list_plane_directories(cindra_root: Path) -> list[Path]: ...
def _discover_available_datasets(cindra_root: Path) -> list[str]: ...
def _check_file_exists(label: str, path: Path, state: _VerificationState, *, required: bool = True) -> bool: ...
def _check_npz_keys(label: str, path: Path, required_keys: list[str], state: _VerificationState) -> None: ...
