from typing import TypeIs
from pathlib import Path

from ..io import (
    TIFF_EXTENSIONS as TIFF_EXTENSIONS,
    MAXIMUM_CHANNEL_COUNT as MAXIMUM_CHANNEL_COUNT,
    find_data_directory as find_data_directory,
)
from ..layout import PARAMETERS_FILENAME as PARAMETERS_FILENAME
from .mcp_instance import mcp as mcp

_MINIMUM_RECOMMENDED_FRAMES_PER_PLANE: int
_MAXIMUM_ROI_LINE_SLICE: int
_ROI_LINE_SLICE_FIELDS: int
_ROI_LINE_SPAN_FIELDS: int

def generate_acquisition_parameters_file_tool(
    raw_data_path: str,
    frame_rate: float,
    plane_number: int = 1,
    channel_number: int = 1,
    roi_number: int = 1,
    roi_lines: list[list[int]] | None = None,
    roi_line_spans: list[list[int]] | None = None,
    roi_x_coordinates: list[int] | None = None,
    roi_y_coordinates: list[int] | None = None,
) -> dict[str, bool | str | list[str] | dict[str, object]]: ...
def validate_acquisition_parameters_file_tool(
    file_path: str, roi_line_slice: list[int] | None = None
) -> dict[str, bool | str | list[str] | dict[str, object]]: ...
def validate_recording_readiness_tool(
    raw_data_path: str, roi_line_slice: list[int] | None = None
) -> dict[str, object]: ...
def _validate_acquisition_parameters(data: dict[str, object]) -> tuple[list[str], list[str]]: ...
def _expand_roi_line_spans(spans: list[list[int]], roi_number: int) -> tuple[list[list[int]] | None, list[str]]: ...
def _compact_acquisition_parameters(data: dict[str, object]) -> dict[str, object]: ...
def _summarize_roi_line_blocks(roi_lines: list[object]) -> list[dict[str, object]]: ...
def _check_roi_line_blocks(roi_lines: list[object], frame_height: int | None) -> tuple[list[str], list[str]]: ...
def _resolve_roi_line_slice(roi_lines: object, request: list[int]) -> tuple[dict[str, object] | None, str]: ...
def _resolve_imaging_directory(directory: Path) -> Path: ...
def _resolve_missing_parameters_message(directory: Path) -> str: ...
def _is_integer_list(value: object) -> TypeIs[list[int]]: ...
