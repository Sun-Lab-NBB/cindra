from ..io import (
    TIFF_EXTENSIONS as TIFF_EXTENSIONS,
    PARAMETERS_FILENAME as PARAMETERS_FILENAME,
    MAXIMUM_CHANNEL_COUNT as MAXIMUM_CHANNEL_COUNT,
)
from .mcp_instance import mcp as mcp

_MINIMUM_RECOMMENDED_FRAMES_PER_PLANE: int

def generate_acquisition_parameters_file_tool(
    output_directory: str,
    frame_rate: float,
    plane_number: int = 1,
    channel_number: int = 1,
    roi_number: int = 1,
    roi_lines: list[list[int]] | None = None,
    roi_x_coordinates: list[int] | None = None,
    roi_y_coordinates: list[int] | None = None,
) -> dict[str, bool | str | list[str] | dict[str, object]]: ...
def validate_acquisition_parameters_file_tool(file_path: str) -> dict[str, bool | str | list[str] | dict[str, object]]: ...
def validate_recording_readiness_tool(recording_directory: str) -> dict[str, object]: ...
def _validate_acquisition_parameters(data: dict[str, object]) -> tuple[list[str], list[str]]: ...
