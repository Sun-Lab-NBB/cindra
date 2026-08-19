from pathlib import Path

from ..layout import (
    PARAMETERS_FILENAME as PARAMETERS_FILENAME,
    OUTPUT_DIRECTORY_NAME as OUTPUT_DIRECTORY_NAME,
    CHANNEL_1_BINARY_FILENAME as CHANNEL_1_BINARY_FILENAME,
    CHANNEL_2_BINARY_FILENAME as CHANNEL_2_BINARY_FILENAME,
    COMBINED_METADATA_FILENAME as COMBINED_METADATA_FILENAME,
    MULTI_RECORDING_DIRECTORY_NAME as MULTI_RECORDING_DIRECTORY_NAME,
    ACQUISITION_PARAMETERS_FILENAME as ACQUISITION_PARAMETERS_FILENAME,
    MULTI_RECORDING_RUNTIME_DATA_FILENAME as MULTI_RECORDING_RUNTIME_DATA_FILENAME,
    SINGLE_RECORDING_RUNTIME_DATA_FILENAME as SINGLE_RECORDING_RUNTIME_DATA_FILENAME,
    resolve_plane_specifier as resolve_plane_specifier,
)
from ..dataclasses import (
    IOData as IOData,
    CombinedData as CombinedData,
    RuntimeContext as RuntimeContext,
    MultiRecordingIOData as MultiRecordingIOData,
    AcquisitionParameters as AcquisitionParameters,
    MultiRecordingRuntimeData as MultiRecordingRuntimeData,
    SingleRecordingRuntimeData as SingleRecordingRuntimeData,
    MultiRecordingConfiguration as MultiRecordingConfiguration,
    MultiRecordingRuntimeContext as MultiRecordingRuntimeContext,
    SingleRecordingConfiguration as SingleRecordingConfiguration,
)

MAXIMUM_CHANNEL_COUNT: int

def find_data_directory(data_path: Path) -> Path: ...
def resolve_single_recording_contexts(
    configuration: SingleRecordingConfiguration, *, persist: bool = True
) -> list[RuntimeContext]: ...
def resolve_multi_recording_contexts(
    configuration: MultiRecordingConfiguration, target_recording_id: str | None = None, *, persist: bool = True
) -> list[MultiRecordingRuntimeContext]: ...
def extract_unique_components(paths: list[Path] | tuple[Path, ...]) -> tuple[str, ...]: ...
def resolve_recording_roots(paths: list[Path] | tuple[Path, ...]) -> tuple[Path, ...]: ...
def load_acquisition_parameters(json_path: Path) -> AcquisitionParameters: ...
def find_cindra_directory(recording_directory: Path) -> Path: ...
def _validate_positive_count(value: object, field_name: str, json_path: Path) -> None: ...
def _find_acquisition_parameters(data_path: Path) -> AcquisitionParameters: ...
def _compute_mroi_region_borders(data_path: Path) -> tuple[int, ...]: ...
