from pathlib import Path

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

PARAMETERS_FILENAME: str
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
def _load_acquisition_parameters(json_path: Path) -> AcquisitionParameters: ...
def _find_acquisition_parameters(data_path: Path) -> AcquisitionParameters: ...
def _find_cindra_directory(recording_directory: Path) -> Path: ...
def _compute_mroi_region_borders(data_path: Path) -> tuple[int, ...]: ...
