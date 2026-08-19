from pathlib import Path

from ..io import (
    combine_planes as combine_planes,
    convert_tiffs_to_binary as convert_tiffs_to_binary,
    resolve_active_binary_marker as resolve_active_binary_marker,
    resolve_tiff_conversion_plan as resolve_tiff_conversion_plan,
    resolve_single_recording_contexts as resolve_single_recording_contexts,
)
from ..layout import (
    CHANNEL_2_BINARY_FILENAME as CHANNEL_2_BINARY_FILENAME,
    COMBINED_METADATA_FILENAME as COMBINED_METADATA_FILENAME,
    DETECTION_DATA_DIRECTORY_NAME as DETECTION_DATA_DIRECTORY_NAME,
    ACQUISITION_PARAMETERS_FILENAME as ACQUISITION_PARAMETERS_FILENAME,
    REGISTRATION_DATA_DIRECTORY_NAME as REGISTRATION_DATA_DIRECTORY_NAME,
    SINGLE_RECORDING_CONFIGURATION_FILENAME as SINGLE_RECORDING_CONFIGURATION_FILENAME,
    DetectionImages as DetectionImages,
    RecordingArrays as RecordingArrays,
    RegistrationArrays as RegistrationArrays,
    resolve_array_path as resolve_array_path,
    resolve_output_path as resolve_output_path,
    parse_plane_specifier as parse_plane_specifier,
)
from ..detection import detect_plane_rois as detect_plane_rois
from ..extraction import extract_traces as extract_traces
from ..dataclasses import (
    TimingData as TimingData,
    DetectionData as DetectionData,
    ExtractionData as ExtractionData,
    RuntimeContext as RuntimeContext,
    RegistrationData as RegistrationData,
    SingleRecordingConfiguration as SingleRecordingConfiguration,
)
from ..registration import register_plane as register_plane

_MINIMUM_PROCESSING_FRAMES: int
_RECOMMENDED_PROCESSING_FRAMES: int
_BINARY_ITEM_SIZE: int

def binarize_recording(configuration: SingleRecordingConfiguration, *, workers: int) -> None: ...
def register_recording_plane(
    configuration: SingleRecordingConfiguration, plane_index: int, *, workers: int
) -> None: ...
def process_plane(configuration: SingleRecordingConfiguration, plane_index: int, *, workers: int) -> None: ...
def save_combined_data(contexts: list[RuntimeContext]) -> None: ...
def _validate_binaries_are_unmarked(contexts: list[RuntimeContext]) -> None: ...
def _validate_second_channel_binaries(contexts: list[RuntimeContext]) -> None: ...
def _validate_binary_sizes(contexts: list[RuntimeContext]) -> None: ...
def _resolve_existing_plane_binaries(context: RuntimeContext) -> tuple[Path, ...]: ...
def _resolve_second_channel_binary(context: RuntimeContext) -> Path | None: ...
def _clear_downstream_data(output_root: Path) -> None: ...
def _clear_result_arrays(directory: Path) -> None: ...
def _resolve_plane_context(
    configuration: SingleRecordingConfiguration,
    plane_index: int,
    *,
    workers: int,
    stage_action: str,
    stage_progressive: str,
    stage_noun: str,
) -> RuntimeContext | None: ...
