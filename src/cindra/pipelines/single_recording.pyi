from pathlib import Path

from ..io import (
    combine_planes as combine_planes,
    convert_tiffs_to_binary as convert_tiffs_to_binary,
    resolve_registration_marker_path as resolve_registration_marker_path,
    resolve_single_recording_contexts as resolve_single_recording_contexts,
)
from ..detection import detect_plane_rois as detect_plane_rois
from ..extraction import extract_traces as extract_traces
from ..dataclasses import (
    RuntimeContext as RuntimeContext,
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
def _resolve_malformed_binaries(contexts: list[RuntimeContext]) -> list[Path]: ...
def _resolve_plane_context(
    configuration: SingleRecordingConfiguration,
    plane_index: int,
    *,
    workers: int,
    stage_action: str,
    stage_progressive: str,
    stage_noun: str,
) -> RuntimeContext | None: ...
