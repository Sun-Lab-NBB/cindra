from ..io import (
    combine_planes as combine_planes,
    convert_tiffs_to_binary as convert_tiffs_to_binary,
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

def binarize_recording(configuration: SingleRecordingConfiguration) -> None: ...
def process_plane(configuration: SingleRecordingConfiguration, plane_index: int) -> None: ...
def save_combined_data(contexts: list[RuntimeContext]) -> None: ...
