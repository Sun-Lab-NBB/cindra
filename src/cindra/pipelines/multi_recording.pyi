from ..io import (
    select_recording_rois as select_recording_rois,
    resolve_multi_recording_contexts as resolve_multi_recording_contexts,
)
from ..detection import track_rois_across_recordings as track_rois_across_recordings
from ..extraction import extract_traces as extract_traces
from ..dataclasses import MultiRecordingConfiguration as MultiRecordingConfiguration
from ..registration import (
    register_recordings as register_recordings,
    project_templates_to_recordings as project_templates_to_recordings,
)

def discover_multi_recording_cells(configuration: MultiRecordingConfiguration) -> None: ...
def extract_multi_recording_fluorescence(configuration: MultiRecordingConfiguration, recording_id: str) -> None: ...
