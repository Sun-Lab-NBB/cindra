from enum import StrEnum
from pathlib import Path

OUTPUT_DIRECTORY_NAME: str
MULTI_RECORDING_DIRECTORY_NAME: str
PLANE_SPECIFIER_PREFIX: str
REGISTRATION_DATA_DIRECTORY_NAME: str
DETECTION_DATA_DIRECTORY_NAME: str
MULTI_RECORDING_ARRAYS_DIRECTORY_NAME: str
PARAMETERS_FILENAME: str
SINGLE_RECORDING_CONFIGURATION_FILENAME: str
MULTI_RECORDING_CONFIGURATION_FILENAME: str
ACQUISITION_PARAMETERS_FILENAME: str
SINGLE_RECORDING_RUNTIME_DATA_FILENAME: str
MULTI_RECORDING_RUNTIME_DATA_FILENAME: str
SINGLE_RECORDING_TRACKER_FILENAME: str
MULTI_RECORDING_TRACKER_FILENAME: str
COMBINED_METADATA_FILENAME: str
TRACKING_TEMPLATE_MASKS_FILENAME: str
DEFORMED_MASKS_FILENAME: str
CHANNEL_1_BINARY_FILENAME: str
CHANNEL_2_BINARY_FILENAME: str
_BINARIZATION_MARKER_SUFFIX: str
_REGISTRATION_MARKER_SUFFIX: str
_CHANNEL_2_ARRAY_SUFFIX: str

class RecordingArrays(StrEnum):
    CELL_FLUORESCENCE = "cell_fluorescence.npy"
    NEUROPIL_FLUORESCENCE = "neuropil_fluorescence.npy"
    SUBTRACTED_FLUORESCENCE = "subtracted_fluorescence.npy"
    SPIKES = "spikes.npy"
    CELL_CLASSIFICATION = "cell_classification.npy"
    CELL_COLOCALIZATION = "cell_colocalization.npy"
    ROI_MASKS = "roi_masks.npz"
    ROI_STATISTICS = "roi_statistics.npz"
    CORRECTED_STRUCTURAL_MEAN_IMAGE = "corrected_structural_mean_image.npy"

class DetectionImages(StrEnum):
    MEAN_IMAGE = "mean_image.npy"
    ENHANCED_MEAN_IMAGE = "enhanced_mean_image.npy"
    MAXIMUM_PROJECTION = "maximum_projection.npy"
    CORRELATION_MAP = "correlation_map.npy"

class RegistrationArrays(StrEnum):
    BAD_FRAMES = "bad_frames.npy"
    REFERENCE_IMAGE = "reference_image.npy"
    RIGID_Y_OFFSETS = "rigid_y_offsets.npy"
    RIGID_X_OFFSETS = "rigid_x_offsets.npy"
    RIGID_CORRELATIONS = "rigid_correlations.npy"
    NONRIGID_Y_OFFSETS = "nonrigid_y_offsets.npy"
    NONRIGID_X_OFFSETS = "nonrigid_x_offsets.npy"
    NONRIGID_CORRELATIONS = "nonrigid_correlations.npy"
    PRINCIPAL_COMPONENT_EXTREME_IMAGES = "principal_component_extreme_images.npy"
    PRINCIPAL_COMPONENT_PROJECTIONS = "principal_component_projections.npy"
    PRINCIPAL_COMPONENT_SHIFT_METRICS = "principal_component_shift_metrics.npy"

class MultiRecordingArrays(StrEnum):
    DEFORM_FIELD_Y = "deform_field_y.npy"
    DEFORM_FIELD_X = "deform_field_x.npy"
    TRANSFORMED_MEAN_IMAGE = "transformed_mean_image.npy"
    TRANSFORMED_ENHANCED_MEAN_IMAGE = "transformed_enhanced_mean_image.npy"
    TRANSFORMED_MAXIMUM_PROJECTION = "transformed_maximum_projection.npy"

type PipelineArray = RecordingArrays | DetectionImages | RegistrationArrays | MultiRecordingArrays

def resolve_output_path(output_root: Path) -> Path: ...
def resolve_plane_path(output_root: Path, plane_index: int) -> Path: ...
def resolve_dataset_path(output_root: Path, dataset_name: str) -> Path: ...
def resolve_channel_2_name(name: str) -> str: ...
def resolve_array_name(array: PipelineArray, *, second_channel: bool = False) -> str: ...
def resolve_array_path(root_path: Path, array: PipelineArray, *, second_channel: bool = False) -> Path: ...
def resolve_binarization_marker_name(binary_name: str) -> str: ...
def resolve_registration_marker_name(binary_name: str) -> str: ...
def resolve_plane_specifier(plane_index: int) -> str: ...
def parse_plane_specifier(specifier: str) -> int | None: ...
