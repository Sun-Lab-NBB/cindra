from pathlib import Path
from dataclasses import dataclass
from collections.abc import Sequence

from ..io import (
    SourceFrameGeometry as SourceFrameGeometry,
    find_data_directory as find_data_directory,
    find_cindra_directory as find_cindra_directory,
    extract_unique_components as extract_unique_components,
    resolve_source_frame_geometry as resolve_source_frame_geometry,
    resolve_acquisition_parameters as resolve_acquisition_parameters,
)
from .jobs import (
    MultiRecordingJobNames as MultiRecordingJobNames,
    SingleRecordingJobNames as SingleRecordingJobNames,
)
from ..layout import (
    PARAMETERS_FILENAME as PARAMETERS_FILENAME,
    COMBINED_METADATA_FILENAME as COMBINED_METADATA_FILENAME,
    ACQUISITION_PARAMETERS_FILENAME as ACQUISITION_PARAMETERS_FILENAME,
    RecordingArrays as RecordingArrays,
    resolve_array_path as resolve_array_path,
    parse_plane_specifier as parse_plane_specifier,
)
from .allocation import resolve_stage_workers as resolve_stage_workers
from ..dataclasses import (
    AcquisitionParameters as AcquisitionParameters,
    MultiRecordingConfiguration as MultiRecordingConfiguration,
    SingleRecordingConfiguration as SingleRecordingConfiguration,
)

MEMORY_ESTIMATE_TOLERANCE: float
WORKER_MEMORY_MB: int
SPAWNED_CHILD_MEMORY_MB: int
_BYTES_PER_MEGABYTE: int
_MEGABYTES_PER_GIGABYTE: int
_SINGLE_PRECISION_BYTES: int
_BINARIZATION_LIVE_BATCHES: int
_INTERNAL_ELEMENT_BYTES: int
_DETECTION_ARRAY_COPIES: float
_DETECTION_DENOISE_ARRAY_COPIES: float
_BIN_BATCH_SIZE: int
_DETECTION_ITERATION_MULTIPLIER: int
_REGISTRATION_METRIC_ARRAY_COPIES: int
_REFERENCE_STAGE_ARRAY_COPIES: int
_MINIMUM_METRIC_SAMPLE_COUNT: int
_MAXIMUM_METRIC_SAMPLE_COUNT: int
_MAXIMUM_EXTENT_FOR_LARGE_SAMPLE: int
_MINIMUM_METRIC_FRAME_COUNT: int
_BLOCK_OVERLAP_FACTOR: float
_UPSAMPLING_PADDING: int
_DEVICE_PIPELINE_SLOTS: int
_DEVICE_STAGING_DIRECTIONS: int
_DEVICE_STAGING_BATCH_PIXEL_BYTES: int
_DEVICE_RIGID_BATCH_PIXEL_BYTES: int
_DEVICE_NONRIGID_BATCH_PIXEL_BYTES: int
_DEVICE_BLOCK_BATCH_PIXEL_BYTES: int
_DEVICE_WINDOW_COPY_BYTES: int
_DEVICE_SUBPIXEL_BLOCK_BYTES: int
_DEVICE_REFERENCE_FRAME_PIXEL_BYTES: int
_DEVICE_REFERENCE_BLOCK_PIXEL_BYTES: int
_DEVICE_COMPLEX_BYTES: int
_DEVICE_UPSAMPLING_MATRIX_BYTES: int
_DEVICE_CONTEXT_BYTES: int
_DEVICE_LIVE_BACKENDS: int
_COMBINATION_TRACE_KINDS: int
_EXTRACTION_TRACE_COPIES: int
_EXTRACTION_BATCH_BYTES_PER_PIXEL: int
_OASIS_WORKSPACE_BYTES: int
_DISCOVERY_PLANES_PER_RECORDING: int
_DISCOVERY_TRANSIENT_PLANES: int
_TRACE_ARRAY_DIMENSIONS: int
_TRACKING_PAIRWISE_BYTES_PER_SQUARED_REGION: float
_TRACKED_REGION_HEADROOM: float

@dataclass(frozen=True, slots=True)
class PlaneGeometry:
    height: int
    width: int
    frame_count: int
    sampling_rate: float
    index: int = ...

@dataclass(frozen=True, slots=True)
class RecordingGeometry:
    planes: tuple[PlaneGeometry, ...] = ...
    raw_frame_pixels: int = ...
    source_element_bytes: int = ...
    combined_pixels: int = ...
    combined_frame_count: int = ...
    two_channels: bool = ...
    region_count: int = ...
    resolved: bool = ...
    acquisition_resolved: bool = ...
    source_resolved: bool = ...
    def _describe_unresolved_inputs(self, data_path: Path | None) -> str: ...

@dataclass(frozen=True, slots=True)
class JobSizing:
    cores: int
    memory_mb: int
    device_memory_mb: int

@dataclass(frozen=True, slots=True)
class _NonrigidBlockGeometry:
    count: int
    height: int
    width: int
    window_size: int

def resolve_recording_geometry(
    output_root: Path, data_path: Path | None = None, ignored_file_names: tuple[str, ...] = ()
) -> RecordingGeometry: ...
def read_tracked_recording_geometry(cindra_root: Path) -> RecordingGeometry: ...
def resolve_maximum_roi_count(plane_count: int, configuration: SingleRecordingConfiguration) -> int: ...
def estimate_single_recording_job_memory_mb(
    job_name: SingleRecordingJobNames,
    specifier: str,
    output_root: Path,
    configuration: SingleRecordingConfiguration,
    data_path: Path | None = None,
    *,
    planned_roi_count: int | None = None,
    gpu_registration: bool = False,
) -> int: ...
def estimate_multi_recording_job_memory_mb(
    job_name: MultiRecordingJobNames,
    specifier: str,
    recording_directories: Sequence[Path],
    configuration: MultiRecordingConfiguration,
    *,
    planned_roi_count: int | None = None,
) -> int: ...
def size_single_recording_job(
    job_name: SingleRecordingJobNames,
    specifier: str,
    output_root: Path,
    configuration: SingleRecordingConfiguration,
    data_path: Path | None = None,
    *,
    planned_roi_count: int | None = None,
    gpu_registration: bool = False,
) -> JobSizing: ...
def size_multi_recording_job(
    job_name: MultiRecordingJobNames,
    specifier: str,
    recording_directories: Sequence[Path],
    configuration: MultiRecordingConfiguration,
    *,
    planned_roi_count: int | None = None,
) -> JobSizing: ...
def _estimate_binarization_mb(geometry: RecordingGeometry, configuration: SingleRecordingConfiguration) -> int: ...
def _estimate_registration_mb(
    plane: PlaneGeometry, configuration: SingleRecordingConfiguration, *, gpu_registration: bool
) -> int: ...
def _estimate_registration_device_memory_mb(
    specifier: str, output_root: Path, configuration: SingleRecordingConfiguration, data_path: Path | None
) -> int: ...
def _estimate_registration_device_mb(plane: PlaneGeometry, configuration: SingleRecordingConfiguration) -> int: ...
def _resolve_device_batch_size(plane: PlaneGeometry, configuration: SingleRecordingConfiguration) -> int: ...
def _resolve_nonrigid_block_geometry(
    plane: PlaneGeometry, configuration: SingleRecordingConfiguration
) -> _NonrigidBlockGeometry: ...
def _estimate_processing_mb(
    plane: PlaneGeometry, configuration: SingleRecordingConfiguration, regions: int, channels: int
) -> int: ...
def _estimate_combination_mb(geometry: RecordingGeometry, regions: int) -> int: ...
def _estimate_discovery_mb(geometries: Sequence[RecordingGeometry]) -> int: ...
def _estimate_extraction_mb(
    geometry: RecordingGeometry, tracked_regions: int, configuration: MultiRecordingConfiguration
) -> int: ...
def _resolve_metric_sample_count(plane: PlaneGeometry) -> int: ...
def _resolve_binned_frame_count(plane: PlaneGeometry, configuration: SingleRecordingConfiguration) -> int: ...
def _resolve_planned_regions(
    geometry: RecordingGeometry, configuration: SingleRecordingConfiguration, planned_roi_count: int | None
) -> int: ...
def _resolve_tracked_regions(
    geometries: Sequence[RecordingGeometry],
    configuration: MultiRecordingConfiguration,
    planned_roi_count: int | None,
) -> int: ...
def _resolve_target_geometry(
    cindra_roots: Sequence[Path], geometries: Sequence[RecordingGeometry], specifier: str
) -> RecordingGeometry: ...
def _derive_plane_geometries(
    acquisition: AcquisitionParameters | None, source: SourceFrameGeometry | None
) -> tuple[PlaneGeometry, ...]: ...
def _read_source_geometry(
    data_path: Path | None, ignored_file_names: tuple[str, ...]
) -> SourceFrameGeometry | None: ...
def _resolve_cindra_directories(recording_directories: Sequence[Path]) -> tuple[Path, ...]: ...
def _read_combined_geometry(metadata_path: Path) -> tuple[int, int, bool]: ...
def _resolve_dataset_geometries(
    job_name: MultiRecordingJobNames, cindra_roots: Sequence[Path]
) -> tuple[RecordingGeometry, ...]: ...
def _read_region_count(array_path: Path) -> int: ...
def _bytes_to_megabytes(byte_count: float) -> int: ...
def _apply_tolerance(memory_mb: int) -> int: ...
