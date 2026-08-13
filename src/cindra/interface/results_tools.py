"""Provides MCP tools for verifying and querying cindra pipeline results.

These tools enable AI agents to verify output completeness, assess processing quality, and inspect specific
results from both single-recording and multi-recording pipelines. All tools load data directly from disk using
lightweight numpy and YAML operations for efficient targeted queries.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path
import contextlib
from dataclasses import field, dataclass

import yaml
import numpy as np
from natsort import natsorted
from ataraxis_data_structures import discover_marker_files

from ..layout import (
    OUTPUT_DIRECTORY_NAME,
    PLANE_SPECIFIER_PREFIX,
    DEFORMED_MASKS_FILENAME,
    CHANNEL_1_BINARY_FILENAME,
    CHANNEL_2_BINARY_FILENAME,
    COMBINED_METADATA_FILENAME,
    DETECTION_DATA_DIRECTORY_NAME,
    MULTI_RECORDING_DIRECTORY_NAME,
    ACQUISITION_PARAMETERS_FILENAME,
    REGISTRATION_DATA_DIRECTORY_NAME,
    TRACKING_TEMPLATE_MASKS_FILENAME,
    MULTI_RECORDING_ARRAYS_DIRECTORY_NAME,
    MULTI_RECORDING_RUNTIME_DATA_FILENAME,
    MULTI_RECORDING_CONFIGURATION_FILENAME,
    SINGLE_RECORDING_RUNTIME_DATA_FILENAME,
    SINGLE_RECORDING_CONFIGURATION_FILENAME,
    DetectionImages,
    RecordingArrays,
    RegistrationArrays,
    MultiRecordingArrays,
    resolve_array_name,
    parse_plane_specifier,
    resolve_channel_2_name,
)
from ..dataclasses import SingleRecordingConfiguration
from .mcp_instance import mcp

if TYPE_CHECKING:
    from numpy.typing import NDArray

_MAX_TRACE_ROIS: int = 50
"""Maximum number of ROIs whose traces can be queried in a single request."""

_MAX_STATS_ROIS: int = 500
"""Maximum number of ROIs whose statistics can be returned in a single request."""

_CELL_LABEL_THRESHOLD: float = 0.5
"""The threshold above which a classification label value is considered a cell."""

_ARRAY_SUMMARY_CHUNK_ELEMENTS: int = 1 << 20
"""The number of elements ``_array_summary`` reduces per accumulation step."""


@dataclass(slots=True)
class _VerificationState:
    """Tracks verification state for output completeness checks."""

    total_checks: int = 0
    """The cumulative number of checks performed."""

    passed: int = 0
    """The number of checks that passed."""

    missing: list[str] = field(default_factory=list)
    """The list of missing file or key names."""

    warnings: list[str] = field(default_factory=list)
    """The list of warning messages for non-critical issues."""


@mcp.tool()
def verify_single_recording_output_tool(recording_path: str) -> dict[str, object]:
    """Verifies completeness of single-recording pipeline output by checking for all expected files and data.

    Runs a systematic file inventory against the expected output structure documented in the
    single-recording-results skill. Reports each expected file as present or missing, validates NPZ key presence
    where applicable, and synthesizes an overall completeness verdict. Use this after processing completes to
    confirm all output was produced before moving to multi-recording processing or analysis.

    Args:
        recording_path: Absolute path to the recording output directory, which is the parent of the cindra/ folder
            and equals the per-recording 'output_path' returned by the prepare tool when the output root differs
            from the raw-data root. The cindra/ subdirectory is resolved automatically, falling back to a recursive
            search for configuration.yaml.

    Returns:
        On success, contains 'complete' flag, 'plane_count', 'two_channels' indicator, total check counts
        ('total_checks', 'passed', 'failed'), 'missing' list of absent required files, and optional 'warnings'. On
        failure, contains an 'error' message. Both cases include a 'success' flag. A 'success' value of True only
        means the tool ran. Callers MUST gate downstream steps on the 'complete' field, which is False whenever
        'missing' is non-empty. Each 'missing' entry is a bare filename for an absent file or 'file[key]' for an
        absent NPZ key. The 'warnings' list holds non-fatal issues such as a registered-binary path that does not
        resolve on disk. A recording whose configuration names flyback planes also carries 'flyback_planes' with
        their indices. Those planes are binarized and never processed, so only their binarization output is required
        and their registration, projection, and extraction files count as optional.
    """
    cindra_root, error = _find_cindra_root(recording_path)
    if cindra_root is None:
        return {"success": False, "error": f"Unable to verify output. {error}"}

    state = _VerificationState()

    # Detects two-channel status from combined metadata if available.
    two_channels = False
    combined_metadata_path = cindra_root / COMBINED_METADATA_FILENAME
    if combined_metadata_path.exists():
        with contextlib.suppress(Exception):
            metadata = np.load(combined_metadata_path, allow_pickle=False)
            if "registered_binary_paths_channel_2" in metadata:
                channel_2_paths = metadata["registered_binary_paths_channel_2"]
                two_channels = len(channel_2_paths) > 0 and str(channel_2_paths[0]) != ""

    # Root-level files.
    _check_file_exists(
        label=SINGLE_RECORDING_CONFIGURATION_FILENAME,
        path=cindra_root / SINGLE_RECORDING_CONFIGURATION_FILENAME,
        state=state,
    )
    _check_file_exists(
        label=ACQUISITION_PARAMETERS_FILENAME, path=cindra_root / ACQUISITION_PARAMETERS_FILENAME, state=state
    )
    _check_file_exists(label=COMBINED_METADATA_FILENAME, path=combined_metadata_path, state=state)
    _check_npz_keys(
        label=COMBINED_METADATA_FILENAME,
        path=combined_metadata_path,
        required_keys=[
            "plane_count",
            "combined_height",
            "combined_width",
            "tau",
            "sampling_rate",
            "registered_binary_paths",
        ],
        state=state,
    )

    # Combined detection images.
    detection_directory = cindra_root / DETECTION_DATA_DIRECTORY_NAME
    name: str
    for name in (
        DetectionImages.MEAN_IMAGE,
        DetectionImages.ENHANCED_MEAN_IMAGE,
        DetectionImages.MAXIMUM_PROJECTION,
        DetectionImages.CORRELATION_MAP,
    ):
        _check_file_exists(label=f"detection_data/{name}", path=detection_directory / name, state=state)
    if two_channels:
        for name in (
            resolve_array_name(array=DetectionImages.MEAN_IMAGE, second_channel=True),
            resolve_array_name(array=DetectionImages.ENHANCED_MEAN_IMAGE, second_channel=True),
            resolve_array_name(array=DetectionImages.MAXIMUM_PROJECTION, second_channel=True),
            resolve_array_name(array=DetectionImages.CORRELATION_MAP, second_channel=True),
        ):
            _check_file_exists(
                label=f"detection_data/{name}", path=detection_directory / name, state=state, required=False
            )

    # Combined extraction data.
    _check_file_exists(label=RecordingArrays.ROI_MASKS, path=cindra_root / RecordingArrays.ROI_MASKS, state=state)
    _check_npz_keys(
        label=RecordingArrays.ROI_MASKS,
        path=cindra_root / RecordingArrays.ROI_MASKS,
        required_keys=["pixel_counts", "y_pixels", "x_pixels", "pixel_weights", "centroids"],
        state=state,
    )
    _check_file_exists(
        label=RecordingArrays.ROI_STATISTICS, path=cindra_root / RecordingArrays.ROI_STATISTICS, state=state
    )
    _check_npz_keys(
        label=RecordingArrays.ROI_STATISTICS,
        path=cindra_root / RecordingArrays.ROI_STATISTICS,
        required_keys=["footprints", "compactness", "plane_index"],
        state=state,
    )
    for name in (
        RecordingArrays.CELL_FLUORESCENCE,
        RecordingArrays.NEUROPIL_FLUORESCENCE,
        RecordingArrays.SUBTRACTED_FLUORESCENCE,
        RecordingArrays.SPIKES,
        RecordingArrays.CELL_CLASSIFICATION,
    ):
        _check_file_exists(label=name, path=cindra_root / name, state=state)
    if two_channels:
        for name in (
            resolve_array_name(array=RecordingArrays.CELL_FLUORESCENCE, second_channel=True),
            resolve_array_name(array=RecordingArrays.NEUROPIL_FLUORESCENCE, second_channel=True),
            resolve_array_name(array=RecordingArrays.SUBTRACTED_FLUORESCENCE, second_channel=True),
            resolve_array_name(array=RecordingArrays.SPIKES, second_channel=True),
            resolve_array_name(array=RecordingArrays.CELL_CLASSIFICATION, second_channel=True),
        ):
            _check_file_exists(label=name, path=cindra_root / name, state=state, required=False)

    # Per-plane directories. A flyback plane is binarized and never registered or processed, so only the files
    # binarization writes are required of it.
    flyback_planes = _resolve_flyback_planes(cindra_root=cindra_root)
    planes = _list_plane_directories(cindra_root)
    plane_count = len(planes)
    for plane_directory in planes:
        plane_name = plane_directory.name
        processed = parse_plane_specifier(specifier=plane_name) not in flyback_planes
        _check_file_exists(
            label=f"{plane_name}/runtime_data.yaml",
            path=plane_directory / SINGLE_RECORDING_RUNTIME_DATA_FILENAME,
            state=state,
        )
        _check_file_exists(
            label=f"{plane_name}/channel_1_data.bin", path=plane_directory / CHANNEL_1_BINARY_FILENAME, state=state
        )
        if two_channels:
            _check_file_exists(
                label=f"{plane_name}/channel_2_data.bin",
                path=plane_directory / CHANNEL_2_BINARY_FILENAME,
                state=state,
                required=False,
            )

        # Per-plane registration data.
        registration_directory = plane_directory / REGISTRATION_DATA_DIRECTORY_NAME
        for name in (
            RegistrationArrays.REFERENCE_IMAGE,
            RegistrationArrays.BAD_FRAMES,
            RegistrationArrays.RIGID_Y_OFFSETS,
            RegistrationArrays.RIGID_X_OFFSETS,
            RegistrationArrays.RIGID_CORRELATIONS,
        ):
            _check_file_exists(
                label=f"{plane_name}/registration_data/{name}",
                path=registration_directory / name,
                state=state,
                required=processed,
            )
        for name in (
            RegistrationArrays.NONRIGID_Y_OFFSETS,
            RegistrationArrays.NONRIGID_X_OFFSETS,
            RegistrationArrays.NONRIGID_CORRELATIONS,
        ):
            _check_file_exists(
                label=f"{plane_name}/registration_data/{name}",
                path=registration_directory / name,
                state=state,
                required=False,
            )
        for name in (
            RegistrationArrays.PRINCIPAL_COMPONENT_EXTREME_IMAGES,
            RegistrationArrays.PRINCIPAL_COMPONENT_PROJECTIONS,
            RegistrationArrays.PRINCIPAL_COMPONENT_SHIFT_METRICS,
        ):
            _check_file_exists(
                label=f"{plane_name}/registration_data/{name}",
                path=registration_directory / name,
                state=state,
                required=False,
            )

        # Per-plane detection and extraction data. Binarization writes the mean image for every plane, while the
        # remaining projections and every extraction array come from the processing stage.
        plane_detection_directory = plane_directory / DETECTION_DATA_DIRECTORY_NAME
        _check_file_exists(
            label=f"{plane_name}/detection_data/{DetectionImages.MEAN_IMAGE}",
            path=plane_detection_directory / DetectionImages.MEAN_IMAGE,
            state=state,
        )
        for name in (
            DetectionImages.ENHANCED_MEAN_IMAGE,
            DetectionImages.MAXIMUM_PROJECTION,
            DetectionImages.CORRELATION_MAP,
        ):
            _check_file_exists(
                label=f"{plane_name}/detection_data/{name}",
                path=plane_detection_directory / name,
                state=state,
                required=processed,
            )
        _check_file_exists(
            label=f"{plane_name}/roi_masks.npz",
            path=plane_directory / RecordingArrays.ROI_MASKS,
            state=state,
            required=processed,
        )
        _check_file_exists(
            label=f"{plane_name}/roi_statistics.npz",
            path=plane_directory / RecordingArrays.ROI_STATISTICS,
            state=state,
            required=processed,
        )
        for name in (
            RecordingArrays.CELL_FLUORESCENCE,
            RecordingArrays.NEUROPIL_FLUORESCENCE,
            RecordingArrays.SUBTRACTED_FLUORESCENCE,
            RecordingArrays.SPIKES,
            RecordingArrays.CELL_CLASSIFICATION,
        ):
            _check_file_exists(
                label=f"{plane_name}/{name}", path=plane_directory / name, state=state, required=processed
            )

    # Multi-recording readiness: validates that registered binary paths exist on disk.
    if combined_metadata_path.exists():
        with contextlib.suppress(Exception):
            metadata = np.load(combined_metadata_path, allow_pickle=False)
            if "registered_binary_paths" in metadata:
                for binary_path_string in metadata["registered_binary_paths"]:
                    binary_path = Path(str(binary_path_string))
                    if not binary_path.is_absolute():
                        binary_path = cindra_root / binary_path
                    if not binary_path.exists():
                        state.warnings.append(f"Registered binary path not found: {binary_path}")

    result: dict[str, object] = {
        "success": True,
        "complete": not state.missing,
        "recording_path": recording_path,
        "cindra_path": str(cindra_root),
        "plane_count": plane_count,
        "two_channels": two_channels,
        "total_checks": state.total_checks,
        "passed": state.passed,
        "failed": state.total_checks - state.passed,
        "missing": state.missing,
        "warnings": state.warnings,
    }

    if flyback_planes:
        result["flyback_planes"] = sorted(flyback_planes)

    return result


@mcp.tool()
def verify_multi_recording_output_tool(recording_path: str, dataset: str) -> dict[str, object]:
    """Verifies completeness of multi-recording pipeline output for a specific dataset.

    Checks the entry recording's output directory for all expected multi-recording files, then enumerates all
    recordings in the dataset and verifies per-recording output completeness. Reports each expected file as
    present or missing, validates NPZ keys, and synthesizes an overall completeness verdict. Use this after
    multi-recording processing completes to confirm all output was produced.

    Args:
        recording_path: Absolute path to the recording output directory, which is the parent of the cindra/ folder
            and equals the per-recording 'output_path' returned by the prepare tool when the output root differs
            from the raw-data root. The cindra/ subdirectory is resolved automatically.
        dataset: The multi-recording dataset name to verify. Matched case-sensitively against the on-disk dataset
            directory, which is lowercased at preparation time. Pass the value returned by resolve_dataset_name_tool
            or prepare_multi_recording_batch_tool.

    Returns:
        On success, contains 'complete' flag, 'recording_count', per-recording verification summaries, 'missing'
        files, and optional 'warnings'. On failure, contains an 'error' message. Both cases include a 'success'
        flag. A 'success' value of True only means the tool ran. Callers MUST gate downstream steps on the
        'complete' field, which is False whenever 'missing' is non-empty. Each 'missing' entry is a bare filename
        for an absent file, 'file[key]' for an absent NPZ key, or a 'recording_i/...' prefixed path for a
        per-recording entry. The 'warnings' list holds non-fatal issues such as a registered-binary path that does
        not resolve on disk.
    """
    cindra_root, error = _find_cindra_root(recording_path)
    if cindra_root is None:
        return {"success": False, "error": f"Unable to verify output. {error}"}

    dataset_path, error = _find_multi_recording_root(cindra_root=cindra_root, dataset=dataset)
    if dataset_path is None:
        return {"success": False, "error": f"Unable to verify output. {error}"}

    state = _VerificationState()

    # Loads entry recording runtime data to discover all recordings in the dataset.
    runtime_yaml = _load_yaml(dataset_path / MULTI_RECORDING_RUNTIME_DATA_FILENAME)
    if runtime_yaml is None:
        return {
            "success": False,
            "error": f"Unable to load runtime data from: {dataset_path / 'multi_recording_runtime_data.yaml'}",
        }

    io_data = runtime_yaml.get("io", {})
    dataset_output_paths = io_data.get("dataset_output_paths", [str(dataset_path)])
    recording_count = len(dataset_output_paths)
    recording_results: list[dict[str, Any]] = []

    # Shared configuration (main recording only, first in natural sort order).
    configuration_found = False
    for output_path_string in dataset_output_paths:
        output_path = Path(output_path_string)
        if (output_path / MULTI_RECORDING_CONFIGURATION_FILENAME).exists():
            _check_file_exists(
                label=MULTI_RECORDING_CONFIGURATION_FILENAME,
                path=output_path / MULTI_RECORDING_CONFIGURATION_FILENAME,
                state=state,
            )
            configuration_found = True
            break
    if not configuration_found:
        state.missing.append(MULTI_RECORDING_CONFIGURATION_FILENAME)
        state.total_checks += 1

    # Per-recording verification.
    for index, output_path_string in enumerate(dataset_output_paths):
        output_path = Path(output_path_string)
        recording_prefix = f"recording_{index}"

        recording_runtime = _load_yaml(output_path / MULTI_RECORDING_RUNTIME_DATA_FILENAME)
        recording_id = (
            recording_runtime.get("io", {}).get("recording_id", f"unknown_{index}")
            if recording_runtime is not None
            else f"unknown_{index}"
        )
        recording_result: dict[str, Any] = {
            "index": index,
            "recording_id": recording_id,
            "output_path": str(output_path),
        }

        if not output_path.exists():
            recording_result["exists"] = False
            recording_result["complete"] = False
            state.missing.append(f"{recording_prefix}/output_directory")
            state.total_checks += 1
            recording_results.append(recording_result)
            continue

        recording_result["exists"] = True

        _check_file_exists(
            label=f"{recording_prefix}/multi_recording_runtime_data.yaml",
            path=output_path / MULTI_RECORDING_RUNTIME_DATA_FILENAME,
            state=state,
        )

        # Registration data.
        registration_directory = output_path / MULTI_RECORDING_ARRAYS_DIRECTORY_NAME
        name: str
        for name in (
            MultiRecordingArrays.DEFORM_FIELD_Y,
            MultiRecordingArrays.DEFORM_FIELD_X,
            MultiRecordingArrays.TRANSFORMED_MEAN_IMAGE,
            MultiRecordingArrays.TRANSFORMED_ENHANCED_MEAN_IMAGE,
            MultiRecordingArrays.TRANSFORMED_MAXIMUM_PROJECTION,
        ):
            _check_file_exists(
                label=f"{recording_prefix}/registration_arrays/{name}",
                path=registration_directory / name,
                state=state,
            )

        _check_file_exists(
            label=f"{recording_prefix}/registration_deformed_masks.npz",
            path=output_path / DEFORMED_MASKS_FILENAME,
            state=state,
        )
        _check_npz_keys(
            label=f"{recording_prefix}/registration_deformed_masks.npz",
            path=output_path / DEFORMED_MASKS_FILENAME,
            required_keys=["pixel_counts", "y_pixels", "x_pixels"],
            state=state,
        )

        # Tracking data.
        _check_file_exists(
            label=f"{recording_prefix}/tracking_template_masks.npz",
            path=output_path / TRACKING_TEMPLATE_MASKS_FILENAME,
            state=state,
        )
        _check_npz_keys(
            label=f"{recording_prefix}/tracking_template_masks.npz",
            path=output_path / TRACKING_TEMPLATE_MASKS_FILENAME,
            required_keys=["pixel_counts", "cluster_id", "recording_count"],
            state=state,
        )

        # Extraction data.
        _check_file_exists(
            label=f"{recording_prefix}/roi_masks.npz", path=output_path / RecordingArrays.ROI_MASKS, state=state
        )
        _check_file_exists(
            label=f"{recording_prefix}/roi_statistics.npz",
            path=output_path / RecordingArrays.ROI_STATISTICS,
            state=state,
        )
        for name in (
            RecordingArrays.CELL_FLUORESCENCE,
            RecordingArrays.NEUROPIL_FLUORESCENCE,
            RecordingArrays.SUBTRACTED_FLUORESCENCE,
            RecordingArrays.SPIKES,
        ):
            _check_file_exists(label=f"{recording_prefix}/{name}", path=output_path / name, state=state)

        # Channel 2 files (optional).
        for name in (
            resolve_channel_2_name(name=DEFORMED_MASKS_FILENAME),
            resolve_channel_2_name(name=TRACKING_TEMPLATE_MASKS_FILENAME),
            resolve_array_name(array=RecordingArrays.ROI_MASKS, second_channel=True),
            resolve_array_name(array=RecordingArrays.ROI_STATISTICS, second_channel=True),
            resolve_array_name(array=RecordingArrays.CELL_FLUORESCENCE, second_channel=True),
            resolve_array_name(array=RecordingArrays.NEUROPIL_FLUORESCENCE, second_channel=True),
            resolve_array_name(array=RecordingArrays.SUBTRACTED_FLUORESCENCE, second_channel=True),
            resolve_array_name(array=RecordingArrays.SPIKES, second_channel=True),
        ):
            _check_file_exists(label=f"{recording_prefix}/{name}", path=output_path / name, state=state, required=False)

        _check_file_exists(
            label=f"{recording_prefix}/cell_colocalization.npy",
            path=output_path / RecordingArrays.CELL_COLOCALIZATION,
            state=state,
            required=False,
        )

        recording_result["complete"] = not any(
            missing_entry.startswith(recording_prefix) for missing_entry in state.missing
        )
        recording_results.append(recording_result)

    return {
        "success": True,
        "complete": not state.missing,
        "recording_path": recording_path,
        "dataset": dataset,
        "recording_count": recording_count,
        "total_checks": state.total_checks,
        "passed": state.passed,
        "failed": state.total_checks - state.passed,
        "recordings": recording_results,
        "missing": state.missing,
        "warnings": state.warnings,
    }


@mcp.tool()
def query_single_recording_metadata_tool(recording_path: str) -> dict[str, object]:
    """Queries metadata and summary information for a cindra-processed single recording.

    Returns recording dimensions, frame count, sampling rate, plane count, ROI count, cell classification
    summary, processing timing data, and available multi-recording datasets. Use this as the first step when
    reviewing processed results to understand the recording's properties and processing status.

    Args:
        recording_path: Absolute path to a cindra pipeline output directory.

    Returns:
        Always contains 'success', 'recording_path', 'cindra_path', and 'available_datasets'. When combined metadata
        loads, also contains 'plane_count', 'combined_height', 'combined_width', 'sampling_rate', 'tau',
        'plane_heights', 'plane_widths', and 'two_channels'. The 'two_channels' flag derives from the registered
        channel-2 binary paths and means channel 2 is present AND functional, not merely that the recording is
        dual-channel. A dual-channel recording whose second channel is structural (non-functional, such as a tdTomato
        anatomy channel) reports 'two_channels' False yet still produces channel-2 colocalization output, so do not use
        'two_channels' to decide whether channel-2 colocalization exists. A 'metadata_error' appears instead when that
        load fails, and 'combined_metadata_available' is False when the combined metadata file is missing. When the
        relevant source files exist, also contains 'roi_count', 'cell_count', 'non_cell_count', 'frame_count', and
        per-plane 'plane_timing' entries. On failure to resolve the recording, contains an 'error' message.
    """
    cindra_root, error = _find_cindra_root(recording_path)
    if cindra_root is None:
        return {"success": False, "error": f"Unable to query metadata. {error}"}

    result: dict[str, Any] = {
        "success": True,
        "recording_path": recording_path,
        "cindra_path": str(cindra_root),
    }

    combined_metadata_path = cindra_root / COMBINED_METADATA_FILENAME
    if combined_metadata_path.exists():
        try:
            metadata = np.load(combined_metadata_path, allow_pickle=False)
            result["plane_count"] = int(metadata["plane_count"][0])
            result["combined_height"] = int(metadata["combined_height"][0])
            result["combined_width"] = int(metadata["combined_width"][0])
            result["tau"] = round(float(metadata["tau"][0]), ndigits=4)
            result["sampling_rate"] = round(float(metadata["sampling_rate"][0]), ndigits=4)
            result["plane_heights"] = [int(height) for height in metadata["plane_heights"]]
            result["plane_widths"] = [int(width) for width in metadata["plane_widths"]]

            two_channels = False
            if "registered_binary_paths_channel_2" in metadata:
                channel_2_paths = metadata["registered_binary_paths_channel_2"]
                two_channels = len(channel_2_paths) > 0 and str(channel_2_paths[0]) != ""
            result["two_channels"] = two_channels
        except Exception as error:
            result["metadata_error"] = str(error)
    else:
        result["combined_metadata_available"] = False

    # ROI count and cell classification summary.
    classification_path = cindra_root / RecordingArrays.CELL_CLASSIFICATION
    if classification_path.exists():
        with contextlib.suppress(Exception):
            classification = np.load(classification_path, mmap_mode="r")
            result["roi_count"] = int(classification.shape[0])
            result["cell_count"] = int(np.sum(classification[:, 0] > _CELL_LABEL_THRESHOLD))
            result["non_cell_count"] = result["roi_count"] - result["cell_count"]

    # Frame count from fluorescence traces (memory-mapped for efficiency).
    fluorescence_path = cindra_root / RecordingArrays.CELL_FLUORESCENCE
    if fluorescence_path.exists():
        with contextlib.suppress(Exception):
            fluorescence = np.load(fluorescence_path, mmap_mode="r")
            result["frame_count"] = int(fluorescence.shape[1])

    # Per-plane timing data from runtime_data.yaml files.
    planes = _list_plane_directories(cindra_root)
    timing_entries: list[dict[str, Any]] = []
    for plane_directory in planes:
        runtime = _load_yaml(plane_directory / SINGLE_RECORDING_RUNTIME_DATA_FILENAME)
        if runtime is None:
            continue
        timing = runtime.get("timing", {})
        io_section = runtime.get("io", {})
        entry: dict[str, Any] = {"plane": plane_directory.name}

        for field_name in ("frame_height", "frame_width", "frame_count"):
            value = io_section.get(field_name)
            if value is not None:
                entry[field_name] = value

        for field_name in (
            "binarization_time",
            "registration_time",
            "detection_time",
            "extraction_time",
            "classification_time",
            "deconvolution_time",
            "total_registration_time",
            "total_processing_time",
            "registration_workers",
            "processing_workers",
            "date_processed",
            "python_version",
            "cindra_version",
        ):
            value = timing.get(field_name)
            if value is not None:
                entry[field_name] = round(value, ndigits=2) if isinstance(value, float) else value
        timing_entries.append(entry)

    if timing_entries:
        result["plane_timing"] = timing_entries

    result["available_datasets"] = _discover_available_datasets(cindra_root)
    return result


@mcp.tool()
def query_registration_quality_tool(
    recording_path: str,
    plane_index: int = 0,
) -> dict[str, object]:
    """Queries registration (motion correction) quality metrics for a specific imaging plane.

    Returns summary statistics for rigid and nonrigid registration offsets, frame-to-reference correlation
    quality, bad frame detection results, and principal component shift metrics. Use this to assess whether
    motion correction was effective and whether registration parameters need adjustment.

    Args:
        recording_path: Absolute path to a cindra pipeline output directory.
        plane_index: The imaging plane index to query (0-based). Registration data is always per-plane.

    Returns:
        On success, contains rigid offset summaries ('rigid_y_offsets', 'rigid_x_offsets'), each a
        {min, max, mean, std, shape} object, and correlation summaries, each a {min, max, mean, std} object. Offsets
        are measured in pixels. Correlation is the phase-correlation coefficient, where higher means better
        alignment. Also contains 'total_frames', 'bad_frame_count', and 'bad_frame_percentage', optional nonrigid
        offset summaries that add 'num_blocks' to the offset object, and optional 'pc_shift_metrics' paired with
        'pc_component_count'. A 'rigid_y_offsets_error' or 'rigid_x_offsets_error' key appears when that array fails
        to load. Every other metric is silently omitted on failure. On failure, contains an 'error' message. Both
        cases include a 'success' flag.
    """
    cindra_root, error = _find_cindra_root(recording_path)
    if cindra_root is None:
        return {"success": False, "error": f"Unable to query registration quality. {error}"}

    plane_path, error = _resolve_data_path(cindra_root=cindra_root, plane_index=plane_index)
    if plane_path is None:
        return {"success": False, "error": f"Unable to query registration quality. {error}"}

    registration_directory = plane_path / REGISTRATION_DATA_DIRECTORY_NAME
    if not registration_directory.exists():
        return {
            "success": False,
            "error": (
                f"Unable to query registration quality. No registration_data directory found for plane_{plane_index}."
            ),
        }

    result: dict[str, Any] = {
        "success": True,
        "recording_path": recording_path,
        "plane_index": plane_index,
    }

    # Rigid registration offsets.
    for name, key in [
        (RegistrationArrays.RIGID_Y_OFFSETS, "rigid_y_offsets"),
        (RegistrationArrays.RIGID_X_OFFSETS, "rigid_x_offsets"),
    ]:
        path = registration_directory / name
        if path.exists():
            try:
                array = np.load(path, mmap_mode="r")
                summary = _array_summary(array)
                summary["shape"] = list(array.shape)
                result[key] = summary
            except Exception as error:
                result[f"{key}_error"] = str(error)

    # Rigid correlations.
    correlation_path = registration_directory / RegistrationArrays.RIGID_CORRELATIONS
    if correlation_path.exists():
        with contextlib.suppress(Exception):
            result["rigid_correlations"] = _array_summary(np.load(correlation_path, mmap_mode="r"))

    # Bad frames.
    bad_frames_path = registration_directory / RegistrationArrays.BAD_FRAMES
    if bad_frames_path.exists():
        with contextlib.suppress(Exception):
            bad_frames = np.load(bad_frames_path, mmap_mode="r")
            total_frames = len(bad_frames)
            bad_count = int(np.sum(bad_frames))
            result["total_frames"] = total_frames
            result["bad_frame_count"] = bad_count
            result["bad_frame_percentage"] = (
                round(100.0 * bad_count / total_frames, ndigits=2) if total_frames > 0 else 0.0
            )

    # Nonrigid registration offsets (optional).
    for name, key in [
        (RegistrationArrays.NONRIGID_Y_OFFSETS, "nonrigid_y_offsets"),
        (RegistrationArrays.NONRIGID_X_OFFSETS, "nonrigid_x_offsets"),
    ]:
        path = registration_directory / name
        if path.exists():
            with contextlib.suppress(Exception):
                array = np.load(path, mmap_mode="r")
                summary = _array_summary(array)
                summary["shape"] = list(array.shape)
                summary["num_blocks"] = int(array.shape[1]) if array.ndim > 1 else 0
                result[key] = summary

    nonrigid_correlation_path = registration_directory / RegistrationArrays.NONRIGID_CORRELATIONS
    if nonrigid_correlation_path.exists():
        with contextlib.suppress(Exception):
            result["nonrigid_correlations"] = _array_summary(np.load(nonrigid_correlation_path, mmap_mode="r"))

    # Principal component shift metrics (optional).
    principal_component_metrics_path = registration_directory / RegistrationArrays.PRINCIPAL_COMPONENT_SHIFT_METRICS
    if principal_component_metrics_path.exists():
        with contextlib.suppress(Exception):
            principal_component_metrics = np.load(principal_component_metrics_path, mmap_mode="r")
            # Shape: (num_components, 3), columns: rigid magnitude, mean nonrigid, max nonrigid.
            result["pc_shift_metrics"] = [
                {
                    "component": component_index,
                    "mean_rigid_shift": round(float(principal_component_metrics[component_index, 0]), ndigits=4),
                    "mean_nonrigid_shift": round(float(principal_component_metrics[component_index, 1]), ndigits=4),
                    "max_nonrigid_shift": round(float(principal_component_metrics[component_index, 2]), ndigits=4),
                }
                for component_index in range(principal_component_metrics.shape[0])
            ]
            result["pc_component_count"] = int(principal_component_metrics.shape[0])

    return result


@mcp.tool()
def query_detection_summary_tool(
    recording_path: str,
    plane_index: int = -1,
) -> dict[str, object]:
    """Queries detection image statistics and ROI detection parameters for a recording.

    Returns intensity statistics (min, max, mean, std) for each detection image (mean image, enhanced mean,
    maximum projection, correlation map), the estimated ROI diameter, and aspect ratio. Use this to assess
    image quality and detection parameter suitability before reviewing individual ROI results.

    Args:
        recording_path: Absolute path to a cindra pipeline output directory.
        plane_index: The plane to query, where -1 selects the combined view (default) and 0 or above selects a
            specific imaging plane.

    Returns:
        On success, per-image statistics are nested under an 'images' mapping. Its keys cover channel-1 images
        ('mean_image', 'enhanced_mean_image', 'maximum_projection', 'correlation_map') and their channel-2 forms
        ('mean_image_channel_2', 'enhanced_mean_image_channel_2', 'maximum_projection_channel_2',
        'correlation_map_channel_2'). Each entry contains {min, max, mean, std, shape}, or {'error': <message>} on
        load failure. The top-level 'roi_diameter' and 'aspect_ratio' appear only when available. This tool returns
        no detected-ROI count. Query query_roi_statistics_tool or query_single_recording_metadata_tool for ROI or
        cell counts. On failure, contains an 'error' message. Both cases include a 'success' flag.
    """
    cindra_root, error = _find_cindra_root(recording_path)
    if cindra_root is None:
        return {"success": False, "error": f"Unable to query detection summary. {error}"}

    data_path, error = _resolve_data_path(cindra_root=cindra_root, plane_index=plane_index)
    if data_path is None:
        return {"success": False, "error": f"Unable to query detection summary. {error}"}

    detection_directory = data_path / DETECTION_DATA_DIRECTORY_NAME
    if not detection_directory.exists():
        return {
            "success": False,
            "error": f"Unable to query detection summary. No detection_data directory found at: {data_path}.",
        }

    result: dict[str, Any] = {
        "success": True,
        "recording_path": recording_path,
        "plane_index": plane_index,
        "images": {},
    }

    # Channel 1 and channel 2 detection images.
    image_files: dict[str, str] = {
        "mean_image": DetectionImages.MEAN_IMAGE,
        "enhanced_mean_image": DetectionImages.ENHANCED_MEAN_IMAGE,
        "maximum_projection": DetectionImages.MAXIMUM_PROJECTION,
        "correlation_map": DetectionImages.CORRELATION_MAP,
        "mean_image_channel_2": resolve_array_name(array=DetectionImages.MEAN_IMAGE, second_channel=True),
        "enhanced_mean_image_channel_2": resolve_array_name(
            array=DetectionImages.ENHANCED_MEAN_IMAGE, second_channel=True
        ),
        "maximum_projection_channel_2": resolve_array_name(
            array=DetectionImages.MAXIMUM_PROJECTION, second_channel=True
        ),
        "correlation_map_channel_2": resolve_array_name(array=DetectionImages.CORRELATION_MAP, second_channel=True),
    }
    for label, filename in image_files.items():
        path = detection_directory / filename
        if path.exists():
            try:
                image = np.load(path, mmap_mode="r")
                statistics = _array_summary(image)
                statistics["shape"] = list(image.shape)
                result["images"][label] = statistics
            except Exception as error:
                result["images"][label] = {"error": str(error)}

    # ROI diameter and aspect ratio from per-plane runtime data.
    source_plane = _list_plane_directories(cindra_root)[0] if plane_index == -1 else data_path
    if source_plane is not None:
        runtime = _load_yaml(source_plane / SINGLE_RECORDING_RUNTIME_DATA_FILENAME)
        if runtime is not None:
            detection_metadata = runtime.get("detection", {})
            if detection_metadata.get("roi_diameter") is not None:
                result["roi_diameter"] = detection_metadata["roi_diameter"]
            if detection_metadata.get("aspect_ratio") is not None:
                result["aspect_ratio"] = round(float(detection_metadata["aspect_ratio"]), ndigits=4)

    return result


@mcp.tool()
def query_roi_statistics_tool(
    recording_path: str,
    roi_indices: list[int] | None = None,
    sort_by: str | None = None,
    top_n: int | None = None,
    plane_index: int = -1,
    dataset: str | None = None,
    recording_index: int | None = None,
) -> dict[str, object]:
    """Queries per-ROI spatial statistics for a cindra-processed recording or multi-recording dataset.

    Returns statistics including pixel count, skewness, compactness, footprint scale, aspect ratio, solidity, and
    centroid coordinates for the requested ROIs. In single-recording mode (dataset is None), also returns cell
    classification labels and, when present, channel-2 colocalization data. In multi-recording mode (dataset is
    provided), enriches entries with cluster ID and recording count from tracking metadata when available.
    Classification and colocalization are not reported in multi-recording mode, because tracked recordings reuse
    the backward-transformed source masks without reclassification. Multi-recording extraction still writes
    cell_colocalization.npy when both channels were functional. Supports sorting by any statistic and limiting to
    top N results for efficient quality assessment.

    Args:
        recording_path: Absolute path to a cindra pipeline output directory.
        roi_indices: Specific ROI indices to query. Returns all ROIs when not provided (up to 500). These are 0-based
            positional row indices, not tracking cluster IDs. Indices outside [0, total_rois) are silently dropped, so
            'queried_count' may be smaller than the number requested and 'rois' may be empty with success True.
        sort_by: Sort results by this statistic name ('skewness', 'compactness', 'footprint', 'aspect_ratio',
            'pixel_count', 'solidity', 'normalized_pixel_count'). Results are returned in descending order.
        top_n: When sort_by is provided, returns the top N after sorting. Otherwise, returns the first N entries.
        plane_index: The plane to query, where -1 selects the combined view (default) and 0 or above selects a
            specific imaging plane. Only used in single-recording mode.
        dataset: The multi-recording dataset name. When provided, switches to multi-recording mode and ignores
            plane_index.
        recording_index: The recording index within the dataset to query (0-based). Only used in multi-recording mode.
            Defaults to 0 (entry recording) when not provided.

    Returns:
        On success, contains 'total_rois', 'queried_count', and 'rois' list with per-ROI statistics. In
        single-recording mode, includes 'total_cells' and 'total_non_cells', per-ROI 'is_cell' and
        'classification_probability' (present only when cell_classification.npy exists), and, when
        cell_colocalization.npy exists, a per-ROI 'colocalization' value pair plus top-level 'colocalization_mode'
        and 'colocalization_columns'. The colocalization column meaning depends on the extraction path:
        'colocalization_mode' is 'intensity' with columns ['is_colocalized', 'probability'] when a structural channel
        was used, or 'spatial' with columns ['matched_channel_2_index', 'overlap_score'] when both channels were
        functional. Single-recording mode also contains 'plane_index'. In multi-recording mode, includes 'dataset',
        'recording_index', 'recording_id', 'has_template_metadata', and optional 'cluster_id' / 'recording_count'
        per ROI. On failure, contains an 'error' message. Both cases include a 'success' flag.
    """
    cindra_root, error = _find_cindra_root(recording_path)
    if cindra_root is None:
        return {"success": False, "error": f"Unable to query ROI statistics. {error}"}

    # Resolves the data path based on mode.
    if dataset is not None:
        data_path, recording_id, error = _resolve_multi_recording_data_path(
            cindra_root=cindra_root, dataset=dataset, recording_index=recording_index
        )
        if data_path is None:
            return {"success": False, "error": f"Unable to query ROI statistics. {error}"}
    else:
        data_path, error = _resolve_data_path(cindra_root=cindra_root, plane_index=plane_index)
        if data_path is None:
            return {"success": False, "error": f"Unable to query ROI statistics. {error}"}
        recording_id = None

    statistics_path = data_path / RecordingArrays.ROI_STATISTICS
    masks_path = data_path / RecordingArrays.ROI_MASKS
    if not statistics_path.exists() or not masks_path.exists():
        return {
            "success": False,
            "error": f"Unable to query ROI statistics. ROI data files not found at: {data_path}.",
        }

    try:
        statistics_data = np.load(statistics_path, allow_pickle=False)
        masks_data = np.load(masks_path, allow_pickle=False)
    except Exception as load_error:
        return {"success": False, "error": f"Unable to load ROI data: {load_error}"}

    entries, total_rois = _build_roi_statistics_entries(
        statistics_data=statistics_data,
        masks_data=masks_data,
        roi_indices=roi_indices,
        include_plane_index=(dataset is None),
    )

    # Enriches entries with mode-specific metadata.
    if dataset is None:
        # Single-recording mode: adds classification data.
        classification_path = data_path / RecordingArrays.CELL_CLASSIFICATION
        classification = None
        if classification_path.exists():
            with contextlib.suppress(Exception):
                classification = np.load(classification_path, mmap_mode="r")

        if classification is not None:
            for _, entry in entries:
                roi_index = entry["roi_index"]
                if roi_index < classification.shape[0]:
                    entry["is_cell"] = bool(classification[roi_index, 0] > _CELL_LABEL_THRESHOLD)
                    entry["classification_probability"] = round(float(classification[roi_index, 1]), ndigits=4)

        # Adds channel-2 colocalization data when present. Column semantics depend on the extraction path:
        # intensity-based colocalization (run when one channel is structural, which also writes
        # corrected_structural_mean_image.npy) stores (is_colocalized, probability), whereas spatial colocalization
        # (run when both channels are functional) stores (matched_channel_2_index, overlap_score).
        colocalization_path = data_path / RecordingArrays.CELL_COLOCALIZATION
        colocalization = None
        if colocalization_path.exists():
            with contextlib.suppress(Exception):
                colocalization = np.load(colocalization_path, mmap_mode="r")

        if colocalization is not None:
            for _, entry in entries:
                roi_index = entry["roi_index"]
                if roi_index < colocalization.shape[0]:
                    entry["colocalization"] = [
                        round(float(colocalization[roi_index, 0]), ndigits=4),
                        round(float(colocalization[roi_index, 1]), ndigits=4),
                    ]
    else:
        # Multi-recording mode: adds tracking template metadata.
        template_data: dict[str, Any] | None = None
        template_path = data_path / TRACKING_TEMPLATE_MASKS_FILENAME
        with contextlib.suppress(Exception):
            if template_path.exists():
                raw_template = np.load(template_path, allow_pickle=False)
                if "cluster_id" in raw_template and "recording_count" in raw_template:
                    template_data = {
                        "cluster_id": raw_template["cluster_id"],
                        "recording_count": raw_template["recording_count"],
                    }

        if template_data is not None:
            for _, entry in entries:
                roi_index = entry["roi_index"]
                if roi_index < len(template_data["cluster_id"]):
                    entry["cluster_id"] = int(template_data["cluster_id"][roi_index])
                    entry["recording_count"] = int(template_data["recording_count"][roi_index])

    entries, sort_error = _sort_and_cap_entries(entries=entries, sort_by=sort_by, top_n=top_n)
    if sort_error is not None:
        return {"success": False, "error": sort_error}

    result: dict[str, Any] = {
        "success": True,
        "total_rois": total_rois,
        "queried_count": len(entries),
        "rois": [entry for _, entry in entries],
    }

    if dataset is None:
        result["plane_index"] = plane_index
        if classification is not None:
            result["total_cells"] = int(np.sum(classification[:, 0] > _CELL_LABEL_THRESHOLD))
            result["total_non_cells"] = total_rois - result["total_cells"]
        if colocalization is not None:
            intensity_based = (data_path / RecordingArrays.CORRECTED_STRUCTURAL_MEAN_IMAGE).exists()
            result["colocalization_mode"] = "intensity" if intensity_based else "spatial"
            result["colocalization_columns"] = (
                ["is_colocalized", "probability"] if intensity_based else ["matched_channel_2_index", "overlap_score"]
            )
    else:
        result["dataset"] = dataset
        result["recording_index"] = recording_index if recording_index is not None else 0
        result["recording_id"] = recording_id
        result["has_template_metadata"] = template_data is not None

    return result


@mcp.tool()
def query_traces_tool(
    recording_path: str,
    roi_indices: list[int],
    trace_type: str = "corrected",
    downsample_factor: int = 1,
    plane_index: int = -1,
    dataset: str | None = None,
    recording_index: int | None = None,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict[str, object]:
    """Queries fluorescence trace data for specific ROIs from a cindra-processed recording or multi-recording dataset.

    Returns trace arrays for up to 50 ROIs at a time. Large traces can be downsampled to reduce response size.
    Supports querying raw cell fluorescence, neuropil fluorescence, neuropil-subtracted corrected traces, or
    deconvolved spike estimates. In single-recording mode (dataset is None), queries from the combined view or a
    specific imaging plane. In multi-recording mode (dataset is provided), queries from the specified recording's
    output directory within the dataset.

    Args:
        recording_path: Absolute path to a cindra pipeline output directory.
        roi_indices: List of ROI indices to retrieve traces for (maximum 50). These are 0-based positional row
            indices into the per-recording trace arrays of shape (num_rois, frames), not tracking cluster IDs. Indices
            outside [0, num_rois) are silently skipped, so compare the returned 'roi_index' values against what you
            requested. The call errors only when none of the indices are valid.
        trace_type: The type of fluorescence trace to return. 'fluorescence' for raw cell fluorescence,
            'neuropil' for neuropil fluorescence, 'corrected' for neuropil-subtracted, 'spikes' for deconvolved. For
            'spikes' and 'corrected', spikes.npy and subtracted_fluorescence.npy are written but zero-filled when
            spike_deconvolution.extract_spikes was disabled at processing time, so an all-zero returned trace can mean
            deconvolution was off rather than absence of activity.
        downsample_factor: Factor by which to downsample traces (1 = no downsampling, 10 = every 10th sample). A
            value below 1 is silently raised to 1 and the clamped value is echoed back.
        plane_index: The plane to query, where -1 selects the combined view (default) and 0 or above selects a
            specific imaging plane. Only used in single-recording mode.
        dataset: The multi-recording dataset name. When provided, switches to multi-recording mode and ignores
            plane_index.
        recording_index: The recording index within the dataset to query (0-based). Only used in multi-recording mode.
            Defaults to 0 (entry recording) when not provided.
        start_frame: The first frame index (inclusive) of the window to return. Applied before downsampling. Defaults
            to 0 (the first frame).
        end_frame: The end frame index (exclusive) of the window to return. Applied before downsampling. Defaults to
            None, meaning through the final frame.

    Returns:
        On success, contains 'trace_type', 'downsample_factor', 'frame_count' (the recording's total frame count
        before any windowing or downsampling), the resolved 'start_frame' (inclusive) and 'end_frame' (exclusive),
        'returned_sample_count' (the length of each returned trace), and 'traces', a list of {'roi_index', 'trace'}
        entries. Each 'trace' is a flat list of float fluorescence values rounded to 4 decimals. Returned sample k
        corresponds to original frame start_frame + k * downsample_factor. Single-recording mode adds 'plane_index'.
        Multi-recording mode adds 'dataset', 'recording_index', 'recording_id', and 'total_rois'. On failure, contains
        an 'error' message. Both cases include a 'success' flag.
    """
    if len(roi_indices) > _MAX_TRACE_ROIS:
        return {
            "success": False,
            "error": f"Unable to query traces. Requested {len(roi_indices)} ROIs, maximum is {_MAX_TRACE_ROIS}.",
        }

    file_map = {
        "fluorescence": RecordingArrays.CELL_FLUORESCENCE,
        "neuropil": RecordingArrays.NEUROPIL_FLUORESCENCE,
        "corrected": RecordingArrays.SUBTRACTED_FLUORESCENCE,
        "spikes": RecordingArrays.SPIKES,
    }
    if trace_type not in file_map:
        return {
            "success": False,
            "error": (
                f"Unable to query traces. Invalid trace_type '{trace_type}'. "
                f"Valid options: {', '.join(file_map.keys())}."
            ),
        }

    cindra_root, error = _find_cindra_root(recording_path)
    if cindra_root is None:
        return {"success": False, "error": f"Unable to query traces. {error}"}

    # Resolves the data path based on mode.
    if dataset is not None:
        data_path, recording_id, error = _resolve_multi_recording_data_path(
            cindra_root=cindra_root, dataset=dataset, recording_index=recording_index
        )
        if data_path is None:
            return {"success": False, "error": f"Unable to query traces. {error}"}
    else:
        data_path, error = _resolve_data_path(cindra_root=cindra_root, plane_index=plane_index)
        if data_path is None:
            return {"success": False, "error": f"Unable to query traces. {error}"}
        recording_id = None

    trace_path = data_path / file_map[trace_type]
    if not trace_path.exists():
        return {
            "success": False,
            "error": f"Unable to query traces. Trace file not found: {file_map[trace_type]}.",
        }

    try:
        traces = np.load(trace_path, mmap_mode="r")
    except Exception as load_error:
        return {"success": False, "error": f"Unable to load trace data: {load_error}"}

    roi_count = traces.shape[0]
    valid_indices = [index for index in roi_indices if 0 <= index < roi_count]
    if not valid_indices:
        return {"success": False, "error": "Unable to query traces. No valid ROI indices provided."}

    frame_count = int(traces.shape[1])
    resolved_start = max(0, start_frame)
    resolved_end = frame_count if end_frame is None else min(end_frame, frame_count)
    if resolved_start >= resolved_end:
        return {
            "success": False,
            "error": (
                f"Unable to query traces. The requested frame range [{start_frame}, {end_frame}) is empty for a "
                f"recording with {frame_count} frames."
            ),
        }

    downsample_factor = max(1, downsample_factor)
    results: list[dict[str, Any]] = []
    for roi_index in valid_indices:
        trace = traces[roi_index][resolved_start:resolved_end]
        if downsample_factor > 1:
            trace = trace[::downsample_factor]
        results.append({"roi_index": roi_index, "trace": [round(float(value), ndigits=4) for value in trace]})

    result: dict[str, object] = {
        "success": True,
        "trace_type": trace_type,
        "downsample_factor": downsample_factor,
        "frame_count": frame_count,
        "start_frame": resolved_start,
        "end_frame": resolved_end,
        "returned_sample_count": len(results[0]["trace"]) if results else 0,
        "traces": results,
    }

    if dataset is not None:
        result["dataset"] = dataset
        result["recording_index"] = recording_index if recording_index is not None else 0
        result["recording_id"] = recording_id
        result["total_rois"] = roi_count
    else:
        result["plane_index"] = plane_index

    return result


@mcp.tool()
def query_multi_recording_overview_tool(
    recording_path: str,
    dataset: str,
) -> dict[str, object]:
    """Queries overview information for a multi-recording dataset.

    Returns the dataset structure including per-recording IDs, mask counts at each processing stage (original
    selected, forward-deformed, consensus template, backward-transformed), processing timing data, and
    extraction completion status. Use this to understand the dataset composition and verify tracking consistency
    across recordings.

    Args:
        recording_path: Absolute path to the recording output directory, which is the parent of the cindra/ folder
            and equals the per-recording 'output_path' returned by the prepare tool when the output root differs
            from the raw-data root. The cindra/ subdirectory is resolved automatically.
        dataset: The multi-recording dataset name to query. Matched case-sensitively against the on-disk dataset
            directory, which is lowercased at preparation time. Pass the value returned by resolve_dataset_name_tool
            or prepare_multi_recording_batch_tool.

    Returns:
        On success, contains 'recording_count', 'template_roi_count', and per-recording summaries with mask
        counts, timing, and completion flags. On failure, contains an 'error' message. Both cases include a
        'success' flag.
    """
    cindra_root, error = _find_cindra_root(recording_path)
    if cindra_root is None:
        return {"success": False, "error": f"Unable to query multi-recording overview. {error}"}

    dataset_path, error = _find_multi_recording_root(cindra_root=cindra_root, dataset=dataset)
    if dataset_path is None:
        return {"success": False, "error": f"Unable to query multi-recording overview. {error}"}

    runtime = _load_yaml(dataset_path / MULTI_RECORDING_RUNTIME_DATA_FILENAME)
    if runtime is None:
        return {"success": False, "error": f"Unable to load runtime data from: {dataset_path}"}

    dataset_output_paths = runtime.get("io", {}).get("dataset_output_paths", [str(dataset_path)])
    recordings: list[dict[str, Any]] = []
    template_roi_count: int | None = None

    for index, output_path_string in enumerate(dataset_output_paths):
        output_path = Path(output_path_string)
        recording_entry: dict[str, Any] = {"index": index, "output_path": str(output_path)}

        if not output_path.exists():
            recording_entry["exists"] = False
            recordings.append(recording_entry)
            continue

        recording_entry["exists"] = True
        recording_runtime = _load_yaml(output_path / MULTI_RECORDING_RUNTIME_DATA_FILENAME)
        if recording_runtime is not None:
            recording_io = recording_runtime.get("io", {})
            recording_entry["recording_id"] = recording_io.get("recording_id", f"unknown_{index}")
            recording_entry["data_path"] = recording_io.get("data_path")

            selected = recording_io.get("selected_roi_indices", [])
            recording_entry["selected_roi_count"] = len(selected) if selected else 0

            recording_timing = recording_runtime.get("timing", {})
            for field_name in (
                "registration_time",
                "tracking_time",
                "backward_transform_time",
                "total_discovery_time",
                "extraction_time",
                "deconvolution_time",
                "total_extraction_time",
                "date_processed",
                "python_version",
                "cindra_version",
            ):
                value = recording_timing.get(field_name)
                if value is not None:
                    recording_entry[field_name] = round(value, ndigits=2) if isinstance(value, float) else value

        # Mask counts from NPZ files.
        for npz_name, key_name in [
            (DEFORMED_MASKS_FILENAME, "deformed_mask_count"),
            (TRACKING_TEMPLATE_MASKS_FILENAME, "template_mask_count"),
            (RecordingArrays.ROI_MASKS, "tracked_mask_count"),
        ]:
            npz_path = output_path / npz_name
            if npz_path.exists():
                with contextlib.suppress(Exception):
                    data = np.load(npz_path, allow_pickle=False)
                    count = len(data["pixel_counts"])
                    recording_entry[key_name] = count
                    if key_name == "template_mask_count" and template_roi_count is None:
                        template_roi_count = count

        recording_entry["has_channel_2"] = (output_path / resolve_channel_2_name(name=DEFORMED_MASKS_FILENAME)).exists()
        recording_entry["extraction_complete"] = (output_path / RecordingArrays.CELL_FLUORESCENCE).exists()
        recordings.append(recording_entry)

    return {
        "success": True,
        "recording_path": recording_path,
        "dataset": dataset,
        "recording_count": len(recordings),
        "template_roi_count": template_roi_count,
        "recordings": recordings,
    }


@mcp.tool()
def query_multi_recording_registration_quality_tool(
    recording_path: str,
    dataset: str,
) -> dict[str, object]:
    """Queries cross-recording deformation field statistics for all recordings in a multi-recording dataset.

    Returns deformation field statistics (displacement magnitude summaries) and transformed image availability
    for each recording. Displacement magnitude reflects how much the field of view shifted between sessions,
    not registration quality. Visual inspection of backward-deformed template overlap is the only reliable way
    to assess cross-day registration quality.

    Args:
        recording_path: Absolute path to a recording directory that belongs to the dataset.
        dataset: The multi-recording dataset name to query. Matched case-sensitively against the on-disk dataset
            directory, which is lowercased at preparation time. Pass the value returned by resolve_dataset_name_tool
            or prepare_multi_recording_batch_tool.

    Returns:
        On success, contains per-recording deformation field statistics and image availability. Each 'deform_field_y'
        and 'deform_field_x' entry adds 'abs_mean' and 'abs_max', the mean and maximum absolute displacement in
        pixels, to its {min, max, mean, std, shape} summary. A 'displacement_magnitude' {min, max, mean, std} summary
        combines both fields. On failure, contains an 'error' message. Both cases include a 'success' flag.
    """
    cindra_root, error = _find_cindra_root(recording_path)
    if cindra_root is None:
        return {"success": False, "error": f"Unable to query registration quality. {error}"}

    dataset_path, error = _find_multi_recording_root(cindra_root=cindra_root, dataset=dataset)
    if dataset_path is None:
        return {"success": False, "error": f"Unable to query registration quality. {error}"}

    runtime = _load_yaml(dataset_path / MULTI_RECORDING_RUNTIME_DATA_FILENAME)
    if runtime is None:
        return {"success": False, "error": f"Unable to load runtime data from: {dataset_path}"}

    dataset_output_paths = runtime.get("io", {}).get("dataset_output_paths", [str(dataset_path)])
    recordings: list[dict[str, Any]] = []

    for index, output_path_string in enumerate(dataset_output_paths):
        output_path = Path(output_path_string)
        recording_entry: dict[str, Any] = {"index": index}

        recording_runtime = _load_yaml(output_path / MULTI_RECORDING_RUNTIME_DATA_FILENAME)
        if recording_runtime is not None:
            recording_entry["recording_id"] = recording_runtime.get("io", {}).get("recording_id", f"unknown_{index}")

        registration_directory = output_path / MULTI_RECORDING_ARRAYS_DIRECTORY_NAME
        if not registration_directory.exists():
            recording_entry["registration_available"] = False
            recordings.append(recording_entry)
            continue

        recording_entry["registration_available"] = True

        # Deformation field statistics.
        for field_name, file_name in [
            ("deform_field_y", MultiRecordingArrays.DEFORM_FIELD_Y),
            ("deform_field_x", MultiRecordingArrays.DEFORM_FIELD_X),
        ]:
            path = registration_directory / file_name
            if path.exists():
                with contextlib.suppress(Exception):
                    field_array = np.load(path, mmap_mode="r")
                    statistics = _array_summary(field_array)
                    statistics["shape"] = list(field_array.shape)
                    statistics["abs_mean"] = round(float(np.mean(np.abs(field_array))), ndigits=4)
                    statistics["abs_max"] = round(float(np.max(np.abs(field_array))), ndigits=4)
                    recording_entry[field_name] = statistics

        # Combined displacement magnitude.
        y_path = registration_directory / MultiRecordingArrays.DEFORM_FIELD_Y
        x_path = registration_directory / MultiRecordingArrays.DEFORM_FIELD_X
        if y_path.exists() and x_path.exists():
            with contextlib.suppress(Exception):
                y_field = np.load(y_path, mmap_mode="r")
                x_field = np.load(x_path, mmap_mode="r")
                magnitude = np.sqrt(y_field**2 + x_field**2)
                recording_entry["displacement_magnitude"] = _array_summary(magnitude)

        # Transformed image availability.
        recording_entry["transformed_images"] = {
            "mean_image": (registration_directory / MultiRecordingArrays.TRANSFORMED_MEAN_IMAGE).exists(),
            "enhanced_mean_image": (
                registration_directory / MultiRecordingArrays.TRANSFORMED_ENHANCED_MEAN_IMAGE
            ).exists(),
            "maximum_projection": (
                registration_directory / MultiRecordingArrays.TRANSFORMED_MAXIMUM_PROJECTION
            ).exists(),
        }
        recording_entry["channel_2_images"] = {
            "mean_image": (
                registration_directory
                / resolve_array_name(array=MultiRecordingArrays.TRANSFORMED_MEAN_IMAGE, second_channel=True)
            ).exists(),
            "enhanced_mean_image": (
                registration_directory
                / resolve_array_name(array=MultiRecordingArrays.TRANSFORMED_ENHANCED_MEAN_IMAGE, second_channel=True)
            ).exists(),
            "maximum_projection": (
                registration_directory
                / resolve_array_name(array=MultiRecordingArrays.TRANSFORMED_MAXIMUM_PROJECTION, second_channel=True)
            ).exists(),
        }

        recordings.append(recording_entry)

    return {
        "success": True,
        "recording_path": recording_path,
        "dataset": dataset,
        "recording_count": len(recordings),
        "recordings": recordings,
    }


@mcp.tool()
def query_multi_recording_tracking_summary_tool(
    recording_path: str,
    dataset: str,
) -> dict[str, object]:
    """Queries ROI tracking summary statistics for a multi-recording dataset.

    Returns template mask count, recording count distribution (how many recordings each tracked ROI spans),
    cluster ID range, and per-ROI centroid and recording count data. Recording count reflects how many sessions
    an ROI was detected in, not tracking reliability. ROIs can be active in some sessions and inactive in
    others.

    Args:
        recording_path: Absolute path to a recording directory that belongs to the dataset.
        dataset: The multi-recording dataset name to query. Matched case-sensitively against the on-disk dataset
            directory, which is lowercased at preparation time. Pass the value returned by resolve_dataset_name_tool
            or prepare_multi_recording_batch_tool.

    Returns:
        On success, contains 'template_count', the 'recording_count_distribution' histogram keyed by recording
        count, 'mean_recording_count', 'median_recording_count', 'min_recording_count', and 'max_recording_count'.
        It also contains a 'pixel_count_summary' {min, max, mean, std} object, 'cluster_id_range' as
        [minimum, maximum], and a 'templates' list of per-template {index, centroid, pixel_count, cluster_id,
        recording_count} entries capped at the first 200 templates. When the cap applies, 'templates_truncated' is
        True and 'templates_shown' reports the cap. A 'channel_2_template_count' appears when channel-2 template
        masks exist. On failure, contains an 'error' message. Both cases include a 'success' flag.
    """
    cindra_root, error = _find_cindra_root(recording_path)
    if cindra_root is None:
        return {"success": False, "error": f"Unable to query tracking summary. {error}"}

    dataset_path, error = _find_multi_recording_root(cindra_root=cindra_root, dataset=dataset)
    if dataset_path is None:
        return {"success": False, "error": f"Unable to query tracking summary. {error}"}

    template_path = dataset_path / TRACKING_TEMPLATE_MASKS_FILENAME
    if not template_path.exists():
        return {
            "success": False,
            "error": f"Unable to query tracking summary. Template masks not found at: {dataset_path}.",
        }

    try:
        data = np.load(template_path, allow_pickle=False)
    except Exception as error:
        return {"success": False, "error": f"Unable to load template masks: {error}"}

    pixel_counts = data["pixel_counts"]
    centroids = data["centroids"]
    cluster_ids = data["cluster_id"]
    recording_counts = data["recording_count"]
    template_count = len(pixel_counts)

    # Recording count distribution: how many templates span N recordings.
    unique_counts, histogram = np.unique(recording_counts, return_counts=True)
    distribution = {int(count): int(frequency) for count, frequency in zip(unique_counts, histogram, strict=True)}

    # Per-template summary (capped for large datasets).
    max_templates = 200
    templates: list[dict[str, Any]] = [
        {
            "index": template_index,
            "centroid": [int(centroids[template_index, 0]), int(centroids[template_index, 1])],
            "pixel_count": int(pixel_counts[template_index]),
            "cluster_id": int(cluster_ids[template_index]),
            "recording_count": int(recording_counts[template_index]),
        }
        for template_index in range(min(template_count, max_templates))
    ]

    result: dict[str, Any] = {
        "success": True,
        "recording_path": recording_path,
        "dataset": dataset,
        "template_count": template_count,
        "recording_count_distribution": distribution,
        "mean_recording_count": round(float(np.mean(recording_counts)), ndigits=2),
        "median_recording_count": int(np.median(recording_counts)),
        "min_recording_count": int(np.min(recording_counts)),
        "max_recording_count": int(np.max(recording_counts)),
        "pixel_count_summary": _array_summary(pixel_counts.astype(np.float32)),
        "cluster_id_range": [int(np.min(cluster_ids)), int(np.max(cluster_ids))],
        "templates": templates,
    }

    if template_count > max_templates:
        result["templates_truncated"] = True
        result["templates_shown"] = max_templates

    # Channel 2 template masks.
    channel_2_path = dataset_path / resolve_channel_2_name(name=TRACKING_TEMPLATE_MASKS_FILENAME)
    if channel_2_path.exists():
        with contextlib.suppress(Exception):
            channel_2_data = np.load(channel_2_path, allow_pickle=False)
            result["channel_2_template_count"] = len(channel_2_data["pixel_counts"])

    return result


@mcp.tool()
def query_cross_recording_traces_tool(
    recording_path: str,
    dataset: str,
    roi_indices: list[int],
    trace_type: str = "corrected",
    downsample_factor: int = 1,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> dict[str, object]:
    """Queries fluorescence traces for specific ROIs across all recordings in a multi-recording dataset.

    For each requested ROI, retrieves trace data from every recording in the dataset, enabling cross-recording
    comparison of tracked ROI activity. Recordings where extraction is incomplete are skipped and reported.
    Use this to compare longitudinal activity patterns for the same ROIs across sessions.

    Args:
        recording_path: Absolute path to a recording directory that belongs to the dataset.
        dataset: The multi-recording dataset name to query. Matched case-sensitively against the on-disk dataset
            directory, which is lowercased at preparation time. Pass the value returned by resolve_dataset_name_tool
            or prepare_multi_recording_batch_tool.
        roi_indices: List of ROI indices to retrieve traces for across all recordings (maximum 50). These are 0-based
            positional per-recording row indices, not the multi-recording tracking cluster_id, which is a separate
            concept. An out-of-range ROI yields an empty 'recordings' list for that ROI with success True.
        trace_type: The type of fluorescence trace to return. 'fluorescence' for raw cell fluorescence,
            'neuropil' for neuropil fluorescence, 'corrected' for neuropil-subtracted, 'spikes' for deconvolved.
        downsample_factor: Factor by which to downsample traces (1 = no downsampling, 10 = every 10th sample). A
            value below 1 is silently raised to 1 and the clamped value is echoed back.
        start_frame: The first frame index (inclusive) of the window to return, applied per recording before
            downsampling. Defaults to 0 (the first frame).
        end_frame: The end frame index (exclusive) of the window to return, applied per recording before
            downsampling. Defaults to None, meaning through each recording's final frame.

    Returns:
        On success, contains 'recording_count' and per-ROI 'rois', each with a 'recordings' list of per-recording
        entries. Each entry holds 'recording_index', 'recording_id', 'frame_count' (that recording's total frames
        before windowing or downsampling), the resolved 'start_frame' and 'end_frame', and 'trace', a flat list of
        float values rounded to 4 decimals. Recordings whose extraction is incomplete are reported under optional
        'skipped_recordings' and excluded from the traces, so the result covers only the subset with complete
        extraction, not necessarily every recording in the dataset. A recording is additionally omitted from an ROI's
        'recordings' list, without a 'skipped_recordings' entry, when the requested window resolves empty for it
        (start_frame at or beyond that recording's frame count, or end_frame not greater than start_frame). On
        failure, contains an 'error' message. Both cases include a 'success' flag.
    """
    if len(roi_indices) > _MAX_TRACE_ROIS:
        return {
            "success": False,
            "error": (
                f"Unable to query cross-recording traces. Requested {len(roi_indices)} ROIs, "
                f"maximum is {_MAX_TRACE_ROIS}."
            ),
        }

    file_map = {
        "fluorescence": RecordingArrays.CELL_FLUORESCENCE,
        "neuropil": RecordingArrays.NEUROPIL_FLUORESCENCE,
        "corrected": RecordingArrays.SUBTRACTED_FLUORESCENCE,
        "spikes": RecordingArrays.SPIKES,
    }
    if trace_type not in file_map:
        return {
            "success": False,
            "error": (
                f"Unable to query cross-recording traces. Invalid trace_type '{trace_type}'. "
                f"Valid options: {', '.join(file_map.keys())}."
            ),
        }

    cindra_root, error = _find_cindra_root(recording_path)
    if cindra_root is None:
        return {"success": False, "error": f"Unable to query cross-recording traces. {error}"}

    dataset_path, error = _find_multi_recording_root(cindra_root=cindra_root, dataset=dataset)
    if dataset_path is None:
        return {"success": False, "error": f"Unable to query cross-recording traces. {error}"}

    runtime = _load_yaml(dataset_path / MULTI_RECORDING_RUNTIME_DATA_FILENAME)
    if runtime is None:
        return {
            "success": False,
            "error": f"Unable to load runtime data from: {dataset_path / 'multi_recording_runtime_data.yaml'}",
        }

    dataset_output_paths = runtime.get("io", {}).get("dataset_output_paths", [str(dataset_path)])

    recording_information: list[tuple[int, str, Path]] = []
    for index, output_path_string in enumerate(dataset_output_paths):
        output_path = Path(output_path_string)
        recording_runtime = _load_yaml(output_path / MULTI_RECORDING_RUNTIME_DATA_FILENAME)
        recording_id = (
            recording_runtime.get("io", {}).get("recording_id", f"unknown_{index}")
            if recording_runtime is not None
            else f"unknown_{index}"
        )
        recording_information.append((index, recording_id, output_path))

    downsample_factor = max(1, downsample_factor)
    skipped_recordings: list[dict[str, object]] = []
    skipped_keys: set[tuple[int, str]] = set()

    rois_result: list[dict[str, object]] = []
    for roi_index in roi_indices:
        per_recording: list[dict[str, object]] = []

        for recording_index, recording_id, output_path in recording_information:
            trace_path = output_path / file_map[trace_type]
            if not trace_path.exists():
                skip_key = (recording_index, f"Trace file not found: {file_map[trace_type]}")
                if skip_key not in skipped_keys:
                    skipped_keys.add(skip_key)
                    skipped_recordings.append(
                        {"recording_index": recording_index, "recording_id": recording_id, "reason": skip_key[1]}
                    )
                continue

            try:
                traces = np.load(trace_path, mmap_mode="r")
            except Exception:
                skip_key = (recording_index, f"Unable to load trace file: {file_map[trace_type]}")
                if skip_key not in skipped_keys:
                    skipped_keys.add(skip_key)
                    skipped_recordings.append(
                        {"recording_index": recording_index, "recording_id": recording_id, "reason": skip_key[1]}
                    )
                continue

            if roi_index < 0 or roi_index >= traces.shape[0]:
                continue

            frame_count = int(traces.shape[1])
            resolved_start = max(0, start_frame)
            resolved_end = frame_count if end_frame is None else min(end_frame, frame_count)
            if resolved_start >= resolved_end:
                continue

            trace = traces[roi_index][resolved_start:resolved_end]
            if downsample_factor > 1:
                trace = trace[::downsample_factor]

            per_recording.append(
                {
                    "recording_index": recording_index,
                    "recording_id": recording_id,
                    "frame_count": frame_count,
                    "start_frame": resolved_start,
                    "end_frame": resolved_end,
                    "trace": [round(float(value), ndigits=4) for value in trace],
                }
            )

        rois_result.append({"roi_index": roi_index, "recordings": per_recording})

    result: dict[str, object] = {
        "success": True,
        "recording_path": recording_path,
        "dataset": dataset,
        "trace_type": trace_type,
        "downsample_factor": downsample_factor,
        "recording_count": len(recording_information),
        "rois": rois_result,
    }

    if skipped_recordings:
        result["skipped_recordings"] = skipped_recordings

    return result


def _resolve_multi_recording_data_path(
    cindra_root: Path, dataset: str, recording_index: int | None
) -> tuple[Path | None, str | None, str | None]:
    """Resolves the data path and recording ID for a multi-recording dataset query.

    Args:
        cindra_root: The cindra output directory path.
        dataset: The multi-recording dataset name.
        recording_index: The recording index within the dataset (0-based). Defaults to 0 when None.

    Returns:
        A tuple of (data_path, recording_id, error_message). If data_path is None, error_message describes the issue.
    """
    dataset_path, error = _find_multi_recording_root(cindra_root=cindra_root, dataset=dataset)
    if dataset_path is None:
        return None, None, error

    runtime = _load_yaml(dataset_path / MULTI_RECORDING_RUNTIME_DATA_FILENAME)
    if runtime is None:
        return None, None, f"Unable to load runtime data from: {dataset_path / 'multi_recording_runtime_data.yaml'}"

    dataset_output_paths = runtime.get("io", {}).get("dataset_output_paths", [str(dataset_path)])
    effective_index = recording_index if recording_index is not None else 0

    if effective_index < 0 or effective_index >= len(dataset_output_paths):
        return (
            None,
            None,
            (
                f"Recording index {effective_index} is out of range "
                f"(dataset has {len(dataset_output_paths)} recordings)."
            ),
        )

    output_path = Path(dataset_output_paths[effective_index])

    # Resolves recording ID from per-recording runtime data.
    recording_runtime = _load_yaml(output_path / MULTI_RECORDING_RUNTIME_DATA_FILENAME)
    recording_id = (
        recording_runtime.get("io", {}).get("recording_id", f"unknown_{effective_index}")
        if recording_runtime is not None
        else f"unknown_{effective_index}"
    )

    return output_path, recording_id, None


def _build_roi_statistics_entries(
    statistics_data: np.lib.npyio.NpzFile,
    masks_data: np.lib.npyio.NpzFile,
    roi_indices: list[int] | None,
    *,
    include_plane_index: bool,
) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    """Builds per-ROI statistics entries from loaded NPZ data.

    Args:
        statistics_data: The loaded roi_statistics.npz data.
        masks_data: The loaded roi_masks.npz data.
        roi_indices: Specific ROI indices to include, or None for all ROIs.
        include_plane_index: Determines whether to include the plane_index field in each entry.

    Returns:
        A tuple of (entries, total_rois) where entries is a list of (index, entry_dict) pairs.
    """
    footprints = statistics_data["footprints"]
    compactness = statistics_data["compactness"]
    solidity = statistics_data["solidity"]
    pixel_count = statistics_data["pixel_count"]
    aspect_ratio = statistics_data["aspect_ratio"]
    normalized_pixel_count = statistics_data["normalized_pixel_count"]
    skewness = statistics_data["skewness"]
    centroids = masks_data["centroids"]
    total_rois = len(footprints)

    indices = (
        list(range(total_rois))
        if roi_indices is None
        else [roi_index for roi_index in roi_indices if 0 <= roi_index < total_rois]
    )
    entries: list[tuple[int, dict[str, Any]]] = []
    for roi_index in indices:
        entry: dict[str, Any] = {
            "roi_index": roi_index,
            "centroid": [int(centroids[roi_index, 0]), int(centroids[roi_index, 1])],
            "pixel_count": int(pixel_count[roi_index]),
            "footprint": int(footprints[roi_index]),
            "compactness": round(float(compactness[roi_index]), ndigits=4),
            "solidity": round(float(solidity[roi_index]), ndigits=4),
            "aspect_ratio": round(float(aspect_ratio[roi_index]), ndigits=4),
            "normalized_pixel_count": round(float(normalized_pixel_count[roi_index]), ndigits=4),
        }
        if include_plane_index:
            entry["plane_index"] = int(statistics_data["plane_index"][roi_index])

        skewness_value = skewness[roi_index]
        entry["skewness"] = round(float(skewness_value), ndigits=4) if not np.isnan(skewness_value) else None
        entries.append((roi_index, entry))

    return entries, total_rois


def _sort_and_cap_entries(
    entries: list[tuple[int, dict[str, Any]]],
    sort_by: str | None,
    top_n: int | None,
) -> tuple[list[tuple[int, dict[str, Any]]], str | None]:
    """Sorts and caps ROI statistics entries by the specified statistic.

    Args:
        entries: The list of (index, entry_dict) pairs to sort and cap.
        sort_by: The statistic name to sort by, or None for no sorting.
        top_n: The maximum number of entries to return after sorting, or None for no limit beyond the global cap.

    Returns:
        A tuple of (sorted_entries, error_message). If error_message is not None, sorting failed.
    """
    if sort_by is not None:
        valid_sort_keys = (
            "skewness",
            "compactness",
            "footprint",
            "aspect_ratio",
            "pixel_count",
            "solidity",
            "normalized_pixel_count",
        )
        if sort_by not in valid_sort_keys:
            return entries, f"Unable to sort by '{sort_by}'. Valid options: {', '.join(valid_sort_keys)}."
        entries.sort(key=lambda pair: pair[1].get(sort_by) or 0, reverse=True)

    if top_n is not None and top_n > 0:
        entries = entries[:top_n]
    if len(entries) > _MAX_STATS_ROIS:
        entries = entries[:_MAX_STATS_ROIS]

    return entries, None


def _find_cindra_root(recording_path: str) -> tuple[Path | None, str | None]:
    """Resolves the cindra output directory from a recording path.

    Notes:
        The ataraxis marker discoverer refuses a subtree it cannot read rather than narrowing its result to the
        readable part, which is the wrong answer for a root the caller chose. A denial therefore falls back to the
        tolerant recursive glob, so an unreadable sibling directory lowers the match count instead of failing the
        whole query.

    Args:
        recording_path: Absolute path to the recording data directory.

    Returns:
        A tuple of (cindra_root, error_message). If cindra_root is None, error_message describes the issue.
    """
    recording = Path(recording_path)
    if not recording.exists():
        return None, f"Recording directory not found: {recording_path}"

    cindra_path = recording / OUTPUT_DIRECTORY_NAME
    if cindra_path.exists():
        return cindra_path, None

    # Falls back to recursive search for configuration.yaml (handles non-standard nesting).
    try:
        matches = discover_marker_files(directory=recording, marker_name=SINGLE_RECORDING_CONFIGURATION_FILENAME)
    except OSError:
        matches = natsorted(recording.rglob(SINGLE_RECORDING_CONFIGURATION_FILENAME))
    if matches:
        return matches[0].parent, None

    return None, f"No cindra output directory found under: {recording_path}"


def _find_multi_recording_root(cindra_root: Path, dataset: str) -> tuple[Path | None, str | None]:
    """Resolves a multi-recording dataset directory from the cindra root.

    Args:
        cindra_root: The cindra output directory path.
        dataset: The multi-recording dataset name.

    Returns:
        A tuple of (dataset_path, error_message).
    """
    dataset_path = cindra_root / MULTI_RECORDING_DIRECTORY_NAME / dataset
    if dataset_path.exists():
        return dataset_path, None

    multi_recording_path = cindra_root / MULTI_RECORDING_DIRECTORY_NAME
    if not multi_recording_path.exists():
        return None, f"No multi_recording directory found under: {cindra_root}"

    available = [directory.name for directory in multi_recording_path.iterdir() if directory.is_dir()]
    if not available:
        return None, "No dataset directories found under multi_recording/"
    return None, f"Dataset '{dataset}' not found. Available datasets: {', '.join(natsorted(available))}"


def _resolve_data_path(cindra_root: Path, plane_index: int) -> tuple[Path | None, str | None]:
    """Resolves the data path for combined or per-plane queries.

    Args:
        cindra_root: The cindra output directory path.
        plane_index: The plane to resolve, where -1 selects the combined view and 0 or above selects a per-plane
            view.

    Returns:
        A tuple of (data_path, error_message).
    """
    if plane_index == -1:
        return cindra_root, None

    plane_path = cindra_root / f"plane_{plane_index}"
    if not plane_path.exists():
        available = natsorted(
            path.name
            for path in cindra_root.iterdir()
            if path.is_dir() and path.name.startswith(PLANE_SPECIFIER_PREFIX)
        )
        return None, f"Plane directory plane_{plane_index} not found. Available: {', '.join(available) or 'none'}"

    return plane_path, None


def _array_summary(array: NDArray[np.float32]) -> dict[str, object]:
    """Computes summary statistics for a numpy array.

    Args:
        array: The data whose distribution is summarized. NaN entries are ignored.

    Returns:
        A dictionary containing the min, max, mean, and standard deviation of the array. The mean and the standard
        deviation are NaN when every entry is NaN.
    """
    # Chunking the accumulation bounds the resident footprint, because most callers pass a memory-mapped array whose
    # pages a whole-array NaN-aware reduction faults in and copies in full.
    values = np.asarray(array).reshape(-1)
    valid_count = 0
    value_sum = 0.0
    for chunk_start in range(0, values.size, _ARRAY_SUMMARY_CHUNK_ELEMENTS):
        chunk = values[chunk_start : chunk_start + _ARRAY_SUMMARY_CHUNK_ELEMENTS]
        valid_mask = ~np.isnan(chunk)
        valid_count += int(np.count_nonzero(valid_mask))
        value_sum += float(np.sum(chunk, where=valid_mask, dtype=np.float64))

    mean = float("nan")
    standard_deviation = float("nan")
    if valid_count > 0:
        mean = value_sum / valid_count
        deviation_sum = 0.0
        # The second pass centers each chunk on the mean before squaring, because a single-pass sum of squares cancels
        # the entire spread of a distribution whose offset is large relative to its width.
        for chunk_start in range(0, values.size, _ARRAY_SUMMARY_CHUNK_ELEMENTS):
            chunk = values[chunk_start : chunk_start + _ARRAY_SUMMARY_CHUNK_ELEMENTS]
            valid_mask = ~np.isnan(chunk)
            deviations = np.subtract(chunk, mean, dtype=np.float64)
            np.square(deviations, out=deviations)
            deviation_sum += float(np.sum(deviations, where=valid_mask, dtype=np.float64))
        standard_deviation = float(np.sqrt(deviation_sum / valid_count))

    return {
        "min": round(float(np.nanmin(array)), ndigits=4),
        "max": round(float(np.nanmax(array)), ndigits=4),
        "mean": round(mean, ndigits=4),
        "std": round(standard_deviation, ndigits=4),
    }


def _load_yaml(file_path: Path) -> dict[str, Any] | None:
    """Loads and parses the YAML file at the specified path.

    Args:
        file_path: The filesystem path to the YAML file to load.

    Returns:
        The parsed YAML dictionary, or None if loading fails.
    """
    try:
        with file_path.open() as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception:
        return None


def _resolve_flyback_planes(cindra_root: Path) -> frozenset[int]:
    """Reads the indices of the planes the recording's configuration excludes from processing.

    Args:
        cindra_root: The cindra output directory holding the recording's configuration file.

    Returns:
        The index of every flyback plane, or an empty set when the configuration is absent or unreadable.
    """
    configuration_path = cindra_root / SINGLE_RECORDING_CONFIGURATION_FILENAME
    with contextlib.suppress(Exception):
        configuration = SingleRecordingConfiguration.load(file_path=configuration_path)
        return frozenset(configuration.main.ignored_flyback_planes)
    return frozenset()


def _list_plane_directories(cindra_root: Path) -> list[Path]:
    """Returns sorted plane directories found under the cindra root.

    Args:
        cindra_root: The cindra output directory path to search for plane directories.

    Returns:
        A naturally-sorted list of plane directory paths found under the given root, so that plane_2 precedes
        plane_10.
    """
    return natsorted(
        (path for path in cindra_root.iterdir() if path.is_dir() and path.name.startswith(PLANE_SPECIFIER_PREFIX)),
        key=lambda path: path.name,
    )


def _discover_available_datasets(cindra_root: Path) -> list[str]:
    """Discovers available multi-recording dataset names under the cindra root.

    Args:
        cindra_root: The cindra output directory path to search for multi-recording datasets.

    Returns:
        A sorted list of dataset names found under the multi_recording subdirectory of the given cindra root.
    """
    multi_recording_path = cindra_root / MULTI_RECORDING_DIRECTORY_NAME
    if not multi_recording_path.exists():
        return []
    return natsorted(directory.name for directory in multi_recording_path.iterdir() if directory.is_dir())


def _check_file_exists(
    label: str,
    path: Path,
    state: _VerificationState,
    *,
    required: bool = True,
) -> bool:
    """Checks whether a file exists and updates verification state accordingly.

    Args:
        label: The descriptive label for the file being checked.
        path: The expected output file whose presence determines the check outcome.
        state: The mutable verification state to update.
        required: Determines whether a missing file is reported as a failure.

    Returns:
        True if the file exists, False otherwise.
    """
    state.total_checks += 1
    exists = path.exists()
    if exists:
        state.passed += 1
    elif required:
        state.missing.append(label)
    return exists


def _check_npz_keys(
    label: str,
    path: Path,
    required_keys: list[str],
    state: _VerificationState,
) -> None:
    """Checks for required keys in an NPZ file and updates verification state.

    Args:
        label: The descriptive label for the NPZ file being checked.
        path: The filesystem path to the NPZ file.
        required_keys: The list of keys that must be present in the NPZ file.
        state: The mutable verification state to update.
    """
    if not path.exists():
        return
    try:
        data = np.load(path, allow_pickle=False)
        for key in required_keys:
            state.total_checks += 1
            if key in data:
                state.passed += 1
            else:
                state.missing.append(f"{label}[{key}]")
    except Exception as error:
        state.warnings.append(f"Unable to read {label}: {error}")
