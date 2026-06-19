"""Provides MCP tools for verifying and querying cindra pipeline results.

These tools enable AI agents to verify output completeness, assess processing quality, and inspect specific
results from both single-recording and multi-recording pipelines. All tools load data directly from disk using
lightweight numpy and YAML operations for efficient targeted queries without the overhead of full data loading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from pathlib import Path
import contextlib
from dataclasses import field, dataclass

import yaml  # type: ignore[import-untyped]
import numpy as np

from .mcp_instance import mcp

if TYPE_CHECKING:
    from numpy.typing import NDArray

_MAX_TRACE_ROIS: int = 50
"""Maximum number of ROIs whose traces can be queried in a single request."""

_MAX_STATS_ROIS: int = 500
"""Maximum number of ROIs whose statistics can be returned in a single request."""

_CELL_LABEL_THRESHOLD: float = 0.5
"""The threshold above which a classification label value is considered a cell."""


@dataclass
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
        recording_path: Absolute path to the recording data directory.

    Returns:
        On success, contains 'complete' flag, 'plane_count', 'two_channels' indicator, total check counts
        ('total_checks', 'passed', 'failed'), 'missing' list of absent required files, and optional 'warnings'. On
        failure, contains an 'error' message. Both cases include a 'success' flag.
    """
    cindra_root, error = _find_cindra_root(recording_path)
    if cindra_root is None:
        return {"success": False, "error": f"Unable to verify output. {error}"}

    state = _VerificationState()

    # Detects two-channel status from combined metadata if available.
    two_channels = False
    combined_metadata_path = cindra_root / "combined_metadata.npz"
    if combined_metadata_path.exists():
        with contextlib.suppress(Exception):
            metadata = np.load(combined_metadata_path, allow_pickle=False)
            if "registered_binary_paths_channel_2" in metadata:
                channel_2_paths = metadata["registered_binary_paths_channel_2"]
                two_channels = len(channel_2_paths) > 0 and str(channel_2_paths[0]) != ""

    # Root-level files.
    _check_file_exists("configuration.yaml", cindra_root / "configuration.yaml", state)
    _check_file_exists("acquisition_parameters.yaml", cindra_root / "acquisition_parameters.yaml", state)
    _check_file_exists("combined_metadata.npz", combined_metadata_path, state)
    _check_npz_keys(
        "combined_metadata.npz",
        combined_metadata_path,
        ["plane_count", "combined_height", "combined_width", "tau", "sampling_rate", "registered_binary_paths"],
        state,
    )

    # Combined detection images.
    detection_directory = cindra_root / "detection_data"
    for name in ("mean_image.npy", "enhanced_mean_image.npy", "maximum_projection.npy", "correlation_map.npy"):
        _check_file_exists(f"detection_data/{name}", detection_directory / name, state)
    if two_channels:
        for name in (
            "mean_image_channel_2.npy",
            "enhanced_mean_image_channel_2.npy",
            "maximum_projection_channel_2.npy",
            "correlation_map_channel_2.npy",
        ):
            _check_file_exists(f"detection_data/{name}", detection_directory / name, state, required=False)

    # Combined extraction data.
    _check_file_exists("roi_masks.npz", cindra_root / "roi_masks.npz", state)
    _check_npz_keys(
        "roi_masks.npz",
        cindra_root / "roi_masks.npz",
        ["pixel_counts", "y_pixels", "x_pixels", "pixel_weights", "centroids"],
        state,
    )
    _check_file_exists("roi_statistics.npz", cindra_root / "roi_statistics.npz", state)
    _check_npz_keys(
        "roi_statistics.npz",
        cindra_root / "roi_statistics.npz",
        ["footprints", "compactness", "plane_index"],
        state,
    )
    for name in (
        "cell_fluorescence.npy",
        "neuropil_fluorescence.npy",
        "subtracted_fluorescence.npy",
        "spikes.npy",
        "cell_classification.npy",
    ):
        _check_file_exists(name, cindra_root / name, state)
    if two_channels:
        for name in (
            "cell_fluorescence_channel_2.npy",
            "neuropil_fluorescence_channel_2.npy",
            "subtracted_fluorescence_channel_2.npy",
            "spikes_channel_2.npy",
            "cell_classification_channel_2.npy",
        ):
            _check_file_exists(name, cindra_root / name, state, required=False)

    # Per-plane directories.
    planes = _list_plane_directories(cindra_root)
    plane_count = len(planes)
    for plane_dir in planes:
        plane_name = plane_dir.name
        _check_file_exists(f"{plane_name}/runtime_data.yaml", plane_dir / "runtime_data.yaml", state)
        _check_file_exists(f"{plane_name}/channel_1_data.bin", plane_dir / "channel_1_data.bin", state)
        if two_channels:
            _check_file_exists(
                f"{plane_name}/channel_2_data.bin",
                plane_dir / "channel_2_data.bin",
                state,
                required=False,
            )

        # Per-plane registration data.
        registration_directory = plane_dir / "registration_data"
        for name in (
            "reference_image.npy",
            "bad_frames.npy",
            "rigid_y_offsets.npy",
            "rigid_x_offsets.npy",
            "rigid_correlations.npy",
        ):
            _check_file_exists(f"{plane_name}/registration_data/{name}", registration_directory / name, state)
        for name in ("nonrigid_y_offsets.npy", "nonrigid_x_offsets.npy", "nonrigid_correlations.npy"):
            _check_file_exists(
                f"{plane_name}/registration_data/{name}",
                registration_directory / name,
                state,
                required=False,
            )
        for name in (
            "principal_component_extreme_images.npy",
            "principal_component_projections.npy",
            "principal_component_shift_metrics.npy",
        ):
            _check_file_exists(
                f"{plane_name}/registration_data/{name}",
                registration_directory / name,
                state,
                required=False,
            )

        # Per-plane detection and extraction data.
        plane_detection_directory = plane_dir / "detection_data"
        for name in ("mean_image.npy", "enhanced_mean_image.npy", "maximum_projection.npy", "correlation_map.npy"):
            _check_file_exists(f"{plane_name}/detection_data/{name}", plane_detection_directory / name, state)
        _check_file_exists(f"{plane_name}/roi_masks.npz", plane_dir / "roi_masks.npz", state)
        _check_file_exists(f"{plane_name}/roi_statistics.npz", plane_dir / "roi_statistics.npz", state)
        for name in (
            "cell_fluorescence.npy",
            "neuropil_fluorescence.npy",
            "subtracted_fluorescence.npy",
            "spikes.npy",
            "cell_classification.npy",
        ):
            _check_file_exists(f"{plane_name}/{name}", plane_dir / name, state)

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

    return {
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


@mcp.tool()
def verify_multi_recording_output_tool(recording_path: str, dataset: str) -> dict[str, object]:
    """Verifies completeness of multi-recording pipeline output for a specific dataset.

    Checks the entry recording's output directory for all expected multi-recording files, then enumerates all
    recordings in the dataset and verifies per-recording output completeness. Reports each expected file as
    present or missing, validates NPZ keys, and synthesizes an overall completeness verdict. Use this after
    multi-recording processing completes to confirm all output was produced.

    Args:
        recording_path: Absolute path to a recording directory that belongs to the dataset.
        dataset: The multi-recording dataset name to verify.

    Returns:
        On success, contains 'complete' flag, 'recording_count', per-recording verification summaries, 'missing'
        files, and optional 'warnings'. On failure, contains an 'error' message. Both cases include a
        'success' flag.
    """
    cindra_root, error = _find_cindra_root(recording_path)
    if cindra_root is None:
        return {"success": False, "error": f"Unable to verify output. {error}"}

    dataset_path, error = _find_multi_recording_root(cindra_root, dataset)
    if dataset_path is None:
        return {"success": False, "error": f"Unable to verify output. {error}"}

    state = _VerificationState()

    # Loads entry recording runtime data to discover all recordings in the dataset.
    runtime_yaml = _load_yaml(dataset_path / "multi_recording_runtime_data.yaml")
    if runtime_yaml is None:
        return {
            "success": False,
            "error": f"Unable to load runtime data from: {dataset_path / 'multi_recording_runtime_data.yaml'}",
        }

    io_data = runtime_yaml.get("io", {})
    dataset_output_paths = io_data.get("dataset_output_paths", [str(dataset_path)])
    recording_count = len(dataset_output_paths)
    recording_results: list[dict[str, Any]] = []

    # Shared configuration (main recording only — first in natural sort order).
    config_found = False
    for output_path_str in dataset_output_paths:
        output_path = Path(output_path_str)
        if (output_path / "multi_recording_configuration.yaml").exists():
            _check_file_exists(
                "multi_recording_configuration.yaml",
                output_path / "multi_recording_configuration.yaml",
                state,
            )
            config_found = True
            break
    if not config_found:
        state.missing.append("multi_recording_configuration.yaml")
        state.total_checks += 1

    # Per-recording verification.
    for i, output_path_str in enumerate(dataset_output_paths):
        output_path = Path(output_path_str)
        recording_prefix = f"recording_{i}"

        recording_runtime = _load_yaml(output_path / "multi_recording_runtime_data.yaml")
        recording_id = (
            recording_runtime.get("io", {}).get("recording_id", f"unknown_{i}") if recording_runtime else f"unknown_{i}"
        )
        recording_result: dict[str, Any] = {
            "index": i,
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
            f"{recording_prefix}/multi_recording_runtime_data.yaml",
            output_path / "multi_recording_runtime_data.yaml",
            state,
        )

        # Registration data.
        registration_directory = output_path / "registration_arrays"
        for name in (
            "deform_field_y.npy",
            "deform_field_x.npy",
            "transformed_mean_image.npy",
            "transformed_enhanced_mean_image.npy",
            "transformed_maximum_projection.npy",
        ):
            _check_file_exists(
                f"{recording_prefix}/registration_arrays/{name}",
                registration_directory / name,
                state,
            )

        _check_file_exists(
            f"{recording_prefix}/registration_deformed_masks.npz",
            output_path / "registration_deformed_masks.npz",
            state,
        )
        _check_npz_keys(
            f"{recording_prefix}/registration_deformed_masks.npz",
            output_path / "registration_deformed_masks.npz",
            ["pixel_counts", "y_pixels", "x_pixels"],
            state,
        )

        # Tracking data.
        _check_file_exists(
            f"{recording_prefix}/tracking_template_masks.npz",
            output_path / "tracking_template_masks.npz",
            state,
        )
        _check_npz_keys(
            f"{recording_prefix}/tracking_template_masks.npz",
            output_path / "tracking_template_masks.npz",
            ["pixel_counts", "cluster_id", "recording_count"],
            state,
        )

        # Extraction data.
        _check_file_exists(f"{recording_prefix}/roi_masks.npz", output_path / "roi_masks.npz", state)
        _check_file_exists(f"{recording_prefix}/roi_statistics.npz", output_path / "roi_statistics.npz", state)
        for name in (
            "cell_fluorescence.npy",
            "neuropil_fluorescence.npy",
            "subtracted_fluorescence.npy",
            "spikes.npy",
        ):
            _check_file_exists(f"{recording_prefix}/{name}", output_path / name, state)

        # Channel 2 files (optional).
        for name in (
            "registration_deformed_masks_channel_2.npz",
            "tracking_template_masks_channel_2.npz",
            "roi_masks_channel_2.npz",
            "roi_statistics_channel_2.npz",
            "cell_fluorescence_channel_2.npy",
            "neuropil_fluorescence_channel_2.npy",
            "subtracted_fluorescence_channel_2.npy",
            "spikes_channel_2.npy",
        ):
            _check_file_exists(f"{recording_prefix}/{name}", output_path / name, state, required=False)

        _check_file_exists(
            f"{recording_prefix}/cell_colocalization.npy",
            output_path / "cell_colocalization.npy",
            state,
            required=False,
        )

        recording_result["complete"] = not any(m.startswith(recording_prefix) for m in state.missing)
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
        On success, contains recording metadata including 'plane_count', 'combined_height', 'combined_width',
        'sampling_rate', 'tau', 'roi_count', 'cell_count', processing 'plane_timing' data, and
        'available_datasets'. On failure, contains an 'error' message. Both cases include a 'success' flag.
    """
    cindra_root, error = _find_cindra_root(recording_path)
    if cindra_root is None:
        return {"success": False, "error": f"Unable to query metadata. {error}"}

    result: dict[str, Any] = {
        "success": True,
        "recording_path": recording_path,
        "cindra_path": str(cindra_root),
    }

    # Loads combined metadata scalars.
    combined_metadata_path = cindra_root / "combined_metadata.npz"
    if combined_metadata_path.exists():
        try:
            metadata = np.load(combined_metadata_path, allow_pickle=False)
            result["plane_count"] = int(metadata["plane_count"][0])
            result["combined_height"] = int(metadata["combined_height"][0])
            result["combined_width"] = int(metadata["combined_width"][0])
            result["tau"] = round(float(metadata["tau"][0]), 4)
            result["sampling_rate"] = round(float(metadata["sampling_rate"][0]), 4)
            result["plane_heights"] = [int(h) for h in metadata["plane_heights"]]
            result["plane_widths"] = [int(w) for w in metadata["plane_widths"]]

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
    classification_path = cindra_root / "cell_classification.npy"
    if classification_path.exists():
        with contextlib.suppress(Exception):
            classification = np.load(classification_path, mmap_mode="r")
            result["roi_count"] = int(classification.shape[0])
            result["cell_count"] = int(np.sum(classification[:, 0] > _CELL_LABEL_THRESHOLD))
            result["non_cell_count"] = result["roi_count"] - result["cell_count"]

    # Frame count from fluorescence traces (memory-mapped for efficiency).
    fluorescence_path = cindra_root / "cell_fluorescence.npy"
    if fluorescence_path.exists():
        with contextlib.suppress(Exception):
            fluorescence = np.load(fluorescence_path, mmap_mode="r")
            result["frame_count"] = int(fluorescence.shape[1])

    # Per-plane timing data from runtime_data.yaml files.
    planes = _list_plane_directories(cindra_root)
    timing_entries: list[dict[str, Any]] = []
    for plane_dir in planes:
        runtime = _load_yaml(plane_dir / "runtime_data.yaml")
        if runtime is None:
            continue
        timing = runtime.get("timing", {})
        io_section = runtime.get("io", {})
        entry: dict[str, Any] = {"plane": plane_dir.name}

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
            "total_plane_time",
            "date_processed",
            "python_version",
            "cindra_version",
        ):
            value = timing.get(field_name)
            if value is not None:
                entry[field_name] = round(value, 2) if isinstance(value, float) else value
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
        On success, contains registration offset summaries ('rigid_y_offsets', 'rigid_x_offsets'), correlation
        summaries, 'bad_frame_count' and 'bad_frame_percentage', optional nonrigid offset summaries, and
        optional 'pc_shift_metrics'. On failure, contains an 'error' message. Both cases include a
        'success' flag.
    """
    cindra_root, error = _find_cindra_root(recording_path)
    if cindra_root is None:
        return {"success": False, "error": f"Unable to query registration quality. {error}"}

    plane_path, error = _resolve_data_path(cindra_root, plane_index)
    if plane_path is None:
        return {"success": False, "error": f"Unable to query registration quality. {error}"}

    registration_directory = plane_path / "registration_data"
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
        ("rigid_y_offsets.npy", "rigid_y_offsets"),
        ("rigid_x_offsets.npy", "rigid_x_offsets"),
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
    correlation_path = registration_directory / "rigid_correlations.npy"
    if correlation_path.exists():
        with contextlib.suppress(Exception):
            result["rigid_correlations"] = _array_summary(np.load(correlation_path, mmap_mode="r"))

    # Bad frames.
    bad_frames_path = registration_directory / "bad_frames.npy"
    if bad_frames_path.exists():
        with contextlib.suppress(Exception):
            bad_frames = np.load(bad_frames_path, mmap_mode="r")
            total_frames = len(bad_frames)
            bad_count = int(np.sum(bad_frames))
            result["total_frames"] = total_frames
            result["bad_frame_count"] = bad_count
            result["bad_frame_percentage"] = round(100.0 * bad_count / total_frames, 2) if total_frames > 0 else 0.0

    # Nonrigid registration offsets (optional).
    for name, key in [
        ("nonrigid_y_offsets.npy", "nonrigid_y_offsets"),
        ("nonrigid_x_offsets.npy", "nonrigid_x_offsets"),
    ]:
        path = registration_directory / name
        if path.exists():
            with contextlib.suppress(Exception):
                array = np.load(path, mmap_mode="r")
                summary = _array_summary(array)
                summary["shape"] = list(array.shape)
                summary["num_blocks"] = int(array.shape[1]) if array.ndim > 1 else 0
                result[key] = summary

    nonrigid_correlation_path = registration_directory / "nonrigid_correlations.npy"
    if nonrigid_correlation_path.exists():
        with contextlib.suppress(Exception):
            result["nonrigid_correlations"] = _array_summary(np.load(nonrigid_correlation_path, mmap_mode="r"))

    # Principal component shift metrics (optional).
    pc_metrics_path = registration_directory / "principal_component_shift_metrics.npy"
    if pc_metrics_path.exists():
        with contextlib.suppress(Exception):
            pc_metrics = np.load(pc_metrics_path, mmap_mode="r")
            # Shape: (num_components, 3) — columns: mean rigid, mean nonrigid, max nonrigid.
            result["pc_shift_metrics"] = [
                {
                    "component": i,
                    "mean_rigid_shift": round(float(pc_metrics[i, 0]), 4),
                    "mean_nonrigid_shift": round(float(pc_metrics[i, 1]), 4),
                    "max_nonrigid_shift": round(float(pc_metrics[i, 2]), 4),
                }
                for i in range(pc_metrics.shape[0])
            ]
            result["pc_component_count"] = int(pc_metrics.shape[0])

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
        plane_index: -1 for combined view (default), 0+ for a specific imaging plane.

    Returns:
        On success, contains per-image intensity statistics, 'roi_diameter', and 'aspect_ratio'. On failure,
        contains an 'error' message. Both cases include a 'success' flag.
    """
    cindra_root, error = _find_cindra_root(recording_path)
    if cindra_root is None:
        return {"success": False, "error": f"Unable to query detection summary. {error}"}

    data_path, error = _resolve_data_path(cindra_root, plane_index)
    if data_path is None:
        return {"success": False, "error": f"Unable to query detection summary. {error}"}

    detection_directory = data_path / "detection_data"
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
        "mean_image": "mean_image.npy",
        "enhanced_mean_image": "enhanced_mean_image.npy",
        "maximum_projection": "maximum_projection.npy",
        "correlation_map": "correlation_map.npy",
        "mean_image_channel_2": "mean_image_channel_2.npy",
        "enhanced_mean_image_channel_2": "enhanced_mean_image_channel_2.npy",
        "maximum_projection_channel_2": "maximum_projection_channel_2.npy",
        "correlation_map_channel_2": "correlation_map_channel_2.npy",
    }
    for label, filename in image_files.items():
        path = detection_directory / filename
        if path.exists():
            try:
                image = np.load(path, mmap_mode="r")
                stats = _array_summary(image)
                stats["shape"] = list(image.shape)
                result["images"][label] = stats
            except Exception as error:
                result["images"][label] = {"error": str(error)}

    # ROI diameter and aspect ratio from per-plane runtime data.
    source_plane = _list_plane_directories(cindra_root)[0] if plane_index == -1 else data_path
    if source_plane is not None:
        runtime = _load_yaml(source_plane / "runtime_data.yaml")
        if runtime is not None:
            detection_metadata = runtime.get("detection", {})
            if detection_metadata.get("roi_diameter") is not None:
                result["roi_diameter"] = detection_metadata["roi_diameter"]
            if detection_metadata.get("aspect_ratio") is not None:
                result["aspect_ratio"] = round(float(detection_metadata["aspect_ratio"]), 4)

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

    Returns statistics including pixel count, skewness, compactness, footprint area, aspect ratio, solidity, and
    centroid coordinates for the requested ROIs. In single-recording mode (dataset is None), also returns cell
    classification labels. In multi-recording mode (dataset is provided), enriches entries with cluster ID and
    recording count from tracking metadata when available. Supports sorting by any statistic and limiting to top N
    results for efficient quality assessment.

    Args:
        recording_path: Absolute path to a cindra pipeline output directory.
        roi_indices: Specific ROI indices to query. Returns all ROIs when not provided (up to 500).
        sort_by: Sort results by this statistic name ('skewness', 'compactness', 'footprint', 'aspect_ratio',
            'pixel_count', 'solidity', 'normalized_pixel_count'). Results are returned in descending order.
        top_n: When sort_by is provided, returns the top N after sorting; otherwise returns the first N entries.
        plane_index: -1 for combined view (default), 0+ for a specific imaging plane. Only used in single-recording
            mode.
        dataset: The multi-recording dataset name. When provided, switches to multi-recording mode and ignores
            plane_index.
        recording_index: The recording index within the dataset to query (0-based). Only used in multi-recording mode.
            Defaults to 0 (entry recording) when not provided.

    Returns:
        On success, contains 'total_rois', 'queried_count', and 'rois' list with per-ROI statistics. In
        single-recording mode, includes 'total_cells', 'total_non_cells', and per-ROI classification data. In
        multi-recording mode, includes 'has_template_metadata' and optional tracking metadata per ROI. On failure,
        contains an 'error' message. Both cases include a 'success' flag.
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
        data_path, error = _resolve_data_path(cindra_root, plane_index)
        if data_path is None:
            return {"success": False, "error": f"Unable to query ROI statistics. {error}"}
        recording_id = None

    stats_path = data_path / "roi_statistics.npz"
    masks_path = data_path / "roi_masks.npz"
    if not stats_path.exists() or not masks_path.exists():
        return {
            "success": False,
            "error": f"Unable to query ROI statistics. ROI data files not found at: {data_path}.",
        }

    try:
        stats_data = np.load(stats_path, allow_pickle=False)
        masks_data = np.load(masks_path, allow_pickle=False)
    except Exception as load_error:
        return {"success": False, "error": f"Unable to load ROI data: {load_error}"}

    entries, total_rois = _build_roi_statistics_entries(
        stats_data=stats_data,
        masks_data=masks_data,
        roi_indices=roi_indices,
        include_plane_index=(dataset is None),
    )

    # Enriches entries with mode-specific metadata.
    if dataset is None:
        # Single-recording mode: adds classification data.
        classification_path = data_path / "cell_classification.npy"
        classification = None
        if classification_path.exists():
            with contextlib.suppress(Exception):
                classification = np.load(classification_path, mmap_mode="r")

        if classification is not None:
            for _, entry in entries:
                i = entry["roi_index"]
                if i < classification.shape[0]:
                    entry["is_cell"] = bool(classification[i, 0] > _CELL_LABEL_THRESHOLD)
                    entry["classification_probability"] = round(float(classification[i, 1]), 4)
    else:
        # Multi-recording mode: adds tracking template metadata.
        template_data: dict[str, Any] | None = None
        template_path = data_path / "tracking_template_masks.npz"
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
                i = entry["roi_index"]
                if i < len(template_data["cluster_id"]):
                    entry["cluster_id"] = int(template_data["cluster_id"][i])
                    entry["recording_count"] = int(template_data["recording_count"][i])

    # Applies sorting and capping.
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
) -> dict[str, object]:
    """Queries fluorescence trace data for specific ROIs from a cindra-processed recording or multi-recording dataset.

    Returns trace arrays for up to 50 ROIs at a time. Large traces can be downsampled to reduce response size.
    Supports querying raw cell fluorescence, neuropil fluorescence, neuropil-subtracted corrected traces, or
    deconvolved spike estimates. In single-recording mode (dataset is None), queries from the combined view or a
    specific imaging plane. In multi-recording mode (dataset is provided), queries from the specified recording's
    output directory within the dataset.

    Args:
        recording_path: Absolute path to a cindra pipeline output directory.
        roi_indices: List of ROI indices to retrieve traces for (maximum 50).
        trace_type: The type of fluorescence trace to return. 'fluorescence' for raw cell fluorescence,
            'neuropil' for neuropil fluorescence, 'corrected' for neuropil-subtracted, 'spikes' for deconvolved.
        downsample_factor: Factor by which to downsample traces (1 = no downsampling, 10 = every 10th sample).
        plane_index: -1 for combined view (default), 0+ for a specific imaging plane. Only used in single-recording
            mode.
        dataset: The multi-recording dataset name. When provided, switches to multi-recording mode and ignores
            plane_index.
        recording_index: The recording index within the dataset to query (0-based). Only used in multi-recording mode.
            Defaults to 0 (entry recording) when not provided.

    Returns:
        On success, contains 'trace_type', 'downsample_factor', 'frame_count', and 'traces' list with per-ROI
        trace arrays. On failure, contains an 'error' message. Both cases include a 'success' flag.
    """
    if len(roi_indices) > _MAX_TRACE_ROIS:
        return {
            "success": False,
            "error": f"Unable to query traces. Requested {len(roi_indices)} ROIs, maximum is {_MAX_TRACE_ROIS}.",
        }

    file_map = {
        "fluorescence": "cell_fluorescence.npy",
        "neuropil": "neuropil_fluorescence.npy",
        "corrected": "subtracted_fluorescence.npy",
        "spikes": "spikes.npy",
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
        data_path, error = _resolve_data_path(cindra_root, plane_index)
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
    valid_indices = [i for i in roi_indices if 0 <= i < roi_count]
    if not valid_indices:
        return {"success": False, "error": "Unable to query traces. No valid ROI indices provided."}

    downsample_factor = max(1, downsample_factor)
    results: list[dict[str, Any]] = []
    for roi_index in valid_indices:
        trace = traces[roi_index]
        if downsample_factor > 1:
            trace = trace[::downsample_factor]
        results.append({"roi_index": roi_index, "trace": [round(float(value), 4) for value in trace]})

    result: dict[str, object] = {
        "success": True,
        "trace_type": trace_type,
        "downsample_factor": downsample_factor,
        "frame_count": int(traces.shape[1]),
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
        recording_path: Absolute path to a recording directory that belongs to the dataset.
        dataset: The multi-recording dataset name to query.

    Returns:
        On success, contains 'recording_count', 'template_roi_count', and per-recording summaries with mask
        counts, timing, and completion flags. On failure, contains an 'error' message. Both cases include a
        'success' flag.
    """
    cindra_root, error = _find_cindra_root(recording_path)
    if cindra_root is None:
        return {"success": False, "error": f"Unable to query multi-recording overview. {error}"}

    dataset_path, error = _find_multi_recording_root(cindra_root, dataset)
    if dataset_path is None:
        return {"success": False, "error": f"Unable to query multi-recording overview. {error}"}

    runtime = _load_yaml(dataset_path / "multi_recording_runtime_data.yaml")
    if runtime is None:
        return {"success": False, "error": f"Unable to load runtime data from: {dataset_path}"}

    dataset_output_paths = runtime.get("io", {}).get("dataset_output_paths", [str(dataset_path)])
    recordings: list[dict[str, Any]] = []
    template_roi_count: int | None = None

    for i, output_path_str in enumerate(dataset_output_paths):
        output_path = Path(output_path_str)
        recording_entry: dict[str, Any] = {"index": i, "output_path": str(output_path)}

        if not output_path.exists():
            recording_entry["exists"] = False
            recordings.append(recording_entry)
            continue

        recording_entry["exists"] = True
        recording_runtime = _load_yaml(output_path / "multi_recording_runtime_data.yaml")
        if recording_runtime is not None:
            recording_io = recording_runtime.get("io", {})
            recording_entry["recording_id"] = recording_io.get("recording_id", f"unknown_{i}")
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
                    recording_entry[field_name] = round(value, 2) if isinstance(value, float) else value

        # Mask counts from NPZ files.
        for npz_name, key_name in [
            ("registration_deformed_masks.npz", "deformed_mask_count"),
            ("tracking_template_masks.npz", "template_mask_count"),
            ("roi_masks.npz", "tracked_mask_count"),
        ]:
            npz_path = output_path / npz_name
            if npz_path.exists():
                with contextlib.suppress(Exception):
                    data = np.load(npz_path, allow_pickle=False)
                    count = len(data["pixel_counts"])
                    recording_entry[key_name] = count
                    if key_name == "template_mask_count" and template_roi_count is None:
                        template_roi_count = count

        recording_entry["has_channel_2"] = (output_path / "registration_deformed_masks_channel_2.npz").exists()
        recording_entry["extraction_complete"] = (output_path / "cell_fluorescence.npy").exists()
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
        dataset: The multi-recording dataset name to query.

    Returns:
        On success, contains per-recording deformation field statistics and image availability. On failure,
        contains an 'error' message. Both cases include a 'success' flag.
    """
    cindra_root, error = _find_cindra_root(recording_path)
    if cindra_root is None:
        return {"success": False, "error": f"Unable to query registration quality. {error}"}

    dataset_path, error = _find_multi_recording_root(cindra_root, dataset)
    if dataset_path is None:
        return {"success": False, "error": f"Unable to query registration quality. {error}"}

    runtime = _load_yaml(dataset_path / "multi_recording_runtime_data.yaml")
    if runtime is None:
        return {"success": False, "error": f"Unable to load runtime data from: {dataset_path}"}

    dataset_output_paths = runtime.get("io", {}).get("dataset_output_paths", [str(dataset_path)])
    recordings: list[dict[str, Any]] = []

    for i, output_path_str in enumerate(dataset_output_paths):
        output_path = Path(output_path_str)
        recording_entry: dict[str, Any] = {"index": i}

        recording_runtime = _load_yaml(output_path / "multi_recording_runtime_data.yaml")
        if recording_runtime is not None:
            recording_entry["recording_id"] = recording_runtime.get("io", {}).get("recording_id", f"unknown_{i}")

        registration_directory = output_path / "registration_arrays"
        if not registration_directory.exists():
            recording_entry["registration_available"] = False
            recordings.append(recording_entry)
            continue

        recording_entry["registration_available"] = True

        # Deformation field statistics.
        for field_name, file_name in [
            ("deform_field_y", "deform_field_y.npy"),
            ("deform_field_x", "deform_field_x.npy"),
        ]:
            path = registration_directory / file_name
            if path.exists():
                with contextlib.suppress(Exception):
                    field_array = np.load(path, mmap_mode="r")
                    stats = _array_summary(field_array)
                    stats["shape"] = list(field_array.shape)
                    stats["abs_mean"] = round(float(np.mean(np.abs(field_array))), 4)
                    stats["abs_max"] = round(float(np.max(np.abs(field_array))), 4)
                    recording_entry[field_name] = stats

        # Combined displacement magnitude.
        y_path = registration_directory / "deform_field_y.npy"
        x_path = registration_directory / "deform_field_x.npy"
        if y_path.exists() and x_path.exists():
            with contextlib.suppress(Exception):
                y_field = np.load(y_path, mmap_mode="r")
                x_field = np.load(x_path, mmap_mode="r")
                magnitude = np.sqrt(y_field**2 + x_field**2)
                recording_entry["displacement_magnitude"] = _array_summary(magnitude)

        # Transformed image availability.
        recording_entry["transformed_images"] = {
            "mean_image": (registration_directory / "transformed_mean_image.npy").exists(),
            "enhanced_mean_image": (registration_directory / "transformed_enhanced_mean_image.npy").exists(),
            "maximum_projection": (registration_directory / "transformed_maximum_projection.npy").exists(),
        }
        recording_entry["channel_2_images"] = {
            "mean_image": (registration_directory / "transformed_mean_image_channel_2.npy").exists(),
            "enhanced_mean_image": (registration_directory / "transformed_enhanced_mean_image_channel_2.npy").exists(),
            "maximum_projection": (registration_directory / "transformed_maximum_projection_channel_2.npy").exists(),
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
    an ROI was detected in, not tracking reliability — ROIs can be active in some sessions and inactive in
    others.

    Args:
        recording_path: Absolute path to a recording directory that belongs to the dataset.
        dataset: The multi-recording dataset name to query.

    Returns:
        On success, contains 'template_count', 'recording_count_distribution' histogram, cluster statistics,
        and per-template summary data. On failure, contains an 'error' message. Both cases include a
        'success' flag.
    """
    cindra_root, error = _find_cindra_root(recording_path)
    if cindra_root is None:
        return {"success": False, "error": f"Unable to query tracking summary. {error}"}

    dataset_path, error = _find_multi_recording_root(cindra_root, dataset)
    if dataset_path is None:
        return {"success": False, "error": f"Unable to query tracking summary. {error}"}

    template_path = dataset_path / "tracking_template_masks.npz"
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
    distribution = {int(count): int(freq) for count, freq in zip(unique_counts, histogram, strict=True)}

    # Per-template summary (capped for large datasets).
    max_templates = 200
    templates: list[dict[str, Any]] = [
        {
            "index": i,
            "centroid": [int(centroids[i, 0]), int(centroids[i, 1])],
            "pixel_count": int(pixel_counts[i]),
            "cluster_id": int(cluster_ids[i]),
            "recording_count": int(recording_counts[i]),
        }
        for i in range(min(template_count, max_templates))
    ]

    result: dict[str, Any] = {
        "success": True,
        "recording_path": recording_path,
        "dataset": dataset,
        "template_count": template_count,
        "recording_count_distribution": distribution,
        "mean_recording_count": round(float(np.mean(recording_counts)), 2),
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
    channel_2_path = dataset_path / "tracking_template_masks_channel_2.npz"
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
) -> dict[str, object]:
    """Queries fluorescence traces for specific ROIs across all recordings in a multi-recording dataset.

    For each requested ROI, retrieves trace data from every recording in the dataset, enabling cross-recording
    comparison of tracked ROI activity. Recordings where extraction is incomplete are skipped and reported.
    Use this to compare longitudinal activity patterns for the same ROIs across sessions.

    Args:
        recording_path: Absolute path to a recording directory that belongs to the dataset.
        dataset: The multi-recording dataset name to query.
        roi_indices: List of ROI indices to retrieve traces for across all recordings (maximum 50).
        trace_type: The type of fluorescence trace to return. 'fluorescence' for raw cell fluorescence,
            'neuropil' for neuropil fluorescence, 'corrected' for neuropil-subtracted, 'spikes' for deconvolved.
        downsample_factor: Factor by which to downsample traces (1 = no downsampling, 10 = every 10th sample).

    Returns:
        On success, contains 'recording_count', per-ROI 'rois' with per-recording traces, and optional
        'skipped_recordings'. On failure, contains an 'error' message. Both cases include a 'success' flag.
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
        "fluorescence": "cell_fluorescence.npy",
        "neuropil": "neuropil_fluorescence.npy",
        "corrected": "subtracted_fluorescence.npy",
        "spikes": "spikes.npy",
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

    dataset_path, error = _find_multi_recording_root(cindra_root, dataset)
    if dataset_path is None:
        return {"success": False, "error": f"Unable to query cross-recording traces. {error}"}

    runtime = _load_yaml(dataset_path / "multi_recording_runtime_data.yaml")
    if runtime is None:
        return {
            "success": False,
            "error": f"Unable to load runtime data from: {dataset_path / 'multi_recording_runtime_data.yaml'}",
        }

    dataset_output_paths = runtime.get("io", {}).get("dataset_output_paths", [str(dataset_path)])

    # Builds recording info list from per-recording runtime data.
    recording_info: list[tuple[int, str, Path]] = []
    for i, output_path_str in enumerate(dataset_output_paths):
        output_path = Path(output_path_str)
        recording_runtime = _load_yaml(output_path / "multi_recording_runtime_data.yaml")
        recording_id = (
            recording_runtime.get("io", {}).get("recording_id", f"unknown_{i}")
            if recording_runtime is not None
            else f"unknown_{i}"
        )
        recording_info.append((i, recording_id, output_path))

    downsample_factor = max(1, downsample_factor)
    skipped_recordings: list[dict[str, object]] = []
    skipped_keys: set[tuple[int, str]] = set()

    # Collects traces for each ROI across all recordings.
    rois_result: list[dict[str, object]] = []
    for roi_index in roi_indices:
        per_recording: list[dict[str, object]] = []

        for recording_index, recording_id, output_path in recording_info:
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

            trace = traces[roi_index]
            if downsample_factor > 1:
                trace = trace[::downsample_factor]

            per_recording.append(
                {
                    "recording_index": recording_index,
                    "recording_id": recording_id,
                    "frame_count": int(traces.shape[1]),
                    "trace": [round(float(value), 4) for value in trace],
                }
            )

        rois_result.append({"roi_index": roi_index, "recordings": per_recording})

    result: dict[str, object] = {
        "success": True,
        "recording_path": recording_path,
        "dataset": dataset,
        "trace_type": trace_type,
        "downsample_factor": downsample_factor,
        "recording_count": len(recording_info),
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
    dataset_path, error = _find_multi_recording_root(cindra_root, dataset)
    if dataset_path is None:
        return None, None, error

    runtime = _load_yaml(dataset_path / "multi_recording_runtime_data.yaml")
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
    recording_runtime = _load_yaml(output_path / "multi_recording_runtime_data.yaml")
    recording_id = (
        recording_runtime.get("io", {}).get("recording_id", f"unknown_{effective_index}")
        if recording_runtime is not None
        else f"unknown_{effective_index}"
    )

    return output_path, recording_id, None


def _build_roi_statistics_entries(
    stats_data: np.lib.npyio.NpzFile,
    masks_data: np.lib.npyio.NpzFile,
    roi_indices: list[int] | None,
    *,
    include_plane_index: bool,
) -> tuple[list[tuple[int, dict[str, Any]]], int]:
    """Builds per-ROI statistics entries from loaded NPZ data.

    Args:
        stats_data: The loaded roi_statistics.npz data.
        masks_data: The loaded roi_masks.npz data.
        roi_indices: Specific ROI indices to include, or None for all ROIs.
        include_plane_index: Determines whether to include the plane_index field in each entry.

    Returns:
        A tuple of (entries, total_rois) where entries is a list of (index, entry_dict) pairs.
    """
    footprints = stats_data["footprints"]
    compactness = stats_data["compactness"]
    solidity = stats_data["solidity"]
    pixel_count = stats_data["pixel_count"]
    aspect_ratio = stats_data["aspect_ratio"]
    normalized_pixel_count = stats_data["normalized_pixel_count"]
    skewness = stats_data["skewness"]
    centroids = masks_data["centroids"]
    total_rois = len(footprints)

    indices = list(range(total_rois)) if roi_indices is None else [i for i in roi_indices if 0 <= i < total_rois]
    entries: list[tuple[int, dict[str, Any]]] = []
    for i in indices:
        entry: dict[str, Any] = {
            "roi_index": i,
            "centroid": [int(centroids[i, 0]), int(centroids[i, 1])],
            "pixel_count": int(pixel_count[i]),
            "footprint": int(footprints[i]),
            "compactness": round(float(compactness[i]), 4),
            "solidity": round(float(solidity[i]), 4),
            "aspect_ratio": round(float(aspect_ratio[i]), 4),
            "normalized_pixel_count": round(float(normalized_pixel_count[i]), 4),
        }
        if include_plane_index:
            entry["plane_index"] = int(stats_data["plane_index"][i])

        skewness_value = skewness[i]
        entry["skewness"] = round(float(skewness_value), 4) if not np.isnan(skewness_value) else None
        entries.append((i, entry))

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

    Args:
        recording_path: Absolute path to the recording data directory.

    Returns:
        A tuple of (cindra_root, error_message). If cindra_root is None, error_message describes the issue.
    """
    recording = Path(recording_path)
    if not recording.exists():
        return None, f"Recording directory not found: {recording_path}"

    cindra_path = recording / "cindra"
    if cindra_path.exists():
        return cindra_path, None

    # Falls back to recursive search for configuration.yaml (handles non-standard nesting).
    matches = list(recording.rglob("configuration.yaml"))
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
    dataset_path = cindra_root / "multi_recording" / dataset
    if dataset_path.exists():
        return dataset_path, None

    multi_recording_path = cindra_root / "multi_recording"
    if not multi_recording_path.exists():
        return None, f"No multi_recording directory found under: {cindra_root}"

    available = [d.name for d in multi_recording_path.iterdir() if d.is_dir()]
    if not available:
        return None, "No dataset directories found under multi_recording/"
    return None, f"Dataset '{dataset}' not found. Available datasets: {', '.join(sorted(available))}"


def _resolve_data_path(cindra_root: Path, plane_index: int) -> tuple[Path | None, str | None]:
    """Resolves the data path for combined or per-plane queries.

    Args:
        cindra_root: The cindra output directory path.
        plane_index: -1 for combined view, 0+ for per-plane view.

    Returns:
        A tuple of (data_path, error_message).
    """
    if plane_index == -1:
        return cindra_root, None

    plane_path = cindra_root / f"plane_{plane_index}"
    if not plane_path.exists():
        available = sorted(p.name for p in cindra_root.iterdir() if p.is_dir() and p.name.startswith("plane_"))
        return None, f"Plane directory plane_{plane_index} not found. Available: {', '.join(available) or 'none'}"

    return plane_path, None


def _array_summary(array: NDArray[np.float32]) -> dict[str, object]:
    """Computes summary statistics for a numpy array.

    Args:
        array: The numpy array to summarize.

    Returns:
        A dictionary containing the min, max, mean, and standard deviation of the array.
    """
    return {
        "min": round(float(np.nanmin(array)), 4),
        "max": round(float(np.nanmax(array)), 4),
        "mean": round(float(np.nanmean(array)), 4),
        "std": round(float(np.nanstd(array)), 4),
    }


def _load_yaml(file_path: Path) -> dict[str, Any] | None:
    """Loads a YAML file and returns the parsed dictionary, or None if loading fails.

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


def _list_plane_directories(cindra_root: Path) -> list[Path]:
    """Returns sorted plane directories found under the cindra root.

    Args:
        cindra_root: The cindra output directory path to search for plane directories.

    Returns:
        A naturally-sorted list of plane directory paths found under the given root.
    """
    return sorted(
        [p for p in cindra_root.iterdir() if p.is_dir() and p.name.startswith("plane_")],
        key=lambda p: p.name,
    )


def _discover_available_datasets(cindra_root: Path) -> list[str]:
    """Discovers available multi-recording dataset names under the cindra root.

    Args:
        cindra_root: The cindra output directory path to search for multi-recording datasets.

    Returns:
        A list of dataset name strings discovered under the given recording path.
    """
    multi_recording_path = cindra_root / "multi_recording"
    if not multi_recording_path.exists():
        return []
    return sorted(d.name for d in multi_recording_path.iterdir() if d.is_dir())


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
        path: The filesystem path to check.
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
