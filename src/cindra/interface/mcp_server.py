"""Provides the MCP server for agentic neural imaging data processing.

Exposes tools that enable AI agents to discover recordings, execute pipelines, and monitor processing status for both
single-recording and multi-recording cindra processing workflows.
"""

from __future__ import annotations

from typing import Any, Literal
from pathlib import Path
from threading import Lock, Thread
import traceback
from dataclasses import field, dataclass

from natsort import natsorted
from ataraxis_time import PrecisionTimer, TimerPrecisions
from mcp.server.fastmcp import FastMCP
from ataraxis_base_utilities import resolve_worker_count, resolve_parallel_job_capacity

from ..io import resolve_multi_recording_contexts
from ..pipelines import run_multi_recording_pipeline, run_single_recording_pipeline
from ..dataclasses import (
    RuntimeContext,
    MultiRecordingConfiguration,
    SingleRecordingConfiguration,
)

mcp = FastMCP(name="cindra-mcp", json_response=True)
"""The MCP server instance initialized with JSON response mode for structured output."""

_RESERVED_CORES: int = 4
"""The number of CPU cores reserved for system operations."""

_MAXIMUM_JOB_CORES: int = 30
"""The maximum number of CPU cores any single job can use."""

_MINIMUM_RECORDING_COUNT: int = 2
"""The minimum number of recordings required for multi-recording processing."""


@dataclass
class _SingleRecordingBatchState:
    """Tracks state for single-recording batch processing operations."""

    recordings: list[Path] = field(default_factory=list)
    """All recordings to process."""
    configuration_path: Path | None = None
    """Template configuration file path."""
    recording_configuration_paths: dict[str, Path] = field(default_factory=dict)
    """Per-recording configuration file paths (recording_key -> configuration_path)."""
    current_phase: str = "binarize"
    """Current processing phase: 'binarize', 'process', or 'combine'."""

    # Tracks binarize phase state.
    binarize_queue: list[Path] = field(default_factory=list)
    """Recordings waiting to binarize."""
    binarize_active: dict[str, Thread] = field(default_factory=dict)
    """Currently binarizing (recording_key -> thread)."""
    binarize_completed: set[str] = field(default_factory=set)
    """Recordings that finished binarizing."""
    binarize_failed: set[str] = field(default_factory=set)
    """Recordings that failed binarization."""
    plane_counts: dict[str, int] = field(default_factory=dict)
    """Recording -> plane count (discovered during binarize)."""

    # Tracks process phase state.
    process_queue: list[tuple[str, int]] = field(default_factory=list)
    """(recording_key, plane_index) pairs to process."""
    process_active: dict[str, Thread] = field(default_factory=dict)
    """Currently processing (recording_plane_key -> thread)."""
    process_completed: set[str] = field(default_factory=set)
    """Completed recording_plane keys."""
    process_failed: set[str] = field(default_factory=set)
    """Failed recording_plane keys."""

    # Tracks combine phase state.
    combine_queue: list[Path] = field(default_factory=list)
    """Recordings waiting to combine."""
    combine_active: dict[str, Thread] = field(default_factory=dict)
    """Currently combining (recording_key -> thread)."""
    combine_completed: set[str] = field(default_factory=set)
    """Recordings that finished combining."""
    combine_failed: set[str] = field(default_factory=set)
    """Recordings that failed combination."""

    # Stores resource allocation settings.
    workers_per_plane: int = 30
    """CPU cores per plane job."""
    max_parallel_planes: int = 1
    """Max concurrent plane jobs."""

    # Stores per-recording and per-plane error messages.
    errors: dict[str, list[str]] = field(default_factory=dict)
    """Recording/plane key -> error messages."""

    lock: Lock = field(default_factory=Lock)
    """Thread synchronization lock."""
    manager_thread: Thread | None = None
    """Background manager thread."""


@dataclass
class _MultiRecordingBatchState:
    """Tracks state for multi-recording batch processing operations."""

    animals: list[tuple[Path, list[Path]]] = field(default_factory=list)
    """(configuration_path, recording_paths) per animal."""
    current_phase: str = "discover"
    """Current processing phase: 'discover' or 'extract'."""

    # Tracks discover phase state per animal.
    discover_queue: list[str] = field(default_factory=list)
    """Animal keys waiting to discover."""
    discover_active: dict[str, Thread] = field(default_factory=dict)
    """Currently discovering (animal_key -> thread)."""
    discover_completed: set[str] = field(default_factory=set)
    """Animals that finished discovery."""
    discover_failed: set[str] = field(default_factory=set)
    """Animals that failed discovery."""

    # Tracks extract phase state per recording across all animals.
    extract_queue: list[tuple[str, str]] = field(default_factory=list)
    """(animal_key, recording_id) pairs."""
    extract_active: dict[str, Thread] = field(default_factory=dict)
    """Currently extracting (animal_recording_key -> thread)."""
    extract_completed: set[str] = field(default_factory=set)
    """Completed extractions."""
    extract_failed: set[str] = field(default_factory=set)
    """Failed extractions."""

    # Stores recording IDs per animal, populated during the discover phase.
    recording_ids: dict[str, list[str]] = field(default_factory=dict)
    """animal_key -> list of recording_ids."""

    # Stores resource allocation settings.
    workers_per_discover: int = 20
    """Workers for discover phase."""
    max_parallel_discovers: int = 1
    """Max concurrent discovers."""
    workers_per_extract: int = 30
    """Workers for extract phase."""
    max_parallel_extracts: int = 1
    """Max concurrent extractions."""
    progress_bars: bool = False
    """Determines whether to display progress bars during processing."""

    # Stores per-animal and per-recording error messages.
    errors: dict[str, list[str]] = field(default_factory=dict)
    """Key -> error messages."""

    lock: Lock = field(default_factory=Lock)
    """Thread synchronization lock."""
    manager_thread: Thread | None = None
    """Background manager thread."""


_single_recording_batch_state: _SingleRecordingBatchState | None = None
"""The module-level batch processing state for single-recording operations."""

_multi_recording_batch_state: _MultiRecordingBatchState | None = None
"""The module-level batch processing state for multi-recording operations."""


def run_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Starts the MCP server with the specified transport.

    Args:
        transport: The transport type to use ('stdio', 'sse', or 'streamable-http').
    """
    mcp.run(transport=transport)


@mcp.tool()
def generate_config_file(
    output_path: str, pipeline_type: Literal["single-recording", "multi-recording"]
) -> dict[str, Any]:
    """Generates a default configuration YAML file for the specified pipeline type.

    Creates a configuration file with sensible defaults that can be used directly or modified before processing.

    Args:
        output_path: The absolute path where the configuration file should be saved.
        pipeline_type: The type of pipeline configuration to generate ('single-recording' or 'multi-recording').
    """
    output = Path(output_path)

    if not output.parent.exists():
        return {"success": False, "error": f"Parent directory does not exist: {output.parent}"}

    if output.suffix != ".yaml":
        output = output.with_suffix(".yaml")

    if pipeline_type == "single-recording":
        configuration: SingleRecordingConfiguration | MultiRecordingConfiguration = SingleRecordingConfiguration()
    else:
        configuration = MultiRecordingConfiguration()

    configuration.save(file_path=output)

    return {"success": True, "file_path": str(output), "pipeline_type": pipeline_type}


@mcp.tool()
def get_single_recording_status(recording_path: str) -> dict[str, Any]:
    """Gets the processing status of a single-recording recording.

    Args:
        recording_path: The absolute path to the recording data directory.
    """
    recording = Path(recording_path)

    if not recording.exists():
        return {"success": False, "error": f"Recording directory not found: {recording_path}"}

    cindra_path = recording / "cindra"
    if not cindra_path.exists():
        # Searches recursively for the RuntimeContext configuration marker.
        matches = list(recording.rglob("configuration.yaml"))
        if matches:
            cindra_path = matches[0].parent

    if not cindra_path.exists():
        return {
            "success": True,
            "recording_path": str(recording),
            "status": "not_started",
            "message": "No cindra output directory found",
        }

    combined_path = cindra_path / "combined"
    planes = [p for p in cindra_path.iterdir() if p.is_dir() and p.name.startswith("plane")]

    status: dict[str, Any] = {
        "success": True,
        "recording_path": str(recording),
        "cindra_path": str(cindra_path),
        "planes_found": len(planes),
        "combined_exists": combined_path.exists(),
    }

    if combined_path.exists():
        status["combined_files"] = {
            "combined_metadata": (combined_path / "combined_metadata.npz").exists(),
            "stat": (combined_path / "stat.npy").exists(),
            "F": (combined_path / "F.npy").exists(),
            "Fneu": (combined_path / "Fneu.npy").exists(),
            "spks": (combined_path / "spks.npy").exists(),
            "iscell": (combined_path / "iscell.npy").exists(),
        }

    return status


@mcp.tool()
def get_multi_recording_status(recording_path: str) -> dict[str, Any]:
    """Gets the multi-recording processing status for a recording.

    Args:
        recording_path: The absolute path to a recording directory.
    """
    recording = Path(recording_path)

    if not recording.exists():
        return {"success": False, "error": f"Recording directory not found: {recording_path}"}

    # Finds the cindra directory first, using the same pattern as get_single_recording_status.
    cindra_path = recording / "cindra"
    if not cindra_path.exists():
        matches = list(recording.rglob("configuration.yaml"))
        if matches:
            cindra_path = matches[0].parent

    multi_recording_base = cindra_path / "multi_recording" if cindra_path.exists() else None

    if multi_recording_base is None or not multi_recording_base.exists():
        return {
            "success": True,
            "recording_path": str(recording),
            "status": "not_started",
            "message": "No multi_recording output directory found",
        }

    datasets = [d for d in multi_recording_base.iterdir() if d.is_dir()]

    if not datasets:
        return {
            "success": True,
            "recording_path": str(recording),
            "status": "not_started",
            "message": "No dataset folders found in multi_recording directory",
        }

    dataset_statuses = {}
    for dataset in datasets:
        dataset_status: dict[str, Any] = {
            "runtime_exists": (dataset / "multi_recording_runtime_data.yaml").exists(),
            "config_exists": (dataset / "multi_recording_cindra_configuration.yaml").exists(),
            "tracker_exists": (dataset / "multi_recording_tracker.json").exists(),
            "template_masks_exists": (dataset / "template_roi_masks.npy").exists(),
            "F_exists": (dataset / "F.npy").exists(),
            "Fneu_exists": (dataset / "Fneu.npy").exists(),
            "spks_exists": (dataset / "spks.npy").exists(),
        }

        if dataset_status["F_exists"]:
            dataset_status["status"] = "completed"
        elif dataset_status["template_masks_exists"]:
            dataset_status["status"] = "discovery_completed"
        elif dataset_status["runtime_exists"]:
            dataset_status["status"] = "initialized"
        else:
            dataset_status["status"] = "unknown"

        dataset_status["is_main_recording"] = dataset_status["tracker_exists"]
        dataset_statuses[dataset.name] = dataset_status

    return {
        "success": True,
        "recording_path": str(recording),
        "multi_recording_path": str(multi_recording_base),
        "datasets": dataset_statuses,
    }


@mcp.tool()
def discover_single_recording_recordings_tool(root_directory: str) -> dict[str, Any]:
    """Discovers recordings containing raw neural imaging data that can be processed by the single-recording pipeline.

    Searches recursively for cindra_parameters.json files (created by sl-experiment), which mark
    directories containing raw recording data suitable for single-recording processing. Returns the
    parent directory of each match as a recording
    candidate path.

    Args:
        root_directory: The absolute path to the root directory to search.
    """
    root_path = Path(root_directory)

    if not root_path.exists():
        return {"error": f"Directory does not exist: {root_directory}"}

    if not root_path.is_dir():
        return {"error": f"Path is not a directory: {root_directory}"}

    recording_paths: list[str] = []
    errors: list[str] = []

    try:
        for marker_file in root_path.rglob("cindra_parameters.json"):
            try:
                recording_paths.append(str(marker_file.parent))
            except Exception as error:
                errors.append(f"{marker_file.parent}: {error}")
    except PermissionError as error:
        errors.append(f"Access denied during search: {error}")

    # Sorts paths for consistent output.
    recording_paths.sort()

    result: dict[str, Any] = {"recordings": recording_paths, "count": len(recording_paths)}

    if errors:
        result["errors"] = errors

    return result


@mcp.tool()
def discover_multi_recording_candidates_tool(root_directory: str) -> dict[str, Any]:
    """Discovers recordings with completed single-recording processing that are candidates for
    multi-recording ROI tracking.

    Searches recursively for combined_metadata.npz files, which mark completed single-recording cindra
    outputs. Returns the
    grandparent directory paths (recording root directories containing cindra output).

    Args:
        root_directory: The absolute path to the root directory to search.
    """
    root_path = Path(root_directory)

    if not root_path.exists():
        return {"error": f"Directory does not exist: {root_directory}"}

    if not root_path.is_dir():
        return {"error": f"Path is not a directory: {root_directory}"}

    recording_paths: list[str] = []
    errors: list[str] = []

    try:
        for marker_file in root_path.rglob("combined_metadata.npz"):
            try:
                # The combined_metadata.npz lives in cindra/combined/; grandparent is the cindra output root,
                # and its parent is the recording directory.
                recording_root = str(marker_file.parent.parent.parent)
                if recording_root not in recording_paths:
                    recording_paths.append(recording_root)
            except Exception as error:
                errors.append(f"{marker_file}: {error}")
    except PermissionError as error:
        errors.append(f"Access denied during search: {error}")

    # Sorts paths for consistent output.
    recording_paths.sort()

    result: dict[str, Any] = {"recordings": recording_paths, "count": len(recording_paths)}

    if errors:
        result["errors"] = errors

    return result


@mcp.tool()
def start_batch_processing_tool(
    recording_paths: list[str],
    configuration_path: str,
    *,
    recording_output_paths: list[str] | None = None,
    workers_per_plane: int = -1,
    max_parallel_planes: int = -1,
    progress_bars: bool = False,
) -> dict[str, Any]:
    """Starts batch single-recording processing for multiple recordings.

    Manages a three-phase batch: binarize (sequential), process (parallel), combine (sequential). Use
    get_batch_processing_status_tool to monitor progress.

    Args:
        recording_paths: List of absolute paths to recording data directories (used as file_io.data_path per recording).
        configuration_path: The absolute path to the template configuration YAML file.
        recording_output_paths: Optional list of absolute paths for per-recording output directories (used as
            file_io.output_path). Must match the length of recording_paths when provided. When not provided, each
            recording's output_path defaults to its data_path.
        workers_per_plane: CPU cores per plane job (-1 for automatic, max 30).
        max_parallel_planes: Max concurrent plane jobs (-1 for automatic).
        progress_bars: Determines whether to display progress bars during processing.
    """
    global _single_recording_batch_state

    if not recording_paths:
        return {"error": "At least one recording path is required"}

    if recording_output_paths is not None and len(recording_output_paths) != len(recording_paths):
        return {
            "error": f"recording_output_paths length ({len(recording_output_paths)}) must match "
            f"recording_paths length ({len(recording_paths)})."
        }

    template_path = Path(configuration_path)
    if not template_path.exists():
        return {"error": f"Configuration file not found: {configuration_path}"}

    if template_path.suffix != ".yaml":
        return {"error": f"Configuration file must be a .yaml file: {configuration_path}"}

    # Validates recording paths.
    valid_indices: list[int] = []
    valid_paths: list[Path] = []
    invalid_paths: list[str] = []

    for index, recording_path in enumerate(recording_paths):
        path = Path(recording_path)
        if path.exists() and path.is_dir():
            valid_paths.append(path)
            valid_indices.append(index)
        else:
            invalid_paths.append(recording_path)

    if not valid_paths:
        return {"error": "No valid recording paths provided", "invalid_paths": invalid_paths}

    # Resolves per-recording output paths. Defaults to data_path when recording_output_paths is not provided.
    resolved_output_paths: list[Path] = []
    for index, data_path in zip(valid_indices, valid_paths, strict=True):
        if recording_output_paths is not None:
            resolved_output_paths.append(Path(recording_output_paths[index]))
        else:
            resolved_output_paths.append(data_path)

    # Checks if batch processing is already active.
    if _single_recording_batch_state is not None:
        with _single_recording_batch_state.lock:
            active_count = (
                len(_single_recording_batch_state.binarize_active)
                + len(_single_recording_batch_state.process_active)
                + len(_single_recording_batch_state.combine_active)
            )
            queue_count = (
                len(_single_recording_batch_state.binarize_queue)
                + len(_single_recording_batch_state.process_queue)
                + len(_single_recording_batch_state.combine_queue)
            )
            if active_count > 0 or queue_count > 0:
                return {
                    "error": "Batch processing already in progress. Wait for current batch to complete.",
                    "active_count": active_count,
                    "queued_count": queue_count,
                }

    # Calculates resource allocation.
    actual_workers = min(
        resolve_worker_count(requested_workers=workers_per_plane, reserved_cores=_RESERVED_CORES), _MAXIMUM_JOB_CORES
    )
    actual_max_parallel = (
        max_parallel_planes
        if max_parallel_planes > 0
        else resolve_parallel_job_capacity(workers_per_job=actual_workers)
    )

    # Creates per-recording configuration copies with recording-specific paths and runtime settings.
    # Each recording gets its own configuration file so that concurrent workers do not interfere with
    # one another. The configuration file is
    # written to the output directory, which is guaranteed to be writable (the pipeline writes output there).
    recording_configuration_paths: dict[str, Path] = {}
    for data_path, output_path in zip(valid_paths, resolved_output_paths, strict=True):
        recording_key = _get_recording_key(data_path)
        recording_configuration = SingleRecordingConfiguration.from_yaml(file_path=template_path)
        recording_configuration.file_io.data_path = data_path
        recording_configuration.file_io.output_path = output_path
        recording_configuration.runtime.parallel_workers = actual_workers
        recording_configuration.runtime.display_progress_bars = progress_bars
        output_path.mkdir(parents=True, exist_ok=True)
        recording_configuration_path = output_path / "_batch_config.yaml"
        recording_configuration.save(file_path=recording_configuration_path)
        recording_configuration_paths[recording_key] = recording_configuration_path

    # Initializes batch state.
    _single_recording_batch_state = _SingleRecordingBatchState(
        recordings=list(valid_paths),
        configuration_path=template_path,
        recording_configuration_paths=recording_configuration_paths,
        current_phase="binarize",
        binarize_queue=list(valid_paths),
        workers_per_plane=actual_workers,
        max_parallel_planes=actual_max_parallel,
        lock=Lock(),
    )

    # Starts the batch manager thread.
    manager = Thread(target=_single_recording_batch_manager, daemon=True)
    manager.start()
    _single_recording_batch_state.manager_thread = manager

    result: dict[str, Any] = {
        "started": True,
        "total_recordings": len(valid_paths),
        "workers_per_plane": actual_workers,
        "max_parallel_planes": actual_max_parallel,
        "message": "Batch processing started. Use get_batch_processing_status_tool to monitor progress.",
    }

    if invalid_paths:
        result["invalid_paths"] = invalid_paths

    return result


@mcp.tool()
def get_batch_processing_status_tool() -> dict[str, Any]:
    """Returns the current status of single-recording batch processing.

    Returns status for all recordings including phase progress (binarize, process, combine).
    """
    if _single_recording_batch_state is None:
        return {
            "current_phase": "none",
            "recordings": [],
            "summary": {
                "total": 0,
                "binarize_completed": 0,
                "process_completed": 0,
                "combine_completed": 0,
                "failed": 0,
            },
        }

    with _single_recording_batch_state.lock:
        recordings_status: list[dict[str, Any]] = []

        for recording_path in _single_recording_batch_state.recordings:
            recording_key = _get_recording_key(recording_path)
            recording_name = recording_path.name

            # Determines binarize status.
            if recording_key in _single_recording_batch_state.binarize_completed:
                binarize_status = "done"
            elif recording_key in _single_recording_batch_state.binarize_failed:
                binarize_status = "failed"
            elif recording_key in _single_recording_batch_state.binarize_active:
                binarize_status = "running"
            elif recording_path in _single_recording_batch_state.binarize_queue:
                binarize_status = "pending"
            else:
                binarize_status = "pending"

            # Determines process status.
            plane_count = _single_recording_batch_state.plane_counts.get(recording_key, 0)
            if plane_count > 0:
                completed_planes = sum(
                    1
                    for p in range(plane_count)
                    if _get_plane_key(recording_path, p) in _single_recording_batch_state.process_completed
                )
                failed_planes = sum(
                    1
                    for p in range(plane_count)
                    if _get_plane_key(recording_path, p) in _single_recording_batch_state.process_failed
                )
                running_planes = sum(
                    1
                    for p in range(plane_count)
                    if _get_plane_key(recording_path, p) in _single_recording_batch_state.process_active
                )

                if failed_planes > 0:
                    process_status = f"{completed_planes}/{plane_count} (failed: {failed_planes})"
                elif running_planes > 0:
                    process_status = f"{completed_planes}/{plane_count} (running: {running_planes})"
                else:
                    process_status = f"{completed_planes}/{plane_count}"
            else:
                process_status = "0/0"

            # Determines combine status.
            if recording_key in _single_recording_batch_state.combine_completed:
                combine_status = "done"
            elif recording_key in _single_recording_batch_state.combine_failed:
                combine_status = "failed"
            elif recording_key in _single_recording_batch_state.combine_active:
                combine_status = "running"
            elif recording_path in _single_recording_batch_state.combine_queue:
                combine_status = "pending"
            else:
                combine_status = "pending"

            # Determines overall status.
            if recording_key in _single_recording_batch_state.combine_completed:
                overall_status = "SUCCEEDED"
            elif (
                recording_key in _single_recording_batch_state.binarize_failed
                or recording_key in _single_recording_batch_state.combine_failed
                or any(
                    _get_plane_key(recording_path, p) in _single_recording_batch_state.process_failed
                    for p in range(plane_count)
                )
            ):
                overall_status = "FAILED"
            elif (
                recording_key in _single_recording_batch_state.binarize_active
                or recording_key in _single_recording_batch_state.combine_active
                or any(
                    _get_plane_key(recording_path, p) in _single_recording_batch_state.process_active
                    for p in range(plane_count)
                )
            ):
                overall_status = "PROCESSING"
            else:
                overall_status = "QUEUED"

            recording_status: dict[str, Any] = {
                "recording_name": recording_name,
                "status": overall_status,
                "binarize": binarize_status,
                "process": process_status,
                "combine": combine_status,
            }

            if recording_key in _single_recording_batch_state.errors:
                recording_status["errors"] = _single_recording_batch_state.errors[recording_key]

            recordings_status.append(recording_status)

        # Computes summary.
        total_failed = len(_single_recording_batch_state.binarize_failed) + len(
            _single_recording_batch_state.combine_failed
        )
        for recording_key in _single_recording_batch_state.binarize_completed:
            recording_path = Path(recording_key)
            plane_count = _single_recording_batch_state.plane_counts.get(recording_key, 0)
            if any(
                _get_plane_key(recording_path, p) in _single_recording_batch_state.process_failed
                for p in range(plane_count)
            ):
                total_failed += 1

        summary = {
            "total": len(_single_recording_batch_state.recordings),
            "binarize_completed": len(_single_recording_batch_state.binarize_completed),
            "process_completed": len(
                {_get_recording_key(Path(key.split("|")[0])) for key in _single_recording_batch_state.process_completed}
            ),
            "combine_completed": len(_single_recording_batch_state.combine_completed),
            "failed": total_failed,
        }

        return {
            "current_phase": _single_recording_batch_state.current_phase,
            "recordings": recordings_status,
            "summary": summary,
        }


@mcp.tool()
def cancel_batch_processing_tool() -> dict[str, Any]:
    """Cancels any running single-recording batch processing.

    Clears all queues and resets the batch state. Active jobs will complete but no new jobs will start.
    """
    global _single_recording_batch_state

    if _single_recording_batch_state is None:
        return {"cancelled": False, "message": "No single-recording batch processing is active."}

    with _single_recording_batch_state.lock:
        active_count = (
            len(_single_recording_batch_state.binarize_active)
            + len(_single_recording_batch_state.process_active)
            + len(_single_recording_batch_state.combine_active)
        )

        # Clears all queues to prevent new jobs from starting.
        _single_recording_batch_state.binarize_queue.clear()
        _single_recording_batch_state.process_queue.clear()
        _single_recording_batch_state.combine_queue.clear()

        # Records final state before reset.
        final_state = {
            "binarize_completed": len(_single_recording_batch_state.binarize_completed),
            "process_completed": len(_single_recording_batch_state.process_completed),
            "combine_completed": len(_single_recording_batch_state.combine_completed),
            "active_jobs_at_cancel": active_count,
        }

    # Resets batch state after releasing lock.
    _single_recording_batch_state = None

    return {
        "cancelled": True,
        "message": "Single-recording batch processing cancelled. Active jobs will complete but no new jobs will start.",
        "final_state": final_state,
    }


@mcp.tool()
def start_multi_recording_batch_processing_tool(
    animal_configurations: list[dict[str, Any]],
    *,
    workers_per_discover: int = 20,
    workers_per_extract: int = -1,
    progress_bars: bool = False,
) -> dict[str, Any]:
    """Starts batch multi-recording processing for multiple animals.

    Manages a two-phase batch: discover (parallel by animal), extract (parallel by recording). Use
    get_multi_recording_batch_processing_status_tool to monitor progress.

    Args:
        animal_configurations: List of animal configurations, each with 'configuration_path' and 'recording_paths'.
        workers_per_discover: Workers for discover phase (default 20).
        workers_per_extract: Workers for extract phase (-1 for automatic, max 30).
        progress_bars: Determines whether to display progress bars during processing.
    """
    global _multi_recording_batch_state

    if not animal_configurations:
        return {"error": "At least one animal configuration is required"}

    # Validates animal configurations.
    valid_animals: list[tuple[Path, list[Path]]] = []
    invalid_configurations: list[str] = []
    animal_keys: list[str] = []

    for animal_configuration in animal_configurations:
        if "configuration_path" not in animal_configuration or "recording_paths" not in animal_configuration:
            invalid_configurations.append(f"Missing required keys: {animal_configuration}")
            continue

        configuration_path = Path(animal_configuration["configuration_path"])
        if not configuration_path.exists():
            invalid_configurations.append(f"Configuration not found: {configuration_path}")
            continue

        recording_paths = [Path(p) for p in animal_configuration["recording_paths"]]
        if len(recording_paths) < _MINIMUM_RECORDING_COUNT:
            invalid_configurations.append(f"Need at least 2 recordings: {configuration_path}")
            continue

        invalid_recordings = [str(p) for p in recording_paths if not p.exists() or not p.is_dir()]
        if invalid_recordings:
            invalid_configurations.append(f"Invalid recordings for {configuration_path}: {invalid_recordings}")
            continue

        # Extracts animal key from the configuration file.
        try:
            configuration = MultiRecordingConfiguration.from_yaml(file_path=configuration_path)
            animal_keys.append(configuration.recording_io.dataset_name)
        except Exception as error:
            invalid_configurations.append(f"Unable to load configuration {configuration_path}: {error}")
            continue

        valid_animals.append((configuration_path, recording_paths))

    if not valid_animals:
        return {"error": "No valid animal configurations provided", "invalid_configurations": invalid_configurations}

    # Checks if batch processing is already active.
    if _multi_recording_batch_state is not None:
        with _multi_recording_batch_state.lock:
            active_count = len(_multi_recording_batch_state.discover_active) + len(
                _multi_recording_batch_state.extract_active
            )
            queue_count = len(_multi_recording_batch_state.discover_queue) + len(
                _multi_recording_batch_state.extract_queue
            )
            if active_count > 0 or queue_count > 0:
                return {
                    "error": "Multi-recording batch processing already in progress.",
                    "active_count": active_count,
                    "queued_count": queue_count,
                }

    # Calculates resource allocation.
    actual_workers_discover = min(
        resolve_worker_count(requested_workers=workers_per_discover, reserved_cores=_RESERVED_CORES),
        _MAXIMUM_JOB_CORES,
    )
    actual_workers_extract = min(
        resolve_worker_count(requested_workers=workers_per_extract, reserved_cores=_RESERVED_CORES), _MAXIMUM_JOB_CORES
    )
    max_parallel_discovers = resolve_parallel_job_capacity(workers_per_job=actual_workers_discover)
    max_parallel_extracts = resolve_parallel_job_capacity(workers_per_job=actual_workers_extract)

    # Initializes batch state.
    _multi_recording_batch_state = _MultiRecordingBatchState(
        animals=valid_animals,
        current_phase="discover",
        discover_queue=list(animal_keys),
        workers_per_discover=actual_workers_discover,
        max_parallel_discovers=max_parallel_discovers,
        workers_per_extract=actual_workers_extract,
        max_parallel_extracts=max_parallel_extracts,
        progress_bars=progress_bars,
        lock=Lock(),
    )

    # Starts the batch manager thread.
    manager = Thread(target=_multi_recording_batch_manager, daemon=True)
    manager.start()
    _multi_recording_batch_state.manager_thread = manager

    total_recordings = sum(len(recordings) for _, recordings in valid_animals)

    result: dict[str, Any] = {
        "started": True,
        "total_animals": len(valid_animals),
        "total_recordings": total_recordings,
        "workers_per_discover": actual_workers_discover,
        "workers_per_extract": actual_workers_extract,
        "message": (
            "Multi-recording batch processing started. Use get_multi_recording_batch_processing_status_tool to monitor."
        ),
    }

    if invalid_configurations:
        result["invalid_configurations"] = invalid_configurations

    return result


@mcp.tool()
def get_multi_recording_batch_processing_status_tool() -> dict[str, Any]:
    """Returns the current status of multi-recording batch processing.

    Returns status for all animals including phase progress (discover, extract).
    """
    if _multi_recording_batch_state is None:
        return {
            "current_phase": "none",
            "animals": [],
            "summary": {
                "total_animals": 0,
                "discover_completed": 0,
                "extract_completed": 0,
                "extract_total": 0,
                "failed": 0,
            },
        }

    with _multi_recording_batch_state.lock:
        animals_status: list[dict[str, Any]] = []

        # Builds an ordered list of animal keys from the configuration files.
        animal_keys_ordered: list[str] = []
        for configuration_path, _ in _multi_recording_batch_state.animals:
            try:
                configuration = MultiRecordingConfiguration.from_yaml(file_path=configuration_path)
                animal_keys_ordered.append(configuration.recording_io.dataset_name)
            except Exception:  # noqa: S112
                continue

        for animal_key in animal_keys_ordered:
            # Determines discover status.
            if animal_key in _multi_recording_batch_state.discover_completed:
                discover_status = "done"
            elif animal_key in _multi_recording_batch_state.discover_failed:
                discover_status = "failed"
            elif animal_key in _multi_recording_batch_state.discover_active:
                discover_status = "running"
            elif animal_key in _multi_recording_batch_state.discover_queue:
                discover_status = "pending"
            else:
                discover_status = "pending"

            # Determines extract progress.
            recording_ids = _multi_recording_batch_state.recording_ids.get(animal_key, [])
            extract_total = len(recording_ids)
            extract_completed = sum(
                1 for sid in recording_ids if f"{animal_key}|{sid}" in _multi_recording_batch_state.extract_completed
            )
            extract_failed = sum(
                1 for sid in recording_ids if f"{animal_key}|{sid}" in _multi_recording_batch_state.extract_failed
            )

            # Determines overall status.
            if animal_key in _multi_recording_batch_state.discover_failed:
                overall_status = "FAILED"
            elif extract_failed > 0:
                overall_status = "PARTIAL"
            elif extract_completed == extract_total and extract_total > 0:
                overall_status = "SUCCEEDED"
            elif animal_key in _multi_recording_batch_state.discover_active or any(
                f"{animal_key}|{sid}" in _multi_recording_batch_state.extract_active for sid in recording_ids
            ):
                overall_status = "PROCESSING"
            else:
                overall_status = "QUEUED"

            animal_status: dict[str, Any] = {
                "animal_key": animal_key,
                "status": overall_status,
                "discover": discover_status,
                "extract_completed": extract_completed,
                "extract_total": extract_total,
            }

            if animal_key in _multi_recording_batch_state.errors:
                animal_status["errors"] = _multi_recording_batch_state.errors[animal_key]

            animals_status.append(animal_status)

        # Computes summary.
        total_extract_completed = len(_multi_recording_batch_state.extract_completed)
        total_extract_total = sum(len(ids) for ids in _multi_recording_batch_state.recording_ids.values())
        total_failed = len(_multi_recording_batch_state.discover_failed) + len(
            _multi_recording_batch_state.extract_failed
        )

        summary = {
            "total_animals": len(_multi_recording_batch_state.animals),
            "discover_completed": len(_multi_recording_batch_state.discover_completed),
            "extract_completed": total_extract_completed,
            "extract_total": total_extract_total,
            "failed": total_failed,
        }

        return {
            "current_phase": _multi_recording_batch_state.current_phase,
            "animals": animals_status,
            "summary": summary,
        }


@mcp.tool()
def cancel_multi_recording_batch_processing_tool() -> dict[str, Any]:
    """Cancels any running multi-recording batch processing.

    Clears all queues and resets the batch state. Active jobs will complete but no new jobs will start.
    """
    global _multi_recording_batch_state

    if _multi_recording_batch_state is None:
        return {"cancelled": False, "message": "No multi-recording batch processing is active."}

    with _multi_recording_batch_state.lock:
        active_count = len(_multi_recording_batch_state.discover_active) + len(
            _multi_recording_batch_state.extract_active
        )

        # Clears all queues to prevent new jobs from starting.
        _multi_recording_batch_state.discover_queue.clear()
        _multi_recording_batch_state.extract_queue.clear()

        # Records final state before reset.
        final_state = {
            "discover_completed": len(_multi_recording_batch_state.discover_completed),
            "extract_completed": len(_multi_recording_batch_state.extract_completed),
            "active_jobs_at_cancel": active_count,
        }

    # Resets batch state after releasing lock.
    _multi_recording_batch_state = None

    return {
        "cancelled": True,
        "message": "Multi-recording batch processing cancelled. Active jobs will complete but no new jobs will start.",
        "final_state": final_state,
    }


def _get_recording_key(recording_path: Path) -> str:
    """Generates a unique key for a recording path.

    Args:
        recording_path: The path to the recording directory.

    Returns:
        A string key for the recording.
    """
    return str(recording_path)


def _get_plane_key(recording_path: Path, plane_index: int) -> str:
    """Generates a unique key for a recording-plane combination.

    Args:
        recording_path: The path to the recording directory.
        plane_index: The plane index.

    Returns:
        A string key for the recording-plane combination.
    """
    return f"{recording_path}|plane_{plane_index}"


def _run_binarize_job(configuration_path: Path) -> tuple[bool, int, str | None]:
    """Runs the binarize phase for a single recording.

    Args:
        configuration_path: The path to the configuration file.

    Returns:
        A tuple containing success status, plane count (0 if failed), and error message if failed.
    """
    try:
        run_single_recording_pipeline(
            configuration_path=configuration_path,
            binarize=True,
            process=False,
            combine=False,
        )

        # Loads the configuration to find the output path, then counts planes via RuntimeContext.
        configuration = SingleRecordingConfiguration.from_yaml(file_path=configuration_path)
        effective_output_path = (
            configuration.file_io.output_path
            if configuration.file_io.output_path is not None
            else configuration.file_io.data_path
        )
        if effective_output_path is None:
            return False, 0, "Configuration error: neither file_io.output_path nor file_io.data_path is set."
        root_path = effective_output_path / "cindra"
        contexts = RuntimeContext.load(root_path=root_path, plane_index=-1)
        if not isinstance(contexts, list):
            contexts = [contexts]
        plane_count = len(contexts)

    except Exception as error:
        frames = traceback.extract_tb(error.__traceback__)
        location = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "unknown"
        return False, 0, f"{type(error).__name__}: {error} ({location})"

    else:
        return True, plane_count, None


def _run_process_job(configuration_path: Path, plane_index: int) -> tuple[bool, str | None]:
    """Runs the process phase for a single plane.

    Args:
        configuration_path: The path to the configuration file.
        plane_index: The plane index to process.

    Returns:
        A tuple containing success status and error message if failed.
    """
    try:
        run_single_recording_pipeline(
            configuration_path=configuration_path,
            binarize=False,
            process=True,
            combine=False,
            target_plane=plane_index,
        )

    except Exception as error:
        frames = traceback.extract_tb(error.__traceback__)
        location = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "unknown"
        return False, f"{type(error).__name__}: {error} ({location})"

    else:
        return True, None


def _run_combine_job(configuration_path: Path) -> tuple[bool, str | None]:
    """Runs the combine phase for a single recording.

    Args:
        configuration_path: The path to the configuration file.

    Returns:
        A tuple containing success status and error message if failed.
    """
    try:
        run_single_recording_pipeline(
            configuration_path=configuration_path,
            binarize=False,
            process=False,
            combine=True,
        )

    except Exception as error:
        frames = traceback.extract_tb(error.__traceback__)
        location = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "unknown"
        return False, f"{type(error).__name__}: {error} ({location})"

    else:
        return True, None


def _binarize_worker(recording_path: Path, configuration_path: Path) -> None:
    """Runs binarization for one recording and updates batch state.

    Args:
        recording_path: The path to the recording directory.
        configuration_path: The path to the configuration file.
    """
    recording_key = _get_recording_key(recording_path)
    success: bool = False
    plane_count: int = 0
    error: str | None = None

    try:
        success, plane_count, error = _run_binarize_job(configuration_path=configuration_path)
    except Exception as exception:
        frames = traceback.extract_tb(exception.__traceback__)
        location = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "unknown"
        success, plane_count, error = (False, 0, f"Worker crash: {type(exception).__name__}: {exception} ({location})")
    finally:
        if _single_recording_batch_state is not None:
            with _single_recording_batch_state.lock:
                _single_recording_batch_state.binarize_active.pop(recording_key, None)
                if success:
                    _single_recording_batch_state.binarize_completed.add(recording_key)
                    _single_recording_batch_state.plane_counts[recording_key] = plane_count
                else:
                    _single_recording_batch_state.binarize_failed.add(recording_key)
                    if error:
                        _single_recording_batch_state.errors.setdefault(recording_key, []).append(f"binarize: {error}")


def _process_worker(recording_path: Path, configuration_path: Path, plane_index: int) -> None:
    """Runs processing for one plane and updates batch state.

    Args:
        recording_path: The path to the recording directory.
        configuration_path: The path to the recording's configuration file.
        plane_index: The plane index to process.
    """
    recording_key = _get_recording_key(recording_path)
    plane_key = _get_plane_key(recording_path, plane_index)
    success: bool = False
    error: str | None = None

    try:
        success, error = _run_process_job(configuration_path=configuration_path, plane_index=plane_index)
    except Exception as exception:
        frames = traceback.extract_tb(exception.__traceback__)
        location = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "unknown"
        success, error = False, f"Worker crash: {type(exception).__name__}: {exception} ({location})"
    finally:
        if _single_recording_batch_state is not None:
            with _single_recording_batch_state.lock:
                _single_recording_batch_state.process_active.pop(plane_key, None)
                if success:
                    _single_recording_batch_state.process_completed.add(plane_key)
                else:
                    _single_recording_batch_state.process_failed.add(plane_key)
                    if error:
                        _single_recording_batch_state.errors.setdefault(recording_key, []).append(
                            f"process_plane_{plane_index}: {error}"
                        )


def _combine_worker(recording_path: Path, configuration_path: Path) -> None:
    """Runs combination for one recording and updates batch state.

    Args:
        recording_path: The path to the recording directory.
        configuration_path: The path to the configuration file.
    """
    recording_key = _get_recording_key(recording_path)
    success: bool = False
    error: str | None = None

    try:
        success, error = _run_combine_job(configuration_path=configuration_path)
    except Exception as exception:
        frames = traceback.extract_tb(exception.__traceback__)
        location = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "unknown"
        success, error = False, f"Worker crash: {type(exception).__name__}: {exception} ({location})"
    finally:
        if _single_recording_batch_state is not None:
            with _single_recording_batch_state.lock:
                _single_recording_batch_state.combine_active.pop(recording_key, None)
                if success:
                    _single_recording_batch_state.combine_completed.add(recording_key)
                else:
                    _single_recording_batch_state.combine_failed.add(recording_key)
                    if error:
                        _single_recording_batch_state.errors.setdefault(recording_key, []).append(f"combine: {error}")


def _single_recording_batch_manager() -> None:
    """Orchestrates three-phase single-recording batch processing: binarize, process, combine."""
    timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

    if _single_recording_batch_state is None:
        return

    while True:
        with _single_recording_batch_state.lock:
            # Phase 1: BINARIZE.
            if _single_recording_batch_state.current_phase == "binarize":
                # Starts new binarize jobs (sequential - I/O bound).
                if not _single_recording_batch_state.binarize_active and _single_recording_batch_state.binarize_queue:
                    next_recording = _single_recording_batch_state.binarize_queue.pop(0)
                    recording_key = _get_recording_key(next_recording)
                    recording_configuration = _single_recording_batch_state.recording_configuration_paths[recording_key]

                    thread = Thread(
                        target=_binarize_worker,
                        kwargs={"recording_path": next_recording, "configuration_path": recording_configuration},
                        daemon=True,
                    )
                    thread.start()
                    _single_recording_batch_state.binarize_active[recording_key] = thread

                # Checks if binarize phase is complete.
                if (
                    not _single_recording_batch_state.binarize_active
                    and not _single_recording_batch_state.binarize_queue
                ):
                    # Builds process queue from completed binarizations (naturally sorted for deterministic order).
                    for recording_key in natsorted(_single_recording_batch_state.binarize_completed):
                        plane_count = _single_recording_batch_state.plane_counts.get(recording_key, 0)
                        for plane in range(plane_count):
                            _single_recording_batch_state.process_queue.append((recording_key, plane))

                    _single_recording_batch_state.current_phase = "process"

            # Phase 2: PROCESS.
            elif _single_recording_batch_state.current_phase == "process":
                # Starts new process jobs (parallel - CPU bound).
                while (
                    len(_single_recording_batch_state.process_active)
                    < _single_recording_batch_state.max_parallel_planes
                    and _single_recording_batch_state.process_queue
                ):
                    recording_key, plane_index = _single_recording_batch_state.process_queue.pop(0)
                    recording_path = Path(recording_key)
                    plane_key = _get_plane_key(recording_path, plane_index)
                    recording_configuration = _single_recording_batch_state.recording_configuration_paths[recording_key]

                    thread = Thread(
                        target=_process_worker,
                        kwargs={
                            "recording_path": recording_path,
                            "configuration_path": recording_configuration,
                            "plane_index": plane_index,
                        },
                        daemon=True,
                    )
                    thread.start()
                    _single_recording_batch_state.process_active[plane_key] = thread

                # Checks if process phase is complete.
                if not _single_recording_batch_state.process_active and not _single_recording_batch_state.process_queue:
                    # Builds combine queue from recordings with all planes processed (naturally sorted).
                    for recording_key in natsorted(_single_recording_batch_state.binarize_completed):
                        plane_count = _single_recording_batch_state.plane_counts.get(recording_key, 0)
                        all_planes_done = all(
                            _get_plane_key(Path(recording_key), p) in _single_recording_batch_state.process_completed
                            for p in range(plane_count)
                        )
                        any_plane_failed = any(
                            _get_plane_key(Path(recording_key), p) in _single_recording_batch_state.process_failed
                            for p in range(plane_count)
                        )

                        if all_planes_done and not any_plane_failed:
                            _single_recording_batch_state.combine_queue.append(Path(recording_key))

                    _single_recording_batch_state.current_phase = "combine"

            # Phase 3: COMBINE.
            elif _single_recording_batch_state.current_phase == "combine":
                # Starts new combine jobs (sequential - I/O bound).
                if not _single_recording_batch_state.combine_active and _single_recording_batch_state.combine_queue:
                    next_recording = _single_recording_batch_state.combine_queue.pop(0)
                    recording_key = _get_recording_key(next_recording)
                    recording_configuration = _single_recording_batch_state.recording_configuration_paths[recording_key]

                    thread = Thread(
                        target=_combine_worker,
                        kwargs={"recording_path": next_recording, "configuration_path": recording_configuration},
                        daemon=True,
                    )
                    thread.start()
                    _single_recording_batch_state.combine_active[recording_key] = thread

                # Checks if all processing is complete.
                if not _single_recording_batch_state.combine_active and not _single_recording_batch_state.combine_queue:
                    break

        # Sleeps briefly before checking again.
        timer.delay(delay=1000, allow_sleep=True)


def _run_discover_job(
    configuration_path: Path, recording_paths: list[Path], workers: int, progress_bars: bool = False
) -> tuple[bool, list[str], str | None]:
    """Runs the discover phase for a single animal.

    Args:
        configuration_path: The path to the configuration file.
        recording_paths: The list of recording paths for this animal.
        workers: The number of workers to use.
        progress_bars: Determines whether to display progress bars during processing.

    Returns:
        A tuple containing success status, list of recording IDs, and error message if failed.
    """
    try:
        # Writes recording directories and runtime settings into the configuration file before running the pipeline.
        configuration = MultiRecordingConfiguration.from_yaml(file_path=configuration_path)
        configuration.recording_io.recording_directories = tuple(natsorted(recording_paths))
        configuration.runtime.parallel_workers = workers
        configuration.runtime.display_progress_bars = progress_bars
        configuration.save(file_path=configuration_path)

        run_multi_recording_pipeline(
            configuration_path=configuration_path,
            discover=True,
            extract=False,
        )

        # Reloads the configuration and resolves contexts to extract recording IDs.
        configuration = MultiRecordingConfiguration.from_yaml(file_path=configuration_path)
        contexts = resolve_multi_recording_contexts(configuration=configuration)
        recording_ids = [ctx.runtime.io.recording_id for ctx in contexts]

    except Exception as error:
        frames = traceback.extract_tb(error.__traceback__)
        location = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "unknown"
        return False, [], f"{type(error).__name__}: {error} ({location})"

    else:
        return True, recording_ids, None


def _run_extract_job(
    configuration_path: Path, recording_paths: list[Path], recording_id: str, workers: int, progress_bars: bool = False
) -> tuple[bool, str | None]:
    """Runs the extract phase for a single recording.

    Args:
        configuration_path: The path to the configuration file.
        recording_paths: The list of recording paths for this animal.
        recording_id: The recording ID to extract.
        workers: The number of workers to use.
        progress_bars: Determines whether to display progress bars during processing.

    Returns:
        A tuple containing success status and error message if failed.
    """
    try:
        # Writes recording directories and runtime settings into the configuration file before running the pipeline.
        configuration = MultiRecordingConfiguration.from_yaml(file_path=configuration_path)
        configuration.recording_io.recording_directories = tuple(natsorted(recording_paths))
        configuration.runtime.parallel_workers = workers
        configuration.runtime.display_progress_bars = progress_bars
        configuration.save(file_path=configuration_path)

        run_multi_recording_pipeline(
            configuration_path=configuration_path,
            discover=False,
            extract=True,
            target_recording=recording_id,
        )

    except Exception as error:
        frames = traceback.extract_tb(error.__traceback__)
        location = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "unknown"
        return False, f"{type(error).__name__}: {error} ({location})"

    else:
        return True, None


def _discover_worker(
    animal_key: str, configuration_path: Path, recording_paths: list[Path], workers: int, progress_bars: bool = False
) -> None:
    """Runs discovery for one animal and updates batch state.

    Args:
        animal_key: The unique key for this animal.
        configuration_path: The path to the configuration file.
        recording_paths: The list of recording paths for this animal.
        workers: The number of workers to use.
        progress_bars: Determines whether to display progress bars during processing.
    """
    success: bool = False
    recording_ids: list[str] = []
    error: str | None = None

    try:
        success, recording_ids, error = _run_discover_job(
            configuration_path=configuration_path,
            recording_paths=recording_paths,
            workers=workers,
            progress_bars=progress_bars,
        )
    except Exception as exception:
        frames = traceback.extract_tb(exception.__traceback__)
        location = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "unknown"
        success, recording_ids, error = (
            False,
            [],
            f"Worker crash: {type(exception).__name__}: {exception} ({location})",
        )
    finally:
        if _multi_recording_batch_state is not None:
            with _multi_recording_batch_state.lock:
                _multi_recording_batch_state.discover_active.pop(animal_key, None)
                if success:
                    _multi_recording_batch_state.discover_completed.add(animal_key)
                    _multi_recording_batch_state.recording_ids[animal_key] = recording_ids
                else:
                    _multi_recording_batch_state.discover_failed.add(animal_key)
                    if error:
                        _multi_recording_batch_state.errors.setdefault(animal_key, []).append(f"discover: {error}")


def _extract_worker(
    animal_key: str,
    configuration_path: Path,
    recording_paths: list[Path],
    recording_id: str,
    workers: int,
    progress_bars: bool = False,
) -> None:
    """Runs extraction for one recording and updates batch state.

    Args:
        animal_key: The unique key for this animal.
        configuration_path: The path to the configuration file.
        recording_paths: The list of recording paths for this animal.
        recording_id: The recording ID to extract.
        workers: The number of workers to use.
        progress_bars: Determines whether to display progress bars during processing.
    """
    extract_key = f"{animal_key}|{recording_id}"
    success: bool = False
    error: str | None = None

    try:
        success, error = _run_extract_job(
            configuration_path=configuration_path,
            recording_paths=recording_paths,
            recording_id=recording_id,
            workers=workers,
            progress_bars=progress_bars,
        )
    except Exception as exception:
        frames = traceback.extract_tb(exception.__traceback__)
        location = f"{frames[-1].filename}:{frames[-1].lineno}" if frames else "unknown"
        success, error = False, f"Worker crash: {type(exception).__name__}: {exception} ({location})"
    finally:
        if _multi_recording_batch_state is not None:
            with _multi_recording_batch_state.lock:
                _multi_recording_batch_state.extract_active.pop(extract_key, None)
                if success:
                    _multi_recording_batch_state.extract_completed.add(extract_key)
                else:
                    _multi_recording_batch_state.extract_failed.add(extract_key)
                    if error:
                        _multi_recording_batch_state.errors.setdefault(animal_key, []).append(
                            f"extract_{recording_id}: {error}"
                        )


def _multi_recording_batch_manager() -> None:
    """Orchestrates two-phase multi-recording batch processing: discover, extract."""
    timer = PrecisionTimer(precision=TimerPrecisions.MILLISECOND)

    if _multi_recording_batch_state is None:
        return

    # Builds animal key to configuration and recordings mapping.
    animal_configurations: dict[str, tuple[Path, list[Path]]] = {}
    for configuration_path, recording_paths in _multi_recording_batch_state.animals:
        configuration = MultiRecordingConfiguration.from_yaml(file_path=configuration_path)
        animal_key = configuration.recording_io.dataset_name
        animal_configurations[animal_key] = (configuration_path, recording_paths)

    while True:
        with _multi_recording_batch_state.lock:
            # Phase 1: DISCOVER.
            if _multi_recording_batch_state.current_phase == "discover":
                # Starts new discover jobs.
                while (
                    len(_multi_recording_batch_state.discover_active)
                    < _multi_recording_batch_state.max_parallel_discovers
                    and _multi_recording_batch_state.discover_queue
                ):
                    animal_key = _multi_recording_batch_state.discover_queue.pop(0)
                    configuration_path, recording_paths = animal_configurations[animal_key]

                    thread = Thread(
                        target=_discover_worker,
                        kwargs={
                            "animal_key": animal_key,
                            "configuration_path": configuration_path,
                            "recording_paths": recording_paths,
                            "workers": _multi_recording_batch_state.workers_per_discover,
                            "progress_bars": _multi_recording_batch_state.progress_bars,
                        },
                        daemon=True,
                    )
                    thread.start()
                    _multi_recording_batch_state.discover_active[animal_key] = thread

                # Checks if discover phase is complete.
                if not _multi_recording_batch_state.discover_active and not _multi_recording_batch_state.discover_queue:
                    # Builds extract queue from completed discoveries (naturally sorted for deterministic order).
                    for animal_key in natsorted(_multi_recording_batch_state.discover_completed):
                        for recording_id in natsorted(_multi_recording_batch_state.recording_ids.get(animal_key, [])):
                            _multi_recording_batch_state.extract_queue.append((animal_key, recording_id))

                    _multi_recording_batch_state.current_phase = "extract"

            # Phase 2: EXTRACT.
            elif _multi_recording_batch_state.current_phase == "extract":
                # Starts new extract jobs.
                while (
                    len(_multi_recording_batch_state.extract_active)
                    < _multi_recording_batch_state.max_parallel_extracts
                    and _multi_recording_batch_state.extract_queue
                ):
                    animal_key, recording_id = _multi_recording_batch_state.extract_queue.pop(0)
                    extract_key = f"{animal_key}|{recording_id}"
                    configuration_path, recording_paths = animal_configurations[animal_key]

                    thread = Thread(
                        target=_extract_worker,
                        kwargs={
                            "animal_key": animal_key,
                            "configuration_path": configuration_path,
                            "recording_paths": recording_paths,
                            "recording_id": recording_id,
                            "workers": _multi_recording_batch_state.workers_per_extract,
                            "progress_bars": _multi_recording_batch_state.progress_bars,
                        },
                        daemon=True,
                    )
                    thread.start()
                    _multi_recording_batch_state.extract_active[extract_key] = thread

                # Checks if all processing is complete.
                if not _multi_recording_batch_state.extract_active and not _multi_recording_batch_state.extract_queue:
                    break

        # Sleeps briefly before checking again.
        timer.delay(delay=1000, allow_sleep=True)
