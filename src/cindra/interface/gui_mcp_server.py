"""Provides the MCP server for launching GUI viewers and querying neural imaging data.

Exposes tools that enable AI agents to launch GUI viewers for the user to interact with and query processed pipeline
data directly. The server avoids importing Qt at module level so that ``cindra-gui mcp`` can start without loading
PySide6.
"""

from __future__ import annotations

import sys
import uuid
from typing import TYPE_CHECKING, Any, Literal
from pathlib import Path
import subprocess
from collections import OrderedDict
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP

if TYPE_CHECKING:
    from ..gui.viewer_context import ViewerData, SingleDayData

gui_mcp = FastMCP(name="cindra-gui-mcp", json_response=True)
"""The GUI MCP server instance initialized with JSON response mode for structured output."""

_DATA_CACHE_MAX_SIZE: int = 4
"""The maximum number of ViewerData instances to cache for data query tools."""

_MAX_TRACE_ROIS: int = 50
"""The maximum number of ROIs whose traces can be queried in a single request."""

_CELL_LABEL_THRESHOLD: float = 0.5
"""The threshold above which a classification label value is considered a cell."""


@dataclass
class _ViewerProcess:
    """Tracks a managed GUI viewer subprocess.

    Attributes:
        viewer_id: The unique identifier for this viewer instance.
        viewer_type: The type of viewer ('roi', 'tracking', or 'registration').
        recording_path: The path to the recording loaded in the viewer.
        dataset: The multi-day dataset name, or None for single-day mode.
        process: The subprocess.Popen instance for the viewer process.
    """

    viewer_id: str
    viewer_type: str
    recording_path: str
    dataset: str | None
    process: subprocess.Popen  # type: ignore[type-arg]


_viewer_registry: dict[str, _ViewerProcess] = {}
"""Tracks active viewer subprocesses keyed by viewer_id."""

_data_cache: OrderedDict[str, Any] = OrderedDict()
"""LRU cache for ViewerData instances used by data query tools, keyed by recording path string."""


def run_gui_server(transport: Literal["stdio", "sse", "streamable-http"] = "stdio") -> None:
    """Starts the GUI MCP server with the specified transport.

    Args:
        transport: The transport type to use ('stdio', 'sse', or 'streamable-http').
    """
    gui_mcp.run(transport=transport)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _get_viewer(viewer_id: str) -> _ViewerProcess | None:
    """Returns the viewer process for the given ID, cleaning up dead processes.

    Args:
        viewer_id: The viewer identifier to look up.

    Returns:
        The _ViewerProcess instance, or None if not found or the process has exited.
    """
    entry = _viewer_registry.get(viewer_id)
    if entry is None:
        return None

    if entry.process.poll() is not None:
        del _viewer_registry[viewer_id]
        return None

    return entry


def _get_or_load_viewer_data(recording_path: str, dataset: str | None = None) -> ViewerData:
    """Returns a cached ViewerData instance or loads one from disk.

    Uses an LRU cache with a maximum size to limit memory usage.

    Args:
        recording_path: The path to the recording's cindra output directory.
        dataset: Optional multi-day dataset name to load.

    Returns:
        A ViewerData instance loaded from the specified path.
    """
    from ..gui.viewer_context import ViewerData  # noqa: PLC0415

    cache_key = f"{recording_path}|{dataset or ''}"

    if cache_key in _data_cache:
        _data_cache.move_to_end(cache_key)
        return _data_cache[cache_key]

    path = Path(recording_path)
    if not path.exists():
        message = f"Unable to load data from '{recording_path}'. The path does not exist."
        raise FileNotFoundError(message)

    data = ViewerData.from_data(root_path=path, dataset=dataset)

    _data_cache[cache_key] = data
    while len(_data_cache) > _DATA_CACHE_MAX_SIZE:
        _data_cache.popitem(last=False)

    return data


def _get_or_load_single_day_data(recording_path: str) -> SingleDayData:
    """Returns a cached SingleDayData instance extracted from ViewerData, or loads one from disk.

    Args:
        recording_path: The path to the recording's cindra output directory.

    Returns:
        A SingleDayData instance loaded from the specified path.
    """
    viewer_data = _get_or_load_viewer_data(recording_path)
    return viewer_data.single_day


# ------------------------------------------------------------------
# Lifecycle tools (3)
# ------------------------------------------------------------------


@gui_mcp.tool()
def launch_viewer_tool(
    viewer_type: Literal["roi", "tracking", "registration"],
    recording_path: str,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Launches a GUI viewer in a subprocess for the user to interact with.

    Spawns the viewer as a child process using the cindra-gui CLI. The viewer window appears on screen for the user
    to interact with directly. Returns a viewer_id that can be used to check status or close the viewer later.

    Args:
        viewer_type: The type of viewer to launch. 'roi' for ROI inspection, 'tracking' for multi-day tracking
            quality, 'registration' for registration quality (binary player + PC viewer).
        recording_path: Absolute path to the cindra pipeline output directory for the recording to visualize.
        dataset: Multi-day dataset name to load on startup. Only used by 'roi' and 'tracking' viewers.
    """
    path = Path(recording_path)
    if not path.exists():
        return {"success": False, "error": f"Unable to launch viewer. Path does not exist: {recording_path}"}

    viewer_id = uuid.uuid4().hex[:12]

    cindra_gui_exe = str(Path(sys.executable).parent / "cindra-gui")
    cmd = [cindra_gui_exe, viewer_type, "--recording-path", str(path)]
    if dataset is not None and viewer_type in ("roi", "tracking"):
        cmd.extend(["--dataset", dataset])

    try:
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)  # noqa: S603
    except OSError as error:
        return {"success": False, "error": f"Unable to launch viewer subprocess. {error}"}

    entry = _ViewerProcess(
        viewer_id=viewer_id,
        viewer_type=viewer_type,
        recording_path=recording_path,
        dataset=dataset,
        process=process,
    )
    _viewer_registry[viewer_id] = entry

    return {
        "success": True,
        "viewer_id": viewer_id,
        "viewer_type": viewer_type,
        "recording_path": recording_path,
        "dataset": dataset,
    }


@gui_mcp.tool()
def list_viewers_tool() -> dict[str, Any]:
    """Lists all active GUI viewer instances managed by this server.

    Returns viewer IDs, types, recording paths, and alive status for each managed viewer. Dead viewers are
    automatically cleaned up.
    """
    viewers: list[dict[str, Any]] = []
    dead_ids: list[str] = []

    for viewer_id, entry in _viewer_registry.items():
        alive = entry.process.poll() is None
        if not alive:
            dead_ids.append(viewer_id)
        viewers.append(
            {
                "viewer_id": viewer_id,
                "viewer_type": entry.viewer_type,
                "recording_path": entry.recording_path,
                "dataset": entry.dataset,
                "alive": alive,
            }
        )

    for dead_id in dead_ids:
        del _viewer_registry[dead_id]

    return {"viewers": viewers, "count": len(viewers)}


@gui_mcp.tool()
def close_viewer_tool(viewer_id: str) -> dict[str, Any]:
    """Closes a GUI viewer and terminates its subprocess.

    Terminates the viewer process, waiting briefly for graceful shutdown before forcing termination.

    Args:
        viewer_id: The unique identifier of the viewer to close, as returned by launch_viewer_tool.
    """
    entry = _get_viewer(viewer_id)
    if entry is None:
        return {"success": False, "error": f"Unable to find viewer with id '{viewer_id}'."}

    entry.process.terminate()
    try:
        entry.process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        entry.process.kill()

    del _viewer_registry[viewer_id]
    return {"success": True, "viewer_id": viewer_id}


# ------------------------------------------------------------------
# Data query tools (5) — no GUI needed
# ------------------------------------------------------------------


@gui_mcp.tool()
def query_session_metadata_tool(recording_path: str) -> dict[str, Any]:
    """Queries metadata for a cindra-processed recording session.

    Returns frame count, sampling rate, plane count, ROI count, cell count, available multi-day datasets, and other
    session properties. Does not require a GUI viewer to be open.

    Args:
        recording_path: Absolute path to a cindra pipeline output directory.
    """
    try:
        data = _get_or_load_viewer_data(recording_path)
    except Exception as error:
        return {
            "success": False,
            "error": (
                f"Unable to load data from '{recording_path}'. Verify the path contains cindra pipeline output. {error}"
            ),
        }

    single_day = data.single_day
    return {
        "success": True,
        "recording_path": recording_path,
        "frame_count": single_day.frame_count,
        "sampling_rate": single_day.sampling_rate,
        "tau": single_day.tau,
        "plane_count": single_day.plane_count,
        "frame_height": single_day.frame_height,
        "frame_width": single_day.frame_width,
        "roi_count": single_day.roi_count,
        "cell_count": single_day.cell_count,
        "two_channels": single_day.two_channels,
        "recording_label": single_day.recording_label,
        "available_datasets": list(data.available_datasets),
    }


@gui_mcp.tool()
def query_roi_statistics_tool(
    recording_path: str,
    roi_indices: list[int] | None = None,
    sort_by: str | None = None,
    top_n: int | None = None,
) -> dict[str, Any]:
    """Queries per-ROI spatial statistics for a cindra-processed recording.

    Returns statistics including pixel count, skewness, compactness, footprint area, aspect ratio, and centroid
    coordinates for the requested ROIs. Does not require a GUI viewer.

    Args:
        recording_path: Absolute path to a cindra pipeline output directory.
        roi_indices: Specific ROI indices to query. Returns all ROIs when not provided.
        sort_by: Sort results by this statistic name ('skewness', 'compactness', 'footprint', 'aspect_ratio',
            'pixel_count'). Results are returned in descending order.
        top_n: Limit results to the top N ROIs after sorting. Only effective when sort_by is also provided.
    """
    try:
        single_day = _get_or_load_single_day_data(recording_path)
    except Exception as error:
        return {
            "success": False,
            "error": f"Unable to load data from '{recording_path}'. {error}",
        }

    all_statistics = single_day.roi_statistics
    if roi_indices is not None:
        valid_indices = [i for i in roi_indices if 0 <= i < len(all_statistics)]
        statistics = [(i, all_statistics[i]) for i in valid_indices]
    else:
        statistics = list(enumerate(all_statistics))

    if sort_by is not None:
        valid_sort_keys = ("skewness", "compactness", "footprint", "aspect_ratio", "pixel_count")
        if sort_by not in valid_sort_keys:
            return {
                "success": False,
                "error": f"Unable to sort by '{sort_by}'. Valid options: {', '.join(valid_sort_keys)}.",
            }
        statistics.sort(key=lambda pair: getattr(pair[1], sort_by, 0), reverse=True)

    if top_n is not None and top_n > 0:
        statistics = statistics[:top_n]

    results: list[dict[str, Any]] = []
    for index, roi in statistics:
        entry: dict[str, Any] = {
            "roi_index": index,
            "centroid": list(roi.mask.centroid),
            "pixel_count": roi.pixel_count,
        }
        for attr in ("skewness", "compactness", "footprint", "aspect_ratio"):
            value = getattr(roi, attr, None)
            if value is not None:
                entry[attr] = round(float(value), 4)
        results.append(entry)

    return {"success": True, "roi_count": len(results), "rois": results}


@gui_mcp.tool()
def query_fluorescence_traces_tool(
    recording_path: str,
    roi_indices: list[int],
    trace_type: Literal["fluorescence", "neuropil", "corrected", "spikes"] = "corrected",
    downsample_factor: int = 1,
) -> dict[str, Any]:
    """Queries fluorescence trace data for specific ROIs from a cindra-processed recording.

    Returns trace arrays for up to 50 ROIs at a time. Large traces can be downsampled to reduce response size.
    Does not require a GUI viewer.

    Args:
        recording_path: Absolute path to a cindra pipeline output directory.
        roi_indices: List of ROI indices to retrieve traces for (maximum 50).
        trace_type: The type of fluorescence trace to return. 'fluorescence' for raw cell fluorescence,
            'neuropil' for neuropil fluorescence, 'corrected' for neuropil-subtracted, 'spikes' for deconvolved.
        downsample_factor: Factor by which to downsample traces (1 = no downsampling, 10 = every 10th sample).
    """
    if len(roi_indices) > _MAX_TRACE_ROIS:
        return {
            "success": False,
            "error": f"Unable to query traces. Requested {len(roi_indices)} ROIs, maximum is {_MAX_TRACE_ROIS}.",
        }

    try:
        single_day = _get_or_load_single_day_data(recording_path)
    except Exception as error:
        return {"success": False, "error": f"Unable to load data from '{recording_path}'. {error}"}

    trace_map = {
        "fluorescence": single_day.cell_fluorescence,
        "neuropil": single_day.neuropil_fluorescence,
        "corrected": single_day.subtracted_fluorescence,
        "spikes": single_day.spikes,
    }
    traces = trace_map.get(trace_type)
    if traces is None or traces.size == 0:
        return {"success": False, "error": f"Unable to retrieve '{trace_type}' traces. Data is not available."}

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
        results.append({"roi_index": roi_index, "trace": [round(float(v), 4) for v in trace]})

    return {
        "success": True,
        "trace_type": trace_type,
        "downsample_factor": downsample_factor,
        "frame_count": int(traces.shape[1]),
        "traces": results,
    }


@gui_mcp.tool()
def query_cell_classification_tool(
    recording_path: str,
    roi_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Queries cell/non-cell classification labels and probabilities for ROIs in a cindra-processed recording.

    Returns binary labels (cell=1.0, non-cell=0.0) and classifier probability estimates for each ROI. Does not
    require a GUI viewer.

    Args:
        recording_path: Absolute path to a cindra pipeline output directory.
        roi_indices: Specific ROI indices to query. Returns all ROIs when not provided.
    """
    try:
        single_day = _get_or_load_single_day_data(recording_path)
    except Exception as error:
        return {"success": False, "error": f"Unable to load data from '{recording_path}'. {error}"}

    classification = single_day.cell_classification
    if classification is None or classification.size == 0:
        return {"success": False, "error": "Unable to retrieve cell classification. Data is not available."}

    total_rois = classification.shape[0]
    indices = [i for i in roi_indices if 0 <= i < total_rois] if roi_indices is not None else list(range(total_rois))

    results: list[dict[str, Any]] = [
        {
            "roi_index": i,
            "is_cell": bool(classification[i, 0] > _CELL_LABEL_THRESHOLD),
            "label": float(classification[i, 0]),
            "probability": round(float(classification[i, 1]), 4),
        }
        for i in indices
    ]

    cell_count = sum(1 for r in results if r["is_cell"])
    return {
        "success": True,
        "total_rois": total_rois,
        "queried_count": len(results),
        "cell_count": cell_count,
        "non_cell_count": len(results) - cell_count,
        "classifications": results,
    }


@gui_mcp.tool()
def query_multi_day_tracking_tool(
    recording_path: str,
    dataset: str,
) -> dict[str, Any]:
    """Queries multi-day tracking data for a cindra-processed recording's dataset.

    Returns the dataset structure including per-session recording IDs and mask counts for each mask layer. Does not
    require a GUI viewer.

    Args:
        recording_path: Absolute path to a cindra pipeline output directory that belongs to the multi-day dataset.
        dataset: The multi-day dataset name to query.
    """
    try:
        data = _get_or_load_viewer_data(recording_path, dataset=dataset)
    except Exception as error:
        return {"success": False, "error": f"Unable to load data from '{recording_path}'. {error}"}

    if not data.is_multi_day:
        available = list(data.available_datasets)
        if dataset and dataset not in available:
            return {
                "success": False,
                "error": f"Unable to load dataset '{dataset}'. Available datasets: {available}.",
            }
        return {"success": False, "error": "Unable to query tracking data. No multi-day dataset is loaded."}

    sessions: list[dict[str, Any]] = []
    for i in range(data.recording_count):
        recording = data.recording(i)
        session_info: dict[str, Any] = {
            "index": i,
            "session_id": recording.session_id,
            "has_channel_2": recording.has_channel_2,
            "original_mask_count": len(recording.original_masks),
            "deformed_mask_count": len(recording.deformed_masks),
            "template_mask_count": len(recording.template_masks),
            "tracked_mask_count": len(recording.tracked_masks),
        }
        sessions.append(session_info)

    return {
        "success": True,
        "dataset": data.active_dataset_name,
        "recording_count": data.recording_count,
        "sessions": sessions,
    }
