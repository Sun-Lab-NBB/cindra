"""Provides MCP tools for pipeline configuration generation and recording discovery.

These tools enable AI agents to generate default configuration files for both single-recording and multi-recording
pipelines, and to discover recordings available for processing under a given root directory.
"""

from __future__ import annotations

from typing import Any, Literal
from pathlib import Path

from ..dataclasses import MultiRecordingConfiguration, SingleRecordingConfiguration
from .mcp_instance import mcp


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
def discover_single_recording_candidates_tool(root_directory: str) -> dict[str, Any]:
    """Discovers recordings containing raw neural imaging data that can be processed by the single-recording pipeline.

    Searches recursively for cindra_parameters.json files (created by sl-experiment), which mark directories
    containing raw recording data suitable for single-recording processing. Returns the parent directory of each
    match as a recording candidate path.

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
    """Discovers recordings with completed single-recording processing that are candidates for multi-recording
    ROI tracking.

    Searches recursively for combined_metadata.npz files, which mark completed single-recording cindra outputs.
    Returns the grandparent directory paths (recording root directories containing cindra output).

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
                # The combined_metadata.npz is saved at the cindra root level (e.g., {recording}/cindra/). Its parent
                # is the cindra output directory, and its grandparent is the recording root directory.
                recording_root = str(marker_file.parent.parent)
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
