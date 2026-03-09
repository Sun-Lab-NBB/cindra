"""Provides MCP tools for creating and validating acquisition parameter files.

These tools enable AI agents to prepare raw imaging data for cindra processing by generating cindra_parameters.json
files and validating existing acquisition parameter files.
"""

from __future__ import annotations

import json
from pathlib import Path

from .mcp_instance import mcp

_PARAMETERS_FILENAME: str = "cindra_parameters.json"
"""The name of the acquisition parameters JSON file expected in each recording's data directory."""

_MAXIMUM_CHANNEL_COUNT: int = 2
"""The maximum number of imaging channels supported by the pipeline."""


@mcp.tool()
def generate_acquisition_parameters_file(
    output_directory: str,
    frame_rate: float,
    plane_number: int = 1,
    channel_number: int = 1,
    roi_number: int = 1,
    roi_lines: list[list[int]] | None = None,
    roi_x_coordinates: list[int] | None = None,
    roi_y_coordinates: list[int] | None = None,
) -> dict[str, bool | str | list[str] | dict[str, object]]:
    """Generates a cindra_parameters.json acquisition parameters file in the specified directory from the provided
    acquisition metadata, validating all fields before writing.

    Args:
        output_directory: The absolute path to the directory where the cindra_parameters.json file should be created,
            typically the same directory containing the raw TIFF files.
        frame_rate: The volume acquisition rate in Hz (rate at which all planes are acquired, not the per-plane rate).
        plane_number: The number of imaging planes per volume.
        channel_number: The number of channels per plane (1 or 2).
        roi_number: The number of ROIs per plane (1 for standard imaging, >1 for MROI data).
        roi_lines: The row indices for each ROI in the raw frame (required when roi_number > 1).
        roi_x_coordinates: The x-pixel offset for each ROI in the combined field of view (required when
            roi_number > 1).
        roi_y_coordinates: The y-pixel offset for each ROI in the combined field of view (required when
            roi_number > 1).

    Returns:
        On success, contains the resolved 'file_path', the validated 'parameters', and any 'warnings' for
        non-critical issues. On failure, contains an 'error' string or 'errors' list describing the issues.
        Both cases include a 'success' flag.
    """
    directory = Path(output_directory)

    if not directory.exists():
        return {
            "success": False,
            "error": f"Unable to generate acquisition parameters file. The directory does not exist: "
            f"{output_directory}",
        }

    if not directory.is_dir():
        return {
            "success": False,
            "error": f"Unable to generate acquisition parameters file. The path is not a directory: {output_directory}",
        }

    # Assembles the parameter dictionary.
    parameters: dict[str, object] = {
        "frame_rate": frame_rate,
        "plane_number": plane_number,
        "channel_number": channel_number,
    }

    if roi_number > 1:
        parameters["roi_number"] = roi_number
        if roi_lines is not None:
            parameters["roi_lines"] = roi_lines
        if roi_x_coordinates is not None:
            parameters["roi_x_coordinates"] = roi_x_coordinates
        if roi_y_coordinates is not None:
            parameters["roi_y_coordinates"] = roi_y_coordinates

    # Validates before writing.
    errors, warnings = _validate_acquisition_parameters(data=parameters)
    if errors:
        return {"success": False, "errors": errors}

    output_path = directory / _PARAMETERS_FILENAME
    with output_path.open("w") as file:
        json.dump(obj=parameters, fp=file, indent=4)

    result: dict[str, bool | str | list[str] | dict[str, object]] = {
        "success": True,
        "file_path": str(output_path),
        "parameters": parameters,
    }

    if warnings:
        result["warnings"] = warnings

    return result


@mcp.tool()
def validate_acquisition_parameters_file(file_path: str) -> dict[str, bool | str | list[str] | dict[str, object]]:
    """Validates an existing cindra_parameters.json file by checking that all required fields are present and have
    valid types and values, and reports any unrecognized fields or inconsistencies.

    Args:
        file_path: The absolute path to the cindra_parameters.json file to validate.

    Returns:
        On success, contains the resolved 'file_path', overall 'valid' status, and the loaded 'parameters', plus
        any validation 'errors' or 'warnings' detected. On failure, contains an 'error' describing the issue.
        Both cases include a 'success' flag.
    """
    path = Path(file_path)

    if not path.exists():
        return {
            "success": False,
            "error": f"Unable to validate acquisition parameters file. The file does not exist: {file_path}",
        }

    try:
        with path.open() as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        return {
            "success": False,
            "error": f"Unable to parse JSON file at '{file_path}': {error}",
        }

    if not isinstance(data, dict):
        return {
            "success": False,
            "error": (
                f"Unable to validate acquisition parameters file. Expected a JSON object at the top level, "
                f"but found {type(data).__name__}: {file_path}"
            ),
        }

    errors, warnings = _validate_acquisition_parameters(data=data)

    result: dict[str, bool | str | list[str] | dict[str, object]] = {
        "success": True,
        "file_path": str(path),
        "valid": not errors,
        "parameters": data,
    }

    if errors:
        result["errors"] = errors
    if warnings:
        result["warnings"] = warnings

    return result


def _validate_acquisition_parameters(
    data: dict[str, object],
) -> tuple[list[str], list[str]]:
    """Validates acquisition parameter values and returns lists of errors and warnings.

    Args:
        data: The acquisition parameter dictionary to validate.

    Returns:
        A tuple of two lists where the first contains error messages for invalid parameters and the second contains
        warning messages for potentially problematic values.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Validates frame_rate.
    frame_rate = data.get("frame_rate")
    if frame_rate is None:
        errors.append("Missing required field 'frame_rate'.")
    elif not isinstance(frame_rate, (int, float)):
        errors.append(f"'frame_rate' must be a number (found: {type(frame_rate).__name__}).")
    elif frame_rate <= 0:
        errors.append(f"'frame_rate' must be positive (found: {frame_rate}).")

    # Validates plane_number.
    plane_number = data.get("plane_number")
    if plane_number is None:
        errors.append("Missing required field 'plane_number'.")
    elif not isinstance(plane_number, int):
        errors.append(f"'plane_number' must be an integer (found: {type(plane_number).__name__}).")
    elif plane_number < 1:
        errors.append(f"'plane_number' must be at least 1 (found: {plane_number}).")

    # Validates channel_number.
    channel_number = data.get("channel_number")
    if channel_number is None:
        errors.append("Missing required field 'channel_number'.")
    elif not isinstance(channel_number, int):
        errors.append(f"'channel_number' must be an integer (found: {type(channel_number).__name__}).")
    elif channel_number < 1 or channel_number > _MAXIMUM_CHANNEL_COUNT:
        errors.append(f"'channel_number' must be 1 or 2 (found: {channel_number}).")

    # Validates roi_number and MROI fields.
    roi_number = data.get("roi_number", 1)
    if not isinstance(roi_number, int):
        errors.append(f"'roi_number' must be an integer (found: {type(roi_number).__name__}).")
    elif roi_number < 1:
        errors.append(f"'roi_number' must be at least 1 (found: {roi_number}).")
    elif roi_number > 1:
        # MROI mode — validates all MROI fields.
        roi_lines = data.get("roi_lines")
        roi_x_coordinates = data.get("roi_x_coordinates")
        roi_y_coordinates = data.get("roi_y_coordinates")

        if roi_lines is None:
            errors.append("Missing required field 'roi_lines' (required when roi_number > 1).")
        elif not isinstance(roi_lines, list) or not all(isinstance(r, list) for r in roi_lines):
            errors.append("'roi_lines' must be a list of lists of integers.")
        elif len(roi_lines) != roi_number:
            errors.append(f"'roi_lines' length ({len(roi_lines)}) must equal 'roi_number' ({roi_number}).")

        if roi_x_coordinates is None:
            errors.append("Missing required field 'roi_x_coordinates' (required when roi_number > 1).")
        elif not isinstance(roi_x_coordinates, list):
            errors.append("'roi_x_coordinates' must be a list of integers.")
        elif len(roi_x_coordinates) != roi_number:
            errors.append(
                f"'roi_x_coordinates' length ({len(roi_x_coordinates)}) must equal 'roi_number' ({roi_number})."
            )

        if roi_y_coordinates is None:
            errors.append("Missing required field 'roi_y_coordinates' (required when roi_number > 1).")
        elif not isinstance(roi_y_coordinates, list):
            errors.append("'roi_y_coordinates' must be a list of integers.")
        elif len(roi_y_coordinates) != roi_number:
            errors.append(
                f"'roi_y_coordinates' length ({len(roi_y_coordinates)}) must equal 'roi_number' ({roi_number})."
            )
    else:
        # Single-ROI mode — warns if MROI fields are present.
        if data.get("roi_lines"):
            warnings.append("'roi_lines' is set but 'roi_number' is 1 (single-ROI mode). Field will be ignored.")
        if data.get("roi_x_coordinates"):
            warnings.append(
                "'roi_x_coordinates' is set but 'roi_number' is 1 (single-ROI mode). Field will be ignored."
            )
        if data.get("roi_y_coordinates"):
            warnings.append(
                "'roi_y_coordinates' is set but 'roi_number' is 1 (single-ROI mode). Field will be ignored."
            )

    # Checks for unrecognized fields.
    known_fields = {
        "frame_rate",
        "plane_number",
        "channel_number",
        "roi_number",
        "roi_lines",
        "roi_x_coordinates",
        "roi_y_coordinates",
    }
    extra_fields = set(data.keys()) - known_fields
    if extra_fields:
        warnings.append(f"Unrecognized fields will be ignored by the pipeline: {sorted(extra_fields)}.")

    return errors, warnings
