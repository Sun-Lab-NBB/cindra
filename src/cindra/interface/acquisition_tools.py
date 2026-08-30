"""Provides MCP tools for creating and validating acquisition parameter files and inspecting raw TIFF data."""

from __future__ import annotations

import json
from typing import TypeIs
from pathlib import Path
from itertools import pairwise

from natsort import natsorted
from tifffile import TiffFile

from ..io import TIFF_EXTENSIONS, MAXIMUM_CHANNEL_COUNT, find_data_directory
from ..layout import PARAMETERS_FILENAME
from .mcp_instance import mcp

_MINIMUM_RECOMMENDED_FRAMES_PER_PLANE: int = 200
"""The minimum recommended number of frames per plane for reliable processing."""

_MAXIMUM_ROI_LINE_SLICE: int = 2000
"""The maximum number of line indices a single 'roi_line_slice' request is allowed to return."""

_ROI_LINE_SLICE_FIELDS: int = 3
"""The number of integers a 'roi_line_slice' request holds: the ROI index, the slice start, and the slice stop."""

_ROI_LINE_SPAN_FIELDS: int = 2
"""The number of integers one 'roi_line_spans' entry holds: the first and the last row index of that region."""


@mcp.tool()
def generate_acquisition_parameters_file_tool(
    raw_data_path: str,
    frame_rate: float,
    plane_number: int = 1,
    channel_number: int = 1,
    roi_number: int = 1,
    roi_lines: list[list[int]] | None = None,
    roi_line_spans: list[list[int]] | None = None,
    roi_x_coordinates: list[int] | None = None,
    roi_y_coordinates: list[int] | None = None,
) -> dict[str, bool | str | list[str] | dict[str, object]]:
    """Generates a cindra_parameters.json acquisition parameters file in the specified raw imaging directory.

    Builds the file from the provided acquisition metadata, validating all fields before writing.

    Args:
        raw_data_path: The absolute path to the raw imaging directory that directly holds the recording's TIFF files,
            which is the same directory the prepare tools take as their raw_data_path. The cindra_parameters.json file
            is written into it.
        frame_rate: The volume acquisition rate in Hz (rate at which all planes are acquired, not the per-plane rate).
            For multi-plane data, the per-plane sampling rate is frame_rate / plane_number.
        plane_number: The number of imaging planes per volume.
        channel_number: The number of channels per plane (1 or 2). By default, channel 1 is treated as the functional
            (calcium) channel and channel 2 as the optional structural channel. This routing is configurable via the
            first_channel_functional and second_channel_functional fields in the pipeline configuration.
        roi_number: The number of ROIs per plane (1 for standard imaging, >1 for MROI data).
        roi_line_spans: The first and last raw frame row index of each ROI, as one inclusive [first, last] pair per
            region. Prefer this form whenever every region occupies a contiguous run of rows, which every real
            mesoscope acquisition does, because three pairs carry what thousands of enumerated indices otherwise
            would. The bounds match the 'span' the validating tools report, so a span read from one of them is
            passed back here unchanged. Supply either this or roi_lines when roi_number > 1, never both.
        roi_lines: The row indices for each ROI in the raw frame, enumerated in full. Use this only for a region
            whose rows are not one contiguous run, which roi_line_spans cannot express. Supply either this or
            roi_line_spans when roi_number > 1, never both.
        roi_x_coordinates: The x-pixel offset for each ROI in the combined field of view (required when
            roi_number > 1).
        roi_y_coordinates: The y-pixel offset for each ROI in the combined field of view (required when
            roi_number > 1).

    Returns:
        On success, contains the resolved 'file_path', the validated 'parameters', and any 'warnings' for
        non-critical issues. The 'roi_lines' field of 'parameters' holds one summary entry per ROI, naming its 'roi'
        index, its 'line_count', its first and last row index as 'span', and whether the block covers a 'contiguous'
        run of rows. The file itself always stores the fully enumerated row indices, so a span request and an
        enumerated request that describe the same regions write byte-identical files. On failure, contains an 'error'
        string or 'errors' list describing the issues. Both cases include a 'success' flag.
    """
    directory = Path(raw_data_path)

    if not directory.exists():
        return {
            "success": False,
            "error": f"Unable to generate acquisition parameters file. The directory does not exist: {raw_data_path}",
        }

    if not directory.is_dir():
        return {
            "success": False,
            "error": f"Unable to generate acquisition parameters file. The path is not a directory: {raw_data_path}",
        }

    parameters: dict[str, object] = {
        "frame_rate": frame_rate,
        "plane_number": plane_number,
        "channel_number": channel_number,
    }

    if roi_lines is not None and roi_line_spans is not None:
        return {
            "success": False,
            "error": (
                "Unable to generate acquisition parameters file. Both 'roi_lines' and 'roi_line_spans' were "
                "supplied, and they describe the same regions two ways. Supply the spans alone, or the enumerated "
                "lines alone for a region whose rows are not one contiguous run."
            ),
        }

    if roi_line_spans is not None:
        expanded, span_errors = _expand_roi_line_spans(spans=roi_line_spans, roi_number=roi_number)
        if span_errors:
            return {"success": False, "errors": span_errors}
        roi_lines = expanded

    if roi_number > 1:
        parameters["roi_number"] = roi_number
        if roi_lines is not None:
            parameters["roi_lines"] = roi_lines
        if roi_x_coordinates is not None:
            parameters["roi_x_coordinates"] = roi_x_coordinates
        if roi_y_coordinates is not None:
            parameters["roi_y_coordinates"] = roi_y_coordinates

    errors, warnings = _validate_acquisition_parameters(data=parameters)
    if errors:
        return {"success": False, "errors": errors}

    output_path = directory / PARAMETERS_FILENAME
    with output_path.open("w") as file:
        json.dump(obj=parameters, fp=file, indent=4)

    result: dict[str, bool | str | list[str] | dict[str, object]] = {
        "success": True,
        "file_path": str(output_path),
        "parameters": _compact_acquisition_parameters(data=parameters),
    }

    if warnings:
        result["warnings"] = warnings

    return result


@mcp.tool()
def validate_acquisition_parameters_file_tool(
    file_path: str, roi_line_slice: list[int] | None = None
) -> dict[str, bool | str | list[str] | dict[str, object]]:
    """Validates an existing cindra_parameters.json file by checking that all required fields are present and have
    valid types and values, and reports any unrecognized fields or inconsistencies.

    Args:
        file_path: The absolute path to the cindra_parameters.json file to validate.
        roi_line_slice: The optional [roi_index, start, stop] triplet naming a half-open slice of one ROI's line list.
            The bounds are positions in that list, counted from zero, rather than the raw frame rows the 'span' field
            of the same response reports.
            When it is provided, the requested line indices are returned alongside the summaries, capped at 2000 lines
            per request.

    Returns:
        On success, contains the resolved 'file_path', overall 'valid' status, and the loaded 'parameters', plus
        any validation 'errors' or 'warnings' detected. The 'roi_lines' field of 'parameters' holds one summary entry
        per ROI, naming its 'roi' index, its 'line_count', its first and last line index as 'span', and whether the
        block covers a 'contiguous' run of rows. The requested line indices appear under a 'roi_line_slice' key. On
        failure, contains an 'error' describing the issue. Both cases include a 'success' flag. A 'success' value of
        True only means the tool ran. Callers MUST gate downstream steps on the 'valid' field, which can be False even
        when 'success' is True.
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

    slice_payload: dict[str, object] | None = None
    if roi_line_slice is not None:
        slice_payload, slice_error = _resolve_roi_line_slice(roi_lines=data.get("roi_lines"), request=roi_line_slice)
        if slice_error:
            return {"success": False, "error": f"Unable to validate acquisition parameters file. {slice_error}"}

    result: dict[str, bool | str | list[str] | dict[str, object]] = {
        "success": True,
        "file_path": str(path),
        "valid": not errors,
        "parameters": _compact_acquisition_parameters(data=data),
    }

    if slice_payload is not None:
        result["roi_line_slice"] = slice_payload
    if errors:
        result["errors"] = errors
    if warnings:
        result["warnings"] = warnings

    return result


@mcp.tool()
def validate_recording_readiness_tool(raw_data_path: str, roi_line_slice: list[int] | None = None) -> dict[str, object]:
    """Validates that a raw imaging directory is fully ready for cindra single-recording processing.

    Verifies that the cindra_parameters.json acquisition parameters file is present and valid, that raw TIFF files exist
    and are readable, and that the TIFF data is compatible with the acquisition parameters. The imaging directory is
    resolved the way the conversion resolves it: the directory holding the acquisition parameters file beneath the named
    path is the one whose TIFF files are read, and that one directory is scanned without descending further. Naming
    either the imaging directory itself or any parent of it therefore yields the verdict the pipeline would reach, which
    is why the recording roots the discovery tool reports are accepted directly. Files whose frame shape differs from
    the shape holding the most frames are reported as warnings rather than errors. A raw directory commonly holds an
    unrelated file such as an anatomical z-stack, which must be excluded through the 'file_io.ignored_file_names'
    configuration parameter.

    Args:
        raw_data_path: The absolute path to the recording's raw imaging directory, or to any parent of it. The
            directory holding the cindra_parameters.json file beneath it is the one validated, and the recording's
            TIFF files must sit directly beside that file.
        roi_line_slice: The optional [roi_index, start, stop] triplet naming a half-open slice of one ROI's line list.
            The bounds are positions in that list, counted from zero, rather than the raw frame rows the 'span' field
            of the same response reports.
            When it is provided, the requested line indices are returned alongside the summaries, capped at 2000 lines
            per request.

    Returns:
        On success, contains the resolved 'raw_data_path' naming the imaging directory that was validated, the overall
        'valid' status, 'tiff_file_count', 'total_frames', validated 'acquisition_parameters', per-file 'files' details,
        and any validation 'errors' or 'warnings'. The resolved 'raw_data_path' is the named path itself unless the
        parameters file was found deeper. The 'roi_lines' field of 'acquisition_parameters' holds one summary entry per
        ROI, naming its 'roi' index, its 'line_count', its first and last line index as 'span', and whether the block
        covers a 'contiguous' run of rows. The requested line indices appear under a 'roi_line_slice' key. The
        'frame_height', 'frame_width', 'dtype', and a meaningful nonzero 'frames_per_plane' appear only when at least
        one TIFF is readable. When no TIFFs are found, the return has 'valid' False with an 'errors' entry and omits
        those frame-dimension keys. On failure, contains an 'error' describing the issue. Both cases include a 'success'
        flag. A 'success' value of True only means the tool ran. Callers MUST gate downstream steps on the 'valid'
        field, which can be False even when 'success' is True.
    """
    directory = Path(raw_data_path)

    if not directory.exists():
        return {
            "success": False,
            "error": f"Unable to validate recording readiness. The directory does not exist: {raw_data_path}",
        }

    if not directory.is_dir():
        return {
            "success": False,
            "error": f"Unable to validate recording readiness. The path is not a directory: {raw_data_path}",
        }

    errors: list[str] = []
    warnings: list[str] = []

    directory = _resolve_imaging_directory(directory=directory)

    parameters_path = directory / PARAMETERS_FILENAME
    if not parameters_path.exists():
        return {"success": False, "error": _resolve_missing_parameters_message(directory=Path(raw_data_path))}

    try:
        with parameters_path.open() as file:
            parameters = json.load(file)
    except json.JSONDecodeError as exception:
        return {
            "success": False,
            "error": f"Unable to validate recording readiness. Failed to parse {PARAMETERS_FILENAME}: {exception}",
        }

    if not isinstance(parameters, dict):
        return {
            "success": False,
            "error": (
                f"Unable to validate recording readiness. Expected a JSON object in {PARAMETERS_FILENAME}, "
                f"but found {type(parameters).__name__}."
            ),
        }

    parameter_errors, parameter_warnings = _validate_acquisition_parameters(data=parameters)
    errors.extend(parameter_errors)
    warnings.extend(parameter_warnings)

    # Extracts validated acquisition fields for cross-validation with TIFF data.
    plane_number = parameters.get("plane_number", 1)
    channel_number = parameters.get("channel_number", 1)
    roi_number = parameters.get("roi_number", 1)
    roi_lines = parameters.get("roi_lines")
    interleave_stride = (
        plane_number * channel_number if isinstance(plane_number, int) and isinstance(channel_number, int) else 0
    )

    slice_payload: dict[str, object] | None = None
    if roi_line_slice is not None:
        slice_payload, slice_error = _resolve_roi_line_slice(roi_lines=roi_lines, request=roi_line_slice)
        if slice_error:
            return {"success": False, "error": f"Unable to validate recording readiness. {slice_error}"}

    # Discovers TIFF files using the same deduplicated globbing as the pipeline, resolving paths so that case-variant
    # extension matches on case-insensitive filesystems are not counted more than once.
    discovered_paths: set[Path] = set()
    for extension in TIFF_EXTENSIONS:
        discovered_paths.update(path.resolve() for path in directory.glob(f"*.{extension}"))
    tiff_paths = natsorted(discovered_paths)

    if not tiff_paths:
        errors.append(f"No TIFF files found in: {raw_data_path}")
        empty_result: dict[str, object] = {
            "success": True,
            "raw_data_path": str(directory),
            "valid": False,
            "errors": errors,
            "warnings": warnings,
            "acquisition_parameters": _compact_acquisition_parameters(data=parameters),
            "tiff_file_count": 0,
            "total_frames": 0,
            "files": [],
        }
        if slice_payload is not None:
            empty_result["roi_line_slice"] = slice_payload
        return empty_result

    file_details: list[dict[str, str | int]] = []
    total_frames: int = 0
    reference_height: int | None = None
    reference_width: int | None = None
    reference_dtype: str | None = None
    shape_groups: dict[tuple[int, int], list[tuple[str, int]]] = {}

    for tiff_path in tiff_paths:
        try:
            with TiffFile(tiff_path) as tiff:
                page_count = len(tiff.pages)

                if page_count == 0:
                    errors.append(f"TIFF file contains zero frames: {tiff_path.name}")
                    file_details.append({"name": tiff_path.name, "frames": 0})
                    continue

                # Reads dimensions and dtype from the first page without loading full frame data.
                first_page = tiff.pages[0]
                height = first_page.shape[0]
                width = first_page.shape[1] if len(first_page.shape) > 1 else 1
                dtype = str(first_page.dtype)

            file_details.append(
                {"name": tiff_path.name, "frames": page_count, "height": height, "width": width, "dtype": dtype}
            )
            shape_groups.setdefault((height, width), []).append((tiff_path.name, page_count))

            # Tracks the reference dtype from the first valid file. Frame dimensions are resolved after the loop, from
            # the shape whose files hold the most frames in total.
            if reference_dtype is None:
                reference_dtype = dtype
            elif dtype != reference_dtype:
                warnings.append(f"Dtype varies in {tiff_path.name}: {dtype} (first file uses {reference_dtype}).")

        except Exception as exception:
            errors.append(f"Unable to read TIFF file {tiff_path.name}: {type(exception).__name__}: {exception}")

    # Resolves the recording's frame shape as the shape whose files hold the most frames in total, and counts
    # frames from those files alone. Receives a directory rather than a configuration, so it cannot read the
    # 'file_io.ignored_file_names' exclusions the pipeline applies. A raw directory commonly holds a differently shaped
    # file that is not part of the recording, such as an anatomical z-stack, and reporting it as an error here would
    # fail a recording the pipeline processes correctly. The conversion stage still rejects a genuinely ragged
    # recording, so this tool reports the outliers and leaves the verdict to the exclusions the caller configures.
    if shape_groups:
        (reference_height, reference_width), majority_files = max(
            shape_groups.items(), key=lambda group: sum(page_count for _, page_count in group[1])
        )
        total_frames = sum(page_count for _, page_count in majority_files)

        for (height, width), outlier_files in shape_groups.items():
            if (height, width) == (reference_height, reference_width):
                continue
            outlier_names = ", ".join(name for name, _ in outlier_files)
            warnings.append(
                f"Frame shape differs in {outlier_names}: {height}x{width}, while the other files hold "
                f"{reference_height}x{reference_width} frames. Every file the pipeline loads must hold frames of the "
                f"same shape, so exclude any file that is not part of the recording, such as an anatomical z-stack, "
                f"through the 'file_io.ignored_file_names' configuration parameter. The frame counts reported below "
                f"cover the {reference_height}x{reference_width} files alone."
            )

    frames_per_plane: int = 0
    if total_frames > 0 and interleave_stride > 0:
        frames_per_plane = total_frames // interleave_stride
        remainder = total_frames % interleave_stride

        if remainder != 0:
            warnings.append(
                f"Total frames ({total_frames}) do not divide evenly by the interleave stride "
                f"({interleave_stride} = {plane_number} planes x {channel_number} channels), which happens when an "
                f"acquisition stops partway through a volume. Binarization discards the {remainder} trailing frames "
                f"of that incomplete cycle, because they reach some planes and channels of the recording and not "
                f"others. Every plane binary then holds the {frames_per_plane} frames reported below."
            )

        if frames_per_plane < _MINIMUM_RECOMMENDED_FRAMES_PER_PLANE:
            warnings.append(
                f"Frames per plane ({frames_per_plane}) is below the recommended minimum of "
                f"{_MINIMUM_RECOMMENDED_FRAMES_PER_PLANE} for reliable processing."
            )

    # Cross-checks the MROI line blocks against the frame height read from the TIFF files and against each other. A
    # malformed entry is reported by the shared validator above, so the cross-check is confined to the entries that
    # hold integer line indices.
    if isinstance(roi_number, int) and roi_number > 1 and isinstance(roi_lines, list):
        block_errors, block_warnings = _check_roi_line_blocks(roi_lines=roi_lines, frame_height=reference_height)
        errors.extend(block_errors)
        warnings.extend(block_warnings)

    if reference_dtype is not None and reference_dtype not in ("uint16", "int16", "int32"):
        warnings.append(
            f"TIFF dtype '{reference_dtype}' is not one of the natively supported types (uint16, int16, int32). "
            f"Data will be cast to int16 during binarization, which may cause precision loss."
        )
    elif reference_dtype in ("uint16", "int32"):
        warnings.append(
            f"TIFF dtype '{reference_dtype}' values are divided by 2 (floor division) during binarization to fit "
            f"the int16 range, so all pixel values are halved before processing."
        )

    result: dict[str, object] = {
        "success": True,
        "raw_data_path": str(directory),
        "valid": not errors,
        "acquisition_parameters": _compact_acquisition_parameters(data=parameters),
        "tiff_file_count": len(tiff_paths),
        "total_frames": total_frames,
        "frames_per_plane": frames_per_plane,
        "files": file_details,
    }

    if slice_payload is not None:
        result["roi_line_slice"] = slice_payload

    if reference_height is not None:
        result["frame_height"] = reference_height
        result["frame_width"] = reference_width
        result["dtype"] = reference_dtype

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
        The error messages for invalid parameters, paired with the warning messages for values that pass validation but
        are likely unintended.
    """
    errors: list[str] = []
    warnings: list[str] = []

    frame_rate = data.get("frame_rate")
    if frame_rate is None:
        errors.append("Missing required field 'frame_rate'.")
    elif not isinstance(frame_rate, (int, float)):
        errors.append(f"'frame_rate' must be a number (found: {type(frame_rate).__name__}).")
    elif frame_rate <= 0:
        errors.append(f"'frame_rate' must be positive (found: {frame_rate}).")

    plane_number = data.get("plane_number")
    if plane_number is None:
        errors.append("Missing required field 'plane_number'.")
    elif not isinstance(plane_number, int):
        errors.append(f"'plane_number' must be an integer (found: {type(plane_number).__name__}).")
    elif plane_number < 1:
        errors.append(f"'plane_number' must be at least 1 (found: {plane_number}).")

    channel_number = data.get("channel_number")
    if channel_number is None:
        errors.append("Missing required field 'channel_number'.")
    elif not isinstance(channel_number, int):
        errors.append(f"'channel_number' must be an integer (found: {type(channel_number).__name__}).")
    elif channel_number < 1 or channel_number > MAXIMUM_CHANNEL_COUNT:
        errors.append(f"'channel_number' must be 1 or 2 (found: {channel_number}).")

    roi_number = data.get("roi_number", 1)
    if not isinstance(roi_number, int):
        errors.append(f"'roi_number' must be an integer (found: {type(roi_number).__name__}).")
    elif roi_number < 1:
        errors.append(f"'roi_number' must be at least 1 (found: {roi_number}).")
    elif roi_number > 1:
        roi_lines = data.get("roi_lines")
        roi_x_coordinates = data.get("roi_x_coordinates")
        roi_y_coordinates = data.get("roi_y_coordinates")

        if roi_lines is None:
            errors.append("Missing required field 'roi_lines' (required when roi_number > 1).")
        elif not isinstance(roi_lines, list) or not all(_is_integer_list(value=lines) for lines in roi_lines):
            errors.append("'roi_lines' must be a list of lists of integers.")
        elif len(roi_lines) != roi_number:
            errors.append(f"'roi_lines' length ({len(roi_lines)}) must equal 'roi_number' ({roi_number}).")

        if roi_x_coordinates is None:
            errors.append("Missing required field 'roi_x_coordinates' (required when roi_number > 1).")
        elif not _is_integer_list(value=roi_x_coordinates):
            errors.append("'roi_x_coordinates' must be a list of integers.")
        elif len(roi_x_coordinates) != roi_number:
            errors.append(
                f"'roi_x_coordinates' length ({len(roi_x_coordinates)}) must equal 'roi_number' ({roi_number})."
            )

        if roi_y_coordinates is None:
            errors.append("Missing required field 'roi_y_coordinates' (required when roi_number > 1).")
        elif not _is_integer_list(value=roi_y_coordinates):
            errors.append("'roi_y_coordinates' must be a list of integers.")
        elif len(roi_y_coordinates) != roi_number:
            errors.append(
                f"'roi_y_coordinates' length ({len(roi_y_coordinates)}) must equal 'roi_number' ({roi_number})."
            )
    else:
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


def _expand_roi_line_spans(spans: list[list[int]], roi_number: int) -> tuple[list[list[int]] | None, list[str]]:
    """Expands the inclusive row bounds of each region into the enumerated row indices the parameters file stores.

    Notes:
        Every region of a real mesoscope acquisition occupies a contiguous run of rows, so its bounds carry what its
        enumerated indices carry. Expanding here keeps the written file in the one shape every reader already parses,
        which is why a span request and an enumerated request describing the same regions write the same file.

        The bounds are inclusive, matching the 'span' the validating tools report, so a span read from one of them is
        passed back unchanged. This differs from the half-open 'roi_line_slice' request, which names a slice.

    Args:
        spans: The inclusive [first, last] row bounds of each region.
        roi_number: The regions the recording declares, which the bounds must cover one for one.

    Returns:
        The enumerated row indices of each region and an empty error list, or None and the errors the bounds carry.

    """
    errors: list[str] = []

    if not isinstance(spans, list) or not spans:
        return None, ["'roi_line_spans' must be a non-empty list of [first, last] row bounds."]

    expanded: list[list[int]] = []
    for index, span in enumerate(spans):
        if not _is_integer_list(value=span) or len(span) != _ROI_LINE_SPAN_FIELDS:
            errors.append(
                f"'roi_line_spans' entry {index} must hold exactly two integers naming the first and the last row "
                f"index of that region, but got {span}."
            )
            continue
        first, last = span
        if first < 0:
            errors.append(f"'roi_line_spans' entry {index} starts at row {first}, but a row index cannot be negative.")
            continue
        if last < first:
            errors.append(
                f"'roi_line_spans' entry {index} ends at row {last}, which precedes its first row {first}. The "
                f"bounds are inclusive, so name a single-row region as [{first}, {first}]."
            )
            continue
        expanded.append(list(range(first, last + 1)))

    if errors:
        return None, errors

    if len(expanded) != roi_number:
        return None, [f"'roi_line_spans' length ({len(expanded)}) must equal 'roi_number' ({roi_number})."]

    return expanded, []


def _compact_acquisition_parameters(data: dict[str, object]) -> dict[str, object]:
    """Replaces the 'roi_lines' arrays of the acquisition parameters with per-ROI summaries.

    Notes:
        A three-region mesoscope recording carries thousands of line indices, whose transport costs far more than the
        summary for which an agent reads them. The values themselves are served through the 'roi_line_slice' request.

    Args:
        data: The acquisition parameters to compact.

    Returns:
        A copy of the acquisition parameters holding every other field verbatim.
    """
    compacted = dict(data)
    roi_lines = compacted.get("roi_lines")
    if isinstance(roi_lines, list):
        compacted["roi_lines"] = _summarize_roi_line_blocks(roi_lines=roi_lines)
    return compacted


def _summarize_roi_line_blocks(roi_lines: list[object]) -> list[dict[str, object]]:
    """Summarizes every ROI line block as its length, its span, and whether it covers a contiguous run of rows.

    Args:
        roi_lines: The 'roi_lines' entries the acquisition parameters hold.

    Returns:
        One entry per ROI, holding the 'roi' index, the 'line_count', the first and last line index as 'span', and the
        'contiguous' flag. An entry holding anything other than integer line indices, or holding no line index at all,
        is summarized with an empty span and a False 'contiguous' flag.
    """
    summaries: list[dict[str, object]] = []
    for roi_index, lines in enumerate(roi_lines):
        if not _is_integer_list(value=lines) or not lines:
            line_count = len(lines) if isinstance(lines, list) else 0
            summaries.append({"roi": roi_index, "line_count": line_count, "span": [], "contiguous": False})
            continue

        first_line = min(lines)
        last_line = max(lines)
        contiguous = len(set(lines)) == len(lines) and last_line - first_line + 1 == len(lines)
        summaries.append(
            {
                "roi": roi_index,
                "line_count": len(lines),
                "span": [first_line, last_line],
                "contiguous": contiguous,
            }
        )
    return summaries


def _check_roi_line_blocks(roi_lines: list[object], frame_height: int | None) -> tuple[list[str], list[str]]:
    """Cross-checks the MROI line blocks against the raw frame height and against each other.

    Args:
        roi_lines: The 'roi_lines' entries the acquisition parameters hold.
        frame_height: The frame height read from the recording's TIFF files, or None when no TIFF file was readable.

    Returns:
        The error messages for the blocks that reach past the last row of the raw frame, paired with the warning
        messages for the gaps and overlaps between the blocks of consecutive ROIs.
    """
    errors: list[str] = []
    warnings: list[str] = []
    spans: list[tuple[int, int, int]] = []

    for roi_index, lines in enumerate(roi_lines):
        if not _is_integer_list(value=lines) or not lines:
            continue

        first_line = min(lines)
        last_line = max(lines)
        spans.append((roi_index, first_line, last_line))

        if frame_height is not None and last_line >= frame_height:
            errors.append(
                f"ROI {roi_index} roi_lines maximum ({last_line}) reaches past the last row of the raw frame, which "
                f"holds {frame_height} rows, so its highest valid line index is {frame_height - 1}."
            )

    for (previous_index, _, previous_last), (current_index, current_first, _) in pairwise(spans):
        if current_first <= previous_last:
            warnings.append(
                f"ROI {current_index} roi_lines start at line {current_first}, which overlaps ROI {previous_index}, "
                f"whose block ends at line {previous_last}. Overlapping blocks assign the same raw rows to two ROIs."
            )
        elif current_first > previous_last + 1:
            warnings.append(
                f"ROI {current_index} roi_lines start at line {current_first}, leaving "
                f"{current_first - previous_last - 1} raw rows unassigned after ROI {previous_index}, whose block "
                f"ends at line {previous_last}. A gap between blocks commonly follows a transcription slip."
            )

    return errors, warnings


def _resolve_roi_line_slice(roi_lines: object, request: list[int]) -> tuple[dict[str, object] | None, str]:
    """Resolves the requested half-open slice of one ROI's line list.

    Args:
        roi_lines: The 'roi_lines' value the acquisition parameters hold.
        request: The [roi_index, start, stop] triplet naming the ROI and the half-open range of its line list.

    Returns:
        The payload holding the 'roi' index, the 'start' and 'stop' bounds, and the requested 'lines', paired with an
        empty message. A rejected request returns None paired with the message stating why it was rejected.
    """
    if not isinstance(roi_lines, list) or not roi_lines:
        return None, "The acquisition parameters hold no 'roi_lines' entries to slice."

    if len(request) != _ROI_LINE_SLICE_FIELDS or not _is_integer_list(value=request):
        return None, (
            f"The 'roi_line_slice' request must hold exactly three integers naming the ROI index, the slice start, "
            f"and the slice stop, but got {request}."
        )

    roi_index, start, stop = request
    if roi_index < 0 or roi_index >= len(roi_lines):
        return None, (
            f"The 'roi_line_slice' ROI index must be in the range [0, {len(roi_lines) - 1}], which is the range the "
            f"acquisition parameters cover, but got {roi_index}."
        )

    lines = roi_lines[roi_index]
    if not _is_integer_list(value=lines):
        return None, f"ROI {roi_index} does not hold a list of integer line indices, so it cannot be sliced."

    if start < 0 or stop <= start or stop > len(lines):
        message = (
            f"The 'roi_line_slice' bounds must satisfy 0 <= start < stop <= {len(lines)}, which is the number of lines "
            f"ROI {roi_index} holds, but got start {start} and stop {stop}."
        )

        # The 'span' this tool reports beside the slice is measured in raw frame rows, so a caller that reads a bound
        # from there and passes it back here lands outside the list-index range the check above covers.
        if lines[0] <= start <= lines[-1]:
            message += (
                f" These bounds index the line list of ROI {roi_index}, counted from zero, rather than the raw frame "
                f"rows its 'span' reports, and that ROI spans rows {lines[0]} to {lines[-1]}."
            )
            if start in lines:
                message += f" Raw row {start} is list index {lines.index(start)}."
        return None, message

    if stop - start > _MAXIMUM_ROI_LINE_SLICE:
        return None, (
            f"The 'roi_line_slice' request covers {stop - start} lines, which exceeds the {_MAXIMUM_ROI_LINE_SLICE} "
            f"line cap of a single request. Request a narrower range."
        )

    return {"roi": roi_index, "start": start, "stop": stop, "lines": lines[start:stop]}, ""


def _resolve_imaging_directory(directory: Path) -> Path:
    """Resolves the imaging directory the conversion reads from the path the caller named.

    Notes:
        Defers to find_data_directory, which is the resolution the conversion itself performs: the directory holding
        the recording's acquisition parameters file is the one scanned for TIFF files. Sharing that resolution keeps
        this tool's verdict identical to the one the pipeline reaches, so a caller that names a recording root has the
        imaging directory beneath it resolved and validated.

    Args:
        directory: The path the caller named as the recording's raw imaging path.

    Returns:
        The directory holding the acquisition parameters file, or the named directory unchanged when its subtree holds
        no such file, cannot be read, or is not a directory.
    """
    if (directory / PARAMETERS_FILENAME).is_file():
        return directory
    try:
        return find_data_directory(data_path=directory)
    except FileNotFoundError, OSError, ValueError:
        return directory


def _resolve_missing_parameters_message(directory: Path) -> str:
    """Composes the readiness error reported when no acquisition parameters file exists beneath the named path.

    Args:
        directory: The path the caller named as the recording's raw imaging path.

    Returns:
        The message asking for a parameters file to be created beside the recording's TIFF files.
    """
    return (
        f"Unable to validate recording readiness. No {PARAMETERS_FILENAME} file is stored inside {directory} or "
        f"anywhere in its subtree. Use generate_acquisition_parameters_file_tool to create one inside the directory "
        f"that directly holds the recording's raw TIFF files, and pass either that directory or a parent of it as "
        f"raw_data_path."
    )


def _is_integer_list(value: object) -> TypeIs[list[int]]:
    """Determines whether the value is a list holding integer elements alone.

    Notes:
        Booleans are rejected, because JSON deserializes 'true' into a subclass of int that the pipeline cannot use
        as a line or pixel index.

    Args:
        value: The deserialized JSON value to check.

    Returns:
        True when the value is a list of integers, and False otherwise.
    """
    return isinstance(value, list) and all(
        isinstance(element, int) and not isinstance(element, bool) for element in value
    )
