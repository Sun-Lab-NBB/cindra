"""Provides MCP tools for pipeline configuration generation, reading, validation, modification, recording discovery,
and dataset name resolution.

These tools enable AI agents to generate default configuration files for both single-recording and multi-recording
pipelines, read, validate, and modify configuration files, discover recordings available for processing under a
given root directory, and construct qualified dataset names for multi-recording processing.
"""

from __future__ import annotations

from enum import Enum
from types import NoneType, UnionType
from typing import Literal, get_args, get_origin, get_type_hints
from os.path import commonpath
from pathlib import Path
from dataclasses import (
    fields as dataclass_fields,
    is_dataclass,
)

import yaml
from natsort import natsorted
from ataraxis_data_structures import discover_marker_roots

from ..io import resolve_recording_roots
from ..layout import (
    PARAMETERS_FILENAME,
    OUTPUT_DIRECTORY_NAME,
    COMBINED_METADATA_FILENAME,
)
from ..dataclasses import (
    BaselineMethod,
    ReferenceImageType,
    MultiRecordingConfiguration,
    SingleRecordingConfiguration,
)
from .mcp_instance import mcp

_MAXIMUM_SPEED_FACTOR: int = 5
"""The upper bound of the typical speed_factor range for diffeomorphic registration."""

_MAXIMUM_PERCENTAGE: int = 100
"""The maximum valid value for percentage-based parameters (prevalence, percentile)."""

_FORBIDDEN_FILESYSTEM_CHARACTERS: frozenset[str] = frozenset('\\/:*?"<>|\x00')
"""Characters that are invalid in directory names on common filesystems."""

_MAXIMUM_CONTROL_CHARACTER_ORDINAL: int = 32
"""The exclusive upper bound of the ASCII control character range."""


@mcp.tool()
def generate_config_file_tool(
    output_path: str, pipeline_type: Literal["single-recording", "multi-recording"]
) -> dict[str, str | bool]:
    """Generates a default configuration YAML file for the specified pipeline type.

    Creates a configuration file with defaults for every user-tunable parameter. The single-recording defaults
    validate as generated, while the multi-recording defaults leave 'recording_io.dataset_name' empty, which
    validate_config_file_tool reports as an error until a name is set (prepare_multi_recording_batch_tool sets it,
    together with 'recording_io.recording_directories', for batch runs). For the full set of writable parameters and
    their constraints, consult the single-recording-configuration and multi-recording-configuration cindra skills.

    Args:
        output_path: The absolute path where the configuration file should be saved.
        pipeline_type: The type of pipeline configuration to generate ('single-recording' or 'multi-recording').

    Returns:
        On success, contains the resolved 'file_path' and the 'pipeline_type'. The resolved 'file_path' may differ
        from 'output_path' when the suffix is normalized to '.yaml' (any non-'.yaml' suffix is replaced), so use the
        returned 'file_path' for subsequent validate and prepare steps. On failure, contains an 'error' describing
        the issue. Both cases include a 'success' flag.
    """
    output = Path(output_path)

    if not output.parent.exists():
        return {
            "success": False,
            "error": f"Unable to generate configuration file. The parent directory does not exist: {output.parent}",
        }

    if output.suffix != ".yaml":
        output = output.with_suffix(".yaml")

    configuration: SingleRecordingConfiguration | MultiRecordingConfiguration
    if pipeline_type == "single-recording":
        configuration = SingleRecordingConfiguration()
    else:
        configuration = MultiRecordingConfiguration()

    configuration.save(file_path=output)

    return {"success": True, "file_path": str(output), "pipeline_type": pipeline_type}


@mcp.tool()
def discover_recordings_tool(root_directory: str) -> dict[str, object]:
    """Discovers recordings available for single-recording and multi-recording processing under a root directory.

    Searches recursively for cindra_parameters.json files (marking raw recordings ready for single-recording
    processing) and combined_metadata.npz files (marking completed single-recording outputs ready for
    multi-recording processing). Every candidate carries the session-level recording root together with the specific
    path the tools that consume it take, so a discovered entry feeds the next step without any path arithmetic by the
    caller. Recording roots are resolved via ``resolve_recording_roots``, which maps a pipeline output directory to
    its parent and strips the trailing components a raw-data directory shares with its peers.

    Notes:
        The marker search keeps only the entries that resolve to files, so a directory carrying a marker's name is not
        reported as a recording. A subtree the process cannot read falls back to a tolerant recursive glob that skips
        it and applies no file check. An unreadable subtree therefore lowers the candidate counts without surfacing a
        distinct diagnostic, and can let a directory carrying a marker's name through.

    Args:
        root_directory: The absolute path to the root directory to search.

    Returns:
        On success, contains the 'single_recording_candidates' and 'multi_recording_candidates' lists together with
        the matching 'single_recording_count' and 'multi_recording_count'. Every single-recording candidate maps
        'recording_root' to the session-level root and 'raw_data_path' to the directory holding the discovered
        cindra_parameters.json marker. That 'raw_data_path' is the value validate_recording_readiness_tool and
        generate_acquisition_parameters_file_tool take, and the value the 'raw_data_paths' argument of
        prepare_single_recording_batch_tool takes. Every multi-recording candidate maps 'recording_root' to the
        session-level root and 'output_root' to the parent of the cindra directory holding the discovered
        combined_metadata.npz marker. That 'output_root' is the value the status, cleaning, and results tools take,
        and the value the 'output_roots' entries of prepare_multi_recording_batch_tool take. On failure, contains an
        'error' describing the issue. Both cases include a 'success' flag.
    """
    root_path = Path(root_directory)

    if not root_path.exists():
        return {
            "success": False,
            "error": f"Unable to discover recordings. The directory does not exist: {root_directory}",
        }

    if not root_path.is_dir():
        return {
            "success": False,
            "error": f"Unable to discover recordings. The path is not a directory: {root_directory}",
        }

    # Discovers single-recording candidates via cindra_parameters.json marker files. The marker-root discoverer keeps
    # only the entries that resolve to files, so a directory carrying a marker's name is not reported as a
    # recording. It refuses a subtree it cannot read, and this tool surveys a root the caller chose rather than a path
    # the pipeline owns, so a denial falls back to the tolerant scan rather than reporting no candidate at all.
    single_marker_parents = _discover_marker_parents(root_path=root_path, marker_name=PARAMETERS_FILENAME)

    # The raw data directory a readiness check and a batch preparation take is the directory holding the marker,
    # which is the recording root only when the recording keeps its TIFF files at the top of its own tree.
    single_recording_candidates = [
        {"recording_root": str(recording_root), "raw_data_path": str(marker_parent)}
        for recording_root, marker_parent in _pair_marker_parents_with_roots(marker_parents=single_marker_parents)
    ]

    multi_marker_parents = _discover_marker_parents(root_path=root_path, marker_name=COMBINED_METADATA_FILENAME)

    multi_recording_candidates = [
        {
            "recording_root": str(recording_root),
            "output_root": str(_resolve_marker_output_root(recording_root=recording_root, marker_parent=marker_parent)),
        }
        for recording_root, marker_parent in _pair_marker_parents_with_roots(marker_parents=multi_marker_parents)
    ]

    return {
        "success": True,
        "single_recording_candidates": single_recording_candidates,
        "single_recording_count": len(single_recording_candidates),
        "multi_recording_candidates": multi_recording_candidates,
        "multi_recording_count": len(multi_recording_candidates),
    }


@mcp.tool()
def resolve_dataset_name_tool(
    dataset_name: str,
    output_roots: list[str],
    specifier: str = "",
) -> dict[str, object]:
    """Constructs a qualified dataset name by combining a shared base name with a batch-specific specifier.

    When multiple groups of recordings share the same analysis type (dataset_name), each group needs a unique qualified
    name for its output directory and batch processing key. Combines the user-provided dataset name with a
    specifier that distinguishes the group.

    When no specifier is provided, one is derived automatically from the deepest common parent directory of the
    output roots. For example, recordings whose output roots are /data/animal_A/rec1 and /data/animal_A/rec2 yield
    specifier 'animal_a', because all returned names are lowercased. The agent can also determine the specifier
    through semantic decomposition of recording names or directory structure, or the user can provide one explicitly.

    Args:
        dataset_name: The shared name identifying the analysis type (e.g., 'learning_task'). This is the base name
            common to all groups in a batch.
        output_roots: The absolute pipeline output root of every recording in this group, each the parent of that
            recording's cindra directory. Used to derive the specifier when none is explicitly provided.
        specifier: An explicit batch-specific label distinguishing this group of recordings (e.g., an animal ID, brain
            region, or session group). When empty, the specifier is derived from the common parent directory of the
            output roots.

    Returns:
        On success, contains the qualified 'dataset_name' (specifier_base), the 'base_name', and the 'specifier'
        used, all lowercased. On failure, contains an 'error' describing the issue. Both cases include a 'success'
        flag.
    """
    if not dataset_name:
        return {
            "success": False,
            "error": "Unable to resolve dataset name. The dataset_name must be a non-empty string.",
        }

    dataset_name_error = _validate_filesystem_name(name=dataset_name, field_label="dataset_name")
    if dataset_name_error is not None:
        return {"success": False, "error": dataset_name_error}

    if not output_roots:
        return {
            "success": False,
            "error": "Unable to resolve dataset name. At least one output root is required.",
        }

    if not specifier:
        resolved_paths = [Path(path) for path in output_roots]
        if len(resolved_paths) == 1:
            specifier = resolved_paths[0].parent.name
        else:
            # The common-path resolver rejects a set of paths that share no root, which covers a list mixing absolute
            # and relative entries and, on Windows, absolute paths on different drives.
            try:
                common = Path(commonpath(paths=resolved_paths))
            except ValueError as error:
                return {
                    "success": False,
                    "error": (
                        f"Unable to resolve dataset name. The output roots must share a common parent directory, "
                        f"but deriving one from {list(output_roots)} failed: {error}."
                    ),
                }
            specifier = common.name

        if not specifier:
            return {
                "success": False,
                "error": "Unable to resolve dataset name. Could not derive a specifier from the output roots.",
            }

    specifier_error = _validate_filesystem_name(name=specifier, field_label="specifier")
    if specifier_error is not None:
        return {"success": False, "error": specifier_error}

    qualified_name = f"{specifier}_{dataset_name}".lower()

    return {
        "success": True,
        "dataset_name": qualified_name,
        "base_name": dataset_name.lower(),
        "specifier": specifier.lower(),
    }


@mcp.tool()
def read_config_file_tool(file_path: str) -> dict[str, str | bool | list[str] | dict[str, object] | None]:
    """Reads a YAML configuration file and returns its raw contents as a dictionary.

    Notes:
        Accepts any YAML mapping regardless of the current cindra configuration schema, which makes it suitable for
        reading legacy cindra configurations, or any other YAML files that need to be inspected or converted.

    Args:
        file_path: The absolute path to the YAML configuration file to read.

    Returns:
        On success, contains the resolved 'file_path', the 'detected_pipeline_type', the top-level 'sections',
        and the raw 'parameters'. On failure, contains an 'error' describing the issue. Both cases include a
        'success' flag.
    """
    path = Path(file_path)

    if not path.exists():
        return {
            "success": False,
            "error": f"Unable to read configuration file. The file does not exist: {file_path}",
        }

    if path.suffix not in (".yaml", ".yml"):
        return {
            "success": False,
            "error": (
                f"Unable to read configuration file. Expected a '.yaml' or '.yml' file, but received: {file_path}"
            ),
        }

    try:
        with path.open() as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as error:
        return {
            "success": False,
            "error": f"Unable to parse YAML file at '{file_path}': {error}",
        }

    if data is None:
        return {"success": False, "error": f"Unable to read configuration file. The file is empty: {file_path}"}

    if not isinstance(data, dict):
        return {
            "success": False,
            "error": (
                f"Unable to read configuration file. Expected a YAML mapping at the top level, but found "
                f"{type(data).__name__}: {file_path}"
            ),
        }

    raw_pipeline_type = data.get("pipeline_type")
    detected_type: str | None = None
    if raw_pipeline_type in ("single-recording", "multi-recording"):
        detected_type = raw_pipeline_type

    return {
        "success": True,
        "file_path": str(path),
        "detected_pipeline_type": detected_type,
        "sections": list(data.keys()),
        "parameters": data,
    }


@mcp.tool()
def validate_config_file_tool(file_path: str) -> dict[str, str | bool | list[str] | dict[str, dict[str, object]]]:
    """Validates a cindra configuration YAML file and reports any problems found.

    Loads the file through the appropriate configuration dataclass, checks parameter values against known constraints,
    and identifies parameters that differ from their defaults.

    Args:
        file_path: The absolute path to the cindra configuration YAML file to validate.

    Returns:
        On success, contains the resolved 'file_path', 'pipeline_type', and overall 'valid' status, plus 'errors',
        'warnings', and 'non_default_parameters' when non-empty (each of these three keys is omitted when empty).
        'errors' and 'warnings' are lists of human-readable strings, and 'non_default_parameters' maps each changed
        'section.parameter' dotted path to a {'current', 'default'} value pair. A 'success' value of True only means
        the tool ran, so gate downstream steps on the 'valid' field, which is False whenever 'errors' is non-empty.
        On failure, contains an 'error' describing the issue. Both cases include a 'success' flag.
    """
    path = Path(file_path)

    if not path.exists():
        return {
            "success": False,
            "error": f"Unable to validate configuration file. The file does not exist: {file_path}",
        }

    if path.suffix not in (".yaml", ".yml"):
        return {
            "success": False,
            "error": (
                f"Unable to validate configuration file. Expected a '.yaml' or '.yml' file, but received: {file_path}"
            ),
        }

    # Parses the raw YAML to detect pipeline type before attempting dataclass deserialization.
    try:
        with path.open() as file:
            raw_data = yaml.safe_load(file)
    except yaml.YAMLError as error:
        return {
            "success": False,
            "error": f"Unable to parse YAML file at '{file_path}': {error}",
        }

    if not isinstance(raw_data, dict):
        return {
            "success": False,
            "error": (
                f"Unable to validate configuration file. Expected a YAML mapping at the top level, but found "
                f"{type(raw_data).__name__ if raw_data is not None else 'empty file'}: {file_path}"
            ),
        }

    raw_pipeline_type = raw_data.get("pipeline_type")
    if raw_pipeline_type not in ("single-recording", "multi-recording"):
        return {
            "success": False,
            "error": (
                f"Unable to validate configuration file. The 'pipeline_type' field is missing or unrecognized "
                f"(found: {raw_pipeline_type!r}). Expected 'single-recording' or 'multi-recording'."
            ),
        }

    # Loads the configuration through the appropriate dataclass to catch deserialization errors.
    config: SingleRecordingConfiguration | MultiRecordingConfiguration
    default: SingleRecordingConfiguration | MultiRecordingConfiguration
    try:
        if raw_pipeline_type == "single-recording":
            config = SingleRecordingConfiguration.load(file_path=path)
            default = SingleRecordingConfiguration()
            errors, warnings = _validate_single_recording(config=config)
        else:
            config = MultiRecordingConfiguration.load(file_path=path)
            default = MultiRecordingConfiguration()
            errors, warnings = _validate_multi_recording(config=config)
    except Exception as error:
        return {
            "success": False,
            "error": (
                f"Unable to deserialize {raw_pipeline_type} configuration from '{file_path}': "
                f"{type(error).__name__}: {error}"
            ),
        }

    non_defaults = _identify_non_default_parameters(config=config, default=default)

    result: dict[str, str | bool | list[str] | dict[str, dict[str, object]]] = {
        "success": True,
        "file_path": str(path),
        "pipeline_type": raw_pipeline_type,
        "valid": not errors,
    }

    if errors:
        result["errors"] = errors
    if warnings:
        result["warnings"] = warnings
    if non_defaults:
        result["non_default_parameters"] = non_defaults

    return result


@mcp.tool()
def set_config_values_tool(file_path: str, values: dict[str, object]) -> dict[str, object]:
    """Writes new values for the named tunable parameters into an existing cindra configuration file.

    Loads the file through the configuration dataclass matching its pipeline type, applies every requested value, and
    writes the whole document back atomically. Each parameter is addressed by the same 'section.parameter' dotted path
    validate_config_file_tool reports under 'non_default_parameters', so the two tools speak one vocabulary.

    Notes:
        The pipeline reads its configuration from disk when it dispatches a job, so a value written while a batch is
        executing reaches the jobs of that batch which have not started yet. Write configuration values before
        preparing and dispatching a batch, and never against a configuration whose jobs are currently running.

        Every requested value is resolved before any of them is applied, so a call naming one unknown section, one
        unknown parameter, or one value of the wrong type writes nothing and reports every rejection it found at once.
        A value is supplied in the form the configuration document carries it, so a path is a string, an enumeration
        is its raw value, and a tuple is a list. An integer is accepted for a parameter typed as a floating point
        number, and no other type substitution is.

        The document is rewritten from the configuration dataclass, so a key the current schema does not declare is
        dropped and the surviving keys follow the dataclass field order.

    Args:
        file_path: The absolute path to the cindra configuration YAML file to modify.
        values: Maps each 'section.parameter' dotted path to the value written to that parameter.

    Returns:
        On success, contains the resolved 'file_path' and the 'changed' map pairing every requested dotted path with its
        'previous' and 'current' values. A successful response also carries the 'valid' status the written file
        validates to, plus the 'errors' and 'warnings' of that validation when either is non-empty. A 'success' value of
        True only means the file was written, so gate downstream steps on 'valid'. On failure, contains an 'error'
        describing the issue, joined by an 'errors' list naming every rejected entry when the failure is a rejected
        value. Both cases include a 'success' flag.
    """
    path = Path(file_path)

    if not path.exists():
        return {
            "success": False,
            "error": f"Unable to set configuration values. The file does not exist: {file_path}",
        }

    if path.suffix not in (".yaml", ".yml"):
        return {
            "success": False,
            "error": (
                f"Unable to set configuration values. Expected a '.yaml' or '.yml' file, but received: {file_path}"
            ),
        }

    if not values:
        return {
            "success": False,
            "error": "Unable to set configuration values. At least one 'section.parameter' entry is required.",
        }

    try:
        with path.open() as file:
            raw_data = yaml.safe_load(file)
    except yaml.YAMLError as error:
        return {
            "success": False,
            "error": f"Unable to parse YAML file at '{file_path}': {error}",
        }

    raw_pipeline_type = raw_data.get("pipeline_type") if isinstance(raw_data, dict) else None
    if raw_pipeline_type not in ("single-recording", "multi-recording"):
        return {
            "success": False,
            "error": (
                f"Unable to set configuration values. The 'pipeline_type' field is missing or unrecognized "
                f"(found: {raw_pipeline_type!r}). Expected 'single-recording' or 'multi-recording'."
            ),
        }

    configuration: SingleRecordingConfiguration | MultiRecordingConfiguration
    try:
        if raw_pipeline_type == "single-recording":
            configuration = SingleRecordingConfiguration.load(file_path=path)
        else:
            configuration = MultiRecordingConfiguration.load(file_path=path)
    except Exception as error:
        return {
            "success": False,
            "error": (
                f"Unable to deserialize {raw_pipeline_type} configuration from '{file_path}': "
                f"{type(error).__name__}: {error}"
            ),
        }

    # Resolves every entry against the schema before writing any of them, so a rejected entry leaves the file as it
    # was rather than applying the entries that happened to precede it.
    assignments: list[tuple[str, object, str, object]] = []
    rejections: list[str] = []
    for dotted_path, value in values.items():
        resolution = _resolve_parameter_assignment(configuration=configuration, dotted_path=dotted_path, value=value)
        if isinstance(resolution, str):
            rejections.append(resolution)
            continue
        owner, field_name, resolved_value = resolution
        assignments.append((dotted_path, owner, field_name, resolved_value))

    if rejections:
        return {
            "success": False,
            "error": (
                f"Unable to set configuration values in '{file_path}'. {len(rejections)} of {len(values)} requested "
                f"entries were rejected, and the file was left unchanged."
            ),
            "errors": rejections,
        }

    changed: dict[str, dict[str, object]] = {}
    for dotted_path, owner, field_name, resolved_value in assignments:
        previous_value = getattr(owner, field_name)
        setattr(owner, field_name, resolved_value)
        changed[dotted_path] = {
            "previous": _convert_to_json_compatible(value=previous_value),
            "current": _convert_to_json_compatible(value=resolved_value),
        }

    # The configuration writer publishes the document through a temporary file, so a reader of the path observes
    # either the previous configuration or the complete new one.
    configuration.save(file_path=path)

    written: SingleRecordingConfiguration | MultiRecordingConfiguration
    try:
        if raw_pipeline_type == "single-recording":
            written = SingleRecordingConfiguration.load(file_path=path)
            validation_errors, validation_warnings = _validate_single_recording(config=written)
        else:
            written = MultiRecordingConfiguration.load(file_path=path)
            validation_errors, validation_warnings = _validate_multi_recording(config=written)
    except Exception as error:
        return {
            "success": False,
            "error": (
                f"Unable to re-read the {raw_pipeline_type} configuration written to '{file_path}': "
                f"{type(error).__name__}: {error}"
            ),
        }

    result: dict[str, object] = {
        "success": True,
        "file_path": str(path),
        "changed": changed,
        "valid": not validation_errors,
    }

    if validation_errors:
        result["errors"] = validation_errors
    if validation_warnings:
        result["warnings"] = validation_warnings

    return result


def _discover_marker_parents(root_path: Path, marker_name: str) -> list[Path]:
    """Discovers the directories owning every marker file with the target name under the root directory.

    Notes:
        The ataraxis marker discoverer refuses a subtree it cannot read rather than narrowing its result to the
        readable part, which is the right answer for a path the pipeline owns and the wrong one for a root the caller
        chose. A denial therefore falls back to the tolerant recursive glob, so an unreadable sibling directory lowers
        the candidate count instead of failing the whole discovery.

    Args:
        root_path: The root directory whose tree is searched.
        marker_name: The exact filename every discovered marker carries.

    Returns:
        The parent directory of every discovered marker file.
    """
    try:
        return discover_marker_roots(directory=root_path, marker_name=marker_name)
    except OSError:
        return [marker_file.parent for marker_file in root_path.rglob(marker_name)]


def _pair_marker_parents_with_roots(marker_parents: list[Path]) -> list[tuple[Path, Path]]:
    """Pairs every discovered marker directory with the recording root that owns it.

    Notes:
        The root resolver deduplicates the directories it receives and reports the roots in its own grouping order, so
        each marker directory is matched back to its root by ancestry rather than by position. A directory matching
        several roots takes the deepest of them, and a directory matching none stands as its own root.

    Args:
        marker_parents: The parent directory of every discovered marker file.

    Returns:
        The (recording root, marker directory) pair of every unique marker directory, ordered by recording root and
        then by marker directory.
    """
    if not marker_parents:
        return []

    roots = resolve_recording_roots(paths=marker_parents)

    pairs: list[tuple[Path, Path]] = []
    for marker_parent in dict.fromkeys(marker_parents):
        ancestors = [root for root in roots if root == marker_parent or root in marker_parent.parents]
        recording_root = max(ancestors, key=lambda root: len(root.parts)) if ancestors else marker_parent
        pairs.append((recording_root, marker_parent))

    return natsorted(pairs, key=lambda pair: (str(pair[0]), str(pair[1])))


def _resolve_marker_output_root(recording_root: Path, marker_parent: Path) -> Path:
    """Resolves the pipeline output root that owns a discovered combined metadata marker.

    Args:
        recording_root: The recording root the marker directory resolved to.
        marker_parent: The directory holding the discovered marker file.

    Returns:
        The parent of the cindra output directory holding the marker, which falls back to the recording root for a
        marker found outside such a directory.
    """
    if marker_parent.name == OUTPUT_DIRECTORY_NAME:
        return marker_parent.parent

    return recording_root


def _convert_to_json_compatible(value: object) -> object:
    """Converts a Python value to a JSON-compatible type for MCP tool output.

    Args:
        value: The Python value to convert. Handles Path, Enum, and tuple types.

    Returns:
        The JSON-compatible representation of the value, or the original value if no conversion is needed.
    """
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_convert_to_json_compatible(value=item) for item in value]
    return value


def _identify_non_default_parameters(config: object, default: object, prefix: str = "") -> dict[str, dict[str, object]]:
    """Compares a loaded configuration against its default instance and reports the parameters that differ.

    Args:
        config: The loaded configuration dataclass instance to compare.
        default: The default configuration dataclass instance to compare against.
        prefix: The dotted path prefix for nested dataclass fields. Defaults to an empty string.

    Returns:
        A dictionary mapping dotted parameter paths to dictionaries containing 'current' and 'default' values for
        each parameter that differs from its default.
    """
    differences: dict[str, dict[str, object]] = {}

    for field in dataclass_fields(config):  # type: ignore[arg-type]
        if not field.init:
            continue

        current_value = getattr(config, field.name)
        default_value = getattr(default, field.name)
        full_path = f"{prefix}.{field.name}" if prefix else field.name

        if is_dataclass(current_value) and is_dataclass(default_value):
            nested = _identify_non_default_parameters(config=current_value, default=default_value, prefix=full_path)
            differences.update(nested)
        elif current_value != default_value:
            differences[full_path] = {
                "current": _convert_to_json_compatible(value=current_value),
                "default": _convert_to_json_compatible(value=default_value),
            }

    return differences


def _validate_single_recording(
    config: SingleRecordingConfiguration,
) -> tuple[list[str], list[str]]:
    """Validates a single-recording configuration and returns lists of errors and warnings.

    Args:
        config: The single-recording configuration to validate.

    Returns:
        A tuple of two lists: the first containing error messages for invalid parameters, and the second containing
        warning messages for potentially problematic parameter values.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if config.main.tau <= 0:
        errors.append(f"main.tau must be positive (current: {config.main.tau}).")
    if not config.main.first_channel_functional and not config.main.second_channel_functional:
        errors.append(
            "Both main.first_channel_functional and main.second_channel_functional are False. No functional "
            "channel available."
        )

    if config.file_io.data_path is not None:
        warnings.append("file_io.data_path is set (pipeline-set parameter, overwritten at runtime).")
    if config.file_io.output_path is not None:
        warnings.append("file_io.output_path is set (pipeline-set parameter, overwritten at runtime).")

    if not config.main.two_channels and not config.registration.align_by_first_channel:
        warnings.append(
            "registration.align_by_first_channel is False but main.two_channels is False. No second channel "
            "available for alignment."
        )
    if config.registration.reference_frame_count <= 0:
        errors.append(
            f"registration.reference_frame_count must be positive "
            f"(current: {config.registration.reference_frame_count})."
        )
    if config.registration.batch_size <= 0:
        errors.append(f"registration.batch_size must be positive (current: {config.registration.batch_size}).")
    if config.registration.maximum_offset_fraction <= 0 or config.registration.maximum_offset_fraction > 1:
        errors.append(
            f"registration.maximum_offset_fraction must be in (0, 1] "
            f"(current: {config.registration.maximum_offset_fraction})."
        )
    if config.registration.spatial_smoothing_sigma < 0:
        errors.append(
            f"registration.spatial_smoothing_sigma must be non-negative "
            f"(current: {config.registration.spatial_smoothing_sigma})."
        )
    if config.registration.temporal_smoothing_sigma < 0:
        errors.append(
            f"registration.temporal_smoothing_sigma must be non-negative "
            f"(current: {config.registration.temporal_smoothing_sigma})."
        )
    if config.registration.bad_frame_threshold <= 0:
        errors.append(
            f"registration.bad_frame_threshold must be positive (current: {config.registration.bad_frame_threshold})."
        )
    if config.registration.registration_metric_principal_components < 0:
        errors.append(
            f"registration.registration_metric_principal_components must be non-negative "
            f"(current: {config.registration.registration_metric_principal_components})."
        )
    if config.registration.gpu_batch_size < 0:
        errors.append(
            f"registration.gpu_batch_size must be non-negative (current: {config.registration.gpu_batch_size})."
        )

    if config.one_photon_registration.enabled:
        if config.one_photon_registration.spatial_highpass_window <= 0:
            errors.append(
                f"one_photon_registration.spatial_highpass_window must be positive when one-photon registration "
                f"is enabled (current: {config.one_photon_registration.spatial_highpass_window})."
            )
        elif config.one_photon_registration.spatial_highpass_window % 2:
            # The spatial smoothing kernel the high-pass filter subtracts is a box filter centered on each pixel, so
            # the registration stage rejects an odd window.
            errors.append(
                f"one_photon_registration.spatial_highpass_window must be an even integer when one-photon "
                f"registration is enabled (current: {config.one_photon_registration.spatial_highpass_window})."
            )
        if config.one_photon_registration.pre_smoothing_sigma < 0:
            errors.append(
                f"one_photon_registration.pre_smoothing_sigma must be non-negative when one-photon registration "
                f"is enabled (current: {config.one_photon_registration.pre_smoothing_sigma})."
            )
        elif int(config.one_photon_registration.pre_smoothing_sigma) % 2:
            # The registration stage truncates this field to an integer box-filter window, which is rejected when odd.
            errors.append(
                f"one_photon_registration.pre_smoothing_sigma must truncate to an even filter window when one-photon "
                f"registration is enabled (current: {config.one_photon_registration.pre_smoothing_sigma})."
            )
        if config.one_photon_registration.edge_taper_pixels < 0:
            errors.append(
                f"one_photon_registration.edge_taper_pixels must be non-negative when one-photon registration "
                f"is enabled (current: {config.one_photon_registration.edge_taper_pixels})."
            )

    if config.nonrigid_registration.enabled:
        if config.nonrigid_registration.signal_to_noise_threshold <= 0:
            errors.append(
                f"nonrigid_registration.signal_to_noise_threshold must be positive when nonrigid registration is "
                f"enabled (current: {config.nonrigid_registration.signal_to_noise_threshold})."
            )
        if config.nonrigid_registration.maximum_block_offset <= 0:
            errors.append(
                f"nonrigid_registration.maximum_block_offset must be positive when nonrigid registration is "
                f"enabled (current: {config.nonrigid_registration.maximum_block_offset})."
            )
        if any(dimension <= 0 for dimension in config.nonrigid_registration.block_size):
            errors.append(
                f"nonrigid_registration.block_size dimensions must be positive "
                f"(current: {list(config.nonrigid_registration.block_size)})."
            )

    if config.roi_detection.enabled:
        if not 0 <= config.roi_detection.preclassification_threshold <= 1:
            errors.append(
                f"roi_detection.preclassification_threshold must be in [0, 1] "
                f"(current: {config.roi_detection.preclassification_threshold})."
            )
        if config.roi_detection.threshold_scaling <= 0:
            errors.append(
                f"roi_detection.threshold_scaling must be positive (current: {config.roi_detection.threshold_scaling})."
            )
        if config.roi_detection.spatial_highpass_window <= 0:
            errors.append(
                f"roi_detection.spatial_highpass_window must be positive "
                f"(current: {config.roi_detection.spatial_highpass_window})."
            )
        if not 0 < config.roi_detection.maximum_overlap <= 1:
            errors.append(
                f"roi_detection.maximum_overlap must be in (0, 1] (current: {config.roi_detection.maximum_overlap})."
            )
        if config.roi_detection.temporal_highpass_window <= 0:
            errors.append(
                f"roi_detection.temporal_highpass_window must be positive "
                f"(current: {config.roi_detection.temporal_highpass_window})."
            )
        if config.roi_detection.maximum_iterations <= 0:
            errors.append(
                f"roi_detection.maximum_iterations must be positive "
                f"(current: {config.roi_detection.maximum_iterations})."
            )
        if config.roi_detection.maximum_binned_frames <= 0:
            errors.append(
                f"roi_detection.maximum_binned_frames must be positive "
                f"(current: {config.roi_detection.maximum_binned_frames})."
            )

    if config.signal_extraction.minimum_neuropil_pixels <= 0:
        errors.append(
            f"signal_extraction.minimum_neuropil_pixels must be positive "
            f"(current: {config.signal_extraction.minimum_neuropil_pixels})."
        )
    if config.signal_extraction.inner_neuropil_border_radius < 0:
        errors.append(
            f"signal_extraction.inner_neuropil_border_radius must be non-negative "
            f"(current: {config.signal_extraction.inner_neuropil_border_radius})."
        )
    if not 0 <= config.signal_extraction.cell_probability_percentile <= _MAXIMUM_PERCENTAGE:
        errors.append(
            f"signal_extraction.cell_probability_percentile must be in [0, {_MAXIMUM_PERCENTAGE}] "
            f"(current: {config.signal_extraction.cell_probability_percentile})."
        )
    if not 0 <= config.signal_extraction.classification_threshold <= 1:
        errors.append(
            f"signal_extraction.classification_threshold must be in [0, 1] "
            f"(current: {config.signal_extraction.classification_threshold})."
        )
    if config.signal_extraction.batch_size <= 0:
        errors.append(
            f"signal_extraction.batch_size must be positive (current: {config.signal_extraction.batch_size})."
        )
    if not 0 <= config.signal_extraction.colocalization_threshold <= 1:
        errors.append(
            f"signal_extraction.colocalization_threshold must be in [0, 1] "
            f"(current: {config.signal_extraction.colocalization_threshold})."
        )

    if not 0 <= config.spike_deconvolution.neuropil_coefficient <= 1:
        warnings.append(
            f"spike_deconvolution.neuropil_coefficient is outside the typical [0, 1] range "
            f"(current: {config.spike_deconvolution.neuropil_coefficient})."
        )
    if config.spike_deconvolution.baseline_window <= 0:
        errors.append(
            f"spike_deconvolution.baseline_window must be positive "
            f"(current: {config.spike_deconvolution.baseline_window})."
        )
    if config.spike_deconvolution.baseline_sigma < 0:
        errors.append(
            f"spike_deconvolution.baseline_sigma must be non-negative "
            f"(current: {config.spike_deconvolution.baseline_sigma})."
        )
    if not 0 <= config.spike_deconvolution.baseline_percentile <= _MAXIMUM_PERCENTAGE:
        errors.append(
            f"spike_deconvolution.baseline_percentile must be in [0, {_MAXIMUM_PERCENTAGE}] "
            f"(current: {config.spike_deconvolution.baseline_percentile})."
        )
    valid_baseline_methods = {member.value for member in BaselineMethod}
    if str(config.spike_deconvolution.baseline_method) not in valid_baseline_methods:
        errors.append(
            f"spike_deconvolution.baseline_method must be one of {sorted(valid_baseline_methods)} "
            f"(current: {config.spike_deconvolution.baseline_method})."
        )

    return errors, warnings


def _validate_multi_recording(
    config: MultiRecordingConfiguration,
) -> tuple[list[str], list[str]]:
    """Validates a multi-recording configuration and returns lists of errors and warnings.

    Args:
        config: The multi-recording configuration to validate.

    Returns:
        A tuple of two lists: the first containing error messages for invalid parameters, and the second containing
        warning messages for potentially problematic parameter values.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not config.recording_io.dataset_name:
        errors.append("recording_io.dataset_name must be a non-empty string.")
    else:
        dataset_name_error = _validate_filesystem_name(
            name=config.recording_io.dataset_name,
            field_label="recording_io.dataset_name",
            action="validate configuration file",
        )
        if dataset_name_error is not None:
            errors.append(dataset_name_error)
    if config.recording_io.recording_directories:
        warnings.append("recording_io.recording_directories is set (pipeline-set parameter, overwritten at runtime).")

    if not 0 <= config.roi_selection.probability_threshold <= 1:
        errors.append(
            f"roi_selection.probability_threshold must be in [0, 1] "
            f"(current: {config.roi_selection.probability_threshold})."
        )
    if config.roi_selection.maximum_size <= 0:
        errors.append(f"roi_selection.maximum_size must be positive (current: {config.roi_selection.maximum_size}).")
    if config.roi_selection.mroi_region_margin < 0:
        errors.append(
            f"roi_selection.mroi_region_margin must be non-negative "
            f"(current: {config.roi_selection.mroi_region_margin})."
        )
    if (
        config.roi_selection.probability_threshold_channel_2 is not None
        and not 0 <= config.roi_selection.probability_threshold_channel_2 <= 1
    ):
        errors.append(
            f"roi_selection.probability_threshold_channel_2 must be in [0, 1] "
            f"(current: {config.roi_selection.probability_threshold_channel_2})."
        )
    if config.roi_selection.maximum_size_channel_2 is not None and config.roi_selection.maximum_size_channel_2 <= 0:
        errors.append(
            f"roi_selection.maximum_size_channel_2 must be positive "
            f"(current: {config.roi_selection.maximum_size_channel_2})."
        )
    if (
        config.roi_selection.mroi_region_margin_channel_2 is not None
        and config.roi_selection.mroi_region_margin_channel_2 < 0
    ):
        errors.append(
            f"roi_selection.mroi_region_margin_channel_2 must be non-negative "
            f"(current: {config.roi_selection.mroi_region_margin_channel_2})."
        )

    if not 0 < config.diffeomorphic_registration.grid_sampling_factor <= 1:
        errors.append(
            f"diffeomorphic_registration.grid_sampling_factor must be in (0, 1] "
            f"(current: {config.diffeomorphic_registration.grid_sampling_factor})."
        )
    if config.diffeomorphic_registration.final_grid_sampling <= 0:
        errors.append(
            f"diffeomorphic_registration.final_grid_sampling must be positive "
            f"(current: {config.diffeomorphic_registration.final_grid_sampling})."
        )
    if config.diffeomorphic_registration.scale_sampling <= 0:
        errors.append(
            f"diffeomorphic_registration.scale_sampling must be positive "
            f"(current: {config.diffeomorphic_registration.scale_sampling})."
        )
    if config.diffeomorphic_registration.speed_factor <= 0:
        errors.append(
            f"diffeomorphic_registration.speed_factor must be positive "
            f"(current: {config.diffeomorphic_registration.speed_factor})."
        )
    elif not 1 <= config.diffeomorphic_registration.speed_factor <= _MAXIMUM_SPEED_FACTOR:
        warnings.append(
            f"diffeomorphic_registration.speed_factor is outside the typical 1-{_MAXIMUM_SPEED_FACTOR} range "
            f"(current: {config.diffeomorphic_registration.speed_factor})."
        )
    valid_image_types = {member.value for member in ReferenceImageType}
    if str(config.diffeomorphic_registration.image_type) not in valid_image_types:
        errors.append(
            f"diffeomorphic_registration.image_type must be one of {sorted(valid_image_types)} "
            f"(current: {config.diffeomorphic_registration.image_type})."
        )

    if not 0 <= config.roi_tracking.threshold <= 1:
        errors.append(f"roi_tracking.threshold must be in [0, 1] (current: {config.roi_tracking.threshold}).")
    if not 0 <= config.roi_tracking.mask_prevalence <= _MAXIMUM_PERCENTAGE:
        errors.append(
            f"roi_tracking.mask_prevalence must be in [0, {_MAXIMUM_PERCENTAGE}] "
            f"(current: {config.roi_tracking.mask_prevalence})."
        )
    if not 0 <= config.roi_tracking.pixel_prevalence <= _MAXIMUM_PERCENTAGE:
        errors.append(
            f"roi_tracking.pixel_prevalence must be in [0, {_MAXIMUM_PERCENTAGE}] "
            f"(current: {config.roi_tracking.pixel_prevalence})."
        )
    if any(dimension <= 0 for dimension in config.roi_tracking.step_sizes):
        errors.append(
            f"roi_tracking.step_sizes dimensions must be positive (current: {list(config.roi_tracking.step_sizes)})."
        )
    if config.roi_tracking.bin_size <= 0:
        errors.append(f"roi_tracking.bin_size must be positive (current: {config.roi_tracking.bin_size}).")
    if config.roi_tracking.maximum_distance <= 0:
        errors.append(
            f"roi_tracking.maximum_distance must be positive (current: {config.roi_tracking.maximum_distance})."
        )
    if config.roi_tracking.minimum_size <= 0:
        errors.append(f"roi_tracking.minimum_size must be positive (current: {config.roi_tracking.minimum_size}).")

    if config.signal_extraction.minimum_neuropil_pixels <= 0:
        errors.append(
            f"signal_extraction.minimum_neuropil_pixels must be positive "
            f"(current: {config.signal_extraction.minimum_neuropil_pixels})."
        )
    if config.signal_extraction.inner_neuropil_border_radius < 0:
        errors.append(
            f"signal_extraction.inner_neuropil_border_radius must be non-negative "
            f"(current: {config.signal_extraction.inner_neuropil_border_radius})."
        )
    if not 0 <= config.signal_extraction.cell_probability_percentile <= _MAXIMUM_PERCENTAGE:
        errors.append(
            f"signal_extraction.cell_probability_percentile must be in [0, {_MAXIMUM_PERCENTAGE}] "
            f"(current: {config.signal_extraction.cell_probability_percentile})."
        )
    if not 0 <= config.signal_extraction.classification_threshold <= 1:
        errors.append(
            f"signal_extraction.classification_threshold must be in [0, 1] "
            f"(current: {config.signal_extraction.classification_threshold})."
        )
    if config.signal_extraction.batch_size <= 0:
        errors.append(
            f"signal_extraction.batch_size must be positive (current: {config.signal_extraction.batch_size})."
        )
    if not 0 <= config.signal_extraction.colocalization_threshold <= 1:
        errors.append(
            f"signal_extraction.colocalization_threshold must be in [0, 1] "
            f"(current: {config.signal_extraction.colocalization_threshold})."
        )

    if not 0 <= config.spike_deconvolution.neuropil_coefficient <= 1:
        warnings.append(
            f"spike_deconvolution.neuropil_coefficient is outside the typical [0, 1] range "
            f"(current: {config.spike_deconvolution.neuropil_coefficient})."
        )
    if config.spike_deconvolution.baseline_window <= 0:
        errors.append(
            f"spike_deconvolution.baseline_window must be positive "
            f"(current: {config.spike_deconvolution.baseline_window})."
        )
    if config.spike_deconvolution.baseline_sigma < 0:
        errors.append(
            f"spike_deconvolution.baseline_sigma must be non-negative "
            f"(current: {config.spike_deconvolution.baseline_sigma})."
        )
    if not 0 <= config.spike_deconvolution.baseline_percentile <= _MAXIMUM_PERCENTAGE:
        errors.append(
            f"spike_deconvolution.baseline_percentile must be in [0, {_MAXIMUM_PERCENTAGE}] "
            f"(current: {config.spike_deconvolution.baseline_percentile})."
        )

    return errors, warnings


def _validate_filesystem_name(name: str, field_label: str, action: str = "resolve dataset name") -> str | None:
    """Validates that a name is safe for use as a filesystem directory name.

    Rejects names containing characters that are invalid in directory names on common filesystems, names that consist
    entirely of whitespace, and reserved names like '.' and '..'.

    Args:
        name: The name string to validate.
        field_label: The label of the field being validated, used in error messages.
        action: The action phrase used in the "Unable to {action}." error prefix. Defaults to "resolve dataset name".

    Returns:
        An error message string if the name is invalid, or None if the name is safe.
    """
    if not name.strip():
        return f"Unable to {action}. The {field_label} must not be empty or consist entirely of whitespace."

    if name in (".", ".."):
        return f"Unable to {action}. The {field_label} must not be '{name}'."

    found = sorted({character for character in name if character in _FORBIDDEN_FILESYSTEM_CHARACTERS})
    if found:
        display = ", ".join(repr(character) for character in found)
        return f"Unable to {action}. The {field_label} contains filesystem-unsafe characters: {display}."

    control_characters = [character for character in name if ord(character) < _MAXIMUM_CONTROL_CHARACTER_ORDINAL]
    if control_characters:
        return f"Unable to {action}. The {field_label} contains control characters."

    return None


def _resolve_parameter_assignment(
    configuration: object, dotted_path: str, value: object
) -> tuple[object, str, object] | str:
    """Resolves one dotted configuration path and the value written to it against the configuration schema.

    Args:
        configuration: The loaded configuration dataclass instance the path is resolved against.
        dotted_path: The 'section.parameter' path naming the parameter to write.
        value: The value the caller asked to write to the parameter.

    Returns:
        The section instance owning the parameter, the parameter's field name, and the value coerced to the field's
        annotated type. Returns an error message string instead when the path names no writable parameter or when the
        value does not match the parameter's annotation.
    """
    owner: object = configuration
    parts = dotted_path.split(".")

    for index, part in enumerate(parts[:-1]):
        writable = _resolve_writable_fields(instance=owner)
        traversed = ".".join(parts[: index + 1])
        if part not in writable:
            return (
                f"Unable to set '{dotted_path}'. The configuration holds no section named '{traversed}'. Available "
                f"names: {sorted(writable)}."
            )
        section = getattr(owner, part)
        if not is_dataclass(section):
            return f"Unable to set '{dotted_path}'. The path component '{traversed}' names a parameter, not a section."
        owner = section

    name = parts[-1]
    writable = _resolve_writable_fields(instance=owner)
    if name not in writable:
        return (
            f"Unable to set '{dotted_path}'. The configuration holds no writable parameter named '{name}'. Available "
            f"names: {sorted(writable)}."
        )

    if is_dataclass(getattr(owner, name)):
        return (
            f"Unable to set '{dotted_path}'. The path names a configuration section, so address the parameters it "
            f"holds one dotted path at a time."
        )

    annotation = writable[name]
    coerced_value, matched = _coerce_parameter_value(value=value, annotation=annotation)
    if not matched:
        return (
            f"Unable to set '{dotted_path}'. The parameter is typed as {_describe_annotation(annotation=annotation)}, "
            f"but received {type(value).__name__}: {value!r}."
        )

    return owner, name, coerced_value


def _resolve_writable_fields(instance: object) -> dict[str, object]:
    """Resolves the fields of a configuration dataclass instance the constructor accepts a value for.

    Notes:
        The configuration modules defer their annotations, so each annotation is resolved against the module that
        defines the dataclass rather than read from the field descriptor, which carries it as a string.

    Args:
        instance: The configuration dataclass instance whose fields are resolved.

    Returns:
        A dictionary mapping the name of every writable field to that field's resolved annotation.
    """
    annotations = get_type_hints(type(instance))
    fields = dataclass_fields(instance)  # type: ignore[arg-type]

    return {field.name: annotations[field.name] for field in fields if field.init}


def _coerce_parameter_value(value: object, annotation: object) -> tuple[object, bool]:
    """Coerces one caller-supplied value to the type the target configuration field is annotated with.

    Notes:
        A value arrives in the form the configuration document carries it, so a path arrives as a string, an
        enumeration as its raw value, and a tuple as a list. An integer is accepted for a field annotated as a
        floating point number, which is the only widening performed here. A boolean is refused for a numeric field
        despite being an integer subclass, because the two carry different meanings in every configuration section.

    Args:
        value: The value the caller asked to write.
        annotation: The resolved annotation of the field the value is written to.

    Returns:
        The coerced value paired with the flag reporting whether the value matches the annotation. The coerced value
        is None whenever that flag is False.
    """
    origin = get_origin(annotation)

    if origin is UnionType:
        for member in get_args(annotation):
            coerced_value, matched = _coerce_parameter_value(value=value, annotation=member)
            if matched:
                return coerced_value, True
        return None, False

    if origin is tuple:
        return _coerce_tuple_value(value=value, annotation=annotation)

    if annotation is NoneType:
        return None, value is None

    if annotation is bool:
        return (value, True) if isinstance(value, bool) else (None, False)

    if annotation is int:
        return (value, True) if isinstance(value, int) and not isinstance(value, bool) else (None, False)

    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None, False
        return float(value), True

    if annotation is str:
        return (value, True) if isinstance(value, str) else (None, False)

    if annotation is Path:
        return (Path(value), True) if isinstance(value, (str, Path)) else (None, False)

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        try:
            return annotation(value), True
        except TypeError, ValueError:
            return None, False

    return None, False


def _coerce_tuple_value(value: object, annotation: object) -> tuple[object, bool]:
    """Coerces one caller-supplied sequence to the tuple type the target configuration field is annotated with.

    Args:
        value: The value the caller asked to write, which matches only when it is a list or a tuple whose length and
            element types the annotation accepts.
        annotation: The resolved tuple annotation of the field the value is written to.

    Returns:
        The coerced tuple paired with the flag reporting whether the value matches the annotation. The coerced value
        is None whenever that flag is False.
    """
    if not isinstance(value, (list, tuple)):
        return None, False

    # A variadic annotation constrains every element to one type, while a fixed-length one constrains the sequence's
    # length as well.
    arguments = get_args(annotation)
    variadic = bool(arguments) and arguments[-1] is Ellipsis
    if not variadic and len(arguments) != len(value):
        return None, False

    element_annotations = [arguments[0]] * len(value) if variadic else list(arguments)

    elements: list[object] = []
    for item, element_annotation in zip(value, element_annotations, strict=True):
        coerced_value, matched = _coerce_parameter_value(value=item, annotation=element_annotation)
        if not matched:
            return None, False
        elements.append(coerced_value)

    return tuple(elements), True


def _describe_annotation(annotation: object) -> str:
    """Describes a resolved field annotation in the form an error message names it.

    Args:
        annotation: The resolved annotation of a configuration field.

    Returns:
        The annotation's own name for a plain class, and its string form for a union or a parameterized generic.
    """
    if isinstance(annotation, type):
        return annotation.__name__

    return str(annotation)
