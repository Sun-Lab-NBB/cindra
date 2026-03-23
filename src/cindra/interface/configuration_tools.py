"""Provides MCP tools for pipeline configuration generation, reading, validation, recording discovery, and dataset name
resolution.

These tools enable AI agents to generate default configuration files for both single-recording and multi-recording
pipelines, read and validate configuration files, discover recordings available for processing under a
given root directory, and construct qualified dataset names for multi-recording processing.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal
from os.path import commonpath
from pathlib import Path
from dataclasses import (
    fields as dataclass_fields,
    is_dataclass,
)

import yaml  # type: ignore[import-untyped]

from ..io import resolve_recording_roots
from ..dataclasses import MultiRecordingConfiguration, SingleRecordingConfiguration
from .mcp_instance import mcp

_MAX_SPEED_FACTOR: int = 5
"""The upper bound of the typical speed_factor range for diffeomorphic registration."""

_MAX_PERCENTAGE: int = 100
"""The maximum valid value for percentage-based parameters (prevalence, percentile)."""

_FORBIDDEN_FILESYSTEM_CHARACTERS: frozenset[str] = frozenset('\\/:*?"<>|\x00')
"""Characters that are invalid in directory names on common filesystems."""

_MAXIMUM_CONTROL_CHARACTER_ORDINAL: int = 32
"""The exclusive upper bound of the ASCII control character range."""


@mcp.tool()
def generate_config_file(
    output_path: str, pipeline_type: Literal["single-recording", "multi-recording"]
) -> dict[str, str | bool]:
    """Generates a default configuration YAML file for the specified pipeline type.

    Creates a configuration file with sensible defaults that can be used directly or modified before processing.

    Args:
        output_path: The absolute path where the configuration file should be saved.
        pipeline_type: The type of pipeline configuration to generate ('single-recording' or 'multi-recording').

    Returns:
        On success, contains the resolved 'file_path' and the 'pipeline_type'. On failure, contains an 'error'
        describing the issue. Both cases include a 'success' flag.
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
    multi-recording processing). Returns recording root directories (not raw data or output subdirectories) so
    that downstream tools receive meaningful session-level paths. Recording roots are resolved by stripping shared
    structural subdirectories via ``resolve_recording_roots``.

    Args:
        root_directory: The absolute path to the root directory to search.

    Returns:
        On success, contains 'single_recording_candidates' and 'multi_recording_candidates' lists of recording root
        paths with their respective counts, and any permission 'errors' encountered during the search. On failure,
        contains an 'error' describing the issue.
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

    errors: list[str] = []

    # Discovers single-recording candidates via cindra_parameters.json marker files.
    single_marker_parents: list[Path] = []
    try:
        single_marker_parents.extend(marker_file.parent for marker_file in root_path.rglob("cindra_parameters.json"))
    except PermissionError as error:
        errors.append(f"Access denied during single-recording search: {error}")

    single_recording_paths = (
        sorted(str(root) for root in resolve_recording_roots(paths=single_marker_parents))
        if single_marker_parents
        else []
    )

    # Discovers multi-recording candidates via combined_metadata.npz marker files.
    multi_marker_parents: list[Path] = []
    try:
        multi_marker_parents.extend(marker_file.parent for marker_file in root_path.rglob("combined_metadata.npz"))
    except PermissionError as error:
        errors.append(f"Access denied during multi-recording search: {error}")

    multi_recording_paths = (
        sorted(str(root) for root in resolve_recording_roots(paths=multi_marker_parents))
        if multi_marker_parents
        else []
    )

    result: dict[str, object] = {
        "success": True,
        "single_recording_candidates": single_recording_paths,
        "single_recording_count": len(single_recording_paths),
        "multi_recording_candidates": multi_recording_paths,
        "multi_recording_count": len(multi_recording_paths),
    }

    if errors:
        result["errors"] = errors

    return result


@mcp.tool()
def resolve_dataset_name_tool(
    dataset_name: str,
    recording_paths: list[str],
    specifier: str = "",
) -> dict[str, object]:
    """Constructs a qualified dataset name by combining a shared base name with a batch-specific specifier.

    When multiple groups of recordings share the same analysis type (dataset_name), each group needs a unique qualified
    name for its output directory and batch processing key. This tool combines the user-provided dataset name with a
    specifier that distinguishes the group.

    When no specifier is provided, one is derived automatically from the deepest common parent directory of the
    recording paths. For example, recordings under /data/animal_A/rec1 and /data/animal_A/rec2 yield specifier
    'animal_A'. The agent can also determine the specifier through semantic decomposition of recording names or
    directory structure, or the user can provide one explicitly.

    Args:
        dataset_name: The shared name identifying the analysis type (e.g., 'learning_task'). This is the base name
            common to all groups in a batch.
        recording_paths: The absolute paths to the recording directories in this group. Used to derive the specifier
            when none is explicitly provided.
        specifier: An explicit batch-specific label distinguishing this group of recordings (e.g., an animal ID, brain
            region, or session group). When empty, the specifier is derived from the common parent directory of the
            recording paths.

    Returns:
        On success, contains the qualified 'dataset_name' (specifier_base), the 'base_name', and the 'specifier'
        used. On failure, contains an 'error' describing the issue. Both cases include a 'success' flag.
    """
    if not dataset_name:
        return {
            "success": False,
            "error": "Unable to resolve dataset name. The dataset_name must be a non-empty string.",
        }

    dataset_name_error = _validate_filesystem_name(name=dataset_name, field_label="dataset_name")
    if dataset_name_error is not None:
        return {"success": False, "error": dataset_name_error}

    if not recording_paths:
        return {
            "success": False,
            "error": "Unable to resolve dataset name. At least one recording path is required.",
        }

    # Derives specifier from the common parent directory when not explicitly provided.
    if not specifier:
        resolved_paths = [Path(p) for p in recording_paths]
        if len(resolved_paths) == 1:
            specifier = resolved_paths[0].parent.name
        else:
            common = Path(commonpath(resolved_paths))
            specifier = common.name

        if not specifier:
            return {
                "success": False,
                "error": "Unable to resolve dataset name. Could not derive a specifier from the recording paths.",
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
def read_config_file(file_path: str) -> dict[str, str | bool | list[str] | dict[str, object] | None]:
    """Reads a YAML configuration file and returns its raw contents as a dictionary.

    Notes:
        This function does not require conformance to the current cindra configuration schema making it suitable for
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

    # Attempts to detect the pipeline type from the raw YAML data.
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
def validate_config_file(file_path: str) -> dict[str, str | bool | list[str] | dict[str, dict[str, object]]]:
    """Validates a cindra configuration YAML file by loading it through the appropriate configuration dataclass,
    checking parameter values against known constraints, and identifying parameters that differ from their defaults.

    Args:
        file_path: The absolute path to the cindra configuration YAML file to validate.

    Returns:
        On success, contains the resolved 'file_path', 'pipeline_type', overall 'valid' status, and any validation
        'errors', 'warnings', or 'non_default_parameters' detected. On failure, contains an 'error' describing the
        issue. Both cases include a 'success' flag.
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
            errors, warnings = _validate_single_recording(config)
        else:
            config = MultiRecordingConfiguration.load(file_path=path)
            default = MultiRecordingConfiguration()
            errors, warnings = _validate_multi_recording(config)
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


def _to_json_compatible(value: object) -> object:
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
        return [_to_json_compatible(item) for item in value]
    return value


def _identify_non_default_parameters(config: object, default: object, prefix: str = "") -> dict[str, dict[str, object]]:
    """Compares a loaded configuration against its default instance and returns a mapping of dotted parameter paths
    to their current and default values for all parameters that differ from the default.

    Args:
        config: The loaded configuration dataclass instance to compare.
        default: The default configuration dataclass instance to compare against.
        prefix: The dotted path prefix for nested dataclass fields. Defaults to an empty string.

    Returns:
        A dictionary mapping dotted parameter paths to dictionaries containing 'current' and 'default' values for
        each parameter that differs from its default.
    """
    differences: dict[str, dict[str, object]] = {}

    # noinspection PyDataclass
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
                "current": _to_json_compatible(current_value),
                "default": _to_json_compatible(default_value),
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

    # Main section validations.
    if config.main.tau <= 0:
        errors.append(f"main.tau must be positive (current: {config.main.tau}).")
    if not config.main.first_channel_functional and not config.main.second_channel_functional:
        errors.append(
            "Both main.first_channel_functional and main.second_channel_functional are False — no functional "
            "channel available."
        )

    # File I/O section — pipeline-set parameter warnings.
    if config.file_io.data_path is not None:
        warnings.append("file_io.data_path is set (pipeline-set parameter — will be overwritten at runtime).")
    if config.file_io.output_path is not None:
        warnings.append("file_io.output_path is set (pipeline-set parameter — will be overwritten at runtime).")

    # Registration section validations.
    if not config.main.two_channels and not config.registration.align_by_first_channel:
        warnings.append(
            "registration.align_by_first_channel is False but main.two_channels is False — no second channel "
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

    # One-photon registration section validations.
    if config.one_photon_registration.enabled:
        if config.one_photon_registration.spatial_highpass_window <= 0:
            errors.append(
                f"one_photon_registration.spatial_highpass_window must be positive when one-photon registration "
                f"is enabled (current: {config.one_photon_registration.spatial_highpass_window})."
            )
        if config.one_photon_registration.pre_smoothing_sigma < 0:
            errors.append(
                f"one_photon_registration.pre_smoothing_sigma must be non-negative when one-photon registration "
                f"is enabled (current: {config.one_photon_registration.pre_smoothing_sigma})."
            )
        if config.one_photon_registration.edge_taper_pixels < 0:
            errors.append(
                f"one_photon_registration.edge_taper_pixels must be non-negative when one-photon registration "
                f"is enabled (current: {config.one_photon_registration.edge_taper_pixels})."
            )

    # Nonrigid registration section validations.
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

    # ROI detection section validations.
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

    # Signal extraction section validations.
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
    if not 0 <= config.signal_extraction.cell_probability_percentile <= _MAX_PERCENTAGE:
        errors.append(
            f"signal_extraction.cell_probability_percentile must be in [0, {_MAX_PERCENTAGE}] "
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

    # Spike deconvolution section validations.
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
    if not 0 <= config.spike_deconvolution.baseline_percentile <= _MAX_PERCENTAGE:
        errors.append(
            f"spike_deconvolution.baseline_percentile must be in [0, {_MAX_PERCENTAGE}] "
            f"(current: {config.spike_deconvolution.baseline_percentile})."
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

    # Recording I/O section validations.
    if not config.recording_io.dataset_name:
        errors.append("recording_io.dataset_name must be a non-empty string.")
    if config.recording_io.recording_directories:
        warnings.append(
            "recording_io.recording_directories is set (pipeline-set parameter — will be overwritten at runtime)."
        )

    # ROI selection section validations.
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

    # Diffeomorphic registration section validations.
    if not 0 < config.diffeomorphic_registration.grid_sampling_factor <= 1:
        errors.append(
            f"diffeomorphic_registration.grid_sampling_factor must be in (0, 1] "
            f"(current: {config.diffeomorphic_registration.grid_sampling_factor})."
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
    elif not 1 <= config.diffeomorphic_registration.speed_factor <= _MAX_SPEED_FACTOR:
        warnings.append(
            f"diffeomorphic_registration.speed_factor is outside the typical 1-{_MAX_SPEED_FACTOR} range "
            f"(current: {config.diffeomorphic_registration.speed_factor})."
        )

    # ROI tracking section validations.
    if not 0 <= config.roi_tracking.threshold <= 1:
        errors.append(f"roi_tracking.threshold must be in [0, 1] (current: {config.roi_tracking.threshold}).")
    if not 0 <= config.roi_tracking.mask_prevalence <= _MAX_PERCENTAGE:
        errors.append(
            f"roi_tracking.mask_prevalence must be in [0, {_MAX_PERCENTAGE}] "
            f"(current: {config.roi_tracking.mask_prevalence})."
        )
    if not 0 <= config.roi_tracking.pixel_prevalence <= _MAX_PERCENTAGE:
        errors.append(
            f"roi_tracking.pixel_prevalence must be in [0, {_MAX_PERCENTAGE}] "
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

    # Signal extraction section validations.
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
    if not 0 <= config.signal_extraction.cell_probability_percentile <= _MAX_PERCENTAGE:
        errors.append(
            f"signal_extraction.cell_probability_percentile must be in [0, {_MAX_PERCENTAGE}] "
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

    # Spike deconvolution section validations.
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
    if not 0 <= config.spike_deconvolution.baseline_percentile <= _MAX_PERCENTAGE:
        errors.append(
            f"spike_deconvolution.baseline_percentile must be in [0, {_MAX_PERCENTAGE}] "
            f"(current: {config.spike_deconvolution.baseline_percentile})."
        )

    return errors, warnings


def _validate_filesystem_name(name: str, field_label: str) -> str | None:
    """Validates that a name is safe for use as a filesystem directory name.

    Rejects names containing characters that are invalid in directory names on common filesystems, names that consist
    entirely of whitespace, and reserved names like '.' and '..'.

    Args:
        name: The name string to validate.
        field_label: The label of the field being validated, used in error messages.

    Returns:
        An error message string if the name is invalid, or None if the name is safe.
    """
    if not name.strip():
        return f"Unable to resolve dataset name. The {field_label} must not be empty or consist entirely of whitespace."

    if name in (".", ".."):
        return f"Unable to resolve dataset name. The {field_label} must not be '{name}'."

    found = sorted({character for character in name if character in _FORBIDDEN_FILESYSTEM_CHARACTERS})
    if found:
        display = ", ".join(repr(character) for character in found)
        return f"Unable to resolve dataset name. The {field_label} contains filesystem-unsafe characters: {display}."

    # Rejects names with control characters (ordinal < 32).
    control_characters = [character for character in name if ord(character) < _MAXIMUM_CONTROL_CHARACTER_ORDINAL]
    if control_characters:
        return f"Unable to resolve dataset name. The {field_label} contains control characters."

    return None
