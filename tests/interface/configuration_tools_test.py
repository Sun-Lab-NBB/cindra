"""Contains tests for the configuration MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cindra.layout import (
    PARAMETERS_FILENAME,
    OUTPUT_DIRECTORY_NAME,
    COMBINED_METADATA_FILENAME,
)
from cindra.dataclasses import BaselineMethod, MultiRecordingConfiguration, SingleRecordingConfiguration
from cindra.interface.configuration_tools import (
    set_config_values_tool,
    discover_recordings_tool,
    resolve_dataset_name_tool,
    validate_config_file_tool,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestResolveDatasetName:
    """Tests the qualified dataset name the multi-recording preparation workflow builds."""

    def test_mixed_absolute_and_relative_paths_are_rejected(self) -> None:
        """Verifies that a path list the common-path resolver refuses reports an error dictionary."""
        result = resolve_dataset_name_tool(dataset_name="learning_task", output_roots=["/data/animal_a/rec1", "rec2"])

        assert result["success"] is False
        assert "must share a common parent directory" in result["error"]

    def test_specifier_is_derived_from_the_common_parent(self) -> None:
        """Verifies that absolute output roots resolve their specifier from the deepest shared directory."""
        result = resolve_dataset_name_tool(
            dataset_name="learning_task", output_roots=["/data/animal_a/rec1", "/data/animal_a/rec2"]
        )

        assert result["success"] is True
        assert result["dataset_name"] == "animal_a_learning_task"

    def test_relative_output_roots_resolve_a_specifier(self) -> None:
        """Verifies that a list of relative paths sharing a parent still derives its specifier from that parent."""
        result = resolve_dataset_name_tool(
            dataset_name="learning_task", output_roots=["data/animal_a/rec1", "data/animal_a/rec2"]
        )

        assert result["success"] is True
        assert result["dataset_name"] == "animal_a_learning_task"

    def test_an_empty_output_root_list_is_rejected(self) -> None:
        """Verifies that a call naming no output root reports the requirement instead of deriving a specifier."""
        result = resolve_dataset_name_tool(dataset_name="learning_task", output_roots=[])

        assert result["success"] is False
        assert "At least one output root is required." in result["error"]


class TestDiscoverRecordings:
    """Tests the per-candidate objects the discovery tool pairs with every recording root it finds."""

    def test_single_recording_candidates_carry_the_raw_data_path(self, tmp_path: Path) -> None:
        """Verifies that every raw candidate maps its recording root to the directory holding the marker file."""
        for session in ("session_a", "session_b"):
            raw_directory = tmp_path / session / "raw_data" / "mesoscope_data"
            raw_directory.mkdir(parents=True)
            (raw_directory / PARAMETERS_FILENAME).write_text("{}")

        result = discover_recordings_tool(root_directory=str(tmp_path))

        assert result["single_recording_count"] == 2
        assert result["single_recording_candidates"] == [
            {
                "recording_root": str(tmp_path / "session_a"),
                "raw_data_path": str(tmp_path / "session_a" / "raw_data" / "mesoscope_data"),
            },
            {
                "recording_root": str(tmp_path / "session_b"),
                "raw_data_path": str(tmp_path / "session_b" / "raw_data" / "mesoscope_data"),
            },
        ]

    def test_multi_recording_candidates_carry_the_output_root(self, tmp_path: Path) -> None:
        """Verifies that a completed output maps its candidate to the parent of the cindra directory."""
        output_root = tmp_path / "session_a" / "processed_data"
        cindra_directory = output_root / OUTPUT_DIRECTORY_NAME
        cindra_directory.mkdir(parents=True)
        (cindra_directory / COMBINED_METADATA_FILENAME).write_bytes(b"")

        result = discover_recordings_tool(root_directory=str(tmp_path))

        assert result["multi_recording_count"] == 1
        assert result["multi_recording_candidates"] == [
            {"recording_root": str(output_root), "output_root": str(output_root)}
        ]

    def test_a_marker_outside_a_cindra_directory_falls_back_to_the_recording_root(self, tmp_path: Path) -> None:
        """Verifies that a stray combined metadata marker reports the recording root as its output root."""
        for session in ("session_a", "session_b"):
            marker_directory = tmp_path / session / "derived" / "analysis"
            marker_directory.mkdir(parents=True)
            (marker_directory / COMBINED_METADATA_FILENAME).write_bytes(b"")

        result = discover_recordings_tool(root_directory=str(tmp_path))

        assert result["multi_recording_candidates"] == [
            {"recording_root": str(tmp_path / "session_a"), "output_root": str(tmp_path / "session_a")},
            {"recording_root": str(tmp_path / "session_b"), "output_root": str(tmp_path / "session_b")},
        ]

    def test_a_root_holding_no_marker_reports_empty_candidate_lists(self, tmp_path: Path) -> None:
        """Verifies that a root carrying neither marker reports both candidate lists as empty."""
        result = discover_recordings_tool(root_directory=str(tmp_path))

        assert result["success"] is True
        assert result["single_recording_candidates"] == []
        assert result["multi_recording_candidates"] == []

    def test_a_missing_root_directory_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that a root directory that does not exist reports an error naming that path."""
        missing = tmp_path / "absent"

        result = discover_recordings_tool(root_directory=str(missing))

        assert result["success"] is False
        assert str(missing) in result["error"]


class TestOnePhotonRegistrationValidation:
    """Tests the pre-flight verdict the validator returns for the one-photon registration filter sizes."""

    def test_odd_spatial_highpass_window_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that an odd high-pass window, which the registration stage refuses, invalidates the file."""
        file_path = tmp_path / "configuration.yaml"
        _write_one_photon_configuration(file_path=file_path, window=25, sigma=0.0)

        result = validate_config_file_tool(file_path=str(file_path))

        assert result["valid"] is False
        assert any("spatial_highpass_window must be an even integer" in error for error in result["errors"])

    def test_odd_pre_smoothing_window_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that a smoothing sigma truncating to an odd window invalidates the file."""
        file_path = tmp_path / "configuration.yaml"
        _write_one_photon_configuration(file_path=file_path, window=42, sigma=3.0)

        result = validate_config_file_tool(file_path=str(file_path))

        assert result["valid"] is False
        assert any("pre_smoothing_sigma must truncate to an even filter window" in error for error in result["errors"])

    def test_even_windows_validate(self, tmp_path: Path) -> None:
        """Verifies that even filter windows leave the configuration valid."""
        file_path = tmp_path / "configuration.yaml"
        _write_one_photon_configuration(file_path=file_path, window=42, sigma=2.0)

        result = validate_config_file_tool(file_path=str(file_path))

        assert result["valid"] is True
        assert "errors" not in result


class TestDeviceBatchSizeValidation:
    """Tests the pre-flight verdict the validator returns for the device batch size."""

    def test_negative_device_batch_size_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that a negative device batch invalidates every single-recording configuration alike."""
        file_path = _write_single_recording_configuration(directory=tmp_path)
        configuration = SingleRecordingConfiguration.load(file_path=file_path)
        configuration.registration.gpu_batch_size = -1
        configuration.save(file_path=file_path)

        result = validate_config_file_tool(file_path=str(file_path))

        assert result["valid"] is False
        assert any("registration.gpu_batch_size must be non-negative" in error for error in result["errors"])

    @pytest.mark.parametrize("gpu_batch_size", [0, 64])
    def test_non_negative_device_batch_size_validates(self, tmp_path: Path, gpu_batch_size: int) -> None:
        """Verifies that a zero and a positive device batch both leave the configuration valid."""
        file_path = _write_single_recording_configuration(directory=tmp_path)
        configuration = SingleRecordingConfiguration.load(file_path=file_path)
        configuration.registration.gpu_batch_size = gpu_batch_size
        configuration.save(file_path=file_path)

        result = validate_config_file_tool(file_path=str(file_path))

        assert result["valid"] is True
        assert "errors" not in result


class TestSetConfigValues:
    """Tests the tool that writes new values into an existing configuration file."""

    def test_requested_values_reach_the_file(self, tmp_path: Path) -> None:
        """Verifies that every accepted value is written to disk and reported with its previous value."""
        file_path = _write_single_recording_configuration(directory=tmp_path)

        result = set_config_values_tool(
            file_path=str(file_path),
            values={"registration.batch_size": 250, "roi_detection.threshold_scaling": 1.5},
        )

        assert result["success"] is True
        assert result["valid"] is True
        assert result["changed"] == {
            "registration.batch_size": {"previous": 100, "current": 250},
            "roi_detection.threshold_scaling": {"previous": 1.0, "current": 1.5},
        }

        written = SingleRecordingConfiguration.load(file_path=file_path)
        assert written.registration.batch_size == 250
        assert written.roi_detection.threshold_scaling == 1.5

    def test_an_integer_widens_to_a_floating_point_parameter(self, tmp_path: Path) -> None:
        """Verifies that an integer written to a float parameter is stored as a floating point value."""
        file_path = _write_single_recording_configuration(directory=tmp_path)

        result = set_config_values_tool(file_path=str(file_path), values={"main.tau": 1})

        assert result["changed"]["main.tau"]["current"] == 1.0
        assert isinstance(SingleRecordingConfiguration.load(file_path=file_path).main.tau, float)

    def test_document_value_forms_are_coerced_to_their_annotated_types(self, tmp_path: Path) -> None:
        """Verifies that a raw enumeration value, a path string, and a list reach their annotated field types."""
        file_path = _write_single_recording_configuration(directory=tmp_path)

        result = set_config_values_tool(
            file_path=str(file_path),
            values={
                "spike_deconvolution.baseline_method": "constant",
                "file_io.data_path": str(tmp_path / "raw_data"),
                "nonrigid_registration.block_size": [64, 64],
                "file_io.ignored_file_names": ["z_stack.tif"],
            },
        )

        assert result["success"] is True

        written = SingleRecordingConfiguration.load(file_path=file_path)
        assert written.spike_deconvolution.baseline_method is BaselineMethod.CONSTANT
        assert written.file_io.data_path == tmp_path / "raw_data"
        assert written.nonrigid_registration.block_size == (64, 64)
        assert written.file_io.ignored_file_names == ("z_stack.tif",)

    def test_a_written_value_the_validator_refuses_reports_an_invalid_file(self, tmp_path: Path) -> None:
        """Verifies that a value written outside its valid domain is reported through the post-write validation."""
        file_path = _write_single_recording_configuration(directory=tmp_path)

        result = set_config_values_tool(file_path=str(file_path), values={"registration.batch_size": 0})

        assert result["success"] is True
        assert result["valid"] is False
        assert any("registration.batch_size must be positive" in error for error in result["errors"])
        assert SingleRecordingConfiguration.load(file_path=file_path).registration.batch_size == 0

    def test_an_unknown_enumeration_value_is_caught_by_the_post_write_validation(self, tmp_path: Path) -> None:
        """Verifies that a baseline method outside the enumeration is written as a string and reported invalid."""
        file_path = _write_single_recording_configuration(directory=tmp_path)

        result = set_config_values_tool(
            file_path=str(file_path), values={"spike_deconvolution.baseline_method": "mean"}
        )

        assert result["success"] is True
        assert result["valid"] is False
        assert any("spike_deconvolution.baseline_method must be one of" in error for error in result["errors"])

    def test_a_multi_recording_configuration_is_written_through_its_own_schema(self, tmp_path: Path) -> None:
        """Verifies that the tool resolves the multi-recording schema from the file's pipeline type discriminator."""
        file_path = tmp_path / "multi_configuration.yaml"
        MultiRecordingConfiguration().save(file_path=file_path)

        result = set_config_values_tool(
            file_path=str(file_path),
            values={
                "recording_io.dataset_name": "animal_a_learning_task",
                "recording_io.recording_directories": [str(tmp_path / "rec1"), str(tmp_path / "rec2")],
            },
        )

        assert result["success"] is True

        written = MultiRecordingConfiguration.load(file_path=file_path)
        assert written.recording_io.dataset_name == "animal_a_learning_task"
        assert written.recording_io.recording_directories == (tmp_path / "rec1", tmp_path / "rec2")

    @pytest.mark.parametrize(
        "dotted_path,value,expected",
        [
            ("nonexistent.batch_size", 1, "holds no section named 'nonexistent'"),
            ("registration.nonexistent", 1, "holds no writable parameter named 'nonexistent'"),
            ("main.tau.value", 1, "names a parameter, not a section"),
            ("registration", 1, "names a configuration section"),
            ("pipeline_type", "single-recording", "holds no writable parameter named 'pipeline_type'"),
            ("registration.batch_size", 1.5, "typed as int, but received float"),
            ("registration.two_step_registration", 1, "typed as bool, but received int"),
            ("main.tau", True, "typed as float, but received bool"),
            ("nonrigid_registration.block_size", [64], "typed as tuple[int, int], but received list"),
            ("diffeomorphic_registration.speed_factor", 3.0, "holds no section named 'diffeomorphic_registration'"),
        ],
    )
    def test_rejected_entries_are_reported(
        self, tmp_path: Path, dotted_path: str, value: object, expected: str
    ) -> None:
        """Verifies that an unknown path, a section target, and a mistyped value are each rejected with a reason."""
        file_path = _write_single_recording_configuration(directory=tmp_path)

        result = set_config_values_tool(file_path=str(file_path), values={dotted_path: value})

        assert result["success"] is False
        assert any(expected in error for error in result["errors"])

    def test_a_rejected_entry_leaves_the_file_unchanged(self, tmp_path: Path) -> None:
        """Verifies that a call pairing an accepted entry with a rejected one writes neither of them."""
        file_path = _write_single_recording_configuration(directory=tmp_path)
        original = file_path.read_bytes()

        result = set_config_values_tool(
            file_path=str(file_path),
            values={"registration.batch_size": 250, "registration.reference_frame_count": "many"},
        )

        assert result["success"] is False
        assert "1 of 2 requested entries were rejected" in result["error"]
        assert file_path.read_bytes() == original

    def test_a_missing_file_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that a configuration path pointing at no file reports an error naming that path."""
        missing = tmp_path / "configuration.yaml"

        result = set_config_values_tool(file_path=str(missing), values={"main.tau": 0.5})

        assert result["success"] is False
        assert str(missing) in result["error"]

    def test_a_non_yaml_file_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that a file carrying a suffix other than '.yaml' or '.yml' is refused."""
        file_path = tmp_path / "configuration.txt"
        file_path.write_text("pipeline_type: single-recording\n")

        result = set_config_values_tool(file_path=str(file_path), values={"main.tau": 0.5})

        assert result["success"] is False
        assert "Expected a '.yaml' or '.yml' file" in result["error"]

    def test_an_empty_value_mapping_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that a call naming no parameter reports the requirement instead of rewriting the file."""
        file_path = _write_single_recording_configuration(directory=tmp_path)

        result = set_config_values_tool(file_path=str(file_path), values={})

        assert result["success"] is False
        assert "At least one 'section.parameter' entry is required." in result["error"]

    def test_a_file_without_a_pipeline_type_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that a YAML file carrying no pipeline type discriminator cannot be modified."""
        file_path = tmp_path / "configuration.yaml"
        file_path.write_text("main:\n  tau: 0.4\n")

        result = set_config_values_tool(file_path=str(file_path), values={"main.tau": 0.5})

        assert result["success"] is False
        assert "The 'pipeline_type' field is missing or unrecognized" in result["error"]


def _write_one_photon_configuration(file_path: Path, window: int, sigma: float) -> None:
    """Saves a single-recording configuration whose one-photon registration section carries the given filter sizes."""
    configuration = SingleRecordingConfiguration()
    configuration.one_photon_registration.enabled = True
    configuration.one_photon_registration.spatial_highpass_window = window
    configuration.one_photon_registration.pre_smoothing_sigma = sigma
    configuration.save(file_path=file_path)


def _write_single_recording_configuration(directory: Path) -> Path:
    """Saves a default single-recording configuration into the directory and returns its path."""
    file_path = directory / "configuration.yaml"
    SingleRecordingConfiguration().save(file_path=file_path)
    return file_path
