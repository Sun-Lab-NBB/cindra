"""Contains tests for the configuration MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cindra.dataclasses import SingleRecordingConfiguration
from cindra.interface.configuration_tools import resolve_dataset_name_tool, validate_config_file_tool

if TYPE_CHECKING:
    from pathlib import Path


class TestResolveDatasetName:
    """Tests the qualified dataset name the multi-recording preparation workflow builds."""

    def test_mixed_absolute_and_relative_paths_are_rejected(self) -> None:
        """Verifies that a path list the common-path resolver refuses reports an error dictionary."""
        result = resolve_dataset_name_tool(
            dataset_name="learning_task", recording_paths=["/data/animal_a/rec1", "rec2"]
        )

        assert result["success"] is False
        assert "must share a common parent directory" in result["error"]

    def test_specifier_is_derived_from_the_common_parent(self) -> None:
        """Verifies that absolute recording paths resolve their specifier from the deepest shared directory."""
        result = resolve_dataset_name_tool(
            dataset_name="learning_task", recording_paths=["/data/animal_a/rec1", "/data/animal_a/rec2"]
        )

        assert result["success"] is True
        assert result["dataset_name"] == "animal_a_learning_task"

    def test_relative_recording_paths_resolve_a_specifier(self) -> None:
        """Verifies that a list of relative paths sharing a parent still derives its specifier from that parent."""
        result = resolve_dataset_name_tool(
            dataset_name="learning_task", recording_paths=["data/animal_a/rec1", "data/animal_a/rec2"]
        )

        assert result["success"] is True
        assert result["dataset_name"] == "animal_a_learning_task"


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


def _write_one_photon_configuration(file_path: Path, window: int, sigma: float) -> None:
    """Saves a single-recording configuration whose one-photon registration section carries the given filter sizes."""
    configuration = SingleRecordingConfiguration()
    configuration.one_photon_registration.enabled = True
    configuration.one_photon_registration.spatial_highpass_window = window
    configuration.one_photon_registration.pre_smoothing_sigma = sigma
    configuration.save(file_path=file_path)
