"""Contains tests for configuration save/load round-trips and pipeline type detection."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from cindra.dataclasses.multi_recording_configuration import MultiRecordingConfiguration
from cindra.dataclasses.single_recording_configuration import (
    PipelineType,
    SingleRecordingConfiguration,
    detect_pipeline_type,
)


class TestSingleRecordingConfigurationRoundTrip:
    """Tests for SingleRecordingConfiguration save/load round-trip."""

    def test_default_configuration_survives_round_trip(self, tmp_path: Path) -> None:
        """Verifies that a default SingleRecordingConfiguration can be saved and loaded without data loss."""
        file_path = tmp_path / "single_config.yaml"
        original = SingleRecordingConfiguration()

        original.save(file_path=file_path)
        loaded = SingleRecordingConfiguration.load(file_path=file_path)

        assert loaded.pipeline_type == original.pipeline_type
        assert loaded.main.tau == original.main.tau
        assert loaded.main.two_channels == original.main.two_channels
        assert loaded.registration.maximum_offset_fraction == original.registration.maximum_offset_fraction
        assert loaded.roi_detection.threshold_scaling == original.roi_detection.threshold_scaling
        assert loaded.signal_extraction.minimum_neuropil_pixels == original.signal_extraction.minimum_neuropil_pixels
        assert loaded.spike_deconvolution.neuropil_coefficient == original.spike_deconvolution.neuropil_coefficient
        assert loaded.runtime.parallel_workers == original.runtime.parallel_workers


class TestMultiRecordingConfigurationRoundTrip:
    """Tests for MultiRecordingConfiguration save/load round-trip."""

    def test_default_configuration_survives_round_trip(self, tmp_path: Path) -> None:
        """Verifies that a default MultiRecordingConfiguration can be saved and loaded without data loss."""
        file_path = tmp_path / "multi_config.yaml"
        original = MultiRecordingConfiguration()

        original.save(file_path=file_path)
        loaded = MultiRecordingConfiguration.load(file_path=file_path)

        assert loaded.pipeline_type == original.pipeline_type
        assert loaded.recording_io.dataset_name == original.recording_io.dataset_name
        assert loaded.roi_selection.probability_threshold == original.roi_selection.probability_threshold
        assert loaded.diffeomorphic_registration.speed_factor == original.diffeomorphic_registration.speed_factor
        assert loaded.roi_tracking.threshold == original.roi_tracking.threshold
        assert loaded.signal_extraction.minimum_neuropil_pixels == original.signal_extraction.minimum_neuropil_pixels
        assert loaded.spike_deconvolution.neuropil_coefficient == original.spike_deconvolution.neuropil_coefficient
        assert loaded.runtime.parallel_workers == original.runtime.parallel_workers


class TestDetectPipelineType:
    """Tests for detect_pipeline_type."""

    def test_detects_single_recording_pipeline(self, tmp_path: Path) -> None:
        """Verifies that a saved SingleRecordingConfiguration file is detected as SINGLE_RECORDING."""
        file_path = tmp_path / "single.yaml"
        configuration = SingleRecordingConfiguration()
        configuration.save(file_path=file_path)

        result = detect_pipeline_type(file_path=file_path)

        assert result == PipelineType.SINGLE_RECORDING

    def test_detects_multi_recording_pipeline(self, tmp_path: Path) -> None:
        """Verifies that a saved MultiRecordingConfiguration file is detected as MULTI_RECORDING."""
        file_path = tmp_path / "multi.yaml"
        configuration = MultiRecordingConfiguration()
        configuration.save(file_path=file_path)

        result = detect_pipeline_type(file_path=file_path)

        assert result == PipelineType.MULTI_RECORDING

    def test_raises_error_for_nonexistent_file(self, tmp_path: Path) -> None:
        """Verifies that a FileNotFoundError is raised when the configuration file does not exist."""
        file_path = tmp_path / "nonexistent.yaml"

        with pytest.raises(FileNotFoundError, match="Unable to detect the pipeline type"):
            detect_pipeline_type(file_path=file_path)

    def test_raises_error_for_non_yaml_file(self, tmp_path: Path) -> None:
        """Verifies that a FileNotFoundError is raised when the file does not have a .yaml extension."""
        file_path = tmp_path / "config.txt"
        file_path.write_text("pipeline_type: single-recording")

        with pytest.raises(FileNotFoundError, match="Unable to detect the pipeline type"):
            detect_pipeline_type(file_path=file_path)

    def test_raises_error_for_unrecognized_pipeline_type(self, tmp_path: Path) -> None:
        """Verifies that a ValueError is raised when the pipeline_type value is not a recognized member."""
        file_path = tmp_path / "bad_config.yaml"
        file_path.write_text("pipeline_type: invalid-pipeline-type\n")

        with pytest.raises(ValueError, match="unrecognized value"):
            detect_pipeline_type(file_path=file_path)

    def test_raises_error_for_missing_pipeline_type_field(self, tmp_path: Path) -> None:
        """Verifies that a ValueError is raised when the pipeline_type field is absent from the YAML."""
        file_path = tmp_path / "no_type.yaml"
        file_path.write_text("some_other_field: 42\n")

        with pytest.raises(ValueError, match="missing or has an unrecognized value"):
            detect_pipeline_type(file_path=file_path)
