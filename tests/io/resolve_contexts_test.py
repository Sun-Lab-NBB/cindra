"""Contains integration tests for the recording-context resolution entry points provided by the context module."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from cindra.io.context import (
    PARAMETERS_FILENAME,
    resolve_multi_recording_contexts,
    resolve_single_recording_contexts,
)
from cindra.dataclasses import (
    CombinedData,
    DetectionData,
    ExtractionData,
    AcquisitionParameters,
    MultiRecordingConfiguration,
    SingleRecordingConfiguration,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_single_configuration(output_path: Path, data_path: Path | None = None) -> SingleRecordingConfiguration:
    """Builds a single-recording configuration with the given output and data paths and serial execution settings."""
    configuration = SingleRecordingConfiguration()
    configuration.file_io.output_path = output_path
    configuration.file_io.data_path = data_path
    configuration.runtime.parallel_workers = 1
    configuration.runtime.display_progress_bars = False
    return configuration


def _write_saved_acquisition(output_path: Path, acquisition: AcquisitionParameters) -> None:
    """Saves acquisition parameters to the processed cindra output directory used by context resolution."""
    cindra_directory = output_path / "cindra"
    cindra_directory.mkdir(parents=True, exist_ok=True)
    acquisition.to_yaml(file_path=cindra_directory / "acquisition_parameters.yaml")


def _write_raw_parameters(data_path: Path, data: dict[str, object]) -> None:
    """Writes a raw cindra_parameters.json acquisition file into the data directory."""
    data_path.mkdir(parents=True, exist_ok=True)
    (data_path / PARAMETERS_FILENAME).write_text(json.dumps(data))


def _make_recording(parent: Path, name: str, acquisition: AcquisitionParameters) -> Path:
    """Creates a fake processed recording with a combined_metadata.npz and acquisition file, returning its root."""
    recording_root = parent / name
    cindra_root = recording_root / "cindra"
    cindra_root.mkdir(parents=True, exist_ok=True)

    combined = CombinedData(detection=DetectionData(), extraction=ExtractionData())
    combined.save(root_path=cindra_root)

    acquisition.to_yaml(file_path=cindra_root / "acquisition_parameters.yaml")
    return recording_root


def _make_multi_configuration(
    recording_directories: tuple[Path, ...], dataset_name: str
) -> MultiRecordingConfiguration:
    """Builds a multi-recording configuration referencing the given recording directories and dataset name."""
    configuration = MultiRecordingConfiguration()
    configuration.recording_io.recording_directories = recording_directories
    configuration.recording_io.dataset_name = dataset_name
    configuration.runtime.parallel_workers = 1
    configuration.runtime.display_progress_bars = False
    return configuration


class TestResolveSingleRecordingContexts:
    """Tests resolve_single_recording_contexts."""

    def test_creates_one_context_per_plane(self, tmp_path: Path) -> None:
        """Verifies that one context per physical plane is created with a derived per-plane sampling rate."""
        output_path = tmp_path / "output"
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=2, channel_number=1)
        _write_saved_acquisition(output_path=output_path, acquisition=acquisition)
        configuration = _make_single_configuration(output_path=output_path)

        contexts = resolve_single_recording_contexts(configuration=configuration)

        assert len(contexts) == 2
        assert all(context.runtime.io.sampling_rate == 15.0 for context in contexts)
        assert (output_path / "cindra" / "plane_0").is_dir()
        assert (output_path / "cindra" / "plane_1").is_dir()
        assert contexts[0].runtime.io.plane_index == 0
        assert contexts[1].runtime.io.plane_index == 1
        assert (
            contexts[0].runtime.io.registered_binary_path == output_path / "cindra" / "plane_0" / "channel_1_data.bin"
        )
        assert contexts[0].runtime.io.registered_binary_path_channel_2 is None
        assert (output_path / "cindra" / "configuration.yaml").exists()
        assert (output_path / "cindra" / "plane_0" / "runtime_data.yaml").exists()
        assert (output_path / "cindra" / "plane_1" / "runtime_data.yaml").exists()

    def test_two_channels_sets_second_binary_path(self, tmp_path: Path) -> None:
        """Verifies that a two-channel recording assigns the channel 2 registered binary path."""
        output_path = tmp_path / "output"
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=2)
        _write_saved_acquisition(output_path=output_path, acquisition=acquisition)
        configuration = _make_single_configuration(output_path=output_path)

        contexts = resolve_single_recording_contexts(configuration=configuration)

        assert len(contexts) == 1
        expected_path = output_path / "cindra" / "plane_0" / "channel_2_data.bin"
        assert contexts[0].runtime.io.registered_binary_path_channel_2 == expected_path

    def test_mroi_creates_one_context_per_virtual_plane(self, tmp_path: Path) -> None:
        """Verifies that MROI data produces one context per virtual plane with populated MROI geometry fields."""
        output_path = tmp_path / "output"
        acquisition = AcquisitionParameters(
            frame_rate=30.0,
            plane_number=1,
            channel_number=1,
            roi_number=3,
            roi_lines=((0, 1), (2, 3), (4, 5)),
            roi_x_coordinates=(0, 100, 200),
            roi_y_coordinates=(10, 20, 30),
        )
        _write_saved_acquisition(output_path=output_path, acquisition=acquisition)
        configuration = _make_single_configuration(output_path=output_path)

        contexts = resolve_single_recording_contexts(configuration=configuration)

        assert len(contexts) == 3
        assert contexts[0].runtime.io.mroi_lines == (0, 1)
        assert contexts[0].runtime.io.mroi_x_offset == 0
        assert contexts[0].runtime.io.mroi_y_offset == 10
        assert contexts[2].runtime.io.mroi_lines == (4, 5)
        assert contexts[2].runtime.io.mroi_x_offset == 200
        assert contexts[2].runtime.io.mroi_y_offset == 30

    def test_loads_acquisition_from_raw_data_path(self, tmp_path: Path) -> None:
        """Verifies that acquisition parameters are loaded from raw data when no processed output exists."""
        output_path = tmp_path / "output"
        data_path = tmp_path / "raw"
        _write_raw_parameters(data_path=data_path, data={"frame_rate": 20.0, "plane_number": 1, "channel_number": 1})
        configuration = _make_single_configuration(output_path=output_path, data_path=data_path)

        contexts = resolve_single_recording_contexts(configuration=configuration)

        assert len(contexts) == 1
        assert contexts[0].runtime.io.sampling_rate == 20.0
        assert contexts[0].acquisition.frame_rate == 20.0

    def test_rejects_more_than_two_channels(self, tmp_path: Path) -> None:
        """Verifies that a recording with more than two channels raises a ValueError."""
        output_path = tmp_path / "output"
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=3)
        _write_saved_acquisition(output_path=output_path, acquisition=acquisition)
        configuration = _make_single_configuration(output_path=output_path)

        with pytest.raises(ValueError, match="supports at most"):
            resolve_single_recording_contexts(configuration=configuration)

    def test_raises_without_output_path(self, tmp_path: Path) -> None:
        """Verifies that a missing output_path raises a ValueError."""
        configuration = _make_single_configuration(output_path=tmp_path / "output")
        configuration.file_io.output_path = None

        with pytest.raises(ValueError, match="output_path must be configured"):
            resolve_single_recording_contexts(configuration=configuration)

    def test_raises_without_processed_or_raw_data(self, tmp_path: Path) -> None:
        """Verifies that a ValueError is raised when neither processed output nor a raw data path is available."""
        configuration = _make_single_configuration(output_path=tmp_path / "output", data_path=None)

        with pytest.raises(ValueError, match="No processed data exists"):
            resolve_single_recording_contexts(configuration=configuration)

    def test_persist_false_round_trip_after_bootstrap(self, tmp_path: Path) -> None:
        """Verifies that a load-only resolution succeeds after a prior persisting resolution wrote the bootstrap."""
        output_path = tmp_path / "output"
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=2, channel_number=1)
        _write_saved_acquisition(output_path=output_path, acquisition=acquisition)
        configuration = _make_single_configuration(output_path=output_path)

        resolve_single_recording_contexts(configuration=configuration, persist=True)
        contexts = resolve_single_recording_contexts(configuration=configuration, persist=False)

        assert len(contexts) == 2
        assert contexts[0].runtime.io.sampling_rate == 15.0

    def test_persist_false_raises_without_bootstrap(self, tmp_path: Path) -> None:
        """Verifies that a load-only resolution raises when no plane runtime data file was written first."""
        output_path = tmp_path / "output"
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        _write_saved_acquisition(output_path=output_path, acquisition=acquisition)
        configuration = _make_single_configuration(output_path=output_path)

        with pytest.raises(FileNotFoundError, match="without bootstrap persistence"):
            resolve_single_recording_contexts(configuration=configuration, persist=False)


class TestResolveMultiRecordingContexts:
    """Tests resolve_multi_recording_contexts."""

    def test_resolves_all_recordings(self, tmp_path: Path) -> None:
        """Verifies that every recording is resolved with shared dataset output paths and persisted bootstrap files."""
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        recording_one = _make_recording(parent=tmp_path, name="rec1", acquisition=acquisition)
        recording_two = _make_recording(parent=tmp_path, name="rec2", acquisition=acquisition)
        configuration = _make_multi_configuration(
            recording_directories=(recording_one, recording_two), dataset_name="test_dataset"
        )

        contexts = resolve_multi_recording_contexts(configuration=configuration)

        assert len(contexts) == 2
        output_one = recording_one / "cindra" / "multi_recording" / "test_dataset"
        output_two = recording_two / "cindra" / "multi_recording" / "test_dataset"
        assert contexts[0].runtime.io.recording_id == "rec1"
        assert contexts[1].runtime.io.recording_id == "rec2"
        assert contexts[0].runtime.io.dataset_output_paths == (output_one, output_two)
        assert contexts[0].runtime.io.dataset_name == "test_dataset"
        assert contexts[0].runtime.io.data_path == recording_one / "cindra"
        assert contexts[0].runtime.combined_data is not None
        assert (output_one / "multi_recording_runtime_data.yaml").exists()
        assert (output_two / "multi_recording_runtime_data.yaml").exists()
        assert (output_one / "multi_recording_configuration.yaml").exists()

    def test_target_recording_filters_to_single(self, tmp_path: Path) -> None:
        """Verifies that providing a target recording identifier resolves only the matching recording."""
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        recording_one = _make_recording(parent=tmp_path, name="rec1", acquisition=acquisition)
        recording_two = _make_recording(parent=tmp_path, name="rec2", acquisition=acquisition)
        configuration = _make_multi_configuration(
            recording_directories=(recording_one, recording_two), dataset_name="test_dataset"
        )

        contexts = resolve_multi_recording_contexts(configuration=configuration, target_recording_id="rec2")

        assert len(contexts) == 1
        assert contexts[0].runtime.io.recording_id == "rec2"

    def test_invalid_target_recording_raises(self, tmp_path: Path) -> None:
        """Verifies that an unknown target recording identifier raises a ValueError."""
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        recording_one = _make_recording(parent=tmp_path, name="rec1", acquisition=acquisition)
        recording_two = _make_recording(parent=tmp_path, name="rec2", acquisition=acquisition)
        configuration = _make_multi_configuration(
            recording_directories=(recording_one, recording_two), dataset_name="test_dataset"
        )

        with pytest.raises(ValueError, match="does not match any resolved"):
            resolve_multi_recording_contexts(configuration=configuration, target_recording_id="missing")

    def test_persist_false_round_trip_after_bootstrap(self, tmp_path: Path) -> None:
        """Verifies that a load-only resolution succeeds after a prior persisting resolution wrote the bootstrap."""
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        recording_one = _make_recording(parent=tmp_path, name="rec1", acquisition=acquisition)
        recording_two = _make_recording(parent=tmp_path, name="rec2", acquisition=acquisition)
        configuration = _make_multi_configuration(
            recording_directories=(recording_one, recording_two), dataset_name="test_dataset"
        )

        resolve_multi_recording_contexts(configuration=configuration, persist=True)
        contexts = resolve_multi_recording_contexts(configuration=configuration, persist=False)

        assert len(contexts) == 2
        assert contexts[0].runtime.combined_data is not None

    def test_persist_false_raises_without_bootstrap(self, tmp_path: Path) -> None:
        """Verifies that a load-only resolution raises when no recording runtime data file was written first."""
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        recording_one = _make_recording(parent=tmp_path, name="rec1", acquisition=acquisition)
        recording_two = _make_recording(parent=tmp_path, name="rec2", acquisition=acquisition)
        configuration = _make_multi_configuration(
            recording_directories=(recording_one, recording_two), dataset_name="test_dataset"
        )

        with pytest.raises(FileNotFoundError, match="without bootstrap persistence"):
            resolve_multi_recording_contexts(configuration=configuration, persist=False)
