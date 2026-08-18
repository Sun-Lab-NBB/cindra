"""Contains tests for the recording and dataset inventory resolvers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from cindra.layout import (
    PARAMETERS_FILENAME,
    COMBINED_METADATA_FILENAME,
    ACQUISITION_PARAMETERS_FILENAME,
    REGISTRATION_DATA_DIRECTORY_NAME,
    TRACKING_TEMPLATE_MASKS_FILENAME,
    RecordingArrays,
    RegistrationArrays,
    resolve_plane_path,
    resolve_output_path,
    resolve_dataset_path,
)
from cindra.io.inventory import (
    is_plane_registered,
    is_dataset_discovered,
    is_recording_processed,
    resolve_recording_planes,
    resolve_dataset_recordings,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestResolveRecordingPlanes:
    """Tests the per-recording plane inventory."""

    def test_absent_recording_resolves_to_an_unresolved_record(self, tmp_path: Path) -> None:
        """Verifies that a recording carrying neither parameters nor output resolves rather than raising."""
        inventory = resolve_recording_planes(output_root=tmp_path)

        assert inventory.plane_count == 0
        assert not inventory.resolved
        assert inventory.plane_paths == ()
        assert inventory.registered_planes == ()
        assert not inventory.processed

    def test_single_roi_recording_counts_physical_planes(self, tmp_path: Path) -> None:
        """Verifies that a single-ROI recording holds one virtual plane per physical plane."""
        _write_acquisition(output_root=tmp_path, plane_number=3)

        inventory = resolve_recording_planes(output_root=tmp_path)

        assert inventory.plane_count == 3
        assert inventory.resolved
        assert inventory.plane_specifiers == ("plane_0", "plane_1", "plane_2")

    def test_multi_roi_recording_counts_virtual_planes(self, tmp_path: Path) -> None:
        """Verifies that a multi-ROI recording holds one virtual plane per ROI and physical plane combination."""
        _write_acquisition(output_root=tmp_path, plane_number=2, roi_number=3)

        inventory = resolve_recording_planes(output_root=tmp_path)

        assert inventory.plane_count == 6

    def test_registered_planes_hold_only_the_planes_carrying_a_reference_image(self, tmp_path: Path) -> None:
        """Verifies that the registered subset names the planes the processing stage can run."""
        _write_acquisition(output_root=tmp_path, plane_number=3)
        _register_plane(output_root=tmp_path, plane_index=0)
        _register_plane(output_root=tmp_path, plane_index=2)

        inventory = resolve_recording_planes(output_root=tmp_path)

        assert inventory.registered_planes == (0, 2)

    def test_processed_flag_follows_the_completion_marker(self, tmp_path: Path) -> None:
        """Verifies that the processed flag follows the combined metadata archive."""
        _write_acquisition(output_root=tmp_path)
        assert not resolve_recording_planes(output_root=tmp_path).processed

        (resolve_output_path(output_root=tmp_path) / COMBINED_METADATA_FILENAME).write_bytes(b"")

        assert resolve_recording_planes(output_root=tmp_path).processed

    def test_raw_parameters_supply_the_plane_count_before_processing(self, tmp_path: Path) -> None:
        """Verifies that an unprocessed recording resolves its plane count from the raw parameters file."""
        data_path = tmp_path / "raw"
        data_path.mkdir()
        (data_path / PARAMETERS_FILENAME).write_text(
            json.dumps({"frame_rate": 30.0, "plane_number": 4, "channel_number": 1})
        )

        inventory = resolve_recording_planes(output_root=tmp_path / "out", data_path=data_path)

        assert inventory.plane_count == 4
        assert inventory.resolved

    def test_missing_raw_directory_resolves_to_an_unresolved_record(self, tmp_path: Path) -> None:
        """Verifies that an unreadable raw directory resolves rather than raising."""
        inventory = resolve_recording_planes(output_root=tmp_path / "out", data_path=tmp_path / "absent")

        assert not inventory.resolved

    def test_raw_directory_without_parameters_resolves_to_an_unresolved_record(self, tmp_path: Path) -> None:
        """Verifies that a raw directory holding no parameters file resolves rather than raising."""
        data_path = tmp_path / "raw"
        data_path.mkdir()

        inventory = resolve_recording_planes(output_root=tmp_path / "out", data_path=data_path)

        assert not inventory.resolved


class TestResolveDatasetRecordings:
    """Tests the per-dataset recording inventory."""

    def test_empty_root_set_resolves_to_an_empty_record(self) -> None:
        """Verifies that a dataset spanning no recordings resolves rather than raising."""
        inventory = resolve_dataset_recordings(recording_roots=[], dataset_name="Animal_One")

        assert inventory.dataset_name == "animal_one"
        assert inventory.recording_ids == ()
        assert not inventory.discovered

    def test_dataset_name_is_lowered_in_the_record_and_the_paths(self, tmp_path: Path) -> None:
        """Verifies that the record and every dataset path carry the folded dataset name."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            root.mkdir()

        inventory = resolve_dataset_recordings(recording_roots=roots, dataset_name="Animal_One")

        assert inventory.dataset_name == "animal_one"
        assert all("animal_one" in str(path) for path in inventory.dataset_paths)
        assert inventory.recording_ids == ("day1", "day2")

    def test_extracted_recordings_hold_only_the_recordings_carrying_traces(self, tmp_path: Path) -> None:
        """Verifies that the extracted subset names the recordings whose tracked fluorescence exists."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            root.mkdir()
        dataset_path = resolve_dataset_path(output_root=roots[1], dataset_name="set")
        dataset_path.mkdir(parents=True)
        (dataset_path / RecordingArrays.CELL_FLUORESCENCE).write_bytes(b"")

        inventory = resolve_dataset_recordings(recording_roots=roots, dataset_name="set")

        assert inventory.extracted_recordings == ("day2",)

    def test_discovered_flag_follows_the_first_recording_template_masks(self, tmp_path: Path) -> None:
        """Verifies that the discovery flag follows the template mask archive of the first recording."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            root.mkdir()
        assert not resolve_dataset_recordings(recording_roots=roots, dataset_name="set").discovered

        dataset_path = resolve_dataset_path(output_root=roots[0], dataset_name="set")
        dataset_path.mkdir(parents=True)
        (dataset_path / TRACKING_TEMPLATE_MASKS_FILENAME).write_bytes(b"")

        assert resolve_dataset_recordings(recording_roots=roots, dataset_name="set").discovered


class TestPredicates:
    """Tests the standalone completion predicates."""

    def test_recording_processed_predicate_follows_the_marker(self, tmp_path: Path) -> None:
        """Verifies that the recording predicate follows the combined metadata archive."""
        assert not is_recording_processed(output_root=tmp_path)

        output_path = resolve_output_path(output_root=tmp_path)
        output_path.mkdir(parents=True)
        (output_path / COMBINED_METADATA_FILENAME).write_bytes(b"")

        assert is_recording_processed(output_root=tmp_path)

    @pytest.mark.parametrize("plane_index", [0, 5])
    def test_plane_registered_predicate_follows_the_reference_image(self, tmp_path: Path, plane_index: int) -> None:
        """Verifies that the plane predicate follows the registration reference image."""
        assert not is_plane_registered(output_root=tmp_path, plane_index=plane_index)

        _register_plane(output_root=tmp_path, plane_index=plane_index)

        assert is_plane_registered(output_root=tmp_path, plane_index=plane_index)

    def test_dataset_discovered_predicate_follows_the_template_masks(self, tmp_path: Path) -> None:
        """Verifies that the dataset predicate follows the template mask archive."""
        assert not is_dataset_discovered(output_root=tmp_path, dataset_name="Set")

        dataset_path = resolve_dataset_path(output_root=tmp_path, dataset_name="Set")
        dataset_path.mkdir(parents=True)
        (dataset_path / TRACKING_TEMPLATE_MASKS_FILENAME).write_bytes(b"")

        assert is_dataset_discovered(output_root=tmp_path, dataset_name="Set")


def _write_acquisition(output_root: Path, plane_number: int = 1, roi_number: int = 1) -> None:
    """Writes an acquisition parameters file into the recording's output directory."""
    output_path = resolve_output_path(output_root=output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / ACQUISITION_PARAMETERS_FILENAME).write_text(
        f"frame_rate: 30.0\nplane_number: {plane_number}\nchannel_number: 1\nroi_number: {roi_number}\n"
        f"roi_lines: []\nroi_x_coordinates: []\nroi_y_coordinates: []\n"
    )


def _register_plane(output_root: Path, plane_index: int) -> None:
    """Writes the reference image that marks one plane as registered."""
    directory = resolve_plane_path(output_root=output_root, plane_index=plane_index) / REGISTRATION_DATA_DIRECTORY_NAME
    directory.mkdir(parents=True, exist_ok=True)
    (directory / RegistrationArrays.REFERENCE_IMAGE).write_bytes(b"")
