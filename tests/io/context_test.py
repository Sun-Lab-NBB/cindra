"""Contains tests for context resolution and path utility functions provided by the context module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from ataraxis_base_utilities import error_format

from cindra.layout import PARAMETERS_FILENAME
from cindra.io.context import (
    OUTPUT_DIRECTORY_NAME,
    find_data_directory,
    find_cindra_directory,
    resolve_recording_roots,
    extract_unique_components,
    load_acquisition_parameters,
    _compute_mroi_region_borders,
    _find_acquisition_parameters,
)
from cindra.dataclasses.single_recording_configuration import AcquisitionParameters


class TestFindDataDirectory:
    """Tests the directory the parameters file search returns, and the missing file and plain file it rejects."""

    def test_finds_directory_with_nested_parameters_file(self, tmp_path: Path) -> None:
        """Verifies that the function locates the correct directory when the parameters file is in a subdirectory."""
        nested_directory = tmp_path / "level_1" / "level_2"
        _write_parameters_json(directory=nested_directory, data={"frame_rate": 30.0})

        result = find_data_directory(data_path=tmp_path)

        assert result == nested_directory

    def test_raises_error_when_parameters_file_missing(self, tmp_path: Path) -> None:
        """Verifies that a FileNotFoundError is raised when no parameters file exists in the directory tree."""
        expected_message = (
            f"Unable to find '{PARAMETERS_FILENAME}' in the data directory or its subdirectories: {tmp_path}. "
            f"This file is required and must contain acquisition metadata."
        )

        with pytest.raises(FileNotFoundError, match=error_format(expected_message)):
            find_data_directory(data_path=tmp_path)

    def test_raises_error_for_non_directory_path(self, tmp_path: Path) -> None:
        """Verifies that a ValueError is raised when the data_path is not a directory."""
        file_path = tmp_path / "not_a_directory.txt"
        file_path.write_text("content")

        expected_message = f"Unable to find data directory. The data_path is not a directory: {file_path}"

        with pytest.raises(ValueError, match=error_format(expected_message)):
            find_data_directory(data_path=file_path)


class TestLoadAcquisitionParameters:
    """Tests the single-ROI and MROI files the loader accepts, and the missing and invalid fields it rejects."""

    def test_loads_valid_single_roi_json(self, tmp_path: Path) -> None:
        """Verifies that a valid single-ROI parameters file is loaded correctly."""
        data = {"frame_rate": 30.0, "plane_number": 2, "channel_number": 1}
        json_path = _write_parameters_json(directory=tmp_path, data=data)

        parameters = load_acquisition_parameters(json_path=json_path)

        assert parameters.frame_rate == 30.0
        assert parameters.plane_number == 2
        assert parameters.channel_number == 1
        assert parameters.roi_number == 1
        assert not parameters.is_mroi

    def test_loads_valid_mroi_json(self, tmp_path: Path) -> None:
        """Verifies that a valid MROI parameters file is loaded correctly with all ROI-specific fields."""
        data = {
            "frame_rate": 15.0,
            "plane_number": 1,
            "channel_number": 2,
            "roi_number": 3,
            "roi_lines": [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
            "roi_x_coordinates": [0, 100, 200],
            "roi_y_coordinates": [10, 20, 30],
        }
        json_path = _write_parameters_json(directory=tmp_path, data=data)

        parameters = load_acquisition_parameters(json_path=json_path)

        assert parameters.frame_rate == 15.0
        assert parameters.roi_number == 3
        assert parameters.is_mroi
        assert parameters.roi_x_coordinates == [0, 100, 200]
        assert parameters.roi_y_coordinates == [10, 20, 30]
        assert parameters.roi_lines == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    def test_raises_error_for_missing_required_field(self, tmp_path: Path) -> None:
        """Verifies that a ValueError is raised when a required field is missing from the JSON data."""
        data = {"frame_rate": 30.0, "plane_number": 2}
        json_path = _write_parameters_json(directory=tmp_path, data=data)

        expected_message = (
            f"Unable to extract the required field 'channel_number' from the acquisition parameters file "
            f"located at {json_path}."
        )

        with pytest.raises(ValueError, match=error_format(expected_message)):
            load_acquisition_parameters(json_path=json_path)

    def test_raises_error_for_nonexistent_file(self, tmp_path: Path) -> None:
        """Verifies that a FileNotFoundError is raised when the JSON file does not exist."""
        json_path = tmp_path / "nonexistent.json"
        expected_message = f"Unable to load acquisition parameters. The file was not found: {json_path}."

        with pytest.raises(FileNotFoundError, match=error_format(expected_message)):
            load_acquisition_parameters(json_path=json_path)

    def test_raises_error_for_mroi_missing_roi_lines(self, tmp_path: Path) -> None:
        """Verifies that a ValueError is raised when an MROI recording is missing the roi_lines field."""
        data = {
            "frame_rate": 15.0,
            "plane_number": 1,
            "channel_number": 1,
            "roi_number": 2,
            "roi_x_coordinates": [0, 100],
            "roi_y_coordinates": [10, 20],
        }
        json_path = _write_parameters_json(directory=tmp_path, data=data)

        expected_message = (
            f"Unable to extract the required field 'roi_lines' from the acquisition parameters file "
            f"located at {json_path}."
        )

        with pytest.raises(ValueError, match=error_format(expected_message)):
            load_acquisition_parameters(json_path=json_path)

    def test_raises_error_for_missing_frame_rate(self, tmp_path: Path) -> None:
        """Verifies that a ValueError is raised when the frame_rate field is missing."""
        data = {"plane_number": 2, "channel_number": 1}
        json_path = _write_parameters_json(directory=tmp_path, data=data)

        expected_message = (
            f"Unable to extract the required field 'frame_rate' from the acquisition parameters file "
            f"located at {json_path}."
        )

        with pytest.raises(ValueError, match=error_format(expected_message)):
            load_acquisition_parameters(json_path=json_path)

    def test_raises_error_for_missing_plane_number(self, tmp_path: Path) -> None:
        """Verifies that a ValueError is raised when the plane_number field is missing."""
        data = {"frame_rate": 30.0, "channel_number": 1}
        json_path = _write_parameters_json(directory=tmp_path, data=data)

        expected_message = (
            f"Unable to extract the required field 'plane_number' from the acquisition parameters file "
            f"located at {json_path}."
        )

        with pytest.raises(ValueError, match=error_format(expected_message)):
            load_acquisition_parameters(json_path=json_path)

    def test_raises_error_for_mroi_missing_roi_x_coordinates(self, tmp_path: Path) -> None:
        """Verifies that a ValueError is raised when an MROI recording is missing roi_x_coordinates."""
        data = {
            "frame_rate": 15.0,
            "plane_number": 1,
            "channel_number": 1,
            "roi_number": 2,
            "roi_lines": [[0, 1], [2, 3]],
            "roi_y_coordinates": [10, 20],
        }
        json_path = _write_parameters_json(directory=tmp_path, data=data)

        expected_message = (
            f"Unable to extract the required field 'roi_x_coordinates' from the acquisition parameters "
            f"file located at {json_path}."
        )

        with pytest.raises(ValueError, match=error_format(expected_message)):
            load_acquisition_parameters(json_path=json_path)

    def test_raises_error_for_mroi_missing_roi_y_coordinates(self, tmp_path: Path) -> None:
        """Verifies that a ValueError is raised when an MROI recording is missing roi_y_coordinates."""
        data = {
            "frame_rate": 15.0,
            "plane_number": 1,
            "channel_number": 1,
            "roi_number": 2,
            "roi_lines": [[0, 1], [2, 3]],
            "roi_x_coordinates": [0, 100],
        }
        json_path = _write_parameters_json(directory=tmp_path, data=data)

        expected_message = (
            f"Unable to extract the required field 'roi_y_coordinates' from the acquisition parameters "
            f"file located at {json_path}."
        )

        with pytest.raises(ValueError, match=error_format(expected_message)):
            load_acquisition_parameters(json_path=json_path)

    def test_raises_error_for_zero_frame_rate(self, tmp_path: Path) -> None:
        """Verifies that a non-positive frame rate is rejected when the file is loaded."""
        data = {"frame_rate": 0.0, "plane_number": 2, "channel_number": 1}
        json_path = _write_parameters_json(directory=tmp_path, data=data)

        expected_message = (
            f"Unable to load the acquisition parameters stored inside the file located at {json_path}. The "
            f"'frame_rate' field must be a positive number, but it is 0.0."
        )

        with pytest.raises(ValueError, match=error_format(expected_message)):
            load_acquisition_parameters(json_path=json_path)

    def test_raises_error_for_zero_plane_number(self, tmp_path: Path) -> None:
        """Verifies that a zero plane count is rejected instead of dividing the frame rate by zero downstream."""
        data = {"frame_rate": 30.0, "plane_number": 0, "channel_number": 1}
        json_path = _write_parameters_json(directory=tmp_path, data=data)

        expected_message = (
            f"Unable to load the acquisition parameters stored inside the file located at {json_path}. The "
            f"'plane_number' field must be a positive integer, but it is 0."
        )

        with pytest.raises(ValueError, match=error_format(expected_message)):
            load_acquisition_parameters(json_path=json_path)

    def test_raises_error_for_negative_channel_number(self, tmp_path: Path) -> None:
        """Verifies that a negative channel count is rejected when the file is loaded."""
        data = {"frame_rate": 30.0, "plane_number": 2, "channel_number": -1}
        json_path = _write_parameters_json(directory=tmp_path, data=data)

        expected_message = (
            f"Unable to load the acquisition parameters stored inside the file located at {json_path}. The "
            f"'channel_number' field must be a positive integer, but it is -1."
        )

        with pytest.raises(ValueError, match=error_format(expected_message)):
            load_acquisition_parameters(json_path=json_path)

    def test_raises_error_for_zero_roi_number(self, tmp_path: Path) -> None:
        """Verifies that a zero ROI count is rejected when the file is loaded."""
        data = {"frame_rate": 30.0, "plane_number": 2, "channel_number": 1, "roi_number": 0}
        json_path = _write_parameters_json(directory=tmp_path, data=data)

        expected_message = (
            f"Unable to load the acquisition parameters stored inside the file located at {json_path}. The "
            f"'roi_number' field must be a positive integer, but it is 0."
        )

        with pytest.raises(ValueError, match=error_format(expected_message)):
            load_acquisition_parameters(json_path=json_path)

    def test_raises_error_for_non_numeric_plane_number(self, tmp_path: Path) -> None:
        """Verifies that a plane count written as text is rejected when the file is loaded."""
        data = {"frame_rate": 30.0, "plane_number": "two", "channel_number": 1}
        json_path = _write_parameters_json(directory=tmp_path, data=data)

        expected_message = (
            f"Unable to load the acquisition parameters stored inside the file located at {json_path}. The "
            f"'plane_number' field must be a positive integer, but it is two."
        )

        with pytest.raises(ValueError, match=error_format(expected_message)):
            load_acquisition_parameters(json_path=json_path)


class TestFindAcquisitionParameters:
    """Tests the nested discovery and load the acquisition parameters wrapper performs for a data path."""

    def test_finds_and_loads_parameters_from_nested_directory(self, tmp_path: Path) -> None:
        """Verifies that the wrapper function correctly discovers and loads acquisition parameters."""
        nested_directory = tmp_path / "data" / "session"
        data = {"frame_rate": 25.0, "plane_number": 3, "channel_number": 1}
        _write_parameters_json(directory=nested_directory, data=data)

        parameters = _find_acquisition_parameters(data_path=tmp_path)

        assert parameters.frame_rate == 25.0
        assert parameters.plane_number == 3


class TestExtractUniqueComponents:
    """Tests the unique component each path contributes as its tracker specifier, and the paths it rejects."""

    def test_extracts_unique_leaf_directories(self) -> None:
        """Verifies that unique leaf directory names are extracted when they differ between paths."""
        paths = [Path("/a/rec1"), Path("/b/rec2")]

        result = extract_unique_components(paths=paths)

        assert result == ("rec1", "rec2")

    def test_extracts_unique_parent_directories(self) -> None:
        """Verifies that unique parent directory names are extracted when leaf names are shared."""
        paths = [Path("/data/day1/recording"), Path("/data/day2/recording")]

        result = extract_unique_components(paths=paths)

        assert result == ("day1", "day2")

    def test_raises_error_for_paths_with_no_unique_components(self) -> None:
        """Verifies that a RuntimeError is raised when paths share all components but are not identical."""
        # Both paths contain exactly the same set of components ("a" and "b"), so neither has a unique one.
        paths = [Path("/a/b"), Path("/b/a")]
        expected_message = f"Unable to extract a unique component from the given path: {paths[0]}."

        with pytest.raises(RuntimeError, match=error_format(expected_message)):
            extract_unique_components(paths=paths)

    def test_raises_error_for_duplicate_paths(self) -> None:
        """Verifies that the same directory listed twice is rejected rather than yielding two identical specifiers."""
        # Each returned component becomes a tracker specifier, so two identical components would collapse the two
        # extraction jobs of the dataset onto one tracker record.
        paths = [Path("/data/day1"), Path("/data/day1")]
        expected_message = f"Unable to extract a unique component from the given path: {paths[0]}."

        with pytest.raises(RuntimeError, match=error_format(expected_message)):
            extract_unique_components(paths=paths)

    def test_raises_error_for_component_containing_a_colon(self) -> None:
        """Verifies that a RuntimeError is raised when the resolved unique component contains a colon."""
        # The tracker joins a job name to its specifier with a colon, so a component carrying one is rejected here.
        paths = [Path("/data/day:1/rec"), Path("/data/day2/rec")]
        expected_message = (
            f"Unable to extract a unique component from the given path: {paths[0]}. The resolved component "
            f"'day:1' contains the ':' character, which the tracker reserves for joining a job "
            f"name to its specifier. Rename the directory to remove the character."
        )

        with pytest.raises(RuntimeError, match=error_format(expected_message)):
            extract_unique_components(paths=paths)

    def test_three_paths_with_unique_components(self) -> None:
        """Verifies correct extraction when three paths each have a unique identifying component."""
        paths = [
            Path("/experiment/mouse_1/session"),
            Path("/experiment/mouse_2/session"),
            Path("/experiment/mouse_3/session"),
        ]

        result = extract_unique_components(paths=paths)

        assert result == ("mouse_1", "mouse_2", "mouse_3")


class TestResolveRecordingRoots:
    """Tests the recording roots resolved from output directories, shared leaf names, and duplicate paths."""

    def test_resolves_roots_from_nested_paths(self) -> None:
        """Verifies that recording roots are resolved by stripping the trailing components shared by every path."""
        paths = [
            Path("/data/day1/recording/cindra/plane_0"),
            Path("/data/day2/recording/cindra/plane_0"),
        ]

        result = resolve_recording_roots(paths=paths)

        assert result == (Path("/data/day1"), Path("/data/day2"))

    def test_resolves_roots_when_leaf_is_unique(self) -> None:
        """Verifies that recording roots match the full paths when the leaf directories are already unique."""
        paths = [
            Path("/data/session_a"),
            Path("/data/session_b"),
        ]

        result = resolve_recording_roots(paths=paths)

        assert result == (Path("/data/session_a"), Path("/data/session_b"))

    def test_deduplicates_identical_paths(self) -> None:
        """Verifies that two identical paths collapse into a single deduplicated recording root."""
        paths = [
            Path("/data/session_a"),
            Path("/data/session_a"),
        ]

        result = resolve_recording_roots(paths=paths)

        assert result == (Path("/data/session_a"),)

    def test_resolves_single_output_directory_to_its_parent(self) -> None:
        """Verifies that a lone output directory resolves to the recording root that contains it."""
        paths = [Path(f"/data/rec1/{OUTPUT_DIRECTORY_NAME}")]

        result = resolve_recording_roots(paths=paths)

        assert result == (Path("/data/rec1"),)

    def test_preserves_single_path_outside_the_output_directory(self) -> None:
        """Verifies that a lone raw-data directory is returned unchanged, because it has no peer for comparison."""
        paths = [Path("/data/rec1/raw_data")]

        result = resolve_recording_roots(paths=paths)

        assert result == (Path("/data/rec1/raw_data"),)

    def test_resolves_multiple_output_directories_to_their_parents(self) -> None:
        """Verifies that every output directory in a batch resolves to its own recording root."""
        paths = [
            Path(f"/data/rec1/{OUTPUT_DIRECTORY_NAME}"),
            Path(f"/data/rec2/{OUTPUT_DIRECTORY_NAME}"),
        ]

        result = resolve_recording_roots(paths=paths)

        assert result == (Path("/data/rec1"), Path("/data/rec2"))

    def test_retains_shared_parent_of_output_directories(self) -> None:
        """Verifies that output directories nested under a shared subdirectory keep that subdirectory in the root."""
        paths = [
            Path(f"/data/rec1/processed/{OUTPUT_DIRECTORY_NAME}"),
            Path(f"/data/rec2/processed/{OUTPUT_DIRECTORY_NAME}"),
        ]

        result = resolve_recording_roots(paths=paths)

        assert result == (Path("/data/rec1/processed"), Path("/data/rec2/processed"))

    def test_deduplicates_identical_output_directories(self) -> None:
        """Verifies that two identical output directories collapse into a single recording root."""
        paths = [
            Path(f"/data/rec1/{OUTPUT_DIRECTORY_NAME}"),
            Path(f"/data/rec1/{OUTPUT_DIRECTORY_NAME}"),
        ]

        result = resolve_recording_roots(paths=paths)

        assert result == (Path("/data/rec1"),)

    def test_resolves_mixed_output_and_raw_directories(self) -> None:
        """Verifies that an output directory resolves to its parent while a raw-data peer is left untouched."""
        paths = [
            Path(f"/data/rec1/{OUTPUT_DIRECTORY_NAME}"),
            Path("/data/rec2/raw_data"),
        ]

        result = resolve_recording_roots(paths=paths)

        assert result == (Path("/data/rec1"), Path("/data/rec2/raw_data"))

    def test_strips_shared_leaf_when_parents_differ(self) -> None:
        """Verifies that a leaf name shared by every raw-data directory is stripped to expose the differing parents."""
        paths = [
            Path("/data/mouse_1/raw_data"),
            Path("/data/mouse_2/raw_data"),
        ]

        result = resolve_recording_roots(paths=paths)

        assert result == (Path("/data/mouse_1"), Path("/data/mouse_2"))

    def test_retains_recordings_stored_under_a_cindra_named_ancestor(self) -> None:
        """Verifies that recordings kept under a directory named after the output directory stay distinct."""
        paths = [
            Path(f"/home/user/{OUTPUT_DIRECTORY_NAME}/data/rec1"),
            Path(f"/home/user/{OUTPUT_DIRECTORY_NAME}/data/rec2"),
        ]

        result = resolve_recording_roots(paths=paths)

        assert result == (
            Path(f"/home/user/{OUTPUT_DIRECTORY_NAME}/data/rec1"),
            Path(f"/home/user/{OUTPUT_DIRECTORY_NAME}/data/rec2"),
        )

    def test_resolves_paths_that_share_every_component(self) -> None:
        """Verifies that paths built from the same component names in different orders are returned unchanged."""
        paths = [
            Path("/data/a/b"),
            Path("/data/b/a"),
        ]

        result = resolve_recording_roots(paths=paths)

        assert result == (Path("/data/a/b"), Path("/data/b/a"))


class TestFindCindraDirectory:
    """Tests the output directory the metadata search returns, and the absent and ambiguous trees it rejects."""

    def test_finds_directory_with_combined_metadata(self, tmp_path: Path) -> None:
        """Verifies that the cindra output directory is found when combined_metadata.npz exists."""
        cindra_directory = tmp_path / "recording" / "cindra"
        cindra_directory.mkdir(parents=True)
        (cindra_directory / "combined_metadata.npz").write_bytes(b"")

        result = find_cindra_directory(recording_directory=tmp_path)

        assert result == cindra_directory

    def test_raises_error_when_no_combined_metadata_found(self, tmp_path: Path) -> None:
        """Verifies that a FileNotFoundError is raised when no combined_metadata.npz exists."""
        expected_message = (
            f"Unable to locate cindra output for recording {tmp_path}. No "
            f"combined_metadata.npz file was found anywhere in the directory tree. Ensure the "
            f"single-recording pipeline has completed successfully for this recording before running "
            f"multi-recording processing."
        )

        with pytest.raises(FileNotFoundError, match=error_format(expected_message)):
            find_cindra_directory(recording_directory=tmp_path)

    def test_raises_error_when_multiple_combined_metadata_found(self, tmp_path: Path) -> None:
        """Verifies that a RuntimeError is raised when multiple combined_metadata.npz files exist."""
        for subdirectory_name in ("cindra_1", "cindra_2"):
            subdirectory = tmp_path / subdirectory_name
            subdirectory.mkdir(parents=True)
            (subdirectory / "combined_metadata.npz").write_bytes(b"")

        expected_message = (
            f"Unable to locate cindra output for recording {tmp_path}. Found 2 "
            f"combined_metadata.npz files, but expected exactly one unique match."
        )

        with pytest.raises(RuntimeError, match=error_format(expected_message)):
            find_cindra_directory(recording_directory=tmp_path)


class TestComputeMroiRegionBorders:
    """Tests the sorted region borders an MROI recording yields, and the empty result a non-MROI recording gives."""

    def test_returns_empty_tuple_for_non_mroi(self, tmp_path: Path) -> None:
        """Verifies that a non-MROI recording returns an empty tuple."""
        acquisition = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        _write_acquisition_yaml(directory=tmp_path, acquisition=acquisition)

        result = _compute_mroi_region_borders(data_path=tmp_path)

        assert result == ()

    def test_returns_sorted_borders_for_mroi(self, tmp_path: Path) -> None:
        """Verifies that MROI recordings return sorted x-coordinates excluding the minimum."""
        acquisition = AcquisitionParameters(
            frame_rate=30.0,
            plane_number=1,
            channel_number=1,
            roi_number=3,
            roi_lines=((0, 1), (2, 3), (4, 5)),
            roi_x_coordinates=(10, 50, 100),
            roi_y_coordinates=(0, 0, 0),
        )
        _write_acquisition_yaml(directory=tmp_path, acquisition=acquisition)

        result = _compute_mroi_region_borders(data_path=tmp_path)

        assert result == (50, 100)


def _write_parameters_json(directory: Path, data: dict[str, object]) -> Path:
    """Writes a cindra_parameters.json file to the given directory."""
    directory.mkdir(parents=True, exist_ok=True)
    file_path = directory / PARAMETERS_FILENAME
    file_path.write_text(json.dumps(data))
    return file_path


def _write_acquisition_yaml(directory: Path, acquisition: AcquisitionParameters) -> None:
    """Saves an AcquisitionParameters instance as acquisition_parameters.yaml in the given directory."""
    directory.mkdir(parents=True, exist_ok=True)
    acquisition.to_yaml(file_path=directory / "acquisition_parameters.yaml")
