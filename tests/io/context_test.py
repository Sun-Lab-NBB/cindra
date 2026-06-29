"""Contains tests for context resolution and path utility functions provided by the context module."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from cindra.io.context import (
    PARAMETERS_FILENAME,
    find_data_directory,
    _find_cindra_directory,
    resolve_recording_roots,
    extract_unique_components,
    _compute_mroi_region_borders,
    _find_acquisition_parameters,
    _load_acquisition_parameters,
)
from cindra.dataclasses.single_recording_configuration import AcquisitionParameters


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


class TestFindDataDirectory:
    """Tests find_data_directory."""

    def test_finds_directory_with_nested_parameters_file(self, tmp_path: Path) -> None:
        """Verifies that the function locates the correct directory when the parameters file is in a subdirectory."""
        nested_directory = tmp_path / "level_1" / "level_2"
        _write_parameters_json(directory=nested_directory, data={"frame_rate": 30.0})

        result = find_data_directory(data_path=tmp_path)

        assert result == nested_directory

    def test_raises_error_when_parameters_file_missing(self, tmp_path: Path) -> None:
        """Verifies that a FileNotFoundError is raised when no parameters file exists in the directory tree."""
        with pytest.raises(FileNotFoundError, match="Unable to find"):
            find_data_directory(data_path=tmp_path)

    def test_raises_error_for_non_directory_path(self, tmp_path: Path) -> None:
        """Verifies that a ValueError is raised when the data_path is not a directory."""
        file_path = tmp_path / "not_a_directory.txt"
        file_path.write_text("content")

        with pytest.raises(ValueError, match="Unable to find data directory"):
            find_data_directory(data_path=file_path)


class TestLoadAcquisitionParameters:
    """Tests _load_acquisition_parameters."""

    def test_loads_valid_single_roi_json(self, tmp_path: Path) -> None:
        """Verifies that a valid single-ROI parameters file is loaded correctly."""
        data = {"frame_rate": 30.0, "plane_number": 2, "channel_number": 1}
        json_path = _write_parameters_json(directory=tmp_path, data=data)

        parameters = _load_acquisition_parameters(json_path=json_path)

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

        parameters = _load_acquisition_parameters(json_path=json_path)

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

        with pytest.raises(ValueError, match="Unable to extract the required field 'channel_number'"):
            _load_acquisition_parameters(json_path=json_path)

    def test_raises_error_for_nonexistent_file(self, tmp_path: Path) -> None:
        """Verifies that a FileNotFoundError is raised when the JSON file does not exist."""
        json_path = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError, match="Unable to load acquisition parameters"):
            _load_acquisition_parameters(json_path=json_path)

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

        with pytest.raises(ValueError, match="Unable to extract the required field 'roi_lines'"):
            _load_acquisition_parameters(json_path=json_path)

    def test_raises_error_for_missing_frame_rate(self, tmp_path: Path) -> None:
        """Verifies that a ValueError is raised when the frame_rate field is missing."""
        data = {"plane_number": 2, "channel_number": 1}
        json_path = _write_parameters_json(directory=tmp_path, data=data)

        with pytest.raises(ValueError, match="Unable to extract the required field 'frame_rate'"):
            _load_acquisition_parameters(json_path=json_path)

    def test_raises_error_for_missing_plane_number(self, tmp_path: Path) -> None:
        """Verifies that a ValueError is raised when the plane_number field is missing."""
        data = {"frame_rate": 30.0, "channel_number": 1}
        json_path = _write_parameters_json(directory=tmp_path, data=data)

        with pytest.raises(ValueError, match="Unable to extract the required field 'plane_number'"):
            _load_acquisition_parameters(json_path=json_path)

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

        with pytest.raises(ValueError, match="Unable to extract the required field 'roi_x_coordinates'"):
            _load_acquisition_parameters(json_path=json_path)

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

        with pytest.raises(ValueError, match="Unable to extract the required field 'roi_y_coordinates'"):
            _load_acquisition_parameters(json_path=json_path)


class TestFindAcquisitionParameters:
    """Tests _find_acquisition_parameters."""

    def test_finds_and_loads_parameters_from_nested_directory(self, tmp_path: Path) -> None:
        """Verifies that the wrapper function correctly discovers and loads acquisition parameters."""
        nested_directory = tmp_path / "data" / "session"
        data = {"frame_rate": 25.0, "plane_number": 3, "channel_number": 1}
        _write_parameters_json(directory=nested_directory, data=data)

        parameters = _find_acquisition_parameters(data_path=tmp_path)

        assert parameters.frame_rate == 25.0
        assert parameters.plane_number == 3


class TestExtractUniqueComponents:
    """Tests extract_unique_components."""

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

        with pytest.raises(RuntimeError, match="Unable to extract a unique component"):
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
    """Tests resolve_recording_roots."""

    def test_resolves_roots_from_nested_paths(self) -> None:
        """Verifies that recording roots are resolved by walking up to the unique component ancestor."""
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


class TestFindCindraDirectory:
    """Tests _find_cindra_directory."""

    def test_finds_directory_with_combined_metadata(self, tmp_path: Path) -> None:
        """Verifies that the cindra output directory is found when combined_metadata.npz exists."""
        cindra_directory = tmp_path / "recording" / "cindra"
        cindra_directory.mkdir(parents=True)
        (cindra_directory / "combined_metadata.npz").write_bytes(b"")

        result = _find_cindra_directory(recording_directory=tmp_path)

        assert result == cindra_directory

    def test_raises_error_when_no_combined_metadata_found(self, tmp_path: Path) -> None:
        """Verifies that a FileNotFoundError is raised when no combined_metadata.npz exists."""
        with pytest.raises(FileNotFoundError, match="Unable to locate cindra output"):
            _find_cindra_directory(recording_directory=tmp_path)

    def test_raises_error_when_multiple_combined_metadata_found(self, tmp_path: Path) -> None:
        """Verifies that a RuntimeError is raised when multiple combined_metadata.npz files exist."""
        for subdirectory_name in ("cindra_1", "cindra_2"):
            subdirectory = tmp_path / subdirectory_name
            subdirectory.mkdir(parents=True)
            (subdirectory / "combined_metadata.npz").write_bytes(b"")

        with pytest.raises(RuntimeError, match="Unable to locate cindra output"):
            _find_cindra_directory(recording_directory=tmp_path)


class TestComputeMroiRegionBorders:
    """Tests _compute_mroi_region_borders."""

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
