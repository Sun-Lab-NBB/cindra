"""Contains tests for the results verification MCP tools."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cindra.layout import (
    OUTPUT_DIRECTORY_NAME,
    CHANNEL_1_BINARY_FILENAME,
    COMBINED_METADATA_FILENAME,
    DETECTION_DATA_DIRECTORY_NAME,
    ACQUISITION_PARAMETERS_FILENAME,
    REGISTRATION_DATA_DIRECTORY_NAME,
    SINGLE_RECORDING_RUNTIME_DATA_FILENAME,
    SINGLE_RECORDING_CONFIGURATION_FILENAME,
    DetectionImages,
    RecordingArrays,
    RegistrationArrays,
    resolve_plane_specifier,
)
from cindra.interface import results_tools
from cindra.dataclasses import SingleRecordingConfiguration
from cindra.interface.results_tools import verify_single_recording_output_tool

if TYPE_CHECKING:
    from pathlib import Path

_EXTRACTION_ARRAYS: tuple[str, ...] = (
    RecordingArrays.ROI_MASKS,
    RecordingArrays.ROI_STATISTICS,
    RecordingArrays.CELL_FLUORESCENCE,
    RecordingArrays.NEUROPIL_FLUORESCENCE,
    RecordingArrays.SUBTRACTED_FLUORESCENCE,
    RecordingArrays.SPIKES,
    RecordingArrays.CELL_CLASSIFICATION,
)
"""The extraction outputs the verification tool requires of the combined dataset and of every processed plane."""

_REGISTRATION_ARRAYS: tuple[str, ...] = (
    RegistrationArrays.REFERENCE_IMAGE,
    RegistrationArrays.BAD_FRAMES,
    RegistrationArrays.RIGID_Y_OFFSETS,
    RegistrationArrays.RIGID_X_OFFSETS,
    RegistrationArrays.RIGID_CORRELATIONS,
)
"""The registration outputs the verification tool requires of every registered plane."""


class TestFlybackPlaneVerification:
    """Tests the completeness verdict for a recording whose configuration excludes a plane from processing."""

    def test_unprocessed_flyback_plane_leaves_the_output_complete(self, tmp_path: Path) -> None:
        """Verifies that a flyback plane holding its binarization files alone does not fail the verification."""
        cindra_root = tmp_path / OUTPUT_DIRECTORY_NAME
        _write_configuration(cindra_root=cindra_root, flyback_planes=(1,))
        _populate_recording(cindra_root=cindra_root, plane_count=2, processed_planes={0})

        result = verify_single_recording_output_tool(recording_path=str(tmp_path))

        assert result["complete"] is True
        assert result["missing"] == []
        assert result["flyback_planes"] == [1]

    def test_every_processed_plane_is_still_required(self, tmp_path: Path) -> None:
        """Verifies that the same file set fails verification when no plane is excluded from processing."""
        cindra_root = tmp_path / OUTPUT_DIRECTORY_NAME
        _write_configuration(cindra_root=cindra_root, flyback_planes=())
        _populate_recording(cindra_root=cindra_root, plane_count=2, processed_planes={0})

        result = verify_single_recording_output_tool(recording_path=str(tmp_path))

        assert result["complete"] is False
        assert any(entry.startswith("plane_1/") for entry in result["missing"])
        assert "flyback_planes" not in result


class TestCindraRootResolution:
    """Tests the output directory search the results tools share."""

    def test_denied_marker_scan_falls_back_to_the_tolerant_glob(self, tmp_path: Path, monkeypatch) -> None:
        """Verifies that a refused marker scan still resolves the output directory through the readable tree."""
        nested_root = tmp_path / "session" / OUTPUT_DIRECTORY_NAME
        nested_root.mkdir(parents=True)
        _touch(path=nested_root / SINGLE_RECORDING_CONFIGURATION_FILENAME)
        monkeypatch.setattr(results_tools, "discover_marker_files", _deny_marker_scan)

        cindra_root, error = results_tools._find_cindra_root(recording_path=str(tmp_path))

        assert error is None
        assert cindra_root == nested_root

    def test_denied_marker_scan_reports_the_documented_failure(self, tmp_path: Path, monkeypatch) -> None:
        """Verifies that a tree holding no configuration reports an error dictionary instead of raising."""
        monkeypatch.setattr(results_tools, "discover_marker_files", _deny_marker_scan)

        result = verify_single_recording_output_tool(recording_path=str(tmp_path))

        assert result["success"] is False
        assert "No cindra output directory found" in result["error"]


def _touch(path: Path) -> None:
    """Creates an empty file, together with the directories leading to it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _write_configuration(cindra_root: Path, flyback_planes: tuple[int, ...]) -> None:
    """Saves a single-recording configuration naming the planes the pipeline binarizes without processing."""
    configuration = SingleRecordingConfiguration()
    configuration.main.ignored_flyback_planes = flyback_planes
    configuration.save(file_path=cindra_root / SINGLE_RECORDING_CONFIGURATION_FILENAME)


def _populate_recording(cindra_root: Path, plane_count: int, processed_planes: set[int]) -> None:
    """Writes the output file set of a finished recording, giving unprocessed planes their binarization files alone."""
    _touch(path=cindra_root / ACQUISITION_PARAMETERS_FILENAME)
    _touch(path=cindra_root / COMBINED_METADATA_FILENAME)
    for image in DetectionImages:
        _touch(path=cindra_root / DETECTION_DATA_DIRECTORY_NAME / image)
    for name in _EXTRACTION_ARRAYS:
        _touch(path=cindra_root / name)

    for plane_index in range(plane_count):
        plane_directory = cindra_root / resolve_plane_specifier(plane_index=plane_index)
        _touch(path=plane_directory / SINGLE_RECORDING_RUNTIME_DATA_FILENAME)
        _touch(path=plane_directory / CHANNEL_1_BINARY_FILENAME)
        _touch(path=plane_directory / DETECTION_DATA_DIRECTORY_NAME / DetectionImages.MEAN_IMAGE)

        if plane_index not in processed_planes:
            continue

        for name in _REGISTRATION_ARRAYS:
            _touch(path=plane_directory / REGISTRATION_DATA_DIRECTORY_NAME / name)
        for image in (
            DetectionImages.ENHANCED_MEAN_IMAGE,
            DetectionImages.MAXIMUM_PROJECTION,
            DetectionImages.CORRELATION_MAP,
        ):
            _touch(path=plane_directory / DETECTION_DATA_DIRECTORY_NAME / image)
        for name in _EXTRACTION_ARRAYS:
            _touch(path=plane_directory / name)


def _deny_marker_scan(directory: Path, marker_name: str) -> list[Path]:
    """Refuses the marker scan the way the ataraxis discoverer refuses an unreadable subtree."""
    raise OSError(13, "Permission denied", str(directory))
