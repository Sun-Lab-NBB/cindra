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

_OPTIONAL_REGISTRATION_ARRAYS: tuple[str, ...] = (
    RegistrationArrays.NONRIGID_Y_OFFSETS,
    RegistrationArrays.NONRIGID_X_OFFSETS,
    RegistrationArrays.NONRIGID_CORRELATIONS,
    RegistrationArrays.PRINCIPAL_COMPONENT_EXTREME_IMAGES,
    RegistrationArrays.PRINCIPAL_COMPONENT_PROJECTIONS,
    RegistrationArrays.PRINCIPAL_COMPONENT_SHIFT_METRICS,
)
"""The registration outputs the verification tool accepts as absent, which a short or rigid-only run never writes."""

_PRINCIPAL_COMPONENT_ARRAYS: tuple[str, ...] = (
    RegistrationArrays.PRINCIPAL_COMPONENT_EXTREME_IMAGES,
    RegistrationArrays.PRINCIPAL_COMPONENT_PROJECTIONS,
    RegistrationArrays.PRINCIPAL_COMPONENT_SHIFT_METRICS,
)
"""The registration-quality outputs a recording holding fewer than 1500 frames never produces."""


class TestFlybackPlaneVerification:
    """Tests the completeness verdict for a recording whose configuration excludes a plane from processing."""

    def test_unprocessed_flyback_plane_leaves_the_output_complete(self, tmp_path: Path) -> None:
        """Verifies that a flyback plane holding its binarization files alone does not fail the verification."""
        cindra_root = tmp_path / OUTPUT_DIRECTORY_NAME
        _write_configuration(cindra_root=cindra_root, flyback_planes=(1,))
        _populate_recording(cindra_root=cindra_root, plane_count=2, processed_planes={0})

        result = verify_single_recording_output_tool(output_root=str(tmp_path))

        assert result["complete"] is True
        assert result["missing"] == []
        assert result["output_root"] == str(tmp_path)
        assert result["flyback_planes"] == [1]

    def test_every_processed_plane_is_still_required(self, tmp_path: Path) -> None:
        """Verifies that the same file set fails verification when no plane is excluded from processing."""
        cindra_root = tmp_path / OUTPUT_DIRECTORY_NAME
        _write_configuration(cindra_root=cindra_root, flyback_planes=())
        _populate_recording(cindra_root=cindra_root, plane_count=2, processed_planes={0})

        result = verify_single_recording_output_tool(output_root=str(tmp_path))

        assert result["complete"] is False
        assert any(entry.startswith("plane_1/") for entry in result["missing"])
        assert "flyback_planes" not in result


class TestOptionalOutputAccounting:
    """Tests the failure count and the optional inventory the verification tool derives from one file sweep."""

    def test_failed_count_matches_the_missing_list(self, tmp_path: Path) -> None:
        """Verifies that the reported failure count covers the required checks alone in both verdicts."""
        complete_root = tmp_path / "complete"
        incomplete_root = tmp_path / "incomplete"
        for output_root, flyback_planes in ((complete_root, (1,)), (incomplete_root, ())):
            cindra_root = output_root / OUTPUT_DIRECTORY_NAME
            _write_configuration(cindra_root=cindra_root, flyback_planes=flyback_planes)
            _populate_recording(cindra_root=cindra_root, plane_count=2, processed_planes={0})

        complete_result = verify_single_recording_output_tool(output_root=str(complete_root))
        incomplete_result = verify_single_recording_output_tool(output_root=str(incomplete_root))

        assert complete_result["failed"] == 0
        assert complete_result["failed"] == len(complete_result["missing"])
        assert incomplete_result["failed"] > 0
        assert incomplete_result["failed"] == len(incomplete_result["missing"])
        assert incomplete_result["passed"] + incomplete_result["failed"] < incomplete_result["total_checks"]

    def test_skipped_registration_metrics_are_reported_as_optionally_absent(self, tmp_path: Path) -> None:
        """Verifies that the principal-component arrays a short recording never writes leave the output complete."""
        cindra_root = tmp_path / OUTPUT_DIRECTORY_NAME
        _write_configuration(cindra_root=cindra_root, flyback_planes=())
        _populate_recording(cindra_root=cindra_root, plane_count=1, processed_planes={0})

        result = verify_single_recording_output_tool(output_root=str(tmp_path))

        optional_absent = result["optional_absent"]
        assert result["complete"] is True
        assert all(f"plane_0/registration_data/{name}" in optional_absent for name in _PRINCIPAL_COMPONENT_ARRAYS)
        assert all(entry not in result["missing"] for entry in optional_absent)

    def test_recording_holding_every_optional_output_omits_the_key(self, tmp_path: Path) -> None:
        """Verifies that the optional inventory is reported only when the recording is missing an optional output."""
        cindra_root = tmp_path / OUTPUT_DIRECTORY_NAME
        _write_configuration(cindra_root=cindra_root, flyback_planes=())
        _populate_recording(
            cindra_root=cindra_root, plane_count=1, processed_planes={0}, include_optional_registration=True
        )

        result = verify_single_recording_output_tool(output_root=str(tmp_path))

        assert result["complete"] is True
        assert "optional_absent" not in result
        assert result["passed"] == result["total_checks"]


class TestCindraRootResolution:
    """Tests the output directory search the results tools share."""

    def test_denied_marker_scan_falls_back_to_the_tolerant_glob(self, tmp_path: Path, monkeypatch) -> None:
        """Verifies that a refused marker scan still resolves the output directory through the readable tree."""
        nested_root = tmp_path / "session" / OUTPUT_DIRECTORY_NAME
        nested_root.mkdir(parents=True)
        _touch(path=nested_root / SINGLE_RECORDING_CONFIGURATION_FILENAME)
        monkeypatch.setattr(results_tools, "discover_marker_files", _deny_marker_scan)

        cindra_root, error = results_tools._find_cindra_root(output_root=str(tmp_path))

        assert error is None
        assert cindra_root == nested_root

    def test_denied_marker_scan_reports_the_documented_failure(self, tmp_path: Path, monkeypatch) -> None:
        """Verifies that a tree holding no configuration reports an error dictionary instead of raising."""
        monkeypatch.setattr(results_tools, "discover_marker_files", _deny_marker_scan)

        result = verify_single_recording_output_tool(output_root=str(tmp_path))

        assert result["success"] is False
        assert "No cindra output directory found" in result["error"]

    def test_absent_output_root_is_reported(self, tmp_path: Path) -> None:
        """Verifies that a path holding no directory names the output root the caller passed."""
        result = verify_single_recording_output_tool(output_root=str(tmp_path / "absent"))

        assert result["success"] is False
        assert "Output root directory not found" in result["error"]


def _touch(path: Path) -> None:
    """Creates an empty file, together with the directories leading to it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def _write_configuration(cindra_root: Path, flyback_planes: tuple[int, ...]) -> None:
    """Saves a single-recording configuration naming the planes the pipeline binarizes without processing."""
    configuration = SingleRecordingConfiguration()
    configuration.main.ignored_flyback_planes = flyback_planes
    configuration.save(file_path=cindra_root / SINGLE_RECORDING_CONFIGURATION_FILENAME)


def _populate_recording(
    cindra_root: Path,
    plane_count: int,
    processed_planes: set[int],
    *,
    include_optional_registration: bool = False,
) -> None:
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
        if include_optional_registration:
            for name in _OPTIONAL_REGISTRATION_ARRAYS:
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
