"""Contains tests for the layout module."""

from __future__ import annotations

from pathlib import Path

import pytest

from cindra.layout import (
    OUTPUT_DIRECTORY_NAME,
    CHANNEL_2_ARRAY_SUFFIX,
    PLANE_SPECIFIER_PREFIX,
    BINARIZATION_MARKER_SUFFIX,
    COMBINED_METADATA_FILENAME,
    REGISTRATION_MARKER_SUFFIX,
    MULTI_RECORDING_DIRECTORY_NAME,
    RecordingArrays,
    resolve_array_path,
    resolve_plane_path,
    resolve_output_path,
    resolve_dataset_path,
    parse_plane_specifier,
    resolve_plane_specifier,
    resolve_binarization_marker_name,
    resolve_registration_marker_name,
)


class TestPathResolvers:
    """Tests the pure path resolvers."""

    def test_output_path_appends_the_output_directory(self) -> None:
        """Verifies that the output path is the output directory inside the caller's root."""
        assert resolve_output_path(output_root=Path("/data/session")) == Path("/data/session") / OUTPUT_DIRECTORY_NAME

    def test_plane_path_nests_the_specifier_under_the_output_directory(self) -> None:
        """Verifies that a plane directory sits inside the output directory under its specifier."""
        resolved = resolve_plane_path(output_root=Path("/data/session"), plane_index=3)

        assert resolved == Path("/data/session") / OUTPUT_DIRECTORY_NAME / "plane_3"

    def test_dataset_path_lowers_the_dataset_name(self) -> None:
        """Verifies that the dataset directory applies the case fold the context resolver applies."""
        resolved = resolve_dataset_path(output_root=Path("/data/session"), dataset_name="Animal_One")

        assert resolved == (
            Path("/data/session") / OUTPUT_DIRECTORY_NAME / MULTI_RECORDING_DIRECTORY_NAME / "animal_one"
        )

    def test_binarization_marker_name_appends_the_suffix(self) -> None:
        """Verifies that the binarization marker name is the binary name plus the binarization suffix."""
        assert resolve_binarization_marker_name(binary_name="channel_1_data.bin") == (
            f"channel_1_data.bin{BINARIZATION_MARKER_SUFFIX}"
        )

    def test_registration_marker_name_appends_the_suffix(self) -> None:
        """Verifies that the registration marker name is the binary name plus the registration suffix."""
        assert resolve_registration_marker_name(binary_name="channel_1_data.bin") == (
            f"channel_1_data.bin{REGISTRATION_MARKER_SUFFIX}"
        )

    def test_marker_suffixes_match_the_reported_job_statuses(self) -> None:
        """Verifies that each marker suffix spells its phase the way the reported job status spells it."""
        assert BINARIZATION_MARKER_SUFFIX == ".binarizing"
        assert REGISTRATION_MARKER_SUFFIX == ".registering"


class TestArrayPaths:
    """Tests the result array path resolver."""

    def test_functional_channel_array_resolves_under_the_root(self) -> None:
        """Verifies that a functional channel array is named exactly as the enum spells it."""
        resolved = resolve_array_path(root_path=Path("/out"), array=RecordingArrays.CELL_FLUORESCENCE)

        assert resolved == Path("/out/cell_fluorescence.npy")

    def test_second_channel_array_carries_the_channel_suffix(self) -> None:
        """Verifies that the second channel copy carries the channel suffix before the extension."""
        resolved = resolve_array_path(
            root_path=Path("/out"), array=RecordingArrays.CELL_FLUORESCENCE, second_channel=True
        )

        assert resolved == Path(f"/out/cell_fluorescence{CHANNEL_2_ARRAY_SUFFIX}.npy")

    def test_second_channel_suffix_applies_to_archives(self) -> None:
        """Verifies that the channel suffix is inserted before an archive extension as well."""
        resolved = resolve_array_path(root_path=Path("/out"), array=RecordingArrays.ROI_MASKS, second_channel=True)

        assert resolved == Path(f"/out/roi_masks{CHANNEL_2_ARRAY_SUFFIX}.npz")

    def test_combined_metadata_filename_is_the_completion_marker(self) -> None:
        """Verifies that the combined metadata filename keeps the name the combination stage publishes."""
        assert COMBINED_METADATA_FILENAME == "combined_metadata.npz"


class TestPlaneSpecifiers:
    """Tests the plane specifier round trip."""

    @pytest.mark.parametrize("plane_index", [0, 1, 7, 42])
    def test_specifier_round_trips_through_the_parser(self, plane_index: int) -> None:
        """Verifies that parsing a resolved specifier recovers the plane index it was built from."""
        assert parse_plane_specifier(specifier=resolve_plane_specifier(plane_index=plane_index)) == plane_index

    def test_specifier_carries_the_shared_prefix(self) -> None:
        """Verifies that a resolved specifier starts with the prefix the directory names share."""
        assert resolve_plane_specifier(plane_index=2).startswith(PLANE_SPECIFIER_PREFIX)

    @pytest.mark.parametrize("specifier", ["", "recording_1", "plane", "planes_1"])
    def test_specifier_without_the_prefix_resolves_to_none(self, specifier: str) -> None:
        """Verifies that a specifier naming no plane resolves to None rather than raising."""
        assert parse_plane_specifier(specifier=specifier) is None

    @pytest.mark.parametrize("specifier", ["plane_", "plane_x", "plane_1a", "plane_-1"])
    def test_specifier_without_a_numeric_index_resolves_to_none(self, specifier: str) -> None:
        """Verifies that a prefixed specifier carrying no whole index resolves to None."""
        assert parse_plane_specifier(specifier=specifier) is None
