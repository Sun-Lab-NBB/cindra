"""Contains tests for path relocation functions provided by the runtime_contexts module."""

from __future__ import annotations

from pathlib import Path

import pytest

from cindra.dataclasses.runtime_contexts import _compute_relocation_prefixes, _relocate_cross_recording_path


class TestComputeRelocationPrefixes:
    """Tests _compute_relocation_prefixes."""

    def test_paths_with_shared_suffix_returns_diverging_prefixes(self) -> None:
        """Verifies that paths sharing a common suffix produce the correct diverging prefixes."""
        old_path = Path("/old/root/project/cindra/plane_0")
        new_path = Path("/new/location/project/cindra/plane_0")

        old_prefix, new_prefix = _compute_relocation_prefixes(old_path=old_path, new_path=new_path)

        assert old_prefix == Path("/old/root")
        assert new_prefix == Path("/new/location")

    def test_paths_with_no_common_suffix_returns_original_paths(self) -> None:
        """Verifies that paths with no common trailing components return the original paths unchanged."""
        old_path = Path("/alpha/beta/gamma")
        new_path = Path("/delta/epsilon/zeta")

        old_prefix, new_prefix = _compute_relocation_prefixes(old_path=old_path, new_path=new_path)

        assert old_prefix == old_path
        assert new_prefix == new_path

    def test_identical_paths_returns_root_prefixes(self) -> None:
        """Verifies that identical paths produce single-component root prefixes since the entire path is a common
        suffix.
        """
        path = Path("/data/recordings/cindra/plane_0")

        old_prefix, new_prefix = _compute_relocation_prefixes(old_path=path, new_path=path)

        # Identical paths share every component, so neither prefix retains content beyond the root.
        assert old_prefix == new_prefix

    def test_single_component_difference(self) -> None:
        """Verifies correct behavior when only the first path component differs between old and new paths."""
        old_path = Path("/old_drive/shared/data/output")
        new_path = Path("/new_drive/shared/data/output")

        old_prefix, new_prefix = _compute_relocation_prefixes(old_path=old_path, new_path=new_path)

        assert old_prefix == Path("/old_drive")
        assert new_prefix == Path("/new_drive")

    def test_different_length_paths_with_shared_suffix(self) -> None:
        """Verifies that paths of different lengths with a shared suffix produce correct prefixes."""
        old_path = Path("/short/cindra/plane_0")
        new_path = Path("/much/longer/path/cindra/plane_0")

        old_prefix, new_prefix = _compute_relocation_prefixes(old_path=old_path, new_path=new_path)

        assert old_prefix == Path("/short")
        assert new_prefix == Path("/much/longer/path")


class TestRelocateCrossRecordingPath:
    """Tests _relocate_cross_recording_path."""

    def test_simple_path_relocation(self) -> None:
        """Verifies that a cross-recording path with the same base structure is correctly relocated."""
        path = Path("/old/root/recording_1/cindra/plane_0/output")
        old_prefix = Path("/old/root")
        new_prefix = Path("/new/location")

        relocated = _relocate_cross_recording_path(path=path, old_prefix=old_prefix, new_prefix=new_prefix)

        assert relocated == Path("/new/location/recording_1/cindra/plane_0/output")

    def test_path_with_different_recording_segment(self) -> None:
        """Verifies that a cross-recording path with a different recording directory segment preserves that segment
        in the relocated path.
        """
        # The entry recording had old_prefix=/old/root from recording_1, but this cross-recording path belongs to
        # recording_2, which has a different second segment.
        path = Path("/old/root_other/recording_2/cindra/plane_0")
        old_prefix = Path("/old/root")
        new_prefix = Path("/new/location")

        relocated = _relocate_cross_recording_path(path=path, old_prefix=old_prefix, new_prefix=new_prefix)

        # The differing segment "root_other" should be substituted into the new prefix at the corresponding position.
        assert relocated == Path("/new/root_other/recording_2/cindra/plane_0")

    def test_raises_error_for_path_shorter_than_prefix(self) -> None:
        """Verifies that a ValueError is raised when the cross-recording path has fewer segments than the prefix."""
        path = Path("/short")
        old_prefix = Path("/old/root/deep")
        new_prefix = Path("/new/location/deep")

        with pytest.raises(ValueError, match="Unable to relocate cross-recording path"):
            _relocate_cross_recording_path(path=path, old_prefix=old_prefix, new_prefix=new_prefix)

    def test_relocation_with_empty_suffix(self) -> None:
        """Verifies correct behavior when the cross-recording path has the same depth as the prefix with no trailing
        suffix.
        """
        path = Path("/old/root")
        old_prefix = Path("/old/root")
        new_prefix = Path("/new/location")

        relocated = _relocate_cross_recording_path(path=path, old_prefix=old_prefix, new_prefix=new_prefix)

        assert relocated == Path("/new/location")
