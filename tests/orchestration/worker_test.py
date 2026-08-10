"""Contains tests for the per-job entry points of the two pipelines."""

from __future__ import annotations

import pytest

from cindra.orchestration import SingleRecordingJobNames
from cindra.orchestration.worker import _resolve_job_plane_index


class TestResolveJobPlaneIndex:
    """Tests the plane index a per-plane job's specifier names."""

    @pytest.mark.parametrize("plane_index", [0, 3, 17])
    def test_specifier_naming_a_plane_resolves_to_its_index(self, plane_index: int) -> None:
        """Verifies that a well-formed specifier resolves to the plane it names."""
        resolved = _resolve_job_plane_index(job_name=SingleRecordingJobNames.REGISTER, specifier=f"plane_{plane_index}")

        assert resolved == plane_index

    @pytest.mark.parametrize("specifier", ["", "plane_", "plane_x", "recording_1"])
    def test_specifier_naming_no_plane_is_rejected(self, specifier: str) -> None:
        """Verifies that a specifier naming no plane raises an error that names the specifier received."""
        with pytest.raises(ValueError, match=r"must name an imaging\s+plane"):
            _resolve_job_plane_index(job_name=SingleRecordingJobNames.PROCESS, specifier=specifier)
