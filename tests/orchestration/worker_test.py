"""Contains tests for the per-job entry points of the two pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from ataraxis_base_utilities import console, error_format
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from cindra.dataclasses import SingleRecordingConfiguration
from cindra.orchestration import SingleRecordingJobNames, worker
from cindra.orchestration.worker import _resolve_job_plane_index, dispatch_single_recording_job

if TYPE_CHECKING:
    from pathlib import Path

_DEVICE_REFUSAL_MESSAGE: str = (
    "Unable to run the registration stage on CUDA device 7. The host exposes no device carrying that index."
)
"""The refusal the stand-in device verification reports, matched against the error the dispatch branch propagates."""


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


class TestRegistrationDeviceGate:
    """Tests the CUDA device verification the registration dispatch branch runs."""

    def test_unusable_device_fails_the_job_on_its_tracker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verifies that a registration job naming an unusable device is recorded as a failure of that job."""
        jobs = [(str(SingleRecordingJobNames.REGISTER), "plane_0")]
        job_id = ProcessingTracker.generate_job_id(job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0")
        tracker = ProcessingTracker(file_path=tmp_path / "single_recording_tracker.yaml")
        tracker.align_jobs(jobs=jobs, universe=jobs)

        def _refuse_device(device: int | None) -> None:
            """Refuses the device index the dispatch branch asked about."""
            assert device == 7
            console.error(message=_DEVICE_REFUSAL_MESSAGE, error=ValueError)

        monkeypatch.setattr(worker, "verify_gpu_runtime", _refuse_device)

        with pytest.raises(ValueError, match=error_format(message=_DEVICE_REFUSAL_MESSAGE)):
            dispatch_single_recording_job(
                configuration=SingleRecordingConfiguration(),
                job_name=SingleRecordingJobNames.REGISTER,
                specifier="plane_0",
                job_id=job_id,
                tracker=tracker,
                workers=None,
                device=7,
            )

        assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED
