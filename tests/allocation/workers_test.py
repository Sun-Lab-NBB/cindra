"""Contains tests for the per-stage worker defaults and the worker count resolver."""

from __future__ import annotations

import pytest
from ataraxis_base_utilities import resolve_worker_count

from cindra.allocation import (
    ALL_CORES_REQUEST,
    DISCOVERY_WORKERS,
    EXTRACTION_WORKERS,
    PROCESSING_WORKERS,
    BINARIZATION_WORKERS,
    REGISTRATION_WORKERS,
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    resolve_stage_workers,
)


class TestStageDefaults:
    """Tests the measured worker default each pipeline stage resolves to."""

    @pytest.mark.parametrize(
        ("job_name", "expected_workers"),
        [
            (SingleRecordingJobNames.BINARIZE, BINARIZATION_WORKERS),
            (SingleRecordingJobNames.REGISTER, REGISTRATION_WORKERS),
            (SingleRecordingJobNames.PROCESS, PROCESSING_WORKERS),
            (MultiRecordingJobNames.DISCOVER, DISCOVERY_WORKERS),
            (MultiRecordingJobNames.EXTRACT, EXTRACTION_WORKERS),
        ],
    )
    def test_unspecified_request_resolves_to_the_stage_default(
        self, job_name: SingleRecordingJobNames | MultiRecordingJobNames, expected_workers: int
    ) -> None:
        """Verifies that omitting the requested count resolves to the measured default of the target stage."""
        assert resolve_stage_workers(job_name=job_name) == expected_workers
        assert resolve_stage_workers(job_name=job_name, requested_workers=None) == expected_workers

    def test_stage_defaults_are_positive(self) -> None:
        """Verifies that every measured stage default is a usable positive worker count."""
        assert BINARIZATION_WORKERS > 0
        assert REGISTRATION_WORKERS > 0
        assert PROCESSING_WORKERS > 0
        assert DISCOVERY_WORKERS > 0
        assert EXTRACTION_WORKERS > 0


class TestExplicitRequests:
    """Tests the worker counts a caller asks for explicitly."""

    @pytest.mark.parametrize("requested_workers", [1, 7, 30, 512])
    def test_positive_request_is_honored_exactly(self, requested_workers: int) -> None:
        """Verifies that a positive requested count reaches the stage unchanged."""
        resolved = resolve_stage_workers(job_name=SingleRecordingJobNames.REGISTER, requested_workers=requested_workers)
        assert resolved == requested_workers

    def test_all_cores_request_resolves_through_the_worker_resolver(self) -> None:
        """Verifies that requesting every available core defers to the ataraxis worker resolver."""
        resolved = resolve_stage_workers(job_name=SingleRecordingJobNames.PROCESS, requested_workers=ALL_CORES_REQUEST)
        assert resolved == resolve_worker_count(requested_workers=ALL_CORES_REQUEST)
        assert resolved >= 1

    def test_all_cores_request_is_honored_for_every_allocating_stage(self) -> None:
        """Verifies that every single-recording stage taking an allocation accepts the all-cores request."""
        expected = resolve_worker_count(requested_workers=ALL_CORES_REQUEST)
        for job_name in (
            SingleRecordingJobNames.BINARIZE,
            SingleRecordingJobNames.REGISTER,
            SingleRecordingJobNames.PROCESS,
        ):
            assert resolve_stage_workers(job_name=job_name, requested_workers=ALL_CORES_REQUEST) == expected


class TestRejectedRequests:
    """Tests the requests the resolver rejects."""

    def test_combination_stage_is_rejected(self) -> None:
        """Verifies that the combination stage, which takes no allocation, raises rather than returning a count."""
        with pytest.raises(ValueError, match=r"does not name a\s+pipeline\s+stage"):
            resolve_stage_workers(job_name=SingleRecordingJobNames.COMBINE)

    def test_combination_stage_is_rejected_before_the_requested_count_is_read(self) -> None:
        """Verifies that the stage check precedes the requested-count check, so a valid count cannot mask it."""
        with pytest.raises(ValueError, match=r"does not name a\s+pipeline\s+stage"):
            resolve_stage_workers(job_name=SingleRecordingJobNames.COMBINE, requested_workers=8)

    @pytest.mark.parametrize("requested_workers", [0, -2, -3, -100])
    def test_invalid_worker_count_is_rejected(self, requested_workers: int) -> None:
        """Verifies that zero and every negative count other than the all-cores request raise an error."""
        with pytest.raises(ValueError, match=r"must be a positive\s+integer"):
            resolve_stage_workers(job_name=SingleRecordingJobNames.PROCESS, requested_workers=requested_workers)
