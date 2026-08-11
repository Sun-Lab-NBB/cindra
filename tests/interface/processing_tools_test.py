"""Contains tests for the batch processing MCP tools."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest
from ataraxis_data_structures import ProcessingTracker

from cindra.layout import (
    OUTPUT_DIRECTORY_NAME,
    DEFORMED_MASKS_FILENAME,
    CHANNEL_1_BINARY_FILENAME,
    COMBINED_METADATA_FILENAME,
    DETECTION_DATA_DIRECTORY_NAME,
    MULTI_RECORDING_DIRECTORY_NAME,
    TRACKING_TEMPLATE_MASKS_FILENAME,
    SINGLE_RECORDING_TRACKER_FILENAME,
    MULTI_RECORDING_ARRAYS_DIRECTORY_NAME,
    MULTI_RECORDING_RUNTIME_DATA_FILENAME,
    DetectionImages,
    RecordingArrays,
    resolve_plane_specifier,
)
from cindra.dataclasses import SingleRecordingConfiguration
from cindra.orchestration import (
    RESOURCE_CLASS_BY_JOB_NAME,
    PendingJob,
    JobExecutionState,
    SingleRecordingJobNames,
    get_execution_state,
    set_execution_state,
)
from cindra.interface import processing_tools
from cindra.interface.processing_tools import (
    clean_processing_output_tool,
    execute_full_pipeline_tool,
    execute_processing_jobs_tool,
    get_active_execution_timing_tool,
    prepare_single_recording_batch_tool,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Iterator

_SETTLE_SECONDS: float = 0.05
"""The pause that places the timed session inside the sub-second window a rounded hour conversion collapses."""

_MINIMUM_THROUGHPUT: float = 1000.0
"""The rate a session measured in tens of milliseconds must exceed, expressed in jobs per hour."""


def _initialize_binarization_job(tracker_path: Path) -> tuple[ProcessingTracker, str]:
    """Registers one binarization job on a fresh tracker and returns the tracker with the job's identifier."""
    tracker = ProcessingTracker(file_path=tracker_path)
    job_id = tracker.initialize_jobs(jobs=[(str(SingleRecordingJobNames.BINARIZE), "")])[0]
    return tracker, job_id


def _build_unpreparable_batch(tmp_path: Path) -> tuple[str, list[str], list[str]]:
    """Creates a template configuration and two recording directories that hold no acquisition parameters file."""
    configuration_path = tmp_path / "template.yaml"
    SingleRecordingConfiguration().save(file_path=configuration_path)
    recording_paths: list[str] = []
    output_paths: list[str] = []
    for name in ("recA", "recB"):
        recording = tmp_path / name
        recording.mkdir()
        recording_paths.append(str(recording))
        output_paths.append(str(tmp_path / f"out_{name}"))
    return str(configuration_path), recording_paths, output_paths


@pytest.fixture(autouse=True)
def _isolated_execution_state() -> Iterator[None]:
    """Clears the module-global execution state around every test, so no session leaks between them."""
    set_execution_state(state=None)
    yield
    set_execution_state(state=None)


class TestActiveExecutionTiming:
    """Tests the session throughput the timing tool reports."""

    @pytest.mark.xdist_group(name="execution_state")
    def test_short_session_reports_a_finite_throughput(self, tmp_path: Path) -> None:
        """Verifies that a session shorter than the converter's rounding step reports a rate instead of raising."""
        tracker_path = tmp_path / SINGLE_RECORDING_TRACKER_FILENAME
        tracker, job_id = _initialize_binarization_job(tracker_path=tracker_path)
        with tracker.run_job(job_id=job_id):
            pass

        pending_job = PendingJob(
            configuration_path=tmp_path / "configuration.yaml",
            tracker_path=tracker_path,
            job_id=job_id,
            single_recording=True,
            resource_class=RESOURCE_CLASS_BY_JOB_NAME[SingleRecordingJobNames.BINARIZE],
        )
        set_execution_state(state=JobExecutionState(all_jobs={pending_job.dispatch_key: pending_job}))
        time.sleep(_SETTLE_SECONDS)

        result = get_active_execution_timing_tool()

        session = result["session"]
        assert session["completed_count"] == 1
        assert session["total_elapsed_seconds"] > 0
        assert session["throughput_jobs_per_hour"] == round(3600.0 / session["total_elapsed_seconds"], 2)
        assert session["throughput_jobs_per_hour"] > _MINIMUM_THROUGHPUT


class TestExecuteProcessingJobs:
    """Tests the descriptor validation the execute tool performs before it starts a session."""

    @pytest.mark.xdist_group(name="execution_state")
    def test_unsizable_job_is_reported_rather_than_raised(self, tmp_path: Path) -> None:
        """Verifies that a descriptor naming a configuration without an output path lands in the invalid list."""
        configuration_path = tmp_path / "template.yaml"
        SingleRecordingConfiguration().save(file_path=configuration_path)
        tracker_path = tmp_path / SINGLE_RECORDING_TRACKER_FILENAME
        _, job_id = _initialize_binarization_job(tracker_path=tracker_path)

        result = execute_processing_jobs_tool(
            jobs=[
                {
                    "configuration_path": str(configuration_path),
                    "tracker_path": str(tracker_path),
                    "job_id": job_id,
                    "pipeline_type": "single-recording",
                }
            ]
        )

        assert result["success"] is False
        assert result["error"] == "Unable to execute jobs. No valid jobs after validation."
        assert result["invalid_jobs"][0]["job_id"] == job_id
        assert "Unable to size the job from its configuration" in result["invalid_jobs"][0]["reason"]
        assert get_execution_state() is None


class TestPrepareSingleRecordingBatch:
    """Tests the manifest the preparation tool builds for a batch holding an unprepared recording."""

    def test_recordings_without_parameters_are_reported(self, tmp_path: Path) -> None:
        """Verifies that a recording whose bootstrap fails is listed with its reason instead of aborting the batch."""
        configuration_path, recording_paths, output_paths = _build_unpreparable_batch(tmp_path=tmp_path)

        result = prepare_single_recording_batch_tool(
            recording_paths=recording_paths,
            configuration_path=configuration_path,
            recording_output_paths=output_paths,
        )

        assert result["success"] is True
        assert result["recordings"] == {}
        assert result["total_jobs"] == 0
        assert len(result["invalid_recordings"]) == 2
        assert all(any(path in entry for entry in result["invalid_recordings"]) for path in recording_paths)


class TestExecuteFullPipeline:
    """Tests the outcome the full-pipeline tool reports for a batch whose preparation accepted no recording."""

    @pytest.mark.xdist_group(name="execution_state")
    def test_batch_without_prepared_recordings_reports_its_rejections(self, tmp_path: Path) -> None:
        """Verifies that a batch whose every recording fails preparation names them instead of claiming completion."""
        configuration_path, recording_paths, output_paths = _build_unpreparable_batch(tmp_path=tmp_path)

        result = execute_full_pipeline_tool(
            pipeline_type="single-recording",
            recording_paths=recording_paths,
            configuration_path=configuration_path,
            recording_output_paths=output_paths,
        )

        assert result["success"] is False
        assert result["started"] is False
        assert "accepted none of the provided inputs" in result["error"]
        assert len(result["invalid_recordings"]) == 2
        assert all(any(path in entry for entry in result["invalid_recordings"]) for path in recording_paths)
        assert get_execution_state() is None


class TestCleanProcessingOutput:
    """Tests the order in which the clean tool removes the completion markers and the data they vouch for."""

    def test_combined_marker_is_unlinked_before_its_payload(self, tmp_path: Path) -> None:
        """Verifies that the single-recording completion marker is the first file the clean removes."""
        cindra_root = tmp_path / OUTPUT_DIRECTORY_NAME
        plane_directory = cindra_root / resolve_plane_specifier(plane_index=0)
        (plane_directory / DETECTION_DATA_DIRECTORY_NAME).mkdir(parents=True)
        (cindra_root / COMBINED_METADATA_FILENAME).write_bytes(b"")
        (cindra_root / RecordingArrays.CELL_FLUORESCENCE).write_bytes(b"")
        (plane_directory / CHANNEL_1_BINARY_FILENAME).write_bytes(b"")
        (plane_directory / DETECTION_DATA_DIRECTORY_NAME / DetectionImages.MEAN_IMAGE).write_bytes(b"")

        result = clean_processing_output_tool(
            recording_path=str(tmp_path), phases=["binarization"], pipeline_type="single-recording"
        )

        deleted_files = result["deleted_files"]
        assert deleted_files[0] == str(cindra_root / COMBINED_METADATA_FILENAME)
        assert deleted_files.index(str(plane_directory / CHANNEL_1_BINARY_FILENAME)) > 0
        assert deleted_files.index(str(cindra_root / RecordingArrays.CELL_FLUORESCENCE)) > 0

    def test_tracking_marker_is_unlinked_before_its_payload(self, tmp_path: Path, monkeypatch) -> None:
        """Verifies that the multi-recording discovery marker is removed ahead of the arrays it vouches for."""
        dataset_path = tmp_path / OUTPUT_DIRECTORY_NAME / MULTI_RECORDING_DIRECTORY_NAME / "animal_a_task"
        arrays_directory = dataset_path / MULTI_RECORDING_ARRAYS_DIRECTORY_NAME
        arrays_directory.mkdir(parents=True)
        (dataset_path / MULTI_RECORDING_RUNTIME_DATA_FILENAME).write_text(
            f"io:\n  dataset_output_paths:\n  - {dataset_path}\n"
        )
        (dataset_path / TRACKING_TEMPLATE_MASKS_FILENAME).write_bytes(b"")
        (dataset_path / DEFORMED_MASKS_FILENAME).write_bytes(b"")
        (arrays_directory / "deformation_field.npy").write_bytes(b"")

        removal_order: list[str] = []
        delete_file = processing_tools._delete_file
        delete_directory = processing_tools._delete_directory

        def _spy_file(path: Path, deleted: list[str], errors: list[str]) -> None:
            if path.exists():
                removal_order.append(path.name)
            delete_file(path=path, deleted=deleted, errors=errors)

        def _spy_directory(path: Path, deleted: list[str], errors: list[str]) -> None:
            if path.exists():
                removal_order.append(path.name)
            delete_directory(path=path, deleted=deleted, errors=errors)

        monkeypatch.setattr(processing_tools, "_delete_file", _spy_file)
        monkeypatch.setattr(processing_tools, "_delete_directory", _spy_directory)

        result = clean_processing_output_tool(
            recording_path=str(tmp_path),
            phases=["discovery"],
            pipeline_type="multi-recording",
            dataset="animal_a_task",
        )

        marker_index = removal_order.index(str(TRACKING_TEMPLATE_MASKS_FILENAME))
        assert result["success"] is True
        assert marker_index < removal_order.index(str(MULTI_RECORDING_ARRAYS_DIRECTORY_NAME))
        assert marker_index < removal_order.index(str(DEFORMED_MASKS_FILENAME))
