"""Contains tests for the batch processing MCP tools."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from tifffile import imwrite
from ataraxis_time import PrecisionTimer, TimerPrecisions
from ataraxis_data_structures import ProcessingTracker

from cindra.layout import (
    PARAMETERS_FILENAME,
    OUTPUT_DIRECTORY_NAME,
    DEFORMED_MASKS_FILENAME,
    CHANNEL_1_BINARY_FILENAME,
    COMBINED_METADATA_FILENAME,
    DETECTION_DATA_DIRECTORY_NAME,
    MULTI_RECORDING_DIRECTORY_NAME,
    ACQUISITION_PARAMETERS_FILENAME,
    REGISTRATION_DATA_DIRECTORY_NAME,
    TRACKING_TEMPLATE_MASKS_FILENAME,
    SINGLE_RECORDING_TRACKER_FILENAME,
    MULTI_RECORDING_ARRAYS_DIRECTORY_NAME,
    MULTI_RECORDING_RUNTIME_DATA_FILENAME,
    SINGLE_RECORDING_CONFIGURATION_FILENAME,
    DetectionImages,
    RecordingArrays,
    RegistrationArrays,
    resolve_array_path,
    resolve_plane_path,
    resolve_output_path,
    resolve_plane_specifier,
)
from cindra.interface import processing_tools
from cindra.dataclasses import (
    AcquisitionParameters,
    MultiRecordingConfiguration,
    SingleRecordingConfiguration,
)
from cindra.orchestration import (
    GPU_REMEDY,
    RESOURCE_CLASS_BY_JOB_NAME,
    GpuDevice,
    GpuStatus,
    GpuSummary,
    PendingJob,
    SingleRecordingJobNames,
    get_execution_state,
    set_execution_state,
)
from cindra.orchestration.execution import JobExecutionState
from cindra.interface.processing_tools import (
    check_gpu_runtime_tool,
    size_pipeline_jobs_tool,
    get_recording_status_tool,
    execute_full_pipeline_tool,
    clean_processing_output_tool,
    execute_processing_jobs_tool,
    reset_processing_phases_tool,
    get_processing_jobs_status_tool,
    get_active_execution_timing_tool,
    prepare_single_recording_batch_tool,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Iterator

_SETTLE_MILLISECONDS: int = 50
"""The millisecond pause that places the timed session inside the sub-second window a rounded hour conversion
collapses."""

_MINIMUM_THROUGHPUT: float = 1000.0
"""The rate a session measured in tens of milliseconds must exceed, expressed in jobs per hour."""

_SOURCE_FRAME_SHAPE: tuple[int, int, int] = (2, 16, 16)
"""The frame count, height, and width of the TIFF file the raw data gate accepts as a readable source file."""

_ABSENT_DEVICE_INDEX: int = 4096
"""The CUDA device index no host exposes, which the device mask rejection test asks a session to run on."""


class TestActiveExecutionTiming:
    """Tests the session throughput the timing tool reports."""

    @pytest.mark.xdist_group(name="execution_state")
    def test_short_session_reports_a_finite_throughput(self, tmp_path: Path) -> None:
        """Verifies that a session shorter than the converter's rounding step reports a rate instead of raising."""
        tracker_path = tmp_path / SINGLE_RECORDING_TRACKER_FILENAME
        tracker, job_id = _initialize_binarization_job(tracker_path=tracker_path)
        with tracker.run_job(job_id=job_id):
            pass

        set_execution_state(state=_build_execution_state(tmp_path=tmp_path, tracker_path=tracker_path, job_id=job_id))
        PrecisionTimer(precision=TimerPrecisions.MILLISECOND).delay(delay=_SETTLE_MILLISECONDS, allow_sleep=True)

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

    @pytest.mark.xdist_group(name="execution_state")
    def test_rejected_override_still_names_the_jobs_validation_refused(self, tmp_path: Path) -> None:
        """Verifies that a session refused for a bad override reports the invalid jobs it validated beforehand."""
        manifest_dict, cindra_root = _prepare_recording(tmp_path=tmp_path)
        tracker_path = cindra_root / SINGLE_RECORDING_TRACKER_FILENAME
        valid_job = {
            "configuration_path": str(cindra_root / SINGLE_RECORDING_CONFIGURATION_FILENAME),
            "tracker_path": str(tracker_path),
            "job_id": manifest_dict["binarize_job"]["job_id"],
            "pipeline_type": "single-recording",
        }

        result = execute_processing_jobs_tool(jobs=[valid_job, {**valid_job, "job_id": "absent"}], workers_per_job=0)

        assert result["success"] is False
        assert result["started"] is False
        assert "'workers_per_job' override must be a positive integer" in result["error"]
        assert result["invalid_jobs"][0]["job_id"] == "absent"
        assert get_execution_state() is None


class TestRegistrationResourceClass:
    """Tests the resource class a registration job receives from the devices the session names."""

    @pytest.mark.parametrize(
        ("gpu_registration", "expected_name"), [(True, "registration_gpu"), (False, "registration")]
    )
    def test_session_device_request_selects_the_class_of_every_register_job(
        self, tmp_path: Path, gpu_registration: bool, expected_name: str
    ) -> None:
        """Verifies that the session's device request reaches the jobs the full-pipeline tool builds."""
        manifest_dict, cindra_root = _prepare_recording(tmp_path=tmp_path)

        _, register_jobs, _, _ = processing_tools._resolve_recording_phase_jobs(
            manifest_dict=manifest_dict,
            configuration_path=cindra_root / SINGLE_RECORDING_CONFIGURATION_FILENAME,
            tracker_path=cindra_root / SINGLE_RECORDING_TRACKER_FILENAME,
            gpu_registration=gpu_registration,
        )

        assert register_jobs
        assert {job.resource_class.name for job in register_jobs} == {expected_name}

    def test_device_planned_register_jobs_carry_the_wider_memory_estimate(self, tmp_path: Path) -> None:
        """Verifies that planning a session for a device reaches the memory figure of every registration job."""
        manifest_dict, cindra_root = _prepare_recording(tmp_path=tmp_path)
        arguments = {
            "manifest_dict": manifest_dict,
            "configuration_path": cindra_root / SINGLE_RECORDING_CONFIGURATION_FILENAME,
            "tracker_path": cindra_root / SINGLE_RECORDING_TRACKER_FILENAME,
        }

        _, host_jobs, _, _ = processing_tools._resolve_recording_phase_jobs(**arguments, gpu_registration=False)
        _, device_jobs, _, _ = processing_tools._resolve_recording_phase_jobs(**arguments, gpu_registration=True)

        assert [job.memory_megabytes for job in device_jobs] >= [job.memory_megabytes for job in host_jobs]

    @pytest.mark.xdist_group(name="execution_state")
    def test_list_naming_an_absent_device_is_reported_rather_than_raised(self, tmp_path: Path) -> None:
        """Verifies that a device list the host cannot satisfy leaves the session unstarted and names the list."""
        manifest_dict, cindra_root = _prepare_recording(tmp_path=tmp_path)

        result = execute_processing_jobs_tool(
            jobs=[
                {
                    "configuration_path": str(cindra_root / SINGLE_RECORDING_CONFIGURATION_FILENAME),
                    "tracker_path": str(cindra_root / SINGLE_RECORDING_TRACKER_FILENAME),
                    "job_id": manifest_dict["binarize_job"]["job_id"],
                    "pipeline_type": "single-recording",
                }
            ],
            gpu_devices=[_ABSENT_DEVICE_INDEX],
        )

        assert result["success"] is False
        assert result["started"] is False
        assert "'gpu_devices' list must name at least one of the CUDA devices the host exposes" in result["error"]
        assert get_execution_state() is None


class TestSizePipelineJobs:
    """Tests the cores, memory, and device memory the sizing tool reports for every job of a pipeline."""

    def test_host_planned_sizing_reports_no_device_memory(self, tmp_path: Path) -> None:
        """Verifies that a sizing pass naming no device reports a zero device figure for every job."""
        configuration_path = _prepare_sizable_recording(tmp_path=tmp_path)

        result = size_pipeline_jobs_tool(configuration_path=str(configuration_path), pipeline_type="single-recording")

        assert result["success"] is True
        assert all(job["device_memory_mb"] == 0 for job in result["jobs"])
        assert result["peak_device_memory_mb"] == 0

    def test_device_planned_sizing_reports_the_device_memory_of_its_registration_jobs(self, tmp_path: Path) -> None:
        """Verifies that the flag reaches the registration jobs alone and sets the reported device peak."""
        configuration_path = _prepare_sizable_recording(tmp_path=tmp_path)

        result = size_pipeline_jobs_tool(
            configuration_path=str(configuration_path), pipeline_type="single-recording", gpu_registration=True
        )

        assert result["success"] is True
        device_figures = {job["name"]: job["device_memory_mb"] for job in result["jobs"] if job["device_memory_mb"] > 0}
        assert set(device_figures) == {SingleRecordingJobNames.REGISTER}
        assert result["peak_device_memory_mb"] == max(device_figures.values())

    def test_multi_recording_sizing_reports_no_device_memory(self, tmp_path: Path) -> None:
        """Verifies that a dataset sizing reports zero device memory whatever the flag holds."""
        configuration_path = _prepare_sizable_dataset(tmp_path=tmp_path)

        result = size_pipeline_jobs_tool(
            configuration_path=str(configuration_path), pipeline_type="multi-recording", gpu_registration=True
        )

        assert result["success"] is True
        assert all(job["device_memory_mb"] == 0 for job in result["jobs"])
        assert result["peak_device_memory_mb"] == 0


class TestCheckGpuRuntime:
    """Tests the flat report the GPU runtime tool shapes from the device resolution."""

    def test_usable_device_is_reported_without_a_remedy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifies that a host carrying a usable device reports it as ready alongside the device's properties."""
        summary = GpuSummary(
            status=GpuStatus.AVAILABLE,
            devices=(GpuDevice(index=0, name="test device", total_memory_mb=1024, compute_capability="8.6"),),
            detail="",
        )
        monkeypatch.setattr(target=processing_tools, name="resolve_gpu_devices", value=lambda: summary)

        result = check_gpu_runtime_tool()

        assert result["success"] is True
        assert result["ready"] is True
        assert result["status"] == GpuStatus.AVAILABLE.value
        assert result["device_count"] == 1
        assert result["devices"] == [
            {"index": 0, "name": "test device", "total_memory_mb": 1024, "compute_capability": "8.6"}
        ]
        assert "remedy" not in result

    def test_missing_runtime_carries_the_installation_remedy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifies that a host carrying no usable device reports the command that resolves the runtime."""
        summary = GpuSummary(
            status=GpuStatus.RUNTIME_MISSING, devices=(), detail="the CuPy distribution is not installed"
        )
        monkeypatch.setattr(target=processing_tools, name="resolve_gpu_devices", value=lambda: summary)

        result = check_gpu_runtime_tool()

        assert result["ready"] is False
        assert result["status"] == GpuStatus.RUNTIME_MISSING.value
        assert result["devices"] == []
        assert result["device_count"] == 0
        assert result["remedy"] == GPU_REMEDY


class TestProcessingJobsStatus:
    """Tests the two response shapes the job status tool returns for one active session."""

    @pytest.mark.xdist_group(name="execution_state")
    def test_summary_only_omits_the_per_job_list(self, tmp_path: Path) -> None:
        """Verifies that the summary counts survive the omission of the per-job entries they summarize."""
        tracker_path = tmp_path / SINGLE_RECORDING_TRACKER_FILENAME
        tracker, job_id = _initialize_binarization_job(tracker_path=tracker_path)
        with tracker.run_job(job_id=job_id):
            pass

        set_execution_state(state=_build_execution_state(tmp_path=tmp_path, tracker_path=tracker_path, job_id=job_id))

        summary_result = get_processing_jobs_status_tool(summary_only=True)
        full_result = get_processing_jobs_status_tool()

        assert "jobs" not in summary_result
        assert summary_result["summary"] == {"pending": 0, "running": 0, "succeeded": 1, "failed": 0}
        assert full_result["summary"] == summary_result["summary"]
        assert [entry["job_id"] for entry in full_result["jobs"]] == [job_id]
        assert full_result["jobs"][0]["tracker_path"] == str(tracker_path)


class TestRecordingStatus:
    """Tests the output root the status tool resolves its trackers under and reports back."""

    def test_absent_output_root_is_reported(self, tmp_path: Path) -> None:
        """Verifies that a path holding no directory reports the output root it was given."""
        result = get_recording_status_tool(output_root=str(tmp_path / "absent"))

        assert result["success"] is False
        assert "Output root not found" in result["error"]

    def test_prepared_recording_reports_its_output_root(self, tmp_path: Path) -> None:
        """Verifies that both the response and its single-recording section name the passed output root."""
        tracker_path = tmp_path / OUTPUT_DIRECTORY_NAME / SINGLE_RECORDING_TRACKER_FILENAME
        tracker_path.parent.mkdir(parents=True)
        _initialize_binarization_job(tracker_path=tracker_path)

        result = get_recording_status_tool(output_root=str(tmp_path))

        assert result["success"] is True
        assert result["output_root"] == str(tmp_path)
        assert result["single_recording"]["output_root"] == str(tmp_path)
        assert result["single_recording"]["status"] == "scheduled"


class TestPrepareSingleRecordingBatch:
    """Tests the manifest the preparation tool builds, the raw data gate it applies through the resolved imaging
    directory, and the stored-path conflicts it reports."""

    def test_recordings_without_parameters_are_reported(self, tmp_path: Path) -> None:
        """Verifies that a recording whose bootstrap fails is listed with its reason instead of aborting the batch."""
        configuration_path, raw_data_paths, output_roots = _build_unpreparable_batch(tmp_path=tmp_path)

        result = prepare_single_recording_batch_tool(
            raw_data_paths=raw_data_paths,
            configuration_path=configuration_path,
            output_roots=output_roots,
        )

        assert result["success"] is True
        assert result["recordings"] == {}
        assert result["total_jobs"] == 0
        assert len(result["invalid_recordings"]) == 2
        assert all(any(path in entry for entry in result["invalid_recordings"]) for path in raw_data_paths)

    def test_a_session_root_carrying_its_parameters_file_deeper_is_prepared(self, tmp_path: Path) -> None:
        """Verifies that a session root is prepared, because the conversion resolves the imaging directory itself."""
        configuration_path = tmp_path / "template.yaml"
        SingleRecordingConfiguration().save(file_path=configuration_path)
        session_root = tmp_path / "session"
        imaging_directory = session_root / "mesoscope_data"
        imaging_directory.mkdir(parents=True)
        _write_source_file(directory=imaging_directory)
        (imaging_directory / PARAMETERS_FILENAME).write_text(
            json.dumps(obj={"frame_rate": 30.0, "plane_number": 1, "channel_number": 1})
        )
        output_root = tmp_path / "output"

        result = prepare_single_recording_batch_tool(
            raw_data_paths=[str(session_root)],
            configuration_path=str(configuration_path),
            output_roots=[str(output_root)],
        )

        assert result["success"] is True
        assert "invalid_recordings" not in result
        assert str(session_root) in result["recordings"]
        assert (output_root / OUTPUT_DIRECTORY_NAME).exists()

    def test_raw_data_path_without_source_files_names_the_subdirectory_holding_them(self, tmp_path: Path) -> None:
        """Verifies that a path whose subtree carries no parameters file is rejected and given the likely path."""
        configuration_path = tmp_path / "template.yaml"
        SingleRecordingConfiguration().save(file_path=configuration_path)
        session_root = tmp_path / "session"
        imaging_directory = session_root / "mesoscope_data"
        imaging_directory.mkdir(parents=True)
        _write_source_file(directory=imaging_directory)
        output_root = tmp_path / "output"

        result = prepare_single_recording_batch_tool(
            raw_data_paths=[str(session_root)],
            configuration_path=str(configuration_path),
            output_roots=[str(output_root)],
        )

        assert result["success"] is True
        assert result["recordings"] == {}
        assert result["total_jobs"] == 0
        rejection = result["invalid_recordings"][0]
        assert str(session_root) in rejection
        assert "without descending further" in rejection
        assert str(imaging_directory) in rejection
        assert not (output_root / OUTPUT_DIRECTORY_NAME).exists()

    def test_outstanding_binarization_rejects_an_unreadable_raw_data_path(self, tmp_path: Path) -> None:
        """Verifies that an existing tracker whose conversion has not run is gated on its raw data like a new one."""
        configuration_path, raw_data_path, output_root, _ = _build_prepared_recording(tmp_path=tmp_path)

        result = prepare_single_recording_batch_tool(
            raw_data_paths=[str(raw_data_path)],
            configuration_path=str(configuration_path),
            output_roots=[str(output_root)],
        )

        assert result["success"] is True
        assert result["recordings"] == {}
        assert f"{raw_data_path}: " in result["invalid_recordings"][0]
        assert "path_conflicts" not in result

    def test_completed_binarization_keeps_its_manifest_and_reports_the_stored_paths(self, tmp_path: Path) -> None:
        """Verifies that a converted recording survives an archived raw directory and names the paths it keeps."""
        configuration_path, raw_data_path, output_root, tracker_path = _build_prepared_recording(tmp_path=tmp_path)
        tracker = ProcessingTracker(file_path=tracker_path)
        job_id = next(iter(tracker.find_jobs(job_name=SingleRecordingJobNames.BINARIZE)))
        with tracker.run_job(job_id=job_id):
            pass
        passed_raw_data_path = tmp_path / "passed_raw"
        passed_raw_data_path.mkdir()

        result = prepare_single_recording_batch_tool(
            raw_data_paths=[str(passed_raw_data_path)],
            configuration_path=str(configuration_path),
            output_roots=[str(output_root)],
        )

        assert result["success"] is True
        assert "invalid_recordings" not in result
        assert result["recordings"][str(passed_raw_data_path)]["output_root"] == str(output_root)
        conflicts = result["path_conflicts"]
        assert len(conflicts) == 1
        assert conflicts[0]["recording"] == str(passed_raw_data_path)
        assert conflicts[0]["field"] == "file_io.data_path"
        assert conflicts[0]["stored"] == str(raw_data_path)
        assert conflicts[0]["passed"] == str(passed_raw_data_path)
        assert str(output_root / OUTPUT_DIRECTORY_NAME) in conflicts[0]["resolution"]


class TestExecuteFullPipeline:
    """Tests the outcome the full-pipeline tool reports for a batch whose preparation accepted no recording."""

    @pytest.mark.xdist_group(name="execution_state")
    def test_batch_without_prepared_recordings_reports_its_rejections(self, tmp_path: Path) -> None:
        """Verifies that a batch whose every recording fails preparation names them instead of claiming completion."""
        configuration_path, raw_data_paths, output_roots = _build_unpreparable_batch(tmp_path=tmp_path)

        result = execute_full_pipeline_tool(
            pipeline_type="single-recording",
            raw_data_paths=raw_data_paths,
            configuration_path=configuration_path,
            output_roots=output_roots,
        )

        assert result["success"] is False
        assert result["started"] is False
        assert "accepted none of the provided inputs" in result["error"]
        assert len(result["invalid_recordings"]) == 2
        assert all(any(path in entry for entry in result["invalid_recordings"]) for path in raw_data_paths)
        assert result["pipeline_type"] == "single-recording"
        assert get_execution_state() is None

    @pytest.mark.parametrize(
        ("pipeline_type", "arguments"),
        [
            ("neither-pipeline", {}),
            ("single-recording", {}),
            ("single-recording", {"raw_data_paths": ["/absent"]}),
            ("single-recording", {"raw_data_paths": ["/absent"], "configuration_path": "/absent.yaml"}),
            ("multi-recording", {}),
        ],
    )
    @pytest.mark.xdist_group(name="execution_state")
    def test_every_argument_rejection_names_the_requested_pipeline(
        self, pipeline_type: str, arguments: dict[str, object]
    ) -> None:
        """Verifies that each argument guard names the pipeline the caller requested, as the Returns block promises."""
        result = execute_full_pipeline_tool(pipeline_type=pipeline_type, **arguments)

        assert result["success"] is False
        assert result["pipeline_type"] == pipeline_type
        assert get_execution_state() is None


class TestResetProcessingPhases:
    """Tests the repeat-flag warnings the reset tool derives from the output the reset phases already hold."""

    def test_disabled_repeat_flag_over_existing_output_is_reported(self, tmp_path: Path) -> None:
        """Verifies that resetting binarization over an existing plane binary names the flag that lifts the skip."""
        _, _, output_root, tracker_path = _build_prepared_recording(tmp_path=tmp_path)
        plane_path = resolve_plane_path(output_root=output_root, plane_index=0)
        plane_path.mkdir(parents=True)
        (plane_path / CHANNEL_1_BINARY_FILENAME).write_bytes(b"")

        result = reset_processing_phases_tool(
            tracker_path=str(tracker_path), phases=["binarization"], pipeline_type="single-recording"
        )

        assert result["success"] is True
        assert result["requested_phases"] == ["binarization"]
        warnings = result["warnings"]
        assert len(warnings) == 1
        assert "file_io.repeat_binarization" in warnings[0]
        assert "set_config_values_tool" in warnings[0]

    def test_registration_reset_over_a_corrected_binary_warns_about_the_second_pass(self, tmp_path: Path) -> None:
        """Verifies that re-registering a plane whose binary is already corrected names the rebuild that avoids it."""
        _, _, output_root, tracker_path = _build_prepared_recording(tmp_path=tmp_path)
        _mark_plane_registered(output_root=output_root)
        _write_repeat_flags(output_root=output_root, repeat_registration=True)

        result = reset_processing_phases_tool(
            tracker_path=str(tracker_path), phases=["registration"], pipeline_type="single-recording"
        )

        warnings = result["warnings"]
        assert len(warnings) == 1
        assert "rewrites the plane binary in place" in warnings[0]
        assert "file_io.repeat_binarization" in warnings[0]

    def test_registration_reset_that_rebuilds_the_binary_reports_no_second_pass_warning(self, tmp_path: Path) -> None:
        """Verifies that resetting binarization with the rebuild enabled leaves the registration reset unflagged."""
        _, _, output_root, tracker_path = _build_prepared_recording(tmp_path=tmp_path)
        _mark_plane_registered(output_root=output_root)
        _write_repeat_flags(output_root=output_root, repeat_registration=True, repeat_binarization=True)

        result = reset_processing_phases_tool(
            tracker_path=str(tracker_path), phases=["binarization"], pipeline_type="single-recording"
        )

        assert "warnings" not in result

    def test_reset_without_existing_output_reports_no_warning(self, tmp_path: Path) -> None:
        """Verifies that a phase whose output does not exist yet carries no skip warning."""
        _, _, _, tracker_path = _build_prepared_recording(tmp_path=tmp_path)

        result = reset_processing_phases_tool(
            tracker_path=str(tracker_path), phases=["binarization"], pipeline_type="single-recording"
        )

        assert result["success"] is True
        assert "warnings" not in result


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
            output_root=str(tmp_path), phases=["binarization"], pipeline_type="single-recording"
        )

        deleted_files = result["deleted_files"]
        assert result["output_root"] == str(tmp_path)
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

        monkeypatch.setattr(target=processing_tools, name="_delete_file", value=_spy_file)
        monkeypatch.setattr(target=processing_tools, name="_delete_directory", value=_spy_directory)

        result = clean_processing_output_tool(
            output_root=str(tmp_path),
            phases=["discovery"],
            pipeline_type="multi-recording",
            dataset="animal_a_task",
        )

        marker_index = removal_order.index(str(TRACKING_TEMPLATE_MASKS_FILENAME))
        assert result["success"] is True
        assert marker_index < removal_order.index(str(MULTI_RECORDING_ARRAYS_DIRECTORY_NAME))
        assert marker_index < removal_order.index(str(DEFORMED_MASKS_FILENAME))


@pytest.fixture(autouse=True)
def _isolated_execution_state() -> Iterator[None]:
    """Clears the module-global execution state around every test, so no session leaks between them."""
    set_execution_state(state=None)
    yield
    set_execution_state(state=None)


def _initialize_binarization_job(tracker_path: Path) -> tuple[ProcessingTracker, str]:
    """Registers one binarization job on a fresh tracker and returns the tracker with the job's identifier."""
    tracker = ProcessingTracker(file_path=tracker_path)
    job_id = tracker.initialize_jobs(jobs=[(str(SingleRecordingJobNames.BINARIZE), "")])[0]
    return tracker, job_id


def _build_execution_state(tmp_path: Path, tracker_path: Path, job_id: str) -> JobExecutionState:
    """Builds a session state holding the single binarization job the given tracker registers."""
    pending_job = PendingJob(
        configuration_path=tmp_path / "configuration.yaml",
        tracker_path=tracker_path,
        job_id=job_id,
        single_recording=True,
        resource_class=RESOURCE_CLASS_BY_JOB_NAME[SingleRecordingJobNames.BINARIZE],
    )
    return JobExecutionState(all_jobs={pending_job.dispatch_key: pending_job})


def _prepare_recording(tmp_path: Path) -> tuple[dict[str, object], Path]:
    """Prepares one recording through the batch tool, returning its manifest entry and its cindra output
    directory."""
    configuration_path = tmp_path / "template.yaml"
    SingleRecordingConfiguration().save(file_path=configuration_path)

    imaging_directory = tmp_path / "session" / "mesoscope_data"
    imaging_directory.mkdir(parents=True)
    _write_source_file(directory=imaging_directory)
    (imaging_directory / PARAMETERS_FILENAME).write_text(
        json.dumps(obj={"frame_rate": 30.0, "plane_number": 1, "channel_number": 1})
    )
    output_root = tmp_path / "output"

    result = prepare_single_recording_batch_tool(
        raw_data_paths=[str(imaging_directory)],
        configuration_path=str(configuration_path),
        output_roots=[str(output_root)],
    )

    return result["recordings"][str(imaging_directory)], output_root / OUTPUT_DIRECTORY_NAME


def _write_source_file(directory: Path) -> None:
    """Writes the TIFF file that makes a directory readable to the conversion's non-recursive source scan."""
    imwrite(directory / "recording.tif", data=np.zeros(_SOURCE_FRAME_SHAPE, dtype=np.uint16))


def _build_unpreparable_batch(tmp_path: Path) -> tuple[str, list[str], list[str]]:
    """Creates a template configuration and two raw data directories holding TIFF files but no parameters file."""
    configuration_path = tmp_path / "template.yaml"
    SingleRecordingConfiguration().save(file_path=configuration_path)
    raw_data_paths: list[str] = []
    output_roots: list[str] = []
    for name in ("recA", "recB"):
        raw_data_path = tmp_path / name
        raw_data_path.mkdir()
        _write_source_file(directory=raw_data_path)
        raw_data_paths.append(str(raw_data_path))
        output_roots.append(str(tmp_path / f"out_{name}"))
    return str(configuration_path), raw_data_paths, output_roots


def _build_prepared_recording(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Writes the template, configuration, acquisition parameters, and tracker a prepared recording carries on disk."""
    template_path = tmp_path / "template.yaml"
    SingleRecordingConfiguration().save(file_path=template_path)

    raw_data_path = tmp_path / "stored_raw"
    raw_data_path.mkdir()
    output_root = tmp_path / "stored_output"
    cindra_root = output_root / OUTPUT_DIRECTORY_NAME
    cindra_root.mkdir(parents=True)

    configuration = SingleRecordingConfiguration()
    configuration.file_io.data_path = raw_data_path
    configuration.file_io.output_path = output_root
    configuration.save(file_path=cindra_root / SINGLE_RECORDING_CONFIGURATION_FILENAME)
    AcquisitionParameters(frame_rate=30.0).to_yaml(file_path=cindra_root / ACQUISITION_PARAMETERS_FILENAME)

    tracker_path = cindra_root / SINGLE_RECORDING_TRACKER_FILENAME
    _initialize_binarization_job(tracker_path=tracker_path)
    return template_path, raw_data_path, output_root, tracker_path


def _mark_plane_registered(output_root: Path, plane_index: int = 0) -> None:
    """Writes the reference image the inventory reads to decide that one plane carries registration output."""
    registration_path = (
        resolve_plane_path(output_root=output_root, plane_index=plane_index) / REGISTRATION_DATA_DIRECTORY_NAME
    )
    registration_path.mkdir(parents=True, exist_ok=True)
    (registration_path / RegistrationArrays.REFERENCE_IMAGE).write_bytes(b"")


def _write_repeat_flags(
    output_root: Path, *, repeat_registration: bool = False, repeat_binarization: bool = False
) -> None:
    """Rewrites the recording's configuration copy with the requested repeat flags."""
    configuration_path = output_root / OUTPUT_DIRECTORY_NAME / SINGLE_RECORDING_CONFIGURATION_FILENAME
    configuration = SingleRecordingConfiguration.from_yaml(file_path=configuration_path)
    configuration.registration.repeat_registration = repeat_registration
    configuration.file_io.repeat_binarization = repeat_binarization
    configuration.save(file_path=configuration_path)


def _prepare_sizable_recording(tmp_path: Path) -> Path:
    """Prepares one recording and returns the per-recording configuration file the sizing tool reads."""
    _, cindra_root = _prepare_recording(tmp_path=tmp_path)
    return cindra_root / SINGLE_RECORDING_CONFIGURATION_FILENAME


def _prepare_sizable_dataset(tmp_path: Path) -> Path:
    """Writes two completed recordings and the dataset configuration naming them, returning that file's path."""
    recording_roots = []
    for name in ("day1", "day2"):
        output_root = tmp_path / name
        output_path = resolve_output_path(output_root=output_root)
        output_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path / COMBINED_METADATA_FILENAME,
            combined_height=np.array([64], dtype=np.uint16),
            combined_width=np.array([64], dtype=np.uint16),
            frame_count=np.array([600], dtype=np.uint32),
        )
        np.save(
            file=resolve_array_path(root_path=output_path, array=RecordingArrays.CELL_FLUORESCENCE),
            arr=np.zeros((40, 600), dtype=np.float32),
        )
        recording_roots.append(output_root)

    configuration = MultiRecordingConfiguration()
    configuration.recording_io.recording_directories = tuple(recording_roots)
    configuration.recording_io.dataset_name = "animal_a_task"
    configuration_path = tmp_path / "dataset.yaml"
    configuration.save(file_path=configuration_path)
    return configuration_path
