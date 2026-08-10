"""Contains tests for the per-stage memory models of the two pipelines."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from cindra.layout import (
    COMBINED_METADATA_FILENAME,
    ACQUISITION_PARAMETERS_FILENAME,
    RecordingArrays,
    resolve_array_path,
    resolve_plane_path,
    resolve_output_path,
    resolve_dataset_path,
)
from cindra.dataclasses import (
    SingleRecordingRuntimeData,
    MultiRecordingConfiguration,
    SingleRecordingConfiguration,
)
from cindra.orchestration import MultiRecordingJobNames, SingleRecordingJobNames
from cindra.orchestration.footprints import (
    WORKER_MEMORY_MB,
    PlaneGeometry,
    RecordingGeometry,
    _apply_tolerance,
    _estimate_discovery_mb,
    _estimate_processing_mb,
    _estimate_registration_mb,
    resolve_recording_geometry,
    _resolve_binned_frame_count,
    _resolve_metric_sample_count,
    estimate_multi_recording_job_memory_mb,
    estimate_single_recording_job_memory_mb,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_recording(
    output_root: Path, plane_count: int = 1, frame_count: int = 6000, height: int = 512, width: int = 512
) -> None:
    """Writes the acquisition parameters and per-plane runtime data of a binarized recording."""
    output_path = resolve_output_path(output_root=output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / ACQUISITION_PARAMETERS_FILENAME).write_text(
        f"frame_rate: 30.0\nplane_number: {plane_count}\nchannel_number: 1\nroi_number: 1\n"
        f"roi_lines: []\nroi_x_coordinates: []\nroi_y_coordinates: []\n"
    )
    for plane_index in range(plane_count):
        plane_path = resolve_plane_path(output_root=output_root, plane_index=plane_index)
        plane_path.mkdir(parents=True, exist_ok=True)
        runtime = SingleRecordingRuntimeData()
        runtime.io.frame_height = height
        runtime.io.frame_width = width
        runtime.io.frame_count = frame_count
        runtime.io.sampling_rate = 30.0
        runtime.io.plane_index = plane_index
        runtime.io.output_path = plane_path
        runtime.save(output_path=plane_path)


def _write_combined(output_root: Path, height: int = 512, width: int = 512, frame_count: int = 6000) -> None:
    """Writes the combined metadata archive that marks a recording processed."""
    output_path = resolve_output_path(output_root=output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path / COMBINED_METADATA_FILENAME,
        combined_height=np.array([height], dtype=np.uint16),
        combined_width=np.array([width], dtype=np.uint16),
        frame_count=np.array([frame_count], dtype=np.uint32),
    )


def _write_traces(root_path: Path, regions: int, samples: int) -> None:
    """Writes a trace array whose header reports the given region and sample counts."""
    root_path.mkdir(parents=True, exist_ok=True)
    np.save(
        resolve_array_path(root_path=root_path, array=RecordingArrays.CELL_FLUORESCENCE),
        np.zeros((regions, samples), dtype=np.float32),
    )


class TestTolerance:
    """Tests the shared reporting scale."""

    def test_estimate_rounds_up_to_a_whole_gigabyte(self) -> None:
        """Verifies that a reportable estimate is a whole number of gigabytes."""
        for memory_mb in (1, 400, 5000):
            assert _apply_tolerance(memory_mb=memory_mb) % 1024 == 0

    def test_estimate_never_understates_the_modeled_figure(self) -> None:
        """Verifies that the reportable estimate is at least the figure the model produced."""
        for memory_mb in (1, 400, 5000, 20000):
            assert _apply_tolerance(memory_mb=memory_mb) >= memory_mb


class TestRegistrationSampleCount:
    """Tests the sample-count step that drives the registration footprint."""

    def test_long_recording_with_small_planes_takes_the_large_sample(self) -> None:
        """Verifies that a long recording of small planes samples the larger frame count."""
        plane = PlaneGeometry(height=512, width=512, frame_count=20000, sampling_rate=30.0)

        assert _resolve_metric_sample_count(plane=plane) == 5000

    @pytest.mark.parametrize(("height", "width"), [(800, 512), (512, 800)])
    def test_large_plane_steps_down_to_the_small_sample(self, height: int, width: int) -> None:
        """Verifies that a plane wider or taller than the threshold samples the smaller frame count."""
        plane = PlaneGeometry(height=height, width=width, frame_count=20000, sampling_rate=30.0)

        assert _resolve_metric_sample_count(plane=plane) == 2000

    def test_short_recording_steps_down_to_the_small_sample(self) -> None:
        """Verifies that a recording shorter than the large-sample threshold samples the smaller frame count."""
        plane = PlaneGeometry(height=512, width=512, frame_count=3000, sampling_rate=30.0)

        assert _resolve_metric_sample_count(plane=plane) == 2000

    def test_sample_count_never_exceeds_the_frames_held(self) -> None:
        """Verifies that a recording shorter than the sample floor samples only the frames it holds."""
        plane = PlaneGeometry(height=512, width=512, frame_count=900, sampling_rate=30.0)

        assert _resolve_metric_sample_count(plane=plane) == 900

    def test_taller_plane_can_hold_a_smaller_working_set(self) -> None:
        """Verifies the inverse-with-extent behavior the sample-count step produces."""
        configuration = SingleRecordingConfiguration()
        small = PlaneGeometry(height=690, width=512, frame_count=20000, sampling_rate=30.0)
        tall = PlaneGeometry(height=710, width=512, frame_count=20000, sampling_rate=30.0)

        assert _estimate_registration_mb(plane=tall, configuration=configuration) < _estimate_registration_mb(
            plane=small, configuration=configuration
        )


class TestProcessingModel:
    """Tests the binned movie model that drives the processing footprint."""

    def test_bin_size_takes_the_coarsest_of_its_three_terms(self) -> None:
        """Verifies that the binned frame count follows the coarsest of the sample, ratio, and decay terms."""
        configuration = SingleRecordingConfiguration()
        plane = PlaneGeometry(height=512, width=512, frame_count=60000, sampling_rate=30.0)

        # The decay term is round(0.4 * 30) = 12, and the ratio term is 60000 // 5000 = 12.
        assert _resolve_binned_frame_count(plane=plane, configuration=configuration) == 5000

    def test_denoising_raises_the_projected_peak(self) -> None:
        """Verifies that enabling PCA denoising raises the projected peak above the plain detection peak."""
        plane = PlaneGeometry(height=512, width=512, frame_count=6000, sampling_rate=30.0)
        plain = SingleRecordingConfiguration()
        denoised = SingleRecordingConfiguration()
        denoised.roi_detection.denoise = True

        assert _estimate_processing_mb(plane=plane, configuration=denoised) > _estimate_processing_mb(
            plane=plane, configuration=plain
        )


class TestDiscoveryModel:
    """Tests the cross-recording registration model."""

    def test_registration_term_grows_linearly_with_the_recording_count(self) -> None:
        """Verifies that adding a recording adds a constant rather than a growing amount of memory."""
        geometry = RecordingGeometry(combined_pixels=512 * 512, combined_frame_count=6000, resolved=True)
        two = _estimate_discovery_mb(geometries=[geometry] * 2)
        three = _estimate_discovery_mb(geometries=[geometry] * 3)
        four = _estimate_discovery_mb(geometries=[geometry] * 4)

        assert (three - two) == (four - three)


class TestSingleRecordingEstimates:
    """Tests the single-recording entry point."""

    def test_absent_recording_reports_the_worker_floor(self, tmp_path: Path) -> None:
        """Verifies that a recording carrying no output reports the baseline and clears the modeled flag."""
        memory_mb, modeled = estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.PROCESS,
            specifier="plane_0",
            output_root=tmp_path,
            configuration=SingleRecordingConfiguration(),
        )

        assert modeled is False
        assert memory_mb == _apply_tolerance(memory_mb=WORKER_MEMORY_MB)

    @pytest.mark.parametrize(
        "job_name",
        [
            SingleRecordingJobNames.BINARIZE,
            SingleRecordingJobNames.REGISTER,
            SingleRecordingJobNames.PROCESS,
            SingleRecordingJobNames.COMBINE,
        ],
    )
    def test_every_stage_models_a_binarized_recording(self, tmp_path: Path, job_name: str) -> None:
        """Verifies that every single-recording stage reports a modeled figure once the recording holds output."""
        _write_recording(output_root=tmp_path, plane_count=2)
        _write_combined(output_root=tmp_path)

        memory_mb, modeled = estimate_single_recording_job_memory_mb(
            job_name=job_name,
            specifier="plane_0",
            output_root=tmp_path,
            configuration=SingleRecordingConfiguration(),
        )

        assert modeled is True
        assert memory_mb > 0

    def test_unresolved_specifier_charges_the_largest_plane(self, tmp_path: Path) -> None:
        """Verifies that a per-plane job whose specifier names no plane is charged the widest per-plane estimate."""
        _write_recording(output_root=tmp_path, plane_count=2)

        named, _ = estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.PROCESS,
            specifier="plane_0",
            output_root=tmp_path,
            configuration=SingleRecordingConfiguration(),
        )
        unnamed, _ = estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.PROCESS,
            specifier="",
            output_root=tmp_path,
            configuration=SingleRecordingConfiguration(),
        )

        assert unnamed >= named

    def test_processing_scales_with_the_plane_extent(self, tmp_path: Path) -> None:
        """Verifies that a wider plane projects a larger processing footprint."""
        small = tmp_path / "small"
        large = tmp_path / "large"
        _write_recording(output_root=small, height=256, width=256)
        _write_recording(output_root=large, height=512, width=512)
        configuration = SingleRecordingConfiguration()

        small_mb, _ = estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.PROCESS,
            specifier="plane_0",
            output_root=small,
            configuration=configuration,
        )
        large_mb, _ = estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.PROCESS,
            specifier="plane_0",
            output_root=large,
            configuration=configuration,
        )

        assert large_mb > small_mb


class TestMultiRecordingEstimates:
    """Tests the multi-recording entry point."""

    def test_absent_dataset_reports_the_clustering_floor(self, tmp_path: Path) -> None:
        """Verifies that a dataset carrying no processed recording reports a floor and clears the modeled flag."""
        memory_mb, modeled = estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.DISCOVER,
            specifier="",
            recording_roots=[tmp_path / "day1"],
            dataset_name="set",
            configuration=MultiRecordingConfiguration(),
        )

        assert modeled is False
        assert memory_mb > 0

    def test_discovery_models_a_processed_dataset(self, tmp_path: Path) -> None:
        """Verifies that discovery reports a modeled figure once the recordings carry combined output."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            _write_combined(output_root=root)

        memory_mb, modeled = estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.DISCOVER,
            specifier="",
            recording_roots=roots,
            dataset_name="set",
            configuration=MultiRecordingConfiguration(),
        )

        assert modeled is True
        assert memory_mb > 0

    def test_extraction_models_the_recording_its_specifier_names(self, tmp_path: Path) -> None:
        """Verifies that extraction reports a modeled figure for the recording its specifier names."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            _write_combined(output_root=root)
        _write_traces(
            root_path=resolve_dataset_path(output_root=roots[1], dataset_name="set"), regions=500, samples=6000
        )

        memory_mb, modeled = estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.EXTRACT,
            specifier="day2",
            recording_roots=roots,
            dataset_name="set",
            configuration=MultiRecordingConfiguration(),
        )

        assert modeled is True
        assert memory_mb > 0


class TestRecordingGeometry:
    """Tests the geometry resolver the estimators read."""

    def test_absent_recording_resolves_unmodeled(self, tmp_path: Path) -> None:
        """Verifies that a recording carrying nothing resolves rather than raising."""
        geometry = resolve_recording_geometry(output_root=tmp_path)

        assert geometry.resolved is False
        assert geometry.planes == ()

    def test_single_region_recording_takes_one_plane_as_its_raw_frame(self, tmp_path: Path) -> None:
        """Verifies that a recording imaging one region holds one plane per acquisition frame."""
        _write_recording(output_root=tmp_path, plane_count=3, height=256, width=256)

        geometry = resolve_recording_geometry(output_root=tmp_path)

        assert geometry.raw_frame_pixels == 256 * 256

    def test_combined_geometry_is_read_from_the_metadata_archive(self, tmp_path: Path) -> None:
        """Verifies that the combined extent and frame count come from the metadata archive."""
        _write_recording(output_root=tmp_path)
        _write_combined(output_root=tmp_path, height=600, width=700, frame_count=4200)

        geometry = resolve_recording_geometry(output_root=tmp_path)

        assert geometry.combined_pixels == 600 * 700
        assert geometry.combined_frame_count == 4200

    def test_region_count_is_read_from_the_trace_header(self, tmp_path: Path) -> None:
        """Verifies that the region count comes from the combined trace array's own header."""
        _write_recording(output_root=tmp_path)
        _write_traces(root_path=resolve_output_path(output_root=tmp_path), regions=321, samples=100)

        geometry = resolve_recording_geometry(output_root=tmp_path)

        assert geometry.region_count == 321


class TestGeometryEdges:
    """Tests the geometry paths a partially written recording exercises."""

    def test_recording_without_planes_reports_the_floor_for_a_per_plane_job(self, tmp_path: Path) -> None:
        """Verifies that a recording carrying combined output but no plane runtime reports the worker floor."""
        _write_combined(output_root=tmp_path)

        memory_mb, modeled = estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.REGISTER,
            specifier="plane_0",
            output_root=tmp_path,
            configuration=SingleRecordingConfiguration(),
        )

        assert modeled is False
        assert memory_mb == _apply_tolerance(memory_mb=WORKER_MEMORY_MB)

    def test_unreadable_plane_runtime_is_skipped(self, tmp_path: Path) -> None:
        """Verifies that a plane whose runtime data cannot be read is left out of the geometry."""
        _write_recording(output_root=tmp_path, plane_count=2)
        (resolve_plane_path(output_root=tmp_path, plane_index=1) / "runtime_data.yaml").unlink()

        geometry = resolve_recording_geometry(output_root=tmp_path)

        assert len(geometry.planes) == 1

    def test_plane_without_frames_is_skipped(self, tmp_path: Path) -> None:
        """Verifies that a plane whose runtime data records no frames is left out of the geometry."""
        _write_recording(output_root=tmp_path, plane_count=1, frame_count=0)

        geometry = resolve_recording_geometry(output_root=tmp_path)

        assert geometry.planes == ()

    def test_multi_region_recording_spans_its_regions_in_one_raw_frame(self, tmp_path: Path) -> None:
        """Verifies that a recording interleaving regions holds every region in each acquisition frame."""
        output_path = resolve_output_path(output_root=tmp_path)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / ACQUISITION_PARAMETERS_FILENAME).write_text(
            "frame_rate: 30.0\nplane_number: 1\nchannel_number: 1\nroi_number: 2\n"
            "roi_lines: [[0, 255], [256, 511]]\nroi_x_coordinates: [0, 0]\nroi_y_coordinates: [0, 256]\n"
        )
        for plane_index in range(2):
            plane_path = resolve_plane_path(output_root=tmp_path, plane_index=plane_index)
            plane_path.mkdir(parents=True, exist_ok=True)
            runtime = SingleRecordingRuntimeData()
            runtime.io.frame_height = 256
            runtime.io.frame_width = 512
            runtime.io.frame_count = 6000
            runtime.io.sampling_rate = 30.0
            runtime.io.plane_index = plane_index
            runtime.io.output_path = plane_path
            runtime.save(output_path=plane_path)

        geometry = resolve_recording_geometry(output_root=tmp_path)

        assert geometry.raw_frame_pixels == 2 * 256 * 512

    def test_metadata_without_a_frame_count_reports_zero_frames(self, tmp_path: Path) -> None:
        """Verifies that a metadata archive predating the frame count field resolves to zero frames."""
        output_path = resolve_output_path(output_root=tmp_path)
        output_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path / COMBINED_METADATA_FILENAME,
            combined_height=np.array([256], dtype=np.uint16),
            combined_width=np.array([256], dtype=np.uint16),
        )

        geometry = resolve_recording_geometry(output_root=tmp_path)

        assert geometry.combined_frame_count == 0

    def test_trace_array_of_another_rank_reports_no_regions(self, tmp_path: Path) -> None:
        """Verifies that a trace array carrying an unexpected rank resolves to no regions."""
        output_path = resolve_output_path(output_root=tmp_path)
        output_path.mkdir(parents=True, exist_ok=True)
        np.save(
            resolve_array_path(root_path=output_path, array=RecordingArrays.CELL_FLUORESCENCE),
            np.zeros((3, 4, 5), dtype=np.float32),
        )

        geometry = resolve_recording_geometry(output_root=tmp_path)

        assert geometry.region_count == 0

    def test_extraction_without_a_matching_specifier_charges_the_widest_recording(self, tmp_path: Path) -> None:
        """Verifies that an extraction job naming no recording is charged the widest readable one."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        _write_combined(output_root=roots[0], height=256, width=256)
        _write_combined(output_root=roots[1], height=512, width=512)

        memory_mb, modeled = estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.EXTRACT,
            specifier="absent",
            recording_roots=roots,
            dataset_name="set",
            configuration=MultiRecordingConfiguration(),
        )

        assert modeled is True
        assert memory_mb > 0

    def test_disabled_metrics_leave_the_reference_stage_as_the_peak(self) -> None:
        """Verifies that a plane whose quality metrics are disabled is charged the reference stage alone."""
        plane = PlaneGeometry(height=512, width=512, frame_count=20000, sampling_rate=30.0)
        configuration = SingleRecordingConfiguration()
        configuration.registration.registration_metric_principal_components = 0

        with_metrics = _estimate_registration_mb(plane=plane, configuration=SingleRecordingConfiguration())

        assert _estimate_registration_mb(plane=plane, configuration=configuration) < with_metrics
