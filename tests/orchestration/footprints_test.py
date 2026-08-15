"""Contains tests for the per-stage memory models of the two pipelines."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from tifffile import TiffWriter
from ataraxis_base_utilities import error_format

from cindra.layout import (
    COMBINED_METADATA_FILENAME,
    ACQUISITION_PARAMETERS_FILENAME,
    RecordingArrays,
    resolve_array_path,
    resolve_plane_path,
    resolve_output_path,
    resolve_dataset_path,
)
from cindra.io.context import PARAMETERS_FILENAME
from cindra.dataclasses import (
    SingleRecordingRuntimeData,
    MultiRecordingConfiguration,
    SingleRecordingConfiguration,
)
from cindra.orchestration import (
    DISCOVERY_WORKERS,
    COMBINATION_WORKERS,
    BINARIZATION_WORKERS,
    MultiRecordingJobNames,
    SingleRecordingJobNames,
)
from cindra.orchestration.footprints import (
    WORKER_MEMORY_MB,
    _DISCOVERY_TRANSIENT_PLANES,
    _DISCOVERY_PLANES_PER_RECORDING,
    _TRACKING_PAIRWISE_BYTES_PER_SQUARED_REGION,
    JobSizing,
    PlaneGeometry,
    RecordingGeometry,
    _apply_tolerance,
    _bytes_to_megabytes,
    _estimate_discovery_mb,
    _estimate_extraction_mb,
    _estimate_processing_mb,
    _estimate_combination_mb,
    size_multi_recording_job,
    _estimate_registration_mb,
    resolve_maximum_roi_count,
    size_single_recording_job,
    resolve_recording_geometry,
    _resolve_binned_frame_count,
    _resolve_metric_sample_count,
    _read_tracked_recording_geometry,
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


def _write_raw_recording(
    data_path: Path,
    pages: int = 600,
    height: int = 24,
    width: int = 16,
    roi_lines: str = "[]",
    roi_number: int = 1,
    plane_number: int = 1,
) -> None:
    """Writes the acquisition metadata and one source file of a recording that has not been converted."""
    data_path.mkdir(parents=True, exist_ok=True)
    (data_path / PARAMETERS_FILENAME).write_text(
        json.dumps(
            {
                "frame_rate": 30.0,
                "plane_number": plane_number,
                "channel_number": 1,
                "roi_number": roi_number,
                "roi_lines": json.loads(roi_lines),
                "roi_x_coordinates": [0] * roi_number,
                "roi_y_coordinates": [0] * roi_number,
            }
        )
    )
    with TiffWriter(data_path / "frames_001.tif") as writer:
        for _ in range(pages):
            writer.write(np.zeros((height, width), dtype=np.int16))


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

        # The decay term is round(0.4 * 30) = 12, and the ratio term is 60000 // 5000 = 12. Each 500 frame read batch
        # bins 41 of its own frames and discards the remaining 8, so 120 batches yield 4920 bins rather than 5000.
        assert _resolve_binned_frame_count(plane=plane, configuration=configuration) == 4920

    def test_binning_truncates_the_remainder_of_every_batch(self) -> None:
        """Verifies that the binned frame count is taken per read batch rather than over the whole movie."""
        configuration = SingleRecordingConfiguration()
        configuration.main.tau = 0.4
        plane = PlaneGeometry(height=512, width=512, frame_count=1000, sampling_rate=30.0)

        # Two 500 frame batches at a bin size of 12 yield 41 bins each, against the 83 a whole-movie division gives.
        assert _resolve_binned_frame_count(plane=plane, configuration=configuration) == 82

    def test_denoising_raises_the_projected_peak(self) -> None:
        """Verifies that enabling PCA denoising raises the projected peak above the plain detection peak."""
        plane = PlaneGeometry(height=512, width=512, frame_count=6000, sampling_rate=30.0)
        plain = SingleRecordingConfiguration()
        denoised = SingleRecordingConfiguration()
        denoised.roi_detection.denoise = True

        assert _estimate_processing_mb(
            plane=plane, configuration=denoised, regions=0, channels=1
        ) > _estimate_processing_mb(plane=plane, configuration=plain, regions=0, channels=1)

    def test_detection_uses_the_registration_crop_when_it_is_known(self) -> None:
        """Verifies that a plane carrying a resolved valid region is charged that region rather than its full extent."""
        configuration = SingleRecordingConfiguration()
        full = PlaneGeometry(height=512, width=512, frame_count=6000, sampling_rate=30.0)
        cropped = PlaneGeometry(height=512, width=512, frame_count=6000, sampling_rate=30.0, valid_pixels=400 * 400)

        assert _estimate_processing_mb(
            plane=cropped, configuration=configuration, regions=0, channels=1
        ) < _estimate_processing_mb(plane=full, configuration=configuration, regions=0, channels=1)

    def test_a_dense_recording_is_charged_its_extraction_traces(self) -> None:
        """Verifies that the extraction term takes over once the regions outgrow the binned movie."""
        configuration = SingleRecordingConfiguration()
        plane = PlaneGeometry(height=64, width=64, frame_count=6000, sampling_rate=30.0)

        assert _estimate_processing_mb(
            plane=plane, configuration=configuration, regions=12500, channels=1
        ) > _estimate_processing_mb(plane=plane, configuration=configuration, regions=1, channels=1)


class TestDiscoveryModel:
    """Tests the cross-recording registration model."""

    def test_registration_term_grows_linearly_with_the_recording_count(self) -> None:
        """Verifies that adding a recording adds a constant rather than a growing amount of memory."""
        geometry = RecordingGeometry(combined_pixels=512 * 512, combined_frame_count=6000, resolved=True)
        two = _estimate_discovery_mb(geometries=[geometry] * 2)
        three = _estimate_discovery_mb(geometries=[geometry] * 3)
        four = _estimate_discovery_mb(geometries=[geometry] * 4)

        assert (three - two) == (four - three)

    def test_clustering_term_grows_with_the_square_of_the_region_count(self) -> None:
        """Verifies that doubling the regions a dataset spans quadruples the clustering term."""
        # The combined frame is small enough that the clustering term dominates the registration term throughout.
        lean = [RecordingGeometry(combined_pixels=64, region_count=20000, resolved=True)] * 2
        dense = [RecordingGeometry(combined_pixels=64, region_count=40000, resolved=True)] * 2

        assert _estimate_discovery_mb(geometries=dense) - WORKER_MEMORY_MB == pytest.approx(
            4 * (_estimate_discovery_mb(geometries=lean) - WORKER_MEMORY_MB), rel=0.01
        )

    def test_both_phases_are_charged_together(self) -> None:
        """Verifies that the registration and clustering terms add, because the first stays resident through the
        second.
        """
        geometries = [RecordingGeometry(combined_pixels=512 * 512, region_count=1000, resolved=True)] * 3

        planes = _DISCOVERY_PLANES_PER_RECORDING * 3 + _DISCOVERY_TRANSIENT_PLANES
        registration = _bytes_to_megabytes(byte_count=planes * 512 * 512 * 4)
        clustering = _bytes_to_megabytes(byte_count=_TRACKING_PAIRWISE_BYTES_PER_SQUARED_REGION * 3000 * 3000)

        assert _estimate_discovery_mb(geometries=geometries) == WORKER_MEMORY_MB + registration + clustering


class TestSingleRecordingEstimates:
    """Tests the single-recording entry point."""

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

        memory_mb = estimate_single_recording_job_memory_mb(
            job_name=job_name,
            specifier="plane_0",
            output_root=tmp_path,
            configuration=SingleRecordingConfiguration(),
        )

        assert memory_mb > 0

    def test_unresolved_specifier_charges_the_largest_plane(self, tmp_path: Path) -> None:
        """Verifies that a per-plane job whose specifier names no plane is charged the widest per-plane estimate."""
        _write_recording(output_root=tmp_path, plane_count=2)

        named = estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.PROCESS,
            specifier="plane_0",
            output_root=tmp_path,
            configuration=SingleRecordingConfiguration(),
        )
        unnamed = estimate_single_recording_job_memory_mb(
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

        small_mb = estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.PROCESS,
            specifier="plane_0",
            output_root=small,
            configuration=configuration,
            planned_roi_count=100,
        )
        large_mb = estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.PROCESS,
            specifier="plane_0",
            output_root=large,
            configuration=configuration,
            planned_roi_count=100,
        )

        assert large_mb > small_mb


class TestMultiRecordingEstimates:
    """Tests the multi-recording entry point."""

    def test_discovery_models_a_processed_dataset(self, tmp_path: Path) -> None:
        """Verifies that discovery reports a modeled figure once the recordings carry combined output."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            _write_combined(output_root=root)

        memory_mb = estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.DISCOVER,
            specifier="",
            recording_directories=roots,
            dataset_name="set",
            configuration=MultiRecordingConfiguration(),
        )

        assert memory_mb > 0

    def test_extraction_models_the_recording_its_specifier_names(self, tmp_path: Path) -> None:
        """Verifies that extraction reports a modeled figure for the recording its specifier names."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            _write_combined(output_root=root)
        _write_traces(
            root_path=resolve_dataset_path(output_root=roots[1], dataset_name="set"), regions=500, samples=6000
        )

        memory_mb = estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.EXTRACT,
            specifier="day2",
            recording_directories=roots,
            dataset_name="set",
            configuration=MultiRecordingConfiguration(),
        )

        assert memory_mb > 0

    def test_extraction_reads_the_tracked_regions_of_the_recording_it_names(self, tmp_path: Path) -> None:
        """Verifies that recordings of identical geometry are charged their own tracked region counts."""
        configuration = MultiRecordingConfiguration()
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            _write_combined(output_root=root, height=64, width=64, frame_count=60000)
        _write_traces(root_path=resolve_dataset_path(output_root=roots[0], dataset_name="set"), regions=40, samples=8)
        _write_traces(
            root_path=resolve_dataset_path(output_root=roots[1], dataset_name="set"), regions=40000, samples=8
        )

        memory_mb = estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.EXTRACT,
            specifier="day2",
            recording_directories=roots,
            dataset_name="set",
            configuration=configuration,
        )

        geometry = RecordingGeometry(combined_pixels=64 * 64, combined_frame_count=60000, resolved=True)
        assert memory_mb == _apply_tolerance(
            memory_mb=_estimate_extraction_mb(geometry=geometry, tracked_regions=40000, configuration=configuration)
        )


class TestJobSizing:
    """Tests the sizing entry points that pair each job's cores with its memory estimate."""

    def test_single_recording_sizing_pairs_the_stage_default_with_the_estimate(self, tmp_path: Path) -> None:
        """Verifies that a sized single-recording job carries its stage's measured cores and its memory estimate."""
        _write_recording(output_root=tmp_path, plane_count=2)
        configuration = SingleRecordingConfiguration()

        sizing = size_single_recording_job(
            job_name=SingleRecordingJobNames.BINARIZE,
            specifier="",
            output_root=tmp_path,
            configuration=configuration,
        )

        assert isinstance(sizing, JobSizing)
        assert sizing.cores == BINARIZATION_WORKERS
        assert sizing.memory_mb == estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.BINARIZE,
            specifier="",
            output_root=tmp_path,
            configuration=configuration,
        )

    def test_multi_recording_sizing_pairs_the_stage_default_with_the_estimate(self, tmp_path: Path) -> None:
        """Verifies that a sized multi-recording job carries its stage's measured cores and its memory estimate."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            _write_combined(output_root=root)
        configuration = MultiRecordingConfiguration()

        sizing = size_multi_recording_job(
            job_name=MultiRecordingJobNames.DISCOVER,
            specifier="",
            recording_directories=roots,
            dataset_name="set",
            configuration=configuration,
        )

        assert sizing.cores == DISCOVERY_WORKERS
        assert sizing.memory_mb == estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.DISCOVER,
            specifier="",
            recording_directories=roots,
            dataset_name="set",
            configuration=configuration,
        )

    @pytest.mark.parametrize("job_name", [SingleRecordingJobNames.BINARIZE, SingleRecordingJobNames.REGISTER])
    def test_a_stage_carrying_no_region_term_is_modeled_before_detection_runs(
        self, tmp_path: Path, job_name: str
    ) -> None:
        """Verifies that a stage whose estimate holds no region term is modeled even with no traces on disk."""
        _write_recording(output_root=tmp_path, plane_count=2)

        sizing = size_single_recording_job(
            job_name=job_name,
            specifier="plane_0",
            output_root=tmp_path,
            configuration=SingleRecordingConfiguration(),
        )

        assert sizing.modeled

    def test_a_budgeted_region_count_reports_an_unmodeled_sizing(self, tmp_path: Path) -> None:
        """Verifies that a region-scaled stage sized against the planning ceiling reports itself unmodeled."""
        _write_recording(output_root=tmp_path, plane_count=2)

        sizing = size_single_recording_job(
            job_name=SingleRecordingJobNames.COMBINE,
            specifier="",
            output_root=tmp_path,
            configuration=SingleRecordingConfiguration(),
        )

        assert sizing.cores == COMBINATION_WORKERS
        assert not sizing.modeled

    def test_a_measured_region_count_reports_a_modeled_sizing(self, tmp_path: Path) -> None:
        """Verifies that a region-scaled stage sized against a written trace array reports itself modeled."""
        _write_recording(output_root=tmp_path, plane_count=2)
        _write_traces(root_path=resolve_output_path(output_root=tmp_path), regions=500, samples=6000)

        sizing = size_single_recording_job(
            job_name=SingleRecordingJobNames.COMBINE,
            specifier="",
            output_root=tmp_path,
            configuration=SingleRecordingConfiguration(),
        )

        assert sizing.modeled

    def test_a_partially_processed_recording_reports_an_unmodeled_sizing(self, tmp_path: Path) -> None:
        """Verifies that a region sum missing a plane's contribution reports itself unmodeled."""
        _write_recording(output_root=tmp_path, plane_count=2)
        _write_traces(root_path=resolve_plane_path(output_root=tmp_path, plane_index=0), regions=300, samples=6000)

        sizing = size_single_recording_job(
            job_name=SingleRecordingJobNames.COMBINE,
            specifier="",
            output_root=tmp_path,
            configuration=SingleRecordingConfiguration(),
        )

        assert not sizing.modeled

    def test_a_fully_processed_recording_reports_a_modeled_sizing(self, tmp_path: Path) -> None:
        """Verifies that a region sum every plane contributed to reports itself modeled."""
        _write_recording(output_root=tmp_path, plane_count=2)
        for plane_index in range(2):
            _write_traces(
                root_path=resolve_plane_path(output_root=tmp_path, plane_index=plane_index),
                regions=300,
                samples=6000,
            )

        sizing = size_single_recording_job(
            job_name=SingleRecordingJobNames.COMBINE,
            specifier="",
            output_root=tmp_path,
            configuration=SingleRecordingConfiguration(),
        )

        assert sizing.modeled

    def test_discovery_without_recording_region_counts_reports_an_unmodeled_sizing(self, tmp_path: Path) -> None:
        """Verifies that discovery sized before any recording reports its regions is unmodeled."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            _write_combined(output_root=root)

        sizing = size_multi_recording_job(
            job_name=MultiRecordingJobNames.DISCOVER,
            specifier="",
            recording_directories=roots,
            dataset_name="set",
            configuration=MultiRecordingConfiguration(),
        )

        assert not sizing.modeled

    def test_discovery_with_every_recording_counted_reports_a_modeled_sizing(self, tmp_path: Path) -> None:
        """Verifies that discovery sized once every recording reports its regions is modeled."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            _write_combined(output_root=root)
            _write_traces(root_path=resolve_output_path(output_root=root), regions=400, samples=6000)

        sizing = size_multi_recording_job(
            job_name=MultiRecordingJobNames.DISCOVER,
            specifier="",
            recording_directories=roots,
            dataset_name="set",
            configuration=MultiRecordingConfiguration(),
        )

        assert sizing.modeled

    def test_extraction_without_tracked_traces_reports_an_unmodeled_sizing(self, tmp_path: Path) -> None:
        """Verifies that extraction sized against the dataset's planning figure reports itself unmodeled."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            _write_combined(output_root=root)

        sizing = size_multi_recording_job(
            job_name=MultiRecordingJobNames.EXTRACT,
            specifier="day2",
            recording_directories=roots,
            dataset_name="set",
            configuration=MultiRecordingConfiguration(),
        )

        assert not sizing.modeled

    def test_extraction_with_tracked_traces_reports_a_modeled_sizing(self, tmp_path: Path) -> None:
        """Verifies that extraction sized against the dataset's own trace array reports itself modeled."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            _write_combined(output_root=root)
        _write_traces(
            root_path=resolve_dataset_path(output_root=roots[1], dataset_name="set"), regions=500, samples=6000
        )

        sizing = size_multi_recording_job(
            job_name=MultiRecordingJobNames.EXTRACT,
            specifier="day2",
            recording_directories=roots,
            dataset_name="set",
            configuration=MultiRecordingConfiguration(),
        )

        assert sizing.modeled


class TestUnsizableJobs:
    """Tests the rejection every estimator issues for a job whose recording can run no stage at all."""

    @pytest.mark.parametrize(
        "job_name",
        [
            SingleRecordingJobNames.BINARIZE,
            SingleRecordingJobNames.REGISTER,
            SingleRecordingJobNames.PROCESS,
            SingleRecordingJobNames.COMBINE,
        ],
    )
    def test_recording_without_output_or_raw_data_is_rejected(self, tmp_path: Path, job_name: str) -> None:
        """Verifies that every single-recording stage rejects a recording that resolves no geometry."""
        message = (
            f"Unable to estimate the memory of the '{job_name}' job. The recording configured with the output root "
            f"{tmp_path} carries neither pipeline output nor readable raw imaging data, so no stage of it can run. "
            f"Verify that the configured data path holds the recording's source files."
        )

        with pytest.raises(FileNotFoundError, match=error_format(message=message)):
            estimate_single_recording_job_memory_mb(
                job_name=job_name,
                specifier="plane_0",
                output_root=tmp_path,
                configuration=SingleRecordingConfiguration(),
            )

    def test_specifier_naming_an_absent_plane_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that a per-plane job naming a plane the recording does not hold is rejected."""
        _write_recording(output_root=tmp_path, plane_count=2)
        message = (
            f"Unable to estimate the memory of the '{SingleRecordingJobNames.PROCESS}' job. Its specifier names "
            f"imaging plane 'plane_7', which the recording configured with the output root {tmp_path} does not hold. "
            f"The recording holds 2 plane(s)."
        )

        with pytest.raises(ValueError, match=error_format(message=message)):
            estimate_single_recording_job_memory_mb(
                job_name=SingleRecordingJobNames.PROCESS,
                specifier="plane_7",
                output_root=tmp_path,
                configuration=SingleRecordingConfiguration(),
            )

    @pytest.mark.parametrize("job_name", [MultiRecordingJobNames.DISCOVER, MultiRecordingJobNames.EXTRACT])
    def test_dataset_without_combined_output_is_rejected(self, tmp_path: Path, job_name: str) -> None:
        """Verifies that both multi-recording stages reject a dataset holding no combined output."""
        with pytest.raises(FileNotFoundError, match="combined metadata archive"):
            estimate_multi_recording_job_memory_mb(
                job_name=job_name,
                specifier="day1",
                recording_directories=[tmp_path / "day1", tmp_path / "day2"],
                dataset_name="set",
                configuration=MultiRecordingConfiguration(),
            )

    def test_dataset_spanning_no_recording_is_rejected(self) -> None:
        """Verifies that a dataset naming no recording at all is rejected."""
        with pytest.raises(FileNotFoundError, match="combined metadata archive"):
            estimate_multi_recording_job_memory_mb(
                job_name=MultiRecordingJobNames.EXTRACT,
                specifier="",
                recording_directories=[],
                dataset_name="set",
                configuration=MultiRecordingConfiguration(),
            )


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

    def test_recording_without_planes_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that a recording carrying no plane runtime is charged its stage's allowance."""
        _write_combined(output_root=tmp_path)

        with pytest.raises((FileNotFoundError, ValueError)):
            estimate_single_recording_job_memory_mb(
                job_name=SingleRecordingJobNames.REGISTER,
                specifier="plane_0",
                output_root=tmp_path,
                configuration=SingleRecordingConfiguration(),
            )

    def test_unreadable_plane_runtime_is_skipped(self, tmp_path: Path) -> None:
        """Verifies that a plane whose runtime data cannot be read is left out of the geometry."""
        _write_recording(output_root=tmp_path, plane_count=2)
        (resolve_plane_path(output_root=tmp_path, plane_index=1) / "runtime_data.yaml").unlink()

        geometry = resolve_recording_geometry(output_root=tmp_path)

        assert len(geometry.planes) == 1

    def test_specifier_naming_an_unreadable_plane_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that a per-plane job naming a plane without runtime data is charged its stage's allowance."""
        _write_recording(output_root=tmp_path, plane_count=2)
        (resolve_plane_path(output_root=tmp_path, plane_index=1) / "runtime_data.yaml").unlink()

        with pytest.raises((FileNotFoundError, ValueError)):
            estimate_single_recording_job_memory_mb(
                job_name=SingleRecordingJobNames.PROCESS,
                specifier="plane_1",
                output_root=tmp_path,
                configuration=SingleRecordingConfiguration(),
            )

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

        memory_mb = estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.EXTRACT,
            specifier="absent",
            recording_directories=roots,
            dataset_name="set",
            configuration=MultiRecordingConfiguration(),
        )

        assert memory_mb > 0

    def test_disabled_metrics_leave_the_reference_stage_as_the_peak(self) -> None:
        """Verifies that a plane whose quality metrics are disabled is charged the reference stage alone."""
        plane = PlaneGeometry(height=512, width=512, frame_count=20000, sampling_rate=30.0)
        configuration = SingleRecordingConfiguration()
        configuration.registration.registration_metric_principal_components = 0

        with_metrics = _estimate_registration_mb(plane=plane, configuration=SingleRecordingConfiguration())

        assert _estimate_registration_mb(plane=plane, configuration=configuration) < with_metrics


class TestDualChannelRecordings:
    """Tests the stages that process both channels inside one job."""

    def test_second_channel_doubles_the_combination_estimate(self) -> None:
        """Verifies that a recording carrying a second channel doubles the traces combination concatenates."""
        single = RecordingGeometry(region_count=500, combined_frame_count=6000, resolved=True)
        dual = RecordingGeometry(region_count=500, combined_frame_count=6000, two_channels=True, resolved=True)

        trace_bytes = _estimate_combination_mb(geometry=dual, regions=500) - WORKER_MEMORY_MB
        assert trace_bytes == pytest.approx(
            2 * (_estimate_combination_mb(geometry=single, regions=500) - WORKER_MEMORY_MB), rel=0.01
        )

    def test_second_channel_raises_the_extraction_estimate(self) -> None:
        """Verifies that a recording carrying a second channel raises the traces extraction holds at once."""
        configuration = MultiRecordingConfiguration()
        single = RecordingGeometry(combined_pixels=512 * 512, combined_frame_count=6000, resolved=True)
        dual = RecordingGeometry(combined_pixels=512 * 512, combined_frame_count=6000, two_channels=True, resolved=True)

        dual_mb = _estimate_extraction_mb(geometry=dual, tracked_regions=500, configuration=configuration)
        single_mb = _estimate_extraction_mb(geometry=single, tracked_regions=500, configuration=configuration)

        assert dual_mb > single_mb

    def test_second_channel_is_read_from_the_combined_metadata(self, tmp_path: Path) -> None:
        """Verifies that the geometry reports a second channel when the metadata archive records its binaries."""
        output_path = resolve_output_path(output_root=tmp_path)
        output_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path / COMBINED_METADATA_FILENAME,
            combined_height=np.array([256], dtype=np.uint16),
            combined_width=np.array([256], dtype=np.uint16),
            frame_count=np.array([100], dtype=np.uint32),
            registered_binary_paths_channel_2=np.array(["a.bin"]),
        )

        assert resolve_recording_geometry(output_root=tmp_path).two_channels is True


class TestRawGeometryDerivation:
    """Tests the geometry the estimator derives before any job has written output."""

    def test_plane_geometry_follows_the_acquisition_and_the_source_header(self, tmp_path: Path) -> None:
        """Verifies that a recording carrying only raw data resolves the shape its conversion will write."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path)

        geometry = resolve_recording_geometry(output_root=tmp_path / "out", data_path=data_path)

        assert geometry.resolved is True
        assert [(plane.height, plane.width) for plane in geometry.planes] == [(24, 16)]
        assert geometry.planes[0].frame_count == 600
        assert geometry.raw_frame_pixels == 24 * 16
        assert geometry.source_element_bytes == 2

    def test_multi_region_planes_take_their_line_spans(self, tmp_path: Path) -> None:
        """Verifies that a multi-region recording derives one plane per region from its line spans."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path, height=30, roi_lines="[[0, 9], [12, 29]]", roi_number=2)

        geometry = resolve_recording_geometry(output_root=tmp_path / "out", data_path=data_path)

        assert [(plane.height, plane.index) for plane in geometry.planes] == [(10, 0), (18, 1)]
        # The raw frame spans the whole acquisition page, including the lines separating the two regions.
        assert geometry.raw_frame_pixels == 30 * 16

    def test_every_stage_resolves_before_the_recording_is_converted(self, tmp_path: Path) -> None:
        """Verifies that each single-recording stage reports a modeled figure from raw data alone."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path)
        configuration = SingleRecordingConfiguration()

        for job_name, specifier in (
            (SingleRecordingJobNames.BINARIZE, ""),
            (SingleRecordingJobNames.REGISTER, "plane_0"),
            (SingleRecordingJobNames.PROCESS, "plane_0"),
            (SingleRecordingJobNames.COMBINE, ""),
        ):
            memory_mb = estimate_single_recording_job_memory_mb(
                job_name=job_name,
                specifier=specifier,
                output_root=tmp_path / "out",
                configuration=configuration,
                data_path=data_path,
            )

            assert memory_mb > 0

    def test_a_recording_short_of_one_interleave_cycle_resolves_no_plane(self, tmp_path: Path) -> None:
        """Verifies that a recording holding fewer frames than one whole cycle derives no plane geometry."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path, pages=2, plane_number=4)

        geometry = resolve_recording_geometry(output_root=tmp_path / "out", data_path=data_path)

        assert geometry.planes == ()

    def test_unreadable_raw_data_leaves_the_recording_unresolved(self, tmp_path: Path) -> None:
        """Verifies that a recording whose raw directory holds no source file resolves no geometry."""
        data_path = tmp_path / "raw"
        data_path.mkdir(parents=True, exist_ok=True)

        geometry = resolve_recording_geometry(output_root=tmp_path / "out", data_path=data_path)

        assert geometry.resolved is False
        assert geometry.planes == ()


class TestPlannedRegionCount:
    """Tests the region budget the region-scaled estimates fall back to."""

    def test_ceiling_scales_with_the_plane_count(self) -> None:
        """Verifies that the provable ceiling is the per-plane detection bound taken across the planes."""
        configuration = SingleRecordingConfiguration()

        single = resolve_maximum_roi_count(plane_count=1, configuration=configuration)
        triple = resolve_maximum_roi_count(plane_count=3, configuration=configuration)

        assert single == 250 * configuration.roi_detection.maximum_iterations
        assert triple == 3 * single

    def test_planned_count_raises_the_combination_estimate(self, tmp_path: Path) -> None:
        """Verifies that a larger planned region count charges the combination job more."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path)
        configuration = SingleRecordingConfiguration()

        small = estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.COMBINE,
            specifier="",
            output_root=tmp_path / "out",
            configuration=configuration,
            data_path=data_path,
            planned_roi_count=100,
        )
        large = estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.COMBINE,
            specifier="",
            output_root=tmp_path / "out",
            configuration=configuration,
            data_path=data_path,
            planned_roi_count=100000,
        )

        assert large > small

    def test_processed_planes_supersede_the_planned_count(self, tmp_path: Path) -> None:
        """Verifies that the per-plane trace headers are preferred once the planes have been processed."""
        _write_recording(output_root=tmp_path, plane_count=2)
        for plane_index in range(2):
            _write_traces(
                root_path=resolve_plane_path(output_root=tmp_path, plane_index=plane_index), regions=7, samples=8
            )
        configuration = SingleRecordingConfiguration()

        geometry = resolve_recording_geometry(output_root=tmp_path)
        planned = estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.COMBINE,
            specifier="",
            output_root=tmp_path,
            configuration=configuration,
            planned_roi_count=100000,
        )

        assert sum(plane.region_count for plane in geometry.planes) == 14
        # The planned count is ignored, so the enormous budget does not reach the estimate.
        assert planned == _apply_tolerance(memory_mb=_estimate_combination_mb(geometry=geometry, regions=14))

    def test_combined_regions_supersede_the_per_plane_sum(self, tmp_path: Path) -> None:
        """Verifies that the combined trace array is preferred once the combination stage has written it."""
        _write_recording(output_root=tmp_path)
        _write_combined(output_root=tmp_path)
        _write_traces(root_path=resolve_plane_path(output_root=tmp_path, plane_index=0), regions=7, samples=8)
        _write_traces(root_path=resolve_output_path(output_root=tmp_path), regions=41, samples=6000)

        geometry = resolve_recording_geometry(output_root=tmp_path)

        assert geometry.region_count == 41

        memory_mb = estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.COMBINE,
            specifier="",
            output_root=tmp_path,
            configuration=SingleRecordingConfiguration(),
            planned_roi_count=100000,
        )

        assert memory_mb == _apply_tolerance(memory_mb=_estimate_combination_mb(geometry=geometry, regions=41))

    def test_non_positive_planned_count_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that a planned region count of zero or less is rejected."""
        message = (
            f"Unable to estimate the memory of the '{SingleRecordingJobNames.BINARIZE}' job. The planned region count "
            f"must be a positive integer counting every plane together, or None to accept the detection ceiling, but "
            f"encountered 0."
        )

        with pytest.raises(ValueError, match=error_format(message=message)):
            estimate_single_recording_job_memory_mb(
                job_name=SingleRecordingJobNames.BINARIZE,
                specifier="",
                output_root=tmp_path,
                configuration=SingleRecordingConfiguration(),
                planned_roi_count=0,
            )


class TestTrackedTemplatePlanning:
    """Tests the tracked template count an extraction job is planned for before discovery produces one."""

    def _write_dataset(self, tmp_path: Path, probabilities: list[float]) -> list[Path]:
        """Writes a two-recording dataset whose classification arrays carry the given probabilities."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            _write_combined(output_root=root)
            _write_traces(root_path=resolve_output_path(output_root=root), regions=len(probabilities), samples=6000)
            np.save(
                resolve_array_path(
                    root_path=resolve_output_path(output_root=root), array=RecordingArrays.CELL_CLASSIFICATION
                ),
                np.stack([np.zeros(len(probabilities), dtype=np.float32), np.array(probabilities, np.float32)], axis=1),
            )
        return [resolve_output_path(output_root=root) for root in roots]

    def test_selection_threshold_narrows_the_clustering_term(self, tmp_path: Path) -> None:
        """Verifies that the regions failing the selection threshold do not enter the discovery estimate."""
        lenient = self._write_dataset(tmp_path=tmp_path / "lenient", probabilities=[0.9] * 400)
        strict = self._write_dataset(tmp_path=tmp_path / "strict", probabilities=[0.9] * 100 + [0.1] * 300)
        configuration = MultiRecordingConfiguration()
        configuration.roi_selection.probability_threshold = 0.8

        lenient_mb = estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.DISCOVER,
            specifier="",
            recording_directories=lenient,
            dataset_name="set",
            configuration=configuration,
        )
        strict_mb = estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.DISCOVER,
            specifier="",
            recording_directories=strict,
            dataset_name="set",
            configuration=configuration,
        )

        assert strict_mb <= lenient_mb

    def test_planned_template_count_raises_the_extraction_estimate(self, tmp_path: Path) -> None:
        """Verifies that planning for more tracked templates charges an extraction job more."""
        directories = self._write_dataset(tmp_path=tmp_path, probabilities=[0.9] * 100)
        configuration = MultiRecordingConfiguration()

        lean = estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.EXTRACT,
            specifier="day1",
            recording_directories=directories,
            dataset_name="set",
            configuration=configuration,
            planned_roi_count=100,
        )
        dense = estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.EXTRACT,
            specifier="day1",
            recording_directories=directories,
            dataset_name="set",
            configuration=configuration,
            planned_roi_count=100000,
        )

        assert dense > lean

    def test_non_positive_planned_template_count_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that a planned tracked template count of zero or less is rejected."""
        message = (
            f"Unable to estimate the memory of the '{MultiRecordingJobNames.EXTRACT}' job. The planned tracked "
            f"template count must be a positive integer, or None to accept the dataset's own planning figure, but "
            f"encountered -5."
        )

        with pytest.raises(ValueError, match=error_format(message=message)):
            estimate_multi_recording_job_memory_mb(
                job_name=MultiRecordingJobNames.EXTRACT,
                specifier="day1",
                recording_directories=[tmp_path],
                dataset_name="set",
                configuration=MultiRecordingConfiguration(),
                planned_roi_count=-5,
            )

    def test_unreadable_classification_leaves_the_selection_count_unresolved(self, tmp_path: Path) -> None:
        """Verifies that a recording whose classification array is malformed contributes no selection count."""
        root = tmp_path / "day1"
        _write_combined(output_root=root)
        np.save(
            resolve_array_path(
                root_path=resolve_output_path(output_root=root), array=RecordingArrays.CELL_CLASSIFICATION
            ),
            np.zeros(7, dtype=np.float32),
        )

        geometry = _read_tracked_recording_geometry(
            cindra_root=resolve_output_path(output_root=root), probability_threshold=0.8
        )

        assert geometry.selected_region_count == 0

    def test_corrupt_classification_leaves_the_selection_count_unresolved(self, tmp_path: Path) -> None:
        """Verifies that a classification file that is not a readable array contributes no selection count."""
        root = tmp_path / "day1"
        _write_combined(output_root=root)
        resolve_array_path(
            root_path=resolve_output_path(output_root=root), array=RecordingArrays.CELL_CLASSIFICATION
        ).write_bytes(b"not an array")

        geometry = _read_tracked_recording_geometry(
            cindra_root=resolve_output_path(output_root=root), probability_threshold=0.8
        )

        assert geometry.selected_region_count == 0


class TestDatasetDirectoryResolution:
    """Tests the latitude the multi-recording estimator allows in the directories a dataset names."""

    def test_a_configured_output_directory_is_used_directly(self, tmp_path: Path) -> None:
        """Verifies that a directory already holding the combined archive is read without a tree search."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            _write_combined(output_root=root)

        memory_mb = estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.DISCOVER,
            specifier="",
            recording_directories=[resolve_output_path(output_root=root) for root in roots],
            dataset_name="set",
            configuration=MultiRecordingConfiguration(),
        )

        assert memory_mb > 0
