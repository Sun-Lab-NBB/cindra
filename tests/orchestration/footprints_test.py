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


def _write_raw_recording(
    data_path: Path,
    pages: int = 600,
    height: int = 24,
    width: int = 16,
    roi_lines: str = "[]",
    roi_number: int = 1,
    plane_number: int = 1,
    channel_number: int = 1,
) -> None:
    """Writes the acquisition metadata and one source file of a recording, which is all a sizing pass reads."""
    data_path.mkdir(parents=True, exist_ok=True)
    (data_path / PARAMETERS_FILENAME).write_text(
        json.dumps(
            {
                "frame_rate": 30.0,
                "plane_number": plane_number,
                "channel_number": channel_number,
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


def _write_recording(
    output_root: Path, plane_count: int = 1, frame_count: int = 6000, height: int = 512, width: int = 512
) -> None:
    """Writes the acquisition parameters and per-plane runtime data a converted recording carries."""
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


def _write_tracked_recording(output_root: Path, height: int, width: int, frame_count: int, regions: int) -> None:
    """Writes the combined archive and trace array a completed single-recording pipeline leaves for the tracker."""
    _write_combined(output_root=output_root, height=height, width=width, frame_count=frame_count)
    _write_traces(root_path=resolve_output_path(output_root=output_root), regions=regions, samples=frame_count)


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

    def test_an_empty_working_set_converts_to_no_megabytes(self) -> None:
        """Verifies that a model holding no bytes is charged no megabytes rather than a rounded-up one."""
        assert _bytes_to_megabytes(byte_count=0) == 0


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

    def test_disabled_metrics_leave_the_reference_stage_as_the_peak(self) -> None:
        """Verifies that a plane whose quality metrics are disabled is charged the reference stage alone."""
        plane = PlaneGeometry(height=512, width=512, frame_count=20000, sampling_rate=30.0)
        configuration = SingleRecordingConfiguration()
        configuration.registration.registration_metric_principal_components = 0

        with_metrics = _estimate_registration_mb(plane=plane, configuration=SingleRecordingConfiguration())

        assert _estimate_registration_mb(plane=plane, configuration=configuration) < with_metrics


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

    def test_detection_scales_with_the_plane_extent(self) -> None:
        """Verifies that a wider plane projects a larger detection working set."""
        configuration = SingleRecordingConfiguration()
        small = PlaneGeometry(height=256, width=256, frame_count=6000, sampling_rate=30.0)
        large = PlaneGeometry(height=512, width=512, frame_count=6000, sampling_rate=30.0)

        assert _estimate_processing_mb(
            plane=large, configuration=configuration, regions=0, channels=1
        ) > _estimate_processing_mb(plane=small, configuration=configuration, regions=0, channels=1)

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


class TestRecordingGeometry:
    """Tests the geometry resolver, which reads the raw acquisition alone."""

    def test_absent_recording_resolves_nothing(self, tmp_path: Path) -> None:
        """Verifies that a recording carrying nothing resolves rather than raising."""
        geometry = resolve_recording_geometry(output_root=tmp_path)

        assert geometry.resolved is False
        assert geometry.planes == ()
        assert geometry.raw_frame_pixels == 0

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

    def test_interleaved_planes_split_the_source_frames(self, tmp_path: Path) -> None:
        """Verifies that every plane of an interleaved recording receives one frame per acquisition cycle."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path, plane_number=3)

        geometry = resolve_recording_geometry(output_root=tmp_path / "out", data_path=data_path)

        assert [plane.index for plane in geometry.planes] == [0, 1, 2]
        assert {plane.frame_count for plane in geometry.planes} == {200}
        assert {plane.sampling_rate for plane in geometry.planes} == {10.0}

    def test_multi_region_planes_take_their_line_spans(self, tmp_path: Path) -> None:
        """Verifies that a multi-region recording derives one plane per region from its line spans."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path, height=30, roi_lines="[[0, 9], [12, 29]]", roi_number=2)

        geometry = resolve_recording_geometry(output_root=tmp_path / "out", data_path=data_path)

        assert [(plane.height, plane.index) for plane in geometry.planes] == [(10, 0), (18, 1)]
        # The raw frame spans the whole acquisition page, including the lines separating the two regions.
        assert geometry.raw_frame_pixels == 30 * 16

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
    def test_every_stage_models_a_recording_before_it_is_converted(self, tmp_path: Path, job_name: str) -> None:
        """Verifies that every single-recording stage reports a figure from the raw acquisition alone."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path, plane_number=2)

        memory_mb = estimate_single_recording_job_memory_mb(
            job_name=job_name,
            specifier="plane_0",
            output_root=tmp_path / "out",
            configuration=SingleRecordingConfiguration(),
            data_path=data_path,
        )

        assert memory_mb > 0

    def test_written_output_does_not_change_the_reported_figure(self, tmp_path: Path) -> None:
        """Verifies that a recording sizes identically before and after its pipeline output exists."""
        data_path = tmp_path / "raw"
        output_root = tmp_path / "out"
        _write_raw_recording(data_path=data_path, plane_number=2)
        configuration = SingleRecordingConfiguration()
        jobs = (
            (SingleRecordingJobNames.BINARIZE, ""),
            (SingleRecordingJobNames.REGISTER, "plane_0"),
            (SingleRecordingJobNames.PROCESS, "plane_1"),
            (SingleRecordingJobNames.COMBINE, ""),
        )

        before = [
            estimate_single_recording_job_memory_mb(
                job_name=job_name,
                specifier=specifier,
                output_root=output_root,
                configuration=configuration,
                data_path=data_path,
            )
            for job_name, specifier in jobs
        ]

        # The written output deliberately disagrees with the raw acquisition on every shape it records, so a sizing
        # pass that consulted any of it would report a different figure once the pipeline has run.
        _write_recording(output_root=output_root, plane_count=2, frame_count=99999, height=999, width=999)
        _write_combined(output_root=output_root, height=999, width=999, frame_count=99999)
        _write_traces(root_path=resolve_output_path(output_root=output_root), regions=41, samples=300)
        for plane_index in range(2):
            _write_traces(
                root_path=resolve_plane_path(output_root=output_root, plane_index=plane_index), regions=7, samples=300
            )
        after = [
            estimate_single_recording_job_memory_mb(
                job_name=job_name,
                specifier=specifier,
                output_root=output_root,
                configuration=configuration,
                data_path=data_path,
            )
            for job_name, specifier in jobs
        ]

        assert after == before

    def test_a_named_plane_is_charged_that_plane_and_an_unnamed_one_the_largest(self, tmp_path: Path) -> None:
        """Verifies that a per-plane job takes its specifier's plane and an unmatched one the widest per-plane peak."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path, height=30, roi_lines="[[0, 9], [12, 29]]", roi_number=2)
        configuration = SingleRecordingConfiguration()
        geometry = resolve_recording_geometry(output_root=tmp_path / "out", data_path=data_path)
        regions = min(
            resolve_maximum_roi_count(plane_count=len(geometry.planes), configuration=configuration),
            resolve_maximum_roi_count(plane_count=1, configuration=configuration),
        )

        named = estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.PROCESS,
            specifier="plane_0",
            output_root=tmp_path / "out",
            configuration=configuration,
            data_path=data_path,
        )
        unnamed = estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.PROCESS,
            specifier="",
            output_root=tmp_path / "out",
            configuration=configuration,
            data_path=data_path,
        )

        assert named == _apply_tolerance(
            memory_mb=_estimate_processing_mb(
                plane=geometry.planes[0], configuration=configuration, regions=regions, channels=1
            )
        )
        assert unnamed == _apply_tolerance(
            memory_mb=max(
                _estimate_processing_mb(plane=plane, configuration=configuration, regions=regions, channels=1)
                for plane in geometry.planes
            )
        )

    def test_registration_is_charged_the_widest_plane_of_an_unnamed_job(self, tmp_path: Path) -> None:
        """Verifies that a registration job naming no plane is charged the largest per-plane estimate."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path, height=30, roi_lines="[[0, 9], [12, 29]]", roi_number=2)
        configuration = SingleRecordingConfiguration()
        geometry = resolve_recording_geometry(output_root=tmp_path / "out", data_path=data_path)

        memory_mb = estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.REGISTER,
            specifier="",
            output_root=tmp_path / "out",
            configuration=configuration,
            data_path=data_path,
        )

        assert memory_mb == _apply_tolerance(
            memory_mb=max(
                _estimate_registration_mb(plane=plane, configuration=configuration) for plane in geometry.planes
            )
        )


class TestMultiRecordingEstimates:
    """Tests the multi-recording entry point."""

    def test_discovery_sums_the_regions_every_recording_reports(self, tmp_path: Path) -> None:
        """Verifies that discovery is charged the combined extent and the summed regions of the whole dataset."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        _write_tracked_recording(output_root=roots[0], height=64, width=64, frame_count=6000, regions=300)
        _write_tracked_recording(output_root=roots[1], height=64, width=64, frame_count=6000, regions=500)

        memory_mb = estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.DISCOVER,
            specifier="",
            recording_directories=roots,
            configuration=MultiRecordingConfiguration(),
        )

        geometries = [
            RecordingGeometry(combined_pixels=4096, combined_frame_count=6000, region_count=300, resolved=True),
            RecordingGeometry(combined_pixels=4096, combined_frame_count=6000, region_count=500, resolved=True),
        ]
        assert memory_mb == _apply_tolerance(memory_mb=_estimate_discovery_mb(geometries=geometries))

    def test_extraction_models_the_recording_its_specifier_names(self, tmp_path: Path) -> None:
        """Verifies that extraction is charged the geometry of the recording its specifier names."""
        configuration = MultiRecordingConfiguration()
        roots = [tmp_path / "day1", tmp_path / "day2"]
        # The named recording is the smaller of the two and holds the fewer regions, so an estimate that ignored the
        # specifier or read its own recording's regions alone would report a different figure.
        _write_tracked_recording(output_root=roots[0], height=256, width=256, frame_count=60000, regions=700)
        _write_tracked_recording(output_root=roots[1], height=64, width=64, frame_count=600, regions=400)

        memory_mb = estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.EXTRACT,
            specifier="day2",
            recording_directories=roots,
            configuration=configuration,
        )

        geometry = RecordingGeometry(combined_pixels=4096, combined_frame_count=600, region_count=400, resolved=True)
        assert memory_mb == _apply_tolerance(
            memory_mb=_estimate_extraction_mb(geometry=geometry, tracked_regions=700, configuration=configuration)
        )

    def test_extraction_without_a_matching_specifier_charges_the_widest_recording(self, tmp_path: Path) -> None:
        """Verifies that an extraction job naming no recording is charged the widest readable one."""
        configuration = MultiRecordingConfiguration()
        roots = [tmp_path / "day1", tmp_path / "day2"]
        _write_tracked_recording(output_root=roots[0], height=64, width=64, frame_count=600, regions=400)
        _write_tracked_recording(output_root=roots[1], height=256, width=256, frame_count=600, regions=500)

        memory_mb = estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.EXTRACT,
            specifier="absent",
            recording_directories=roots,
            configuration=configuration,
        )

        widest = RecordingGeometry(combined_pixels=65536, combined_frame_count=600, region_count=500, resolved=True)
        assert memory_mb == _apply_tolerance(
            memory_mb=_estimate_extraction_mb(geometry=widest, tracked_regions=500, configuration=configuration)
        )


class TestTrackedRecordingGeometry:
    """Tests the reader every multi-recording model resolves one recording's shape through."""

    def test_region_count_is_read_from_the_trace_header(self, tmp_path: Path) -> None:
        """Verifies that the region count comes from the combined trace array's own header."""
        _write_tracked_recording(output_root=tmp_path, height=64, width=64, frame_count=600, regions=321)

        geometry = _read_tracked_recording_geometry(cindra_root=resolve_output_path(output_root=tmp_path))

        assert geometry.region_count == 321
        assert geometry.resolved is True

    def test_a_recording_without_combined_output_resolves_nothing(self, tmp_path: Path) -> None:
        """Verifies that a recording carrying no metadata archive contributes no geometry."""
        geometry = _read_tracked_recording_geometry(cindra_root=resolve_output_path(output_root=tmp_path))

        assert geometry.resolved is False
        assert geometry.combined_pixels == 0

    def test_an_absent_trace_array_reports_no_regions(self, tmp_path: Path) -> None:
        """Verifies that a recording whose combination stage wrote no trace array reports no regions."""
        _write_combined(output_root=tmp_path)

        geometry = _read_tracked_recording_geometry(cindra_root=resolve_output_path(output_root=tmp_path))

        assert geometry.region_count == 0

    def test_a_trace_array_of_another_rank_reports_no_regions(self, tmp_path: Path) -> None:
        """Verifies that a trace array carrying an unexpected rank resolves to no regions."""
        _write_combined(output_root=tmp_path)
        np.save(
            resolve_array_path(
                root_path=resolve_output_path(output_root=tmp_path), array=RecordingArrays.CELL_FLUORESCENCE
            ),
            np.zeros((3, 4, 5), dtype=np.float32),
        )

        geometry = _read_tracked_recording_geometry(cindra_root=resolve_output_path(output_root=tmp_path))

        assert geometry.region_count == 0

    def test_metadata_without_a_frame_count_reports_zero_frames(self, tmp_path: Path) -> None:
        """Verifies that a metadata archive predating the frame count field resolves to zero frames."""
        output_path = resolve_output_path(output_root=tmp_path)
        output_path.mkdir(parents=True, exist_ok=True)
        np.savez(
            output_path / COMBINED_METADATA_FILENAME,
            combined_height=np.array([256], dtype=np.uint16),
            combined_width=np.array([256], dtype=np.uint16),
        )

        geometry = _read_tracked_recording_geometry(cindra_root=output_path)

        assert geometry.combined_frame_count == 0

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

        assert _read_tracked_recording_geometry(cindra_root=output_path).two_channels is True


class TestJobSizing:
    """Tests the sizing entry points that pair each job's cores with its memory estimate."""

    def test_single_recording_sizing_pairs_the_stage_default_with_the_estimate(self, tmp_path: Path) -> None:
        """Verifies that a sized single-recording job carries its stage's measured cores and its memory estimate."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path, plane_number=2)
        configuration = SingleRecordingConfiguration()

        sizing = size_single_recording_job(
            job_name=SingleRecordingJobNames.BINARIZE,
            specifier="",
            output_root=tmp_path / "out",
            configuration=configuration,
            data_path=data_path,
        )

        assert isinstance(sizing, JobSizing)
        assert sizing.cores == BINARIZATION_WORKERS
        assert sizing.memory_mb == estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.BINARIZE,
            specifier="",
            output_root=tmp_path / "out",
            configuration=configuration,
            data_path=data_path,
        )

    def test_combination_sizing_carries_the_core_its_serial_merge_occupies(self, tmp_path: Path) -> None:
        """Verifies that a sized combination job carries its stage's measured cores and its memory estimate."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path)
        configuration = SingleRecordingConfiguration()

        sizing = size_single_recording_job(
            job_name=SingleRecordingJobNames.COMBINE,
            specifier="",
            output_root=tmp_path / "out",
            configuration=configuration,
            data_path=data_path,
        )

        assert sizing.cores == COMBINATION_WORKERS
        assert sizing.memory_mb == estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.COMBINE,
            specifier="",
            output_root=tmp_path / "out",
            configuration=configuration,
            data_path=data_path,
        )

    def test_a_sized_job_carries_the_planned_region_count_through(self, tmp_path: Path) -> None:
        """Verifies that the planned region count a sizing pass receives reaches the estimate it reports."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path)
        configuration = SingleRecordingConfiguration()

        sizing = size_single_recording_job(
            job_name=SingleRecordingJobNames.COMBINE,
            specifier="",
            output_root=tmp_path / "out",
            configuration=configuration,
            data_path=data_path,
            planned_roi_count=100,
        )

        assert sizing.memory_mb == estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.COMBINE,
            specifier="",
            output_root=tmp_path / "out",
            configuration=configuration,
            data_path=data_path,
            planned_roi_count=100,
        )

    def test_multi_recording_sizing_pairs_the_stage_default_with_the_estimate(self, tmp_path: Path) -> None:
        """Verifies that a sized multi-recording job carries its stage's measured cores and its memory estimate."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            _write_tracked_recording(output_root=root, height=64, width=64, frame_count=600, regions=400)
        configuration = MultiRecordingConfiguration()

        sizing = size_multi_recording_job(
            job_name=MultiRecordingJobNames.DISCOVER,
            specifier="",
            recording_directories=roots,
            configuration=configuration,
        )

        assert sizing.cores == DISCOVERY_WORKERS
        assert sizing.memory_mb == estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.DISCOVER,
            specifier="",
            recording_directories=roots,
            configuration=configuration,
        )


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
    def test_recording_without_readable_raw_data_is_rejected(self, tmp_path: Path, job_name: str) -> None:
        """Verifies that every single-recording stage rejects a recording that resolves no geometry."""
        message = (
            f"Unable to estimate the memory of the '{job_name}' job. The recording configured with the output root "
            f"{tmp_path} carries no readable raw imaging data, so no stage of it can run. Verify that the configured "
            f"data path holds the recording's source files."
        )

        with pytest.raises(FileNotFoundError, match=error_format(message=message)):
            estimate_single_recording_job_memory_mb(
                job_name=job_name,
                specifier="plane_0",
                output_root=tmp_path,
                configuration=SingleRecordingConfiguration(),
            )

    def test_a_written_output_does_not_rescue_an_unreadable_recording(self, tmp_path: Path) -> None:
        """Verifies that pipeline output on disk does not stand in for the raw data a sizing pass reads."""
        _write_recording(output_root=tmp_path, plane_count=2)
        _write_combined(output_root=tmp_path)
        message = (
            f"Unable to estimate the memory of the '{SingleRecordingJobNames.COMBINE}' job. The recording configured "
            f"with the output root {tmp_path} carries no readable raw imaging data, so no stage of it can run. Verify "
            f"that the configured data path holds the recording's source files."
        )

        with pytest.raises(FileNotFoundError, match=error_format(message=message)):
            estimate_single_recording_job_memory_mb(
                job_name=SingleRecordingJobNames.COMBINE,
                specifier="",
                output_root=tmp_path,
                configuration=SingleRecordingConfiguration(),
            )

    def test_specifier_naming_an_absent_plane_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that a per-plane job naming a plane the recording does not hold is rejected."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path, plane_number=2)
        output_root = tmp_path / "out"
        message = (
            f"Unable to estimate the memory of the '{SingleRecordingJobNames.PROCESS}' job. Its specifier names "
            f"imaging plane 'plane_7', which the recording configured with the output root {output_root} does not "
            f"hold. The recording holds 2 plane(s)."
        )

        with pytest.raises(ValueError, match=error_format(message=message)):
            estimate_single_recording_job_memory_mb(
                job_name=SingleRecordingJobNames.PROCESS,
                specifier="plane_7",
                output_root=output_root,
                configuration=SingleRecordingConfiguration(),
                data_path=data_path,
            )

    def test_discovery_over_a_dataset_without_combined_output_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that discovery rejects a dataset whose recordings hold no combined output."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        message = (
            f"Unable to estimate the memory of the '{MultiRecordingJobNames.DISCOVER}' job. 2 of the 2 recording(s) "
            f"the dataset spans carry no combined metadata archive: {roots[0]}, {roots[1]}. Run the single-recording "
            f"pipeline to completion over every recording of the dataset first."
        )

        with pytest.raises(FileNotFoundError, match=error_format(message=message)):
            estimate_multi_recording_job_memory_mb(
                job_name=MultiRecordingJobNames.DISCOVER,
                specifier="",
                recording_directories=roots,
                configuration=MultiRecordingConfiguration(),
            )

    def test_extraction_over_a_dataset_without_combined_output_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that extraction rejects a dataset whose recordings hold no combined output."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        message = (
            f"Unable to estimate the memory of the '{MultiRecordingJobNames.EXTRACT}' job. 2 of the 2 recording(s) "
            f"the dataset spans carry no combined metadata archive: {roots[0]}, {roots[1]}. Run the single-recording "
            f"pipeline to completion over every recording of the dataset first."
        )

        with pytest.raises(FileNotFoundError, match=error_format(message=message)):
            estimate_multi_recording_job_memory_mb(
                job_name=MultiRecordingJobNames.EXTRACT,
                specifier="day1",
                recording_directories=roots,
                configuration=MultiRecordingConfiguration(),
            )

    def test_dataset_spanning_no_recording_is_rejected(self) -> None:
        """Verifies that a dataset naming no recording at all is rejected."""
        message = (
            f"Unable to estimate the memory of the '{MultiRecordingJobNames.EXTRACT}' job. The dataset names no "
            f"recording directory, so the stage has nothing to size against."
        )

        with pytest.raises(FileNotFoundError, match=error_format(message=message)):
            estimate_multi_recording_job_memory_mb(
                job_name=MultiRecordingJobNames.EXTRACT,
                specifier="",
                recording_directories=[],
                configuration=MultiRecordingConfiguration(),
            )

    @pytest.mark.parametrize(
        ("job_name", "specifier"),
        [(MultiRecordingJobNames.DISCOVER, ""), (MultiRecordingJobNames.EXTRACT, "day1")],
    )
    def test_a_dataset_reporting_no_regions_is_rejected(self, tmp_path: Path, job_name: str, specifier: str) -> None:
        """Verifies that both multi-recording stages reject a dataset whose recordings wrote no trace array."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            _write_combined(output_root=root)
        message = (
            f"Unable to estimate the memory of the '{job_name}' job. 2 of the 2 recording(s) the dataset spans "
            f"report no regions in their combined trace array: "
            f"{resolve_output_path(output_root=roots[0])}, {resolve_output_path(output_root=roots[1])}. Run the "
            f"single-recording pipeline to completion over every recording of the dataset first."
        )

        with pytest.raises(FileNotFoundError, match=error_format(message=message)):
            estimate_multi_recording_job_memory_mb(
                job_name=job_name,
                specifier=specifier,
                recording_directories=roots,
                configuration=MultiRecordingConfiguration(),
            )

    def test_one_unreported_recording_rejects_the_whole_dataset(self, tmp_path: Path) -> None:
        """Verifies that a dataset is rejected whole when even one of its recordings reports no regions."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        _write_combined(output_root=roots[0])
        _write_tracked_recording(output_root=roots[1], height=64, width=64, frame_count=600, regions=400)
        message = (
            f"Unable to estimate the memory of the '{MultiRecordingJobNames.DISCOVER}' job. 1 of the 2 recording(s) "
            f"the dataset spans report no regions in their combined trace array: "
            f"{resolve_output_path(output_root=roots[0])}. Run the single-recording pipeline to completion over "
            f"every recording of the dataset first."
        )

        with pytest.raises(FileNotFoundError, match=error_format(message=message)):
            estimate_multi_recording_job_memory_mb(
                job_name=MultiRecordingJobNames.DISCOVER,
                specifier="",
                recording_directories=roots,
                configuration=MultiRecordingConfiguration(),
            )


class TestDualChannelRecordings:
    """Tests the stages that process both channels inside one job."""

    def test_second_channel_doubles_the_combination_estimate(self) -> None:
        """Verifies that a recording carrying a second channel doubles the traces combination concatenates."""
        planes = (PlaneGeometry(height=1, width=1, frame_count=6000, sampling_rate=30.0),)
        single = RecordingGeometry(planes=planes, resolved=True)
        dual = RecordingGeometry(planes=planes, two_channels=True, resolved=True)

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

    def test_second_channel_is_read_from_the_acquisition_parameters(self, tmp_path: Path) -> None:
        """Verifies that the geometry reports a second channel when the acquisition metadata declares one."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path, channel_number=2)

        geometry = resolve_recording_geometry(output_root=tmp_path / "out", data_path=data_path)

        assert geometry.two_channels is True
        # The interleave cycle carries one frame per plane and channel, so each plane receives half the source pages.
        assert geometry.planes[0].frame_count == 300

    def test_a_two_channel_acquisition_doubles_its_combination_estimate(self, tmp_path: Path) -> None:
        """Verifies that the second channel an acquisition declares reaches the recording's combination estimate."""
        single_path, dual_path = tmp_path / "single", tmp_path / "dual"
        # The dual-channel recording carries twice the source pages, so both recordings hold 600 frames per plane and
        # the channel factor is the only term left between their estimates.
        _write_raw_recording(data_path=single_path, pages=600)
        _write_raw_recording(data_path=dual_path, pages=1200, channel_number=2)

        single = resolve_recording_geometry(output_root=tmp_path / "out", data_path=single_path)
        dual = resolve_recording_geometry(output_root=tmp_path / "out", data_path=dual_path)

        assert single.planes[0].frame_count == dual.planes[0].frame_count
        assert _estimate_combination_mb(geometry=dual, regions=500) - WORKER_MEMORY_MB == 2 * (
            _estimate_combination_mb(geometry=single, regions=500) - WORKER_MEMORY_MB
        )


class TestPlannedRegionCount:
    """Tests the region term the single-recording estimates are sized for."""

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

    def test_planned_count_is_the_region_term_the_estimate_uses(self, tmp_path: Path) -> None:
        """Verifies that the planned region count is exactly what the region-scaled model is sized for."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path)
        configuration = SingleRecordingConfiguration()

        geometry = resolve_recording_geometry(output_root=tmp_path / "out", data_path=data_path)
        memory_mb = estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.COMBINE,
            specifier="",
            output_root=tmp_path / "out",
            configuration=configuration,
            data_path=data_path,
            planned_roi_count=137,
        )

        assert memory_mb == _apply_tolerance(memory_mb=_estimate_combination_mb(geometry=geometry, regions=137))

    def test_absent_planned_count_falls_back_to_the_detection_ceiling(self, tmp_path: Path) -> None:
        """Verifies that a recording named no planned count is sized for the provable detection ceiling."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path, plane_number=2)
        configuration = SingleRecordingConfiguration()

        geometry = resolve_recording_geometry(output_root=tmp_path / "out", data_path=data_path)
        ceiling = resolve_maximum_roi_count(plane_count=2, configuration=configuration)
        memory_mb = estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.COMBINE,
            specifier="",
            output_root=tmp_path / "out",
            configuration=configuration,
            data_path=data_path,
        )

        assert memory_mb == _apply_tolerance(memory_mb=_estimate_combination_mb(geometry=geometry, regions=ceiling))

    def test_a_per_plane_job_is_capped_at_the_per_plane_ceiling(self, tmp_path: Path) -> None:
        """Verifies that a planned count above what one plane can detect does not inflate a per-plane estimate."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path)
        configuration = SingleRecordingConfiguration()
        ceiling = resolve_maximum_roi_count(plane_count=1, configuration=configuration)

        capped = estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.PROCESS,
            specifier="plane_0",
            output_root=tmp_path / "out",
            configuration=configuration,
            data_path=data_path,
            planned_roi_count=ceiling,
        )
        beyond = estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.PROCESS,
            specifier="plane_0",
            output_root=tmp_path / "out",
            configuration=configuration,
            data_path=data_path,
            planned_roi_count=10 * ceiling,
        )

        assert beyond == capped

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


class TestDatasetDirectoryResolution:
    """Tests the latitude the multi-recording estimator allows in the directories a dataset names."""

    def test_a_configured_output_directory_is_used_directly(self, tmp_path: Path) -> None:
        """Verifies that a directory already holding the combined archive is read without a tree search."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            _write_tracked_recording(output_root=root, height=64, width=64, frame_count=600, regions=400)

        memory_mb = estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.DISCOVER,
            specifier="",
            recording_directories=[resolve_output_path(output_root=root) for root in roots],
            configuration=MultiRecordingConfiguration(),
        )

        assert memory_mb > 0
