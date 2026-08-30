"""Contains tests for the per-stage memory models of the two pipelines."""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
from tifffile import TiffWriter
from ataraxis_base_utilities import error_format

from cindra.layout import (
    PARAMETERS_FILENAME,
    COMBINED_METADATA_FILENAME,
    ACQUISITION_PARAMETERS_FILENAME,
    RecordingArrays,
    resolve_array_path,
    resolve_plane_path,
    resolve_output_path,
)
from cindra.dataclasses import (
    SingleRecordingRuntimeData,
    MultiRecordingConfiguration,
    SingleRecordingConfiguration,
)
from cindra.orchestration import (
    DISCOVERY_WORKERS,
    EXTRACTION_WORKERS,
    COMBINATION_WORKERS,
    BINARIZATION_WORKERS,
    REGISTRATION_WORKERS,
    REGISTRATION_GPU_WORKERS,
    MultiRecordingJobNames,
    SingleRecordingJobNames,
)
from cindra.orchestration.footprints import (
    WORKER_MEMORY_MB,
    _BYTES_PER_MEGABYTE,
    _DEVICE_CONTEXT_BYTES,
    _DEVICE_PIPELINE_SLOTS,
    _SINGLE_PRECISION_BYTES,
    _DEVICE_STAGING_DIRECTIONS,
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
    _resolve_tracked_regions,
    size_multi_recording_job,
    _estimate_registration_mb,
    resolve_maximum_roi_count,
    size_single_recording_job,
    _resolve_device_batch_size,
    resolve_recording_geometry,
    _resolve_binned_frame_count,
    _resolve_metric_sample_count,
    read_tracked_recording_geometry,
    _estimate_registration_device_mb,
    _resolve_nonrigid_block_geometry,
    estimate_multi_recording_job_memory_mb,
    _estimate_registration_device_memory_mb,
    estimate_single_recording_job_memory_mb,
)

if TYPE_CHECKING:
    from pathlib import Path


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
        """Verifies that a model holding no bytes is charged no megabytes."""
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

        assert _estimate_registration_mb(
            plane=tall, configuration=configuration, gpu_registration=False
        ) < _estimate_registration_mb(plane=small, configuration=configuration, gpu_registration=False)

    def test_disabled_metrics_leave_the_reference_stage_as_the_peak(self) -> None:
        """Verifies that a plane whose quality metrics are disabled is charged the reference stage alone."""
        plane = PlaneGeometry(height=512, width=512, frame_count=20000, sampling_rate=30.0)
        configuration = SingleRecordingConfiguration()
        configuration.registration.registration_metric_principal_components = 0

        with_metrics = _estimate_registration_mb(
            plane=plane, configuration=SingleRecordingConfiguration(), gpu_registration=False
        )

        assert (
            _estimate_registration_mb(plane=plane, configuration=configuration, gpu_registration=False) < with_metrics
        )


class TestDeviceBatchSize:
    """Tests the frame batch a plane's registration stages at once while it runs on a CUDA device."""

    def test_device_batch_reads_the_wider_of_the_two_configured_sizes(self) -> None:
        """Verifies that a configured device batch above the shared one is what the device stages."""
        plane = PlaneGeometry(height=512, width=512, frame_count=20000, sampling_rate=30.0)
        configuration = SingleRecordingConfiguration()
        configuration.registration.batch_size = 100
        configuration.registration.gpu_batch_size = 250

        assert _resolve_device_batch_size(plane=plane, configuration=configuration) == 250

    def test_shared_batch_applies_while_no_device_batch_is_configured(self) -> None:
        """Verifies that a device batch of zero leaves the shared batch as the size the device stages."""
        plane = PlaneGeometry(height=512, width=512, frame_count=20000, sampling_rate=30.0)
        configuration = SingleRecordingConfiguration()
        configuration.registration.batch_size = 100
        configuration.registration.gpu_batch_size = 0

        assert _resolve_device_batch_size(plane=plane, configuration=configuration) == 100

    def test_batch_never_exceeds_the_frames_the_plane_holds(self) -> None:
        """Verifies that a plane shorter than the configured batch is charged the frames it actually holds."""
        plane = PlaneGeometry(height=512, width=512, frame_count=12, sampling_rate=30.0)

        assert _resolve_device_batch_size(plane=plane, configuration=SingleRecordingConfiguration()) == 12


class TestNonrigidBlockGeometry:
    """Tests the block tiling over which the device memory model resolves its nonrigid terms."""

    def test_disabled_nonrigid_registration_tiles_no_block(self) -> None:
        """Verifies that a configuration running rigid registration alone reports an empty block geometry."""
        plane = PlaneGeometry(height=512, width=512, frame_count=20000, sampling_rate=30.0)
        configuration = SingleRecordingConfiguration()
        configuration.nonrigid_registration.enabled = False

        blocks = _resolve_nonrigid_block_geometry(plane=plane, configuration=configuration)

        assert (blocks.count, blocks.height, blocks.width, blocks.window_size) == (0, 0, 0, 0)

    def test_block_smaller_than_the_plane_tiles_both_axes_with_overlap(self) -> None:
        """Verifies that a block below both plane extents tiles each axis at roughly half a block of overlap."""
        plane = PlaneGeometry(height=512, width=512, frame_count=20000, sampling_rate=30.0)

        blocks = _resolve_nonrigid_block_geometry(plane=plane, configuration=SingleRecordingConfiguration())

        assert (blocks.count, blocks.height, blocks.width, blocks.window_size) == (36, 128, 128, 17)

    def test_block_covering_the_plane_tiles_each_axis_once(self) -> None:
        """Verifies that a block at least as large as the plane spans it as one block on both axes."""
        plane = PlaneGeometry(height=96, width=96, frame_count=20000, sampling_rate=30.0)

        blocks = _resolve_nonrigid_block_geometry(plane=plane, configuration=SingleRecordingConfiguration())

        assert (blocks.count, blocks.height, blocks.width) == (1, 96, 96)

    def test_block_covering_one_axis_alone_tiles_the_other(self) -> None:
        """Verifies that each axis resolves its own tiling from the block extent with which that axis was configured."""
        plane = PlaneGeometry(height=96, width=512, frame_count=20000, sampling_rate=30.0)

        blocks = _resolve_nonrigid_block_geometry(plane=plane, configuration=SingleRecordingConfiguration())

        assert (blocks.count, blocks.height, blocks.width) == (6, 96, 128)

    def test_small_block_narrows_the_correlation_window_below_the_configured_offset(self) -> None:
        """Verifies that a block too small to hold the configured offset bounds the window on its own extent."""
        plane = PlaneGeometry(height=256, width=256, frame_count=20000, sampling_rate=30.0)
        configuration = SingleRecordingConfiguration()
        configuration.nonrigid_registration.block_size = (14, 14)

        blocks = _resolve_nonrigid_block_geometry(plane=plane, configuration=configuration)

        assert (blocks.count, blocks.height, blocks.width, blocks.window_size) == (784, 14, 14, 15)


class TestDeviceRegistrationModel:
    """Tests the device memory model and the host staging term a device-planned registration job adds."""

    def test_host_peak_gains_the_staging_buffers(self) -> None:
        """Verifies that a job planned for a device adds its page-locked staging buffers to the host peak."""
        # A 512 by 512 single-precision frame is exactly one megabyte, so the staging term converts without rounding.
        plane = PlaneGeometry(height=512, width=512, frame_count=20000, sampling_rate=30.0)
        configuration = SingleRecordingConfiguration()

        host_only = _estimate_registration_mb(plane=plane, configuration=configuration, gpu_registration=False)
        device_backed = _estimate_registration_mb(plane=plane, configuration=configuration, gpu_registration=True)

        staging_bytes = (
            _DEVICE_PIPELINE_SLOTS
            * _DEVICE_STAGING_DIRECTIONS
            * _resolve_device_batch_size(plane=plane, configuration=configuration)
            * plane.height
            * plane.width
            * _SINGLE_PRECISION_BYTES
        )
        assert device_backed - host_only == staging_bytes // _BYTES_PER_MEGABYTE

    def test_nonrigid_plane_holds_its_block_and_window_terms(self) -> None:
        """Verifies the device figure a tiled 512 pixel plane holds while nonrigid registration runs."""
        plane = PlaneGeometry(height=512, width=512, frame_count=20000, sampling_rate=30.0)

        assert _estimate_registration_device_mb(plane=plane, configuration=SingleRecordingConfiguration()) == 4453

    def test_two_step_refinement_charges_the_plane_two_live_backends(self) -> None:
        """Verifies that enabling the refinement pass charges the plane both backends and one context."""
        plane = PlaneGeometry(height=512, width=512, frame_count=20000, sampling_rate=30.0)
        configuration = SingleRecordingConfiguration()
        configuration.registration.two_step_registration = True

        assert _estimate_registration_device_mb(plane=plane, configuration=configuration) == 8393

    def test_rigid_plane_holds_the_frame_terms_alone(self) -> None:
        """Verifies that disabling nonrigid registration drops every block-shaped term from the device figure."""
        plane = PlaneGeometry(height=512, width=512, frame_count=20000, sampling_rate=30.0)
        configuration = SingleRecordingConfiguration()
        configuration.nonrigid_registration.enabled = False

        assert _estimate_registration_device_mb(plane=plane, configuration=configuration) == 1922

    def test_device_figure_scales_with_the_batch_the_plane_stages(self) -> None:
        """Verifies that the device batch is the one setting that fits a registration job to a card."""
        plane = PlaneGeometry(height=512, width=512, frame_count=20000, sampling_rate=30.0)
        narrow = SingleRecordingConfiguration()
        narrow.registration.gpu_batch_size = 20
        wide = SingleRecordingConfiguration()
        wide.registration.gpu_batch_size = 200

        assert _estimate_registration_device_mb(plane=plane, configuration=narrow) < _estimate_registration_device_mb(
            plane=plane, configuration=wide
        )

    def test_every_plane_holds_the_context_floor(self) -> None:
        """Verifies that even a plane holding one frame is charged the context the driver keeps resident."""
        plane = PlaneGeometry(height=8, width=8, frame_count=1, sampling_rate=30.0)

        estimate = _estimate_registration_device_mb(plane=plane, configuration=SingleRecordingConfiguration())

        assert estimate >= _DEVICE_CONTEXT_BYTES // _BYTES_PER_MEGABYTE


class TestDeviceRegistrationJobEstimate:
    """Tests the per-job device estimate that reads the recording's own plane geometry."""

    def test_named_plane_is_charged_its_own_geometry(self, tmp_path: Path) -> None:
        """Verifies that a job naming one plane is charged the device figure that plane's extent produces."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path, height=30, roi_lines="[[0, 9], [12, 29]]", roi_number=2)
        configuration = SingleRecordingConfiguration()
        geometry = resolve_recording_geometry(output_root=tmp_path / "out", data_path=data_path)

        memory_mb = _estimate_registration_device_memory_mb(
            specifier="plane_1", output_root=tmp_path / "out", configuration=configuration, data_path=data_path
        )

        assert memory_mb == _apply_tolerance(
            memory_mb=_estimate_registration_device_mb(plane=geometry.planes[1], configuration=configuration)
        )

    def test_unnamed_job_is_charged_the_widest_plane(self, tmp_path: Path) -> None:
        """Verifies that a job naming no plane is charged the largest per-plane device figure."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path, height=30, roi_lines="[[0, 9], [12, 29]]", roi_number=2)
        configuration = SingleRecordingConfiguration()
        geometry = resolve_recording_geometry(output_root=tmp_path / "out", data_path=data_path)

        memory_mb = _estimate_registration_device_memory_mb(
            specifier="", output_root=tmp_path / "out", configuration=configuration, data_path=data_path
        )

        assert memory_mb == _apply_tolerance(
            memory_mb=max(
                _estimate_registration_device_mb(plane=plane, configuration=configuration) for plane in geometry.planes
            )
        )


class TestProcessingModel:
    """Tests the binned movie model that drives the processing footprint."""

    def test_bin_size_takes_the_coarsest_of_its_three_terms(self) -> None:
        """Verifies that the binned frame count follows the coarsest of the sample, ratio, and decay terms."""
        configuration = SingleRecordingConfiguration()
        plane = PlaneGeometry(height=512, width=512, frame_count=60000, sampling_rate=30.0)

        # The decay term is round(0.4 * 30) = 12, and the ratio term is 60000 // 5000 = 12. Each 500 frame read batch
        # yields 41 bins from 492 of its own frames and discards the remaining 8, so 120 batches yield 4920 bins
        # rather than 5000.
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

        assert not geometry.resolved
        assert geometry.planes == ()
        assert geometry.raw_frame_pixels == 0

    def test_plane_geometry_follows_the_acquisition_and_the_source_header(self, tmp_path: Path) -> None:
        """Verifies that a recording carrying only raw data resolves the shape its conversion will write."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path)

        geometry = resolve_recording_geometry(output_root=tmp_path / "out", data_path=data_path)

        assert geometry.resolved
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

        assert not geometry.resolved
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
                _estimate_registration_mb(plane=plane, configuration=configuration, gpu_registration=False)
                for plane in geometry.planes
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

        # Two recordings at the default 50% prevalence demand one recording per template, so the pooled ceiling is
        # the 1100 regions the dataset holds and the headroom bound, ceil(700 * 1.5), is the smaller of the two.
        geometry = RecordingGeometry(combined_pixels=4096, combined_frame_count=600, region_count=400, resolved=True)
        assert memory_mb == _apply_tolerance(
            memory_mb=_estimate_extraction_mb(geometry=geometry, tracked_regions=1050, configuration=configuration)
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

        # The pooled ceiling is the 900 regions the dataset holds and the headroom bound is ceil(500 * 1.5).
        widest = RecordingGeometry(combined_pixels=65536, combined_frame_count=600, region_count=500, resolved=True)
        assert memory_mb == _apply_tolerance(
            memory_mb=_estimate_extraction_mb(geometry=widest, tracked_regions=750, configuration=configuration)
        )


class TestTrackedRecordingGeometry:
    """Tests the reader through which every multi-recording model resolves one recording's shape."""

    def test_region_count_is_read_from_the_trace_header(self, tmp_path: Path) -> None:
        """Verifies that the region count comes from the combined trace array's own header."""
        _write_tracked_recording(output_root=tmp_path, height=64, width=64, frame_count=600, regions=321)

        geometry = read_tracked_recording_geometry(cindra_root=resolve_output_path(output_root=tmp_path))

        assert geometry.region_count == 321
        assert geometry.resolved

    def test_a_recording_without_combined_output_resolves_nothing(self, tmp_path: Path) -> None:
        """Verifies that a recording carrying no metadata archive contributes no geometry."""
        geometry = read_tracked_recording_geometry(cindra_root=resolve_output_path(output_root=tmp_path))

        assert not geometry.resolved
        assert geometry.combined_pixels == 0

    def test_an_absent_trace_array_reports_no_regions(self, tmp_path: Path) -> None:
        """Verifies that a recording whose combination stage wrote no trace array reports no regions."""
        _write_combined(output_root=tmp_path)

        geometry = read_tracked_recording_geometry(cindra_root=resolve_output_path(output_root=tmp_path))

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

        geometry = read_tracked_recording_geometry(cindra_root=resolve_output_path(output_root=tmp_path))

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

        geometry = read_tracked_recording_geometry(cindra_root=output_path)

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

        assert read_tracked_recording_geometry(cindra_root=output_path).two_channels


class TestUnresolvedInputReporting:
    """Tests the unresolved-input description a recording geometry reports for each of its input states."""

    def test_a_resolved_geometry_describes_nothing(self) -> None:
        """Verifies that a recording holding a plane reports no unresolved input."""
        geometry = RecordingGeometry(
            planes=(PlaneGeometry(height=512, width=512, frame_count=1000, sampling_rate=30.0),),
            resolved=True,
            acquisition_resolved=True,
            source_resolved=True,
        )

        assert geometry._describe_unresolved_inputs(data_path=None) == ""

    def test_unreadable_acquisition_parameters_are_named_alone(self, tmp_path: Path) -> None:
        """Verifies that a recording whose source files resolved blames its acquisition parameters alone."""
        geometry = RecordingGeometry(acquisition_resolved=False, source_resolved=True)

        description = geometry._describe_unresolved_inputs(data_path=tmp_path)

        assert "Its acquisition parameters were not readable" in description
        assert ACQUISITION_PARAMETERS_FILENAME in description
        assert "holds no readable source files" not in description

    def test_two_readable_inputs_describing_no_whole_cycle_are_named(self, tmp_path: Path) -> None:
        """Verifies that a recording whose inputs both resolved blames the interleave cycle they describe."""
        geometry = RecordingGeometry(acquisition_resolved=True, source_resolved=True)

        description = geometry._describe_unresolved_inputs(data_path=tmp_path)

        assert "fewer frames than one whole plane and channel interleave cycle" in description


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
        assert sizing.device_memory_mb == 0


class TestDevicePlannedJobSizing:
    """Tests the cores and the device memory a single-recording job reports while it is planned for a device."""

    def test_registration_reports_its_device_cores_and_device_memory(self, tmp_path: Path) -> None:
        """Verifies that a registration planned for a device carries its host-side cores and its device figure."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path)
        configuration = SingleRecordingConfiguration()

        sizing = size_single_recording_job(
            job_name=SingleRecordingJobNames.REGISTER,
            specifier="plane_0",
            output_root=tmp_path / "out",
            configuration=configuration,
            data_path=data_path,
            gpu_registration=True,
        )

        assert sizing.cores == REGISTRATION_GPU_WORKERS
        assert sizing.device_memory_mb == _estimate_registration_device_memory_mb(
            specifier="plane_0", output_root=tmp_path / "out", configuration=configuration, data_path=data_path
        )
        assert sizing.memory_mb == estimate_single_recording_job_memory_mb(
            job_name=SingleRecordingJobNames.REGISTER,
            specifier="plane_0",
            output_root=tmp_path / "out",
            configuration=configuration,
            data_path=data_path,
            gpu_registration=True,
        )

    def test_host_planned_registration_reports_no_device_memory(self, tmp_path: Path) -> None:
        """Verifies that a registration planned for the host CPU carries its host cores and no device figure."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path)

        sizing = size_single_recording_job(
            job_name=SingleRecordingJobNames.REGISTER,
            specifier="plane_0",
            output_root=tmp_path / "out",
            configuration=SingleRecordingConfiguration(),
            data_path=data_path,
        )

        assert sizing.cores == REGISTRATION_WORKERS
        assert sizing.device_memory_mb == 0

    @pytest.mark.parametrize(
        "job_name",
        [SingleRecordingJobNames.BINARIZE, SingleRecordingJobNames.PROCESS, SingleRecordingJobNames.COMBINE],
    )
    def test_every_other_stage_reports_no_device_memory(
        self, tmp_path: Path, job_name: SingleRecordingJobNames
    ) -> None:
        """Verifies that the registration stage alone reports a device figure while the flag is set."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path)

        sizing = size_single_recording_job(
            job_name=job_name,
            specifier="plane_0",
            output_root=tmp_path / "out",
            configuration=SingleRecordingConfiguration(),
            data_path=data_path,
            gpu_registration=True,
        )

        assert sizing.device_memory_mb == 0

    def test_device_plan_raises_the_host_estimate_of_a_registration_job(self, tmp_path: Path) -> None:
        """Verifies that the staging buffers reach the reported host figure of a device-planned registration."""
        data_path = tmp_path / "raw"
        _write_raw_recording(data_path=data_path, height=512, width=512, pages=60)
        configuration = SingleRecordingConfiguration()

        arguments = {
            "job_name": SingleRecordingJobNames.REGISTER,
            "specifier": "plane_0",
            "output_root": tmp_path / "out",
            "configuration": configuration,
            "data_path": data_path,
        }

        assert estimate_single_recording_job_memory_mb(
            **arguments, gpu_registration=True
        ) > estimate_single_recording_job_memory_mb(**arguments)


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
            f"{tmp_path} declares no imaging plane, so no stage of it can run. Neither of its two inputs resolved. The "
            f"recording's configuration names no raw imaging data path, and its acquisition parameters were not "
            f"readable either. Point the configured data path at the recording's imaging directory, or at any parent "
            f"of it that carries the acquisition parameters file beneath it. Verify that "
            f"{ACQUISITION_PARAMETERS_FILENAME} sits in the recording's output directory or that {PARAMETERS_FILENAME} "
            f"sits under its raw imaging directory."
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
            f"with the output root {tmp_path} declares no imaging plane, so no stage of it can run. The recording's "
            f"configuration names no raw imaging data path, so the frames its conversion reads cannot be counted. "
            f"Point the configured data path at the recording's imaging directory, or at any parent of it that "
            f"carries the acquisition parameters file beneath it."
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

        assert geometry.two_channels
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
    """Tests the region term for which the single-recording estimates are sized."""

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
        """Verifies that the region-scaled model is sized by exactly the planned region count."""
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


class TestTrackedRegionBound:
    """Tests the bound the tracked extraction estimate substitutes for a template count no plan can read."""

    def test_headroom_bound_holds_when_the_dataset_spans_many_recordings(self) -> None:
        """Verifies that the domain headroom term is the bound whenever the pooled ceiling sits above it."""
        configuration = MultiRecordingConfiguration()

        tracked = _resolve_tracked_regions(
            geometries=self._dataset((1000, 1000, 1000, 1000)), configuration=configuration, planned_roi_count=None
        )

        # Four recordings at 50% prevalence demand two recordings per template, so the pooled ceiling is 4000 // 2.
        assert tracked == 1500

    def test_pooled_ceiling_holds_when_the_dataset_spans_few_recordings(self) -> None:
        """Verifies that the combinatorial ceiling is the bound whenever it sits below the headroom term."""
        configuration = MultiRecordingConfiguration()
        configuration.roi_tracking.mask_prevalence = 100

        tracked = _resolve_tracked_regions(
            geometries=self._dataset((1000, 1000)), configuration=configuration, planned_roi_count=None
        )

        # Every template consumes one region of each of the two recordings exclusively, so the dataset admits 1000
        # of them where the headroom term alone would allow 1500.
        assert tracked == 1000

    def test_minimum_recordings_mirrors_the_count_tracking_derives(self) -> None:
        """Verifies that the pooled ceiling divides by the rounded-up recording count tracking derives."""
        configuration = MultiRecordingConfiguration()
        configuration.roi_tracking.mask_prevalence = 50

        tracked = _resolve_tracked_regions(
            geometries=self._dataset((300, 300, 300)), configuration=configuration, planned_roi_count=None
        )

        # Tracking takes the ceiling of 50% of three recordings, which is two, so the pooled ceiling is 900 // 2.
        assert tracked == 450

    def test_prevalence_of_zero_still_divides_the_pooled_count(self) -> None:
        """Verifies that a prevalence demanding no recording charges one rather than dividing by nothing."""
        configuration = MultiRecordingConfiguration()
        configuration.roi_tracking.mask_prevalence = 0

        tracked = _resolve_tracked_regions(
            geometries=self._dataset((100, 100)), configuration=configuration, planned_roi_count=None
        )

        assert tracked == 150

    def test_bound_stays_below_the_pooled_count_of_a_broad_dataset(self) -> None:
        """Verifies that a dataset of many populated recordings is not sized for its pooled region count."""
        configuration = MultiRecordingConfiguration()
        geometries = self._dataset((15000,) * 20)

        tracked = _resolve_tracked_regions(geometries=geometries, configuration=configuration, planned_roi_count=None)

        # The pooled ceiling of 300000 // 10 is what a pooled-only bound would charge, and the headroom term keeps
        # the reservation well under it.
        assert tracked == 22500
        assert tracked < sum(geometry.region_count for geometry in geometries) // 10

    def test_bound_reaches_the_extraction_estimate(self, tmp_path: Path) -> None:
        """Verifies that the extraction model is sized by exactly the template count the bound reports."""
        configuration = MultiRecordingConfiguration()
        roots = [tmp_path / "day1", tmp_path / "day2", tmp_path / "day3"]
        for root, regions in zip(roots, (300, 500, 400), strict=True):
            _write_tracked_recording(output_root=root, height=64, width=64, frame_count=600, regions=regions)

        memory_mb = estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.EXTRACT,
            specifier="day1",
            recording_directories=roots,
            configuration=configuration,
        )

        geometry = RecordingGeometry(combined_pixels=4096, combined_frame_count=600, region_count=300, resolved=True)
        # Three recordings at 50% prevalence demand two per template, so the pooled ceiling of 1200 // 2 is the
        # smaller of the two bounds and ceil(500 * 1.5) is not reached.
        assert memory_mb == _apply_tolerance(
            memory_mb=_estimate_extraction_mb(geometry=geometry, tracked_regions=600, configuration=configuration)
        )

    def test_bound_never_exceeds_either_of_its_two_terms(self) -> None:
        """Verifies that the bound is the smaller of the two terms across a spread of datasets and prevalences."""
        for counts in ((10, 10), (1000, 200, 30), (700,), (15000,) * 8):
            for prevalence in (0, 25, 50, 75, 100):
                configuration = MultiRecordingConfiguration()
                configuration.roi_tracking.mask_prevalence = prevalence
                geometries = self._dataset(counts)

                tracked = _resolve_tracked_regions(
                    geometries=geometries, configuration=configuration, planned_roi_count=None
                )

                minimum_recordings = max(1, math.ceil(prevalence / 100 * len(counts)))
                assert tracked <= sum(counts) // minimum_recordings
                assert tracked <= math.ceil(max(counts) * 1.5)
                assert tracked >= 1

    @staticmethod
    def _dataset(region_counts: tuple[int, ...]) -> tuple[RecordingGeometry, ...]:
        """Builds the geometry of a dataset whose recordings hold the given region counts."""
        return tuple(
            RecordingGeometry(combined_pixels=4096, combined_frame_count=600, region_count=count, resolved=True)
            for count in region_counts
        )


class TestPlannedTrackedRegionCount:
    """Tests the override a caller that knows its template count states instead of taking the bound."""

    def test_planned_count_is_the_tracked_term_the_estimate_uses(self, tmp_path: Path) -> None:
        """Verifies that the extraction model is sized by exactly the planned count."""
        configuration = MultiRecordingConfiguration()
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            _write_tracked_recording(output_root=root, height=64, width=64, frame_count=600, regions=400)

        memory_mb = estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.EXTRACT,
            specifier="day1",
            recording_directories=roots,
            configuration=configuration,
            planned_roi_count=137,
        )

        geometry = RecordingGeometry(combined_pixels=4096, combined_frame_count=600, region_count=400, resolved=True)
        assert memory_mb == _apply_tolerance(
            memory_mb=_estimate_extraction_mb(geometry=geometry, tracked_regions=137, configuration=configuration)
        )

    def test_planned_count_overrides_the_bound_in_both_directions(self, tmp_path: Path) -> None:
        """Verifies that a caller's figure is taken whether it sits above or below the bound the dataset implies."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            _write_tracked_recording(output_root=root, height=64, width=64, frame_count=60000, regions=400)

        def estimate(planned_roi_count: int | None) -> int:
            return estimate_multi_recording_job_memory_mb(
                job_name=MultiRecordingJobNames.EXTRACT,
                specifier="day1",
                recording_directories=roots,
                configuration=MultiRecordingConfiguration(),
                planned_roi_count=planned_roi_count,
            )

        assert estimate(planned_roi_count=10) < estimate(planned_roi_count=None) < estimate(planned_roi_count=100000)

    def test_absent_planned_count_falls_back_to_the_bound(self, tmp_path: Path) -> None:
        """Verifies that a dataset named no planned count is sized for the bound its region counts provide."""
        configuration = MultiRecordingConfiguration()
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            _write_tracked_recording(output_root=root, height=64, width=64, frame_count=600, regions=400)

        geometries = [
            RecordingGeometry(combined_pixels=4096, combined_frame_count=600, region_count=400, resolved=True)
        ] * 2
        bound = _resolve_tracked_regions(geometries=geometries, configuration=configuration, planned_roi_count=None)
        memory_mb = estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.EXTRACT,
            specifier="day1",
            recording_directories=roots,
            configuration=configuration,
        )

        assert memory_mb == estimate_multi_recording_job_memory_mb(
            job_name=MultiRecordingJobNames.EXTRACT,
            specifier="day1",
            recording_directories=roots,
            configuration=configuration,
            planned_roi_count=bound,
        )

    def test_discovery_does_not_read_the_planned_count(self, tmp_path: Path) -> None:
        """Verifies that discovery is sized for the regions each recording reports, whatever count the caller plans."""
        configuration = MultiRecordingConfiguration()
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            _write_tracked_recording(output_root=root, height=64, width=64, frame_count=600, regions=400)

        def estimate(planned_roi_count: int | None) -> int:
            return estimate_multi_recording_job_memory_mb(
                job_name=MultiRecordingJobNames.DISCOVER,
                specifier="",
                recording_directories=roots,
                configuration=configuration,
                planned_roi_count=planned_roi_count,
            )

        assert estimate(planned_roi_count=None) == estimate(planned_roi_count=1) == estimate(planned_roi_count=500000)

    def test_sizing_forwards_the_planned_count_to_the_estimate(self, tmp_path: Path) -> None:
        """Verifies that the sizing entry point reports the memory the planned count implies."""
        configuration = MultiRecordingConfiguration()
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            _write_tracked_recording(output_root=root, height=64, width=64, frame_count=60000, regions=400)

        sizing = size_multi_recording_job(
            job_name=MultiRecordingJobNames.EXTRACT,
            specifier="day1",
            recording_directories=roots,
            configuration=configuration,
            planned_roi_count=97,
        )

        assert sizing == JobSizing(
            cores=EXTRACTION_WORKERS,
            memory_mb=estimate_multi_recording_job_memory_mb(
                job_name=MultiRecordingJobNames.EXTRACT,
                specifier="day1",
                recording_directories=roots,
                configuration=configuration,
                planned_roi_count=97,
            ),
            device_memory_mb=0,
        )

    def test_non_positive_planned_count_is_rejected(self, tmp_path: Path) -> None:
        """Verifies that a planned template count of zero or less is rejected before the dataset is read."""
        message = (
            f"Unable to estimate the memory of the '{MultiRecordingJobNames.EXTRACT}' job. The planned region count "
            f"must be a positive integer counting the templates the dataset tracks, or None to accept the bound the "
            f"per-recording region counts provide, but encountered -1."
        )

        with pytest.raises(ValueError, match=error_format(message=message)):
            estimate_multi_recording_job_memory_mb(
                job_name=MultiRecordingJobNames.EXTRACT,
                specifier="day1",
                recording_directories=[tmp_path / "day1"],
                configuration=MultiRecordingConfiguration(),
                planned_roi_count=-1,
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
