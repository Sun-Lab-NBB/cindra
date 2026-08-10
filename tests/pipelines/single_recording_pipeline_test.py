"""Contains integration tests for the single-recording pipeline orchestration entry points."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
import pytest
from tifffile import TiffWriter
from ataraxis_base_utilities import ensure_directory_exists
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from cindra.io import (
    create_registration_marker,
    resolve_registration_marker_path,
    resolve_single_recording_contexts,
)
from cindra.io.context import PARAMETERS_FILENAME
from cindra.dataclasses import RuntimeContext, SingleRecordingConfiguration
from cindra.orchestration import SingleRecordingJobNames
from cindra.orchestration.pipeline import (
    execute_single_recording_job,
    _execute_single_recording_job,
    run_single_recording_pipeline,
)
from cindra.pipelines.single_recording import (
    process_plane,
    binarize_recording,
    save_combined_data,
    register_recording_plane,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from numpy.typing import NDArray

_FRAME_HEIGHT: int = 128
"""The synthetic frame height in pixels, large enough for phase-correlation registration to converge."""

_FRAME_WIDTH: int = 128
"""The synthetic frame width in pixels, large enough for phase-correlation registration to converge."""

_FRAME_COUNT: int = 60
"""The default synthetic frame count, above the processing minimum but below the recommended threshold."""

_BLOB_CENTERS: tuple[tuple[int, int], ...] = ((32, 32), (76, 44), (50, 90), (96, 96))
"""The planted blob centroids, spaced far apart so detection resolves each into a distinct ROI."""

_BLOB_SIGMA: float = 3.0
"""The Gaussian blob radius in pixels, matching a compact soma-sized fluorescence source."""

_BLOB_AMPLITUDE: float = 1500.0
"""The peak blob intensity scale added on top of the flat background during active frames."""

_BACKGROUND_LEVEL: float = 100.0
"""The flat baseline intensity of the synthetic field outside of the blobs."""

_MAXIMUM_PIXEL_VALUE: int = 32766
"""The clipping ceiling for the synthetic movie, keeping intensities within the signed 16-bit range."""

_TEST_WORKERS: int = 1
"""The worker allocation every stage entry point receives in these tests, which keeps the synthetic runs serial."""


def _build_flickering_movie(*, frame_count: int, seed: int) -> NDArray[np.int16]:
    """Builds a synthetic movie whose spatially fixed Gaussian blobs flicker independently across frames.

    Notes:
        Detection keys on temporal variance, so a movie of identical frames yields no detectable ROIs. Each blob is
        therefore scaled by an independent positive random amplitude per frame to plant localized temporal signal.
    """
    generator = np.random.default_rng(seed=seed)
    rows, columns = np.mgrid[0:_FRAME_HEIGHT, 0:_FRAME_WIDTH]
    movie = np.full((frame_count, _FRAME_HEIGHT, _FRAME_WIDTH), fill_value=_BACKGROUND_LEVEL, dtype=np.float64)
    for center_row, center_column in _BLOB_CENTERS:
        blob = np.exp(-(((rows - center_row) ** 2 + (columns - center_column) ** 2) / (2.0 * _BLOB_SIGMA**2)))
        amplitudes = _BLOB_AMPLITUDE * (0.5 + np.abs(generator.standard_normal(frame_count)))
        movie += amplitudes[:, np.newaxis, np.newaxis] * blob[np.newaxis, :, :]
    return np.clip(movie, 0, _MAXIMUM_PIXEL_VALUE).astype(np.int16)


def _write_raw_recording(data_directory: Path, *, frame_count: int = _FRAME_COUNT, seed: int = 0) -> None:
    """Writes a multi-page TIFF and a raw acquisition parameters file describing one single-channel imaging plane."""
    ensure_directory_exists(data_directory)
    movie = _build_flickering_movie(frame_count=frame_count, seed=seed)
    with TiffWriter(data_directory / "recording.tif") as writer:
        for frame_index in range(frame_count):
            writer.write(movie[frame_index])
    parameters = {"frame_rate": 30.0, "plane_number": 1, "channel_number": 1}
    (data_directory / PARAMETERS_FILENAME).write_text(json.dumps(parameters))


def _make_configuration(*, data_directory: Path | None, output_directory: Path | None) -> SingleRecordingConfiguration:
    """Builds a tuned single-recording configuration that runs serially and detects the planted blobs."""
    configuration = SingleRecordingConfiguration()
    configuration.file_io.data_path = data_directory
    configuration.file_io.output_path = output_directory
    configuration.runtime.display_progress_bars = False
    configuration.registration.registration_metric_principal_components = 0
    configuration.nonrigid_registration.enabled = False
    configuration.one_photon_registration.enabled = False
    configuration.roi_detection.denoise = False
    configuration.roi_detection.preclassification_threshold = 0.0
    configuration.roi_detection.crop_to_soma = False
    configuration.roi_detection.threshold_scaling = 0.5
    configuration.main.tau = 0.01
    return configuration


def _prepare_pipeline_inputs(
    root: Path, *, frame_count: int = _FRAME_COUNT, seed: int = 0, display_progress_bars: bool = False
) -> tuple[Path, Path]:
    """Writes a raw recording and a saved configuration file, returning the configuration path and output directory."""
    data_directory = root / "data"
    output_directory = root / "output"
    _write_raw_recording(data_directory=data_directory, frame_count=frame_count, seed=seed)
    configuration = _make_configuration(data_directory=data_directory, output_directory=output_directory)
    configuration.runtime.display_progress_bars = display_progress_bars
    configuration_path = root / "configuration.yaml"
    configuration.save(file_path=configuration_path)
    return configuration_path, output_directory


def _binarize_to_disk(root: Path, *, frame_count: int = _FRAME_COUNT, seed: int = 0) -> SingleRecordingConfiguration:
    """Writes a raw recording and binarizes it, returning the configuration bound to the on-disk binary outputs."""
    data_directory = root / "data"
    output_directory = root / "output"
    _write_raw_recording(data_directory=data_directory, frame_count=frame_count, seed=seed)
    configuration = _make_configuration(data_directory=data_directory, output_directory=output_directory)
    # Writes the single-threaded filesystem bootstrap that binarize_recording's load-only resolution depends on.
    resolve_single_recording_contexts(configuration=configuration, persist=True)
    binarize_recording(configuration=configuration, workers=_TEST_WORKERS)
    return configuration


def _register_to_disk(root: Path, *, frame_count: int = _FRAME_COUNT, seed: int = 0) -> SingleRecordingConfiguration:
    """Binarizes and registers plane 0, returning the configuration bound to the registered on-disk outputs."""
    configuration = _binarize_to_disk(root=root, frame_count=frame_count, seed=seed)
    register_recording_plane(configuration=configuration, plane_index=0, workers=_TEST_WORKERS)
    return configuration


class TestRunSingleRecordingPipeline:
    """Tests run_single_recording_pipeline."""

    def test_runs_all_phases_when_no_flags_set(self, tmp_path: Path) -> None:
        """Verifies that omitting every phase flag runs the four phases end-to-end in their dependency order."""
        configuration_path, output_directory = _prepare_pipeline_inputs(tmp_path)

        run_single_recording_pipeline(configuration_path=configuration_path)

        combined = output_directory / "cindra" / "combined_metadata.npz"
        assert combined.exists()
        tracker = ProcessingTracker(file_path=output_directory / "cindra" / "single_recording_tracker.yaml")
        assert tracker.complete

        # A full run registers the plane and stamps the allocation each stage ran with onto its runtime context.
        context = RuntimeContext.load(root_path=output_directory / "cindra", plane_index=0)
        assert not isinstance(context, list)
        assert context.runtime.registration.is_registered(output_path=context.runtime.io.output_path)
        assert context.runtime.timing.registration_workers > 0
        assert context.runtime.timing.processing_workers > 0

    def test_runs_all_four_phase_jobs_for_every_plane(self, tmp_path: Path) -> None:
        """Verifies that the tracker records a registration job per plane, sequenced before the processing job."""
        configuration_path, output_directory = _prepare_pipeline_inputs(tmp_path)

        run_single_recording_pipeline(configuration_path=configuration_path)

        tracker = ProcessingTracker(file_path=output_directory / "cindra" / "single_recording_tracker.yaml")
        register_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.REGISTER)
        process_jobs = tracker.find_jobs(job_name=SingleRecordingJobNames.PROCESS)

        assert {specifier for _, specifier in register_jobs.values()} == {"plane_0"}
        assert {specifier for _, specifier in process_jobs.values()} == {"plane_0"}

        register_id = ProcessingTracker.generate_job_id(job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0")
        process_id = ProcessingTracker.generate_job_id(job_name=SingleRecordingJobNames.PROCESS, specifier="plane_0")
        assert tracker.get_job_status(job_id=register_id) == ProcessingStatus.SUCCEEDED
        assert tracker.get_job_status(job_id=process_id) == ProcessingStatus.SUCCEEDED

        # Registration is the prerequisite of processing, so it must have completed before processing started.
        register_info = tracker.get_job_info(job_id=register_id)
        process_info = tracker.get_job_info(job_id=process_id)
        assert register_info.completed_at is not None
        assert process_info.started_at is not None
        assert int(register_info.completed_at) <= int(process_info.started_at)

    def test_runs_explicit_flags_for_single_target_plane(self, tmp_path: Path) -> None:
        """Verifies that explicit phase flags with a specific target plane process only that plane and combine it."""
        configuration_path, output_directory = _prepare_pipeline_inputs(root=tmp_path, display_progress_bars=True)

        run_single_recording_pipeline(
            configuration_path=configuration_path,
            binarize=True,
            register=True,
            process=True,
            combine=True,
            target_plane=0,
        )

        combined = output_directory / "cindra" / "combined_metadata.npz"
        assert combined.exists()

    def test_out_of_range_target_plane_raises(self, tmp_path: Path) -> None:
        """Verifies that a target plane the recording does not hold raises a ValueError before the tracker is built."""
        configuration_path, output_directory = _prepare_pipeline_inputs(tmp_path)

        # The synthetic recording holds a single plane, so index 1 falls outside the resolved plane range.
        with pytest.raises(ValueError, match="The requested 'target_plane' must be"):
            run_single_recording_pipeline(configuration_path=configuration_path, register=True, target_plane=1)

        # The guard fires before align_jobs, so the pipeline leaves no tracker file behind.
        assert not (output_directory / "cindra" / "single_recording_tracker.yaml").exists()

    def test_remote_mode_executes_individual_jobs(self, tmp_path: Path) -> None:
        """Verifies that remote mode executes each of the four phase jobs addressed by its own job ID."""
        configuration_path, output_directory = _prepare_pipeline_inputs(tmp_path)

        # Bootstraps the per-plane runtime data and binaries so that the remote (load-only) resolutions succeed.
        run_single_recording_pipeline(configuration_path=configuration_path, binarize=True)

        binarize_id = ProcessingTracker.generate_job_id(job_name=SingleRecordingJobNames.BINARIZE, specifier="")
        register_id = ProcessingTracker.generate_job_id(job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0")
        process_id = ProcessingTracker.generate_job_id(job_name=SingleRecordingJobNames.PROCESS, specifier="plane_0")
        combine_id = ProcessingTracker.generate_job_id(job_name=SingleRecordingJobNames.COMBINE, specifier="")

        run_single_recording_pipeline(configuration_path=configuration_path, job_id=binarize_id, binarize=True)
        run_single_recording_pipeline(configuration_path=configuration_path, job_id=register_id, register=True)
        run_single_recording_pipeline(configuration_path=configuration_path, job_id=process_id, process=True)
        run_single_recording_pipeline(configuration_path=configuration_path, job_id=combine_id, combine=True)

        combined = output_directory / "cindra" / "combined_metadata.npz"
        assert combined.exists()

    def test_remote_mode_rejects_process_before_register(self, tmp_path: Path) -> None:
        """Verifies that addressing the processing job of an unregistered plane raises from the registration guard."""
        configuration_path, _ = _prepare_pipeline_inputs(tmp_path)

        run_single_recording_pipeline(configuration_path=configuration_path, binarize=True)

        process_id = ProcessingTracker.generate_job_id(job_name=SingleRecordingJobNames.PROCESS, specifier="plane_0")

        with pytest.raises(RuntimeError, match="must be registered before ROI detection"):
            run_single_recording_pipeline(configuration_path=configuration_path, job_id=process_id, process=True)

    def test_invalid_job_id_raises(self, tmp_path: Path) -> None:
        """Verifies that a job identifier outside the configuration's job universe raises a ValueError."""
        configuration_path, _ = _prepare_pipeline_inputs(tmp_path)

        # Bootstraps the runtime data so that the remote resolution reaches the job identifier validation.
        run_single_recording_pipeline(configuration_path=configuration_path, binarize=True)

        with pytest.raises(ValueError, match="must name a job the pipeline could produce"):
            run_single_recording_pipeline(configuration_path=configuration_path, job_id="deadbeefdeadbeef")

    def test_missing_configuration_file_raises(self, tmp_path: Path) -> None:
        """Verifies that a configuration path that does not exist raises a FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Expected the configuration file to"):
            run_single_recording_pipeline(configuration_path=tmp_path / "missing.yaml")

    def test_non_yaml_configuration_raises(self, tmp_path: Path) -> None:
        """Verifies that an existing configuration file without a .yaml extension raises a FileNotFoundError."""
        configuration_path = tmp_path / "configuration.txt"
        configuration_path.write_text("placeholder")

        with pytest.raises(FileNotFoundError, match="Expected the configuration file to"):
            run_single_recording_pipeline(configuration_path=configuration_path)

    def test_unparseable_configuration_raises(self, tmp_path: Path) -> None:
        """Verifies that a malformed configuration file raises a FileNotFoundError from the load guard."""
        configuration_path = tmp_path / "configuration.yaml"
        configuration_path.write_text("not a valid configuration: [unterminated\n  - {{{\n")

        with pytest.raises(FileNotFoundError, match="is not a valid"):
            run_single_recording_pipeline(configuration_path=configuration_path)

    def test_missing_output_path_raises(self, tmp_path: Path) -> None:
        """Verifies that a configuration without an output path raises a ValueError before context resolution."""
        configuration = _make_configuration(data_directory=tmp_path / "data", output_directory=None)
        configuration_path = tmp_path / "configuration.yaml"
        configuration.save(file_path=configuration_path)

        with pytest.raises(ValueError, match="output_path must be configured"):
            run_single_recording_pipeline(configuration_path=configuration_path, binarize=True)


class TestBinarizeRecording:
    """Tests binarize_recording."""

    def test_missing_data_path_raises(self, tmp_path: Path) -> None:
        """Verifies that a configuration without a data path raises a ValueError."""
        configuration = _make_configuration(data_directory=None, output_directory=tmp_path / "output")

        with pytest.raises(ValueError, match="data_path must be configured"):
            binarize_recording(configuration=configuration, workers=_TEST_WORKERS)

    def test_missing_output_path_raises(self, tmp_path: Path) -> None:
        """Verifies that a configuration without an output path raises a ValueError."""
        configuration = _make_configuration(data_directory=tmp_path / "data", output_directory=None)

        with pytest.raises(ValueError, match="output_path must be configured"):
            binarize_recording(configuration=configuration, workers=_TEST_WORKERS)

    def test_without_bootstrap_raises(self, tmp_path: Path) -> None:
        """Verifies that binarizing before the filesystem bootstrap exists raises a FileNotFoundError."""
        data_directory = tmp_path / "data"
        _write_raw_recording(data_directory)
        configuration = _make_configuration(data_directory=data_directory, output_directory=tmp_path / "output")

        with pytest.raises(FileNotFoundError, match="bootstrap persistence"):
            binarize_recording(configuration=configuration, workers=_TEST_WORKERS)

    def test_skips_existing_valid_binaries(self, tmp_path: Path) -> None:
        """Verifies that a second binarization is skipped when valid binaries already exist on disk."""
        configuration = _binarize_to_disk(tmp_path)
        binary_path = tmp_path / "output" / "cindra" / "plane_0" / "channel_1_data.bin"
        first_size = binary_path.stat().st_size

        binarize_recording(configuration=configuration, workers=_TEST_WORKERS)

        assert binary_path.exists()
        assert binary_path.stat().st_size == first_size

    def test_repeat_binarization_recreates_binaries(self, tmp_path: Path) -> None:
        """Verifies that the repeat_binarization flag forces a fresh conversion over existing valid binaries."""
        configuration = _binarize_to_disk(tmp_path)
        configuration.file_io.repeat_binarization = True
        binary_path = tmp_path / "output" / "cindra" / "plane_0" / "channel_1_data.bin"

        binarize_recording(configuration=configuration, workers=_TEST_WORKERS)

        assert binary_path.exists()

    def test_recreates_missing_binaries(self, tmp_path: Path) -> None:
        """Verifies that binarization recreates the binaries when a previously written binary file is deleted."""
        configuration = _binarize_to_disk(tmp_path)
        binary_path = tmp_path / "output" / "cindra" / "plane_0" / "channel_1_data.bin"
        binary_path.unlink()

        binarize_recording(configuration=configuration, workers=_TEST_WORKERS)

        assert binary_path.exists()

    def test_recreates_truncated_binaries(self, tmp_path: Path) -> None:
        """Verifies that binarization rebuilds a binary whose size disagrees with its recorded frame geometry."""
        configuration = _binarize_to_disk(tmp_path)
        binary_path = tmp_path / "output" / "cindra" / "plane_0" / "channel_1_data.bin"
        full_size = binary_path.stat().st_size

        # Simulates a conversion that died partway, which leaves a binary holding fewer frames than the plane's
        # runtime data records.
        with binary_path.open(mode="r+b") as binary_file:
            binary_file.truncate(full_size // 2)

        binarize_recording(configuration=configuration, workers=_TEST_WORKERS)

        assert binary_path.stat().st_size == full_size

    def test_recreates_binaries_left_mid_registration(self, tmp_path: Path) -> None:
        """Verifies that binarization rebuilds a marked binary and clears its marker without a caller request."""
        configuration = _binarize_to_disk(tmp_path)
        binary_path = tmp_path / "output" / "cindra" / "plane_0" / "channel_1_data.bin"
        create_registration_marker(binary_path=binary_path)

        binarize_recording(configuration=configuration, workers=_TEST_WORKERS)

        assert binary_path.exists()
        assert not resolve_registration_marker_path(binary_path=binary_path).exists()

    def test_skips_a_registered_binary(self, tmp_path: Path) -> None:
        """Verifies that a binary whose contents registration rewrote in place is treated as valid and left intact."""
        configuration = _binarize_to_disk(tmp_path)
        binary_path = tmp_path / "output" / "cindra" / "plane_0" / "channel_1_data.bin"

        # Registration rewrites a binary in place, so it changes the contents while preserving the size. Overwriting
        # every byte here reproduces that, and binarization must still treat the file as valid.
        original_bytes = binary_path.read_bytes()
        binary_path.write_bytes(bytes(len(original_bytes)))

        binarize_recording(configuration=configuration, workers=_TEST_WORKERS)

        assert binary_path.stat().st_size == len(original_bytes)
        assert binary_path.read_bytes() == bytes(len(original_bytes))


class TestRegisterRecordingPlane:
    """Tests register_recording_plane."""

    def test_skips_flyback_plane(self, tmp_path: Path) -> None:
        """Verifies that a plane listed as a flyback plane returns early without loading any runtime data."""
        configuration = _make_configuration(data_directory=None, output_directory=tmp_path / "output")
        configuration.main.ignored_flyback_planes = (0,)

        register_recording_plane(configuration=configuration, plane_index=0, workers=_TEST_WORKERS)

        assert not (tmp_path / "output" / "cindra" / "plane_0").exists()

    def test_missing_output_path_raises(self) -> None:
        """Verifies that a configuration without an output path raises a ValueError before loading runtime data."""
        configuration = _make_configuration(data_directory=None, output_directory=None)

        with pytest.raises(ValueError, match="output_path must be configured"):
            register_recording_plane(configuration=configuration, plane_index=0, workers=_TEST_WORKERS)

    def test_loaded_context_list_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifies that a multi-context load result for a single plane raises a TypeError."""
        configuration = _make_configuration(data_directory=None, output_directory=tmp_path / "output")

        def _fake_load(**_kwargs: object) -> list[object]:
            """Returns a two-element list, standing in for a load that resolved multiple plane contexts."""
            return [object(), object()]

        monkeypatch.setattr(RuntimeContext, "load", _fake_load)

        with pytest.raises(TypeError, match="Expected a single RuntimeContext"):
            register_recording_plane(configuration=configuration, plane_index=0, workers=_TEST_WORKERS)

    def test_frame_count_below_minimum_raises(self, tmp_path: Path) -> None:
        """Verifies that a plane with fewer than the minimum required frames raises a ValueError."""
        configuration = _binarize_to_disk(root=tmp_path, frame_count=40)

        with pytest.raises(ValueError, match="at least"):
            register_recording_plane(configuration=configuration, plane_index=0, workers=_TEST_WORKERS)

    def test_registers_plane_and_records_allocation(self, tmp_path: Path) -> None:
        """Verifies that registering a binarized plane writes registration output and records the used allocation."""
        _register_to_disk(tmp_path)

        context = RuntimeContext.load(root_path=tmp_path / "output" / "cindra", plane_index=0)
        assert not isinstance(context, list)
        assert context.runtime.registration.is_registered(output_path=context.runtime.io.output_path)
        assert (tmp_path / "output" / "cindra" / "plane_0" / "registration_data" / "reference_image.npy").exists()
        assert context.runtime.timing.registration_workers == _TEST_WORKERS

        # ROI detection is the processing stage's work, so a registered plane carries registration output only.
        assert not (tmp_path / "output" / "cindra" / "plane_0" / "roi_statistics.npz").exists()


class TestProcessPlane:
    """Tests process_plane."""

    def test_skips_flyback_plane(self, tmp_path: Path) -> None:
        """Verifies that a plane listed as a flyback plane returns early without loading any runtime data."""
        configuration = _make_configuration(data_directory=None, output_directory=tmp_path / "output")
        configuration.main.ignored_flyback_planes = (0,)

        process_plane(configuration=configuration, plane_index=0, workers=_TEST_WORKERS)

        assert not (tmp_path / "output" / "cindra" / "plane_0").exists()

    def test_missing_output_path_raises(self) -> None:
        """Verifies that a configuration without an output path raises a ValueError before loading runtime data."""
        configuration = _make_configuration(data_directory=None, output_directory=None)

        with pytest.raises(ValueError, match="output_path must be configured"):
            process_plane(configuration=configuration, plane_index=0, workers=_TEST_WORKERS)

    def test_loaded_context_list_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifies that a multi-context load result for a single plane raises a TypeError."""
        configuration = _make_configuration(data_directory=None, output_directory=tmp_path / "output")

        def _fake_load(**_kwargs: object) -> list[object]:
            """Returns a two-element list, standing in for a load that resolved multiple plane contexts."""
            return [object(), object()]

        monkeypatch.setattr(RuntimeContext, "load", _fake_load)

        with pytest.raises(TypeError, match="Expected a single RuntimeContext"):
            process_plane(configuration=configuration, plane_index=0, workers=_TEST_WORKERS)

    def test_frame_count_below_minimum_raises(self, tmp_path: Path) -> None:
        """Verifies that a plane with fewer than the minimum required frames raises a ValueError."""
        configuration = _binarize_to_disk(root=tmp_path, frame_count=40)

        with pytest.raises(ValueError, match="at least"):
            process_plane(configuration=configuration, plane_index=0, workers=_TEST_WORKERS)

    def test_unregistered_plane_raises(self, tmp_path: Path) -> None:
        """Verifies that processing a binarized but unregistered plane raises from the registration guard."""
        configuration = _binarize_to_disk(tmp_path)

        with pytest.raises(RuntimeError, match="must be registered before ROI detection"):
            process_plane(configuration=configuration, plane_index=0, workers=_TEST_WORKERS)

        # The guard fires before any detection output is written, so the plane holds no extraction results.
        plane_directory = tmp_path / "output" / "cindra" / "plane_0"
        assert not (plane_directory / "roi_statistics.npz").exists()
        assert not (plane_directory / "cell_fluorescence.npy").exists()

    def test_registered_plane_detects_rois(self, tmp_path: Path) -> None:
        """Verifies that processing a registered plane detects ROIs and records the used allocation."""
        configuration = _register_to_disk(tmp_path)

        process_plane(configuration=configuration, plane_index=0, workers=_TEST_WORKERS)

        # The extraction results are written to disk rather than held in the reloaded context, which memory-maps its
        # arrays on demand, so the on-disk artifacts are what prove detection and extraction both ran.
        plane_directory = tmp_path / "output" / "cindra" / "plane_0"
        assert (plane_directory / "roi_statistics.npz").exists()
        assert (plane_directory / "cell_fluorescence.npy").exists()
        assert (plane_directory / "spikes.npy").exists()

        context = RuntimeContext.load(root_path=tmp_path / "output" / "cindra", plane_index=0)
        assert not isinstance(context, list)
        assert context.runtime.timing.processing_workers == _TEST_WORKERS
        assert context.runtime.timing.date_processed

    def test_detection_disabled_skips_detection(self, tmp_path: Path) -> None:
        """Verifies that disabling ROI detection skips it on a registered plane at the recommended frame count."""
        configuration = _register_to_disk(root=tmp_path, frame_count=200)
        configuration.roi_detection.enabled = False

        process_plane(configuration=configuration, plane_index=0, workers=_TEST_WORKERS)

        plane_directory = tmp_path / "output" / "cindra" / "plane_0"
        assert not (plane_directory / "roi_statistics.npz").exists()
        assert not (plane_directory / "cell_fluorescence.npy").exists()

        context = RuntimeContext.load(root_path=tmp_path / "output" / "cindra", plane_index=0)
        assert not isinstance(context, list)
        assert context.runtime.timing.processing_workers == _TEST_WORKERS
        assert context.runtime.timing.date_processed


class TestSaveCombinedData:
    """Tests save_combined_data."""

    def test_empty_contexts_raises(self) -> None:
        """Verifies that combining an empty context list raises a ValueError."""
        with pytest.raises(ValueError, match="At least one RuntimeContext"):
            save_combined_data(contexts=[])

    def test_missing_output_path_raises(
        self, tmp_path: Path, single_recording_context: Callable[..., RuntimeContext]
    ) -> None:
        """Verifies that combining contexts whose configuration lacks an output path raises a ValueError."""
        context = single_recording_context(tmp_path)
        context.configuration.file_io.output_path = None

        with pytest.raises(ValueError, match="output_path must be configured"):
            save_combined_data(contexts=[context])


class TestAlignJobs:
    """Tests tracker job-registry alignment via ProcessingTracker.align_jobs."""

    def test_first_run_initializes_jobs(self, tmp_path: Path) -> None:
        """Verifies that a missing tracker file is initialized with the requested jobs."""
        tracker = ProcessingTracker(file_path=tmp_path / "tracker.yaml")
        jobs = [(SingleRecordingJobNames.BINARIZE, ""), (SingleRecordingJobNames.PROCESS, "plane_0")]
        universe = [*jobs, (SingleRecordingJobNames.COMBINE, "")]

        tracker.align_jobs(jobs=jobs, universe=universe)

        assert tracker.file_path.exists()
        assert len(tracker.find_jobs(job_name="")) == 2

    def test_foreign_entry_resets_tracker(self, tmp_path: Path) -> None:
        """Verifies that tracker entries outside the job universe trigger a reset before reinitialization."""
        tracker = ProcessingTracker(file_path=tmp_path / "tracker.yaml")
        tracker.initialize_jobs(jobs=[("foreign_job", "")])
        jobs = [(SingleRecordingJobNames.BINARIZE, "")]
        universe = [(SingleRecordingJobNames.BINARIZE, ""), (SingleRecordingJobNames.COMBINE, "")]

        tracker.align_jobs(jobs=jobs, universe=universe)

        assert not tracker.find_jobs(job_name="foreign_job")
        assert len(tracker.find_jobs(job_name="binarization")) == 1

    def test_additive_subset_registers_missing_jobs(self, tmp_path: Path) -> None:
        """Verifies that a tracker missing a requested universe job has the missing job added without a reset."""
        tracker = ProcessingTracker(file_path=tmp_path / "tracker.yaml")
        tracker.initialize_jobs(jobs=[(SingleRecordingJobNames.BINARIZE, "")])
        jobs = [(SingleRecordingJobNames.BINARIZE, ""), (SingleRecordingJobNames.PROCESS, "plane_0")]
        universe = [*jobs, (SingleRecordingJobNames.COMBINE, "")]

        tracker.align_jobs(jobs=jobs, universe=universe)

        assert len(tracker.find_jobs(job_name="")) == 2

    def test_fully_aligned_is_noop(self, tmp_path: Path) -> None:
        """Verifies that a tracker already holding every requested job preserves prior job state."""
        tracker = ProcessingTracker(file_path=tmp_path / "tracker.yaml")
        jobs = [(SingleRecordingJobNames.BINARIZE, ""), (SingleRecordingJobNames.PROCESS, "plane_0")]
        tracker.initialize_jobs(jobs=jobs)
        binarize_id = ProcessingTracker.generate_job_id(job_name=SingleRecordingJobNames.BINARIZE, specifier="")
        tracker.complete_job(job_id=binarize_id)
        universe = [*jobs, (SingleRecordingJobNames.COMBINE, "")]

        tracker.align_jobs(jobs=jobs, universe=universe)

        assert tracker.get_job_status(job_id=binarize_id) == ProcessingStatus.SUCCEEDED


class TestExecuteSingleRecordingJob:
    """Tests _execute_single_recording_job."""

    def test_unknown_job_fails_and_reraises(self, tmp_path: Path) -> None:
        """Verifies that an unrecognized job name marks the job failed and re-raises the ValueError."""
        tracker = ProcessingTracker(file_path=tmp_path / "tracker.yaml")
        tracker.initialize_jobs(jobs=[("unrecognized_job", "")])
        job_id = ProcessingTracker.generate_job_id(job_name="unrecognized_job", specifier="")
        configuration = _make_configuration(data_directory=None, output_directory=tmp_path / "output")

        with pytest.raises(ValueError, match="not recognized"):
            _execute_single_recording_job(
                configuration=configuration,
                job_name="unrecognized_job",  # type: ignore[arg-type]
                specifier="",
                job_id=job_id,
                tracker=tracker,
                workers=None,
            )

        assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED

    def test_combine_without_output_path_fails(self, tmp_path: Path) -> None:
        """Verifies that a combination job without an output path marks the job failed and re-raises the ValueError."""
        tracker = ProcessingTracker(file_path=tmp_path / "tracker.yaml")
        tracker.initialize_jobs(jobs=[(SingleRecordingJobNames.COMBINE, "")])
        job_id = ProcessingTracker.generate_job_id(job_name=SingleRecordingJobNames.COMBINE, specifier="")
        configuration = _make_configuration(data_directory=None, output_directory=None)

        with pytest.raises(ValueError, match="output_path must be configured"):
            _execute_single_recording_job(
                configuration=configuration,
                job_name=SingleRecordingJobNames.COMBINE,
                specifier="",
                job_id=job_id,
                tracker=tracker,
                workers=None,
            )

        assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED

    def test_invalid_worker_count_fails_and_reraises(self, tmp_path: Path) -> None:
        """Verifies that an invalid worker request marks the job failed on the tracker before the error propagates."""
        tracker = ProcessingTracker(file_path=tmp_path / "tracker.yaml")
        tracker.initialize_jobs(jobs=[(SingleRecordingJobNames.REGISTER, "plane_0")])
        job_id = ProcessingTracker.generate_job_id(job_name=SingleRecordingJobNames.REGISTER, specifier="plane_0")
        configuration = _make_configuration(data_directory=None, output_directory=tmp_path / "output")

        with pytest.raises(ValueError, match="but encountered 0"):
            _execute_single_recording_job(
                configuration=configuration,
                job_name=SingleRecordingJobNames.REGISTER,
                specifier="plane_0",
                job_id=job_id,
                tracker=tracker,
                workers=0,
            )

        assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED


class TestExecuteSingleRecordingJobInjection:
    """Tests execute_single_recording_job against a caller-owned tracker."""

    def test_injected_tracker_records_jobs_without_disturbing_foreign_entries(self, tmp_path: Path) -> None:
        """Verifies that all four phases stamp a foreign tracker while preserving its other jobs."""
        configuration_path, output_directory = _prepare_pipeline_inputs(tmp_path)

        # Builds a caller-owned tracker whose universe uses the caller's own job names and includes a foreign job the
        # injected executor leaves in place, mirroring how an owning pipeline would drive cindra.
        owner_job = ("session_owner", "")
        universe = [
            owner_job,
            ("recording_binarize", ""),
            ("recording_register", "plane_0"),
            ("recording_process", "plane_0"),
            ("recording_combine", ""),
        ]
        tracker = ProcessingTracker(file_path=tmp_path / "owner_tracker.yaml")
        tracker.align_jobs(jobs=universe, universe=universe)

        binarize_id = ProcessingTracker.generate_job_id(job_name="recording_binarize", specifier="")
        register_id = ProcessingTracker.generate_job_id(job_name="recording_register", specifier="plane_0")
        process_id = ProcessingTracker.generate_job_id(job_name="recording_process", specifier="plane_0")
        combine_id = ProcessingTracker.generate_job_id(job_name="recording_combine", specifier="")

        execute_single_recording_job(
            configuration_path=configuration_path,
            job_name=SingleRecordingJobNames.BINARIZE,
            specifier="",
            job_id=binarize_id,
            tracker=tracker,
            persist_bootstrap=True,
        )
        execute_single_recording_job(
            configuration_path=configuration_path,
            job_name=SingleRecordingJobNames.REGISTER,
            specifier="plane_0",
            job_id=register_id,
            tracker=tracker,
            workers=_TEST_WORKERS,
        )
        execute_single_recording_job(
            configuration_path=configuration_path,
            job_name=SingleRecordingJobNames.PROCESS,
            specifier="plane_0",
            job_id=process_id,
            tracker=tracker,
            workers=_TEST_WORKERS,
        )
        execute_single_recording_job(
            configuration_path=configuration_path,
            job_name=SingleRecordingJobNames.COMBINE,
            specifier="",
            job_id=combine_id,
            tracker=tracker,
        )

        assert (output_directory / "cindra" / "combined_metadata.npz").exists()
        assert tracker.get_job_status(job_id=binarize_id) == ProcessingStatus.SUCCEEDED
        assert tracker.get_job_status(job_id=register_id) == ProcessingStatus.SUCCEEDED
        assert tracker.get_job_status(job_id=process_id) == ProcessingStatus.SUCCEEDED
        assert tracker.get_job_status(job_id=combine_id) == ProcessingStatus.SUCCEEDED

        # Each stage records the allocation the caller passed to it, so the loaded context reports the injected count.
        context = RuntimeContext.load(root_path=output_directory / "cindra", plane_index=0)
        assert not isinstance(context, list)
        assert context.runtime.timing.registration_workers == _TEST_WORKERS
        assert context.runtime.timing.processing_workers == _TEST_WORKERS

        # The injected cindra jobs stamp only the identifiers they are given, so the caller's owner job is left in its
        # initial scheduled state.
        owner_id = ProcessingTracker.generate_job_id(job_name="session_owner", specifier="")
        assert tracker.get_job_status(job_id=owner_id) == ProcessingStatus.SCHEDULED

    def test_injected_process_before_register_raises(self, tmp_path: Path) -> None:
        """Verifies that a caller that skips the registration job fails loudly at the registration guard."""
        configuration_path, _ = _prepare_pipeline_inputs(tmp_path)

        universe = [("recording_binarize", ""), ("recording_process", "plane_0")]
        tracker = ProcessingTracker(file_path=tmp_path / "owner_tracker.yaml")
        tracker.align_jobs(jobs=universe, universe=universe)

        binarize_id = ProcessingTracker.generate_job_id(job_name="recording_binarize", specifier="")
        process_id = ProcessingTracker.generate_job_id(job_name="recording_process", specifier="plane_0")

        execute_single_recording_job(
            configuration_path=configuration_path,
            job_name=SingleRecordingJobNames.BINARIZE,
            specifier="",
            job_id=binarize_id,
            tracker=tracker,
            persist_bootstrap=True,
        )

        with pytest.raises(RuntimeError, match="must be registered before ROI detection"):
            execute_single_recording_job(
                configuration_path=configuration_path,
                job_name=SingleRecordingJobNames.PROCESS,
                specifier="plane_0",
                job_id=process_id,
                tracker=tracker,
                workers=_TEST_WORKERS,
            )

        assert tracker.get_job_status(job_id=process_id) == ProcessingStatus.FAILED

    def test_missing_configuration_file_raises(self, tmp_path: Path) -> None:
        """Verifies that a missing configuration path raises a FileNotFoundError through the injected executor."""
        tracker = ProcessingTracker(file_path=tmp_path / "owner_tracker.yaml")

        with pytest.raises(FileNotFoundError, match="Expected the configuration file to"):
            execute_single_recording_job(
                configuration_path=tmp_path / "missing.yaml",
                job_name=SingleRecordingJobNames.BINARIZE,
                specifier="",
                job_id="deadbeefdeadbeef",
                tracker=tracker,
            )
