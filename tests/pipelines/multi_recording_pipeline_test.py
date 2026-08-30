"""Contains integration tests for the multi-recording pipeline orchestration entry points."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from tifffile import TiffWriter
from ataraxis_base_utilities import error_format, ensure_directory_exists
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from cindra.layout import PARAMETERS_FILENAME
from cindra.dataclasses import MultiRecordingConfiguration, SingleRecordingConfiguration
from cindra.orchestration import (
    MULTI_RECORDING_TRACKER_FILENAME,
    MultiRecordingJobNames,
    resolve_multi_recording_jobs,
)
from cindra.orchestration.worker import (
    prime_dataset,
    execute_multi_recording_job,
    dispatch_multi_recording_job,
)
from cindra.orchestration.pipeline import run_multi_recording_pipeline, run_single_recording_pipeline

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

_FRAME_HEIGHT: int = 128
"""The synthetic frame height in pixels, large enough for phase-correlation registration to converge."""

_FRAME_WIDTH: int = 128
"""The synthetic frame width in pixels, large enough for phase-correlation registration to converge."""

_FRAME_COUNT: int = 60
"""The synthetic frame count used to process each recording feeding the multi-recording pipeline."""

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

_DATASET_NAME: str = "tracked_cells"
"""The dataset name under which multi-recording outputs are written for every test recording."""


class TestRunMultiRecordingPipeline:
    """Tests the phases a multi-recording run dispatches, the recordings it targets, and its configuration guards."""

    def test_runs_all_phases_when_no_flags_set(self, tmp_path: Path) -> None:
        """Verifies that omitting every phase flag runs discovery and extraction across both recordings."""
        configuration_path, first_output, second_output = _prepare_dataset(tmp_path=tmp_path)

        run_multi_recording_pipeline(configuration_path=configuration_path)

        assert (_multi_output(first_output) / "cell_fluorescence.npy").exists()
        assert (_multi_output(second_output) / "cell_fluorescence.npy").exists()

    def test_remote_and_target_recording_extraction(self, tmp_path: Path) -> None:
        """Verifies that discovery, remote extraction, and targeted local extraction populate both recordings."""
        configuration_path, first_output, second_output = _prepare_dataset(
            tmp_path=tmp_path, display_progress_bars=True
        )

        run_multi_recording_pipeline(configuration_path=configuration_path, discover=True)

        extract_first_id = ProcessingTracker.generate_job_id(job_name=MultiRecordingJobNames.EXTRACT, specifier="rec1")
        run_multi_recording_pipeline(configuration_path=configuration_path, job_id=extract_first_id, extract=True)
        run_multi_recording_pipeline(configuration_path=configuration_path, extract=True, target_recording="rec2")

        assert (_multi_output(first_output) / "cell_fluorescence.npy").exists()
        assert (_multi_output(second_output) / "cell_fluorescence.npy").exists()

    def test_invalid_job_id_raises(self, tmp_path: Path) -> None:
        """Verifies that a job identifier outside the configuration's job universe raises a ValueError."""
        configuration_path, first_output, _ = _prepare_dataset(tmp_path=tmp_path)

        # Bootstraps the multi-recording runtime data so that the remote resolution reaches job identifier validation.
        run_multi_recording_pipeline(configuration_path=configuration_path, discover=True)

        tracker_path = _multi_output(first_output) / MULTI_RECORDING_TRACKER_FILENAME
        universe = resolve_multi_recording_jobs(recording_ids=("rec1", "rec2"))
        universe_ids = sorted(
            ProcessingTracker.generate_job_id(job_name=job_name, specifier=specifier)
            for job_name, specifier in universe
        )
        expected_message = (
            f"Unable to resolve the job with ID 'deadbeefdeadbeef' against the job universe of the processing tracker "
            f"at '{tracker_path}'. The identifier must name a job the pipeline could produce, but the universe "
            f"holds only the jobs with IDs: {', '.join(universe_ids)}."
        )

        with pytest.raises(ValueError, match=error_format(expected_message)):
            run_multi_recording_pipeline(configuration_path=configuration_path, job_id="deadbeefdeadbeef")

    def test_unknown_target_recording_raises(self, tmp_path: Path) -> None:
        """Verifies that a target recording the dataset does not span raises a ValueError."""
        configuration_path, _, _ = _prepare_dataset(tmp_path=tmp_path)

        # Bootstraps the multi-recording runtime data so that the local resolution reaches the target validation.
        run_multi_recording_pipeline(configuration_path=configuration_path, discover=True)

        expected_message = (
            f"Unable to run the multi-recording cindra processing pipeline. The requested 'target_recording' must "
            f"name one of the recordings the dataset spans, but encountered 'rec3'. Resolved "
            f"recording identifiers: {['rec1', 'rec2']}."
        )

        with pytest.raises(ValueError, match=error_format(expected_message)):
            run_multi_recording_pipeline(configuration_path=configuration_path, extract=True, target_recording="rec3")

    def test_extract_without_discovery_raises(self, tmp_path: Path) -> None:
        """Verifies that extracting before discovery completes raises a RuntimeError from the statistics guard."""
        configuration_path, _, _ = _prepare_dataset(tmp_path=tmp_path)

        with pytest.raises(RuntimeError, match="Backward-transformed"):
            run_multi_recording_pipeline(configuration_path=configuration_path, extract=True)

    def test_missing_configuration_file_raises(self, tmp_path: Path) -> None:
        """Verifies that a configuration path that does not exist raises a FileNotFoundError."""
        configuration_path = tmp_path / "missing.yaml"
        expected_message = (
            f"Unable to run the multi-recording cindra processing pipeline. "
            f"Expected the configuration file to end with a '.yaml' extension and "
            f"exist at the specified path, but encountered: {configuration_path}."
        )

        with pytest.raises(FileNotFoundError, match=error_format(expected_message)):
            run_multi_recording_pipeline(configuration_path=configuration_path)

    def test_non_yaml_configuration_raises(self, tmp_path: Path) -> None:
        """Verifies that an existing configuration file without a .yaml extension raises a FileNotFoundError."""
        configuration_path = tmp_path / "configuration.txt"
        configuration_path.write_text("placeholder")
        expected_message = (
            f"Unable to run the multi-recording cindra processing pipeline. "
            f"Expected the configuration file to end with a '.yaml' extension and "
            f"exist at the specified path, but encountered: {configuration_path}."
        )

        with pytest.raises(FileNotFoundError, match=error_format(expected_message)):
            run_multi_recording_pipeline(configuration_path=configuration_path)

    def test_unparseable_configuration_raises(self, tmp_path: Path) -> None:
        """Verifies that a malformed configuration file raises a FileNotFoundError from the load guard."""
        configuration_path = tmp_path / "configuration.yaml"
        configuration_path.write_text("not a valid configuration: [unterminated\n  - {{{\n")
        expected_message = (
            "Unable to run the multi-recording cindra processing pipeline, as the input configuration file is not a "
            "valid multi-recording pipeline configuration file. Specifically, failed to load the file's data as a "
            "MultiRecordingConfiguration dataclass instance. Ensure that the 'configuration_path' argument points to a "
            "valid multi-recording configuration .yaml file."
        )

        with pytest.raises(FileNotFoundError, match=error_format(expected_message)):
            run_multi_recording_pipeline(configuration_path=configuration_path)

    @pytest.mark.parametrize("recording_count", [0, 1])
    def test_undersized_recording_set_raises(self, tmp_path: Path, recording_count: int) -> None:
        """Verifies that a configuration naming fewer than two recording directories raises a ValueError."""
        directories = tuple(tmp_path / f"rec{index}" for index in range(recording_count))
        configuration = _make_multi_configuration(recording_directories=directories)
        configuration_path = tmp_path / "configuration.yaml"
        configuration.save(file_path=configuration_path)

        expected_message = (
            f"Unable to run the multi-recording cindra processing pipeline. The "
            f"configuration file must specify at least two recording directories "
            f"under 'recording_io.recording_directories'. The provided configuration "
            f"specifies {recording_count}."
        )

        with pytest.raises(ValueError, match=error_format(expected_message)):
            run_multi_recording_pipeline(configuration_path=configuration_path)

    def test_empty_dataset_name_raises(self, tmp_path: Path) -> None:
        """Verifies that a configuration without a dataset name raises a ValueError."""
        configuration = _make_multi_configuration(
            recording_directories=(tmp_path / "rec1", tmp_path / "rec2"), dataset_name=""
        )
        configuration_path = tmp_path / "configuration.yaml"
        configuration.save(file_path=configuration_path)

        expected_message = (
            "Unable to run the multi-recording cindra processing pipeline. The "
            "configuration file must specify a dataset name under "
            "'recording_io.dataset_name'. The provided configuration has no "
            "dataset name specified."
        )

        with pytest.raises(ValueError, match=error_format(expected_message)):
            run_multi_recording_pipeline(configuration_path=configuration_path)

    def test_missing_main_output_path_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifies that a resolved context without a configured output path raises a ValueError."""
        configuration = _make_multi_configuration(recording_directories=(tmp_path / "rec1", tmp_path / "rec2"))
        configuration_path = tmp_path / "configuration.yaml"
        configuration.save(file_path=configuration_path)

        def _fake_resolve(**_kwargs: object) -> list[object]:
            """Returns a single stand-in context whose runtime output path is unset."""
            return [SimpleNamespace(runtime=SimpleNamespace(io=SimpleNamespace(recording_id="rec1"), output_path=None))]

        monkeypatch.setattr("cindra.orchestration.pipeline.resolve_multi_recording_contexts", _fake_resolve)

        expected_message = (
            "Unable to run the multi-recording pipeline. The main recording's "
            "output path is not configured in the resolved runtime context."
        )

        with pytest.raises(ValueError, match=error_format(expected_message)):
            run_multi_recording_pipeline(configuration_path=configuration_path, discover=True)


class TestDispatchMultiRecordingJob:
    """Tests the failed status the dispatcher records for a job name the multi-recording pipeline does not define."""

    def test_unknown_job_fails_and_reraises(self, tmp_path: Path) -> None:
        """Verifies that an unrecognized job name marks the job failed and re-raises the ValueError."""
        tracker = ProcessingTracker(file_path=tmp_path / "tracker.yaml")
        tracker.initialize_jobs(jobs=[("unrecognized_job", "")])
        job_id = ProcessingTracker.generate_job_id(job_name="unrecognized_job", specifier="")
        configuration = _make_multi_configuration(recording_directories=(tmp_path / "rec1",))

        expected_message = (
            f"Unable to execute the requested job 'unrecognized_job' with ID '{job_id}'. The input job name is not "
            f"recognized. Use one of the valid Job names: {list(MultiRecordingJobNames)}."
        )

        with pytest.raises(ValueError, match=error_format(expected_message)):
            dispatch_multi_recording_job(
                configuration=configuration,
                job_name="unrecognized_job",  # type: ignore[arg-type]  # The invalid name is the input under test.
                specifier="",
                job_id=job_id,
                tracker=tracker,
                workers=None,
            )

        assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED


class TestExecuteMultiRecordingJobInjection:
    """Tests execute_multi_recording_job against a caller-owned tracker."""

    def test_injected_tracker_records_jobs_without_disturbing_foreign_entries(self, tmp_path: Path) -> None:
        """Verifies that discovery and extraction stamp a foreign tracker while preserving its other jobs."""
        configuration_path, first_output, second_output = _prepare_dataset(tmp_path=tmp_path)

        # Builds a caller-owned tracker whose universe uses the caller's own job names and includes a foreign job the
        # injected executor leaves in place, mirroring how the forging pipeline drives cindra.
        define_job = ("dataset_definition", "")
        universe = [
            define_job,
            ("multiday_discovery", "animal"),
            ("multiday_extraction", "rec1"),
            ("multiday_extraction", "rec2"),
        ]
        tracker = ProcessingTracker(file_path=tmp_path / "forging_tracker.yaml")
        tracker.align_jobs(jobs=universe, universe=universe)

        discovery_id = ProcessingTracker.generate_job_id(job_name="multiday_discovery", specifier="animal")
        first_id = ProcessingTracker.generate_job_id(job_name="multiday_extraction", specifier="rec1")
        second_id = ProcessingTracker.generate_job_id(job_name="multiday_extraction", specifier="rec2")

        prime_dataset(configuration_path=configuration_path)
        execute_multi_recording_job(
            configuration_path=configuration_path,
            job_name=MultiRecordingJobNames.DISCOVER,
            specifier="",
            job_id=discovery_id,
            tracker=tracker,
        )
        execute_multi_recording_job(
            configuration_path=configuration_path,
            job_name=MultiRecordingJobNames.EXTRACT,
            specifier="rec1",
            job_id=first_id,
            tracker=tracker,
        )
        execute_multi_recording_job(
            configuration_path=configuration_path,
            job_name=MultiRecordingJobNames.EXTRACT,
            specifier="rec2",
            job_id=second_id,
            tracker=tracker,
        )

        assert (_multi_output(first_output) / "cell_fluorescence.npy").exists()
        assert (_multi_output(second_output) / "cell_fluorescence.npy").exists()
        assert tracker.get_job_status(job_id=discovery_id) == ProcessingStatus.SUCCEEDED
        assert tracker.get_job_status(job_id=first_id) == ProcessingStatus.SUCCEEDED
        assert tracker.get_job_status(job_id=second_id) == ProcessingStatus.SUCCEEDED

        # The injected cindra jobs stamp only the identifiers they are given, so the caller's definition job is left in
        # its initial scheduled state.
        define_id = ProcessingTracker.generate_job_id(job_name="dataset_definition", specifier="")
        assert tracker.get_job_status(job_id=define_id) == ProcessingStatus.SCHEDULED

    def test_missing_configuration_file_raises(self, tmp_path: Path) -> None:
        """Verifies that a missing configuration path raises a FileNotFoundError through the injected executor."""
        tracker = ProcessingTracker(file_path=tmp_path / "forging_tracker.yaml")
        configuration_path = tmp_path / "missing.yaml"
        expected_message = (
            f"Unable to run the multi-recording cindra processing pipeline. "
            f"Expected the configuration file to end with a '.yaml' extension and "
            f"exist at the specified path, but encountered: {configuration_path}."
        )

        with pytest.raises(FileNotFoundError, match=error_format(expected_message)):
            execute_multi_recording_job(
                configuration_path=configuration_path,
                job_name=MultiRecordingJobNames.DISCOVER,
                specifier="",
                job_id="deadbeefdeadbeef",
                tracker=tracker,
            )


def _build_flickering_movie(*, seed: int) -> NDArray[np.int16]:
    """Builds a synthetic movie whose spatially fixed Gaussian blobs flicker independently across frames."""
    generator = np.random.default_rng(seed=seed)
    rows, columns = np.mgrid[0:_FRAME_HEIGHT, 0:_FRAME_WIDTH]
    movie = np.full((_FRAME_COUNT, _FRAME_HEIGHT, _FRAME_WIDTH), fill_value=_BACKGROUND_LEVEL, dtype=np.float64)
    for center_row, center_column in _BLOB_CENTERS:
        blob = np.exp(-(((rows - center_row) ** 2 + (columns - center_column) ** 2) / (2.0 * _BLOB_SIGMA**2)))
        amplitudes = _BLOB_AMPLITUDE * (0.5 + np.abs(generator.standard_normal(_FRAME_COUNT)))
        movie += amplitudes[:, np.newaxis, np.newaxis] * blob[np.newaxis, :, :]
    return np.clip(a=movie, a_min=0, a_max=_MAXIMUM_PIXEL_VALUE).astype(np.int16)


def _build_processed_recording(root: Path, *, seed: int) -> Path:
    """Processes one synthetic recording through the single-recording pipeline and returns its output directory."""
    data_directory = root / "data"
    output_directory = root / "output"
    ensure_directory_exists(data_directory)
    movie = _build_flickering_movie(seed=seed)
    with TiffWriter(data_directory / "recording.tif") as writer:
        for frame_index in range(_FRAME_COUNT):
            writer.write(movie[frame_index])
    parameters = {"frame_rate": 30.0, "plane_number": 1, "channel_number": 1}
    (data_directory / PARAMETERS_FILENAME).write_text(json.dumps(parameters))

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
    configuration_path = root / "configuration.yaml"
    configuration.save(file_path=configuration_path)

    run_single_recording_pipeline(
        configuration_path=configuration_path, binarize=True, register=True, process=True, combine=True
    )
    return output_directory


def _make_multi_configuration(
    *, recording_directories: tuple[Path, ...], dataset_name: str = _DATASET_NAME, display_progress_bars: bool = False
) -> MultiRecordingConfiguration:
    """Builds a multi-recording configuration referencing the given recording directories and dataset name."""
    configuration = MultiRecordingConfiguration()
    configuration.recording_io.recording_directories = recording_directories
    configuration.recording_io.dataset_name = dataset_name
    configuration.runtime.display_progress_bars = display_progress_bars
    return configuration


def _prepare_dataset(tmp_path: Path, *, display_progress_bars: bool = False) -> tuple[Path, Path, Path]:
    """Processes two synthetic recordings and writes a multi-recording configuration referencing both of them."""
    first_output = _build_processed_recording(root=tmp_path / "rec1", seed=0)
    second_output = _build_processed_recording(root=tmp_path / "rec2", seed=1)
    configuration = _make_multi_configuration(
        recording_directories=(first_output, second_output), display_progress_bars=display_progress_bars
    )
    configuration_path = tmp_path / "multi_configuration.yaml"
    configuration.save(file_path=configuration_path)
    return configuration_path, first_output, second_output


def _multi_output(recording_output: Path) -> Path:
    """Returns the dataset-specific multi-recording output directory for a processed recording."""
    return recording_output / "cindra" / "multi_recording" / _DATASET_NAME
