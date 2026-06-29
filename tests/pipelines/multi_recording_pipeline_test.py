"""Contains integration tests for the multi-recording pipeline orchestration entry points."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import pytest
from tifffile import TiffWriter
from ataraxis_base_utilities import ensure_directory_exists
from ataraxis_data_structures import ProcessingStatus, ProcessingTracker

from cindra.io.context import PARAMETERS_FILENAME
from cindra.dataclasses import MultiRecordingConfiguration, SingleRecordingConfiguration
import cindra.pipelines.pipeline as pipeline_module
from cindra.pipelines.pipeline import (
    MultiRecordingJobNames,
    _execute_multi_recording_job,
    run_multi_recording_pipeline,
    run_single_recording_pipeline,
)

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


def _build_flickering_movie(*, seed: int) -> NDArray[np.int16]:
    """Builds a synthetic movie whose spatially fixed Gaussian blobs flicker independently across frames."""
    generator = np.random.default_rng(seed=seed)
    rows, columns = np.mgrid[0:_FRAME_HEIGHT, 0:_FRAME_WIDTH]
    movie = np.full((_FRAME_COUNT, _FRAME_HEIGHT, _FRAME_WIDTH), _BACKGROUND_LEVEL, dtype=np.float64)
    for center_row, center_column in _BLOB_CENTERS:
        blob = np.exp(-(((rows - center_row) ** 2 + (columns - center_column) ** 2) / (2.0 * _BLOB_SIGMA**2)))
        amplitudes = _BLOB_AMPLITUDE * (0.5 + np.abs(generator.standard_normal(_FRAME_COUNT)))
        movie += amplitudes[:, np.newaxis, np.newaxis] * blob[np.newaxis, :, :]
    return np.clip(movie, 0, _MAXIMUM_PIXEL_VALUE).astype(np.int16)


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
    configuration.runtime.parallel_workers = 1
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

    run_single_recording_pipeline(configuration_path=configuration_path, binarize=True, process=True, combine=True)
    return output_directory


def _make_multi_configuration(
    *, recording_directories: tuple[Path, ...], dataset_name: str = _DATASET_NAME, display_progress_bars: bool = False
) -> MultiRecordingConfiguration:
    """Builds a serial multi-recording configuration referencing the given recording directories and dataset name."""
    configuration = MultiRecordingConfiguration()
    configuration.recording_io.recording_directories = recording_directories
    configuration.recording_io.dataset_name = dataset_name
    configuration.runtime.parallel_workers = 1
    configuration.runtime.display_progress_bars = display_progress_bars
    return configuration


def _prepare_dataset(tmp_path: Path, *, display_progress_bars: bool = False) -> tuple[Path, Path, Path]:
    """Processes two synthetic recordings and writes a multi-recording configuration referencing both of them."""
    first_output = _build_processed_recording(tmp_path / "rec1", seed=0)
    second_output = _build_processed_recording(tmp_path / "rec2", seed=1)
    configuration = _make_multi_configuration(
        recording_directories=(first_output, second_output), display_progress_bars=display_progress_bars
    )
    configuration_path = tmp_path / "multi_configuration.yaml"
    configuration.save(file_path=configuration_path)
    return configuration_path, first_output, second_output


def _multi_output(recording_output: Path) -> Path:
    """Returns the dataset-specific multi-recording output directory for a processed recording."""
    return recording_output / "cindra" / "multi_recording" / _DATASET_NAME


class TestRunMultiRecordingPipeline:
    """Tests run_multi_recording_pipeline."""

    def test_runs_all_phases_when_no_flags_set(self, tmp_path: Path) -> None:
        """Verifies that omitting every phase flag runs discovery and extraction across both recordings."""
        configuration_path, first_output, second_output = _prepare_dataset(tmp_path)

        run_multi_recording_pipeline(configuration_path=configuration_path)

        assert (_multi_output(first_output) / "cell_fluorescence.npy").exists()
        assert (_multi_output(second_output) / "cell_fluorescence.npy").exists()

    def test_remote_and_target_recording_extraction(self, tmp_path: Path) -> None:
        """Verifies that discovery, remote extraction, and targeted local extraction populate both recordings."""
        configuration_path, first_output, second_output = _prepare_dataset(tmp_path, display_progress_bars=True)

        run_multi_recording_pipeline(configuration_path=configuration_path, discover=True)

        extract_first_id = ProcessingTracker.generate_job_id(job_name=MultiRecordingJobNames.EXTRACT, specifier="rec1")
        run_multi_recording_pipeline(configuration_path=configuration_path, job_id=extract_first_id, extract=True)
        run_multi_recording_pipeline(configuration_path=configuration_path, extract=True, target_recording="rec2")

        assert (_multi_output(first_output) / "cell_fluorescence.npy").exists()
        assert (_multi_output(second_output) / "cell_fluorescence.npy").exists()

    def test_invalid_job_id_raises(self, tmp_path: Path) -> None:
        """Verifies that a job identifier outside the configuration's job universe raises a ValueError."""
        configuration_path, _, _ = _prepare_dataset(tmp_path)

        # Bootstraps the multi-recording runtime data so that the remote resolution reaches job identifier validation.
        run_multi_recording_pipeline(configuration_path=configuration_path, discover=True)

        with pytest.raises(ValueError, match="does not match"):
            run_multi_recording_pipeline(configuration_path=configuration_path, job_id="deadbeefdeadbeef")

    def test_extract_without_discovery_raises(self, tmp_path: Path) -> None:
        """Verifies that extracting before discovery completes raises a RuntimeError from the statistics guard."""
        configuration_path, _, _ = _prepare_dataset(tmp_path)

        with pytest.raises(RuntimeError, match="Backward-transformed"):
            run_multi_recording_pipeline(configuration_path=configuration_path, extract=True)

    def test_missing_configuration_file_raises(self, tmp_path: Path) -> None:
        """Verifies that a configuration path that does not exist raises a FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="Expected the configuration file to"):
            run_multi_recording_pipeline(configuration_path=tmp_path / "missing.yaml")

    def test_non_yaml_configuration_raises(self, tmp_path: Path) -> None:
        """Verifies that an existing configuration file without a .yaml extension raises a FileNotFoundError."""
        configuration_path = tmp_path / "configuration.txt"
        configuration_path.write_text("placeholder")

        with pytest.raises(FileNotFoundError, match="Expected the configuration file to"):
            run_multi_recording_pipeline(configuration_path=configuration_path)

    def test_unparseable_configuration_raises(self, tmp_path: Path) -> None:
        """Verifies that a malformed configuration file raises a FileNotFoundError from the load guard."""
        configuration_path = tmp_path / "configuration.yaml"
        configuration_path.write_text("not a valid configuration: [unterminated\n  - {{{\n")

        with pytest.raises(FileNotFoundError, match="is not a valid"):
            run_multi_recording_pipeline(configuration_path=configuration_path)

    def test_empty_recording_directories_raises(self, tmp_path: Path) -> None:
        """Verifies that a configuration without recording directories raises a ValueError."""
        configuration = _make_multi_configuration(recording_directories=())
        configuration_path = tmp_path / "configuration.yaml"
        configuration.save(file_path=configuration_path)

        with pytest.raises(ValueError, match="must specify at least two recording"):
            run_multi_recording_pipeline(configuration_path=configuration_path)

    def test_empty_dataset_name_raises(self, tmp_path: Path) -> None:
        """Verifies that a configuration without a dataset name raises a ValueError."""
        configuration = _make_multi_configuration(
            recording_directories=(tmp_path / "rec1", tmp_path / "rec2"), dataset_name=""
        )
        configuration_path = tmp_path / "configuration.yaml"
        configuration.save(file_path=configuration_path)

        with pytest.raises(ValueError, match="must specify a dataset name"):
            run_multi_recording_pipeline(configuration_path=configuration_path)

    def test_missing_main_output_path_raises(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verifies that a resolved context without a configured output path raises a ValueError."""
        configuration = _make_multi_configuration(recording_directories=(tmp_path / "rec1", tmp_path / "rec2"))
        configuration_path = tmp_path / "configuration.yaml"
        configuration.save(file_path=configuration_path)

        def _fake_resolve(**_kwargs: object) -> list[object]:
            return [SimpleNamespace(runtime=SimpleNamespace(io=SimpleNamespace(recording_id="rec1"), output_path=None))]

        monkeypatch.setattr(pipeline_module, "resolve_multi_recording_contexts", _fake_resolve)

        with pytest.raises(ValueError, match="output path is not configured"):
            run_multi_recording_pipeline(configuration_path=configuration_path, discover=True)


class TestExecuteMultiRecordingJob:
    """Tests _execute_multi_recording_job."""

    def test_unknown_job_fails_and_reraises(self, tmp_path: Path) -> None:
        """Verifies that an unrecognized job name marks the job failed and re-raises the ValueError."""
        tracker = ProcessingTracker(file_path=tmp_path / "tracker.yaml")
        tracker.initialize_jobs(jobs=[("unrecognized_job", "")])
        job_id = ProcessingTracker.generate_job_id(job_name="unrecognized_job", specifier="")
        configuration = _make_multi_configuration(recording_directories=(tmp_path / "rec1",))

        with pytest.raises(ValueError, match="not recognized"):
            _execute_multi_recording_job(
                configuration=configuration,
                job_name="unrecognized_job",  # type: ignore[arg-type]
                specifier="",
                job_id=job_id,
                tracker=tracker,
            )

        assert tracker.get_job_status(job_id=job_id) == ProcessingStatus.FAILED
