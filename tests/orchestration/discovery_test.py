"""Contains tests for the job universe resolvers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cindra.layout import (
    CHANNEL_1_BINARY_FILENAME,
    COMBINED_METADATA_FILENAME,
    ACQUISITION_PARAMETERS_FILENAME,
    REGISTRATION_DATA_DIRECTORY_NAME,
    TRACKING_TEMPLATE_MASKS_FILENAME,
    RecordingArrays,
    RegistrationArrays,
    resolve_plane_path,
    resolve_output_path,
    resolve_dataset_path,
)
from cindra.orchestration import (
    MultiRecordingJobNames,
    SingleRecordingJobNames,
    resolve_multi_recording_job_universe,
    resolve_single_recording_job_universe,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_parameters(output_root: Path, plane_count: int) -> None:
    """Writes the acquisition parameters that declare a recording's plane count."""
    output_path = resolve_output_path(output_root=output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / ACQUISITION_PARAMETERS_FILENAME).write_text(
        f"frame_rate: 30.0\nplane_number: {plane_count}\nchannel_number: 1\nroi_number: 1\n"
        f"roi_lines: []\nroi_x_coordinates: []\nroi_y_coordinates: []\n"
    )


def _convert_plane(output_root: Path, plane_index: int) -> None:
    """Writes the channel binary the conversion stage produces for one plane."""
    plane_path = resolve_plane_path(output_root=output_root, plane_index=plane_index)
    plane_path.mkdir(parents=True, exist_ok=True)
    (plane_path / CHANNEL_1_BINARY_FILENAME).write_bytes(b"")


def _process_plane(output_root: Path, plane_index: int) -> None:
    """Writes the extracted trace that marks one plane as processed."""
    plane_path = resolve_plane_path(output_root=output_root, plane_index=plane_index)
    plane_path.mkdir(parents=True, exist_ok=True)
    (plane_path / RecordingArrays.CELL_FLUORESCENCE).write_bytes(b"")


def _project_masks(output_root: Path, dataset_name: str) -> None:
    """Writes the projected ROI statistics one recording's extraction job reads."""
    dataset_path = resolve_dataset_path(output_root=output_root, dataset_name=dataset_name)
    dataset_path.mkdir(parents=True, exist_ok=True)
    (dataset_path / RecordingArrays.ROI_STATISTICS).write_bytes(b"")


def _register_plane(output_root: Path, plane_index: int) -> None:
    """Writes the reference image that marks one plane as registered."""
    directory = resolve_plane_path(output_root=output_root, plane_index=plane_index) / REGISTRATION_DATA_DIRECTORY_NAME
    directory.mkdir(parents=True, exist_ok=True)
    (directory / RegistrationArrays.REFERENCE_IMAGE).write_bytes(b"")


def _mark_processed(output_root: Path) -> None:
    """Writes the combined metadata archive that marks a recording's pipeline complete."""
    output_path = resolve_output_path(output_root=output_root)
    output_path.mkdir(parents=True, exist_ok=True)
    np.savez(output_path / COMBINED_METADATA_FILENAME, combined_height=np.array([1], dtype=np.uint16))


def _names(jobs: tuple[tuple[str, str], ...]) -> set[str]:
    """Returns the distinct job names the given job pairs carry."""
    return {job_name for job_name, _ in jobs}


class TestSingleRecordingJobUniverse:
    """Tests the single-recording job universe."""

    def test_absent_recording_resolves_to_an_empty_universe(self, tmp_path: Path) -> None:
        """Verifies that a recording carrying no parameters resolves rather than raising."""
        universe = resolve_single_recording_job_universe(output_root=tmp_path)

        assert universe.resolved is False
        assert universe.universe == ()
        assert universe.possible == ()

    def test_universe_covers_every_declared_job(self, tmp_path: Path) -> None:
        """Verifies that the universe holds one job per stage and per plane whatever exists on disk."""
        _write_parameters(output_root=tmp_path, plane_count=2)

        universe = resolve_single_recording_job_universe(output_root=tmp_path)

        assert universe.resolved is True
        assert universe.plane_count == 2
        assert len(universe.universe) == 6
        assert _names(universe.universe) == {
            SingleRecordingJobNames.BINARIZE,
            SingleRecordingJobNames.REGISTER,
            SingleRecordingJobNames.PROCESS,
            SingleRecordingJobNames.COMBINE,
        }

    def test_only_conversion_is_ready_before_anything_runs(self, tmp_path: Path) -> None:
        """Verifies that an unconverted recording offers only its conversion job."""
        _write_parameters(output_root=tmp_path, plane_count=2)

        universe = resolve_single_recording_job_universe(output_root=tmp_path)

        assert universe.possible == ((SingleRecordingJobNames.BINARIZE, ""),)

    def test_a_plane_directory_alone_does_not_make_registration_ready(self, tmp_path: Path) -> None:
        """Verifies that the priming step's plane directories do not by themselves report registration ready."""
        _write_parameters(output_root=tmp_path, plane_count=2)
        resolve_plane_path(output_root=tmp_path, plane_index=0).mkdir(parents=True, exist_ok=True)

        universe = resolve_single_recording_job_universe(output_root=tmp_path)

        assert (SingleRecordingJobNames.REGISTER, "plane_0") not in universe.possible

    def test_conversion_makes_registration_ready(self, tmp_path: Path) -> None:
        """Verifies that a plane carrying its channel binary offers registration and not yet processing."""
        _write_parameters(output_root=tmp_path, plane_count=2)
        _convert_plane(output_root=tmp_path, plane_index=0)

        universe = resolve_single_recording_job_universe(output_root=tmp_path)

        assert (SingleRecordingJobNames.REGISTER, "plane_0") in universe.possible
        assert (SingleRecordingJobNames.REGISTER, "plane_1") not in universe.possible
        assert (SingleRecordingJobNames.PROCESS, "plane_0") not in universe.possible

    def test_registration_makes_processing_ready(self, tmp_path: Path) -> None:
        """Verifies that a registered plane offers its processing job."""
        _write_parameters(output_root=tmp_path, plane_count=2)
        _register_plane(output_root=tmp_path, plane_index=1)

        universe = resolve_single_recording_job_universe(output_root=tmp_path)

        assert (SingleRecordingJobNames.PROCESS, "plane_1") in universe.possible
        assert (SingleRecordingJobNames.PROCESS, "plane_0") not in universe.possible

    def test_combination_waits_for_every_plane_to_be_processed(self, tmp_path: Path) -> None:
        """Verifies that combination waits for the traces it concatenates rather than for registration."""
        _write_parameters(output_root=tmp_path, plane_count=2)
        for plane_index in range(2):
            _register_plane(output_root=tmp_path, plane_index=plane_index)

        # Every plane is registered, but none carries the traces the combination stage reads.
        assert (SingleRecordingJobNames.COMBINE, "") not in resolve_single_recording_job_universe(
            output_root=tmp_path
        ).possible

        _process_plane(output_root=tmp_path, plane_index=0)
        assert (SingleRecordingJobNames.COMBINE, "") not in resolve_single_recording_job_universe(
            output_root=tmp_path
        ).possible

        _process_plane(output_root=tmp_path, plane_index=1)

        assert (SingleRecordingJobNames.COMBINE, "") in resolve_single_recording_job_universe(
            output_root=tmp_path
        ).possible

    def test_possible_is_always_a_subset_of_the_universe(self, tmp_path: Path) -> None:
        """Verifies that no ready job falls outside the declared universe."""
        _write_parameters(output_root=tmp_path, plane_count=3)
        for plane_index in range(3):
            _register_plane(output_root=tmp_path, plane_index=plane_index)

        universe = resolve_single_recording_job_universe(output_root=tmp_path)

        assert set(universe.possible) <= set(universe.universe)


class TestMultiRecordingJobUniverse:
    """Tests the multi-recording job universe."""

    def test_empty_dataset_resolves_to_an_empty_universe(self) -> None:
        """Verifies that a dataset spanning no recordings resolves rather than raising."""
        universe = resolve_multi_recording_job_universe(recording_roots=[], dataset_name="Set")

        assert universe.resolved is False
        assert universe.dataset_name == "set"
        assert universe.universe == ()

    def test_universe_covers_discovery_and_every_extraction(self, tmp_path: Path) -> None:
        """Verifies that the universe holds one discovery job and one extraction job per recording."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            root.mkdir()

        universe = resolve_multi_recording_job_universe(recording_roots=roots, dataset_name="set")

        assert universe.recording_ids == ("day1", "day2")
        assert _names(universe.universe) == {MultiRecordingJobNames.DISCOVER, MultiRecordingJobNames.EXTRACT}
        assert len(universe.universe) == 3

    def test_discovery_waits_for_every_recording_to_be_processed(self, tmp_path: Path) -> None:
        """Verifies that discovery becomes ready only once every recording carries its output."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            root.mkdir()
        _mark_processed(output_root=roots[0])
        assert not resolve_multi_recording_job_universe(recording_roots=roots, dataset_name="set").possible

        _mark_processed(output_root=roots[1])

        assert _names(resolve_multi_recording_job_universe(recording_roots=roots, dataset_name="set").possible) == {
            MultiRecordingJobNames.DISCOVER
        }

    def test_extraction_waits_for_its_own_recording_projection(self, tmp_path: Path) -> None:
        """Verifies that each extraction job waits for the masks projected into its own recording."""
        roots = [tmp_path / "day1", tmp_path / "day2"]
        for root in roots:
            root.mkdir()
            _mark_processed(output_root=root)

        # The dataset-wide template archive marks the clustering step, not any recording's projection.
        dataset_path = resolve_dataset_path(output_root=roots[0], dataset_name="set")
        dataset_path.mkdir(parents=True)
        (dataset_path / TRACKING_TEMPLATE_MASKS_FILENAME).write_bytes(b"")
        assert MultiRecordingJobNames.EXTRACT not in _names(
            resolve_multi_recording_job_universe(recording_roots=roots, dataset_name="set").possible
        )

        _project_masks(output_root=roots[1], dataset_name="set")
        universe = resolve_multi_recording_job_universe(recording_roots=roots, dataset_name="set")

        assert (MultiRecordingJobNames.EXTRACT, "day2") in universe.possible
        assert (MultiRecordingJobNames.EXTRACT, "day1") not in universe.possible
