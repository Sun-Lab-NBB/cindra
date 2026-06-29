"""Contains integration tests for the RuntimeContext and MultiRecordingRuntimeContext disk persistence entry points."""

from __future__ import annotations

import shutil
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
import pytest
from ataraxis_base_utilities import ensure_directory_exists

from cindra.dataclasses import (
    ROIMask,
    CombinedData,
    DetectionData,
    ROIStatistics,
    ExtractionData,
    RuntimeContext,
    MultiRecordingRuntimeData,
    MultiRecordingConfiguration,
    MultiRecordingRuntimeContext,
)
from cindra.dataclasses.runtime_contexts import _relocate_runtime_paths, _load_multi_recording_data
from cindra.dataclasses.single_recording_data import SingleRecordingRuntimeData, is_memory_mapped

if TYPE_CHECKING:
    from collections.abc import Callable


def _make_roi_statistics(frame_width: int) -> ROIStatistics:
    """Creates a minimal ROIStatistics instance for round-trip verification."""
    mask = ROIMask(
        y_pixels=np.array([1, 2, 3], dtype=np.int32),
        x_pixels=np.array([4, 5, 6], dtype=np.int32),
        pixel_weights=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        centroid=(2, 5),
        frame_width=frame_width,
        radius=2.0,
    )
    return ROIStatistics(
        mask=mask,
        footprint=4,
        compactness=0.8,
        solidity=0.9,
        pixel_count=3,
        aspect_ratio=1.0,
        normalized_pixel_count=0.5,
        skewness=1.2,
    )


def _populate_single_runtime_arrays(
    runtime: SingleRecordingRuntimeData, *, height: int, width: int, frame_count: int
) -> None:
    """Populates registration, detection, and extraction arrays so that a save/load round-trip carries data."""
    rng = np.random.default_rng(seed=7)
    runtime.registration.reference_image = rng.random(size=(height, width)).astype(np.float32)
    runtime.registration.rigid_y_offsets = np.arange(frame_count, dtype=np.int32)
    runtime.registration.rigid_x_offsets = (np.arange(frame_count, dtype=np.int32) * 2).astype(np.int32)
    runtime.detection.mean_image = rng.random(size=(height, width)).astype(np.float32)
    runtime.extraction.roi_statistics = [_make_roi_statistics(frame_width=width)]


def _build_multi_dataset(
    base: Path, *, recording_count: int = 2, write_config: bool = True, set_data_path: bool = True
) -> tuple[Path, ...]:
    """Builds and saves a multi-recording dataset under base and returns the natural-sorted output paths.

    Each recording owns a multi_recording output directory holding multi_recording_runtime_data.yaml. When
    set_data_path is True, each recording also owns a single-recording cindra root containing combined_metadata.npz, a
    valid plane_0 directory, and an empty decoy plane directory that exercises the plane discovery skip branch during
    relocation. The first recording's output directory holds the shared multi_recording_configuration.yaml.
    """
    data_paths: list[Path] = []
    multi_directories: list[Path] = []
    for index in range(recording_count):
        data_path = base / f"rec{index}" / "cindra"
        multi_directory = data_path / "multi_recording" / "dataset"
        ensure_directory_exists(multi_directory)
        data_paths.append(data_path)
        multi_directories.append(multi_directory)

    multi_directories_tuple = tuple(multi_directories)
    for index in range(recording_count):
        if set_data_path:
            CombinedData(
                detection=DetectionData(),
                extraction=ExtractionData(),
                plane_count=1,
                combined_height=16,
                combined_width=16,
                tau=1.0,
                sampling_rate=15.0,
            ).save(root_path=data_paths[index])

            # Writes a minimal single-recording runtime under a plane_0 directory so the multi-recording relocation
            # logic discovers and relocates underlying single-recording data during a move. The empty decoy plane
            # directory exercises the discovery skip branch (a plane_* entry without a runtime_data.yaml file).
            SingleRecordingRuntimeData().save(output_path=data_paths[index] / "plane_0")
            ensure_directory_exists(data_paths[index] / "plane_decoy")

        runtime = MultiRecordingRuntimeData()
        runtime.io.recording_id = f"rec{index}"
        runtime.io.dataset_name = "dataset"
        runtime.io.data_path = data_paths[index] if set_data_path else None
        runtime.io.dataset_output_paths = multi_directories_tuple
        runtime.save(output_path=multi_directories[index])

    if write_config:
        configuration = MultiRecordingConfiguration()
        configuration.runtime.parallel_workers = 3
        configuration.save(file_path=multi_directories[0] / "multi_recording_configuration.yaml")

    return multi_directories_tuple


class TestRuntimeContextSaveShared:
    """Tests RuntimeContext.save_shared."""

    def test_save_shared_writes_configuration_and_acquisition(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that save_shared writes configuration.yaml and acquisition_parameters.yaml into the cindra root."""
        context = single_recording_context(tmp_path)
        context.save_shared()

        cindra_root = context.configuration.file_io.output_path / "cindra"
        assert (cindra_root / "configuration.yaml").exists()
        assert (cindra_root / "acquisition_parameters.yaml").exists()

    def test_save_shared_raises_when_output_path_none(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that save_shared raises ValueError when the configuration output_path is None."""
        context = single_recording_context(tmp_path)
        context.configuration.file_io.output_path = None

        with pytest.raises(ValueError, match="Unable to save shared configuration data"):
            context.save_shared()


class TestRuntimeContextSaveRuntime:
    """Tests RuntimeContext.save_runtime."""

    def test_save_runtime_writes_runtime_yaml(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that save_runtime writes the runtime_data.yaml file into the plane output directory."""
        context = single_recording_context(tmp_path)
        context.save_runtime()

        assert (context.runtime.io.output_path / "runtime_data.yaml").exists()

    def test_save_runtime_raises_when_output_path_none(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that save_runtime raises ValueError when the runtime IOData output_path is None."""
        context = single_recording_context(tmp_path)
        context.runtime.io.output_path = None

        with pytest.raises(ValueError, match="Unable to save runtime data"):
            context.save_runtime()


class TestRuntimeContextLoad:
    """Tests RuntimeContext.load."""

    def test_load_all_planes_round_trip(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that load with plane_index=-1 returns a list of contexts with matching scalars and arrays."""
        context = single_recording_context(tmp_path, frame_height=24, frame_width=24, frame_count=12)
        _populate_single_runtime_arrays(context.runtime, height=24, width=24, frame_count=12)
        context.save_shared()
        context.save_runtime()

        output_root = context.configuration.file_io.output_path
        loaded = RuntimeContext.load(root_path=output_root)

        assert isinstance(loaded, list)
        assert len(loaded) == 1
        reloaded = loaded[0]

        assert reloaded.runtime.io.frame_height == 24
        assert reloaded.runtime.io.frame_width == 24
        assert reloaded.runtime.io.frame_count == 12
        assert reloaded.runtime.io.plane_index == 0
        assert reloaded.runtime.io.sampling_rate == pytest.approx(30.0)
        assert reloaded.configuration.file_io.output_path == output_root
        assert reloaded.acquisition.frame_rate == pytest.approx(30.0)

        reloaded.runtime.load_arrays()
        np.testing.assert_allclose(
            context.runtime.registration.reference_image, reloaded.runtime.registration.reference_image, rtol=1e-6
        )
        np.testing.assert_array_equal(
            context.runtime.registration.rigid_y_offsets, reloaded.runtime.registration.rigid_y_offsets
        )
        np.testing.assert_array_equal(
            context.runtime.registration.rigid_x_offsets, reloaded.runtime.registration.rigid_x_offsets
        )
        np.testing.assert_allclose(
            context.runtime.detection.mean_image, reloaded.runtime.detection.mean_image, rtol=1e-6
        )
        assert reloaded.runtime.extraction.roi_statistics is not None
        assert len(reloaded.runtime.extraction.roi_statistics) == 1

    def test_load_specific_plane_with_memory_map(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that load with a specific plane index returns a single context whose arrays memory-map."""
        context = single_recording_context(tmp_path, frame_height=20, frame_width=20, frame_count=10)
        _populate_single_runtime_arrays(context.runtime, height=20, width=20, frame_count=10)
        context.save_shared()
        context.save_runtime()

        output_root = context.configuration.file_io.output_path
        reloaded = RuntimeContext.load(root_path=output_root, plane_index=0)

        assert isinstance(reloaded, RuntimeContext)
        assert reloaded.runtime.io.plane_index == 0

        reloaded.runtime.memory_map_arrays()
        assert reloaded.runtime.registration.reference_image is not None
        assert is_memory_mapped(reloaded.runtime.registration.reference_image)
        np.testing.assert_allclose(
            context.runtime.registration.reference_image, reloaded.runtime.registration.reference_image, rtol=1e-6
        )

    def test_load_after_relocation_corrects_paths(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that load relocates stale runtime paths after the dataset directory tree is moved."""
        source = tmp_path / "source"
        generator = np.random.default_rng(seed=3)
        movie_channel_2 = generator.integers(low=100, high=1000, size=(10, 18, 18)).astype(np.int16)
        context = single_recording_context(
            source, frame_height=18, frame_width=18, frame_count=10, movie_channel_2=movie_channel_2
        )
        _populate_single_runtime_arrays(context.runtime, height=18, width=18, frame_count=10)
        context.save_shared()
        context.save_runtime()

        destination = tmp_path / "destination"
        shutil.copytree(source, destination)

        reloaded = RuntimeContext.load(root_path=destination / "output", plane_index=0)
        assert isinstance(reloaded, RuntimeContext)

        # The relocated runtime points at the destination tree, and its arrays load from the moved files.
        assert reloaded.runtime.output_path is not None
        assert reloaded.runtime.output_path.is_relative_to(destination)
        assert reloaded.runtime.io.registered_binary_path is not None
        assert reloaded.runtime.io.registered_binary_path.is_relative_to(destination)
        assert reloaded.runtime.io.registered_binary_path_channel_2 is not None
        assert reloaded.runtime.io.registered_binary_path_channel_2.is_relative_to(destination)

        reloaded.runtime.load_arrays()
        np.testing.assert_allclose(
            context.runtime.detection.mean_image, reloaded.runtime.detection.mean_image, rtol=1e-6
        )

    def test_load_raises_when_no_configuration_found(self, tmp_path: Path) -> None:
        """Verifies that load raises FileNotFoundError when no configuration.yaml exists under the root."""
        with pytest.raises(FileNotFoundError, match="No configuration"):
            RuntimeContext.load(root_path=tmp_path)

    def test_load_raises_when_multiple_configurations_found(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that load raises RuntimeError when more than one configuration.yaml exists under the root."""
        single_recording_context(tmp_path / "r1").save_shared()
        single_recording_context(tmp_path / "r2").save_shared()

        with pytest.raises(RuntimeError, match="expected exactly one"):
            RuntimeContext.load(root_path=tmp_path)

    def test_load_raises_when_acquisition_missing(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that load raises FileNotFoundError when the acquisition parameters file is absent."""
        context = single_recording_context(tmp_path)
        context.save_shared()
        (context.configuration.file_io.output_path / "cindra" / "acquisition_parameters.yaml").unlink()

        with pytest.raises(FileNotFoundError, match="Acquisition parameters file does not exist"):
            RuntimeContext.load(root_path=context.configuration.file_io.output_path)

    def test_load_raises_when_plane_directory_missing(
        self, single_recording_context: Callable[..., RuntimeContext], tmp_path: Path
    ) -> None:
        """Verifies that load raises FileNotFoundError when the requested plane directory does not exist."""
        context = single_recording_context(tmp_path)
        context.save_shared()
        context.save_runtime()

        with pytest.raises(FileNotFoundError, match="Plane directory does not exist"):
            RuntimeContext.load(root_path=context.configuration.file_io.output_path, plane_index=3)


class TestMultiRecordingRuntimeContextSaveShared:
    """Tests MultiRecordingRuntimeContext.save_shared."""

    def test_save_shared_writes_configuration(self, tmp_path: Path) -> None:
        """Verifies that save_shared writes the configuration into the main recording's output directory."""
        main_directory = tmp_path / "rec0" / "cindra" / "multi_recording" / "dataset"
        ensure_directory_exists(main_directory)
        runtime = MultiRecordingRuntimeData()
        runtime.output_path = main_directory
        runtime.io.dataset_output_paths = (main_directory,)
        context = MultiRecordingRuntimeContext(configuration=MultiRecordingConfiguration(), runtime=runtime)

        context.save_shared()

        assert (main_directory / "multi_recording_configuration.yaml").exists()

    def test_save_shared_raises_when_output_path_none(self, tmp_path: Path) -> None:
        """Verifies that save_shared raises ValueError when the runtime output_path is None."""
        runtime = MultiRecordingRuntimeData()
        runtime.output_path = None
        runtime.io.dataset_output_paths = (tmp_path,)
        context = MultiRecordingRuntimeContext(configuration=MultiRecordingConfiguration(), runtime=runtime)

        with pytest.raises(ValueError, match="Unable to save configuration"):
            context.save_shared()


class TestMultiRecordingRuntimeContextSaveRuntime:
    """Tests MultiRecordingRuntimeContext.save_runtime."""

    def test_save_runtime_writes_runtime_yaml(self, tmp_path: Path) -> None:
        """Verifies that save_runtime writes multi_recording_runtime_data.yaml into the recording output directory."""
        output_directory = tmp_path / "rec0" / "cindra" / "multi_recording" / "dataset"
        ensure_directory_exists(output_directory)
        runtime = MultiRecordingRuntimeData()
        runtime.output_path = output_directory
        context = MultiRecordingRuntimeContext(configuration=MultiRecordingConfiguration(), runtime=runtime)

        context.save_runtime()

        assert (output_directory / "multi_recording_runtime_data.yaml").exists()

    def test_save_runtime_raises_when_output_path_none(self) -> None:
        """Verifies that save_runtime raises ValueError when the runtime output_path is None."""
        runtime = MultiRecordingRuntimeData()
        runtime.output_path = None
        context = MultiRecordingRuntimeContext(configuration=MultiRecordingConfiguration(), runtime=runtime)

        with pytest.raises(ValueError, match="Unable to save runtime data"):
            context.save_runtime()


class TestMultiRecordingRuntimeContextLoad:
    """Tests MultiRecordingRuntimeContext.load."""

    def test_load_all_recordings_round_trip(self, tmp_path: Path) -> None:
        """Verifies that load with recording_index=-1 returns all recording contexts with combined data loaded."""
        _build_multi_dataset(tmp_path, recording_count=2)

        loaded = MultiRecordingRuntimeContext.load(root_path=tmp_path / "rec0")

        assert isinstance(loaded, list)
        assert len(loaded) == 2
        assert loaded[0].configuration.runtime.parallel_workers == 3
        assert loaded[0].runtime.io.recording_id == "rec0"
        assert loaded[1].runtime.io.recording_id == "rec1"

        # Combined metadata is eagerly loaded onto each recording's runtime.
        assert loaded[0].runtime.combined_data is not None
        assert loaded[0].runtime.combined_data.sampling_rate == pytest.approx(15.0)
        assert loaded[1].runtime.combined_data is not None

    def test_load_entry_recording_by_index(self, tmp_path: Path) -> None:
        """Verifies that load returns the entry recording directly when its index matches the resolved path."""
        _build_multi_dataset(tmp_path, recording_count=2)

        loaded = MultiRecordingRuntimeContext.load(root_path=tmp_path / "rec0", recording_index=0)

        assert isinstance(loaded, MultiRecordingRuntimeContext)
        assert loaded.runtime.io.recording_id == "rec0"
        assert loaded.runtime.combined_data is not None

    def test_load_other_recording_by_index(self, tmp_path: Path) -> None:
        """Verifies that load returns a non-entry recording by re-reading its runtime data from disk."""
        _build_multi_dataset(tmp_path, recording_count=2)

        loaded = MultiRecordingRuntimeContext.load(root_path=tmp_path / "rec0", recording_index=1)

        assert isinstance(loaded, MultiRecordingRuntimeContext)
        assert loaded.runtime.io.recording_id == "rec1"
        assert loaded.runtime.combined_data is not None

    def test_load_after_relocation_corrects_paths(self, tmp_path: Path) -> None:
        """Verifies that load relocates stale dataset paths across recordings after the dataset tree is moved."""
        source = tmp_path / "source"
        _build_multi_dataset(source, recording_count=2)

        destination = tmp_path / "destination"
        shutil.copytree(source, destination)

        loaded = MultiRecordingRuntimeContext.load(root_path=destination / "rec0")

        assert isinstance(loaded, list)
        assert len(loaded) == 2
        for context in loaded:
            assert context.runtime.output_path is not None
            assert context.runtime.output_path.is_relative_to(destination)
            assert context.runtime.io.data_path is not None
            assert context.runtime.io.data_path.is_relative_to(destination)
            for output_path in context.runtime.io.dataset_output_paths:
                assert output_path.is_relative_to(destination)
            assert context.runtime.combined_data is not None

    def test_load_after_relocation_with_one_recording_already_relocated(self, tmp_path: Path) -> None:
        """Verifies that load skips re-relocating a recording whose stored output path already matches its location."""
        source = tmp_path / "source"
        output_paths = _build_multi_dataset(source, recording_count=2)

        destination = tmp_path / "destination"
        shutil.copytree(source, destination)

        # Re-saves the non-entry recording in place at the destination so its stored output_path already matches the
        # resolved location, exercising the relocation skip branch for an already-correct recording.
        non_entry_output_path = destination / output_paths[1].relative_to(source)
        already_correct = MultiRecordingRuntimeData.load(output_path=non_entry_output_path)
        already_correct.save(output_path=non_entry_output_path)

        loaded = MultiRecordingRuntimeContext.load(root_path=destination / "rec0")

        assert isinstance(loaded, list)
        assert len(loaded) == 2
        for context in loaded:
            assert context.runtime.output_path is not None
            assert context.runtime.output_path.is_relative_to(destination)

    def test_load_after_relocation_without_data_paths(self, tmp_path: Path) -> None:
        """Verifies that load relocates a moved dataset whose recordings carry no single-recording data path."""
        source = tmp_path / "source"
        _build_multi_dataset(source, recording_count=2, set_data_path=False)

        destination = tmp_path / "destination"
        shutil.copytree(source, destination)

        loaded = MultiRecordingRuntimeContext.load(root_path=destination / "rec0")

        assert isinstance(loaded, list)
        assert len(loaded) == 2
        for context in loaded:
            assert context.runtime.output_path is not None
            assert context.runtime.output_path.is_relative_to(destination)
            assert context.runtime.io.data_path is None
            assert context.runtime.combined_data is None

    def test_load_raises_when_no_runtime_found(self, tmp_path: Path) -> None:
        """Verifies that load raises FileNotFoundError when no multi-recording runtime file exists under the root."""
        with pytest.raises(FileNotFoundError, match="No multi_recording_runtime_data"):
            MultiRecordingRuntimeContext.load(root_path=tmp_path)

    def test_load_raises_when_multiple_runtimes_found(self, tmp_path: Path) -> None:
        """Verifies that load raises RuntimeError when more than one runtime file exists under the root."""
        _build_multi_dataset(tmp_path, recording_count=2)

        with pytest.raises(RuntimeError, match="expected exactly one"):
            MultiRecordingRuntimeContext.load(root_path=tmp_path)

    def test_load_raises_when_dataset_output_paths_missing(self, tmp_path: Path) -> None:
        """Verifies that load raises FileNotFoundError when the runtime lacks dataset_output_paths."""
        output_directory = tmp_path / "rec0" / "cindra" / "multi_recording" / "dataset"
        ensure_directory_exists(output_directory)
        runtime = MultiRecordingRuntimeData()
        runtime.io.dataset_output_paths = ()
        runtime.save(output_path=output_directory)

        with pytest.raises(FileNotFoundError, match="resolve_multi_recording_contexts"):
            MultiRecordingRuntimeContext.load(root_path=tmp_path / "rec0")

    def test_load_raises_when_configuration_missing(self, tmp_path: Path) -> None:
        """Verifies that load raises FileNotFoundError when the configuration file is absent."""
        _build_multi_dataset(tmp_path, recording_count=1, write_config=False)

        with pytest.raises(FileNotFoundError, match="Configuration file does not exist"):
            MultiRecordingRuntimeContext.load(root_path=tmp_path / "rec0")

    def test_load_raises_when_recording_index_out_of_range(self, tmp_path: Path) -> None:
        """Verifies that load raises IndexError when the requested recording index is out of range."""
        _build_multi_dataset(tmp_path, recording_count=1)

        with pytest.raises(IndexError, match="is out of range"):
            MultiRecordingRuntimeContext.load(root_path=tmp_path / "rec0", recording_index=99)


class TestRelocateRuntimePaths:
    """Tests the _relocate_runtime_paths helper across its single-recording and multi-recording branches."""

    def test_single_recording_relocates_all_paths(self) -> None:
        """Verifies that all populated single-recording paths under the old prefix are relocated to the new prefix."""
        runtime = SingleRecordingRuntimeData()
        runtime.output_path = Path("/old/root/cindra/plane_0")
        runtime.io.registered_binary_path = Path("/old/root/cindra/plane_0/registered.bin")
        runtime.io.registered_binary_path_channel_2 = Path("/old/root/cindra/plane_0/registered_channel_2.bin")
        runtime.io.output_path = Path("/old/root/cindra/plane_0")

        _relocate_runtime_paths(runtime=runtime, old_prefix=Path("/old/root"), new_prefix=Path("/new/location"))

        assert runtime.output_path == Path("/new/location/cindra/plane_0")
        assert runtime.io.registered_binary_path == Path("/new/location/cindra/plane_0/registered.bin")
        assert runtime.io.registered_binary_path_channel_2 == Path(
            "/new/location/cindra/plane_0/registered_channel_2.bin"
        )
        assert runtime.io.output_path == Path("/new/location/cindra/plane_0")

    def test_single_recording_skips_none_and_relocated_paths(self) -> None:
        """Verifies that None paths and paths already under the new prefix are left unchanged."""
        none_runtime = SingleRecordingRuntimeData()
        none_runtime.output_path = None

        _relocate_runtime_paths(runtime=none_runtime, old_prefix=Path("/old"), new_prefix=Path("/new"))

        assert none_runtime.output_path is None
        assert none_runtime.io.registered_binary_path is None
        assert none_runtime.io.registered_binary_path_channel_2 is None
        assert none_runtime.io.output_path is None

        relocated_runtime = SingleRecordingRuntimeData()
        relocated_runtime.output_path = Path("/old/root/cindra/plane_0")
        relocated_runtime.io.registered_binary_path = Path("/new/location/cindra/plane_0/registered.bin")
        relocated_runtime.io.registered_binary_path_channel_2 = Path("/new/location/cindra/plane_0/channel_2.bin")
        relocated_runtime.io.output_path = Path("/new/location/cindra/plane_0")

        _relocate_runtime_paths(
            runtime=relocated_runtime, old_prefix=Path("/old/root"), new_prefix=Path("/new/location")
        )

        assert relocated_runtime.output_path == Path("/new/location/cindra/plane_0")
        assert relocated_runtime.io.registered_binary_path == Path("/new/location/cindra/plane_0/registered.bin")
        assert relocated_runtime.io.registered_binary_path_channel_2 == Path(
            "/new/location/cindra/plane_0/channel_2.bin"
        )
        assert relocated_runtime.io.output_path == Path("/new/location/cindra/plane_0")

    def test_multi_recording_relocates_data_and_output_paths(self) -> None:
        """Verifies that the multi-recording branch relocates data_path and every dataset output path variant."""
        runtime = MultiRecordingRuntimeData()
        runtime.output_path = Path("/old/root/cindra/multi_recording/dataset")
        runtime.io.data_path = Path("/old/root/cindra")
        runtime.io.dataset_output_paths = (
            Path("/old/root/cindra/multi_recording/dataset"),
            Path("/new/location/other/cindra/multi_recording/dataset"),
            Path("/old/elsewhere/cindra/multi_recording/dataset"),
        )

        _relocate_runtime_paths(runtime=runtime, old_prefix=Path("/old/root"), new_prefix=Path("/new/location"))

        assert runtime.output_path == Path("/new/location/cindra/multi_recording/dataset")
        assert runtime.io.data_path == Path("/new/location/cindra")
        relocated = runtime.io.dataset_output_paths
        # Path under the old prefix is relocated directly.
        assert relocated[0] == Path("/new/location/cindra/multi_recording/dataset")
        # Path already under the new prefix is preserved unchanged.
        assert relocated[1] == Path("/new/location/other/cindra/multi_recording/dataset")
        # Divergent cross-recording path keeps its recording-specific segment via the fallback substitution.
        assert relocated[2] == Path("/new/elsewhere/cindra/multi_recording/dataset")

    def test_multi_recording_skips_none_and_relocated_data_path(self) -> None:
        """Verifies that the multi-recording branch handles None and already-relocated data paths."""
        none_runtime = MultiRecordingRuntimeData()
        none_runtime.output_path = None
        none_runtime.io.data_path = None
        none_runtime.io.dataset_output_paths = ()

        _relocate_runtime_paths(runtime=none_runtime, old_prefix=Path("/old"), new_prefix=Path("/new"))

        assert none_runtime.output_path is None
        assert none_runtime.io.data_path is None
        assert none_runtime.io.dataset_output_paths == ()

        relocated_runtime = MultiRecordingRuntimeData()
        relocated_runtime.output_path = Path("/old/root/cindra")
        relocated_runtime.io.data_path = Path("/new/location/cindra")
        relocated_runtime.io.dataset_output_paths = ()

        _relocate_runtime_paths(
            runtime=relocated_runtime, old_prefix=Path("/old/root"), new_prefix=Path("/new/location")
        )

        assert relocated_runtime.output_path == Path("/new/location/cindra")
        assert relocated_runtime.io.data_path == Path("/new/location/cindra")


class TestLoadMultiRecordingData:
    """Tests the _load_multi_recording_data helper."""

    def test_loads_combined_data_when_present(self, tmp_path: Path) -> None:
        """Verifies that combined data is loaded onto the runtime when a data path with metadata is provided."""
        data_path = tmp_path / "cindra"
        ensure_directory_exists(data_path)
        CombinedData(
            detection=DetectionData(),
            extraction=ExtractionData(),
            plane_count=1,
            combined_height=8,
            combined_width=8,
            tau=1.0,
            sampling_rate=20.0,
        ).save(root_path=data_path)

        runtime = MultiRecordingRuntimeData()
        runtime.io.data_path = data_path

        _load_multi_recording_data(runtime)

        assert runtime.combined_data is not None
        assert runtime.combined_data.sampling_rate == pytest.approx(20.0)

    def test_skips_loading_when_data_path_none(self) -> None:
        """Verifies that combined data remains None when the runtime has no data path."""
        runtime = MultiRecordingRuntimeData()
        runtime.io.data_path = None

        _load_multi_recording_data(runtime)

        assert runtime.combined_data is None
