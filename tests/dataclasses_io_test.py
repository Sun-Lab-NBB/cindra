"""Contains round-trip tests for all save/load/memory_map methods in cindra dataclass files."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

from cindra.dataclasses.multi_recording_data import (
    MultiRecordingRuntimeData,
    MultiRecordingTrackingData,
    MultiRecordingRegistrationData,
)
from cindra.dataclasses.single_recording_data import (
    IOData,
    ROIMask,
    CombinedData,
    DetectionData,
    ROIStatistics,
    ExtractionData,
    RegistrationData,
    SingleRecordingRuntimeData,
    is_memory_mapped,
    _load_optional_array_field,
    _save_optional_array_field,
)


def _make_roi_mask(
    pixel_count: int = 5,
    frame_width: int = 128,
    cluster_id: int = 0,
    recording_count: int = 0,
) -> ROIMask:
    """Creates a synthetic ROIMask with the given pixel count."""
    rng = np.random.default_rng(seed=pixel_count + cluster_id)
    y_pixels = rng.integers(low=0, high=64, size=pixel_count).astype(np.int32)
    x_pixels = rng.integers(low=0, high=frame_width, size=pixel_count).astype(np.int32)
    pixel_weights = rng.random(size=pixel_count).astype(np.float32)
    centroid = (int(np.median(y_pixels)), int(np.median(x_pixels)))
    return ROIMask(
        y_pixels=y_pixels,
        x_pixels=x_pixels,
        pixel_weights=pixel_weights,
        centroid=centroid,
        frame_width=frame_width,
        radius=5.5,
        cluster_id=cluster_id,
        recording_count=recording_count,
    )


def _make_roi_statistics(
    pixel_count: int = 8,
    frame_width: int = 128,
    include_soma: bool = True,
    include_neuropil: bool = True,
    include_overlap: bool = True,
    skewness_value: float | None = 1.5,
    plane_index: int = 0,
) -> ROIStatistics:
    """Creates a synthetic ROIStatistics instance."""
    mask = _make_roi_mask(pixel_count=pixel_count, frame_width=frame_width)
    rng = np.random.default_rng(seed=pixel_count + plane_index)
    soma_mask = rng.choice(a=[True, False], size=pixel_count).astype(np.bool_) if include_soma else None
    neuropil_mask = (
        rng.integers(low=0, high=frame_width * 64, size=pixel_count * 3).astype(np.int32) if include_neuropil else None
    )
    overlap_mask = rng.choice(a=[True, False], size=pixel_count).astype(np.bool_) if include_overlap else None
    mask.overlap_mask = overlap_mask
    return ROIStatistics(
        mask=mask,
        footprint=4,
        compactness=0.85,
        solidity=0.92,
        pixel_count=pixel_count,
        soma_mask=soma_mask,
        aspect_ratio=1.1,
        normalized_pixel_count=0.75,
        skewness=skewness_value,
        neuropil_mask=neuropil_mask,
        plane_index=plane_index,
    )


def _populate_registration_data() -> RegistrationData:
    """Creates a RegistrationData instance with all arrays populated."""
    frame_count = 20
    height = 32
    width = 32
    block_count = 4
    component_count = 3
    return RegistrationData(
        valid_y_range=(2, 30),
        valid_x_range=(2, 30),
        bad_frames=np.array([True, False] * (frame_count // 2), dtype=np.bool_),
        bidirectional_phase_offset=3,
        bidirectional_phase_corrected=True,
        normalization_minimum=100,
        normalization_maximum=5000,
        reference_image=np.random.default_rng(seed=0).random(size=(height, width)).astype(np.float32),
        rigid_y_offsets=np.arange(frame_count, dtype=np.int32),
        rigid_x_offsets=np.arange(frame_count, dtype=np.int32) * 2,
        rigid_correlations=np.linspace(start=0.8, stop=1.0, num=frame_count).astype(np.float32),
        nonrigid_y_offsets=np.random.default_rng(seed=1).random(size=(frame_count, block_count)).astype(np.float32),
        nonrigid_x_offsets=np.random.default_rng(seed=2).random(size=(frame_count, block_count)).astype(np.float32),
        nonrigid_correlations=np.random.default_rng(seed=3).random(size=(frame_count, block_count)).astype(np.float32),
        principal_component_extreme_images=np.random.default_rng(seed=4)
        .random(size=(2, component_count, height, width))
        .astype(np.float32),
        principal_component_projections=np.random.default_rng(seed=5)
        .random(size=(frame_count, component_count))
        .astype(np.float32),
        principal_component_shift_metrics=np.random.default_rng(seed=6)
        .random(size=(component_count, 3))
        .astype(np.float32),
    )


def _populate_detection_data() -> DetectionData:
    """Creates a DetectionData instance with all arrays populated."""
    height = 32
    width = 32
    rng = np.random.default_rng(seed=42)
    return DetectionData(
        roi_diameter=12,
        aspect_ratio=1.2,
        mean_image=rng.random(size=(height, width)).astype(np.float32),
        enhanced_mean_image=rng.random(size=(height, width)).astype(np.float32),
        maximum_projection=rng.random(size=(height, width)).astype(np.float32),
        correlation_map=rng.random(size=(height, width)).astype(np.float32),
        roi_diameter_channel_2=10,
        mean_image_channel_2=rng.random(size=(height, width)).astype(np.float32),
        enhanced_mean_image_channel_2=rng.random(size=(height, width)).astype(np.float32),
        maximum_projection_channel_2=rng.random(size=(height, width)).astype(np.float32),
        correlation_map_channel_2=rng.random(size=(height, width)).astype(np.float32),
    )


def _populate_extraction_data(
    roi_count: int = 3,
    frame_count: int = 50,
) -> ExtractionData:
    """Creates an ExtractionData instance with all arrays populated."""
    rng = np.random.default_rng(seed=99)
    roi_statistics = [_make_roi_statistics(pixel_count=5 + i) for i in range(roi_count)]
    return ExtractionData(
        roi_statistics=roi_statistics,
        cell_fluorescence=rng.random(size=(roi_count, frame_count)).astype(np.float32),
        neuropil_fluorescence=rng.random(size=(roi_count, frame_count)).astype(np.float32),
        subtracted_fluorescence=rng.random(size=(roi_count, frame_count)).astype(np.float32),
        spikes=rng.random(size=(roi_count, frame_count)).astype(np.float32),
        cell_classification=rng.random(size=(roi_count, 2)).astype(np.float32),
        cell_colocalization=rng.random(size=(roi_count, 2)).astype(np.float32),
        corrected_structural_mean_image=rng.random(size=(32, 32)).astype(np.float32),
    )


def _assert_roi_masks_equal(original: ROIMask, loaded: ROIMask) -> None:
    """Asserts that two ROIMask instances have identical data."""
    np.testing.assert_array_equal(original.y_pixels, loaded.y_pixels)
    np.testing.assert_array_equal(original.x_pixels, loaded.x_pixels)
    np.testing.assert_allclose(original.pixel_weights, loaded.pixel_weights, rtol=1e-6)
    assert original.centroid == loaded.centroid
    assert original.frame_width == loaded.frame_width
    assert original.radius == pytest.approx(loaded.radius, abs=1e-4)
    assert original.cluster_id == loaded.cluster_id
    assert original.recording_count == loaded.recording_count


def _assert_roi_statistics_equal(original: ROIStatistics, loaded: ROIStatistics) -> None:
    """Asserts that two ROIStatistics instances have identical data."""
    _assert_roi_masks_equal(original.mask, loaded.mask)
    assert original.footprint == loaded.footprint
    assert original.compactness == pytest.approx(loaded.compactness, abs=1e-5)
    assert original.solidity == pytest.approx(loaded.solidity, abs=1e-5)
    assert original.pixel_count == loaded.pixel_count
    assert original.aspect_ratio == pytest.approx(loaded.aspect_ratio, abs=1e-5)
    assert original.normalized_pixel_count == pytest.approx(loaded.normalized_pixel_count, abs=1e-5)
    assert original.plane_index == loaded.plane_index

    # Skewness: both None or both equal.
    if original.skewness is None:
        assert loaded.skewness is None
    else:
        assert loaded.skewness is not None
        assert original.skewness == pytest.approx(loaded.skewness, abs=1e-5)

    # Optional array fields.
    if original.soma_mask is not None:
        assert loaded.soma_mask is not None
        np.testing.assert_array_equal(original.soma_mask, loaded.soma_mask)
    else:
        assert loaded.soma_mask is None

    if original.neuropil_mask is not None:
        assert loaded.neuropil_mask is not None
        np.testing.assert_array_equal(original.neuropil_mask, loaded.neuropil_mask)
    else:
        assert loaded.neuropil_mask is None

    if original.mask.overlap_mask is not None:
        assert loaded.mask.overlap_mask is not None
        np.testing.assert_array_equal(original.mask.overlap_mask, loaded.mask.overlap_mask)
    else:
        assert loaded.mask.overlap_mask is None


class TestRegistrationDataSaveLoad:
    """Tests RegistrationData save_arrays, load_arrays, and memory_map_arrays round-trips."""

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        """Verifies that RegistrationData arrays survive a save/load round-trip with correct dtypes."""
        original = _populate_registration_data()
        original.save_arrays(output_path=tmp_path)

        loaded = RegistrationData()
        loaded.load_arrays(output_path=tmp_path)

        assert loaded.bad_frames is not None
        np.testing.assert_array_equal(original.bad_frames, loaded.bad_frames)
        assert loaded.bad_frames.dtype == np.bool_

        assert loaded.reference_image is not None
        np.testing.assert_allclose(original.reference_image, loaded.reference_image, rtol=1e-6)
        assert loaded.reference_image.dtype == np.float32

        assert loaded.rigid_y_offsets is not None
        np.testing.assert_array_equal(original.rigid_y_offsets, loaded.rigid_y_offsets)
        assert loaded.rigid_y_offsets.dtype == np.int32

        assert loaded.rigid_x_offsets is not None
        np.testing.assert_array_equal(original.rigid_x_offsets, loaded.rigid_x_offsets)
        assert loaded.rigid_x_offsets.dtype == np.int32

        assert loaded.rigid_correlations is not None
        np.testing.assert_allclose(original.rigid_correlations, loaded.rigid_correlations, rtol=1e-6)
        assert loaded.rigid_correlations.dtype == np.float32

        assert loaded.nonrigid_y_offsets is not None
        np.testing.assert_allclose(original.nonrigid_y_offsets, loaded.nonrigid_y_offsets, rtol=1e-6)

        assert loaded.nonrigid_x_offsets is not None
        np.testing.assert_allclose(original.nonrigid_x_offsets, loaded.nonrigid_x_offsets, rtol=1e-6)

        assert loaded.nonrigid_correlations is not None
        np.testing.assert_allclose(original.nonrigid_correlations, loaded.nonrigid_correlations, rtol=1e-6)

        assert loaded.principal_component_extreme_images is not None
        np.testing.assert_allclose(
            original.principal_component_extreme_images, loaded.principal_component_extreme_images, rtol=1e-6
        )

        assert loaded.principal_component_projections is not None
        np.testing.assert_allclose(
            original.principal_component_projections, loaded.principal_component_projections, rtol=1e-6
        )

        assert loaded.principal_component_shift_metrics is not None
        np.testing.assert_allclose(
            original.principal_component_shift_metrics, loaded.principal_component_shift_metrics, rtol=1e-6
        )

    def test_is_registered_returns_true_after_save(self, tmp_path: Path) -> None:
        """Verifies that is_registered returns True via disk check after save_arrays writes registration files."""
        empty_data = RegistrationData()
        assert not empty_data.is_registered(output_path=tmp_path)

        data = _populate_registration_data()
        data.save_arrays(output_path=tmp_path)
        assert empty_data.is_registered(output_path=tmp_path)

    def test_memory_map_round_trip(self, tmp_path: Path) -> None:
        """Verifies that memory_map_arrays produces memory-mapped arrays matching the original data."""
        original = _populate_registration_data()
        original.save_arrays(output_path=tmp_path)

        mapped = RegistrationData()
        mapped.memory_map_arrays(output_path=tmp_path)

        assert mapped.reference_image is not None
        assert is_memory_mapped(mapped.reference_image)
        np.testing.assert_allclose(original.reference_image, mapped.reference_image, rtol=1e-6)

        assert mapped.rigid_y_offsets is not None
        assert is_memory_mapped(mapped.rigid_y_offsets)
        np.testing.assert_array_equal(original.rigid_y_offsets, mapped.rigid_y_offsets)

        assert mapped.bad_frames is not None
        assert is_memory_mapped(mapped.bad_frames)
        np.testing.assert_array_equal(original.bad_frames, mapped.bad_frames)

    def test_load_returns_early_when_directory_missing(self, tmp_path: Path) -> None:
        """Verifies that load_arrays returns without error when the registration_data directory does not exist."""
        data = RegistrationData()
        data.load_arrays(output_path=tmp_path)
        assert data.reference_image is None

    def test_memory_map_returns_early_when_directory_missing(self, tmp_path: Path) -> None:
        """Verifies that memory_map_arrays returns without error when the registration_data directory does not exist."""
        data = RegistrationData()
        data.memory_map_arrays(output_path=tmp_path)
        assert data.reference_image is None


class TestDetectionDataSaveLoad:
    """Tests DetectionData save_arrays, load_arrays, and memory_map_arrays round-trips."""

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        """Verifies that DetectionData arrays survive a save/load round-trip with correct dtypes."""
        original = _populate_detection_data()
        original.save_arrays(output_path=tmp_path)

        loaded = DetectionData()
        loaded.load_arrays(output_path=tmp_path)

        assert loaded.mean_image is not None
        np.testing.assert_allclose(original.mean_image, loaded.mean_image, rtol=1e-6)
        assert loaded.mean_image.dtype == np.float32

        assert loaded.enhanced_mean_image is not None
        np.testing.assert_allclose(original.enhanced_mean_image, loaded.enhanced_mean_image, rtol=1e-6)

        assert loaded.maximum_projection is not None
        np.testing.assert_allclose(original.maximum_projection, loaded.maximum_projection, rtol=1e-6)

        assert loaded.correlation_map is not None
        np.testing.assert_allclose(original.correlation_map, loaded.correlation_map, rtol=1e-6)

        # Channel 2.
        assert loaded.mean_image_channel_2 is not None
        np.testing.assert_allclose(original.mean_image_channel_2, loaded.mean_image_channel_2, rtol=1e-6)

        assert loaded.enhanced_mean_image_channel_2 is not None
        np.testing.assert_allclose(
            original.enhanced_mean_image_channel_2, loaded.enhanced_mean_image_channel_2, rtol=1e-6
        )

        assert loaded.maximum_projection_channel_2 is not None
        np.testing.assert_allclose(
            original.maximum_projection_channel_2, loaded.maximum_projection_channel_2, rtol=1e-6
        )

        assert loaded.correlation_map_channel_2 is not None
        np.testing.assert_allclose(original.correlation_map_channel_2, loaded.correlation_map_channel_2, rtol=1e-6)

    def test_memory_map_round_trip(self, tmp_path: Path) -> None:
        """Verifies that memory_map_arrays produces memory-mapped arrays matching the original data."""
        original = _populate_detection_data()
        original.save_arrays(output_path=tmp_path)

        mapped = DetectionData()
        mapped.memory_map_arrays(output_path=tmp_path)

        assert mapped.mean_image is not None
        assert is_memory_mapped(mapped.mean_image)
        np.testing.assert_allclose(original.mean_image, mapped.mean_image, rtol=1e-6)

        assert mapped.correlation_map_channel_2 is not None
        assert is_memory_mapped(mapped.correlation_map_channel_2)
        np.testing.assert_allclose(original.correlation_map_channel_2, mapped.correlation_map_channel_2, rtol=1e-6)

    def test_load_returns_early_when_directory_missing(self, tmp_path: Path) -> None:
        """Verifies that load_arrays returns without error when detection_data directory does not exist."""
        data = DetectionData()
        data.load_arrays(output_path=tmp_path)
        assert data.mean_image is None

    def test_memory_map_returns_early_when_directory_missing(self, tmp_path: Path) -> None:
        """Verifies that memory_map_arrays returns without error when detection_data directory does not exist."""
        data = DetectionData()
        data.memory_map_arrays(output_path=tmp_path)
        assert data.mean_image is None


class TestROIMaskSaveLoadList:
    """Tests ROIMask.save_list and ROIMask.load_list round-trips."""

    def test_save_load_round_trip_multiple_masks(self, tmp_path: Path) -> None:
        """Verifies that a list of ROIMask instances survives a save/load round-trip."""
        masks = [
            _make_roi_mask(pixel_count=3, cluster_id=1, recording_count=2),
            _make_roi_mask(pixel_count=7, cluster_id=2, recording_count=5),
            _make_roi_mask(pixel_count=12, cluster_id=0, recording_count=0),
        ]

        file_path = tmp_path / "test_masks.npz"
        ROIMask.save_list(mask_list=masks, file_path=file_path)
        loaded_masks = ROIMask.load_list(file_path=file_path)

        assert len(loaded_masks) == len(masks)
        for original, loaded in zip(masks, loaded_masks, strict=True):
            _assert_roi_masks_equal(original, loaded)

    def test_save_empty_list_does_not_create_file(self, tmp_path: Path) -> None:
        """Verifies that saving an empty list does not create an output file."""
        file_path = tmp_path / "empty_masks.npz"
        ROIMask.save_list(mask_list=[], file_path=file_path)
        assert not file_path.exists()

    def test_save_load_single_mask(self, tmp_path: Path) -> None:
        """Verifies that a single ROIMask survives a save/load round-trip."""
        mask = _make_roi_mask(pixel_count=10, cluster_id=42, recording_count=3)
        file_path = tmp_path / "single_mask.npz"
        ROIMask.save_list(mask_list=[mask], file_path=file_path)
        loaded_masks = ROIMask.load_list(file_path=file_path)

        assert len(loaded_masks) == 1
        _assert_roi_masks_equal(mask, loaded_masks[0])


class TestROIStatisticsSaveLoadList:
    """Tests ROIStatistics.save_list and ROIStatistics.load_list round-trips."""

    def test_save_load_round_trip_with_all_fields(self, tmp_path: Path) -> None:
        """Verifies that ROIStatistics instances with all optional fields survive a save/load round-trip."""
        roi_list = [
            _make_roi_statistics(pixel_count=5, skewness_value=1.2, plane_index=0),
            _make_roi_statistics(pixel_count=8, skewness_value=-0.5, plane_index=1),
            _make_roi_statistics(pixel_count=12, skewness_value=0.0, plane_index=2),
        ]

        masks_path = tmp_path / "test_roi_masks.npz"
        stats_path = tmp_path / "test_roi_statistics.npz"
        ROIStatistics.save_list(roi_list=roi_list, masks_path=masks_path, stats_path=stats_path)
        loaded_list = ROIStatistics.load_list(masks_path=masks_path, stats_path=stats_path)

        assert len(loaded_list) == len(roi_list)
        for original, loaded in zip(roi_list, loaded_list, strict=True):
            _assert_roi_statistics_equal(original, loaded)

    def test_save_load_round_trip_with_none_optional_fields(self, tmp_path: Path) -> None:
        """Verifies that ROIStatistics instances with None optional fields survive a save/load round-trip."""
        roi_list = [
            _make_roi_statistics(
                pixel_count=6,
                include_soma=False,
                include_neuropil=False,
                include_overlap=False,
                skewness_value=None,
            ),
            _make_roi_statistics(
                pixel_count=10,
                include_soma=False,
                include_neuropil=False,
                include_overlap=False,
                skewness_value=None,
            ),
        ]

        masks_path = tmp_path / "masks_no_optional.npz"
        stats_path = tmp_path / "stats_no_optional.npz"
        ROIStatistics.save_list(roi_list=roi_list, masks_path=masks_path, stats_path=stats_path)
        loaded_list = ROIStatistics.load_list(masks_path=masks_path, stats_path=stats_path)

        assert len(loaded_list) == len(roi_list)
        for original, loaded in zip(roi_list, loaded_list, strict=True):
            _assert_roi_statistics_equal(original, loaded)

    def test_save_load_round_trip_mixed_optional_fields(self, tmp_path: Path) -> None:
        """Verifies that ROIStatistics instances with a mix of present and absent optional fields survive a
        save/load round-trip.
        """
        roi_list = [
            _make_roi_statistics(pixel_count=5, include_soma=True, include_neuropil=True, include_overlap=True),
            _make_roi_statistics(pixel_count=7, include_soma=False, include_neuropil=False, include_overlap=False),
            _make_roi_statistics(pixel_count=9, include_soma=True, include_neuropil=False, include_overlap=True),
        ]

        masks_path = tmp_path / "masks_mixed.npz"
        stats_path = tmp_path / "stats_mixed.npz"
        ROIStatistics.save_list(roi_list=roi_list, masks_path=masks_path, stats_path=stats_path)
        loaded_list = ROIStatistics.load_list(masks_path=masks_path, stats_path=stats_path)

        assert len(loaded_list) == len(roi_list)
        for original, loaded in zip(roi_list, loaded_list, strict=True):
            _assert_roi_statistics_equal(original, loaded)

    def test_save_empty_list_does_not_create_files(self, tmp_path: Path) -> None:
        """Verifies that saving an empty list does not create output files."""
        masks_path = tmp_path / "empty_masks.npz"
        stats_path = tmp_path / "empty_stats.npz"
        ROIStatistics.save_list(roi_list=[], masks_path=masks_path, stats_path=stats_path)
        assert not masks_path.exists()
        assert not stats_path.exists()


class TestExtractionDataSaveLoad:
    """Tests ExtractionData save_arrays, load_arrays, load_results, memory_map_arrays, and
    memory_map_results round-trips.
    """

    def test_save_load_arrays_loads_only_statistics_and_classification(self, tmp_path: Path) -> None:
        """Verifies that load_arrays loads roi_statistics and classification but not fluorescence traces."""
        original = _populate_extraction_data()
        original.save_arrays(output_path=tmp_path)

        loaded = ExtractionData()
        loaded.load_arrays(output_path=tmp_path)

        # ROI statistics should be loaded.
        assert loaded.roi_statistics is not None
        assert len(loaded.roi_statistics) == len(original.roi_statistics)

        # Classification should be loaded.
        assert loaded.cell_classification is not None
        np.testing.assert_allclose(original.cell_classification, loaded.cell_classification, rtol=1e-6)

        # Traces should NOT be loaded by load_arrays.
        assert loaded.cell_fluorescence is None
        assert loaded.neuropil_fluorescence is None
        assert loaded.subtracted_fluorescence is None
        assert loaded.spikes is None

        # Colocalization should NOT be loaded by load_arrays.
        assert loaded.cell_colocalization is None
        assert loaded.corrected_structural_mean_image is None

    def test_save_load_results_loads_all_arrays(self, tmp_path: Path) -> None:
        """Verifies that load_results loads all extraction result arrays including traces and colocalization."""
        original = _populate_extraction_data()
        original.save_arrays(output_path=tmp_path)

        loaded = ExtractionData()
        # First load roi_statistics via load_arrays (load_results does not load roi_statistics).
        loaded.load_arrays(output_path=tmp_path)
        loaded.load_results(output_path=tmp_path)

        # Traces should be loaded.
        assert loaded.cell_fluorescence is not None
        np.testing.assert_allclose(original.cell_fluorescence, loaded.cell_fluorescence, rtol=1e-6)

        assert loaded.neuropil_fluorescence is not None
        np.testing.assert_allclose(original.neuropil_fluorescence, loaded.neuropil_fluorescence, rtol=1e-6)

        assert loaded.subtracted_fluorescence is not None
        np.testing.assert_allclose(original.subtracted_fluorescence, loaded.subtracted_fluorescence, rtol=1e-6)

        assert loaded.spikes is not None
        np.testing.assert_allclose(original.spikes, loaded.spikes, rtol=1e-6)

        # Classification.
        assert loaded.cell_classification is not None
        np.testing.assert_allclose(original.cell_classification, loaded.cell_classification, rtol=1e-6)

        # Colocalization.
        assert loaded.cell_colocalization is not None
        np.testing.assert_allclose(original.cell_colocalization, loaded.cell_colocalization, rtol=1e-6)

        assert loaded.corrected_structural_mean_image is not None
        np.testing.assert_allclose(
            original.corrected_structural_mean_image, loaded.corrected_structural_mean_image, rtol=1e-6
        )

    def test_memory_map_arrays_produces_memory_mapped_classification(self, tmp_path: Path) -> None:
        """Verifies that memory_map_arrays produces memory-mapped classification arrays."""
        original = _populate_extraction_data()
        original.save_arrays(output_path=tmp_path)

        mapped = ExtractionData()
        mapped.memory_map_arrays(output_path=tmp_path)

        # ROI statistics should be loaded (eagerly, since npz does not support mmap).
        assert mapped.roi_statistics is not None
        assert len(mapped.roi_statistics) == len(original.roi_statistics)

        # Classification should be memory-mapped.
        assert mapped.cell_classification is not None
        assert is_memory_mapped(mapped.cell_classification)
        np.testing.assert_allclose(original.cell_classification, mapped.cell_classification, rtol=1e-6)

    def test_memory_map_results_produces_memory_mapped_traces(self, tmp_path: Path) -> None:
        """Verifies that memory_map_results produces memory-mapped trace and colocalization arrays."""
        original = _populate_extraction_data()
        original.save_arrays(output_path=tmp_path)

        mapped = ExtractionData()
        mapped.memory_map_results(output_path=tmp_path)

        assert mapped.cell_fluorescence is not None
        assert is_memory_mapped(mapped.cell_fluorescence)
        np.testing.assert_allclose(original.cell_fluorescence, mapped.cell_fluorescence, rtol=1e-6)

        assert mapped.spikes is not None
        assert is_memory_mapped(mapped.spikes)

        assert mapped.cell_colocalization is not None
        assert is_memory_mapped(mapped.cell_colocalization)

        assert mapped.corrected_structural_mean_image is not None
        assert is_memory_mapped(mapped.corrected_structural_mean_image)

    def test_save_load_without_optional_channel_2(self, tmp_path: Path) -> None:
        """Verifies that extraction data without channel 2 survives a save/load round-trip."""
        original = ExtractionData(
            roi_statistics=[_make_roi_statistics(pixel_count=5)],
            cell_fluorescence=np.ones(shape=(1, 10), dtype=np.float32),
            cell_classification=np.array([[1.0, 0.9]], dtype=np.float32),
        )
        original.save_arrays(output_path=tmp_path)

        loaded = ExtractionData()
        loaded.load_arrays(output_path=tmp_path)
        loaded.load_results(output_path=tmp_path)

        assert loaded.roi_statistics is not None
        assert len(loaded.roi_statistics) == 1
        assert loaded.cell_fluorescence is not None
        assert loaded.roi_statistics_channel_2 is None
        assert loaded.cell_fluorescence_channel_2 is None


class TestSingleRecordingRuntimeDataSaveLoad:
    """Tests SingleRecordingRuntimeData save and load round-trips."""

    def test_save_load_preserves_scalar_fields_and_arrays_none(self, tmp_path: Path) -> None:
        """Verifies that save/load preserves scalar fields while arrays remain None after YAML-only load."""
        data = SingleRecordingRuntimeData()
        data.io = IOData(
            frame_height=256,
            frame_width=256,
            frame_count=1000,
            sampling_rate=15.0,
            registered_binary_path=tmp_path / "registered.bin",
            output_path=tmp_path / "plane0",
            plane_index=0,
        )
        data.registration = _populate_registration_data()
        data.detection = _populate_detection_data()
        data.extraction = _populate_extraction_data()

        data.save(output_path=tmp_path)

        loaded = SingleRecordingRuntimeData.load(output_path=tmp_path)

        # Scalar IO fields should be preserved.
        assert loaded.io.frame_height == 256
        assert loaded.io.frame_width == 256
        assert loaded.io.frame_count == 1000
        assert loaded.io.sampling_rate == pytest.approx(15.0)
        assert loaded.io.plane_index == 0
        assert loaded.output_path == tmp_path

        # Registration scalar fields should be preserved.
        assert loaded.registration.valid_y_range == (2, 30)
        assert loaded.registration.valid_x_range == (2, 30)
        assert loaded.registration.bidirectional_phase_offset == 3
        assert loaded.registration.bidirectional_phase_corrected is True
        assert loaded.registration.normalization_minimum == 100
        assert loaded.registration.normalization_maximum == 5000

        # Detection scalar fields should be preserved.
        assert loaded.detection.roi_diameter == 12
        assert loaded.detection.aspect_ratio == pytest.approx(1.2)

        # Arrays should be None after YAML-only load.
        assert loaded.registration.reference_image is None
        assert loaded.registration.rigid_y_offsets is None
        assert loaded.detection.mean_image is None
        assert loaded.extraction.roi_statistics is None

    def test_save_does_not_corrupt_original_arrays(self, tmp_path: Path) -> None:
        """Verifies that the shallow-copy pattern in save does not null the original instance arrays."""
        data = SingleRecordingRuntimeData()
        data.registration = _populate_registration_data()
        data.detection = _populate_detection_data()
        data.extraction = _populate_extraction_data()

        data.save(output_path=tmp_path)

        # Original arrays should still be present.
        assert data.registration.reference_image is not None
        assert data.detection.mean_image is not None
        assert data.extraction.roi_statistics is not None

    def test_load_then_load_arrays_round_trip(self, tmp_path: Path) -> None:
        """Verifies that load + load_arrays recovers registration, detection, and extraction arrays."""
        data = SingleRecordingRuntimeData()
        data.registration = _populate_registration_data()
        data.detection = _populate_detection_data()
        data.extraction = _populate_extraction_data()
        data.save(output_path=tmp_path)

        loaded = SingleRecordingRuntimeData.load(output_path=tmp_path)
        loaded.load_arrays()

        assert loaded.registration.reference_image is not None
        np.testing.assert_allclose(data.registration.reference_image, loaded.registration.reference_image, rtol=1e-6)

        assert loaded.detection.mean_image is not None
        np.testing.assert_allclose(data.detection.mean_image, loaded.detection.mean_image, rtol=1e-6)

        assert loaded.extraction.roi_statistics is not None
        assert len(loaded.extraction.roi_statistics) == len(data.extraction.roi_statistics)

    def test_load_then_memory_map_arrays(self, tmp_path: Path) -> None:
        """Verifies that load + memory_map_arrays produces memory-mapped arrays."""
        data = SingleRecordingRuntimeData()
        data.registration = _populate_registration_data()
        data.detection = _populate_detection_data()
        data.save(output_path=tmp_path)

        loaded = SingleRecordingRuntimeData.load(output_path=tmp_path)
        loaded.memory_map_arrays()

        assert loaded.registration.reference_image is not None
        assert is_memory_mapped(loaded.registration.reference_image)
        assert loaded.detection.mean_image is not None
        assert is_memory_mapped(loaded.detection.mean_image)

    def test_release_arrays_clears_all_child_arrays(self) -> None:
        """Verifies that release_arrays sets all child array fields to None."""
        data = SingleRecordingRuntimeData()
        data.registration = _populate_registration_data()
        data.detection = _populate_detection_data()
        data.extraction = _populate_extraction_data()

        data.release_arrays()

        assert data.registration.reference_image is None
        assert data.registration.rigid_y_offsets is None
        assert data.detection.mean_image is None
        assert data.extraction.roi_statistics is None
        # Scalar fields should be preserved.
        assert data.registration.valid_y_range == (2, 30)

    def test_load_arrays_noop_when_output_path_none(self) -> None:
        """Verifies that load_arrays returns without error when output_path is None."""
        data = SingleRecordingRuntimeData()
        data.output_path = None
        data.load_arrays()
        assert data.registration.reference_image is None

    def test_memory_map_arrays_noop_when_output_path_none(self) -> None:
        """Verifies that memory_map_arrays returns without error when output_path is None."""
        data = SingleRecordingRuntimeData()
        data.output_path = None
        data.memory_map_arrays()
        assert data.registration.reference_image is None


class TestMultiRecordingRegistrationDataSaveLoad:
    """Tests MultiRecordingRegistrationData save_arrays, load_arrays, and memory_map_arrays round-trips."""

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        """Verifies that multi-recording registration arrays survive a save/load round-trip."""
        rng = np.random.default_rng(seed=7)
        height = 32
        width = 32
        original = MultiRecordingRegistrationData(
            deform_field_y=rng.random(size=(height, width)).astype(np.float32),
            deform_field_x=rng.random(size=(height, width)).astype(np.float32),
            transformed_mean_image=rng.random(size=(height, width)).astype(np.float32),
            transformed_enhanced_mean_image=rng.random(size=(height, width)).astype(np.float32),
            transformed_maximum_projection=rng.random(size=(height, width)).astype(np.float32),
            transformed_mean_image_channel_2=rng.random(size=(height, width)).astype(np.float32),
            transformed_enhanced_mean_image_channel_2=rng.random(size=(height, width)).astype(np.float32),
            transformed_maximum_projection_channel_2=rng.random(size=(height, width)).astype(np.float32),
            deformed_roi_masks=[_make_roi_mask(pixel_count=5), _make_roi_mask(pixel_count=8, cluster_id=1)],
            deformed_roi_masks_channel_2=[_make_roi_mask(pixel_count=4, cluster_id=2)],
        )
        original.save_arrays(output_path=tmp_path)

        loaded = MultiRecordingRegistrationData()
        loaded.load_arrays(output_path=tmp_path)

        assert loaded.deform_field_y is not None
        np.testing.assert_allclose(original.deform_field_y, loaded.deform_field_y, rtol=1e-6)

        assert loaded.deform_field_x is not None
        np.testing.assert_allclose(original.deform_field_x, loaded.deform_field_x, rtol=1e-6)

        assert loaded.transformed_mean_image is not None
        np.testing.assert_allclose(original.transformed_mean_image, loaded.transformed_mean_image, rtol=1e-6)

        assert loaded.transformed_enhanced_mean_image is not None
        np.testing.assert_allclose(
            original.transformed_enhanced_mean_image, loaded.transformed_enhanced_mean_image, rtol=1e-6
        )

        assert loaded.transformed_maximum_projection is not None
        np.testing.assert_allclose(
            original.transformed_maximum_projection, loaded.transformed_maximum_projection, rtol=1e-6
        )

        # Channel 2 images.
        assert loaded.transformed_mean_image_channel_2 is not None
        np.testing.assert_allclose(
            original.transformed_mean_image_channel_2, loaded.transformed_mean_image_channel_2, rtol=1e-6
        )
        assert loaded.transformed_enhanced_mean_image_channel_2 is not None
        assert loaded.transformed_maximum_projection_channel_2 is not None

        # Deformed ROI masks.
        assert loaded.deformed_roi_masks is not None
        assert len(loaded.deformed_roi_masks) == 2
        _assert_roi_masks_equal(original.deformed_roi_masks[0], loaded.deformed_roi_masks[0])
        _assert_roi_masks_equal(original.deformed_roi_masks[1], loaded.deformed_roi_masks[1])

        assert loaded.deformed_roi_masks_channel_2 is not None
        assert len(loaded.deformed_roi_masks_channel_2) == 1
        _assert_roi_masks_equal(original.deformed_roi_masks_channel_2[0], loaded.deformed_roi_masks_channel_2[0])

    def test_memory_map_round_trip(self, tmp_path: Path) -> None:
        """Verifies that memory_map_arrays produces memory-mapped .npy arrays and eagerly loaded .npz masks."""
        rng = np.random.default_rng(seed=8)
        original = MultiRecordingRegistrationData(
            deform_field_y=rng.random(size=(16, 16)).astype(np.float32),
            deform_field_x=rng.random(size=(16, 16)).astype(np.float32),
            transformed_mean_image=rng.random(size=(16, 16)).astype(np.float32),
            deformed_roi_masks=[_make_roi_mask(pixel_count=4)],
        )
        original.save_arrays(output_path=tmp_path)

        mapped = MultiRecordingRegistrationData()
        mapped.memory_map_arrays(output_path=tmp_path)

        assert mapped.deform_field_y is not None
        assert is_memory_mapped(mapped.deform_field_y)
        np.testing.assert_allclose(original.deform_field_y, mapped.deform_field_y, rtol=1e-6)

        assert mapped.transformed_mean_image is not None
        assert is_memory_mapped(mapped.transformed_mean_image)

        # ROI masks are eagerly loaded (npz does not support mmap).
        assert mapped.deformed_roi_masks is not None
        assert len(mapped.deformed_roi_masks) == 1

    def test_is_registered_returns_true_after_save(self, tmp_path: Path) -> None:
        """Verifies that is_registered returns True via disk check after save_arrays writes deformation fields."""
        data = MultiRecordingRegistrationData(
            deform_field_y=np.zeros(shape=(4, 4), dtype=np.float32),
            deform_field_x=np.zeros(shape=(4, 4), dtype=np.float32),
        )
        assert not data.is_registered(output_path=tmp_path)
        data.save_arrays(output_path=tmp_path)
        assert data.is_registered(output_path=tmp_path)


class TestMultiRecordingTrackingDataSaveLoad:
    """Tests MultiRecordingTrackingData save_arrays, load_arrays, and memory_map_arrays round-trips."""

    def test_save_load_round_trip(self, tmp_path: Path) -> None:
        """Verifies that tracking template masks survive a save/load round-trip."""
        masks_channel_1 = [_make_roi_mask(pixel_count=6, cluster_id=1), _make_roi_mask(pixel_count=9, cluster_id=2)]
        masks_channel_2 = [_make_roi_mask(pixel_count=4, cluster_id=3)]

        original = MultiRecordingTrackingData(
            template_masks=masks_channel_1,
            template_masks_channel_2=masks_channel_2,
            template_diameter=12,
            template_diameter_channel_2=10,
        )
        original.save_arrays(output_path=tmp_path)

        loaded = MultiRecordingTrackingData()
        loaded.load_arrays(output_path=tmp_path)

        assert loaded.template_masks is not None
        assert len(loaded.template_masks) == 2
        _assert_roi_masks_equal(masks_channel_1[0], loaded.template_masks[0])
        _assert_roi_masks_equal(masks_channel_1[1], loaded.template_masks[1])

        assert loaded.template_masks_channel_2 is not None
        assert len(loaded.template_masks_channel_2) == 1
        _assert_roi_masks_equal(masks_channel_2[0], loaded.template_masks_channel_2[0])

    def test_memory_map_delegates_to_load(self, tmp_path: Path) -> None:
        """Verifies that memory_map_arrays loads template masks identically to load_arrays."""
        masks = [_make_roi_mask(pixel_count=5)]
        original = MultiRecordingTrackingData(template_masks=masks)
        original.save_arrays(output_path=tmp_path)

        mapped = MultiRecordingTrackingData()
        mapped.memory_map_arrays(output_path=tmp_path)

        assert mapped.template_masks is not None
        assert len(mapped.template_masks) == 1
        _assert_roi_masks_equal(masks[0], mapped.template_masks[0])

    def test_load_no_file_leaves_none(self, tmp_path: Path) -> None:
        """Verifies that load_arrays leaves fields None when no files exist."""
        data = MultiRecordingTrackingData()
        data.load_arrays(output_path=tmp_path)
        assert data.template_masks is None
        assert data.template_masks_channel_2 is None

    def test_save_without_channel_2(self, tmp_path: Path) -> None:
        """Verifies that saving without channel 2 data does not create a channel 2 file."""
        original = MultiRecordingTrackingData(template_masks=[_make_roi_mask(pixel_count=3)])
        original.save_arrays(output_path=tmp_path)

        assert (tmp_path / "tracking_template_masks.npz").exists()
        assert not (tmp_path / "tracking_template_masks_channel_2.npz").exists()


def _populate_extraction_data_with_channel_2(
    roi_count: int = 3,
    frame_count: int = 50,
) -> ExtractionData:
    """Creates an ExtractionData instance with both channel 1 and channel 2 data populated."""
    rng = np.random.default_rng(seed=77)
    roi_statistics = [_make_roi_statistics(pixel_count=5 + i) for i in range(roi_count)]
    roi_statistics_channel_2 = [_make_roi_statistics(pixel_count=6 + i, plane_index=1) for i in range(roi_count)]
    return ExtractionData(
        roi_statistics=roi_statistics,
        cell_fluorescence=rng.random(size=(roi_count, frame_count)).astype(np.float32),
        neuropil_fluorescence=rng.random(size=(roi_count, frame_count)).astype(np.float32),
        subtracted_fluorescence=rng.random(size=(roi_count, frame_count)).astype(np.float32),
        spikes=rng.random(size=(roi_count, frame_count)).astype(np.float32),
        cell_classification=rng.random(size=(roi_count, 2)).astype(np.float32),
        roi_statistics_channel_2=roi_statistics_channel_2,
        cell_fluorescence_channel_2=rng.random(size=(roi_count, frame_count)).astype(np.float32),
        neuropil_fluorescence_channel_2=rng.random(size=(roi_count, frame_count)).astype(np.float32),
        subtracted_fluorescence_channel_2=rng.random(size=(roi_count, frame_count)).astype(np.float32),
        spikes_channel_2=rng.random(size=(roi_count, frame_count)).astype(np.float32),
        cell_classification_channel_2=rng.random(size=(roi_count, 2)).astype(np.float32),
        cell_colocalization=rng.random(size=(roi_count, 2)).astype(np.float32),
        corrected_structural_mean_image=rng.random(size=(32, 32)).astype(np.float32),
    )


class TestExtractionDataChannel2:
    """Tests ExtractionData save/load/memory_map round-trips with channel 2 data populated."""

    def test_save_load_arrays_loads_channel_2_statistics_and_classification(self, tmp_path: Path) -> None:
        """Verifies that load_arrays loads channel 2 ROI statistics and classification from disk."""
        original = _populate_extraction_data_with_channel_2()
        original.save_arrays(output_path=tmp_path)

        loaded = ExtractionData()
        loaded.load_arrays(output_path=tmp_path)

        # Channel 1 statistics and classification should be loaded.
        assert loaded.roi_statistics is not None
        assert len(loaded.roi_statistics) == len(original.roi_statistics)
        assert loaded.cell_classification is not None

        # Channel 2 statistics and classification should be loaded.
        assert loaded.roi_statistics_channel_2 is not None
        assert len(loaded.roi_statistics_channel_2) == len(original.roi_statistics_channel_2)
        for original_roi, loaded_roi in zip(
            original.roi_statistics_channel_2, loaded.roi_statistics_channel_2, strict=True
        ):
            _assert_roi_statistics_equal(original_roi, loaded_roi)

        assert loaded.cell_classification_channel_2 is not None
        np.testing.assert_allclose(
            original.cell_classification_channel_2, loaded.cell_classification_channel_2, rtol=1e-6
        )

        # Channel 2 traces should NOT be loaded by load_arrays.
        assert loaded.cell_fluorescence_channel_2 is None
        assert loaded.neuropil_fluorescence_channel_2 is None
        assert loaded.subtracted_fluorescence_channel_2 is None
        assert loaded.spikes_channel_2 is None

    def test_save_load_results_loads_all_channel_2_traces(self, tmp_path: Path) -> None:
        """Verifies that load_results loads all channel 2 fluorescence trace arrays from disk."""
        original = _populate_extraction_data_with_channel_2()
        original.save_arrays(output_path=tmp_path)

        loaded = ExtractionData()
        loaded.load_results(output_path=tmp_path)

        # Channel 2 traces should be loaded.
        assert loaded.cell_fluorescence_channel_2 is not None
        np.testing.assert_allclose(original.cell_fluorescence_channel_2, loaded.cell_fluorescence_channel_2, rtol=1e-6)

        assert loaded.neuropil_fluorescence_channel_2 is not None
        np.testing.assert_allclose(
            original.neuropil_fluorescence_channel_2, loaded.neuropil_fluorescence_channel_2, rtol=1e-6
        )

        assert loaded.subtracted_fluorescence_channel_2 is not None
        np.testing.assert_allclose(
            original.subtracted_fluorescence_channel_2, loaded.subtracted_fluorescence_channel_2, rtol=1e-6
        )

        assert loaded.spikes_channel_2 is not None
        np.testing.assert_allclose(original.spikes_channel_2, loaded.spikes_channel_2, rtol=1e-6)

        # Channel 2 classification should also be loaded by load_results.
        assert loaded.cell_classification_channel_2 is not None
        np.testing.assert_allclose(
            original.cell_classification_channel_2, loaded.cell_classification_channel_2, rtol=1e-6
        )

        # Colocalization should be loaded.
        assert loaded.cell_colocalization is not None
        np.testing.assert_allclose(original.cell_colocalization, loaded.cell_colocalization, rtol=1e-6)

        assert loaded.corrected_structural_mean_image is not None
        np.testing.assert_allclose(
            original.corrected_structural_mean_image, loaded.corrected_structural_mean_image, rtol=1e-6
        )

    def test_memory_map_arrays_loads_channel_2_statistics_and_classification(self, tmp_path: Path) -> None:
        """Verifies that memory_map_arrays loads channel 2 ROI statistics and memory-maps classification."""
        original = _populate_extraction_data_with_channel_2()
        original.save_arrays(output_path=tmp_path)

        mapped = ExtractionData()
        mapped.memory_map_arrays(output_path=tmp_path)

        # Channel 2 ROI statistics should be eagerly loaded (npz cannot be memory-mapped).
        assert mapped.roi_statistics_channel_2 is not None
        assert len(mapped.roi_statistics_channel_2) == len(original.roi_statistics_channel_2)

        # Channel 2 classification should be memory-mapped.
        assert mapped.cell_classification_channel_2 is not None
        assert is_memory_mapped(mapped.cell_classification_channel_2)
        np.testing.assert_allclose(
            original.cell_classification_channel_2, mapped.cell_classification_channel_2, rtol=1e-6
        )

    def test_memory_map_results_produces_memory_mapped_channel_2_traces(self, tmp_path: Path) -> None:
        """Verifies that memory_map_results produces memory-mapped channel 2 trace arrays."""
        original = _populate_extraction_data_with_channel_2()
        original.save_arrays(output_path=tmp_path)

        mapped = ExtractionData()
        mapped.memory_map_results(output_path=tmp_path)

        # Channel 2 traces should be memory-mapped.
        assert mapped.cell_fluorescence_channel_2 is not None
        assert is_memory_mapped(mapped.cell_fluorescence_channel_2)
        np.testing.assert_allclose(original.cell_fluorescence_channel_2, mapped.cell_fluorescence_channel_2, rtol=1e-6)

        assert mapped.neuropil_fluorescence_channel_2 is not None
        assert is_memory_mapped(mapped.neuropil_fluorescence_channel_2)

        assert mapped.subtracted_fluorescence_channel_2 is not None
        assert is_memory_mapped(mapped.subtracted_fluorescence_channel_2)

        assert mapped.spikes_channel_2 is not None
        assert is_memory_mapped(mapped.spikes_channel_2)

        # Channel 2 classification should be memory-mapped.
        assert mapped.cell_classification_channel_2 is not None
        assert is_memory_mapped(mapped.cell_classification_channel_2)


class TestMultiRecordingRegistrationDataChannel2:
    """Tests MultiRecordingRegistrationData save/load/memory_map round-trips with channel 2 data."""

    def test_memory_map_loads_channel_2_images_and_masks(self, tmp_path: Path) -> None:
        """Verifies that memory_map_arrays loads channel 2 transformed images and deformed masks."""
        rng = np.random.default_rng(seed=12)
        height = 16
        width = 16
        original = MultiRecordingRegistrationData(
            deform_field_y=rng.random(size=(height, width)).astype(np.float32),
            deform_field_x=rng.random(size=(height, width)).astype(np.float32),
            transformed_mean_image=rng.random(size=(height, width)).astype(np.float32),
            transformed_enhanced_mean_image=rng.random(size=(height, width)).astype(np.float32),
            transformed_maximum_projection=rng.random(size=(height, width)).astype(np.float32),
            transformed_mean_image_channel_2=rng.random(size=(height, width)).astype(np.float32),
            transformed_enhanced_mean_image_channel_2=rng.random(size=(height, width)).astype(np.float32),
            transformed_maximum_projection_channel_2=rng.random(size=(height, width)).astype(np.float32),
            deformed_roi_masks=[_make_roi_mask(pixel_count=5)],
            deformed_roi_masks_channel_2=[_make_roi_mask(pixel_count=4, cluster_id=3)],
        )
        original.save_arrays(output_path=tmp_path)

        mapped = MultiRecordingRegistrationData()
        mapped.memory_map_arrays(output_path=tmp_path)

        # Channel 2 transformed images should be memory-mapped.
        assert mapped.transformed_mean_image_channel_2 is not None
        assert is_memory_mapped(mapped.transformed_mean_image_channel_2)
        np.testing.assert_allclose(
            original.transformed_mean_image_channel_2, mapped.transformed_mean_image_channel_2, rtol=1e-6
        )

        assert mapped.transformed_enhanced_mean_image_channel_2 is not None
        assert is_memory_mapped(mapped.transformed_enhanced_mean_image_channel_2)

        assert mapped.transformed_maximum_projection_channel_2 is not None
        assert is_memory_mapped(mapped.transformed_maximum_projection_channel_2)

        # Channel 2 deformed ROI masks should be eagerly loaded (npz cannot be memory-mapped).
        assert mapped.deformed_roi_masks_channel_2 is not None
        assert len(mapped.deformed_roi_masks_channel_2) == 1
        _assert_roi_masks_equal(original.deformed_roi_masks_channel_2[0], mapped.deformed_roi_masks_channel_2[0])


class TestMultiRecordingRuntimeDataReleaseWithCombinedData:
    """Tests MultiRecordingRuntimeData.release_arrays when combined_data is loaded."""

    def test_release_arrays_clears_combined_data_child_arrays(self) -> None:
        """Verifies that release_arrays delegates to combined_data detection and extraction release_arrays."""
        rng = np.random.default_rng(seed=42)
        detection = DetectionData(
            mean_image=rng.random(size=(16, 16)).astype(np.float32),
            enhanced_mean_image=rng.random(size=(16, 16)).astype(np.float32),
        )
        extraction = ExtractionData(
            roi_statistics=[_make_roi_statistics(pixel_count=5)],
            cell_fluorescence=rng.random(size=(1, 20)).astype(np.float32),
            cell_classification=rng.random(size=(1, 2)).astype(np.float32),
        )
        combined_data = CombinedData(detection=detection, extraction=extraction)

        runtime = MultiRecordingRuntimeData()
        runtime.combined_data = combined_data

        runtime.release_arrays()

        # Combined data detection arrays should be released.
        assert runtime.combined_data.detection.mean_image is None
        assert runtime.combined_data.detection.enhanced_mean_image is None

        # Combined data extraction arrays should be released.
        assert runtime.combined_data.extraction.roi_statistics is None
        assert runtime.combined_data.extraction.cell_fluorescence is None
        assert runtime.combined_data.extraction.cell_classification is None


class TestMultiRecordingRuntimeDataSaveLoad:
    """Tests MultiRecordingRuntimeData save and load round-trips."""

    def test_save_load_preserves_scalar_fields_and_arrays_none(self, tmp_path: Path) -> None:
        """Verifies that save/load preserves scalar fields while arrays remain None after YAML-only load."""
        data = MultiRecordingRuntimeData()
        data.io.recording_id = "session_001"
        data.io.dataset_name = "test_dataset"
        data.io.data_path = tmp_path / "single_rec"
        data.io.selected_roi_indices = (0, 3, 7)
        data.timing.registration_time = 120
        data.timing.total_discovery_time = 300

        rng = np.random.default_rng(seed=10)
        data.registration = MultiRecordingRegistrationData(
            deform_field_y=rng.random(size=(16, 16)).astype(np.float32),
            deform_field_x=rng.random(size=(16, 16)).astype(np.float32),
        )
        data.tracking = MultiRecordingTrackingData(
            template_masks=[_make_roi_mask(pixel_count=4)],
            template_diameter=10,
        )

        data.save(output_path=tmp_path)

        loaded = MultiRecordingRuntimeData.load(output_path=tmp_path)

        # Scalar fields should be preserved.
        assert loaded.io.recording_id == "session_001"
        assert loaded.io.dataset_name == "test_dataset"
        assert loaded.io.selected_roi_indices == (0, 3, 7)
        assert loaded.timing.registration_time == 120
        assert loaded.timing.total_discovery_time == 300
        assert loaded.output_path == tmp_path

        # Arrays should be None after YAML-only load.
        assert loaded.registration.deform_field_y is None
        assert loaded.tracking.template_masks is None

        # combined_data should be None.
        assert loaded.combined_data is None

    def test_save_does_not_corrupt_original_arrays(self, tmp_path: Path) -> None:
        """Verifies that the shallow-copy pattern in save does not null the original instance arrays."""
        data = MultiRecordingRuntimeData()
        rng = np.random.default_rng(seed=11)
        data.registration = MultiRecordingRegistrationData(
            deform_field_y=rng.random(size=(8, 8)).astype(np.float32),
            deform_field_x=rng.random(size=(8, 8)).astype(np.float32),
        )
        data.tracking = MultiRecordingTrackingData(template_masks=[_make_roi_mask(pixel_count=3)])

        data.save(output_path=tmp_path)

        # Original arrays should still be present.
        assert data.registration.deform_field_y is not None
        assert data.tracking.template_masks is not None

    def test_load_then_load_arrays_round_trip(self, tmp_path: Path) -> None:
        """Verifies that load + load_arrays recovers registration, tracking, and extraction arrays."""
        data = MultiRecordingRuntimeData()
        rng = np.random.default_rng(seed=12)
        data.registration = MultiRecordingRegistrationData(
            deform_field_y=rng.random(size=(16, 16)).astype(np.float32),
            deform_field_x=rng.random(size=(16, 16)).astype(np.float32),
            deformed_roi_masks=[_make_roi_mask(pixel_count=5)],
        )
        data.tracking = MultiRecordingTrackingData(template_masks=[_make_roi_mask(pixel_count=6)])
        data.extraction = _populate_extraction_data()

        data.save(output_path=tmp_path)

        loaded = MultiRecordingRuntimeData.load(output_path=tmp_path)
        loaded.load_arrays()

        assert loaded.registration.deform_field_y is not None
        np.testing.assert_allclose(data.registration.deform_field_y, loaded.registration.deform_field_y, rtol=1e-6)

        assert loaded.tracking.template_masks is not None
        assert len(loaded.tracking.template_masks) == 1

        assert loaded.extraction.roi_statistics is not None

    def test_load_then_memory_map_arrays(self, tmp_path: Path) -> None:
        """Verifies that load + memory_map_arrays produces memory-mapped arrays."""
        data = MultiRecordingRuntimeData()
        rng = np.random.default_rng(seed=13)
        data.registration = MultiRecordingRegistrationData(
            deform_field_y=rng.random(size=(16, 16)).astype(np.float32),
            deform_field_x=rng.random(size=(16, 16)).astype(np.float32),
        )
        data.save(output_path=tmp_path)

        loaded = MultiRecordingRuntimeData.load(output_path=tmp_path)
        loaded.memory_map_arrays()

        assert loaded.registration.deform_field_y is not None
        assert is_memory_mapped(loaded.registration.deform_field_y)

    def test_release_arrays_clears_child_arrays(self) -> None:
        """Verifies that release_arrays clears all child array fields."""
        data = MultiRecordingRuntimeData()
        rng = np.random.default_rng(seed=14)
        data.registration = MultiRecordingRegistrationData(
            deform_field_y=rng.random(size=(8, 8)).astype(np.float32),
            deform_field_x=rng.random(size=(8, 8)).astype(np.float32),
        )
        data.tracking = MultiRecordingTrackingData(template_masks=[_make_roi_mask(pixel_count=3)])
        data.extraction = _populate_extraction_data()

        data.release_arrays()

        assert data.registration.deform_field_y is None
        assert data.tracking.template_masks is None
        assert data.extraction.roi_statistics is None

    def test_load_arrays_noop_when_output_path_none(self) -> None:
        """Verifies that load_arrays returns without error when output_path is None."""
        data = MultiRecordingRuntimeData()
        data.output_path = None
        data.load_arrays()
        assert data.registration.deform_field_y is None

    def test_memory_map_arrays_noop_when_output_path_none(self) -> None:
        """Verifies that memory_map_arrays returns without error when output_path is None."""
        data = MultiRecordingRuntimeData()
        data.output_path = None
        data.memory_map_arrays()
        assert data.registration.deform_field_y is None


class TestCombinedDataSaveLoad:
    """Tests CombinedData save and load round-trips."""

    def test_save_load_metadata_round_trip(self, tmp_path: Path) -> None:
        """Verifies that CombinedData metadata survives a save/load round-trip."""
        # Creates fake binary paths inside tmp_path so relative_to works.
        binary_path_1 = tmp_path / "plane0" / "registered.bin"
        binary_path_2 = tmp_path / "plane1" / "registered.bin"
        binary_path_1.parent.mkdir(parents=True, exist_ok=True)
        binary_path_2.parent.mkdir(parents=True, exist_ok=True)
        binary_path_1.touch()
        binary_path_2.touch()

        original = CombinedData(
            detection=_populate_detection_data(),
            extraction=_populate_extraction_data(),
            plane_count=2,
            combined_height=64,
            combined_width=128,
            tau=1.5,
            sampling_rate=15.0,
            plane_heights=np.array([32, 32], dtype=np.uint16),
            plane_widths=np.array([64, 64], dtype=np.uint16),
            plane_y_offsets=np.array([0, 32], dtype=np.int32),
            plane_x_offsets=np.array([0, 0], dtype=np.int32),
            registered_binary_paths=(binary_path_1, binary_path_2),
        )
        original.save(root_path=tmp_path)

        loaded = CombinedData.load(root_path=tmp_path)

        assert loaded.plane_count == 2
        assert loaded.combined_height == 64
        assert loaded.combined_width == 128
        assert loaded.tau == pytest.approx(1.5, abs=1e-5)
        assert loaded.sampling_rate == pytest.approx(15.0, abs=1e-5)
        np.testing.assert_array_equal(loaded.plane_heights, np.array([32, 32], dtype=np.uint16))
        np.testing.assert_array_equal(loaded.plane_widths, np.array([64, 64], dtype=np.uint16))
        np.testing.assert_array_equal(loaded.plane_y_offsets, np.array([0, 32], dtype=np.int32))
        np.testing.assert_array_equal(loaded.plane_x_offsets, np.array([0, 0], dtype=np.int32))

        # Binary paths should be reconstructed as absolute paths.
        assert len(loaded.registered_binary_paths) == 2
        assert loaded.registered_binary_paths[0] == binary_path_1
        assert loaded.registered_binary_paths[1] == binary_path_2

        # Arrays should be None after metadata-only load.
        assert loaded.detection.mean_image is None
        assert loaded.extraction.roi_statistics is None

    def test_save_load_with_channel_2_binary_paths(self, tmp_path: Path) -> None:
        """Verifies that channel 2 binary paths survive a save/load round-trip."""
        binary_path_ch1 = tmp_path / "plane0" / "registered.bin"
        binary_path_ch2 = tmp_path / "plane0" / "registered_channel_2.bin"
        binary_path_ch1.parent.mkdir(parents=True, exist_ok=True)
        binary_path_ch1.touch()
        binary_path_ch2.touch()

        original = CombinedData(
            detection=DetectionData(),
            extraction=ExtractionData(),
            plane_count=1,
            combined_height=32,
            combined_width=32,
            tau=1.0,
            sampling_rate=10.0,
            plane_heights=np.array([32], dtype=np.uint16),
            plane_widths=np.array([32], dtype=np.uint16),
            plane_y_offsets=np.array([0], dtype=np.int32),
            plane_x_offsets=np.array([0], dtype=np.int32),
            registered_binary_paths=(binary_path_ch1,),
            registered_binary_paths_channel_2=(binary_path_ch2,),
        )
        original.save(root_path=tmp_path)

        loaded = CombinedData.load(root_path=tmp_path)

        assert loaded.registered_binary_paths_channel_2 is not None
        assert len(loaded.registered_binary_paths_channel_2) == 1
        assert loaded.registered_binary_paths_channel_2[0] == binary_path_ch2

    def test_save_load_then_load_detection_extraction_arrays(self, tmp_path: Path) -> None:
        """Verifies that detection and extraction arrays can be loaded after CombinedData.load."""
        binary_path = tmp_path / "plane0" / "registered.bin"
        binary_path.parent.mkdir(parents=True, exist_ok=True)
        binary_path.touch()

        original = CombinedData(
            detection=_populate_detection_data(),
            extraction=_populate_extraction_data(),
            plane_count=1,
            combined_height=32,
            combined_width=32,
            tau=1.0,
            sampling_rate=10.0,
            registered_binary_paths=(binary_path,),
        )
        original.save(root_path=tmp_path)

        loaded = CombinedData.load(root_path=tmp_path)

        # Load child arrays.
        loaded.detection.load_arrays(output_path=tmp_path)
        loaded.extraction.load_arrays(output_path=tmp_path)

        assert loaded.detection.mean_image is not None
        np.testing.assert_allclose(original.detection.mean_image, loaded.detection.mean_image, rtol=1e-6)

        assert loaded.extraction.roi_statistics is not None
        assert len(loaded.extraction.roi_statistics) == len(original.extraction.roi_statistics)

    def test_load_raises_when_metadata_missing(self, tmp_path: Path) -> None:
        """Verifies that CombinedData.load raises FileNotFoundError when metadata file does not exist."""
        with pytest.raises(FileNotFoundError):
            CombinedData.load(root_path=tmp_path)

    def test_save_load_without_channel_2_binary_paths(self, tmp_path: Path) -> None:
        """Verifies that CombinedData without channel 2 binary paths stores None for that field."""
        binary_path = tmp_path / "plane0" / "registered.bin"
        binary_path.parent.mkdir(parents=True, exist_ok=True)
        binary_path.touch()

        original = CombinedData(
            detection=DetectionData(),
            extraction=ExtractionData(),
            plane_count=1,
            combined_height=32,
            combined_width=32,
            tau=1.0,
            sampling_rate=10.0,
            registered_binary_paths=(binary_path,),
            registered_binary_paths_channel_2=None,
        )
        original.save(root_path=tmp_path)

        loaded = CombinedData.load(root_path=tmp_path)
        assert loaded.registered_binary_paths_channel_2 is None


class TestOptionalArrayFieldSerialization:
    """Tests _save_optional_array_field and _load_optional_array_field round-trips."""

    def test_round_trip_with_mix_of_none_and_arrays(self, tmp_path: Path) -> None:
        """Verifies that a mix of None and non-None arrays survives a save/load round-trip."""
        arrays: list[NDArray[np.int32] | None] = [
            np.array([1, 2, 3], dtype=np.int32),
            None,
            np.array([10, 20], dtype=np.int32),
        ]

        save_dict: dict[str, np.ndarray] = {}
        _save_optional_array_field(field_name="test_field", arrays=arrays, save_dictionary=save_dict, dtype=np.int32)

        # Saves to npz to simulate the full workflow.
        file_path = tmp_path / "optional_test.npz"
        np.savez(file_path, allow_pickle=False, **save_dict)
        data = np.load(file_path, allow_pickle=False)

        result = _load_optional_array_field(field_name="test_field", item_count=3, data=data, dtype=np.int32)

        assert result[0] is not None
        np.testing.assert_array_equal(result[0], np.array([1, 2, 3], dtype=np.int32))
        assert result[1] is None
        assert result[2] is not None
        np.testing.assert_array_equal(result[2], np.array([10, 20], dtype=np.int32))

    def test_all_none_arrays_produce_no_keys(self) -> None:
        """Verifies that all-None arrays produce no keys in the save dictionary."""
        arrays: list[None] = [None, None, None]
        save_dict: dict[str, np.ndarray] = {}
        _save_optional_array_field(field_name="empty_field", arrays=arrays, save_dictionary=save_dict, dtype=np.bool_)

        assert "empty_field" not in save_dict
        assert "empty_field_counts" not in save_dict

    def test_load_returns_all_none_when_key_missing(self, tmp_path: Path) -> None:
        """Verifies that loading a missing field returns a list of None values."""
        file_path = tmp_path / "no_field.npz"
        np.savez(file_path, allow_pickle=False, dummy=np.array([1]))
        data = np.load(file_path, allow_pickle=False)

        result = _load_optional_array_field(field_name="missing", item_count=5, data=data, dtype=np.float32)

        assert len(result) == 5
        assert all(item is None for item in result)

    def test_round_trip_with_bool_arrays(self, tmp_path: Path) -> None:
        """Verifies that boolean optional arrays survive a save/load round-trip."""
        arrays: list[NDArray[np.bool_] | None] = [
            np.array([True, False, True], dtype=np.bool_),
            np.array([False, False], dtype=np.bool_),
            None,
        ]

        save_dict: dict[str, np.ndarray] = {}
        _save_optional_array_field(field_name="bool_field", arrays=arrays, save_dictionary=save_dict, dtype=np.bool_)

        file_path = tmp_path / "bool_test.npz"
        np.savez(file_path, allow_pickle=False, **save_dict)
        data = np.load(file_path, allow_pickle=False)

        result = _load_optional_array_field(field_name="bool_field", item_count=3, data=data, dtype=np.bool_)

        assert result[0] is not None
        np.testing.assert_array_equal(result[0], np.array([True, False, True], dtype=np.bool_))
        assert result[1] is not None
        np.testing.assert_array_equal(result[1], np.array([False, False], dtype=np.bool_))
        assert result[2] is None
