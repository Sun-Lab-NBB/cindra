"""Contains tests for the dataclass state management methods in single_recording_data and multi_recording_data."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from cindra.dataclasses.multi_recording_data import MultiRecordingTrackingData, MultiRecordingRegistrationData
from cindra.dataclasses.single_recording_data import (
    ROIMask,
    DetectionData,
    ExtractionData,
    RegistrationData,
    is_memory_mapped,
)
from cindra.dataclasses.multi_recording_configuration import RecordingIO
from cindra.dataclasses.single_recording_configuration import AcquisitionParameters


class TestIsMemoryMapped:
    """Tests for the is_memory_mapped() function."""

    def test_returns_false_for_regular_array(self) -> None:
        """Verifies that a standard numpy array is not identified as memory-mapped."""
        array = np.zeros(10, dtype=np.float32)
        assert not is_memory_mapped(array=array)

    def test_returns_true_for_memory_mapped_array(self) -> None:
        """Verifies that a numpy memmap array is correctly identified as memory-mapped."""
        with tempfile.NamedTemporaryFile(suffix=".dat") as temporary_file:
            memory_mapped_array = np.memmap(
                temporary_file.name, dtype=np.float32, mode="w+", shape=(10,)
            )
            assert is_memory_mapped(array=memory_mapped_array)

    def test_returns_false_for_none(self) -> None:
        """Verifies that None input returns False."""
        assert not is_memory_mapped(array=None)


class TestRegistrationDataIsRegistered:
    """Tests for RegistrationData.is_registered()."""

    def test_returns_false_for_default_instance(self) -> None:
        """Verifies that a default RegistrationData instance is not registered."""
        data = RegistrationData()
        assert not data.is_registered()

    def test_returns_true_when_arrays_present(self) -> None:
        """Verifies that registration is detected when all required arrays are set."""
        data = RegistrationData()
        data.reference_image = np.zeros((64, 64), dtype=np.float32)
        data.rigid_y_offsets = np.zeros(10, dtype=np.int32)
        data.rigid_x_offsets = np.zeros(10, dtype=np.int32)
        assert data.is_registered()

    def test_returns_false_when_only_reference_image_present(self) -> None:
        """Verifies that partial arrays do not indicate registration."""
        data = RegistrationData()
        data.reference_image = np.zeros((64, 64), dtype=np.float32)
        assert not data.is_registered()

    def test_returns_true_when_has_registration_data_flag_set(self) -> None:
        """Verifies that the has_registration_data flag alone indicates registration even without arrays."""
        data = RegistrationData()
        data.has_registration_data = True
        assert data.is_registered()


class TestRegistrationDataClear:
    """Tests for RegistrationData.clear()."""

    def test_resets_all_fields_to_defaults(self) -> None:
        """Verifies that clear() resets all fields including the registration flag and scalar values."""
        data = RegistrationData()
        data.has_registration_data = True
        data.valid_y_range = (10, 200)
        data.valid_x_range = (5, 150)
        data.bad_frames = np.ones(10, dtype=np.bool_)
        data.bidirectional_phase_offset = 3
        data.bidirectional_phase_corrected = True
        data.normalization_minimum = 100
        data.normalization_maximum = 4000
        data.reference_image = np.zeros((64, 64), dtype=np.float32)
        data.rigid_y_offsets = np.zeros(10, dtype=np.int32)
        data.rigid_x_offsets = np.zeros(10, dtype=np.int32)
        data.rigid_correlations = np.zeros(10, dtype=np.float32)
        data.nonrigid_y_offsets = np.zeros((10, 4), dtype=np.float32)
        data.nonrigid_x_offsets = np.zeros((10, 4), dtype=np.float32)
        data.nonrigid_correlations = np.zeros((10, 4), dtype=np.float32)
        data.principal_component_extreme_images = np.zeros((2, 3, 64, 64), dtype=np.float32)
        data.principal_component_projections = np.zeros((10, 3), dtype=np.float32)
        data.principal_component_shift_metrics = np.zeros((3, 3), dtype=np.float32)

        data.clear()

        assert not data.has_registration_data
        assert data.valid_y_range == (0, 0)
        assert data.valid_x_range == (0, 0)
        assert data.bad_frames is None
        assert data.bidirectional_phase_offset == 0
        assert not data.bidirectional_phase_corrected
        assert data.normalization_minimum == 0
        assert data.normalization_maximum == 0
        assert data.reference_image is None
        assert data.rigid_y_offsets is None
        assert data.rigid_x_offsets is None
        assert data.rigid_correlations is None
        assert data.nonrigid_y_offsets is None
        assert data.nonrigid_x_offsets is None
        assert data.nonrigid_correlations is None
        assert data.principal_component_extreme_images is None
        assert data.principal_component_projections is None
        assert data.principal_component_shift_metrics is None

    def test_instance_not_registered_after_clear(self) -> None:
        """Verifies that is_registered() returns False after clearing."""
        data = RegistrationData()
        data.has_registration_data = True
        data.reference_image = np.zeros((64, 64), dtype=np.float32)
        data.rigid_y_offsets = np.zeros(10, dtype=np.int32)
        data.rigid_x_offsets = np.zeros(10, dtype=np.int32)

        data.clear()

        assert not data.is_registered()


class TestRegistrationDataPrepareForSaving:
    """Tests for RegistrationData.prepare_for_saving()."""

    def test_sets_all_array_fields_to_none(self) -> None:
        """Verifies that prepare_for_saving() nullifies all array fields."""
        data = RegistrationData()
        data.bad_frames = np.ones(10, dtype=np.bool_)
        data.reference_image = np.zeros((64, 64), dtype=np.float32)
        data.rigid_y_offsets = np.zeros(10, dtype=np.int32)
        data.rigid_x_offsets = np.zeros(10, dtype=np.int32)
        data.rigid_correlations = np.zeros(10, dtype=np.float32)
        data.nonrigid_y_offsets = np.zeros((10, 4), dtype=np.float32)
        data.nonrigid_x_offsets = np.zeros((10, 4), dtype=np.float32)
        data.nonrigid_correlations = np.zeros((10, 4), dtype=np.float32)
        data.principal_component_extreme_images = np.zeros((2, 3, 64, 64), dtype=np.float32)
        data.principal_component_projections = np.zeros((10, 3), dtype=np.float32)
        data.principal_component_shift_metrics = np.zeros((3, 3), dtype=np.float32)

        data.prepare_for_saving()

        assert data.bad_frames is None
        assert data.reference_image is None
        assert data.rigid_y_offsets is None
        assert data.rigid_x_offsets is None
        assert data.rigid_correlations is None
        assert data.nonrigid_y_offsets is None
        assert data.nonrigid_x_offsets is None
        assert data.nonrigid_correlations is None
        assert data.principal_component_extreme_images is None
        assert data.principal_component_projections is None
        assert data.principal_component_shift_metrics is None

    def test_preserves_scalar_fields(self) -> None:
        """Verifies that prepare_for_saving() does not modify scalar fields."""
        data = RegistrationData()
        data.has_registration_data = True
        data.valid_y_range = (10, 200)
        data.valid_x_range = (5, 150)
        data.bidirectional_phase_offset = 3
        data.normalization_minimum = 100
        data.normalization_maximum = 4000

        data.prepare_for_saving()

        assert data.has_registration_data
        assert data.valid_y_range == (10, 200)
        assert data.valid_x_range == (5, 150)
        assert data.bidirectional_phase_offset == 3
        assert data.normalization_minimum == 100
        assert data.normalization_maximum == 4000


class TestRegistrationDataReleaseArrays:
    """Tests for RegistrationData.release_arrays()."""

    def test_sets_all_array_fields_to_none(self) -> None:
        """Verifies that release_arrays() nullifies all array fields."""
        data = RegistrationData()
        data.bad_frames = np.ones(10, dtype=np.bool_)
        data.reference_image = np.zeros((64, 64), dtype=np.float32)
        data.rigid_y_offsets = np.zeros(10, dtype=np.int32)
        data.rigid_x_offsets = np.zeros(10, dtype=np.int32)
        data.rigid_correlations = np.zeros(10, dtype=np.float32)
        data.nonrigid_y_offsets = np.zeros((10, 4), dtype=np.float32)
        data.nonrigid_x_offsets = np.zeros((10, 4), dtype=np.float32)
        data.nonrigid_correlations = np.zeros((10, 4), dtype=np.float32)
        data.principal_component_extreme_images = np.zeros((2, 3, 64, 64), dtype=np.float32)
        data.principal_component_projections = np.zeros((10, 3), dtype=np.float32)
        data.principal_component_shift_metrics = np.zeros((3, 3), dtype=np.float32)

        data.release_arrays()

        assert data.bad_frames is None
        assert data.reference_image is None
        assert data.rigid_y_offsets is None
        assert data.rigid_x_offsets is None
        assert data.rigid_correlations is None
        assert data.nonrigid_y_offsets is None
        assert data.nonrigid_x_offsets is None
        assert data.nonrigid_correlations is None
        assert data.principal_component_extreme_images is None
        assert data.principal_component_projections is None
        assert data.principal_component_shift_metrics is None

    def test_preserves_has_registration_data_flag(self) -> None:
        """Verifies that release_arrays() preserves the has_registration_data flag."""
        data = RegistrationData()
        data.has_registration_data = True
        data.reference_image = np.zeros((64, 64), dtype=np.float32)
        data.rigid_y_offsets = np.zeros(10, dtype=np.int32)
        data.rigid_x_offsets = np.zeros(10, dtype=np.int32)

        data.release_arrays()

        assert data.has_registration_data

    def test_is_registered_remains_true_after_release(self) -> None:
        """Verifies that is_registered() returns True after release when has_registration_data is set."""
        data = RegistrationData()
        data.has_registration_data = True
        data.reference_image = np.zeros((64, 64), dtype=np.float32)
        data.rigid_y_offsets = np.zeros(10, dtype=np.int32)
        data.rigid_x_offsets = np.zeros(10, dtype=np.int32)

        data.release_arrays()

        assert data.is_registered()

    def test_preserves_scalar_fields(self) -> None:
        """Verifies that release_arrays() preserves all scalar fields."""
        data = RegistrationData()
        data.valid_y_range = (10, 200)
        data.valid_x_range = (5, 150)
        data.bidirectional_phase_offset = 3
        data.bidirectional_phase_corrected = True
        data.normalization_minimum = 100
        data.normalization_maximum = 4000

        data.release_arrays()

        assert data.valid_y_range == (10, 200)
        assert data.valid_x_range == (5, 150)
        assert data.bidirectional_phase_offset == 3
        assert data.bidirectional_phase_corrected
        assert data.normalization_minimum == 100
        assert data.normalization_maximum == 4000


class TestDetectionDataPrepareForSaving:
    """Tests for DetectionData.prepare_for_saving()."""

    def test_sets_all_array_fields_to_none(self) -> None:
        """Verifies that prepare_for_saving() nullifies all array fields for both channels."""
        data = DetectionData()
        data.mean_image = np.zeros((64, 64), dtype=np.float32)
        data.enhanced_mean_image = np.zeros((64, 64), dtype=np.float32)
        data.maximum_projection = np.zeros((64, 64), dtype=np.float32)
        data.correlation_map = np.zeros((64, 64), dtype=np.float32)
        data.mean_image_channel_2 = np.zeros((64, 64), dtype=np.float32)
        data.enhanced_mean_image_channel_2 = np.zeros((64, 64), dtype=np.float32)
        data.maximum_projection_channel_2 = np.zeros((64, 64), dtype=np.float32)
        data.correlation_map_channel_2 = np.zeros((64, 64), dtype=np.float32)

        data.prepare_for_saving()

        assert data.mean_image is None
        assert data.enhanced_mean_image is None
        assert data.maximum_projection is None
        assert data.correlation_map is None
        assert data.mean_image_channel_2 is None
        assert data.enhanced_mean_image_channel_2 is None
        assert data.maximum_projection_channel_2 is None
        assert data.correlation_map_channel_2 is None

    def test_preserves_scalar_fields(self) -> None:
        """Verifies that prepare_for_saving() does not modify scalar fields."""
        data = DetectionData()
        data.roi_diameter = 12
        data.aspect_ratio = 1.5
        data.roi_diameter_channel_2 = 10

        data.prepare_for_saving()

        assert data.roi_diameter == 12
        assert data.aspect_ratio == 1.5
        assert data.roi_diameter_channel_2 == 10


class TestDetectionDataReleaseArrays:
    """Tests for DetectionData.release_arrays()."""

    def test_sets_all_array_fields_to_none(self) -> None:
        """Verifies that release_arrays() nullifies all array fields for both channels."""
        data = DetectionData()
        data.mean_image = np.zeros((64, 64), dtype=np.float32)
        data.enhanced_mean_image = np.zeros((64, 64), dtype=np.float32)
        data.maximum_projection = np.zeros((64, 64), dtype=np.float32)
        data.correlation_map = np.zeros((64, 64), dtype=np.float32)
        data.mean_image_channel_2 = np.zeros((64, 64), dtype=np.float32)
        data.enhanced_mean_image_channel_2 = np.zeros((64, 64), dtype=np.float32)
        data.maximum_projection_channel_2 = np.zeros((64, 64), dtype=np.float32)
        data.correlation_map_channel_2 = np.zeros((64, 64), dtype=np.float32)

        data.release_arrays()

        assert data.mean_image is None
        assert data.enhanced_mean_image is None
        assert data.maximum_projection is None
        assert data.correlation_map is None
        assert data.mean_image_channel_2 is None
        assert data.enhanced_mean_image_channel_2 is None
        assert data.maximum_projection_channel_2 is None
        assert data.correlation_map_channel_2 is None

    def test_preserves_scalar_fields(self) -> None:
        """Verifies that release_arrays() does not modify scalar fields."""
        data = DetectionData()
        data.roi_diameter = 12
        data.aspect_ratio = 1.5
        data.roi_diameter_channel_2 = 10

        data.release_arrays()

        assert data.roi_diameter == 12
        assert data.aspect_ratio == 1.5
        assert data.roi_diameter_channel_2 == 10


class TestROIMaskRaveledPixels:
    """Tests for ROIMask.raveled_pixels cached property."""

    def test_computes_correct_raveled_indices(self) -> None:
        """Verifies that raveled pixel indices are computed as y * frame_width + x."""
        mask = ROIMask(
            y_pixels=np.array([0, 1, 2], dtype=np.int32),
            x_pixels=np.array([3, 4, 5], dtype=np.int32),
            pixel_weights=np.ones(3, dtype=np.float32),
            centroid=(1, 4),
            frame_width=10,
        )
        expected = np.array([3, 14, 25], dtype=np.int32)
        np.testing.assert_array_equal(mask.raveled_pixels, expected)

    def test_output_dtype_is_int32(self) -> None:
        """Verifies that the raveled pixel array has int32 dtype."""
        mask = ROIMask(
            y_pixels=np.array([0, 1], dtype=np.int32),
            x_pixels=np.array([0, 1], dtype=np.int32),
            pixel_weights=np.ones(2, dtype=np.float32),
            centroid=(0, 0),
            frame_width=100,
        )
        assert mask.raveled_pixels.dtype == np.int32

    def test_cached_property_returns_same_object(self) -> None:
        """Verifies that repeated access returns the same cached array object."""
        mask = ROIMask(
            y_pixels=np.array([5], dtype=np.int32),
            x_pixels=np.array([3], dtype=np.int32),
            pixel_weights=np.ones(1, dtype=np.float32),
            centroid=(5, 3),
            frame_width=20,
        )
        first_access = mask.raveled_pixels
        second_access = mask.raveled_pixels
        assert first_access is second_access


class TestROIMaskCirclePixels:
    """Tests for ROIMask.circle_pixels cached property."""

    def test_returns_tuple_of_two_arrays(self) -> None:
        """Verifies that circle_pixels returns a tuple of (y_circle, x_circle) arrays."""
        mask = ROIMask(
            y_pixels=np.array([10], dtype=np.int32),
            x_pixels=np.array([10], dtype=np.int32),
            pixel_weights=np.ones(1, dtype=np.float32),
            centroid=(10, 10),
            frame_width=100,
            radius=5.0,
        )
        result = mask.circle_pixels
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_output_arrays_have_100_points(self) -> None:
        """Verifies that each circle coordinate array has exactly 100 sample points."""
        mask = ROIMask(
            y_pixels=np.array([10], dtype=np.int32),
            x_pixels=np.array([10], dtype=np.int32),
            pixel_weights=np.ones(1, dtype=np.float32),
            centroid=(10, 10),
            frame_width=100,
            radius=5.0,
        )
        y_circle, x_circle = mask.circle_pixels
        assert y_circle.shape == (100,)
        assert x_circle.shape == (100,)

    def test_output_dtype_is_int32(self) -> None:
        """Verifies that circle pixel coordinate arrays have int32 dtype."""
        mask = ROIMask(
            y_pixels=np.array([10], dtype=np.int32),
            x_pixels=np.array([10], dtype=np.int32),
            pixel_weights=np.ones(1, dtype=np.float32),
            centroid=(10, 10),
            frame_width=100,
            radius=5.0,
        )
        y_circle, x_circle = mask.circle_pixels
        assert y_circle.dtype == np.int32
        assert x_circle.dtype == np.int32

    def test_circle_centered_around_centroid(self) -> None:
        """Verifies that the circle coordinates are approximately centered around the ROI centroid."""
        centroid_y, centroid_x = 50, 60
        mask = ROIMask(
            y_pixels=np.array([50], dtype=np.int32),
            x_pixels=np.array([60], dtype=np.int32),
            pixel_weights=np.ones(1, dtype=np.float32),
            centroid=(centroid_y, centroid_x),
            frame_width=200,
            radius=10.0,
        )
        y_circle, x_circle = mask.circle_pixels
        # The mean of points on a circle should be near the center.
        assert abs(float(np.mean(y_circle)) - centroid_y) < 2
        assert abs(float(np.mean(x_circle)) - centroid_x) < 2

    def test_uses_scaled_radius(self) -> None:
        """Verifies that the circle uses 1.25 times the ROI radius."""
        radius = 10.0
        scaled_radius = radius * 1.25
        centroid_y, centroid_x = 100, 100
        mask = ROIMask(
            y_pixels=np.array([100], dtype=np.int32),
            x_pixels=np.array([100], dtype=np.int32),
            pixel_weights=np.ones(1, dtype=np.float32),
            centroid=(centroid_y, centroid_x),
            frame_width=200,
            radius=radius,
        )
        y_circle, x_circle = mask.circle_pixels
        # Checks that the maximum distance from centroid is approximately the scaled radius.
        delta_y = y_circle.astype(np.float64) - centroid_y
        delta_x = x_circle.astype(np.float64) - centroid_x
        distances = np.sqrt(delta_y**2 + delta_x**2)
        np.testing.assert_allclose(np.max(distances), scaled_radius, atol=1.5)

    def test_cached_property_returns_same_object(self) -> None:
        """Verifies that repeated access returns the same cached tuple object."""
        mask = ROIMask(
            y_pixels=np.array([10], dtype=np.int32),
            x_pixels=np.array([10], dtype=np.int32),
            pixel_weights=np.ones(1, dtype=np.float32),
            centroid=(10, 10),
            frame_width=100,
            radius=5.0,
        )
        first_access = mask.circle_pixels
        second_access = mask.circle_pixels
        assert first_access is second_access


class TestExtractionDataPrepareForSaving:
    """Tests for ExtractionData.prepare_for_saving()."""

    def test_sets_all_fields_to_none(self) -> None:
        """Verifies that prepare_for_saving() nullifies all array and list fields across both channels."""
        data = ExtractionData()
        # Channel 1.
        data.roi_statistics = []
        data.cell_fluorescence = np.zeros((5, 100), dtype=np.float32)
        data.neuropil_fluorescence = np.zeros((5, 100), dtype=np.float32)
        data.subtracted_fluorescence = np.zeros((5, 100), dtype=np.float32)
        data.spikes = np.zeros((5, 100), dtype=np.float32)
        data.cell_classification = np.zeros((5, 2), dtype=np.float32)
        # Channel 2.
        data.roi_statistics_channel_2 = []
        data.cell_fluorescence_channel_2 = np.zeros((5, 100), dtype=np.float32)
        data.neuropil_fluorescence_channel_2 = np.zeros((5, 100), dtype=np.float32)
        data.subtracted_fluorescence_channel_2 = np.zeros((5, 100), dtype=np.float32)
        data.spikes_channel_2 = np.zeros((5, 100), dtype=np.float32)
        data.cell_classification_channel_2 = np.zeros((5, 2), dtype=np.float32)
        # Colocalization.
        data.cell_colocalization = np.zeros((5, 2), dtype=np.float32)
        data.corrected_structural_mean_image = np.zeros((64, 64), dtype=np.float32)

        data.prepare_for_saving()

        # Channel 1.
        assert data.roi_statistics is None
        assert data.cell_fluorescence is None
        assert data.neuropil_fluorescence is None
        assert data.subtracted_fluorescence is None
        assert data.spikes is None
        assert data.cell_classification is None
        # Channel 2.
        assert data.roi_statistics_channel_2 is None
        assert data.cell_fluorescence_channel_2 is None
        assert data.neuropil_fluorescence_channel_2 is None
        assert data.subtracted_fluorescence_channel_2 is None
        assert data.spikes_channel_2 is None
        assert data.cell_classification_channel_2 is None
        # Colocalization.
        assert data.cell_colocalization is None
        assert data.corrected_structural_mean_image is None


class TestExtractionDataReleaseArrays:
    """Tests for ExtractionData.release_arrays()."""

    def test_sets_all_fields_to_none(self) -> None:
        """Verifies that release_arrays() nullifies all array and list fields across both channels."""
        data = ExtractionData()
        # Channel 1.
        data.roi_statistics = []
        data.cell_fluorescence = np.zeros((5, 100), dtype=np.float32)
        data.neuropil_fluorescence = np.zeros((5, 100), dtype=np.float32)
        data.subtracted_fluorescence = np.zeros((5, 100), dtype=np.float32)
        data.spikes = np.zeros((5, 100), dtype=np.float32)
        data.cell_classification = np.zeros((5, 2), dtype=np.float32)
        # Channel 2.
        data.roi_statistics_channel_2 = []
        data.cell_fluorescence_channel_2 = np.zeros((5, 100), dtype=np.float32)
        data.neuropil_fluorescence_channel_2 = np.zeros((5, 100), dtype=np.float32)
        data.subtracted_fluorescence_channel_2 = np.zeros((5, 100), dtype=np.float32)
        data.spikes_channel_2 = np.zeros((5, 100), dtype=np.float32)
        data.cell_classification_channel_2 = np.zeros((5, 2), dtype=np.float32)
        # Colocalization.
        data.cell_colocalization = np.zeros((5, 2), dtype=np.float32)
        data.corrected_structural_mean_image = np.zeros((64, 64), dtype=np.float32)

        data.release_arrays()

        # Channel 1.
        assert data.roi_statistics is None
        assert data.cell_fluorescence is None
        assert data.neuropil_fluorescence is None
        assert data.subtracted_fluorescence is None
        assert data.spikes is None
        assert data.cell_classification is None
        # Channel 2.
        assert data.roi_statistics_channel_2 is None
        assert data.cell_fluorescence_channel_2 is None
        assert data.neuropil_fluorescence_channel_2 is None
        assert data.subtracted_fluorescence_channel_2 is None
        assert data.spikes_channel_2 is None
        assert data.cell_classification_channel_2 is None
        # Colocalization.
        assert data.cell_colocalization is None
        assert data.corrected_structural_mean_image is None


class TestMultiRecordingRegistrationDataIsRegistered:
    """Tests for MultiRecordingRegistrationData.is_registered()."""

    def test_returns_false_for_default_instance(self) -> None:
        """Verifies that a default MultiRecordingRegistrationData instance is not registered."""
        data = MultiRecordingRegistrationData()
        assert not data.is_registered()

    def test_returns_true_when_arrays_present(self) -> None:
        """Verifies that registration is detected when deformation fields and ROI masks are set."""
        data = MultiRecordingRegistrationData()
        data.deform_field_y = np.zeros((64, 64), dtype=np.float32)
        data.deformed_roi_masks = []
        assert data.is_registered()

    def test_returns_false_when_only_deform_field_present(self) -> None:
        """Verifies that a single deformation field alone does not indicate registration."""
        data = MultiRecordingRegistrationData()
        data.deform_field_y = np.zeros((64, 64), dtype=np.float32)
        assert not data.is_registered()

    def test_returns_true_when_has_registration_data_flag_set(self) -> None:
        """Verifies that the has_registration_data flag alone indicates registration even without arrays."""
        data = MultiRecordingRegistrationData()
        data.has_registration_data = True
        assert data.is_registered()


class TestMultiRecordingRegistrationDataClear:
    """Tests for MultiRecordingRegistrationData.clear()."""

    def test_resets_flag_and_releases_arrays(self) -> None:
        """Verifies that clear() resets the registration flag and nullifies all array fields."""
        data = MultiRecordingRegistrationData()
        data.has_registration_data = True
        data.deform_field_y = np.zeros((64, 64), dtype=np.float32)
        data.deform_field_x = np.zeros((64, 64), dtype=np.float32)
        data.transformed_mean_image = np.zeros((64, 64), dtype=np.float32)
        data.deformed_roi_masks = []

        data.clear()

        assert not data.has_registration_data
        assert data.deform_field_y is None
        assert data.deform_field_x is None
        assert data.transformed_mean_image is None
        assert data.deformed_roi_masks is None

    def test_not_registered_after_clear(self) -> None:
        """Verifies that is_registered() returns False after clearing."""
        data = MultiRecordingRegistrationData()
        data.has_registration_data = True

        data.clear()

        assert not data.is_registered()


class TestMultiRecordingRegistrationDataPrepareForSaving:
    """Tests for MultiRecordingRegistrationData.prepare_for_saving()."""

    def test_delegates_to_release_arrays(self) -> None:
        """Verifies that prepare_for_saving() nullifies all array fields identically to release_arrays()."""
        data = MultiRecordingRegistrationData()
        data.deform_field_y = np.zeros((64, 64), dtype=np.float32)
        data.deform_field_x = np.zeros((64, 64), dtype=np.float32)
        data.transformed_mean_image = np.zeros((64, 64), dtype=np.float32)
        data.transformed_enhanced_mean_image = np.zeros((64, 64), dtype=np.float32)
        data.transformed_maximum_projection = np.zeros((64, 64), dtype=np.float32)
        data.transformed_mean_image_channel_2 = np.zeros((64, 64), dtype=np.float32)
        data.transformed_enhanced_mean_image_channel_2 = np.zeros((64, 64), dtype=np.float32)
        data.transformed_maximum_projection_channel_2 = np.zeros((64, 64), dtype=np.float32)
        data.deformed_roi_masks = []
        data.deformed_roi_masks_channel_2 = []

        data.prepare_for_saving()

        assert data.deform_field_y is None
        assert data.deform_field_x is None
        assert data.transformed_mean_image is None
        assert data.transformed_enhanced_mean_image is None
        assert data.transformed_maximum_projection is None
        assert data.transformed_mean_image_channel_2 is None
        assert data.transformed_enhanced_mean_image_channel_2 is None
        assert data.transformed_maximum_projection_channel_2 is None
        assert data.deformed_roi_masks is None
        assert data.deformed_roi_masks_channel_2 is None

    def test_preserves_has_registration_data_flag(self) -> None:
        """Verifies that prepare_for_saving() preserves the has_registration_data flag."""
        data = MultiRecordingRegistrationData()
        data.has_registration_data = True
        data.deform_field_y = np.zeros((64, 64), dtype=np.float32)

        data.prepare_for_saving()

        assert data.has_registration_data


class TestMultiRecordingRegistrationDataReleaseArrays:
    """Tests for MultiRecordingRegistrationData.release_arrays()."""

    def test_sets_all_array_fields_to_none(self) -> None:
        """Verifies that release_arrays() nullifies all array and list fields."""
        data = MultiRecordingRegistrationData()
        data.deform_field_y = np.zeros((64, 64), dtype=np.float32)
        data.deform_field_x = np.zeros((64, 64), dtype=np.float32)
        data.transformed_mean_image = np.zeros((64, 64), dtype=np.float32)
        data.transformed_enhanced_mean_image = np.zeros((64, 64), dtype=np.float32)
        data.transformed_maximum_projection = np.zeros((64, 64), dtype=np.float32)
        data.transformed_mean_image_channel_2 = np.zeros((64, 64), dtype=np.float32)
        data.transformed_enhanced_mean_image_channel_2 = np.zeros((64, 64), dtype=np.float32)
        data.transformed_maximum_projection_channel_2 = np.zeros((64, 64), dtype=np.float32)
        data.deformed_roi_masks = []
        data.deformed_roi_masks_channel_2 = []

        data.release_arrays()

        assert data.deform_field_y is None
        assert data.deform_field_x is None
        assert data.transformed_mean_image is None
        assert data.transformed_enhanced_mean_image is None
        assert data.transformed_maximum_projection is None
        assert data.transformed_mean_image_channel_2 is None
        assert data.transformed_enhanced_mean_image_channel_2 is None
        assert data.transformed_maximum_projection_channel_2 is None
        assert data.deformed_roi_masks is None
        assert data.deformed_roi_masks_channel_2 is None

    def test_preserves_has_registration_data_flag(self) -> None:
        """Verifies that release_arrays() preserves the has_registration_data flag."""
        data = MultiRecordingRegistrationData()
        data.has_registration_data = True
        data.deform_field_y = np.zeros((64, 64), dtype=np.float32)
        data.deformed_roi_masks = []

        data.release_arrays()

        assert data.has_registration_data

    def test_is_registered_remains_true_after_release(self) -> None:
        """Verifies that is_registered() returns True after release when has_registration_data is set."""
        data = MultiRecordingRegistrationData()
        data.has_registration_data = True
        data.deform_field_y = np.zeros((64, 64), dtype=np.float32)
        data.deformed_roi_masks = []

        data.release_arrays()

        assert data.is_registered()


class TestMultiRecordingTrackingDataPrepareForSaving:
    """Tests for MultiRecordingTrackingData.prepare_for_saving() and release_arrays()."""

    def test_prepare_for_saving_nullifies_template_masks(self) -> None:
        """Verifies that prepare_for_saving() sets template mask fields to None."""
        data = MultiRecordingTrackingData()
        data.template_masks = []
        data.template_masks_channel_2 = []

        data.prepare_for_saving()

        assert data.template_masks is None
        assert data.template_masks_channel_2 is None

    def test_release_arrays_nullifies_template_masks(self) -> None:
        """Verifies that release_arrays() sets template mask fields to None."""
        data = MultiRecordingTrackingData()
        data.template_masks = []
        data.template_masks_channel_2 = []

        data.release_arrays()

        assert data.template_masks is None
        assert data.template_masks_channel_2 is None

    def test_release_preserves_scalar_fields(self) -> None:
        """Verifies that release_arrays() does not affect scalar fields like template_diameter."""
        data = MultiRecordingTrackingData()
        data.template_diameter = 15
        data.template_masks = []

        data.release_arrays()

        assert data.template_diameter == 15


class TestRecordingIOPostInit:
    """Tests for RecordingIO.__post_init__() natural sorting."""

    def test_natural_sorts_recording_directories(self) -> None:
        """Verifies that __post_init__ natural-sorts recording directories."""
        paths = (Path("/data/recording_10"), Path("/data/recording_2"), Path("/data/recording_1"))
        recording_io = RecordingIO(recording_directories=paths)

        assert recording_io.recording_directories == (
            Path("/data/recording_1"),
            Path("/data/recording_2"),
            Path("/data/recording_10"),
        )


class TestAcquisitionParametersProperties:
    """Tests for AcquisitionParameters computed properties."""

    def test_is_mroi_returns_false_for_single_roi(self) -> None:
        """Verifies that is_mroi returns False when roi_number is 1."""
        parameters = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1)
        assert not parameters.is_mroi

    def test_is_mroi_returns_true_for_multiple_rois(self) -> None:
        """Verifies that is_mroi returns True when roi_number exceeds 1."""
        parameters = AcquisitionParameters(frame_rate=30.0, plane_number=1, channel_number=1, roi_number=3)
        assert parameters.is_mroi

    def test_virtual_plane_count_single_roi(self) -> None:
        """Verifies that virtual_plane_count equals plane_number when roi_number is 1."""
        parameters = AcquisitionParameters(frame_rate=30.0, plane_number=4, channel_number=1)
        assert parameters.virtual_plane_count == 4

    def test_virtual_plane_count_multiple_rois(self) -> None:
        """Verifies that virtual_plane_count is roi_number * plane_number."""
        parameters = AcquisitionParameters(frame_rate=30.0, plane_number=2, channel_number=1, roi_number=3)
        assert parameters.virtual_plane_count == 6
