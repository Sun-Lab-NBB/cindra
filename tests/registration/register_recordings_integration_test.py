"""Contains integration tests for the cross-recording registration stage entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING
from importlib import import_module

import numpy as np
import pytest
from ataraxis_base_utilities import error_format, ensure_directory_exists

from cindra.dataclasses import (
    ROIMask,
    CombinedData,
    DetectionData,
    ROIStatistics,
    ExtractionData,
    ReferenceImageType,
    MultiRecordingRuntimeData,
    MultiRecordingConfiguration,
    MultiRecordingRuntimeContext,
)
from cindra.registration.deformation import Deformation
from cindra.registration.diffeomorphic import DiffeomorphicDemonsRegistration
from cindra.registration.register_recordings import (
    register_recordings,
    _apply_forward_deformation,
    _apply_backward_deformation,
    project_templates_to_recordings,
)

if TYPE_CHECKING:
    from pathlib import Path
    from collections.abc import Callable

    from numpy.typing import NDArray

_FRAME_SIZE: int = 64
"""The synthetic combined field-of-view dimension in pixels used for all registration contexts."""

_BASE_CENTERS: tuple[tuple[int, int], ...] = ((18, 18), (40, 22), (24, 44), (46, 46))
"""The blob centroids for the reference recording, also reused as channel 1 ROI mask centroids."""


class TestRegisterRecordings:
    """Tests register_recordings."""

    def test_serial_path_writes_deformation_outputs(
        self, gaussian_blob_image: Callable[..., NDArray[np.float64]], tmp_path: Path
    ) -> None:
        """Verifies that serial registration writes deformation fields, transformed images, and deformed masks."""
        configuration = _make_configuration()
        contexts = _build_recording_pair(tmp_path=tmp_path, builder=gaussian_blob_image, configuration=configuration)

        register_recordings(contexts=contexts, workers=1)

        for context in contexts:
            output_path = context.runtime.output_path
            assert output_path is not None
            assert (output_path / "registration_arrays" / "deform_field_y.npy").exists()
            assert (output_path / "registration_arrays" / "deform_field_x.npy").exists()
            assert (output_path / "registration_arrays" / "transformed_enhanced_mean_image.npy").exists()
            assert (output_path / "registration_deformed_masks.npz").exists()
            assert (output_path / "multi_recording_runtime_data.yaml").exists()
            assert context.runtime.timing.registration_time >= 0

            field_y, field_x = _read_deform_fields(context=context)
            assert field_y.shape == (_FRAME_SIZE, _FRAME_SIZE)
            assert field_x.shape == (_FRAME_SIZE, _FRAME_SIZE)
            assert np.all(np.isfinite(field_y))
            assert np.all(np.isfinite(field_x))

            deformed_masks = ROIMask.load_list(file_path=output_path / "registration_deformed_masks.npz")
            assert len(deformed_masks) == len(_BASE_CENTERS)

    def test_forwards_configured_registration_parameters(
        self,
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verifies that the configured diffeomorphic registration parameters reach the registration algorithm."""
        configuration = _make_configuration()
        configuration.diffeomorphic_registration.final_grid_sampling = 8.0
        configuration.diffeomorphic_registration.grid_sampling_factor = 0.5
        configuration.diffeomorphic_registration.scale_sampling = 4
        configuration.diffeomorphic_registration.speed_factor = 2.0
        contexts = _build_recording_pair(tmp_path=tmp_path, builder=gaussian_blob_image, configuration=configuration)
        registrations = _capture_registrations(monkeypatch=monkeypatch)

        register_recordings(contexts=contexts, workers=1)

        # Each configured value differs from the algorithm's own default for that parameter, so dropping any one of
        # the forwarded arguments leaves the algorithm holding its default and fails the matching assertion.
        assert len(registrations) == 1
        registration = registrations[0]
        assert registration._final_grid_sampling == 8.0
        assert registration._grid_sampling_factor == 0.5
        assert registration._scale_sampling == 4
        assert registration._speed_factor == 2.0

    def test_identical_images_produce_near_zero_deformation(
        self, gaussian_blob_image: Callable[..., NDArray[np.float64]], tmp_path: Path
    ) -> None:
        """Verifies that registering two identical reference images yields near-zero deformation fields."""
        configuration = _make_configuration()
        contexts = _build_recording_pair(
            tmp_path=tmp_path, builder=gaussian_blob_image, configuration=configuration, shift=0
        )

        register_recordings(contexts=contexts, workers=1)

        for context in contexts:
            field_y, field_x = _read_deform_fields(context=context)
            assert float(np.abs(field_y).max()) < 1.0
            assert float(np.abs(field_x).max()) < 1.0

    def test_maximum_projection_image_type(
        self, gaussian_blob_image: Callable[..., NDArray[np.float64]], tmp_path: Path
    ) -> None:
        """Verifies registration against maximum-projection images while skipping forward mask deformation."""
        configuration = _make_configuration(image_type=ReferenceImageType.MAXIMUM_PROJECTION)
        contexts = _build_recording_pair(
            tmp_path=tmp_path,
            builder=gaussian_blob_image,
            configuration=configuration,
            image_kinds=("maximum_projection",),
            selected_indices=(),
        )

        register_recordings(contexts=contexts, workers=1)

        for context in contexts:
            output_path = context.runtime.output_path
            assert output_path is not None
            assert (output_path / "registration_arrays" / "transformed_maximum_projection.npy").exists()
            # No ROIs were selected, so no deformed mask file is produced.
            assert not (output_path / "registration_deformed_masks.npz").exists()

    def test_two_channel_writes_channel_2_outputs(
        self, gaussian_blob_image: Callable[..., NDArray[np.float64]], tmp_path: Path
    ) -> None:
        """Verifies that a two-channel recording transforms and saves channel 2 images and deformed masks."""
        configuration = _make_configuration()
        contexts = _build_recording_pair(
            tmp_path=tmp_path,
            builder=gaussian_blob_image,
            configuration=configuration,
            image_kinds=("mean", "enhanced_mean", "maximum_projection"),
            two_channel=True,
            selected_indices_channel_2=(0, 1, 2, 3),
            write_channel_2_masks=True,
        )

        register_recordings(contexts=contexts, workers=1)

        for context in contexts:
            output_path = context.runtime.output_path
            assert output_path is not None
            arrays_directory = output_path / "registration_arrays"
            assert (arrays_directory / "transformed_mean_image.npy").exists()
            assert (arrays_directory / "transformed_mean_image_channel_2.npy").exists()
            assert (arrays_directory / "transformed_enhanced_mean_image_channel_2.npy").exists()
            assert (arrays_directory / "transformed_maximum_projection_channel_2.npy").exists()
            assert (output_path / "registration_deformed_masks.npz").exists()
            assert (output_path / "registration_deformed_masks_channel_2.npz").exists()

    def test_missing_mask_files_skip_forward_mask_deformation(
        self, gaussian_blob_image: Callable[..., NDArray[np.float64]], tmp_path: Path
    ) -> None:
        """Verifies that selected ROI indices without on-disk mask files leave deformed masks unwritten."""
        configuration = _make_configuration()
        contexts = _build_recording_pair(
            tmp_path=tmp_path,
            builder=gaussian_blob_image,
            configuration=configuration,
            selected_indices=(0, 1, 2, 3),
            selected_indices_channel_2=(0, 1, 2, 3),
            write_channel_1_masks=False,
            write_channel_2_masks=False,
        )

        register_recordings(contexts=contexts, workers=1)

        for context in contexts:
            output_path = context.runtime.output_path
            assert output_path is not None
            assert not (output_path / "registration_deformed_masks.npz").exists()
            assert not (output_path / "registration_deformed_masks_channel_2.npz").exists()

    def test_skips_when_already_registered(
        self, gaussian_blob_image: Callable[..., NDArray[np.float64]], tmp_path: Path
    ) -> None:
        """Verifies that a second registration call short-circuits when registration data already exists on disk."""
        configuration = _make_configuration()
        contexts = _build_recording_pair(tmp_path=tmp_path, builder=gaussian_blob_image, configuration=configuration)
        register_recordings(contexts=contexts, workers=1)

        # Removing combined data would break a re-run. The skip path must not touch it, so no error proves the skip.
        for context in contexts:
            context.runtime.combined_data = None

        register_recordings(contexts=contexts, workers=1)

        for context in contexts:
            output_path = context.runtime.output_path
            assert output_path is not None
            assert (output_path / "registration_arrays" / "deform_field_y.npy").exists()

    def test_forced_repeat_clears_and_reregisters(
        self, gaussian_blob_image: Callable[..., NDArray[np.float64]], tmp_path: Path
    ) -> None:
        """Verifies that enabling repeat_registration re-runs registration despite existing registration data."""
        configuration = _make_configuration()
        contexts = _build_recording_pair(tmp_path=tmp_path, builder=gaussian_blob_image, configuration=configuration)
        register_recordings(contexts=contexts, workers=1)

        configuration.diffeomorphic_registration.repeat_registration = True
        register_recordings(contexts=contexts, workers=1)

        for context in contexts:
            field_y, field_x = _read_deform_fields(context=context)
            assert field_y.shape == (_FRAME_SIZE, _FRAME_SIZE)
            assert np.all(np.isfinite(field_x))

    def test_missing_combined_data_raises(
        self, gaussian_blob_image: Callable[..., NDArray[np.float64]], tmp_path: Path
    ) -> None:
        """Verifies that a recording without loaded combined data raises a ValueError during registration."""
        configuration = _make_configuration()
        contexts = _build_recording_pair(tmp_path=tmp_path, builder=gaussian_blob_image, configuration=configuration)
        contexts[0].runtime.combined_data = None

        expected_message = (
            "Unable to register recording 'rec0' to shared visual space. The recording's combined_data must be "
            "loaded before registration."
        )
        with pytest.raises(ValueError, match=error_format(expected_message)):
            register_recordings(contexts=contexts, workers=1)

    def test_missing_reference_image_raises(
        self, gaussian_blob_image: Callable[..., NDArray[np.float64]], tmp_path: Path
    ) -> None:
        """Verifies that requesting an unavailable reference image type raises a ValueError."""
        configuration = _make_configuration(image_type=ReferenceImageType.MEAN)
        contexts = _build_recording_pair(
            tmp_path=tmp_path,
            builder=gaussian_blob_image,
            configuration=configuration,
            image_kinds=("enhanced_mean",),
        )

        expected_message = (
            "Unable to register recording 'rec0' to shared visual space. The required reference image "
            f"({ReferenceImageType.MEAN!s}) is not available in combined_data."
        )
        with pytest.raises(ValueError, match=error_format(expected_message)):
            register_recordings(contexts=contexts, workers=1)


class TestRegisterRecordingsParallelPath:
    """Tests the thread-pool branch register_recordings takes when it is allocated more than one worker."""

    def test_matches_the_serial_path_and_keeps_each_deformation_with_its_recording(
        self,
        gaussian_blob_image: Callable[..., NDArray[np.float64]],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Verifies the parallel branch writes the same fields as the serial one, indexed to the same recordings."""
        configuration = _make_configuration()
        registrations = _capture_registrations(monkeypatch=monkeypatch)

        serial_contexts = _build_recording_pair(
            tmp_path=tmp_path / "serial", builder=gaussian_blob_image, configuration=configuration
        )
        parallel_contexts = _build_recording_pair(
            tmp_path=tmp_path / "parallel", builder=gaussian_blob_image, configuration=configuration
        )

        register_recordings(contexts=serial_contexts, workers=1)
        register_recordings(contexts=parallel_contexts, workers=4)

        assert len(registrations) == 2
        serial_registration, parallel_registration = registrations

        for index in range(2):
            serial_field_y, serial_field_x = _read_deform_fields(context=serial_contexts[index])
            parallel_field_y, parallel_field_x = _read_deform_fields(context=parallel_contexts[index])

            # The groupwise registration itself runs on the calling thread in both branches, so the fields the pool
            # distributes are bit-identical to the ones the serial loop distributes.
            np.testing.assert_array_equal(parallel_field_y, serial_field_y)
            np.testing.assert_array_equal(parallel_field_x, serial_field_x)

            # Equality alone cannot see a mapping error applied to both branches, so each recording's saved field is
            # also matched against the deformation the algorithm resolved for that recording's own image index.
            np.testing.assert_array_equal(
                parallel_field_y, parallel_registration.get_deformation(image_index=index).get_field(dimension=0)
            )
            np.testing.assert_array_equal(
                parallel_field_x, parallel_registration.get_deformation(image_index=index).get_field(dimension=1)
            )
            np.testing.assert_array_equal(
                serial_field_y, serial_registration.get_deformation(image_index=index).get_field(dimension=0)
            )

        # The two recordings differ by an imposed shift, so their deformations differ. Without that, matching each
        # recording against its own index would hold under a swapped mapping as well.
        first_field_y, _ = _read_deform_fields(context=parallel_contexts[0])
        second_field_y, _ = _read_deform_fields(context=parallel_contexts[1])
        assert float(np.max(np.abs(first_field_y - second_field_y))) > 0.1


class TestProjectTemplatesParallelPath:
    """Tests the thread-pool branch project_templates_to_recordings takes when allocated more than one worker."""

    def test_matches_the_serial_path_and_keeps_each_projection_with_its_recording(self, tmp_path: Path) -> None:
        """Verifies the parallel branch projects the same centroids the serial branch does, per recording."""
        configuration = _make_configuration()
        serial_contexts = self._build_contexts(tmp_path=tmp_path / "serial", configuration=configuration)
        parallel_contexts = self._build_contexts(tmp_path=tmp_path / "parallel", configuration=configuration)

        project_templates_to_recordings(contexts=serial_contexts, workers=1)
        project_templates_to_recordings(contexts=parallel_contexts, workers=4)

        # The templates sit at (20, 20) and (42, 24) in the shared space. Backward projection applies the inverse of
        # each recording's own uniform field, which moves them by that field: rec0 down four rows, rec1 five columns
        # left. A pool that paired a recording with another one's context would report the other recording's shift.
        assert self._read_centroids(context=parallel_contexts[0]) == [(24, 20), (46, 24)]
        assert self._read_centroids(context=parallel_contexts[1]) == [(20, 15), (42, 19)]

        for index in range(2):
            assert self._read_centroids(context=parallel_contexts[index]) == self._read_centroids(
                context=serial_contexts[index]
            )

    @staticmethod
    def _build_contexts(
        tmp_path: Path, configuration: MultiRecordingConfiguration
    ) -> list[MultiRecordingRuntimeContext]:
        """Builds two projection contexts whose deformation fields displace their templates along different axes."""
        return [
            _build_projection_context(
                tmp_path=tmp_path, configuration=configuration, recording_id="rec0", displacement=(4.0, 0.0)
            ),
            _build_projection_context(
                tmp_path=tmp_path, configuration=configuration, recording_id="rec1", displacement=(0.0, -5.0)
            ),
        ]

    @staticmethod
    def _read_centroids(context: MultiRecordingRuntimeContext) -> list[tuple[int, int]]:
        """Reads the projected ROI centroids a recording's output directory holds."""
        output_path = context.runtime.output_path
        assert output_path is not None
        roi_statistics = ROIStatistics.load_list(
            masks_path=output_path / "roi_masks.npz", statistics_path=output_path / "roi_statistics.npz"
        )
        return sorted(roi.mask.centroid for roi in roi_statistics)


class TestApplyForwardDeformation:
    """Tests _apply_forward_deformation."""

    def test_raises_without_combined_data(self) -> None:
        """Verifies that forward deformation raises a ValueError when combined data is not loaded."""
        runtime = MultiRecordingRuntimeData()
        runtime.io.recording_id = "rec0"
        context = MultiRecordingRuntimeContext(configuration=MultiRecordingConfiguration(), runtime=runtime)
        deformation = Deformation.identity(height=_FRAME_SIZE, width=_FRAME_SIZE)

        expected_message = (
            "Unable to register recording 'rec0' to shared visual space. The recording's combined_data must be "
            "loaded before transforming images and ROI masks."
        )
        with pytest.raises(ValueError, match=error_format(expected_message)):
            _apply_forward_deformation(context=context, deformation=deformation)


class TestProjectTemplatesToRecordings:
    """Tests project_templates_to_recordings."""

    def test_projects_channel_1_templates(self, tmp_path: Path) -> None:
        """Verifies that backward projection writes channel 1 ROI statistics for the tracked templates."""
        configuration = _make_configuration()
        contexts = [
            _build_projection_context(tmp_path=tmp_path, configuration=configuration, recording_id="rec0"),
            _build_projection_context(tmp_path=tmp_path, configuration=configuration, recording_id="rec1"),
        ]

        project_templates_to_recordings(contexts=contexts, workers=1)

        for context in contexts:
            output_path = context.runtime.output_path
            assert output_path is not None
            assert (output_path / "roi_statistics.npz").exists()
            assert (output_path / "roi_masks.npz").exists()
            assert not (output_path / "roi_statistics_channel_2.npz").exists()
            assert context.runtime.timing.backward_transform_time >= 0

            roi_statistics = ROIStatistics.load_list(
                masks_path=output_path / "roi_masks.npz", statistics_path=output_path / "roi_statistics.npz"
            )
            assert len(roi_statistics) == 2
            # Identity deformation preserves template centroids, and tracked ROIs carry a zeroed footprint.
            assert {roi.mask.centroid for roi in roi_statistics} == {(20, 20), (42, 24)}
            assert all(roi.footprint == 0 for roi in roi_statistics)

    def test_projects_channel_2_templates(self, tmp_path: Path) -> None:
        """Verifies that backward projection writes channel 2 ROI statistics when only channel 2 templates exist."""
        configuration = _make_configuration()
        contexts = [
            _build_projection_context(
                tmp_path=tmp_path,
                configuration=configuration,
                recording_id="rec0",
                channel_1_templates=False,
                channel_2_templates=True,
            ),
            _build_projection_context(
                tmp_path=tmp_path,
                configuration=configuration,
                recording_id="rec1",
                channel_1_templates=False,
                channel_2_templates=True,
            ),
        ]

        project_templates_to_recordings(contexts=contexts, workers=1)

        for context in contexts:
            output_path = context.runtime.output_path
            assert output_path is not None
            assert (output_path / "roi_statistics_channel_2.npz").exists()
            assert not (output_path / "roi_statistics.npz").exists()

    def test_no_templates_produces_no_statistics(self, tmp_path: Path) -> None:
        """Verifies that projection without any template masks completes and writes no ROI statistics."""
        configuration = _make_configuration()
        contexts = [
            _build_projection_context(
                tmp_path=tmp_path,
                configuration=configuration,
                recording_id="rec0",
                channel_1_templates=False,
                channel_2_templates=False,
            ),
            _build_projection_context(
                tmp_path=tmp_path,
                configuration=configuration,
                recording_id="rec1",
                channel_1_templates=False,
                channel_2_templates=False,
            ),
        ]

        project_templates_to_recordings(contexts=contexts, workers=1)

        for context in contexts:
            output_path = context.runtime.output_path
            assert output_path is not None
            assert not (output_path / "roi_statistics.npz").exists()
            assert not (output_path / "roi_statistics_channel_2.npz").exists()

    def test_skips_when_output_exists(self, tmp_path: Path) -> None:
        """Verifies that a second projection call short-circuits when the projection output already exists."""
        configuration = _make_configuration()
        contexts = [
            _build_projection_context(tmp_path=tmp_path, configuration=configuration, recording_id="rec0"),
            _build_projection_context(tmp_path=tmp_path, configuration=configuration, recording_id="rec1"),
        ]
        project_templates_to_recordings(contexts=contexts, workers=1)

        # Clearing the in-memory templates would break a re-run. The skip path must not reach them.
        for context in contexts:
            context.runtime.tracking.template_masks = None

        project_templates_to_recordings(contexts=contexts, workers=1)

        for context in contexts:
            output_path = context.runtime.output_path
            assert output_path is not None
            assert (output_path / "roi_statistics.npz").exists()

    def test_reprojects_when_a_later_output_is_missing(self, tmp_path: Path) -> None:
        """Verifies that a projection missing one recording's output re-projects instead of reporting completion."""
        configuration = _make_configuration()
        contexts = [
            _build_projection_context(tmp_path=tmp_path, configuration=configuration, recording_id="rec0"),
            _build_projection_context(tmp_path=tmp_path, configuration=configuration, recording_id="rec1"),
        ]
        project_templates_to_recordings(contexts=contexts, workers=1)

        # Reproduces the state a run killed between the two per-recording writes leaves behind, where the first
        # recording carries its projected statistics and the second does not.
        second_output = contexts[1].runtime.output_path
        assert second_output is not None
        (second_output / "roi_statistics.npz").unlink()

        project_templates_to_recordings(contexts=contexts, workers=1)

        assert (second_output / "roi_statistics.npz").exists()


class TestApplyBackwardDeformation:
    """Tests _apply_backward_deformation."""

    def test_raises_without_combined_data(self) -> None:
        """Verifies that backward deformation raises a ValueError when combined data is not loaded."""
        runtime = MultiRecordingRuntimeData()
        runtime.io.recording_id = "rec0"
        context = MultiRecordingRuntimeContext(configuration=MultiRecordingConfiguration(), runtime=runtime)

        expected_message = (
            "Unable to project templates to recording 'rec0'. The recording's combined_data must be loaded before "
            "transforming template masks."
        )
        with pytest.raises(ValueError, match=error_format(expected_message)):
            _apply_backward_deformation(context=context)

    def test_raises_without_deformation_fields(self) -> None:
        """Verifies that backward deformation raises a ValueError when deformation fields are not populated."""
        runtime = MultiRecordingRuntimeData()
        runtime.io.recording_id = "rec0"
        runtime.combined_data = CombinedData(
            detection=DetectionData(),
            extraction=ExtractionData(),
            plane_count=1,
            combined_height=_FRAME_SIZE,
            combined_width=_FRAME_SIZE,
            tau=1.0,
            sampling_rate=30.0,
        )
        context = MultiRecordingRuntimeContext(configuration=MultiRecordingConfiguration(), runtime=runtime)

        expected_message = (
            "Unable to project templates to recording 'rec0'. Deformation fields must be computed by "
            "register_recordings() before applying backward transformation."
        )
        with pytest.raises(ValueError, match=error_format(expected_message)):
            _apply_backward_deformation(context=context)


def _build_blob_image(
    builder: Callable[..., NDArray[np.float64]], centers: tuple[tuple[int, int], ...]
) -> NDArray[np.float32]:
    """Builds a structured Gaussian-blob reference image cast to float32."""
    image = builder(
        height=_FRAME_SIZE,
        width=_FRAME_SIZE,
        centers=centers,
        sigma=4.0,
        amplitude=2000.0,
        background=100.0,
    )
    return image.astype(np.float32)


def _make_circle_mask(
    centroid: tuple[int, int],
    radius: int,
    frame_width: int,
    cluster_id: int = 0,
    recording_count: int = 0,
) -> ROIMask:
    """Creates a filled circular ROIMask centered on the given centroid."""
    delta_y, delta_x = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    inside_circle = delta_y**2 + delta_x**2 <= radius**2
    y_array = (centroid[0] + delta_y[inside_circle]).astype(np.int32)
    x_array = (centroid[1] + delta_x[inside_circle]).astype(np.int32)
    pixel_weights = np.ones(y_array.size, dtype=np.float32)
    pixel_weights /= np.linalg.norm(pixel_weights)
    return ROIMask(
        y_pixels=y_array,
        x_pixels=x_array,
        pixel_weights=pixel_weights,
        centroid=centroid,
        frame_width=frame_width,
        radius=float(radius),
        cluster_id=cluster_id,
        recording_count=recording_count,
    )


def _make_configuration(
    *,
    image_type: ReferenceImageType = ReferenceImageType.ENHANCED_MEAN,
    repeat_registration: bool = False,
) -> MultiRecordingConfiguration:
    """Builds a multi-recording configuration with fast diffeomorphic registration settings."""
    configuration = MultiRecordingConfiguration()
    configuration.runtime.display_progress_bars = False
    configuration.diffeomorphic_registration.image_type = image_type
    configuration.diffeomorphic_registration.scale_sampling = 5
    configuration.diffeomorphic_registration.repeat_registration = repeat_registration
    return configuration


def _make_detection(
    builder: Callable[..., NDArray[np.float64]],
    centers: tuple[tuple[int, int], ...],
    image_kinds: tuple[str, ...],
    *,
    two_channel: bool,
) -> DetectionData:
    """Builds DetectionData populated with the requested reference image variants for one or both channels."""
    detection = DetectionData()
    detection.roi_diameter = 8
    if "mean" in image_kinds:
        detection.mean_image = _build_blob_image(builder=builder, centers=centers)
    if "enhanced_mean" in image_kinds:
        detection.enhanced_mean_image = _build_blob_image(builder=builder, centers=centers)
    if "maximum_projection" in image_kinds:
        detection.maximum_projection = _build_blob_image(builder=builder, centers=centers)
    if two_channel:
        detection.roi_diameter_channel_2 = 8
        detection.mean_image_channel_2 = _build_blob_image(builder=builder, centers=centers)
        detection.enhanced_mean_image_channel_2 = _build_blob_image(builder=builder, centers=centers)
        detection.maximum_projection_channel_2 = _build_blob_image(builder=builder, centers=centers)
    return detection


def _build_recording_context(
    tmp_path: Path,
    builder: Callable[..., NDArray[np.float64]],
    configuration: MultiRecordingConfiguration,
    *,
    recording_id: str,
    centers: tuple[tuple[int, int], ...],
    image_kinds: tuple[str, ...] = ("enhanced_mean",),
    two_channel: bool = False,
    selected_indices: tuple[int, ...] = (0, 1, 2, 3),
    selected_indices_channel_2: tuple[int, ...] = (),
    write_channel_1_masks: bool = True,
    write_channel_2_masks: bool = False,
) -> MultiRecordingRuntimeContext:
    """Builds a single registration context backed by on-disk single-recording combined data and ROI masks."""
    data_path = tmp_path / recording_id / "cindra"
    output_path = data_path / "multi_recording" / "dataset"

    detection = _make_detection(builder=builder, centers=centers, image_kinds=image_kinds, two_channel=two_channel)
    CombinedData(
        detection=detection,
        extraction=ExtractionData(),
        plane_count=1,
        combined_height=_FRAME_SIZE,
        combined_width=_FRAME_SIZE,
        tau=1.0,
        sampling_rate=30.0,
    ).save(root_path=data_path)

    masks = [_make_circle_mask(centroid=center, radius=4, frame_width=_FRAME_SIZE) for center in centers]
    if write_channel_1_masks:
        ROIMask.save_list(mask_list=masks, file_path=data_path / "roi_masks.npz")
    if write_channel_2_masks:
        ROIMask.save_list(mask_list=masks, file_path=data_path / "roi_masks_channel_2.npz")

    runtime = MultiRecordingRuntimeData()
    runtime.output_path = output_path
    runtime.io.recording_id = recording_id
    runtime.io.data_path = data_path
    runtime.io.dataset_name = "dataset"
    runtime.io.selected_roi_indices = selected_indices
    runtime.io.selected_roi_indices_channel_2 = selected_indices_channel_2
    runtime.combined_data = CombinedData.load(root_path=data_path)

    return MultiRecordingRuntimeContext(configuration=configuration, runtime=runtime)


def _build_recording_pair(
    tmp_path: Path,
    builder: Callable[..., NDArray[np.float64]],
    configuration: MultiRecordingConfiguration,
    *,
    shift: int = 2,
    **keyword_arguments: object,
) -> list[MultiRecordingRuntimeContext]:
    """Builds two registration contexts whose reference images differ by a small uniform translation."""
    reference = _build_recording_context(
        tmp_path=tmp_path,
        builder=builder,
        configuration=configuration,
        recording_id="rec0",
        centers=_BASE_CENTERS,
        **keyword_arguments,
    )
    shifted_centers = tuple((center[0] + shift, center[1] + shift) for center in _BASE_CENTERS)
    moved = _build_recording_context(
        tmp_path=tmp_path,
        builder=builder,
        configuration=configuration,
        recording_id="rec1",
        centers=shifted_centers,
        **keyword_arguments,
    )
    return [reference, moved]


def _build_projection_context(
    tmp_path: Path,
    configuration: MultiRecordingConfiguration,
    *,
    recording_id: str,
    channel_1_templates: bool = True,
    channel_2_templates: bool = False,
    displacement: tuple[float, float] = (0.0, 0.0),
) -> MultiRecordingRuntimeContext:
    """Builds a projection context with identity deformation fields on disk and in-memory template masks."""
    output_path = tmp_path / recording_id / "cindra" / "multi_recording" / "dataset"
    ensure_directory_exists(path=output_path)

    runtime = MultiRecordingRuntimeData()
    runtime.output_path = output_path
    runtime.io.recording_id = recording_id
    runtime.combined_data = CombinedData(
        detection=DetectionData(),
        extraction=ExtractionData(),
        plane_count=1,
        combined_height=_FRAME_SIZE,
        combined_width=_FRAME_SIZE,
        tau=1.0,
        sampling_rate=30.0,
    )

    # Persists uniform deformation fields, which default to zero displacement so backward projection preserves
    # template positions. A non-zero displacement makes each recording's own field visible in its projected output.
    runtime.registration.deform_field_y = np.full(
        (_FRAME_SIZE, _FRAME_SIZE), fill_value=displacement[0], dtype=np.float32
    )
    runtime.registration.deform_field_x = np.full(
        (_FRAME_SIZE, _FRAME_SIZE), fill_value=displacement[1], dtype=np.float32
    )
    runtime.registration.save_arrays(output_path=output_path)
    runtime.registration.release_arrays()

    if channel_1_templates:
        runtime.tracking.template_masks = [
            _make_circle_mask(centroid=(20, 20), radius=4, frame_width=_FRAME_SIZE, cluster_id=1, recording_count=2),
            _make_circle_mask(centroid=(42, 24), radius=4, frame_width=_FRAME_SIZE, cluster_id=2, recording_count=2),
        ]
        runtime.tracking.template_diameter = 8
    if channel_2_templates:
        runtime.tracking.template_masks_channel_2 = [
            _make_circle_mask(centroid=(30, 30), radius=4, frame_width=_FRAME_SIZE, cluster_id=3, recording_count=2),
        ]
        runtime.tracking.template_diameter_channel_2 = 8

    return MultiRecordingRuntimeContext(configuration=configuration, runtime=runtime)


def _read_deform_fields(context: MultiRecordingRuntimeContext) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Memory-maps the saved deformation fields and returns in-memory copies of the Y and X components."""
    output_path = context.runtime.output_path
    assert output_path is not None
    context.runtime.registration.memory_map_arrays(output_path=output_path)
    field_y = np.array(context.runtime.registration.deform_field_y, dtype=np.float32)
    field_x = np.array(context.runtime.registration.deform_field_x, dtype=np.float32)
    context.runtime.registration.release_arrays()
    return field_y, field_x


def _capture_registrations(monkeypatch: pytest.MonkeyPatch) -> list[DiffeomorphicDemonsRegistration]:
    """Wraps the registration algorithm so every instance register_recordings builds is captured for inspection."""
    registrations: list[DiffeomorphicDemonsRegistration] = []

    def _build(*arguments: object, **keyword_arguments: object) -> DiffeomorphicDemonsRegistration:
        registration = DiffeomorphicDemonsRegistration(*arguments, **keyword_arguments)
        registrations.append(registration)
        return registration

    # Resolves the module through import_module rather than through a dotted string. The registration package binds
    # the name 'register_recordings' to the function it re-exports, which shadows the module of the same name, so a
    # dotted string reaches the function and finds no algorithm attribute on it.
    monkeypatch.setattr(
        import_module("cindra.registration.register_recordings"), "DiffeomorphicDemonsRegistration", _build
    )
    return registrations
