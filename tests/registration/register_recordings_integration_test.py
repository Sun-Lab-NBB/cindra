"""Contains integration tests for the cross-recording registration stage entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ataraxis_base_utilities import ensure_directory_exists

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


def _blob(builder: Callable[..., NDArray[np.float64]], centers: tuple[tuple[int, int], ...]) -> NDArray[np.float32]:
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


def _circle_mask(
    centroid: tuple[int, int],
    radius: int,
    frame_width: int,
    cluster_id: int = 0,
    recording_count: int = 0,
) -> ROIMask:
    """Creates a filled circular ROIMask centered on the given centroid."""
    y_pixels: list[int] = []
    x_pixels: list[int] = []
    for delta_y in range(-radius, radius + 1):
        for delta_x in range(-radius, radius + 1):
            if delta_y**2 + delta_x**2 <= radius**2:
                y_pixels.append(centroid[0] + delta_y)
                x_pixels.append(centroid[1] + delta_x)
    pixel_weights = np.ones(len(y_pixels), dtype=np.float32)
    pixel_weights /= np.linalg.norm(pixel_weights)
    return ROIMask(
        y_pixels=np.array(y_pixels, dtype=np.int32),
        x_pixels=np.array(x_pixels, dtype=np.int32),
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
    """Builds a serial multi-recording configuration with fast diffeomorphic registration settings."""
    configuration = MultiRecordingConfiguration()
    configuration.runtime.parallel_workers = 1
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
        detection.mean_image = _blob(builder, centers)
    if "enhanced_mean" in image_kinds:
        detection.enhanced_mean_image = _blob(builder, centers)
    if "maximum_projection" in image_kinds:
        detection.maximum_projection = _blob(builder, centers)
    if two_channel:
        detection.roi_diameter_channel_2 = 8
        detection.mean_image_channel_2 = _blob(builder, centers)
        detection.enhanced_mean_image_channel_2 = _blob(builder, centers)
        detection.maximum_projection_channel_2 = _blob(builder, centers)
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

    detection = _make_detection(builder, centers, image_kinds, two_channel=two_channel)
    CombinedData(
        detection=detection,
        extraction=ExtractionData(),
        plane_count=1,
        combined_height=_FRAME_SIZE,
        combined_width=_FRAME_SIZE,
        tau=1.0,
        sampling_rate=30.0,
    ).save(root_path=data_path)

    masks = [_circle_mask(centroid=center, radius=4, frame_width=_FRAME_SIZE) for center in centers]
    if write_channel_1_masks:
        ROIMask.save_list(masks, data_path / "roi_masks.npz")
    if write_channel_2_masks:
        ROIMask.save_list(masks, data_path / "roi_masks_channel_2.npz")

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
    **kwargs: object,
) -> list[MultiRecordingRuntimeContext]:
    """Builds two registration contexts whose reference images differ by a small uniform translation."""
    reference = _build_recording_context(
        tmp_path, builder, configuration, recording_id="rec0", centers=_BASE_CENTERS, **kwargs
    )
    shifted_centers = tuple((center[0] + shift, center[1] + shift) for center in _BASE_CENTERS)
    moved = _build_recording_context(
        tmp_path, builder, configuration, recording_id="rec1", centers=shifted_centers, **kwargs
    )
    return [reference, moved]


def _build_projection_context(
    tmp_path: Path,
    configuration: MultiRecordingConfiguration,
    *,
    recording_id: str,
    channel_1_templates: bool = True,
    channel_2_templates: bool = False,
) -> MultiRecordingRuntimeContext:
    """Builds a projection context with identity deformation fields on disk and in-memory template masks."""
    output_path = tmp_path / recording_id / "cindra" / "multi_recording" / "dataset"
    ensure_directory_exists(output_path)

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

    # Persists identity (zero-displacement) deformation fields so backward projection preserves template positions.
    runtime.registration.deform_field_y = np.zeros((_FRAME_SIZE, _FRAME_SIZE), dtype=np.float32)
    runtime.registration.deform_field_x = np.zeros((_FRAME_SIZE, _FRAME_SIZE), dtype=np.float32)
    runtime.registration.save_arrays(output_path)
    runtime.registration.release_arrays()

    if channel_1_templates:
        runtime.tracking.template_masks = [
            _circle_mask(centroid=(20, 20), radius=4, frame_width=_FRAME_SIZE, cluster_id=1, recording_count=2),
            _circle_mask(centroid=(42, 24), radius=4, frame_width=_FRAME_SIZE, cluster_id=2, recording_count=2),
        ]
        runtime.tracking.template_diameter = 8
    if channel_2_templates:
        runtime.tracking.template_masks_channel_2 = [
            _circle_mask(centroid=(30, 30), radius=4, frame_width=_FRAME_SIZE, cluster_id=3, recording_count=2),
        ]
        runtime.tracking.template_diameter_channel_2 = 8

    return MultiRecordingRuntimeContext(configuration=configuration, runtime=runtime)


def _read_deform_fields(context: MultiRecordingRuntimeContext) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Memory-maps the saved deformation fields and returns in-memory copies of the Y and X components."""
    output_path = context.runtime.output_path
    assert output_path is not None
    context.runtime.registration.memory_map_arrays(output_path)
    field_y = np.array(context.runtime.registration.deform_field_y, dtype=np.float32)
    field_x = np.array(context.runtime.registration.deform_field_x, dtype=np.float32)
    context.runtime.registration.release_arrays()
    return field_y, field_x


class TestRegisterRecordings:
    """Tests register_recordings."""

    def test_serial_path_writes_deformation_outputs(
        self, gaussian_blob_image: Callable[..., NDArray[np.float64]], tmp_path: Path
    ) -> None:
        """Verifies that serial registration writes deformation fields, transformed images, and deformed masks."""
        configuration = _make_configuration()
        contexts = _build_recording_pair(tmp_path, gaussian_blob_image, configuration)

        register_recordings(contexts)

        for context in contexts:
            output_path = context.runtime.output_path
            assert output_path is not None
            assert (output_path / "registration_arrays" / "deform_field_y.npy").exists()
            assert (output_path / "registration_arrays" / "deform_field_x.npy").exists()
            assert (output_path / "registration_arrays" / "transformed_enhanced_mean_image.npy").exists()
            assert (output_path / "registration_deformed_masks.npz").exists()
            assert (output_path / "multi_recording_runtime_data.yaml").exists()
            assert context.runtime.timing.registration_time >= 0

            field_y, field_x = _read_deform_fields(context)
            assert field_y.shape == (_FRAME_SIZE, _FRAME_SIZE)
            assert field_x.shape == (_FRAME_SIZE, _FRAME_SIZE)
            assert np.all(np.isfinite(field_y))
            assert np.all(np.isfinite(field_x))

            deformed_masks = ROIMask.load_list(output_path / "registration_deformed_masks.npz")
            assert len(deformed_masks) == len(_BASE_CENTERS)

    def test_identical_images_produce_near_zero_deformation(
        self, gaussian_blob_image: Callable[..., NDArray[np.float64]], tmp_path: Path
    ) -> None:
        """Verifies that registering two identical reference images yields near-zero deformation fields."""
        configuration = _make_configuration()
        contexts = _build_recording_pair(tmp_path, gaussian_blob_image, configuration, shift=0)

        register_recordings(contexts)

        for context in contexts:
            field_y, field_x = _read_deform_fields(context)
            assert float(np.abs(field_y).max()) < 1.0
            assert float(np.abs(field_x).max()) < 1.0

    def test_maximum_projection_image_type(
        self, gaussian_blob_image: Callable[..., NDArray[np.float64]], tmp_path: Path
    ) -> None:
        """Verifies registration against maximum-projection images while skipping forward mask deformation."""
        configuration = _make_configuration(image_type=ReferenceImageType.MAXIMUM_PROJECTION)
        contexts = _build_recording_pair(
            tmp_path,
            gaussian_blob_image,
            configuration,
            image_kinds=("maximum_projection",),
            selected_indices=(),
        )

        register_recordings(contexts)

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
            tmp_path,
            gaussian_blob_image,
            configuration,
            image_kinds=("mean", "enhanced_mean", "maximum_projection"),
            two_channel=True,
            selected_indices_channel_2=(0, 1, 2, 3),
            write_channel_2_masks=True,
        )

        register_recordings(contexts)

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
            tmp_path,
            gaussian_blob_image,
            configuration,
            selected_indices=(0, 1, 2, 3),
            selected_indices_channel_2=(0, 1, 2, 3),
            write_channel_1_masks=False,
            write_channel_2_masks=False,
        )

        register_recordings(contexts)

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
        contexts = _build_recording_pair(tmp_path, gaussian_blob_image, configuration)
        register_recordings(contexts)

        # Removing combined data would break a re-run; the skip path must not touch it, so no error proves the skip.
        for context in contexts:
            context.runtime.combined_data = None

        register_recordings(contexts)

        for context in contexts:
            output_path = context.runtime.output_path
            assert output_path is not None
            assert (output_path / "registration_arrays" / "deform_field_y.npy").exists()

    def test_forced_repeat_clears_and_reregisters(
        self, gaussian_blob_image: Callable[..., NDArray[np.float64]], tmp_path: Path
    ) -> None:
        """Verifies that enabling repeat_registration re-runs registration despite existing registration data."""
        configuration = _make_configuration()
        contexts = _build_recording_pair(tmp_path, gaussian_blob_image, configuration)
        register_recordings(contexts)

        configuration.diffeomorphic_registration.repeat_registration = True
        register_recordings(contexts)

        for context in contexts:
            field_y, field_x = _read_deform_fields(context)
            assert field_y.shape == (_FRAME_SIZE, _FRAME_SIZE)
            assert np.all(np.isfinite(field_x))

    def test_missing_combined_data_raises(
        self, gaussian_blob_image: Callable[..., NDArray[np.float64]], tmp_path: Path
    ) -> None:
        """Verifies that a recording without loaded combined data raises a ValueError during registration."""
        configuration = _make_configuration()
        contexts = _build_recording_pair(tmp_path, gaussian_blob_image, configuration)
        contexts[0].runtime.combined_data = None

        with pytest.raises(ValueError, match="combined_data must be loaded"):
            register_recordings(contexts)

    def test_missing_reference_image_raises(
        self, gaussian_blob_image: Callable[..., NDArray[np.float64]], tmp_path: Path
    ) -> None:
        """Verifies that requesting an unavailable reference image type raises a ValueError."""
        configuration = _make_configuration(image_type=ReferenceImageType.MEAN)
        contexts = _build_recording_pair(tmp_path, gaussian_blob_image, configuration, image_kinds=("enhanced_mean",))

        with pytest.raises(ValueError, match="required reference image"):
            register_recordings(contexts)


class TestApplyForwardDeformation:
    """Tests _apply_forward_deformation."""

    def test_raises_without_combined_data(self) -> None:
        """Verifies that forward deformation raises a ValueError when combined data is not loaded."""
        runtime = MultiRecordingRuntimeData()
        runtime.io.recording_id = "rec0"
        context = MultiRecordingRuntimeContext(configuration=MultiRecordingConfiguration(), runtime=runtime)
        deformation = Deformation.identity(height=_FRAME_SIZE, width=_FRAME_SIZE)

        with pytest.raises(ValueError, match="combined_data must be loaded"):
            _apply_forward_deformation(context=context, deformation=deformation)


class TestProjectTemplatesToRecordings:
    """Tests project_templates_to_recordings."""

    def test_projects_channel_1_templates(self, tmp_path: Path) -> None:
        """Verifies that backward projection writes channel 1 ROI statistics for the tracked templates."""
        configuration = _make_configuration()
        contexts = [
            _build_projection_context(tmp_path, configuration, recording_id="rec0"),
            _build_projection_context(tmp_path, configuration, recording_id="rec1"),
        ]

        project_templates_to_recordings(contexts)

        for context in contexts:
            output_path = context.runtime.output_path
            assert output_path is not None
            assert (output_path / "roi_statistics.npz").exists()
            assert (output_path / "roi_masks.npz").exists()
            assert not (output_path / "roi_statistics_channel_2.npz").exists()
            assert context.runtime.timing.backward_transform_time >= 0

            roi_statistics = ROIStatistics.load_list(
                masks_path=output_path / "roi_masks.npz", stats_path=output_path / "roi_statistics.npz"
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
                tmp_path, configuration, recording_id="rec0", channel_1_templates=False, channel_2_templates=True
            ),
            _build_projection_context(
                tmp_path, configuration, recording_id="rec1", channel_1_templates=False, channel_2_templates=True
            ),
        ]

        project_templates_to_recordings(contexts)

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
                tmp_path, configuration, recording_id="rec0", channel_1_templates=False, channel_2_templates=False
            ),
            _build_projection_context(
                tmp_path, configuration, recording_id="rec1", channel_1_templates=False, channel_2_templates=False
            ),
        ]

        project_templates_to_recordings(contexts)

        for context in contexts:
            output_path = context.runtime.output_path
            assert output_path is not None
            assert not (output_path / "roi_statistics.npz").exists()
            assert not (output_path / "roi_statistics_channel_2.npz").exists()

    def test_skips_when_output_exists(self, tmp_path: Path) -> None:
        """Verifies that a second projection call short-circuits when the projection output already exists."""
        configuration = _make_configuration()
        contexts = [
            _build_projection_context(tmp_path, configuration, recording_id="rec0"),
            _build_projection_context(tmp_path, configuration, recording_id="rec1"),
        ]
        project_templates_to_recordings(contexts)

        # Clearing the in-memory templates would break a re-run; the skip path must not reach them.
        for context in contexts:
            context.runtime.tracking.template_masks = None

        project_templates_to_recordings(contexts)

        for context in contexts:
            output_path = context.runtime.output_path
            assert output_path is not None
            assert (output_path / "roi_statistics.npz").exists()


class TestApplyBackwardDeformation:
    """Tests _apply_backward_deformation."""

    def test_raises_without_combined_data(self) -> None:
        """Verifies that backward deformation raises a ValueError when combined data is not loaded."""
        runtime = MultiRecordingRuntimeData()
        runtime.io.recording_id = "rec0"
        context = MultiRecordingRuntimeContext(configuration=MultiRecordingConfiguration(), runtime=runtime)

        with pytest.raises(ValueError, match="combined_data must be loaded"):
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

        with pytest.raises(ValueError, match="Deformation fields must be computed"):
            _apply_backward_deformation(context=context)
