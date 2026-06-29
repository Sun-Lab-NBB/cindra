"""Contains integration tests for the multi-recording ROI tracking entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from cindra.dataclasses import (
    ROIMask,
    CombinedData,
    DetectionData,
    ExtractionData,
    MultiRecordingRuntimeData,
    MultiRecordingConfiguration,
    MultiRecordingRuntimeContext,
)
from cindra.detection.tracking import track_rois_across_recordings

if TYPE_CHECKING:
    from pathlib import Path


def _make_block_mask(
    *,
    y_origin: int,
    x_origin: int,
    size: int,
    frame_width: int,
    weight: float = 1.0,
    radius: float = 5.0,
    cluster_id: int = 0,
) -> ROIMask:
    """Creates a square block ROIMask spanning a size-by-size pixel region anchored at the given origin."""
    rows, cols = np.meshgrid(
        np.arange(y_origin, y_origin + size),
        np.arange(x_origin, x_origin + size),
        indexing="ij",
    )
    y_pixels = rows.ravel().astype(np.int32)
    x_pixels = cols.ravel().astype(np.int32)
    weights = np.full(shape=y_pixels.shape, fill_value=weight, dtype=np.float32)
    centroid = (int(np.median(y_pixels)), int(np.median(x_pixels)))
    return ROIMask(
        y_pixels=y_pixels,
        x_pixels=x_pixels,
        pixel_weights=weights,
        centroid=centroid,
        frame_width=frame_width,
        radius=radius,
        cluster_id=cluster_id,
    )


def _make_combined_data(image_size: int) -> CombinedData:
    """Builds a minimal CombinedData defining the shared visual space dimensions used by tracking."""
    return CombinedData(
        detection=DetectionData(),
        extraction=ExtractionData(),
        plane_count=1,
        combined_height=image_size,
        combined_width=image_size,
        tau=1.0,
        sampling_rate=15.0,
    )


def _make_context(
    output_path: Path,
    configuration: MultiRecordingConfiguration,
    *,
    image_size: int = 400,
    deformed_masks: list[ROIMask] | None = None,
    deformed_masks_channel_2: list[ROIMask] | None = None,
    with_combined_data: bool = True,
    set_output_path: bool = True,
) -> MultiRecordingRuntimeContext:
    """Builds a multi-recording runtime context wired with deformed masks and shared-space dimensions for tracking."""
    runtime = MultiRecordingRuntimeData()
    runtime.output_path = output_path if set_output_path else None
    runtime.io.dataset_output_paths = (output_path,)
    runtime.registration.deformed_roi_masks = deformed_masks
    runtime.registration.deformed_roi_masks_channel_2 = deformed_masks_channel_2
    if with_combined_data:
        runtime.combined_data = _make_combined_data(image_size)
    return MultiRecordingRuntimeContext(configuration=configuration, runtime=runtime)


def _assert_valid_template(template: ROIMask, image_size: int) -> None:
    """Asserts that a template mask exposes a consistent, in-bounds pixel set."""
    assert template.y_pixels.shape == template.x_pixels.shape
    assert template.pixel_weights.shape == template.y_pixels.shape
    assert template.y_pixels.size > 0
    assert int(template.y_pixels.min()) >= 0
    assert int(template.x_pixels.min()) >= 0
    assert int(template.y_pixels.max()) < image_size
    assert int(template.x_pixels.max()) < image_size


class TestTrackRoisAcrossRecordings:
    """Tests track_rois_across_recordings."""

    def test_returns_early_for_empty_contexts(self) -> None:
        """Verifies that an empty context list returns without error."""
        track_rois_across_recordings(contexts=[])

    def test_clusters_overlapping_pair_and_keeps_isolated_roi(self, tmp_path: Path) -> None:
        """Verifies that an overlapping ROI pair collapses to one template while an isolated ROI keeps its own."""
        image_size = 400
        configuration = MultiRecordingConfiguration()

        # Recording 0 contributes the overlapping ROI, an isolated ROI, and a pre-clustered ROI that must be ignored.
        recording_0_masks = [
            _make_block_mask(y_origin=48, x_origin=48, size=6, frame_width=image_size),
            _make_block_mask(y_origin=148, x_origin=148, size=6, frame_width=image_size),
            _make_block_mask(y_origin=300, x_origin=300, size=6, frame_width=image_size, cluster_id=9),
        ]
        # Recording 1 contributes an ROI identical to recording 0's overlapping ROI so the pair clusters together.
        recording_1_masks = [
            _make_block_mask(y_origin=48, x_origin=48, size=6, frame_width=image_size),
        ]

        context_0 = _make_context(
            tmp_path / "rec0", configuration, image_size=image_size, deformed_masks=recording_0_masks
        )
        context_1 = _make_context(
            tmp_path / "rec1", configuration, image_size=image_size, deformed_masks=recording_1_masks
        )
        contexts = [context_0, context_1]

        track_rois_across_recordings(contexts=contexts)

        templates = context_0.runtime.tracking.template_masks
        assert templates is not None
        assert len(templates) == 2
        for template in templates:
            _assert_valid_template(template=template, image_size=image_size)

        recording_counts = sorted(template.recording_count for template in templates)
        assert recording_counts == [1, 2]

        # Both recordings share the identical consensus template list and a positive estimated diameter.
        assert context_1.runtime.tracking.template_masks is templates
        assert context_0.runtime.tracking.template_diameter > 0
        assert context_1.runtime.tracking.template_diameter == context_0.runtime.tracking.template_diameter

        # The runtime data is persisted to each recording's output directory.
        assert (tmp_path / "rec0" / "tracking_template_masks.npz").exists()
        assert (tmp_path / "rec0" / "multi_recording_runtime_data.yaml").exists()
        assert (tmp_path / "rec1" / "tracking_template_masks.npz").exists()

    def test_tracks_both_channels_independently(self, tmp_path: Path) -> None:
        """Verifies that channel 1 and channel 2 deformed masks each produce their own template list."""
        image_size = 400
        configuration = MultiRecordingConfiguration()

        channel_1_recording_0 = [_make_block_mask(y_origin=48, x_origin=48, size=6, frame_width=image_size)]
        channel_1_recording_1 = [_make_block_mask(y_origin=48, x_origin=48, size=6, frame_width=image_size)]
        channel_2_recording_0 = [_make_block_mask(y_origin=148, x_origin=148, size=6, frame_width=image_size)]
        channel_2_recording_1 = [_make_block_mask(y_origin=148, x_origin=148, size=6, frame_width=image_size)]

        context_0 = _make_context(
            tmp_path / "rec0",
            configuration,
            image_size=image_size,
            deformed_masks=channel_1_recording_0,
            deformed_masks_channel_2=channel_2_recording_0,
        )
        context_1 = _make_context(
            tmp_path / "rec1",
            configuration,
            image_size=image_size,
            deformed_masks=channel_1_recording_1,
            deformed_masks_channel_2=channel_2_recording_1,
        )
        contexts = [context_0, context_1]

        track_rois_across_recordings(contexts=contexts)

        assert context_0.runtime.tracking.template_masks is not None
        assert len(context_0.runtime.tracking.template_masks) == 1
        assert context_0.runtime.tracking.template_masks_channel_2 is not None
        assert len(context_0.runtime.tracking.template_masks_channel_2) == 1
        assert context_0.runtime.tracking.template_diameter_channel_2 > 0
        assert (tmp_path / "rec0" / "tracking_template_masks_channel_2.npz").exists()

    def test_tracks_channel_2_only_when_channel_1_absent(self, tmp_path: Path) -> None:
        """Verifies that recordings carrying only channel 2 deformed masks track channel 2 templates alone."""
        image_size = 400
        configuration = MultiRecordingConfiguration()

        channel_2_recording_0 = [_make_block_mask(y_origin=48, x_origin=48, size=6, frame_width=image_size)]
        channel_2_recording_1 = [_make_block_mask(y_origin=48, x_origin=48, size=6, frame_width=image_size)]

        context_0 = _make_context(
            tmp_path / "rec0",
            configuration,
            image_size=image_size,
            deformed_masks=None,
            deformed_masks_channel_2=channel_2_recording_0,
        )
        context_1 = _make_context(
            tmp_path / "rec1",
            configuration,
            image_size=image_size,
            deformed_masks=None,
            deformed_masks_channel_2=channel_2_recording_1,
        )
        contexts = [context_0, context_1]

        track_rois_across_recordings(contexts=contexts)

        assert context_0.runtime.tracking.template_masks is None
        assert context_0.runtime.tracking.template_masks_channel_2 is not None
        assert len(context_0.runtime.tracking.template_masks_channel_2) == 1

    def test_filters_clusters_below_recording_prevalence(self, tmp_path: Path) -> None:
        """Verifies that a cluster appearing in too few recordings is dropped while a prevalent cluster survives."""
        image_size = 400
        configuration = MultiRecordingConfiguration()

        # Every recording observes the prevalent ROI; only recording 0 observes the isolated ROI. With three
        # recordings and a 50% mask prevalence, a cluster must appear in at least two recordings to be retained.
        contexts = []
        for index in range(3):
            masks = [_make_block_mask(y_origin=48, x_origin=48, size=6, frame_width=image_size)]
            if index == 0:
                masks.append(_make_block_mask(y_origin=148, x_origin=148, size=6, frame_width=image_size))
            contexts.append(
                _make_context(tmp_path / f"rec{index}", configuration, image_size=image_size, deformed_masks=masks)
            )

        track_rois_across_recordings(contexts=contexts)

        templates = contexts[0].runtime.tracking.template_masks
        assert templates is not None
        assert len(templates) == 1
        assert templates[0].recording_count == 3

    def test_assigns_boundary_cluster_to_owning_bin(self, tmp_path: Path) -> None:
        """Verifies that a cluster straddling a bin boundary is created once by the bin that owns its center."""
        image_size = 400
        configuration = MultiRecordingConfiguration()

        # The overlapping pair sits just past the y=200 bin boundary. The first bin sees it through its overlap
        # margin but does not own it, while the lower bin both sees and owns it, producing exactly one template.
        recording_0_masks = [_make_block_mask(y_origin=203, x_origin=48, size=6, frame_width=image_size)]
        recording_1_masks = [_make_block_mask(y_origin=203, x_origin=48, size=6, frame_width=image_size)]

        context_0 = _make_context(
            tmp_path / "rec0", configuration, image_size=image_size, deformed_masks=recording_0_masks
        )
        context_1 = _make_context(
            tmp_path / "rec1", configuration, image_size=image_size, deformed_masks=recording_1_masks
        )
        contexts = [context_0, context_1]

        track_rois_across_recordings(contexts=contexts)

        templates = context_0.runtime.tracking.template_masks
        assert templates is not None
        assert len(templates) == 1
        assert templates[0].recording_count == 2
        assert templates[0].centroid[0] >= 200

    def test_returns_when_channel_has_no_rois(self, tmp_path: Path) -> None:
        """Verifies that an empty deformed mask list yields no templates without raising."""
        configuration = MultiRecordingConfiguration()
        context_0 = _make_context(tmp_path / "rec0", configuration, deformed_masks=[])
        context_1 = _make_context(tmp_path / "rec1", configuration, deformed_masks=[])

        track_rois_across_recordings(contexts=[context_0, context_1])

        assert context_0.runtime.tracking.template_masks is None

    def test_returns_when_combined_data_missing(self, tmp_path: Path) -> None:
        """Verifies that tracking exits without templates when the shared-space dimensions are unavailable."""
        image_size = 400
        configuration = MultiRecordingConfiguration()
        masks = [_make_block_mask(y_origin=48, x_origin=48, size=6, frame_width=image_size)]
        context_0 = _make_context(tmp_path / "rec0", configuration, deformed_masks=masks, with_combined_data=False)
        context_1 = _make_context(tmp_path / "rec1", configuration, deformed_masks=masks, with_combined_data=False)

        track_rois_across_recordings(contexts=[context_0, context_1])

        assert context_0.runtime.tracking.template_masks is None

    def test_handles_recording_without_deformed_masks(self, tmp_path: Path) -> None:
        """Verifies that a recording lacking deformed masks is skipped during collection without raising."""
        image_size = 400
        configuration = MultiRecordingConfiguration()
        masks = [_make_block_mask(y_origin=48, x_origin=48, size=6, frame_width=image_size)]

        # Recording 0 contributes no deformed masks at all, so channel detection and ROI collection must both skip it.
        context_0 = _make_context(tmp_path / "rec0", configuration, image_size=image_size, deformed_masks=None)
        context_1 = _make_context(tmp_path / "rec1", configuration, image_size=image_size, deformed_masks=masks)

        track_rois_across_recordings(contexts=[context_0, context_1])

        # A single recording yields no cross-recording clusters, so the consensus template list is empty.
        assert context_1.runtime.tracking.template_masks == []
        assert context_1.runtime.tracking.template_diameter == 0

    def test_handles_dataset_without_any_deformed_masks(self, tmp_path: Path) -> None:
        """Verifies that a dataset where no recording carries deformed masks tracks nothing without raising."""
        configuration = MultiRecordingConfiguration()
        context_0 = _make_context(tmp_path / "rec0", configuration, deformed_masks=None, deformed_masks_channel_2=None)
        context_1 = _make_context(tmp_path / "rec1", configuration, deformed_masks=None, deformed_masks_channel_2=None)

        track_rois_across_recordings(contexts=[context_0, context_1])

        assert context_0.runtime.tracking.template_masks is None
        assert context_0.runtime.tracking.template_masks_channel_2 is None
        # Runtime data is still persisted for every recording even when no channel produced templates.
        assert (tmp_path / "rec0" / "multi_recording_runtime_data.yaml").exists()
        assert (tmp_path / "rec1" / "multi_recording_runtime_data.yaml").exists()

    def test_skips_when_template_masks_exist(self, tmp_path: Path) -> None:
        """Verifies that tracking loads existing templates and returns early when re-registration is disabled."""
        image_size = 400
        configuration = MultiRecordingConfiguration()
        existing_template = _make_block_mask(y_origin=48, x_origin=48, size=6, frame_width=image_size)

        output_0 = tmp_path / "rec0"
        output_0.mkdir(parents=True, exist_ok=True)
        ROIMask.save_list([existing_template], output_0 / "tracking_template_masks.npz")

        context_0 = _make_context(output_0, configuration, image_size=image_size, deformed_masks=None)
        # The second recording carries no output_path, exercising the load-skip branch within the early return.
        context_1 = _make_context(
            tmp_path / "rec1", configuration, image_size=image_size, deformed_masks=None, set_output_path=False
        )

        track_rois_across_recordings(contexts=[context_0, context_1])

        assert context_0.runtime.tracking.template_masks is not None
        assert len(context_0.runtime.tracking.template_masks) == 1
        assert context_1.runtime.tracking.template_masks is None
