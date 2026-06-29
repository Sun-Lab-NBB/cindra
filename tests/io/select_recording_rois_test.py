"""Contains integration tests for the select_recording_rois stage entry point."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from ataraxis_base_utilities import ensure_directory_exists

from cindra.io.select import select_recording_rois
from cindra.dataclasses import (
    ROIMask,
    CombinedData,
    DetectionData,
    ROIStatistics,
    ExtractionData,
    MultiRecordingRuntimeData,
    MultiRecordingConfiguration,
    MultiRecordingRuntimeContext,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_roi(centroid: tuple[int, int] = (10, 10), pixel_count: int = 50) -> ROIStatistics:
    """Creates a minimal ROIStatistics instance with the given centroid and pixel count."""
    y_pixels = np.arange(pixel_count, dtype=np.int32) % 10
    x_pixels = np.arange(pixel_count, dtype=np.int32) // 10
    mask = ROIMask(
        y_pixels=y_pixels,
        x_pixels=x_pixels,
        pixel_weights=np.ones(pixel_count, dtype=np.float32),
        centroid=centroid,
        frame_width=200,
    )
    roi = ROIStatistics(mask=mask)
    roi.pixel_count = pixel_count
    return roi


def _classification(probabilities: list[float]) -> np.ndarray:
    """Builds a (cells, 2) classification array placing each probability in the second column."""
    return np.array([[1.0, probability] for probability in probabilities], dtype=np.float32)


def _make_context(
    tmp_path: Path,
    recording_id: str,
    *,
    rois: list[ROIStatistics],
    classification: np.ndarray,
    rois_channel_2: list[ROIStatistics] | None = None,
    classification_channel_2: np.ndarray | None = None,
    persist: bool = True,
    data_path_none: bool = False,
    combined_data_none: bool = False,
    repeat_selection: bool = False,
    probability_threshold: float = 0.5,
    maximum_size: int = 10000,
    mroi_region_margin: int = 0,
    mroi_region_borders: tuple[int, ...] = (),
    selected_roi_indices: tuple[int, ...] = (),
    selected_roi_indices_channel_2: tuple[int, ...] = (),
) -> MultiRecordingRuntimeContext:
    """Builds a MultiRecordingRuntimeContext with combined data either saved to disk or held in memory.

    Args:
        tmp_path: The pytest temporary directory used to host the recording tree.
        recording_id: The identifier used for the recording and its output directory.
        rois: The channel 1 ROIStatistics list stored in the combined data.
        classification: The channel 1 classification array stored in the combined data.
        rois_channel_2: The optional channel 2 ROIStatistics list to enable two-channel selection.
        classification_channel_2: The optional channel 2 classification array.
        persist: When True, saves the combined data to disk and attaches a metadata-only copy that
            forces on-demand memory mapping during selection.
        data_path_none: When True (with persist False), leaves the runtime data path unset so the
            memory-mapping branch is skipped and selection runs on the in-memory combined data.
        combined_data_none: When True, attaches no combined data to exercise the missing-data guard.
        repeat_selection: The value assigned to the configuration repeat_selection flag.
        probability_threshold: The minimum classifier probability required for selection.
        maximum_size: The maximum allowed ROI pixel count.
        mroi_region_margin: The minimum distance between an ROI centroid and an MROI border.
        mroi_region_borders: The x-coordinates of MROI region borders.
        selected_roi_indices: The pre-existing channel 1 selection used by the skip guard.
        selected_roi_indices_channel_2: The pre-existing channel 2 selection used by the skip guard.

    Returns:
        The assembled MultiRecordingRuntimeContext ready to pass to select_recording_rois.
    """
    base = tmp_path / recording_id / "cindra"
    output_directory = base / "multi_recording" / "dataset"
    ensure_directory_exists(output_directory)

    runtime_combined: CombinedData | None = None
    data_path: Path | None = None
    if not combined_data_none:
        extraction = ExtractionData()
        extraction.roi_statistics = rois
        extraction.cell_classification = classification
        if rois_channel_2 is not None:
            extraction.roi_statistics_channel_2 = rois_channel_2
            extraction.cell_classification_channel_2 = classification_channel_2
        combined = CombinedData(detection=DetectionData(), extraction=extraction)

        if persist:
            ensure_directory_exists(base)
            combined.save(root_path=base)
            data_path = base
            # Loads a metadata-only copy so select_recording_rois memory-maps arrays from disk.
            runtime_combined = CombinedData.load(root_path=base)
        else:
            runtime_combined = combined
            data_path = None if data_path_none else base

    runtime = MultiRecordingRuntimeData()
    runtime.output_path = output_directory
    runtime.io.recording_id = recording_id
    runtime.io.dataset_name = "dataset"
    runtime.io.data_path = data_path
    runtime.io.mroi_region_borders = mroi_region_borders
    runtime.io.selected_roi_indices = selected_roi_indices
    runtime.io.selected_roi_indices_channel_2 = selected_roi_indices_channel_2
    runtime.combined_data = runtime_combined

    configuration = MultiRecordingConfiguration()
    configuration.recording_io.repeat_selection = repeat_selection
    configuration.roi_selection.probability_threshold = probability_threshold
    configuration.roi_selection.maximum_size = maximum_size
    configuration.roi_selection.mroi_region_margin = mroi_region_margin
    configuration.runtime.parallel_workers = 1

    return MultiRecordingRuntimeContext(configuration=configuration, runtime=runtime)


class TestSelectRecordingRois:
    """Tests select_recording_rois."""

    def test_empty_contexts_returns_without_error(self) -> None:
        """Verifies that an empty context list returns without raising."""
        select_recording_rois(contexts=[])

    def test_probability_filter_selects_expected_subset(self, tmp_path: Path) -> None:
        """Verifies that channel 1 ROIs below the probability threshold are excluded from selection."""
        rois = [_make_roi() for _ in range(4)]
        classification = _classification([0.9, 0.4, 0.95, 0.2])
        context = _make_context(tmp_path, "rec0", rois=rois, classification=classification, probability_threshold=0.5)

        select_recording_rois(contexts=[context])

        assert context.runtime.io.selected_roi_indices == (0, 2)
        assert context.runtime.io.selected_roi_indices_channel_2 == ()
        assert context.runtime.output_path is not None
        assert (context.runtime.output_path / "multi_recording_runtime_data.yaml").exists()

    def test_size_filter_excludes_large_rois(self, tmp_path: Path) -> None:
        """Verifies that ROIs exceeding the maximum pixel count are excluded from selection."""
        rois = [_make_roi(pixel_count=10), _make_roi(pixel_count=100), _make_roi(pixel_count=150)]
        classification = _classification([0.9, 0.9, 0.9])
        context = _make_context(
            tmp_path, "rec0", rois=rois, classification=classification, probability_threshold=0.0, maximum_size=100
        )

        select_recording_rois(contexts=[context])

        # The maximum size is inclusive, so the 150-pixel ROI is the only one excluded.
        assert context.runtime.io.selected_roi_indices == (0, 1)

    def test_mroi_border_filter_excludes_near_border(self, tmp_path: Path) -> None:
        """Verifies that ROIs whose centroid is within the margin of an MROI border are excluded."""
        rois = [_make_roi(centroid=(10, 50)), _make_roi(centroid=(10, 100))]
        classification = _classification([0.9, 0.9])
        context = _make_context(
            tmp_path,
            "rec0",
            rois=rois,
            classification=classification,
            probability_threshold=0.0,
            mroi_region_margin=10,
            mroi_region_borders=(50,),
        )

        select_recording_rois(contexts=[context])

        # The first ROI sits on the border (distance 0 < margin 10) and is dropped.
        assert context.runtime.io.selected_roi_indices == (1,)

    def test_channel_2_selection_runs_independently(self, tmp_path: Path) -> None:
        """Verifies that channel 2 ROIs are selected independently when channel 2 data is present."""
        rois = [_make_roi() for _ in range(3)]
        classification = _classification([0.9, 0.4, 0.95])
        rois_channel_2 = [_make_roi() for _ in range(3)]
        classification_channel_2 = _classification([0.9, 0.2, 0.8])
        context = _make_context(
            tmp_path,
            "rec0",
            rois=rois,
            classification=classification,
            rois_channel_2=rois_channel_2,
            classification_channel_2=classification_channel_2,
            probability_threshold=0.5,
        )

        select_recording_rois(contexts=[context])

        assert context.runtime.io.selected_roi_indices == (0, 2)
        assert context.runtime.io.selected_roi_indices_channel_2 == (0, 2)

    def test_in_memory_combined_data_skips_memory_mapping(self, tmp_path: Path) -> None:
        """Verifies that selection runs on in-memory combined data when the runtime data path is unset."""
        rois = [_make_roi() for _ in range(3)]
        classification = _classification([0.9, 0.3, 0.8])
        context = _make_context(
            tmp_path,
            "rec0",
            rois=rois,
            classification=classification,
            persist=False,
            data_path_none=True,
            probability_threshold=0.5,
        )

        select_recording_rois(contexts=[context])

        assert context.runtime.io.selected_roi_indices == (0, 2)

    def test_skips_recording_with_existing_channel_1_selection(self, tmp_path: Path) -> None:
        """Verifies that a recording with an existing channel 1 selection is skipped when repeat is disabled."""
        rois = [_make_roi() for _ in range(3)]
        classification = _classification([0.9, 0.9, 0.9])
        context = _make_context(
            tmp_path,
            "rec0",
            rois=rois,
            classification=classification,
            persist=False,
            data_path_none=True,
            selected_roi_indices=(99,),
        )

        select_recording_rois(contexts=[context])

        # The sentinel selection survives untouched, proving the filter did not run.
        assert context.runtime.io.selected_roi_indices == (99,)
        assert context.runtime.output_path is not None
        assert not (context.runtime.output_path / "multi_recording_runtime_data.yaml").exists()

    def test_skips_recording_with_existing_channel_2_selection(self, tmp_path: Path) -> None:
        """Verifies that a two-channel recording with existing selections for both channels is skipped."""
        rois = [_make_roi() for _ in range(3)]
        classification = _classification([0.9, 0.9, 0.9])
        rois_channel_2 = [_make_roi() for _ in range(3)]
        classification_channel_2 = _classification([0.9, 0.9, 0.9])
        context = _make_context(
            tmp_path,
            "rec0",
            rois=rois,
            classification=classification,
            rois_channel_2=rois_channel_2,
            classification_channel_2=classification_channel_2,
            persist=False,
            data_path_none=True,
            selected_roi_indices=(99,),
            selected_roi_indices_channel_2=(88,),
        )

        select_recording_rois(contexts=[context])

        assert context.runtime.io.selected_roi_indices == (99,)
        assert context.runtime.io.selected_roi_indices_channel_2 == (88,)

    def test_repeat_selection_reruns_existing_selection(self, tmp_path: Path) -> None:
        """Verifies that repeat_selection re-runs filtering and overwrites an existing selection."""
        rois = [_make_roi() for _ in range(4)]
        classification = _classification([0.9, 0.4, 0.95, 0.2])
        context = _make_context(
            tmp_path,
            "rec0",
            rois=rois,
            classification=classification,
            repeat_selection=True,
            probability_threshold=0.5,
            selected_roi_indices=(99,),
        )

        select_recording_rois(contexts=[context])

        # The stale sentinel is replaced by the freshly computed selection.
        assert context.runtime.io.selected_roi_indices == (0, 2)

    def test_raises_when_combined_data_missing(self, tmp_path: Path) -> None:
        """Verifies that a recording without combined data raises a ValueError during selection."""
        context = _make_context(
            tmp_path, "rec0", rois=[_make_roi()], classification=_classification([0.9]), combined_data_none=True
        )

        with pytest.raises(ValueError, match="Unable to select ROIs"):
            select_recording_rois(contexts=[context])

    def test_processes_multiple_contexts(self, tmp_path: Path) -> None:
        """Verifies that every context in the list is processed with independent selections."""
        context_one = _make_context(
            tmp_path, "rec0", rois=[_make_roi() for _ in range(3)], classification=_classification([0.9, 0.4, 0.95])
        )
        context_two = _make_context(
            tmp_path, "rec1", rois=[_make_roi() for _ in range(3)], classification=_classification([0.2, 0.9, 0.95])
        )

        select_recording_rois(contexts=[context_one, context_two])

        assert context_one.runtime.io.selected_roi_indices == (0, 2)
        assert context_two.runtime.io.selected_roi_indices == (1, 2)
