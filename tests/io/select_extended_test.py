"""Contains tests for extended select module helper functions."""

from __future__ import annotations

import numpy as np
import pytest

from cindra.io.select import _filter_rois
from cindra.dataclasses import (
    ROIMask,
    CombinedData,
    DetectionData,
    ROIStatistics,
    ExtractionData,
    MultiRecordingRuntimeData,
    MultiRecordingConfiguration,
)


def _make_roi(centroid: tuple[int, int] = (20, 20), pixel_count: int = 50) -> ROIStatistics:
    """Creates a minimal ROIStatistics instance for testing."""
    y_pixels = np.arange(pixel_count, dtype=np.int32) % 10
    x_pixels = np.arange(pixel_count, dtype=np.int32) // 10
    mask = ROIMask(
        y_pixels=y_pixels,
        x_pixels=x_pixels,
        pixel_weights=np.ones(pixel_count, dtype=np.float32),
        centroid=centroid,
        frame_width=100,
    )
    roi = ROIStatistics(mask=mask)
    roi.pixel_count = pixel_count
    return roi


def _make_runtime_and_config(
    roi_count: int = 3,
    probabilities: list[float] | None = None,
    probability_threshold: float = 0.5,
    maximum_size: int = 10000,
) -> tuple[MultiRecordingRuntimeData, MultiRecordingConfiguration]:
    """Creates minimal MultiRecordingRuntimeData and MultiRecordingConfiguration for testing.

    Args:
        roi_count: The number of ROIs to create.
        probabilities: The probability values for each ROI. If None, all ROIs get probability 0.9.
        probability_threshold: The minimum probability threshold for ROI selection.
        maximum_size: The maximum allowed ROI size in pixels.

    Returns:
        A tuple of (runtime, configuration) instances with combined_data populated.
    """
    rois = [_make_roi(centroid=(20 + i * 15, 20 + i * 15)) for i in range(roi_count)]
    if probabilities is None:
        probabilities = [0.9] * roi_count
    classification = np.array(
        [[probability, 1.0 if probability > 0.5 else 0.0] for probability in probabilities], dtype=np.float32
    )

    extraction = ExtractionData()
    extraction.roi_statistics = rois
    extraction.cell_classification = classification

    combined_data = CombinedData(
        detection=DetectionData(),
        extraction=extraction,
    )

    runtime = MultiRecordingRuntimeData()
    runtime.io.recording_id = "test_recording"
    runtime.combined_data = combined_data

    configuration = MultiRecordingConfiguration()
    configuration.roi_selection.probability_threshold = probability_threshold
    configuration.roi_selection.maximum_size = maximum_size
    configuration.roi_selection.mroi_region_margin = 0

    return runtime, configuration


class TestFilterRois:
    """Tests _filter_rois."""

    def test_selects_all_rois_with_permissive_filters(self) -> None:
        """Verifies that all ROIs are selected when filters are permissive."""
        runtime, configuration = _make_runtime_and_config(
            roi_count=3,
            probability_threshold=0.0,
            maximum_size=100000,
        )

        channel_1_count, channel_2_count = _filter_rois(runtime=runtime, configuration=configuration)

        assert channel_1_count == 3
        assert channel_2_count == 0
        assert len(runtime.io.selected_roi_indices) == 3

    def test_probability_filter_excludes_low_probability_rois(self) -> None:
        """Verifies that ROIs below the probability threshold are excluded."""
        runtime, configuration = _make_runtime_and_config(
            roi_count=3,
            probabilities=[0.9, 0.3, 0.8],
            probability_threshold=0.5,
        )

        channel_1_count, _ = _filter_rois(runtime=runtime, configuration=configuration)

        assert channel_1_count == 2
        assert runtime.io.selected_roi_indices == (0, 2)

    def test_size_filter_excludes_large_rois(self) -> None:
        """Verifies that ROIs exceeding the maximum size are excluded."""
        runtime, configuration = _make_runtime_and_config(
            roi_count=2,
            probability_threshold=0.0,
            maximum_size=49,
        )

        # Default ROIs have pixel_count=50; maximum_size is inclusive, so 50 > 49 is excluded.
        channel_1_count, _ = _filter_rois(runtime=runtime, configuration=configuration)

        assert channel_1_count == 0

    def test_stores_indices_in_runtime(self) -> None:
        """Verifies that selected indices are stored in runtime.io.selected_roi_indices."""
        runtime, configuration = _make_runtime_and_config(
            roi_count=4,
            probabilities=[0.9, 0.1, 0.8, 0.05],
            probability_threshold=0.5,
        )

        _filter_rois(runtime=runtime, configuration=configuration)

        # Only ROIs at indices 0 and 2 have probability above 0.5.
        assert 0 in runtime.io.selected_roi_indices
        assert 2 in runtime.io.selected_roi_indices
        assert 1 not in runtime.io.selected_roi_indices
        assert 3 not in runtime.io.selected_roi_indices

    def test_filters_channel_2_rois_when_present(self) -> None:
        """Verifies that channel 2 ROIs are filtered independently when channel 2 data is present."""
        roi_count = 3
        runtime, configuration = _make_runtime_and_config(
            roi_count=roi_count,
            probability_threshold=0.5,
            maximum_size=10000,
        )

        # Adds channel 2 ROI statistics and classification to the combined data.
        channel_2_rois = [_make_roi(centroid=(30 + i * 10, 30 + i * 10)) for i in range(roi_count)]
        channel_2_classification = np.array([[0.9, 1.0], [0.2, 0.0], [0.8, 1.0]], dtype=np.float32)
        assert runtime.combined_data is not None
        runtime.combined_data.extraction.roi_statistics_channel_2 = channel_2_rois
        runtime.combined_data.extraction.cell_classification_channel_2 = channel_2_classification

        channel_1_count, channel_2_count = _filter_rois(runtime=runtime, configuration=configuration)

        # All channel 1 ROIs pass (all probabilities are 0.9).
        assert channel_1_count == roi_count
        # Channel 2: ROI at index 1 has probability 0.2 < 0.5, so excluded.
        assert channel_2_count == 2
        assert runtime.io.selected_roi_indices_channel_2 == (0, 2)

    def test_raises_error_when_combined_data_is_none(self) -> None:
        """Verifies that a ValueError is raised when combined_data is not available."""
        runtime = MultiRecordingRuntimeData()
        runtime.io.recording_id = "test_recording"
        runtime.combined_data = None

        configuration = MultiRecordingConfiguration()

        with pytest.raises(ValueError, match="Unable to select ROIs"):
            _filter_rois(runtime=runtime, configuration=configuration)

    def test_raises_error_when_roi_statistics_is_none(self) -> None:
        """Verifies that a ValueError is raised when combined_data has no ROI statistics."""
        extraction = ExtractionData()
        extraction.roi_statistics = None
        combined_data = CombinedData(detection=DetectionData(), extraction=extraction)

        runtime = MultiRecordingRuntimeData()
        runtime.io.recording_id = "test_recording"
        runtime.combined_data = combined_data

        configuration = MultiRecordingConfiguration()

        with pytest.raises(ValueError, match="does not contain ROI statistics"):
            _filter_rois(runtime=runtime, configuration=configuration)

    def test_raises_error_when_cell_classification_is_none(self) -> None:
        """Verifies that a ValueError is raised when combined_data has no classification results."""
        extraction = ExtractionData()
        extraction.roi_statistics = [_make_roi()]
        extraction.cell_classification = None
        combined_data = CombinedData(detection=DetectionData(), extraction=extraction)

        runtime = MultiRecordingRuntimeData()
        runtime.io.recording_id = "test_recording"
        runtime.combined_data = combined_data

        configuration = MultiRecordingConfiguration()

        with pytest.raises(ValueError, match="does not contain cell"):
            _filter_rois(runtime=runtime, configuration=configuration)

    def test_raises_error_when_channel_2_classification_missing(self) -> None:
        """Verifies that a ValueError is raised when channel 2 statistics exist but classification is missing."""
        roi_count = 2
        runtime, configuration = _make_runtime_and_config(
            roi_count=roi_count,
            probability_threshold=0.5,
        )

        # Adds channel 2 statistics but no classification.
        channel_2_rois = [_make_roi() for _ in range(roi_count)]
        assert runtime.combined_data is not None
        runtime.combined_data.extraction.roi_statistics_channel_2 = channel_2_rois
        runtime.combined_data.extraction.cell_classification_channel_2 = None

        with pytest.raises(ValueError, match="Unable to select channel 2 ROIs"):
            _filter_rois(runtime=runtime, configuration=configuration)
