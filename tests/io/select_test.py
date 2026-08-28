"""Contains tests for the select module."""

from __future__ import annotations

import numpy as np

from cindra.layout import (
    OUTPUT_DIRECTORY_NAME,
    MULTI_RECORDING_DIRECTORY_NAME,
    MULTI_RECORDING_RUNTIME_DATA_FILENAME,
)
from cindra.io.select import _filter_channel_rois, clear_dataset_selection, clear_recording_selections
from cindra.dataclasses import ROIMask, ROIStatistics, MultiRecordingRuntimeData


class TestFilterChannelRois:
    """Tests _filter_channel_rois."""

    def test_all_pass(self) -> None:
        """Verifies that all ROIs pass when no filters are restrictive."""
        rois = [_make_roi() for _ in range(3)]
        classification = np.ones((3, 2), dtype=np.float32)
        result = _filter_channel_rois(
            roi_statistics=rois,
            cell_classification=classification,
            mroi_region_borders=(),
            probability_threshold=0.0,
            maximum_size=10000,
            region_margin=0,
        )
        assert result == (0, 1, 2)

    def test_probability_filter(self) -> None:
        """Verifies that ROIs below the probability threshold are excluded."""
        rois = [_make_roi() for _ in range(3)]
        classification = np.array([[0.9, 1.0], [0.3, 0.0], [0.8, 1.0]], dtype=np.float32)
        result = _filter_channel_rois(
            roi_statistics=rois,
            cell_classification=classification,
            mroi_region_borders=(),
            probability_threshold=0.5,
            maximum_size=10000,
            region_margin=0,
        )
        assert result == (0, 2)

    def test_size_filter(self) -> None:
        """Verifies that ROIs exceeding maximum size are excluded while the boundary value is retained."""
        rois = [_make_roi(pixel_count=10), _make_roi(pixel_count=100), _make_roi(pixel_count=150)]
        classification = np.ones((3, 2), dtype=np.float32)
        result = _filter_channel_rois(
            roi_statistics=rois,
            cell_classification=classification,
            mroi_region_borders=(),
            probability_threshold=0.0,
            maximum_size=100,
            region_margin=0,
        )
        # maximum_size is inclusive: only pixel_count > maximum_size is excluded, so 10 and the boundary 100 pass.
        assert result == (0, 1)

    def test_mroi_border_filter(self) -> None:
        """Verifies that ROIs near MROI region borders are excluded."""
        rois = [
            _make_roi(centroid=(10, 50)),  # Near border at x=50.
            _make_roi(centroid=(10, 100)),  # Far from border.
        ]
        classification = np.ones((2, 2), dtype=np.float32)
        result = _filter_channel_rois(
            roi_statistics=rois,
            cell_classification=classification,
            mroi_region_borders=(50,),
            probability_threshold=0.0,
            maximum_size=10000,
            region_margin=10,
        )
        # ROI at x=50 is exactly at the border (distance=0 < 10), excluded.
        assert 0 not in result
        assert 1 in result

    def test_no_mroi_borders(self) -> None:
        """Verifies that empty MROI borders disable the border filter."""
        rois = [_make_roi(centroid=(10, 0))]
        classification = np.ones((1, 2), dtype=np.float32)
        result = _filter_channel_rois(
            roi_statistics=rois,
            cell_classification=classification,
            mroi_region_borders=(),
            probability_threshold=0.0,
            maximum_size=10000,
            region_margin=100,
        )
        assert result == (0,)

    def test_empty_rois(self) -> None:
        """Verifies that empty input produces empty output."""
        classification = np.empty((0, 2), dtype=np.float32)
        result = _filter_channel_rois(
            roi_statistics=[],
            cell_classification=classification,
            mroi_region_borders=(),
            probability_threshold=0.0,
            maximum_size=10000,
            region_margin=0,
        )
        assert result == ()


def _make_roi(
    centroid: tuple[int, int] = (10, 10),
    pixel_count: int = 50,
) -> ROIStatistics:
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


class TestClearSelections:
    """Tests the selection clearing that returns a dataset to an unselected state."""

    def test_clearing_removes_both_channel_selections(self, tmp_path):
        """Verifies that clearing empties both channel selections while the recording identity stays in place."""
        dataset_path = _write_dataset_runtime(tmp_path=tmp_path, indices=(0, 3, 7), channel_2_indices=(1, 2))

        assert clear_dataset_selection(dataset_path=dataset_path)

        runtime_data = MultiRecordingRuntimeData.from_yaml(
            file_path=dataset_path / MULTI_RECORDING_RUNTIME_DATA_FILENAME
        )
        assert runtime_data.io.selected_roi_indices == ()
        assert runtime_data.io.selected_roi_indices_channel_2 == ()
        assert runtime_data.io.recording_id == "recording_a"

    def test_clearing_an_unselected_dataset_reports_no_change(self, tmp_path):
        """Verifies that a dataset holding no selection is left untouched."""
        dataset_path = _write_dataset_runtime(tmp_path=tmp_path, indices=())

        assert not clear_dataset_selection(dataset_path=dataset_path)

    def test_clearing_a_directory_without_runtime_data_reports_no_change(self, tmp_path):
        """Verifies that a dataset directory carrying no runtime data file is skipped."""
        dataset_path = tmp_path / "empty_dataset"
        dataset_path.mkdir()

        assert not clear_dataset_selection(dataset_path=dataset_path)

    def test_recording_clearing_covers_every_dataset(self, tmp_path):
        """Verifies that the recording-level sweep clears the selection of each dataset it holds."""
        cindra_root = tmp_path / OUTPUT_DIRECTORY_NAME
        for name in ("dataset_a", "dataset_b"):
            _write_dataset_runtime(
                tmp_path=cindra_root / MULTI_RECORDING_DIRECTORY_NAME, indices=(0, 1), dataset_name=name
            )

        assert clear_recording_selections(cindra_root=cindra_root) == 2

    def test_recording_clearing_without_datasets_reports_zero(self, tmp_path):
        """Verifies that a recording belonging to no dataset reports nothing cleared."""
        assert clear_recording_selections(cindra_root=tmp_path) == 0


def _write_dataset_runtime(tmp_path, indices, channel_2_indices=(), dataset_name="dataset_a"):
    """Writes one dataset runtime data file carrying the requested selections and returns its directory."""
    dataset_path = tmp_path / dataset_name
    dataset_path.mkdir(parents=True)
    runtime_data = MultiRecordingRuntimeData()
    runtime_data.io.recording_id = "recording_a"
    runtime_data.io.selected_roi_indices = tuple(indices)
    runtime_data.io.selected_roi_indices_channel_2 = tuple(channel_2_indices)
    runtime_data.to_yaml(file_path=dataset_path / MULTI_RECORDING_RUNTIME_DATA_FILENAME)
    return dataset_path
