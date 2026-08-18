"""Contains tests for the single-recording ROI viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cindra.gui.roi_viewer import ROIViewer
from cindra.gui.viewer_context import ViewerData

if TYPE_CHECKING:
    from collections.abc import Callable

    from PySide6.QtWidgets import QApplication

_ROI_COUNT: int = 4
"""The ROI count of the synthetic recording the viewer fixtures build."""

_CLASSIFIED_LABELS: list[float] = [1.0, 0.0, 0.0, 0.0]
"""The classification labels used by every test that is not about the initial selection itself."""


@pytest.mark.xdist_group("gui_viewers")
class TestInitialSelection:
    """Tests the ROI selection the viewer opens with."""

    def test_recording_without_any_classified_cell_opens_on_the_first_roi(
        self,
        qt_application: QApplication,
        single_recording_stub: Callable[..., object],
    ) -> None:
        """Verifies that labeling every ROI a non-cell opens the viewer instead of aborting the process."""
        viewer = _make_viewer(single_recording_stub=single_recording_stub, cell_labels=[0.0, 0.0, 0.0, 0.0])

        assert viewer._selected_roi_index == 0
        assert viewer._selected_roi_indices == [0]

        viewer.close()

    def test_first_classified_cell_is_selected(
        self,
        qt_application: QApplication,
        single_recording_stub: Callable[..., object],
    ) -> None:
        """Verifies that the viewer opens on the first ROI the classifier labeled as a cell."""
        viewer = _make_viewer(single_recording_stub=single_recording_stub, cell_labels=[0.0, 0.0, 1.0, 1.0])

        assert viewer._selected_roi_index == 2
        assert viewer._selected_roi_indices == [2]

        viewer.close()


@pytest.mark.xdist_group("gui_viewers")
class TestCorrelationCache:
    """Tests the correlation coloring cache the viewer builds while loading."""

    def test_binned_fluorescence_is_materialized_by_the_load(
        self,
        qt_application: QApplication,
        single_recording_stub: Callable[..., object],
    ) -> None:
        """Verifies that loading a recording fills the cache the activity correlation color mode reads."""
        viewer = _make_viewer(single_recording_stub=single_recording_stub, cell_labels=_CLASSIFIED_LABELS)

        assert viewer._binned_fluorescence is not None
        assert viewer._binned_fluorescence.shape[0] == _ROI_COUNT
        assert viewer._binned_fluorescence.shape[1] >= 1
        assert viewer._fluorescence_standard_deviation is not None
        assert viewer._fluorescence_standard_deviation.shape == (_ROI_COUNT,)

        viewer.close()


@pytest.mark.xdist_group("gui_viewers")
class TestRoiIndexField:
    """Tests the ROI index entry field."""

    def test_negative_entry_clamps_to_the_first_roi(
        self,
        qt_application: QApplication,
        single_recording_stub: Callable[..., object],
    ) -> None:
        """Verifies that a negative ROI number selects the first ROI instead of indexing past the list start."""
        viewer = _make_viewer(single_recording_stub=single_recording_stub, cell_labels=_CLASSIFIED_LABELS)

        viewer._roi_index_edit.setText("-500")
        viewer._on_number_chosen()

        assert viewer._selected_roi_index == 0
        assert viewer._selected_roi_indices == [0]

        viewer.close()

    def test_over_range_entry_clamps_to_the_last_roi(
        self,
        qt_application: QApplication,
        single_recording_stub: Callable[..., object],
    ) -> None:
        """Verifies that an ROI number above the ROI count selects the last ROI."""
        viewer = _make_viewer(single_recording_stub=single_recording_stub, cell_labels=_CLASSIFIED_LABELS)

        viewer._roi_index_edit.setText("999")
        viewer._on_number_chosen()

        assert viewer._selected_roi_index == _ROI_COUNT - 1
        assert viewer._selected_roi_indices == [_ROI_COUNT - 1]

        viewer.close()


@pytest.mark.xdist_group("gui_viewers")
class TestRankedSelection:
    """Tests the top-n and bottom-n ranked ROI selection."""

    def test_zero_count_selects_no_roi(
        self,
        qt_application: QApplication,
        single_recording_stub: Callable[..., object],
    ) -> None:
        """Verifies that a ranked count of zero clears the selection instead of selecting every ROI."""
        viewer = _make_viewer(single_recording_stub=single_recording_stub, cell_labels=_CLASSIFIED_LABELS)
        viewer._ranked_count_edit.setText("0")

        viewer._on_ranked_selection(top=True)
        assert viewer._selected_roi_indices == []

        viewer._on_ranked_selection(top=False)
        assert viewer._selected_roi_indices == []

        viewer.close()

    def test_positive_count_selects_the_requested_number_of_rois(
        self,
        qt_application: QApplication,
        single_recording_stub: Callable[..., object],
    ) -> None:
        """Verifies that the ranked selection returns as many ROIs as the field requests."""
        viewer = _make_viewer(single_recording_stub=single_recording_stub, cell_labels=_CLASSIFIED_LABELS)
        viewer._ranked_count_edit.setText("2")

        viewer._on_ranked_selection(top=True)
        assert len(viewer._selected_roi_indices) == 2

        viewer._on_ranked_selection(top=False)
        assert len(viewer._selected_roi_indices) == 2

        viewer.close()


def _make_viewer(single_recording_stub: Callable[..., object], cell_labels: list[float]) -> ROIViewer:
    """Builds an ROI viewer over a synthetic single recording carrying the requested classification labels."""
    data = ViewerData(single_recording=single_recording_stub(cell_labels=cell_labels, roi_count=_ROI_COUNT))
    return ROIViewer(data=data)
