"""Contains tests for the multi-recording tracking viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

import pytest

from cindra.gui.viewer_context import ViewerData
from cindra.gui.tracking_viewer import TrackingViewer

if TYPE_CHECKING:
    from collections.abc import Callable

    from PySide6.QtWidgets import QApplication


@pytest.mark.xdist_group("gui_viewers")
class TestTrackingViewerLoad:
    """Tests the data population step of the tracking viewer."""

    def test_recording_dropdown_opens_on_the_displayed_recording(
        self,
        qt_application: QApplication,
        monkeypatch: pytest.MonkeyPatch,
        tracked_viewer_data: Callable[..., ViewerData],
    ) -> None:
        """Verifies that the dropdown and the published state name the recording the viewer was launched from."""
        monkeypatch.setattr(TrackingViewer, "_refresh_display", lambda self: None)
        data = tracked_viewer_data()

        viewer = TrackingViewer(data=data)
        state = viewer.get_state()

        assert viewer._recording_combo.currentIndex() == 2
        assert state["current_recording_index"] == 2
        assert state["current_recording_id"] == "c"

        viewer.close()

    def test_dataset_free_data_is_rejected(
        self,
        qt_application: QApplication,
        single_recording_stub: Callable[..., object],
    ) -> None:
        """Verifies that opening the tracking viewer without a tracked dataset reports a readable failure."""
        data = ViewerData(single_recording=single_recording_stub(output_path=Path("/data/recording_c/cindra")))

        with pytest.raises(ValueError, match="Unable to populate the multi-recording tracking viewer"):
            TrackingViewer(data=data)
