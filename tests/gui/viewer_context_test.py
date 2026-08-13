"""Contains tests for the viewer data hierarchy shared by the GUI viewers."""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    from collections.abc import Callable

    from cindra.gui.viewer_context import ViewerData


class TestDatasetActivation:
    """Tests the dataset activation and deactivation transitions."""

    def test_reactivating_a_cached_dataset_restores_the_anchor_recording(
        self,
        tracked_viewer_data: Callable[..., ViewerData],
    ) -> None:
        """Verifies that the fast re-activation path re-derives the anchor the deactivation dropped."""
        data = tracked_viewer_data()

        data.unload_dataset()
        assert data.current_recording_index == 0

        data.load_dataset(dataset_name="d")

        assert data.current_recording_index == 2
        assert data.current_recording_id == "c"
        assert data.is_multi_recording

    def test_anchor_falls_back_to_the_first_recording(
        self,
        tracked_viewer_data: Callable[..., ViewerData],
    ) -> None:
        """Verifies that a dataset excluding the visualized recording opens on its first recording."""
        data = tracked_viewer_data(anchor_root=Path("/data/recording_z/cindra"))

        data.unload_dataset()
        data.load_dataset(dataset_name="d")

        assert data.current_recording_index == 0
        assert data.current_recording_id == "a"

    def test_unknown_dataset_leaves_the_selection_unchanged(
        self,
        tracked_viewer_data: Callable[..., ViewerData],
    ) -> None:
        """Verifies that requesting an undiscovered dataset name does not touch the recording selection."""
        data = tracked_viewer_data()

        data.load_dataset(dataset_name="missing")

        assert data.current_recording_index == 2
        assert data.active_dataset_name == "d"
