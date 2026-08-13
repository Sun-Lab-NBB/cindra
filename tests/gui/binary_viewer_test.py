"""Contains tests for the registered binary recording viewer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from cindra.gui.binary_viewer import BinaryPlayer

if TYPE_CHECKING:
    from collections.abc import Callable

    from PySide6.QtWidgets import QApplication


@pytest.mark.xdist_group("gui_viewers")
class TestAverageOffsetCurves:
    """Tests the average rigid offset curves the viewer plots under the frame display."""

    def test_planes_longer_than_the_combined_movie_are_trimmed(
        self,
        qt_application: QApplication,
        registration_recording_stub: Callable[..., object],
    ) -> None:
        """Verifies that a plane longer than the combined movie still opens and averages over the combined frames."""
        data = registration_recording_stub(plane_frame_counts=(21, 20), combined_frame_count=20)

        viewer = BinaryPlayer(data=data)

        assert viewer._average_rigid_y_offsets.shape == (20,)
        assert viewer._average_rigid_x_offsets.shape == (20,)
        # Plane 0 contributes an offset of 1 and plane 1 an offset of 2 on every retained frame, and the x offsets
        # double both, so the two averages are 1.5 and 3.0.
        assert viewer._average_rigid_y_offsets == pytest.approx(1.5)
        assert viewer._average_rigid_x_offsets == pytest.approx(3.0)

        viewer.close()

    def test_equal_length_planes_average_over_every_frame(
        self,
        qt_application: QApplication,
        registration_recording_stub: Callable[..., object],
    ) -> None:
        """Verifies that planes of equal length keep contributing every frame to the average offset curves."""
        data = registration_recording_stub(plane_frame_counts=(20, 20), combined_frame_count=20)

        viewer = BinaryPlayer(data=data)

        assert viewer._average_rigid_y_offsets.shape == (20,)
        assert viewer._average_rigid_y_offsets == pytest.approx(1.5)

        viewer.close()
