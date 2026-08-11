"""Contains tests for the principal component registration metrics viewer."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from PySide6 import QtGui, QtCore
from PySide6.QtWidgets import QApplication

from cindra.gui.pc_viewer import PCViewer


def _arrow_event(key: QtCore.Qt.Key) -> QtGui.QKeyEvent:
    """Builds an unmodified key press event for the requested arrow key."""
    return QtGui.QKeyEvent(QtCore.QEvent.Type.KeyPress, key, QtCore.Qt.KeyboardModifier.NoModifier)


@pytest.mark.xdist_group("gui_viewers")
class TestPrincipalComponentField:
    """Tests the principal component number field reads."""

    def test_arrow_navigation_survives_an_emptied_field(
        self,
        qt_application: QApplication,
        principal_component_stub: Callable[..., object],
    ) -> None:
        """Verifies that stepping through principal components with an empty field resumes from the first one."""
        viewer = PCViewer(data=principal_component_stub())
        viewer._pc_edit.setText("")

        viewer.keyPressEvent(_arrow_event(key=QtCore.Qt.Key.Key_Right))
        assert viewer._pc_edit.text() == "2"

        viewer._pc_edit.setText("")
        viewer.keyPressEvent(_arrow_event(key=QtCore.Qt.Key.Key_Left))
        assert viewer._pc_edit.text() == "1"

        viewer.close()

    def test_state_and_redraw_read_the_same_clamped_number(
        self,
        qt_application: QApplication,
        principal_component_stub: Callable[..., object],
    ) -> None:
        """Verifies that an empty or over-range field resolves to a number inside the available range."""
        viewer = PCViewer(data=principal_component_stub(pc_count=3))

        viewer._pc_edit.setText("")
        assert viewer.get_state()["current_pc"] == 1
        viewer._plot_frame()

        viewer._pc_edit.setText("99")
        assert viewer.get_state()["current_pc"] == 3
        viewer._plot_frame()

        viewer.close()


@pytest.mark.xdist_group("gui_viewers")
class TestAnimatedPanelLabel:
    """Tests the caption of the animated principal component panel."""

    def test_redraw_labels_the_extreme_it_displays(
        self,
        qt_application: QApplication,
        principal_component_stub: Callable[..., object],
    ) -> None:
        """Verifies that a redraw after a pause on an odd tick captions the high-projection extreme as bottom."""
        viewer = PCViewer(data=principal_component_stub())

        viewer._current_frame = 0
        viewer._plot_frame()
        assert viewer._title_labels[2].toPlainText() == "top"

        viewer._current_frame = 1
        viewer._plot_frame()
        assert viewer._title_labels[2].toPlainText() == "bottom"

        viewer.close()
