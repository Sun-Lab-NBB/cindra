"""Provides selection, size-toggle, and quadrant-zoom button widgets for the main GUI."""

from __future__ import annotations

from typing import TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
from PySide6 import QtGui, QtCore
from PySide6.QtWidgets import QLabel, QGroupBox, QLineEdit, QGridLayout, QPushButton, QButtonGroup

from .styles import STYLE, label_font_bold

if TYPE_CHECKING:
    from collections.abc import Callable

    from PySide6.QtWidgets import QWidget

    from .signals import GUISignals


_STRETCH_FACTOR: int = 100
"""The layout stretch factor applied when toggling single-panel vs dual-panel view."""

_MAX_TOP_N: int = 500
"""The maximum number of cells allowed in the top-n / bottom-n selection input."""

_DEFAULT_TOP_N: int = 40
"""The default number of cells selected by top-n / bottom-n."""

_QUADRANT_ZOOM_MARGIN: float = 0.15
"""The margin fraction added to quadrant zoom ranges to provide padding."""

_QUADRANT_COLUMNS: int = 3
"""The number of columns in the quadrant grid."""

_VIEW_CELLS_ONLY: int = 0
"""The view mode index for displaying cells only."""

_VIEW_BOTH: int = 1
"""The view mode index for displaying both cells and non-cells."""

_VIEW_NONCELLS_ONLY: int = 2
"""The view mode index for displaying non-cells only."""


@dataclass
class SelectionControls:
    """Holds references to cell selection widgets and their mutable state.

    Attributes:
        selection_buttons: Button group with draw/top-n/bottom-n selection modes.
        top_count_edit: Text input for the number of top/bottom cells to select.
        top_count: Current top-n/bottom-n count value.
    """

    selection_buttons: QButtonGroup
    top_count_edit: QLineEdit
    top_count: int = _DEFAULT_TOP_N


@dataclass
class CellToggleControls:
    """Holds references to cell/non-cell/both toggle widgets and ROI count labels.

    Attributes:
        size_buttons: Button group with cells/both/non-cells toggle modes.
        cell_count_label: Label showing the number of classified cells.
        noncell_count_label: Label showing the number of non-cells.
    """

    size_buttons: QButtonGroup
    cell_count_label: QLabel
    noncell_count_label: QLabel


@dataclass
class QuadrantControls:
    """Holds references to quadrant zoom navigation widgets.

    Attributes:
        quadrant_buttons: Button group with the 3x3 quadrant zoom buttons.
    """

    quadrant_buttons: QButtonGroup


def create_selection_buttons(
    owner: QWidget,
    signals: GUISignals,
) -> tuple[QGroupBox, SelectionControls]:
    """Creates the cell selection buttons and the top-n input field.

    Builds a group box containing a button group with three selection modes (draw selection,
    select top n, select bottom n) and a numeric input field for specifying how many cells
    to select.

    Args:
        owner: The parent widget for ownership of created widgets.
        signals: The central signal bus for GUI events.

    Returns:
        Tuple of (group box, selection controls container).
    """
    group_box = QGroupBox("Cell Selection")
    group_box.setStyleSheet("QGroupBox { color: white; }")
    layout = QGridLayout(group_box)

    selection_buttons = QButtonGroup()

    labels = [" draw selection", " select top n", " select bottom n"]
    for button_index in range(3):
        button = _SelectionButton(
            button_id=button_index,
            text=labels[button_index],
            owner=owner,
            button_group=selection_buttons,
            signals=signals,
        )
        selection_buttons.addButton(button, button_index)
        layout.addWidget(button, button_index, 0, 1, 1)
        button.setEnabled(False)
    selection_buttons.setExclusive(True)

    count_label = QLabel("n=")
    count_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
    count_label.setStyleSheet(STYLE.white_label)
    count_label.setFont(label_font_bold())
    layout.addWidget(count_label, 1, 1, 1, 1)

    controls = SelectionControls(
        selection_buttons=selection_buttons,
        top_count_edit=QLineEdit(owner),
    )

    controls.top_count_edit.setValidator(QtGui.QIntValidator(0, _MAX_TOP_N))
    controls.top_count_edit.setText(str(_DEFAULT_TOP_N))
    controls.top_count_edit.setFixedWidth(STYLE.small_edit_width)
    controls.top_count_edit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
    controls.top_count_edit.returnPressed.connect(signals.roi_selection_changed.emit)
    layout.addWidget(controls.top_count_edit, 2, 1, 1, 1)

    return group_box, controls


def create_cell_toggle_buttons(
    owner: QWidget,
    signals: GUISignals,
) -> tuple[QGroupBox, CellToggleControls]:
    """Creates the cell / not-cell / both size-toggle buttons and ROI count labels.

    Builds a group box containing labels showing the number of cells and non-cells, plus a
    button group with three mutually exclusive view modes.

    Args:
        owner: The parent widget for ownership of created widgets.
        signals: The central signal bus for GUI events.

    Returns:
        Tuple of (group box, cell toggle controls container).
    """
    group_box = QGroupBox("View Toggle")
    group_box.setStyleSheet("QGroupBox { color: white; }")
    layout = QGridLayout(group_box)

    cell_count_label = QLabel("")
    cell_count_label.setStyleSheet(STYLE.white_label)
    cell_count_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
    layout.addWidget(cell_count_label, 0, 0, 1, 1)

    noncell_count_label = QLabel("")
    noncell_count_label.setStyleSheet(STYLE.white_label)
    layout.addWidget(noncell_count_label, 0, 2, 1, 1)

    size_buttons = QButtonGroup(owner)
    labels = [" cells", " both", " not cells"]
    for button_index, label_text in enumerate(labels):
        button = _SizeButton(
            button_id=button_index,
            text=label_text,
            owner=owner,
            button_group=size_buttons,
            signals=signals,
        )
        size_buttons.addButton(button, button_index)
        layout.addWidget(button, 1, button_index, 1, 1)
        button.setEnabled(button_index == _VIEW_BOTH)
    size_buttons.setExclusive(True)

    return group_box, CellToggleControls(
        size_buttons=size_buttons,
        cell_count_label=cell_count_label,
        noncell_count_label=noncell_count_label,
    )


def create_quadrant_buttons(
    owner: QWidget,
    signals: GUISignals,
) -> tuple[QGroupBox, QuadrantControls]:
    """Creates the 3x3 quadrant zoom navigation buttons.

    Builds a group box containing nine buttons that zoom the field-of-view plots to the
    corresponding ninth of the image.

    Args:
        owner: The parent widget for ownership of created widgets.
        signals: The central signal bus for GUI events.

    Returns:
        Tuple of (group box, quadrant controls container).
    """
    group_box = QGroupBox("Navigation")
    group_box.setStyleSheet("QGroupBox { color: white; }")
    layout = QGridLayout(group_box)

    quadrant_buttons = QButtonGroup(owner)
    for button_index in range(9):
        button = _QuadButton(
            button_id=button_index,
            text=" " + str(button_index + 1),
            owner=owner,
            button_group=quadrant_buttons,
            signals=signals,
        )
        quadrant_buttons.addButton(button, button_index)
        row = button_index // _QUADRANT_COLUMNS
        col = button_index % _QUADRANT_COLUMNS
        layout.addWidget(button, row, col, 1, 1)
        button.setEnabled(False)
    quadrant_buttons.setExclusive(True)

    return group_box, QuadrantControls(quadrant_buttons=quadrant_buttons)


def apply_quadrant_zoom(
    quadrant_buttons: QButtonGroup,
    button_index: int,
    frame_width: int,
    frame_height: int,
    set_view_range: Callable[[float, float, float, float], None],
) -> None:
    """Zooms the image views to the specified quadrant.

    Args:
        quadrant_buttons: The quadrant button group.
        button_index: Index of the pressed button.
        frame_width: Width of the field of view in pixels.
        frame_height: Height of the field of view in pixels.
        set_view_range: Callback that sets the x/y range on both image panels. Called
            with (x_min, x_max, y_min, y_max).
    """
    for index in range(9):
        if quadrant_buttons.button(index).isEnabled():
            quadrant_buttons.button(index).setStyleSheet(STYLE.button_unpressed)
    quadrant_buttons.button(button_index).setStyleSheet(STYLE.button_pressed)

    x_column = button_index % _QUADRANT_COLUMNS
    y_row = button_index // _QUADRANT_COLUMNS
    x_range = (
        np.array([x_column - _QUADRANT_ZOOM_MARGIN, x_column + 1 + _QUADRANT_ZOOM_MARGIN])
        * frame_width
        / _QUADRANT_COLUMNS
    )
    y_range = (
        np.array([y_row - _QUADRANT_ZOOM_MARGIN, y_row + 1 + _QUADRANT_ZOOM_MARGIN]) * frame_height / _QUADRANT_COLUMNS
    )

    set_view_range(float(x_range[0]), float(x_range[1]), float(y_range[0]), float(y_range[1]))


class _QuadButton(QPushButton):
    """Implements a quadrant zoom button for navigating field-of-view subregions.

    Each button maps to one of nine quadrant positions in a 3x3 grid. Pressing a button
    emits the roi_selection_changed signal for the orchestrator to handle.

    Args:
        button_id: Index identifying this button's grid position.
        text: Display label for the button.
        owner: The parent widget for ownership.
        button_group: The button group this button belongs to.
        signals: The central signal bus for GUI events.
    """

    def __init__(
        self,
        button_id: int,
        text: str,
        owner: QWidget,
        button_group: QButtonGroup,  # noqa: ARG002
        signals: GUISignals,
    ) -> None:
        super().__init__(owner)
        self.setText(text)
        self.setCheckable(True)
        self.setStyleSheet(STYLE.button_inactive)
        self.setFont(label_font_bold())
        self.resize(self.minimumSizeHint())
        self.setMaximumWidth(STYLE.small_edit_width)
        self._button_id: int = button_id
        self._signals = signals
        self.clicked.connect(self._press)
        self.show()

    def _press(self) -> None:
        """Emits the plot_needs_update signal for the orchestrator to handle zoom."""
        self._signals.plot_needs_update.emit()


class _SizeButton(QPushButton):
    """Implements a view size toggle button for switching between cell/both/non-cell panels.

    Controls which image panels are visible. Pressing emits the plot_needs_update signal
    so the orchestrator can apply the panel layout change.

    Args:
        button_id: View mode index (0=cells only, 1=both, 2=non-cells only).
        text: Display label for the button.
        owner: The parent widget for ownership.
        button_group: The button group this button belongs to.
        signals: The central signal bus for GUI events.

    Attributes:
        _button_id: Cached view mode index.
    """

    def __init__(
        self,
        button_id: int,
        text: str,
        owner: QWidget,
        button_group: QButtonGroup,  # noqa: ARG002
        signals: GUISignals,
    ) -> None:
        super().__init__(owner)
        self.setText(text)
        self.setCheckable(True)
        self.setStyleSheet(STYLE.button_inactive)
        self.setFont(label_font_bold())
        self.resize(self.minimumSizeHint())
        self._button_id: int = button_id
        self._signals = signals
        self.clicked.connect(self._press)
        self.show()

    def _press(self) -> None:
        """Emits the plot_needs_update signal for the orchestrator to handle panel toggling."""
        self._signals.plot_needs_update.emit()


class _SelectionButton(QPushButton):
    """Implements a cell selection mode button (draw, top-n, bottom-n).

    Controls the cell selection behavior in the main viewer. Pressing emits the
    roi_selection_changed signal so the orchestrator can activate the appropriate mode.

    Args:
        button_id: Selection mode index (0=draw, 1=top n, 2=bottom n).
        text: Display label for the button.
        owner: The parent widget for ownership.
        button_group: The button group this button belongs to.
        signals: The central signal bus for GUI events.

    Attributes:
        _button_id: Cached selection mode index.
    """

    def __init__(
        self,
        button_id: int,
        text: str,
        owner: QWidget,
        button_group: QButtonGroup,  # noqa: ARG002
        signals: GUISignals,
    ) -> None:
        super().__init__(owner)
        self._button_id: int = button_id
        self.setText(text)
        self.setCheckable(True)
        self.setStyleSheet(STYLE.button_inactive)
        self.setFont(label_font_bold())
        self.resize(self.minimumSizeHint())
        self._signals = signals
        self.clicked.connect(self._press)
        self.show()

    def _press(self) -> None:
        """Emits the roi_selection_changed signal for the orchestrator to handle mode activation."""
        self._signals.roi_selection_changed.emit()
