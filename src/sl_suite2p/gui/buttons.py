"""This module provides various reusable GUI button classes and setup functions for integrating them into the
sl-suite2p GUI.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from qtpy import QtGui, QtCore
import numpy as np
from qtpy.QtWidgets import QLabel, QLineEdit, QPushButton, QButtonGroup

# Import guard to avoid circular imports if needed
if TYPE_CHECKING:
    from .gui2p import MainWindow


def add_cell_selection_buttons(gui: MainWindow) -> None:
    """Creates the buttons that allow selecting a subset of visualized cells (ROIs) for further analysis.

    When pressed, the buttons generate a resizable rectangular selection box that can be used to visually sample the
    ROIs displayed in the GUI.

    Notes:
        This set of buttons is added to the top of the main GUI window.

    """
    gui.topbtns = QButtonGroup()

    label = QLabel("select cells")
    label.setStyleSheet("color: white;")
    label.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Weight.Bold))
    gui.l0.addWidget(label, 0, 2, 1, 2)

    positions = [2, 3, 4]
    for bid in range(3):
        btn = TopButton(bid, gui)
        btn.setFont(QtGui.QFont("Arial", 8))
        gui.topbtns.addButton(btn, bid)
        gui.l0.addWidget(btn, 0, positions[bid] * 2, 1, 2)
        btn.setEnabled(False)

    gui.topbtns.setExclusive(True)
    gui.isROI = False
    gui.ROIplot = 0

    n_label = QLabel("n=")
    n_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignVCenter)
    n_label.setStyleSheet("color: white;")
    n_label.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Weight.Bold))
    gui.l0.addWidget(n_label, 0, 10, 1, 1)

    gui.topedit = QLineEdit(gui)
    gui.topedit.setValidator(QtGui.QIntValidator(0, 500))
    gui.topedit.setText("40")
    gui.ntop = 40
    gui.topedit.setFixedWidth(35)
    gui.topedit.setAlignment(QtCore.Qt.AlignmentFlag.AlignRight)
    gui.topedit.returnPressed.connect(gui.top_number_chosen)
    gui.l0.addWidget(gui.topedit, 0, 11, 1, 1)


def make_cellnotcell(parent) -> None:
    """Create buttons for toggling cell / not-cell views."""
    parent.lcell0 = QLabel("")
    parent.lcell0.setStyleSheet("color: white;")
    parent.lcell0.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
    parent.l0.addWidget(parent.lcell0, 0, 12, 1, 2)

    parent.lcell1 = QLabel("")
    parent.lcell1.setStyleSheet("color: white;")
    parent.l0.addWidget(parent.lcell1, 0, 20, 1, 2)

    parent.sizebtns = QButtonGroup(parent)
    labels = [" cells", " both", " not cells"]

    for bid, text in enumerate(labels):
        btn = SizeButton(bid, text, parent)
        parent.sizebtns.addButton(btn, bid)
        parent.l0.addWidget(btn, 0, 14 + 2 * bid, 1, 2)
        btn.setEnabled(bid == 1)

    parent.sizebtns.setExclusive(True)


def make_quadrants(parent) -> None:
    """Create quadrant buttons for view selection."""
    parent.quadbtns = QButtonGroup(parent)
    for bid in range(9):
        btn = QuadButton(bid, f" {bid + 1}", parent)
        parent.quadbtns.addButton(btn, bid)
        parent.l0.addWidget(
            btn,
            0 + btn.ypos,
            29 + btn.xpos,
            1,
            1,
        )
        btn.setEnabled(False)

    parent.quadbtns.setExclusive(True)


class QuadButton(QPushButton):
    """Custom QPushButton for quadrant plotting.

    Allows selecting a quadrant to zoom into. Only one quadrant
    button can be active at a time within a QButtonGroup.
    """

    def __init__(self, bid: int, text: str, parent: object | None = None) -> None:
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setStyleSheet(parent.styleInactive)
        self.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.resize(self.minimumSizeHint())
        self.setMaximumWidth(22)

        self.xpos = bid % 3
        self.ypos = bid // 3

        self.clicked.connect(lambda: self.press(parent, bid))
        self.show()

    def press(self, parent, bid: int) -> None:
        """Handle quadrant selection and update plots."""
        for b in range(9):
            if parent.quadbtns.button(b).isEnabled():
                parent.quadbtns.button(b).setStyleSheet(parent.styleUnpressed)

        self.setStyleSheet(parent.stylePressed)

        self.xrange = np.array([self.xpos - 0.15, self.xpos + 1.15]) * parent.ops["Lx"] / 3
        self.yrange = np.array([self.ypos - 0.15, self.ypos + 1.15]) * parent.ops["Ly"] / 3

        parent.p1.setXRange(*self.xrange)
        parent.p1.setYRange(*self.yrange)
        parent.p2.setXRange(*self.xrange)
        parent.p2.setYRange(*self.yrange)

        parent.p2.setXLink("plot1")
        parent.p2.setYLink("plot1")
        parent.show()


class SizeButton(QPushButton):
    """Custom QPushButton to adjust trace box view size."""

    def __init__(self, bid: int, text: str, parent: object | None = None) -> None:
        super().__init__(parent)
        self.setText(text)
        self.setCheckable(True)
        self.setStyleSheet(parent.styleInactive)
        self.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.resize(self.minimumSizeHint())

        self.bid = bid
        self.clicked.connect(lambda: self.press(parent))
        self.show()

    def press(self, parent) -> None:
        """Update layout and selection button states when size is changed."""
        ts = 100

        if self.bid == 0:  # left-only view
            parent.p2.linkView(parent.p2.XAxis, view=None)
            parent.p2.linkView(parent.p2.YAxis, view=None)
            parent.win.ci.layout.setColumnStretchFactor(0, ts)
            parent.win.ci.layout.setColumnStretchFactor(1, 0)

        elif self.bid == 1:  # both views
            parent.win.ci.layout.setColumnStretchFactor(0, ts)
            parent.win.ci.layout.setColumnStretchFactor(1, ts)
            parent.p2.setXLink("plot1")
            parent.p2.setYLink("plot1")

        elif self.bid == 2:  # right-only view
            parent.p2.linkView(parent.p2.XAxis, view=None)
            parent.p2.linkView(parent.p2.YAxis, view=None)
            parent.win.ci.layout.setColumnStretchFactor(0, 0)
            parent.win.ci.layout.setColumnStretchFactor(1, ts)

        # Enable/disable selection buttons depending on view
        if self.bid != 1:
            if parent.ops_plot["color"] != 0:
                for btn in parent.topbtns.buttons():
                    btn.setStyleSheet(parent.styleUnpressed)
                    btn.setEnabled(True)
            else:
                parent.topbtns.button(0).setStyleSheet(parent.styleUnpressed)
                parent.topbtns.button(0).setEnabled(True)
        else:
            parent.ROI_remove()
            for btn in parent.topbtns.buttons():
                btn.setEnabled(False)
                btn.setStyleSheet(parent.styleInactive)

        parent.win.show()
        parent.show()


class TopButton(QPushButton):
    """Custom QPushButton for selecting top/bottom neurons."""

    def __init__(self, bid: int, parent: object | None = None) -> None:
        super().__init__(parent)

        labels = [" draw selection", " select top n", " select bottom n"]
        self.bid = bid
        self.setText(labels[bid])
        self.setCheckable(True)
        self.setStyleSheet(parent.styleInactive)
        self.setFont(QtGui.QFont("Arial", 8, QtGui.QFont.Bold))
        self.resize(self.minimumSizeHint())

        self.clicked.connect(lambda: self.press(parent))
        self.show()

    def press(self, parent) -> None:
        """Handle top-button press logic."""
        if not parent.sizebtns.button(1).isChecked():
            if parent.ops_plot["color"] == 0:
                for b in [1, 2]:
                    parent.topbtns.button(b).setEnabled(False)
                    parent.topbtns.button(b).setStyleSheet(parent.styleInactive)
            else:
                for b in [1, 2]:
                    parent.topbtns.button(b).setEnabled(True)
                    parent.topbtns.button(b).setStyleSheet(parent.styleUnpressed)
        else:
            for b in range(3):
                parent.topbtns.button(b).setEnabled(False)
                parent.topbtns.button(b).setStyleSheet(parent.styleInactive)

        self.setStyleSheet(parent.stylePressed)

        if self.bid == 0:
            parent.ROI_selection()
        else:
            self.top_selection(parent)

    def top_selection(self, parent) -> None:
        """Perform top/bottom cell selection and update plots."""
        parent.ROI_remove()
        draw = False
        ncells = len(parent.stat)
        icells = min(ncells, parent.ntop)

        top = self.bid == 1

        if parent.sizebtns.button(0).isChecked():
            wplot = 0
            draw = True
        elif parent.sizebtns.button(2).isChecked():
            wplot = 1
            draw = True

        if draw and parent.ops_plot["color"] != 0:
            c = parent.ops_plot["color"]
            istat = parent.colors["istat"][c]

            if wplot == 0:
                icell = np.array(parent.iscell.nonzero()).flatten()
                istat = istat[parent.iscell]
            else:
                icell = np.array((~parent.iscell).nonzero()).flatten()
                istat = istat[~parent.iscell]

            inds = istat.argsort()

            if top:
                inds = inds[-icells:]
                parent.ichosen = icell[inds[-1]]
            else:
                inds = inds[:icells]
                parent.ichosen = icell[inds[0]]

            parent.imerge = [icell[n] for n in inds]

            parent.update_plot()
            parent.show()
