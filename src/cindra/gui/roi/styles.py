"""Provides centralized style constants and font factories for the ROI viewer GUI."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtGui


@dataclass(frozen=True, slots=True)
class _ROIViewerStyle:
    """Encapsulates visual and dimensional constants for the ROI viewer window.

    Attributes:
        main_window: Stylesheet applied to the main window background.
        button_pressed: Stylesheet for a button in the pressed (active/selected) state.
        button_unpressed: Stylesheet for a button in the unpressed (enabled, not selected) state.
        button_inactive: Stylesheet for a button in the inactive (disabled/grayed-out) state.
        white_label: Stylesheet for white label text on a dark background.
        red_label: Stylesheet for red label text (used for neuropil trace indicators).
        cyan_label: Stylesheet for cyan label text (used for raw fluorescence trace indicators).
        range_slider: Stylesheet for the dual-handle range slider used in saturation controls.
        small_edit_width: Width for small numeric input fields (top-n count, diameter, max-cells).
        roi_edit_width: Width for ROI index input fields.
        medium_edit_width: Width for medium input fields (PC number, button labels in drawroi).
        parameter_edit_width: Width for parameter input fields in settings dialogs.
        combo_box_width: Width for the activity mode combo box in the trace panel.
        square_button_max_width: Maximum width for small square buttons (quadrant navigation, trace arrows).
        colorbar_max_height: Maximum height for the colorbar widget.
        colorbar_max_width: Maximum width for the colorbar widget.
        font_family: Standard font family used throughout the GUI.
        alt_font_family: Alternative font family used for colorbar and merge dialog labels.
    """

    main_window: str = "QMainWindow {background: 'black';}"
    button_pressed: str = "QPushButton {Text-align: left; background-color: rgb(100,50,100); color:white;}"
    button_unpressed: str = "QPushButton {Text-align: left; background-color: rgb(50,50,50); color:white;}"
    button_inactive: str = "QPushButton {Text-align: left; background-color: rgb(50,50,50); color:gray;}"
    white_label: str = "color: white;"
    red_label: str = "color: red;"
    cyan_label: str = "color: cyan;"
    range_slider: str = (
        "QSlider::handle:horizontal {"
        "background-color: white;"
        "border: 1px solid #5c5c5c;"
        "border-radius: 0px;"
        "border-color: black;"
        "height: 8px;"
        "width: 6px;"
        "margin: -8px 2;"
        "}"
    )
    small_edit_width: int = 35
    roi_edit_width: int = 45
    medium_edit_width: int = 60
    parameter_edit_width: int = 90
    combo_box_width: int = 100
    square_button_max_width: int = 22
    colorbar_max_height: int = 60
    colorbar_max_width: int = 150
    font_family: str = "Arial"
    alt_font_family: str = "Times"


STYLE: _ROIViewerStyle = _ROIViewerStyle()
"""Module-level singleton providing all ROI viewer style constants."""


def label_font() -> QtGui.QFont:
    """Creates the standard label font (Arial 8pt)."""
    return QtGui.QFont(STYLE.font_family, pointSize=8)


def label_font_bold() -> QtGui.QFont:
    """Creates the standard bold label font (Arial 8pt bold)."""
    return QtGui.QFont(STYLE.font_family, pointSize=8, weight=QtGui.QFont.Weight.Bold)


def header_font() -> QtGui.QFont:
    """Creates the section header font (Arial 10pt bold)."""
    return QtGui.QFont(STYLE.font_family, pointSize=10, weight=QtGui.QFont.Weight.Bold)


def arrow_button_font() -> QtGui.QFont:
    """Creates the font for trace expand/collapse arrow buttons (Arial 11pt bold)."""
    return QtGui.QFont(STYLE.font_family, pointSize=11, weight=QtGui.QFont.Weight.Bold)


def colorbar_font() -> QtGui.QFont:
    """Creates the font for colorbar tick labels (Times 8pt bold)."""
    return QtGui.QFont(STYLE.alt_font_family, pointSize=8, weight=QtGui.QFont.Weight.Bold)


def merge_label_font() -> QtGui.QFont:
    """Creates the font for merge dialog parameter labels (Times bold)."""
    return QtGui.QFont(STYLE.alt_font_family, weight=QtGui.QFont.Weight.Bold)
