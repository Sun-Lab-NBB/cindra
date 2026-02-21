"""Provides centralized style constants and font factories for GUI widgets."""

from PySide6 import QtGui

# Stylesheet applied to the main window background.
MAIN_WINDOW_STYLESHEET: str = "QMainWindow {background: 'black';}"

# Stylesheet for a button in the pressed (active/selected) state.
BUTTON_PRESSED_STYLESHEET: str = "QPushButton {Text-align: left; background-color: rgb(100,50,100); color:white;}"

# Stylesheet for a button in the unpressed (enabled, not selected) state.
BUTTON_UNPRESSED_STYLESHEET: str = "QPushButton {Text-align: left; background-color: rgb(50,50,50); color:white;}"

# Stylesheet for a button in the inactive (disabled/grayed-out) state.
BUTTON_INACTIVE_STYLESHEET: str = "QPushButton {Text-align: left; background-color: rgb(50,50,50); color:gray;}"

# Stylesheet for white label text on a dark background.
WHITE_LABEL_STYLESHEET: str = "color: white;"

# Stylesheet for red label text (used for neuropil trace indicators).
RED_LABEL_STYLESHEET: str = "color: red;"

# Stylesheet for cyan label text (used for raw fluorescence trace indicators).
CYAN_LABEL_STYLESHEET: str = "color: cyan;"

# Stylesheet for the dual-handle range slider used in saturation and visualization controls.
RANGE_SLIDER_STYLESHEET: str = (
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

# Width for small numeric input fields (top-n count, diameter, max-cells).
SMALL_EDIT_WIDTH: int = 35

# Width for ROI index input fields.
ROI_EDIT_WIDTH: int = 45

# Width for medium input fields (PC number, button labels in drawroi).
MEDIUM_EDIT_WIDTH: int = 60

# Width for parameter input fields in settings dialogs.
PARAMETER_EDIT_WIDTH: int = 90

# Width for the activity mode combo box in the trace panel.
COMBO_BOX_WIDTH: int = 100

# Maximum width for small square buttons (quadrant navigation, trace arrows).
SQUARE_BUTTON_MAX_WIDTH: int = 22

# Maximum height for the colorbar widget.
COLORBAR_MAX_HEIGHT: int = 60

# Maximum width for the colorbar widget.
COLORBAR_MAX_WIDTH: int = 150

# Standard font family used throughout the GUI.
_FONT_FAMILY: str = "Arial"

# Alternative font family used for colorbar and merge dialog labels.
_ALT_FONT_FAMILY: str = "Times"


def label_font() -> QtGui.QFont:
    """Creates the standard label font (Arial 8pt)."""
    return QtGui.QFont(_FONT_FAMILY, pointSize=8)


def label_font_bold() -> QtGui.QFont:
    """Creates the standard bold label font (Arial 8pt bold)."""
    return QtGui.QFont(_FONT_FAMILY, pointSize=8, weight=QtGui.QFont.Weight.Bold)


def header_font() -> QtGui.QFont:
    """Creates the section header font (Arial 10pt bold)."""
    return QtGui.QFont(_FONT_FAMILY, pointSize=10, weight=QtGui.QFont.Weight.Bold)


def arrow_button_font() -> QtGui.QFont:
    """Creates the font for trace expand/collapse arrow buttons (Arial 11pt bold)."""
    return QtGui.QFont(_FONT_FAMILY, pointSize=11, weight=QtGui.QFont.Weight.Bold)


def colorbar_font() -> QtGui.QFont:
    """Creates the font for colorbar tick labels (Times 8pt bold)."""
    return QtGui.QFont(_ALT_FONT_FAMILY, pointSize=8, weight=QtGui.QFont.Weight.Bold)


def merge_label_font() -> QtGui.QFont:
    """Creates the font for merge dialog parameter labels (Times bold)."""
    return QtGui.QFont(_ALT_FONT_FAMILY, weight=QtGui.QFont.Weight.Bold)
