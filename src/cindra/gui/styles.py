"""Provides visual style constants shared across all GUI viewer windows."""

from __future__ import annotations

from dataclasses import field, dataclass

from PySide6 import QtGui


@dataclass(frozen=True, slots=True)
class _CommonStyle:
    """Encapsulates static visual parameters shared by all viewer windows."""

    white_label: str = "color: white;"
    """The stylesheet for QLabel text that sits on the dark window background. Applied to status labels, axis labels,
    checkbox captions, and group headings across all viewer windows."""
    icon_size: int = 30
    """The pixel dimension (width and height) for QIcon-based media control buttons (play, pause, step-forward,
    step-backward) in the Binary, PC, and Tracking viewer playback toolbars."""
    default_mask_opacity: int = 127
    """The default mask overlay opacity (0-255 uint8 range). Determines the initial transparency of ROI masks rendered
    over background images in the ROI viewer and Tracking viewer."""
    group_box: str = "QGroupBox { color: white; }"
    """The stylesheet for QGroupBox title text. Applied to every QGroupBox across all viewer windows so section titles
    read clearly on the dark background."""
    edit_width: int = 50
    """The fixed pixel width for small numeric QLineEdit input fields across all viewer windows."""
    button_pressed: str = (
        "QPushButton { text-align: left; background-color: rgb(100,50,100); color: white; }"
        "QPushButton:disabled { background-color: rgb(50,50,50); color: gray; }"
    )
    """The stylesheet for a QPushButton in the pressed (active/selected) state with a disabled fallback."""
    button_unpressed: str = (
        "QPushButton { text-align: left; background-color: rgb(50,50,50); color: white; }"
        "QPushButton:disabled { background-color: rgb(50,50,50); color: gray; }"
    )
    """The stylesheet for a QPushButton in the unpressed (enabled but not selected) state with a disabled fallback."""
    menu: str = (
        "QMenu { background-color: rgb(50,50,50); color: white; }"
        "QMenu::item:selected { background-color: rgb(100,50,100); }"
    )
    """The stylesheet for QMenu dropdowns to match the dark viewer theme."""


@dataclass(frozen=True, slots=True)
class _PlotStyle:
    """Encapsulates static pyqtgraph plot parameters shared by all viewer windows."""

    scatter_point_size: int = 10
    """The marker diameter in pixels for pyqtgraph scatter plot overlays."""
    legend_headroom: float = 0.25
    """The fraction of the y-axis data range added as extra top padding when a plot contains a legend. Prevents
    legend entries from overlapping the topmost data points."""
    legend_horizontal_spacing: int = 20
    """The horizontal pixel spacing between adjacent legend entries in pyqtgraph plot widgets."""
    legend_offset: tuple[int, int] = (-10, 1)
    """The (x, y) pixel offset that positions the pyqtgraph legend relative to the top-right corner of the plot area.
    A negative x value pulls the legend inward from the right edge to avoid clipping."""
    left_axis_width: int = 80
    """The fixed pixel width for the left (y) axis in plot widgets. Provides consistent spacing between the axis label
    and tick labels across all viewers."""
    bottom_axis_height: int = 50
    """The fixed pixel height for the bottom (x) axis in plot widgets. Provides consistent spacing between the axis
    label and tick labels across all viewers."""


@dataclass(frozen=True, slots=True)
class _FontStyle:
    """Encapsulates static font and text size parameters shared by all viewer windows."""

    family: str = "Arial"
    """The standard font family used throughout the GUI."""

    plot_title_size: str = "14pt"
    """The font size for plot titles."""

    label_size: str = "12pt"
    """The font size for axis labels and legend entry labels."""

    small: QtGui.QFont = field(default_factory=lambda: QtGui.QFont("Arial", 14))
    """Small font (Arial 14pt) for general GUI widgets and the application default."""

    small_bold: QtGui.QFont = field(default_factory=lambda: QtGui.QFont("Arial", 14, QtGui.QFont.Weight.Bold.value))
    """Small bold font (Arial 14pt bold) for combo boxes, buttons, and overlay text."""

    large: QtGui.QFont = field(default_factory=lambda: QtGui.QFont("Arial", 16))
    """Large font (Arial 16pt) for prominent GUI input fields."""

    large_bold: QtGui.QFont = field(default_factory=lambda: QtGui.QFont("Arial", 16, QtGui.QFont.Weight.Bold.value))
    """Large bold font (Arial 16pt bold) for prominent GUI labels."""


@dataclass(frozen=True, slots=True)
class _ROIViewerStyle:
    """Encapsulates static visual parameters for the ROI viewer and editor windows."""

    colorbar_max_height: int = 60
    """The maximum height for the colorbar widget."""
    color_edit_width: int = 65
    """The width for color panel edit fields and the colormap combo box."""
    colorbar_sample_count: int = 101
    """The number of samples for the colorbar gradient."""
    colorbar_row_count: int = 20
    """The number of rows in the colorbar image."""
    window_geometry: tuple[int, int, int, int] = (50, 50, 1500, 800)
    """The initial window position (x, y) and size (width, height) for the ROI viewer."""


@dataclass(frozen=True, slots=True)
class _TrackingViewerStyle:
    """Encapsulates static visual parameters for the tracking viewer window."""

    window_geometry: tuple[int, int, int, int] = (50, 50, 1200, 800)
    """The initial window position (x, y) and size (width, height) for the tracking viewer."""


@dataclass(frozen=True, slots=True)
class _BinaryPlayerStyle:
    """Encapsulates static visual parameters for the binary player window."""

    image_plot_stretch: tuple[int, int] = (7, 3)
    """Row stretch factors for the image panel (index 0) and plot panel (index 1). A higher value gives that row
    proportionally more vertical space in the graphics layout."""
    window_geometry: tuple[int, int, int, int] = (50, 50, 1400, 1070)
    """The initial window position (x, y) and size (width, height) for the binary player."""
    legend_column_count: int = 2
    """The number of columns in the registration offset plot legend. This ensures that the legend uses the 
    horizontal, rather than a vertical layout."""


@dataclass(frozen=True, slots=True)
class _PCViewerStyle:
    """Encapsulates static visual parameters for the PC viewer window."""

    group_spacing: int = 20
    """The pixel spacing inserted between logical widget groups in the bottom control panel."""
    title_gutter_fraction: float = 0.08
    """The fraction of image height added as black space below each Principal Component extreme image. This ensures 
    that the image titles anchored to the bottom of the image are not clipped by the image or any other GUI elements."""
    window_geometry: tuple[int, int, int, int] = (50, 50, 1300, 800)
    """The initial window position (x, y) and size (width, height) for the PC viewer."""
    legend_column_count: int = 3
    """The number of columns in the PC metrics plot legend. This ensures that the legend uses the horizontal, 
    rather than a vertical layout."""


@dataclass(frozen=True, slots=True)
class _Colors:
    """Defines the RGB color palette shared across all viewer windows.

    All colors are expressed as ``(R, G, B)`` integer tuples in the 0-255 range.  Viewers select
    from this palette at their call sites, giving each entry a local semantic name.
    """

    cyan: tuple[int, int, int] = (0, 255, 255)
    """The cyan color (#00FFFF)."""
    magenta: tuple[int, int, int] = (255, 0, 255)
    """The magenta color (#FF00FF)."""
    red: tuple[int, int, int] = (255, 0, 0)
    """The red color (#FF0000)."""
    white: tuple[int, int, int] = (255, 255, 255)
    """The white color (#FFFFFF)."""
    black: tuple[int, int, int] = (0, 0, 0)
    """The black color (#000000)."""
    gray: tuple[int, int, int] = (100, 100, 100)
    """The gray color (#646464)."""
    silver: tuple[int, int, int] = (192, 192, 192)
    """The silver color (#C0C0C0)."""
    green: tuple[int, int, int] = (0, 255, 0)
    """The green color (#00FF00)."""
    gold: tuple[int, int, int] = (255, 215, 0)
    """The gold color (#FFD700)."""


COLORS: _Colors = _Colors()
"""The module-level singleton providing the shared color palette."""

STYLE: _CommonStyle = _CommonStyle()
"""The module-level singleton providing all shared viewer style constants."""

PLOT_STYLE: _PlotStyle = _PlotStyle()
"""The module-level singleton providing all shared pyqtgraph plot style constants."""

FONTS: _FontStyle = _FontStyle()
"""The module-level singleton providing all shared font and text size constants."""

ROI_STYLE: _ROIViewerStyle = _ROIViewerStyle()
"""The module-level singleton providing ROI viewer style constants."""

TRACKING_STYLE: _TrackingViewerStyle = _TrackingViewerStyle()
"""The module-level singleton providing tracking viewer style constants."""

BINARY_STYLE: _BinaryPlayerStyle = _BinaryPlayerStyle()
"""The module-level singleton providing binary player style constants."""

PC_STYLE: _PCViewerStyle = _PCViewerStyle()
"""The module-level singleton providing PC viewer style constants."""
