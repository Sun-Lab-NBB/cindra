"""Provides visual style constants shared across all GUI viewer windows."""

from __future__ import annotations

from dataclasses import field, dataclass

from PySide6 import QtGui


@dataclass(frozen=True, slots=True)
class _CommonStyle:
    """Encapsulates visual constants shared by all viewer windows."""

    main_window: str = "QMainWindow {background: 'black';}"
    """The stylesheet applied to the main window background."""
    white_label: str = "color: white;"
    """The stylesheet for white label text on a dark background."""
    scatter_point_size: int = 10
    """The marker size in pixels for scatter plot overlays."""
    icon_size: int = 30
    """The dimension in pixels for media control button icons."""
    legend_headroom: float = 0.25
    """The fraction of the y-axis data range added as top padding so legends never overlap traces."""
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
    """The stylesheet for the dual-handle range slider used in saturation controls."""
    group_box: str = "QGroupBox { color: white; }"
    """The stylesheet for QGroupBox title text on dark backgrounds."""
    legend_horizontal_spacing: int = 20
    """The horizontal spacing between legend entries in plot widgets."""
    legend_offset: tuple[int, int] = (-10, 1)
    """The position offset for the legend widget relative to the plot corner."""
    axis_fixed_width: int = 60
    """The fixed pixel width for the y-axis so the plot area stays stable when tick label digit counts change."""
    roi_edit_width: int = 50
    """The width for ROI index input fields."""
    small_edit_width: int = 40
    """The width for small numeric input fields (top-n count, max plotted count, PC number)."""
    square_button_width: int = 30
    """The maximum width for small square buttons (trace arrows, scale +/- buttons)."""
    group_spacing: int = 20
    """The pixel spacing between widget groups in control panels."""
    button_pressed: str = "QPushButton {Text-align: left; background-color: rgb(100,50,100); color:white;}"
    """The stylesheet for a button in the pressed (active/selected) state."""
    button_unpressed: str = "QPushButton {Text-align: left; background-color: rgb(50,50,50); color:white;}"
    """The stylesheet for a button in the unpressed (enabled, not selected) state."""
    button_inactive: str = "QPushButton {Text-align: left; background-color: rgb(50,50,50); color:gray;}"
    """The stylesheet for a button in the inactive (disabled/grayed-out) state."""
    combo_box_width: int = 100
    """The fixed width for combo box widgets."""


@dataclass(frozen=True, slots=True)
class _FontStyle:
    """Encapsulates all font and text size constants shared by all viewer windows."""

    family: str = "Arial"
    """The standard font family used throughout the GUI."""

    plot_title_size: str = "14pt"
    """The font size for plot titles."""

    label_size: str = "12pt"
    """The font size for axis labels and legend entry labels."""

    small: QtGui.QFont = field(default_factory=lambda: QtGui.QFont("Arial", 8))
    """Small font (Arial 8pt) for ROI statistic display."""

    small_bold: QtGui.QFont = field(default_factory=lambda: QtGui.QFont("Arial", 8, QtGui.QFont.Weight.Bold.value))
    """Small bold font (Arial 8pt bold) for combo boxes, buttons, and overlay text."""

    medium_bold: QtGui.QFont = field(default_factory=lambda: QtGui.QFont("Arial", 11, QtGui.QFont.Weight.Bold.value))
    """Medium bold font (Arial 11pt bold) for arrow and scale buttons."""

    large: QtGui.QFont = field(default_factory=lambda: QtGui.QFont("Arial", 14))
    """Large font (Arial 14pt) for prominent GUI input fields."""

    large_bold: QtGui.QFont = field(default_factory=lambda: QtGui.QFont("Arial", 14, QtGui.QFont.Weight.Bold.value))
    """Large bold font (Arial 14pt bold) for prominent GUI labels."""


@dataclass(frozen=True, slots=True)
class _ROIViewerStyle:
    """Encapsulates visual constants specific to the ROI viewer and editor windows."""

    colorbar_max_height: int = 60
    """The maximum height for the colorbar widget."""
    colorbar_max_width: int = 150
    """The maximum width for the colorbar widget."""
    color_edit_width: int = 65
    """The width for color panel edit fields and the colormap combo box."""
    colorbar_sample_count: int = 101
    """The number of samples for the colorbar gradient."""
    colorbar_row_count: int = 20
    """The number of rows in the colorbar image."""
    default_roi_opacity: tuple[int, int] = (127, 255)
    """Default (unselected, selected) ROI opacity range."""
    default_saturation_range: tuple[int, int] = (0, 255)
    """Default background saturation min/max range."""
    window_geometry: tuple[int, int, int, int] = (50, 50, 1500, 800)
    """The initial window position (x, y) and size (width, height) for the ROI viewer."""


@dataclass(frozen=True, slots=True)
class _TrackingViewerStyle:
    """Encapsulates visual constants specific to the tracking viewer window."""

    default_mask_opacity: int = 127
    """The default mask overlay opacity (0-255 uint8 range). This determines the initial transparency of the ROI masks 
    rendered over the chosen background image in the Tracker GUI."""
    window_geometry: tuple[int, int, int, int] = (50, 50, 1200, 800)
    """The initial window position (x, y) and size (width, height) for the tracking viewer."""


@dataclass(frozen=True, slots=True)
class _BinaryPlayerStyle:
    """Encapsulates visual constants specific to the binary player window."""

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
    """Encapsulates visual constants specific to the PC viewer window."""

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
