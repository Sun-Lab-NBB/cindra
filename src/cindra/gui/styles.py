"""Provides visual style constants shared across all GUI viewer windows."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6 import QtGui


@dataclass(frozen=True, slots=True)
class _CommonStyle:
    """Encapsulates visual constants shared by all viewer windows."""

    main_window: str = "QMainWindow {background: 'black';}"
    """Stylesheet applied to the main window background."""
    white_label: str = "color: white;"
    """Stylesheet for white label text on a dark background."""
    font_family: str = "Arial"
    """Standard font family used throughout the GUI."""
    scatter_point_size: int = 10
    """Marker size in pixels for scatter plot overlays."""
    icon_size: int = 30
    """Dimension in pixels for media control button icons."""
    plot_title_size: str = "14pt"
    """Font size for plot titles."""
    axis_label_size: str = "12pt"
    """Font size for axis labels."""
    legend_label_size: str = "12pt"
    """Font size for legend entry labels."""
    legend_headroom: float = 0.25
    """Fraction of the y-axis data range added as top padding so legends never overlap traces."""
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
    """Stylesheet for the dual-handle range slider used in saturation controls."""
    deconvolved_alpha: int = 150
    """Alpha value for the deconvolved trace pen."""
    average_gray: int = 140
    """Gray intensity for the average trace pen."""
    group_box: str = "QGroupBox { color: white; }"
    """Stylesheet for QGroupBox title text on dark backgrounds."""
    legend_horizontal_spacing: int = 20
    """Horizontal spacing between legend entries in plot widgets."""
    legend_offset: tuple[int, int] = (-10, 1)
    """Position offset for the legend widget relative to the plot corner."""
    fluorescence_pen: str = "c"
    """Pen color for cell fluorescence traces (cyan)."""
    neuropil_pen: str = "r"
    """Pen color for neuropil fluorescence traces (red)."""
    deconvolved_pen_color: tuple[int, int, int] = (255, 255, 255)
    """RGB color for deconvolved spike traces (alpha handled separately via deconvolved_alpha)."""

    def label_font(self) -> QtGui.QFont:
        """Creates the standard label font (Arial 8pt).

        Returns:
            The configured QFont instance.
        """
        return QtGui.QFont(self.font_family, 8)

    def label_font_bold(self) -> QtGui.QFont:
        """Creates the standard bold label font (Arial 8pt bold).

        Returns:
            The configured QFont instance.
        """
        return QtGui.QFont(self.font_family, 8, QtGui.QFont.Weight.Bold.value)

    def header_font(self) -> QtGui.QFont:
        """Creates the section header font (Arial 10pt bold).

        Returns:
            The configured QFont instance.
        """
        return QtGui.QFont(self.font_family, 10, QtGui.QFont.Weight.Bold.value)

    def arrow_button_font(self) -> QtGui.QFont:
        """Creates the font for trace expand/collapse arrow buttons (Arial 11pt bold).

        Returns:
            The configured QFont instance.
        """
        return QtGui.QFont(self.font_family, 11, QtGui.QFont.Weight.Bold.value)


@dataclass(frozen=True, slots=True)
class _ROIViewerStyle:
    """Encapsulates visual constants specific to the ROI viewer and editor windows."""

    button_pressed: str = "QPushButton {Text-align: left; background-color: rgb(100,50,100); color:white;}"
    """Stylesheet for a button in the pressed (active/selected) state."""
    button_unpressed: str = "QPushButton {Text-align: left; background-color: rgb(50,50,50); color:white;}"
    """Stylesheet for a button in the unpressed (enabled, not selected) state."""
    button_inactive: str = "QPushButton {Text-align: left; background-color: rgb(50,50,50); color:gray;}"
    """Stylesheet for a button in the inactive (disabled/grayed-out) state."""
    red_label: str = "color: red;"
    """Stylesheet for red label text (used for neuropil trace indicators)."""
    cyan_label: str = "color: cyan;"
    """Stylesheet for cyan label text (used for raw fluorescence trace indicators)."""
    small_edit_width: int = 35
    """Width for small input fields and quadrant buttons (top-n count, diameter, max plotted count)."""
    roi_edit_width: int = 45
    """Width for ROI index input fields."""
    medium_edit_width: int = 60
    """Width for medium-sized widgets in the ROI editor (add-ROI button, diameter label)."""
    combo_box_width: int = 100
    """Width for the activity mode combo box in the trace panel."""
    square_button_max_width: int = 22
    """Maximum width for small square buttons (trace arrows, scale buttons)."""
    colorbar_max_height: int = 60
    """Maximum height for the colorbar widget."""
    colorbar_max_width: int = 150
    """Maximum width for the colorbar widget."""
    color_edit_width: int = 65
    """Width for color panel edit fields and the colormap combo box."""
    roi_text_size: int = 8
    """Font size for ROI text labels."""
    roi_text_color: tuple[int, int, int] = (180, 180, 180)
    """Color for ROI text labels."""
    colorbar_sample_count: int = 101
    """Number of samples for the colorbar gradient."""
    colorbar_row_count: int = 20
    """Number of rows in the colorbar image."""
    save_button_width: int = 100
    """Fixed width for the save-and-quit button in the ROI draw editor."""
    window_geometry: tuple[int, int, int, int] = (50, 50, 1500, 800)
    """Initial window position (x, y) and size (width, height) for the ROI viewer."""
    view_box_border: tuple[int, int, int] = (100, 100, 100)
    """RGB border color for the ROI image ViewBox."""
    roi_text_font_family: str = "Times"
    """Font family for ROI index text labels at centroids."""
    selection_pen: str = "w"
    """Pen color for the interactive ROI selection rectangle."""


@dataclass(frozen=True, slots=True)
class _TrackingViewerStyle:
    """Encapsulates visual constants specific to the tracking viewer window."""

    default_mask_opacity: int = 127
    """The default mask overlay opacity (0-255 uint8 range)."""
    roi_edit_width: int = 50
    """The fixed pixel width of the ROI index input field."""
    window_size: tuple[int, int] = (1200, 800)
    """Initial window size (width, height) for the tracking viewer."""
    scale_button_width: int = 30
    """Maximum pixel width for trace scale +/- buttons."""


@dataclass(frozen=True, slots=True)
class _BinaryPlayerStyle:
    """Encapsulates visual constants specific to the binary player window."""

    axis_fixed_width: int = 50
    """Fixed pixel width for the y-axis so the plot area stays stable when tick label digit counts change."""
    window_geometry: tuple[int, int, int, int] = (70, 70, 1400, 1070)
    """Initial window position (x, y) and size (width, height) for the binary player."""
    y_offset_pen: str = "g"
    """Pen color for the Y rigid registration offset trace (green)."""
    x_offset_pen: str = "y"
    """Pen color for the X rigid registration offset trace (yellow)."""
    scatter_brush_color: tuple[int, int, int] = (255, 0, 0)
    """RGB brush color for the current-frame scatter indicator (red)."""
    legend_column_count: int = 2
    """Number of columns in the registration offset plot legend."""


@dataclass(frozen=True, slots=True)
class _PCViewerStyle:
    """Encapsulates visual constants specific to the PC viewer window."""

    metrics_font_size: int = 14
    """Point size for metric value labels and PC input field font."""
    pc_edit_width: int = 40
    """Width in pixels for the principal component number input field."""
    axis_fixed_width: int = 60
    """Fixed pixel width for the y-axis so the plot area stays stable when tick label digit counts change."""
    title_gutter_fraction: float = 0.08
    """Fraction of image height added as black space below the image for title labels."""
    window_geometry: tuple[int, int, int, int] = (70, 70, 1300, 800)
    """Initial window position (x, y) and size (width, height) for the PC viewer."""
    metric_colors: tuple[tuple[int, int, int], ...] = ((200, 200, 255), (255, 100, 100), (100, 50, 200))
    """RGB pen colors for rigid, nonrigid, and nonrigid-max metric curves."""
    scatter_brush_color: tuple[int, int, int] = (255, 255, 255)
    """RGB brush color for the selected-PC scatter indicator (white)."""
    group_spacing: int = 20
    """Pixel spacing between widget groups in the bottom control panel."""
    legend_column_count: int = 3
    """Number of columns in the PC metrics plot legend."""


STYLE: _CommonStyle = _CommonStyle()
"""Module-level singleton providing all shared viewer style constants."""

ROI_STYLE: _ROIViewerStyle = _ROIViewerStyle()
"""Module-level singleton providing ROI viewer style constants."""

TRACKING_STYLE: _TrackingViewerStyle = _TrackingViewerStyle()
"""Module-level singleton providing tracking viewer style constants."""

BINARY_STYLE: _BinaryPlayerStyle = _BinaryPlayerStyle()
"""Module-level singleton providing binary player style constants."""

PC_STYLE: _PCViewerStyle = _PCViewerStyle()
"""Module-level singleton providing PC viewer style constants."""
