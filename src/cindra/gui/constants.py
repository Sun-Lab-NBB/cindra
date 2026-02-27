"""Provides constants, enumerations, and style definitions shared by the ROI viewer and ROI editor."""

from __future__ import annotations

from enum import IntEnum
from dataclasses import field, dataclass

from PySide6 import QtGui


class ROIColorMode(IntEnum):
    """Selects the statistic used to color ROI overlays in the image panels."""

    RANDOM = 0
    """Assigns each ROI a random color from the active colormap."""

    SKEWNESS = 1
    """Colors ROIs by the skewness of their spatial footprint pixel distribution."""

    COMPACTNESS = 2
    """Colors ROIs by the compactness (circularity) of their spatial footprint."""

    FOOTPRINT = 3
    """Colors ROIs by their total spatial footprint area in pixels."""

    ASPECT_RATIO = 4
    """Colors ROIs by the aspect ratio of their bounding ellipse."""

    COLOCALIZATION_PROBABILITY = 5
    """Colors ROIs by their channel 2 colocalization probability."""

    CLASSIFIER_PROBABILITY = 6
    """Colors ROIs by the trained classifier's cell-probability estimate."""

    CORRELATIONS = 7
    """Colors ROIs by pairwise activity correlation with the selected ROI."""

    CELL_NON_CELL = 8
    """Colors ROIs by their cell vs non-cell classification status."""


class BackgroundView(IntEnum):
    """Selects the background image displayed behind ROI overlays in the image panels."""

    ROIS_ONLY = 0
    """Displays a blank background with ROI overlays only."""

    # Channel 1 reference images.
    MEAN_IMAGE = 1
    """Displays the temporal mean of all registered channel 1 frames."""

    ENHANCED_MEAN_IMAGE = 2
    """Displays the high-pass filtered channel 1 mean image used for cell boundary detection."""

    CORRELATION_MAP = 3
    """Displays the pixel-wise activity correlation map computed during channel 1 detection."""

    MAXIMUM_PROJECTION = 4
    """Displays the maximum intensity projection across all channel 1 frames."""

    # Channel 2 reference images.
    MEAN_IMAGE_CHANNEL_2 = 5
    """Displays the temporal mean of all registered channel 2 frames."""

    ENHANCED_MEAN_IMAGE_CHANNEL_2 = 6
    """Displays the high-pass filtered channel 2 mean image used for cell boundary detection."""

    CORRELATION_MAP_CHANNEL_2 = 7
    """Displays the pixel-wise activity correlation map computed during channel 2 detection."""

    MAXIMUM_PROJECTION_CHANNEL_2 = 8
    """Displays the maximum intensity projection across all channel 2 frames."""

    # Structural reference images.
    CORRECTED_STRUCTURAL_MEAN_IMAGE = 9
    """Displays the bleed-through-corrected structural channel mean image computed during
    functional-to-structural channel colocalization."""


class TraceMode(IntEnum):
    """Selects the fluorescence trace type displayed in the trace panel."""

    RAW_FLUORESCENCE = 0
    """Displays the raw cell_fluorescence trace."""

    NEUROPIL = 1
    """Displays the neuropil_fluorescence trace."""

    NEUROPIL_CORRECTED = 2
    """Displays the neuropil-corrected trace (cell_fluorescence - neuropil_coefficient *
    neuropil_fluorescence)."""

    DECONVOLVED = 3
    """Displays the deconvolved spikes trace."""


@dataclass(frozen=True, slots=True)
class _ViewerStyle:
    """Encapsulates visual and dimensional constants shared by the ROI viewer and editor windows."""

    main_window: str = "QMainWindow {background: 'black';}"
    """Stylesheet applied to the main window background."""
    button_pressed: str = "QPushButton {Text-align: left; background-color: rgb(100,50,100); color:white;}"
    """Stylesheet for a button in the pressed (active/selected) state."""
    button_unpressed: str = "QPushButton {Text-align: left; background-color: rgb(50,50,50); color:white;}"
    """Stylesheet for a button in the unpressed (enabled, not selected) state."""
    button_inactive: str = "QPushButton {Text-align: left; background-color: rgb(50,50,50); color:gray;}"
    """Stylesheet for a button in the inactive (disabled/grayed-out) state."""
    white_label: str = "color: white;"
    """Stylesheet for white label text on a dark background."""
    red_label: str = "color: red;"
    """Stylesheet for red label text (used for neuropil trace indicators)."""
    cyan_label: str = "color: cyan;"
    """Stylesheet for cyan label text (used for raw fluorescence trace indicators)."""
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
    small_edit_width: int = 35
    """Width for small input fields and quadrant buttons (top-n count, diameter, max plotted count)."""
    roi_edit_width: int = 45
    """Width for ROI index input fields."""
    medium_edit_width: int = 60
    """Width for medium-sized widgets in the ROI editor (add-ROI button, diameter label)."""
    parameter_edit_width: int = 90
    """Width for parameter input fields in the merge dialog."""
    combo_box_width: int = 100
    """Width for the activity mode combo box in the trace panel."""
    square_button_max_width: int = 22
    """Maximum width for small square buttons (trace arrows, scale buttons)."""
    colorbar_max_height: int = 60
    """Maximum height for the colorbar widget."""
    colorbar_max_width: int = 150
    """Maximum width for the colorbar widget."""
    font_family: str = "Arial"
    """Standard font family used throughout the GUI."""
    alternative_font_family: str = "Times"
    """Alternative font family used for colorbar and merge dialog labels."""
    color_edit_width: int = 65
    """Width for color panel edit fields and the colormap combo box."""
    roi_text_size: int = 8
    """Font size for ROI text labels."""
    roi_text_color: tuple[int, int, int] = (180, 180, 180)
    """Color for ROI text labels."""
    deconvolved_alpha: int = 150
    """Alpha value for the deconvolved trace pen."""
    average_gray: int = 140
    """Gray intensity for the average trace pen."""
    colorbar_sample_count: int = 101
    """Number of samples for the colorbar gradient."""
    colorbar_row_count: int = 20
    """Number of rows in the colorbar image."""
    save_button_width: int = 100
    """Fixed width for the save-and-quit button in the ROI draw editor."""

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

    def colorbar_font(self) -> QtGui.QFont:
        """Creates the font for colorbar tick labels (Times 8pt bold).

        Returns:
            The configured QFont instance.
        """
        return QtGui.QFont(self.alternative_font_family, 8, QtGui.QFont.Weight.Bold.value)

    def merge_label_font(self) -> QtGui.QFont:
        """Creates the font for merge dialog parameter labels (Times bold).

        Returns:
            The configured QFont instance.
        """
        return QtGui.QFont(self.alternative_font_family, -1, QtGui.QFont.Weight.Bold.value)


STYLE: _ViewerStyle = _ViewerStyle()
"""Module-level singleton providing all viewer style constants."""


@dataclass(frozen=True, slots=True)
class _ViewerConfig:
    """Encapsulates behavioral and algorithmic constants shared by the ROI viewer and editor."""

    # Overlay constants.
    overlap_layers: int = 3
    """Number of overlap layers stored in the ROI index map."""
    lambda_norm_scale: float = 0.75
    """ROI weight normalization scale factor."""
    lambda_threshold: float = 1e-10
    """Minimum lambda value threshold for computing the mean weight."""
    color_stat_count: int = 9
    """Number of color statistics (random, skew, compact, footprint, aspect, chan2, class, corr, cell/non-cell)."""
    fixed_colorbar_range: tuple[float, ...] = (0.0, 0.5, 1.0)
    """Fixed colorbar range for statistics without computed percentiles."""
    lower_percentile: float = 2.0
    """Lower percentile value for computing istat normalization bounds."""
    upper_percentile: float = 98.0
    """Upper percentile value for computing istat normalization bounds."""
    channel_2_color_divisor: float = 1.4
    """Random color adjustment divisor for channel 2 data."""
    channel_2_color_offset: float = 0.1
    """Random color adjustment offset for channel 2 data."""
    hsv_divisor: float = 1.4
    """HSV transform normalization divisor."""
    hsv_offset: float = 0.4
    """HSV transform normalization offset."""
    flip_threshold: int = 100
    """Minimum number of changed cells before incremental flip is used over full reinit."""

    # Color constants.
    colormaps: tuple[str, ...] = (
        "hsv",
        "viridis",
        "plasma",
        "inferno",
        "magma",
        "cividis",
        "viridis_r",
        "plasma_r",
        "inferno_r",
        "magma_r",
        "cividis_r",
    )
    """Available colormaps for the colormap chooser."""
    color_names: tuple[str, ...] = (
        "A: random",
        "S: skew",
        "D: compact",
        "F: footprint",
        "G: aspect_ratio",
        "H: chan2_prob",
        "J: classifier, cell prob=",
        "K: correlations, bin=",
        "L: cell / non-cell",
    )
    """Color statistic names displayed on the color buttons (with keyboard shortcut prefixes)."""
    color_short_names: tuple[str, ...] = (
        "random",
        "skew",
        "compact",
        "footprint",
        "aspect_ratio",
        "chan2_prob",
        "classifier, cell prob=",
        "correlations, bin=",
        "cell / non-cell",
    )
    """Short color statistic names without keyboard shortcut prefixes."""
    color_narrow_range_start: int = 5
    """Starting index of the color buttons that require the adjacent edit field column."""
    color_narrow_range_end: int = 9
    """Ending index of the color buttons that require the adjacent edit field column."""
    color_channel_2: int = 5
    """Channel 2 color index."""
    color_classifier: int = 6
    """Classifier probability color index."""
    color_correlation: int = 7
    """Correlation color index."""
    color_cell_non_cell: int = 8
    """Cell vs non-cell classification color index."""
    cell_color: tuple[int, int, int] = (0, 200, 0)
    """RGB color for classified cells in cell/non-cell mode."""
    noncell_color: tuple[int, int, int] = (200, 0, 200)
    """RGB color for non-cells in cell/non-cell mode."""
    channel_2_threshold_epsilon: float = 1e-3
    """Minimum change in channel 2 threshold to trigger a recoloring update."""
    stat_field_map: dict[str, str] = field(default_factory=lambda: {
        "skew": "skewness",
        "compact": "compactness",
        "footprint": "footprint",
        "aspect_ratio": "aspect_ratio",
        "chan2_prob": "colocalization_probability",
    })
    """Mapping from color statistic display names to ROIStatistics attribute names."""

    # View constants.
    view_count: int = 7
    """Number of background view types available."""
    view_names: tuple[str, ...] = (
        "Q: ROIs",
        "W: mean img",
        "E: mean img (enhanced)",
        "R: correlation map",
        "T: max projection",
        "Y: mean img chan2, corr",
        "U: mean img chan2",
    )
    """Names displayed on view selection buttons (with keyboard shortcut prefixes)."""
    view_short_names: tuple[str, ...] = (
        "ROIs",
        "mean img",
        "mean img (enhanced)",
        "correlation map",
        "max projection",
        "mean img chan2, corr",
        "mean img chan2",
    )
    """Short view names without keyboard shortcut prefixes."""

    # Trace constants.
    default_activity_mode: int = 3
    """Default activity mode index (deconvolved)."""
    max_plotted_count: int = 400
    """Maximum number of traces that can be plotted simultaneously."""
    default_plotted_count: int = 40
    """Default number of traces plotted."""
    default_scale_factor: float = 2.0
    """Default vertical scale factor for multi-trace stacking."""
    scale_step: float = 0.5
    """Scale factor adjustment step per button press."""
    min_scale: float = 0.5
    """Minimum allowed scale factor."""
    max_scale: float = 10.0
    """Maximum allowed scale factor."""
    default_trace_level: int = 1
    """Default trace panel row stretch level."""
    min_trace_level: int = 1
    """Minimum trace panel stretch level."""
    max_trace_level: int = 5
    """Maximum trace panel stretch level."""
    activity_mode_subtracted: int = 2
    """Activity mode index for neuropil-subtracted fluorescence (F - 0.7*Fneu)."""
    neuropil_coefficient: float = 0.7
    """Neuropil subtraction coefficient for the F - 0.7*Fneu activity mode."""
    average_threshold: int = 5
    """Minimum number of selected cells before the average trace is displayed."""
    average_scale_divisor: float = 25.0
    """Ratio of selected cells to determine average trace vertical scale."""

    # Selection constants.
    max_top_n: int = 500
    """Maximum number of cells allowed in the top-n / bottom-n selection input."""
    default_top_n: int = 40
    """Default number of cells selected by top-n / bottom-n."""
    # Viewer constants.
    centroid_stat_index: int = 1
    """1-based stat index for the centroid field (used to display ROI position)."""
    pixel_count_stat_index: int = 2
    """1-based stat index for the pixel count field."""

    # Context loader constants.
    default_channel_2_threshold: float = 0.6
    """Default colocalization threshold for channel 2 data."""
    bin_size_divisor: int = 2
    """Divisor for computing the temporal bin size from tau * sampling_rate."""
    basic_color_count: int = 9
    """Number of basic (non-dynamic) color mode buttons."""
    default_context_activity_mode: int = 2
    """Default activity mode index used during context loading (neuropil-corrected)."""

    # Classifier constants.
    classifier_color_index: int = 6
    """Index of the classifier probability color mode in the color button group."""
    classification_features: tuple[str, ...] = ("normalized_pixel_count", "compactness", "skewness")
    """Feature names used by the classifier, matching ROIStatistics attribute names."""

    # ROI draw editor constants.
    image_percentile_low: int = 1
    """Percentile range lower bound used for image normalization in the draw editor."""
    image_percentile_high: int = 99
    """Percentile range upper bound used for image normalization in the draw editor."""
    draw_view_count: int = 4
    """Number of reference images available in the draw editor (mean, enhanced, correlation, max projection)."""
    max_diameter_fraction: float = 0.2
    """Maximum fraction of field of view used as the default ROI diameter."""
    min_diameter: int = 3
    """Minimum ROI diameter in pixels."""
    reference_roi_count: int = 100
    """Number of reference ROIs used for normalized pixel count computation."""
    roi_pen_width: int = 3
    """Pen width for ROI ellipse outlines in the draw editor."""
    roi_position_offset: int = 5
    """Default initial position offset for the ROI ellipse in the draw editor."""
    correlation_map_view_index: int = 2
    """View index for the correlation map in the draw editor reference image selector."""

    # Merge constants.
    sentinel_distance: float = 1e6
    """Large sentinel distance used to initialize the distance matrix upper triangle."""
    correlation_epsilon: float = 1e-3
    """Small epsilon added to denominators to prevent division by zero in merge computations."""
    default_correlation_threshold: float = 0.8
    """Default correlation threshold for automated merge suggestions."""
    default_distance_threshold: float = 100.0
    """Default euclidean distance threshold for automated merge suggestions."""
    scatter_pen_width: int = 3
    """Scatter plot pen width for merge suggestion visualization."""


CONFIG: _ViewerConfig = _ViewerConfig()
"""Module-level singleton providing all viewer behavioral constants."""
