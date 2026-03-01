"""Provides constants and enumerations shared across all GUI viewer windows."""

from __future__ import annotations

from enum import IntEnum
from dataclasses import field, dataclass


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
    """Selects the background image displayed behind ROI overlays in the image panels.

    When channel 2 is toggled on, slots 1-4 display channel 2 images instead of channel 1.
    """

    ROIS_ONLY = 0
    """Displays a blank background with ROI overlays only."""

    MEAN_IMAGE = 1
    """Displays the temporal mean image (channel 1 or channel 2 depending on channel toggle)."""

    ENHANCED_MEAN_IMAGE = 2
    """Displays the enhanced mean image (channel 1 or channel 2 depending on channel toggle)."""

    CORRELATION_MAP = 3
    """Displays the pixel-wise activity correlation map (channel 1 or channel 2 depending on channel toggle)."""

    MAXIMUM_PROJECTION = 4
    """Displays the maximum intensity projection (channel 1 or channel 2 depending on channel toggle)."""

    CORRECTED_STRUCTURAL = 5
    """Displays the bleed-through-corrected structural channel mean image computed during
    functional-to-structural channel colocalization. Only enabled when colocalization data exists."""


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


class MaskLayer(IntEnum):
    """Selects the active ROI mask layer."""

    ORIGINAL = 0
    """Displays the original ROI masks from single-day extraction in native recording coordinates."""

    DEFORMED = 1
    """Displays the original ROI masks warped to the shared cross-recording coordinate space via multi-day registration
    deformation fields."""

    TEMPLATE = 2
    """Displays the consensus template ROI masks derived from cross-recording clustering, defined in the shared
    coordinate space."""

    TRACKED = 3
    """Displays the template ROI masks backward-deformed to each recording's native coordinate space."""


class CoordinateSpace(IntEnum):
    """Selects the coordinate space for reference images."""

    NATIVE = 0
    """Displays reference images in the original recording coordinate space."""

    TRANSFORMED = 1
    """Displays reference images warped to align with the cross-recording template coordinate space."""


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
    random_color_seed: int = 0
    """Seed for reproducible random ROI color generation."""

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
    """Color statistic names displayed in the color mode dropdown."""
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
    cell_color: tuple[int, int, int] = (0, 255, 0)
    """RGB color for classified cells in cell/non-cell mode (green)."""
    noncell_color: tuple[int, int, int] = (255, 0, 255)
    """RGB color for non-cells in cell/non-cell mode (magenta)."""
    channel_2_threshold_epsilon: float = 1e-3
    """Minimum change in channel 2 threshold to trigger a recoloring update."""
    stat_field_map: dict[str, str] = field(
        default_factory=lambda: {
            "skew": "skewness",
            "compact": "compactness",
            "footprint": "footprint",
            "aspect_ratio": "aspect_ratio",
            "chan2_prob": "colocalization_probability",
        }
    )
    """Mapping from color statistic display names to ROIStatistics attribute names."""

    # View constants.
    view_count: int = 6
    """Number of background view types available."""
    view_names: tuple[str, ...] = (
        "ROIs",
        "mean img",
        "mean img (enhanced)",
        "correlation map",
        "max projection",
        "corrected structural",
    )
    """Names displayed in the background view dropdown."""

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
    activity_mode_labels: tuple[str, ...] = ("Fluorescence", "Neuropil", "Neuropil Subtracted", "Spikes")
    """Display labels for the activity mode combo box, indexed by TraceMode value."""
    average_threshold: int = 5
    """Minimum number of selected cells before the average trace is displayed."""
    average_scale_divisor: float = 25.0
    """Ratio of selected cells to determine average trace vertical scale."""

    # Selection constants.
    max_top_n: int = 500
    """Maximum number of cells allowed in the top-n / bottom-n selection input."""
    default_top_n: int = 40
    """Default number of cells selected by top-n / bottom-n."""
    roi_selection_overlap_threshold: float = 0.6
    """Minimum fraction of an ROI's pixels that must fall inside the selection rectangle for it to be included."""
    roi_selection_max_dimension: int = 300
    """Maximum pixel dimension for the ROI selection rectangle."""
    zoom_to_cell_fraction: float = 0.1
    """Fraction of the maximum image dimension used as padding when zooming to a cell."""

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


CONFIG: _ViewerConfig = _ViewerConfig()
"""Module-level singleton providing all viewer behavioral constants."""


@dataclass(frozen=True, slots=True)
class _TrackingViewerConfig:
    """Encapsulates behavioral constants for the tracking viewer window."""

    cycle_interval: int = 500
    """The millisecond interval for auto-cycling between recordings."""

    lower_percentile: float = 1.0
    """The lower percentile value for normalizing background images."""

    upper_percentile: float = 99.0
    """The upper percentile value for normalizing background images."""


TRACKING_CONFIG: _TrackingViewerConfig = _TrackingViewerConfig()
"""Module-level singleton providing tracking viewer behavioral constants."""


@dataclass(frozen=True, slots=True)
class _BinaryPlayerConfig:
    """Encapsulates behavioral constants for the binary player window."""

    playback_speed_multiplier: int = 5
    """Factor by which the real-time frame period is divided to compute the playback timer interval."""

    subsample_frame_count: int = 100
    """Number of evenly-spaced frames subsampled for dynamic range estimation."""

    min_frame_delta: int = 5
    """Minimum frame increment for arrow key navigation."""

    frame_delta_divisor: int = 200
    """Divisor for computing frame slider step size from total frame count."""

    frame_slider_tick_interval: int = 5
    """Tick interval for the frame navigation slider. Determines the number of frames represented by each tick of
    the binary viewer slider."""

    display_range_low_sigma: float = 2.0
    """Standard deviations below mean for display range lower bound."""

    display_range_high_sigma: float = 5.0
    """Standard deviations above mean for display range upper bound."""


BINARY_CONFIG: _BinaryPlayerConfig = _BinaryPlayerConfig()
"""Module-level singleton providing binary player behavioral constants."""


@dataclass(frozen=True, slots=True)
class _PCViewerConfig:
    """Encapsulates behavioral constants for the PC viewer window."""

    animation_interval_ms: int = 200
    """Interval in milliseconds between PC extreme image animation updates."""


PC_CONFIG: _PCViewerConfig = _PCViewerConfig()
"""Module-level singleton providing PC viewer behavioral constants."""
