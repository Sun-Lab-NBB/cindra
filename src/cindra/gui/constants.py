"""Provides constants and enumerations shared across all GUI viewer windows."""

from __future__ import annotations

from enum import IntEnum, StrEnum
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

    CLASSIFIER_LABEL = 8
    """Colors ROIs by their classifier-assigned label (cell vs non-cell)."""


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


class Colormap(StrEnum):
    """Defines the available colormaps for ROI overlay coloring."""

    HSV = "hsv"
    """The hue-saturation-value cyclic colormap."""
    VIRIDIS = "viridis"
    """The viridis perceptually uniform sequential colormap."""
    PLASMA = "plasma"
    """The plasma perceptually uniform sequential colormap."""
    INFERNO = "inferno"
    """The inferno perceptually uniform sequential colormap."""
    MAGMA = "magma"
    """The magma perceptually uniform sequential colormap."""
    CIVIDIS = "cividis"
    """The cividis colorblind-friendly sequential colormap."""
    VIRIDIS_R = "viridis_r"
    """The reversed viridis colormap."""
    PLASMA_R = "plasma_r"
    """The reversed plasma colormap."""
    INFERNO_R = "inferno_r"
    """The reversed inferno colormap."""
    MAGMA_R = "magma_r"
    """The reversed magma colormap."""
    CIVIDIS_R = "cividis_r"
    """The reversed cividis colormap."""


@dataclass(frozen=True, slots=True)
class _CommonConstants:
    """Encapsulates static runtime parameters shared by all viewer windows."""

    lower_percentile: float = 1.0
    """The lower percentile bound for normalizing image data and ROI statistics."""
    upper_percentile: float = 99.0
    """The upper percentile bound for normalizing image data and ROI statistics."""


@dataclass(frozen=True, slots=True)
class _ROIViewerConstants:
    """Encapsulates static runtime parameters for the ROI viewer and editor."""

    overlap_layers: int = 3
    """The number of overlap layers stored in the ROI index map. Each pixel tracks up to this many overlapping ROIs in a
    depth stack, enabling click-through selection and brightness-based overlap visualization. ROIs beyond this depth are
    silently dropped."""
    fixed_colorbar_range: tuple[float, ...] = (0.0, 0.5, 1.0)
    """The (low, mid, high) colorbar tick values used for color modes that lack data-driven percentile ranges. Applied
    to the random, classifier probability, correlation, and classifier label modes where the statistic is either
    categorical or already normalized to [0, 1]."""
    channel_2_color_divisor: float = 1.4
    """The random color adjustment divisor for channel 2 data."""
    channel_2_color_offset: float = 0.1
    """The random color adjustment offset for channel 2 data."""
    hsv_divisor: float = 1.4
    """The HSV transform normalization divisor."""
    hsv_offset: float = 0.4
    """The HSV transform normalization offset."""
    random_color_seed: int = 0
    """The seed for reproducible random ROI color generation."""
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
    """The color statistic names displayed in the color mode dropdown."""
    cell_color: tuple[int, int, int] = (0, 255, 0)
    """The RGB color for classified cells in cell/non-cell mode (green)."""
    non_cell_color: tuple[int, int, int] = (255, 0, 255)
    """The RGB color for non-cells in cell/non-cell mode (magenta)."""
    statistic_field_map: dict[str, str] = field(
        default_factory=lambda: {
            "skew": "skewness",
            "compact": "compactness",
            "footprint": "footprint",
            "aspect_ratio": "aspect_ratio",
            "chan2_prob": "colocalization_probability",
        }
    )
    """The mapping from color statistic display names to ROIStatistics attribute names."""
    view_names: tuple[str, ...] = (
        "ROIs",
        "mean img",
        "mean img (enhanced)",
        "correlation map",
        "max projection",
        "corrected structural",
    )
    """The names displayed in the background view dropdown."""
    max_plotted_count: int = 400
    """The maximum number of traces that can be plotted simultaneously."""
    default_plotted_count: int = 40
    """The default number of traces plotted."""
    default_scale_factor: float = 2.0
    """The default vertical scale factor for multi-trace stacking."""
    neuropil_coefficient: float = 0.7
    """The neuropil subtraction coefficient for the F - 0.7*Fneu activity mode."""
    activity_mode_labels: tuple[str, ...] = ("Fluorescence", "Neuropil", "Neuropil Subtracted", "Spikes")
    """The display labels for the activity mode combo box, indexed by TraceMode value."""
    average_threshold: int = 5
    """The minimum number of selected cells before the average trace is displayed."""
    average_scale_divisor: float = 25.0
    """The ratio of selected cells to determine average trace vertical scale."""
    maximum_top_count: int = 500
    """The maximum number of cells allowed in the top-n / bottom-n selection input."""
    default_top_count: int = 40
    """The default number of cells selected by top-n / bottom-n."""
    roi_selection_overlap_threshold: float = 0.6
    """The minimum fraction of an ROI's pixels that must fall inside the selection rectangle for it to be included."""
    roi_selection_max_dimension: int = 300
    """The maximum pixel dimension for the ROI selection rectangle."""
    zoom_to_cell_fraction: float = 0.1
    """The fraction of the maximum image dimension used as padding when zooming to a cell."""
    centroid_statistic_index: int = 1
    """The 1-based stat index for the centroid field (used to display ROI position)."""
    pixel_count_statistic_index: int = 2
    """The 1-based stat index for the pixel count field."""
    default_channel_2_threshold: float = 0.6
    """The default colocalization threshold for channel 2 data."""
    default_classifier_threshold: float = 0.5
    """The default probability threshold above which an ROI is labeled as a cell."""
    bin_size_divisor: int = 2
    """The divisor for computing the temporal bin size from tau * sampling_rate."""


@dataclass(frozen=True, slots=True)
class _TrackingViewerConstants:
    """Encapsulates static runtime parameters for the tracking viewer window."""

    cycle_interval: int = 500
    """The millisecond interval for auto-cycling between recordings."""


@dataclass(frozen=True, slots=True)
class _BinaryPlayerConstants:
    """Encapsulates static runtime parameters for the binary player window."""

    playback_speed_multiplier: int = 5
    """The factor by which playback runs faster than the real time recording. A value of 5 means the binary viewer 
    plays frames at 5x the original recording speed."""
    subsample_frame_count: int = 100
    """The number of evenly-spaced frames subsampled from the recording for dynamic range estimation. These frames
    are used to compute the mean and standard deviation that define the display intensity range."""
    minimum_frame_delta: int = 5
    """The minimum frame step size for arrow key navigation. When the recording is short enough that the computed
    step (frame_count / frame_delta_divisor) falls below this value, this minimum is used instead."""
    frame_delta_divisor: int = 200
    """The divisor applied to the total frame count to compute the arrow key frame step size. Larger values produce
    smaller steps, giving finer frame-by-frame navigation in long recordings."""
    frame_slider_tick_interval: int = 5
    """The tick spacing for the frame navigation slider. Controls how many frame positions are represented by each
    discrete tick mark on the slider widget."""
    display_range_low_sigma: float = 2.0
    """The number of standard deviations below the subsampled mean used as the display intensity floor. Pixels below
    this bound are clipped to black in the binary viewer image."""
    display_range_high_sigma: float = 5.0
    """The number of standard deviations above the subsampled mean used as the display intensity ceiling. Pixels above
    this bound are clipped to white in the binary viewer image."""


@dataclass(frozen=True, slots=True)
class _PCViewerConstants:
    """Encapsulates static runtime parameters for the PC viewer window."""

    animation_interval_milliseconds: int = 200
    """The interval in milliseconds between PC extreme image animation updates."""


COMMON_CONFIG: _CommonConstants = _CommonConstants()
"""The module-level singleton providing shared behavioral constants."""

ROI_CONFIG: _ROIViewerConstants = _ROIViewerConstants()
"""The module-level singleton providing ROI viewer behavioral constants."""

TRACKING_CONFIG: _TrackingViewerConstants = _TrackingViewerConstants()
"""The module-level singleton providing tracking viewer behavioral constants."""

BINARY_CONFIG: _BinaryPlayerConstants = _BinaryPlayerConstants()
"""The module-level singleton providing binary player behavioral constants."""

PC_CONFIG: _PCViewerConstants = _PCViewerConstants()
"""The module-level singleton providing PC viewer behavioral constants."""
