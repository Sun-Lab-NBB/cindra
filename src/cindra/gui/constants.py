"""Provides constants and enumerations shared across all GUI viewer windows."""

from __future__ import annotations

from enum import IntEnum, StrEnum
from dataclasses import dataclass


class ROIColorMode(IntEnum):
    """Selects the statistic used to color ROI overlays in the image panels."""

    RANDOM = 0
    """Assigns each ROI a random color from the active colormap."""

    SKEWNESS = 1
    """Colors ROIs by the skewness of their baseline-subtracted fluorescence time series."""

    COMPACTNESS = 2
    """Colors ROIs by the compactness (circularity) of their spatial footprint."""

    FOOTPRINT = 3
    """Colors ROIs by the spatial detection scale (hop size) used during sparse detection."""

    ASPECT_RATIO = 4
    """Colors ROIs by the aspect ratio of their bounding ellipse."""

    SOLIDITY = 5
    """Colors ROIs by the solidity (soma-to-convex-hull area ratio) of their spatial footprint."""

    COLOCALIZATION_PROBABILITY = 6
    """Colors ROIs by their channel 2 colocalization probability."""

    RECORDING_COUNT = 7
    """Colors ROIs by the number of recordings in which they were tracked, indicating tracking reliability."""

    CELL_PROBABILITY = 8
    """Colors ROIs by the trained classifier's cell-probability estimate using a colormap gradient."""

    CORRELATIONS = 9
    """Colors ROIs by pairwise activity correlation with the selected ROI."""

    CELL_CLASSIFICATION = 10
    """Colors ROIs by binary cell/non-cell labels (non-cell uses the active colormap low endpoint, cell uses the high
    endpoint)."""


class ROIColorModeLabel(StrEnum):
    """Provides human-readable display labels for the ROIColorMode dropdown, indexed by ROIColorMode value."""

    RANDOM = "Random"
    """The display label for random hue-based ROI coloring."""

    SKEWNESS = "Skewness"
    """The display label for skewness-based ROI coloring."""

    COMPACTNESS = "Compactness"
    """The display label for compactness-based ROI coloring."""

    FOOTPRINT = "Footprint"
    """The display label for footprint-based ROI coloring."""

    ASPECT_RATIO = "Aspect Ratio"
    """The display label for aspect ratio-based ROI coloring."""

    SOLIDITY = "Solidity"
    """The display label for solidity-based ROI coloring."""

    COLOCALIZATION_PROBABILITY = "Colocalization"
    """The display label for channel 2 colocalization probability-based ROI coloring."""

    RECORDING_COUNT = "Recording Count"
    """The display label for recording count-based ROI coloring."""

    CELL_PROBABILITY = "Cell Probability"
    """The display label for classifier probability gradient ROI coloring."""

    CORRELATIONS = "Activity Correlation"
    """The display label for pairwise correlation-based ROI coloring."""

    CELL_CLASSIFICATION = "Classification"
    """The display label for binary cell/non-cell label ROI coloring."""


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
    """Displays the pre-computed baseline-and-neuropil-subtracted fluorescence trace."""

    DECONVOLVED = 3
    """Displays the deconvolved spikes trace."""


class TraceModeLabel(StrEnum):
    """Provides human-readable display labels for the TraceMode dropdown, indexed by TraceMode value."""

    RAW_FLUORESCENCE = "fluorescence"
    """The display label for the raw fluorescence trace mode."""

    NEUROPIL = "neuropil"
    """The display label for the neuropil fluorescence trace mode."""

    NEUROPIL_CORRECTED = "corrected"
    """The display label for the neuropil-corrected trace mode."""

    DECONVOLVED = "spikes"
    """The display label for the deconvolved spikes trace mode."""


class MaskLayer(IntEnum):
    """Selects the active ROI mask layer."""

    ORIGINAL = 0
    """Displays the original ROI masks from single-recording extraction in native recording coordinates."""

    DEFORMED = 1
    """Displays the original ROI masks warped to the shared cross-recording coordinate space via
    multi-recording registration deformation fields."""

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


class BackgroundViewLabel(StrEnum):
    """Provides human-readable display labels for the BackgroundView dropdown, indexed by BackgroundView value."""

    ROIS_ONLY = "ROIs"
    """The display label for the ROIs-only background mode."""

    MEAN_IMAGE = "Mean Image"
    """The display label for the temporal mean image background mode."""

    ENHANCED_MEAN_IMAGE = "Mean Image (Enhanced)"
    """The display label for the contrast-enhanced mean image background mode."""

    CORRELATION_MAP = "Correlation Map"
    """The display label for the pixel-wise activity correlation map background mode."""

    MAXIMUM_PROJECTION = "Maximum Projection"
    """The display label for the maximum intensity projection background mode."""

    CORRECTED_STRUCTURAL = "Corrected Structural"
    """The display label for the bleed-through-corrected structural channel background mode."""


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
    to the random, cell probability, cell classification, and correlation modes where the statistic is either
    categorical or already normalized to [0, 1]."""
    channel_2_color_divisor: float = 1.4
    """The divisor applied to random hue values when channel 2 data is present. Compresses the hue range so that
    channel 1 and channel 2 ROIs occupy visually distinct color bands in the random color overlay."""
    channel_2_color_offset: float = 0.1
    """The offset added to random hue values after division when channel 2 data is present. Shifts the compressed hue
    range away from zero, ensuring channel 2 ROIs are colored in a distinct hue band from channel 1 ROIs."""
    hsv_divisor: float = 1.4
    """The normalization divisor for percentile-based statistic values before HSV color mapping. Scales the statistic
    range to occupy a visually informative portion of the hue spectrum, preventing extreme hues from dominating the
    overlay."""
    hsv_offset: float = 0.4
    """The normalization offset applied before HSV division for percentile-based statistics. Shifts the scaled statistic
    values to center the color mapping within a perceptually useful region of the hue spectrum."""
    random_color_seed: int = 0
    """The seed for the random number generator used to assign ROI hue values. Ensures reproducible color assignments
    across recordings and viewer reloads, so the same ROI always receives the same random color."""
    plotted_trace_count: int = 40
    """The default and maximum number of simultaneously rendered fluorescence traces. Used as the initial value in the
    max-plotted input field, the QIntValidator upper bound, and the fallback when the field is empty."""
    default_scale_factor: float = 2.0
    """The default vertical spacing multiplier used to separate stacked fluorescence traces. Controls the Y-axis
    distance between adjacent traces in both single-recording and multi-recording trace plots, with larger values
    decreasing separation (trace spacing is computed as 1.0 / scale_factor)."""
    average_threshold: int = 5
    """An average trace is rendered only when more than this many ROIs are selected; at or below this count, only
    individual traces are shown to avoid displaying a noisy average from too few samples."""
    average_scale_divisor: float = 25.0
    """The divisor used to compute the vertical scale of the average trace relative to the number of selected ROIs.
    The average scale is calculated as (selected_count / divisor) + 1, producing a gradually increasing amplitude as
    more ROIs are selected."""
    top_selection_count: int = 40
    """The default and maximum number of ROIs selectable via top-n / bottom-n statistic ranking. Used as the initial
    value in the ranked count input field and the QIntValidator upper bound."""
    default_channel_2_threshold: float = 0.6
    """The default colocalization probability threshold for classifying ROIs as channel 2 positive. ROIs with a
    colocalization probability above this value are assigned to channel 2, and the threshold resets to this default
    on each recording load."""
    bin_size_divisor: int = 2
    """The divisor applied to the product of tau and sampling rate when computing the default temporal bin size. The
    bin size is calculated as max(1, int(tau * sampling_rate / divisor)) and controls the time window used for
    activity-correlation coloring (pairwise ROI correlation) computation."""


@dataclass(frozen=True, slots=True)
class _TrackingViewerConstants:
    """Encapsulates static runtime parameters for the tracking viewer window."""

    cycle_interval: int = 500
    """The millisecond interval for auto-cycling between recordings."""


@dataclass(frozen=True, slots=True)
class _BinaryPlayerConstants:
    """Encapsulates static runtime parameters for the binary player window."""

    playback_speed_multiplier: int = 5
    """The factor by which playback runs faster than the recording's real-time rate. A value of 5 means the binary
    viewer plays frames at 5x the original recording speed."""
    subsample_frame_count: int = 100
    """The number of evenly-spaced frames subsampled from the recording for dynamic range estimation. These frames
    are used to compute the mean and standard deviation that define the display intensity range."""
    default_frame_delta: int = 100
    """The default frame step size for arrow key navigation, playback advancement, and slider single-step."""
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
