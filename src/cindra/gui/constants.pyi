from enum import IntEnum, StrEnum
from dataclasses import dataclass

class ROIColorMode(IntEnum):
    RANDOM = 0
    SKEWNESS = 1
    COMPACTNESS = 2
    FOOTPRINT = 3
    ASPECT_RATIO = 4
    SOLIDITY = 5
    COLOCALIZATION_PROBABILITY = 6
    RECORDING_COUNT = 7
    CELL_PROBABILITY = 8
    CORRELATIONS = 9
    CELL_CLASSIFICATION = 10

class ROIColorModeLabel(StrEnum):
    RANDOM = "Random"
    SKEWNESS = "Skewness"
    COMPACTNESS = "Compactness"
    FOOTPRINT = "Footprint"
    ASPECT_RATIO = "Aspect Ratio"
    SOLIDITY = "Solidity"
    COLOCALIZATION_PROBABILITY = "Colocalization"
    RECORDING_COUNT = "Recording Count"
    CELL_PROBABILITY = "Cell Probability"
    CORRELATIONS = "Activity Correlation"
    CELL_CLASSIFICATION = "Classification"

class BackgroundView(IntEnum):
    ROIS_ONLY = 0
    MEAN_IMAGE = 1
    ENHANCED_MEAN_IMAGE = 2
    CORRELATION_MAP = 3
    MAXIMUM_PROJECTION = 4
    CORRECTED_STRUCTURAL = 5

class TraceMode(IntEnum):
    RAW_FLUORESCENCE = 0
    NEUROPIL = 1
    NEUROPIL_CORRECTED = 2
    DECONVOLVED = 3

class TraceModeLabel(StrEnum):
    RAW_FLUORESCENCE = "fluorescence"
    NEUROPIL = "neuropil"
    NEUROPIL_CORRECTED = "corrected"
    DECONVOLVED = "spikes"

class MaskLayer(IntEnum):
    ORIGINAL = 0
    DEFORMED = 1
    TEMPLATE = 2
    TRACKED = 3

class CoordinateSpace(IntEnum):
    NATIVE = 0
    TRANSFORMED = 1

class BackgroundViewLabel(StrEnum):
    ROIS_ONLY = "ROIs"
    MEAN_IMAGE = "Mean Image"
    ENHANCED_MEAN_IMAGE = "Mean Image (Enhanced)"
    CORRELATION_MAP = "Correlation Map"
    MAXIMUM_PROJECTION = "Maximum Projection"
    CORRECTED_STRUCTURAL = "Corrected Structural"

class Colormap(StrEnum):
    HSV = "hsv"
    VIRIDIS = "viridis"
    PLASMA = "plasma"
    INFERNO = "inferno"
    MAGMA = "magma"
    CIVIDIS = "cividis"
    VIRIDIS_R = "viridis_r"
    PLASMA_R = "plasma_r"
    INFERNO_R = "inferno_r"
    MAGMA_R = "magma_r"
    CIVIDIS_R = "cividis_r"

@dataclass(frozen=True, slots=True)
class _CommonConstants:
    lower_percentile: float = ...
    upper_percentile: float = ...

@dataclass(frozen=True, slots=True)
class _ROIViewerConstants:
    overlap_layers: int = ...
    fixed_colorbar_range: tuple[float, ...] = ...
    channel_2_color_divisor: float = ...
    channel_2_color_offset: float = ...
    hsv_divisor: float = ...
    hsv_offset: float = ...
    random_color_seed: int = ...
    plotted_trace_count: int = ...
    default_scale_factor: float = ...
    average_threshold: int = ...
    average_scale_divisor: float = ...
    top_selection_count: int = ...
    default_channel_2_threshold: float = ...
    bin_size_divisor: int = ...

@dataclass(frozen=True, slots=True)
class _TrackingViewerConstants:
    cycle_interval: int = ...

@dataclass(frozen=True, slots=True)
class _BinaryPlayerConstants:
    playback_speed_multiplier: int = ...
    subsample_frame_count: int = ...
    default_frame_delta: int = ...
    frame_slider_tick_interval: int = ...
    display_range_low_sigma: float = ...
    display_range_high_sigma: float = ...

@dataclass(frozen=True, slots=True)
class _PCViewerConstants:
    animation_interval_milliseconds: int = ...

COMMON_CONFIG: _CommonConstants
ROI_CONFIG: _ROIViewerConstants
TRACKING_CONFIG: _TrackingViewerConstants
BINARY_CONFIG: _BinaryPlayerConstants
PC_CONFIG: _PCViewerConstants
