"""Provides the ViewState dataclass and display-mode enums for GUI state tracking."""

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


class ROIToolPanel(IntEnum):
    """Identifies which image panel hosts the rectangular ROI selection tool."""

    CELLS = 0
    """The ROI selection tool is active on the cell image panel."""

    NON_CELLS = 1
    """The ROI selection tool is active on the non-cell image panel."""


class MaskLayer(IntEnum):
    """Selects the active ROI mask layer in multi-day mode."""

    ORIGINAL_ROI_MASKS = 0
    """Displays the original ROI masks from single-day extraction in native session coordinates."""

    DEFORMED_ROI_MASKS = 1
    """Displays the original ROI masks warped to the shared cross-session coordinate space via
    multi-day registration deformation fields."""

    TEMPLATE_ROI_MASKS = 2
    """Displays the consensus template ROI masks derived from cross-session clustering, defined in
    the shared coordinate space. These masks specify the ROIs tracked across sessions."""

    SESSION_TEMPLATE_ROI_MASKS = 3
    """Displays the template ROI masks warped back to native session coordinates via the inverse
    deformation fields."""


class CoordinateSpace(IntEnum):
    """Selects the coordinate space for reference images in multi-day mode."""

    NATIVE = 0
    """Displays reference images in the original recording session coordinate space."""

    TRANSFORMED = 1
    """Displays reference images warped to align with the cross-session template coordinate space."""


@dataclass
class ViewState:
    """Tracks all mutable GUI display and selection state.

    This dataclass decouples UI element state from the MainWindow orchestrator, allowing
    independent submodules to read and write display settings through a shared, typed object
    rather than through direct parent references.
    """

    # Image panel display state.
    rois_visible: bool = True
    """Controls whether ROI overlays are drawn on the cell and non-cell image panels."""

    roi_color_mode: ROIColorMode = ROIColorMode.RANDOM
    """Selects the statistic used to color ROI overlays. Must be a valid ``ROIColorMode``
    member."""

    background_view: BackgroundView = BackgroundView.ROIS_ONLY
    """Selects the background image displayed behind ROI overlays. Must be a valid
    ``BackgroundView`` member."""

    roi_opacity: list[int] = field(default_factory=lambda: [127, 255])
    """Alpha values for ROI overlays in [circle-view, filled-ROI-view] rendering modes. Each value
    ranges from 0 (fully transparent) to 255 (fully opaque)."""

    background_saturation: list[int] = field(default_factory=lambda: [0, 255])
    """Intensity range [min, max] applied to the background image before display. Pixel values
    outside this range are clipped, stretching contrast within the window."""

    roi_colormap: str = "hsv"
    """Name of the matplotlib colormap applied when mapping ROI statistics to overlay colors."""

    # ROI selection state.
    selected_roi_index: int = 0
    """Index of the currently highlighted ROI. This ROI's trace is plotted in the trace panel, and
    its spatial footprint is highlighted in the image panels."""

    merge_roi_indices: list[int] = field(default_factory=lambda: [0])
    """Indices of all ROIs staged for a merge operation. The merge dialog combines these ROIs into a
    single ROI when the user confirms the merge."""

    last_reclassified_index: int = 0
    """Index of the most recently reclassified ROI, used to briefly highlight the ROI that was moved
    between the cell and non-cell panels."""

    roi_tool_active: bool = False
    """Controls whether the rectangular ROI selection tool is active, allowing the user to drag a
    rectangle on an image panel to select ROIs within the region."""

    roi_tool_panel: ROIToolPanel = ROIToolPanel.CELLS
    """Identifies which image panel hosts the active rectangular ROI selection tool. Must be a
    valid ``ROIToolPanel`` member."""

    # Trace panel display state.
    trace_mode: TraceMode = TraceMode.NEUROPIL_CORRECTED
    """Selects which fluorescence trace type is displayed and used for correlation computations.
    Must be a valid ``TraceMode`` member."""

    temporal_bin_size: int = 1
    """Number of consecutive frames averaged together when computing binned activity traces. Higher
    values smooth the trace and reduce noise at the cost of temporal resolution."""

    fluorescence_visible: bool = True
    """Controls whether the cell_fluorescence trace (raw or neuropil-corrected) is drawn in the
    trace panel."""

    neuropil_visible: bool = True
    """Controls whether the neuropil_fluorescence trace is drawn in the trace panel."""

    deconvolved_visible: bool = True
    """Controls whether the deconvolved spike trace is drawn in the trace panel."""

    # Display toggles.
    auto_zoom_to_roi: bool = False
    """Controls whether the image panels automatically zoom to center on the currently selected
    ROI when the selection changes."""

    roi_labels_visible: bool = False
    """Controls whether numeric ROI index labels are drawn at each ROI centroid on the image
    panels."""

    session_loaded: bool = False
    """Indicates that a data context (single-day or multi-day) has been loaded into the GUI. When
    False, most interactive features are disabled."""

    # Channel 2 colocalization state.
    colocalization_threshold: float = 0.6
    """Display threshold applied to cell_colocalization_probabilities when rendering ROI overlays.
    ROIs with a probability above this value are visually distinguished as channel 2 cells in the
    color overlay. This threshold only affects visualization and does not modify the underlying
    classification data."""

    # Multi-day specific state.
    mask_layer: MaskLayer = MaskLayer.ORIGINAL_ROI_MASKS
    """Selects the active ROI mask layer in multi-day mode. Must be a valid ``MaskLayer``
    member."""

    coordinate_space: CoordinateSpace = CoordinateSpace.NATIVE
    """Selects the coordinate space for reference images in multi-day mode. Must be a valid
    ``CoordinateSpace`` member."""

    mask_opacity: float = 0.5
    """Alpha value for blending ROI mask overlays onto reference images in multi-day views. Ranges
    from 0.0 (fully transparent) to 1.0 (fully opaque)."""
