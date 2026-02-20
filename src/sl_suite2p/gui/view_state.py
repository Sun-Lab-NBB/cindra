"""Provides the ViewState dataclass that holds all GUI display and selection state."""

from __future__ import annotations

from dataclasses import field, dataclass


@dataclass
class ViewState:
    """Tracks all mutable GUI display state.

    Replaces the scattered ``parent.ops_plot``, ``parent.ichosen``, ``parent.activityMode``,
    and similar attributes that were previously stored directly on the MainWindow instance.
    """

    # Plot display state.
    rois_visible: bool = True
    """Determines whether ROI overlays are drawn on the image panels."""

    color_mode: int = 0
    """Index into the color statistics array (0-9)."""

    view_mode: int = 0
    """Index into the background view array (0-6)."""

    opacity: list[int] = field(default_factory=lambda: [127, 255])
    """Opacity values for ROI overlays in [circle-view, roi-view] modes."""

    saturation: list[int] = field(default_factory=lambda: [0, 255])
    """Saturation range for background image display."""

    colormap: str = "hsv"
    """Name of the active matplotlib colormap for ROI coloring."""

    # ROI selection state.
    chosen_index: int = 0
    """Index of the currently selected (highlighted) ROI."""

    merge_indices: list[int] = field(default_factory=lambda: [0])
    """Indices of all ROIs in the current merge selection."""

    flipped_index: int = 0
    """Index of the most recently flipped ROI."""

    is_roi_active: bool = False
    """Determines whether a rectangular ROI selection tool is active."""

    roi_plot_panel: int = 0
    """Panel index where the ROI selection tool is displayed (0=cells, 1=non-cells)."""

    # Trace display state.
    activity_mode: int = 2
    """Trace activity mode (0=F, 1=Fneu, 2=F-0.7*Fneu, 3=spks)."""

    bin_size: int = 1
    """Temporal bin size for activity computation."""

    traces_visible: bool = True
    """Determines whether the deconvolved spike trace is drawn."""

    neuropil_visible: bool = True
    """Determines whether the neuropil fluorescence trace is drawn."""

    deconvolved_visible: bool = True
    """Determines whether the deconvolved trace is drawn."""

    # View toggles.
    zoom_to_cell: bool = False
    """Determines whether the image panels auto-zoom to the selected cell."""

    show_roi_labels: bool = False
    """Determines whether ROI index labels are drawn on the image panels."""

    behavior_loaded: bool = False
    """Determines whether a behavioral trace has been loaded."""

    rastermap_loaded: bool = False
    """Determines whether rastermap sorting has been applied."""

    is_loaded: bool = False
    """Determines whether any data context has been loaded into the GUI."""

    # Channel 2 state.
    channel_2_threshold: float = 0.6
    """Probability threshold for classifying channel 2 (red) cells."""

    # Multi-day specific state.
    mask_set: int = 0
    """Active mask set index (0=detected, 1=registered, 2=template, 3=session-template)."""

    image_space: int = 0
    """Active image space index (0=native, 1=transformed/deformed)."""

    mask_opacity: float = 0.5
    """Overlay opacity for mask blending in multi-day views."""
