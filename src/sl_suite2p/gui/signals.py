"""Provides the central Qt Signal bus for cross-module GUI communication."""

from __future__ import annotations

from PySide6.QtCore import Signal, QObject


class GUISignals(QObject):
    """Defines the central signal bus for cross-module GUI events.

    Modules emit and connect to signals on this object instead of calling methods on
    a parent window directly. This decouples modules from each other and from the
    MainWindow class.

    Notes:
        Signals that carry an ``int`` payload use ``IntEnum`` values defined in
        ``view_state``.
    """

    # Session lifecycle signals.
    context_loaded = Signal()
    """Emitted after ContextData has been populated with all pipeline arrays and metadata for the
    loaded session. Subscribers should use this signal to initialize their widgets and bind to the
    newly available data."""

    context_closing = Signal()
    """Emitted when the current ContextData is about to be unloaded. Subscribers should use this
    signal to release references to pipeline arrays, save pending edits, and reset widget state."""

    # ROI selection signals.
    roi_selection_changed = Signal()
    """Emitted when the actively selected ROI (selected_roi_index) or the set of ROIs staged for
    merging (merge_roi_indices) has changed. Subscribers should refresh highlights, trace plots, and
    any selection-dependent UI elements."""

    roi_reclassified = Signal(int)
    """Emitted when a single ROI is moved between the cell and non-cell panels via manual
    reclassification. The int argument is the index of the reclassified ROI."""

    # Data mutation signals.
    cells_reclassified = Signal()
    """Emitted after a bulk modification to the cell_classification_labels array, such as applying a
    probability threshold or restoring a saved classification. Subscribers should rebuild their cell
    and non-cell ROI lists from the updated labels."""

    rois_merged = Signal()
    """Emitted after two or more ROIs have been combined into a single ROI, which updates the pixel
    masks, fluorescence traces, and classification arrays. Subscribers should reload the full ROI
    set, as indices will have shifted."""

    classifier_applied = Signal()
    """Emitted after the trained classifier has recalculated the cell_classification_probabilities
    array. Subscribers that display or sort by classifier confidence should refresh their views."""

    # View change signals.
    background_view_changed = Signal(int)
    """Emitted when the background image displayed behind ROI overlays changes. The int argument is
    a ``BackgroundView`` member."""

    roi_color_mode_changed = Signal(int)
    """Emitted when the statistic used to color ROI overlays changes. The int argument is a
    ``ROIColorMode`` member."""

    trace_mode_changed = Signal(int)
    """Emitted when the fluorescence trace type shown in the trace panel changes. The int argument
    is a ``TraceMode`` member."""

    # Multi-day specific signals.
    session_switched = Signal(int)
    """Emitted when the user navigates to a different recording session in multi-day mode. The int
    argument is the new session index. Subscribers should expect all mutable pipeline arrays in
    ContextData to have been reloaded from the newly selected session."""

    mask_layer_changed = Signal(int)
    """Emitted when the displayed ROI mask layer changes in multi-day mode. The int argument is a
    ``MaskLayer`` member."""

    coordinate_space_changed = Signal(int)
    """Emitted when the coordinate space for reference images changes in multi-day mode. The int
    argument is a ``CoordinateSpace`` member."""

    # Supplementary data signals.
    behavior_loaded = Signal()
    """Emitted after a 1D behavioral trace has been loaded and resampled to match the imaging frame
    count. Subscribers should enable behavior-related UI elements such as the behavior correlation
    color mode and the behavioral trace overlay in the visualization window."""

    rastermap_loaded = Signal()
    """Emitted after rastermap sorting has been applied to reorder ROIs by embedding similarity.
    Subscribers should enable the rastermap color mode and update any ROI ordering displays."""

    # Display update signals.
    plot_needs_update = Signal()
    """Emitted to request a full redraw of all image panels, ROI overlays, and colorbars. Use this
    when underlying data has changed in a way that affects multiple visual elements simultaneously."""

    trace_needs_update = Signal()
    """Emitted to request a redraw of the fluorescence trace panel without refreshing image panels.
    Use this for changes that only affect trace display, such as toggling trace visibility or
    adjusting the temporal bin size."""
