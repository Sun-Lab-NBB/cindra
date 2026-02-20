"""Provides the central Qt Signal bus for cross-module GUI communication."""

from __future__ import annotations

from PySide6.QtCore import Signal, QObject


class GUISignals(QObject):
    """Defines the central signal bus for cross-module GUI events.

    Modules emit and connect to signals on this object instead of calling methods on
    a parent window directly. This decouples modules from each other and from the
    MainWindow class.
    """

    # Session lifecycle signals.
    context_loaded = Signal()
    """Emitted after ContextData has been populated (single-day or multi-day)."""

    context_closing = Signal()
    """Emitted when the current context is about to be unloaded."""

    # ROI selection signals.
    selection_changed = Signal()
    """Emitted when chosen_index or merge_indices changed."""

    roi_flipped = Signal(int)
    """Emitted when an ROI is moved between the cell and non-cell panels."""

    # Data mutation signals.
    cells_reclassified = Signal()
    """Emitted when the is_cell array has been modified."""

    rois_merged = Signal()
    """Emitted when ROIs have been merged."""

    classifier_applied = Signal()
    """Emitted when the classifier has updated cell_probability."""

    # View change signals.
    view_mode_changed = Signal(int)
    """Emitted when the background view index changes."""

    color_mode_changed = Signal(int)
    """Emitted when the color statistic index changes."""

    activity_mode_changed = Signal(int)
    """Emitted when the trace activity mode changes."""

    # Multi-day specific signals.
    session_switched = Signal(int)
    """Emitted when the user switches to a different multi-day session index."""

    mask_set_changed = Signal(int)
    """Emitted when the user switches between detected, registered, or template masks."""

    image_space_changed = Signal(int)
    """Emitted when the user toggles between native and transformed reference images."""

    # Display update signals.
    plot_needs_update = Signal()
    """Emitted to request a full redraw of all plot panels."""

    trace_needs_update = Signal()
    """Emitted to request a redraw of the trace panel only."""
