"""Provides data loading, saving, and GUI initialization for cindra sessions."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QFileDialog, QMessageBox
from ataraxis_base_utilities import LogLevel, console

from ..constants import STYLE, CONFIG, ROIColorMode, BackgroundView
from ..widgets import plot_trace, initialize_ranges
from ..single_day_context import ROIViewerData
from ..overlays import (
    build_views,
    draw_masks,
    display_masks,
    display_views,
    draw_colorbar,
    init_roi_maps,
    compute_colors,
    render_colorbar,
)

if TYPE_CHECKING:
    from .viewer import ROIEditor


def export_fig(parent: ROIEditor) -> None:
    """Opens the pyqtgraph export dialog for the current plot.

    Args:
        parent: The main GUI window.
    """
    parent._graphics_widget.scene().contextMenuItem = parent._cells_view_box
    parent._graphics_widget.scene().showExportDialog()


def save_cell_classification(parent: ROIEditor) -> None:
    """Saves the current cell classification labels to cell_classification.npy.

    Writes both the boolean classification and probability arrays, filtered by the
    not-merged mask, to the session directory. Updates the cell count labels in the
    GUI after saving.

    Args:
        parent: The main GUI window.
    """
    if parent.context_data is None:
        return
    context = parent.context_data
    output_path = context.output_path
    if output_path is None:
        return
    np.save(
        str(output_path / "cell_classification.npy"),
        np.concatenate(
            (
                np.expand_dims(context.cell_classification_labels[context.not_merged], axis=1),
                np.expand_dims(context.cell_classification_probabilities[context.not_merged], axis=1),
            ),
            axis=1,
        ),
    )
    parent._cell_toggle_controls.cell_count_label.setText(f"{int(context.cell_classification_labels.sum())}")
    parent._cell_toggle_controls.noncell_count_label.setText(
        f"{int(context.cell_classification_labels.size - context.cell_classification_labels.sum())}"
    )


def save_cell_colocalization(parent: ROIEditor) -> None:
    """Saves the current cell colocalization labels to cell_colocalization.npy.

    Args:
        parent: The main GUI window.
    """
    if parent.context_data is None:
        return
    context = parent.context_data
    output_path = context.output_path
    if output_path is None:
        return
    np.save(
        str(output_path / "cell_colocalization.npy"),
        np.concatenate(
            (
                np.expand_dims(context.cell_colocalization_labels[context.not_merged], axis=1),
                np.expand_dims(context.cell_colocalization_probabilities[context.not_merged], axis=1),
            ),
            axis=1,
        ),
    )


def save_merge(parent: ROIEditor) -> None:
    """Saves all session data arrays after a merge operation.

    Writes fluorescence traces, spike deconvolution, and classification arrays to the
    session directory. Also saves channel 2 data if channel 2 is available. Resets
    the not-merged mask after saving.

    Args:
        parent: The main GUI window.
    """
    if parent.context_data is None:
        return
    context = parent.context_data
    output_path = context.output_path
    if output_path is None:
        return

    console.echo(message="Saving to NPY files...", level=LogLevel.SUCCESS)
    np.save(str(output_path / "F.npy"), context.cell_fluorescence)
    np.save(str(output_path / "Fneu.npy"), context.neuropil_fluorescence)
    np.save(str(output_path / "spks.npy"), context.spikes)
    if context.has_channel_2:
        if context.cell_fluorescence_channel_2 is not None:
            np.save(str(output_path / "F_chan2.npy"), context.cell_fluorescence_channel_2)
        if context.neuropil_fluorescence_channel_2 is not None:
            np.save(str(output_path / "Fneu_chan2.npy"), context.neuropil_fluorescence_channel_2)
        np.save(
            str(output_path / "cell_colocalization.npy"),
            np.concatenate(
                (
                    np.expand_dims(context.cell_colocalization_labels, axis=1),
                    np.expand_dims(context.cell_colocalization_probabilities, axis=1),
                ),
                axis=1,
            ),
        )
    cell_classification = np.concatenate(
        (
            context.cell_classification_labels[:, np.newaxis],
            context.cell_classification_probabilities[:, np.newaxis],
        ),
        axis=1,
    )
    np.save(str(output_path / "cell_classification.npy"), cell_classification)
    context.not_merged = np.ones(context.cell_classification_labels.size, dtype=np.bool_)


def load_session(parent: ROIEditor, session_path: Path | None = None) -> None:
    """Loads a pipeline output directory into the GUI.

    Detects whether the directory contains single-day or multi-day pipeline output and
    creates the appropriate ROIViewerData wrapper. Then initializes all GUI components
    with the loaded data.

    Args:
        parent: The main GUI window.
        session_path: Path to the cindra output directory. If None, opens a dialog.
    """
    if session_path is None:
        session_path = _select_directory(parent=parent)
        if session_path is None:
            return

    console.echo(message=f"Loading session: {session_path}")

    try:
        context_data = ROIViewerData.from_single_day(root_path=session_path, mutable=True)
    except Exception:
        console.echo(message="Failed to load session data.", level=LogLevel.ERROR)
        _load_again(parent=parent, text="Failed to load session. Try another directory?")
        return

    parent.context_data = context_data

    # Resets display state attributes to defaults.
    parent.rois_visible = True
    parent.roi_color_mode = ROIColorMode.RANDOM
    parent.background_view = BackgroundView.ROIS_ONLY
    parent.roi_opacity = [127, 255]
    parent.background_saturation = [0, 255]
    parent.roi_colormap = "hsv"
    parent.selected_roi_index = 0
    parent.merge_roi_indices = [0]
    parent.last_reclassified_index = 0
    parent.roi_tool_active = False
    parent.trace_mode = 2
    parent.temporal_bin_size = 1
    parent.fluorescence_visible = True
    parent.neuropil_visible = True
    parent.deconvolved_visible = True
    parent.auto_zoom_to_roi = False
    parent.roi_labels_visible = False
    parent.session_loaded = False
    parent.colocalization_threshold = 0.6

    _initialize_gui(parent=parent)


def load_dialog(parent: ROIEditor) -> None:
    """Opens a directory dialog to select and load a cindra output directory.

    Args:
        parent: The main GUI window.
    """
    load_session(parent=parent)


def load_dialog_folder(parent: ROIEditor) -> None:
    """Opens a directory dialog to select and load a cindra output directory.

    This is an alias for ``load_dialog`` maintained for menu bar compatibility.

    Args:
        parent: The main GUI window.
    """
    load_session(parent=parent)


def _initialize_gui(parent: ROIEditor) -> None:
    """Initializes all GUI components after loading context data.

    Builds background views and color arrays, initializes plot ranges, and enables
    all interactive controls.

    Args:
        parent: The main GUI window with context_data and view state attributes already assigned.
    """
    context = parent.context_data
    if context is None:
        return

    # Resets display state.
    parent.roi_color_mode = ROIColorMode(0)
    parent.background_view = BackgroundView(0)
    parent.selected_roi_index = 0
    parent._roi_visibility_checkbox.setChecked(True)
    if parent._roi_labels_checkbox.isChecked():
        parent._roi_text(False)
    parent._roi_labels_checkbox.setChecked(False)
    parent._roi_labels_checkbox.setEnabled(True)
    parent.saveMerge.setEnabled(True)
    parent.sugMerge.setEnabled(True)
    parent.manual.setEnabled(True)
    parent._roi_remove()

    session_title = str(context.output_path) if context.output_path is not None else "unknown session"
    parent.setWindowTitle(session_title)

    # Computes default bin size from tau and sampling rate.
    parent.temporal_bin_size = max(1, int(context.tau * context.sampling_rate / CONFIG.bin_size_divisor))
    parent._color_controls.bin_edit.setText(str(parent.temporal_bin_size))
    parent.colocalization_threshold = CONFIG.default_channel_2_threshold
    parent._color_controls.channel_2_edit.setText(str(parent.colocalization_threshold))

    # Enables buttons and menu items.
    _enable_views_and_classifier(parent=parent)

    # Builds background views from detection images.
    parent.views = build_views(
        frame_height=context.frame_height,
        frame_width=context.frame_width,
        mean_image=context.mean_image,
        enhanced_mean_image=context.enhanced_mean_image,
        correlation_map=context.correlation_map,
        maximum_projection=context.maximum_projection,
        corrected_channel_2_image=context.corrected_structural_mean_image,
        channel_2_mean_image=context.mean_image_channel_2,
        valid_y_range=context.valid_y_range,
        valid_x_range=context.valid_x_range,
    )

    # Computes color statistics and builds ROI index maps.
    parent.color_arrays = compute_colors(
        context=context,
        roi_colormap=parent.roi_colormap,
        colocalization_threshold=parent.colocalization_threshold,
    )
    parent.roi_maps = init_roi_maps(context=context, color_arrays=parent.color_arrays)

    # Selects the first classified cell as the initial selection.
    first_cell = int(np.nonzero(context.cell_classification_labels)[0][0]) if context.cell_count > 0 else 0
    parent.selected_roi_index = first_cell
    parent.merge_roi_indices = [first_cell]
    parent.last_reclassified_index = first_cell
    parent._ichosen_stats()
    parent._trace_controls.activity_combo.setCurrentIndex(CONFIG.default_context_activity_mode)

    # Draws the colorbar and initial mask overlays.
    parent.colorbar_image = draw_colorbar(colormap=parent.roi_colormap)
    if parent.colorbar_widgets is None or parent.colorbar_image is None:
        return
    render_colorbar(
        roi_color_mode=parent.roi_color_mode,
        color_arrays=parent.color_arrays,
        colorbar_widgets=parent.colorbar_widgets,
        colorbar_image=parent.colorbar_image,
    )

    tic = time.time()
    masks = draw_masks(
        context=context,
        color_arrays=parent.color_arrays,
        roi_maps=parent.roi_maps,
        roi_color_mode=parent.roi_color_mode,
        background_view=parent.background_view,
        roi_opacity=parent.roi_opacity,
        selected_roi_index=parent.selected_roi_index,
        merge_roi_indices=parent.merge_roi_indices,
    )
    display_masks(
        color1=parent._cells_overlay,
        color2=parent._noncells_overlay,
        masks=masks,
    )
    console.echo(message=f"Time to draw and plot masks: {time.time() - tic:.4f} sec")

    # Updates cell count labels.
    parent._cell_toggle_controls.cell_count_label.setText(f"{int(context.cell_count)}")
    parent._cell_toggle_controls.noncell_count_label.setText(f"{int(context.roi_count - context.cell_count)}")

    # Initializes plot ranges and displays background views.
    parent.frame_indices = initialize_ranges(
        cells_view=parent._cells_view_box,
        noncells_view=parent._noncells_view_box,
        trace_box=parent._trace_box,
        frame_width=context.frame_width,
        frame_height=context.frame_height,
        frame_count=context.frame_count,
    )
    display_views(
        view1=parent._cells_background,
        view2=parent._noncells_background,
        views=parent.views,
        view_index=parent.background_view,
        saturation=parent.background_saturation,
    )
    plot_trace(
        trace_box=parent._trace_box,
        cell_fluorescence=context.cell_fluorescence,
        neuropil_fluorescence=context.neuropil_fluorescence,
        spikes=context.spikes,
        frame_indices=parent.frame_indices,
        merge_indices=parent.merge_roi_indices,
        activity_mode=parent.trace_mode,
    )

    # Sets aspect ratio on both panels.
    parent._cells_view_box.setAspectLocked(lock=True, ratio=context.aspect_ratio)
    parent._noncells_view_box.setAspectLocked(lock=True, ratio=context.aspect_ratio)

    parent.session_loaded = True

    # Computes binned activity and triggers initial full redraw.
    parent.mode_change(CONFIG.default_context_activity_mode)
    parent.show()


def _enable_views_and_classifier(parent: ROIEditor) -> None:
    """Enables all view, color, and selection buttons after data loading.

    Configures button styles, enables channel 2 views if available, and activates the
    classifier menu items.

    Args:
        parent: The main GUI window with context_data already assigned.
    """
    if parent.context_data is None:
        return
    context = parent.context_data

    # Enables quadrant buttons.
    for b in range(9):
        parent._quadrant_controls.quadrant_buttons.button(b).setEnabled(True)
        parent._quadrant_controls.quadrant_buttons.button(b).setStyleSheet(STYLE.button_unpressed)

    # Enables view buttons.
    for b in range(len(parent._view_controls.view_names)):
        parent._view_controls.view_buttons.button(b).setEnabled(True)
        parent._view_controls.view_buttons.button(b).setStyleSheet(STYLE.button_unpressed)
        if b == 0:
            parent._view_controls.view_buttons.button(b).setChecked(True)
            parent._view_controls.view_buttons.button(b).setStyleSheet(STYLE.button_pressed)

    # Disables channel 2 views if no channel 2 data is available.
    if context.corrected_structural_mean_image is None:
        parent._view_controls.view_buttons.button(5).setEnabled(False)
        parent._view_controls.view_buttons.button(5).setStyleSheet(STYLE.button_inactive)
        if context.mean_image_channel_2 is None:
            parent._view_controls.view_buttons.button(6).setEnabled(False)
            parent._view_controls.view_buttons.button(6).setStyleSheet(STYLE.button_inactive)

    # Enables color mode buttons.
    color_button_count = len(parent._color_controls.color_buttons.buttons())
    for b in range(color_button_count):
        if b == CONFIG.color_channel_2:
            if context.has_channel_2:
                parent._color_controls.color_buttons.button(b).setEnabled(True)
                parent._color_controls.color_buttons.button(b).setStyleSheet(STYLE.button_unpressed)
        elif b == 0:
            parent._color_controls.color_buttons.button(b).setEnabled(True)
            parent._color_controls.color_buttons.button(b).setChecked(True)
            parent._color_controls.color_buttons.button(b).setStyleSheet(STYLE.button_pressed)
        elif b < CONFIG.basic_color_count:
            parent._color_controls.color_buttons.button(b).setEnabled(True)
            parent._color_controls.color_buttons.button(b).setStyleSheet(STYLE.button_unpressed)

    # Enables size toggle buttons.
    for button_index, btn in enumerate(parent._cell_toggle_controls.size_buttons.buttons()):
        btn.setStyleSheet(STYLE.button_unpressed)
        btn.setEnabled(True)
        if button_index == 0:
            btn.setChecked(True)
            btn.setStyleSheet(STYLE.button_pressed)

    # Enables selection buttons (draw enabled, top/bottom disabled until data analyzed).
    for b in range(3):
        if b == 0:
            parent._selection_controls.selection_buttons.button(b).setEnabled(True)
            parent._selection_controls.selection_buttons.button(b).setStyleSheet(STYLE.button_unpressed)
        else:
            parent._selection_controls.selection_buttons.button(b).setEnabled(False)
            parent._selection_controls.selection_buttons.button(b).setStyleSheet(STYLE.button_inactive)

    # Enables classifier menu items.
    parent.loadClass.setEnabled(True)
    parent.loadTrain.setEnabled(True)
    parent.loadUClass.setEnabled(True)
    parent.loadSClass.setEnabled(True)
    parent.resetDefault.setEnabled(True)


def _select_directory(parent: ROIEditor) -> Path | None:
    """Opens a directory dialog to select a cindra output directory.

    Args:
        parent: The main GUI window.

    Returns:
        The selected directory as a Path, or None if the dialog was cancelled.
    """
    name = QFileDialog.getExistingDirectory(
        parent=parent,
        caption="Open cindra output directory",
    )
    if not name:
        return None
    return Path(name)


def _load_again(parent: ROIEditor, text: str) -> None:
    """Shows an error dialog and optionally reopens the directory selection dialog.

    Args:
        parent: The main GUI window.
        text: The error message to display.
    """
    result = QMessageBox.question(parent, "ERROR", text, QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    if result == QMessageBox.StandardButton.Yes:
        load_dialog(parent=parent)
