"""Provides data loading, saving, and GUI initialization for suite2p sessions."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from PySide6.QtWidgets import QFileDialog, QMessageBox
from scipy.interpolate import interp1d
from ataraxis_base_utilities import LogLevel, console

from .styles import (
    BUTTON_PRESSED_STYLESHEET,
    BUTTON_INACTIVE_STYLESHEET,
    BUTTON_UNPRESSED_STYLESHEET,
)
from .view_state import ViewState
from .trace_panel import plot_trace
from .context_data import ContextData
from .plot_widgets import initialize_ranges
from .roi_geometry import circle, boundary
from .roi_overlays import (
    draw_masks,
    display_masks,
    draw_colorbar,
    init_roi_maps,
    compute_colors,
    render_colorbar,
    update_custom_masks,
    update_behavior_masks,
)
from .background_views import build_views, display_views

if TYPE_CHECKING:
    from .main_window import MainWindow

# Default channel 2 probability threshold.
_DEFAULT_CHANNEL_2_THRESHOLD: float = 0.6

# Divisor for computing the default trace bin size from tau and sampling rate.
_BIN_SIZE_DIVISOR: int = 2

# Minimum number of columns in a behavior array to treat it as (data, time) format.
_MIN_BEHAVIOR_COLUMNS: int = 2

# Index of the channel 2 color mode button in the color button group.
_CHANNEL_2_COLOR_INDEX: int = 5

# Number of basic (non-dynamic) color mode buttons.
_BASIC_COLOR_COUNT: int = 8

# Index of the behavior color mode button.
_BEHAVIOR_COLOR_INDEX: int = 8

# Index of the rastermap / custom color mode button.
_CUSTOM_COLOR_INDEX: int = 9

# Default activity mode index (neuropil-corrected: F - 0.7*Fneu).
_DEFAULT_ACTIVITY_MODE: int = 2


def export_fig(parent: MainWindow) -> None:
    """Opens the pyqtgraph export dialog for the current plot.

    Args:
        parent: The main GUI window.
    """
    parent.win.scene().contextMenuItem = parent.p1
    parent.win.scene().showExportDialog()


def save_cell_classification(parent: MainWindow) -> None:
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
    save_path = context.save_path
    if save_path is None:
        return
    np.save(
        str(save_path / "cell_classification.npy"),
        np.concatenate(
            (
                np.expand_dims(context.cell_classification_labels[context.not_merged], axis=1),
                np.expand_dims(context.cell_classification_probabilities[context.not_merged], axis=1),
            ),
            axis=1,
        ),
    )
    parent.lcell0.setText(f"{int(context.cell_classification_labels.sum())}")
    parent.lcell1.setText(
        f"{int(context.cell_classification_labels.size - context.cell_classification_labels.sum())}"
    )


def save_cell_colocalization(parent: MainWindow) -> None:
    """Saves the current cell colocalization labels to cell_colocalization.npy.

    Args:
        parent: The main GUI window.
    """
    if parent.context_data is None:
        return
    context = parent.context_data
    save_path = context.save_path
    if save_path is None:
        return
    np.save(
        str(save_path / "cell_colocalization.npy"),
        np.concatenate(
            (
                np.expand_dims(context.cell_colocalization_labels[context.not_merged], axis=1),
                np.expand_dims(context.cell_colocalization_probabilities[context.not_merged], axis=1),
            ),
            axis=1,
        ),
    )


def save_merge(parent: MainWindow) -> None:
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
    save_path = context.save_path
    if save_path is None:
        return

    console.echo(message="Saving to NPY files...", level=LogLevel.SUCCESS)
    np.save(str(save_path / "F.npy"), context.cell_fluorescence)
    np.save(str(save_path / "Fneu.npy"), context.neuropil_fluorescence)
    np.save(str(save_path / "spks.npy"), context.spikes)
    if context.has_channel_2:
        if context.cell_fluorescence_channel_2 is not None:
            np.save(str(save_path / "F_chan2.npy"), context.cell_fluorescence_channel_2)
        if context.neuropil_fluorescence_channel_2 is not None:
            np.save(str(save_path / "Fneu_chan2.npy"), context.neuropil_fluorescence_channel_2)
        np.save(
            str(save_path / "cell_colocalization.npy"),
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
    np.save(str(save_path / "cell_classification.npy"), cell_classification)
    context.not_merged = np.ones(context.cell_classification_labels.size, dtype=np.bool_)


def load_session(parent: MainWindow, session_path: Path | None = None) -> None:
    """Loads a pipeline output directory into the GUI.

    Detects whether the directory contains single-day or multi-day pipeline output and
    creates the appropriate ContextData wrapper. Then initializes all GUI components
    with the loaded data.

    Args:
        parent: The main GUI window.
        session_path: Path to the suite2p output directory. If None, opens a dialog.
    """
    if session_path is None:
        session_path = _select_directory(parent=parent)
        if session_path is None:
            return

    console.echo(message=f"Loading session: {session_path}")

    is_multi_day = _is_multi_day_session(session_path=session_path)

    try:
        if is_multi_day:
            context_data = ContextData.from_multi_day(root_path=session_path)
        else:
            context_data = ContextData.from_single_day(root_path=session_path)
    except Exception:
        console.echo(message="Failed to load session data.", level=LogLevel.ERROR)
        _load_again(parent=parent, text="Failed to load session. Try another directory?")
        return

    parent.context_data = context_data
    parent.view_state = ViewState()
    _initialize_gui(parent=parent)


def load_dialog(parent: MainWindow) -> None:
    """Opens a directory dialog to select and load a suite2p output directory.

    Args:
        parent: The main GUI window.
    """
    load_session(parent=parent)


def load_dialog_folder(parent: MainWindow) -> None:
    """Opens a directory dialog to select and load a suite2p output directory.

    This is an alias for ``load_dialog`` maintained for menu bar compatibility.

    Args:
        parent: The main GUI window.
    """
    load_session(parent=parent)


def load_behavior(parent: MainWindow) -> None:
    """Opens a file dialog to load a behavioral trace array.

    The behavioral data is resampled to match the fluorescence frame count if timestamps
    are provided in a second column. After loading, the behavior color mode button is
    enabled and the behavior correlation masks are computed.

    Args:
        parent: The main GUI window.
    """
    if parent.context_data is None:
        return
    context = parent.context_data

    name = QFileDialog.getOpenFileName(parent, "Open *.npy", filter="*.npy")
    name = name[0]
    if not name:
        return

    behavior_loaded = False
    beh_time = np.arange(0, context.frame_count, dtype=np.float32)
    needs_resample = False

    try:
        beh = np.load(name)
        if beh.ndim > 1:
            if beh.shape[1] < _MIN_BEHAVIOR_COLUMNS:
                beh = beh.flatten()
                if beh.shape[0] == context.frame_count:
                    behavior_loaded = True
                    beh_time = np.arange(0, context.frame_count, dtype=np.float32)
            else:
                behavior_loaded = True
                beh_time = beh[:, 1].astype(np.float32)
                beh = beh[:, 0]
                needs_resample = True
        elif beh.shape[0] == context.frame_count:
            behavior_loaded = True
            beh_time = np.arange(0, context.frame_count, dtype=np.float32)
    except (ValueError, KeyError, OSError, RuntimeError, TypeError, NameError):
        console.echo(
            message="ERROR: this is not a 1D array with length of data",
            level=LogLevel.ERROR,
        )

    if not behavior_loaded:
        console.echo(
            message="ERROR: this is not a 1D array with length of data",
            level=LogLevel.ERROR,
        )
        return

    beh = beh.astype(np.float32)
    beh -= beh.min()
    beh_max = beh.max()
    if beh_max > 0:
        beh /= beh_max

    context.behavior = beh
    context.behavior_time = beh_time
    if needs_resample:
        context.behavior_resampled = _resample_frames(
            signal=beh,
            time_points=beh_time,
            target_times=np.arange(0, context.frame_count, dtype=np.float32),
        ).astype(np.float32)
    else:
        context.behavior_resampled = beh

    parent.view_state.behavior_loaded = True
    parent.colorbtns.button(_BEHAVIOR_COLOR_INDEX).setEnabled(True)
    parent.colorbtns.button(_BEHAVIOR_COLOR_INDEX).setStyleSheet(BUTTON_UNPRESSED_STYLESHEET)

    # Computes behavior correlation masks if binned activity data is available.
    if parent.Fbin is not None and parent.Fstd is not None and parent.color_arrays is not None:
        update_behavior_masks(
            color_arrays=parent.color_arrays,
            roi_maps=parent.roi_maps,
            binned_fluorescence=parent.Fbin,
            fluorescence_std=parent.Fstd,
            behavior_resampled=context.behavior_resampled,
            bin_size=parent.view_state.bin_size,
            merge_indices=parent.view_state.merge_indices,
            colormap=parent.view_state.colormap,
        )

    parent.update_plot()

    if hasattr(parent, "VW"):
        parent.VW.bloaded = parent.view_state.behavior_loaded
        parent.VW.beh = context.behavior
        parent.VW.beh_time = context.behavior_time
        parent.VW.plot_traces()

    parent.show()


def load_custom_mask(parent: MainWindow) -> None:
    """Opens a file dialog to load a custom ROI mask overlay.

    The mask must be a 1D array with one value per ROI. After loading, the custom color
    mode button is enabled and the mask overlay is displayed.

    Args:
        parent: The main GUI window.
    """
    if parent.context_data is None or parent.color_arrays is None or parent.roi_maps is None:
        return
    context = parent.context_data

    name = QFileDialog.getOpenFileName(parent, "Open *.npy", filter="*.npy")
    name = name[0]
    if not name:
        return

    custom_loaded = False
    try:
        mask = np.load(name).flatten().astype(np.float32)
        if mask.size == context.roi_count:
            custom_loaded = True
    except (ValueError, KeyError, OSError, RuntimeError, TypeError, NameError):
        console.echo(
            message="ERROR: this is not a 1D array with length of data",
            level=LogLevel.ERROR,
        )

    if not custom_loaded:
        console.echo(
            message="ERROR: this is not a 1D array with length of # of ROIs",
            level=LogLevel.ERROR,
        )
        return

    parent.custom_mask = mask
    update_custom_masks(
        color_arrays=parent.color_arrays,
        roi_maps=parent.roi_maps,
        custom_mask=mask,
        colormap=parent.view_state.colormap,
    )
    masks = draw_masks(
        context=context,
        state=parent.view_state,
        color_arrays=parent.color_arrays,
        roi_maps=parent.roi_maps,
    )
    display_masks(
        color1=parent.color1,
        color2=parent.color2,
        masks=masks,
    )

    parent.colorbtns.button(_CUSTOM_COLOR_INDEX).setEnabled(True)
    parent.colorbtns.button(_CUSTOM_COLOR_INDEX).setStyleSheet(BUTTON_UNPRESSED_STYLESHEET)
    parent.colorbtns.button(_CUSTOM_COLOR_INDEX).setChecked(True)
    parent.view_state.color_mode = _CUSTOM_COLOR_INDEX
    parent.update_plot()
    parent.show()


def _initialize_gui(parent: MainWindow) -> None:
    """Initializes all GUI components after loading context data.

    Computes boundary and circle geometry for each ROI, builds background views and
    color arrays, initializes plot ranges, and enables all interactive controls.

    Args:
        parent: The main GUI window with context_data and view_state already assigned.
    """
    context = parent.context_data
    state = parent.view_state
    if context is None:
        return

    # Resets display state.
    state.color_mode = 0
    state.view_mode = 0
    state.chosen_index = 0
    parent.checkBox.setChecked(True)
    if parent.checkBoxN.isChecked():
        parent._roi_text(False)
    parent.checkBoxN.setChecked(False)
    parent.checkBoxN.setEnabled(True)
    parent.loadBeh.setEnabled(True)
    parent.saveMerge.setEnabled(True)
    parent.sugMerge.setEnabled(True)
    parent.manual.setEnabled(True)
    parent._roi_remove()

    session_title = str(context.save_path) if context.save_path is not None else "unknown session"
    parent.setWindowTitle(session_title)

    # Computes default bin size from tau and sampling rate.
    state.bin_size = max(1, int(context.tau * context.sampling_rate / _BIN_SIZE_DIVISOR))
    parent.binedit.setText(str(state.bin_size))
    state.channel_2_threshold = _DEFAULT_CHANNEL_2_THRESHOLD
    parent.chan2edit.setText(str(state.channel_2_threshold))

    # Computes boundary and circle geometry for each ROI.
    _compute_roi_geometry(context=context)

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
        valid_y_range=(context.valid_y_range[0], context.valid_y_range[1]),
        valid_x_range=(context.valid_x_range[0], context.valid_x_range[1]),
    )

    # Computes color statistics and builds ROI index maps.
    parent.color_arrays = compute_colors(context=context, state=state)
    parent.roi_maps = init_roi_maps(context=context, color_arrays=parent.color_arrays)

    # Selects the first classified cell as the initial selection.
    first_cell = int(np.nonzero(context.cell_classification_labels)[0][0]) if context.cell_count > 0 else 0
    state.chosen_index = first_cell
    state.merge_indices = [first_cell]
    state.flipped_index = first_cell
    parent._ichosen_stats()
    parent.comboBox.setCurrentIndex(_DEFAULT_ACTIVITY_MODE)

    # Draws the colorbar and initial mask overlays.
    parent.colorbar_image = draw_colorbar(colormap=state.colormap)
    render_colorbar(
        state=state,
        color_arrays=parent.color_arrays,
        colorbar_widgets=parent.colorbar_widgets,
        colorbar_image=parent.colorbar_image,
    )

    tic = time.time()
    masks = draw_masks(
        context=context,
        state=state,
        color_arrays=parent.color_arrays,
        roi_maps=parent.roi_maps,
    )
    display_masks(
        color1=parent.color1,
        color2=parent.color2,
        masks=masks,
    )
    console.echo(message=f"Time to draw and plot masks: {time.time() - tic:.4f} sec")

    # Updates cell count labels.
    parent.lcell0.setText(f"{int(context.cell_count)}")
    parent.lcell1.setText(f"{int(context.roi_count - context.cell_count)}")

    # Initializes plot ranges and displays background views.
    parent.trange = initialize_ranges(
        cells_view=parent.p1,
        noncells_view=parent.p2,
        trace_box=parent.p3,
        frame_width=context.frame_width,
        frame_height=context.frame_height,
        frame_count=context.frame_count,
    )
    display_views(
        view1=parent.view1,
        view2=parent.view2,
        views=parent.views,
        view_index=state.view_mode,
        saturation=state.saturation,
    )
    plot_trace(
        trace_box=parent.p3,
        cell_fluorescence=context.cell_fluorescence,
        neuropil_fluorescence=context.neuropil_fluorescence,
        spikes=context.spikes,
        time_range=parent.trange,
        merge_indices=state.merge_indices,
        activity_mode=state.activity_mode,
    )

    # Sets aspect ratio on both panels.
    parent.p1.setAspectLocked(lock=True, ratio=context.aspect_ratio)
    parent.p2.setAspectLocked(lock=True, ratio=context.aspect_ratio)

    # Backward-compat alias for visualization_window.py.
    parent.color_names = [btn.text() for btn in parent.colorbtns.buttons()]

    state.is_loaded = True

    # Computes binned activity and triggers initial full redraw.
    parent.mode_change(_DEFAULT_ACTIVITY_MODE)
    parent.show()


def _enable_views_and_classifier(parent: MainWindow) -> None:
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
        parent.quadbtns.button(b).setEnabled(True)
        parent.quadbtns.button(b).setStyleSheet(BUTTON_UNPRESSED_STYLESHEET)

    # Enables view buttons.
    for b in range(len(parent.view_names)):
        parent.viewbtns.button(b).setEnabled(True)
        parent.viewbtns.button(b).setStyleSheet(BUTTON_UNPRESSED_STYLESHEET)
        if b == 0:
            parent.viewbtns.button(b).setChecked(True)
            parent.viewbtns.button(b).setStyleSheet(BUTTON_PRESSED_STYLESHEET)

    # Disables channel 2 views if no channel 2 data is available.
    if context.corrected_structural_mean_image is None:
        parent.viewbtns.button(5).setEnabled(False)
        parent.viewbtns.button(5).setStyleSheet(BUTTON_INACTIVE_STYLESHEET)
        if context.mean_image_channel_2 is None:
            parent.viewbtns.button(6).setEnabled(False)
            parent.viewbtns.button(6).setStyleSheet(BUTTON_INACTIVE_STYLESHEET)

    # Enables color mode buttons.
    color_button_count = len(parent.colorbtns.buttons())
    for b in range(color_button_count):
        if b == _CHANNEL_2_COLOR_INDEX:
            if context.has_channel_2:
                parent.colorbtns.button(b).setEnabled(True)
                parent.colorbtns.button(b).setStyleSheet(BUTTON_UNPRESSED_STYLESHEET)
        elif b == 0:
            parent.colorbtns.button(b).setEnabled(True)
            parent.colorbtns.button(b).setChecked(True)
            parent.colorbtns.button(b).setStyleSheet(BUTTON_PRESSED_STYLESHEET)
        elif b < _BASIC_COLOR_COUNT:
            parent.colorbtns.button(b).setEnabled(True)
            parent.colorbtns.button(b).setStyleSheet(BUTTON_UNPRESSED_STYLESHEET)

    # Enables size toggle buttons.
    for button_index, btn in enumerate(parent.sizebtns.buttons()):
        btn.setStyleSheet(BUTTON_UNPRESSED_STYLESHEET)
        btn.setEnabled(True)
        if button_index == 0:
            btn.setChecked(True)
            btn.setStyleSheet(BUTTON_PRESSED_STYLESHEET)

    # Enables selection buttons (draw enabled, top/bottom disabled until data analyzed).
    for b in range(3):
        if b == 0:
            parent.topbtns.button(b).setEnabled(True)
            parent.topbtns.button(b).setStyleSheet(BUTTON_UNPRESSED_STYLESHEET)
        else:
            parent.topbtns.button(b).setEnabled(False)
            parent.topbtns.button(b).setStyleSheet(BUTTON_INACTIVE_STYLESHEET)

    # Enables classifier menu items.
    parent.loadClass.setEnabled(True)
    parent.loadTrain.setEnabled(True)
    parent.loadUClass.setEnabled(True)
    parent.loadSClass.setEnabled(True)
    parent.resetDefault.setEnabled(True)
    parent.visualizations.setEnabled(True)
    parent.custommask.setEnabled(True)


def _compute_roi_geometry(context: ContextData) -> None:
    """Computes boundary and circle geometry for each ROI in the context.

    Populates the boundary_y_pixels, boundary_x_pixels, circle_y_pixels, and
    circle_x_pixels fields on each ROIStatistics instance.

    Args:
        context: The loaded data context whose ROI statistics will be modified.
    """
    for roi in context.roi_statistics:
        y_pixels = roi.y_pixels.flatten()
        x_pixels = roi.x_pixels.flatten()
        roi.boundary_y_pixels, roi.boundary_x_pixels = boundary(
            y_pixels=y_pixels,
            x_pixels=x_pixels,
        )
        y_circle, x_circle = circle(
            centroid=roi.centroid,
            radius=roi.radius,
        )
        valid = (
            (y_circle >= 0)
            & (x_circle >= 0)
            & (y_circle < context.frame_height)
            & (x_circle < context.frame_width)
        )
        roi.circle_y_pixels = y_circle[valid]
        roi.circle_x_pixels = x_circle[valid]


def _resample_frames(
    signal: np.ndarray,
    time_points: np.ndarray,
    target_times: np.ndarray,
) -> np.ndarray:
    """Resamples a behavioral signal to match fluorescence frame times.

    Applies Gaussian smoothing proportional to the resampling ratio before interpolation
    to prevent aliasing artifacts.

    Args:
        signal: The behavioral signal values.
        time_points: The original time points of the signal.
        target_times: The target time points to interpolate onto.

    Returns:
        Resampled signal at the target time points.
    """
    resampling_ratio = time_points.size / target_times.size
    smoothed = gaussian_filter1d(signal, np.ceil(resampling_ratio / 2), axis=0)
    interpolator = interp1d(time_points, smoothed, fill_value="extrapolate")
    return interpolator(target_times)


def _select_directory(parent: MainWindow) -> Path | None:
    """Opens a directory dialog to select a suite2p output directory.

    Args:
        parent: The main GUI window.

    Returns:
        The selected directory as a Path, or None if the dialog was cancelled.
    """
    name = QFileDialog.getExistingDirectory(
        parent=parent,
        caption="Open suite2p output directory",
    )
    if not name:
        return None
    return Path(name)


def _load_again(parent: MainWindow, text: str) -> None:
    """Shows an error dialog and optionally reopens the directory selection dialog.

    Args:
        parent: The main GUI window.
        text: The error message to display.
    """
    result = QMessageBox.question(parent, "ERROR", text, QMessageBox.Yes | QMessageBox.No)
    if result == QMessageBox.Yes:
        load_dialog(parent=parent)


def _is_multi_day_session(session_path: Path) -> bool:
    """Checks whether the session directory contains multi-day pipeline output.

    Args:
        session_path: Path to the suite2p output directory.

    Returns:
        True if a multiday_runtime_data.yaml file is found in the directory tree.
    """
    return any(session_path.rglob("multiday_runtime_data.yaml"))
